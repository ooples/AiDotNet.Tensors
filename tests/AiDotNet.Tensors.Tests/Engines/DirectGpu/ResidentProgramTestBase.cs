using System;
using System.Linq;
using AiDotNet.Tensors.Engines.DirectGpu;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// #890 phase 3 on every backend (#1448-#1452): a resident program runs a declared chain - per-column affine + ReLU,
/// row softmax, residual add - in one launch (CUDA, HIP) or one submission (OpenCL, Vulkan, Metal, WebGPU). It must
/// equal a double-precision reference, equal the same stages run one by one bit for bit (both run the same device
/// code), and stay correct run after run. Every intermediate is NaN-poisoned first, so a stage that reads before the
/// one before it finished (a missing or broken barrier, or dispatches the API did not order) cannot pass on values a
/// previous run left behind.
/// </summary>
public abstract class ResidentProgramTestBase<TBackend> where TBackend : class, IDirectGpuBackend
{
    /// <summary>The backend, or a skip when it (or its resident program) is unavailable here.</summary>
    private protected abstract TBackend RequireBackend();

    /// <summary>Runs the chain resident; returns the number of launches (dispatches) it made.</summary>
    private protected abstract int RunResident(TBackend backend, ResidentStage[] stages);

    /// <summary>Whether a row-local chain runs as the single-launch row program here (CUDA keeps its grid-barrier
    /// program for every chain).</summary>
    private protected virtual bool HasRowProgram => true;

    private protected abstract void RunSeparately(TBackend backend, ResidentStage[] stages);

    /// <summary>Releases what <see cref="RequireBackend"/> created; a subclass holding a barrier frees it first.</summary>
    private protected virtual void ReleaseBackend(TBackend backend) => backend.Dispose();

    private protected sealed class Chain : IDisposable
    {
        public readonly int Rows, Cols;
        public readonly float[] X, Scale, Bias, Residual;
        public readonly IGpuBuffer DX, DScale, DBias, DResidual, DHidden, DProbs, DOut, DScale2;
        private readonly IDirectGpuBackend _b;

        public Chain(IDirectGpuBackend b, int rows, int cols, int seed)
        {
            _b = b; Rows = rows; Cols = cols;
            var rng = new Random(seed);
            float[] R(int len, double scale, double shift = 0) =>
                Enumerable.Range(0, len).Select(_ => (float)(shift + (rng.NextDouble() * 2 - 1) * scale)).ToArray();
            X = R(rows * cols, 2); Scale = R(cols, 0.5, 1); Bias = R(cols, 0.5); Residual = R(rows * cols, 1);
            DX = b.AllocateBuffer(X); DScale = b.AllocateBuffer(Scale); DBias = b.AllocateBuffer(Bias);
            DResidual = b.AllocateBuffer(Residual);
            DHidden = b.AllocateBuffer(rows * cols); DProbs = b.AllocateBuffer(rows * cols); DOut = b.AllocateBuffer(rows * cols);
            DScale2 = b.AllocateBuffer(cols);
        }

        public ResidentStage[] Stages => new[]
        {
            new ResidentStage(ResidentOp.AffineRelu, Rows, Cols, DX, DScale, DBias, DHidden),
            new ResidentStage(ResidentOp.RowSoftmax, Rows, Cols, DHidden, null, null, DProbs),
            new ResidentStage(ResidentOp.Add, Rows, Cols, DProbs, DResidual, null, DOut),
        };

        /// <summary>
        /// A chain that is not row-local: its first stage computes a new per-column scale, scale * bias + bias over
        /// [1, cols], which every row of the next stage reads. It must run stage after stage.
        /// </summary>
        public ResidentStage[] StagesNotRowLocal => new[]
        {
            new ResidentStage(ResidentOp.Affine, 1, Cols, DScale, DBias, DBias, DScale2),
            new ResidentStage(ResidentOp.AffineRelu, Rows, Cols, DX, DScale2, DBias, DHidden),
            new ResidentStage(ResidentOp.RowSoftmax, Rows, Cols, DHidden, null, null, DProbs),
            new ResidentStage(ResidentOp.Add, Rows, Cols, DProbs, DResidual, null, DOut),
        };

        public double[] Reference(bool notRowLocal = false)
        {
            // The device computes the new scale in fp32 with a fused multiply-add: the exact product plus the bias,
            // rounded to float (here through double, which differs by at most one rounding, far inside the tolerance).
            var scale = new double[Cols];
            for (int c = 0; c < Cols; c++)
                scale[c] = notRowLocal ? (double)(float)((double)Scale[c] * Bias[c] + Bias[c]) : Scale[c];
            var y = new double[Rows * Cols];
            for (int r = 0; r < Rows; r++)
            {
                double max = double.NegativeInfinity;
                var h = new double[Cols];
                for (int c = 0; c < Cols; c++)
                {
                    h[c] = Math.Max(0.0, (double)X[r * Cols + c] * scale[c] + Bias[c]);
                    max = Math.Max(max, h[c]);
                }
                double sum = 0;
                for (int c = 0; c < Cols; c++) sum += Math.Exp(h[c] - max);
                for (int c = 0; c < Cols; c++) y[r * Cols + c] = Math.Exp(h[c] - max) / sum + Residual[r * Cols + c];
            }
            return y;
        }

        public float[] Download() { _b.Synchronize(); return _b.DownloadBuffer(DOut); }

        private float[]? _nan;

        /// <summary>NaN in every intermediate and the output, uploaded from the host so the poison does not depend on
        /// a backend's fill kernel (WebGPU's dispatches at most 65535 workgroups, too few for the largest case).</summary>
        public void Poison()
        {
            var nan = _nan ??= Enumerable.Repeat(float.NaN, Rows * Cols).ToArray();
            _b.UploadBufferInPlace(nan, DHidden); _b.UploadBufferInPlace(nan, DProbs); _b.UploadBufferInPlace(nan, DOut);
        }

        public void Dispose()
        {
            foreach (var d in new[] { DX, DScale, DBias, DResidual, DHidden, DProbs, DOut, DScale2 }) d.Dispose();
        }
    }

    private protected static void AssertBitIdentical(float[] expected, float[] actual, int cols, string what)
    {
        for (int i = 0; i < expected.Length; i++)
            Assert.True(TestHelpers.MathCompat.SingleToInt32Bits(expected[i]) == TestHelpers.MathCompat.SingleToInt32Bits(actual[i]),
                $"[{i / cols},{i % cols}]: {what} {expected[i]:G9} vs {actual[i]:G9}");
    }

    private protected static void AssertMatchesReference(Chain chain, float[] actual, bool notRowLocal = false)
    {
        var expected = chain.Reference(notRowLocal);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= 2e-6 + 2e-6 * Math.Abs(expected[i]),
                $"[{i / chain.Cols},{i % chain.Cols}]: expected {expected[i]:G9}, got {actual[i]:G9}");
    }
    [SkippableTheory]
    [InlineData(64, 256)]     // the launch-bound size the phase-3 benchmark measures
    [InlineData(300, 77)]     // ragged columns; rows not a multiple of anything
    [InlineData(1, 4096)]     // one long row: a single group reduces it
    [InlineData(5000, 32)]    // more rows than a cooperative grid holds: groups stride over the rows
    [InlineData(70000, 2)]    // more softmax rows than one launch's group cap: groups stride over the rows
    [InlineData(4097, 4096)]  // more elements than 65535 groups of 256 cover: the elementwise stages grid-stride too
    public void MatchesAReference_AndTheSameStagesRunSeparately(int rows, int cols)
    {
        var backend = RequireBackend();
        try
        {
            using var chain = new Chain(backend, rows, cols, rows * 7 + cols);
            chain.Poison();
            RunResident(backend, chain.Stages);
            var resident = chain.Download();
            chain.Poison();
            RunSeparately(backend, chain.Stages);
            var separate = chain.Download();

            AssertBitIdentical(separate, resident, cols, "separate vs resident");
            AssertMatchesReference(chain, resident);
        }
        finally
        {
            ReleaseBackend(backend);
        }
    }

    // Hundreds of back-to-back chains with nothing waited on in between, every one poisoned first: a chain whose
    // stages ran out of order, or overlapped the next chain's poison, would leave a NaN or a stale value.
    [SkippableFact]
    public void ManyChainsInARow_StayCorrect()
    {
        var backend = RequireBackend();
        try
        {
            using var chain = new Chain(backend, 128, 512, 11);
            for (int i = 0; i < 300; i++) { chain.Poison(); RunResident(backend, chain.Stages); }
            var resident = chain.Download();
            chain.Poison();
            RunSeparately(backend, chain.Stages);
            AssertBitIdentical(chain.Download(), resident, chain.Cols, "separate vs 300th resident");
        }
        finally
        {
            ReleaseBackend(backend);
        }
    }

    [SkippableFact]
    public void RejectsAnEmptyOrOverlongChainAndAMissingOperand()
    {
        var backend = RequireBackend();
        try
        {
            using var chain = new Chain(backend, 4, 32, 3);
            var nine = Enumerable.Repeat(chain.Stages[0], ResidentProgramKernelTemplate.MaxStages + 1).ToArray();
            Assert.Throws<ArgumentException>(() => RunResident(backend, nine));
            Assert.Throws<ArgumentException>(() => RunResident(backend, Array.Empty<ResidentStage>()));
            Assert.Throws<ArgumentNullException>(() => new ResidentStage(ResidentOp.Add, 4, 32, chain.DX, null, null, chain.DOut));
            Assert.Throws<ArgumentException>(() => new ResidentStage(ResidentOp.Add, 4, 64, chain.DX, chain.DResidual, null, chain.DOut));
        }
        finally
        {
            ReleaseBackend(backend);
        }
    }

    [SkippableFact]
    public void RowLocalChain_RunsAsOneLaunch()
    {
        var backend = RequireBackend();
        try
        {
            Skip.IfNot(HasRowProgram, "This backend runs every chain as its grid-barrier program.");
            using var chain = new Chain(backend, 64, 256, 21);
            Assert.True(ResidentProgramKernelTemplate.TryPlanRowProgram(chain.Stages, out var plan));
            Assert.Equal(7, plan?.Buffers.Length);
            chain.Poison();
            Assert.Equal(1, RunResident(backend, chain.Stages));
            AssertMatchesReference(chain, chain.Download());
        }
        finally
        {
            ReleaseBackend(backend);
        }
    }

    // A stage whose output every row of the next stage reads cannot run row by row; the backend falls back, and the
    // result must still match the reference and the stages run one by one.
    [SkippableTheory]
    [InlineData(64, 256)]
    [InlineData(300, 77)]
    public void ChainThatIsNotRowLocal_MatchesAReference(int rows, int cols)
    {
        var backend = RequireBackend();
        try
        {
            using var chain = new Chain(backend, rows, cols, rows * 5 + cols);
            Assert.False(ResidentProgramKernelTemplate.TryPlanRowProgram(chain.StagesNotRowLocal, out _));
            var nanColumns = Enumerable.Repeat(float.NaN, cols).ToArray();
            chain.Poison();
            backend.UploadBufferInPlace(nanColumns, chain.DScale2);
            RunResident(backend, chain.StagesNotRowLocal);
            var resident = chain.Download();
            chain.Poison();
            backend.UploadBufferInPlace(nanColumns, chain.DScale2);
            RunSeparately(backend, chain.StagesNotRowLocal);
            AssertBitIdentical(chain.Download(), resident, cols, "separate vs resident");
            AssertMatchesReference(chain, resident, notRowLocal: true);
        }
        finally
        {
            ReleaseBackend(backend);
        }
    }

    [SkippableFact]
    public void ThePlanRefusesAWrittenColumnOperand_AndTooManyBuffers()
    {
        var backend = RequireBackend();
        try
        {
            using var chain = new Chain(backend, 1, 32, 9);
            // Same shapes, but the second stage reads as its scale what the first wrote: every row would need it whole.
            var writesScale = new[]
            {
                new ResidentStage(ResidentOp.Add, 1, 32, chain.DX, chain.DResidual, null, chain.DScale),
                new ResidentStage(ResidentOp.AffineRelu, 1, 32, chain.DX, chain.DScale, chain.DBias, chain.DOut),
            };
            Assert.False(ResidentProgramKernelTemplate.TryPlanRowProgram(writesScale, out _));
            var extra = Enumerable.Range(0, 8).Select(_ => backend.AllocateBuffer(32)).ToArray();
            try
            {
                // Nine distinct buffers: one more than a row program binds.
                var wide = new[]
                {
                    new ResidentStage(ResidentOp.Add, 1, 32, chain.DX, extra[0], null, extra[1]),
                    new ResidentStage(ResidentOp.Add, 1, 32, extra[2], extra[3], null, extra[4]),
                    new ResidentStage(ResidentOp.Add, 1, 32, extra[5], extra[6], null, extra[7]),
                };
                Assert.False(ResidentProgramKernelTemplate.TryPlanRowProgram(wide, out _));
            }
            finally
            {
                foreach (var e in extra) e.Dispose();
            }
        }
        finally
        {
            ReleaseBackend(backend);
        }
    }
}
