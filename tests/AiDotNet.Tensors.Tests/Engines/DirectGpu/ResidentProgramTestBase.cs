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

    private protected abstract void RunResident(TBackend backend, ResidentStage[] stages);

    private protected abstract void RunSeparately(TBackend backend, ResidentStage[] stages);

    /// <summary>Releases what <see cref="RequireBackend"/> created; a subclass holding a barrier frees it first.</summary>
    private protected virtual void ReleaseBackend(TBackend backend) => backend.Dispose();

    private protected sealed class Chain : IDisposable
    {
        public readonly int Rows, Cols;
        public readonly float[] X, Scale, Bias, Residual;
        public readonly IGpuBuffer DX, DScale, DBias, DResidual, DHidden, DProbs, DOut;
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
        }

        public ResidentStage[] Stages => new[]
        {
            new ResidentStage(ResidentOp.AffineRelu, Rows, Cols, DX, DScale, DBias, DHidden),
            new ResidentStage(ResidentOp.RowSoftmax, Rows, Cols, DHidden, null, null, DProbs),
            new ResidentStage(ResidentOp.Add, Rows, Cols, DProbs, DResidual, null, DOut),
        };

        public double[] Reference()
        {
            var y = new double[Rows * Cols];
            for (int r = 0; r < Rows; r++)
            {
                double max = double.NegativeInfinity;
                var h = new double[Cols];
                for (int c = 0; c < Cols; c++)
                {
                    h[c] = Math.Max(0.0, (double)X[r * Cols + c] * Scale[c] + Bias[c]);
                    max = Math.Max(max, h[c]);
                }
                double sum = 0;
                for (int c = 0; c < Cols; c++) sum += Math.Exp(h[c] - max);
                for (int c = 0; c < Cols; c++) y[r * Cols + c] = Math.Exp(h[c] - max) / sum + Residual[r * Cols + c];
            }
            return y;
        }

        public float[] Download() { _b.Synchronize(); return _b.DownloadBuffer(DOut); }

        /// <summary>NaN in every intermediate and the output.</summary>
        public void Poison()
        {
            int n = Rows * Cols;
            _b.Fill(DHidden, float.NaN, n); _b.Fill(DProbs, float.NaN, n); _b.Fill(DOut, float.NaN, n);
        }

        public void Dispose()
        {
            foreach (var d in new[] { DX, DScale, DBias, DResidual, DHidden, DProbs, DOut }) d.Dispose();
        }
    }

    private protected static void AssertBitIdentical(float[] expected, float[] actual, int cols, string what)
    {
        for (int i = 0; i < expected.Length; i++)
            Assert.True(TestHelpers.MathCompat.SingleToInt32Bits(expected[i]) == TestHelpers.MathCompat.SingleToInt32Bits(actual[i]),
                $"[{i / cols},{i % cols}]: {what} {expected[i]:G9} vs {actual[i]:G9}");
    }

    private protected static void AssertMatchesReference(Chain chain, float[] actual)
    {
        var expected = chain.Reference();
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
}
