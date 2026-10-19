using System;
using System.Linq;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// #890 phase 3: the resident program (<see cref="CudaBackend.RunResidentProgram"/>) runs a declared chain - per-column
/// affine + ReLU, row softmax, residual add - as one persistent launch. It must equal a double-precision reference,
/// equal the same stages launched one by one bit for bit (they share the device code), and stay correct launch after
/// launch on one barrier (the barrier resets itself), including when the rows outnumber the grid.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class CudaResidentProgramTests
{
    private static CudaBackend RequireBackend()
    {
        var backend = new CudaBackend();
        if (!backend.IsAvailable || !backend.SupportsResidentProgram)
        {
            backend.Dispose();
            Skip.If(true, "CUDA with cooperative launch is not available.");
        }
        return backend;
    }

    private sealed class Chain : IDisposable
    {
        public readonly int Rows, Cols;
        public readonly float[] X, Scale, Bias, Residual;
        public readonly IGpuBuffer DX, DScale, DBias, DResidual, DHidden, DProbs, DOut;
        private readonly CudaBackend _b;

        public Chain(CudaBackend b, int rows, int cols, int seed)
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

        /// <summary>NaN in every intermediate and the output, so a stage that reads before the previous one finished
        /// (a missing or broken barrier) cannot pass on values an earlier run left behind.</summary>
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

    [SkippableTheory]
    [InlineData(64, 256)]     // the launch-bound size the phase-3 benchmark measures
    [InlineData(300, 77)]     // ragged columns; rows not a multiple of anything
    [InlineData(1, 4096)]     // one long row: a single block reduces it while the rest wait at the barriers
    [InlineData(5000, 32)]    // more rows than any cooperative grid: blocks stride over the rows
    public void MatchesAReference_AndTheSameStagesLaunchedSeparately(int rows, int cols)
    {
        using var backend = RequireBackend();
        using var chain = new Chain(backend, rows, cols, rows * 7 + cols);
        using var barrier = backend.AllocateResidentBarrier();

        chain.Poison();
        backend.RunResidentProgram(chain.Stages, barrier);
        var resident = chain.Download();
        chain.Poison();
        backend.RunStagesSeparately(chain.Stages);
        var separate = chain.Download();

        for (int i = 0; i < separate.Length; i++)
            Assert.True(BitConverter.SingleToInt32Bits(separate[i]) == BitConverter.SingleToInt32Bits(resident[i]),
                $"[{i / cols},{i % cols}]: separate {separate[i]:G9}, resident {resident[i]:G9}");
        var expected = chain.Reference();
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - resident[i]) <= 2e-6 + 2e-6 * Math.Abs(expected[i]),
                $"[{i / cols},{i % cols}]: expected {expected[i]:G9}, got {resident[i]:G9}");
    }

    // The barrier's count must return to zero at every barrier and its generation keep counting, or a later launch
    // would release early (a race) or hang. Hundreds of back-to-back launches on one barrier, then the result.
    [SkippableFact]
    public void OneBarrier_StaysCorrectAcrossManyLaunches()
    {
        using var backend = RequireBackend();
        using var chain = new Chain(backend, 128, 512, 11);
        using var barrier = backend.AllocateResidentBarrier();
        for (int i = 0; i < 300; i++) { chain.Poison(); backend.RunResidentProgram(chain.Stages, barrier); }
        var resident = chain.Download();
        backend.Synchronize();
        var counts = backend.DownloadBuffer(barrier);
        Assert.Equal(0f, counts[0]);   // the arrival count (read as float bits: zero is zero)

        chain.Poison();
        backend.RunStagesSeparately(chain.Stages);
        var separate = chain.Download();
        Assert.True(resident.Zip(separate).All(p => BitConverter.SingleToInt32Bits(p.First) == BitConverter.SingleToInt32Bits(p.Second)),
            "the 300th resident launch differs from the separate stages");
    }

    [SkippableFact]
    public void RejectsTooManyStagesAndAMissingOperand()
    {
        using var backend = RequireBackend();
        using var chain = new Chain(backend, 4, 32, 3);
        using var barrier = backend.AllocateResidentBarrier();
        var nine = Enumerable.Repeat(chain.Stages[0], 9).ToArray();
        Assert.Throws<ArgumentException>(() => backend.RunResidentProgram(nine, barrier));
        Assert.Throws<ArgumentNullException>(() => new ResidentStage(ResidentOp.Add, 4, 32, chain.DX, null, null, chain.DOut));
    }
}
