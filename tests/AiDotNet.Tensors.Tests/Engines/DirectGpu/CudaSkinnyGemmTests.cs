using System;
using System.Linq;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// The narrow row-major GEMMs CudaBackend runs before cuBLAS (skinny_gemm, skinny_gemm_tn): each case asserts the
/// kernel engaged -- a declined shape would silently fall back to cuBLAS and prove nothing -- and compares every
/// element with a double-precision reference.
/// </summary>
public class CudaSkinnyGemmTests
{
    [SkippableTheory]
    [InlineData(2048, 64, 64, false)]      // a d_model-64 Transformer's Q/K/V/output projection
    [InlineData(2048, 128, 64, false)]     // its FFN up-projection
    [InlineData(2048, 64, 128, false)]     // its FFN down-projection
    [InlineData(2048, 64, 64, true)]       // the projections' input gradient, dY · Wᵀ
    [InlineData(2048, 64, 128, true)]
    [InlineData(300, 4, 8, false)]         // rows not a multiple of the 32-row tile, one partial column tile
    [InlineData(257, 256, 256, true)]      // the widest shape taken, K swept in eight chunks
    [InlineData(333, 68, 36, false)]       // N and K past a tile edge
    [InlineData(256, 64, 128, false)]      // a 256-row FFN down-projection: few tiles
    [InlineData(256, 192, 64, true)]       // a 256-row in-projection, B transposed
    public void SkinnyGemm_EqualsTheReference(int m, int n, int k, bool transB)
    {
        using var backend = new CudaBackend();
        Skip.IfNot(backend.IsAvailable, "CUDA is not available.");
        float[] a = Fill(m * k, 1), b = Fill(k * n, 2), c0 = Fill(m * n, 3);
        using var aBuf = backend.AllocateBuffer(a);
        using var bBuf = backend.AllocateBuffer(b);
        // Every row tile on every shape (0: the device's own choice, which depends on its SM count).
        foreach (int tileRows in new[] { 0, 32, 16, 8 })
        foreach (var (alpha, beta) in new[] { (1f, 0f), (0.5f, 0.25f) })
        {
            // beta 0 must not read C: NaN there would poison every output.
            using var cBuf = backend.AllocateBuffer(beta == 0f ? Enumerable.Repeat(float.NaN, m * n).ToArray() : c0);
            Assert.True(backend.TrySkinnyGemm(aBuf, bBuf, cBuf, m, n, k, transB, alpha, beta, forceTileRows: tileRows),
                $"skinny_gemm declined {m}x{n}x{k} (tile rows {tileRows})");
            AssertClose(backend.DownloadBuffer(cBuf), i =>
            {
                int row = i / n, col = i % n;
                double s = 0, bound = 0;
                for (int t = 0; t < k; t++)
                {
                    double p = (double)a[row * k + t] * (transB ? b[col * k + t] : b[t * n + col]);
                    s += p; bound += Math.Abs(p);
                }
                return (alpha * s + beta * c0[i], Math.Abs(alpha) * bound + Math.Abs(beta * c0[i]));
            }, $"{m}x{n}x{k} transB={transB} tileRows={tileRows} alpha={alpha} beta={beta}");
        }
    }

    /// <summary>
    /// A linear layer's forward in one kernel (bias and ReLU in skinny_gemm's epilogue) must equal the GEMM followed by
    /// BiasAdd and Relu bit for bit -- including a NaN input, which the ReLU passes through.
    /// </summary>
    [SkippableTheory]
    [InlineData(2048, 64, 64, true)]
    [InlineData(2048, 128, 64, false)]
    [InlineData(300, 4, 8, true)]
    public void GemmBiasActivation_EqualsGemmThenBiasAddThenRelu(int m, int n, int k, bool relu)
    {
        using var backend = new CudaBackend();
        Skip.IfNot(backend.IsAvailable, "CUDA is not available.");
        float[] a = Fill(m * k, 9), w = Fill(k * n, 10), bias = Fill(n, 11);
        a[k + 1] = float.NaN;   // row 1 of the output is NaN throughout
        using var aBuf = backend.AllocateBuffer(a);
        using var wBuf = backend.AllocateBuffer(w);
        using var biasBuf = backend.AllocateBuffer(bias);
        using var fused = backend.AllocateBuffer(m * n);
        using var separate = backend.AllocateBuffer(m * n);
        var activation = relu ? AiDotNet.Tensors.Engines.FusedActivationType.ReLU : AiDotNet.Tensors.Engines.FusedActivationType.None;

        Assert.True(((AiDotNet.Tensors.Engines.DirectGpu.IGemmBiasActivationBackend)backend)
            .TryGemmBiasActivation(aBuf, wBuf, biasBuf, fused, m, n, k, activation), $"declined {m}x{n}x{k}");
        backend.Gemm(aBuf, wBuf, separate, m, n, k);
        backend.BiasAdd(separate, biasBuf, separate, m, n);
        if (relu) backend.Relu(separate, separate, m * n);

        var got = backend.DownloadBuffer(fused);
        var expected = backend.DownloadBuffer(separate);
        Assert.True(float.IsNaN(got[n]), "the NaN row did not stay NaN");
        for (int i = 0; i < got.Length; i++)
            if (TestHelpers.MathCompat.SingleToInt32Bits(got[i]) != TestHelpers.MathCompat.SingleToInt32Bits(expected[i]))
                Assert.Fail($"element {i}: one kernel {got[i]} != gemm + bias{(relu ? " + relu" : "")} {expected[i]}");
    }

    [SkippableTheory]
    [InlineData(64, 64, 2048)]     // a projection's weight gradient: two tiles, 32 splits
    [InlineData(64, 128, 2048)]
    [InlineData(128, 64, 2048)]
    [InlineData(256, 256, 4096)]   // 32 tiles, the most the per-thread counters hold
    [InlineData(4, 8, 513)]        // a partial tile and a reduction that is not a multiple of the 32-row chunk
    [InlineData(36, 68, 1000)]
    public void SkinnyGemmTransposedA_EqualsTheReference(int m, int n, int k)
    {
        using var backend = new CudaBackend();
        Skip.IfNot(backend.IsAvailable, "CUDA is not available.");
        float[] a = Fill(k * m, 4), b = Fill(k * n, 5), c0 = Fill(m * n, 6);
        using var aBuf = backend.AllocateBuffer(a);
        using var bBuf = backend.AllocateBuffer(b);
        foreach (var (alpha, beta) in new[] { (1f, 0f), (0.5f, 1f) })
        {
            using var cBuf = backend.AllocateBuffer(beta == 0f ? Enumerable.Repeat(float.NaN, m * n).ToArray() : c0);
            Assert.True(backend.TrySkinnyGemmTransposedA(aBuf, bBuf, cBuf, m, n, k, alpha, beta),
                $"skinny_gemm_tn declined {m}x{n}x{k}");
            AssertClose(backend.DownloadBuffer(cBuf), i => TransposedAReference(a, b, c0, m, n, k, alpha, beta, i),
                $"{m}x{n}x{k} alpha={alpha} beta={beta}");
        }
    }

    /// <summary>
    /// skinny_gemm_tn's last block per tile resets that tile's arrival counter. Without the reset a later launch
    /// never finds its last block and writes nothing -- so every launch here starts from a sentinel C and new
    /// inputs, inside a captured graph replayed several times as the compiled training step replays it, and must
    /// give the reference result with the same bits each replay of the same data.
    /// </summary>
    [SkippableFact]
    public void SkinnyGemmTransposedA_EveryReplayOfACapturedLaunchIsComplete()
    {
        using var backend = new CudaBackend();
        Skip.IfNot(backend.IsAvailable, "CUDA is not available.");
        const int m = 64, n = 64, k = 2048;
        using var aBuf = backend.AllocateBuffer(Fill(k * m, 7));
        using var bBuf = backend.AllocateBuffer(Fill(k * n, 8));
        using var cBuf = backend.AllocateBuffer(m * n);
        // Warm the thread's cuBLAS workspace and counters outside the capture, as the training step's eager pass does.
        Assert.True(backend.TrySkinnyGemmTransposedA(aBuf, bBuf, cBuf, m, n, k, 1f, 0f));
        // The kernel itself, not MatMulTransposedA: a fallback to cuBLAS would pass without exercising the counters.
        bool engaged = false;
        IntPtr exec = backend.CaptureGraph(() => engaged = backend.TrySkinnyGemmTransposedA(aBuf, bBuf, cBuf, m, n, k, 1f, 0f));
        Assert.True(engaged, "skinny_gemm_tn declined the captured launch");
        Assert.NotEqual(IntPtr.Zero, exec);
        try
        {
            float[]? previous = null;
            for (int replay = 0; replay < 4; replay++)
            {
                // Replays 0 and 1 run the same data (bits must repeat); later ones run new data.
                float[] a = Fill(k * m, 7 + Math.Max(0, replay - 1)), b = Fill(k * n, 8);
                backend.UploadBufferInPlace(a, aBuf);
                backend.UploadBufferInPlace(Enumerable.Repeat(123.25f, m * n).ToArray(), cBuf);
                backend.LaunchCapturedGraph(exec);
                var got = backend.DownloadBuffer(cBuf);
                AssertClose(got, i => TransposedAReference(a, b, new float[m * n], m, n, k, 1f, 0f, i), $"replay {replay}");
                if (replay == 1 && previous is not null)
                    for (int i = 0; i < got.Length; i++)
                        Assert.Equal(TestHelpers.MathCompat.SingleToInt32Bits(previous[i]), TestHelpers.MathCompat.SingleToInt32Bits(got[i]));
                previous = got;
            }
        }
        finally
        {
            backend.DestroyCapturedGraph(exec);
        }
    }

    [SkippableTheory]
    [InlineData("skinny_gemm", 256)]
    [InlineData("skinny_gemm_m16", 256)]
    [InlineData("skinny_gemm_m8", 256)]
    [InlineData("skinny_gemm_tn", 256)]
    public void SkinnyGemmKernels_KeepTheirAccumulatorsInRegisters(string kernel, int blockThreads)
    {
        using var backend = new CudaBackend();
        Skip.IfNot(backend.IsAvailable, "CUDA is not available.");
        var usage = backend.KernelResourceUsage(kernel, blockThreads);
        Assert.True(usage.HasValue, $"{kernel} is not loaded");
        Assert.True(usage.Value.LocalBytes == 0,
            $"{kernel} spills {usage.Value.LocalBytes} bytes per thread to local memory ({usage.Value.Registers} registers)");
    }

    [SkippableTheory]
    [InlineData(2048, 6, 64, false)]     // N not a multiple of four
    [InlineData(2048, 64, 300, false)]   // K past the narrow limit
    [InlineData(128, 64, 64, false)]     // too few rows to beat cuBLAS's tiling
    public void SkinnyGemm_DeclinesShapesItDoesNotServe(int m, int n, int k, bool transB)
    {
        using var backend = new CudaBackend();
        Skip.IfNot(backend.IsAvailable, "CUDA is not available.");
        using var aBuf = backend.AllocateBuffer(m * k);
        using var bBuf = backend.AllocateBuffer(k * n);
        using var cBuf = backend.AllocateBuffer(m * n);
        Assert.False(backend.TrySkinnyGemm(aBuf, bBuf, cBuf, m, n, k, transB, 1f, 0f));
    }

    private static (double Expected, double Bound) TransposedAReference(float[] a, float[] b, float[] c0,
        int m, int n, int k, float alpha, float beta, int i)
    {
        int row = i / n, col = i % n;
        double s = 0, bound = 0;
        for (int t = 0; t < k; t++)
        {
            double p = (double)a[t * m + row] * b[t * n + col];
            s += p; bound += Math.Abs(p);
        }
        return (alpha * s + beta * c0[i], Math.Abs(alpha) * bound + Math.Abs(beta * c0[i]));
    }

    // Relative to the sum of |terms|: fp32 accumulation error grows with it, not with the (possibly cancelling) result.
    /// <summary>
    /// The small products (TrySmallGemm): few outputs over a long reduction (a classifier head over a flattened
    /// sequence, small_gemm_rowreduce) and a reduction of at most 32 (its gradients, small_gemm_tinyk), with A or B
    /// transposed, the beta term and the bias/ReLU epilogue.
    /// </summary>
    [SkippableTheory]
    [InlineData(16, 10, 1024, false, false)]   // the head's forward
    [InlineData(16, 10, 1024, true, false)]
    [InlineData(16, 10, 1024, false, true)]
    [InlineData(3, 37, 300, false, false)]     // ragged column groups
    [InlineData(16, 1024, 10, false, true)]    // the head's input gradient, dY · Wᵀ
    [InlineData(1024, 10, 16, true, false)]    // its weight gradient, Xᵀ · dY
    [InlineData(5, 7, 1, false, false)]
    [InlineData(32, 10, 64, false, false)]    // an LSTM head over its last hidden state: a short reduction, few outputs
    [InlineData(32, 64, 128, true, false)]
    public void SmallGemm_EqualsTheReference(int m, int n, int k, bool transA, bool transB)
    {
        using var backend = new CudaBackend();
        Skip.IfNot(backend.IsAvailable, "CUDA is not available.");
        float[] a = Fill(m * k, 11), b = Fill(k * n, 12), c0 = Fill(m * n, 13), bias = Fill(n, 14);
        using var aBuf = backend.AllocateBuffer(a);
        using var bBuf = backend.AllocateBuffer(b);
        using var biasBuf = backend.AllocateBuffer(bias);
        foreach (var (alpha, beta, withBias, relu) in new[] { (1f, 0f, false, false), (0.5f, 0.25f, false, false), (1f, 0f, true, true) })
        {
            using var cBuf = backend.AllocateBuffer(beta == 0f ? Enumerable.Repeat(float.NaN, m * n).ToArray() : c0);
            Assert.True(backend.TrySmallGemm(aBuf, bBuf, cBuf, m, n, k, transA, transB, alpha, beta, withBias ? biasBuf : null, relu),
                $"small GEMM declined {m}x{n}x{k}");
            AssertClose(backend.DownloadBuffer(cBuf), i =>
            {
                int row = i / n, col = i % n;
                double s = 0, bound = 0;
                for (int t = 0; t < k; t++)
                {
                    double av = transA ? a[t * m + row] : a[row * k + t];
                    double bv = transB ? b[col * k + t] : b[t * n + col];
                    s += av * bv; bound += Math.Abs(av * bv);
                }
                double r = alpha * s + beta * c0[i] + (withBias ? bias[col] : 0);
                if (relu && r < 0) r = 0;
                return (r, Math.Abs(alpha) * bound + Math.Abs(beta * c0[i]) + (withBias ? Math.Abs(bias[col]) : 0));
            }, $"{m}x{n}x{k} transA={transA} transB={transB} alpha={alpha} beta={beta} bias={withBias} relu={relu}");
        }
    }

    [SkippableTheory]
    [InlineData(128, 64, 64)]     // neither few outputs nor a short reduction: cuBLAS's
    [InlineData(65, 10, 1024)]    // more rows than the row-reduction takes
    [InlineData(64, 64, 256)]     // too many outputs for the row-reduction (a 64x64 weight gradient)
    [InlineData(64, 64, 129)]     // a reduction just past the short limit at its most outputs
    public void SmallGemm_DeclinesShapesItDoesNotServe(int m, int n, int k)
    {
        using var backend = new CudaBackend();
        Skip.IfNot(backend.IsAvailable, "CUDA is not available.");
        using var aBuf = backend.AllocateBuffer(new float[m * k]);
        using var bBuf = backend.AllocateBuffer(new float[k * n]);
        using var cBuf = backend.AllocateBuffer(new float[m * n]);
        Assert.False(backend.TrySmallGemm(aBuf, bBuf, cBuf, m, n, k, false, false, 1f, 0f));
    }

    private static void AssertClose(float[] got, Func<int, (double Expected, double Bound)> reference, string what)
    {
        for (int i = 0; i < got.Length; i++)
        {
            var (expected, bound) = reference(i);
            if (!(Math.Abs(got[i] - expected) <= 1e-5 * bound + 1e-6))
                Assert.Fail($"{what}: element {i} = {got[i]}, expected {expected} (term magnitude {bound})");
        }
    }

    private static float[] Fill(int length, int seed)
    {
        var random = new Random(seed);
        return Enumerable.Range(0, length).Select(_ => (float)(random.NextDouble() * 2 - 1)).ToArray();
    }
}
