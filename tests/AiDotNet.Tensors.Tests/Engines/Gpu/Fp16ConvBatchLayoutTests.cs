// System.Half (the FP16 oracle) only exists on .NET 5+; net471/net462 builds skip this file.
#if NET6_0_OR_GREATER

using System;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Tests.Engines.DirectGpu;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Gpu;

/// <summary>
/// The compiled step's FP16 im2col conv (<c>Conv2DInto</c> on the capture path, eviction suspended) must write NCHW for
/// any batch. Its im2col buffer is <c>col[K, batch * oh * ow]</c>, so a single GEMM over the batch wrote
/// <c>[outC][batch][oh * ow]</c>: every output past the first image's first channel was misplaced. Checked against
/// <see cref="CpuEngine"/> at batch 1 and 4, with FP32 and FP16-resident output.
/// </summary>
[Collection("DirectGpuSerial")]
public class Fp16ConvBatchLayoutTests : IClassFixture<DirectGpuTensorEngineTestFixture>
{
    // The test's own engine: the process-wide current engine is shared and other tests reset it to the CPU,
    // so reading it made these tests skip on a CUDA machine depending on test order.
    private readonly DirectGpuTensorEngineTestFixture _fixture;

    public Fp16ConvBatchLayoutTests(DirectGpuTensorEngineTestFixture fixture) => _fixture = fixture;

    // The backend primitive alone: out[b][M][P] = W[M,K] @ col[K, b*P + p] against an fp64 reference, both outputs.
    [SkippableTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void GemmFp16ConvBatched_WritesEveryImageInNchw(bool halfOutput)
    {
        Skip.IfNot(CudaNativeBindings.IsAvailable, "CUDA driver not available");
        using var backend = new CudaBackend();
        Skip.IfNot(backend.IsAvailable && backend.SupportsHgemm, "Needs a CUDA device with FP16 GEMM.");
        const int m = 24, pixels = 40, k = 72, batch = 3;
        int n = batch * pixels;
        var rng = new Random(7);
        Half[] w = Enumerable.Range(0, m * k).Select(_ => (Half)(rng.NextDouble() * 2 - 1)).ToArray();
        Half[] col = Enumerable.Range(0, k * n).Select(_ => (Half)(rng.NextDouble() * 2 - 1)).ToArray();
        static byte[] Bytes(Half[] v) => v.SelectMany(h => BitConverter.GetBytes(BitConverter.HalfToUInt16Bits(h))).ToArray();
        using var dw = backend.AllocateByteBuffer(m * k * 2);
        using var dcol = backend.AllocateByteBuffer(k * n * 2);
        using var dout = halfOutput ? backend.AllocateByteBuffer(m * n * 2) : backend.AllocateBuffer(m * n);
        backend.UploadBytes(dw, Bytes(w));
        backend.UploadBytes(dcol, Bytes(col));
        backend.GemmFp16ConvBatched(dw, dcol, dout, m, pixels, k, batch, halfOutput);
        float[] actual;
        if (halfOutput)
        {
            byte[] raw = backend.DownloadByteBuffer(dout, m * n * 2);
            actual = Enumerable.Range(0, m * n).Select(i => (float)BitConverter.UInt16BitsToHalf(BitConverter.ToUInt16(raw, 2 * i))).ToArray();
        }
        else
        {
            actual = backend.DownloadBuffer(dout);
        }
        for (int b = 0; b < batch; b++)
            for (int i = 0; i < m; i++)
                for (int p = 0; p < pixels; p++)
                {
                    double sum = 0;
                    for (int q = 0; q < k; q++) sum += (double)w[i * k + q] * (double)col[q * n + b * pixels + p];
                    float got = actual[(b * m + i) * pixels + p];
                    Assert.True(Math.Abs(got - sum) <= 1e-3 + (halfOutput ? Math.Abs(sum) * Math.Pow(2, -10) : 0),
                        $"out[{b}][{i}][{p}] = {got}, expected {sum}");
                }
    }

    // Opt-in (AIDOTNET_RUN_CONV_LAYOUT_PROBE=1): the per-image strided-batched GEMM against the single whole-batch GEMM
    // it replaces (same FLOPs, wrong layout), CUDA events, best of 3 rounds of 20, on ResNet-style conv GEMM shapes.
    [SkippableFact]
    public void Probe_BatchedConvGemmAgainstSingleGemm()
    {
        Skip.IfNot(Environment.GetEnvironmentVariable("AIDOTNET_RUN_CONV_LAYOUT_PROBE") == "1", "Set AIDOTNET_RUN_CONV_LAYOUT_PROBE=1.");
        Skip.IfNot(CudaNativeBindings.IsAvailable, "CUDA driver not available");
        using var backend = new CudaBackend();
        Skip.IfNot(backend.IsAvailable && backend.SupportsHgemm, "Needs a CUDA device with FP16 GEMM.");
        foreach (var (m, pixels, k, batch) in new[] { (64, 3136, 576, 8), (128, 784, 1152, 8), (256, 196, 2304, 8), (64, 3136, 576, 32) })
        {
            int n = batch * pixels;
            using var dw = backend.AllocateByteBuffer(m * k * 2);
            using var dcol = backend.AllocateByteBuffer(k * n * 2);
            using var dout = backend.AllocateBuffer(m * n);
            double Time(Action launch)
            {
                for (int i = 0; i < 3; i++) launch();
                backend.Synchronize();
                double best = double.MaxValue;
                for (int round = 0; round < 3; round++)
                {
                    using var start = backend.CreateEvent(enableTiming: true);
                    using var end = backend.CreateEvent(enableTiming: true);
                    backend.RecordEvent(start, backend.DefaultStream);
                    for (int i = 0; i < 20; i++) launch();
                    backend.RecordEvent(end, backend.DefaultStream);
                    end.Synchronize();
                    best = Math.Min(best, backend.GetEventElapsedTime(start, end) * 1000 / 20);
                }
                return best;
            }
            double single = Time(() => backend.GemmFp16In32fOut(dw, dcol, dout, m, n, k));
            double batched = Time(() => backend.GemmFp16ConvBatched(dw, dcol, dout, m, pixels, k, batch, halfOutput: false));
            Console.WriteLine($"M{m} P{pixels} K{k} batch {batch}: single GEMM {single:F1} us, per-image batched {batched:F1} us ({single / batched:F2}x)");
        }
    }

    // #1549: a conv large enough for the FP16-accumulate route (M 128, P 784, K 1152, batch 8) reaches it through the engine
    // (the route evaluates the per-image GEMM's shape) and still matches CpuEngine in NCHW.
    [SkippableFact]
    public void LargeBatchedFp16Conv_ReachesTheFp16AccumulateRoute_AndMatchesTheCpu()
    {
        Skip.IfNot(CudaNativeBindings.IsAvailable, "CUDA driver not available");
        var eng = _fixture.Engine;
        Skip.IfNot(eng is not null, "the DirectGpu engine did not initialize");
        var warm = new Tensor<float>(new float[] { 1, 2, 3, 4 }, new[] { 2, 2 });
        _ = eng!.TensorMatMul(warm, warm);
        var backend = eng!.TestBackend as CudaBackend;
        Skip.IfNot(backend is not null && backend.IsAvailable && backend.SupportsHgemm && backend.Fp16Im2colAvailable
                   && backend.GetDirectPtxRuntime().ComputeCapabilityMajor >= 8, "Needs the CUDA FP16 im2col conv path on Ampere+.");
        const int batch = 8, inC = 128, h = 28, w = 28, outC = 128;
        var rng = new Random(15490);
        var input = new Tensor<float>(Enumerable.Range(0, batch * inC * h * w).Select(_ => (float)(rng.NextDouble() * 2 - 1)).ToArray(),
            new[] { batch, inC, h, w });
        var kernel = new Tensor<float>(Enumerable.Range(0, outC * inC * 9).Select(_ => (float)(rng.NextDouble() * 2 - 1)).ToArray(),
            new[] { outC, inC, 3, 3 });
        float[] expected = new CpuEngine().Conv2D(input, kernel, 1, 1, 1).ToArray();
        var gpu = (DirectGpuTensorEngine)eng;
        var result = new Tensor<float>(new[] { batch, outC, h, w });
        bool? previousConv = DirectGpuTensorEngine.Fp16ConvOverride;
        DirectGpuTensorEngine.Fp16ConvOverride = true;
        gpu.SuspendActivationEviction();
        try
        {
            using (gpu.EnterCompiledCapturePath())
                ((IEngine)gpu).Conv2DInto(result, input, kernel, 1, 1, 1);
        }
        finally
        {
            gpu.ResumeActivationEviction();
            DirectGpuTensorEngine.Fp16ConvOverride = previousConv;
        }
        float[] actual = result.ToArray();
        var verdict = backend!.Fp16AccumulateVerdict(outC, h * w, inC * 9, halfOutput: false, batch: batch);
        Assert.True(verdict.HasValue, "the batched conv GEMM never reached the FP16-accumulate route");
        // FP16 operands, K = 1152 products in [-1, 1), FP16 accumulation per tile when routed.
        int bad = 0;
        for (int i = 0; i < expected.Length; i++)
            if (!(Math.Abs(actual[i] - expected[i]) <= 0.1)) bad++;
        Assert.True(bad == 0, $"{bad}/{expected.Length} outputs off (route: {verdict!.Value.Routed}, {verdict.Value.Reason})");
    }

    [SkippableTheory]
    [InlineData(1, false)]
    [InlineData(4, false)]
    [InlineData(4, true)]
    public void Fp16Im2colConv_WritesNchw_ForEveryBatch(int batch, bool fp16Activations)
    {
        Skip.IfNot(CudaNativeBindings.IsAvailable, "CUDA driver not available");
        var eng = _fixture.Engine;
        Skip.IfNot(eng is not null, "the DirectGpu engine did not initialize");
        var warm = new Tensor<float>(new float[] { 1, 2, 3, 4 }, new[] { 2, 2 });
        _ = eng!.TensorMatMul(warm, warm);
        var backend = eng!.TestBackend as CudaBackend;
        Skip.IfNot(backend is not null && backend.IsAvailable && backend.SupportsHgemm && backend.Fp16Im2colAvailable,
            "Needs the CUDA FP16 im2col conv path.");

        const int inC = 64, h = 28, w = 28, outC = 64;   // GEMM N = batch * 784 > the tiny-N cutoff, K = 576
        var rng = new Random(1549 + batch);
        var input = new Tensor<float>(Enumerable.Range(0, batch * inC * h * w).Select(_ => (float)(rng.NextDouble() * 2 - 1)).ToArray(),
            new[] { batch, inC, h, w });
        var kernel = new Tensor<float>(Enumerable.Range(0, outC * inC * 9).Select(_ => (float)(rng.NextDouble() * 2 - 1)).ToArray(),
            new[] { outC, inC, 3, 3 });
        float[] expected = new CpuEngine().Conv2D(input, kernel, 1, 1, 1).ToArray();

        var gpu = (DirectGpuTensorEngine)eng;
        var result = new Tensor<float>(new[] { batch, outC, h, w });
        bool? previousConv = DirectGpuTensorEngine.Fp16ConvOverride, previousAct = DirectGpuTensorEngine.Fp16ActOverride;
        long before = backend!.Fp16ConvBatchedDispatchCount;
        DirectGpuTensorEngine.Fp16ConvOverride = true;
        DirectGpuTensorEngine.Fp16ActOverride = fp16Activations;
        gpu.SuspendActivationEviction();
        try
        {
            using (gpu.EnterCompiledCapturePath())
                ((IEngine)gpu).Conv2DInto(result, input, kernel, 1, 1, 1);
        }
        finally
        {
            gpu.ResumeActivationEviction();
            DirectGpuTensorEngine.Fp16ConvOverride = previousConv;
            DirectGpuTensorEngine.Fp16ActOverride = previousAct;
        }
        float[] actual = result.ToArray();

        // Positive control: a batch above one ran the per-image GEMM (not the FP32 conv fallback).
        if (batch > 1) Assert.True(backend.Fp16ConvBatchedDispatchCount > before, "the batched FP16 conv GEMM did not run");
        // FP16 operands (2^-11 each) with FP32 accumulation over K = 576 products in [-1, 1), plus FP16 output rounding.
        double tolerance = fp16Activations ? 0.08 : 0.05;
        int bad = 0, firstBad = -1;
        for (int i = 0; i < expected.Length; i++)
            if (!(Math.Abs(actual[i] - expected[i]) <= tolerance)) { if (firstBad < 0) firstBad = i; bad++; }
        Assert.True(bad == 0, $"{bad}/{expected.Length} outputs off; first at {firstBad}: {(firstBad >= 0 ? $"{actual[firstBad]} vs {expected[firstBad]}" : "")}");
    }
}

#endif
