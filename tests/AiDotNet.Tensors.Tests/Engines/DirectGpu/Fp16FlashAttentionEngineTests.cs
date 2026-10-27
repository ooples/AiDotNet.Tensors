#if NET5_0_OR_GREATER
using System;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// #1549 end to end: with the direct-PTX attention opt-in, <c>FlashAttention</c> at lengths the short online kernel does
/// not take (S = 300) and at head dimension 128 runs the FP16 flash kernel; its output and per-row log-sum-exp must
/// match CpuEngine's FP32 attention to FP16 accuracy, and the device backward that reads those statistics must match
/// CpuEngine's gradients.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class Fp16FlashAttentionEngineTests
{
    private readonly ITestOutputHelper _output;

    public Fp16FlashAttentionEngineTests(ITestOutputHelper output) => _output = output;

    private static Tensor<float> Random(Random rng, int[] shape, double range) =>
        new(Enumerable.Range(0, shape.Aggregate(1, (a, b) => a * b)).Select(_ => (float)((rng.NextDouble() * 2 - 1) * range)).ToArray(), shape);

    private static double MaxAbsDiff(Tensor<float> a, Tensor<float> b)
    {
        float[] x = a.ToArray(), y = b.ToArray();
        Assert.Equal(x.Length, y.Length);
        double worst = 0;
        for (int i = 0; i < x.Length; i++) worst = Math.Max(worst, Math.Abs((double)x[i] - y[i]));
        return worst;
    }

    // Default routing without the direct-PTX opt-in: an FP16 autocast scope (the caller chose FP16) takes the flash
    // kernel; plain FP32 attention stays on the exact FP32 kernel.
    [SkippableTheory]
    [InlineData(true)]
    [InlineData(false)]
    public void WithoutTheOptIn_Fp16AutocastTakesTheFlashKernel_AndFp32StaysExact(bool autocast)
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        bool? previous = DirectPtxFeatureGate.TestOverride;
        DirectPtxFeatureGate.TestOverride = false;
        try
        {
            using var engine = new DirectGpuTensorEngine();
            Skip.IfNot(engine.IsGpuAvailable, "CUDA tensor engine is unavailable.");
            var backend = Assert.IsType<CudaBackend>(engine.GetBackend());
            Skip.If(backend.IsDirectPtxAttentionEnabled, "The opt-in must be off for this test.");
            Skip.IfNot(backend.GetDirectPtxRuntime().ComputeCapabilityMajor >= 8,
                "The flash kernel needs compute capability 8.0+ (older devices keep the FP32 kernel).");
            var rng = new Random(77);
            int[] shape = { 2, 4, 257, 64 };
            Tensor<float> q = Random(rng, shape, 1.0), k = Random(rng, shape, 1.0), v = Random(rng, shape, 1.0);
            const double scale = 0.125;
            long before = backend.DirectPtxFlashAttentionDispatchCount;
            Tensor<float> output, stats;
            if (autocast)
            {
                using var scope = new AiDotNet.Tensors.Engines.Gpu.AutocastScope(AiDotNet.Tensors.Engines.Gpu.PrecisionMode.Float16);
                output = engine.FlashAttention(q, k, v, scale, true, out stats);
            }
            else
            {
                output = engine.FlashAttention(q, k, v, scale, true, out stats);
            }
            Tensor<float> expected = new CpuEngine().FlashAttention(q, k, v, scale, true, out Tensor<float> expectedStats);
            double outputError = MaxAbsDiff(output, expected), statsError = MaxAbsDiff(stats, expectedStats);
            _output.WriteLine($"autocast={autocast}: flash launches {backend.DirectPtxFlashAttentionDispatchCount - before}, " +
                              $"output max |err| {outputError:E2}, logsumexp max |err| {statsError:E2}");
            if (autocast)
            {
                Assert.Equal(before + 1, backend.DirectPtxFlashAttentionDispatchCount);
                Assert.True(outputError <= 1.5e-2 && statsError <= 1.5e-2, $"FP16 attention error {outputError:E2} / {statsError:E2}");
            }
            else
            {
                Assert.Equal(before, backend.DirectPtxFlashAttentionDispatchCount);
                Assert.True(outputError <= 1e-4 && statsError <= 1e-4, $"FP32 attention must stay exact: {outputError:E2} / {statsError:E2}");
            }
        }
        finally
        {
            DirectPtxFeatureGate.TestOverride = previous;
        }
    }

    [SkippableTheory]
    [InlineData(2, 4, 300, 64, false)]
    [InlineData(2, 4, 300, 64, true)]
    [InlineData(1, 2, 200, 128, false)]
    [InlineData(1, 2, 200, 128, true)]
    public void FlashAttention_LongOrWideShapes_RunTheFlashKernel_AndMatchTheCpu(int b, int h, int s, int d, bool causal)
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        bool? previous = DirectPtxFeatureGate.TestOverride;
        DirectPtxFeatureGate.TestOverride = true;
        try
        {
            using var engine = new DirectGpuTensorEngine();
            Skip.IfNot(engine.IsGpuAvailable, "CUDA tensor engine is unavailable.");
            var backend = Assert.IsType<CudaBackend>(engine.GetBackend());
            Skip.IfNot(backend.IsDirectPtxAttentionEnabled, "Requires a validated Ampere CUDA backend.");
            var rng = new Random(b * 1000 + s + d + (causal ? 1 : 0));
            int[] shape = { b, h, s, d };
            Tensor<float> q = Random(rng, shape, 1.0), k = Random(rng, shape, 1.0), v = Random(rng, shape, 1.0);
            double scale = 1 / Math.Sqrt(d);

            long before = backend.DirectPtxFlashAttentionDispatchCount;
            Tensor<float> output = engine.FlashAttention(q, k, v, scale, causal, out Tensor<float> stats);
            Assert.Equal(before + 1, backend.DirectPtxFlashAttentionDispatchCount);

            var cpu = new CpuEngine();
            Tensor<float> expected = cpu.FlashAttention(q, k, v, scale, causal, out Tensor<float> expectedStats);
            double outputError = MaxAbsDiff(output, expected), statsError = MaxAbsDiff(stats, expectedStats);
            _output.WriteLine($"B{b} H{h} S{s} D{d} causal={causal}: output max |err| {outputError:E2}, logsumexp max |err| {statsError:E2}");
            // FP16-rounded inputs (2^-11 relative) and FP16 accumulation per 64-key tile.
            Assert.True(outputError <= 1.5e-2, $"output max |err| {outputError:E2}");
            Assert.True(statsError <= 1.5e-2, $"logsumexp max |err| {statsError:E2}");

            Tensor<float> gradOutput = Random(rng, shape, 1.0);
            engine.FlashAttentionBackward(gradOutput, q, k, v, output, stats, scale, causal,
                out Tensor<float> gq, out Tensor<float> gk, out Tensor<float> gv);
            cpu.FlashAttentionBackward(gradOutput, q, k, v, expected, expectedStats, scale, causal,
                out Tensor<float> eq, out Tensor<float> ek, out Tensor<float> ev);
            double gradError = Math.Max(MaxAbsDiff(gq, eq), Math.Max(MaxAbsDiff(gk, ek), MaxAbsDiff(gv, ev)));
            _output.WriteLine($"  gradients max |err| {gradError:E2}");
            Assert.True(gradError <= 5e-2, $"gradient max |err| {gradError:E2}");
        }
        finally
        {
            DirectPtxFeatureGate.TestOverride = previous;
        }
    }
}
#endif