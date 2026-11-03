using System;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// The eager GPU optimizer step (what a tape-trained model's AdamW calls each step) with compressed moments: bf16
/// Adam/AdamW (half of fp32 moment memory) and int8 Adam/AdamW with weight decay (a quarter). Eager training had
/// only fp32 Adam/AdamW moments on the device and an int8 Adam step without decay, so AdamW -- every LM recipe's
/// optimizer -- could not shrink its state at all.
/// </summary>
[Collection("DirectGpuSerial")]
public class GpuOptimizerCompressedMomentStepTests
{
    private const int Length = 300;
    private const float Lr = 1e-2f, Beta1 = 0.9f, Beta2 = 0.99f, Eps = 1e-8f, WeightDecay = 0.1f;

    private static float[] Rand(int n, int seed, double scale = 1.0)
    {
        var rng = new Random(seed);
        return Enumerable.Range(0, n).Select(_ => (float)((rng.NextDouble() * 2 - 1) * scale)).ToArray();
    }

    private static DirectGpuTensorEngine? TryGpu()
    {
        bool required = Environment.GetEnvironmentVariable("AIDOTNET_REQUIRE_GPU_TESTS") == "1";
        try
        {
            var gpu = new DirectGpuTensorEngine();
            if (gpu.IsGpuAvailable) return gpu;
            gpu.Dispose();
            if (required) throw new InvalidOperationException("GPU tests require an available engine.");
        }
        catch (Exception ex) when (!required && (ex is PlatformNotSupportedException or DllNotFoundException)) { }
        return null;
    }

    [Fact]
    public void CompressedSteps_RejectGradientLengthMismatchBeforeFallback()
    {
        var p = new Tensor<float>(new[] { 2 });
        var g = new Tensor<float>(new[] { 1 });
        Assert.Throws<ArgumentException>(() => GpuOptimizer.TryAdamBf16Step(p, g, null, null,
            Lr, Beta1, Beta2, Eps, WeightDecay, 1, true));
        Assert.Throws<ArgumentException>(() => GpuOptimizer.TryAdam8BitStep(p, g, null, null, null, null,
            Lr, Beta1, Beta2, Eps, 0.1f, 0.01f, 64));
        Assert.Throws<ArgumentException>(() => GpuOptimizer.TryAdam8BitStep(p, g, null, null, null, null,
            Lr, Beta1, Beta2, Eps, 0.1f, 0.01f, 64, WeightDecay, true));
    }

    /// <summary>Puts a host array on the device as a GPU-resident tensor (the result of a device op).</summary>
    private static Tensor<float> OnDevice(DirectGpuTensorEngine gpu, float[] values)
        => gpu.TensorMultiplyScalar(new Tensor<float>(new[] { values.Length }, new Vector<float>(values)), 1f);

    [SkippableTheory]
    [InlineData(true, true)]
    [InlineData(true, false)]
    [InlineData(false, true)]
    [InlineData(false, false)]
    public void CompressedSteps_UpdateFlatViewsAndSharedHostStorage(bool bf16, bool decay)
    {
        using var gpu = TryGpu();
        Skip.If(gpu is null, "GPU engine unavailable.");
        var saved = AiDotNetEngine.Current;
        AiDotNetEngine.Current = gpu!;
        try
        {
            var buffer = new ParameterBuffer<float>(new[] { new[] { 2 }, new[] { 2 } });
            buffer.CopyFrom(new[] {
                new Tensor<float>(new[] { 2 }, new Vector<float>(new[] { 7f, 8f })),
                new Tensor<float>(new[] { 2 }, new Vector<float>(new[] { 3f, 4f })) });
            var p = buffer.CreateAllViews()[1];
            var g = OnDevice(gpu!, new[] { 0.5f, -0.25f });
            IGpuBuffer? m = null, v = null, ms = null, vs = null;
            try
            {
                var expected = new[] { 3f, 4f };
                var hm = new byte[4]; var hv = new byte[4];
                Assert.True(bf16
                    ? GpuOptimizer.TryAllocBf16MomentState(2, out m, out v)
                    : GpuOptimizer.TryAllocAdam8BitState(2, 64, out m, out v, out ms, out vs));
                GpuLaunchProbe.Reset();
                bool ran = bf16
                    ? GpuOptimizer.TryAdamBf16Step(p, g, m, v, Lr, Beta1, Beta2, Eps, WeightDecay, 1, decay)
                    : GpuOptimizer.TryAdam8BitStep(p, g, m, v, ms, vs, Lr, Beta1, Beta2, Eps,
                        1f - Beta1, 1f - Beta2, 64, WeightDecay, decay);
                Assert.True(ran, "compressed optimizer declined a flat parameter view");
                Assert.Equal(0L, GpuLaunchProbe.ReadbackBytes);
                Assert.True(gpu!.IsDeviceAuthoritative(p));
                if (bf16) HostCompressedMomentAdam.Bf16Step(expected, new[] { 0.5f, -0.25f }, hm, hv,
                    2, Lr, Beta1, Beta2, Eps, WeightDecay, 1, decay);
                else HostCompressedMomentAdam.Int8Step(expected, new[] { 0.5f, -0.25f }, hm, hv,
                    new float[1], new float[1], 2, 64, Lr, Beta1, Beta2, Eps, WeightDecay, 1, decay);
                var actual = p.ToArray();
                for (int i = 0; i < 2; i++) Assert.True(Math.Abs(expected[i] - actual[i]) < 1e-5f);
                Assert.Equal(new[] { 7f, 8f, actual[0], actual[1] }, buffer.AsVector().ToArray());
            }
            finally
            {
                GpuOptimizer.FreeGpuBuffer(m); GpuOptimizer.FreeGpuBuffer(v);
                GpuOptimizer.FreeGpuBuffer(ms); GpuOptimizer.FreeGpuBuffer(vs);
            }
        }
        finally { AiDotNetEngine.Current = saved; }
    }

    [SkippableFact]
    public void Bf16Step_AdamAndAdamW_MatchTheHostReference()
    {
        var gpu = TryGpu();
        Skip.If(gpu is null, "GPU engine unavailable.");
        var saved = AiDotNetEngine.Current;
        AiDotNetEngine.Current = gpu!;
        try
        {
            foreach (bool decoupled in new[] { true, false })
            {
                var initial = Rand(Length, 1);
                var p = OnDevice(gpu!, initial);
                Assert.True(GpuOptimizer.TryAllocBf16MomentState(Length, out var m, out var v));
                try
                {
                var refP = (float[])initial.Clone();
                var refM = new byte[Length * 2];
                var refV = new byte[Length * 2];
                for (int step = 1; step <= 3; step++)
                {
                    var grad = Rand(Length, 10 + step, 0.5);
                    var g = OnDevice(gpu!, grad);
                    Assert.True(GpuOptimizer.TryAdamBf16Step(p, g, m, v, Lr, Beta1, Beta2, Eps, WeightDecay, step, decoupled),
                        "the bf16 step fell back to the host");
                    HostCompressedMomentAdam.Bf16Step(refP, grad, refM, refV, Length, Lr, Beta1, Beta2, Eps, WeightDecay,
                        step, decoupled);
                }
                // The host API keeps the device's exact state layout, so state moves between them by upload/download.
                var backend = gpu!.GetBackend()!;
                Assert.Equal(refM, backend.DownloadByteBuffer(m!, Length * 2));
                Assert.Equal(refV, backend.DownloadByteBuffer(v!, Length * 2));
                var got = p.ToArray();
                for (int i = 0; i < Length; i++)
                    Assert.True(Math.Abs(refP[i] - got[i]) < 1e-5f,
                        $"bf16 {(decoupled ? "AdamW" : "Adam")} element {i}: device {got[i]} vs host {refP[i]}");
                }
                finally
                {
                    GpuOptimizer.FreeGpuBuffer(m); GpuOptimizer.FreeGpuBuffer(v);
                }
            }
        }
        finally
        {
            AiDotNetEngine.Current = saved;
            gpu!.Dispose();
        }
    }

    [SkippableFact]
    public void Int8Step_WithWeightDecay_MatchesTheHostReference_AndDecayActs()
    {
        var gpu = TryGpu();
        Skip.If(gpu is null, "GPU engine unavailable.");
        var saved = AiDotNetEngine.Current;
        AiDotNetEngine.Current = gpu!;
        const int blockSize = 64;
        int blocks = (Length + blockSize - 1) / blockSize;
        try
        {
            foreach (bool decoupled in new[] { true, false })
            {
                float[] Run(float weightDecay)
                {
                    var p = OnDevice(gpu!, Rand(Length, 2));
                    Assert.True(GpuOptimizer.TryAllocAdam8BitState(Length, blockSize, out var mQ, out var vQ, out var mS, out var vS));
                    try
                    {
                        for (int step = 1; step <= 3; step++)
                        {
                            var g = OnDevice(gpu!, Rand(Length, 20 + step, 0.5));
                            Assert.True(GpuOptimizer.TryAdam8BitStep(p, g, mQ, vQ, mS, vS, Lr, Beta1, Beta2, Eps,
                                1f - MathF.Pow(Beta1, step), 1f - MathF.Pow(Beta2, step), blockSize, weightDecay, decoupled),
                                "the int8 step fell back to the host");
                        }
                        if (weightDecay != 0f)
                        {
                            // The public host twin reproduces the device state byte for byte.
                            var hp = Rand(Length, 2);
                            var hmQ = new byte[Length]; var hvQ = new byte[Length];
                            var hmS = new float[blocks]; var hvS = new float[blocks];
                            for (int step = 1; step <= 3; step++)
                                HostCompressedMomentAdam.Int8Step(hp, Rand(Length, 20 + step, 0.5), hmQ, hvQ, hmS, hvS, Length, blockSize,
                                    Lr, Beta1, Beta2, Eps, weightDecay, step, decoupled);
                            var backend = gpu!.GetBackend()!;
                            Assert.Equal(hmQ, backend.DownloadByteBuffer(mQ!, Length));
                            Assert.Equal(hvQ, backend.DownloadByteBuffer(vQ!, Length));
                            var dev = p.ToArray();
                            for (int i = 0; i < Length; i++)
                                Assert.True(Math.Abs(hp[i] - dev[i]) < 1e-6f, $"host twin element {i}: {hp[i]} vs device {dev[i]}");
                        }
                        return p.ToArray();
                    }
                    finally
                    {
                        GpuOptimizer.FreeGpuBuffer(mQ); GpuOptimizer.FreeGpuBuffer(vQ);
                        GpuOptimizer.FreeGpuBuffer(mS); GpuOptimizer.FreeGpuBuffer(vS);
                    }
                }

                var got = Run(WeightDecay);
                var refP = Rand(Length, 2);
                var refMQ = new byte[Length]; var refVQ = new byte[Length];
                var refMS = new float[blocks]; var refVS = new float[blocks];
                for (int step = 1; step <= 3; step++)
                {
                    var grad = Rand(Length, 20 + step, 0.5);
                    // The decay applied outside the decay-free host kernel, the way each optimizer defines it.
                    for (int i = 0; i < Length; i++)
                    {
                        if (!decoupled) grad[i] += WeightDecay * refP[i];
                        else refP[i] *= 1f - Lr * WeightDecay;
                    }
                    CompressedMomentHostFallback.Adam8Bit(refP, grad, refMQ, refVQ, refMS, refVS, Lr, Beta1, Beta2, Eps,
                        1f - Beta1, 1f - Beta2, 1f - MathF.Pow(Beta1, step), 1f - MathF.Pow(Beta2, step), blockSize, Length, blocks);
                }
                string label = decoupled ? "AdamW" : "Adam + L2";
                for (int i = 0; i < Length; i++)
                    Assert.True(Math.Abs(refP[i] - got[i]) < 1e-5f, $"int8 {label} element {i}: device {got[i]} vs host {refP[i]}");

                var noDecay = Run(0f);
                float moved = got.Zip(noDecay, (a, b) => Math.Abs(a - b)).Max();
                Assert.True(moved > 1e-4f, $"int8 {label}: weight decay changed the weights by only {moved:E3}");
            }
        }
        finally
        {
            AiDotNetEngine.Current = saved;
            gpu!.Dispose();
        }
    }
}
