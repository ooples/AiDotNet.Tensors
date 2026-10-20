using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// The OpenCL inference routes added for GPU Predict, each against a CPU reference: the row-vector GEMV, the split-K
/// register-blocked GEMM, the batch-norm inference kernel (plain and with ReLU), and the engine caches that keep
/// persistent weights and transformed Winograd filters resident (they must still see in-place host writes).
/// </summary>
[Collection("DirectGpuSerial")]
public class OpenClInferenceKernelTests
{
    private static OpenClBackend RequireBackend()
    {
        Skip.IfNot(DirectOpenClContext.IsAvailable && DirectOpenClContext.GetDeviceCount() > 0, "No OpenCL device.");
        var backend = new OpenClBackend(deviceIndex: 0);
        if (!backend.IsAvailable)
        {
            backend.Dispose();
            Skip.If(true, "OpenCL backend did not initialize.");
        }
        return backend;
    }

    private static float[] Fill(Random rng, int count)
    {
        var a = new float[count];
        for (int i = 0; i < count; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        return a;
    }

    private static void AssertProduct(float[] a, float[] b, float[] actual, int m, int n, int k, string what)
    {
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
            {
                double sum = 0;
                for (int q = 0; q < k; q++) sum += (double)a[i * k + q] * b[q * n + j];
                Assert.True(Math.Abs(sum - actual[i * n + j]) <= 1e-5 * (k + 1),
                    $"{what} [{i},{j}]: expected {sum}, got {actual[i * n + j]}");
            }
    }

    // M = 1 takes the split-K GEMV when N is a multiple of 4 and K >= 256; the other shapes check the fallback.
    [SkippableTheory]
    [InlineData(4096, 300)]
    [InlineData(1000, 2048)]
    [InlineData(1002, 512)]
    [InlineData(64, 100)]
    public void Row_vector_gemm_matches_the_cpu_product(int n, int k)
    {
        using var backend = RequireBackend();
        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(11);
        var a = Fill(rng, k);
        var b = Fill(rng, k * n);
        using var ba = backend.AllocateBuffer(a);
        using var bb = backend.AllocateBuffer(b);
        using var bc = backend.AllocateBuffer(n);
        backend.Gemm(ba, bb, bc, 1, n, k);
        AssertProduct(a, b, backend.DownloadBuffer(bc), 1, n, k, $"gemv 1x{n}x{k}");
    }

    // One slice (a large tile grid), several slices with a short last one, and edges that fill no whole tile.
    [SkippableTheory]
    [InlineData(256, 3136, 64)]
    [InlineData(512, 49, 2048)]
    [InlineData(70, 13, 1000)]
    [InlineData(1, 1, 1)]
    public void Split_k_gemm_matches_the_cpu_product(int m, int n, int k)
    {
        using var backend = RequireBackend();
        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(13);
        var a = Fill(rng, m * k);
        var b = Fill(rng, k * n);
        using var ba = backend.AllocateBuffer(a);
        using var bb = backend.AllocateBuffer(b);
        using var bc = backend.AllocateBuffer(m * n);
        backend.RbSplitKGemm(ba, bb, bc, m, n, k);
        AssertProduct(a, b, backend.DownloadBuffer(bc), m, n, k, $"split-k {m}x{n}x{k}");
    }

    [SkippableTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void Batch_norm_inference_matches_the_cpu_reference(bool relu)
    {
        using var backend = RequireBackend();
        Assert.True(backend.HasCompiledKernel("batchnorm_inference"), "batchnorm_inference did not compile.");
        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(17);
        int batch = 2, channels = 5, spatial = 37;
        var x = Fill(rng, batch * channels * spatial);
        var gamma = Fill(rng, channels);
        var beta = Fill(rng, channels);
        var mean = Fill(rng, channels);
        var variance = new float[channels];
        for (int c = 0; c < channels; c++) variance[c] = (float)(rng.NextDouble() + 0.1);
        const float eps = 1e-5f;
        using var bx = backend.AllocateBuffer(x);
        using var by = backend.AllocateBuffer(x.Length);
        using var bg = backend.AllocateBuffer(gamma);
        using var bb = backend.AllocateBuffer(beta);
        using var bm = backend.AllocateBuffer(mean);
        using var bv = backend.AllocateBuffer(variance);
        using var sm = backend.AllocateBuffer(channels);
        using var sv = backend.AllocateBuffer(channels);
        if (relu)
            Assert.True(backend.TryFusedBatchNormActivation(bx, by, bg, bb, bm, bv, sm, sv, batch, channels, spatial, eps,
                0.1f, training: false, FusedActivationType.ReLU));
        else
            backend.BatchNorm(bx, by, bg, bb, bm, bv, sm, sv, batch, channels, spatial, eps, 0.1f, training: false);
        var actual = backend.DownloadBuffer(by);
        for (int i = 0; i < x.Length; i++)
        {
            int c = (i / spatial) % channels;
            double y = gamma[c] * (x[i] - mean[c]) / Math.Sqrt(variance[c] + eps) + beta[c];
            if (relu) y = Math.Max(0, y);
            Assert.True(Math.Abs(y - actual[i]) <= 1e-5 * (1 + Math.Abs(y)), $"[{i}]: expected {y}, got {actual[i]}");
        }
    }

    // A GPU engine on the OpenCL backend. Only an unavailable GPU skips: any other construction failure fails the test.
    private static DirectGpuTensorEngine RequireEngine()
    {
        Skip.IfNot(DirectOpenClContext.IsAvailable && DirectOpenClContext.GetDeviceCount() > 0, "No OpenCL device.");
        DirectGpuTensorEngine engine;
        try
        {
            engine = new DirectGpuTensorEngine();
        }
        catch (GpuCapabilityUnavailableException)
        {
            Skip.If(true, "No GPU engine.");
            throw;
        }
        if (!engine.SupportsGpu || engine.GetBackend() is not OpenClBackend)
        {
            engine.Dispose();
            Skip.If(true, "The GPU engine did not select the OpenCL backend.");
        }
        return engine;
    }

    private static Tensor<float> Random(Random rng, params int[] shape)
    {
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = (float)(rng.NextDouble() * 2 - 1);
        return t;
    }

    private static void AssertClose(Tensor<float> expected, Tensor<float> actual, double tolerance, string what)
    {
        var e = expected.ToArray();
        var a = actual.ToArray();
        Assert.Equal(e.Length, a.Length);
        for (int i = 0; i < e.Length; i++)
            Assert.True(Math.Abs(e[i] - a[i]) <= tolerance * (1 + Math.Abs(e[i])), $"{what} [{i}]: expected {e[i]}, got {a[i]}");
    }

    // A registered weight read by a generic op (TensorMatMul) is cached once; an in-place host write must still reach
    // the next product.
    [SkippableFact]
    public void Registered_weight_cached_by_matmul_sees_host_writes()
    {
        using var gpu = RequireEngine();
        var cpu = new CpuEngine();
        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(19);
        var w = Random(rng, 48, 40);
        var x = Random(rng, 40, 33);
        gpu.RegisterPersistentTensor(w, PersistentTensorRole.Weights);
        try
        {
            AssertClose(cpu.TensorMatMul(w, x), gpu.TensorMatMul(w, x), 1e-4, "first product");
            w[0] = 5f;
            w[w.Length - 1] = -3f;
            AssertClose(cpu.TensorMatMul(w, x), gpu.TensorMatMul(w, x), 1e-4, "after a host write");
        }
        finally
        {
            gpu.UnregisterPersistentTensor(w);
        }
    }

    // The F(4x4,3x3) route caches each weight's transformed filter; an in-place weight write must invalidate it.
    [SkippableFact]
    public void Winograd_filter_cache_sees_host_writes()
    {
        using var gpu = RequireEngine();
        var cpu = new CpuEngine();
        var previous = AiDotNet.Tensors.Helpers.Autotune.TunedKernels.TunedKernelPolicy.OverrideFor(
            AiDotNet.Tensors.Helpers.Autotune.TunedKernels.TunedKernelOp.Conv2DForward);
        AiDotNet.Tensors.Helpers.Autotune.TunedKernels.TunedKernelPolicy.SetOverride(
            AiDotNet.Tensors.Helpers.Autotune.TunedKernels.TunedKernelOp.Conv2DForward, "opencl.conv2d.winograd_f43_gemm");
        try
        {
            var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(23);
            var x = Random(rng, 1, 8, 13, 11);
            var w = Random(rng, 6, 8, 3, 3);
            AssertClose(cpu.Conv2D(x, w, 1, 1, 1), gpu.Conv2D(x, w, 1, 1, 1), 1e-4, "first convolution");
            AssertClose(cpu.Conv2D(x, w, 1, 1, 1), gpu.Conv2D(x, w, 1, 1, 1), 1e-4, "cached filter");
            for (int i = 0; i < w.Length; i += 7) w[i] = -w[i];
            AssertClose(cpu.Conv2D(x, w, 1, 1, 1), gpu.Conv2D(x, w, 1, 1, 1), 1e-4, "after a host write");
        }
        finally
        {
            AiDotNet.Tensors.Helpers.Autotune.TunedKernels.TunedKernelPolicy.SetOverride(
                AiDotNet.Tensors.Helpers.Autotune.TunedKernels.TunedKernelOp.Conv2DForward, previous);
        }
    }

    [SkippableTheory]
    [InlineData(FusedActivationType.None)]
    [InlineData(FusedActivationType.ReLU)]
    public void Engine_batch_norm_inference_with_activation_matches_cpu(FusedActivationType activation)
    {
        using var gpu = RequireEngine();
        var cpu = new CpuEngine();
        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(29);
        var x = Random(rng, 2, 6, 9, 7);
        var gamma = Random(rng, 6);
        var beta = Random(rng, 6);
        var mean = Random(rng, 6);
        var variance = new Tensor<float>(new[] { 6 });
        for (int i = 0; i < 6; i++) variance[i] = (float)(rng.NextDouble() + 0.1);
        AssertClose(cpu.BatchNormInference(x, gamma, beta, mean, variance, 1e-5, activation),
            gpu.BatchNormInference(x, gamma, beta, mean, variance, 1e-5, activation), 1e-5, $"bn {activation}");
    }

    // An empty batch reaches the six-argument override through the fused overload's CPU fallback; it must not launch a
    // kernel (which divides by batch * channels).
    [SkippableTheory]
    [InlineData(FusedActivationType.None)]
    [InlineData(FusedActivationType.ReLU)]
    public void Engine_batch_norm_inference_with_an_empty_batch_matches_cpu(FusedActivationType activation)
    {
        using var gpu = RequireEngine();
        var cpu = new CpuEngine();
        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(31);
        var x = new Tensor<float>(new[] { 0, 6, 9, 7 });
        var gamma = Random(rng, 6);
        var beta = Random(rng, 6);
        var mean = Random(rng, 6);
        var variance = new Tensor<float>(new[] { 6 });
        for (int i = 0; i < 6; i++) variance[i] = (float)(rng.NextDouble() + 0.1);
        var actual = gpu.BatchNormInference(x, gamma, beta, mean, variance, 1e-5, activation);
        Assert.Equal(new[] { 0, 6, 9, 7 }, actual.Shape.ToArray());
        AssertClose(cpu.BatchNormInference(x, gamma, beta, mean, variance, 1e-5, activation), actual, 1e-5, $"empty bn {activation}");
    }
// Every convolution route with the bias and ReLU: the fused routes apply them in their output kernel, the others
    // after it, and all must equal the CPU's convolution, bias add and ReLU. Shapes: 3x3 stride 1 (the Winograd routes),
    // 1x1 (the split-K GEMM on the input itself, one slice and several), and 3x3 stride 2 (im2col + GEMM).
    [SkippableTheory]
    [InlineData("opencl.conv2d.generic")]
    [InlineData("opencl.conv2d.im2col_gemm")]
    [InlineData("opencl.conv2d.tiled")]
    [InlineData("opencl.conv2d.direct")]
    [InlineData("opencl.conv2d.winograd_gemm")]
    [InlineData("opencl.conv2d.winograd_f43_gemm")]
    [InlineData("opencl.conv2d.winograd_fused")]
    [InlineData("opencl.conv2d.im2col_rb_gemm")]
    public void Fused_convolution_bias_and_relu_match_cpu_on_every_route(string route)
    {
        using var gpu = RequireEngine();
        var cpu = new CpuEngine();
        var op = AiDotNet.Tensors.Helpers.Autotune.TunedKernels.TunedKernelOp.Conv2DForward;
        var previous = AiDotNet.Tensors.Helpers.Autotune.TunedKernels.TunedKernelPolicy.OverrideFor(op);
        AiDotNet.Tensors.Helpers.Autotune.TunedKernels.TunedKernelPolicy.SetOverride(op, route);
        try
        {
            var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(37);
            var shapes = new (int C, int H, int W, int O, int K, int S, int P)[]
            {
                (8, 13, 11, 6, 3, 1, 1), (24, 9, 10, 40, 3, 1, 1), (16, 7, 7, 12, 1, 1, 0), (512, 7, 7, 70, 1, 1, 0),
                (6, 15, 13, 9, 3, 2, 1),
            };
            foreach (var (c, h, w, o, k, s, pad) in shapes)
            {
                var x = Random(rng, 1, c, h, w);
                var kernel = Random(rng, o, c, k, k);
                var bias = Random(rng, o);
                foreach (var withBias in new[] { true, false })
                    foreach (var activation in new[] { FusedActivationType.None, FusedActivationType.ReLU })
                    {
                        var b = withBias ? bias : null;
                        AssertClose(cpu.FusedConv2D(x, kernel, b, s, s, pad, pad, 1, 1, activation),
                            gpu.FusedConv2D(x, kernel, b, s, s, pad, pad, 1, 1, activation), 1e-4,
                            $"{route} {c}x{h}x{w}->{o} k{k} s{s} bias={withBias} {activation}");
                    }
            }
        }
        finally
        {
            AiDotNet.Tensors.Helpers.Autotune.TunedKernels.TunedKernelPolicy.SetOverride(op, previous);
        }
    }
}