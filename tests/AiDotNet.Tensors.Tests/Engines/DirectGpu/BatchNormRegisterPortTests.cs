using System;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.Metal;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;
using AiDotNet.Tensors.Engines.DirectGpu.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// The register-resident training BatchNorm (<see cref="IBatchNormVarianceBackend"/>) on each non-CUDA backend
/// (<see cref="PortableBatchNorm"/>, #1489), against a CPU BatchNorm accumulated in double: every forward form (plain,
/// +ReLU, +residual+ReLU) and backward form (plain, ReLU mask read, ReLU mask rebuilt from x, +residual, a second output
/// gradient summed in), over slices that take each per-thread variant. The ReLU masks of the reference are the GPU
/// forward's own (its output > 0), so the rebuilt mask must reproduce the read one exactly.
/// </summary>
[Collection("DirectGpuSerial")]
public class BatchNormRegisterPortTests
{
    private readonly ITestOutputHelper _output;

    public BatchNormRegisterPortTests(ITestOutputHelper output) => _output = output;

    // batch, channels, spatial: slices of 10 (k4), 27 (k4, ragged), 512 and 2000 (k4, k8), 4096 (k16) and 8192 (k32).
    public static readonly TheoryData<int, int, int> Shapes = new()
    {
        { 2, 3, 5 },
        { 3, 7, 9 },
        { 8, 16, 64 },
        { 4, 5, 500 },
        { 4, 3, 1024 },
        { 8, 4, 1024 },
    };

    [SkippableTheory]
    [MemberData(nameof(Shapes))]
    public void OpenCl_register_batchnorm_matches_the_cpu_batchnorm(int batch, int channels, int spatial)
    {
        Skip.IfNot(DirectOpenClContext.IsAvailable && DirectOpenClContext.GetDeviceCount() > 0, "No OpenCL device.");
        using var backend = new OpenClBackend(deviceIndex: 0);
        Skip.IfNot(backend.IsAvailable, "OpenCL backend did not initialize.");
        Check(backend, batch, channels, spatial);
    }

    [SkippableTheory]
    [MemberData(nameof(Shapes))]
    public void Vulkan_register_batchnorm_matches_the_cpu_batchnorm(int batch, int channels, int spatial)
    {
        var backend = VulkanBackend.Instance;
        Skip.IfNot(backend.Initialize() && backend.IsGlslCompilerAvailable, "Vulkan with a GLSL compiler is not available.");
        Check(backend, batch, channels, spatial);
    }

    [SkippableTheory]
    [MemberData(nameof(Shapes))]
    public void Metal_register_batchnorm_matches_the_cpu_batchnorm(int batch, int channels, int spatial)
    {
        Skip.IfNot(System.Runtime.InteropServices.RuntimeInformation.IsOSPlatform(System.Runtime.InteropServices.OSPlatform.OSX),
            "Metal is only available on macOS.");
        using var backend = new MetalBackend();
        Skip.IfNot(backend.IsAvailable, "Metal is not available.");
        Check(backend, batch, channels, spatial);
    }

#if NET7_0_OR_GREATER
    [SkippableTheory]
    [MemberData(nameof(Shapes))]
    public void WebGpu_register_batchnorm_matches_the_cpu_batchnorm(int batch, int channels, int spatial)
    {
        var backend = DawnHostedWebGpu.BackendOrSkip();
        Check(backend, batch, channels, spatial);
    }
#endif

    /// <summary>The HIP module compiles as CUDA C++ under NVRTC (the dialect has nothing HIP-specific), on any CUDA box.</summary>
    [SkippableFact]
    public void Hip_register_batchnorm_module_compiles_as_cuda_cxx()
    {
        Skip.IfNot(AiDotNet.Tensors.Engines.DirectGpu.CUDA.NvrtcNativeBindings.IsAvailable, "NVRTC is not available.");
        string source = BatchNormRegisterKernelTemplate.GetModuleSource(WinogradKernelDialect.Hip);
        IntPtr program = IntPtr.Zero;
        Assert.Equal(AiDotNet.Tensors.Engines.DirectGpu.CUDA.NvrtcResult.Success,
            AiDotNet.Tensors.Engines.DirectGpu.CUDA.NvrtcNativeBindings.nvrtcCreateProgram(ref program, source, "bn_hip",
                0, IntPtr.Zero, IntPtr.Zero));
        try
        {
            var result = AiDotNet.Tensors.Engines.DirectGpu.CUDA.NvrtcNativeBindings.nvrtcCompileProgram(program, 1,
                new[] { "--gpu-architecture=compute_75" });
            Assert.True(result == AiDotNet.Tensors.Engines.DirectGpu.CUDA.NvrtcResult.Success, $"the HIP dialect did not compile ({result})");
        }
        finally
        {
            AiDotNet.Tensors.Engines.DirectGpu.CUDA.NvrtcNativeBindings.nvrtcDestroyProgram(ref program);
        }
    }

    private const float Epsilon = 1e-5f;

    private void Check(IDirectGpuBackend backend, int batch, int channels, int spatial)
    {
        var bn = Assert.IsAssignableFrom<IBatchNormVarianceBackend>(backend);
        int n = batch * channels * spatial;
        var x = Random(n, 1, 2.0, 0.3);
        var residual = Random(n, 2, 1.0, 0.0);
        var gamma = Random(channels, 3, 1.0, 1.0);
        var beta = Random(channels, 4, 0.5, 0.0);
        var dy = Random(n, 5, 1.0, 0.0);
        var dy2 = Random(n, 6, 1.0, 0.0);
        using var bx = backend.AllocateBuffer(x);
        using var bres = backend.AllocateBuffer(residual);
        using var bg = backend.AllocateBuffer(gamma);
        using var bb = backend.AllocateBuffer(beta);
        using var bdy = backend.AllocateBuffer(dy);
        using var bdy2 = backend.AllocateBuffer(dy2);
        using var by = backend.AllocateBuffer(n);
        using var bmean = backend.AllocateBuffer(channels);
        using var bvar = backend.AllocateBuffer(channels);
        using var bdx = backend.AllocateBuffer(n);
        using var bdg = backend.AllocateBuffer(channels);
        using var bdb = backend.AllocateBuffer(channels);
        using var bdr = backend.AllocateBuffer(n);
        string label = $"N{batch} C{channels} S{spatial}";
        var (refMean, refVar) = Statistics(x, batch, channels, spatial);

        foreach (var (relu, withResidual) in new[] { (false, false), (true, false), (true, true) })
        {
            string form = withResidual ? "add+relu" : relu ? "relu" : "plain";
            Assert.True(bn.TryBatchNormTrainingWithVariance(bx, by, bg, bb, bmean, bvar, batch, channels, spatial, Epsilon,
                relu, withResidual ? bres : null), $"the forward refused {label}");
            backend.Synchronize();
            var y = backend.DownloadBuffer(by);
            Compare($"fwd {form} mean {label}", refMean, backend.DownloadBuffer(bmean));
            Compare($"fwd {form} var {label}", refVar, backend.DownloadBuffer(bvar));
            Compare($"fwd {form} y {label}", ForwardReference(x, withResidual ? residual : null, relu, gamma, beta, refMean, refVar,
                batch, channels, spatial), y);
            if (!relu)
            {
                RunBackward("bwd plain", reluOutput: null, residualGrad: false, rebuild: false, second: false, mask: null);
                continue;
            }
            var mask = new bool[n];
            for (int i = 0; i < n; i++) mask[i] = y[i] > 0f;
            if (withResidual)
            {
                RunBackward("bwd add+relu", by, residualGrad: true, rebuild: false, second: false, mask);
                RunBackward("bwd add+relu +dy2", by, residualGrad: true, rebuild: false, second: true, mask);
            }
            else
            {
                RunBackward("bwd relu read", by, residualGrad: false, rebuild: false, second: false, mask);
                RunBackward("bwd relu rebuilt", by, residualGrad: false, rebuild: true, second: false, mask);
                RunBackward("bwd relu rebuilt +dy2", by, residualGrad: false, rebuild: true, second: true, mask);
            }
        }

        void RunBackward(string form, IGpuBuffer? reluOutput, bool residualGrad, bool rebuild, bool second, bool[]? mask)
        {
            backend.Fill(bdx, float.NaN, n);
            backend.Fill(bdg, float.NaN, channels);
            backend.Fill(bdb, float.NaN, channels);
            if (residualGrad) backend.Fill(bdr, float.NaN, n);
            Assert.True(bn.TryBatchNormBackwardFromVariance(bdy, bx, bg, bmean, bvar, bdx, bdg, bdb, batch, channels, spatial,
                Epsilon, reluOutput, residualGrad ? bdr : null, rebuild ? bb : null, second ? bdy2 : null),
                $"the backward refused {form} {label}");
            backend.Synchronize();
            var (dx, dgamma, dbeta, dres) = BackwardReference(x, dy, second ? dy2 : null, mask, gamma, refMean, refVar,
                batch, channels, spatial);
            Compare($"{form} dX {label}", dx, backend.DownloadBuffer(bdx));
            Compare($"{form} dGamma {label}", dgamma, backend.DownloadBuffer(bdg));
            Compare($"{form} dBeta {label}", dbeta, backend.DownloadBuffer(bdb));
            if (residualGrad) Compare($"{form} dResidual {label}", dres, backend.DownloadBuffer(bdr));
        }
    }

    private static (double[] Mean, double[] Var) Statistics(float[] x, int batch, int channels, int spatial)
    {
        var mean = new double[channels];
        var variance = new double[channels];
        int count = batch * spatial;
        for (int c = 0; c < channels; c++)
        {
            double s = 0;
            for (int b = 0; b < batch; b++)
                for (int i = 0; i < spatial; i++) s += x[(b * channels + c) * spatial + i];
            double m = s / count, q = 0;
            for (int b = 0; b < batch; b++)
                for (int i = 0; i < spatial; i++) { double d = x[(b * channels + c) * spatial + i] - m; q += d * d; }
            mean[c] = m;
            variance[c] = q / count;
        }
        return (mean, variance);
    }

    private static double[] ForwardReference(float[] x, float[]? residual, bool relu, float[] gamma, float[] beta, double[] mean,
        double[] variance, int batch, int channels, int spatial)
    {
        var y = new double[x.Length];
        for (int b = 0; b < batch; b++)
            for (int c = 0; c < channels; c++)
            {
                double inv = 1.0 / Math.Sqrt(variance[c] + Epsilon);
                for (int i = 0; i < spatial; i++)
                {
                    int idx = (b * channels + c) * spatial + i;
                    double v = gamma[c] * (x[idx] - mean[c]) * inv + beta[c];
                    if (residual is not null) v += residual[idx];
                    if (relu && v < 0) v = 0;
                    y[idx] = v;
                }
            }
        return y;
    }

    private static (double[] Dx, double[] DGamma, double[] DBeta, double[] DRes) BackwardReference(float[] x, float[] dy,
        float[]? dy2, bool[]? mask, float[] gamma, double[] mean, double[] variance, int batch, int channels, int spatial)
    {
        int n = x.Length, count = batch * spatial;
        var d = new double[n];
        for (int i = 0; i < n; i++)
        {
            double v = dy2 is null ? dy[i] : (double)(dy[i] + dy2[i]);   // the kernels sum the two in float
            d[i] = mask is null || mask[i] ? v : 0.0;
        }
        var dx = new double[n];
        var dgamma = new double[channels];
        var dbeta = new double[channels];
        for (int c = 0; c < channels; c++)
        {
            double inv = 1.0 / Math.Sqrt(variance[c] + Epsilon), sdy = 0, sdyx = 0;
            for (int b = 0; b < batch; b++)
                for (int i = 0; i < spatial; i++)
                {
                    int idx = (b * channels + c) * spatial + i;
                    sdy += d[idx];
                    sdyx += d[idx] * (x[idx] - mean[c]) * inv;
                }
            dgamma[c] = sdyx;
            dbeta[c] = sdy;
            for (int b = 0; b < batch; b++)
                for (int i = 0; i < spatial; i++)
                {
                    int idx = (b * channels + c) * spatial + i;
                    double xhat = (x[idx] - mean[c]) * inv;
                    dx[idx] = inv * (d[idx] * gamma[c] - (gamma[c] * sdy + xhat * gamma[c] * sdyx) / count);
                }
        }
        return (dx, dgamma, dbeta, d);
    }

    // fp32 against a double reference: error relative to the largest magnitude of the result.
    private const double Tolerance = 2e-4;

    private void Compare(string what, double[] expected, float[] actual)
    {
        Assert.True(actual.Length >= expected.Length, $"{what}: {actual.Length} < {expected.Length}");
        double scale = 1e-12, maxErr = 0;
        foreach (var e in expected) scale = Math.Max(scale, Math.Abs(e));
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.False(float.IsNaN(actual[i]) || float.IsInfinity(actual[i]), $"{what}: element {i} is {actual[i]}");
            maxErr = Math.Max(maxErr, Math.Abs(expected[i] - actual[i]) / scale);
        }
        _output.WriteLine($"[{what}] max err / max|ref| {maxErr:E2}");
        Assert.True(maxErr < Tolerance, $"{what}: max err / max|ref| {maxErr:E2}");
    }

    private static float[] Random(int n, int seed, double scale, double offset)
    {
        var rng = new Random(seed);
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)(offset + (rng.NextDouble() * 2 - 1) * scale);
        return a;
    }
}
