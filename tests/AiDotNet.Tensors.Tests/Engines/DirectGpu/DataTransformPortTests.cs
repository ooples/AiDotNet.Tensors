using System;
using System.Linq;
using System.Runtime.InteropServices;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.HIP;
using AiDotNet.Tensors.Engines.DirectGpu.Metal;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;
using AiDotNet.Tensors.Engines.DirectGpu.Vulkan;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// The data-transform kernels (<see cref="IDataTransformBackend"/>, <see cref="DataTransformKernelTemplate"/>) on every
/// backend against the CPU engine's double-precision host code: Box-Muller in both branches with and without the clamp,
/// the exact double-vs-float threshold (ties and near-ties included), the lfilter recursion over more signals than one
/// 256-wide group (FIR, all-pole, zi and the guard), the row gather / scatter with skipped ids, and the integral rounding
/// under every mode at ties, range edges, infinities and NaN. Outputs start as NaN, so every element must be written.
/// </summary>
[Collection("DirectGpuSerial")]
public class DataTransformPortTests
{
    [SkippableFact]
    public void Cuda_data_transforms_match_the_host()
    {
        Skip.IfNot(CudaBackend.IsCudaAvailable, "CUDA is not available.");
        using var backend = new CudaBackend();
        Skip.IfNot(backend.IsAvailable, "CUDA backend did not initialize.");
        CheckAll(backend);
    }

    [SkippableFact]
    public void Hip_data_transforms_match_the_host()
    {
        Skip.IfNot(HipBackend.IsHipAvailable, "HIP is not available.");
        using var backend = new HipBackend();
        Skip.IfNot(backend.IsAvailable, "HIP backend did not initialize.");
        CheckAll(backend);
    }

    [SkippableFact]
    public void OpenCl_data_transforms_match_the_host()
    {
        Skip.IfNot(DirectOpenClContext.IsAvailable && DirectOpenClContext.GetDeviceCount() > 0, "No OpenCL device.");
        using var backend = new OpenClBackend(deviceIndex: 0);
        Skip.IfNot(backend.IsAvailable, "OpenCL backend did not initialize.");
        CheckAll(backend);
    }

    [SkippableFact]
    public void Vulkan_data_transforms_match_the_host()
    {
        var backend = VulkanBackend.Instance;
        Skip.IfNot(backend.Initialize() && backend.IsGlslCompilerAvailable, "Vulkan with a GLSL compiler is not available.");
        CheckAll(backend);
    }

    [SkippableFact]
    public void Metal_data_transforms_match_the_host()
    {
        Skip.IfNot(RuntimeInformation.IsOSPlatform(OSPlatform.OSX), "Metal is only available on macOS.");
        using var backend = new MetalBackend();
        Skip.IfNot(backend.IsAvailable, "Metal is not available.");
        CheckAll(backend);
    }

#if NET7_0_OR_GREATER
    [SkippableFact]
    public void WebGpu_data_transforms_match_the_host()
    {
        var backend = DawnHostedWebGpu.BackendOrSkip();
        CheckAll(backend);
    }
#endif

    /// <summary>Every GLSL kernel compiles under shaderc (the hardware tests skip without a device).</summary>
    [SkippableFact]
    public void Vulkan_data_transform_shaders_compile()
    {
        using var compiler = new VulkanGlslCompiler();
        Skip.IfNot(compiler.IsAvailable, "libshaderc is unavailable.");
        foreach (var name in DataTransformKernelTemplate.GetKernelNames())
            Assert.True(compiler.CompileToSpirv(DataTransformKernelTemplate.GetGlslSource(name)) is not null,
                $"{name}: {compiler.LastError}");
    }

    // The HIP dialect is what CUDA compiles through NVRTC: it must compile as CUDA C++.
    [SkippableFact]
    public void Hip_data_transform_module_compiles_as_cuda_cxx()
    {
        Skip.IfNot(NvrtcNativeBindings.IsAvailable, "NVRTC is not available.");
        string source = DataTransformKernelTemplate.GetModuleSource(WinogradKernelDialect.Hip);
        IntPtr program = IntPtr.Zero;
        Assert.Equal(NvrtcResult.Success, NvrtcNativeBindings.nvrtcCreateProgram(ref program, source, "data_transform",
            0, IntPtr.Zero, IntPtr.Zero));
        try
        {
            var result = NvrtcNativeBindings.nvrtcCompileProgram(program, 1, new[] { "--gpu-architecture=compute_75" });
            Assert.True(result == NvrtcResult.Success, $"the HIP dialect did not compile ({result})");
        }
        finally
        {
            NvrtcNativeBindings.nvrtcDestroyProgram(ref program);
        }
    }

    /// <summary>Every kernel's source comes from its embedded resource and names its entry point, in every dialect.</summary>
    [Fact]
    public void Every_dialect_carries_every_kernel()
    {
        foreach (var dialect in new[] { WinogradKernelDialect.OpenCl, WinogradKernelDialect.Hip, WinogradKernelDialect.Metal })
        {
            string module = DataTransformKernelTemplate.GetModuleSource(dialect);
            foreach (var name in DataTransformKernelTemplate.GetKernelNames()) Assert.Contains(name, module);
        }
        foreach (var name in DataTransformKernelTemplate.GetKernelNames())
        {
            Assert.Contains("void main", DataTransformKernelTemplate.GetGlslSource(name));
            Assert.Contains("fn " + name, AiDotNet.Tensors.Engines.DirectGpu.WebGpu.WebGpuDataTransformKernels.SourceFor(name));
        }
    }

    private static void CheckAll(IDirectGpuBackend backend)
    {
        var dt = Assert.IsAssignableFrom<IDataTransformBackend>(backend);
        foreach (var branch in new[] { BoxMullerBranch.Cosine, BoxMullerBranch.Sine })
        {
            CheckBoxMuller(backend, dt, 3001, branch, clamp: false);
            CheckBoxMuller(backend, dt, 777, branch, clamp: true);
        }
        CheckThreshold(backend, dt, above: false, perElement: false);
        CheckThreshold(backend, dt, above: true, perElement: false);
        CheckThreshold(backend, dt, above: false, perElement: true);
        CheckThreshold(backend, dt, above: true, perElement: true);
        CheckLinearFilter(backend, dt, 300, 40, 3, 4, withZi: true, guard: false);
        CheckLinearFilter(backend, dt, 5, 17, 1, 3, withZi: false, guard: false);   // all-pole
        CheckLinearFilter(backend, dt, 3, 9, 4, 1, withZi: true, guard: false);     // FIR
        CheckLinearFilter(backend, dt, 2, 12, 1, 1, withZi: false, guard: false);   // a scaling, no state
        CheckLinearFilter(backend, dt, 4, 30, 1, 2, withZi: false, guard: true);    // a runaway clamped by the guard
        CheckRows(backend, dt);
        CheckRound(backend, dt);
    }

    private static IGpuBuffer Output(IDirectGpuBackend backend, int count)
    {
        var b = backend.AllocateBuffer(count);
        backend.Fill(b, float.NaN, count);
        return b;
    }

    private static float[] Download(IDirectGpuBackend backend, IGpuBuffer buffer, int count)
    {
        backend.Synchronize();
        return backend.DownloadBuffer(buffer).Take(count).ToArray();
    }

    private static void CheckBoxMuller(IDirectGpuBackend backend, IDataTransformBackend dt, int n, BoxMullerBranch branch, bool clamp)
    {
        var rng = RandomHelper.CreateSeededRandom(51 + n);
        var uv = new float[2 * n];
        for (int i = 0; i < uv.Length; i++) uv[i] = (float)(1.0 - rng.NextDouble());   // (0, 1]
        uv[0] = 1f; uv[2] = 1e-30f; uv[3] = 0.25f;   // r = 0, a large r (normal u1), an exact quarter turn
        const float mean = 0.5f, std = 1.75f, low = -1.25f, high = 2.0f;
        using var bu = backend.AllocateBuffer(uv);
        using var bo = Output(backend, n);
        Assert.True(dt.TryBoxMuller(bu, bo, n, branch, mean, std, clamp, low, high), $"box-muller {branch} refused");
        var actual = Download(backend, bo, n);
        for (int i = 0; i < n; i++)
        {
            double z = CpuEngine.BoxMullerNormal(uv[2 * i], uv[2 * i + 1], branch);
            double e = mean + std * z;
            if (clamp) e = CpuEngine.ClampNormal(e, low, high);
            // Float log / sqrt / sin / cos of float arguments: a few ulps of r and of the angle 2 pi u2.
            Assert.True(Math.Abs(e - actual[i]) <= 5e-5 * Math.Max(1.0, Math.Abs(e)),
                $"box-muller {branch} clamp={clamp} [{i}] (u1 {uv[2 * i]:R}, u2 {uv[2 * i + 1]:R}): expected {e:R}, got {actual[i]:R}");
        }
    }

    // Each double draw u travels as a = (float)u and r = (float)(u - a); the kernel must decide u < t (or u > t) for the
    // float t exactly as the double comparison does, so draws equal to t and one double ulp either side are included.
    private static void CheckThreshold(IDirectGpuBackend backend, IDataTransformBackend dt, bool above, bool perElement)
    {
        const int n = 2048;
        var rng = RandomHelper.CreateSeededRandom(above ? 61 : 62);
        var thresholds = new float[n];
        var draws = new double[n];
        for (int i = 0; i < n; i++)
        {
            float t = perElement ? (float)rng.NextDouble() : 0.3f;
            thresholds[i] = t;
            draws[i] = (i % 4) switch
            {
                0 => t,
                1 => BitConverter.Int64BitsToDouble(BitConverter.DoubleToInt64Bits(t) + 1),
                2 => BitConverter.Int64BitsToDouble(BitConverter.DoubleToInt64Bits(t) - 1),
                _ => rng.NextDouble(),
            };
        }
        var uv = new float[2 * n];
        for (int i = 0; i < n; i++)
        {
            float a = (float)draws[i];
            uv[2 * i] = a;
            uv[2 * i + 1] = (float)(draws[i] - a);
        }
        using var bu = backend.AllocateBuffer(uv);
        using var bt = backend.AllocateBuffer(thresholds);
        using var bo = Output(backend, n);
        Assert.True(dt.TryUniformThreshold(bu, perElement ? bt : null, bo, n, above, 0.3f, 7f, -2f), "threshold refused");
        var actual = Download(backend, bo, n);
        for (int i = 0; i < n; i++)
        {
            bool hit = above ? draws[i] > thresholds[i] : draws[i] < thresholds[i];
            Assert.True((hit ? 7f : -2f) == actual[i],
                $"threshold above={above} per={perElement} [{i}] u {draws[i]:R} t {thresholds[i]:R}: got {actual[i]:R}");
        }
    }

    private static void CheckLinearFilter(IDirectGpuBackend backend, IDataTransformBackend dt, int signals, int length, int nb,
        int na, bool withZi, bool guard)
    {
        int order = Math.Max(nb, na);
        var rng = RandomHelper.CreateSeededRandom(71 + signals + nb * 10 + na);
        float[] Rand(int count, double scale) =>
            Enumerable.Range(0, count).Select(_ => (float)(scale * (rng.NextDouble() * 2 - 1))).ToArray();
        var x = Rand(signals * length, 2);
        var b = Rand(nb, 1);
        var a = Rand(na, 0.3);
        a[0] = 1.5f;
        if (guard) { a = new[] { 1f, -2f }; b = new[] { 1f }; }   // y[n] = x[n] + 2 y[n-1]: runs away without the guard
        if (guard) x[3] = float.NaN;
        var zi = withZi ? Rand(signals * (order - 1), 0.5) : null;
        using var bx = backend.AllocateBuffer(x);
        using var bb = backend.AllocateBuffer(b);
        using var ba = backend.AllocateBuffer(a);
        using var bz = backend.AllocateBuffer(zi ?? new float[1]);
        using var work = Output(backend, Math.Max(1, signals * (order - 1)));
        using var by = Output(backend, signals * length);
        const float maxAbs = 3f;
        Assert.True(dt.TryLinearFilter(bx, bb, ba, withZi ? bz : null, by, work, signals, length, nb, na, guard, maxAbs),
            $"lfilter S{signals} NB{nb} NA{na} refused");
        var actual = Download(backend, by, signals * length);
        double[] D(float[] v) => v.Select(f => (double)f).ToArray();
        var expected = CpuEngine.LinearFilterHost(D(b), D(a), D(x), zi is null ? null : D(zi), signals, length, order,
            guard ? maxAbs : double.PositiveInfinity);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= 1e-4 * Math.Max(1.0, Math.Abs(expected[i])),
                $"lfilter S{signals} L{length} NB{nb} NA{na} zi={withZi} guard={guard} [{i}]: expected {expected[i]:R}, got {actual[i]:R}");
    }

    private static void CheckRows(IDirectGpuBackend backend, IDataTransformBackend dt)
    {
        const int vocab = 50, d = 37, k = 300;
        var rng = RandomHelper.CreateSeededRandom(81);
        var table = Enumerable.Range(0, vocab * d).Select(i => (float)(i * 0.25 - 100)).ToArray();
        var ids = Enumerable.Range(0, k).Select(i => i % 11 == 3 ? -1f : rng.Next(vocab)).ToArray();
        using var bt = backend.AllocateBuffer(table);
        using var bi = backend.AllocateBuffer(ids);
        using var bo = Output(backend, k * d);
        Assert.True(dt.TryGatherRows(bt, bi, bo, k, d), "gather refused");
        var gathered = Download(backend, bo, k * d);
        for (int r = 0; r < k; r++)
            for (int c = 0; c < d; c++)
                Assert.Equal(ids[r] < 0 ? 0f : table[(int)ids[r] * d + c], gathered[r * d + c]);

        // Scatter: distinct ids (a permutation of a subset), some skipped.
        var perm = Enumerable.Range(0, vocab).OrderBy(_ => rng.Next()).ToArray();
        var sids = Enumerable.Range(0, 30).Select(i => i % 7 == 6 ? -1f : perm[i]).ToArray();
        var source = Enumerable.Range(0, 30 * d).Select(i => (float)(1000 + i)).ToArray();
        using var bs = backend.AllocateBuffer(source);
        using var bsi = backend.AllocateBuffer(sids);
        Assert.True(dt.TryScatterRows(bs, bsi, bt, 30, d), "scatter refused");
        var scattered = Download(backend, bt, vocab * d);
        var expected = (float[])table.Clone();
        for (int r = 0; r < 30; r++)
            if (sids[r] >= 0)
                Array.Copy(source, r * d, expected, (int)sids[r] * d, d);
        Assert.Equal(expected, scattered);
    }

    private static void CheckRound(IDirectGpuBackend backend, IDataTransformBackend dt)
    {
        var values = new[]
        {
            0.5f, -0.5f, 1.5f, -1.5f, 2.5f, -2.5f, 0.49999997f, -0.49999997f, 8388607.5f, -8388607.5f, 16777216f, 1e10f,
            -1e10f, 127.5f, -128.5f, 255.5f, float.PositiveInfinity, float.NegativeInfinity, float.NaN, -0f, 3.75f, -3.25f,
        }.Concat(Enumerable.Range(0, 1000).Select(i => (float)Math.Sin(i) * 300f)).ToArray();
        int n = values.Length;
        using var bx = backend.AllocateBuffer(values);
        foreach (RoundingMode mode in Enum.GetValues(typeof(RoundingMode)))
        foreach (var (low, high) in new[] { (-128f, 128f), (-2147483648f, 2147483648f), (0f, 256f) })
        {
            using var by = Output(backend, n);
            Assert.True(dt.TryRoundIntegral(bx, by, n, mode, low, high), $"round {mode} refused");
            var actual = Download(backend, by, n);
            for (int i = 0; i < n; i++)
            {
                double e = float.IsNaN(values[i]) ? 0.0 : Math.Max(low, Math.Min(high, CpuEngine.RoundToIntegral(values[i], mode)));
                Assert.True(e == actual[i], $"round {mode} [{low}, {high}] [{i}] {values[i]:R}: expected {e:R}, got {actual[i]:R}");
            }
        }
    }
}
