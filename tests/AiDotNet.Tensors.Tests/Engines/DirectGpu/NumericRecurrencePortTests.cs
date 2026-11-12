using System;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.HIP;
using AiDotNet.Tensors.Engines.DirectGpu.Metal;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;
using AiDotNet.Tensors.Engines.DirectGpu.Vulkan;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// The batched numeric recurrences (<see cref="INumericRecurrenceBackend"/>) on every backend against the CPU engine's
/// double-precision host kernels: the tridiagonal solve and the Fritsch-Carlson limiter over more systems than one
/// 256-wide group (one unknown included), the CTC lattice with an empty target, a repeated label, a shorter input, an
/// empty input and an item with no alignment, and the assignment at a size past one group's 256 threads and with NaN
/// costs. Outputs start as NaN, so every element must be written.
/// </summary>
[Collection("DirectGpuSerial")]
public class NumericRecurrencePortTests
{
    [SkippableFact]
    public void Cuda_numeric_recurrences_match_the_host()
    {
        Skip.IfNot(CudaBackend.IsCudaAvailable, "CUDA is not available.");
        using var backend = new CudaBackend();
        Skip.IfNot(backend.IsAvailable, "CUDA backend did not initialize.");
        CheckAll(backend);
    }

    [SkippableFact]
    public void Hip_numeric_recurrences_match_the_host()
    {
        Skip.IfNot(HipBackend.IsHipAvailable, "HIP is not available.");
        using var backend = new HipBackend();
        Skip.IfNot(backend.IsAvailable, "HIP backend did not initialize.");
        CheckAll(backend);
    }

    [SkippableFact]
    public void OpenCl_numeric_recurrences_match_the_host()
    {
        Skip.IfNot(DirectOpenClContext.IsAvailable && DirectOpenClContext.GetDeviceCount() > 0, "No OpenCL device.");
        using var backend = new OpenClBackend(deviceIndex: 0);
        Skip.IfNot(backend.IsAvailable, "OpenCL backend did not initialize.");
        CheckAll(backend);
    }

    [SkippableFact]
    public void Vulkan_numeric_recurrences_match_the_host()
    {
        var backend = VulkanBackend.Instance;
        Skip.IfNot(backend.Initialize() && backend.IsGlslCompilerAvailable, "Vulkan with a GLSL compiler is not available.");
        CheckAll(backend);
    }

    [SkippableFact]
    public void Metal_numeric_recurrences_match_the_host()
    {
        Skip.IfNot(System.Runtime.InteropServices.RuntimeInformation.IsOSPlatform(System.Runtime.InteropServices.OSPlatform.OSX),
            "Metal is only available on macOS.");
        using var backend = new MetalBackend();
        Skip.IfNot(backend.IsAvailable, "Metal is not available.");
        CheckAll(backend);
    }

#if NET7_0_OR_GREATER
    [SkippableFact]
    public void WebGpu_numeric_recurrences_match_the_host()
    {
        var backend = DawnHostedWebGpu.BackendOrSkip();
        CheckAll(backend);
    }
#endif

    // The HIP dialect is what CUDA compiles through NVRTC: it must compile as CUDA C++.
    [SkippableFact]
    public void Hip_module_compiles_as_cuda_cxx()
    {
        Skip.IfNot(NvrtcNativeBindings.IsAvailable, "NVRTC is not available.");
        string source = NumericRecurrenceKernelTemplate.GetModuleSource(WinogradKernelDialect.Hip);
        IntPtr program = IntPtr.Zero;
        Assert.Equal(NvrtcResult.Success, NvrtcNativeBindings.nvrtcCreateProgram(ref program, source, "numeric_recurrence",
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

    // Every kernel fits WebGPU's guaranteed 16 KB of workgroup storage.
    [Fact]
    public void Every_kernel_fits_16_KB_of_shared_memory()
    {
        foreach (var spec in NumericRecurrenceKernelTemplate.Kernels)
            Assert.True(spec.SharedFloats * sizeof(float) <= 16384, $"{spec.Name} declares {spec.SharedFloats} floats");
    }

    private static void CheckAll(IDirectGpuBackend backend)
    {
        var nr = Assert.IsAssignableFrom<INumericRecurrenceBackend>(backend);
        CheckTridiagonal(backend, nr, 300, 17);
        CheckTridiagonal(backend, nr, 3, 1);
        CheckFritschCarlson(backend, nr, 300, 33);
        CheckFritschCarlson(backend, nr, 2, 1);
        CheckCtc(backend, nr);
        CheckLongCtc(backend, nr);
        CheckAssignment(backend, nr, 3, 7, withNaN: false);
        CheckAssignment(backend, nr, 2, 260, withNaN: false);
        CheckAssignment(backend, nr, 2, 9, withNaN: true);
        // Past the 400 rows a shared-memory solver could hold: the state lives in the global workspace.
        CheckAssignment(backend, nr, 1, 450, withNaN: false);
        CheckExtremeAssignment(backend, nr);
    }

    private static float[] Random(int count, int seed, double scale, double shift)
    {
        var rng = RandomHelper.CreateSeededRandom(seed);
        return Enumerable.Range(0, count).Select(_ => (float)(shift + scale * (rng.NextDouble() * 2 - 1))).ToArray();
    }

    private static double[] D(float[] v) => v.Select(x => (double)x).ToArray();

    private static IGpuBuffer Output(IDirectGpuBackend backend, int count)
    {
        var b = backend.AllocateBuffer(count);
        backend.Fill(b, float.NaN, count);
        return b;
    }

    private static void CheckTridiagonal(IDirectGpuBackend backend, INumericRecurrenceBackend nr, int batch, int n)
    {
        int bands = Math.Max(1, batch * (n - 1));
        var lo = Random(bands, 1, 1, 0); var up = Random(bands, 2, 1, 0);
        var dg = Random(batch * n, 3, 1, 4); var rh = Random(batch * n, 4, 1, 0);
        using var bl = backend.AllocateBuffer(lo); using var bu = backend.AllocateBuffer(up);
        using var bd = backend.AllocateBuffer(dg); using var br = backend.AllocateBuffer(rh);
        using var work = backend.AllocateBuffer(bands);
        using var bx = Output(backend, batch * n);
        Assert.True(nr.TryTridiagonalSolve(bl, bd, bu, br, bx, work, batch, n), $"the solve refused B{batch} N{n}");
        backend.Synchronize();
        var expected = CpuEngine.TridiagonalSolveHost(D(lo), D(dg), D(up), D(rh), batch, n);
        Compare($"tridiagonal B{batch} N{n}", expected, backend.DownloadBuffer(bx), 1e-4);
    }

    private static void CheckFritschCarlson(IDirectGpuBackend backend, INumericRecurrenceBackend nr, int batch, int n)
    {
        var m = Random(batch * n, 5, 3, 0);
        var d = Random(Math.Max(1, batch * (n - 1)), 6, 1, 0);
        for (int i = 0; i < d.Length; i += 7) d[i] = 0f;   // flat sections
        using var bm = backend.AllocateBuffer(m); using var bd = backend.AllocateBuffer(d);
        using var bo = Output(backend, batch * n);
        Assert.True(nr.TryFritschCarlsonLimit(bm, bd, bo, batch, n), $"the limiter refused B{batch} N{n}");
        backend.Synchronize();
        var expected = CpuEngine.FritschCarlsonHost(D(m), D(d), batch, n);
        Compare($"fritsch-carlson B{batch} N{n}", expected, backend.DownloadBuffer(bo), 1e-5);
    }

    // Items: a 3-label target with a repeat over 9 steps, an empty target over 5, a 2-label target over 4, an empty
    // input, and [2, 2] over 2 steps (a repeat needs 3: no alignment).
    private static void CheckCtc(IDirectGpuBackend backend, INumericRecurrenceBackend nr) =>
        CheckCtc(backend, nr, 9, 5, new[] { 9, 5, 4, 0, 2 }, new[] { 3, 0, 2, 1, 2 }, new[] { 1, 3, 3, 4, 2, 1, 2, 2 },
            1e-4, 1e-4);

    // A lattice of 1041 positions (520 labels, every fifth a repeat) beside a short item: longer than the 1024 rows a
    // shared-memory kernel could double-buffer, so the rows run through global memory. Over 1100 float steps the
    // posteriors exp(alpha + beta - logP) carry the float rounding of log values near -1000 (about 1e-3 absolute in
    // the exponent), hence their looser tolerance; alpha and beta keep the relative 1e-4.
    private static void CheckLongCtc(IDirectGpuBackend backend, INumericRecurrenceBackend nr)
    {
        var rng = RandomHelper.CreateSeededRandom(11);
        var targets = new int[520 + 3];
        for (int i = 0; i < 520; i++) targets[i] = i % 5 == 4 ? targets[i - 1] : 1 + rng.Next(4);
        targets[520] = 2; targets[521] = 3; targets[522] = 1;
        CheckCtc(backend, nr, 1100, 2, new[] { 1100, 40 }, new[] { 520, 3 }, targets, 1e-4, 2e-2);
    }

    private static void CheckCtc(IDirectGpuBackend backend, INumericRecurrenceBackend nr, int T, int N, int[] inputLengths,
        int[] targetLengths, int[] targets, double latticeTolerance, double posteriorTolerance)
    {
        int C = 5, blank = 0;
        var raw = Random(T * N * C, 7, 2, 0);
        var lp = new float[raw.Length];
        for (int r = 0; r < T * N; r++)
        {
            double max = double.NegativeInfinity, sum = 0;
            for (int c = 0; c < C; c++) max = Math.Max(max, raw[r * C + c]);
            for (int c = 0; c < C; c++) sum += Math.Exp(raw[r * C + c] - max);
            for (int c = 0; c < C; c++) lp[r * C + c] = (float)(raw[r * C + c] - max - Math.Log(sum));
        }
        var cpu = new CpuEngine();
        var lpT = new Tensor<double>(D(lp), new[] { T, N, C });
        var tgT = new Tensor<int>(targets, new[] { targets.Length });
        var (alpha, beta) = cpu.TensorCTCForwardBackward(lpT, tgT, inputLengths, targetLengths, blank);
        var gamma = cpu.TensorCTCLabelPosteriors(lpT, tgT, inputLengths, targetLengths, blank);

        int sm = 2 * targetLengths.Max() + 1, count = N * T * sm;
        var lengths = new float[3 * N];
        int off = 0;
        for (int n = 0; n < N; n++)
        {
            lengths[n] = inputLengths[n]; lengths[N + n] = targetLengths[n]; lengths[2 * N + n] = off;
            off += targetLengths[n];
        }
        using var blp = backend.AllocateBuffer(lp);
        using var btg = backend.AllocateBuffer(targets.Select(v => (float)v).ToArray());
        using var blen = backend.AllocateBuffer(lengths);
        using var bla = Output(backend, count);
        using var blb = Output(backend, count);
        using var blz = Output(backend, N);
        using var bg = Output(backend, T * N * C);
        Assert.True(nr.TryCtcForwardBackward(blp, btg, blen, bla, blb, blz, T, N, C, sm, blank), "the forward-backward refused");
        Assert.True(nr.TryCtcLabelPosteriors(bla, blb, btg, blen, blz, bg, T, N, C, sm, blank), "the posteriors refused");
        backend.Synchronize();
        Compare($"log-alpha T{T}", alpha.ToArray(), backend.DownloadBuffer(bla), latticeTolerance);
        Compare($"log-beta T{T}", beta.ToArray(), backend.DownloadBuffer(blb), latticeTolerance);
        Compare($"gamma T{T}", gamma.ToArray(), backend.DownloadBuffer(bg), posteriorTolerance);
    }

    private static void CheckAssignment(IDirectGpuBackend backend, INumericRecurrenceBackend nr, int batch, int size, bool withNaN)
    {
        var cost = Random(batch * size * size, 8 + size, 5, 0);
        if (withNaN)
        {
            for (int j = 0; j < size; j++) cost[j] = float.NaN;                       // item 0, row 0: forced onto NaN
            for (int i = 0; i < size; i += 2) cost[(size + i) * size + i] = float.NaN;   // item 1: scattered, avoidable
        }
        using var bc = backend.AllocateBuffer(cost);
        using var ba = Output(backend, batch * size);
        using var work = Output(backend, batch * 8 * size);
        Assert.True(nr.TryLinearSumAssignment(bc, ba, work, batch, size), $"the assignment refused B{batch} S{size}");
        backend.Synchronize();
        var expected = CpuEngine.LinearSumAssignmentHost(D(cost), batch, size);
        var actual = backend.DownloadBuffer(ba);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(expected[i] == actual[i], $"assignment B{batch} S{size}[{i}]: expected {expected[i]}, got {actual[i]}");
    }

    private static void CheckExtremeAssignment(IDirectGpuBackend backend, INumericRecurrenceBackend nr)
    {
        float h = float.MaxValue;
        // Finite costs of both signs, and a forced non-finite row in a second batch item.
        var cost = new[] { h, -h, -h, h, float.NaN, float.NaN, h, -h };
        using var bc = backend.AllocateBuffer(cost);
        using var ba = Output(backend, 4);
        using var work = Output(backend, 32);
        Assert.True(nr.TryLinearSumAssignment(bc, ba, work, 2, 2));
        backend.Synchronize();
        Assert.Equal(new[] { 1f, 0f, 0f, 1f }, backend.DownloadBuffer(ba));
    }

    private static void Compare(string what, double[] expected, float[] actual, double tolerance)
    {
        Assert.True(actual.Length >= expected.Length, $"{what}: {actual.Length} values, expected {expected.Length}");
        for (int i = 0; i < expected.Length; i++)
        {
            if (double.IsNegativeInfinity(expected[i]))
                Assert.True(float.IsNegativeInfinity(actual[i]), $"{what}[{i}]: expected -inf, got {actual[i]:R}");
            else if (double.IsNaN(expected[i]))
                Assert.True(float.IsNaN(actual[i]), $"{what}[{i}]: expected NaN, got {actual[i]:R}");
            else
                Assert.True(Math.Abs(expected[i] - actual[i]) <= tolerance * Math.Max(1, Math.Abs(expected[i])),
                    $"{what}[{i}]: expected {expected[i]:R}, got {actual[i]:R}");
        }
    }
}
