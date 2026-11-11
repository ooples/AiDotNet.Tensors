using System;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.Metal;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;
using AiDotNet.Tensors.Engines.DirectGpu.Vulkan;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Ref = AiDotNet.Tensors.Tests.Engines.MixtureOfExpertsTests;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// The fused MoE kernels (<see cref="IMoeKernelBackend"/>, <see cref="PortableMoe"/>) on every backend against the host
/// reference (<see cref="CpuEngine"/> in double): routing (ids, gates), the counting sort (perm, pos, ends), the per-sequence
/// loads, the forward output, and every gradient. Shapes put expert boundaries inside 64-row tiles, leave experts empty
/// (a bias of -100 keeps some unchosen), and use D and F that are not multiples of the 16-deep / 64-wide tiles. Every
/// output buffer starts as NaN, so an element a kernel fails to write fails the comparison.
/// </summary>
[Collection("DirectGpuSerial")]
public class MoeFusedPortTests
{
    // n, d, g, k, f, sequences, emptyExperts
    public static readonly TheoryData<int, int, int, int, int, int, int> Shapes = new()
    {
        { 5, 3, 4, 1, 2, 1, 0 },
        { 37, 20, 7, 3, 13, 1, 2 },
        { 200, 70, 16, 4, 33, 4, 3 },
        { 130, 64, 64, 8, 64, 2, 10 },
    };

    [SkippableTheory]
    [MemberData(nameof(Shapes))]
    public void Cuda_fused_moe_matches_the_reference(int n, int d, int g, int k, int f, int b, int empty)
    {
        Skip.IfNot(CudaBackend.IsCudaAvailable, "CUDA is not available.");
        using var backend = new CudaBackend();
        Skip.IfNot(backend.IsAvailable, "CUDA backend did not initialize.");
        Check(backend, n, d, g, k, f, b, empty);
    }

    [SkippableTheory]
    [MemberData(nameof(Shapes))]
    public void OpenCl_fused_moe_matches_the_reference(int n, int d, int g, int k, int f, int b, int empty)
    {
        Skip.IfNot(DirectOpenClContext.IsAvailable && DirectOpenClContext.GetDeviceCount() > 0, "No OpenCL device.");
        using var backend = new OpenClBackend(deviceIndex: 0);
        Skip.IfNot(backend.IsAvailable, "OpenCL backend did not initialize.");
        Check(backend, n, d, g, k, f, b, empty);
    }

    [SkippableTheory]
    [MemberData(nameof(Shapes))]
    public void Vulkan_fused_moe_matches_the_reference(int n, int d, int g, int k, int f, int b, int empty)
    {
        var backend = VulkanBackend.Instance;
        Skip.IfNot(backend.Initialize() && backend.IsGlslCompilerAvailable, "Vulkan with a GLSL compiler is not available.");
        Check(backend, n, d, g, k, f, b, empty);
    }

    [SkippableTheory]
    [MemberData(nameof(Shapes))]
    public void Metal_fused_moe_matches_the_reference(int n, int d, int g, int k, int f, int b, int empty)
    {
        Skip.IfNot(System.Runtime.InteropServices.RuntimeInformation.IsOSPlatform(System.Runtime.InteropServices.OSPlatform.OSX),
            "Metal is only available on macOS.");
        using var backend = new MetalBackend();
        Skip.IfNot(backend.IsAvailable, "Metal is not available.");
        Check(backend, n, d, g, k, f, b, empty);
    }

#if NET7_0_OR_GREATER
    [SkippableTheory]
    [MemberData(nameof(Shapes))]
    public void WebGpu_fused_moe_matches_the_reference(int n, int d, int g, int k, int f, int b, int empty)
    {
        var backend = DawnHostedWebGpu.BackendOrSkip();
        Check(backend, n, d, g, k, f, b, empty);
    }
#endif

    // HipBackend compiles the template's HIP dialect with hiprtc; it must also be valid CUDA C++ (NVRTC on this box).
    [SkippableFact]
    public void Hip_module_compiles_as_cuda_cxx()
    {
        Skip.IfNot(NvrtcNativeBindings.IsAvailable, "NVRTC is not available.");
        string source = MoeKernelTemplate.GetModuleSource(WinogradKernelDialect.Hip);
        IntPtr program = IntPtr.Zero;
        Assert.Equal(NvrtcResult.Success, NvrtcNativeBindings.nvrtcCreateProgram(ref program, source, "moe_hip",
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

    [Fact]
    public void Every_kernel_fits_16_KB_of_shared_memory()
    {
        foreach (var spec in MoeKernelTemplate.Kernels)
            Assert.True(spec.SharedFloats * sizeof(float) <= 16384, $"{spec.Name} declares {spec.SharedFloats} floats");
    }

    private static void Check(IDirectGpuBackend backend, int n, int d, int g, int k, int f, int b, int empty)
    {
        var moe = Assert.IsAssignableFrom<IMoeKernelBackend>(backend);
        Assert.True(moe.MoeKernelsReady, "the fused MoE kernels did not build");
        string label = $"N{n} D{d} G{g} K{k} F{f} B{b}";
        var s = new MoeShape(n, d, g, k, f, b);
        Assert.True(s.Admitted, label);
        int nk = n * k;

        var x = Ref.Random<float>(1, -1, 1, n, d);
        var sc = Ref.Random<float>(2, 0.05, 0.95, n, g);
        var bias = Ref.Random<float>(3, -0.3, 0.3, g);
        for (int e = 0; e < empty; e++) bias.GetDataArray()[(e * 7 + 1) % g] = -100f;   // never chosen
        var w1 = Ref.Random<float>(4, -0.5, 0.5, g, d, f);
        var w3 = Ref.Random<float>(5, -0.5, 0.5, g, d, f);
        var w2 = Ref.Random<float>(6, -0.5, 0.5, g, f, d);
        var dy = Ref.Random<float>(7, -1, 1, n, d);

        // The host reference, through the internal forward that also returns its routing.
        var cpu = new CpuEngine();
        var ry = cpu.MoeSwiGluHost(x, sc, bias, w1, w3, w2, k, b, out var rLoads, out var saved);
        var (rdx, rds, rdw1, rdw3, rdw2) = cpu.MoeSwiGluGradients(dy, x, w1, w3, w2, saved);

        IGpuBuffer Up(Tensor<float> t) => backend.AllocateBuffer(t.ToArray());
        IGpuBuffer Out(int count)
        {
            var buf = backend.AllocateBuffer(count);
            backend.Fill(buf, float.NaN, count);
            return buf;
        }
        using var bx = Up(x); using var bs = Up(sc); using var bb = Up(bias);
        using var bw1 = Up(w1); using var bw3 = Up(w3); using var bw2 = Up(w2); using var bdy = Up(dy);
        using var ids = Out(nk); using var gates = Out(nk); using var ssum = Out(n); using var perm = Out(nk);
        using var pos = Out(nk); using var ends = Out(g); using var loads = Out(b * g); using var h13 = Out(nk * 2 * f);
        using var o = Out(nk * d); using var y = Out(n * d);
        using var ds = Out(n * g); using var dh13 = Out(nk * 2 * f); using var dxs = Out(nk * d); using var dx = Out(n * d);
        using var dw1 = Out(g * d * f); using var dw3 = Out(g * d * f); using var dw2 = Out(g * f * d);

        PortableMoe.Forward(moe, s, bx, bs, bb, bw1, bw3, bw2, ids, gates, ssum, perm, pos, ends, loads, h13, o, y);
        PortableMoe.Backward(moe, s, bdy, bx, bw1, bw3, bw2, ids, gates, ssum, perm, pos, ends, h13, o, ds, dh13, dxs, dx,
            dw1, dw3, dw2);
        backend.Synchronize();

        Exact($"ids {label}", saved.Ids!.Select(v => (double)v).ToArray(), backend.DownloadBuffer(ids));
        Exact($"perm {label}", saved.Perm!.Select(v => (double)v).ToArray(), backend.DownloadBuffer(perm));
        Exact($"pos {label}", saved.Pos!.Select(v => (double)v).ToArray(), backend.DownloadBuffer(pos));
        var cumEnds = saved.Ends!.Select(v => (double)v).ToArray();
        Exact($"ends {label}", cumEnds, backend.DownloadBuffer(ends));
        Assert.True(empty == 0 || Enumerable.Range(0, g).Any(e => cumEnds[e] == (e == 0 ? 0 : cumEnds[e - 1])),
            $"{label}: no expert was left empty");
        Exact($"loads {label}", Ref.D(rLoads), backend.DownloadBuffer(loads));
        Close($"gates {label}", saved.Gates!, backend.DownloadBuffer(gates), 1e-5);
        Close($"y {label}", Ref.D(ry), backend.DownloadBuffer(y), 1e-4);
        Close($"dscores {label}", Ref.D(rds), backend.DownloadBuffer(ds), 1e-4);
        Close($"dx {label}", Ref.D(rdx), backend.DownloadBuffer(dx), 1e-4);
        Close($"dw1 {label}", Ref.D(rdw1), backend.DownloadBuffer(dw1), 1e-4);
        Close($"dw3 {label}", Ref.D(rdw3), backend.DownloadBuffer(dw3), 1e-4);
        Close($"dw2 {label}", Ref.D(rdw2), backend.DownloadBuffer(dw2), 1e-4);
    }

    private static void Exact(string what, double[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(expected[i] == actual[i], $"{what}[{i}]: expected {expected[i]}, got {actual[i]}");
    }

    private static void Close(string what, double[] expected, float[] actual, double tolerance)
    {
        Assert.Equal(expected.Length, actual.Length);
        double scale = Math.Max(1, expected.Max(Math.Abs));
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= tolerance * scale,
                $"{what}[{i}]: expected {expected[i]:R}, got {actual[i]:R}");
    }
}
