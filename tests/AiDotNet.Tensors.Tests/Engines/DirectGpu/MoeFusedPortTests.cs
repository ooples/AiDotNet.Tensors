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

    private sealed class MaxWorkgroupAssertingMoeBackend : IMoeKernelBackend
    {
        private readonly IMoeKernelBackend _inner;
        public int MaxObservedGroups { get; private set; }
        public MaxWorkgroupAssertingMoeBackend(IMoeKernelBackend inner) => _inner = inner;
        public bool MoeKernelsReady => _inner.MoeKernelsReady;

        public void LaunchMoeKernel(string kernelName, int groups, IGpuBuffer[] buffers, int[] ints)
        {
            Assert.True(groups > 0 && groups <= 65535, $"{kernelName} launched {groups} X workgroups (> 65535).");
            if (groups > MaxObservedGroups) MaxObservedGroups = groups;
            _inner.LaunchMoeKernel(kernelName, groups, buffers, ints);
        }
    }

    [SkippableFact]
    public void Vulkan_elementwise_moe_splits_launches_past_65535_workgroups()
    {
        var backend = VulkanBackend.Instance;
        Skip.IfNot(backend.Initialize() && backend.IsGlslCompilerAvailable, "Vulkan with a GLSL compiler is not available.");
        CheckLargeDispatchLimit(backend);
    }

    [SkippableFact]
    public void Cuda_elementwise_moe_splits_launches_past_65535_workgroups()
    {
        Skip.IfNot(CudaBackend.IsCudaAvailable, "CUDA is not available.");
        using var backend = new CudaBackend();
        Skip.IfNot(backend.IsAvailable, "CUDA backend did not initialize.");
        CheckLargeDispatchLimit(backend);
    }

    private static void CheckLargeDispatchLimit(IDirectGpuBackend backend)
    {
        var innerMoe = Assert.IsAssignableFrom<IMoeKernelBackend>(backend);
        Assert.True(innerMoe.MoeKernelsReady, "the fused MoE kernels did not build");
        var capped = new MaxWorkgroupAssertingMoeBackend(innerMoe);

        // 4,097 tokens x D = 4,096 = 16,781,312 elements -> 65,552 Combine workgroups (> 65,535).
        const int n = 4097, d = 4096, g = 2, k = 1, f = 1, b = 1;
        var s = new MoeShape(n, d, g, k, f, b);
        Assert.True(s.Admitted);

        var x = Ref.Random<float>(31, -0.5, 0.5, n, d);
        var sc = Ref.Random<float>(32, 0.1, 0.9, n, g);
        var w1 = Ref.Random<float>(33, -0.5, 0.5, g, d, f);
        var w3 = Ref.Random<float>(34, -0.5, 0.5, g, d, f);
        var w2 = Ref.Random<float>(35, -0.5, 0.5, g, f, d);
        var dy = Ref.Random<float>(36, -0.5, 0.5, n, d);

        var cpu = new CpuEngine();
        var ry = cpu.MoeSwiGluHost(x, sc, null, w1, w3, w2, k, b, out _, out var saved);
        var (rdx, rds, _, _, _) = cpu.MoeSwiGluGradients(dy, x, w1, w3, w2, saved);

        using var bx = backend.AllocateBuffer(x.ToArray());
        using var bs = backend.AllocateBuffer(sc.ToArray());
        using var bw1 = backend.AllocateBuffer(w1.ToArray());
        using var bw3 = backend.AllocateBuffer(w3.ToArray());
        using var bw2 = backend.AllocateBuffer(w2.ToArray());
        using var bdy = backend.AllocateBuffer(dy.ToArray());
        using var ids = backend.AllocateBuffer(n * k);
        using var gates = backend.AllocateBuffer(n * k);
        using var ssum = backend.AllocateBuffer(n);
        using var perm = backend.AllocateBuffer(n * k);
        using var pos = backend.AllocateBuffer(n * k);
        using var ends = backend.AllocateBuffer(g);
        using var loads = backend.AllocateBuffer(b * g);
        using var h13 = backend.AllocateBuffer(n * k * 2 * f);
        using var o = backend.AllocateBuffer(n * k * d);
        using var y = backend.AllocateBuffer(n * d);
        using var ds = backend.AllocateBuffer(n * g);
        using var dh13 = backend.AllocateBuffer(n * k * 2 * f);
        using var dxs = backend.AllocateBuffer(n * k * d);
        using var dx = backend.AllocateBuffer(n * d);
        using var dw1 = backend.AllocateBuffer(g * d * f);
        using var dw3 = backend.AllocateBuffer(g * d * f);
        using var dw2 = backend.AllocateBuffer(g * f * d);
        backend.Fill(y, float.NaN, n * d);
        backend.Fill(dx, float.NaN, n * d);

        PortableMoe.Forward(capped, s, bx, bs, null, bw1, bw3, bw2, ids, gates, ssum, perm, pos, ends, loads, h13, o, y);
        PortableMoe.Backward(capped, s, bdy, bx, bw1, bw3, bw2, ids, gates, ssum, perm, pos, ends, h13, o, ds, dh13, dxs, dx,
            dw1, dw3, dw2);
        backend.Synchronize();
        Assert.Equal(65535, capped.MaxObservedGroups);

        // Verify both chunk 0 and tail chunk (> 65535 * 256 = 16,776,960) match host reference.
        var yHost = Ref.D(ry);
        var yDev = backend.DownloadBuffer(y);
        var dxHost = Ref.D(rdx);
        var dxDev = backend.DownloadBuffer(dx);
        foreach (int idx in new[] { 0, 1023, 16_776_959, 16_776_960, 16_776_961, n * d - 1 })
        {
            Assert.True(Math.Abs(yHost[idx] - yDev[idx]) <= 1e-3 * Math.Max(1.0, Math.Abs(yHost[idx])),
                $"y[{idx}]: expected {yHost[idx]:R}, got {yDev[idx]:R}");
            Assert.True(Math.Abs(dxHost[idx] - dxDev[idx]) <= 1e-3 * Math.Max(1.0, Math.Abs(dxHost[idx])),
                $"dx[{idx}]: expected {dxHost[idx]:R}, got {dxDev[idx]:R}");
        }
        Close("dscores large", Ref.D(rds), backend.DownloadBuffer(ds), 1e-3);

        // Also cross 65,535 workgroups on Route and DScores: N = 65535 * 256 + 1 = 16,776,961 (< 2^24).
        const int nRoute = 65535 * 256 + 1;
        var sRoute = new MoeShape(nRoute, 1, 1, 1, 1, 1);
        Assert.True(sRoute.Admitted);
        var ones = new float[nRoute];
        for (int i = 0; i < nRoute; i++) ones[i] = 0.5f;
        using var rX = backend.AllocateBuffer(ones);
        using var rScores = backend.AllocateBuffer(ones);
        using var rIds = backend.AllocateBuffer(nRoute);
        using var rGates = backend.AllocateBuffer(nRoute);
        using var rSsum = backend.AllocateBuffer(nRoute);
        using var rO = backend.AllocateBuffer(nRoute);
        using var rDy = backend.AllocateBuffer(ones);
        using var rPos = backend.AllocateBuffer(nRoute);
        using var rDs = backend.AllocateBuffer(nRoute);
        backend.Fill(rGates, float.NaN, nRoute);
        backend.Fill(rDs, float.NaN, nRoute);
        using var rPerm = backend.AllocateBuffer(nRoute);
        using var rEnds = backend.AllocateBuffer(1);
        using var rLoads = backend.AllocateBuffer(1);
        using var rW1 = backend.AllocateBuffer(new[] { 1f });
        using var rW3 = backend.AllocateBuffer(new[] { 1f });
        using var rW2 = backend.AllocateBuffer(new[] { 1f });
        using var rH13 = backend.AllocateBuffer(2 * nRoute);
        using var rY = backend.AllocateBuffer(nRoute);
        using var rDh13 = backend.AllocateBuffer(2 * nRoute);
        using var rDxs = backend.AllocateBuffer(nRoute);
        using var rDx = backend.AllocateBuffer(nRoute);
        using var rDw1 = backend.AllocateBuffer(1);
        using var rDw3 = backend.AllocateBuffer(1);
        using var rDw2 = backend.AllocateBuffer(1);

        PortableMoe.Forward(capped, sRoute, rX, rScores, null, rW1, rW3, rW2, rIds, rGates, rSsum, rPerm, rPos, rEnds, rLoads, rH13, rO, rY);
        PortableMoe.Backward(capped, sRoute, rDy, rX, rW1, rW3, rW2, rIds, rGates, rSsum, rPerm, rPos, rEnds, rH13, rO, rDs, rDh13, rDxs, rDx,
            rDw1, rDw3, rDw2);
        backend.Synchronize();
        var gatesOut = backend.DownloadBuffer(rGates);
        var dsOut = backend.DownloadBuffer(rDs);
        Assert.Equal(1.0f, gatesOut[nRoute - 1]);
        Assert.Equal(0.0f, dsOut[nRoute - 1]);
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
