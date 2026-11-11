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

    [SkippableFact]
    public void Vulkan_full_forward_sorts_16776961_slots_in_bounded_groups()
    {
        var backend = VulkanBackend.Instance;
        Skip.IfNot(backend.Initialize() && backend.IsGlslCompilerAvailable, "Vulkan with a GLSL compiler is not available.");
        CheckLargeSort(backend);
    }

    [SkippableFact]
    public void Cuda_full_forward_sorts_16776961_slots_in_bounded_groups()
    {
        Skip.IfNot(CudaBackend.IsCudaAvailable, "CUDA is not available.");
        using var backend = new CudaBackend();
        Skip.IfNot(backend.IsAvailable, "CUDA backend did not initialize.");
        CheckLargeSort(backend);
    }

    [SkippableFact]
    public void OpenCl_full_forward_sorts_16776961_slots_in_bounded_groups()
    {
        Skip.IfNot(DirectOpenClContext.IsAvailable && DirectOpenClContext.GetDeviceCount() > 0, "No OpenCL device.");
        using var backend = new OpenClBackend(deviceIndex: 0);
        Skip.IfNot(backend.IsAvailable, "OpenCL backend did not initialize.");
        CheckLargeSort(backend);
    }

    /// <summary>
    /// The largest admitted slot count with one expert (N = 16,776,961, D = G = K = F = B = 1): every slot lands on expert
    /// 0, so the sort is the identity, ends [NK] and the load NK. One sort group per expert used to scan all NK slots
    /// here (over a million barriers in a single group); each sort dispatch now covers one bounded slice.
    /// </summary>
    private static void CheckLargeSort(IDirectGpuBackend backend)
    {
        var moe = Assert.IsAssignableFrom<IMoeKernelBackend>(backend);
        Assert.True(moe.MoeKernelsReady, "the fused MoE kernels did not build");
        var capped = new MaxWorkgroupAssertingMoeBackend(moe);
        const int n = 16_776_961;
        var s = new MoeShape(n, 1, 1, 1, 1, 1);
        Assert.True(s.Admitted);

        var xHost = new float[n];
        for (int i = 0; i < n; i++) xHost[i] = (i % 1999) / 1999f - 0.5f;
        var scoresHost = new float[n];
        for (int i = 0; i < n; i++) scoresHost[i] = 0.5f;
        const float w1 = 0.75f, w3 = -1.25f, w2 = 0.5f;
        using var bx = backend.AllocateBuffer(xHost);
        using var bs = backend.AllocateBuffer(scoresHost);
        using var bw1 = backend.AllocateBuffer(new[] { w1 });
        using var bw3 = backend.AllocateBuffer(new[] { w3 });
        using var bw2 = backend.AllocateBuffer(new[] { w2 });
        IGpuBuffer Out(int count)
        {
            var buf = backend.AllocateBuffer(count);
            backend.Fill(buf, float.NaN, count);
            return buf;
        }
        using var ids = Out(n); using var gates = Out(n); using var ssum = Out(n); using var perm = Out(n);
        using var pos = Out(n); using var ends = Out(1); using var loads = Out(1);
        using var h13 = Out(2 * n); using var o = Out(n); using var y = Out(n);

        PortableMoe.Forward(capped, s, bx, bs, null, bw1, bw3, bw2, ids, gates, ssum, perm, pos, ends, loads, h13, o,
            y);
        backend.Synchronize();

        Assert.Equal((float)n, backend.DownloadBuffer(ends)[0]);
        Assert.Equal((float)n, backend.DownloadBuffer(loads)[0]);
        var permOut = backend.DownloadBuffer(perm);
        var posOut = backend.DownloadBuffer(pos);
        for (int i = 0; i < n; i++)
        {
            Assert.True(permOut[i] == i, $"perm[{i}] = {permOut[i]}, expected {i}");
            Assert.True(posOut[i] == i, $"pos[{i}] = {posOut[i]}, expected {i}");
        }
        var yOut = backend.DownloadBuffer(y);
        foreach (int t in new[] { 0, 4095, 4096, 8_388_607, 16_773_119, 16_773_120, n - 1 })
        {
            double h1 = xHost[t] * (double)w1, h3 = xHost[t] * (double)w3;
            double expected = h1 / (1 + Math.Exp(-h1)) * h3 * w2;
            Assert.True(Math.Abs(expected - yOut[t]) <= 1e-5 * Math.Max(1.0, Math.Abs(expected)),
                $"y[{t}]: expected {expected:R}, got {yOut[t]:R}");
        }
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

        // Multi-slice Sort + full Forward/Backward parity when NK > MaxSortSlotsPerSlice (66,000 > 65,536) and B > 1.
        Check(backend, 33000, 2, 4, 2, 2, 3, 1);

        // Full Forward at N = 65535 * 256 + 1 = 16,776,961 (crosses 65,535 X-workgroups on Route & Combine and 257 Sort slices).
        const int nRoute = 65535 * 256 + 1;
        var largeShape = new MoeShape(nRoute, 1, 2, 1, 1, 1);
        Assert.True(largeShape.Admitted);
        var xLargeHost = new float[nRoute];
        var scoresHost = new float[nRoute * 2];
        for (int i = 0; i < nRoute; i++)
        {
            if ((i & 1) == 0)
            {
                xLargeHost[i] = 2.0f;
                scoresHost[i * 2] = 0.8f;
                scoresHost[i * 2 + 1] = 0.2f;
            }
            else
            {
                xLargeHost[i] = -1.0f;
                scoresHost[i * 2] = 0.2f;
                scoresHost[i * 2 + 1] = 0.8f;
            }
        }
        using var rX = backend.AllocateBuffer(xLargeHost);
        using var rScores = backend.AllocateBuffer(scoresHost);
        using var rW1 = backend.AllocateBuffer(new[] { 1.0f, 1.0f });
        using var rW3 = backend.AllocateBuffer(new[] { 2.0f, 3.0f });
        using var rW2 = backend.AllocateBuffer(new[] { 0.5f, 1.5f });
        using var rIds = backend.AllocateBuffer(nRoute);
        using var rGates = backend.AllocateBuffer(nRoute);
        using var rSsum = backend.AllocateBuffer(nRoute);
        using var rPerm = backend.AllocateBuffer(nRoute);
        using var rPos = backend.AllocateBuffer(nRoute);
        using var rEnds = backend.AllocateBuffer(2);
        using var rLoads = backend.AllocateBuffer(2);
        using var rH13 = backend.AllocateBuffer(nRoute * 2);
        using var rO = backend.AllocateBuffer(nRoute);
        using var rY = backend.AllocateBuffer(nRoute);
        backend.Fill(rY, float.NaN, nRoute);
        backend.Fill(rEnds, float.NaN, 2);
        backend.Fill(rLoads, float.NaN, 2);

        PortableMoe.Forward(capped, largeShape, rX, rScores, null, rW1, rW3, rW2,
            rIds, rGates, rSsum, rPerm, rPos, rEnds, rLoads, rH13, rO, rY);
        backend.Synchronize();

        const int expectedE0 = (nRoute + 1) / 2; // 8,388,481 even tokens
        const int expectedE1 = nRoute / 2;       // 8,388,480 odd tokens
        var endsOut = backend.DownloadBuffer(rEnds);
        var loadsOut = backend.DownloadBuffer(rLoads);
        Assert.Equal((float)expectedE0, endsOut[0]);
        Assert.Equal((float)nRoute, endsOut[1]);
        Assert.Equal((float)expectedE0, loadsOut[0]);
        Assert.Equal((float)expectedE1, loadsOut[1]);

        var permOut = backend.DownloadBuffer(rPerm);
        Assert.Equal(0.0f, permOut[0]);
        Assert.Equal((float)(nRoute - 1), permOut[expectedE0 - 1]);
        Assert.Equal(1.0f, permOut[expectedE0]);
        Assert.Equal((float)(nRoute - 2), permOut[nRoute - 1]);

        float expectedEvenY = (float)((2.0 / (1.0 + Math.Exp(-2.0))) * 4.0 * 0.5);
        float expectedOddY = (float)((-1.0 / (1.0 + Math.Exp(1.0))) * (-3.0) * 1.5);
        var yLargeOut = backend.DownloadBuffer(rY);
        Assert.InRange(Math.Abs(yLargeOut[0] - expectedEvenY), 0f, 1e-4f);
        Assert.InRange(Math.Abs(yLargeOut[1] - expectedOddY), 0f, 1e-4f);
        Assert.InRange(Math.Abs(yLargeOut[nRoute - 2] - expectedOddY), 0f, 1e-4f);
        Assert.InRange(Math.Abs(yLargeOut[nRoute - 1] - expectedEvenY), 0f, 1e-4f);

        // DScores across > 65,535 workgroups: dg[0] = o[0] * dy[t] = 3 * 5 = 15, dot = 15 * 0.6 = 9 -> ds[t, e] = (15 - 9) / 2 = 3.
        using var rDy = backend.AllocateBuffer(nRoute);
        using var rDs = backend.AllocateBuffer(nRoute * 2);
        backend.Fill(rO, 3.0f, nRoute);
        backend.Fill(rDy, 5.0f, nRoute);
        backend.Fill(rGates, 0.6f, nRoute);
        backend.Fill(rSsum, 2.0f, nRoute);
        backend.Fill(rDs, float.NaN, nRoute * 2);

        PortableMoe.LaunchDScores(capped, nRoute, 1, 2, 1, rO, rDy, rPos, rGates, rSsum, rIds, rDs);
        backend.Synchronize();
        var dsOut = backend.DownloadBuffer(rDs);
        Assert.Equal(3.0f, dsOut[(nRoute - 1) * 2]);
        Assert.Equal(0.0f, dsOut[(nRoute - 1) * 2 + 1]);
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
