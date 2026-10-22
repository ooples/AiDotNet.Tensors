using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.Metal;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;
using AiDotNet.Tensors.Engines.DirectGpu.Vulkan;
using AiDotNet.Tensors.Helpers.Autotune.TunedKernels;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// The stride-2 input gradients of <see cref="ConvDgradS2KernelTemplate"/> on each non-CUDA backend
/// (<see cref="PortableConvDgradS2"/>, #1479-#1483) against a float64 reference, over the shapes and split counts
/// <see cref="ConvDgradS2Tests"/> runs on CUDA: the 3 x 3 four-phase kernel, the 1 x 1 kernel and the projection-shortcut
/// pair, each within 1e-5 of sum|terms|, every element written, bit-reproducible across two runs. HIP's module is
/// compiled as CUDA C++ under NVRTC (the dialect has nothing HIP-specific), so it is checked on any CUDA box.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class ConvDgradS2PortTests
{
    private readonly ITestOutputHelper _out;
    public ConvDgradS2PortTests(ITestOutputHelper output) => _out = output;

    // family ("k3", "k1", "pair"), batch, C, K, K2 (pair only), input size.
    private static readonly object[][] BaseShapes =
    {
        new object[] { "k3", 8, 64, 128, 0, 32 },      // ResNet-18 stage entries
        new object[] { "k3", 8, 256, 512, 0, 8 },      // deep: the split choice splits the channels
        new object[] { "k3", 3, 40, 24, 0, 16 },       // a partial 32-channel tile; three eight-channel steps
        new object[] { "k3", 2, 72, 16, 0, 64 },       // output width 32: one row per group
        new object[] { "k3", 2, 12, 20, 0, 8 },        // output width 4: eight rows (two images) per group
        new object[] { "k1", 8, 128, 256, 0, 16 },     // a ResNet-18 shortcut
        new object[] { "k1", 3, 40, 24, 0, 24 },       // output width 12: groups straddle rows
        new object[] { "k1", 2, 12, 20, 0, 8 },        // output width 4; K off the eight-channel step
        new object[] { "k1", 1, 8, 16, 0, 128 },       // output width 64: a group is half a row
        new object[] { "pair", 8, 128, 256, 256, 16 }, // a ResNet-18 downsampling block
        new object[] { "pair", 3, 40, 24, 16, 16 },    // K2 < K: the shortcut's reduction ends first
        new object[] { "pair", 2, 12, 8, 36, 8 },      // K2 > K, off the eight-channel step; output width 4
    };

    public static IEnumerable<object[]> Cases() =>
        BaseShapes.SelectMany(s => new[] { 0, 1, 3 }.Select(split => s.Concat(new object[] { split }).ToArray()));

    [SkippableTheory]
    [MemberData(nameof(Cases))]
    public void OpenCl_matches_the_float64_reference(string family, int batch, int inC, int outC, int k2, int size, int splits)
    {
        Skip.IfNot(DirectOpenClContext.IsAvailable && DirectOpenClContext.GetDeviceCount() > 0, "No OpenCL device.");
        using var backend = new OpenClBackend(deviceIndex: 0);
        Skip.IfNot(backend.IsAvailable, "OpenCL backend did not initialize.");
        Check(backend, family, batch, inC, outC, k2, size, splits, backend.TryConv2DBackwardInputS2K3,
            backend.TryConv2DBackwardInputS2K1, backend.TryConv2DBackwardInputS2K3Pair);
    }

    [SkippableTheory]
    [MemberData(nameof(Cases))]
    public void Vulkan_matches_the_float64_reference(string family, int batch, int inC, int outC, int k2, int size, int splits)
    {
        var backend = VulkanBackend.Instance;
        Skip.IfNot(backend.Initialize() && backend.IsGlslCompilerAvailable, "Vulkan with a GLSL compiler is not available.");
        Check(backend, family, batch, inC, outC, k2, size, splits, backend.TryConv2DBackwardInputS2K3,
            backend.TryConv2DBackwardInputS2K1, backend.TryConv2DBackwardInputS2K3Pair);
    }

    [SkippableTheory]
    [MemberData(nameof(Cases))]
    public void Metal_matches_the_float64_reference(string family, int batch, int inC, int outC, int k2, int size, int splits)
    {
        Skip.IfNot(System.Runtime.InteropServices.RuntimeInformation.IsOSPlatform(System.Runtime.InteropServices.OSPlatform.OSX),
            "Metal is only available on macOS.");
        using var backend = new MetalBackend();
        Skip.IfNot(backend.IsAvailable, "Metal is not available.");
        Check(backend, family, batch, inC, outC, k2, size, splits, backend.TryConv2DBackwardInputS2K3,
            backend.TryConv2DBackwardInputS2K1, backend.TryConv2DBackwardInputS2K3Pair);
    }

#if NET7_0_OR_GREATER
    [SkippableTheory]
    [MemberData(nameof(Cases))]
    public void WebGpu_matches_the_float64_reference(string family, int batch, int inC, int outC, int k2, int size, int splits)
    {
        var backend = DawnHostedWebGpu.BackendOrSkip();
        Check(backend, family, batch, inC, outC, k2, size, splits, backend.TryConv2DBackwardInputS2K3,
            backend.TryConv2DBackwardInputS2K1, backend.TryConv2DBackwardInputS2K3Pair);
    }
#endif

    [SkippableFact]
    public void Hip_module_compiles_as_cuda_cxx()
    {
        Skip.IfNot(AiDotNet.Tensors.Engines.DirectGpu.CUDA.NvrtcNativeBindings.IsAvailable, "NVRTC is not available.");
        string source = ConvDgradS2KernelTemplate.GetModuleSource(WinogradKernelDialect.Hip);
        IntPtr program = IntPtr.Zero;
        Assert.Equal(AiDotNet.Tensors.Engines.DirectGpu.CUDA.NvrtcResult.Success,
            AiDotNet.Tensors.Engines.DirectGpu.CUDA.NvrtcNativeBindings.nvrtcCreateProgram(ref program, source, "s2_hip",
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

    // The shared-memory budget: every kernel fits WebGPU's guaranteed 16 KB of workgroup storage.
    [Fact]
    public void Every_kernel_fits_16_KB_of_shared_memory()
    {
        foreach (var spec in ConvDgradS2KernelTemplate.Kernels)
            Assert.True(spec.SharedFloats * sizeof(float) <= 16384, $"{spec.Name} declares {spec.SharedFloats} floats");
    }

    [Fact]
    public void Shapes_the_kernels_do_not_compute_are_refused()
    {
        static TunedConv2DBackwardArgs A(int inC, int h, int outH, int k, int s, int p) =>
            new(null!, null!, null!, false, 2, inC, h, h, 8, outH, outH, k, k, s, s, p, p, 1, 1);
        Assert.True(PortableConvDgradS2.AdmitsK3(A(16, 32, 16, 3, 2, 1)), "the control shape");
        Assert.False(PortableConvDgradS2.AdmitsK3(A(16, 32, 32, 3, 1, 1)), "stride 1");
        Assert.False(PortableConvDgradS2.AdmitsK3(A(16, 32, 16, 1, 2, 0)), "1x1");
        Assert.False(PortableConvDgradS2.AdmitsK3(A(18, 32, 16, 3, 2, 1)), "channels off a multiple of four");
        Assert.False(PortableConvDgradS2.AdmitsK3(A(16, 31, 16, 3, 2, 1)), "odd input");
        Assert.False(PortableConvDgradS2.AdmitsK3(A(16, 24, 12, 3, 2, 1)), "output width not dividing the group");
        Assert.False(PortableConvDgradS2.AdmitsK3(A(16, 128, 64, 3, 2, 1)), "output width beyond the group");
        Assert.True(PortableConvDgradS2.AdmitsK1(A(16, 24, 12, 1, 2, 0)), "the 1x1 control shape");
        Assert.False(PortableConvDgradS2.AdmitsK1(A(16, 32, 16, 3, 2, 1)), "3x3");
        Assert.False(PortableConvDgradS2.AdmitsK1(A(16, 20, 10, 1, 2, 0)), "output width off a multiple of four");
        Assert.False(PortableConvDgradS2.AdmitsPair(2, 16, 32, 32, 8, 0, 16, 16), "no shortcut channels");
        Assert.False(PortableConvDgradS2.AdmitsPair(2, 16, 24, 24, 8, 8, 12, 12), "pair output width not dividing the group");
    }

    // The tuned-slot shape predicates agree with the argument predicates on every shape the tests run.
    [Fact]
    public void The_slot_shape_predicates_agree_with_the_argument_predicates()
    {
        foreach (var s in BaseShapes)
        {
            string family = (string)s[0];
            int batch = (int)s[1], inC = (int)s[2], outC = (int)s[3], size = (int)s[5], o = size / 2;
            int k = family == "k1" ? 1 : 3, pad = family == "k1" ? 0 : 1;
            var a = new TunedConv2DBackwardArgs(null!, null!, null!, false, batch, inC, size, size, outC, o, o, k, k, 2, 2,
                pad, pad, 1, 1);
            Assert.Equal(PortableConvDgradS2.AdmitsK3(a), PortableConvDgradS2.AdmitsK3Shape(a.Shape()));
            Assert.Equal(PortableConvDgradS2.AdmitsK1(a), PortableConvDgradS2.AdmitsK1Shape(a.Shape()));
        }
    }

    [Theory]
    [InlineData(64, 128, 68, 1)]    // 64 -> 128 at 32x32: enough tiles alone
    [InlineData(16, 512, 68, 4)]    // 256 -> 512 at 8x8: four splits fill the device
    [InlineData(16, 24, 68, 1)]     // three eight-channel steps: never split
    [InlineData(4, 512, 0, 1)]      // no compute-unit count: one split
    public void The_split_choice_fills_the_device(long tiles, int outC, int target, int splits)
        => Assert.Equal(splits, PortableConvDgradS2.ChooseSplits(tiles, outC, target));

    // The wiring: the OpenCL input-gradient slot offers both kernels as candidates, never as its reference.
    [SkippableFact]
    public void The_opencl_input_gradient_slot_registers_both_candidates()
    {
        Skip.IfNot(DirectOpenClContext.IsAvailable && DirectOpenClContext.GetDeviceCount() > 0, "No OpenCL device.");
        using var backend = new OpenClBackend(deviceIndex: 0);
        Skip.IfNot(backend.IsAvailable, "OpenCL backend did not initialize.");
        var ids = backend.Conv2DBackwardDataSlot.CandidateIds;
        foreach (var id in new[] { "opencl.conv2d_dx.s2k3", "opencl.conv2d_dx.s2k1" })
        {
            Assert.Contains(id, ids);
            Assert.NotEqual(id, ids[0]);
        }
    }

    /// <summary>
    /// ResNet-18's downsampling input gradients at batch 8 on each available backend, interleaved round by round (min /
    /// median us per call): the ported kernels against the backend's direct input-gradient route, and the route the tuned
    /// slot settles on. Gated on AIDOTNET_PERF_RUNNER=1.
    /// </summary>
    [SkippableFact]
    public void Bench_PortVsDirect()
    {
        Skip.IfNot(Environment.GetEnvironmentVariable("AIDOTNET_PERF_RUNNER") == "1", "Set AIDOTNET_PERF_RUNNER=1 on a quiet GPU machine.");
        bool any = false;
        if (DirectOpenClContext.IsAvailable && DirectOpenClContext.GetDeviceCount() > 0)
        {
            using var ocl = new OpenClBackend(deviceIndex: 0);
            if (ocl.IsAvailable) { Bench(ocl, ocl.Conv2DBackwardDataSlot, "opencl"); any = true; }
        }
        var vk = VulkanBackend.Instance;
        if (vk.Initialize() && vk.IsGlslCompilerAvailable) { Bench(vk, vk.Conv2DBackwardDataSlot, "vulkan"); any = true; }
#if NET7_0_OR_GREATER
        if (!string.IsNullOrEmpty(Environment.GetEnvironmentVariable("AIDOTNET_WEBGPU_NODE_DIR")))
        {
            var wg = DawnHostedWebGpu.BackendOrSkip();
            Bench(wg, wg.Conv2DBackwardDataSlot, "webgpu");
            any = true;
        }
#endif
        Skip.IfNot(any, "No OpenCL, Vulkan or WebGPU device.");
    }

    private void Bench(IDirectGpuBackend backend, TunedKernelSlot<TunedConv2DBackwardArgs> slot, string prefix)
    {
        const int rounds = 21, burst = 10;
        var rng = new Random(41);
        foreach (var (k, inC, size, outC) in new[] { (3, 64, 32, 128), (3, 128, 16, 256), (3, 256, 8, 512),
                     (1, 64, 32, 128), (1, 128, 16, 256), (1, 256, 8, 512) })
        {
            int o = size / 2, pad = k == 3 ? 1 : 0;
            using var dy = backend.AllocateBuffer(Rand(8 * outC * o * o, rng));
            using var w = backend.AllocateBuffer(Rand(outC * inC * k * k, rng));
            using var dx = backend.AllocateBuffer(8 * inC * size * size);
            var a = new TunedConv2DBackwardArgs(dy, w, dx, false, 8, inC, size, size, outC, o, o, k, k, 2, 2, pad, pad, 1, 1);
            var routes = new[] { slot.Candidate($"{prefix}.conv2d_dx.s2k{k}"), slot.Candidate($"{prefix}.conv2d_dx.direct") };
            Assert.All(routes, r => Assert.NotNull(r));
            var times = routes.Select(_ => new List<double>()).ToArray();
            for (int r = 0; r < rounds; r++)
                for (int i = 0; i < routes.Length; i++)
                {
                    for (int j = 0; j < 2; j++) routes[i]!.Execute(a);
                    backend.Synchronize();
                    var sw = System.Diagnostics.Stopwatch.StartNew();
                    for (int j = 0; j < burst; j++) routes[i]!.Execute(a);
                    backend.Synchronize();
                    times[i].Add(sw.Elapsed.TotalMilliseconds * 1000 / burst);
                }
            string Stat(List<double> v) { var s = v.OrderBy(x => x).ToArray(); return $"{s[0]:F1} / {s[s.Length / 2]:F1}"; }
            string chosen = slot.Resolve(a.Shape(), a)?.Id ?? "(none)";
            _out.WriteLine($"[{prefix} {k}x{k} {inC}->{outC} at {size}x{size}] s2k{k} {Stat(times[0])} us, direct {Stat(times[1])} us; slot: {chosen}");
        }
    }

    internal delegate bool Single(in TunedConv2DBackwardArgs a, int splits);
    internal delegate bool Pair(in TunedConv2DBackwardArgs a, IGpuBuffer dy2, IGpuBuffer w2, int k2, int splits);

    private static float[] Rand(int n, Random rng)
    {
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        return a;
    }

    // The reference of one case, shared by every backend and split count that runs it.
    private static readonly ConcurrentDictionary<string, (float[] gy, float[] wt, float[] gy2, float[] wt2, double[] sum, double[] mag)>
        s_references = new();

    private void Check(IDirectGpuBackend backend, string family, int batch, int inC, int outC, int k2, int size, int splits,
        Single k3, Single k1, Pair pair)
    {
        int o = size / 2;
        var r = s_references.GetOrAdd($"{family} {batch} {inC} {outC} {k2} {size}",
            _ => Reference(family, batch, inC, outC, k2, size));
        using var dy = backend.AllocateBuffer(r.gy);
        using var w = backend.AllocateBuffer(r.wt);
        using var dy2 = backend.AllocateBuffer(r.gy2.Length > 0 ? r.gy2 : new float[1]);
        using var w2 = backend.AllocateBuffer(r.wt2.Length > 0 ? r.wt2 : new float[1]);
        var poison = Enumerable.Repeat(float.NaN, batch * inC * size * size).ToArray();
        using var x1 = backend.AllocateBuffer(poison);
        using var x2 = backend.AllocateBuffer(poison);
        int kk = family == "k1" ? 1 : 3, pad = family == "k1" ? 0 : 1;
        foreach (var x in new[] { x1, x2 })
        {
            var a = new TunedConv2DBackwardArgs(dy, w, x, false, batch, inC, size, size, outC, o, o, kk, kk, 2, 2, pad, pad, 1, 1);
            bool ran = family switch
            {
                "k3" => k3(a, splits),
                "k1" => k1(a, splits),
                _ => pair(a, dy2, w2, k2, splits),
            };
            Assert.True(ran, $"the {family} kernel refused an admitted shape");
        }
        backend.Synchronize();
        float[] got = backend.DownloadBuffer(x1), again = backend.DownloadBuffer(x2);
        double worst = 0;
        for (int i = 0; i < got.Length; i++)
        {
            Assert.False(float.IsNaN(got[i]), $"dX[{i}] was never written");
            Assert.True(TestHelpers.MathCompat.SingleToInt32Bits(got[i]) == TestHelpers.MathCompat.SingleToInt32Bits(again[i]),
                $"dX[{i}] differs between two identical runs");
            double rel = Math.Abs(got[i] - r.sum[i]) / Math.Max(1.0, r.mag[i]);
            worst = Math.Max(worst, rel);
            Assert.True(rel <= 1e-5, $"dX[{i}] = {got[i]} vs {r.sum[i]} (|terms| {r.mag[i]})");
        }
        _out.WriteLine($"{family} s{splits}: max err / sum|terms| = {worst:E2}");
    }

    private static (float[], float[], float[], float[], double[], double[]) Reference(string family, int batch, int inC,
        int outC, int k2, int size)
    {
        int o = size / 2;
        var rng = new Random(7270 + batch * 7 + inC * 3 + outC + k2 + family.Length);
        bool k1 = family == "k1";
        var gy = Rand(batch * outC * o * o, rng);
        var wt = Rand(outC * inC * (k1 ? 1 : 9), rng);
        var gy2 = family == "pair" ? Rand(batch * k2 * o * o, rng) : Array.Empty<float>();
        var wt2 = family == "pair" ? Rand(k2 * inC, rng) : Array.Empty<float>();
        var sum = new double[batch * inC * size * size];
        var mag = new double[sum.Length];
        for (int b = 0; b < batch; b++)
            for (int c = 0; c < inC; c++)
                for (int ih = 0; ih < size; ih++)
                    for (int iw = 0; iw < size; iw++)
                    {
                        double s = 0, m = 0;
                        bool even = (ih & 1) == 0 && (iw & 1) == 0;
                        if (k1)
                        {
                            if (even)
                                for (int k = 0; k < outC; k++)
                                {
                                    double t = (double)gy[((b * outC + k) * o + ih / 2) * o + iw / 2] * wt[k * inC + c];
                                    s += t; m += Math.Abs(t);
                                }
                        }
                        else
                        {
                            for (int kh = 0; kh < 3; kh++)
                            {
                                int th = ih + 1 - kh;
                                if (th < 0 || (th & 1) != 0 || th / 2 >= o) continue;
                                for (int kw = 0; kw < 3; kw++)
                                {
                                    int tw = iw + 1 - kw;
                                    if (tw < 0 || (tw & 1) != 0 || tw / 2 >= o) continue;
                                    for (int k = 0; k < outC; k++)
                                    {
                                        double t = (double)gy[((b * outC + k) * o + th / 2) * o + tw / 2] * wt[((k * inC + c) * 3 + kh) * 3 + kw];
                                        s += t; m += Math.Abs(t);
                                    }
                                }
                            }
                            if (family == "pair" && even)
                                for (int k = 0; k < k2; k++)
                                {
                                    double t = (double)gy2[((b * k2 + k) * o + ih / 2) * o + iw / 2] * wt2[k * inC + c];
                                    s += t; m += Math.Abs(t);
                                }
                        }
                        int idx = ((b * inC + c) * size + ih) * size + iw;
                        sum[idx] = s; mag[idx] = m;
                    }
        return (gy, wt, gy2, wt2, sum, mag);
    }
}
