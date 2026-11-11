using System;
using System.Linq;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.Metal;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;
using AiDotNet.Tensors.Engines.DirectGpu.Vulkan;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// The jagged-rows grouped GEMM (<see cref="IGroupedGemmBackend"/>) on every backend against a double-precision reference:
/// the forward (b plain and transposed; no bias, an [N] bias, a [G, N] bias) and the per-group weight gradient. The
/// shapes put group boundaries inside 64-row tiles, leave the first, a middle and the last group empty, leave rows past
/// the last offset, and use K and N that are not multiples of the 16-deep / 64-wide tiles. One case has more tiles than a
/// single 65535-group dispatch holds, so the portable kernels' split launch is exercised.
/// </summary>
[Collection("DirectGpuSerial")]
public class GroupedGemmPortTests
{
    // m, k, n, ends: boundaries mid-tile, empty groups (first, middle, last), rows past the last offset.
    public static readonly TheoryData<int, int, int, int[]> Shapes = new()
    {
        { 7, 5, 3, new[] { 2, 7 } },
        { 150, 37, 70, new[] { 0, 30, 30, 95, 101, 140, 140 } },
        { 300, 64, 128, new[] { 64, 128, 200, 250 } },
        { 513, 17, 65, new[] { 1, 2, 3, 400, 500 } },
    };

    [SkippableTheory]
    [MemberData(nameof(Shapes))]
    public void Cuda_grouped_gemm_matches_the_reference(int m, int k, int n, int[] ends)
    {
        Skip.IfNot(CudaBackend.IsCudaAvailable, "CUDA is not available.");
        using var backend = new CudaBackend();
        Skip.IfNot(backend.IsAvailable, "CUDA backend did not initialize.");
        Check(backend, m, k, n, ends);
    }

    [SkippableTheory]
    [MemberData(nameof(Shapes))]
    public void OpenCl_grouped_gemm_matches_the_reference(int m, int k, int n, int[] ends)
    {
        Skip.IfNot(DirectOpenClContext.IsAvailable && DirectOpenClContext.GetDeviceCount() > 0, "No OpenCL device.");
        using var backend = new OpenClBackend(deviceIndex: 0);
        Skip.IfNot(backend.IsAvailable, "OpenCL backend did not initialize.");
        Check(backend, m, k, n, ends);
    }

    [SkippableTheory]
    [MemberData(nameof(Shapes))]
    public void Vulkan_grouped_gemm_matches_the_reference(int m, int k, int n, int[] ends)
    {
        var backend = VulkanBackend.Instance;
        Skip.IfNot(backend.Initialize() && backend.IsGlslCompilerAvailable, "Vulkan with a GLSL compiler is not available.");
        Check(backend, m, k, n, ends);
    }

    [SkippableTheory]
    [MemberData(nameof(Shapes))]
    public void Metal_grouped_gemm_matches_the_reference(int m, int k, int n, int[] ends)
    {
        Skip.IfNot(System.Runtime.InteropServices.RuntimeInformation.IsOSPlatform(System.Runtime.InteropServices.OSPlatform.OSX),
            "Metal is only available on macOS.");
        using var backend = new MetalBackend();
        Skip.IfNot(backend.IsAvailable, "Metal is not available.");
        Check(backend, m, k, n, ends);
    }

#if NET7_0_OR_GREATER
    [SkippableTheory]
    [MemberData(nameof(Shapes))]
    public void WebGpu_grouped_gemm_matches_the_reference(int m, int k, int n, int[] ends)
    {
        var backend = DawnHostedWebGpu.BackendOrSkip();
        Check(backend, m, k, n, ends);
    }
#endif

    // HipBackend runs ViewGroupedGemm (rocBLAS per group), but the template's HIP dialect must still compile as C++.
    [SkippableFact]
    public void Hip_module_compiles_as_cuda_cxx()
    {
        Skip.IfNot(NvrtcNativeBindings.IsAvailable, "NVRTC is not available.");
        string source = GroupedGemmKernelTemplate.GetModuleSource(WinogradKernelDialect.Hip);
        IntPtr program = IntPtr.Zero;
        Assert.Equal(NvrtcResult.Success, NvrtcNativeBindings.nvrtcCreateProgram(ref program, source, "grouped_hip",
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
        foreach (var spec in GroupedGemmKernelTemplate.Kernels)
            Assert.True(spec.SharedFloats * sizeof(float) <= 16384, $"{spec.Name} declares {spec.SharedFloats} floats");
    }

    /// <summary>70,000 groups of a 1 x 1 weight gradient: more tiles than one dispatch's 65535 groups.</summary>
    [SkippableFact]
    public void OpenCl_weight_gradient_splits_a_grid_past_one_dispatch()
    {
        Skip.IfNot(DirectOpenClContext.IsAvailable && DirectOpenClContext.GetDeviceCount() > 0, "No OpenCL device.");
        using var backend = new OpenClBackend(deviceIndex: 0);
        Skip.IfNot(backend.IsAvailable, "OpenCL backend did not initialize.");
        CheckManyGroups(backend);
    }

    [SkippableFact]
    public void Vulkan_weight_gradient_splits_a_grid_past_one_dispatch()
    {
        var backend = VulkanBackend.Instance;
        Skip.IfNot(backend.Initialize() && backend.IsGlslCompilerAvailable, "Vulkan with a GLSL compiler is not available.");
        CheckManyGroups(backend);
    }

    [SkippableFact]
    public void Vulkan_grouped_gemm_inside_active_batch_preserves_prior_dispatches()
    {
        var backend = VulkanBackend.Instance;
        Skip.IfNot(backend.Initialize() && backend.IsGlslCompilerAvailable, "Vulkan with a GLSL compiler is not available.");
        var gg = Assert.IsAssignableFrom<IGroupedGemmBackend>(backend);

        const int m = 16, k = 8, n = 6;
        var ends = new[] { 5, 12, 16 };
        var a = Random(m * k, 41);
        var b = Random(ends.Length * k * n, 42);
        var c = Random(m * n, 43);
        var preLeft = new float[] { 10f, 20f, 30f, 40f };
        var preRight = new float[] { 1f, 2f, 3f, 4f };

        using var ba = backend.AllocateBuffer(a);
        using var bb = backend.AllocateBuffer(b);
        using var bc = backend.AllocateBuffer(c);
        using var by = backend.AllocateBuffer(m * n);
        using var bw = backend.AllocateBuffer(ends.Length * k * n);
        using var bPreL = backend.AllocateBuffer(preLeft);
        using var bPreR = backend.AllocateBuffer(preRight);
        using var bPreOut = backend.AllocateBuffer(preLeft.Length);
        backend.Fill(bPreOut, -999f, preLeft.Length);
        backend.Fill(by, float.NaN, m * n);
        backend.Fill(bw, float.NaN, ends.Length * k * n);

        backend.BeginBatch();
        try
        {
            // Earlier command recorded into the active batch must NOT be discarded by grouped GEMM's offsets upload.
            backend.Add(bPreL, bPreR, bPreOut, preLeft.Length);
            Assert.True(gg.TryGroupedRowsGemm(ba, bb, by, ends, m, n, k, false, null, false));
            Assert.True(gg.TryGroupedRowsWeightGrad(ba, bc, bw, ends, m, k, n));
        }
        finally
        {
            backend.EndBatch();
        }

        Assert.Equal(new float[] { 11f, 22f, 33f, 44f }, backend.DownloadBuffer(bPreOut));
        Compare("vulkan batch fwd", ForwardReference(a, b, ends, m, k, n, false, null, false), backend.DownloadBuffer(by), 1e-4);
        Compare("vulkan batch wgrad", WeightGradReference(a, c, ends, m, k, n), backend.DownloadBuffer(bw), 1e-4);
    }

    private static void CheckManyGroups(IDirectGpuBackend backend)
    {
        var gg = Assert.IsAssignableFrom<IGroupedGemmBackend>(backend);
        const int groups = 70_000, m = 9;
        var ends = new int[groups];
        for (int i = 0; i < groups; i++) ends[i] = i < 3 ? (i + 1) * 3 : m;   // rows 0..8 in groups 0, 1, 2; the rest empty
        var a = Random(m, 1); var c = Random(m, 2);
        using var ba = backend.AllocateBuffer(a);
        using var bc = backend.AllocateBuffer(c);
        using var bw = backend.AllocateBuffer(groups);
        backend.Fill(bw, float.NaN, groups);   // every element must be written
        Assert.True(gg.TryGroupedRowsWeightGrad(ba, bc, bw, ends, m, 1, 1), "the weight gradient refused the shape");
        backend.Synchronize();
        var w = backend.DownloadBuffer(bw);
        var expected = WeightGradReference(a, c, ends, m, 1, 1);
        Compare("70000-group wgrad", expected, w, 1e-5);
    }

    private static void Check(IDirectGpuBackend backend, int m, int k, int n, int[] ends)
    {
        var gg = Assert.IsAssignableFrom<IGroupedGemmBackend>(backend);
        int g = ends.Length;
        string label = $"M{m} K{k} N{n} G{g}";
        var a = Random(m * k, 1);
        var biasN = Random(n, 4);
        var biasGn = Random(g * n, 5);
        var c = Random(m * n, 6);
        using var ba = backend.AllocateBuffer(a);
        using var bbN = backend.AllocateBuffer(biasN);
        using var bbGn = backend.AllocateBuffer(biasGn);
        using var bc = backend.AllocateBuffer(c);
        using var by = backend.AllocateBuffer(m * n);
        foreach (bool transposeB in new[] { false, true })
        {
            var b = Random(g * k * n, transposeB ? 3 : 2);
            using var bb = backend.AllocateBuffer(b);
            foreach (int biasMode in new[] { 0, 1, 2 })
            {
                backend.Fill(by, float.NaN, m * n);   // every element must be written, the tail rows included
                var bias = biasMode == 1 ? bbN : biasMode == 2 ? bbGn : null;
                Assert.True(gg.TryGroupedRowsGemm(ba, bb, by, ends, m, n, k, transposeB, bias, biasMode == 2),
                    $"the forward refused {label}");
                backend.Synchronize();
                var expected = ForwardReference(a, b, ends, m, k, n, transposeB,
                    biasMode == 1 ? biasN : biasMode == 2 ? biasGn : null, biasMode == 2);
                Compare($"fwd transB={transposeB} bias={biasMode} {label}", expected, backend.DownloadBuffer(by), 1e-4);
            }
        }
        using var bw = backend.AllocateBuffer(g * k * n);
        backend.Fill(bw, float.NaN, g * k * n);
        Assert.True(gg.TryGroupedRowsWeightGrad(ba, bc, bw, ends, m, k, n), $"the weight gradient refused {label}");
        backend.Synchronize();
        Compare($"wgrad {label}", WeightGradReference(a, c, ends, m, k, n), backend.DownloadBuffer(bw), 1e-4);
        CheckNaNStaysInItsGroup(backend, gg);
    }

    // A non-finite weight in one group must not reach another group's rows or the unrouted tail (rows share a tile).
    private static void CheckNaNStaysInItsGroup(IDirectGpuBackend backend, IGroupedGemmBackend gg)
    {
        using var ba = backend.AllocateBuffer(new[] { 1f, 1f, 1f });
        using var bb = backend.AllocateBuffer(new[] { float.NaN, 2f });
        using var by = backend.AllocateBuffer(3);
        backend.Fill(by, -7f, 3);
        Assert.True(gg.TryGroupedRowsGemm(ba, bb, by, new[] { 1, 2 }, 3, 1, 1, false, null, false), "the forward refused M3 K1 N1");
        backend.Synchronize();
        var y = backend.DownloadBuffer(by);
        Assert.True(float.IsNaN(y[0]), $"row 0 (group 0) should be NaN, got {y[0]}");
        Assert.Equal(2f, y[1]);
        Assert.Equal(0f, y[2]);
    }

    private static int GroupOf(int[] ends, int row)
    {
        for (int g = 0; g < ends.Length; g++) if (row < ends[g]) return g;
        return -1;
    }

    private static double[] ForwardReference(float[] a, float[] b, int[] ends, int m, int k, int n, bool transposeB,
        float[]? bias, bool biasPerGroup)
    {
        var y = new double[m * n];
        for (int r = 0; r < m; r++)
        {
            int g = GroupOf(ends, r);
            if (g < 0) continue;
            for (int j = 0; j < n; j++)
            {
                double s = 0;
                for (int q = 0; q < k; q++) s += (double)a[r * k + q] * (transposeB ? b[(g * n + j) * k + q] : b[(g * k + q) * n + j]);
                if (bias is not null) s += bias[(biasPerGroup ? g * n : 0) + j];
                y[r * n + j] = s;
            }
        }
        return y;
    }

    private static double[] WeightGradReference(float[] a, float[] c, int[] ends, int m, int k, int n)
    {
        var w = new double[ends.Length * k * n];
        for (int r = 0; r < m; r++)
        {
            int g = GroupOf(ends, r);
            if (g < 0) continue;
            for (int p = 0; p < k; p++)
                for (int j = 0; j < n; j++)
                    w[(g * k + p) * n + j] += (double)a[r * k + p] * c[r * n + j];
        }
        return w;
    }

    private static float[] Random(int count, int seed)
    {
        var rng = new Random(seed);
        return Enumerable.Range(0, count).Select(_ => (float)(rng.NextDouble() * 2 - 1)).ToArray();
    }

    private static void Compare(string what, double[] expected, float[] actual, double tolerance)
    {
        Assert.True(actual.Length >= expected.Length, $"{what}: {actual.Length} values, expected {expected.Length}");
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= tolerance * Math.Max(1, Math.Abs(expected[i])),
                $"{what}[{i}]: expected {expected[i]:R}, got {actual[i]:R}");
    }
}
