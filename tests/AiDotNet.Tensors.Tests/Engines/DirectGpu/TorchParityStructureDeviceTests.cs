using System;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Tests.TestHelpers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// #1112 tier 8: complex / view_as_complex, nonzero_static, unravel_index, unique_dim and grid_sample 3-D run on the
/// device. Each case must match the CPU element by element (shape included), launch device work, and record no host
/// fallback anywhere in the op; argument and per-element range errors must raise the CPU's exceptions.
/// </summary>
[Collection("VulkanGlobalState")]
public class TorchParityStructureDeviceTests : IClassFixture<DirectGpuTensorEngineTestFixture>
{
    private readonly DirectGpuTensorEngineTestFixture _fixture;
    public TorchParityStructureDeviceTests(DirectGpuTensorEngineTestFixture fixture) => _fixture = fixture;

    private DirectGpuTensorEngine GpuOrSkip()
    {
        Skip.If(!_fixture.IsAvailable, "No DirectGpu backend available.");
        var gpu = _fixture.Engine!;
        Skip.IfNot(gpu.GetBackend() is ITorchParityElementwiseBackend, "This backend has no PyTorch-parity device kernels (CUDA, OpenCL and Vulkan do).");
        return gpu;
    }

    private static long TotalFallbacks() => GpuLaunchProbe.Fallbacks.Sum(entry =>
    {
        int x = entry.IndexOf("x ", StringComparison.Ordinal);
        return x > 0 && long.TryParse(entry.Substring(0, x), out long count) ? count : 0L;
    });

    private static T OnDevice<T>(Func<T> run, Action<T> touch)
    {
        long launches = GpuLaunchProbe.Count, fallbacks = TotalFallbacks();
        var r = run();
        touch(r);
        Assert.True(GpuLaunchProbe.Count > launches, "no device work launched");
        Assert.Equal(fallbacks, TotalFallbacks());
        return r;
    }

    private static Tensor<float> Random(int salt, float scale, params int[] shape)
    {
        var rng = new Random(salt);
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = (float)((rng.NextDouble() - 0.5) * scale);
        return t;
    }

    private static void AssertClose(string what, Tensor<float> cpu, Tensor<float> gpu, double tolerance = 1e-5)
    {
        Assert.Equal(cpu._shape, gpu._shape);
        for (int i = 0; i < cpu.Length; i++)
        {
            float c = cpu[i], g = gpu[i];
            if (c == g) continue;   // also equal infinities, whose difference is NaN
            if (float.IsNaN(c) || float.IsNaN(g)) { Assert.True(float.IsNaN(c) && float.IsNaN(g), $"{what}[{i}]: cpu {c} gpu {g}"); continue; }
            Assert.True(Math.Abs(c - g) <= tolerance * Math.Max(1.0, Math.Abs(c)), $"{what}[{i}]: cpu {c:R} gpu {g:R}");
        }
    }

    [SkippableFact]
    public void ComplexAndViewAsComplex_MatchCpu_OnDevice()
    {
        var gpu = GpuOrSkip();
        var cpu = new CpuEngine();
        var re = Random(1, 4, 3, 7);
        var im = Random(2, 4, 3, 7);
        var pairs = Random(3, 4, 5, 6, 2);
        foreach (var (what, run) in new (string, Func<IEngine, Tensor<Complex<float>>>)[]
        {
            ("complex", e => e.TensorComplex(re, im)),
            ("view_as_complex", e => e.TensorViewAsComplex(pairs)),
        })
        {
            var expected = run(cpu);
            var actual = OnDevice(() => run(gpu), r => _ = r[0]);
            Assert.Equal(expected._shape, actual._shape);
            for (int i = 0; i < expected.Length; i++)
            {
                Assert.Equal(expected[i].Real, actual[i].Real);
                Assert.Equal(expected[i].Imaginary, actual[i].Imaginary);
            }
        }
    }

    [SkippableTheory]
    [InlineData(6, -1)]     // fewer non-zeros than size: trailing rows keep the fill value
    [InlineData(200, 7)]    // more than size: the first `size` in row-major order
    public void NonzeroStatic_MatchesCpu_OnDevice(int size, int fill)
    {
        var gpu = GpuOrSkip();
        var cpu = new CpuEngine();
        var x = new Tensor<float>(new[] { 4, 5, 6 });
        var rng = new Random(4);
        for (int i = 0; i < x.Length; i++) x[i] = rng.Next(4) switch { 0 => 0f, 1 => -0f, 2 => float.NaN, _ => rng.Next(1, 9) };
        var expected = cpu.TensorNonzeroStatic(x, size, fill);
        var actual = OnDevice(() => gpu.TensorNonzeroStatic(x, size, fill), r => _ = r[0]);
        Assert.Equal(expected._shape, actual._shape);
        for (int i = 0; i < expected.Length; i++) Assert.Equal(expected[i], actual[i]);
    }

    // 1.1M elements: the device prefix sum recurses through three levels of 1024-element blocks.
    [SkippableFact]
    public void NonzeroStatic_LargeInput_MatchesCpu_OnDevice()
    {
        var gpu = GpuOrSkip();
        var cpu = new CpuEngine();
        var x = new Tensor<float>(new[] { 1100, 1000 });
        var rng = new Random(8);
        for (int i = 0; i < x.Length; i++) x[i] = rng.Next(3) == 0 ? 0f : 1f;
        var expected = cpu.TensorNonzeroStatic(x, 700_000, -5);
        var actual = OnDevice(() => gpu.TensorNonzeroStatic(x, 700_000, -5), r => _ = r[0]);
        Assert.Equal(expected._shape, actual._shape);
        for (int i = 0; i < expected.Length; i++) Assert.True(expected[i] == actual[i], $"[{i}]: cpu {expected[i]} gpu {actual[i]}");
    }

    // Both adjoints (input and grid) on the device: the tape gradient of sum(w * grid_sample(x, g)) must equal the CPU's
    // for each mode / padding / alignment, with no host fallback in forward or backward.
    [SkippableTheory]
    [InlineData(GridSampleMode.Bilinear, GridSamplePadding.Zeros, false)]
    [InlineData(GridSampleMode.Bilinear, GridSamplePadding.Border, true)]
    [InlineData(GridSampleMode.Bilinear, GridSamplePadding.Reflection, false)]
    [InlineData(GridSampleMode.Bilinear, GridSamplePadding.Reflection, true)]
    [InlineData(GridSampleMode.Nearest, GridSamplePadding.Zeros, false)]
    public void GridSample3D_Adjoints_MatchCpu_OnDevice(GridSampleMode mode, GridSamplePadding padding, bool alignCorners)
    {
        var gpu = GpuOrSkip();
        var cpu = new CpuEngine();
        var input = Random(10, 4, 2, 3, 4, 5, 6);
        var grid = Random(11, 3.2f, 2, 3, 4, 2, 3);
        var weights = Random(12, 2, 2, 3, 3, 4, 2);
        (float[] Dx, float[] Dg) Gradients(IEngine e)
        {
            var previous = AiDotNetEngine.Current;
            AiDotNetEngine.Current = e;
            try
            {
                using var tape = new GradientTape<float>();
                var y = e.TensorGridSample3D(input, grid, mode, padding, alignCorners);
                var loss = e.ReduceSum(e.TensorMultiply(y, weights), new[] { 0, 1, 2, 3, 4 }, keepDims: false);
                var grads = tape.ComputeGradients(loss, new[] { input, grid });
                return (grads[input].ToArray(), grads[grid].ToArray());
            }
            finally
            {
                AiDotNetEngine.Current = previous;
            }
        }
        var expected = Gradients(cpu);
        long fallbacks = TotalFallbacks();
        var actual = Gradients(gpu);
        Assert.Equal(fallbacks, TotalFallbacks());
        foreach (var (what, c, g) in new[] { ("dx", expected.Dx, actual.Dx), ("dgrid", expected.Dg, actual.Dg) })
        {
            Assert.Equal(c.Length, g.Length);
            for (int i = 0; i < c.Length; i++)
                Assert.True(Math.Abs(c[i] - g[i]) <= 1e-4 * Math.Max(1.0, Math.Abs(c[i])), $"{what}[{i}]: cpu {c[i]:R} gpu {g[i]:R}");
        }
    }

    // PyTorch maps non-finite and out-of-int-range voxel coordinates out of bounds, so such points sample nothing; the
    // CPU's bare (int) cast used to read voxel 0 on .NET 9+ (NaN saturates to 0).
    [Theory]
    [InlineData(GridSampleMode.Bilinear, GridSamplePadding.Zeros)]
    [InlineData(GridSampleMode.Bilinear, GridSamplePadding.Border)]
    [InlineData(GridSampleMode.Nearest, GridSamplePadding.Reflection)]
    public void GridSample3D_Cpu_NonFiniteCoordinatesSampleNothing(GridSampleMode mode, GridSamplePadding padding)
    {
        var cpu = new CpuEngine();
        var input = Random(9, 4, 1, 2, 3, 3, 3);
        for (int i = 0; i < input.Length; i++) input[i] = Math.Abs(input[i]) + 1;   // every voxel non-zero
        var grid = new Tensor<float>(new[] { 1, 1, 1, 3, 3 });
        float[] bad = { float.NaN, float.PositiveInfinity, float.NegativeInfinity };
        for (int q = 0; q < 3; q++) { grid[q * 3] = 0.1f; grid[q * 3 + 1] = -0.2f; grid[q * 3 + 2] = bad[q]; }
        var y = cpu.TensorGridSample3D(input, grid, mode, padding);
        for (int i = 0; i < y.Length; i++)
        {
            // Border / reflection clamp an infinite coordinate onto the volume (a real sample); NaN never lands.
            bool infinite = padding != GridSamplePadding.Zeros && i % 3 != 0;
            if (!infinite) Assert.Equal(0f, y[i]);
            else Assert.False(float.IsNaN(y[i]));
        }
    }

    [SkippableFact]
    public void UnravelIndex_MatchesCpu_AndRejectsOutOfRange()
    {
        var gpu = GpuOrSkip();
        var cpu = new CpuEngine();
        var shape = new[] { 3, 4, 5 };
        var indices = new Tensor<float>(new[] { 6, 10 });
        for (int i = 0; i < indices.Length; i++) indices[i] = (i * 7 % 120) - 60;   // negative indices wrap
        indices[4] = -60.5f;   // truncates toward zero to -60: in range on both engines
        var expected = cpu.TensorUnravelIndex(indices, shape);
        var actual = OnDevice(() => gpu.TensorUnravelIndex(indices, shape), r => _ = r[0][0]);
        Assert.Equal(expected.Length, actual.Length);
        for (int d = 0; d < expected.Length; d++) AssertClose($"unravel[{d}]", expected[d], actual[d], 0);

        indices[17] = 60;
        var e = Assert.Throws<ArgumentOutOfRangeException>(() => gpu.TensorUnravelIndex(indices, shape));
        Assert.Contains("index 60", e.Message);
        indices[17] = float.NaN;   // rejected on both engines, whatever the runtime's (long) cast of NaN
        Assert.Throws<ArgumentOutOfRangeException>(() => cpu.TensorUnravelIndex(indices, shape));
        Assert.Throws<ArgumentOutOfRangeException>(() => gpu.TensorUnravelIndex(indices, shape));
    }

    [SkippableTheory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(-1)]
    public void UniqueDim_MatchesCpu_OnDevice(int dim)
    {
        var gpu = GpuOrSkip();
        var cpu = new CpuEngine();
        var x = new Tensor<float>(new[] { 9, 11, 4 });
        var rng = new Random(5);
        // Few distinct values so slices repeat; NaN (ordered first, equal to NaN) and infinities included.
        float[] pool = { 1, 2, float.NaN, float.PositiveInfinity, float.NegativeInfinity };
        for (int i = 0; i < x.Length; i++) x[i] = pool[rng.Next(dim == 0 ? 2 : pool.Length)];
        // Duplicate some slices along dim so unique has something to drop.
        int d = dim < 0 ? dim + 3 : dim;
        for (int i = 0; i < x.Length; i++)
        {
            var coord = new[] { i / 44, i / 4 % 11, i % 4 };
            if (coord[d] % 3 == 1)
            {
                var from = (int[])coord.Clone();
                from[d] -= 1;
                x[i] = x[(from[0] * 11 + from[1]) * 4 + from[2]];
            }
        }
        var expected = cpu.TensorUniqueDim(x, dim);
        var actual = OnDevice(() => gpu.TensorUniqueDim(x, dim), r => _ = r[0]);
        AssertClose($"unique_dim({dim})", expected, actual, 0);
    }

    [SkippableTheory]
    [InlineData(GridSampleMode.Bilinear, GridSamplePadding.Zeros, false)]
    [InlineData(GridSampleMode.Bilinear, GridSamplePadding.Border, true)]
    [InlineData(GridSampleMode.Bilinear, GridSamplePadding.Reflection, false)]
    [InlineData(GridSampleMode.Bilinear, GridSamplePadding.Reflection, true)]
    [InlineData(GridSampleMode.Nearest, GridSamplePadding.Zeros, true)]
    [InlineData(GridSampleMode.Nearest, GridSamplePadding.Reflection, false)]
    public void GridSample3D_MatchesCpu_OnDevice(GridSampleMode mode, GridSamplePadding padding, bool alignCorners)
    {
        var gpu = GpuOrSkip();
        var cpu = new CpuEngine();
        var input = Random(6, 4, 2, 3, 5, 6, 7);
        var grid = Random(7, 3.2f, 2, 4, 3, 5, 3);   // coordinates well outside [-1, 1] exercise the padding modes
        grid[5] = float.NaN;                          // a NaN coordinate samples nothing
        grid[11] = 1e12f;                             // so does one far outside the int range
        var expected = cpu.TensorGridSample3D(input, grid, mode, padding, alignCorners);
        var actual = OnDevice(() => gpu.TensorGridSample3D(input, grid, mode, padding, alignCorners), r => _ = r[0]);
        AssertClose($"grid_sample_3d {mode}/{padding}/{alignCorners}", expected, actual);
    }
}
