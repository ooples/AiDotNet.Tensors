using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// GPU-vs-CPU parity for the geometry / sampling ops added by Issue #217.
/// Same skip semantics as <c>DetectionGpuParityTests</c>: without an
/// <c>IGeometryBackend</c>-implementing backend the tests skip cleanly
/// and the engine falls through to CpuEngine.
/// </summary>
[Collection("VulkanGlobalState")]
public class GeometryGpuParityTests : IClassFixture<DirectGpuTensorEngineTestFixture>
{
    private readonly DirectGpuTensorEngineTestFixture _fixture;
    private readonly CpuEngine _cpu = new();
    private readonly bool _gpuAvailable;
    private const float Tolerance = 1e-3f;
    // Typed as IEngine on purpose: most GPU overrides are explicit IEngine implementations, so a call on the
    // concrete DirectGpuTensorEngine binds to the inherited CpuEngine method and the test compares the CPU with itself.
    private IEngine Gpu => _fixture.Engine ?? throw new InvalidOperationException(
        "Direct GPU engine was not initialized.", _fixture.InitializationException);

    public GeometryGpuParityTests(DirectGpuTensorEngineTestFixture fixture)
    {
        _fixture = fixture;
        _gpuAvailable = fixture.IsAvailable && BackendImplementsGeometry();
    }

    private bool BackendImplementsGeometry()
    {
        // The runtime backend is _directGpu.Backend (a DirectGpuEngine), NOT the _backend field (which is
        // null on the lazy path). Probing _backend made these parity tests silently skip everywhere.
        var directGpuField = typeof(DirectGpuTensorEngine).GetField(
            "_directGpu",
            System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic);
        var directGpu = directGpuField?.GetValue(_fixture.Engine);
        var backend = directGpu?.GetType().GetProperty("Backend")?.GetValue(directGpu);
        return backend is IGeometryBackend;
    }

    private void SkipIfUnavailable() => Skip.If(!_gpuAvailable,
        "GPU backend without IGeometryBackend support — CPU fallback is exercised by GeometryOpsTests instead.");

    private static Tensor<float> Rand4D(int seed, int N, int C, int H, int W, float range = 1f)
    {
        var rng = new Random(seed);
        var data = new float[N * C * H * W];
        for (int i = 0; i < data.Length; i++)
            data[i] = (float)(rng.NextDouble() * range * 2 - range);
        return new Tensor<float>(data, new[] { N, C, H, W });
    }

    private static void AssertClose(Tensor<float> g, Tensor<float> c, float tol = Tolerance)
    {
        Assert.Equal(c.Shape.ToArray(), g.Shape.ToArray());
        var gs = g.AsSpan(); var cs = c.AsSpan();
        for (int i = 0; i < gs.Length; i++)
        {
            // Non-finite values must match exactly: NaN fails every tolerance comparison, so without this a NaN GPU
            // value would pass.
            if (float.IsNaN(gs[i]) || float.IsInfinity(gs[i]) || float.IsNaN(cs[i]) || float.IsInfinity(cs[i]))
            {
                bool same = float.IsNaN(gs[i]) ? float.IsNaN(cs[i]) : gs[i].Equals(cs[i]);
                if (!same)
                    throw new Xunit.Sdk.XunitException($"GPU vs CPU non-finite mismatch at [{i}]: gpu={gs[i]}, cpu={cs[i]}");
                continue;
            }

            float d = Math.Abs(gs[i] - cs[i]);
            float scale = 1 + Math.Abs(cs[i]);
            if (d > tol * scale)
                throw new Xunit.Sdk.XunitException(
                    $"GPU vs CPU mismatch at [{i}]: gpu={gs[i]}, cpu={cs[i]}, diff={d}");
        }
    }

    private static void AssertNonDegenerate(Tensor<float> tensor, string name)
    {
        foreach (float value in tensor.AsSpan())
            if (Math.Abs(value) > 1e-6f) return;
        throw new Xunit.Sdk.XunitException(
            $"{name} is entirely zero; parity would pass vacuously.");
    }

    [SkippableTheory]
    [InlineData(InterpolateMode.Nearest, false)]
    [InlineData(InterpolateMode.Bilinear, false)]
    [InlineData(InterpolateMode.Bilinear, true)]
    [InlineData(InterpolateMode.Bicubic, false)]
    [InlineData(InterpolateMode.Area, false)]
    public void Interpolate2D_GpuMatchesCpu(InterpolateMode mode, bool alignCorners)
    {
        SkipIfUnavailable();
        var input = Rand4D(1, 2, 3, 8, 10);
        var g = Gpu.Interpolate(input, new[] { 12, 15 }, mode, alignCorners);
        var c = _cpu.Interpolate(input, new[] { 12, 15 }, mode, alignCorners);
        AssertClose(g, c);
    }

    [SkippableTheory]
    [InlineData(PadMode.Constant)]
    [InlineData(PadMode.Reflect)]
    [InlineData(PadMode.Replicate)]
    [InlineData(PadMode.Circular)]
    public void Pad4D_GpuMatchesCpu(PadMode mode)
    {
        SkipIfUnavailable();
        var input = Rand4D(2, 1, 2, 4, 5);
        int[] pad = { 1, 2, 1, 1, 0, 0, 0, 0 };
        var g = Gpu.PadNd(input, pad, mode, 0.5f);
        var c = _cpu.PadNd(input, pad, mode, 0.5f);
        AssertClose(g, c);
    }

    [SkippableFact]
    public void GridSample2D_Bilinear_Zeros_GpuMatchesCpu()
    {
        SkipIfUnavailable();
        var input = Rand4D(3, 1, 2, 4, 4);  // NHWC: [1, 4, 4, 2]
        input = new Tensor<float>(input.AsSpan().ToArray(), new[] { 1, 4, 4, 2 });
        var rng = new Random(4);
        var gridData = new float[1 * 3 * 3 * 2];
        for (int i = 0; i < gridData.Length; i++) gridData[i] = (float)(rng.NextDouble() * 2 - 1);
        var grid = new Tensor<float>(gridData, new[] { 1, 3, 3, 2 });
        var g = Gpu.GridSample(input, grid, GridSampleMode.Bilinear, GridSamplePadding.Zeros, false);
        var c = _cpu.GridSample(input, grid, GridSampleMode.Bilinear, GridSamplePadding.Zeros, false);
        AssertClose(g, c);
    }

    // #1134: a NaN / infinite / out-of-int-range grid point samples nothing on the device too, in every mode and
    // padding; finite points are unaffected.
    [SkippableTheory]
    [MemberData(nameof(NonFiniteSortAndGridSampleTests.AllModes), MemberType = typeof(NonFiniteSortAndGridSampleTests))]
    public void GridSample2D_NonFinitePoints_GpuMatchesCpu(GridSampleMode mode, GridSamplePadding padding, bool alignCorners)
    {
        SkipIfUnavailable();
        var input = new Tensor<float>(Enumerable.Range(1, 24).Select(v => (float)v).ToArray(), new[] { 1, 2, 3, 4 });
        var (grid, bad) = NonFiniteSortAndGridSampleTests.MixedGrid();
        var g = Gpu.GridSample(input, grid, mode, padding, alignCorners);
        var c = _cpu.GridSample(input, grid, mode, padding, alignCorners);
        for (int ch = 0; ch < 2; ch++)
            for (int i = 0; i < bad.Length; i++)
                if (bad[i]) Assert.True(g[ch * bad.Length + i] == 0f, $"{mode}/{padding}/align={alignCorners} point {i}: {g[ch * bad.Length + i]}");
        AssertClose(g, c);
    }

    [SkippableFact]
    public void GridSample2D_TwoArgument_NonFinitePoints_ForwardAndGradientsMatchCpu()
    {
        SkipIfUnavailable();
        var input = new Tensor<float>(Enumerable.Range(1, 24).Select(v => (float)v).ToArray(), new[] { 1, 2, 3, 4 });
        var (grid, bad) = NonFiniteSortAndGridSampleTests.MixedGrid();
        var (go, gdx, gdg) = NonFiniteSortAndGridSampleTests.BilinearWithGradients(Gpu, input, grid);
        var (co, cdx, cdg) = NonFiniteSortAndGridSampleTests.BilinearWithGradients(_cpu, input, grid);
        foreach (var (name, gv, cv) in new[] { ("output", go, co), ("dInput", gdx, cdx), ("dGrid", gdg, cdg) })
        {
            Assert.Equal(cv.Length, gv.Length);
            for (int i = 0; i < cv.Length; i++)
            {
                // A NaN on either side fails here (a bare tolerance test would pass it: NaN > x is false).
                Assert.False(float.IsNaN(gv[i]) || float.IsInfinity(gv[i]), $"{name}[{i}]: gpu {gv[i]} is not finite");
                Assert.False(float.IsNaN(cv[i]) || float.IsInfinity(cv[i]), $"{name}[{i}]: cpu {cv[i]} is not finite");
                Assert.True(Math.Abs(gv[i] - cv[i]) <= Tolerance * (1 + Math.Abs(cv[i])), $"{name}[{i}]: gpu {gv[i]} cpu {cv[i]}");
            }
        }
        for (int i = 0; i < bad.Length; i++)
            if (bad[i])
            {
                Assert.Equal(0f, gdg[2 * i]);
                Assert.Equal(0f, gdg[2 * i + 1]);
            }
    }

    // The GPU results of the mode/padding GridSample and of AffineGrid3D returned without a tape edge, so a tape
    // around them gave the input, the grid and theta no gradient at all on the GPU.
    private static float[] TapeGradient(IEngine engine, Tensor<float> source, Func<IEngine, Tensor<float>> forward)
    {
        using var tape = new AiDotNet.Tensors.Engines.Autodiff.GradientTape<float>();
        var loss = forward(engine);
        var grads = tape.ComputeGradients(loss, new[] { source });
        Assert.True(grads.ContainsKey(source), "The tape found no gradient path to the source.");
        return grads[source].GetFlattenedData();
    }

    private static void AssertGradientsClose(float[] gpu, float[] cpu, string what)
    {
        Assert.Equal(cpu.Length, gpu.Length);
        Assert.True(cpu.Any(v => v != 0f), $"{what}: the CPU gradient is zero, so the comparison proves nothing.");
        for (int i = 0; i < cpu.Length; i++)
            Assert.True(Math.Abs(gpu[i] - cpu[i]) <= Tolerance * (1 + Math.Abs(cpu[i])), $"{what}[{i}]: gpu {gpu[i]} cpu {cpu[i]}");
    }

    [SkippableFact]
    public void GridSample2D_WithModes_TapeGradientsOnGpu_MatchCpu()
    {
        SkipIfUnavailable();
        var input = Rand4D(21, 2, 3, 8, 8);
        var grid = new Tensor<float>(Rand4D(22, 2, 6, 6, 2, 0.8f).GetFlattenedData(), new[] { 2, 6, 6, 2 });
        var weights = Rand4D(23, 2, 3, 6, 6);
        Tensor<float> Loss(IEngine e, Tensor<float> x, Tensor<float> g) => e.ReduceSum(
            e.TensorMultiply(e.GridSample(x, g, GridSampleMode.Bilinear, GridSamplePadding.Zeros, false), weights), null);
        AssertGradientsClose(TapeGradient(Gpu, input, e => Loss(e, input, grid)), TapeGradient(_cpu, input, e => Loss(e, input, grid)), "dInput");
        AssertGradientsClose(TapeGradient(Gpu, grid, e => Loss(e, input, grid)), TapeGradient(_cpu, grid, e => Loss(e, input, grid)), "dGrid");
    }

    [SkippableFact]
    public void AffineGrid3D_TapeThetaGradientOnGpu_MatchesCpu()
    {
        SkipIfUnavailable();
        var rng = new Random(24);
        var t = new float[2 * 3 * 4];
        for (int i = 0; i < t.Length; i++) t[i] = (float)(rng.NextDouble() * 2 - 1);
        var theta = new Tensor<float>(t, new[] { 2, 3, 4 });
        var w = new float[2 * 3 * 4 * 5 * 3];
        for (int i = 0; i < w.Length; i++) w[i] = (float)(rng.NextDouble() * 2 - 1);
        var weights = new Tensor<float>(w, new[] { 2, 3, 4, 5, 3 });
        Tensor<float> Loss(IEngine e) => e.ReduceSum(e.TensorMultiply(e.AffineGrid3D(theta, 3, 4, 5, alignCorners: false), weights), null);
        AssertGradientsClose(TapeGradient(Gpu, theta, Loss), TapeGradient(_cpu, theta, Loss), "dTheta");
    }

    [SkippableFact]
    public void AffineGrid3D_GpuMatchesCpu()
    {
        SkipIfUnavailable();
        var rng = new Random(5);
        var t = new float[1 * 3 * 4];
        for (int i = 0; i < t.Length; i++) t[i] = (float)(rng.NextDouble() * 2 - 1);
        var theta = new Tensor<float>(t, new[] { 1, 3, 4 });
        var g = Gpu.AffineGrid3D(theta, 2, 3, 3, alignCorners: false);
        var c = _cpu.AffineGrid3D(theta, 2, 3, 3, alignCorners: false);
        AssertClose(g, c);
    }

    [SkippableFact]
    public void PartialCorrelationVolume_GpuForwardAndBackwardMatchCpu()
    {
        SkipIfUnavailable();
        var gpuFirst = Rand4D(14, 1, 2, 4, 5);
        var gpuSecond = Rand4D(15, 1, 2, 4, 5);
        var cpuFirst = new Tensor<float>(gpuFirst.GetDataArray(), gpuFirst.Shape.ToArray());
        var cpuSecond = new Tensor<float>(gpuSecond.GetDataArray(), gpuSecond.Shape.ToArray());
        Tensor<float> gpuOutput;
        Dictionary<Tensor<float>, Tensor<float>> gpuGradients;
        using (var tape = new AiDotNet.Tensors.Engines.Autodiff.GradientTape<float>())
        {
            gpuOutput = Gpu.PartialCorrelationVolume(gpuFirst, gpuSecond, radius: 1);
            var loss = Gpu.ReduceSum(gpuOutput, null);
            gpuGradients = tape.ComputeGradients(loss, [gpuFirst, gpuSecond]);
        }
        Assert.True(gpuOutput.IsGpuResident, "Correlation output must remain GPU-resident.");
        Assert.True(gpuGradients[gpuFirst].IsGpuResident,
            "First correlation gradient must remain GPU-resident.");
        Assert.True(gpuGradients[gpuSecond].IsGpuResident,
            "Second correlation gradient must remain GPU-resident.");

        Tensor<float> cpuOutput;
        Dictionary<Tensor<float>, Tensor<float>> cpuGradients;
        using (var tape = new AiDotNet.Tensors.Engines.Autodiff.GradientTape<float>())
        {
            cpuOutput = _cpu.PartialCorrelationVolume(cpuFirst, cpuSecond, radius: 1);
            var loss = _cpu.ReduceSum(cpuOutput, null);
            cpuGradients = tape.ComputeGradients(loss, [cpuFirst, cpuSecond]);
        }

        AssertClose(gpuOutput, cpuOutput, 2e-3f);
        AssertClose(gpuGradients[gpuFirst], cpuGradients[cpuFirst], 3e-3f);
        AssertClose(gpuGradients[gpuSecond], cpuGradients[cpuSecond], 3e-3f);
        AssertNonDegenerate(cpuOutput, nameof(cpuOutput));
        AssertNonDegenerate(cpuGradients[cpuFirst], "first gradient");
        AssertNonDegenerate(cpuGradients[cpuSecond], "second gradient");
    }

    [SkippableTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void ForwardSplat_GpuMatchesCpu(bool normalize)
    {
        SkipIfUnavailable();
        var input = Rand4D(6, 1, 2, 4, 5);
        var flow = Rand4D(7, 1, 2, 4, 5, range: 0.35f);

        var gpu = Gpu.ForwardSplat(input, flow, normalize);
        var cpu = _cpu.ForwardSplat(input, flow, normalize);

        AssertClose(gpu, cpu, 2e-3f);
    }

    [SkippableFact]
    public void ForwardSplat_GpuGraphModeRejectsCpuFallback()
    {
        SkipIfUnavailable();
        var input = Rand4D(16, 1, 2, 4, 5);
        var flow = Rand4D(17, 1, 2, 4, 5, range: 0.35f);
        using var cache = new CompiledModelCache<float>();
        Func<Tensor<float>> forward = () => Gpu.ForwardSplat(input, flow);

        var error = Assert.Throws<NotSupportedException>(() =>
            cache.GetOrCompileInference([input, flow], forward));

        Assert.Contains("ForwardSplat", error.Message, StringComparison.Ordinal);
        Assert.Contains("graph capture", error.Message, StringComparison.Ordinal);
    }

    [SkippableTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void ForwardSplatBackwardInput_GpuMatchesCpu(bool normalize)
    {
        SkipIfUnavailable();
        var input = Rand4D(8, 1, 2, 4, 5);
        var flow = Rand4D(9, 1, 2, 4, 5, range: 0.35f);
        var gradOutput = Rand4D(10, 1, 2, 4, 5);

        var gpu = Gpu.ForwardSplatBackwardInput(gradOutput, input, flow, normalize);
        var cpu = _cpu.ForwardSplatBackwardInput(gradOutput, input, flow, normalize);

        AssertClose(gpu, cpu, 2e-3f);
    }

    [SkippableTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void ForwardSplatBackwardFlow_GpuMatchesCpu(bool normalize)
    {
        SkipIfUnavailable();
        var input = Rand4D(11, 1, 2, 4, 5);
        var flow = Rand4D(12, 1, 2, 4, 5, range: 0.35f);
        var gradOutput = Rand4D(13, 1, 2, 4, 5);
        var gpuOutput = Gpu.ForwardSplat(input, flow, normalize);
        var cpuOutput = _cpu.ForwardSplat(input, flow, normalize);

        var gpu = Gpu.ForwardSplatBackwardFlow(
            gradOutput, input, flow, gpuOutput, normalize);
        var cpu = _cpu.ForwardSplatBackwardFlow(
            gradOutput, input, flow, cpuOutput, normalize);

        AssertClose(gpu, cpu, 3e-3f);
    }

    [SkippableFact]
    public void ForwardSplatBackwardFlow_GpuValidationNamesArgumentsByMode()
    {
        SkipIfUnavailable();
        var input = Rand4D(18, 1, 2, 4, 5);
        var flow = Rand4D(19, 1, 2, 4, 5, range: 0.35f);
        var gradOutput = Rand4D(20, 1, 2, 4, 5);
        var wrongShape = Rand4D(21, 1, 2, 2, 2);

        var gradError = Assert.Throws<ArgumentException>(() =>
            Gpu.ForwardSplatBackwardFlow(wrongShape, input, flow, input));
        Assert.Equal("gradOutput", gradError.ParamName);

        var outputError = Assert.Throws<ArgumentException>(() =>
            Gpu.ForwardSplatBackwardFlow(gradOutput, input, flow, wrongShape));
        Assert.Equal("output", outputError.ParamName);

        var nullOutputError = Assert.Throws<ArgumentNullException>(() =>
            Gpu.ForwardSplatBackwardFlow(gradOutput, input, flow, null));
        Assert.Equal("output", nullOutputError.ParamName);

        var unnormalized = Gpu.ForwardSplatBackwardFlow(
            gradOutput, input, flow, null, normalize: false);
        Assert.Equal(flow.Shape.ToArray(), unnormalized.Shape.ToArray());
    }

    public static System.Collections.Generic.IEnumerable<object[]> GridSampleBackwardCombinations()
    {
        foreach (GridSampleMode mode in Enum.GetValues(typeof(GridSampleMode)))
        foreach (GridSamplePadding padding in Enum.GetValues(typeof(GridSamplePadding)))
        foreach (bool align in new[] { false, true })
            yield return new object[] { mode, padding, align };
    }

    /// <summary>
    /// Sampling points between -3 and S+2 pixels (so every padding rule is hit), each at least 0.15 from a tap
    /// boundary and, for Nearest, from the .5 rounding tie: float and double then pick the same taps.
    /// </summary>
    private static Tensor<float> PaddingExercisingGrid(int seed, int N, int outH, int outW, int H, int W,
        GridSampleMode mode, bool align)
    {
        var rng = new Random(seed);
        var data = new float[N * outH * outW * 2];
        for (int i = 0; i < data.Length; i++)
        {
            int size = i % 2 == 0 ? W : H;
            int whole = rng.Next(-3, size + 2);
            double frac;
            do frac = 0.15 + rng.NextDouble() * 0.7;
            while (mode == GridSampleMode.Nearest && Math.Abs(frac - 0.5) < 0.15);
            double pixel = whole + frac;
            data[i] = (float)(align ? pixel * 2.0 / (size - 1) - 1.0 : (2.0 * pixel + 1.0) / size - 1.0);
        }
        return new Tensor<float>(data, new[] { N, outH, outW, 2 });
    }

    private bool BackendImplementsGeometryBackward()
    {
        var directGpuField = typeof(DirectGpuTensorEngine).GetField(
            "_directGpu",
            System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic);
        var directGpu = directGpuField?.GetValue(_fixture.Engine);
        return directGpu?.GetType().GetProperty("Backend")?.GetValue(directGpu) is IGeometryBackwardBackend;
    }

    /// <summary>
    /// The device GridSample backward for every mode, padding and convention (#1680) matches the CpuEngine
    /// reference, and runs on the device: no GridSample backward fallback is recorded.
    /// </summary>
    [SkippableTheory]
    [MemberData(nameof(GridSampleBackwardCombinations))]
    public void GridSampleModeBackward_GpuMatchesCpu(GridSampleMode mode, GridSamplePadding padding, bool alignCorners)
    {
        SkipIfUnavailable();
        Skip.If(!BackendImplementsGeometryBackward(), "Backend has no IGeometryBackwardBackend kernels.");
        const int N = 2, C = 3, H = 5, W = 6, outH = 4, outW = 5;
        var input = Rand4D(31, N, C, H, W);
        var gradOut = Rand4D(32, N, C, outH, outW);
        var grid = PaddingExercisingGrid(33, N, outH, outW, H, W, mode, alignCorners);

        GpuLaunchProbe.Reset();
        var gpuInput = Gpu.GridSampleBackwardInput(gradOut, grid, new[] { N, C, H, W }, mode, padding, alignCorners);
        var gpuGrid = Gpu.GridSampleBackwardGrid(gradOut, input, grid, mode, padding, alignCorners);
        var fallbacks = GpuLaunchProbe.Fallbacks.Where(f => f.Contains("GridSampleBackward")).ToArray();
        Assert.True(fallbacks.Length == 0, "GridSample backward fell back to the CPU: " + string.Join("; ", fallbacks));

        var cpuInput = _cpu.GridSampleBackwardInput(gradOut, grid, new[] { N, C, H, W }, mode, padding, alignCorners);
        var cpuGrid = _cpu.GridSampleBackwardGrid(gradOut, input, grid, mode, padding, alignCorners);
        AssertNonDegenerate(cpuInput, "CPU input gradient");
        if (mode != GridSampleMode.Nearest) AssertNonDegenerate(cpuGrid, "CPU grid gradient");
        AssertClose(gpuInput, cpuInput, 1e-4f);
        AssertClose(gpuGrid, cpuGrid, 1e-4f);
    }

    /// <summary>
    /// Nearest's grid gradient is exactly 0 on the device even when the input holds infinities and the upstream
    /// gradient is huge: a product of a non-finite value with the zero derivative would otherwise be NaN.
    /// </summary>
    [SkippableTheory]
    [InlineData(GridSamplePadding.Zeros)]
    [InlineData(GridSamplePadding.Border)]
    [InlineData(GridSamplePadding.Reflection)]
    public void GridSampleNearestBackward_NonFiniteInput_GridGradientIsExactlyZero(GridSamplePadding padding)
    {
        SkipIfUnavailable();
        Skip.If(!BackendImplementsGeometryBackward(), "Backend has no IGeometryBackwardBackend kernels.");
        const int N = 1, C = 2, H = 4, W = 5, outH = 3, outW = 4;
        var input = Rand4D(41, N, C, H, W);
        var span = input.AsWritableSpan();
        for (int i = 0; i < span.Length; i += 3) span[i] = i % 2 == 0 ? float.PositiveInfinity : float.NaN;
        var gradOut = Rand4D(42, N, C, outH, outW);
        foreach (ref float v in gradOut.AsWritableSpan()) v *= 1e30f;
        var grid = PaddingExercisingGrid(43, N, outH, outW, H, W, GridSampleMode.Nearest, false);

        var gpuGrid = Gpu.GridSampleBackwardGrid(gradOut, input, grid, GridSampleMode.Nearest, padding, false);
        foreach (float v in gpuGrid.AsSpan()) Assert.Equal(0f, v);
        AssertClose(gpuGrid, _cpu.GridSampleBackwardGrid(gradOut, input, grid, GridSampleMode.Nearest, padding, false));
    }
}
