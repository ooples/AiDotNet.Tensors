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
public class GeometryGpuParityTests : IDisposable
{
    private readonly DirectGpuTensorEngine? _gpu;
    private readonly CpuEngine _cpu = new();
    private readonly bool _gpuAvailable;
    private const float Tolerance = 1e-3f;

    public GeometryGpuParityTests()
    {
        try
        {
            _gpu = new DirectGpuTensorEngine();
            _gpuAvailable = _gpu.IsGpuAvailable && BackendImplementsGeometry();
        }
        catch (PlatformNotSupportedException) { _gpuAvailable = false; }
        catch (System.DllNotFoundException) { _gpuAvailable = false; }
    }

    private bool BackendImplementsGeometry()
    {
        // The runtime backend is _directGpu.Backend (a DirectGpuEngine), NOT the _backend field (which is
        // null on the lazy path). Probing _backend made these parity tests silently skip everywhere.
        var directGpuField = typeof(DirectGpuTensorEngine).GetField(
            "_directGpu",
            System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic);
        var directGpu = directGpuField?.GetValue(_gpu);
        var backend = directGpu?.GetType().GetProperty("Backend")?.GetValue(directGpu);
        return backend is IGeometryBackend;
    }

    public void Dispose() => (_gpu as IDisposable)?.Dispose();

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
        var g = _gpu!.Interpolate(input, new[] { 12, 15 }, mode, alignCorners);
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
        var g = _gpu!.PadNd(input, pad, mode, 0.5f);
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
        var g = _gpu!.GridSample(input, grid, GridSampleMode.Bilinear, GridSamplePadding.Zeros, false);
        var c = _cpu.GridSample(input, grid, GridSampleMode.Bilinear, GridSamplePadding.Zeros, false);
        AssertClose(g, c);
    }

    [SkippableFact]
    public void AffineGrid3D_GpuMatchesCpu()
    {
        SkipIfUnavailable();
        var rng = new Random(5);
        var t = new float[1 * 3 * 4];
        for (int i = 0; i < t.Length; i++) t[i] = (float)(rng.NextDouble() * 2 - 1);
        var theta = new Tensor<float>(t, new[] { 1, 3, 4 });
        var g = _gpu!.AffineGrid3D(theta, 2, 3, 3, alignCorners: false);
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
            gpuOutput = _gpu!.PartialCorrelationVolume(gpuFirst, gpuSecond, radius: 1);
            var loss = _gpu.ReduceSum(gpuOutput, null);
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

        var gpu = _gpu!.ForwardSplat(input, flow, normalize);
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
        Func<Tensor<float>> forward = () => _gpu!.ForwardSplat(input, flow);

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

        var gpu = _gpu!.ForwardSplatBackwardInput(gradOutput, input, flow, normalize);
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
        var gpuOutput = _gpu!.ForwardSplat(input, flow, normalize);
        var cpuOutput = _cpu.ForwardSplat(input, flow, normalize);

        var gpu = _gpu.ForwardSplatBackwardFlow(
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
            _gpu!.ForwardSplatBackwardFlow(wrongShape, input, flow, input));
        Assert.Equal("gradOutput", gradError.ParamName);

        var outputError = Assert.Throws<ArgumentException>(() =>
            _gpu!.ForwardSplatBackwardFlow(gradOutput, input, flow, wrongShape));
        Assert.Equal("output", outputError.ParamName);

        var nullOutputError = Assert.Throws<ArgumentNullException>(() =>
            _gpu!.ForwardSplatBackwardFlow(gradOutput, input, flow, null!));
        Assert.Equal("output", nullOutputError.ParamName);

        var unnormalized = _gpu!.ForwardSplatBackwardFlow(
            gradOutput, input, flow, null!, normalize: false);
        Assert.Equal(flow.Shape.ToArray(), unnormalized.Shape.ToArray());
    }
}
