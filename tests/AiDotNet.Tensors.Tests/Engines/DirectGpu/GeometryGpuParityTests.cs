using AiDotNet.Tensors.Engines;
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
        catch { _gpuAvailable = false; }
    }

    private bool BackendImplementsGeometry()
    {
        var backendField = typeof(DirectGpuTensorEngine).GetField(
            "_backend",
            System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic);
        var backend = backendField?.GetValue(_gpu);
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
}
