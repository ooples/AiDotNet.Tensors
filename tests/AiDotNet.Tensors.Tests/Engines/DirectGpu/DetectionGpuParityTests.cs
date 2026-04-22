using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// GPU-vs-CPU parity for the vision detection ops added by Issue #217.
/// Skipped automatically when no GPU backend is available, or when the
/// active backend doesn't implement <c>IDetectionBackend</c> — in that
/// configuration <c>DirectGpuTensorEngine</c> falls through to
/// <c>CpuEngine</c> and there's nothing GPU-side to validate.
/// </summary>
[Collection("VulkanGlobalState")]
public class DetectionGpuParityTests : IDisposable
{
    private readonly DirectGpuTensorEngine? _gpu;
    private readonly CpuEngine _cpu = new();
    private readonly bool _gpuAvailable;
    private const float Tolerance = 1e-4f;

    public DetectionGpuParityTests()
    {
        // Only swallow PlatformNotSupportedException / DllNotFoundException
        // (no native GPU runtime on this machine). A real kernel / module
        // compilation regression should surface as a test failure, not a
        // silent skip.
        try
        {
            _gpu = new DirectGpuTensorEngine();
            _gpuAvailable = _gpu.IsGpuAvailable && BackendImplementsDetection();
        }
        catch (PlatformNotSupportedException) { _gpuAvailable = false; }
        catch (System.DllNotFoundException) { _gpuAvailable = false; }
    }

    private bool BackendImplementsDetection()
    {
        var backendField = typeof(DirectGpuTensorEngine).GetField(
            "_backend",
            System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic);
        var backend = backendField?.GetValue(_gpu);
        return backend is IDetectionBackend;
    }

    public void Dispose() => (_gpu as IDisposable)?.Dispose();

    private void SkipIfUnavailable() => Skip.If(!_gpuAvailable,
        "GPU backend without IDetectionBackend support — CPU fallback is exercised by BoxOpsTests instead.");

    private static Tensor<float> RandBoxes(int seed, int n, float range = 100f)
    {
        var rng = new Random(seed);
        var data = new float[n * 4];
        for (int i = 0; i < n; i++)
        {
            float x1 = (float)(rng.NextDouble() * range);
            float y1 = (float)(rng.NextDouble() * range);
            float w = (float)(rng.NextDouble() * range * 0.5);
            float h = (float)(rng.NextDouble() * range * 0.5);
            data[i * 4] = x1;
            data[i * 4 + 1] = y1;
            data[i * 4 + 2] = x1 + w;
            data[i * 4 + 3] = y1 + h;
        }
        return new Tensor<float>(data, new[] { n, 4 });
    }

    private static void AssertClose(Tensor<float> g, Tensor<float> c, float tol = Tolerance)
    {
        Assert.Equal(c.Shape.ToArray(), g.Shape.ToArray());
        var gs = g.AsSpan();
        var cs = c.AsSpan();
        for (int i = 0; i < gs.Length; i++)
        {
            float d = Math.Abs(gs[i] - cs[i]);
            float scale = 1 + Math.Abs(cs[i]);
            if (d > tol * scale)
                throw new Xunit.Sdk.XunitException(
                    $"GPU vs CPU mismatch at [{i}]: gpu={gs[i]}, cpu={cs[i]}, diff={d}");
        }
    }

    [SkippableFact]
    public void BoxIou_GpuMatchesCpu()
    {
        SkipIfUnavailable();
        var a = RandBoxes(1, 16);
        var b = RandBoxes(2, 12);
        AssertClose(_gpu!.BoxIou(a, b), _cpu.BoxIou(a, b));
    }

    [SkippableFact]
    public void GeneralizedBoxIou_GpuMatchesCpu()
    {
        SkipIfUnavailable();
        var a = RandBoxes(3, 10);
        var b = RandBoxes(4, 14);
        AssertClose(_gpu!.GeneralizedBoxIou(a, b), _cpu.GeneralizedBoxIou(a, b));
    }

    [SkippableFact]
    public void DistanceBoxIou_GpuMatchesCpu()
    {
        SkipIfUnavailable();
        var a = RandBoxes(5, 8);
        var b = RandBoxes(6, 10);
        AssertClose(_gpu!.DistanceBoxIou(a, b), _cpu.DistanceBoxIou(a, b));
    }

    [SkippableFact]
    public void CompleteBoxIou_GpuMatchesCpu()
    {
        SkipIfUnavailable();
        var a = RandBoxes(7, 6);
        var b = RandBoxes(8, 9);
        // CIoU's atan term loses some FP precision on GPU vs CPU; relax
        // tolerance very slightly to absorb 1-2 ULPs of accumulated error.
        AssertClose(_gpu!.CompleteBoxIou(a, b), _cpu.CompleteBoxIou(a, b), tol: 5e-4f);
    }

    [SkippableFact]
    public void BoxArea_GpuMatchesCpu()
    {
        SkipIfUnavailable();
        var boxes = RandBoxes(9, 32);
        AssertClose(_gpu!.BoxArea(boxes), _cpu.BoxArea(boxes));
    }

    [SkippableFact]
    public void BoxConvert_AllPairs_GpuMatchesCpu()
    {
        SkipIfUnavailable();
        var boxes = RandBoxes(10, 17);
        // Enum.GetValues<T>() is net5+; cast the non-generic form for net471.
        foreach (BoxFormat from in (BoxFormat[])Enum.GetValues(typeof(BoxFormat)))
            foreach (BoxFormat to in (BoxFormat[])Enum.GetValues(typeof(BoxFormat)))
            {
                if (from == to) continue;
                AssertClose(_gpu!.BoxConvert(boxes, from, to), _cpu.BoxConvert(boxes, from, to));
            }
    }

    [SkippableFact]
    public void BoxIou_SquareN_GpuMatchesCpu_RegressionShape()
    {
        // Exact same input on both sides — common case for self-IoU
        // (e.g. NMS upstream). Exercises the diagonal where IoU should be
        // exactly 1.0 modulo FP roundoff.
        SkipIfUnavailable();
        var boxes = RandBoxes(11, 24);
        AssertClose(_gpu!.BoxIou(boxes, boxes), _cpu.BoxIou(boxes, boxes));
    }

    // ========================================================================
    // Backward parity (Issue #217). GPU backward kernel runs in fp32, CPU
    // reference runs in fp64 internally — tolerance relaxed to 1e-3 to absorb
    // the precision delta (especially CIoU with its atan term).
    // ========================================================================

    private static Tensor<float> RandGrad(int seed, int n, int m)
    {
        var rng = new Random(seed ^ 0x55);
        var data = new float[n * m];
        for (int i = 0; i < data.Length; i++)
            data[i] = (float)(rng.NextDouble() * 2 - 1);
        return new Tensor<float>(data, new[] { n, m });
    }

    [SkippableFact]
    public void BoxIouBackward_GpuMatchesCpu()
    {
        SkipIfUnavailable();
        var a = RandBoxes(21, 6);
        var b = RandBoxes(22, 8);
        var go = RandGrad(23, 6, 8);
        var (gpuA, gpuB) = _gpu!.BoxIouBackward(go, a, b);
        var (cpuA, cpuB) = _cpu.BoxIouBackward(go, a, b);
        AssertClose(gpuA, cpuA, tol: 1e-3f);
        AssertClose(gpuB, cpuB, tol: 1e-3f);
    }

    [SkippableFact]
    public void GeneralizedBoxIouBackward_GpuMatchesCpu()
    {
        SkipIfUnavailable();
        var a = RandBoxes(24, 5);
        var b = RandBoxes(25, 7);
        var go = RandGrad(26, 5, 7);
        var (gpuA, gpuB) = _gpu!.GeneralizedBoxIouBackward(go, a, b);
        var (cpuA, cpuB) = _cpu.GeneralizedBoxIouBackward(go, a, b);
        AssertClose(gpuA, cpuA, tol: 1e-3f);
        AssertClose(gpuB, cpuB, tol: 1e-3f);
    }

    [SkippableFact]
    public void DistanceBoxIouBackward_GpuMatchesCpu()
    {
        SkipIfUnavailable();
        var a = RandBoxes(27, 4);
        var b = RandBoxes(28, 6);
        var go = RandGrad(29, 4, 6);
        var (gpuA, gpuB) = _gpu!.DistanceBoxIouBackward(go, a, b);
        var (cpuA, cpuB) = _cpu.DistanceBoxIouBackward(go, a, b);
        AssertClose(gpuA, cpuA, tol: 1e-3f);
        AssertClose(gpuB, cpuB, tol: 1e-3f);
    }

    [SkippableFact]
    public void CompleteBoxIouBackward_GpuMatchesCpu()
    {
        SkipIfUnavailable();
        var a = RandBoxes(30, 4);
        var b = RandBoxes(31, 5);
        var go = RandGrad(32, 4, 5);
        var (gpuA, gpuB) = _gpu!.CompleteBoxIouBackward(go, a, b);
        var (cpuA, cpuB) = _cpu.CompleteBoxIouBackward(go, a, b);
        // CIoU has an atan term — fp32 on GPU vs fp64 on CPU gives a slightly
        // larger delta than the other variants.
        AssertClose(gpuA, cpuA, tol: 2e-3f);
        AssertClose(gpuB, cpuB, tol: 2e-3f);
    }
}
