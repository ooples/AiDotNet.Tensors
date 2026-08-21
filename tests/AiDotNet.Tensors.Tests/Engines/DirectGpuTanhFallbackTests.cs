using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// The CPU fallback inside <see cref="DirectGpuTensorEngine.TensorTanh{T}"/> must not re-enter the
/// virtual <c>Tanh</c> dispatch (ooples/AiDotNet#2010).
/// </summary>
/// <remarks>
/// <para>
/// The cycle these tests guard: <c>DirectGpuTensorEngine.Tanh</c> forwards a contiguous eager tensor
/// to <c>TensorTanh</c> (#775). When the device path yields nothing, <c>TensorTanh</c> fell back to
/// <c>base.TensorTanh</c>, whose CpuEngine body is <c>var r = Tanh(tensor);</c> — a VIRTUAL call
/// that lands back on this engine's <c>Tanh</c> override and round-trips forever.
/// </para>
/// <para>
/// It ends in a StackOverflowException, which .NET does not allow to be caught, so the
/// <c>catch (Exception)</c> guarding the fallback never runs and the process dies outright. A
/// consumer saw exactly that: every call to <c>TensorOperations&lt;T&gt;.Tanh</c> killed the process
/// on a machine whose engine had auto-selected DirectGpu.
/// </para>
/// <para>
/// Why the existing tanh tests did not catch it: on a host with no usable device
/// <c>AiDotNetEngine.Current</c> is CpuEngine, this override is never in play, and the library's own
/// suite pins the CPU engine for determinism. The fault needs a real backend AND a tensor the kernel
/// declines — which is what the fallback is for.
/// </para>
/// </remarks>
[Collection("DirectGpuSerial")]
public class DirectGpuTanhFallbackTests
{
    private static Tensor<T> Filled<T>(int count, Func<int, double> value)
    {
        var tensor = new Tensor<T>(new[] { count });
        for (int i = 0; i < count; i++)
        {
            tensor[i] = (T)Convert.ChangeType(value(i), typeof(T));
        }

        return tensor;
    }

    private static double AsDouble<T>(T value) => Convert.ToDouble(value);

    [SkippableFact]
    public void TensorTanh_WhenThePolicyForcesTheCpuFallback_IsExactRatherThanRecursing()
    {
        using var gpu = new DirectGpuTensorEngine();
        Skip.IfNot(gpu.IsGpuAvailable, "No DirectGpu backend available");

        // PreserveInputType makes ShouldFallbackForPrecision<double>() true, so TryRunUnary returns
        // null and TensorTanh takes the fallback — the exact path that used to recurse forever.
        // This is the supported way to demand the fallback, rather than hoping a kernel declines.
        using var scope = new GpuExecutionPolicyScope(GpuExecutionPolicy.Preserve);

        var input = Filled<double>(64, i => -3.0 + 6.0 * i / 63.0);
        var result = gpu.TensorTanh(input);
        var host = new CpuEngine().TensorTanh(input);

        Assert.Equal(input.Length, result.Length);
        for (int i = 0; i < input.Length; i++)
        {
            // BIT-IDENTICAL to CpuEngine, which is the exact promise of the fix: the fallback must
            // reach the CPU implementation itself. Comparing against Math.Tanh instead would be the
            // wrong oracle — CpuEngine's vectorised tanh is a polynomial approximation and differs
            // from Math.Tanh by about 1.3e-15, so that comparison measures the CPU kernel's accuracy
            // rather than whether the fallback routed correctly.
            Assert.Equal(host[i], result[i]);
        }
    }

    [SkippableFact]
    public void Tanh_WhenThePolicyForcesTheCpuFallback_IsExactRatherThanRecursing()
    {
        using var gpu = new DirectGpuTensorEngine();
        Skip.IfNot(gpu.IsGpuAvailable, "No DirectGpu backend available");

        // The other door into the same cycle: Tanh forwards contiguous eager tensors to TensorTanh,
        // so a consumer calling the plain primitive reached the recursion just as directly.
        using var scope = new GpuExecutionPolicyScope(GpuExecutionPolicy.Preserve);

        var input = Filled<double>(64, i => -3.0 + 6.0 * i / 63.0);
        var result = gpu.Tanh(input);
        var host = new CpuEngine().Tanh(input);

        for (int i = 0; i < input.Length; i++)
        {
            Assert.Equal(host[i], result[i]);
        }
    }

    [SkippableFact]
    public void TensorTanh_UnderTheDefaultSpeedFirstPolicy_ComputesDoubleAtSinglePrecision()
    {
        using var gpu = new DirectGpuTensorEngine();
        Skip.IfNot(gpu.IsGpuAvailable, "No DirectGpu backend available");

        // Pinned because it is surprising and it is DELIBERATE, not the bug above. The package
        // default is GpuAccuracyMode.SpeedFirst, which converts a Tensor<double> across the FP32
        // device boundary — measured error about 2.8e-08, i.e. float epsilon, not double's.
        //
        // Callers who need double to stay double ask for it: GpuExecutionPolicy.Preserve, as the
        // two tests above do, or AIDOTNET_DIRECTGPU_STRICT_FP64=1 for the legacy process-wide
        // switch. This test exists so that contract is discoverable from the test suite rather than
        // from a debugging session.
        var input = Filled<double>(64, i => -3.0 + 6.0 * i / 63.0);
        var result = gpu.TensorTanh(input);

        double worst = 0.0;
        for (int i = 0; i < input.Length; i++)
        {
            worst = Math.Max(worst, Math.Abs(result[i] - Math.Tanh(input[i])));

            // Right answer to single precision — a real tanh, not garbage.
            Assert.True(Math.Abs(result[i] - Math.Tanh(input[i])) < 1e-6,
                $"at {i}: got {result[i]:G17}, expected {Math.Tanh(input[i]):G17}");
        }

        Assert.True(worst > 1e-12,
            $"worst error {worst:E3}: if the default path became exact, the speed-first conversion " +
            "documented here no longer happens and this test should be re-read rather than relaxed");
    }

    [SkippableFact]
    public void TensorTanh_Float_MatchesTheCpuEngine()
    {
        using var gpu = new DirectGpuTensorEngine();
        Skip.IfNot(gpu.IsGpuAvailable, "No DirectGpu backend available");

        // float normally takes the device kernel rather than the fallback, so this pins that the
        // fix left the fast path alone: the two engines must still agree.
        var input = Filled<float>(64, i => -3.0 + 6.0 * i / 63.0);

        var onDevice = gpu.TensorTanh(input);
        var onHost = new CpuEngine().TensorTanh(input);

        for (int i = 0; i < input.Length; i++)
        {
            Assert.True(Math.Abs(AsDouble(onDevice[i]) - AsDouble(onHost[i])) < 1e-5,
                $"at {i}: device {onDevice[i]}, host {onHost[i]}");
        }
    }

    [SkippableFact]
    public void TensorTanh_PreservesLayout()
    {
        using var gpu = new DirectGpuTensorEngine();
        Skip.IfNot(gpu.IsGpuAvailable, "No DirectGpu backend available");

        // CpuEngine.TensorTanh copies Layout across, and the fallback replaces that method rather
        // than merely calling something adjacent to it — so the copy has to survive the fix.
        var input = Filled<double>(32, i => 0.1 * i);
        var expected = input.Layout;

        var result = gpu.TensorTanh(input);

        Assert.Equal(expected, result.Layout);
    }

    [Fact]
    public void CpuEngineTensorTanh_IsStillTheVirtualDispatchTheFallbackMustAvoid()
    {
        // Not a test of the fix, a test of the PREMISE behind it. CpuEngine.TensorTanh reaching
        // tanh through a virtual call is what makes base.TensorTanh unsafe to call from a derived
        // engine whose own Tanh routes back into TensorTanh.
        //
        // If someone later devirtualises CpuEngine.TensorTanh, this assertion is the note that the
        // fallback in DirectGpuTensorEngine could be simplified back.
        var method = typeof(CpuEngine).GetMethod(nameof(CpuEngine.TensorTanh));

        Assert.NotNull(method);
        Assert.True(method!.IsVirtual, "CpuEngine.TensorTanh is expected to be virtual");

        var cpu = new CpuEngine();
        var input = Filled<double>(8, i => 0.25 * i);
        var result = cpu.TensorTanh(input);

        for (int i = 0; i < input.Length; i++)
        {
            Assert.True(Math.Abs(result[i] - Math.Tanh(input[i])) < 1e-12,
                $"at {i}: got {result[i]:G17}");
        }
    }
}
