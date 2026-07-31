using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// The backward pass must run on the device the taped data actually lives on.
/// </summary>
/// <remarks>
/// <para>
/// GradientTape used to default to <c>AiDotNetEngine.Current</c> and rely on ops to rebind via
/// BindEngineIfUnset. Only 36 of ~365 differentiable CpuEngine ops make that call, so about nine in
/// ten tapes kept the default — the DirectGpu engine on any auto-detect host. A caller explicitly
/// holding a CpuEngine and working in <c>double</c> got its backward run on the GPU in fp32, which
/// shows up as ~1e-7 per-element drift that reads like FMA noise but is two different backward
/// kernels on two different devices.
/// </para>
/// <para>
/// The fix keys on the DATA rather than on which ops remembered to bind, so it needs no per-op
/// upkeep. These tests pin both directions: CPU data stays on CPU, an explicit binding still wins,
/// and gradients remain numerically correct either way.
/// </para>
/// </remarks>
public class TapeEngineFollowsDataDeviceTests
{
    private readonly ITestOutputHelper _out;
    public TapeEngineFollowsDataDeviceTests(ITestOutputHelper o) => _out = o;

    private static Tensor<double> Rand(int n, int seed)
    {
        var rng = new Random(seed);
        var t = new Tensor<double>([n]);
        for (int i = 0; i < n; i++) t[i] = rng.NextDouble() * 2 - 1;
        return t;
    }

    /// <summary>
    /// A tape whose data is entirely CPU-resident must not dispatch its backward to a GPU engine,
    /// even when the process-wide default engine is one.
    /// </summary>
    [Fact]
    public void CpuResidentTape_DoesNotRunBackwardOnGpu()
    {
        var cpu = new CpuEngine();
        var x = Rand(8, 3);
        var y = Rand(8, 5);

        using var tape = new GradientTape<double>();
        // TensorMultiply is one of the ~90% that never call BindEngineIfUnset, so this is exactly the
        // path that used to inherit AiDotNetEngine.Current.
        var product = cpu.TensorMultiply(x, y);
        var loss = cpu.ReduceSum(product, null);
        var grads = tape.ComputeGradients(loss, [x, y]);

        _out.WriteLine($"tape engine after backward: {tape.Engine.GetType().Name}");
        Assert.False(tape.Engine is DirectGpuTensorEngine,
            $"backward dispatched to {tape.Engine.GetType().Name} for a tape whose data is entirely " +
            $"CPU-resident — a double workload would silently be narrowed to fp32 on the device.");

        // The gradient must also be RIGHT, not merely computed on the expected engine: d(sum(x*y))/dx = y.
        for (int i = 0; i < x.Length; i++)
        {
            Assert.True(Math.Abs(grads[x][i] - y[i]) < 1e-12,
                $"d/dx[{i}] = {grads[x][i]:G17}, expected y[{i}] = {y[i]:G17}");
            Assert.True(Math.Abs(grads[y][i] - x[i]) < 1e-12,
                $"d/dy[{i}] = {grads[y][i]:G17}, expected x[{i}] = {x[i]:G17}");
        }
    }

    /// <summary>An engine that binds itself explicitly still wins over the data-derived choice.</summary>
    [Fact]
    public void ExplicitBindingWinsOverDataInference()
    {
        var cpu = new CpuEngine();
        var x = Rand(4, 7);

        using var tape = new GradientTape<double>();
        tape.BindEngineIfUnset(cpu);
        var loss = cpu.ReduceSum(cpu.TensorMultiply(x, x), null);
        tape.ComputeGradients(loss, [x]);

        Assert.Same(cpu, tape.Engine);
    }

    /// <summary>
    /// A CPU tensor reports no pending GPU data, so the signal the dispatch keys on is not simply
    /// always-true.
    /// </summary>
    /// <remarks>
    /// Without this, <c>CpuResidentTape_DoesNotRunBackwardOnGpu</c> would still pass if
    /// HasPendingGpuData were hardwired to false, which would break the GPU direction silently.
    /// </remarks>
    [Fact]
    public void PlainCpuTensor_ReportsNoPendingGpuData()
    {
        var t = Rand(4, 11);
        Assert.False(t.HasPendingGpuData);
        Assert.False(t.IsGpuResident);
    }

    /// <summary>
    /// A GPU-resident tensor reports pending GPU data, so a GPU tape is still recognised as one.
    /// </summary>
    [SkippableFact]
    public void GpuResidentTensor_ReportsPendingGpuData()
    {
        DirectGpuTensorEngine? gpu = null;
        try { gpu = new DirectGpuTensorEngine(); Skip.If(!gpu.IsGpuAvailable, "no GPU backend"); }
        catch { Skip.If(true, "no GPU backend"); }

        using (gpu)
        {
            var a = new Tensor<float>([64]);
            var b = new Tensor<float>([64]);
            for (int i = 0; i < 64; i++) { a[i] = i * 0.01f; b[i] = i * 0.02f; }
            var result = gpu!.TensorMultiply(a, b);

            _out.WriteLine($"device={result.Device} isGpuResident={result.IsGpuResident} " +
                           $"hasPending={result.HasPendingGpuData}");
            Assert.True(result.HasPendingGpuData,
                "a freshly-computed GPU result reports neither a GPU device nor a pending download, so " +
                "the tape cannot tell that its data is on the device and would move the backward to CPU.");
        }
    }
}
