using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// GPU and CPU <c>TensorCartesianProd</c> must produce the same gradient.
/// </summary>
/// <remarks>
/// <para>
/// The two engines build this op differently, and that difference decides whether an explicit tape
/// record is correct. CpuEngine.TensorMeshgrid fills its output spans directly, so nothing on the tape
/// connects the grids back to the caller's inputs and an explicit record is REQUIRED. The GPU override
/// instead returns <c>TensorBroadcastTo(tensors[k].Reshape(rshape), outShape)</c>, which chains from the
/// caller's tensor — so ReshapeBackward and BroadcastToBackward already accumulate into it, and an
/// explicit record on top DOUBLES the gradient.
/// </para>
/// <para>
/// The CPU-only gradcheck sweep cannot see that: it exercises the CpuEngine composition, where the
/// record is right. Only a cross-engine comparison catches it, which is why this exists.
/// </para>
/// </remarks>
[Collection("DirectGpuSerial")]
public class CartesianProdGpuTapeParityTests : IDisposable
{
    private readonly ITestOutputHelper _out;
    private readonly CpuEngine _cpu = new();
    private readonly DirectGpuTensorEngine? _gpu;
    private readonly bool _available;

    public CartesianProdGpuTapeParityTests(ITestOutputHelper o)
    {
        _out = o;
        try { _gpu = new DirectGpuTensorEngine(); _available = _gpu.IsGpuAvailable; }
        catch { _available = false; }
    }

    public void Dispose() { _gpu?.Dispose(); GC.SuppressFinalize(this); }

    private static Tensor<float> Vec(params float[] v)
    {
        var t = new Tensor<float>([v.Length]);
        for (int i = 0; i < v.Length; i++) t[i] = v[i];
        return t;
    }

    private static Tensor<float>[] GradsOn(IEngine engine, Tensor<float> a, Tensor<float> b)
    {
        using var tape = new GradientTape<float>();
        tape.BindEngineIfUnset(engine);
        var prod = engine.TensorCartesianProd([a, b]);
        var loss = engine.ReduceSum(prod, null);
        var grads = tape.ComputeGradients(loss, [a, b]);
        return [grads[a], grads[b]];
    }

    /// <summary>
    /// CLOSED FORM: each element of <c>a</c> appears once per element of <c>b</c> in the product, so
    /// <c>d(sum)/da[i] == b.Length</c>, and symmetrically <c>d(sum)/db[j] == a.Length</c>.
    /// </summary>
    /// <remarks>
    /// Stated as a closed form rather than only as CPU-vs-GPU agreement so that both engines being
    /// wrong by the same factor cannot pass. A doubled record shows up here as exactly 2x.
    /// </remarks>
    [SkippableFact]
    public void CartesianProdGradient_MatchesClosedFormOnBothEngines()
    {
        Skip.If(!_available, "GPU backend not available");
        var a = Vec(0.5f, -1.25f, 2.0f);
        var b = Vec(3.0f, -0.75f);

        var cpu = GradsOn(_cpu, a, b);
        var gpu = GradsOn(_gpu!, a, b);

        _out.WriteLine($"cpu d/da[0]={cpu[0][0]:G9} gpu d/da[0]={gpu[0][0]:G9} (expected {b.Length})");
        _out.WriteLine($"cpu d/db[0]={cpu[1][0]:G9} gpu d/db[0]={gpu[1][0]:G9} (expected {a.Length})");

        for (int i = 0; i < a.Length; i++)
        {
            Assert.True(Math.Abs((double)cpu[0][i] - b.Length) < 1e-5,
                $"CPU d/da[{i}] = {cpu[0][i]:G9}, expected {b.Length}");
            Assert.True(Math.Abs((double)gpu[0][i] - b.Length) < 1e-5,
                $"GPU d/da[{i}] = {gpu[0][i]:G9}, expected {b.Length}. A ratio of exactly 2 means the " +
                $"explicit CartesianProd record is stacking on top of the gradient the " +
                $"meshgrid/reshape/broadcast composition already provides.");
        }
        for (int j = 0; j < b.Length; j++)
        {
            Assert.True(Math.Abs((double)cpu[1][j] - a.Length) < 1e-5,
                $"CPU d/db[{j}] = {cpu[1][j]:G9}, expected {a.Length}");
            Assert.True(Math.Abs((double)gpu[1][j] - a.Length) < 1e-5,
                $"GPU d/db[{j}] = {gpu[1][j]:G9}, expected {a.Length}");
        }
    }
}
