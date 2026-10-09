#if !NETFRAMEWORK
// GpuCpuCorrectnessFixture (GpuCpuCorrectnessTests.cs) is compiled only off .NET Framework.
using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// GPU-vs-CPU accuracy for the PyTorch-parity ops the generic differential harness cannot drive (constant
/// constructors, integer-valued bitwise/gcd inputs, shape-coupled convolution, pooling and sampling operands).
/// Composed ops run on the device's primitives; host fall-throughs must still agree exactly.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class TorchParityGpuCorrectnessTests : IClassFixture<GpuCpuCorrectnessFixture>
{
    private readonly GpuCpuCorrectnessFixture _fixture;

    public TorchParityGpuCorrectnessTests(GpuCpuCorrectnessFixture fixture) => _fixture = fixture;

    // ((i * mul mod m) - shift) * scale: deterministic, distinct-enough values.
    private static Tensor<float> W(int[] shape, int mod, int mul, int shift, float scale = 0.1f)
        => new Tensor<float>(Enumerable.Range(0, shape.Aggregate(1, (a, b) => a * b)).Select(i => (i * mul % mod - shift) * scale).ToArray(), shape);

    private static Tensor<float> Ints(int[] shape, int mod, int mul, int shift)
        => new Tensor<float>(Enumerable.Range(0, shape.Aggregate(1, (a, b) => a * b)).Select(i => (float)(i * mul % mod - shift)).ToArray(), shape);

    // Bitwise ops and gcd/lcm are integer-only, as in PyTorch.
    private static Tensor<int> IntsI(int[] shape, int mod, int mul, int shift)
        => new Tensor<int>(Enumerable.Range(0, shape.Aggregate(1, (a, b) => a * b)).Select(i => i * mul % mod - shift).ToArray(), shape);

    private static float[] Flat(Tensor<float> t) => t.ToArray();

    private static float[] FlatI(Tensor<int> t) => t.ToArray().Select(v => (float)v).ToArray();

    private static readonly Dictionary<string, (Func<IEngine, float[]> Run, double Tolerance)> Cases = new()
    {
        ["TensorArange"] = (e => Flat(e.TensorArange<float>(-1.5, 2.2, 0.5)), 0),
        ["TensorRange"] = (e => Flat(e.TensorRange<float>(1, 3, 0.25)), 0),
        ["TensorLogspace"] = (e => Flat(e.TensorLogspace<float>(-1, 2, 7)), 1e-6),
        ["TensorHannWindow"] = (e => Flat(e.TensorHannWindow<float>(9)), 1e-6),
        ["TensorHammingWindow"] = (e => Flat(e.TensorHammingWindow<float>(9, periodic: false)), 1e-6),
        ["TensorBlackmanWindow"] = (e => Flat(e.TensorBlackmanWindow<float>(9)), 1e-6),
        ["TensorBartlettWindow"] = (e => Flat(e.TensorBartlettWindow<float>(9)), 1e-6),
        ["TensorKaiserWindow"] = (e => Flat(e.TensorKaiserWindow<float>(9, beta: 8)), 1e-6),
        ["TensorTrilIndices"] = (e => Flat(e.TensorTrilIndices<float>(4, 5, 1)), 0),
        ["TensorTriuIndices"] = (e => Flat(e.TensorTriuIndices<float>(4, 5, -1)), 0),
        ["TensorBitwiseAnd"] = (e => FlatI(e.TensorBitwiseAnd(IntsI(new[] { 4, 6 }, 29, 7, 9), IntsI(new[] { 4, 6 }, 31, 5, 4))), 0),
        ["TensorBitwiseOr"] = (e => FlatI(e.TensorBitwiseOr(IntsI(new[] { 4, 6 }, 29, 7, 9), IntsI(new[] { 4, 6 }, 31, 5, 4))), 0),
        ["TensorBitwiseXor"] = (e => FlatI(e.TensorBitwiseXor(IntsI(new[] { 4, 6 }, 29, 7, 9), IntsI(new[] { 4, 6 }, 31, 5, 4))), 0),
        ["TensorBitwiseNot"] = (e => FlatI(e.TensorBitwiseNot(IntsI(new[] { 4, 6 }, 29, 7, 9))), 0),
        ["TensorBitwiseLeftShift"] = (e => FlatI(e.TensorBitwiseLeftShift(IntsI(new[] { 4, 6 }, 29, 7, 9), IntsI(new[] { 4, 6 }, 4, 3, 0))), 0),
        ["TensorBitwiseRightShift"] = (e => FlatI(e.TensorBitwiseRightShift(IntsI(new[] { 4, 6 }, 29, 7, 9), IntsI(new[] { 4, 6 }, 4, 3, 0))), 0),
        ["TensorGcd"] = (e => FlatI(e.TensorGcd(IntsI(new[] { 4, 6 }, 29, 7, 9), IntsI(new[] { 4, 6 }, 31, 5, 4))), 0),
        ["TensorLcm"] = (e => FlatI(e.TensorLcm(IntsI(new[] { 4, 6 }, 13, 7, 6), IntsI(new[] { 4, 6 }, 11, 5, 4))), 0),
        ["TensorAddmv"] = (e => Flat(e.TensorAddmv(W(new[] { 4 }, 7, 3, 3), W(new[] { 4, 5 }, 17, 5, 8), W(new[] { 5 }, 11, 4, 5), 0.5, 2)), 1e-5),
        ["TensorBilinear"] = (e => Flat(e.TensorBilinear(W(new[] { 3, 4 }, 13, 5, 6), W(new[] { 3, 2 }, 7, 3, 3), W(new[] { 5, 4, 2 }, 19, 7, 9), W(new[] { 5 }, 5, 2, 2))), 1e-5),
        ["TensorConvolution"] = (e => Flat(e.TensorConvolution(W(new[] { 1, 4, 3, 4 }, 23, 7, 11), W(new[] { 4, 3, 2, 3 }, 19, 5, 9), W(new[] { 6 }, 7, 3, 3),
            new[] { 2, 1 }, new[] { 1, 0 }, new[] { 1, 2 }, transposed: true, outputPadding: new[] { 1, 0 }, groups: 2)), 1e-5),
        ["TensorConvTranspose1D"] = (e => Flat(e.TensorConvTranspose1D(W(new[] { 2, 3, 5 }, 23, 7, 11), W(new[] { 3, 2, 3 }, 19, 5, 9), 2, 1, 1)), 1e-5),
        ["TensorConvTbc"] = (e => Flat(e.TensorConvTbc(W(new[] { 6, 2, 3 }, 23, 7, 11), W(new[] { 3, 3, 4 }, 19, 5, 9), W(new[] { 4 }, 7, 3, 3), 1)), 1e-5),
        ["TensorDiagonalScatter"] = (e => Flat(e.TensorDiagonalScatter(W(new[] { 3, 4 }, 13, 5, 6), W(new[] { 3 }, 7, 3, 3), 1)), 0),
        ["TensorAdaptiveAvgPool3D"] = (e => Flat(e.TensorAdaptiveAvgPool3D(W(new[] { 1, 2, 3, 4, 5 }, 127, 37, 60), new[] { 2, 3, 2 })), 1e-6),
        ["TensorAdaptiveMaxPool3D"] = (e => Flat(e.TensorAdaptiveMaxPool3D(W(new[] { 1, 2, 3, 4, 5 }, 127, 37, 60), new[] { 2, 3, 2 })), 0),
        // Indices as a max pool emits them (plane-local, a leading channel axis); an index repeated within a plane keeps
        // the last value, as in PyTorch.
        ["TensorMaxUnpool"] = (e => Flat(e.TensorMaxUnpool(W(new[] { 2, 3 }, 13, 5, 6),
            new Tensor<int>(new[] { 0, 3, 5, 1, 1, 4 }, new[] { 2, 3 }), new[] { 6 })), 0),
        ["TensorGridSample3D"] = (e => Flat(e.TensorGridSample3D(W(new[] { 1, 2, 3, 4, 5 }, 37, 7, 18), W(new[] { 1, 2, 3, 2, 3 }, 29, 11, 14, 0.09f),
            GridSampleMode.Bilinear, GridSamplePadding.Reflection, true)), 1e-6),
        ["TensorNonzeroStatic"] = (e => e.TensorNonzeroStatic(Ints(new[] { 4, 6 }, 5, 3, 2), 30).ToArray().Select(i => (float)i).ToArray(), 0),
        ["TensorViewAsComplex"] = (e => Flat(e.TensorViewAsReal(e.TensorViewAsComplex(W(new[] { 4, 3, 2 }, 23, 7, 11)))), 0),
    };

    public static IEnumerable<object[]> CaseNames() => Cases.Keys.Select(k => new object[] { k });

    [SkippableTheory]
    [MemberData(nameof(CaseNames))]
    public void GpuMatchesCpu(string op)
    {
        Skip.IfNot(_fixture.IsGpuReady, "No DirectGpu backend available on this system.");
        var (run, tolerance) = Cases[op];
        var cpu = run(_fixture.Cpu);
        var gpu = run(_fixture.Gpu);
        Assert.Equal(cpu.Length, gpu.Length);
        for (int i = 0; i < cpu.Length; i++)
            Assert.True(Math.Abs(cpu[i] - gpu[i]) <= tolerance * Math.Max(1, Math.Abs(cpu[i])),
                $"{op}[{i}]: cpu {cpu[i]:R}, gpu {gpu[i]:R}");
    }
}
#endif
