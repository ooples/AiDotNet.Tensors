// Copyright (c) AiDotNet. All rights reserved.

using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// An in-place binary op whose target <c>a</c> is GPU-resident but whose operand <c>b</c> lives on the host (the
/// optimizer update <c>param -= lr * grad</c> after a GPU forward made the parameter resident) must not lose the write.
/// The resident fast path requires both operands on the device, so this case falls to the CPU base implementation,
/// which writes through <c>a.Data</c> -- host storage that the still-authoritative device buffer later overwrites.
/// </summary>
[Collection("VulkanGlobalState")]
public sealed class ResidentInPlaceHostOperandTests : IClassFixture<DirectGpuTensorEngineTestFixture>
{
    private readonly DirectGpuTensorEngineTestFixture _fixture;

    public ResidentInPlaceHostOperandTests(DirectGpuTensorEngineTestFixture fixture) => _fixture = fixture;

    private static Tensor<float> Rand(int length, int seed)
    {
        var rng = new Random(seed);
        var t = new Tensor<float>([length]);
        for (int i = 0; i < length; i++) t[i] = (float)(rng.NextDouble() - 0.5);
        return t;
    }

    public static IEnumerable<object[]> Ops() =>
    [
        ["subtract", 1000], ["add", 1000], ["multiply", 1000],
        ["subtract", 7], ["add", 7], ["multiply", 7],
    ];

    [SkippableTheory]
    [MemberData(nameof(Ops))]
    public void InPlace_ResidentTarget_HostOperand_KeepsTheWrite(string op, int length)
    {
        Skip.IfNot(_fixture.IsAvailable, "No GPU device.");
        IEngine gpu = _fixture.Engine!;
        var x = Rand(length, 1);
        var y = Rand(length, 2);
        var b = Rand(length, 3);

        // `a` is the output of a GPU op: resident, device buffer authoritative.
        var a = gpu.TensorAdd(x, y);
        Assert.True(a.IsGpuResident || a.HasPendingGpuData, "precondition: the target must be resident");
        Assert.False(b.IsGpuResident || b.HasPendingGpuData, "precondition: the operand must be host-only");
        RunAndCheck(gpu, op, a, b, x, y, length);
    }

    /// <summary>The mirror case: a HOST parameter updated in place by a resident update tensor (param -= lr * adamStep,
    /// where the step was computed by GPU ops). Measured in AiDotNet: the version bumps but the values never change.</summary>
    [SkippableTheory]
    [MemberData(nameof(Ops))]
    public void InPlace_HostTarget_ResidentOperand_KeepsTheWrite(string op, int length)
    {
        Skip.IfNot(_fixture.IsAvailable, "No GPU device.");
        IEngine gpu = _fixture.Engine!;
        var p = Rand(length, 4);
        var u = Rand(length, 5);
        var zero = new Tensor<float>([length]);
        var b = gpu.TensorAdd(u, zero);   // resident operand
        Assert.True(b.IsGpuResident || b.HasPendingGpuData, "precondition: the operand must be resident");
        Assert.False(p.IsGpuResident || p.HasPendingGpuData, "precondition: the target must be host-only");
        RunAndCheck(gpu, op, p, b, p.Clone(), zero, length);
    }

    private static void RunAndCheck(IEngine gpu, string op, Tensor<float> a, Tensor<float> b, Tensor<float> x, Tensor<float> y, int length)
    {

        var xs = x.ToArray(); var ys = y.ToArray(); var bs = b.ToArray();
        var expected = new float[length];
        for (int i = 0; i < length; i++)
        {
            float av = xs[i] + ys[i];
            expected[i] = op switch { "subtract" => av - bs[i], "add" => av + bs[i], _ => av * bs[i] };
        }

        switch (op)
        {
            case "subtract": gpu.TensorSubtractInPlace(a, b); break;
            case "add": gpu.TensorAddInPlace(a, b); break;
            default: gpu.TensorMultiplyInPlace(a, b); break;
        }

        // Read it back twice and through a subsequent GPU op: the host view and the device view must both agree.
        var host = a.ToArray();
        var viaDevice = gpu.TensorAdd(a, new Tensor<float>([length])).ToArray();
        for (int i = 0; i < length; i++)
        {
            Assert.True(Math.Abs(expected[i] - host[i]) <= 1e-5, $"{op} host[{i}]: expected {expected[i]}, got {host[i]}");
            Assert.True(Math.Abs(expected[i] - viaDevice[i]) <= 1e-5, $"{op} device[{i}]: expected {expected[i]}, got {viaDevice[i]}");
        }
    }
}
