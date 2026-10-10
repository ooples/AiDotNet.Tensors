using System;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Tests.TestHelpers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// #1112 tier 9: the integer ops (bitwise and / or / xor / not / shifts, gcd, lcm, floor_divide) on int32 tensors run
/// on raw int32 device storage. Each must equal the CPU bit for bit -- including int.MinValue / MaxValue, negative and
/// oversized shift counts and zero operands -- launch device work, record no host fallback, and floor_divide by zero
/// must raise the CPU's DivideByZeroException.
/// </summary>
[Collection("VulkanGlobalState")]
public class TorchParityIntegerDeviceTests : IClassFixture<DirectGpuTensorEngineTestFixture>
{
    private readonly DirectGpuTensorEngineTestFixture _fixture;
    public TorchParityIntegerDeviceTests(DirectGpuTensorEngineTestFixture fixture) => _fixture = fixture;

    private DirectGpuTensorEngine GpuOrSkip()
    {
        Skip.If(!_fixture.IsAvailable, "No DirectGpu backend available.");
        var gpu = _fixture.Engine!;
        Skip.IfNot(gpu.GetBackend() is ITorchParityElementwiseBackend, "These device kernels are CUDA-only.");
        return gpu;
    }

    private static long TotalFallbacks() => GpuLaunchProbe.Fallbacks.Sum(entry =>
    {
        int x = entry.IndexOf("x ", StringComparison.Ordinal);
        return x > 0 && long.TryParse(entry.Substring(0, x), out long count) ? count : 0L;
    });

    private static readonly int[] Edges = { int.MinValue, int.MinValue + 1, -65536, -7, -1, 0, 1, 2, 7, 12, 65536, int.MaxValue - 1, int.MaxValue };

    private static (Tensor<int> A, Tensor<int> B) Operands(bool shifts, bool nonZeroDivisor)
    {
        var rng = new Random(shifts ? 3 : 4);
        int n = Edges.Length * Edges.Length + 2000;
        var a = new Tensor<int>(new[] { n });
        var b = new Tensor<int>(new[] { n });
        int[] shiftCounts = { -65, -1, 0, 1, 5, 31, 32, 33, 63, 64, 65, 1000 };
        for (int i = 0; i < n; i++)
        {
            a[i] = i < Edges.Length * Edges.Length ? Edges[i / Edges.Length] : rng.Next(int.MinValue, int.MaxValue);
            b[i] = shifts ? shiftCounts[i % shiftCounts.Length]
                : i < Edges.Length * Edges.Length ? Edges[i % Edges.Length] : rng.Next(-1000, 1000);
            if (nonZeroDivisor && b[i] == 0) b[i] = 3;
        }
        return (a, b);
    }

    public static TheoryData<string> Ops() => new()
    {
        "and", "or", "xor", "not", "leftShift", "rightShift", "gcd", "lcm", "floorDivide",
    };

    [SkippableTheory]
    [MemberData(nameof(Ops))]
    public void IntegerOp_EqualsCpu_OnDevice(string op)
    {
        var gpu = GpuOrSkip();
        var cpu = new CpuEngine();
        var (a, b) = Operands(op.EndsWith("Shift"), op == "floorDivide");
        Func<IEngine, Tensor<int>> f = op switch
        {
            "and" => e => e.TensorBitwiseAnd(a, b),
            "or" => e => e.TensorBitwiseOr(a, b),
            "xor" => e => e.TensorBitwiseXor(a, b),
            "not" => e => e.TensorBitwiseNot(a),
            "leftShift" => e => e.TensorBitwiseLeftShift(a, b),
            "rightShift" => e => e.TensorBitwiseRightShift(a, b),
            "gcd" => e => e.TensorGcd(a, b),
            "lcm" => e => e.TensorLcm(a, b),
            _ => e => e.TensorFloorDivide(a, b),
        };
        var expected = f(cpu);
        long launches = GpuLaunchProbe.Count, fallbacks = TotalFallbacks();
        var actual = f(gpu);
        _ = actual[0];
        Assert.True(GpuLaunchProbe.Count > launches, $"{op}: no device work launched");
        Assert.Equal(fallbacks, TotalFallbacks());
        Assert.Equal(expected._shape, actual._shape);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(expected[i] == actual[i], $"{op}[{i}] ({a[i]}, {b[i]}): cpu {expected[i]} gpu {actual[i]}");
    }

    [SkippableFact]
    public void FloorDivideByZero_ThrowsLikeCpu()
    {
        var gpu = GpuOrSkip();
        var (a, b) = Operands(false, true);
        b[1234] = 0;
        Assert.Throws<DivideByZeroException>(() => new CpuEngine().TensorFloorDivide(a, b));
        Assert.Throws<DivideByZeroException>(() => gpu.TensorFloorDivide(a, b));
    }
}
