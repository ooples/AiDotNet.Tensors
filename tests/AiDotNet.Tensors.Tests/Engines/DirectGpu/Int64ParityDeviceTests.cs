using System;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Tests.TestHelpers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// #1135: the integer parity ops on long, ulong, short, ushort, sbyte, byte and uint tensors run on the device (the raw
/// int64 layout) and equal the CPU element for element -- min / max values, negative and oversized shift counts
/// included -- with the CPU's exceptions for a zero divisor and for Int64 overflow.
/// </summary>
public abstract class Int64ParityDeviceTestsBase
{
    private readonly IDirectGpuTensorEngineFixture _fixture;
    protected Int64ParityDeviceTestsBase(IDirectGpuTensorEngineFixture fixture) => _fixture = fixture;

    // Typed as IEngine: the GPU overrides are reached through the interface (a call on the concrete type can bind to
    // the inherited CPU method).
    private IEngine GpuOrSkip()
    {
        Skip.If(!_fixture.IsAvailable, "No DirectGpu backend available.");
        DirectGpuTensorEngine gpu = _fixture.Engine!;
        Skip.IfNot(gpu.GetBackend() is ITorchParityElementwiseBackend, "This backend has no PyTorch-parity device kernels.");
        return gpu;
    }

    private static long TotalFallbacks() => GpuLaunchProbe.Fallbacks.Sum(entry =>
    {
        int x = entry.IndexOf("x ", StringComparison.Ordinal);
        return x > 0 && long.TryParse(entry.Substring(0, x), out long count) ? count : 0L;
    });

    private static readonly string[] Ops =
        { "and", "or", "xor", "not", "lshift", "rshift", "gcd", "lcm", "floordiv" };

    private static Tensor<T> Run<T>(IEngine e, string op, Tensor<T> a, Tensor<T> b) => op switch
    {
        "and" => e.TensorBitwiseAnd(a, b),
        "or" => e.TensorBitwiseOr(a, b),
        "xor" => e.TensorBitwiseXor(a, b),
        "not" => e.TensorBitwiseNot(a),
        "lshift" => e.TensorBitwiseLeftShift(a, b),
        "rshift" => e.TensorBitwiseRightShift(a, b),
        "gcd" => e.TensorGcd(a, b),
        "lcm" => e.TensorLcm(a, b),
        _ => e.TensorFloorDivide(a, b),
    };

    // Operands: the type's extremes, small values of both signs, and (for shifts) counts that are negative, zero,
    // in range, the width, 63 and >= 64. Divisors avoid zero; overflow cases are tested separately.
    private static (Tensor<T> A, Tensor<T> B) Operands<T>(string op, long min, long max, Func<long, T> make)
    {
        long[] a = { min, max, 0, 1, -1, 7, -7, 12, -18, 100, max / 3, min / 5, 255, 1L << 20 };
        long[] b;
        if (op is "lshift" or "rshift")
            b = new long[] { 1, 3, 0, 63, 64, 65, -1, 8, 16, 31, 32, 200, 5, 2 };
        else if (op is "gcd" or "lcm" or "floordiv")
            b = new long[] { 3, -7, 5, 1, -1, 2, 9, -4, 6, 30, 11, -13, 17, 4 };
        else
            b = new long[] { max, min, -1, 0x55, 0x0F, 3, -2, 128, 1000, 1L << 33, 6, -6, 1, 2 };
        static long Clamp(long v, long lo, long hi) => v < lo ? lo : v > hi ? hi : v;
        if (op is "gcd" or "lcm" or "floordiv")
        {
            // Nonzero divisors in the type's range (|d| for unsigned types); long.MinValue operands overflow on the CPU
            // and are covered by Int64Overflow_ThrowsLikeCpu.
            b = b.Select(v => min == 0 ? Math.Abs(v) : v).ToArray();
            a = a.Select(v => v == long.MinValue ? long.MinValue + 1 : v).ToArray();
        }
        var av = a.Select(v => make(Clamp(v, min, max))).ToArray();
        var bv = b.Select(v => make(Clamp(v, min, max))).ToArray();
        return (new Tensor<T>(av, new[] { av.Length }), new Tensor<T>(bv, new[] { bv.Length }));
    }

    private void CheckType<T>(long min, long max, Func<long, T> make)
    {
        var gpu = GpuOrSkip();
        var cpu = new CpuEngine();
        foreach (var op in Ops)
        {
            var (a, b) = Operands(op, min, max, make);
            if (op is "lcm" && typeof(T) == typeof(long))
            {
                // lcm(long.MaxValue, x) can wrap to long.MinValue, which the CPU's Math.Abs rejects; keep it in range.
                a = new Tensor<T>(a.ToArray().Select((v, i) => i < 2 ? make(1000 + i) : v).ToArray(), a._shape);
            }
            Tensor<T> expected;
            try { expected = Run(cpu, op, a, b); }
            catch (Exception ex) { throw new InvalidOperationException($"{typeof(T).Name} {op}: CPU reference threw {ex.GetType().Name}", ex); }
            long before = TotalFallbacks();
            var got = Run(gpu, op, a, b);
            Assert.True(TotalFallbacks() == before, $"{typeof(T).Name} {op} fell back to the CPU: {string.Join("; ", GpuLaunchProbe.Fallbacks)}");
            var e = expected.ToArray(); var g = got.ToArray();
            for (int i = 0; i < e.Length; i++)
                Assert.True(Equals(e[i], g[i]), $"{typeof(T).Name} {op}[{i}] ({a[i]}, {b[i]}): cpu {e[i]}, gpu {g[i]}");
        }
    }

    [SkippableFact] public void Long_MatchesCpu() => CheckType(long.MinValue, long.MaxValue, v => v);
    [SkippableFact] public void ULong_MatchesCpu() => CheckType(0, long.MaxValue, v => unchecked((ulong)v));
    [SkippableFact] public void Short_MatchesCpu() => CheckType(short.MinValue, short.MaxValue, v => (short)v);
    [SkippableFact] public void UShort_MatchesCpu() => CheckType(0, ushort.MaxValue, v => (ushort)v);
    [SkippableFact] public void SByte_MatchesCpu() => CheckType(sbyte.MinValue, sbyte.MaxValue, v => (sbyte)v);
    [SkippableFact] public void Byte_MatchesCpu() => CheckType(0, byte.MaxValue, v => (byte)v);
    [SkippableFact] public void UInt_MatchesCpu() => CheckType(0, uint.MaxValue, v => (uint)v);

    [SkippableFact]
    public void ULong_HighBit_MatchesCpu()
    {
        // ulong values above long.MaxValue: the CPU computes on their int64 bits (negative), so must the device.
        var gpu = GpuOrSkip();
        var cpu = new CpuEngine();
        var a = new Tensor<ulong>(new[] { ulong.MaxValue, 1UL << 63, (1UL << 63) + 5, 12UL }, new[] { 4 });
        var b = new Tensor<ulong>(new[] { 1UL, 3UL, 7UL, ulong.MaxValue }, new[] { 4 });
        foreach (var op in new[] { "and", "or", "xor", "not", "lshift", "rshift" })
            Assert.Equal(Run(cpu, op, a, b).ToArray(), Run(gpu, op, a, b).ToArray());
    }

    [SkippableFact]
    public void ZeroDivisor_ThrowsLikeCpu()
    {
        var gpu = GpuOrSkip();
        var a = new Tensor<long>(new[] { 5L, 6L }, new[] { 2 });
        var b = new Tensor<long>(new[] { 2L, 0L }, new[] { 2 });
        Assert.Throws<DivideByZeroException>(() => new CpuEngine().TensorFloorDivide(a, b).ToArray());
        Assert.Throws<DivideByZeroException>(() => gpu.TensorFloorDivide(a, b).ToArray());
        var s = new Tensor<short>(new short[] { 5, 6 }, new[] { 2 });
        var z = new Tensor<short>(new short[] { 0, 1 }, new[] { 2 });
        Assert.Throws<DivideByZeroException>(() => gpu.TensorFloorDivide(s, z).ToArray());
    }

    [SkippableTheory]
    [InlineData("gcd")]
    [InlineData("lcm")]
    [InlineData("floordiv")]
    public void Int64Overflow_ThrowsLikeCpu(string op)
    {
        var gpu = GpuOrSkip();
        var a = new Tensor<long>(new[] { 4L, long.MinValue }, new[] { 2 });
        var b = new Tensor<long>(new[] { 2L, op == "floordiv" ? -1L : 3L }, new[] { 2 });
        Assert.ThrowsAny<ArithmeticException>(() => Run(new CpuEngine(), op, a, b).ToArray());
        Assert.Throws<OverflowException>(() => Run(gpu, op, a, b).ToArray());
    }

    // A resident result is already wrapped to its element type, so the next op sees what the CPU's does: shifting left
    // past the width and back right, or right-shifting a bitwise not, differs from the unwrapped 64-bit value.
    private void CheckChainWraps<T>(Func<long, T> make, int width)
    {
        var gpu = GpuOrSkip();
        var cpu = new CpuEngine();
        var a = new Tensor<T>(new long[] { 1, 3, 100, 77, 5 }.Select(make).ToArray(), new[] { 5 });
        var k = new Tensor<T>(Enumerable.Repeat(make(width - 3), 5).ToArray(), new[] { 5 });
        var expected = cpu.TensorBitwiseRightShift(cpu.TensorBitwiseLeftShift(a, k), k).ToArray();
        Assert.Equal(expected, gpu.TensorBitwiseRightShift(gpu.TensorBitwiseLeftShift(a, k), k).ToArray());
        var one = new Tensor<T>(Enumerable.Repeat(make(width - 1), 5).ToArray(), new[] { 5 });
        Assert.Equal(cpu.TensorBitwiseRightShift(cpu.TensorBitwiseNot(a), one).ToArray(),
            gpu.TensorBitwiseRightShift(gpu.TensorBitwiseNot(a), one).ToArray());
    }

    [SkippableFact] public void Short_ChainWraps() => CheckChainWraps(v => (short)v, 16);
    [SkippableFact] public void UShort_ChainWraps() => CheckChainWraps(v => (ushort)v, 16);
    [SkippableFact] public void SByte_ChainWraps() => CheckChainWraps(v => (sbyte)v, 8);
    [SkippableFact] public void Byte_ChainWraps() => CheckChainWraps(v => (byte)v, 8);
    [SkippableFact] public void UInt_ChainWraps() => CheckChainWraps(v => (uint)v, 32);

    [SkippableFact]
    public void ChainedOps_StayResident()
    {
        // A result is left in the raw int64 layout, so the next op reads it on the device without an upload.
        var gpu = GpuOrSkip();
        var cpu = new CpuEngine();
        var a = new Tensor<long>(Enumerable.Range(0, 1000).Select(i => (long)i * 7919 - 3_000_000_000L).ToArray(), new[] { 1000 });
        var b = new Tensor<long>(Enumerable.Range(0, 1000).Select(i => (long)(i % 13) + 1).ToArray(), new[] { 1000 });
        var first = gpu.TensorBitwiseXor(a, b);
        Assert.True(first._gpuBufferContainsRawInt64, "the result is not resident in the raw int64 layout");
        long uploads = GpuLaunchProbe.Uploads;
        var second = gpu.TensorFloorDivide(first, b);   // b uploads again; `first` must not
        long secondUploads = GpuLaunchProbe.Uploads - uploads;
        Assert.True(secondUploads <= 1, $"the chained op uploaded {secondUploads} operands; the resident one should be reused");
        Assert.Equal(cpu.TensorFloorDivide(cpu.TensorBitwiseXor(a, b), b).ToArray(), second.ToArray());
    }
}

/// <summary>The cases above on the machine's default DirectGpu backend.</summary>
[Collection("VulkanGlobalState")]
public class Int64ParityDeviceTests : Int64ParityDeviceTestsBase, IClassFixture<DirectGpuTensorEngineTestFixture>
{
    public Int64ParityDeviceTests(DirectGpuTensorEngineTestFixture fixture) : base(fixture) { }
}

/// <summary>The cases above on the Vulkan backend.</summary>
[Collection("VulkanGlobalState")]
public class Int64ParityVulkanDeviceTests : Int64ParityDeviceTestsBase, IClassFixture<VulkanTensorEngineTestFixture>
{
    public Int64ParityVulkanDeviceTests(VulkanTensorEngineTestFixture fixture) : base(fixture) { }
}

[Collection("DirectGpuSerial")]
public class Int64ParityHipDeviceTests : Int64ParityDeviceTestsBase, IClassFixture<HipTensorEngineTestFixture>
{
    public Int64ParityHipDeviceTests(HipTensorEngineTestFixture fixture) : base(fixture) { }
}
