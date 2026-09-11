using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Gpu;

/// <summary>
/// <see cref="GpuExecutionPolicy.Preserve"/> keeps an exact numeric type exact, end to end.
/// </summary>
/// <remarks>
/// <para>
/// The default policy is SpeedFirst, and it converts every ordinary public type through FP32 -
/// <c>float</c>, <c>double</c>, <c>int</c>, <c>long</c> and <c>decimal</c> alike. That is deliberate
/// and <see cref="GpuPrecisionPolicyTests.SpeedFirst_DefaultConvertsEveryOrdinaryPublicTypeThroughFp32"/>
/// asserts it by name. The consequence is easy to walk into and worth stating plainly: on a machine
/// with a GPU, <c>0.1234567890123456m + 1e-16m</c> returns <c>0.1234568</c> under the default policy,
/// and <c>16777217</c> comes back as <c>16777216</c> for <c>int</c> and <c>long</c>, since
/// <c>2^24 + 1</c> is the first integer FP32 cannot represent.
/// </para>
/// <para>
/// <c>Preserve</c> is the documented way out, and these pin that it actually delivers exact values
/// rather than merely planning to. <see cref="GpuPrecisionPolicyTests"/> covers the planner's
/// decision against a fake backend; this covers the numbers that come back from the real current
/// engine, which is what a caller choosing <c>decimal</c> is relying on.
/// </para>
/// <para>
/// Only the <c>Preserve</c> half is asserted. The SpeedFirst behaviour depends on a GPU actually
/// being present - on a CPU-only host the current engine is already exact - so asserting the lossy
/// side would encode the test machine rather than the contract.
/// </para>
/// </remarks>
public class ExactTypePrecisionTests
{
    private readonly IEngine _engine = AiDotNetEngine.Current;

    /// <summary>The first integer FP32 cannot represent.</summary>
    private const int FirstUnrepresentableInFloat32 = 16777217;

    [Fact]
    public void Preserve_KeepsDecimalAdditionExact()
    {
        var a = new Tensor<decimal>(new[] { 0.1234567890123456m }, new[] { 1 });
        var b = new Tensor<decimal>(new[] { 0.0000000000000001m }, new[] { 1 });

        using var policy = new GpuExecutionPolicyScope(GpuExecutionPolicy.Preserve);
        var sum = _engine.TensorAdd(a, b);

        Assert.Equal(0.1234567890123456m + 0.0000000000000001m, sum[0]);
    }

    [Fact]
    public void Preserve_KeepsDecimalMultiplicationExact()
    {
        var a = new Tensor<decimal>(new[] { 1.0000000000000001m }, new[] { 1 });
        var b = new Tensor<decimal>(new[] { 3.0m }, new[] { 1 });

        using var policy = new GpuExecutionPolicyScope(GpuExecutionPolicy.Preserve);
        var product = _engine.TensorMultiply(a, b);

        Assert.Equal(1.0000000000000001m * 3.0m, product[0]);
    }

    [Fact]
    public void Preserve_KeepsIntegersAboveTheFloat32Limit()
    {
        var a = new Tensor<int>(new[] { FirstUnrepresentableInFloat32 }, new[] { 1 });
        var b = new Tensor<int>(new[] { 0 }, new[] { 1 });

        using var policy = new GpuExecutionPolicyScope(GpuExecutionPolicy.Preserve);
        var sum = _engine.TensorAdd(a, b);

        Assert.Equal(FirstUnrepresentableInFloat32, sum[0]);
    }

    [Fact]
    public void Preserve_KeepsLongsWellAboveTheFloat32Limit()
    {
        // Past 2^53, where even FP64 would start losing integers.
        const long large = 9007199254740993L;
        var a = new Tensor<long>(new[] { large }, new[] { 1 });
        var b = new Tensor<long>(new[] { 0L }, new[] { 1 });

        using var policy = new GpuExecutionPolicyScope(GpuExecutionPolicy.Preserve);
        var sum = _engine.TensorAdd(a, b);

        Assert.Equal(large, sum[0]);
    }
}
