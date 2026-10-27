using System;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Onnx.Operators;
using Xunit;

namespace AiDotNet.Tensors.Onnx.Tests;

/// <summary>
/// The ONNX index helpers convert through the numeric operations, not <c>Convert.ToDouble(object)</c> (#1559). That
/// needed IConvertible, which <see cref="Half"/> does not implement, so a half-precision index tensor threw
/// InvalidCastException; the int64 "to end" sentinels still clamp to the int range.
/// </summary>
public class OnnxIndexConversionTests
{
    [Fact]
    public void FloatIndicesConvertAndTheInt64SentinelsClamp()
    {
        var t = new Tensor<float>(new[] { 3f, -2f, 9.2e18f, -9.2e18f, 2.5f }, new[] { 5 });
        // 2.5 rounds half to even, as Convert.ToInt32 and Math.Round do.
        Assert.Equal(new[] { 3, -2, int.MaxValue, int.MinValue, 2 }, TensorManipOperators.ToInt32Array(t));
    }

#if NET5_0_OR_GREATER
    [Fact]
    public void HalfIndicesConvertInsteadOfThrowing()
    {
        var t = new Tensor<Half>(new[] { (Half)4f, (Half)(-1f), (Half)7f }, new[] { 3 });
        Assert.Equal(new[] { 4, -1, 7 }, TensorManipOperators.ToInt32Array(t));
    }
#endif
}
