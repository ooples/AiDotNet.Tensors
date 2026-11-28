// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using AiDotNet.Tensors.Engines.Compilation.GeneratedKernels;
#if NET8_0_OR_GREATER
using System.Runtime.Intrinsics;
#endif
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// The compile-time generated elementwise kernels give the same answer on every CPU path (AVX lanes, portable
/// vectors, the scalar tail) as the CUDA kernel they are generated alongside: one rounding for fma, the device's NaN
/// rule for max / min / ReLU, and a sign-bit negation.
/// </summary>
public sealed class GeneratedKernelSemanticsTests
{
    // 37 elements: four AVX blocks of 8, then a scalar tail of 5, so every path writes some of the output.
    private const int Length = 37;

    [Fact]
    public void ResidualScaleAdd_RoundsOnceLikeTheDeviceFma()
    {
        // fma(MaxValue, 2, -MaxValue) = MaxValue exactly; rounding the product first gives +Infinity.
        var branch = Filled(float.MaxValue);
        var residual = Filled(-float.MaxValue);
        var destination = new float[Length];

        CompiledFusedKernels.ResidualScaleAdd(branch, residual, 2.0f, destination);

        for (int i = 0; i < Length; i++)
            Assert.Equal(float.MaxValue, destination[i]);
    }

    [Fact]
    public void AddRelu_MapsNaNToZeroOnEveryLane()
    {
        var a = Filled(float.NaN);
        var b = Filled(1.0f);
        var destination = new float[Length];

        CompiledFusedKernels.AddRelu(a, b, destination);

        for (int i = 0; i < Length; i++)
            Assert.Equal(0.0f, destination[i]);
    }

    [Fact]
    public void ScalarHelpers_FollowTheDeviceNaNRule()
    {
        Assert.Equal(2.0f, GeneratedKernelSimdMath.MaxScalar(float.NaN, 2.0f));
        Assert.Equal(2.0f, GeneratedKernelSimdMath.MaxScalar(2.0f, float.NaN));
        Assert.Equal(-1.0f, GeneratedKernelSimdMath.MinScalar(float.NaN, -1.0f));
        Assert.Equal(-1.0f, GeneratedKernelSimdMath.MinScalar(-1.0f, float.NaN));
        Assert.True(float.IsNaN(GeneratedKernelSimdMath.MaxScalar(float.NaN, float.NaN)));
        Assert.Equal(3.0f, GeneratedKernelSimdMath.MaxScalar(3.0f, -4.0f));
        Assert.Equal(-4.0f, GeneratedKernelSimdMath.MinScalar(3.0f, -4.0f));
        Assert.Equal(float.MaxValue, GeneratedKernelSimdMath.FmaScalar(float.MaxValue, 2.0f, -float.MaxValue));
    }

    [Fact]
    public void PortableHelpers_FollowTheDeviceNaNRule()
    {
        var nan = new System.Numerics.Vector<float>(float.NaN);
        var two = new System.Numerics.Vector<float>(2.0f);
        Assert.Equal(two, GeneratedKernelSimdMath.MaxPortable(nan, two));
        Assert.Equal(two, GeneratedKernelSimdMath.MaxPortable(two, nan));
        Assert.Equal(two, GeneratedKernelSimdMath.MinPortable(nan, two));
        Assert.Equal(two, GeneratedKernelSimdMath.MinPortable(two, nan));
        Assert.Equal(System.Numerics.Vector<float>.Zero, GeneratedKernelSimdMath.ReluPortable(nan));
    }

#if NET8_0_OR_GREATER
    [Fact]
    public void AvxHelpers_FollowTheDeviceNaNRuleAndFlipTheSignBit()
    {
        Skip.IfNot(System.Runtime.Intrinsics.X86.Avx.IsSupported, "AVX is not available on this CPU.");
        var nan = System.Runtime.Intrinsics.Vector256.Create(float.NaN);
        var two = System.Runtime.Intrinsics.Vector256.Create(2.0f);
        Assert.Equal(two, GeneratedKernelSimdMath.Max256(nan, two));
        Assert.Equal(two, GeneratedKernelSimdMath.Max256(two, nan));
        Assert.Equal(two, GeneratedKernelSimdMath.Min256(nan, two));
        Assert.Equal(two, GeneratedKernelSimdMath.Min256(two, nan));

        var negatedZero = GeneratedKernelSimdMath.Negate256(System.Runtime.Intrinsics.Vector256<float>.Zero);
        for (int lane = 0; lane < 8; lane++)
            Assert.True(float.IsNegative(negatedZero.GetElement(lane)), $"lane {lane} kept +0");

        var fused = GeneratedKernelSimdMath.Fma256(
            System.Runtime.Intrinsics.Vector256.Create(float.MaxValue), two,
            System.Runtime.Intrinsics.Vector256.Create(-float.MaxValue));
        Assert.Equal(System.Runtime.Intrinsics.Vector256.Create(float.MaxValue), fused);
    }
#endif

    [Fact]
    public void CudaSource_EmitsTheSwishOperandOnce()
    {
        // The operand goes through a device helper; a textual x / (1 + exp(-x)) doubled it per nested swish.
        string source = CompiledFusedKernels.AffineSwishCudaSource;
        Assert.Contains("aidn_swish(", source);
        Assert.DoesNotContain("expf(-((", source);
        // 64-bit grid-stride indices.
        Assert.Contains("long long idx", source);
    }

    private static float[] Filled(float value)
    {
        var data = new float[Length];
        for (int i = 0; i < Length; i++) data[i] = value;
        return data;
    }
}
