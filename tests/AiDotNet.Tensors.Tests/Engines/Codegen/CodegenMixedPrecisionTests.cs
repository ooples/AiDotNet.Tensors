// Copyright (c) AiDotNet. All rights reserved.
// Narrow storage, wide arithmetic.
//
// The emitter hardcoded fp32 in 69 places and bindings carried no element type at all, so
// no mixed-precision kernel could be expressed and every such PR had to hand-write one.
// That is a foundation gap, not a missing operator.
//
// Storage is now per BINDING, which is what lets one kernel read fp16 activations against
// fp32 weights -- the common decode shape. Arithmetic stays fp32 throughout: accumulating
// a long reduction in fp16 loses roughly three decimal digits and is never what a caller
// wants from a narrow input.

using System;
using AiDotNet.Tensors.Engines.Compilation.Codegen;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ptx;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Codegen;

public class CodegenMixedPrecisionTests
{
    private static CodegenKernelSpec MatMul(
        int m, int k, int n,
        CodegenElementType aType = CodegenElementType.Float32,
        CodegenElementType bType = CodegenElementType.Float32,
        CodegenElementType outType = CodegenElementType.Float32)
    {
        var space = new CodegenIterationSpace(
            CodegenAxis.Parallel("m", m), CodegenAxis.Parallel("n", n),
            CodegenAxis.Reduce("k", k));

        var a = new CodegenTensorBinding(0, "a", new[] { m, k },
            new[] { CodegenAffineExpr.Axis(0), CodegenAffineExpr.Axis(2) },
            elementType: aType);
        var b = new CodegenTensorBinding(1, "b", new[] { k, n },
            new[] { CodegenAffineExpr.Axis(2), CodegenAffineExpr.Axis(1) },
            elementType: bType);
        var output = new CodegenTensorBinding(2, "out", new[] { m, n },
            new[] { CodegenAffineExpr.Axis(0), CodegenAffineExpr.Axis(1) },
            isOutput: true, elementType: outType);

        return new CodegenKernelSpec("mixed_matmul", space, new[] { a, b }, output,
            new[] { 0, 1 }, CodegenReduceKind.Sum);
    }

    /// <summary>An fp16 binding is two bytes and needs conversion; fp32 is neither.</summary>
    [Theory]
    [InlineData(CodegenElementType.Float32, 4, false)]
    [InlineData(CodegenElementType.Float16, 2, true)]
    [InlineData(CodegenElementType.BFloat16, 2, true)]
    public void ElementSizeAndConversion_FollowTheType(
        CodegenElementType type, int bytes, bool converts)
    {
        var binding = new CodegenTensorBinding(0, "x", new[] { 4 },
            new[] { CodegenAffineExpr.Axis(0) }, elementType: type);

        Assert.Equal(bytes, binding.ElementBytes);
        Assert.Equal(converts, binding.NeedsConversion);
    }

    /// <summary>An fp16 operand must load through a 16-bit load and a widening convert.</summary>
    /// <remarks>
    /// The VECTOR path is disabled here so this covers the scalar form. A unit-stride narrow
    /// binding now vectorises through v2.u32, which is a different instruction and is covered
    /// by NarrowBinding_VectorisesThroughTwoWords.
    /// </remarks>
    [Fact]
    public void Fp16Operand_LoadsNarrowAndWidens()
    {
        string ptx = new PtxAffineEmitter { EnableVectorLoads = false }.Emit(
            MatMul(32, 16, 32, aType: CodegenElementType.Float16), 8, 6);

        Assert.Contains("ld.global.nc.u16", ptx, StringComparison.Ordinal);
        Assert.Contains("cvt.f32.f16", ptx, StringComparison.Ordinal);

        // The address must scale by TWO bytes, not four.
        Assert.Contains("mul.wide.s32", ptx, StringComparison.Ordinal);
        Assert.Contains(", 2;", ptx, StringComparison.Ordinal);
    }

    /// <summary>An fp16 output must narrow on the way out, rounding to nearest.</summary>
    [Fact]
    public void Fp16Output_NarrowsOnStore()
    {
        string ptx = new PtxAffineEmitter().Emit(
            MatMul(32, 16, 32, outType: CodegenElementType.Float16), 8, 6);

        Assert.Contains("cvt.rn.f16.f32", ptx, StringComparison.Ordinal);
        Assert.Contains("st.global.u16", ptx, StringComparison.Ordinal);
    }

    /// <summary>
    /// bf16 is the TOP half of the fp32 pattern, so it must shift rather than use a
    /// conversion instruction -- none exists for it on this architecture.
    /// </summary>
    [Fact]
    public void BFloat16_ShiftsRatherThanConverts()
    {
        string ptx = new PtxAffineEmitter().Emit(
            MatMul(32, 16, 32, aType: CodegenElementType.BFloat16), 8, 6);

        Assert.Contains("shl.b32", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("cvt.f32.bf16", ptx, StringComparison.Ordinal);
    }

    /// <summary>
    /// A MIXED kernel -- fp16 activations, fp32 weights -- is the decode shape, and must
    /// emit both a widening load and a plain one.
    /// </summary>
    [Fact]
    public void MixedOperands_EmitBothLoadForms()
    {
        string ptx = new PtxAffineEmitter { EnableVectorLoads = false }.Emit(
            MatMul(32, 16, 32, aType: CodegenElementType.Float16), 8, 6);

        Assert.Contains("ld.global.nc.u16", ptx, StringComparison.Ordinal);   // fp16 operand
        Assert.Contains("ld.global.nc.f32", ptx, StringComparison.Ordinal);   // fp32 operand
    }

    /// <summary>
    /// The accumulator stays fp32 whatever the operands are. An fp16 accumulator would be
    /// a different, worse operator.
    /// </summary>
    [Fact]
    public void AccumulatorStaysFp32_WhateverTheOperands()
    {
        string ptx = new PtxAffineEmitter().Emit(
            MatMul(32, 16, 32,
                aType: CodegenElementType.Float16,
                bType: CodegenElementType.Float16,
                outType: CodegenElementType.Float16), 8, 6);

        Assert.Contains("add.rn.f32", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("add.rn.f16", ptx, StringComparison.Ordinal);
    }

    /// <summary>
    /// A narrow binding vectorises through v2.u32 -- four halves, the same four elements a
    /// v4.f32 carries -- and never through the f32 form.
    /// </summary>
    /// <remarks>
    /// The f32 form scales by four bytes an element, so on a 16-bit tensor it would read
    /// twice the intended span and return neighbouring data without complaint. That is why
    /// narrow bindings were excluded from vectorising at all; the profile then showed what
    /// the exclusion cost -- 2 sectors per request against fp32's 4, and 66.6% of DRAM peak
    /// against 89.1% -- because 32 lanes at 2 bytes is a 64-byte request, half a cache line.
    /// </remarks>
    [Fact]
    public void NarrowBinding_VectorisesThroughTwoWords()
    {
        var emitter = new PtxAffineEmitter { EnableVectorLoads = true };
        string ptx = emitter.Emit(MatMul(64, 64, 64, bType: CodegenElementType.Float16), 8, 6);

        Assert.Contains("ld.global.nc.v2.u32", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("ld.global.nc.v4.f32", ptx, StringComparison.Ordinal);

        // Each word carries TWO halves, so the upper one is shifted down before widening.
        Assert.Contains("shr.b32", ptx, StringComparison.Ordinal);
        Assert.Contains("cvt.f32.f16", ptx, StringComparison.Ordinal);
        Assert.True(emitter.VectorisedLoads > 0);
    }

    /// <summary>An element type the emitter cannot store is refused, not silently widened.</summary>
    [Theory]
    [InlineData(CodegenElementType.Float64)]
    public void UnsupportedElementType_IsRefused(CodegenElementType type)
    {
        Assert.Throws<ArgumentException>(() => new CodegenTensorBinding(
            0, "x", new[] { 4 }, new[] { CodegenAffineExpr.Axis(0) }, elementType: type));
    }

    /// <summary>
    /// Int32 IS accepted, but only as an index tensor -- it arrived with gather/scatter.
    /// Letting an index buffer masquerade as fp32 is what the separate type prevents; see
    /// CodegenGatherScatterTests for the refusal to read one as an operand.
    /// </summary>
    [Fact]
    public void Int32_IsAcceptedAsAnIndexTensor()
    {
        var binding = new CodegenTensorBinding(0, "ids", new[] { 4 },
            new[] { CodegenAffineExpr.Axis(0) }, elementType: CodegenElementType.Int32);

        Assert.True(binding.IsIndexTensor);
        Assert.Equal(4, binding.ElementBytes);
        Assert.False(binding.NeedsConversion);
    }

    /// <summary>fp32 kernels must be untouched -- no conversion, no 2-byte addressing.</summary>
    [Fact]
    public void Fp32Kernels_AreUnchanged()
    {
        string ptx = new PtxAffineEmitter().Emit(MatMul(32, 16, 32), 8, 6);

        Assert.DoesNotContain("cvt.f32.f16", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("ld.global.nc.u16", ptx, StringComparison.Ordinal);
        Assert.Contains("ld.global.nc.f32", ptx, StringComparison.Ordinal);
    }
}
