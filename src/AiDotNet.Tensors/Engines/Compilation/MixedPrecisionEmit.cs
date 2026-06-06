using System;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Compilation;

/// <summary>
/// Forward emission for FP16 activation storage (Tensors #555 Phase 2c,
/// docs/fp16-activation-storage-design.md). Under an active FP16 <see cref="Gpu.AutocastScope"/>, an
/// autocast-eligible op (matmul/FF) is recorded onto the lazy graph as a cast-with-backward triple:
///
///   a(fp32) ─┐
///            ├─ down-cast ─▶ a(fp16), b(fp16) ─▶ MatMul[fp16] ─▶ c(fp16) ─▶ up-cast ─▶ c(fp32)
///   b(fp32) ─┘
///
/// The matmul OUTPUT node is a <see cref="Tensor{Half}"/> — i.e. the dominant activation buffer is 2
/// bytes, which is the resident-memory win the FP16-compute-only autocast could not give (it kept FP32
/// buffers). The cast nodes carry the verified <see cref="MixedPrecisionCast"/> bridge so
/// <see cref="MixedPrecisionGraphBackward"/> backprops across the dtype boundary; the consuming FP32 op
/// sees an ordinary FP32 tensor.
///
/// <para>NOTE: this graph is mixed-dtype and must be consumed by a mixed-dtype-aware backward
/// (<see cref="MixedPrecisionGraphBackward"/>), NOT the single-type <c>CompiledTrainingPlan&lt;float&gt;</c>
/// — wiring it into that plan's replay is the remaining production step. Gated behind autocast; the
/// default FP32 recording path is untouched.</para>
/// </summary>
internal static class MixedPrecisionEmit
{
    /// <summary>
    /// True when graph tracing should emit FP16 activation buffers: an FP16 autocast scope is active.
    /// </summary>
    public static bool ShouldEmitFp16<T>() =>
        typeof(T) == typeof(float)
        && Gpu.AutocastScope.IsEnabled
        && Gpu.AutocastScope.ActivePrecision == Gpu.PrecisionMode.Float16;

    /// <summary>
    /// Emits <c>a · b</c> onto <paramref name="scope"/> with the matmul activation stored as FP16.
    /// Returns the FP32 output tensor (the up-cast result). <paramref name="execHalfMatMul"/> runs the
    /// actual FP16 matmul during realize (on GPU this is <c>Hgemm</c>; on CPU a generic Half matmul).
    /// </summary>
    public static Tensor<float> MatMul(
        LazyTensorScope scope,
        Tensor<float> a,
        Tensor<float> b,
        int[] outShape,
        string opName = "TensorMatMul")
    {
        if (scope is null) throw new ArgumentNullException(nameof(scope));

        // 1. Down-cast inputs FP32 -> FP16. Backward: bridge the FP16 grad up to the FP32 input space.
        var aH = scope.RecordCrossTypeWithBackward<float, Half>(
            LazyNodeType.Custom, opName + ".castA16", a, a._shape,
            (e, o) => MixedPrecisionCast.CastToFp16(a).AsSpan().CopyTo(o.AsWritableSpan()),
            (gradOut, input, output, state, e) => MixedPrecisionCast.CastToFp16Backward(gradOut));

        var bH = scope.RecordCrossTypeWithBackward<float, Half>(
            LazyNodeType.Custom, opName + ".castB16", b, b._shape,
            (e, o) => MixedPrecisionCast.CastToFp16(b).AsSpan().CopyTo(o.AsWritableSpan()),
            (gradOut, input, output, state, e) => MixedPrecisionCast.CastToFp16Backward(gradOut));

        // 2. FP16 matmul — the OUTPUT BUFFER is Half (the 2-byte activation = the memory win).
        var cH = scope.RecordBinary<Half>(
            LazyNodeType.MatMul, opName + ".fp16", aH, bH, outShape,
            (e, o) =>
            {
                var r = e.TensorMatMul(aH, bH);
                r.AsSpan().CopyTo(o.AsWritableSpan());
            },
            BackwardFunctions<Half>.MatMulBackward);

        // 3. Up-cast FP16 -> FP32 for the next FP32 op. Backward: bridge the FP32 grad down to FP16.
        var cF = scope.RecordCrossTypeWithBackward<Half, float>(
            LazyNodeType.Custom, opName + ".castOut32", cH, outShape,
            (e, o) => MixedPrecisionCast.CastToFp32(cH).AsSpan().CopyTo(o.AsWritableSpan()),
            (gradOut, input, output, state, e) => MixedPrecisionCast.CastToFp32Backward(gradOut));

        return cF;
    }
}
