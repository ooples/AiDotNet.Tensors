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
    /// Opt-in master switch for FP16 ACTIVATION STORAGE (the buffer-level win), distinct from the
    /// shipped FP16-compute autocast. Read once: <c>AIDOTNET_FP16_ACTIVATIONS=1</c>. It is separate from
    /// the autocast flag because emitting <see cref="Tensor{Half}"/> activation nodes produces a
    /// mixed-dtype graph that ONLY a mixed-dtype-aware backward (<see cref="MixedPrecisionGraphBackward"/>)
    /// can consume — the single-type fused <c>CompiledTrainingPlan&lt;float&gt;</c> must NOT see it. So a
    /// caller turns this on exactly when it drives the mixed-dtype path. Default off ⇒ existing autocast
    /// (FP16 compute / FP32 storage) and the fused plan are byte-identical.
    /// </summary>
    /// <summary>Test-only override; null ⇒ read the env var. Not part of the public API.</summary>
    internal static bool? TestOverrideEnabled;

    public static bool Fp16ActivationStorageEnabled =>
        TestOverrideEnabled ?? (Environment.GetEnvironmentVariable("AIDOTNET_FP16_ACTIVATIONS") == "1");

    /// <summary>
    /// True when graph tracing should emit FP16 activation buffers: an FP16 autocast scope is active.
    /// </summary>
    public static bool ShouldEmitFp16<T>() =>
        typeof(T) == typeof(float)
        && Gpu.AutocastScope.IsEnabled
        && Gpu.AutocastScope.ActivePrecision == Gpu.PrecisionMode.Float16;

    /// <summary>Both the autocast scope and the activation-storage opt-in are active.</summary>
    public static bool ActivationStorageActive<T>() => Fp16ActivationStorageEnabled && ShouldEmitFp16<T>();

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
        if (a is null) throw new ArgumentNullException(nameof(a));
        if (b is null) throw new ArgumentNullException(nameof(b));
        if (outShape is null) throw new ArgumentNullException(nameof(outShape));

        // 1. Get inputs as FP16. CRITICAL for the memory win: if an input is already the up-cast output
        // of a prior FP16 op (a matmul-stack intermediate), REUSE that op's existing Half tensor instead
        // of down-casting the float copy again. This keeps activations FP16 across the whole eligible
        // region — the intermediate float up-cast becomes unreachable from the loss and is dropped by the
        // plan's topo-from-output (so it's never resident). Without this, every matmul kept BOTH its Half
        // output AND a float up-cast, so the net footprint went UP, not down (measured: +1GB vs FP32).
        var aH = ToFp16Input(scope, a, opName + ".castA16");
        var bH = ToFp16Input(scope, b, opName + ".castB16");

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
        return UpCast(scope, cH, outShape, opName + ".castOut32");
    }

    /// <summary>
    /// Emits a unary FP16-NATIVE op (GELU / ReLU / activation / norm-as-unary) so its activation stays
    /// <see cref="Half"/> end-to-end. This is the keystone of the resident-memory win
    /// (docs/fp16-activation-storage-design.md §"SEVENTH FINDING"): a transformer separates matmuls with
    /// FP32 ops, and the EARLIER matmul-only emission still up-cast each output to an FP32 tensor that the
    /// consuming op then SAVED for its own backward — so the dominant activation stayed FP32 and footprint
    /// went UP. By taking the input as Half (reusing the upstream FP16 op's output, never re-widening),
    /// running the op's FP32 math on a transient up-cast-in-realize copy, and storing BOTH the saved input
    /// and the output as Half, the activation chain never widens. The transient FP32 inside realize is freed
    /// immediately, so the saved-activation set is genuinely Half.
    /// <para>Returns the FP32 up-cast of the Half output for the next consumer. When that consumer is also
    /// FP16-native, its <see cref="ToFp16Input"/> reuses the Half output directly and this up-cast is left
    /// dead (unreachable from the loss) → dropped by the plan's topo-from-output, so it is never resident.</para>
    /// </summary>
    /// <param name="fp32Forward">Runs the op's FP32 math: <c>(engine, inputFp32) =&gt; outputFp32</c>.</param>
    /// <param name="fp32Backward">The op's FP32 backward: <c>(engine, gradOutFp32, inputFp32) =&gt; gradInFp32</c>.</param>
    public static Tensor<float> Unary(
        LazyTensorScope scope,
        Tensor<float> x,
        int[] outShape,
        string opName,
        LazyNodeType nodeType,
        Func<IEngine, Tensor<float>, Tensor<float>> fp32Forward,
        Func<IEngine, Tensor<float>, Tensor<float>, Tensor<float>> fp32Backward)
    {
        if (scope is null) throw new ArgumentNullException(nameof(scope));
        if (x is null) throw new ArgumentNullException(nameof(x));
        if (outShape is null) throw new ArgumentNullException(nameof(outShape));
        if (fp32Forward is null) throw new ArgumentNullException(nameof(fp32Forward));
        if (fp32Backward is null) throw new ArgumentNullException(nameof(fp32Backward));

        var xH = ToFp16Input(scope, x, opName + ".castIn16");

        var yH = scope.RecordUnary<Half>(
            nodeType, opName + ".fp16", xH, outShape,
            (e, o) =>
            {
                // FP16-native realize: up-cast the Half activation only for the FP32 math, then store the
                // result back as Half so the saved activation chain never widens to FP32.
                var xf = MixedPrecisionCast.CastToFp32(xH);
                var yf = fp32Forward(e, xf);
                MixedPrecisionCast.CastToFp16(yf).AsSpan().CopyTo(o.AsWritableSpan());
            },
            (gradOutH, inputs, output, state, e, grads) =>
            {
                // Backward in FP32: up-cast the saved Half input + incoming Half grad, run the op's FP32
                // gradient, down-cast the input-grad back to Half and accumulate into the Half grad map.
                var inH = inputs[0];
                var gradOutF = MixedPrecisionCast.CastToFp32(gradOutH);
                var inF = MixedPrecisionCast.CastToFp32(inH);
                var gradInF = fp32Backward(e, gradOutF, inF);
                Autodiff.DifferentiableOps.AccumulateGrad(grads, inH, MixedPrecisionCast.CastToFp16(gradInF), e);
            });

        return UpCast(scope, yH, outShape, opName + ".castOut32");
    }

    /// <summary>
    /// Emits a binary FP16-NATIVE elementwise op (residual <c>Add</c>, elementwise <c>Multiply</c>, ...) so
    /// both operand activations and the output stay <see cref="Half"/>. Same rationale and mechanics as
    /// <see cref="Unary"/>; the backward returns the gradient for EACH input, each down-cast to Half and
    /// accumulated into the matching node.
    /// </summary>
    /// <param name="fp32Backward">
    /// The op's FP32 backward: <c>(engine, gradOutFp32, aFp32, bFp32) =&gt; (gradAFp32, gradBFp32)</c>.
    /// </param>
    public static Tensor<float> Binary(
        LazyTensorScope scope,
        Tensor<float> a,
        Tensor<float> b,
        int[] outShape,
        string opName,
        LazyNodeType nodeType,
        Func<IEngine, Tensor<float>, Tensor<float>, Tensor<float>> fp32Forward,
        Func<IEngine, Tensor<float>, Tensor<float>, Tensor<float>, (Tensor<float> gradA, Tensor<float> gradB)> fp32Backward)
    {
        if (scope is null) throw new ArgumentNullException(nameof(scope));
        if (a is null) throw new ArgumentNullException(nameof(a));
        if (b is null) throw new ArgumentNullException(nameof(b));
        if (outShape is null) throw new ArgumentNullException(nameof(outShape));
        if (fp32Forward is null) throw new ArgumentNullException(nameof(fp32Forward));
        if (fp32Backward is null) throw new ArgumentNullException(nameof(fp32Backward));

        var aH = ToFp16Input(scope, a, opName + ".castA16");
        var bH = ToFp16Input(scope, b, opName + ".castB16");

        var cH = scope.RecordBinary<Half>(
            nodeType, opName + ".fp16", aH, bH, outShape,
            (e, o) =>
            {
                var af = MixedPrecisionCast.CastToFp32(aH);
                var bf = MixedPrecisionCast.CastToFp32(bH);
                var cf = fp32Forward(e, af, bf);
                MixedPrecisionCast.CastToFp16(cf).AsSpan().CopyTo(o.AsWritableSpan());
            },
            (gradOutH, inputs, output, state, e, grads) =>
            {
                var inA = inputs[0];
                var inB = inputs[1];
                var gradOutF = MixedPrecisionCast.CastToFp32(gradOutH);
                var (gradAF, gradBF) = fp32Backward(
                    e, gradOutF, MixedPrecisionCast.CastToFp32(inA), MixedPrecisionCast.CastToFp32(inB));
                Autodiff.DifferentiableOps.AccumulateGrad(grads, inA, MixedPrecisionCast.CastToFp16(gradAF), e);
                Autodiff.DifferentiableOps.AccumulateGrad(grads, inB, MixedPrecisionCast.CastToFp16(gradBF), e);
            });

        return UpCast(scope, cH, outShape, opName + ".castOut32");
    }

    /// <summary>
    /// Records the FP16→FP32 up-cast that bridges a Half activation node back to the FP32 consumer surface,
    /// carrying the verified <see cref="MixedPrecisionCast"/> backward so <see cref="MixedPrecisionGraphBackward"/>
    /// bridges the gradient down to the Half grad space. Shared by <see cref="MatMul"/>, <see cref="Unary"/>,
    /// and <see cref="Binary"/>.
    /// </summary>
    private static Tensor<float> UpCast(LazyTensorScope scope, Tensor<Half> yH, int[] outShape, string name)
    {
        return scope.RecordCrossTypeWithBackward<Half, float>(
            LazyNodeType.Custom, name, yH, outShape,
            (e, o) => MixedPrecisionCast.CastToFp32(yH).AsSpan().CopyTo(o.AsWritableSpan()),
            (gradOut, input, output, state, e) => MixedPrecisionCast.CastToFp32Backward(gradOut));
    }

    /// <summary>
    /// Returns <paramref name="x"/> as an FP16 tensor for an FP16-eligible op input. If <paramref name="x"/>
    /// is the float output of a prior FP16 op's up-cast, reuses that op's Half output directly (no new
    /// cast node) so consecutive FP16 ops keep their activations in FP16 and the intermediate float
    /// up-cast is left dead. Otherwise records a down-cast (the FP32→FP16 boundary into the region).
    /// </summary>
    private static Tensor<Half> ToFp16Input(LazyTensorScope scope, Tensor<float> x, string name)
    {
        if (x.LazySource is CrossTypeLazyNode<Half, float> up)
            return up.Input; // already FP16 upstream — reuse it, skip the float round-trip
        return scope.RecordCrossTypeWithBackward<float, Half>(
            LazyNodeType.Custom, name, x, x._shape,
            (e, o) => MixedPrecisionCast.CastToFp16(x).AsSpan().CopyTo(o.AsWritableSpan()),
            (gradOut, input, output, state, e) => MixedPrecisionCast.CastToFp16Backward(gradOut));
    }
}
