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

    // AIDOTNET_FP16_ACTIVATIONS routes to the separate (capture-incompatible) MixedPrecisionCompiledPlan in
    // AiDotNet; AIDOTNET_FP16_CAPTURE instead keeps the normal CompiledTrainingPlan but emits the same Half
    // activation nodes, which that plan now consumes via its heterogeneous forward/backward (FP16-in-capture,
    // task #30). Either enables the FP16-activation EMISSION here; the routing differs at the AiDotNet caller.
    public static bool Fp16ActivationStorageEnabled =>
        TestOverrideEnabled ?? (Environment.GetEnvironmentVariable("AIDOTNET_FP16_ACTIVATIONS") == "1"
                                || Environment.GetEnvironmentVariable("AIDOTNET_FP16_CAPTURE") == "1");

    /// <summary>
    /// True when graph tracing should emit FP16 activation buffers: an FP16 autocast scope is active.
    /// </summary>
    public static bool ShouldEmitFp16<T>() =>
        typeof(T) == typeof(float)
        && Gpu.AutocastScope.IsEnabled
        && Gpu.AutocastScope.ActivePrecision == Gpu.PrecisionMode.Float16;

    /// <summary>
    /// Both the autocast scope and the activation-storage opt-in are active. The CpuEngine GraphMode
    /// branches (MatMul/GELU/ReLU/Softmax/TensorAdd/LayerNorm) call this to decide whether to record the
    /// FP16-native emit instead of the FP32 node.
    /// <para><b>GPU-engine behavior (documents the deliberate fallback for issue #558).</b> The recorded
    /// realize delegates are engine-agnostic: at realize they call <c>e.GELU</c>/<c>e.TensorMatMul</c>/etc.,
    /// which dispatch to whatever engine runs the plan — so on <c>DirectGpuTensorEngine</c> the op math runs
    /// on the GPU, NOT silently on the CPU. The only host-side work is the FP16↔FP32 dtype cast
    /// (<see cref="MixedPrecisionCast"/>); the GPU does the matmul/activation. This is the same engine-agnostic
    /// pattern the already-shipped FP16 <see cref="MatMul"/> emit uses. The zero-copy GPU path — having the
    /// realize call the per-backend FP16-native kernels (<c>Fp16Gelu</c>/<c>Fp16Relu</c>/<c>Fp16Add</c>,
    /// added for all six backends in this PR) so no host round-trip happens — is the remaining wiring tracked
    /// under issue #558; until then the GPU path is correct (ops on GPU) but not yet host-round-trip-free.</para>
    /// </summary>
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
                LandHalf(e, r, o);
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
    /// <param name="fp32Backward">
    /// The op's FP32 backward: <c>(engine, gradOutFp32, inputFp32, outputFp32, savedState) =&gt; gradInFp32</c>.
    /// <paramref name="outputFp32"/> is the up-cast realized output (softmax/sigmoid backward need it);
    /// <c>savedState</c> carries op params (e.g. the softmax axis). Either may be ignored.
    /// </param>
    public static Tensor<float> Unary(
        LazyTensorScope scope,
        Tensor<float> x,
        int[] outShape,
        string opName,
        LazyNodeType nodeType,
        Func<IEngine, Tensor<float>, Tensor<float>> fp32Forward,
        Func<IEngine, Tensor<float>, Tensor<float>, Tensor<float>, object[], Tensor<float>> fp32Backward,
        object[]? savedState = null)
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
                // result back as Half so the saved activation chain never widens to FP32. The up-cast/op
                // results are fresh, non-pooled Tensors local to this realize — dispose them so the
                // per-replay-step transient FP32 buffers don't accumulate GC pressure in the training loop.
                using var xf = MixedPrecisionCast.CastToFp32(xH);
                using var yf = fp32Forward(e, xf);
                using var yh = MixedPrecisionCast.CastToFp16(yf);
                LandHalf(e, yh, o);
            },
            (gradOutH, inputs, output, state, e, grads) =>
            {
                // Backward in FP32: up-cast the saved Half input + incoming Half grad (+ Half output), run
                // the op's FP32 gradient, down-cast the input-grad back to Half and accumulate. All upcasts
                // and the op-gradient output are fresh transients → dispose them; the down-cast grad handed
                // to AccumulateGrad is retained by the grad map and must NOT be disposed here.
                var inH = inputs[0];
                using var gradOutF = MixedPrecisionCast.CastToFp32(gradOutH);
                using var inF = MixedPrecisionCast.CastToFp32(inH);
                using var outF = MixedPrecisionCast.CastToFp32(output);
                using var gradInF = fp32Backward(e, gradOutF, inF, outF, state);
                Autodiff.DifferentiableOps.AccumulateGrad(grads, inH, MixedPrecisionCast.CastToFp16(gradInF), e);
            },
            savedState: savedState ?? Array.Empty<object>());

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
                // Fresh, non-pooled transients local to this realize — dispose to bound per-step GC churn.
                using var af = MixedPrecisionCast.CastToFp32(aH);
                using var bf = MixedPrecisionCast.CastToFp32(bH);
                using var cf = fp32Forward(e, af, bf);
                using var ch = MixedPrecisionCast.CastToFp16(cf);
                LandHalf(e, ch, o);
            },
            (gradOutH, inputs, output, state, e, grads) =>
            {
                var inA = inputs[0];
                var inB = inputs[1];
                // Only dispose the operand up-casts: they are fresh and never returned by fp32Backward.
                // gradOutF and the returned (gradAF, gradBF) are NOT disposed — a linear op (e.g. residual
                // Add: (gOut, gOut)) returns gradOutF itself for both inputs, so disposing any of them would
                // free a buffer the CastToFp16 reads (use-after-free) or double-free the alias. Leaving them
                // to GC is the safe choice; the dominant transients (the two operand up-casts) are freed.
                var gradOutF = MixedPrecisionCast.CastToFp32(gradOutH);
                using var aF = MixedPrecisionCast.CastToFp32(inA);
                using var bF = MixedPrecisionCast.CastToFp32(inB);
                var (gradAF, gradBF) = fp32Backward(e, gradOutF, aF, bF);
                Autodiff.DifferentiableOps.AccumulateGrad(grads, inA, MixedPrecisionCast.CastToFp16(gradAF), e);
                Autodiff.DifferentiableOps.AccumulateGrad(grads, inB, MixedPrecisionCast.CastToFp16(gradBF), e);
            });

        return UpCast(scope, cH, outShape, opName + ".castOut32");
    }

    /// <summary>Mutable carrier for the realize-time FP32 mean/variance a norm backward needs.</summary>
    private sealed class MeanVarHolder
    {
        public Tensor<float>? Mean;
        public Tensor<float>? Variance;
    }

    /// <summary>
    /// Emits an FP16-NATIVE affine LayerNorm so the dominant activation (the [B,S,D] input) is saved as
    /// <see cref="Half"/>. Input and the gamma/beta params are taken as Half (params down-cast once per step
    /// via <see cref="ToFp16Input"/> — the cast node bridges their Half grad back to the FP32 master grad,
    /// exactly as matmul handles its weight); the normalize math runs in FP32 on an up-cast-in-realize copy,
    /// the output is Half, and the realize-time mean/variance flow to the backward through a holder (the
    /// #1331 freshness contract — trace-time mean/var are stale under replay). gradInput/gradGamma/gradBeta
    /// are computed in FP32 then down-cast into the Half grad map.
    /// </summary>
    public static Tensor<float> LayerNorm(
        LazyTensorScope scope,
        Tensor<float> x,
        Tensor<float> gamma,
        Tensor<float> beta,
        double epsilon,
        int[] outShape,
        string opName = "LayerNorm")
    {
        if (scope is null) throw new ArgumentNullException(nameof(scope));
        if (x is null) throw new ArgumentNullException(nameof(x));
        if (gamma is null) throw new ArgumentNullException(nameof(gamma));
        if (beta is null) throw new ArgumentNullException(nameof(beta));
        if (outShape is null) throw new ArgumentNullException(nameof(outShape));

        var xH = ToFp16Input(scope, x, opName + ".castIn16");
        var gH = ToFp16Input(scope, gamma, opName + ".castGamma16");
        var bH = ToFp16Input(scope, beta, opName + ".castBeta16");
        var mv = new MeanVarHolder();

        var yH = scope.RecordVariadic<Half>(
            LazyNodeType.Custom, opName + ".fp16", new[] { xH, gH, bH }, outShape,
            (e, o) =>
            {
                // x/gamma/beta up-casts and the normalized output are fresh transients — dispose them. The
                // mean/variance out-params are NOT disposed: they're handed to the backward via the holder.
                using var xf = MixedPrecisionCast.CastToFp32(xH);
                using var gf = MixedPrecisionCast.CastToFp32(gH);
                using var bf = MixedPrecisionCast.CastToFp32(bH);
                using var yf = e.LayerNorm(xf, gf, bf, epsilon, out var mean, out var variance);
                mv.Mean = mean;
                mv.Variance = variance;
                using var yh = MixedPrecisionCast.CastToFp16(yf);
                LandHalf(e, yh, o);
            },
            (gradOutH, inputs, output, state, e, grads) =>
            {
                if (mv.Mean is null || mv.Variance is null)
                    throw new InvalidOperationException("LayerNorm backward ran before its forward realize populated mean/variance.");
                // All up-casts and the three FP32 grad outputs are fresh transients → dispose. The down-cast
                // grads handed to AccumulateGrad are retained by the grad map and must NOT be disposed here.
                using var xf = MixedPrecisionCast.CastToFp32(inputs[0]);
                using var gf = MixedPrecisionCast.CastToFp32(inputs[1]);
                using var gradOutF = MixedPrecisionCast.CastToFp32(gradOutH);
                using var gradIn = e.LayerNormBackward(
                    gradOutF, xf, gf, mv.Mean, mv.Variance, epsilon, out var gradGamma, out var gradBeta);
                using var gradGammaD = gradGamma;
                using var gradBetaD = gradBeta;
                Autodiff.DifferentiableOps.AccumulateGrad(grads, inputs[0], MixedPrecisionCast.CastToFp16(gradIn), e);
                Autodiff.DifferentiableOps.AccumulateGrad(grads, inputs[1], MixedPrecisionCast.CastToFp16(gradGammaD), e);
                Autodiff.DifferentiableOps.AccumulateGrad(grads, inputs[2], MixedPrecisionCast.CastToFp16(gradBetaD), e);
            });

        return UpCast(scope, yH, outShape, opName + ".castOut32");
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
            (e, o) => LandHalf(e, MixedPrecisionCast.CastToFp16(x), o),
            (gradOut, input, output, state, e) => MixedPrecisionCast.CastToFp16Backward(gradOut));
    }

    /// <summary>
    /// GPU-resident landing of a Half op result into the stable node-output <paramref name="o"/> (#34): on a
    /// DirectGpu engine, an on-device DtoD copy keeps o resident so the next op reads it from GPU with no
    /// re-upload (eliminates the per-op host round-trip that dominated the hetero replay). Falls back to the
    /// host <c>CopyTo</c> on CPU / non-resident results — identical semantics, just without the GPU fast path.
    /// </summary>
    private static void LandHalf(IEngine e, Tensor<Half> r, Tensor<Half> o)
    {
        if (e is DirectGpuTensorEngine gpu && gpu.TryLandResidentHalf(r, o)) return;
        r.AsSpan().CopyTo(o.AsWritableSpan());
    }
}
