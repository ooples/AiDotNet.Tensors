using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.Compilation.Ops;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Optimization;

/// <summary>
/// Fusion patterns 11-14 for diffusion UNet hot sequences. Operates on the
/// compiled step array and replaces multi-step sequences with single fused
/// operations that eliminate intermediate tensor materializations.
///
/// <para><b>Pattern 11 — GroupNorm + SiLU:</b>
/// <c>GroupNorm → Swish</c> → <see cref="FusedGroupNormActivationOp{T}"/>{SiLU}.
/// Hit rate: 80 per SD15 forward (~40 ResBlocks × 2). Saves ~41 MB intermediates.</para>
///
/// <para><b>Pattern 12 — Conv2D + Bias + SiLU:</b>
/// <c>Conv2D → BroadcastAdd → Swish</c> → <see cref="FusedConv2DBiasActivationOp{T}"/>{SiLU}.
/// Hit rate: ~40 per SD15 forward. 2-4× faster than 3-op sequence.</para>
///
/// <para><b>Pattern 14 — Residual Add + GroupNorm:</b>
/// <c>Add → GroupNorm</c> → fused step using <c>IEngine.AddGroupNormInto</c>.
/// Hit rate: ~40 per SD15 forward. Eliminates skip-connection intermediate.</para>
///
/// <para>Pattern 13 (GroupNorm+SiLU+Conv2D 3-op fusion) is deferred —
/// Patterns 11+12 already capture most of the win.</para>
/// </summary>
internal sealed class DiffusionFusionPass : ICpuOptimizationPass
{
    public string Name => "DiffusionFusion";

    // Always enabled — diffusion patterns are high-value and zero-risk
    // (the fused ops delegate to existing engine kernels).
    public bool IsEnabled => true;

    public CompiledStep<T>[]? TryOptimize<T>(CompiledStep<T>[] steps, IEngine engine)
    {
        if (typeof(T) != typeof(float)) return null; // Float-only for now

        var result = new List<CompiledStep<T>>(steps.Length);
        bool anyFused = false;
        int i = 0;

        while (i < steps.Length)
        {
            // Try Pattern 12 first (3-op, greedy) before Pattern 11 (2-op)
            if (TryMatchPattern12(steps, i, engine, out var fused12, out int consumed12))
            {
                result.Add(fused12!);
                i += consumed12;
                anyFused = true;
                continue;
            }

            // Pattern 11: GroupNorm + Swish → FusedGroupNormActivation{SiLU}
            if (TryMatchPattern11(steps, i, engine, out var fused11, out int consumed11))
            {
                result.Add(fused11!);
                i += consumed11;
                anyFused = true;
                continue;
            }

            // Pattern 14: Add + GroupNorm → FusedAddGroupNorm
            if (TryMatchPattern14(steps, i, engine, out var fused14, out int consumed14))
            {
                result.Add(fused14!);
                i += consumed14;
                anyFused = true;
                continue;
            }

            // No match — pass through
            result.Add(steps[i]);
            i++;
        }

        return anyFused ? result.ToArray() : null;
    }

    // ════════════════════════════════════════════════════════════════════
    // Pattern 11: GroupNorm → Swish → FusedGroupNormActivation{SiLU}
    // ════════════════════════════════════════════════════════════════════

    private static bool TryMatchPattern11<T>(
        CompiledStep<T>[] steps, int index, IEngine engine,
        out CompiledStep<T>? fused, out int consumed)
    {
        fused = null;
        consumed = 0;

        if (index + 1 >= steps.Length) return false;

        var gnStep    = steps[index];
        var swishStep = steps[index + 1];

        // Match: GroupNorm followed by Swish
        if (gnStep.OpType != OpType.GroupNorm) return false;
        if (swishStep.OpType != OpType.Swish) return false;

        // Data dependency: GroupNorm output feeds Swish input
        if (!ReferenceEquals(gnStep.OutputBuffer, swishStep.Inputs[0])) return false;

        // Extract GroupNorm params from SavedState
        var gnOp = GroupNormOp<T>.TryFromStep(gnStep);
        if (gnOp is null) return false;

        // Build fused op
        var fusedOp = new FusedGroupNormActivationOp<T>(
            gnOp.Input, gnOp.NumGroups, gnOp.Gamma, gnOp.Beta, gnOp.Epsilon,
            GroupNormActivation.SiLU);

        fused = fusedOp.ToCompiledStep(swishStep.OutputBuffer);
        consumed = 2;
        return true;
    }

    // ════════════════════════════════════════════════════════════════════
    // Pattern 12: Conv2D → BroadcastAdd → Swish → FusedConv2DBiasActivation{SiLU}
    // ════════════════════════════════════════════════════════════════════

    private static bool TryMatchPattern12<T>(
        CompiledStep<T>[] steps, int index, IEngine engine,
        out CompiledStep<T>? fused, out int consumed)
    {
        fused = null;
        consumed = 0;

        if (index + 2 >= steps.Length) return false;

        var convStep  = steps[index];
        var addStep   = steps[index + 1];
        var swishStep = steps[index + 2];

        // Match: Conv2D → BroadcastAdd → Swish
        if (convStep.OpType != OpType.Conv2D) return false;
        if (addStep.OpType != OpType.TensorBroadcastAdd) return false;
        if (swishStep.OpType != OpType.Swish) return false;

        // Data dependency chain
        if (!ReferenceEquals(convStep.OutputBuffer, addStep.Inputs[0])) return false;
        if (!ReferenceEquals(addStep.OutputBuffer, swishStep.Inputs[0])) return false;

        // Extract conv params from SavedState. Strict: if the shape of the
        // saved state doesn't match what Conv2DOp writes, bail out of fusion
        // rather than silently substituting defaults — defaults would produce
        // numerically wrong output for any conv with stride≠1, padding≠0, or
        // dilation≠1.
        var convState = convStep.SavedState;
        if (convState is null || convState.Length < 3) return false;
        if (convState[0] is not int[] s || s.Length == 0) return false;
        if (convState[1] is not int[] p || p.Length == 0) return false;
        if (convState[2] is not int[] d || d.Length == 0) return false;
        int stride = s[0];
        int padding = p[0];
        int dilation = d[0];

        // The bias is the second input to BroadcastAdd
        var bias = addStep.Inputs.Length > 1 ? addStep.Inputs[1] : null;

        // Build fused op
        var fusedOp = new FusedConv2DBiasActivationOp<T>(
            convStep.Inputs[0], convStep.Inputs[1], bias,
            stride, stride, padding, padding, dilation, dilation,
            FusedActivationType.Swish);

        fused = fusedOp.ToCompiledStep(swishStep.OutputBuffer);
        consumed = 3;
        return true;
    }

    // ════════════════════════════════════════════════════════════════════
    // Pattern 14: Add → GroupNorm → FusedAddGroupNorm
    // ════════════════════════════════════════════════════════════════════

    private static bool TryMatchPattern14<T>(
        CompiledStep<T>[] steps, int index, IEngine engine,
        out CompiledStep<T>? fused, out int consumed)
    {
        fused = null;
        consumed = 0;

        if (index + 1 >= steps.Length) return false;

        var addStep = steps[index];
        var gnStep  = steps[index + 1];

        // Match: Add/BroadcastAdd followed by GroupNorm
        if (addStep.OpType is not (OpType.TensorAdd or OpType.TensorBroadcastAdd)) return false;
        if (gnStep.OpType != OpType.GroupNorm) return false;

        // Data dependency: Add output feeds GroupNorm input
        if (!ReferenceEquals(addStep.OutputBuffer, gnStep.Inputs[0])) return false;

        // Extract GroupNorm params
        var gnOp = GroupNormOp<T>.TryFromStep(gnStep);
        if (gnOp is null) return false;

        // Build fused step using IEngine.AddGroupNormInto. A well-formed Add
        // has exactly two distinct operands; a single-input Add is malformed
        // and fusing it as GroupNorm(a + a) would silently double the input.
        if (addStep.Inputs.Length < 2) return false;
        var addA = addStep.Inputs[0];
        var addB = addStep.Inputs[1];
        var gamma = gnOp.Gamma;
        var beta = gnOp.Beta;
        int numGroups = gnOp.NumGroups;
        double epsilon = gnOp.Epsilon;

        fused = new CompiledStep<T>(
            opName: "FusedAddGroupNorm",
            execute: (eng, output) =>
            {
                eng.AddGroupNormInto(output, addA, addB, numGroups, gamma, beta, epsilon);
            },
            outputBuffer: gnStep.OutputBuffer,
            inputs: new[] { addA, addB, gamma, beta },
            backwardFn: null, // Backward decomposes: GroupNorm backward + Add backward
            savedState: new object[] { numGroups, epsilon });

        consumed = 2;
        return true;
    }
}
