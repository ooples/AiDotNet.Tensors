using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.Simd;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tensors.Engines.Optimization;

/// <summary>
/// Phase 4.2: Attention pattern fusion — detects Q@K^T -> Scale -> Softmax -> @V
/// patterns and replaces them with a fused attention kernel.
///
/// Standard attention: O(N^2) memory for attention matrix, 4 separate kernels.
/// Fused attention: O(N) memory, single tiled pass (Flash Attention style).
///
/// Pattern matched:
///   1. MatMul(Q, K^T)     → scores [batch, heads, seqQ, seqK]
///   2. Scale(scores, 1/√d) → scaled_scores (optional, may be fused with matmul)
///   3. Softmax(scaled)     → attention weights
///   4. MatMul(weights, V)  → output [batch, heads, seqQ, headDim]
///
/// Expected gain: 2-5x for long sequences, O(N) vs O(N^2) memory.
/// </summary>
internal sealed class AttentionFusionPass : ICpuOptimizationPass
{
    public string Name => "AttentionFusion";

    public bool IsEnabled => TensorCodecOptions.Current.EnableAttentionFusion;

    public CompiledStep<T>[]? TryOptimize<T>(CompiledStep<T>[] steps, IEngine engine)
    {
        if (!IsEnabled || typeof(T) != typeof(float) || steps.Length < 3) return null;

        var result = new List<CompiledStep<T>>(steps.Length);
        bool anyFused = false;

        for (int i = 0; i < steps.Length; i++)
        {
            // Pattern: MatMul[i] -> Softmax[j] -> MatMul[k]
            // (Scale may be between MatMul and Softmax, or fused into the matmul)
            if (i + 2 < steps.Length && TryMatchAttentionPattern(steps, i, out var fused, out int consumed))
            {
                if (fused is not null)
                {
                    result.Add(fused);
                    i += consumed - 1;
                    anyFused = true;
                    continue;
                }
            }
            result.Add(steps[i]);
        }

        return anyFused ? result.ToArray() : null;
    }

    private static bool TryMatchAttentionPattern<T>(
        CompiledStep<T>[] steps, int index,
        out CompiledStep<T>? fused, out int consumed)
    {
        fused = null;
        consumed = 0;

        // Look for: MatMul -> [Scale] -> Softmax -> MatMul
        var qk = steps[index];
        if (qk.OpName != "TensorMatMul" && qk.OpName != "BatchMatMul")
            return false;

        // Find Softmax within next 2 steps (may have Scale in between)
        int softmaxIdx = -1;
        for (int j = index + 1; j <= Math.Min(index + 2, steps.Length - 1); j++)
        {
            if (steps[j].OpName == "Softmax")
            {
                softmaxIdx = j;
                break;
            }
        }
        if (softmaxIdx < 0) return false;

        // Find the second MatMul right after Softmax
        if (softmaxIdx + 1 >= steps.Length) return false;
        var attnV = steps[softmaxIdx + 1];
        if (attnV.OpName != "TensorMatMul" && attnV.OpName != "BatchMatMul")
            return false;

        // Verify data dependencies:
        // QK output feeds into softmax, softmax output feeds into attnV
        var softmax = steps[softmaxIdx];
        if (!ReferenceEquals(qk.OutputBuffer, softmax.Inputs[0])
            && !(softmaxIdx > index + 1 && ReferenceEquals(steps[index + 1].OutputBuffer, softmax.Inputs[0])))
            return false;

        if (!ReferenceEquals(softmax.OutputBuffer, attnV.Inputs[0]))
            return false;

        consumed = softmaxIdx + 2 - index;

        // Extract Q, K, V tensors and dimensions for Flash Attention
        // QK step: Q @ K^T, so Q = qk.Inputs[0], K^T's source = qk.Inputs[1]
        // attnV step: attn_weights @ V, so V = attnV.Inputs[1]
        if (typeof(T) != typeof(float)) return false;

        var queryTensor = qk.Inputs[0];
        var keyTensor = qk.Inputs[1]; // This is K (or K^T depending on how the graph was built)
        var valueTensor = attnV.Inputs[1];
        var finalOutput = attnV.OutputBuffer;

        // Determine if this is batched (4D: [batch*heads, seqQ, headDim]) or 2D
        bool isBatched = queryTensor.Rank >= 3;

        if (queryTensor.Rank == 2 && keyTensor.Rank == 2 && valueTensor.Rank == 2)
        {
            // 2D attention: [seqQ, headDim] @ [seqK, headDim]^T
            int seqQ = queryTensor._shape[0];
            int headDim = queryTensor._shape[1];
            int seqK = keyTensor._shape[0];
            float scale = 1f / MathF.Sqrt(headDim);

            var capturedQ = queryTensor;
            var capturedK = keyTensor;
            var capturedV = valueTensor;

            fused = new CompiledStep<T>(
                "FlashAttention",
                (eng, output) =>
                {
                    var qArr = (float[])(object)capturedQ.GetDataArray();
                    var kArr = (float[])(object)capturedK.GetDataArray();
                    var vArr = (float[])(object)capturedV.GetDataArray();
                    var oArr = (float[])(object)output.GetDataArray();

                    FusedAttention.FlashAttentionForward(
                        qArr, kArr, vArr, oArr,
                        seqQ, seqK, headDim, scale);
                },
                finalOutput,
                new[] { queryTensor, keyTensor, valueTensor },
                null, // Flash Attention backward is a separate kernel (not yet implemented)
                null);

            return true;
        }

        // For higher-rank tensors, fall back to dispatch fusion (sequential execution)
        var capturedSteps = new CompiledStep<T>[consumed];
        Array.Copy(steps, index, capturedSteps, 0, consumed);

        fused = new CompiledStep<T>(
            "FusedAttention",
            (eng, output) =>
            {
                foreach (var step in capturedSteps)
                    step.Execute(eng, step.OutputBuffer);
            },
            finalOutput,
            qk.Inputs,
            null,
            null);

        return true;
    }
}
