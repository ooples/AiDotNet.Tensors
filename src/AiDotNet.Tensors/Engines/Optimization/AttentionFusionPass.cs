using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.Simd;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

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
        if (qk.OpType is not (OpType.TensorMatMul or OpType.BatchMatMul))
            return false;

        // Find Softmax within next 2 steps (may have Scale in between)
        int softmaxIdx = -1;
        for (int j = index + 1; j <= Math.Min(index + 2, steps.Length - 1); j++)
        {
            if (steps[j].OpType == OpType.Softmax)
            {
                softmaxIdx = j;
                break;
            }
        }
        if (softmaxIdx < 0) return false;

        // Find the second MatMul right after Softmax
        if (softmaxIdx + 1 >= steps.Length) return false;
        var attnV = steps[softmaxIdx + 1];
        if (attnV.OpType is not (OpType.TensorMatMul or OpType.BatchMatMul))
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

        // The matched QK op is a plain TensorMatMul/BatchMatMul (no transpose flag),
        // so to compute scores = Q @ K^T its second operand is K^T — the output of a
        // TensorTranspose(K) step. FlashAttention computes Q @ K^T INTERNALLY and so
        // needs the NATURAL key K ([.., seqK, headDim]); feeding it the already-
        // transposed K^T makes it transpose twice (→ Q @ K), which is silently wrong
        // whenever seqK == headDim (equal shapes pass every downstream dim check) and
        // also breaks the seqK/headDim extraction below (both were written for natural K).
        // Recover natural K by tracing the transpose that produced the operand; if the
        // operand is not a clean last-two-dim transpose output we cannot prove its
        // layout, so refuse to fuse rather than risk corruption.
        if (!TryRecoverNaturalKey(steps, index, qk.Inputs[1], out var keyTensor))
            return false;
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

        // 3D / 4D attention: real FlashAttention kernel via
        // FusedAttention.BatchedFlashAttention. BERT attention has Q/K/V
        // shaped [batch, heads, seqQ, headDim] = [1, 12, 256, 64]. We
        // collapse (batch, heads) into a single batch-heads dim, then
        // dispatch per-head in parallel. Skips materialising the full
        // [batch, heads, seqQ, seqK] attention matrix (3 MB per head in
        // BERT = 36 MB of DRAM traffic per forward pass).
        //
        // Gate conditions: rank 3 or 4 tensors with matching inner dims.
        // rank-4 input [batch, heads, seq, d]: collapse to [batch*heads,
        //   seq, d].
        // rank-3 input [batch_heads, seq, d]: pass through directly.
        if (queryTensor.Rank >= 3 && keyTensor.Rank == queryTensor.Rank
            && valueTensor.Rank == queryTensor.Rank
            && queryTensor.Rank <= 4)
        {
            int rank = queryTensor.Rank;
            int seqQ = queryTensor._shape[rank - 2];
            int headDim = queryTensor._shape[rank - 1];
            int seqK = keyTensor._shape[rank - 2];
            int batchHeads = 1;
            for (int i = 0; i < rank - 2; i++) batchHeads *= queryTensor._shape[i];
            float scale = 1f / MathF.Sqrt(headDim);

            // Sanity: K and V must share seqK and headDim with the
            // expected attention layout. The QK step output is the
            // attention-weights tensor whose last two dims are [seqQ, seqK].
            if (keyTensor._shape[rank - 1] != headDim) return false;
            if (valueTensor._shape[rank - 2] != seqK) return false;
            if (valueTensor._shape[rank - 1] != headDim) return false;

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

                    FusedAttention.BatchedFlashAttention(
                        qArr, kArr, vArr, oArr,
                        batchHeads, seqQ, seqK, headDim, scale);
                },
                finalOutput,
                new[] { queryTensor, keyTensor, valueTensor },
                null, null);

            return true;
        }

        // Last-resort fallback: dispatch fusion (sequential execution)
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

    /// <summary>
    /// Recovers the natural key tensor K from the second operand of a Q @ K^T matmul.
    /// Because <see cref="OpType.TensorMatMul"/>/<see cref="OpType.BatchMatMul"/> carry no
    /// transpose flag, an attention graph materialises K^T with a <see cref="OpType.TensorTranspose"/>
    /// step and feeds its output as the matmul's second operand. The FlashAttention kernels transpose
    /// K internally, so they require natural K — we trace that transpose to its input.
    /// <see cref="CpuEngine.TensorTranspose{T}"/> is 2D-only and always swaps the two dims, so a
    /// TensorTranspose producer's input is exactly the natural [seqK, headDim] key. Returns false
    /// (caller must NOT fuse) when the operand has no producing transpose or was produced by some other
    /// op — there the orientation cannot be proven and fusing would risk a silently-transposed key. The
    /// downstream rank checks validate the recovered key's shape, so no extra shape guard is needed here.
    /// </summary>
    private static bool TryRecoverNaturalKey<T>(
        CompiledStep<T>[] steps, int qkIndex, Tensor<T> keyOperand, out Tensor<T> naturalKey)
    {
        naturalKey = keyOperand;

        for (int s = qkIndex - 1; s >= 0; s--)
        {
            var producer = steps[s];
            if (producer.OutputBuffer is null || !ReferenceEquals(producer.OutputBuffer, keyOperand))
                continue;

            // The K operand is K^T only if a TensorTranspose produced it; recover that transpose's input
            // (the natural key). Any other producer leaves the orientation unprovable -> refuse to fuse.
            if (producer.OpType == OpType.TensorTranspose && producer.Inputs is { Length: >= 1 })
            {
                naturalKey = producer.Inputs[0];
                return true;
            }
            return false;
        }

        // No producing step found (the key is a direct plan input): orientation unknown -> refuse.
        return false;
    }
}
