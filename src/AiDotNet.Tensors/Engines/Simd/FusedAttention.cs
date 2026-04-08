using System;
using System.Runtime.CompilerServices;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tensors.Engines.Simd;

/// <summary>
/// Flash Attention: tiled fused attention kernel that computes
/// O = softmax(Q @ K^T / sqrt(d)) @ V without materializing the full N×N attention matrix.
///
/// Standard attention: O(N^2) memory for the attention matrix, 4 separate kernels.
/// Flash Attention: O(N) memory, single tiled pass with online softmax.
///
/// Algorithm (Dao et al., 2022):
///   For each query tile Qi (size Br × d):
///     For each key/value tile Kj, Vj (size Bc × d):
///       Sij = Qi @ Kj^T (local scores, Br × Bc)
///       mij = rowmax(Sij)
///       Pij = exp(Sij - mij) (local attention weights)
///       lij = rowsum(Pij)
///       Update running max: mi_new = max(mi_old, mij)
///       Rescale: Oi = exp(mi_old - mi_new) * Oi + Pij @ Vj
///       Update normalizer: li = exp(mi_old - mi_new) * li_old + lij
///     Final: Oi = Oi / li
///
/// Key insight: online softmax maintains numerical stability without needing
/// the full attention matrix. Each tile computes local exp values and rescales
/// previous results as the global max evolves.
/// </summary>
internal static class FusedAttention
{
    /// <summary>Default tile sizes for the tiled attention kernel.</summary>
    private const int DefaultBr = 64;  // Query tile size (rows of Q processed together)
    private const int DefaultBc = 64;  // Key tile size (columns of K^T processed together)

    /// <summary>
    /// Computes Flash Attention: O = softmax(Q @ K^T / sqrt(d)) @ V
    /// using tiled online softmax algorithm.
    /// </summary>
    /// <param name="q">Query matrix [seqQ, headDim]</param>
    /// <param name="k">Key matrix [seqK, headDim]</param>
    /// <param name="v">Value matrix [seqK, headDim]</param>
    /// <param name="output">Output matrix [seqQ, headDim] (pre-allocated)</param>
    /// <param name="scale">Scale factor, typically 1/sqrt(headDim)</param>
    /// <param name="isCausal">Whether to apply causal masking (upper triangle = -inf)</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static void FlashAttentionForward(
        float[] q, float[] k, float[] v, float[] output,
        int seqQ, int seqK, int headDim,
        float scale, bool isCausal = false)
    {
        int br = Math.Min(DefaultBr, seqQ);
        int bc = Math.Min(DefaultBc, seqK);

        // Per-row running state for online softmax
        var rowMax = new float[br];    // Running max per query row
        var rowSum = new float[br];    // Running sum of exp per query row
        var localScores = new float[br * bc]; // Tile of Q @ K^T scores

        // Initialize output to zero
        Array.Clear(output, 0, seqQ * headDim);

        // Outer loop: iterate over query tiles
        for (int qi = 0; qi < seqQ; qi += br)
        {
            int actualBr = Math.Min(br, seqQ - qi);

            // Initialize running max to -inf and sum to 0 for this query tile
            for (int r = 0; r < actualBr; r++)
            {
                rowMax[r] = float.NegativeInfinity;
                rowSum[r] = 0f;
            }

            // Inner loop: iterate over key/value tiles
            for (int kj = 0; kj < seqK; kj += bc)
            {
                int actualBc = Math.Min(bc, seqK - kj);

                // Step 1: Compute local scores Sij = Qi @ Kj^T (Br × Bc)
                ComputeScoreTile(q, k, localScores, qi, kj, actualBr, actualBc, headDim, scale);

                // Step 1b: Apply causal mask if needed
                if (isCausal)
                    ApplyCausalMask(localScores, qi, kj, actualBr, actualBc);

                // Step 2: Online softmax update
                // For each query row, update running max, rescale previous output,
                // compute local exp weights, and accumulate into output
                for (int r = 0; r < actualBr; r++)
                {
                    int globalRow = qi + r;

                    // Find local row max
                    float localMax = float.NegativeInfinity;
                    for (int c = 0; c < actualBc; c++)
                    {
                        float s = localScores[r * actualBc + c];
                        if (s > localMax) localMax = s;
                    }

                    // Update global max
                    float prevMax = rowMax[r];
                    float newMax = Math.Max(prevMax, localMax);
                    rowMax[r] = newMax;

                    // Rescale factor for previous accumulations
                    float rescale = prevMax == float.NegativeInfinity
                        ? 0f
                        : MathF.Exp(prevMax - newMax);

                    // Rescale previous output and sum
                    if (rescale > 0f && rescale < 1f)
                    {
                        int outOffset = globalRow * headDim;
                        for (int d = 0; d < headDim; d++)
                            output[outOffset + d] *= rescale;
                        rowSum[r] *= rescale;
                    }

                    // Compute exp weights and accumulate: O += exp(S - max) @ V
                    float localSum = 0f;
                    for (int c = 0; c < actualBc; c++)
                    {
                        float expVal = MathF.Exp(localScores[r * actualBc + c] - newMax);
                        localSum += expVal;

                        // Accumulate: output[row, :] += expVal * V[kj + c, :]
                        int vOffset = (kj + c) * headDim;
                        int oOffset = globalRow * headDim;
                        for (int d = 0; d < headDim; d++)
                            output[oOffset + d] += expVal * v[vOffset + d];
                    }

                    rowSum[r] += localSum;
                }
            }

            // Step 3: Normalize output by row sum
            for (int r = 0; r < actualBr; r++)
            {
                int globalRow = qi + r;
                float invSum = rowSum[r] > 0f ? 1f / rowSum[r] : 0f;
                int oOffset = globalRow * headDim;
                for (int d = 0; d < headDim; d++)
                    output[oOffset + d] *= invSum;
            }
        }
    }

    /// <summary>
    /// Batched Flash Attention: processes multiple batch×head combinations.
    /// Q, K, V are [batch * heads, seq, headDim].
    /// </summary>
    internal static void BatchedFlashAttention(
        float[] q, float[] k, float[] v, float[] output,
        int batchHeads, int seqQ, int seqK, int headDim,
        float scale, bool isCausal = false)
    {
        int qStride = seqQ * headDim;
        int kStride = seqK * headDim;

        // Process each batch×head independently (embarrassingly parallel)
        System.Threading.Tasks.Parallel.For(0, batchHeads, bh =>
        {
            int qOffset = bh * qStride;
            int kOffset = bh * kStride;
            int vOffset = bh * kStride; // V has same layout as K

            // Create per-thread views (avoid allocation by using ArraySegment-like offsets)
            var qSlice = new float[qStride];
            var kSlice = new float[kStride];
            var vSlice = new float[kStride];
            var oSlice = new float[qStride];

            Array.Copy(q, qOffset, qSlice, 0, qStride);
            Array.Copy(k, kOffset, kSlice, 0, kStride);
            Array.Copy(v, vOffset, vSlice, 0, kStride);

            FlashAttentionForward(qSlice, kSlice, vSlice, oSlice,
                seqQ, seqK, headDim, scale, isCausal);

            Array.Copy(oSlice, 0, output, qOffset, qStride);
        });
    }

    /// <summary>Compute score tile: Sij = Q[qi:qi+br, :] @ K[kj:kj+bc, :]^T * scale</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void ComputeScoreTile(
        float[] q, float[] k, float[] scores,
        int qi, int kj, int br, int bc, int headDim, float scale)
    {
        for (int r = 0; r < br; r++)
        {
            int qRowOffset = (qi + r) * headDim;
            for (int c = 0; c < bc; c++)
            {
                int kRowOffset = (kj + c) * headDim;
                float dot = 0f;
                for (int d = 0; d < headDim; d++)
                    dot += q[qRowOffset + d] * k[kRowOffset + d];
                scores[r * bc + c] = dot * scale;
            }
        }
    }

    /// <summary>Apply causal mask: set scores[r, c] = -inf where qi+r < kj+c</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void ApplyCausalMask(float[] scores, int qi, int kj, int br, int bc)
    {
        for (int r = 0; r < br; r++)
        {
            for (int c = 0; c < bc; c++)
            {
                if (qi + r < kj + c)
                    scores[r * bc + c] = float.NegativeInfinity;
            }
        }
    }
}
