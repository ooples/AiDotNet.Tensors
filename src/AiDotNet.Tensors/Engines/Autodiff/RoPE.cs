using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Autodiff;

/// <summary>
/// Rotary position embeddings — the fused in-place op used by LLaMA,
/// GPT-NeoX, Mistral, and virtually every modern decoder-only LLM.
/// Applies a 2D rotation to every pair of adjacent dims in Q / K
/// where the rotation angle is <c>pos × (base ^ (−2i / headDim))</c>
/// for pair index <c>i</c>.
///
/// <para><b>Why fused:</b> RoPE runs on every token's Q and K vectors
/// per layer. Doing it as (sin, cos, multiply, add) broadcasts materialises
/// three intermediate seq×dim tensors — for long contexts that's
/// O(seqLen × headDim) of wasted memory traffic. The fused kernel
/// rotates in place with two floats of state per (pos, pair).</para>
///
/// <para>Two common variants:</para>
/// <list type="bullet">
/// <item><b>Interleaved</b> — GPT-NeoX / LLaMA default. Dim pair
/// <c>(2i, 2i+1)</c> rotated together.</item>
/// <item><b>Half-rotated</b> — GPT-J / Meta newer. First half and second
/// half of the dim are rotated as two halves (dim i rotates with dim
/// i + headDim/2). Controlled by <see cref="RoPEStyle.HalfRotated"/>.</item>
/// </list>
/// </summary>
public enum RoPEStyle
{
    /// <summary>Rotate adjacent dim pairs — GPT-NeoX / LLaMA.</summary>
    Interleaved,
    /// <summary>Rotate first-half / second-half dim pairs — GPT-J.</summary>
    HalfRotated,
}

/// <summary>
/// Rotary position embeddings helper. Operates on rank-3
/// <c>[batch, seqLen, headDim]</c> or rank-4
/// <c>[batch, heads, seqLen, headDim]</c> tensors in place.
/// </summary>
public static class RoPE
{
    /// <summary>
    /// Apply RoPE to <paramref name="tensor"/> in place.
    /// </summary>
    /// <param name="tensor">Q or K; rank 3 <c>[B, S, D]</c> or rank 4
    /// <c>[B, H, S, D]</c>.</param>
    /// <param name="startPosition">Token-position offset of the first
    /// element on the seq axis. Lets the caller apply RoPE to a single
    /// decode step with the correct absolute position.</param>
    /// <param name="base">Base frequency — <c>10000</c> for LLaMA /
    /// Mistral, <c>500000</c> for LLaMA-3.1.</param>
    /// <param name="style">Pair-rotation layout.</param>
    public static void Apply(
        Tensor<float> tensor,
        int startPosition = 0,
        float @base = 10000f,
        RoPEStyle style = RoPEStyle.Interleaved)
    {
        if (tensor is null) throw new ArgumentNullException(nameof(tensor));
        int rank = tensor.Rank;
        if (rank != 3 && rank != 4)
            throw new ArgumentException(
                $"RoPE expects rank 3 [B, S, D] or rank 4 [B, H, S, D]; got rank {rank}.",
                nameof(tensor));
        if (!tensor.IsContiguous) tensor = tensor.Contiguous();

        int headDim = tensor._shape[rank - 1];
        if ((headDim & 1) != 0)
            throw new ArgumentException(
                $"RoPE requires even headDim (paired rotation); got {headDim}.", nameof(tensor));

        int batch = tensor._shape[0];
        int heads, seqLen;
        if (rank == 3)
        {
            heads = 1;
            seqLen = tensor._shape[1];
        }
        else
        {
            heads = tensor._shape[1];
            seqLen = tensor._shape[2];
        }

        var data = tensor.GetDataArray();
        int innerStride = headDim;
        int seqStride = heads * headDim; // [B, S, H, D] would differ — but canonical is [B, H, S, D]

        if (style == RoPEStyle.Interleaved)
        {
            int halfDim = headDim / 2;
            for (int b = 0; b < batch; b++)
            {
                for (int h = 0; h < heads; h++)
                {
                    for (int s = 0; s < seqLen; s++)
                    {
                        int pos = s + startPosition;
                        int rowBase = HeadRowBase(b, h, s, rank, heads, seqLen, headDim);
                        for (int i = 0; i < halfDim; i++)
                        {
                            float theta = pos / (float)Math.Pow(@base, 2.0 * i / headDim);
                            float cosT = (float)Math.Cos(theta);
                            float sinT = (float)Math.Sin(theta);
                            int a = rowBase + 2 * i;
                            int c = rowBase + 2 * i + 1;
                            float x = data[a], y = data[c];
                            data[a] = x * cosT - y * sinT;
                            data[c] = x * sinT + y * cosT;
                        }
                    }
                }
            }
        }
        else // HalfRotated
        {
            int halfDim = headDim / 2;
            for (int b = 0; b < batch; b++)
            {
                for (int h = 0; h < heads; h++)
                {
                    for (int s = 0; s < seqLen; s++)
                    {
                        int pos = s + startPosition;
                        int rowBase = HeadRowBase(b, h, s, rank, heads, seqLen, headDim);
                        for (int i = 0; i < halfDim; i++)
                        {
                            float theta = pos / (float)Math.Pow(@base, 2.0 * i / headDim);
                            float cosT = (float)Math.Cos(theta);
                            float sinT = (float)Math.Sin(theta);
                            int a = rowBase + i;
                            int c = rowBase + halfDim + i;
                            float x = data[a], y = data[c];
                            data[a] = x * cosT - y * sinT;
                            data[c] = x * sinT + y * cosT;
                        }
                    }
                }
            }
        }
    }

    private static int HeadRowBase(int b, int h, int s, int rank, int heads, int seqLen, int headDim)
    {
        // Canonical layouts:
        //   rank 3: [B, S, D]        → offset = ((b * S) + s) * D
        //   rank 4: [B, H, S, D]     → offset = (((b * H) + h) * S + s) * D
        if (rank == 3) return (b * seqLen + s) * headDim;
        return (((b * heads) + h) * seqLen + s) * headDim;
    }
}
