using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// The attention mask is broadcastable to <c>[batch, heads, seq_q, seq_k]</c> per the IEngine
/// contract, so a shared plane such as <c>[1, 1, seq, seq]</c> must produce the same result as the
/// fully-materialized <c>[batch, heads, seq, seq]</c> mask.
/// </summary>
/// <remarks>
/// <para>
/// The MHA float and double fast paths honoured this through <c>BroadcastMaskIndex</c>; the generic
/// SDPA workers indexed <c>mask[b, h, i, j]</c> directly, which reads past a size-1 axis. Because
/// <c>Tensor&lt;bool&gt;</c>'s indexer computes a flat offset from strides rather than bounds-checking
/// each axis, that did not throw — it silently read a DIFFERENT element, so a broadcast mask masked
/// the wrong positions and the attention output was quietly wrong.
/// </para>
/// <para>
/// Asserted at every shape the contract permits, and against the materialized mask rather than
/// against hand-computed numbers: the full mask exercises the path that was always correct, so any
/// disagreement is the broadcast path being wrong rather than a fixture encoding today's behaviour.
/// Driven at <c>double</c> and <c>float</c> because the two take different kernels.
/// </para>
/// </remarks>
public class SdpaMaskBroadcastTests
{
    private const int Batch = 2;
    private const int Heads = 3;
    private const int Seq = 5;
    private const int HeadDim = 4;

    /// <summary>Deterministic, non-degenerate values so a mis-indexed mask changes the output.</summary>
    private static Tensor<T> Fill<T>(Func<double, T> conv)
    {
        var data = new T[Batch * Heads * Seq * HeadDim];
        for (int i = 0; i < data.Length; i++)
        {
            data[i] = conv(Math.Sin(i * 0.7) * 1.5 + 0.25 * ((i % 7) - 3));
        }

        return new Tensor<T>(data, new[] { Batch, Heads, Seq, HeadDim });
    }

    /// <summary>A causal plane at an arbitrary broadcastable shape.</summary>
    private static Tensor<bool> CausalMask(int b, int h)
    {
        var data = new bool[b * h * Seq * Seq];
        int idx = 0;
        for (int bi = 0; bi < b; bi++)
            for (int hi = 0; hi < h; hi++)
                for (int i = 0; i < Seq; i++)
                    for (int j = 0; j < Seq; j++)
                        data[idx++] = j <= i;

        return new Tensor<bool>(data, new[] { b, h, Seq, Seq });
    }

    public static TheoryData<int, int> BroadcastShapes => new()
    {
        { 1, 1 },          // fully shared plane — the common causal case
        { Batch, 1 },      // per-batch, shared across heads
        { 1, Heads },      // per-head, shared across batch
    };

    /// <summary>
    /// A broadcast mask must give the same output as the equivalent materialized mask (double).
    /// </summary>
    [Theory]
    [MemberData(nameof(BroadcastShapes))]
    public void DoubleSdpa_BroadcastMask_MatchesMaterializedMask(int maskBatch, int maskHeads)
    {
        var engine = new CpuEngine();
        var q = Fill<double>(v => v);
        var k = Fill<double>(v => v * 0.5 + 0.1);
        var v2 = Fill<double>(v => v * -0.3 + 0.2);

        var expected = engine.ScaledDotProductAttention(
            q, k, v2, CausalMask(Batch, Heads), scale: null, out _);
        var actual = engine.ScaledDotProductAttention(
            q, k, v2, CausalMask(maskBatch, maskHeads), scale: null, out _);

        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.Equal(expected[i], actual[i], precision: 12);
        }
    }

    /// <summary>
    /// The same contract at <c>float</c>, which takes a different kernel.
    /// </summary>
    [Theory]
    [MemberData(nameof(BroadcastShapes))]
    public void FloatSdpa_BroadcastMask_MatchesMaterializedMask(int maskBatch, int maskHeads)
    {
        var engine = new CpuEngine();
        var q = Fill<float>(v => (float)v);
        var k = Fill<float>(v => (float)(v * 0.5 + 0.1));
        var v2 = Fill<float>(v => (float)(v * -0.3 + 0.2));

        var expected = engine.ScaledDotProductAttention(
            q, k, v2, CausalMask(Batch, Heads), scale: null, out _);
        var actual = engine.ScaledDotProductAttention(
            q, k, v2, CausalMask(maskBatch, maskHeads), scale: null, out _);

        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.Equal(expected[i], actual[i], precision: 5);
        }
    }

    /// <summary>
    /// The mask must actually be doing something, or the theories above would pass on a no-op.
    /// </summary>
    /// <remarks>
    /// Without this, a broadcast bug that resolved to "everything visible" would agree with a
    /// materialized mask that also resolved to "everything visible", and both theories would pass
    /// while masking was silently disabled.
    /// </remarks>
    [Fact]
    public void CausalMask_ChangesTheOutput_SoTheBroadcastAssertionsAreNotVacuous()
    {
        var engine = new CpuEngine();
        var q = Fill<double>(v => v);
        var k = Fill<double>(v => v * 0.5 + 0.1);
        var v2 = Fill<double>(v => v * -0.3 + 0.2);

        var unmasked = engine.ScaledDotProductAttention(q, k, v2, null, scale: null, out _);
        var masked = engine.ScaledDotProductAttention(
            q, k, v2, CausalMask(1, 1), scale: null, out _);

        bool differs = false;
        for (int i = 0; i < unmasked.Length && !differs; i++)
        {
            if (Math.Abs(unmasked[i] - masked[i]) > 1e-9) differs = true;
        }

        Assert.True(differs, "The causal mask changed nothing, so the broadcast tests prove nothing.");
    }
}
