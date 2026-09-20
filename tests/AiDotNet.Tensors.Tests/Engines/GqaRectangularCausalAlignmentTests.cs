using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// The two grouped-query attention entry points on one engine must place the causal window
/// identically when the query block is shorter than the KV history.
///
/// <para><b>The divergence.</b> <see cref="CpuEngine.ScaledDotProductAttentionGqa{T}"/> builds its
/// mask as <c>j &lt;= i + (seqK - seqQ)</c> — bottom-right, carrying the comment "KV-cache offset:
/// query i is at absolute key position i + offset" — on the eager path, on the graph-traced path,
/// and in the <c>scaled_dot_product_attention</c> kernels of every backend.
/// <see cref="CpuEngine.GroupedQueryAttention{T}"/> masks with <c>ki &gt; qi</c> — top-left, no
/// offset term — on its eager path, on its graph path (<c>queryOffset: 0</c>) and in the
/// <c>grouped_query_attention</c> kernels. Neither documents which alignment it implements, and
/// nothing rejects a rectangular call, so the same operation on the same inputs returns two
/// different tensors depending only on which overload the caller reached for.</para>
///
/// <para><b>Why the bottom-right side is the one that is right.</b> These tests do not assert that
/// the two agree with each other, which would only pin them together and could be satisfied by
/// making both wrong. They assert an independent fact about attention: a single decode query
/// appended to a KV cache of length N sits at absolute position N-1, so nothing in the cache is in
/// its future and the causal mask must remove no keys at all. Its causal output therefore has to
/// equal its noncausal output. Top-left alignment instead lets that query see only key 0, which is
/// the whole cache discarded — the failure is not a rounding difference but the model losing its
/// context.</para>
/// </summary>
public sealed class GqaRectangularCausalAlignmentTests
{
    private const int Batch = 1;
    private const int Heads = 2;
    private const int HeadDim = 4;
    private const int CacheLength = 4;
    private const float Tolerance = 1e-5f;

    private static readonly double Scale = 1.0 / Math.Sqrt(HeadDim);

    /// <summary>Deterministic, non-degenerate values: a constant or symmetric fill would make
    /// every masking convention agree and the test vacuous.</summary>
    private static Tensor<float> Data(int[] shape, double phase)
    {
        int length = 1;
        foreach (int d in shape) length *= d;
        var data = new float[length];
        for (int i = 0; i < length; i++)
            data[i] = (float)Math.Sin(phase + (i * 0.37));

        return new Tensor<float>(data, shape);
    }

    /// <summary>
    /// THE REGRESSION, stated without reference to either implementation. With one query and a
    /// four-key cache the causal mask is vacuous, so causal and noncausal must coincide.
    /// <see cref="CpuEngine.ScaledDotProductAttentionGqa{T}"/> satisfies this; before the fix
    /// <see cref="CpuEngine.GroupedQueryAttention{T}"/> returns attention over key 0 alone.
    /// </summary>
    [Fact]
    public void ASingleDecodeQuerySeesTheWholeKvCache()
    {
        var engine = new CpuEngine();
        var query = Data(new[] { Batch, Heads, 1, HeadDim }, 0.3);
        var key = Data(new[] { Batch, Heads, CacheLength, HeadDim }, 1.1);
        var value = Data(new[] { Batch, Heads, CacheLength, HeadDim }, 2.5);

        var causal = engine.GroupedQueryAttention(query, key, value, 1, Scale, true, out _);
        var noncausal = engine.GroupedQueryAttention(query, key, value, 1, Scale, false, out _);

        // Guard against a vacuous pass: if attending key 0 alone happened to equal attending all
        // four, no alignment could be distinguished and agreement would prove nothing.
        Assert.False(
            Close(noncausal.ToArray(), FirstKeyRow(value)),
            "Fixture is degenerate: attending the whole cache already equals attending key 0, so " +
            "this test cannot tell the two causal alignments apart.");

        AssertClose(noncausal.ToArray(), causal.ToArray());
    }

    /// <summary>
    /// The same fact through the sibling entry point, which pins the reference arm rather than
    /// assuming it: were this to fail, the oracle above would be measuring nothing.
    /// </summary>
    [Fact]
    public void ASingleDecodeQuerySeesTheWholeKvCacheThroughTheSdpaGqaEntryPoint()
    {
        var engine = new CpuEngine();
        var query = Data(new[] { Batch, Heads, 1, HeadDim }, 0.3);
        var key = Data(new[] { Batch, Heads, CacheLength, HeadDim }, 1.1);
        var value = Data(new[] { Batch, Heads, CacheLength, HeadDim }, 2.5);

        var causal = engine.ScaledDotProductAttentionGqa(query, key, value, Scale, true);
        var noncausal = engine.ScaledDotProductAttentionGqa(query, key, value, Scale, false);

        AssertClose(noncausal.ToArray(), causal.ToArray());
    }

    /// <summary>
    /// Two entry points, one operation: having pinned each to the independent oracle above, they
    /// must also agree with each other on a rectangular causal call.
    /// </summary>
    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    public void BothGqaEntryPointsAgreeOnRectangularCausalAttention(int queryCount)
    {
        var engine = new CpuEngine();
        var query = Data(new[] { Batch, Heads, queryCount, HeadDim }, 0.3);
        var key = Data(new[] { Batch, Heads, CacheLength, HeadDim }, 1.1);
        var value = Data(new[] { Batch, Heads, CacheLength, HeadDim }, 2.5);

        var viaGqa = engine.GroupedQueryAttention(query, key, value, 1, Scale, true, out _);
        var viaSdpa = engine.ScaledDotProductAttentionGqa(query, key, value, Scale, true);

        AssertClose(viaSdpa.ToArray(), viaGqa.ToArray());
    }

    /// <summary>
    /// Ordinary square self-attention must be untouched: seqK - seqQ is zero there, so the two
    /// alignments coincide and the fix is a no-op. Without this, a "fix" that shifted every mask
    /// would look green above while silently changing every training run.
    /// </summary>
    [Fact]
    public void SquareCausalAttentionIsUnchangedAndStillMasksTheFuture()
    {
        var engine = new CpuEngine();
        var query = Data(new[] { Batch, Heads, CacheLength, HeadDim }, 0.3);
        var key = Data(new[] { Batch, Heads, CacheLength, HeadDim }, 1.1);
        var value = Data(new[] { Batch, Heads, CacheLength, HeadDim }, 2.5);

        var causal = engine.GroupedQueryAttention(query, key, value, 1, Scale, true, out _);
        var noncausal = engine.GroupedQueryAttention(query, key, value, 1, Scale, false, out _);

        // Square causal genuinely hides the future, so it must NOT equal noncausal -- the opposite
        // of the rectangular case, and the check that the mask still exists at all.
        Assert.False(
            Close(noncausal.ToArray(), causal.ToArray()),
            "Square causal attention matched noncausal attention: the causal mask is not being applied.");

        AssertClose(
            engine.ScaledDotProductAttentionGqa(query, key, value, Scale, true).ToArray(),
            causal.ToArray());
    }

    /// <summary>Output of attending only key 0: softmax over a single key is 1, so the result is
    /// value row 0 repeated for every query.</summary>
    private static float[] FirstKeyRow(Tensor<float> value)
    {
        var v = value.ToArray();
        var expected = new float[Batch * Heads * HeadDim];
        int index = 0;
        for (int b = 0; b < Batch; b++)
            for (int h = 0; h < Heads; h++)
                for (int d = 0; d < HeadDim; d++)
                    expected[index++] = v[(((b * Heads) + h) * CacheLength * HeadDim) + d];

        return expected;
    }

    private static bool Close(float[] expected, float[] actual)
    {
        if (expected.Length != actual.Length) return false;
        for (int i = 0; i < expected.Length; i++)
            if (Math.Abs(expected[i] - actual[i]) > Tolerance) return false;

        return true;
    }

    private static void AssertClose(float[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        int worst = -1;
        float worstDiff = 0f;
        for (int i = 0; i < expected.Length; i++)
        {
            float diff = Math.Abs(expected[i] - actual[i]);
            if (diff > worstDiff) { worstDiff = diff; worst = i; }
        }

        Assert.True(
            worstDiff <= Tolerance,
            worst < 0
                ? "unexpected comparison failure"
                : $"Worst mismatch at [{worst}]: expected={expected[worst]:G6}, actual={actual[worst]:G6}, diff={worstDiff:G6}");
    }
}
