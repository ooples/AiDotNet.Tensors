using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// Correctness tests for <c>IEngine.TensorBroadcastTo</c> — the three-tier
/// broadcast primitive that replaces the
/// <c>TensorBroadcastAdd(x, new Tensor&lt;T&gt;(targetShape))</c> anti-pattern.
///
/// <para>Tier 1 (identity): shapes already equal → returned as-is, zero cost.
/// Tier 2 (leading-1s): source shape matches target tail, target prepends only
/// size-1 axes → dispatched to Reshape (zero data copy).
/// Tier 3 (general): genuine size-1 → size-N expansion → full materialisation
/// via existing broadcast kernel.</para>
///
/// <para>Each test both verifies the numerical result and, where possible,
/// asserts the fast-path was taken (tier-1 returns the same instance;
/// tier-2 returns a reshape-view backed by the same storage).</para>
/// </summary>
public class TensorBroadcastToTests
{
    private readonly CpuEngine _engine = new();

    [Fact]
    public void Tier1_ShapesEqual_ReturnsSameInstance()
    {
        var input = MakeSequential(new[] { 2, 3, 4 });
        var result = _engine.TensorBroadcastTo(input, new[] { 2, 3, 4 });
        Assert.Same(input, result);
    }

    [Fact]
    public void Tier2_LeadingOneAdded_ReshapesWithSameValues()
    {
        // [3, 4] → [1, 3, 4] is a pure rank-insert. Data is the same.
        var input = MakeSequential(new[] { 3, 4 });
        var result = _engine.TensorBroadcastTo(input, new[] { 1, 3, 4 });

        Assert.Equal(new[] { 1, 3, 4 }, result.Shape.ToArray());
        Assert.Equal(12, result.Length);
        for (int i = 0; i < 12; i++)
            Assert.Equal(input.AsSpan()[i], result.AsSpan()[i]);
    }

    [Fact]
    public void Tier2_MultipleLeadingOnesAdded_StillReshapes()
    {
        var input = MakeSequential(new[] { 5, 6 });
        var result = _engine.TensorBroadcastTo(input, new[] { 1, 1, 1, 5, 6 });

        Assert.Equal(new[] { 1, 1, 1, 5, 6 }, result.Shape.ToArray());
        for (int i = 0; i < input.Length; i++)
            Assert.Equal(input.AsSpan()[i], result.AsSpan()[i]);
    }

    [Fact]
    public void Tier3_MiddleAxisExpand_ReplicatesValues()
    {
        // [2, 1, 3] → [2, 4, 3]: middle axis expands from 1 to 4.
        // Element at (i, j, k) should equal input at (i, 0, k).
        var input = MakeSequential(new[] { 2, 1, 3 });
        var result = _engine.TensorBroadcastTo(input, new[] { 2, 4, 3 });

        Assert.Equal(new[] { 2, 4, 3 }, result.Shape.ToArray());
        var rSpan = result.AsSpan();
        var iSpan = input.AsSpan();
        // For each [i, j, k] in output, compare to input[i, 0, k].
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 4; j++)
                for (int k = 0; k < 3; k++)
                {
                    int outIdx = ((i * 4) + j) * 3 + k;
                    int inIdx  = (i * 1 + 0) * 3 + k;
                    Assert.Equal(iSpan[inIdx], rSpan[outIdx]);
                }
    }

    [Fact]
    public void Tier3_LeadingAxisExpand_ReplicatesValues()
    {
        // [1, 3] → [4, 3]: leading axis expands from 1 to 4.
        // Each row of output equals the single input row.
        var input = MakeSequential(new[] { 1, 3 });
        var result = _engine.TensorBroadcastTo(input, new[] { 4, 3 });

        Assert.Equal(new[] { 4, 3 }, result.Shape.ToArray());
        for (int row = 0; row < 4; row++)
            for (int col = 0; col < 3; col++)
                Assert.Equal(input.AsSpan()[col], result.AsSpan()[row * 3 + col]);
    }

    [Fact]
    public void Tier3_ScalarLikeInput_FillsTarget()
    {
        // [1, 1] → [3, 4]: every output element equals the single input value.
        var input = new Tensor<float>(new[] { 1, 1 });
        input.AsWritableSpan()[0] = 7.25f;
        var result = _engine.TensorBroadcastTo(input, new[] { 3, 4 });

        Assert.Equal(new[] { 3, 4 }, result.Shape.ToArray());
        for (int i = 0; i < 12; i++) Assert.Equal(7.25f, result.AsSpan()[i]);
    }

    [Fact]
    public void IncompatibleShapes_Throws()
    {
        // [3, 4] cannot broadcast to [3, 5] — dim 1 is 4 vs 5, neither is 1.
        var input = MakeSequential(new[] { 3, 4 });
        var ex = Assert.Throws<ArgumentException>(() =>
            _engine.TensorBroadcastTo(input, new[] { 3, 5 }));
        Assert.Contains("broadcast", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void TierParityVsBroadcastAdd_MiddleAxisExpand()
    {
        // Semantic parity: TensorBroadcastTo(x, target) must produce the
        // same values as the old idiom TensorBroadcastAdd(x, zeros(target)).
        var input = MakeSequential(new[] { 2, 1, 3 });
        var target = new[] { 2, 5, 3 };
        var newPath = _engine.TensorBroadcastTo(input, target);
        var oldPath = _engine.TensorBroadcastAdd(input, new Tensor<float>(target));

        Assert.Equal(newPath.Shape.ToArray(), oldPath.Shape.ToArray());
        for (int i = 0; i < newPath.Length; i++)
            Assert.Equal(oldPath.AsSpan()[i], newPath.AsSpan()[i]);
    }

    [Fact]
    public void NullInputTensor_Throws()
    {
        Assert.Throws<ArgumentNullException>(() =>
            _engine.TensorBroadcastTo<float>(null!, new[] { 1, 2 }));
    }

    [Fact]
    public void NullTargetShape_Throws()
    {
        var input = new Tensor<float>(new[] { 1, 2 });
        Assert.Throws<ArgumentNullException>(() =>
            _engine.TensorBroadcastTo(input, null!));
    }

    // ─── helpers ────────────────────────────────────────────────────────────

    private static Tensor<float> MakeSequential(int[] shape)
    {
        int total = 1; foreach (var d in shape) total *= d;
        var t = new Tensor<float>(shape);
        var s = t.AsWritableSpan();
        for (int i = 0; i < total; i++) s[i] = i + 1f;
        return t;
    }
}
