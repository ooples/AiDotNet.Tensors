using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

public class RoPETests
{
    [Fact]
    public void PositionZero_IsIdentity()
    {
        // pos 0 → theta = 0 → cos=1 sin=0 → no rotation.
        var t = UnitRows(new[] { 1, 1, 1, 4 });
        var before = t.AsSpan().ToArray();
        RoPE.Apply(t, startPosition: 0);
        Assert.Equal(before, t.AsSpan().ToArray());
    }

    [Fact]
    public void Interleaved_ProducesRotation_PreservesNorm()
    {
        // A pure rotation preserves L2 norm of each pair.
        var t = Random(new[] { 1, 1, 4, 8 }, seed: 1);
        var before = t.AsSpan().ToArray();
        RoPE.Apply(t, startPosition: 0, style: RoPEStyle.Interleaved);

        int headDim = 8;
        for (int pos = 0; pos < 4; pos++)
        {
            int rowBase = pos * headDim;
            for (int i = 0; i < headDim / 2; i++)
            {
                float xB = before[rowBase + 2 * i];
                float yB = before[rowBase + 2 * i + 1];
                float xA = t.AsSpan()[rowBase + 2 * i];
                float yA = t.AsSpan()[rowBase + 2 * i + 1];
                float normBefore = xB * xB + yB * yB;
                float normAfter = xA * xA + yA * yA;
                Assert.Equal(normBefore, normAfter, 3);
            }
        }
    }

    [Fact]
    public void HalfRotated_ProducesRotation_PreservesNorm()
    {
        var t = Random(new[] { 2, 2, 4, 6 }, seed: 2);
        var before = t.AsSpan().ToArray();
        RoPE.Apply(t, startPosition: 0, style: RoPEStyle.HalfRotated);

        // For any (batch, head, pos), dim i with dim i+halfDim must preserve norm.
        int B = 2, H = 2, S = 4, D = 6;
        int halfDim = D / 2;
        for (int b = 0; b < B; b++)
        for (int h = 0; h < H; h++)
        for (int s = 0; s < S; s++)
        {
            int rowBase = (((b * H) + h) * S + s) * D;
            for (int i = 0; i < halfDim; i++)
            {
                int a = rowBase + i;
                int c = rowBase + halfDim + i;
                float normBefore = before[a] * before[a] + before[c] * before[c];
                float normAfter = t.AsSpan()[a] * t.AsSpan()[a] + t.AsSpan()[c] * t.AsSpan()[c];
                Assert.Equal(normBefore, normAfter, 3);
            }
        }
    }

    [Fact]
    public void StartPosition_ShiftsAngle()
    {
        // Applying RoPE with start=0 to position 5 should equal applying
        // with start=5 to position 0. (Same absolute position.)
        var a = Random(new[] { 1, 1, 6, 4 }, seed: 3);
        var b = new Tensor<float>(a._shape);
        // b's first row is initialized from a's row 5.
        int headDim = 4;
        var aSpan = a.AsSpan();
        var bSpan = b.AsWritableSpan();
        for (int i = 0; i < headDim; i++) bSpan[i] = aSpan[5 * headDim + i];

        RoPE.Apply(a, startPosition: 0);
        RoPE.Apply(b, startPosition: 5);

        for (int i = 0; i < headDim; i++)
            Assert.Equal(a.AsSpan()[5 * headDim + i], b.AsSpan()[i], 4);
    }

    [Fact]
    public void OddHeadDim_Throws()
    {
        var t = new Tensor<float>(new[] { 1, 1, 1, 5 });
        Assert.Throws<ArgumentException>(() => RoPE.Apply(t));
    }

    [Fact]
    public void WrongRank_Throws()
    {
        var t = new Tensor<float>(new[] { 4 });
        Assert.Throws<ArgumentException>(() => RoPE.Apply(t));
    }

    private static Tensor<float> UnitRows(int[] shape)
    {
        var t = new Tensor<float>(shape);
        var s = t.AsWritableSpan();
        for (int i = 0; i < s.Length; i++) s[i] = i + 1f;
        return t;
    }

    private static Tensor<float> Random(int[] shape, int seed)
    {
        var t = new Tensor<float>(shape);
        var rng = new Random(seed);
        var s = t.AsWritableSpan();
        for (int i = 0; i < s.Length; i++) s[i] = (float)(rng.NextDouble() * 2 - 1);
        return t;
    }
}
