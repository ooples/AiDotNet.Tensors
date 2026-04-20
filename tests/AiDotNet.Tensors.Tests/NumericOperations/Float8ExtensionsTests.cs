using AiDotNet.Tensors.NumericOperations;
using Xunit;

namespace AiDotNet.Tensors.Tests.NumericOperations;

/// <summary>
/// Issue #197 sub-feature 1 — Float8Extensions.
/// </summary>
public class Float8ExtensionsTests
{
    [Fact]
    public void ToE4M3_FromFloat_MatchesDirectFromFloat()
    {
        Assert.Equal(Float8E4M3.FromFloat(3.14f), 3.14f.ToE4M3());
    }

    [Fact]
    public void ToE4M3_FromDouble_CastsToFloatFirst()
    {
        Assert.Equal(Float8E4M3.FromFloat(2.5f), 2.5.ToE4M3());
    }

    [Fact]
    public void ToE4M3_FromHalf_CastsToFloatFirst()
    {
        Half h = (Half)1.5f;
        Assert.Equal(Float8E4M3.FromFloat(1.5f), h.ToE4M3());
    }

    [Fact]
    public void E4M3_ToHalf_RoundTrips()
    {
        var e4 = Float8E4M3.FromFloat(2.0f);
        Assert.Equal(2.0f, (float)e4.ToHalf(), 1);
    }

    [Fact]
    public void E4M3_ToDouble_RoundTrips()
    {
        var e4 = Float8E4M3.FromFloat(8.0f);
        Assert.Equal(8.0, e4.ToDouble(), 1);
    }

    [Fact]
    public void BulkE4M3_RoundTripsWithinTolerance()
    {
        var src = new float[] { 0f, 1f, -1f, 0.5f, -0.5f, 100f };
        var packed = new Float8E4M3[src.Length];
        Float8Extensions.ToE4M3Array(src, packed);
        var back = new float[src.Length];
        Float8Extensions.ToFloatArray(packed, back);
        for (int i = 0; i < src.Length; i++)
        {
            if (src[i] == 0f) Assert.Equal(0f, back[i]);
            else
            {
                float relErr = Math.Abs(back[i] - src[i]) / Math.Abs(src[i]);
                Assert.True(relErr < 0.14f);
            }
        }
    }

    [Fact]
    public void BulkE4M3_Dst_TooSmall_Throws()
    {
        var src = new float[4];
        var dst = new Float8E4M3[2];
        Assert.Throws<ArgumentException>(() => Float8Extensions.ToE4M3Array(src, dst));
    }

    [Fact]
    public void ToE5M2_FromFloat_MatchesDirectFromFloat()
    {
        Assert.Equal(Float8E5M2.FromFloat(5.5f), 5.5f.ToE5M2());
    }

    [Fact]
    public void BulkE5M2_RoundTripsWithinTolerance()
    {
        var src = new float[] { 0f, 1f, -1f, 1000f, -1000f };
        var packed = new Float8E5M2[src.Length];
        Float8Extensions.ToE5M2Array(src, packed);
        var back = new float[src.Length];
        Float8Extensions.ToFloatArray(packed, back);
        for (int i = 0; i < src.Length; i++)
        {
            if (src[i] == 0f) Assert.Equal(0f, back[i]);
            else
            {
                float relErr = Math.Abs(back[i] - src[i]) / Math.Abs(src[i]);
                Assert.True(relErr < 0.26f, $"rel err {relErr} on {src[i]}");
            }
        }
    }
}
