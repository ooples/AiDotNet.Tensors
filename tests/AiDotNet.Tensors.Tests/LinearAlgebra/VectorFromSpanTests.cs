using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.LinearAlgebra;

public class VectorFromSpanTests
{
    [Fact]
    public void FromSpan_Float_CopiesCorrectly()
    {
        float[] source = [1.5f, 2.5f, 3.5f, 4.5f, 5.5f];
        var vec = Vector<float>.FromSpan(source.AsSpan());

        Assert.Equal(5, vec.Length);
        for (int i = 0; i < source.Length; i++)
            Assert.Equal(source[i], vec[i]);
    }

    [Fact]
    public void FromSpan_Double_CopiesCorrectly()
    {
        double[] source = [1.1, 2.2, 3.3];
        var vec = Vector<double>.FromSpan(source.AsSpan());

        Assert.Equal(3, vec.Length);
        for (int i = 0; i < source.Length; i++)
            Assert.Equal(source[i], vec[i]);
    }

    [Fact]
    public void FromSpan_EmptySpan_CreatesEmptyVector()
    {
        var vec = Vector<float>.FromSpan(ReadOnlySpan<float>.Empty);
        Assert.Equal(0, vec.Length);
    }

    [Fact]
    public void FromSpan_IsDeepCopy_SourceMutationDoesNotAffectVector()
    {
        float[] source = [10f, 20f, 30f];
        var vec = Vector<float>.FromSpan(source.AsSpan());

        // Mutate source after creating vector
        source[0] = 999f;
        source[1] = 888f;

        // Vector should be unaffected
        Assert.Equal(10f, vec[0]);
        Assert.Equal(20f, vec[1]);
        Assert.Equal(30f, vec[2]);
    }

    [Fact]
    public void FromSpan_LargeSpan_WorksCorrectly()
    {
        var source = new double[10000];
        for (int i = 0; i < source.Length; i++) source[i] = i * 0.001;

        var vec = Vector<double>.FromSpan(source.AsSpan());

        Assert.Equal(10000, vec.Length);
        Assert.Equal(0.0, vec[0]);
        Assert.Equal(9.999, vec[9999], 6);
    }
}
