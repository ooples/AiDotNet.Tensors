using System;
using System.Buffers;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// Storage-contract coverage for standalone CPU gradient clipping. These cases
/// deliberately exercise layouts for which GetDataArray() is not a writable
/// alias of the tensor's logical region.
/// </summary>
public sealed class GradientClippingStorageContractTests
{
    [Fact]
    public void ClipGradNorm_PoolPaddedTensor_IgnoresPaddingAndPreservesTail()
    {
        var backing = ArrayPool<float>.Shared.Rent(3);
        var gradient = Tensor<float>.FromPooledMemory(
            new Memory<float>(backing, 0, 3), new[] { 3 }, backing);

        try
        {
            Assert.True(backing.Length > gradient.Length);
            gradient[0] = 3f;
            gradient[1] = 4f;
            gradient[2] = 0f;
            for (int i = gradient.Length; i < backing.Length; i++)
                backing[i] = 1000f + i;
            var expectedTail = backing.AsSpan(gradient.Length).ToArray();
            int versionBefore = gradient.Version;

            float norm = GradientClipping.ClipGradNorm(new[] { gradient }, 2.5f);

            Assert.Equal(5f, norm, 5);
            Assert.Equal(1.5f, gradient[0], 4);
            Assert.Equal(2f, gradient[1], 4);
            Assert.Equal(0f, gradient[2]);
            Assert.Equal(expectedTail, backing.AsSpan(gradient.Length).ToArray());
            Assert.True(gradient.Version > versionBefore);
        }
        finally
        {
            TensorAllocator.Return(gradient);
        }
    }

    [Fact]
    public void ClipGradNorm_OffsetView_WritesOnlyLogicalSlice()
    {
        var backing = new[] { 777f, 3f, 4f, 888f };
        var gradient = Tensor<float>.FromMemory(
            new Memory<float>(backing, 1, 2), new[] { 2 });

        float norm = GradientClipping.ClipGradNorm(new[] { gradient }, 1f);

        float scale = 1f / (5f + 1e-6f);
        Assert.Equal(5f, norm, 5);
        Assert.Equal(777f, backing[0]);
        Assert.Equal(3f * scale, backing[1], 5);
        Assert.Equal(4f * scale, backing[2], 5);
        Assert.Equal(888f, backing[3]);
    }

    [Fact]
    public void ClipGradNorm_LargeFiniteValues_DoesNotOverflowScaleToZero()
    {
        var gradient = new Tensor<float>(new[] { 1e20f, 1e20f }, new[] { 2 });

        float norm = GradientClipping.ClipGradNorm(new[] { gradient }, 1f);

        Assert.False(float.IsNaN(norm));
        Assert.False(float.IsInfinity(norm));
        Assert.Equal(MathF.Sqrt(2f) * 1e20f, norm, precision: 5);
        Assert.Equal(1f / MathF.Sqrt(2f), gradient[0], precision: 5);
        Assert.Equal(1f / MathF.Sqrt(2f), gradient[1], precision: 5);
    }

    [Fact]
    public void ClipGradValue_NonContiguousView_UpdatesUnderlyingStorage()
    {
        var source = new Tensor<float>(
            new[] { 100f, -3f, 2f, -4f, 5f, -100f }, new[] { 2, 3 });
        var transposedView = source.Transpose();
        Assert.False(transposedView.IsContiguous);

        GradientClipping.ClipGradValue(new[] { transposedView }, 2f);

        Assert.Equal(new[] { 2f, -2f, 2f, -2f, 2f, -2f }, source.AsSpan().ToArray());
    }

    [Fact]
    public void ClipGradValue_CopyOnWriteClone_PrivatizesBeforeMutation()
    {
        var untouchedPeer = new Tensor<float>(new[] { -5f, 0.5f, 7f }, new[] { 3 });
        var gradient = (Tensor<float>)untouchedPeer.CloneShared();
        Assert.True(gradient.IsCowShared);

        GradientClipping.ClipGradValue(new[] { gradient }, 1f);

        Assert.Equal(new[] { -1f, 0.5f, 1f }, gradient.AsSpan().ToArray());
        Assert.Equal(new[] { -5f, 0.5f, 7f }, untouchedPeer.AsSpan().ToArray());
        Assert.False(gradient.IsCowShared);
    }

    [Fact]
    public void ClipGradValue_OverlappingZeroStrideView_ThrowsWithoutMutation()
    {
        var source = new Tensor<float>(new[] { 3f, -4f }, new[] { 2 });
        var overlappingView = new Tensor<float>(
            source.DataVector,
            new[] { 2, 2 },
            new[] { 0, 1 },
            storageOffset: 0,
            parentStorage: source._storage);
        int versionBefore = source.Version;

        var error = Assert.Throws<InvalidOperationException>(() =>
            GradientClipping.ClipGradValue(new[] { overlappingView }, 1f));

        Assert.Contains("overlapping tensor views", error.Message);
        Assert.Equal(new[] { 3f, -4f }, source.ToArray());
        Assert.Equal(versionBefore, source.Version);
    }

    [Theory]
    [InlineData(-1f)]
    [InlineData(float.NaN)]
    public void ClipGradNorm_InvalidMaximum_Throws(float maxNorm)
    {
        var gradient = new Tensor<float>(new[] { 1f }, new[] { 1 });
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            GradientClipping.ClipGradNorm(new[] { gradient }, maxNorm));
    }

    [Fact]
    public void ClipGradNorm_InvalidMaximumWithEmptyInput_StillThrows()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            GradientClipping.ClipGradNorm(Array.Empty<Tensor<float>>(), float.NaN));
    }

    [Theory]
    [InlineData(-1f)]
    [InlineData(float.NaN)]
    public void ClipGradValue_InvalidMaximum_Throws(float clipValue)
    {
        var gradient = new Tensor<float>(new[] { 1f }, new[] { 1 });
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            GradientClipping.ClipGradValue(new[] { gradient }, clipValue));
    }
}
