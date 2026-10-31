using AiDotNet.Tensors.Engines.DirectGpu;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// The portable fused attention kernels (OpenCL, HIP, Vulkan, Metal, WebGPU) run only where the composed attention would
/// materialize large score matrices (over 64M elements, 256 MB): below that two batched GEMMs around a softmax win (a
/// ViT's 65-token attention trained 5.7x faster composed on OpenCL, with the same loss). Long sequences keep the fused
/// kernels, whose point is never building the score matrices.
/// </summary>
public class PortableFusedAttentionPreferenceTests
{
    [Theory]
    [InlineData(64, 3, 65, 65, false)]        // ViT-small at batch 64: 0.8M scores
    [InlineData(8, 12, 512, 512, false)]      // a 512-token encoder: 25M
    [InlineData(1, 32, 2048, 2048, true)]     // a 2K-token decoder: 134M
    [InlineData(8, 16, 4096, 4096, true)]     // 2.1G: the composed form could not hold it
    public void PrefersFused_OnlyWhereTheScoresWouldBeLarge(int batch, int heads, int seqQ, int seqK, bool fused)
        => Assert.Equal(fused, PortableFusedAttention.PrefersFused(batch, heads, seqQ, seqK));

    [Fact]
    public void PrefersFused_TheBoundaryIs64MScores()
    {
        Assert.False(PortableFusedAttention.PrefersFused(1, 1, 8192, 8192));   // exactly 64M
        Assert.True(PortableFusedAttention.PrefersFused(1, 1, 8192, 8193));
    }
}
