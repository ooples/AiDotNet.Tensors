using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// The high-channel (inChannels >= 256) float forward conv routes through the FUSED
/// (implicit-GEMM) path <c>Conv2DWithImplicitGemmFloat</c>: it blocks on output rows and
/// im2col's each block into a cache-resident panel that is GEMM'd straight into the output,
/// instead of materialising the full [colH, outH*outW] im2col matrix. Because every output
/// column is the exact same kernel·im2col K-reduction issued over disjoint column ranges
/// (same BlasManaged GEMM, just blocked), the result MUST be BIT-IDENTICAL to the full
/// im2col path. This guards that across stride / padding / dilation / batch / spatial-size
/// variations — including deep-bottleneck shapes where the row-block clamps to a single row.
/// </summary>
public class Conv2DImplicitGemmParityTests
{
    private static Tensor<float> Rand(int[] shape, int seed)
    {
        var rng = new Random(seed);
        var t = new Tensor<float>(shape);
        var s = t.AsWritableSpan();
        for (int i = 0; i < s.Length; i++) s[i] = (float)(rng.NextDouble() - 0.5);
        return t;
    }

    [Theory]
    // batch, inC, outC, k, hw, stride, pad, dilation
    [InlineData(1, 256, 256, 3, 16, 1, 1, 1)]   // canonical diffusion ResBlock 3×3
    [InlineData(1, 320, 320, 3, 32, 1, 1, 1)]   // larger spatial → multiple row-blocks
    [InlineData(2, 256, 128, 3, 16, 1, 1, 1)]   // batched
    [InlineData(1, 384, 256, 3, 8, 1, 1, 1)]    // deep bottleneck (small spatial)
    [InlineData(1, 256, 256, 3, 16, 2, 1, 1)]   // stride 2 (general im2col path)
    [InlineData(1, 256, 256, 3, 16, 1, 0, 1)]   // no padding
    [InlineData(1, 256, 256, 3, 16, 1, 2, 2)]   // dilation 2
    [InlineData(1, 512, 256, 1, 16, 1, 0, 1)]   // 1×1 high-channel (routes to full path)
    [InlineData(1, 256, 300, 5, 12, 1, 2, 1)]   // 5×5, non-power-of-two outC
    public void FusedConv_IsBitIdentical_ToFullIm2Col(
        int batch, int inC, int outC, int k, int hw, int stride, int pad, int dilation)
    {
        var e = new CpuEngine();
        var x = Rand(new[] { batch, inC, hw, hw }, 101);
        var kernel = Rand(new[] { outC, inC, k, k }, 202);
        var strideArr = new[] { stride, stride };
        var padArr = new[] { pad, pad };
        var dilArr = new[] { dilation, dilation };

        // Fused (implicit-GEMM) path is the production default.
        float[] fused = e.Conv2D(x, kernel, strideArr, padArr, dilArr).ToArray();

        // Full im2col baseline via a thread-local, auto-restoring scope — no process-wide flag, so
        // this can't perturb a concurrent Conv2D or leak into sibling tests.
        float[] full;
        using (CpuEngine.ForceFullIm2ColScope())
        {
            full = e.Conv2D(x, kernel, strideArr, padArr, dilArr).ToArray();
        }

        Assert.Equal(full.Length, fused.Length);
        // Bit-exact: identical GEMM, identical per-element K-reduction, disjoint column blocks.
        for (int i = 0; i < full.Length; i++)
        {
            if (full[i] != fused[i])
                Assert.Fail($"fused conv diverged at index {i}: full={full[i]:R} fused={fused[i]:R} " +
                            $"(batch={batch} inC={inC} outC={outC} k={k} hw={hw} s={stride} p={pad} d={dilation})");
        }
    }
}
