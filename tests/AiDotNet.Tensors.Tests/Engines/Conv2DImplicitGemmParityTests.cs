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
[Collection("BlasManaged-Stats-Serial")]
public class Conv2DImplicitGemmParityTests
{
    // The oneDNN / implicit / full route chain is compiled only on .NET Core (CpuEngine.Conv2DWithIm2ColFloat), so the
    // routes these tests compare and pin do not exist on .NET Framework.
#if NETCOREAPP
    private const bool RouteChainCompiled = true;
#else
    private const bool RouteChainCompiled = false;
#endif

    // An unpinned conv reaches that chain only on Windows.
    private static bool UnpinnedRouteChainRuns =>
        RouteChainCompiled
        && System.Runtime.InteropServices.RuntimeInformation.IsOSPlatform(System.Runtime.InteropServices.OSPlatform.Windows);

    // oneDNN, where installed, takes the conv before the chain's own routes; its run counter says whether it did.
    private static int OneDnnConvRuns =>
#if NETCOREAPP
        AiDotNet.Tensors.Helpers.OneDnnProvider.Conv2DRunsOnThisThread;
#else
        0;
#endif

    private static Tensor<float> Rand(int[] shape, int seed)
    {
        var rng = new Random(seed);
        var t = new Tensor<float>(shape);
        var s = t.AsWritableSpan();
        for (int i = 0; i < s.Length; i++) s[i] = (float)(rng.NextDouble() - 0.5);
        return t;
    }

    [Fact]
    public void RowBlockedIm2Col_IsBitIdentical_ToFullIm2Col()
    {
        const int channels = 320;
        const int height = 32;
        const int width = 32;
        const int kernelSize = 3;
        const int padding = 1;
        const int outputHeight = 32;
        const int outputWidth = 32;
        const int rowsPerBlock = 22;
        const int colH = channels * kernelSize * kernelSize;
        const int colW = outputHeight * outputWidth;

        var input = Rand(new[] { 1, channels, height, width }, 101);
        var full = new float[colH * colW];
        AiDotNet.Tensors.Helpers.Im2ColHelper.Im2ColChannelParallel(
            input.AsSpan(), full,
            channels, height, width,
            kernelSize, kernelSize, 1, 1, padding, padding, 1, 1,
            outputHeight, outputWidth);

        for (int ohStart = 0; ohStart < outputHeight; ohStart += rowsPerBlock)
        {
            int ohEnd = Math.Min(ohStart + rowsPerBlock, outputHeight);
            int blockWidth = (ohEnd - ohStart) * outputWidth;
            var block = new float[colH * blockWidth];
            AiDotNet.Tensors.Helpers.Im2ColHelper.Im2ColRowBlockFloat(
                input.AsSpan(), block,
                channels, height, width,
                kernelSize, kernelSize, 1, 1, padding, padding, 1, 1,
                outputHeight, outputWidth, ohStart, ohEnd);

            int fullColumnOffset = ohStart * outputWidth;
            for (int row = 0; row < colH; row++)
            {
                var expected = full.AsSpan(row * colW + fullColumnOffset, blockWidth);
                var actual = block.AsSpan(row * blockWidth, blockWidth);
                Assert.True(expected.SequenceEqual(actual),
                    $"blocked im2col diverged in row {row}, output rows [{ohStart}, {ohEnd})");
            }
        }
    }

    [SkippableTheory]
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
        Skip.IfNot(RouteChainCompiled, "The implicit and full conv routes are compiled only on .NET Core.");
        var e = new CpuEngine();
        var x = Rand(new[] { batch, inC, hw, hw }, 101);
        var kernel = Rand(new[] { outC, inC, k, k }, 202);
        var strideArr = new[] { stride, stride };
        var padArr = new[] { pad, pad };
        var dilArr = new[] { dilation, dilation };

        // Fused (implicit-GEMM) path, pinned: short-wide shapes now default to the full im2col path,
        // and pinning keeps this a real comparison if that routing ever widens to these shapes.
        float[] fused;
        bool pointwise = k == 1 && stride == 1 && pad == 0 && dilation == 1;   // 1x1 identity always takes the full path
        int implicitBefore = CpuEngine.ImplicitGemmConvRunsOnThisThread;
        using (CpuEngine.ForceImplicitGemmScope())
        {
            fused = e.Conv2D(x, kernel, strideArr, padArr, dilArr).ToArray();
        }
        // The positive control: at 8+ threads the default route is the full im2col GEMM too, and a comparison of the full
        // path with itself would pass whatever the implicit path computes.
        if (!pointwise)
            Assert.True(CpuEngine.ImplicitGemmConvRunsOnThisThread > implicitBefore, "the implicit-GEMM path did not run");

        // Full im2col baseline via a thread-local, auto-restoring scope — no process-wide flag, so
        // this can't perturb a concurrent Conv2D or leak into sibling tests.
        float[] full;
        int fullBefore = CpuEngine.FullIm2ColConvRunsOnThisThread;
        using (CpuEngine.ForceFullIm2ColScope())
        {
            full = e.Conv2D(x, kernel, strideArr, padArr, dilArr).ToArray();
        }
        Assert.True(CpuEngine.FullIm2ColConvRunsOnThisThread > fullBefore, "the full im2col path did not run");

        Assert.Equal(full.Length, fused.Length);
        // Bit-exact: identical GEMM, identical per-element K-reduction, disjoint column blocks.
        for (int i = 0; i < full.Length; i++)
        {
            if (full[i] != fused[i])
                Assert.Fail($"fused conv diverged at index {i}: full={full[i]:R} fused={fused[i]:R} " +
                            $"(batch={batch} inC={inC} outC={outC} k={k} hw={hw} s={stride} p={pad} d={dilation})");
        }
    }

    // #653: the GEMM plan follows the shape and the thread count (the full route's own GEMM where it is the default, at
    // 8+ threads or on a 64x64+ plane; parallel blocks on fixed-blocking GEMMs elsewhere), so the routes must agree under
    // each plan. The shapes cover a 14-wide row (a block edge inside a 16-column tile), output channels that are not a
    // multiple of the 48-row chunk, an 8x8 plane split over output channels, and a batch.
    [SkippableTheory]
    [InlineData(1, 256, 256, 3, 16, 1)]
    [InlineData(1, 256, 256, 3, 16, 4)]
    [InlineData(1, 256, 256, 3, 16, 16)]
    [InlineData(1, 320, 300, 3, 14, 4)]
    [InlineData(1, 320, 300, 3, 14, 16)]
    [InlineData(1, 512, 512, 3, 8, 4)]
    [InlineData(1, 512, 512, 3, 8, 16)]
    [InlineData(2, 256, 136, 3, 24, 3)]
    [InlineData(1, 128, 64, 3, 64, 16)]
    [InlineData(1, 512, 48, 3, 63, 16)]    // 73 MB matrix keeps the implicit route; 16-row tile blocks cap it at 3 workers
    [InlineData(1, 64, 64, 3, 64, 4)]      // K = 576: past the row-block conv, a low-channel shape on the implicit route
    [InlineData(1, 64, 64, 3, 64, 1)]      // single thread: both routes' block GEMMs take the default (Goto) route
    [InlineData(1, 32, 32, 3, 128, 1)]
    public void FusedConv_IsBitIdentical_ToFullIm2Col_AtThreadCount(int batch, int inC, int outC, int k, int hw, int threads)
    {
        int previous = AiDotNet.Tensors.Helpers.CpuParallelSettings.MaxDegreeOfParallelism;
        try
        {
            AiDotNet.Tensors.Helpers.CpuParallelSettings.MaxDegreeOfParallelism = threads;
            FusedConv_IsBitIdentical_ToFullIm2Col(batch, inC, outC, k, hw, 1, 1, 1);
        }
        finally
        {
            AiDotNet.Tensors.Helpers.CpuParallelSettings.MaxDegreeOfParallelism = previous;
        }
    }

    // #653: a large-spatial 3x3 whose K (inChannels * 9) is past the row-block conv's 512 takes the parallel implicit GEMM,
    // not the full-matrix N-parallel GEMM that collapsed there (64->64ch@128x128: 27.5 ms at 16 threads against 3.1).
    [SkippableTheory]
    [InlineData(64, 64, 128, 4)]
    [InlineData(64, 32, 128, 16)]
    [InlineData(96, 96, 64, 16)]
    public void LargeSpatialConv_PastRowBlockK_TakesImplicitGemm(int inC, int outC, int hw, int threads)
    {
        Skip.IfNot(UnpinnedRouteChainRuns, "The adaptive conv route chain runs only on Windows under .NET Core.");
        var e = new CpuEngine();
        var x = Rand(new[] { 1, inC, hw, hw }, 505);
        var kernel = Rand(new[] { outC, inC, 3, 3 }, 606);
        int previous = AiDotNet.Tensors.Helpers.CpuParallelSettings.MaxDegreeOfParallelism;
        int fullBefore = CpuEngine.FullIm2ColConvRunsOnThisThread, implicitBefore = CpuEngine.ImplicitGemmConvRunsOnThisThread, oneDnnBefore = OneDnnConvRuns;
        try
        {
            AiDotNet.Tensors.Helpers.CpuParallelSettings.MaxDegreeOfParallelism = threads;
            // This pins the choice between the two im2col routes; single-image 64x64 planes now go to the direct
            // kernels first (DirectConvRouteTests), so those are kept out of the way on this thread.
            using (DirectConvAvx2Suppression())
                e.Conv2D(x, kernel, new[] { 1, 1 }, new[] { 1, 1 }, new[] { 1, 1 });
        }
        finally
        {
            AiDotNet.Tensors.Helpers.CpuParallelSettings.MaxDegreeOfParallelism = previous;
        }

        Skip.If(OneDnnConvRuns > oneDnnBefore, "oneDNN handled the convolution on this host.");
        Assert.Equal(0, CpuEngine.FullIm2ColConvRunsOnThisThread - fullBefore);
        Assert.Equal(1, CpuEngine.ImplicitGemmConvRunsOnThisThread - implicitBefore);
    }

    private static IDisposable DirectConvAvx2Suppression()
    {
#if NET5_0_OR_GREATER
        return AiDotNet.Tensors.Engines.Simd.DirectConvAvx2.SuppressOnThisThread();
#else
        return new NoScope();
#endif
    }

    private sealed class NoScope : IDisposable { public void Dispose() { } }

    // #653: high-channel 3x3 convs take the full im2col GEMM at 8+ threads, on one thread (where its matrix feeds the
    // serial Goto GEMM) or on a 64x64+ plane, while the column matrix fits ThreadLocalTensorCache's 64 MB; the implicit
    // GEMM otherwise. Single-threaded 320ch@32x32 measured full 36.8-44.4 ms vs implicit 41.4-48.5 (medians).
    [SkippableTheory]
    [InlineData(320, 32, 16, true)]    // 320ch@32x32 at 16 threads: 11.8 MB matrix -> full
    [InlineData(320, 32, 1, true)]     // single-threaded: full im2col + serial Goto GEMM
    [InlineData(320, 32, 4, false)]    // 2-7 threads keep the implicit path (equal or better there)
    [InlineData(320, 64, 1, true)]     // a 64x64 plane takes the full path at any thread count (47 MB)
    [InlineData(640, 64, 16, false)]   // 94 MB matrix: past the cache, keeps the implicit path
    public void HighChannelConv_RoutesByThreadsAndMatrixSize(int channels, int hw, int threads, bool expectFull)
    {
        Skip.IfNot(UnpinnedRouteChainRuns, "The adaptive conv route chain runs only on Windows under .NET Core.");
        var e = new CpuEngine();
        var x = Rand(new[] { 1, channels, hw, hw }, 303);
        var kernel = Rand(new[] { channels, channels, 3, 3 }, 404);
        int previous = AiDotNet.Tensors.Helpers.CpuParallelSettings.MaxDegreeOfParallelism;
        int fullBefore = CpuEngine.FullIm2ColConvRunsOnThisThread, implicitBefore = CpuEngine.ImplicitGemmConvRunsOnThisThread, oneDnnBefore = OneDnnConvRuns;
        try
        {
            AiDotNet.Tensors.Helpers.CpuParallelSettings.MaxDegreeOfParallelism = threads;
            // This pins the choice between the two im2col routes; single-image 64x64 planes now go to the direct
            // kernels first (DirectConvRouteTests), so those are kept out of the way on this thread.
            using (DirectConvAvx2Suppression())
                e.Conv2D(x, kernel, new[] { 1, 1 }, new[] { 1, 1 }, new[] { 1, 1 });
        }
        finally
        {
            AiDotNet.Tensors.Helpers.CpuParallelSettings.MaxDegreeOfParallelism = previous;
        }
        // On that chain oneDNN, where installed, may take the conv first; otherwise one of the two routes must have run.
        Skip.If(OneDnnConvRuns > oneDnnBefore, "oneDNN handled the convolution on this host.");
        Assert.Equal(expectFull ? 1 : 0, CpuEngine.FullIm2ColConvRunsOnThisThread - fullBefore);
        Assert.Equal(expectFull ? 0 : 1, CpuEngine.ImplicitGemmConvRunsOnThisThread - implicitBefore);
    }
}
