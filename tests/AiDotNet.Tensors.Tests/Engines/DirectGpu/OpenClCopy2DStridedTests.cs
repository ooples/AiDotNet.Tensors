#if NET6_0_OR_GREATER
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// OpenCL Copy2DStrided: each source row into a column slice of a wider destination row (a concatenation along the last
/// axis), by the four-column kernel where every row starts 16-byte aligned and by the scalar kernel otherwise. Columns
/// outside the slice keep their values.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class OpenClCopy2DStridedTests
{
    [SkippableTheory]
    [InlineData(3, 8, 20, 4)]       // aligned: the float4 kernel
    [InlineData(32, 1024, 3072, 2048)]
    [InlineData(3, 7, 19, 5)]       // unaligned: the scalar kernel
    [InlineData(5, 12, 30, 2)]      // aligned columns, unaligned offset: scalar
    [InlineData(1, 300, 300, 0)]
    public void OpenCl_Copy2DStrided_MatchesReference(int rows, int srcCols, int destTotalCols, int destColOffset)
    {
        using var backend = new OpenClBackend();
        Skip.IfNot(backend.IsAvailable, "OpenCL is not available.");
        var src = new float[rows * srcCols];
        for (int i = 0; i < src.Length; i++) src[i] = i * 0.5f + 1f;
        var dst = new float[rows * destTotalCols];
        for (int i = 0; i < dst.Length; i++) dst[i] = -1f - i;
        var expected = (float[])dst.Clone();
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < srcCols; c++)
                expected[r * destTotalCols + destColOffset + c] = src[r * srcCols + c];
        using var srcBuffer = backend.AllocateBuffer(src);
        using var dstBuffer = backend.AllocateBuffer(dst);
        backend.Copy2DStrided(srcBuffer, dstBuffer, rows, srcCols, destTotalCols, destColOffset);
        Assert.Equal(expected, backend.DownloadBuffer(dstBuffer));
    }
}
#endif