using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// A narrow that keeps the LEADING elements of its source (start 0) is a contiguous view at storage offset 0 that
/// covers only a prefix of the shared storage. The GPU engine's buffer lookup treated it as the whole tensor: uploading
/// the view cached a buffer of the VIEW's length under the shared storage key, and the next device read of the source
/// got that short buffer ("Buffer A is too small for the specified dimensions" in a GEMM, or a short read). Found in a
/// decode loop that narrows a one-hot position [1, P] to its first column and multiplies the one-hot with a table.
/// A reshape of a prefix view and a shared clone of one are prefix views too. Skips without a GPU backend.
/// </summary>
[Collection("DirectGpuSerial")]
public class NarrowPrefixViewGpuCacheTests
{
    [SkippableFact]
    public void PrefixNarrow_DoesNotPoisonTheSourceDeviceBuffer()
    {
        using var gpu = new DirectGpuTensorEngine();
        Skip.IfNot(gpu.IsGpuAvailable, "No DirectGpu backend available");
        int p = 24;
        var oneHot = new Tensor<float>(new[] { 1, p });
        oneHot[0, 3] = 1f;
        var tableData = new float[p * 2];
        for (int i = 0; i < tableData.Length; i++) tableData[i] = i;
        var table = new Tensor<float>(new[] { p, 2 }, new Vector<float>(tableData));
        var xData = new float[64];
        for (int i = 0; i < xData.Length; i++) xData[i] = i;
        var x = new Tensor<float>(new[] { 1, 64 }, new Vector<float>(xData));

        for (int rep = 0; rep < 3; rep++)
        {
            var before = gpu.TensorMatMul(oneHot, table).ToArray();
            var first = gpu.TensorNarrow(oneHot, 1, 0, 1);                      // [1, 1] prefix view, value 0
            var scaled = gpu.TensorMultiply(x, gpu.TensorBroadcastTo(first, new[] { 1, 64 })).ToArray();
            var after = gpu.TensorMatMul(oneHot, table).ToArray();
            Assert.Equal(new[] { 6f, 7f }, before);
            Assert.Equal(new[] { 6f, 7f }, after);
            Assert.All(scaled, v => Assert.Equal(0f, v));
        }
    }

    [SkippableFact]
    public void PrefixNarrow_ReadsOnlyItsOwnElements()
    {
        using var gpu = new DirectGpuTensorEngine();
        Skip.IfNot(gpu.IsGpuAvailable, "No DirectGpu backend available");
        var data = new float[12];
        for (int i = 0; i < data.Length; i++) data[i] = i + 1;
        var source = new Tensor<float>(new[] { 3, 4 }, new Vector<float>(data));
        _ = gpu.TensorAddScalar(source, 0f).ToArray();                          // the source has a device copy
        var rows = gpu.TensorNarrow(source, 0, 0, 2);                           // first two rows: a prefix view
        var doubled = gpu.TensorMultiplyScalar(rows, 2f).ToArray();
        Assert.Equal(new[] { 2f, 4f, 6f, 8f, 10f, 12f, 14f, 16f }, doubled);
        var whole = gpu.TensorMultiplyScalar(source, 1f).ToArray();
        Assert.Equal(data, whole);
    }

    [SkippableFact]
    public void A_reshaped_or_cloned_prefix_view_does_not_poison_the_source_either()
    {
        using var gpu = new DirectGpuTensorEngine();
        Skip.IfNot(gpu.IsGpuAvailable, "No DirectGpu backend available");
        var data = new float[24];
        for (int i = 0; i < data.Length; i++) data[i] = i + 1;
        var source = new Tensor<float>(new[] { 4, 6 }, new Vector<float>(data));
        var ones = new Tensor<float>(new[] { 6, 1 }, new Vector<float>(new float[] { 1, 1, 1, 1, 1, 1 }));
        var rowSums = new float[] { 21, 57, 93, 129 };
        var firstRow = new[] { 1f, 2f, 3f, 4f, 5f, 6f, 1f, 2f, 3f, 4f, 5f, 6f };
        Assert.Equal(rowSums, gpu.TensorMatMul(source, ones).ToArray());
        // The first row (6 of 24 elements), reshaped: still offset 0, still a prefix of the shared storage. The
        // broadcast uploads it, as in the narrow case.
        var reshaped = gpu.Reshape(gpu.TensorNarrow(source, 0, 0, 1), new[] { 1, 6 });
        Assert.Equal(firstRow, gpu.TensorBroadcastTo(reshaped, new[] { 2, 6 }).ToArray());
        Assert.Equal(rowSums, gpu.TensorMatMul(source, ones).ToArray());
        var cloned = (Tensor<float>)gpu.TensorNarrow(source, 0, 0, 1).CloneShared();
        Assert.Equal(firstRow, gpu.TensorBroadcastTo(cloned, new[] { 2, 6 }).ToArray());
        Assert.Equal(rowSums, gpu.TensorMatMul(source, ones).ToArray());
    }
}
