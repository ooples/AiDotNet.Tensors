using System;
using System.Linq;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// CUDA BiasAdd (C = A + bias per row) runs over the flat [rows * cols] range: a float4 per thread when cols is a
/// multiple of four, one float otherwise, grid-striding when the range needs more blocks than one launch has. It used
/// one block per row, which left most of each block idle for a linear layer's 64-128 columns.
/// </summary>
public class CudaBiasAddLaunchTests
{
    [SkippableTheory]
    [InlineData(2048, 64)]      // float4: a Transformer linear layer's output
    [InlineData(37, 6)]         // one float per thread: cols not a multiple of four
    [InlineData(3, 1)]
    [InlineData(70000, 260)]    // float4, a large range (its stride loop would need over 67M floats to engage)
    [InlineData(70001, 263)]    // one float per thread, beyond 65535 blocks of 256: the grid-stride loop
    public void BiasAdd_AddsTheBiasToEveryElement(int rows, int cols)
    {
        using var backend = new CudaBackend();
        Skip.IfNot(backend.IsAvailable, "CUDA is not available.");
        int n = rows * cols;
        float[] a = Enumerable.Range(0, n).Select(i => 0.5f * (i % 13) - 3f).ToArray();
        float[] bias = Enumerable.Range(0, cols).Select(j => 0.125f * (j % 11) - 0.5f).ToArray();
        using var aBuf = backend.AllocateBuffer(a);
        using var biasBuf = backend.AllocateBuffer(bias);
        using var cBuf = backend.AllocateBuffer(Enumerable.Repeat(123.25f, n).ToArray());   // unwritten shows as 123.25

        backend.BiasAdd(aBuf, biasBuf, cBuf, rows, cols);

        var got = backend.DownloadBuffer(cBuf);
        for (int i = 0; i < n; i++)
            if (got[i] != a[i] + bias[i % cols])
                Assert.Fail($"C[{i / cols},{i % cols}] = {got[i]}, expected {a[i] + bias[i % cols]}");
    }

    /// <summary>The fused bias + ReLU pass must equal BiasAdd followed by Relu bit for bit -- negative sums, NaN
    /// propagation (relu_nan_safe) and both the float4 and the scalar layouts.</summary>
    [SkippableTheory]
    [InlineData(2048, 256)]
    [InlineData(37, 6)]
    public void BiasAddRelu_EqualsBiasAddThenRelu(int rows, int cols)
    {
        using var backend = new CudaBackend();
        Skip.IfNot(backend.IsAvailable, "CUDA is not available.");
        int n = rows * cols;
        float[] a = Enumerable.Range(0, n).Select(i => i % 101 == 0 ? float.NaN : 0.37f * (i % 23) - 4f).ToArray();
        float[] bias = Enumerable.Range(0, cols).Select(j => 0.25f * (j % 9) - 1f).ToArray();
        using var aBuf = backend.AllocateBuffer(a);
        using var biasBuf = backend.AllocateBuffer(bias);
        using var fused = backend.AllocateBuffer(new float[n]);
        using var separate = backend.AllocateBuffer(new float[n]);

        Assert.True(((AiDotNet.Tensors.Engines.DirectGpu.IBiasActivationBackend)backend)
            .TryBiasAddActivation(aBuf, biasBuf, fused, rows, cols, AiDotNet.Tensors.Engines.FusedActivationType.ReLU));
        backend.BiasAdd(aBuf, biasBuf, separate, rows, cols);
        backend.Relu(separate, separate, n);

        var got = backend.DownloadBuffer(fused);
        var expected = backend.DownloadBuffer(separate);
        for (int i = 0; i < n; i++)
            if (TestHelpers.MathCompat.SingleToInt32Bits(got[i]) != TestHelpers.MathCompat.SingleToInt32Bits(expected[i]))
                Assert.Fail($"element {i}: fused {got[i]} != separate {expected[i]}");
    }

    [SkippableFact]
    public void BiasAdd_InPlace_AddsTheBiasOnce()
    {
        using var backend = new CudaBackend();
        Skip.IfNot(backend.IsAvailable, "CUDA is not available.");
        const int rows = 5, cols = 8;
        float[] a = Enumerable.Range(0, rows * cols).Select(i => (float)i).ToArray();
        float[] bias = Enumerable.Range(0, cols).Select(j => 100f * j).ToArray();
        using var buf = backend.AllocateBuffer(a);
        using var biasBuf = backend.AllocateBuffer(bias);

        backend.BiasAdd(buf, biasBuf, buf, rows, cols);   // the resident FusedLinear adds its bias in place

        var got = backend.DownloadBuffer(buf);
        for (int i = 0; i < a.Length; i++) Assert.Equal(a[i] + bias[i % cols], got[i]);
    }
}
