using System;
using System.Linq;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// gridDim.y and gridDim.z are capped at 65535 by the hardware. These kernels put batch x channels (or batch x heads)
/// on one of those axes, so ResNet-18 bias-add and VGG-11 max-pool inference at batch 128 x 512 channels failed with
/// "cuLaunchKernel failed: Invalid value". The host now clamps the axis and each kernel grid-strides over the rest.
/// Every case here launches more than 65535 rows on the strided axis and checks every element against a host
/// reference, so a launch that silently stopped at the clamp would leave the tail unwritten and fail.
/// </summary>
public class CudaGridDimensionLimitTests
{
    // 129 x 512 = 66048 rows: one more batch than ResNet's batch-128 x 512-channel stage, past the 65535 cap.
    private const int Batch = 129;
    private const int Channels = 512;
    private const float Unwritten = 123.25f;

    private static CudaBackend OpenBackendOrSkip()
    {
        Skip.IfNot(CudaNativeBindings.IsAvailable, "CUDA driver is unavailable.");
        var backend = new CudaBackend();
        if (!backend.IsAvailable)
        {
            backend.Dispose();
            throw new InvalidOperationException("CUDA driver was detected, but the CUDA backend failed to initialize.");
        }
        return backend;
    }

    private static float[] Fill(int count, int salt) =>
        Enumerable.Range(0, count).Select(i => (float)((((i * 37 + salt * 101) % 97) - 48) / 64.0)).ToArray();

    private static void AssertEqual(float[] expected, float[] actual, string label, float tolerance = 0f)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            if (Math.Abs(expected[i] - actual[i]) > tolerance * Math.Max(1f, Math.Abs(expected[i])))
                Assert.Fail($"{label}: element {i} of {expected.Length} expected {expected[i]}, got {actual[i]}");
    }

    [SkippableFact]
    public void Conv2DBiasAdd_InPlace_CoversEveryBatchChannelRow()
    {
        using var backend = OpenBackendOrSkip();
        const int spatial = 4;
        var data = Fill(Batch * Channels * spatial, 1);
        var bias = Fill(Channels, 2);
        using var buf = backend.AllocateBuffer(data);
        using var biasBuf = backend.AllocateBuffer(bias);

        backend.Conv2DBiasAdd(buf, biasBuf, Batch, Channels, spatial);

        var expected = data.Select((v, i) => v + bias[i / spatial % Channels]).ToArray();
        AssertEqual(expected, backend.DownloadBuffer(buf), "conv2d_bias_add");
    }

    [SkippableFact]
    public void MaxPool2D_ForwardAndBackward_CoverEveryBatchChannelRow()
    {
        using var backend = OpenBackendOrSkip();
        const int h = 4, w = 4, oh = 2, ow = 2;
        int planes = Batch * Channels;
        var input = Fill(planes * h * w, 3);
        using var inBuf = backend.AllocateBuffer(input);
        using var outBuf = backend.AllocateBuffer(Enumerable.Repeat(Unwritten, planes * oh * ow).ToArray());
        using var idxBuf = backend.AllocateIntBuffer(planes * oh * ow);

        backend.MaxPool2D(inBuf, outBuf, idxBuf, Batch, Channels, h, w, oh, ow, 2, 2, 2, 2, 0, 0);

        var expected = new float[planes * oh * ow];
        var argmax = new int[expected.Length];
        for (int p = 0; p < planes; p++)
        for (int y = 0; y < oh; y++)
        for (int x = 0; x < ow; x++)
        {
            float best = float.NegativeInfinity;
            int at = -1;
            for (int ky = 0; ky < 2; ky++)
            for (int kx = 0; kx < 2; kx++)
            {
                int i = (p * h + y * 2 + ky) * w + x * 2 + kx;
                if (input[i] > best) { best = input[i]; at = i; }
            }
            expected[(p * oh + y) * ow + x] = best;
            argmax[(p * oh + y) * ow + x] = at;
        }
        AssertEqual(expected, backend.DownloadBuffer(outBuf), "maxpool2d forward");

        // Backward routes each output gradient to its argmax; every input plane must receive its share.
        var gradOut = Fill(expected.Length, 4);
        using var gradOutBuf = backend.AllocateBuffer(gradOut);
        using var gradInBuf = backend.AllocateBuffer(new float[input.Length]);
        backend.MaxPool2DBackward(gradOutBuf, idxBuf, gradInBuf, Batch, Channels, h, w, oh, ow, 2, 2, 2, 2, 0, 0);
        var expectedGrad = new float[input.Length];
        for (int o = 0; o < gradOut.Length; o++) expectedGrad[argmax[o]] += gradOut[o];
        AssertEqual(expectedGrad, backend.DownloadBuffer(gradInBuf), "maxpool2d backward");
    }

    [SkippableFact]
    public void AvgPool2D_ForwardAndBackward_CoverEveryBatchChannelRow()
    {
        using var backend = OpenBackendOrSkip();
        const int h = 4, w = 4, oh = 2, ow = 2;
        int planes = Batch * Channels;
        var input = Fill(planes * h * w, 5);
        using var inBuf = backend.AllocateBuffer(input);
        using var outBuf = backend.AllocateBuffer(Enumerable.Repeat(Unwritten, planes * oh * ow).ToArray());

        backend.AvgPool2D(inBuf, outBuf, Batch, Channels, h, w, oh, ow, 2, 2, 2, 2, 0, 0, countIncludePad: true);

        var expected = new float[planes * oh * ow];
        for (int p = 0; p < planes; p++)
        for (int y = 0; y < oh; y++)
        for (int x = 0; x < ow; x++)
        {
            float sum = 0f;
            for (int ky = 0; ky < 2; ky++)
            for (int kx = 0; kx < 2; kx++) sum += input[(p * h + y * 2 + ky) * w + x * 2 + kx];
            expected[(p * oh + y) * ow + x] = sum / 4f;
        }
        AssertEqual(expected, backend.DownloadBuffer(outBuf), "avgpool2d forward", 1e-6f);

        var gradOut = Fill(expected.Length, 6);
        using var gradOutBuf = backend.AllocateBuffer(gradOut);
        using var gradInBuf = backend.AllocateBuffer(Enumerable.Repeat(Unwritten, input.Length).ToArray());
        backend.AvgPool2DBackward(gradOutBuf, gradInBuf, Batch, Channels, h, w, oh, ow, 2, 2, 2, 2, 0, 0, countIncludePad: true);
        var expectedGrad = new float[input.Length];
        for (int p = 0; p < planes; p++)
        for (int iy = 0; iy < h; iy++)
        for (int ix = 0; ix < w; ix++)
            expectedGrad[(p * h + iy) * w + ix] = gradOut[(p * oh + iy / 2) * ow + ix / 2] / 4f;
        AssertEqual(expectedGrad, backend.DownloadBuffer(gradInBuf), "avgpool2d backward", 1e-6f);
    }

    [SkippableFact]
    public void DepthwiseConv2D_CoversEveryBatchChannelRow()
    {
        using var backend = OpenBackendOrSkip();
        const int h = 3, w = 3, k = 3, oh = 3, ow = 3;
        int planes = Batch * Channels;
        var input = Fill(planes * h * w, 7);
        var kernel = Fill(Channels * k * k, 8);
        using var inBuf = backend.AllocateBuffer(input);
        using var kBuf = backend.AllocateBuffer(kernel);
        using var outBuf = backend.AllocateBuffer(Enumerable.Repeat(Unwritten, planes * oh * ow).ToArray());

        // Off the declared-program route, so the hand-written depthwise_conv2d kernel serves the call.
        backend.DeclaredKernelsSwitch = false;
        backend.DepthwiseConv2D(inBuf, kBuf, outBuf, Batch, Channels, h, w, oh, ow, k, k, 1, 1, 1, 1);
        Assert.Equal(0, backend.DeclaredProgramLaunches);

        var expected = new float[planes * oh * ow];
        for (int p = 0; p < planes; p++)
        for (int y = 0; y < oh; y++)
        for (int x = 0; x < ow; x++)
        {
            int c = p % Channels;
            double sum = 0;
            for (int ky = 0; ky < k; ky++)
            for (int kx = 0; kx < k; kx++)
            {
                int iy = y - 1 + ky, ix = x - 1 + kx;
                if (iy < 0 || iy >= h || ix < 0 || ix >= w) continue;
                sum += input[(p * h + iy) * w + ix] * kernel[(c * k + ky) * k + kx];
            }
            expected[(p * oh + y) * ow + x] = (float)sum;
        }
        AssertEqual(expected, backend.DownloadBuffer(outBuf), "depthwise_conv2d", 1e-5f);
    }

    [SkippableFact]
    public void BatchedTranspose_CoversEveryBatch()
    {
        using var backend = OpenBackendOrSkip();
        const int batch = 70000, rows = 2, cols = 3;
        var a = Fill(batch * rows * cols, 9);
        using var aBuf = backend.AllocateBuffer(a);
        using var bBuf = backend.AllocateBuffer(Enumerable.Repeat(Unwritten, a.Length).ToArray());

        backend.BatchedTranspose(aBuf, bBuf, batch, rows, cols);

        var expected = new float[a.Length];
        for (int b = 0; b < batch; b++)
        for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++)
            expected[(b * cols + c) * rows + r] = a[(b * rows + r) * cols + c];
        AssertEqual(expected, backend.DownloadBuffer(bBuf), "batched_transpose");
    }

    [SkippableFact]
    public void FlashAttentionV2_CoversEveryBatchHeadRow()
    {
        using var backend = OpenBackendOrSkip();
        // 16385 x 4 = 65540 (batch, head) rows on grid Y; seq 2 and headDim 4 keep the buffers small.
        const int batch = 16385, heads = 4, seq = 2, dim = 4;
        int n = batch * heads * seq * dim;
        float scale = 1f / MathF.Sqrt(dim);
        var q = Fill(n, 10);
        var kk = Fill(n, 11);
        var v = Fill(n, 12);
        using var qBuf = backend.AllocateBuffer(q);
        using var kBuf = backend.AllocateBuffer(kk);
        using var vBuf = backend.AllocateBuffer(v);
        using var oBuf = backend.AllocateBuffer(Enumerable.Repeat(Unwritten, n).ToArray());
        using var statsBuf = backend.AllocateBuffer(batch * heads * seq);

        // Off the declared-program route, so the hand-written flash_attention_v2 kernel serves the call.
        backend.DeclaredKernelsSwitch = false;
        backend.FlashAttentionV2(qBuf, kBuf, vBuf, oBuf, statsBuf, batch, heads, seq, seq, dim, scale, isCausal: false);
        Assert.Equal(0, backend.DeclaredProgramLaunches);

        var expected = new float[n];
        for (int bh = 0; bh < batch * heads; bh++)
        for (int i = 0; i < seq; i++)
        {
            var logits = new double[seq];
            for (int j = 0; j < seq; j++)
            {
                double dot = 0;
                for (int d = 0; d < dim; d++) dot += q[(bh * seq + i) * dim + d] * kk[(bh * seq + j) * dim + d];
                logits[j] = dot * scale;
            }
            double max = logits.Max(), denom = logits.Sum(l => Math.Exp(l - max));
            for (int d = 0; d < dim; d++)
            {
                double acc = 0;
                for (int j = 0; j < seq; j++) acc += Math.Exp(logits[j] - max) / denom * v[(bh * seq + j) * dim + d];
                expected[(bh * seq + i) * dim + d] = (float)acc;
            }
        }
        AssertEqual(expected, backend.DownloadBuffer(oBuf), "flash_attention_v2", 1e-4f);
    }
}
