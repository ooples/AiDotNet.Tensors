#if NET6_0_OR_GREATER
using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.HIP;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;
using AiDotNet.Tensors.Engines.DirectGpu.Vulkan;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// The #1154 backend contracts on every non-CUDA backend, against CPU references of the CUDA kernels' semantics:
/// the FP16 activation operations (FP32 compute, IEEE half storage), the fused index kernels (round-half-to-even
/// indices, NaN and out-of-range reading nothing), the fused residual BatchNorm + ReLU, the device-scalar ops, and
/// the in-place upload, zero fill and byte copy the engine now reaches through contracts.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class BackendContractParityTests
{
    private static float[] Random(int count, int seed, float scale = 2f)
    {
        var rng = RandomHelper.CreateSeededRandom(seed);
        var values = new float[count];
        for (int i = 0; i < count; i++) values[i] = (float)((rng.NextDouble() * 2 - 1) * scale);
        return values;
    }

    private static float H(float value) => (float)(Half)value;

    private static IGpuBuffer UploadHalf(IDirectGpuBackend backend, float[] values)
    {
        var bytes = new byte[values.Length * 2];
        for (int i = 0; i < values.Length; i++) BitConverter.TryWriteBytes(bytes.AsSpan(2 * i), BitConverter.HalfToUInt16Bits((Half)values[i]));
        var buffer = backend.AllocateByteBuffer(Math.Max(4, bytes.Length + (bytes.Length & 2)));
        backend.UploadByteBuffer(buffer, bytes);
        return buffer;
    }

    private static float[] DownloadHalf(IDirectGpuBackend backend, IGpuBuffer buffer, int count)
    {
        byte[] bytes = backend.DownloadByteBuffer(buffer, count * 2);
        var values = new float[count];
        for (int i = 0; i < count; i++) values[i] = (float)BitConverter.UInt16BitsToHalf(BitConverter.ToUInt16(bytes, 2 * i));
        return values;
    }

    private static void Close(double expected, float actual, double tolerance, string what)
        => Assert.True(Math.Abs(expected - actual) <= tolerance * (1 + Math.Abs(expected)), $"{what}: expected {expected}, got {actual}");

    private static double Gelu(double x) => 0.5 * x * (1 + Math.Tanh(Math.Sqrt(2 / Math.PI) * (x + 0.044715 * x * x * x)));

    private static void CheckFp16(IDirectGpuBackend backend)
    {
        var fp16 = Assert.IsAssignableFrom<IFp16ActivationBackend>(backend);
        const int n = 37;
        float[] a = Array.ConvertAll(Random(n, 1), H), b = Array.ConvertAll(Random(n, 2), H);
        using var ah = UploadHalf(backend, a);
        using var bh = UploadHalf(backend, b);

        using (var o = UploadHalf(backend, new float[n]))
        {
            fp16.Fp16Add(ah, bh, o, n);
            var got = DownloadHalf(backend, o, n);
            for (int i = 0; i < n; i++) Assert.Equal(H(a[i] + b[i]), got[i]);
            fp16.Fp16Relu(ah, o, n);
            got = DownloadHalf(backend, o, n);
            for (int i = 0; i < n; i++) Assert.Equal(H(Math.Max(a[i], 0f)), got[i]);
            fp16.Fp16Gelu(ah, o, n);
            got = DownloadHalf(backend, o, n);
            for (int i = 0; i < n; i++) Close(Gelu(a[i]), got[i], 2e-3, $"gelu[{i}]");
            fp16.Fp16ReluBackward(bh, ah, o, n);
            got = DownloadHalf(backend, o, n);
            for (int i = 0; i < n; i++) Assert.Equal(a[i] > 0 ? b[i] : 0f, got[i]);
        }

        // io[b, c, s] += bias[b, c]
        const int batch = 2, channels = 3, spatial = 5;
        float[] io = Array.ConvertAll(Random(batch * channels * spatial, 3), H), bias = Array.ConvertAll(Random(batch * channels, 4), H);
        using (var ioh = UploadHalf(backend, io))
        using (var biash = UploadHalf(backend, bias))
        {
            fp16.Fp16BroadcastAddChannel(ioh, biash, batch, channels, spatial);
            var got = DownloadHalf(backend, ioh, io.Length);
            for (int i = 0; i < io.Length; i++) Assert.Equal(H(io[i] + bias[i / spatial]), got[i]);
        }

        // C = A[M,K] B[K,N], FP16 in and out
        const int m = 3, nn = 4, k = 5;
        float[] ga = Array.ConvertAll(Random(m * k, 5), H), gb = Array.ConvertAll(Random(k * nn, 6), H);
        using (var gah = UploadHalf(backend, ga))
        using (var gbh = UploadHalf(backend, gb))
        using (var gch = UploadHalf(backend, new float[m * nn]))
        {
            fp16.GemmFp16HalfOut(gah, gbh, gch, m, nn, k);
            var got = DownloadHalf(backend, gch, m * nn);
            for (int r = 0; r < m; r++)
                for (int c = 0; c < nn; c++)
                {
                    double s = 0;
                    for (int t = 0; t < k; t++) s += (double)ga[r * k + t] * gb[t * nn + c];
                    Close(s, got[r * nn + c], 2e-3, $"gemm[{r},{c}]");
                }
        }

        // im2col [K, N] of a 1x2x4x4 input, 3x3 kernel, pad 1
        const int ch = 2, hgt = 4, wid = 4;
        float[] img = Array.ConvertAll(Random(ch * hgt * wid, 7), H);
        int kRows = ch * 9, cols = hgt * wid;
        using (var imgh = UploadHalf(backend, img))
        using (var colh = UploadHalf(backend, new float[kRows * cols]))
        {
            fp16.UnfoldKNFp16FromFp16(imgh, colh, 1, ch, hgt, wid, 3, 3, 1, 1, 1, 1, 1, 1);
            var got = DownloadHalf(backend, colh, kRows * cols);
            for (int row = 0; row < kRows; row++)
                for (int col = 0; col < cols; col++)
                {
                    int kw = row % 3, kh = (row / 3) % 3, c = row / 9, oh = col / wid, ow = col % wid;
                    int ih = oh - 1 + kh, iw = ow - 1 + kw;
                    float expected = ih < 0 || ih >= hgt || iw < 0 || iw >= wid ? 0f : img[(c * hgt + ih) * wid + iw];
                    Assert.Equal(expected, got[row * cols + col]);
                }
        }

        // An odd count of halves crosses the host so nothing past the range is written.
        using (var dst = UploadHalf(backend, new float[] { 9f, 9f, 9f, 9f }))
        {
            fp16.CopyBufferDtoD(ah, dst, 3 * 2);
            var got = DownloadHalf(backend, dst, 4);
            Assert.Equal(new[] { a[0], a[1], a[2], 9f }, got);
        }
    }

    private static void CheckFusedIndex(IDirectGpuBackend backend)
    {
        var fused = Assert.IsAssignableFrom<IFusedIndexKernels>(backend);
        const int rows = 6, classes = 4;
        float[] values = Random(rows * classes, 8);
        // round half to even (2.5 -> 2, 1.5 -> 2), NaN and out of range read nothing
        float[] cls = { 2.5f, 1.5f, float.NaN, -1f, 4f, 0.4f };
        int[] expectedClass = { 2, 2, -1, -1, -1, 0 };
        float[] grad = Random(rows, 9);
        using var v = backend.AllocateBuffer(values);
        using var c = backend.AllocateBuffer(cls);
        using var g = backend.AllocateBuffer(grad);
        using var outRows = backend.AllocateBuffer(new float[rows]);
        fused.GatherClassValues(v, c, outRows, rows, classes);
        var gathered = backend.DownloadBuffer(outRows);
        for (int r = 0; r < rows; r++)
            Assert.Equal(expectedClass[r] < 0 ? 0f : values[r * classes + expectedClass[r]], gathered[r]);
        using var outGrid = backend.AllocateBuffer(Random(rows * classes, 10));
        fused.ScatterClassGrad(g, c, outGrid, rows, classes);
        var scattered = backend.DownloadBuffer(outGrid);
        for (int r = 0; r < rows; r++)
            for (int k = 0; k < classes; k++)
                Assert.Equal(k == expectedClass[r] ? grad[r] : 0f, scattered[r * classes + k]);

        const int vocab = 3, dim = 2;
        float[] table = Random(vocab * dim, 11), ids = { 2f, 0.5f, float.NaN, 2.4f, 7f };
        int[] row = { 2, 0, -1, 2, -1 };
        float[] dOut = Random(ids.Length * dim, 12);
        using var t = backend.AllocateBuffer(table);
        using var idb = backend.AllocateBuffer(ids);
        using var emb = backend.AllocateBuffer(new float[ids.Length * dim]);
        fused.EmbeddingFromFloatIds(idb, t, emb, ids.Length, dim, vocab);
        var looked = backend.DownloadBuffer(emb);
        for (int i = 0; i < ids.Length; i++)
            for (int e = 0; e < dim; e++)
                Assert.Equal(row[i] < 0 ? 0f : table[row[i] * dim + e], looked[i * dim + e]);
        var expectedGrad = new double[vocab * dim];
        for (int i = 0; i < ids.Length; i++)
            if (row[i] >= 0)
                for (int e = 0; e < dim; e++) expectedGrad[row[i] * dim + e] += dOut[i * dim + e];
        using var dO = backend.AllocateBuffer(dOut);
        using var gt = backend.AllocateBuffer(new float[vocab * dim]);
        fused.EmbeddingBackwardFromFloatIds(dO, idb, gt, ids.Length, dim, vocab);
        var gotGrad = backend.DownloadBuffer(gt);
        for (int i = 0; i < vocab * dim; i++) Close(expectedGrad[i], gotGrad[i], 1e-6, $"embedding grad[{i}]");
    }

    private static void CheckDeviceOps(IDirectGpuBackend backend)
    {
        // relu(batchnorm_inference(x + residual)) from running statistics
        var fusedBn = Assert.IsAssignableFrom<IFusedResidualBatchNormRelu>(backend);
        const int batch = 2, channels = 3, spatial = 4, count = batch * channels * spatial;
        float[] x = Random(count, 13), res = Random(count, 14), gamma = Random(channels, 15), beta = Random(channels, 16);
        float[] mean = Random(channels, 17, 0.5f), variance = Array.ConvertAll(Random(channels, 18), v => Math.Abs(v) + 0.5f);
        using var xb = backend.AllocateBuffer(x);
        using var rb = backend.AllocateBuffer(res);
        using var ob = backend.AllocateBuffer(new float[count]);
        using var gb = backend.AllocateBuffer(gamma);
        using var bb = backend.AllocateBuffer(beta);
        using var mb = backend.AllocateBuffer(mean);
        using var vb = backend.AllocateBuffer(variance);
        Assert.True(fusedBn.TryResidualBatchNormRelu(xb, rb, ob, gb, bb, mb, vb, batch, channels, spatial, 1e-5f));
        var got = backend.DownloadBuffer(ob);
        for (int i = 0; i < count; i++)
        {
            int c = (i / spatial) % channels;
            double y = gamma[c] * (x[i] + res[i] - mean[c]) / Math.Sqrt(variance[c] + 1e-5) + beta[c];
            Close(Math.Max(y, 0), got[i], 1e-4, $"residual bn relu[{i}]");
        }

        var scalarOps = Assert.IsAssignableFrom<IDeviceScalarOps>(backend);
        float[] data = Random(9, 19);
        using var db = backend.AllocateBuffer(data);
        using var sb = backend.AllocateBuffer(new[] { -1.5f });
        scalarOps.ScaleByDeviceScalar(db, sb, data.Length);
        var scaled = backend.DownloadBuffer(db);
        for (int i = 0; i < data.Length; i++) Close(data[i] * -1.5, scaled[i], 1e-6, $"scale[{i}]");
        float[] second = Random(5, 20);
        using var b2 = backend.AllocateBuffer(second);
        using var sum = backend.AllocateBuffer(new float[2]);
        scalarOps.MultiTensorSumOfSquares(new List<IGpuBuffer> { db, b2 }, new List<int> { data.Length, 3 }, sum);
        var words = backend.DownloadBuffer(sum);
        double expected = 0;
        foreach (var value in scaled) expected += (double)value * value;
        for (int i = 0; i < 3; i++) expected += (double)second[i] * second[i];
        var bytes = new byte[8];
        BitConverter.TryWriteBytes(bytes.AsSpan(0), words[0]);
        BitConverter.TryWriteBytes(bytes.AsSpan(4), words[1]);
        Close(expected, (float)BitConverter.ToDouble(bytes, 0), 1e-6, "multi-tensor sum of squares");

        var zero = Assert.IsAssignableFrom<IDeviceZeroFill>(backend);
        using var z = backend.AllocateBuffer(Random(8, 21));
        zero.ZeroBytes(z, 6 * sizeof(float));
        var zeroed = backend.DownloadBuffer(z);
        for (int i = 0; i < 6; i++) Assert.Equal(0f, zeroed[i]);
        Assert.NotEqual(0f, zeroed[7]);

        using var target = backend.AllocateBuffer(new float[6]);
        float[] fresh = Random(4, 22);
        backend.UploadBufferInPlace(fresh, target);
        var uploaded = backend.DownloadBuffer(target);
        for (int i = 0; i < fresh.Length; i++) Assert.Equal(fresh[i], uploaded[i]);
    }

    private static void Check(IDirectGpuBackend backend)
    {
        CheckFp16(backend);
        CheckFusedIndex(backend);
        CheckDeviceOps(backend);
    }

    [SkippableFact]
    public void OpenCl_BackendContracts_MatchTheCudaSemantics()
    {
        using var backend = new OpenClBackend();
        Skip.IfNot(backend.IsAvailable, "OpenCL is not available.");
        Check(backend);
    }

    [SkippableFact]
    public void Vulkan_BackendContracts_MatchTheCudaSemantics()
    {
        var backend = VulkanBackend.Instance;
        Skip.IfNot(backend.Initialize() && backend.IsGlslCompilerAvailable, "Vulkan with a GLSL compiler is not available.");
        Check(backend);
    }

    [SkippableFact]
    public void Hip_BackendContracts_MatchTheCudaSemantics()
    {
        using var backend = new HipBackend();
        Skip.IfNot(backend.IsAvailable, "HIP is not available.");
        Check(backend);
    }
}
#endif
