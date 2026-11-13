using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// Half sums, matmuls, cumulative sums, norms, dot products and softmax denominators on
/// <see cref="DirectGpuTensorEngine"/> must accumulate in float32 like the CPU engine (PyTorch 2.x: float32 accumulation
/// for half reductions and cuBLAS <c>CUBLAS_COMPUTE_32F</c> for half matmul). A Half accumulator stops growing at 2048
/// (2048 + 1 rounds to 2048), so 4096 ones must sum to exactly 4096 and a matmul with an inner dimension of 4096 of
/// ones must give 4096. Every result is also compared bit for bit with the CPU engine: the GPU either reduces the
/// widened float32 buffers on the device and rounds once, or falls back to the CPU engine's float32 kernels.
/// <para>The FP16-storage GEMM (<see cref="IGpuHalfPrecisionBackend.Hgemm"/>: FP16 operands and output) is checked on
/// the backend directly; cublasHgemm and hipblasHgemm compute in FP16 and returned 2048.</para>
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class LowPrecisionAccumulationGpuTests
{
    private const int K = 4096;

    private static DirectGpuTensorEngine? CreateGpu()
    {
        DirectGpuTensorEngine gpu;
        try { gpu = new DirectGpuTensorEngine(); }
        catch (InvalidOperationException) { return null; }
        catch (NotSupportedException) { return null; }
        if (!gpu.IsGpuAvailable) { gpu.Dispose(); return null; }
        return gpu;
    }

    private static Tensor<Half> Fill(double value, params int[] shape)
    {
        int n = 1;
        foreach (int d in shape) n *= d;
        var data = new Half[n];
        for (int i = 0; i < n; i++) data[i] = (Half)value;
        return new Tensor<Half>(data, shape);
    }

    private static void AssertAll(double expected, Half[] actual, string what)
    {
        Assert.NotEmpty(actual);
        for (int i = 0; i < actual.Length; i++)
            Assert.True((double)actual[i] == expected, $"{what}[{i}]: GPU engine gave {(double)actual[i]}, expected {expected}");
    }

    private static void AssertBitEqual(Half[] cpu, Half[] gpu, string what)
    {
        Assert.Equal(cpu.Length, gpu.Length);
        for (int i = 0; i < cpu.Length; i++)
            Assert.True(cpu[i].Equals(gpu[i]), $"{what}[{i}]: GPU engine {(double)gpu[i]} differs from CPU engine {(double)cpu[i]}");
    }

    [SkippableFact]
    public void Sums_Half_4096Ones_Are4096_AndMatchCpu()
    {
        using var gpu = CreateGpu();
        Skip.If(gpu is null, "No GPU available");
        var cpu = new CpuEngine();
        IEngine g = gpu;

        Assert.Equal(4096.0, (double)g.TensorSum(Fill(1, K)));
        Assert.Equal(4096.0, (double)g.TensorSumOfSquares(Fill(1, K)));
        var rows = Fill(1, 3, K);
        AssertAll(K, g.ReduceSum(rows, new[] { 1 }, false).ToArray(), "ReduceSum last axis");
        AssertBitEqual(cpu.ReduceSum(rows, new[] { 1 }, false).ToArray(), g.ReduceSum(rows, new[] { 1 }, false).ToArray(), "ReduceSum");
        var cols = Fill(1, K, 3);
        AssertAll(K, g.ReduceSum(cols, new[] { 0 }, false).ToArray(), "ReduceSum leading axis");
        Assert.Equal(4096.0, (double)g.TensorVecDot(Fill(1, K), Fill(1, K)));
        AssertAll(64, g.TensorNorm(rows, 1, false).ToArray(), "TensorNorm");
    }

    [SkippableFact]
    public void Means_Half_70000Ones_AreExactlyOne()
    {
        using var gpu = CreateGpu();
        Skip.If(gpu is null, "No GPU available");
        IEngine g = gpu;
        var ones = Fill(1, 70_000);
        Assert.Equal(1.0, (double)g.TensorMean(ones));
        AssertAll(1, g.ReduceMean(ones, new[] { 0 }, false).ToArray(), "ReduceMean");
    }

    [SkippableFact]
    public void CumSum_Half_4096Ones_CountsPast2048_AndMatchesCpu()
    {
        using var gpu = CreateGpu();
        Skip.If(gpu is null, "No GPU available");
        var cpu = new CpuEngine();
        var x = Fill(1, 2, K);
        var r = ((IEngine)gpu).TensorCumSum(x, 1).ToArray();
        Assert.Equal(4096.0, (double)r[K - 1]);
        Assert.Equal(4096.0, (double)r[2 * K - 1]);
        AssertBitEqual(cpu.TensorCumSum(x, 1).ToArray(), r, "TensorCumSum");
    }

    [SkippableFact]
    public void MatMul_Half_InnerDim4096OfOnes_Is4096_AndMatchesCpu()
    {
        using var gpu = CreateGpu();
        Skip.If(gpu is null, "No GPU available");
        var cpu = new CpuEngine();
        IEngine g = gpu;
        var a = Fill(1, 2, K);
        var b = Fill(1, K, 3);
        AssertAll(K, g.TensorMatMul(a, b).ToArray(), "TensorMatMul 2D");
        AssertAll(K, g.TensorMatMul(Fill(1, 2, 2, K), b).ToArray(), "TensorMatMul ND x 2D");
        AssertAll(K, g.BatchMatMul(Fill(1, 2, 2, K), Fill(1, 2, K, 3)).ToArray(), "BatchMatMul");
        AssertAll(K, g.TensorMatMulTransposed(a, Fill(1, 3, K)).ToArray(), "TensorMatMulTransposed");
        AssertBitEqual(cpu.TensorMatMul(a, b).ToArray(), g.TensorMatMul(a, b).ToArray(), "TensorMatMul");
    }

    [SkippableFact]
    public void SoftmaxFamily_Half_4096EqualLogits_MatchCpu()
    {
        using var gpu = CreateGpu();
        Skip.If(gpu is null, "No GPU available");
        var cpu = new CpuEngine();
        IEngine g = gpu;
        var x = Fill(0, 2, K);
        AssertAll(1.0 / K, g.Softmax(x, -1).ToArray(), "Softmax");
        AssertAll(1.0 / K, g.TensorSoftmax(x, 1).ToArray(), "TensorSoftmax");
        AssertAll((double)(Half)(-Math.Log(K)), g.TensorLogSoftmax(x, 1).ToArray(), "TensorLogSoftmax");
        AssertAll((double)(Half)Math.Log(K), g.TensorLogSumExp(x, 1, false).ToArray(), "TensorLogSumExp");
        AssertBitEqual(cpu.TensorLogSoftmax(x, 1).ToArray(), g.TensorLogSoftmax(x, 1).ToArray(), "TensorLogSoftmax");
        AssertBitEqual(cpu.TensorLogSumExp(x, 1, false).ToArray(), g.TensorLogSumExp(x, 1, false).ToArray(), "TensorLogSumExp");
    }

    [SkippableFact]
    public void BackendHgemm_Fp16Storage_InnerDim4096OfOnes_Is4096()
    {
        using var gpu = CreateGpu();
        Skip.If(gpu is null, "No GPU available");
        var maybeBackend = gpu.GetBackend();
        Skip.If(maybeBackend is null, "No GPU backend");
        IDirectGpuBackend backend = maybeBackend;
        var half = backend as IGpuHalfPrecisionBackend;
        Skip.If(half is null || !half.SupportsHgemm, "Backend has no FP16 GEMM");

        const int m = 2, n = 3;
        IGpuBuffer HalfOnes(int count)
        {
            var ones = new float[count];
            for (int i = 0; i < count; i++) ones[i] = 1f;
            using var f = backend.AllocateBuffer(ones);
            var h = backend.AllocateByteBuffer(count * 2);
            backend.ConvertToFp16(f, h, count);
            return h;
        }

        using var aH = HalfOnes(m * K);
        using var bH = HalfOnes(K * n);
        using var cH = backend.AllocateByteBuffer(m * n * 2);
        half.Hgemm(aH, bH, cH, m, n, K);
        using var cF = backend.AllocateBuffer(m * n);
        backend.ConvertToFp32(cH, cF, m * n);
        var c = backend.DownloadBuffer(cF);
        for (int i = 0; i < m * n; i++) Assert.True(c[i] == K, $"Hgemm[{i}]: got {c[i]}, expected {K}");

        using var cF32 = backend.AllocateBuffer(m * n);
        half.GemmFp16In32fOut(aH, bH, cF32, m, n, K);
        var c32 = backend.DownloadBuffer(cF32);
        for (int i = 0; i < m * n; i++) Assert.True(c32[i] == K, $"GemmFp16In32fOut[{i}]: got {c32[i]}, expected {K}");
    }
}
