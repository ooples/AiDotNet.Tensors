using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.BlasManaged;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using BlasManagedLib = AiDotNet.Tensors.Engines.BlasManaged.BlasManaged;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

/// <summary>
/// Sub-E (#373) acceptance — verify the production-caller adoption hooks
/// (CpuEngine.TensorMatMul2DWithPrePackedB and
/// BackwardFunctions.MatMulBackwardGradAOnly_PrePackedBT) produce
/// numerically-correct output that matches the non-pre-packed reference path.
/// </summary>
[Collection("BlasManaged-Stats-Serial")]
public class PrePackAdoptionTest
{
    [Fact]
    public void CpuEngine_TensorMatMul2DWithPrePackedB_Matches_Reference_FP32()
    {
        const int M = 32, K = 128, N = 64;
        var rng = new Random(42);
        var a = new float[M * K];
        var b = new float[K * N];
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);

        // Naïve reference — no engine dependency.
        var refResult = new float[M * N];
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++)
            {
                float sum = 0;
                for (int kk = 0; kk < K; kk++) sum += a[i * K + kk] * b[kk * N + j];
                refResult[i * N + j] = sum;
            }

        var aT = new Tensor<float>(a, new[] { M, K });
        var bT = new Tensor<float>(b, new[] { K, N });
        var engine = new CpuEngine();
        var handle = BlasManagedLib.PrePackB<float>(b, N, false, K, N);
        try
        {
            var prePackResult = engine.TensorMatMul2DWithPrePackedB(aT, bT, handle);
            double maxDelta = 0;
            for (int i = 0; i < refResult.Length; i++)
                maxDelta = Math.Max(maxDelta, Math.Abs(refResult[i] - prePackResult.Data.Span[i]));
            Assert.True(maxDelta < 1e-3,
                $"Pre-pack adoption produced drift {maxDelta:G6} > 1e-3 vs naïve reference");
        }
        finally
        {
            handle.Dispose();
        }
    }

    [Fact]
    public void CpuEngine_TensorMatMul2DWithPrePackedB_Matches_Reference_FP64()
    {
        const int M = 32, K = 128, N = 64;
        var rng = new Random(7);
        var a = new double[M * K];
        var b = new double[K * N];
        for (int i = 0; i < a.Length; i++) a[i] = rng.NextDouble() * 2 - 1;
        for (int i = 0; i < b.Length; i++) b[i] = rng.NextDouble() * 2 - 1;

        var refResult = new double[M * N];
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++)
            {
                double sum = 0;
                for (int kk = 0; kk < K; kk++) sum += a[i * K + kk] * b[kk * N + j];
                refResult[i * N + j] = sum;
            }

        var aT = new Tensor<double>(a, new[] { M, K });
        var bT = new Tensor<double>(b, new[] { K, N });
        var engine = new CpuEngine();
        var handle = BlasManagedLib.PrePackB<double>(b, N, false, K, N);
        try
        {
            var prePackResult = engine.TensorMatMul2DWithPrePackedB(aT, bT, handle);
            double maxDelta = 0;
            for (int i = 0; i < refResult.Length; i++)
                maxDelta = Math.Max(maxDelta, Math.Abs(refResult[i] - prePackResult.Data.Span[i]));
            Assert.True(maxDelta < 1e-9,
                $"FP64 pre-pack adoption produced drift {maxDelta:G6} > 1e-9 vs naïve reference");
        }
        finally
        {
            handle.Dispose();
        }
    }

    [Fact]
    public void Direct_BlasManaged_Gemm_NO_PackedB_Matches_Reference_FP32()
    {
        const int M = 32, K = 128, N = 64;
        var rng = new Random(42);
        var a = new float[M * K];
        var b = new float[K * N];
        var c = new float[M * N];
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);

        var refResult = new float[M * N];
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++)
            {
                float sum = 0;
                for (int kk = 0; kk < K; kk++) sum += a[i * K + kk] * b[kk * N + j];
                refResult[i * N + j] = sum;
            }

        BlasManagedLib.Gemm<float>(a, K, false, b, N, false, c, N, M, N, K);
        double maxDelta = 0;
        for (int i = 0; i < refResult.Length; i++)
            maxDelta = Math.Max(maxDelta, Math.Abs(refResult[i] - c[i]));
        Assert.True(maxDelta < 1e-3,
            $"BlasManaged.Gemm WITHOUT pre-pack drift {maxDelta:G6} > 1e-3");
    }

    [Fact]
    public void Direct_BlasManaged_Gemm_With_PackedB_Matches_Reference_FP32()
    {
        // Diagnostic: bypass CpuEngine wrapper. Call BlasManaged.Gemm directly
        // with the same shape (M=32, K=128, N=64) as the failing adoption test.
        const int M = 32, K = 128, N = 64;
        var rng = new Random(42);
        var a = new float[M * K];
        var b = new float[K * N];
        var c = new float[M * N];
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);

        var refResult = new float[M * N];
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++)
            {
                float sum = 0;
                for (int kk = 0; kk < K; kk++) sum += a[i * K + kk] * b[kk * N + j];
                refResult[i * N + j] = sum;
            }

        var handle = BlasManagedLib.PrePackB<float>(b, N, false, K, N);
        try
        {
            var opts = new BlasOptions<float> { PackedB = handle };
            BlasManagedLib.Gemm<float>(a, K, false, b, N, false, c, N, M, N, K, opts);
            double maxDelta = 0;
            for (int i = 0; i < refResult.Length; i++)
                maxDelta = Math.Max(maxDelta, Math.Abs(refResult[i] - c[i]));
            Assert.True(maxDelta < 1e-3,
                $"Direct BlasManaged.Gemm with PackedB drift {maxDelta:G6} > 1e-3");
        }
        finally
        {
            handle.Dispose();
        }
    }

    [Fact]
    public void BackwardFunctions_MatMulBackwardGradAOnly_PrePackedBT_Matches_Reference()
    {
        const int M = 32, K = 128, N = 64;
        var rng = new Random(13);
        var dC = new float[M * N];
        var a = new float[M * K];
        var bT = new float[N * K];  // pre-transposed: row-major [N, K]
        for (int i = 0; i < dC.Length; i++) dC[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < bT.Length; i++) bT[i] = (float)(rng.NextDouble() * 2 - 1);

        var dCT = new Tensor<float>(dC, new[] { M, N });
        var aT = new Tensor<float>(a, new[] { M, K });
        var bTT = new Tensor<float>(bT, new[] { N, K });

        // Reference: gradA[M,K] = dC[M,N] · Bᵀ[N,K] computed naïvely.
        var refGradA = new float[M * K];
        for (int i = 0; i < M; i++)
            for (int kk = 0; kk < K; kk++)
            {
                float sum = 0;
                for (int j = 0; j < N; j++) sum += dC[i * N + j] * bT[j * K + kk];
                refGradA[i * K + kk] = sum;
            }

        // Pre-pack Bᵀ (the [N, K] matrix as "B" of the backward GEMM: k=N, n=K).
        var handle = BlasManagedLib.PrePackB<float>(bT, K, false, N, K);
        try
        {
            var adoptedGradA = BackwardFunctions<float>.MatMulBackwardGradAOnly_PrePackedBT(
                dCT, aT, bTT, handle);
            double maxDelta = 0;
            for (int i = 0; i < refGradA.Length; i++)
                maxDelta = Math.Max(maxDelta, Math.Abs(refGradA[i] - adoptedGradA.Data.Span[i]));
            Assert.True(maxDelta < 1e-3,
                $"Backward pre-pack adoption produced drift {maxDelta:G6} > 1e-3 vs naïve reference");
        }
        finally
        {
            handle.Dispose();
        }
    }
}
