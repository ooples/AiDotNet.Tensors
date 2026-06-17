using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Gpu;

/// <summary>
/// #558 layer 7 — validates the FP16 Tensor-Core GEMM (<see cref="CudaBackend.GemmFp16"/>) on real hardware.
/// The hot single-matmul path uses TF32 <c>cublasSgemm</c>; <c>GemmFp16</c> routes the same op through
/// <c>cublasGemmEx</c> with FP16 inputs + FP32 accumulate (full-rate Tensor Cores, half input bandwidth).
/// This proves the FP16 result matches a CPU FP32 reference within FP16 rounding — i.e. the Tensor-Core
/// path is correct, not just fast. Inputs are produced by the driver-only <c>ConvertToFp16Native</c> kernel,
/// so the whole chain works without the CUDA Toolkit (no cuda_fp16.h).
/// </summary>
public class Fp16TensorCoreGemmTests
{
    [SkippableFact]
    public void GemmFp16_matches_fp32_reference_within_fp16_tolerance()
    {
        Skip.IfNot(CudaNativeBindings.IsAvailable, "CUDA driver not available");
        var eng = AiDotNetEngine.Current;
        Skip.IfNot(eng is DirectGpuTensorEngine, "active engine is not the CUDA backend");
        // Warm up: a real GPU op forces the engine's kernel modules (incl. the FP16 module) to compile.
        var warm = new Tensor<float>(new float[] { 1, 2, 3, 4 }, new[] { 2, 2 });
        _ = eng.TensorMatMul(warm, warm);
        var b = ((DirectGpuTensorEngine)eng).TestBackend as CudaBackend;
        Skip.IfNot(b is not null && b.IsAvailable, "CudaBackend not available");

        const int M = 64, K = 128, N = 48;
        var rng = new Random(11);
        var A = new float[M * K];
        var B = new float[K * N];
        for (int i = 0; i < A.Length; i++) A[i] = (float)(rng.NextDouble() * 2 - 1); // [-1,1] — safe FP16 range
        for (int i = 0; i < B.Length; i++) B[i] = (float)(rng.NextDouble() * 2 - 1);

        // Authoritative CPU FP32 reference: row-major C[M,N] = A[M,K] · B[K,N].
        var cpu = new float[M * N];
        for (int m = 0; m < M; m++)
            for (int n = 0; n < N; n++)
            {
                double acc = 0;
                for (int k = 0; k < K; k++) acc += (double)A[m * K + k] * B[k * N + n];
                cpu[m * N + n] = (float)acc;
            }

        using var dA = b!.AllocateBuffer(A);
        using var dB = b.AllocateBuffer(B);

        // FP32 path (TF32 cublasSgemm) — the current hot path, as a sanity anchor.
        using var dC32 = b.AllocateBuffer(M * N);
        b.Gemm(dA, dB, dC32, M, N, K);
        var got32 = b.DownloadBuffer(dC32);

        // FP16 Tensor-Core path: convert A,B → half (driver-only native kernel), then GemmFp16.
        using var hA = b.AllocateByteBuffer(M * K * 2);
        using var hB = b.AllocateByteBuffer(K * N * 2);
        b.ConvertToFp16Native(dA, hA, M * K);
        b.ConvertToFp16Native(dB, hB, K * N);
        Assert.Equal((long)M * K * 2, hA.SizeInBytes); // genuinely half
        Assert.Equal((long)K * N * 2, hB.SizeInBytes);

        using var dC16 = b.AllocateBuffer(M * N); // FP32 accumulate output
        // Not every CUDA device/driver supports the FP16-in / FP32-compute cublasGemmEx
        // config (it returns CUBLAS_STATUS_NOT_SUPPORTED, surfaced as NotSupportedException).
        // The compute-capability guards above can't predict it precisely, so skip on the
        // authoritative runtime signal — the test validates correctness only where the
        // Tensor-Core path actually runs.
        try
        {
            b.GemmFp16(hA, hB, dC16, M, N, K);
        }
        catch (NotSupportedException ex)
        {
            Skip.If(true, $"FP16 Tensor-Core GEMM not supported on this device: {ex.Message}");
            return;
        }
        var got16 = b.DownloadBuffer(dC16);

        // Relative Frobenius error ||got - ref|| / ||ref|| — the standard GEMM accuracy metric.
        // Per-entry relative error is meaningless here because C entries straddle zero (sums of
        // ±products), so a near-zero entry inflates relative error despite a tiny absolute error.
        double num32 = 0, num16 = 0, den = 0;
        for (int i = 0; i < cpu.Length; i++)
        {
            double e32 = got32[i] - cpu[i], e16 = got16[i] - cpu[i];
            num32 += e32 * e32; num16 += e16 * e16; den += (double)cpu[i] * cpu[i];
        }
        double rel32 = Math.Sqrt(num32 / den), rel16 = Math.Sqrt(num16 / den);

        // FP16 inputs carry ~2^-11 relative; with FP32 accumulate over K=128 the Frobenius
        // error stays ~1e-3. A gross mismatch (layout/dtype bug) would be O(1).
        Assert.True(rel32 < 1e-3, $"TF32 GEMM drifted from FP32 reference: relFro={rel32:E3}");
        Assert.True(rel16 < 1e-2, $"FP16 Tensor-Core GEMM exceeded FP16 tolerance: relFro={rel16:E3}");
    }

    [SkippableFact]
    public void GemmFp16HalfOut_matches_fp32_reference_within_fp16_tolerance()
    {
        // The Half-OUTPUT forward GEMM (CudaBackend.GemmFp16HalfOut) — the building block that lets the forward
        // keep the matmul activation resident as Half instead of up-casting to FP32. FP16 in, FP32 accumulate,
        // Half stored output; up-cast back to FP32 (ConvertToFp32Native) to compare against the CPU reference.
        Skip.IfNot(CudaNativeBindings.IsAvailable, "CUDA driver not available");
        var eng = AiDotNetEngine.Current;
        Skip.IfNot(eng is DirectGpuTensorEngine, "active engine is not the CUDA backend");
        var warm = new Tensor<float>(new float[] { 1, 2, 3, 4 }, new[] { 2, 2 });
        _ = eng.TensorMatMul(warm, warm);
        var b = ((DirectGpuTensorEngine)eng).TestBackend as CudaBackend;
        Skip.IfNot(b is not null && b.IsAvailable, "CudaBackend not available");

        const int M = 64, K = 96, N = 48; // non-square, distinct dims
        var rng = new Random(13);
        var A = new float[M * K];
        var B = new float[K * N];
        for (int i = 0; i < A.Length; i++) A[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < B.Length; i++) B[i] = (float)(rng.NextDouble() * 2 - 1);

        var cpu = new float[M * N];
        for (int m = 0; m < M; m++)
            for (int n = 0; n < N; n++)
            {
                double acc = 0;
                for (int k = 0; k < K; k++) acc += (double)A[m * K + k] * B[k * N + n];
                cpu[m * N + n] = (float)acc;
            }

        using var dA = b!.AllocateBuffer(A);
        using var dB = b.AllocateBuffer(B);
        using var hA = b.AllocateByteBuffer(M * K * 2);
        using var hB = b.AllocateByteBuffer(K * N * 2);
        b.ConvertToFp16Native(dA, hA, M * K);
        b.ConvertToFp16Native(dB, hB, K * N);

        using var hC = b.AllocateByteBuffer(M * N * 2); // HALF output
        try { b.GemmFp16HalfOut(hA, hB, hC, M, N, K); }
        catch (NotSupportedException ex) { Skip.If(true, $"not supported: {ex.Message}"); return; }

        using var fC = b.AllocateBuffer(M * N);
        b.ConvertToFp32Native(hC, fC, M * N);
        var got = b.DownloadBuffer(fC);

        double num = 0, den = 0;
        for (int i = 0; i < cpu.Length; i++) { double e = got[i] - cpu[i]; num += e * e; den += (double)cpu[i] * cpu[i]; }
        double rel = Math.Sqrt(num / den);
        // Half-stored output rounds the result to FP16 (~2^-11 relative) on top of FP16-input GEMM error.
        Assert.True(rel < 2e-2, $"GemmFp16HalfOut exceeded FP16 tolerance: relFro={rel:E3}");
    }
}
