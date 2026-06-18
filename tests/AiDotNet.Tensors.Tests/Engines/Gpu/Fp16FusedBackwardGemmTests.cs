using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Gpu;

/// <summary>
/// Validates <see cref="CudaBackend.MatMulBackwardFp16Fused"/> on real hardware against an authoritative CPU
/// FP32 reference. This is the fused FP16 backward that replaces the eager path's two TensorTranspose + two
/// TensorMatMul FP32 dispatches (the measured dominant VRAM cost of the FP16 hetero path) with two
/// transpose-free Tensor-Core Half GEMMs into caller-owned buffers. Because the gradients are derived through
/// cuBLAS's column-major convention (a layout/transpose-flag bug would silently emit WRONG gradients), this
/// parity test runs FIRST — before the kernel is wired into training — to catch any convention error.
/// </summary>
public class Fp16FusedBackwardGemmTests
{
    [SkippableTheory]
    [InlineData(false)] // FP32 gradient outputs
    [InlineData(true)]  // FP16 gradient outputs (FP32 accumulate) — the dtype the hetero backward needs
    public void MatMulBackwardFp16Fused_matches_fp32_reference(bool gradOutHalf)
    {
        Skip.IfNot(CudaNativeBindings.IsAvailable, "CUDA driver not available");
        var eng = AiDotNetEngine.Current;
        Skip.IfNot(eng is DirectGpuTensorEngine, "active engine is not the CUDA backend");
        var warm = new Tensor<float>(new float[] { 1, 2, 3, 4 }, new[] { 2, 2 });
        _ = eng.TensorMatMul(warm, warm); // force kernel-module compile
        var b = ((DirectGpuTensorEngine)eng).TestBackend as CudaBackend;
        Skip.IfNot(b is not null && b.IsAvailable, "CudaBackend not available");

        // Forward: C[M,N] = A[M,K] · B[K,N]. Non-square + all dims distinct so a transpose/layout bug can't
        // hide behind symmetry.
        const int M = 64, K = 96, N = 48;
        var rng = new Random(7);
        var A = new float[M * K];
        var B = new float[K * N];
        var gradC = new float[M * N];
        for (int i = 0; i < A.Length; i++) A[i] = (float)(rng.NextDouble() * 2 - 1);     // [-1,1] FP16-safe
        for (int i = 0; i < B.Length; i++) B[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < gradC.Length; i++) gradC[i] = (float)(rng.NextDouble() * 2 - 1);

        // CPU FP32 reference: gradA[M,K] = gradC · Bᵀ ; gradB[K,N] = Aᵀ · gradC.
        var refGradA = new float[M * K];
        for (int m = 0; m < M; m++)
            for (int k = 0; k < K; k++)
            {
                double acc = 0;
                for (int n = 0; n < N; n++) acc += (double)gradC[m * N + n] * B[k * N + n];
                refGradA[m * K + k] = (float)acc;
            }
        var refGradB = new float[K * N];
        for (int k = 0; k < K; k++)
            for (int n = 0; n < N; n++)
            {
                double acc = 0;
                for (int m = 0; m < M; m++) acc += (double)A[m * K + k] * gradC[m * N + n];
                refGradB[k * N + n] = (float)acc;
            }

        using var dA = b!.AllocateBuffer(A);
        using var dB = b.AllocateBuffer(B);
        using var dGradC = b.AllocateBuffer(gradC);
        using var hA = b.AllocateByteBuffer(M * K * 2);
        using var hB = b.AllocateByteBuffer(K * N * 2);
        using var hGradC = b.AllocateByteBuffer(M * N * 2);
        b.ConvertToFp16Native(dA, hA, M * K);
        b.ConvertToFp16Native(dB, hB, K * N);
        b.ConvertToFp16Native(dGradC, hGradC, M * N);

        // Outputs: FP32 buffers, or FP16 byte buffers up-cast back to FP32 for comparison.
        using var dGradA = gradOutHalf ? b.AllocateByteBuffer(M * K * 2) : b.AllocateBuffer(M * K);
        using var dGradB = gradOutHalf ? b.AllocateByteBuffer(K * N * 2) : b.AllocateBuffer(K * N);
        try
        {
            b.MatMulBackwardFp16Fused(hGradC, hA, hB, dGradA, dGradB, M, N, K, gradOutHalf);
        }
        catch (NotSupportedException ex)
        {
            Skip.If(true, $"FP16 Tensor-Core GEMM not supported on this device: {ex.Message}");
            return;
        }
        float[] gotGradA, gotGradB;
        if (gradOutHalf)
        {
            using var fA = b.AllocateBuffer(M * K);
            using var fB = b.AllocateBuffer(K * N);
            b.ConvertToFp32Native(dGradA, fA, M * K);
            b.ConvertToFp32Native(dGradB, fB, K * N);
            gotGradA = b.DownloadBuffer(fA);
            gotGradB = b.DownloadBuffer(fB);
        }
        else
        {
            gotGradA = b.DownloadBuffer(dGradA);
            gotGradB = b.DownloadBuffer(dGradB);
        }

        // Relative Frobenius error (entries straddle zero, so per-entry relative error is meaningless).
        double RelFro(float[] got, float[] re)
        {
            double num = 0, den = 0;
            for (int i = 0; i < re.Length; i++) { double e = got[i] - re[i]; num += e * e; den += (double)re[i] * re[i]; }
            return Math.Sqrt(num / Math.Max(den, 1e-30));
        }
        double relA = RelFro(gotGradA, refGradA);
        double relB = RelFro(gotGradB, refGradB);
        // FP16 inputs (~2^-11 relative) with FP32 accumulate over K/M ≈ 1e-3. A layout/transpose bug = O(1).
        Assert.True(relA < 1e-2, $"fused gradA exceeded FP16 tolerance: relFro={relA:E3}");
        Assert.True(relB < 1e-2, $"fused gradB exceeded FP16 tolerance: relFro={relB:E3}");
    }
}
