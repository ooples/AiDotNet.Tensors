using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Gpu;

/// <summary>
/// Pilot for the FP16-NATIVE kernel path (Tensors #558 layer 7). The convert-based activation compression
/// (layers 5–6) was proven to RAISE peak VRAM because ConvertToFp16/Fp32 need src+dst live simultaneously
/// (a transient double). The real win is FP16-native kernels: each op reads the FP16 buffer DIRECTLY,
/// computes in FP32 in-register, and writes FP16 — no separate FP32 buffer, no transient, half the
/// bandwidth, Tensor-Core-friendly. The fp16_gelu kernel already exists + compiles; this pilot wires a
/// launcher (CudaBackend.Fp16Gelu) and confirms it (1) runs correctly on genuinely HALF buffers and
/// (2) matches the FP32 GELU within FP16 tolerance — validating the rewrite path end-to-end on the GPU.
/// </summary>
public class Fp16NativeGeluPilotTests
{
    [SkippableFact]
    public void Fp16NativeGelu_OnHalfBuffers_MatchesFp32Gelu()
    {
        Skip.IfNot(CudaNativeBindings.IsAvailable, "CUDA driver not available");
        // Use the engine's fully-initialized backend (kernel modules compiled), not a bare CudaBackend.
        var eng = AiDotNetEngine.Current;
        Skip.IfNot(eng is DirectGpuTensorEngine, "active engine is not the CUDA backend");
        // Warm up: a real GPU op forces the engine's kernel modules (incl. the FP16 module) to compile.
        var warm = new Tensor<float>(new float[] { 1, 2, 3, 4 }, new[] { 2, 2 });
        _ = eng.TensorMatMul(warm, warm);
        var b = ((DirectGpuTensorEngine)eng).TestBackend as CudaBackend;
        Skip.IfNot(b is not null && b.IsAvailable, "CudaBackend not available");
        // The header-based FP16 module (cuda_fp16.h) is EXPECTED to fail on a driver-only box — that is
        // the bug this pilot exposes. The self-contained native kernels below must work regardless.

        const int n = 2048;
        var rng = new Random(7);
        var x = new float[n];
        for (int i = 0; i < n; i++) x[i] = (float)((rng.NextDouble() * 2 - 1) * 4.0); // span FP16 normal range

        // Upload FP32 → compress to a genuine HALF buffer → run the FP16-NATIVE gelu (no FP32 materialized)
        // → upcast for readback. The activation that fp16_gelu reads AND writes is half-size throughout.
        using var fp32In = b.AllocateBuffer(x);
        using var fp16In = b.AllocateByteBuffer(n * 2);   // 2 bytes/elem
        b.ConvertToFp16Native(fp32In, fp16In, n);         // self-contained (no cuda_fp16.h)
        using var fp16Out = b.AllocateByteBuffer(n * 2);  // 2 bytes/elem
        b.Fp16Gelu(fp16In, fp16Out, n);                   // ← the pilot fp16-native kernel
        using var fp32Out = b.AllocateBuffer(n);
        b.ConvertToFp32Native(fp16Out, fp32Out, n);
        var got = b.DownloadBuffer(fp32Out);

        // The activation buffers are genuinely HALF (2 bytes/elem) — the property the memory win needs.
        Assert.Equal((long)n * 2, fp16In.SizeInBytes);
        Assert.Equal((long)n * 2, fp16Out.SizeInBytes);

        // Correctness: matches FP32 GELU within FP16 rounding (the kernel computes in FP32 in-register).
        int bad = 0;
        for (int i = 0; i < n; i++)
        {
            float xi = x[i], x3 = xi * xi * xi;
            float inner = 0.7978845608f * (xi + 0.044715f * x3);
            float exp = 0.5f * xi * (1f + (float)Math.Tanh(inner));
            Assert.False(float.IsNaN(got[i]) || float.IsInfinity(got[i]), $"non-finite at {i}");
            if (Math.Abs(got[i] - exp) > 3e-2f + 4e-2f * Math.Abs(exp)) bad++;
        }
        Assert.True(bad == 0, $"{bad}/{n} fp16-native GELU outputs exceeded FP16 tolerance vs FP32");
    }
}
