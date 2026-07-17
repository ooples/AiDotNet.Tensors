// Copyright (c) AiDotNet. All rights reserved.
// Weight-only fused dequant-GEMM dispatch (P0) for the Metal backend.

namespace AiDotNet.Tensors.Engines.DirectGpu.Metal;

public sealed partial class MetalBackend
{
    /// <summary>
    /// Weight-only fused dequant-GEMM for integer weights (int8 or unpacked int4): C[M,N] =
    /// act[M,K] · (scale · W[K,N]). Symmetric per-tensor/per-group scales, matching the CPU oracle
    /// FusedDequantMatmulKernels. <paramref name="weightsInt"/> is an int buffer (AllocateIntBuffer)
    /// of decoded integer weight values.
    /// </summary>
    public IGpuBuffer DequantGemmInt(IGpuBuffer activations, IGpuBuffer weightsInt, IGpuBuffer scales,
        int M, int K, int N, int groupSize, int scaleCount)
        => LaunchDequantGemm("dequant_gemm_int", activations, weightsInt, scales, M, K, N, groupSize, scaleCount);

    /// <summary>
    /// Weight-only fused dequant-GEMM for OCP FP8 E4M3 weights: C[M,N] = act[M,K] ·
    /// (scale · decode_e4m3(W[K,N])). <paramref name="weightsFp8Raw"/> is an int buffer of raw fp8
    /// bytes (0..255); decode matches Float8E4M3.ToFloat.
    /// </summary>
    public IGpuBuffer DequantGemmFp8E4M3(IGpuBuffer activations, IGpuBuffer weightsFp8Raw, IGpuBuffer scales,
        int M, int K, int N, int groupSize, int scaleCount)
        => LaunchDequantGemm("dequant_gemm_fp8", activations, weightsFp8Raw, scales, M, K, N, groupSize, scaleCount);

    private IGpuBuffer LaunchDequantGemm(string kernelName, IGpuBuffer act, IGpuBuffer weights, IGpuBuffer scales,
        int M, int K, int N, int groupSize, int scaleCount)
    {
        ThrowIfDisposed();
        var output = AllocateBuffer(M * N);
        var pipeline = GetPipeline("QuantGemm", _quantGemmLibrary, kernelName);
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(M * N);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer((MetalGpuBuffer)act, 0);
        encoder.SetBuffer((MetalGpuBuffer)weights, 1);
        encoder.SetBuffer((MetalGpuBuffer)scales, 2);
        encoder.SetBuffer((MetalGpuBuffer)output, 3);
        encoder.SetBytes(M, 4);
        encoder.SetBytes(K, 5);
        encoder.SetBytes(N, 6);
        encoder.SetBytes(groupSize, 7);
        encoder.SetBytes(scaleCount, 8);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
        return output;
    }
}
