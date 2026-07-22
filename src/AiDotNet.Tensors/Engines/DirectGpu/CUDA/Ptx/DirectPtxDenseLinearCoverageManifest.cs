#if NET5_0_OR_GREATER
using System;
using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

internal enum DirectPtxDenseLinearCoverageStatus
{
    BaselineOnly,
    ExperimentalDirectPtx,
    PlannedDirectPtx
}

internal sealed record DirectPtxDenseLinearCoverageCell(
    string Api,
    string ExistingImplementation,
    string Semantics,
    string PhysicalLayout,
    string DTypes,
    DirectPtxDenseLinearCoverageStatus Status,
    string DirectPtxAssignment);

/// <summary>
/// Executable issue-#836 inventory. Each dense-product and linear entry point
/// has one explicit owner, including work that remains on the established path.
/// </summary>
internal static class DirectPtxDenseLinearCoverageManifest
{
    private const string Dense = "contiguous row-major A[M,K], B[K,N], C[M,N]";
    private const string DirectGemm =
        "exact-shape pointer-only PTX; canonical layout is validated once before dispatch; unqualified cells fail closed";
    private const string DirectLinear =
        "exact-shape pointer-only fused PTX; epilogue/intermediate remains in registers/shared memory; unqualified cells fail closed";

    internal static IReadOnlyList<DirectPtxDenseLinearCoverageCell> All { get; } =
    [
        Cell("CudaBackend.Gemm", "cuBLAS SGEMM", "C=alpha*A@B+beta*C", Dense, "FP32", DirectGemm),
        Cell("CudaBackend.GemmAsync", "cuBLAS SGEMM on caller stream", "asynchronous C=alpha*A@B+beta*C", Dense, "FP32", DirectGemm),
        Cell("CudaBackend.MatMul", "allocated output plus CudaBackend.Gemm", "C=A@B", Dense, "FP32", DirectGemm),
        Cell("CudaBackend.MatMulTransposed", "cuBLAS SGEMM", "C=A@transpose(B)", "A[M,K], output-major B[N,K], C[M,N]", "FP32", DirectGemm),
        Cell("CudaBackend.BatchedGemm", "cuBLAS strided-batched SGEMM", "independent dense batches", "contiguous strided [batch,M,K]/[batch,K,N]", "FP32", DirectGemm),
        Cell("CudaBackend.BatchedGemmFanout", "cuBLAS per-slice stream fanout", "independent dense batches", "contiguous strided [batch,M,K]/[batch,K,N]/[batch,M,N]", "FP32", DirectGemm),
        Cell("CudaBackend.BatchedGemmExFanout", "cuBLAS GemmEx per-slice stream fanout", "mixed-precision independent dense batches", "contiguous strided 16-bit A/B and FP32 C", "FP16/BF16->FP32", DirectGemm),
        Cell("CudaBackend.GemmFp16", "cuBLAS GemmEx", "FP16 inputs with FP32 output contract", Dense, "FP16->FP32", DirectGemm),
        Cell("CudaBackend.Hgemm", "cuBLAS Hgemm", "FP16 inputs and output", Dense, "FP16", DirectGemm),
        Cell("CudaBackend.GemmFp16In32fOut", "cuBLAS GemmEx", "FP16 multiply, FP32 accumulation/output", Dense, "FP16->FP32", DirectGemm),
        Cell("CudaBackend.GemmFp16HalfOut", "cuBLAS GemmEx", "FP16 multiply/accumulate mode and half output", Dense, "FP16", DirectGemm),
        Cell("CudaBackend.MatMulBackwardFp16Fused", "two cuBLAS GemmEx operations", "dA=dC@B^T and dB=A^T@dC", "contiguous forward tensors and two gradients", "FP16/FP32 gradients", DirectGemm),
        Cell("CudaBackend.GemmBias", "cuBLAS SGEMM plus NVRTC bias", "A@B+bias", Dense + ", bias[N]", "FP32", DirectLinear),
        Cell("CudaBackend.GemmBiasRelu", "cuBLAS/NVRTC or cuBLASLt", "relu(A@B+bias)", Dense + ", bias[N]", "FP32", DirectLinear),
        Cell("CudaBackend.GemmBiasGelu", "cuBLAS/NVRTC or cuBLASLt", "gelu_tanh(A@B+bias)", Dense + ", bias[N]", "FP32", DirectLinear),
        Cell("CudaBackend.GemmBiasSigmoid", "cuBLAS plus NVRTC sigmoid", "sigmoid(A@B+bias)", Dense + ", bias[N]", "FP32", DirectLinear),
        Cell("CudaBackend.GemmBiasTanh", "cuBLAS plus NVRTC tanh", "tanh(A@B+bias)", Dense + ", bias[N]", "FP32", DirectLinear),
        Cell("CudaBackend.GemmBiasSwish", "cuBLAS plus NVRTC swish", "swish(A@B+bias)", Dense + ", bias[N]", "FP32", DirectLinear),
        Cell("CudaBackend.GemmBiasLeakyRelu", "cuBLAS plus NVRTC leaky-ReLU", "leaky_relu(A@B+bias)", Dense + ", bias[N]", "FP32", DirectLinear),
        Cell("CudaBackend.FusedGemmBiasActivationAsync", "NVRTC fused GEMM epilogue on caller stream", "A@B+bias plus selected pointwise activation", Dense + ", bias[N]", "FP32", DirectLinear),
        Cell("CudaBackend.FusedLinearGELUTransposedM1", "direct PTX or MatMulTransposed+BiasAdd+Gelu", "gelu_tanh(x@transpose(W)+bias)", "x[K], W[N,K], bias/output[N]", "FP32", "v1 Ampere M=1 K/N=512/2048 is performance-qualified; 256/256 and 1024/4096 remain measured candidates", DirectPtxDenseLinearCoverageStatus.ExperimentalDirectPtx),
        Cell("CudaBackend.FusedLinearReLU", "NVRTC fused linear kernel", "relu(input@weight+bias)", Dense + ", bias[N]", "FP32", DirectLinear),
        Cell("CudaBackend.FusedLinearSigmoid", "NVRTC fused linear kernel", "sigmoid(input@weight+bias)", Dense + ", bias[N]", "FP32", DirectLinear),
        Cell("CudaBackend.FusedLinearTanh", "NVRTC fused linear kernel", "tanh(input@weight+bias)", Dense + ", bias[N]", "FP32", DirectLinear),
        Cell("CudaBackend.FusedLinearGELU", "NVRTC fused linear kernel", "gelu_tanh(input@weight+bias)", Dense + ", bias[N]", "FP32", "input-major exact-shape tiled PTX; M=1 output-major specialization remains separately performance-qualified", DirectPtxDenseLinearCoverageStatus.ExperimentalDirectPtx),
        Cell("CudaBackend.FusedLinearSwish", "NVRTC fused linear kernel", "swish(input@weight+bias)", Dense + ", bias[N]", "FP32", DirectLinear),
        Cell("CudaBackend.FusedLinearReLUBackward", "NVRTC backward kernels", "dInput/dWeight/dBias for ReLU linear", "contiguous forward tensors and gradients", "FP32", DirectLinear),
        Cell("CudaBackend.FusedLinearSigmoidBackward", "NVRTC backward kernels", "dInput/dWeight/dBias for sigmoid linear", "contiguous forward tensors and gradients", "FP32", DirectLinear),
        Cell("CudaBackend.FusedLinearTanhBackward", "NVRTC backward kernels", "dInput/dWeight/dBias for tanh linear", "contiguous forward tensors and gradients", "FP32", DirectLinear),
        Cell("CudaBackend.FusedLinearGELUBackward", "NVRTC backward kernels", "dInput/dWeight/dBias for GELU linear", "contiguous forward tensors and gradients", "FP32", DirectLinear),
        Cell("CudaBackend.FusedLinearSwishBackward", "NVRTC backward kernels", "dInput/dWeight/dBias for swish linear", "contiguous forward tensors and gradients", "FP32", DirectLinear),
        Cell("CudaBackend.FusedLinearCrossEntropyIndex", "NVRTC fused projection/loss", "linear plus indexed cross entropy", "hidden[M,K], W[K,N], bias[N], targets[M]", "FP32", DirectLinear),
        Cell("CudaBackend.FusedLinearCrossEntropyDense", "NVRTC fused projection/loss", "linear plus dense-target cross entropy", "hidden[M,K], W[K,N], bias[N], targets[M,N]", "FP32", DirectLinear),
        Cell("CudaBackend.FusedLoRAForward", "NVRTC fused LoRA composition", "baseOutput+scale*((input@A)@B)", "input[M,K], A[K,R], B[R,N], output[M,N]", "FP32", DirectLinear),
        Cell("CudaBackend.OuterProduct", "NVRTC outer-product kernel", "output=a outer b", "a[M], b[N], output[M,N]", "FP32", DirectGemm),
        Cell("CudaBackend.DotProduct", "NVRTC reduction", "scalar sum(a*b)", "a[K], b[K], output[1]", "FP32", DirectGemm),
        Cell("CudaBackend.StridedDotProduct", "NVRTC strided reduction", "sum(a[i]*b[offset+i*step]) for valid indices", "canonical a/b allocations with exact baked b offset and step", "FP32", DirectGemm),
        Cell("CudaBackend.BatchOuterProduct", "NVRTC batched outer-product kernel", "output[b]=a[b] outer b[b]", "a[B,M], b[B,N], output[B,M,N]", "FP32", DirectGemm),
        Cell("CudaBackend.BatchDotProduct", "NVRTC batched reduction", "output[b]=sum(a[b]*b[b])", "a[B,K], b[B,K], output[B]", "FP32", DirectGemm),
        Cell("CudaBackend.BatchedDotProduct", "NVRTC batched reduction", "output[b]=sum(a[b]*b[b])", "a[B,K], b[B,K], output[B]", "FP32", DirectGemm),
        Cell("DirectGpuTensorEngine.TensorMatMul", "backend Gemm/BatchedGemm/Hgemm routing", "public rank-aware matrix product", "canonical contiguous 2D/3D tensors", "generic public; GPU FP16/FP32", DirectGemm),
        Cell("DirectGpuTensorEngine.FusedLinear", "GemmBias family or baseline fallback", "public activation-selecting linear", "input[...,K], weight[K,N], bias[N]", "generic public; GPU FP32", DirectLinear),
        Cell("DirectGpuTensorEngine.FusedLinearBackward", "backend fused backward or baseline", "public linear activation backward", "canonical forward tensors plus gradients", "generic public; GPU FP32", DirectLinear),
        Cell("DirectGpuTensorEngine.FusedLoRAForward", "CudaBackend.FusedLoRAForward or CPU fallback", "public LoRA forward", "input[M,K], base[M,N], A[K,R], B[R,N]", "FP32", DirectLinear)
    ];

    private static DirectPtxDenseLinearCoverageCell Cell(
        string api, string existing, string semantics, string layout, string dtypes,
        string assignment,
        DirectPtxDenseLinearCoverageStatus status = DirectPtxDenseLinearCoverageStatus.ExperimentalDirectPtx) =>
        new(api, existing, semantics, layout, dtypes, status, assignment);

    internal static DirectPtxDenseLinearCoverageCell Get(string api)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(api);
        foreach (DirectPtxDenseLinearCoverageCell cell in All)
            if (string.Equals(cell.Api, api, StringComparison.Ordinal)) return cell;
        throw new KeyNotFoundException(
            $"Dense-linear API '{api}' is not assigned in the #836 coverage manifest.");
    }
}
#endif
