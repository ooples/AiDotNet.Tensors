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
    private const string PlannedGemm =
        "shape/dtype-specific PTX GEMM tile; no generic runtime-stride kernel";
    private const string PlannedLinear =
        "shape/dtype/activation-specific PTX fusion with a prepacked weight contract";
    private const string TiledLinear =
        "general-M output-major register-blocked PTX tile (PtxFusedLinearTiledKernel) " +
        "with a fused activation epilogue; fails closed until GPU-validated and promoted";
    private const string TiledGemmBias =
        "standard-B register-blocked PTX tile (PtxFusedGemmBiasKernel) with a fused " +
        "bias+activation epilogue; fails closed until GPU-validated and promoted";
    private const string TiledBackward =
        "direct PTX backward set: activation-gradient dZ (PtxLinearActivationBackwardKernel) " +
        "then dInput=dZ@W (plain GEMM tile), dWeight=transpose(dZ)@X (PtxGemmContractMKernel) " +
        "and dBias=colsum(dZ) (PtxBiasGradientKernel); fails closed until GPU-validated and promoted";

    internal static IReadOnlyList<DirectPtxDenseLinearCoverageCell> All { get; } =
    [
        Cell("CudaBackend.Gemm", "cuBLAS SGEMM", "C=alpha*A@B+beta*C", Dense, "FP32", "direct PTX GEMM tile (PtxGemmKernel), alpha=1/beta=0 no-bias epilogue; fail-closed pending GPU validation", DirectPtxDenseLinearCoverageStatus.ExperimentalDirectPtx),
        Cell("CudaBackend.GemmAsync", "cuBLAS SGEMM on caller stream", "asynchronous C=alpha*A@B+beta*C", Dense, "FP32", "direct PTX GEMM tile (PtxGemmKernel) on the caller stream; fail-closed pending GPU validation", DirectPtxDenseLinearCoverageStatus.ExperimentalDirectPtx),
        Cell("CudaBackend.MatMul", "allocated output plus CudaBackend.Gemm", "C=A@B", Dense, "FP32", "direct PTX GEMM tile (PtxGemmKernel); fail-closed pending GPU validation", DirectPtxDenseLinearCoverageStatus.ExperimentalDirectPtx),
        Cell("CudaBackend.MatMulTransposed", "cuBLAS SGEMM", "C=A@transpose(B)", "A[M,K], output-major B[N,K], C[M,N]", "FP32", "direct PTX transpose-B tile (PtxMatMulTransposedKernel); fail-closed pending GPU validation", DirectPtxDenseLinearCoverageStatus.ExperimentalDirectPtx),
        Cell("CudaBackend.BatchedGemm", "cuBLAS strided-batched SGEMM", "independent dense batches", "contiguous strided [batch,M,K]/[batch,K,N]", "FP32", "direct PTX strided-batched tile (PtxBatchedGemmKernel); fail-closed pending GPU validation", DirectPtxDenseLinearCoverageStatus.ExperimentalDirectPtx),
        Cell("CudaBackend.BatchedGemmFanout", "cuBLAS batched pointer fanout", "one A with multiple B/C products", "contiguous A plus pointer-array B/C", "FP32", "direct PTX pointer-fanout tile (PtxBatchedGemmFanoutKernel), B/C bases dereferenced from device pointer arrays per ctaid.z; fail-closed pending GPU validation", DirectPtxDenseLinearCoverageStatus.ExperimentalDirectPtx),
        Cell("CudaBackend.BatchedGemmExFanout", "cuBLAS GemmEx pointer fanout", "mixed-precision fanout products", "contiguous A plus pointer-array B/C", "FP16/BF16/FP32", "direct PTX fp16 pointer-fanout tile (PtxBatchedGemmExFanoutKernel), fp16-in fp32-accumulate fp16-out, B/C bases from device pointer arrays per ctaid.z; fail-closed pending GPU validation", DirectPtxDenseLinearCoverageStatus.ExperimentalDirectPtx),
        Cell("CudaBackend.GemmFp16", "cuBLAS GemmEx", "FP16 inputs with FP32 output contract", Dense, "FP16->FP32", "direct PTX FP16-input FP32-accumulate tile (PtxGemmFp16Kernel); fail-closed pending GPU validation", DirectPtxDenseLinearCoverageStatus.ExperimentalDirectPtx),
        Cell("CudaBackend.Hgemm", "cuBLAS Hgemm", "FP16 inputs and output", Dense, "FP16", "direct PTX fp16-output tile (PtxHgemmKernel); fail-closed pending GPU validation", DirectPtxDenseLinearCoverageStatus.ExperimentalDirectPtx),
        Cell("CudaBackend.GemmFp16In32fOut", "cuBLAS GemmEx", "FP16 multiply, FP32 accumulation/output", Dense, "FP16->FP32", "direct PTX FP16-input FP32-accumulate tile (PtxGemmFp16Kernel); fail-closed pending GPU validation", DirectPtxDenseLinearCoverageStatus.ExperimentalDirectPtx),
        Cell("CudaBackend.GemmFp16HalfOut", "cuBLAS GemmEx", "FP16 multiply/accumulate mode and half output", Dense, "FP16", "direct PTX fp16-output tile (PtxHgemmKernel); fail-closed pending GPU validation", DirectPtxDenseLinearCoverageStatus.ExperimentalDirectPtx),
        Cell("CudaBackend.MatMulBackwardFp16Fused", "two cuBLAS GemmEx operations", "dA=dC@B^T and dB=A^T@dC", "contiguous forward tensors and two gradients", "FP16/FP32 gradients", "direct PTX fp16 transpose-B dA (PtxGemmFp16TransposedKernel) and fp16 contract-M dB (PtxGemmFp16ContractMKernel); fail-closed pending GPU validation", DirectPtxDenseLinearCoverageStatus.ExperimentalDirectPtx),
        Cell("CudaBackend.GemmBias", "cuBLAS SGEMM plus NVRTC bias", "A@B+bias", Dense + ", bias[N]", "FP32", TiledGemmBias, DirectPtxDenseLinearCoverageStatus.ExperimentalDirectPtx),
        Cell("CudaBackend.GemmBiasRelu", "cuBLAS/NVRTC or cuBLASLt", "relu(A@B+bias)", Dense + ", bias[N]", "FP32", TiledGemmBias, DirectPtxDenseLinearCoverageStatus.ExperimentalDirectPtx),
        Cell("CudaBackend.GemmBiasGelu", "cuBLAS/NVRTC or cuBLASLt", "gelu_tanh(A@B+bias)", Dense + ", bias[N]", "FP32", TiledGemmBias, DirectPtxDenseLinearCoverageStatus.ExperimentalDirectPtx),
        Cell("CudaBackend.GemmBiasSigmoid", "cuBLAS plus NVRTC sigmoid", "sigmoid(A@B+bias)", Dense + ", bias[N]", "FP32", TiledGemmBias, DirectPtxDenseLinearCoverageStatus.ExperimentalDirectPtx),
        Cell("CudaBackend.GemmBiasTanh", "cuBLAS plus NVRTC tanh", "tanh(A@B+bias)", Dense + ", bias[N]", "FP32", TiledGemmBias, DirectPtxDenseLinearCoverageStatus.ExperimentalDirectPtx),
        Cell("CudaBackend.GemmBiasSwish", "cuBLAS plus NVRTC swish", "swish(A@B+bias)", Dense + ", bias[N]", "FP32", TiledGemmBias, DirectPtxDenseLinearCoverageStatus.ExperimentalDirectPtx),
        Cell("CudaBackend.GemmBiasLeakyRelu", "cuBLAS plus NVRTC leaky-ReLU", "leaky_relu(A@B+bias)", Dense + ", bias[N]", "FP32", TiledGemmBias, DirectPtxDenseLinearCoverageStatus.ExperimentalDirectPtx),
        Cell("CudaBackend.FusedLinearGELUTransposedM1", "direct PTX or MatMulTransposed+BiasAdd+Gelu", "gelu_tanh(x@transpose(W)+bias)", "x[K], W[N,K], bias/output[N]", "FP32", "v1 Ampere M=1 K/N=512/2048 is performance-qualified; 256/256 and 1024/4096 remain measured candidates", DirectPtxDenseLinearCoverageStatus.ExperimentalDirectPtx),
        Cell("CudaBackend.FusedLinearReLU", "NVRTC fused linear kernel", "relu(input@weight+bias)", Dense + ", bias[N]", "FP32", TiledLinear, DirectPtxDenseLinearCoverageStatus.ExperimentalDirectPtx),
        Cell("CudaBackend.FusedLinearSigmoid", "NVRTC fused linear kernel", "sigmoid(input@weight+bias)", Dense + ", bias[N]", "FP32", TiledLinear, DirectPtxDenseLinearCoverageStatus.ExperimentalDirectPtx),
        Cell("CudaBackend.FusedLinearTanh", "NVRTC fused linear kernel", "tanh(input@weight+bias)", Dense + ", bias[N]", "FP32", TiledLinear, DirectPtxDenseLinearCoverageStatus.ExperimentalDirectPtx),
        Cell("CudaBackend.FusedLinearGELU", "NVRTC fused linear kernel", "gelu_tanh(input@weight+bias)", Dense + ", bias[N]", "FP32", "input-major general-M contract remains baseline; M=1 output-major route is separate", DirectPtxDenseLinearCoverageStatus.ExperimentalDirectPtx),
        Cell("CudaBackend.FusedLinearSwish", "NVRTC fused linear kernel", "swish(input@weight+bias)", Dense + ", bias[N]", "FP32", TiledLinear, DirectPtxDenseLinearCoverageStatus.ExperimentalDirectPtx),
        Cell("CudaBackend.FusedLinearReLUBackward", "NVRTC backward kernels", "dInput/dWeight/dBias for ReLU linear", "contiguous forward tensors and gradients", "FP32", TiledBackward, DirectPtxDenseLinearCoverageStatus.ExperimentalDirectPtx),
        Cell("CudaBackend.FusedLinearSigmoidBackward", "NVRTC backward kernels", "dInput/dWeight/dBias for sigmoid linear", "contiguous forward tensors and gradients", "FP32", TiledBackward, DirectPtxDenseLinearCoverageStatus.ExperimentalDirectPtx),
        Cell("CudaBackend.FusedLinearTanhBackward", "NVRTC backward kernels", "dInput/dWeight/dBias for tanh linear", "contiguous forward tensors and gradients", "FP32", TiledBackward, DirectPtxDenseLinearCoverageStatus.ExperimentalDirectPtx),
        Cell("CudaBackend.FusedLinearGELUBackward", "NVRTC backward kernels", "dInput/dWeight/dBias for GELU linear", "contiguous forward tensors and gradients", "FP32", TiledBackward, DirectPtxDenseLinearCoverageStatus.ExperimentalDirectPtx),
        Cell("CudaBackend.FusedLinearSwishBackward", "NVRTC backward kernels", "dInput/dWeight/dBias for swish linear", "contiguous forward tensors and gradients", "FP32", TiledBackward, DirectPtxDenseLinearCoverageStatus.ExperimentalDirectPtx),
        Cell("CudaBackend.FusedLinearCrossEntropyIndex", "NVRTC fused projection/loss", "linear plus indexed cross entropy", "hidden[M,K], W[K,N], bias[N], targets[M]", "FP32", "GemmBias tile for logits plus fused row-wise softmax-CE loss/gradient (PtxSoftmaxCrossEntropyIndexKernel); fail-closed pending GPU validation", DirectPtxDenseLinearCoverageStatus.ExperimentalDirectPtx),
        Cell("CudaBackend.FusedLinearCrossEntropyDense", "NVRTC fused projection/loss", "linear plus dense-target cross entropy", "hidden[M,K], W[K,N], bias[N], targets[M,N]", "FP32", "GemmBias tile for logits plus fused row-wise dense softmax-CE loss/gradient (PtxSoftmaxCrossEntropyDenseKernel); fail-closed pending GPU validation", DirectPtxDenseLinearCoverageStatus.ExperimentalDirectPtx),
        Cell("CudaBackend.FusedLoRAForward", "NVRTC fused LoRA composition", "base+scale*(X@transpose(A))@transpose(B)", "X[M,K], output-major A[R,K], output-major B[N,R], base/output[M,N]", "FP32", "direct PTX single-launch two-stage tile with a shared-resident Z intermediate (PtxFusedLoRAForwardKernel), r=64; fail-closed pending GPU validation", DirectPtxDenseLinearCoverageStatus.ExperimentalDirectPtx),
        Cell("CudaBackend.OuterProduct", "NVRTC outer-product kernel", "output=a outer b", "a[M], b[N], output[M,N]", "FP32", "direct PTX kernel (PtxOuterProductKernel); fail-closed pending GPU validation", DirectPtxDenseLinearCoverageStatus.ExperimentalDirectPtx),
        Cell("CudaBackend.DotProduct", "NVRTC reduction", "scalar sum(a*b)", "a[K], b[K], output[1]", "FP32", "direct PTX warp+CTA reduction (PtxDotProductKernel); fail-closed pending GPU validation", DirectPtxDenseLinearCoverageStatus.ExperimentalDirectPtx),
        Cell("DirectGpuTensorEngine.TensorMatMul", "backend Gemm/BatchedGemm/Hgemm routing", "public rank-aware matrix product", "canonical contiguous 2D/3D tensors", "generic public; GPU FP16/FP32", PlannedGemm),
        Cell("DirectGpuTensorEngine.FusedLinear", "GemmBias family or baseline fallback", "public activation-selecting linear", "input[...,K], output-major weight[N,K], bias[N]", "generic public; GPU FP32", "backend FusedLinear{ReLU,Sigmoid,Tanh,GELU} route through the tiled PTX kernel via a fail-closed TryDirectPtxFusedLinearTiled guard (returns false until GPU-promoted), then fall through to the NVRTC path", DirectPtxDenseLinearCoverageStatus.ExperimentalDirectPtx),
        Cell("DirectGpuTensorEngine.FusedLinearBackward", "backend fused backward or baseline", "public linear activation backward", "canonical forward tensors plus gradients", "generic public; GPU FP32", PlannedLinear),
        Cell("DirectGpuTensorEngine.FusedLoRAForward", "CudaBackend.FusedLoRAForward or CPU fallback", "public LoRA forward", "input[M,K], base[M,N], A[K,R], B[R,N]", "FP32", PlannedLinear)
    ];

    private static DirectPtxDenseLinearCoverageCell Cell(
        string api, string existing, string semantics, string layout, string dtypes,
        string assignment,
        DirectPtxDenseLinearCoverageStatus status = DirectPtxDenseLinearCoverageStatus.PlannedDirectPtx) =>
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
