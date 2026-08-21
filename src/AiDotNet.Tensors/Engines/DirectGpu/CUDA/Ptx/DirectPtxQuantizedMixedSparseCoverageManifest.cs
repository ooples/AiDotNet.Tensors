using System;
using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

internal enum DirectPtxQuantizedMixedSparseCoverageStatus
{
    ExperimentalDirectPtx,
    PlannedDirectPtx
}

internal sealed record DirectPtxQuantizedMixedSparseCoverageCell(
    string Api,
    string ExistingImplementation,
    string Semantics,
    string PhysicalLayout,
    string DTypes,
    string DirectPtxAssignment,
    DirectPtxQuantizedMixedSparseCoverageStatus Status);

/// <summary>
/// Executable #837 inventory. Every in-scope CUDA-facing mixed-precision,
/// weight-only quantized, and 2:4/CSR sparse operation has one PTX assignment.
/// </summary>
internal static class DirectPtxQuantizedMixedSparseCoverageManifest
{
    internal static IReadOnlyList<DirectPtxQuantizedMixedSparseCoverageCell> All { get; } =
    [
        Planned("CudaBackend.Hgemm", "cublasHgemm", "C=A@B", "A[M,K], B[K,N], C[M,N]", "FP16->FP16", "dense-m-ge-16-fp16-mma"),
        Planned("CudaBackend.GemmFp16In32fOut", "cublasGemmEx", "C=fp32(A@B)", "A[M,K], B[K,N], C[M,N]", "FP16->FP32", "dense-m-ge-16-fp16-mma-fp32acc"),
        Planned("CudaBackend.GemmFp16HalfOut", "cublasGemmEx", "C=fp16(fp32(A@B))", "A[M,K], B[K,N], C[M,N]", "FP16->FP16", "dense-m-ge-16-fp16-mma-half-store"),
        Planned("CudaBackend.GemmFp16", "cublasGemmEx", "C=fp32(A@B)", "A[M,K], B[K,N], C[M,N]", "FP16->FP32", "dense-m-ge-16-fp16-mma-fp32acc"),
        Planned("CudaBackend.MatMulBackwardFp16Fused", "two cublasGemmEx launches", "dA=dC@transpose(B); dB=transpose(A)@dC", "canonical row-major, logical transpose", "FP16->FP16/FP32", "fused-backward-two-mma-kernels"),
        Direct("CudaBackend.FusedLinearGELUFp16TransposedM1", "direct PTX or cublasGemmEx+bias+GELU", "gelu_tanh(fp32(x@transpose(W))+bias)", "x[K], W[N,K], bias/output[N]", "FP16/FP16/FP32->FP32", "v2 Ampere M=1 half2 K/N=512/2048 and 1024/4096 promoted; 256/256 rejected"),
        Direct("CudaBackend.FusedLinearGELUFp16TransposedM16", "direct PTX async MMA or cublasGemmEx+bias+GELU", "gelu_tanh(fp32(X@transpose(W))+bias)", "X[16,K], W[N,K], bias[N], output[16,N]", "FP16/FP16/FP32->FP32", "v3 Ampere M=16 bank-conflict-free cp.async+mma: 1024/4096 promoted; 512/2048 experiment-only"),
        Direct("CudaBackend.FusedLinearGELUW8A8TransposedM1", "direct PTX DP4A or resident NVRTC fallback", "gelu_tanh(s32(x@transpose(W))*activationScale*weightScale+bias)", "x[K], W[N,K], activationScale[1], weightScale/bias/output[N]", "INT8/INT8/FP32->FP32", "v4 Ampere M=1 DP4A 1024/4096 promoted at 1.507x conservative median"),
        Planned("CudaBackend.DequantGemmInt8", "NVRTC dequant GEMM", "C=A@dequant(W,scale)", "group-scaled packed weights", "FP32/INT8->FP32", "w8a16-or-w8a8-dp4a/mma"),
        Planned("CudaBackend.DequantGemmInt4", "NVRTC dequant GEMM", "C=A@dequant(W,scale)", "two signed nibbles/byte", "FP32/INT4->FP32", "w4a16-mma-with-register-dequant"),
        Planned("CudaBackend.DequantGemmFp8E4M3", "NVRTC dequant GEMM", "C=A@dequant(W,scale)", "E4M3 bytes plus scales", "FP32/FP8->FP32", "fp8-family-specialization"),
        Planned("CudaBackend.Enforce2x4Sparsity", "NVRTC top-2-of-4 packing", "retain two largest magnitudes per group of four", "values plus 2-bit indices", "FP32->2:4", "offline-2x4-pack"),
        Planned("CudaBackend.Decompress2x4Sparse", "NVRTC unpack", "dense=decompress(values,indices)", "values plus 2-bit indices", "2:4->FP32", "diagnostic-unpack"),
        Planned("CudaBackend.SparseGemm", "NVRTC mma.sp path or scalar fallback", "C=alpha*sparseA@B+beta*C", "2:4 values/indices plus dense B", "FP32", "mma-sp-shape-family"),
        Planned("CudaBackend.SparseGemmBiasRelu", "SparseGemm then bias/ReLU", "relu(sparseA@B+bias)", "2:4 values/indices plus dense B", "FP32", "fused-mma-sp-bias-relu"),
        Planned("CudaBackend.FusedSparseLinear", "NVRTC CSR row kernel", "activation(input@transpose(CSR(W))+bias)", "CSR row offsets/columns/values", "FP32", "csr-density-bucket-family")
    ];

    private static readonly Dictionary<string, DirectPtxQuantizedMixedSparseCoverageCell> ByApi =
        CreateByApi();

    internal static DirectPtxQuantizedMixedSparseCoverageCell Get(string api)
    {
        PtxCompat.ThrowIfNullOrWhiteSpace(api, nameof(api));
        return ByApi.TryGetValue(api, out DirectPtxQuantizedMixedSparseCoverageCell? cell)
            ? cell
            : throw new KeyNotFoundException($"No #837 coverage cell is assigned to '{api}'.");
    }

    private static DirectPtxQuantizedMixedSparseCoverageCell Planned(
        string api, string implementation, string semantics, string layout,
        string dtypes, string assignment) =>
        new(api, implementation, semantics, layout, dtypes, assignment,
            DirectPtxQuantizedMixedSparseCoverageStatus.PlannedDirectPtx);

    private static Dictionary<string, DirectPtxQuantizedMixedSparseCoverageCell> CreateByApi()
    {
        var result = new Dictionary<string, DirectPtxQuantizedMixedSparseCoverageCell>(StringComparer.Ordinal);
        foreach (DirectPtxQuantizedMixedSparseCoverageCell cell in All)
            result.Add(cell.Api, cell);
        return result;
    }

    private static DirectPtxQuantizedMixedSparseCoverageCell Direct(
        string api, string implementation, string semantics, string layout,
        string dtypes, string assignment) =>
        new(api, implementation, semantics, layout, dtypes, assignment,
            DirectPtxQuantizedMixedSparseCoverageStatus.ExperimentalDirectPtx);
}
