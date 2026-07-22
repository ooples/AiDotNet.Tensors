using System;
using System.Collections.Generic;
using System.Linq;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

internal enum DirectPtxLayoutCoverageStatus
{
    ExistingBackend,
    ExperimentalDirectPtx,
    PromotedDirectPtx,
    PlannedDirectPtx
}

internal sealed record DirectPtxLayoutCoverageCell(
    string Api,
    string ExistingImplementation,
    string Semantics,
    string PhysicalLayout,
    string DTypes,
    string DirectPtxAssignment,
    DirectPtxLayoutCoverageStatus Status);

/// <summary>
/// Executable issue-#845 inventory. Every CUDA tensor-layout, copy, transpose,
/// packing, and dtype-conversion boundary has one explicit direct-PTX lane.
/// Shape and layout families remain closed sets: a planned row must be split
/// into exact physical ABI cells before it can be promoted.
/// </summary>
internal static class DirectPtxLayoutCoverageManifest
{
    private const string Vector = "canonical contiguous vector";
    private const string Rows = "canonical contiguous row-major [rows,cols]";

    internal static IReadOnlyList<DirectPtxLayoutCoverageCell> All { get; } =
    [
        Experimental("CudaBackend.ConvertToFp16", "NVRTC convert_fp32_to_fp16 / _native", "round-to-nearest-even FP32 to FP16", Vector, "FP32 in; FP16 out", "v1 Ampere linear-vec4 exact-size cells"),
        Experimental("CudaBackend.ConvertToFp32", "NVRTC convert_fp16_to_fp32", "widen FP16 to FP32", Vector, "FP16 in; FP32 out", "v1 Ampere linear-vec4 exact-size cells"),
        Planned("CudaBackend.ConvertToBf16", "none - no backend op exists yet", "round FP32 to BF16", Vector, "FP32 in; BF16 out", "needs a public backend op first, then linear-vec4-bf16-families"),
        Planned("CudaBackend.ConvertBf16ToFp32", "none - no backend op exists yet", "widen BF16 to FP32", Vector, "BF16 in; FP32 out", "needs a public backend op first, then linear-vec4-widen-families"),
        Planned("CudaBackend.Copy", "NVRTC memcpy_device / cudaMemcpyAsync", "contiguous device copy", Vector, "FP32/FP16", "linear-vec4-copy-families"),
        Planned("CudaBackend.Transpose", "NVRTC transpose", "2D transpose [rows,cols] to [cols,rows]", Rows, "FP32", "shared-tile-32x32-transpose-families"),
        Planned("CudaBackend.TransposeBatched", "NVRTC transpose_batched", "batched 2D transpose", "canonical contiguous [batch,rows,cols]", "FP32", "shared-tile-batched-transpose-families"),
        Planned("CudaBackend.Permute", "NVRTC permute", "arbitrary axis permutation with baked strides", "canonical contiguous with baked permutation", "FP32", "baked-permutation-stride-families"),
        Planned("CudaBackend.Reshape", "view/metadata copy", "contiguous reinterpretation", Vector, "FP32/FP16", "no-op-or-copy-routing"),
        Planned("CudaBackend.PackInterleave", "NVRTC pack_interleave", "interleave channels into packed layout", Rows, "FP32/FP16", "baked-interleave-packing-families"),
        Planned("CudaBackend.UnpackInterleave", "NVRTC unpack_interleave", "de-interleave packed layout", Rows, "FP32/FP16", "baked-deinterleave-families"),
        Planned("CudaBackend.PackQ8", "NVRTC pack_q8_0", "block-quantize FP32 to Q8_0", Rows, "FP32 in; INT8 blocks out", "block-quantize-packing-families"),
        Planned("CudaBackend.UnpackQ8", "NVRTC unpack_q8_0", "dequantize Q8_0 to FP32", Rows, "INT8 blocks in; FP32 out", "block-dequantize-families"),
        Planned("CudaBackend.PadContiguous", "NVRTC pad_contiguous", "pad to an aligned physical extent", Rows, "FP32/FP16", "baked-pad-extent-families"),
        Planned("CudaBackend.SliceContiguous", "NVRTC slice_contiguous", "extract a contiguous sub-view", Rows, "FP32/FP16", "baked-slice-offset-families"),
        Planned("CudaBackend.Concat", "NVRTC concat_axis", "concatenate along a baked axis", Rows, "FP32/FP16", "baked-axis-concat-families"),
        Planned("DirectGpuEngine.ConvertToFp16", "array upload + CudaBackend.ConvertToFp16 + download", "public array FP16 cast", Vector, "generic public; CUDA FP32/FP16", "public-array-cast-routing"),
        Planned("DirectGpuTensorEngine.TensorToHalf", "resident CudaBackend.ConvertToFp16 or CPU fallback", "public tensor FP16 cast", "logical strided tensor; canonical contiguous fast path", "generic public; CUDA FP32/FP16", "public-tensor-cast-routing-and-materialization"),
        Planned("DirectGpuTensorEngine.TensorTranspose", "CudaBackend.Transpose or CPU fallback", "public 2D tensor transpose", "logical 2D; canonical contiguous admitted view", "generic public; CUDA FP32", "public-transpose-routing"),
        Planned("DirectGpuTensorEngine.TensorPermute", "CudaBackend.Permute or CPU fallback", "public axis permutation", "logical strided tensor; canonical contiguous admitted view", "generic public; CUDA FP32", "public-permute-routing"),
    ];

    private static readonly IReadOnlyDictionary<string, DirectPtxLayoutCoverageCell> ByApi =
        All.ToDictionary(cell => cell.Api, StringComparer.Ordinal);

    internal static DirectPtxLayoutCoverageCell Get(string api) =>
        ByApi.TryGetValue(api, out DirectPtxLayoutCoverageCell? cell)
            ? cell
            : throw new KeyNotFoundException($"No direct-PTX layout coverage cell for '{api}'.");

    private static DirectPtxLayoutCoverageCell Experimental(
        string api, string existing, string semantics, string layout, string dtypes, string assignment) =>
        new(api, existing, semantics, layout, dtypes, assignment, DirectPtxLayoutCoverageStatus.ExperimentalDirectPtx);

    private static DirectPtxLayoutCoverageCell Planned(
        string api, string existing, string semantics, string layout, string dtypes, string assignment) =>
        new(api, existing, semantics, layout, dtypes, assignment, DirectPtxLayoutCoverageStatus.PlannedDirectPtx);
}
