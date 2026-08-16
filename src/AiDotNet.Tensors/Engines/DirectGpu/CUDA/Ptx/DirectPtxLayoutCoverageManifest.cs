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

    /// <summary>
    /// Issue-#845 inventory. Acceptance criterion 1 requires every scoped entry
    /// point to be enumerated against a real existing implementation, so each
    /// row below was verified against the backend rather than assumed. Nine
    /// rows previously named NVRTC kernels and backend ops that exist nowhere
    /// in the tree (convert_fp32_to_bf16, transpose_batched, pack_interleave,
    /// pack_q8_0, pad_contiguous, slice_contiguous, concat_axis, and the
    /// DirectGpuEngine/TensorToHalf entry points). Those cells cannot be
    /// "ported" - they need a public backend op written first - and now say so
    /// instead of reading as straightforward PTX ports.
    /// </summary>
    internal static IReadOnlyList<DirectPtxLayoutCoverageCell> All { get; } =
    [
        // ---- Implemented: direct-PTX specializations exist and are wired ----
        Experimental("CudaBackend.ConvertToFp16", "NVRTC convert_fp32_to_fp16 / _native", "round-to-nearest-even FP32 to FP16", Vector, "FP32 in; FP16 out", "v1 Ampere linear-vec4 exact-size cells"),
        Experimental("CudaBackend.ConvertToFp32", "NVRTC convert_fp16_to_fp32", "widen FP16 to FP32", Vector, "FP16 in; FP32 out", "v1 Ampere linear-vec4 exact-size cells"),
        Experimental("CudaBackend.Transpose", "NVRTC transpose_2d (naive 16x16, uncoalesced stores)", "2D transpose [rows,cols] to [cols,rows]", Rows, "FP32", "v1 Ampere shared-tile 32x32 exact-extent cells"),

        // ---- Deliberately left on the established backend ----
        // Copy is cuMemcpyDtoDAsync: the driver's DMA copy engine, not a kernel.
        // A hand-written PTX copy is pure bandwidth-bound work with no fusion
        // opportunity, so it has no headroom against the copy engine and would
        // have to be measurably faster to clear the promotion gate. Porting it
        // would be a pessimization, so this cell stays on the existing path.
        ExistingBackend("CudaBackend.Copy", "cuMemcpyDtoDAsync (driver DMA copy engine)", "contiguous device copy", Vector, "FP32/FP16", "no direct-PTX lane - the DMA engine already saturates the link"),

        // ---- Portable, not yet ported: a real backend op exists ----
        Planned("CudaBackend.Permute", "NVRTC permute, with 2D and 3D fast paths delegating to Transpose/BatchedTranspose", "arbitrary axis permutation with baked strides", "canonical contiguous with baked permutation", "FP32", "baked-permutation-stride-families; the rank-2 case already routes through the ported Transpose"),
        Planned("CudaBackend.BatchedTranspose", "NVRTC batched transpose", "batched 2D transpose", "canonical contiguous [batch,rows,cols]", "FP32", "shared-tile-batched-transpose-families; a grid.z extension of the ported 2D tile"),

        // ---- Blocked: no backend op exists to port ----
        Planned("CudaBackend.ConvertToBf16", "none - no backend op exists", "round FP32 to BF16", Vector, "FP32 in; BF16 out", "blocked: needs a public backend op first, then linear-vec4-bf16-families"),
        Planned("CudaBackend.ConvertBf16ToFp32", "none - no backend op exists", "widen BF16 to FP32", Vector, "BF16 in; FP32 out", "blocked: needs a public backend op first, then linear-vec4-widen-families"),
        Planned("CudaBackend.Reshape", "none - reshape is host-side metadata, no device op exists", "contiguous reinterpretation", Vector, "FP32/FP16", "blocked: nothing to port - a contiguous reshape moves no data"),
        Planned("CudaBackend.PackInterleave", "none - no backend op exists", "interleave channels into packed layout", Rows, "FP32/FP16", "blocked: needs a public backend op first, then baked-interleave-packing-families"),
        Planned("CudaBackend.UnpackInterleave", "none - no backend op exists", "de-interleave packed layout", Rows, "FP32/FP16", "blocked: needs a public backend op first, then baked-deinterleave-families"),
        Planned("CudaBackend.PackQ8", "none - no backend op exists", "block-quantize FP32 to Q8_0", Rows, "FP32 in; INT8 blocks out", "blocked: needs a public backend op first, then block-quantize-packing-families"),
        Planned("CudaBackend.UnpackQ8", "none - no backend op exists", "dequantize Q8_0 to FP32", Rows, "INT8 blocks in; FP32 out", "blocked: needs a public backend op first, then block-dequantize-families"),
        Planned("CudaBackend.PadContiguous", "none - no backend op exists", "pad to an aligned physical extent", Rows, "FP32/FP16", "blocked: needs a public backend op first, then baked-pad-extent-families"),
        Planned("CudaBackend.SliceContiguous", "none - no backend op exists", "extract a contiguous sub-view", Rows, "FP32/FP16", "blocked: needs a public backend op first, then baked-slice-offset-families"),
        Planned("CudaBackend.Concat", "none - no backend op exists", "concatenate along a baked axis", Rows, "FP32/FP16", "blocked: needs a public backend op first, then baked-axis-concat-families"),

        // ---- Engine-level routing cells ----
        Planned("DirectGpuEngine.ConvertToFp16", "DirectGpuEngine exists but exposes no ConvertToFp16 entry point", "public array FP16 cast", Vector, "generic public; CUDA FP32/FP16", "blocked: needs the public array entry point before a routing lane means anything"),
        Planned("DirectGpuTensorEngine.TensorToHalf", "none - no such method exists on the engine", "public tensor FP16 cast", "logical strided tensor; canonical contiguous fast path", "generic public; CUDA FP32/FP16", "blocked: the named entry point does not exist; the tensor-level FP16 cast has no public surface yet"),
        Planned("DirectGpuTensorEngine.TensorTranspose", "DirectGpuTensorEngine.TensorTranspose -> CudaBackend.Transpose or CPU fallback", "public 2D tensor transpose", "logical 2D; canonical contiguous admitted view", "generic public; CUDA FP32", "routes through the ported Transpose whenever the tensor is contiguous and the extents are admitted"),
        Planned("DirectGpuTensorEngine.TensorPermute", "DirectGpuTensorEngine.TensorPermute -> CudaBackend.Permute or CPU fallback", "public axis permutation", "logical strided tensor; canonical contiguous admitted view", "generic public; CUDA FP32", "public-permute-routing; follows the CudaBackend.Permute lane"),
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

    private static DirectPtxLayoutCoverageCell ExistingBackend(
        string api, string existing, string semantics, string layout, string dtypes, string assignment) =>
        new(api, existing, semantics, layout, dtypes, assignment, DirectPtxLayoutCoverageStatus.ExistingBackend);

    private static DirectPtxLayoutCoverageCell Planned(
        string api, string existing, string semantics, string layout, string dtypes, string assignment) =>
        new(api, existing, semantics, layout, dtypes, assignment, DirectPtxLayoutCoverageStatus.PlannedDirectPtx);
}
