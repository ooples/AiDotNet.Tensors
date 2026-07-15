#if !NETFRAMEWORK
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.OpParity;

/// <summary>
/// Cross-backend kernel-source coverage guard (#775). Only the OpenCL backend is runtime-validated
/// on this machine; CUDA/HIP/Metal/Vulkan/WebGPU are source-validated (they compile, but their
/// kernels are never executed here). A source-only backend can silently FAIL to mirror a kernel the
/// OpenCL reference has — nothing else in the suite catches that. This test reflects every backend's
/// static <c>GetKernelNames()</c> registries and flags kernels present in OpenCL but absent from a
/// mirror backend.
///
/// CUDA and HIP mirror OpenCL's string-named <c>GetKernelNames()</c> convention (identical snake_case
/// kernel names), so the set difference is meaningful and ASSERTED against a regression floor that
/// ratchets DOWN only: a newly-added OpenCL compute kernel with no CUDA/HIP twin raises the diff
/// above the floor and fails the build. The non-zero baselines cover kernels that are legitimately
/// OpenCL-specific (e.g. CLBlast packing) plus the current known mirror gaps — drive them down by
/// adding the kernel to the backend, never by raising the floor.
///
/// Metal/Vulkan/WebGPU register shaders through non-name-based mechanisms (Metal functions / SPIR-V /
/// WGSL pipelines), so a raw name diff is not a faithful coverage signal there; their enumerable
/// coverage is REPORTED for visibility but not asserted.
/// </summary>
public sealed class CrossBackendKernelCoverageTests
{
    // GENUINE functional-gap floor (#775). The RAW OpenCL-vs-CUDA/HIP name diff is ~154, but that is a
    // 15x over-count: the CUDA/HIP backends already run those ops on-device, just registered under a
    // different kernel name. The raw diff is reconciled by (a) OpenClOnlyByDesign (cuBLAS/CLBlast GEMM
    // plumbing + cudaMemset — never mirrored by design) and (b) a name normalizer + CoveredUnderDifferentName
    // allowlist (each entry a verified same-op rename or a helper sub-kernel of a resident op). What
    // survives reconciliation is the GENUINE gap: an OpenCL kernel whose OP has no on-device CUDA/HIP path
    // under ANY name. This floor ratchets DOWN only — drive it to 0 by porting the kernel to CUDA+HIP.
    private const int GenuineGapFloor = 0;

    // OpenCL-specific plumbing CUDA/HIP cover via cuBLAS / cudaMemset — mirrored by design, never ported.
    private static readonly HashSet<string> OpenClOnlyByDesign = new(StringComparer.Ordinal)
    {
        "CopyMatrix", "CopyMatrixFast", "CopyPadMatrix", "TransposeMatrix", "TransposeMatrixFast",
        "TransposePadMatrix", "XgemmDirectNN", "XgemmDirectNT", "XgemmDirectTN", "XgemmDirectTT",
        "copy_block_2d", "copy_rows", "copy_submatrix", "fill_buffer", "gemm_batched_persistent",
        "gemm_clblast_rdna1", "gemm_coalesced", "gemm_double_buffered", "gemm_fp16_backward",
        "gemm_fp16in_fp16out", "gemm_fp16in_fp32out", "gemm_kreg4", "gemm_low_register", "gemm_medium_tile",
        "gemm_prefetch", "gemm_small", "gemm_small_tile", "gemm_tiled_simple", "gemm_vectorized",
        "gemm_vectorized_tile", "gemm_wide_vec", "pad_copy", "pad_copy_from_column_major",
        "pad_copy_transpose", "sparse_gemm_2_4", "transpose2d",
    };

    // OpenCL kernels the normalizer cannot reconcile by name shape, but which are VERIFIED resident on
    // CUDA/HIP under a different name (same op) or are a helper sub-kernel of a resident op. Comment gives
    // the CUDA home. Adding an entry here is an assertion "this op already runs on-device on CUDA/HIP".
    private static readonly HashSet<string> CoveredUnderDifferentName = new(StringComparer.Ordinal)
    {
        "add_gelu", "add_relu", "add_sigmoid",                 // fused bias+activation epilogue (CudaBackend.cs)
        "clip_by_value", "where_select", "max_axis", "var_axis", // elementwise/axis-reduce (CudaBackend.cs)
        "mixed_precision_forward", "mixed_precision_backward",  // CudaBackend.MixedPrecisionConv.cs
        "global_maxpool2d_with_indices",                       // -> global_maxpool2d (+backward)
        "conv_transpose2d_backward_weights",                   // -> conv_transpose2d_backward_kernel
        "gru_accumulate_bias_gradients", "gru_accumulate_weight_gradients_hh",
        "gru_accumulate_weight_gradients_ih",                  // -> gru_accumulate_weight_gradients (fused hh/ih/bias)
        "lstm_accumulate_bias_gradients", "lstm_accumulate_weight_gradients_hh",
        "lstm_accumulate_weight_gradients_ih",                 // -> lstm_accumulate_weight_gradients
        "sparse_categorical_cross_entropy_gradient",
        "sparse_categorical_cross_entropy_loss",               // -> categorical_cross_entropy_* (int-label form)
        "weighted_cross_entropy_gradient", "weighted_cross_entropy_loss", // -> categorical_cross_entropy (class weights)
        "scatter_add_kernel_deterministic",                    // -> parity210_index_add_deterministic / resident scatter
        "softmax_fused", "softmax_exp_sub", "softmax_div_scalar", // helper sub-kernels of resident softmax
        "fft_scale", "reduce_partial_sums", "scale_add", "fused_mul_add",
        "accumulate_gradient_fp32",                            // helpers of resident fft/reduce/elementwise/optimizer
        "normalize_overlap_add", "bit_reverse_cols", "bit_reverse_rows", // helpers of resident istft/fft
        "cosine_similarity_gradient",                          // -> cosine_similarity (backward composes)
        "embedding_lookup",                                    // -> CUDA/HIP backend.Embedding launches embedding_forward
        "sddmm_collab",                                        // -> CUDA CsrSddmm resident via 'sddmm'; _collab is an
                                                               // OpenCL-specific collaborative-tiling perf variant
    };

    // OpenCL kernels registered in GetKernelNames() but launched from NOWHERE (referenced only in their own
    // *Kernels.cs) — dead/legacy code, not a residency gap. Either the op composes from resident primitives
    // or no IEngine op dispatches them at all. Porting them to CUDA would just add dead kernels. (A cleaner
    // future cleanup is to delete these from the OpenCL registry; until then they are reconciled here.)
    private static readonly HashSet<string> UnusedLegacyOpenClKernels = new(StringComparer.Ordinal)
    {
        "create_hann_window", "create_hamming_window",  // CreateWindow composes (TensorLinspace+cos+scalar ops); unused
        "dice_gradient", "jaccard_gradient",            // no DiceLoss/JaccardLoss IEngine op dispatches these (unused)
        "compute_sparsity_ratio",                       // referenced only in SparseGemmKernels.cs (unused)
    };

    /// <summary>Reduce a kernel name to a convention-independent token set so an OpenCL name and its
    /// CUDA/HIP twin (different prefix/suffix/word-order) normalize to the same value. Strips the
    /// parity210_/resident_/native_/fused_ registration prefixes and _kernel/_forward/... suffixes,
    /// canonicalizes gradient->backward and 2_4->2x4 and fp16/fp32->f16/f32, and drops filler tokens that
    /// vary between conventions (loss, local, batched, ...). Deliberately lossy: it powers coverage
    /// reconciliation, not dispatch.</summary>
    private static SortedSet<string> Normalize(string s)
    {
        foreach (var p in new[] { "parity210_", "resident_", "native_", "fused_" })
            if (s.StartsWith(p, StringComparison.Ordinal)) { s = s.Substring(p.Length); break; }
        foreach (var suf in new[] { "_kernel", "_forward", "_native", "_counts", "_f", "_l2", "_vec2" })
            while (s.EndsWith(suf, StringComparison.Ordinal)) s = s.Substring(0, s.Length - suf.Length);
        s = s.Replace("_gradient", "_backward").Replace("2_4", "2x4");
        var drop = new HashSet<string>(StringComparer.Ordinal)
            { "loss", "values", "local", "from", "squared", "sum", "legacy", "batched", "scores" };
        var toks = new SortedSet<string>(StringComparer.Ordinal);
        foreach (var t in s.Split('_'))
        {
            if (t.Length == 0 || drop.Contains(t)) continue;
            toks.Add(t.Replace("fp16", "f16").Replace("fp32", "f32"));
        }
        return toks;
    }

    [Fact]
    public void MirrorBackends_CoverEveryOpenClComputeKernel()
    {
        var ocl = KernelNames("OpenCL");
        var cuda = KernelNames("CUDA");
        var hip = KernelNames("HIP");
        var metal = KernelNames("Metal");
        var vulkan = KernelNames("Vulkan");
        var webgpu = KernelNames("WebGpu");

        var cudaRawMissing = ocl.Except(cuda).OrderBy(x => x, StringComparer.Ordinal).ToArray();
        var hipRawMissing = ocl.Except(hip).OrderBy(x => x, StringComparer.Ordinal).ToArray();

        // Normalized token-sets of every CUDA+HIP registered kernel — a name is "covered by shape" if its
        // normalized form equals, or is a subset of, some registry kernel's normalized form.
        var mirrorNorm = cuda.Concat(hip).Select(Normalize).ToArray();
        bool CoveredByShape(string name)
        {
            var nm = Normalize(name);
            return mirrorNorm.Any(k => k.SetEquals(nm) || nm.IsSubsetOf(k));
        }
        bool Reconciled(string name) =>
            OpenClOnlyByDesign.Contains(name) || CoveredUnderDifferentName.Contains(name)
            || UnusedLegacyOpenClKernels.Contains(name) || CoveredByShape(name);

        // Genuine gaps = OpenCL kernels present in neither CUDA nor HIP whose op has no on-device twin.
        var genuine = ocl.Except(cuda).Union(ocl.Except(hip))
            .Where(n => !Reconciled(n)).OrderBy(x => x, StringComparer.Ordinal).ToArray();

        var sb = new StringBuilder();
        sb.AppendLine("# Cross-backend kernel-name coverage vs OpenCL (#775)");
        sb.AppendLine($"OpenCL kernels: {ocl.Count}");
        sb.AppendLine($"CUDA: {cuda.Count}  HIP: {hip.Count}  Metal: {metal.Count}  " +
            $"Vulkan: {vulkan.Count}  WebGpu: {webgpu.Count}");
        sb.AppendLine();
        sb.AppendLine($"Raw name diff (SUPERSET — reconciled below): CUDA {cudaRawMissing.Length}, HIP {hipRawMissing.Length}");
        sb.AppendLine($"  by-design (cuBLAS/CLBlast/memset): {OpenClOnlyByDesign.Count}");
        sb.AppendLine($"  covered under a different name / helper of a resident op: reconciled");
        sb.AppendLine();
        sb.AppendLine($"## GENUINE gaps: {genuine.Length} (floor {GenuineGapFloor}) — op has NO on-device CUDA/HIP path");
        foreach (var n in genuine) sb.AppendLine($"  [ ] {n}");
        sb.AppendLine();
        sb.AppendLine("## Raw CUDA-missing (visibility only, not asserted)");
        foreach (var n in cudaRawMissing) sb.AppendLine($"  [{(Reconciled(n) ? "x" : " ")}] {n}");
        sb.AppendLine();
        sb.AppendLine("Reported-only backends (non-name-based shader registration — not asserted):");
        sb.AppendLine($"  Metal enumerable kernels: {metal.Count}");
        sb.AppendLine($"  Vulkan enumerable kernels: {vulkan.Count}");
        sb.AppendLine($"  WebGpu enumerable kernels: {webgpu.Count}");
        WriteRaw("gpu-cross-backend-coverage.md", sb.ToString());

        Assert.True(genuine.Length <= GenuineGapFloor,
            $"{genuine.Length} OpenCL kernels have no on-device CUDA/HIP twin under any name (floor " +
            $"{GenuineGapFloor}). Port the kernel to CUDA+HIP (do not raise the floor); if it is actually " +
            $"covered, add it to CoveredUnderDifferentName with the CUDA home. See " +
            $"gpu-cross-backend-coverage.md. Gaps: {string.Join(", ", genuine)}");
    }

    /// <summary>Union of every <c>public static string[] GetKernelNames()</c> registry declared under
    /// the <c>.DirectGpu.&lt;segment&gt;</c> namespace of the product assembly.</summary>
    private static SortedSet<string> KernelNames(string backendSegment)
    {
        var set = new SortedSet<string>(StringComparer.Ordinal);
        var asm = typeof(AiDotNet.Tensors.Engines.DirectGpu.OpenCL.OpenClBackend).Assembly;
        string marker = $".DirectGpu.{backendSegment}";
        foreach (var t in asm.GetTypes())
        {
            var ns = t.Namespace;
            if (ns is null || ns.IndexOf(marker, StringComparison.Ordinal) < 0) continue;
            var m = t.GetMethod("GetKernelNames", BindingFlags.Public | BindingFlags.Static,
                binder: null, types: Type.EmptyTypes, modifiers: null);
            if (m is null || m.ReturnType != typeof(string[])) continue;
            try
            {
                if (m.Invoke(null, null) is string[] arr)
                    foreach (var n in arr)
                        if (!string.IsNullOrWhiteSpace(n)) set.Add(n);
            }
            catch { /* a registry that needs a device/context — skip, best effort */ }
        }
        return set;
    }

    private static void WriteRaw(string file, string content)
    {
        try
        {
            var dir = Environment.GetEnvironmentVariable("AIDOTNET_OPPARITY_REPORT_DIR")
                      ?? Path.Combine(Path.GetTempPath(), "aidotnet-opparity");
            Directory.CreateDirectory(dir);
            File.WriteAllText(Path.Combine(dir, file), content);
        }
        catch { /* best-effort */ }
    }
}
#endif
