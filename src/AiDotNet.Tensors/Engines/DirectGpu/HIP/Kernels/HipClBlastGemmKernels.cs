// Copyright (c) AiDotNet. All rights reserved.
// Portions of the kernel sources are derived from the CLBlast project (Apache 2.0): https://github.com/CNugteren/CLBlast
using System;
using System.Text;
using System.Text.RegularExpressions;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL.Kernels;

namespace AiDotNet.Tensors.Engines.DirectGpu.HIP.Kernels;

/// <summary>
/// The CLBlast GEMM kernels OpenCL runs (XgemmDirect), compiled for HIP from the same OpenCL C source (#1266). CLBlast
/// was written to build as CUDA too (LOCAL_PTR is a macro "because CUDA doesn't need them"); a preamble maps the
/// remaining OpenCL built-ins to HIP and two definitions are rewritten so helper functions are device functions. Vector
/// widths are limited to 1, 2 and 4: HIP has no float8 / float16 vector types (stubs only keep their typedefs valid).
/// </summary>
internal static class HipClBlastGemmKernels
{
    private static readonly TimeSpan RegexTimeout = TimeSpan.FromSeconds(2);

    private const string Preamble = @"
// OpenCL C -> HIP for the CLBlast GEMM kernels.
#define __kernel extern ""C"" __global__
#define __global
#define __local __shared__
#define LOCAL_PTR
#define restrict __restrict__
#define CLK_LOCAL_MEM_FENCE 0
#define CLK_GLOBAL_MEM_FENCE 0
#define barrier(flags) __syncthreads()
#define get_local_id(d) ((int)((d) == 0 ? threadIdx.x : ((d) == 1 ? threadIdx.y : threadIdx.z)))
#define get_group_id(d) ((int)((d) == 0 ? blockIdx.x : ((d) == 1 ? blockIdx.y : blockIdx.z)))
#define get_local_size(d) ((int)((d) == 0 ? blockDim.x : ((d) == 1 ? blockDim.y : blockDim.z)))
#define get_num_groups(d) ((int)((d) == 0 ? gridDim.x : ((d) == 1 ? gridDim.y : gridDim.z)))
#define get_global_id(d) (get_group_id(d) * get_local_size(d) + get_local_id(d))
#define mad(a, b, c) fmaf((a), (b), (c))
// Only the typedefs of the 8- and 16-wide vectors appear outside the VW 8/16 branches, which HIP does not compile.
typedef struct { float s0, s1, s2, s3, s4, s5, s6, s7; } float8;
typedef struct { float s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, sA, sB, sC, sD, sE, sF; } float16;
";

    /// <summary>Whether a parameter set can be built for HIP (vector widths of 1, 2 or 4).</summary>
    internal static bool Supports(ClBlastXgemmDirectParameters p) =>
        p.Wgd > 0 && p.MdimCd > 0 && p.NdimCd > 0 && p.Vwm is 1 or 2 or 4 && p.Vwn is 1 or 2 or 4;

    internal static string BuildDirectSource(ClBlastXgemmDirectParameters parameters)
    {
        if (!Supports(parameters))
            throw new ArgumentException("HIP builds the CLBlast direct GEMM only with vector widths 1, 2 or 4.", nameof(parameters));
        return Translate(ClBlastXgemmDirectKernel.BuildSource(parameters));
    }

    internal static string[] DirectKernelNames() => ClBlastXgemmDirectKernel.GetKernelNames();

    /// <summary>Whether the indirect Xgemm configuration can be built for HIP.</summary>
    internal static bool Supports(GemmConfig config) =>
        config.KernelTemplate == GemmKernelTemplate.ClBlastBaselineK0 &&
        config.TileM > 0 && config.TileN > 0 && config.TileK > 0 && config.ThreadTileM > 0 && config.ThreadTileN > 0 &&
        config.VectorWidthM is 0 or 1 or 2 or 4 && config.VectorWidthN is 0 or 1 or 2 or 4;

    /// <summary>The indirect GEMM kernel (clblast_xgemm, GEMMK 0) that OpenCL runs for sizes past MinIndirectSize.</summary>
    internal static string BuildIndirectSource(GemmConfig config)
    {
        if (!Supports(config))
            throw new ArgumentException("HIP builds the CLBlast indirect GEMM only for GEMMK 0 with vector widths 1, 2 or 4.", nameof(config));
        return Translate(ClBlastXgemmKernel.BuildSource(config, 0));
    }

    /// <summary>The pad and transpose kernels the indirect GEMM packs its operands with (the non-Fast variants).</summary>
    internal static string BuildPackingSource(ClBlastPadParameters pad, ClBlastPadTransposeParameters padTranspose,
        ClBlastCopyParameters copy, ClBlastTransposeParameters transpose) =>
        Translate(ClBlastPackingKernels.BuildSource(pad, padTranspose, copy, transpose));

    internal static readonly string[] PackingKernelNames = { "CopyPadMatrix", "CopyMatrix", "TransposePadMatrix" };

    private static string Translate(string source)
    {
        string openCl = source.Replace("#define RELAX_WORKGROUP_SIZE 0", "#define RELAX_WORKGROUP_SIZE 1");
        // Device-side helpers: CLBlast defines INLINE_FUNC unguarded in its common header.
        openCl = Regex.Replace(openCl, @"(?m)^(\s*)#define INLINE_FUNC(\s+inline)?\s*$", "$1#define INLINE_FUNC __device__ inline",
            RegexOptions.None, RegexTimeout);
        // OpenCL extension pragmas mean nothing to hiprtc.
        openCl = Regex.Replace(openCl, @"(?m)^\s*#pragma OPENCL[^\n]*$", string.Empty, RegexOptions.None, RegexTimeout);
        return new StringBuilder(Preamble.Length + openCl.Length).Append(Preamble).Append(openCl).ToString();
    }
}