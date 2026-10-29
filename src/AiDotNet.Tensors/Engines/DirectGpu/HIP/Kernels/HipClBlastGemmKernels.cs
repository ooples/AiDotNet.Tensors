// Copyright (c) CrowdTrainer LLC. All rights reserved.
// Portions of the kernel sources are derived from the CLBlast project (Apache 2.0): https://github.com/CNugteren/CLBlast
using System;
using System.Text;
using System.Text.RegularExpressions;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL.Kernels;

namespace AiDotNet.Tensors.Engines.DirectGpu.HIP.Kernels;

/// <summary>
/// The CLBlast GEMM kernels OpenCL runs (the indirect Xgemm and its packing kernels), compiled for HIP from the same OpenCL C source (#1266). CLBlast
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

    /// <summary>
    /// Whether the indirect Xgemm configuration can be built for HIP: GEMMK 0, vector widths 1, 2 or 4, and CLBlast's own
    /// xgemm tuner constraints (src/tuning/kernels/xgemm.hpp). The cooperative-load ones (MDIMA, NDIMB against KWG and
    /// the tiles) are not in DynamicGemmKernel.ValidateConfig; a configuration that breaks them indexes outside its tiles.
    /// </summary>
    internal static bool Supports(GemmConfig config)
    {
        if (config.KernelTemplate != GemmKernelTemplate.ClBlastBaselineK0) return false;
        int mwg = config.TileM, nwg = config.TileN, kwg = config.TileK;
        int mdimc = config.ThreadTileM, ndimc = config.ThreadTileN;
        int mdima = config.MdimaSize > 0 ? config.MdimaSize : mdimc, ndimb = config.NdimbSize > 0 ? config.NdimbSize : ndimc;
        int vwm = config.VectorWidthM > 0 ? config.VectorWidthM : 1, vwn = config.VectorWidthN > 0 ? config.VectorWidthN : 1;
        int kwi = config.KUnroll > 0 ? config.KUnroll : 1;
        if (mwg <= 0 || nwg <= 0 || kwg <= 0 || mdimc <= 0 || ndimc <= 0 || mdima <= 0 || ndimb <= 0) return false;
        if (vwm is not (1 or 2 or 4) || vwn is not (1 or 2 or 4) || config.KReg > 1) return false;
        int threads = mdimc * ndimc;
        if (threads % mdima != 0 || threads % ndimb != 0) return false;
        return kwg % kwi == 0 &&
            mwg % (mdimc * vwm) == 0 && nwg % (ndimc * vwn) == 0 &&
            mwg % (mdima * vwm) == 0 && nwg % (ndimb * vwn) == 0 &&
            kwg % (threads / mdima) == 0 && kwg % (threads / ndimb) == 0;
    }
    /// <summary>The indirect GEMM kernel (clblast_xgemm, GEMMK 0) that OpenCL runs for sizes past MinIndirectSize.</summary>
    internal static string BuildIndirectSource(GemmConfig config, string kernelName = IndirectKernelName)
    {
        if (!Supports(config))
            throw new ArgumentException("HIP builds the CLBlast indirect GEMM only for GEMMK 0 with vector widths 1, 2 or 4.", nameof(config));
        // CLBlast names its entry point through "#define Xgemm clblast_xgemm"; a distinct name lets variants coexist.
        return Translate(ClBlastXgemmKernel.BuildSource(config, 0)
            .Replace("#define Xgemm clblast_xgemm", "#define Xgemm " + kernelName));
    }

    /// <summary>The pad and transpose kernels the indirect GEMM packs its operands with (the non-Fast variants).</summary>
    internal static string BuildPackingSource(ClBlastPadParameters pad, ClBlastPadTransposeParameters padTranspose,
        ClBlastCopyParameters copy, ClBlastTransposeParameters transpose) =>
        Translate(ClBlastPackingKernels.BuildSource(pad, padTranspose, copy, transpose));

    internal static readonly string[] PackingKernelNames = { "CopyPadMatrix", "CopyMatrix", "TransposePadMatrix" };

    internal const string IndirectKernelName = "clblast_xgemm";

    /// <summary>Whether <paramref name="name"/> belongs to the modules HipBackend builds on first use.</summary>
    internal static bool IsLazyKernel(string name) =>
        name == IndirectKernelName || Array.IndexOf(PackingKernelNames, name) >= 0;

    /// <summary>OpenCL C from the CLBlast builders, made compilable by hiprtc: the built-in preamble, relaxed work-group
    /// sizes, device-side helper functions, and no OpenCL pragmas.</summary>
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