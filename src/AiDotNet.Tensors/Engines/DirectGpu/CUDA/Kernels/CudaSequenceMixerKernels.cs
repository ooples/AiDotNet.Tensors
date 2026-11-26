// Copyright (c) CrowdTrainer LLC. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels;

/// <summary>
/// Single-launch register/warp-shuffle forward and deterministic zero-atomic backward CUDA kernels
/// for <see cref="IFusedSequenceMixerOps"/>.
/// </summary>
internal static class CudaSequenceMixerKernels
{
    public const int MaxRopeL2HeadDim = 512;
    public const int MaxRopeQuadFeatureDim = 64;
    public const int MaxProjResDim = 1024;
    public const int MaxSpecCircDim = 1024;
    public const int MaxTopKExperts = 256;
    public const int MaxTopK = 16;
    public const int WarpBlockThreads = 128;
    public const int WarpsPerBlock = WarpBlockThreads / 32;
    public const int ElemBlockThreads = 256;

    public static string[] GetKernelNames() => new[]
    {
        "smx_cuda_givens_fwd",
        "smx_cuda_givens_bwd_dx",
        "smx_cuda_givens_bwd_dphi",
        "smx_cuda_conv2_fwd",
        "smx_cuda_conv2_bwd_dx",
        "smx_cuda_conv2_bwd_dw",
        "smx_cuda_ropel2_fwd",
        "smx_cuda_ropel2_bwd",
        "smx_cuda_ropequad_fwd",
        "smx_cuda_ropequad_bwd",
        "smx_cuda_projres_fwd",
        "smx_cuda_projres_bwd_rows",
        "smx_cuda_projres_bwd_dgate",
        "smx_cuda_speccirc_fwd",
        "smx_cuda_speccirc_bwd",
        "smx_cuda_sigtopk_mix_fwd",
        "smx_cuda_sigtopk_bias_upd",
        "smx_cuda_sigtopk_mix_bwd_dlog",
        "smx_cuda_sigtopk_mix_bwd_dtbl",
        "smx_cuda_dualswiglu_fwd",
        "smx_cuda_dualswiglu_bwd_elem",
        "smx_cuda_dualswiglu_bwd_dbase",
        "smx_cuda_bornratio_fwd",
        "smx_cuda_bornratio_bwd_dscan",
        "smx_cuda_bornratio_bwd_dsink",
        "smx_cuda_boundsig_fwd",
        "smx_cuda_boundsig_bwd_dlog",
        "smx_cuda_boundsig_bwd_dbias",
    };

    public static string GetSource() => KernelSourceResources.Read("CudaSequenceMixer.c");
}
