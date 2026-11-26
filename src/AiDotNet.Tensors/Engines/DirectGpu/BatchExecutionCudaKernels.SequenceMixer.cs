// Copyright (c) CrowdTrainer LLC. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu;

internal static partial class BatchExecutionCudaKernels
{
    public const string SmxGivensForward = "smx_p_givens_fwd",
        SmxGivensBackwardDx = "smx_p_givens_bwd_dx",
        SmxGivensBackwardDphi = "smx_p_givens_bwd_dphi",
        SmxConv2TapForward = "smx_p_conv2_fwd",
        SmxConv2TapBackwardDx = "smx_p_conv2_bwd_dx",
        SmxConv2TapBackwardDw = "smx_p_conv2_bwd_dw",
        SmxRopeL2Forward = "smx_p_ropel2_fwd",
        SmxRopeL2Backward = "smx_p_ropel2_bwd",
        SmxRopeQuadForward = "smx_p_ropequad_fwd",
        SmxRopeQuadBackward = "smx_p_ropequad_bwd",
        SmxProjResForward = "smx_p_projres_fwd",
        SmxProjResBackwardRows = "smx_p_projres_bwd_rows",
        SmxProjResBackwardDgate = "smx_p_projres_bwd_dgate",
        SmxSpecCircH = "smx_p_speccirc_h",
        SmxSpecCircW = "smx_p_speccirc_w",
        SmxSpecCircBackwardDhDd = "smx_p_speccirc_bwd_dhdd",
        SmxSpecCircBackwardDspec = "smx_p_speccirc_bwd_dspec",
        SmxSigTopKMixForward = "smx_p_sigtopk_mix_fwd",
        SmxSigTopKBiasUpdate = "smx_p_sigtopk_bias_upd",
        SmxSigTopKMixBackwardDlogits = "smx_p_sigtopk_mix_bwd_dlog",
        SmxSigTopKMixBackwardDtable = "smx_p_sigtopk_mix_bwd_dtbl",
        SmxDualSwiGluForward = "smx_p_dualswiglu_fwd",
        SmxDualSwiGluBackwardElem = "smx_p_dualswiglu_bwd_elem",
        SmxDualSwiGluBackwardDbase = "smx_p_dualswiglu_bwd_dbase",
        SmxBornRatioForward = "smx_p_bornratio_fwd",
        SmxBornRatioBackwardDscan = "smx_p_bornratio_bwd_dscan",
        SmxBornRatioBackwardDsink = "smx_p_bornratio_bwd_dsink",
        SmxBoundSigForward = "smx_p_boundsig_fwd",
        SmxBoundSigBackwardDlog = "smx_p_boundsig_bwd_dlog",
        SmxBoundSigBackwardDbias = "smx_p_boundsig_bwd_dbias";

    internal static string SequenceMixerSource => KernelSourceResources.Read("SequenceMixerBatch.c");
}
