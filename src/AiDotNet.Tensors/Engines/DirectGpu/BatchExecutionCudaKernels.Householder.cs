// Copyright (c) CrowdTrainer LLC. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu;

internal static partial class BatchExecutionCudaKernels
{
    public const string HouseholderNorm = "hh_p_norm",
        HouseholderForward = "hh_p_fwd",
        HouseholderBackwardInput = "hh_p_bwd_dx",
        HouseholderBackwardReflectorsRaw = "hh_p_bwd_du_raw",
        HouseholderBackwardReflectorsNorm = "hh_p_bwd_du_norm";

    internal static string HouseholderSource => KernelSourceResources.Read("HouseholderBatch.c");
}
