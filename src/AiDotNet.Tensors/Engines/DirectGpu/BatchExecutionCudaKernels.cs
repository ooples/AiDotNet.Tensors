// Copyright (c) CrowdTrainer LLC. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// CUDA C source for <see cref="IGpuBatchExecution"/>, compiled at runtime by both NVRTC (CUDA) and hiprtc (HIP)
/// (#1142). Related operations share one kernel with a <c>mode</c> argument so each backend language ports the same
/// small set. Semantics are the parity test's: whole-tensor reductions overwrite <c>output[0]</c> and return the
/// norm or the full log-sum-exp; the crop backward writes zero outside the window; sparsemax has no row-length cap;
/// the random kernels use the shared counter hash so a seed gives the same values on every backend.
/// </summary>
internal static partial class BatchExecutionCudaKernels
{
    public const int FullReduceBlockSize = 256;

    // Mode numbering shared with every backend's launcher.
    public const int FullSumOfSquares = 0, FullNormL2 = 1, FullProduct = 2, FullMaxMagnitude = 3, FullMinMagnitude = 4, FullLogSumExp = 5;
    public const int RowVariance = 0, RowStd = 1, RowProduct = 2, RowNorm = 3, RowLogSumExp = 4, RowLogVariance = 5;
    public const int MapCumSum = 0, MapNormalizeL2 = 1, MapLogSoftmax = 2, MapTaylorSoftmax = 3, MapSphericalSoftmax = 4, MapSparsemax = 5;
    public const int ScalarMinusTensor = 0, DivideByScalar = 1;
    public const int UnaryRsqrt = 0, UnaryReluDerivative = 1, UnarySigmoidDerivative = 2, UnaryTanhDerivative = 3;
    public const int BackwardSum = 0, BackwardMax = 1, BackwardVariance = 2, BackwardLogVariance = 3;
    public const int BroadcastAdd = 0, BroadcastMul = 1;
    public const int GateGlu = 0, GateGeGlu = 1, GateReGlu = 2, GateSwiGlu = 3;

    public static string GetSource() => CoreSource + SolverSource + AttentionSource + LstmSource + GruSource + DecodeLinearSource + DirectConvSource + WeightNormSource + RfftMagnitudeSource + ChannelBiasLeakySource + GroupedConv2DSource + ChannelBiasResidualSource;

    private static string CoreSource => KernelSourceResources.Load("BatchExecution.core.c");

    public static string[] GetKernelNames() => new[]
    {
        ChannelBiasLeaky, ChannelBiasResidual,
        GroupedConv2D, GroupedConv2DInputGrad, GroupedConv2DKernelGrad,
        RfftMagnitude, RfftMagnitudeSeed, RfftRealGradient,
        WeightNorm, WeightNormStats, WeightNormDirectionGrad, WeightNormGainGrad,
        "bx_full_reduce", "bx_row_reduce", "bx_row_map", "bx_scalar", "bx_unary", "bx_sincos", "bx_reduce_backward",
        "bx_broadcast_first", "bx_gated_backward", "bx_concat_axis", "bx_slice_last_axis", "bx_set_slice_last_axis",
        "bx_slice_axis", "bx_set_slice_axis", "bx_stack2", "bx_window_2d", "bx_tile_last_axis", "bx_repeat_elements",
        "bx_linspace", "bx_diag", "bx_extract_diag", "bx_triangular_mask", "bx_index_select", "bx_row_loss",
        "bx_dropout_mask", "bx_gaussian_noise", "bx_gumbel_softmax", "bx_batch_outer_product", "bx_meshgrid2d", "bx_fft_chirp", "bx_fft_bluestein_pre",
        "bx_fft_cmul", "bx_fft_bluestein_post", "bx_fft_to_storage", "bx_fft_from_storage", "bx_fft_bit_reverse", "bx_fft_butterfly",
        LaSymmetrize, LaLuSolve, LaLdlFactor, LaLdlSolve, LaSolveVector, LaTriangularSolveVector, LaCholeskyBackwardLower,
        LaSolveBackwardVector, LaSvdReduced, LaQrReduced, FaLse, FaOut, FaDelta, FaDq, FaDk, FaDv,
        StridedViewCopy, LstmGates, LstmCell, LstmCellBackward, LstmHiddenBackward, LstmInputGrad, LstmWeightIhGrad, LstmWeightHhGrad, LstmBiasGrad, LstmGateCell, LstmGateCellBackward, XLstmTrainingForward, XLstmTrainingBackward,
        GruProjection, GruCell, GruCellBackward, GruHiddenBackward, GruInputGrad, GruWeightIhGrad, GruWeightHhGrad, GruBiasGrad,
        GruRbResetHidden, GruRbCandidateProjection, GruRbCell, GruRbCellBackward, GruRbResetBackward, GruRbHiddenBackward,
        GruRbWeightHhGrad,
        DlLinearGelu, DlLinearGeluFp16, DlW8A8LinearGelu, DlW8A8Epilogue, DcConv2dDirectFp16Hw
    };
}
