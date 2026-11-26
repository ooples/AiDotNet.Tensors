// Copyright (c) CrowdTrainer LLC. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// Fused GPU sequence-mixer primitives:
/// <list type="bullet">
/// <item><description><see cref="SplitHalfGivensRotationForward"/> / <see cref="SplitHalfGivensRotationBackward"/>: split-half 2D Givens/phase rotation.</description></item>
/// <item><description><see cref="DepthwiseCausalConv1d2TapForward"/> / <see cref="DepthwiseCausalConv1d2TapBackward"/>: 2-tap depthwise causal 1D convolution.</description></item>
/// <item><description><see cref="RopeL2NormalizeHeadsForward"/> / <see cref="RopeL2NormalizeHeadsBackward"/>: interleaved RoPE followed by per-head L2 QK-normalization.</description></item>
/// <item><description><see cref="RopeQuadraticOuterFeaturesForward"/> / <see cref="RopeQuadraticOuterFeaturesBackward"/>: interleaved RoPE followed by 2nd-order outer-product feature map with optional zero-padding.</description></item>
/// <item><description><see cref="ProjectiveGatedResidualForward"/> / <see cref="ProjectiveGatedResidualBackward"/>: orthogonal-projection gated residual addition.</description></item>
/// </list>
/// </summary>
internal interface IFusedSequenceMixerOps
{
    void SplitHalfGivensRotationForward(
        IGpuBuffer input,
        IGpuBuffer angles,
        IGpuBuffer output,
        int rows,
        int halfDim);

    void SplitHalfGivensRotationBackward(
        IGpuBuffer gradOutput,
        IGpuBuffer input,
        IGpuBuffer angles,
        IGpuBuffer gradInput,
        IGpuBuffer gradAngles,
        int rows,
        int halfDim);

    void DepthwiseCausalConv1d2TapForward(
        IGpuBuffer input,
        IGpuBuffer weight0,
        IGpuBuffer weight1,
        IGpuBuffer output,
        int batch,
        int seqLen,
        int channels);

    void DepthwiseCausalConv1d2TapBackward(
        IGpuBuffer gradOutput,
        IGpuBuffer input,
        IGpuBuffer weight0,
        IGpuBuffer weight1,
        IGpuBuffer gradInput,
        IGpuBuffer gradWeight0,
        IGpuBuffer gradWeight1,
        int batch,
        int seqLen,
        int channels);

    void RopeL2NormalizeHeadsForward(
        IGpuBuffer input,
        IGpuBuffer cos,
        IGpuBuffer sin,
        IGpuBuffer output,
        int batch,
        int seqLen,
        int numHeads,
        int headDim,
        int startPosition,
        float epsilon,
        float postScale);

    void RopeL2NormalizeHeadsBackward(
        IGpuBuffer gradOutput,
        IGpuBuffer input,
        IGpuBuffer cos,
        IGpuBuffer sin,
        IGpuBuffer gradInput,
        int batch,
        int seqLen,
        int numHeads,
        int headDim,
        int startPosition,
        float epsilon,
        float postScale);

    void RopeQuadraticOuterFeaturesForward(
        IGpuBuffer input,
        IGpuBuffer cos,
        IGpuBuffer sin,
        IGpuBuffer output,
        int batch,
        int seqLen,
        int numHeads,
        int featureDim,
        int outputHeadDim,
        int startPosition);

    void RopeQuadraticOuterFeaturesBackward(
        IGpuBuffer gradOutput,
        IGpuBuffer input,
        IGpuBuffer cos,
        IGpuBuffer sin,
        IGpuBuffer gradInput,
        int batch,
        int seqLen,
        int numHeads,
        int featureDim,
        int outputHeadDim,
        int startPosition);

    void ProjectiveGatedResidualForward(
        IGpuBuffer baseHidden,
        IGpuBuffer proposal,
        IGpuBuffer gateLogit,
        IGpuBuffer output,
        int rows,
        int dim,
        float epsilon);

    void ProjectiveGatedResidualBackward(
        IGpuBuffer gradOutput,
        IGpuBuffer baseHidden,
        IGpuBuffer proposal,
        IGpuBuffer gateLogit,
        IGpuBuffer gradBaseHidden,
        IGpuBuffer gradProposal,
        IGpuBuffer gradGateLogit,
        int rows,
        int dim,
        float epsilon);

    void SpectralCirculantMatrixForward(
        IGpuBuffer diagScale,
        IGpuBuffer spectrumRe,
        IGpuBuffer spectrumIm,
        IGpuBuffer output,
        int batch,
        int dim);

    void SpectralCirculantMatrixBackward(
        IGpuBuffer gradOutput,
        IGpuBuffer diagScale,
        IGpuBuffer spectrumRe,
        IGpuBuffer spectrumIm,
        IGpuBuffer gradDiagScale,
        IGpuBuffer gradSpectrumRe,
        IGpuBuffer gradSpectrumIm,
        int batch,
        int dim);

    void SigmoidTopKExpertMixForward(
        IGpuBuffer logits,
        IGpuBuffer expertTable,
        IGpuBuffer routingBias,
        IGpuBuffer output,
        IGpuBuffer savedInfo,
        int rows,
        int numExperts,
        int topK,
        int channels,
        float epsilon,
        float biasUpdateRate);

    void SigmoidTopKExpertMixBackward(
        IGpuBuffer gradOutput,
        IGpuBuffer logits,
        IGpuBuffer expertTable,
        IGpuBuffer savedInfo,
        IGpuBuffer gradLogits,
        IGpuBuffer gradExpertTable,
        int rows,
        int numExperts,
        int topK,
        int channels,
        float epsilon);

    void DualModulatedSwiGluForward(
        IGpuBuffer gateUpProj,
        IGpuBuffer gateUpGainDelta,
        IGpuBuffer upBase,
        IGpuBuffer output,
        int rows,
        int hiddenDim);

    void DualModulatedSwiGluBackward(
        IGpuBuffer gradOutput,
        IGpuBuffer gateUpProj,
        IGpuBuffer gateUpGainDelta,
        IGpuBuffer upBase,
        IGpuBuffer gradGateUpProj,
        IGpuBuffer gradGateUpGainDelta,
        IGpuBuffer gradUpBase,
        int rows,
        int hiddenDim);

    void BornRatioNormalizeForward(
        IGpuBuffer scanOutput,
        IGpuBuffer sinkLogAlphaOrDummy,
        IGpuBuffer output,
        int totalRows,
        int numHeads,
        int headDim,
        int scanDim,
        int hasSink,
        float epsilon);

    void BornRatioNormalizeBackward(
        IGpuBuffer gradOutput,
        IGpuBuffer scanOutput,
        IGpuBuffer sinkLogAlphaOrDummy,
        IGpuBuffer gradScanOutput,
        IGpuBuffer gradSinkOrDummy,
        int totalRows,
        int numHeads,
        int headDim,
        int scanDim,
        int hasSink,
        float epsilon);

    void BoundedSigmoidGateForward(
        IGpuBuffer logits,
        IGpuBuffer bias,
        IGpuBuffer output,
        int rows,
        int cols,
        float floor);

    void BoundedSigmoidGateBackward(
        IGpuBuffer gradOutput,
        IGpuBuffer logits,
        IGpuBuffer bias,
        IGpuBuffer gradLogits,
        IGpuBuffer gradBias,
        int rows,
        int cols,
        float floor);
}
