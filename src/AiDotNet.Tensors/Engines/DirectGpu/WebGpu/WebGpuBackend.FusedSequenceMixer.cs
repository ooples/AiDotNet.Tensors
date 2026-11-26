#if NET7_0_OR_GREATER
// Copyright (c) CrowdTrainer LLC. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

public sealed partial class WebGpuBackend : IFusedSequenceMixerOps
{
    void IFusedSequenceMixerOps.SplitHalfGivensRotationForward(IGpuBuffer input, IGpuBuffer angles, IGpuBuffer output, int rows, int halfDim)
        => PortableSequenceMixerOps.SplitHalfGivensRotationForward(this, input, angles, output, rows, halfDim);
    void IFusedSequenceMixerOps.SplitHalfGivensRotationBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer angles, IGpuBuffer gradInput, IGpuBuffer gradAngles, int rows, int halfDim)
        => PortableSequenceMixerOps.SplitHalfGivensRotationBackward(this, gradOutput, input, angles, gradInput, gradAngles, rows, halfDim);
    void IFusedSequenceMixerOps.DepthwiseCausalConv1d2TapForward(IGpuBuffer input, IGpuBuffer weight0, IGpuBuffer weight1, IGpuBuffer output, int batch, int seqLen, int channels)
        => PortableSequenceMixerOps.DepthwiseCausalConv1d2TapForward(this, input, weight0, weight1, output, batch, seqLen, channels);
    void IFusedSequenceMixerOps.DepthwiseCausalConv1d2TapBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer weight0, IGpuBuffer weight1, IGpuBuffer gradInput, IGpuBuffer gradWeight0, IGpuBuffer gradWeight1, int batch, int seqLen, int channels)
        => PortableSequenceMixerOps.DepthwiseCausalConv1d2TapBackward(this, gradOutput, input, weight0, weight1, gradInput, gradWeight0, gradWeight1, batch, seqLen, channels);
    void IFusedSequenceMixerOps.RopeL2NormalizeHeadsForward(IGpuBuffer input, IGpuBuffer cos, IGpuBuffer sin, IGpuBuffer output, int batch, int seqLen, int numHeads, int headDim, int startPosition, float epsilon, float postScale)
        => PortableSequenceMixerOps.RopeL2NormalizeHeadsForward(this, input, cos, sin, output, batch, seqLen, numHeads, headDim, startPosition, epsilon, postScale);
    void IFusedSequenceMixerOps.RopeL2NormalizeHeadsBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer cos, IGpuBuffer sin, IGpuBuffer gradInput, int batch, int seqLen, int numHeads, int headDim, int startPosition, float epsilon, float postScale)
        => PortableSequenceMixerOps.RopeL2NormalizeHeadsBackward(this, gradOutput, input, cos, sin, gradInput, batch, seqLen, numHeads, headDim, startPosition, epsilon, postScale);
    void IFusedSequenceMixerOps.RopeQuadraticOuterFeaturesForward(IGpuBuffer input, IGpuBuffer cos, IGpuBuffer sin, IGpuBuffer output, int batch, int seqLen, int numHeads, int featureDim, int outputHeadDim, int startPosition)
        => PortableSequenceMixerOps.RopeQuadraticOuterFeaturesForward(this, input, cos, sin, output, batch, seqLen, numHeads, featureDim, outputHeadDim, startPosition);
    void IFusedSequenceMixerOps.RopeQuadraticOuterFeaturesBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer cos, IGpuBuffer sin, IGpuBuffer gradInput, int batch, int seqLen, int numHeads, int featureDim, int outputHeadDim, int startPosition)
        => PortableSequenceMixerOps.RopeQuadraticOuterFeaturesBackward(this, gradOutput, input, cos, sin, gradInput, batch, seqLen, numHeads, featureDim, outputHeadDim, startPosition);
    void IFusedSequenceMixerOps.ProjectiveGatedResidualForward(IGpuBuffer baseHidden, IGpuBuffer proposal, IGpuBuffer gateLogit, IGpuBuffer output, int rows, int dim, float epsilon)
        => PortableSequenceMixerOps.ProjectiveGatedResidualForward(this, baseHidden, proposal, gateLogit, output, rows, dim, epsilon);
    void IFusedSequenceMixerOps.ProjectiveGatedResidualBackward(IGpuBuffer gradOutput, IGpuBuffer baseHidden, IGpuBuffer proposal, IGpuBuffer gateLogit, IGpuBuffer gradBaseHidden, IGpuBuffer gradProposal, IGpuBuffer gradGateLogit, int rows, int dim, float epsilon)
        => PortableSequenceMixerOps.ProjectiveGatedResidualBackward(this, gradOutput, baseHidden, proposal, gateLogit, gradBaseHidden, gradProposal, gradGateLogit, rows, dim, epsilon);
    void IFusedSequenceMixerOps.SpectralCirculantMatrixForward(IGpuBuffer diagScale, IGpuBuffer spectrumRe, IGpuBuffer spectrumIm, IGpuBuffer output, int batch, int dim)
        => PortableSequenceMixerOps.SpectralCirculantMatrixForward(this, diagScale, spectrumRe, spectrumIm, output, batch, dim);
    void IFusedSequenceMixerOps.SpectralCirculantMatrixBackward(IGpuBuffer gradOutput, IGpuBuffer diagScale, IGpuBuffer spectrumRe, IGpuBuffer spectrumIm, IGpuBuffer gradDiagScale, IGpuBuffer gradSpectrumRe, IGpuBuffer gradSpectrumIm, int batch, int dim)
        => PortableSequenceMixerOps.SpectralCirculantMatrixBackward(this, gradOutput, diagScale, spectrumRe, spectrumIm, gradDiagScale, gradSpectrumRe, gradSpectrumIm, batch, dim);
    void IFusedSequenceMixerOps.SigmoidTopKExpertMixForward(IGpuBuffer logits, IGpuBuffer expertTable, IGpuBuffer routingBias, IGpuBuffer output, IGpuBuffer savedInfo, int rows, int numExperts, int topK, int channels, float epsilon, float biasUpdateRate)
        => PortableSequenceMixerOps.SigmoidTopKExpertMixForward(this, logits, expertTable, routingBias, output, savedInfo, rows, numExperts, topK, channels, epsilon, biasUpdateRate);
    void IFusedSequenceMixerOps.SigmoidTopKExpertMixBackward(IGpuBuffer gradOutput, IGpuBuffer logits, IGpuBuffer expertTable, IGpuBuffer savedInfo, IGpuBuffer gradLogits, IGpuBuffer gradExpertTable, int rows, int numExperts, int topK, int channels, float epsilon)
        => PortableSequenceMixerOps.SigmoidTopKExpertMixBackward(this, gradOutput, logits, expertTable, savedInfo, gradLogits, gradExpertTable, rows, numExperts, topK, channels, epsilon);
    void IFusedSequenceMixerOps.DualModulatedSwiGluForward(IGpuBuffer gateUpProj, IGpuBuffer gateUpGainDelta, IGpuBuffer upBase, IGpuBuffer output, int rows, int hiddenDim)
        => PortableSequenceMixerOps.DualModulatedSwiGluForward(this, gateUpProj, gateUpGainDelta, upBase, output, rows, hiddenDim);
    void IFusedSequenceMixerOps.DualModulatedSwiGluBackward(IGpuBuffer gradOutput, IGpuBuffer gateUpProj, IGpuBuffer gateUpGainDelta, IGpuBuffer upBase, IGpuBuffer gradGateUpProj, IGpuBuffer gradGateUpGainDelta, IGpuBuffer gradUpBase, int rows, int hiddenDim)
        => PortableSequenceMixerOps.DualModulatedSwiGluBackward(this, gradOutput, gateUpProj, gateUpGainDelta, upBase, gradGateUpProj, gradGateUpGainDelta, gradUpBase, rows, hiddenDim);
    void IFusedSequenceMixerOps.BornRatioNormalizeForward(IGpuBuffer scanOutput, IGpuBuffer sinkLogAlphaOrDummy, IGpuBuffer output, int totalRows, int numHeads, int headDim, int scanDim, int hasSink, float epsilon)
        => PortableSequenceMixerOps.BornRatioNormalizeForward(this, scanOutput, sinkLogAlphaOrDummy, output, totalRows, numHeads, headDim, scanDim, hasSink, epsilon);
    void IFusedSequenceMixerOps.BornRatioNormalizeBackward(IGpuBuffer gradOutput, IGpuBuffer scanOutput, IGpuBuffer sinkLogAlphaOrDummy, IGpuBuffer gradScanOutput, IGpuBuffer gradSinkOrDummy, int totalRows, int numHeads, int headDim, int scanDim, int hasSink, float epsilon)
        => PortableSequenceMixerOps.BornRatioNormalizeBackward(this, gradOutput, scanOutput, sinkLogAlphaOrDummy, gradScanOutput, gradSinkOrDummy, totalRows, numHeads, headDim, scanDim, hasSink, epsilon);
    void IFusedSequenceMixerOps.BoundedSigmoidGateForward(IGpuBuffer logits, IGpuBuffer bias, IGpuBuffer output, int rows, int cols, float floor)
        => PortableSequenceMixerOps.BoundedSigmoidGateForward(this, logits, bias, output, rows, cols, floor);
    void IFusedSequenceMixerOps.BoundedSigmoidGateBackward(IGpuBuffer gradOutput, IGpuBuffer logits, IGpuBuffer bias, IGpuBuffer gradLogits, IGpuBuffer gradBias, int rows, int cols, float floor)
        => PortableSequenceMixerOps.BoundedSigmoidGateBackward(this, gradOutput, logits, bias, gradLogits, gradBias, rows, cols, floor);
}
#endif
