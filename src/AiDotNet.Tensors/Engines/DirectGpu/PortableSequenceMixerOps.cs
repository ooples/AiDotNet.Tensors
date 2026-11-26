// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;
using K = AiDotNet.Tensors.Engines.DirectGpu.BatchExecutionCudaKernels;

namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// Portable device implementation of <see cref="IFusedSequenceMixerOps"/> over the shared <c>smx_p_*</c> batch kernels,
/// executed via <see cref="IFusedLstmSequenceTraining.StatefulLstmLauncher"/> on HIP, Metal, OpenCL, Vulkan, WebGPU,
/// and as CUDA's fallback for oversized shapes.
/// </summary>
internal static class PortableSequenceMixerOps
{
    public static void SplitHalfGivensRotationForward(
        IDirectGpuBackend backend,
        IGpuBuffer input,
        IGpuBuffer angles,
        IGpuBuffer output,
        int rows,
        int halfDim)
    {
        if (rows < 0) throw new ArgumentOutOfRangeException(nameof(rows));
        if (halfDim <= 0) throw new ArgumentOutOfRangeException(nameof(halfDim));
        int totalHalf = checked(rows * halfDim);
        if (totalHalf == 0) return;
        var launch = ((IFusedLstmSequenceTraining)backend).StatefulLstmLauncher;
        launch(K.SmxGivensForward, totalHalf, input, angles, output, totalHalf, halfDim);
    }

    public static void SplitHalfGivensRotationBackward(
        IDirectGpuBackend backend,
        IGpuBuffer gradOutput,
        IGpuBuffer input,
        IGpuBuffer angles,
        IGpuBuffer gradInput,
        IGpuBuffer gradAngles,
        int rows,
        int halfDim)
    {
        if (rows < 0) throw new ArgumentOutOfRangeException(nameof(rows));
        if (halfDim <= 0) throw new ArgumentOutOfRangeException(nameof(halfDim));
        int totalHalf = checked(rows * halfDim);
        if (totalHalf == 0)
        {
            backend.Fill(gradAngles, 0f, halfDim);
            return;
        }
        var launch = ((IFusedLstmSequenceTraining)backend).StatefulLstmLauncher;
        launch(K.SmxGivensBackwardDx, totalHalf, gradOutput, angles, gradInput, totalHalf, halfDim);
        launch(K.SmxGivensBackwardDphi, halfDim, gradOutput, input, angles, gradAngles, rows, halfDim);
    }

    public static void DepthwiseCausalConv1d2TapForward(
        IDirectGpuBackend backend,
        IGpuBuffer input,
        IGpuBuffer weight0,
        IGpuBuffer weight1,
        IGpuBuffer output,
        int batch,
        int seqLen,
        int channels)
    {
        if (batch < 0) throw new ArgumentOutOfRangeException(nameof(batch));
        if (seqLen < 0) throw new ArgumentOutOfRangeException(nameof(seqLen));
        if (channels <= 0) throw new ArgumentOutOfRangeException(nameof(channels));
        int total = checked(batch * seqLen * channels);
        if (total == 0) return;
        var launch = ((IFusedLstmSequenceTraining)backend).StatefulLstmLauncher;
        launch(K.SmxConv2TapForward, total, input, weight0, weight1, output, total, seqLen, channels);
    }

    public static void DepthwiseCausalConv1d2TapBackward(
        IDirectGpuBackend backend,
        IGpuBuffer gradOutput,
        IGpuBuffer input,
        IGpuBuffer weight0,
        IGpuBuffer weight1,
        IGpuBuffer gradInput,
        IGpuBuffer gradWeight0,
        IGpuBuffer gradWeight1,
        int batch,
        int seqLen,
        int channels)
    {
        if (batch < 0) throw new ArgumentOutOfRangeException(nameof(batch));
        if (seqLen < 0) throw new ArgumentOutOfRangeException(nameof(seqLen));
        if (channels <= 0) throw new ArgumentOutOfRangeException(nameof(channels));
        int total = checked(batch * seqLen * channels);
        if (total == 0)
        {
            backend.Fill(gradWeight0, 0f, channels);
            backend.Fill(gradWeight1, 0f, channels);
            return;
        }
        var launch = ((IFusedLstmSequenceTraining)backend).StatefulLstmLauncher;
        launch(K.SmxConv2TapBackwardDx, total, gradOutput, weight0, weight1, gradInput, total, seqLen, channels);
        launch(K.SmxConv2TapBackwardDw, channels, gradOutput, input, gradWeight0, gradWeight1, batch, seqLen, channels);
    }

    public static void RopeL2NormalizeHeadsForward(
        IDirectGpuBackend backend,
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
        float postScale)
    {
        int totalHeads = checked(batch * seqLen * numHeads);
        if (totalHeads == 0) return;
        var launch = ((IFusedLstmSequenceTraining)backend).StatefulLstmLauncher;
        int epsBits = BitConverter.SingleToInt32Bits(epsilon);
        int scaleBits = BitConverter.SingleToInt32Bits(postScale);
        launch(K.SmxRopeL2Forward, totalHeads, input, cos, sin, output, totalHeads, seqLen, numHeads, headDim, startPosition, epsBits, scaleBits);
    }

    public static void RopeL2NormalizeHeadsBackward(
        IDirectGpuBackend backend,
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
        float postScale)
    {
        int totalHeads = checked(batch * seqLen * numHeads);
        if (totalHeads == 0) return;
        var launch = ((IFusedLstmSequenceTraining)backend).StatefulLstmLauncher;
        int epsBits = BitConverter.SingleToInt32Bits(epsilon);
        int scaleBits = BitConverter.SingleToInt32Bits(postScale);
        launch(K.SmxRopeL2Backward, totalHeads, gradOutput, input, cos, sin, gradInput, totalHeads, seqLen, numHeads, headDim, startPosition, epsBits, scaleBits);
    }

    public static void RopeQuadraticOuterFeaturesForward(
        IDirectGpuBackend backend,
        IGpuBuffer input,
        IGpuBuffer cos,
        IGpuBuffer sin,
        IGpuBuffer output,
        int batch,
        int seqLen,
        int numHeads,
        int featureDim,
        int outputHeadDim,
        int startPosition)
    {
        int totalHeads = checked(batch * seqLen * numHeads);
        if (totalHeads == 0) return;
        var launch = ((IFusedLstmSequenceTraining)backend).StatefulLstmLauncher;
        launch(K.SmxRopeQuadForward, totalHeads, input, cos, sin, output, totalHeads, seqLen, numHeads, featureDim, outputHeadDim, startPosition);
    }

    public static void RopeQuadraticOuterFeaturesBackward(
        IDirectGpuBackend backend,
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
        int startPosition)
    {
        int totalHeads = checked(batch * seqLen * numHeads);
        if (totalHeads == 0) return;
        var launch = ((IFusedLstmSequenceTraining)backend).StatefulLstmLauncher;
        launch(K.SmxRopeQuadBackward, totalHeads, gradOutput, input, cos, sin, gradInput, totalHeads, seqLen, numHeads, featureDim, outputHeadDim, startPosition);
    }

    public static void ProjectiveGatedResidualForward(
        IDirectGpuBackend backend,
        IGpuBuffer baseHidden,
        IGpuBuffer proposal,
        IGpuBuffer gateLogit,
        IGpuBuffer output,
        int rows,
        int dim,
        float epsilon)
    {
        if (rows < 0) throw new ArgumentOutOfRangeException(nameof(rows));
        if (dim <= 0) throw new ArgumentOutOfRangeException(nameof(dim));
        if (rows == 0) return;
        var launch = ((IFusedLstmSequenceTraining)backend).StatefulLstmLauncher;
        int epsBits = BitConverter.SingleToInt32Bits(epsilon);
        launch(K.SmxProjResForward, rows, baseHidden, proposal, gateLogit, output, rows, dim, epsBits);
    }

    public static void ProjectiveGatedResidualBackward(
        IDirectGpuBackend backend,
        IGpuBuffer gradOutput,
        IGpuBuffer baseHidden,
        IGpuBuffer proposal,
        IGpuBuffer gateLogit,
        IGpuBuffer gradBaseHidden,
        IGpuBuffer gradProposal,
        IGpuBuffer gradGateLogit,
        int rows,
        int dim,
        float epsilon)
    {
        if (rows < 0) throw new ArgumentOutOfRangeException(nameof(rows));
        if (dim <= 0) throw new ArgumentOutOfRangeException(nameof(dim));
        if (rows == 0)
        {
            backend.Fill(gradGateLogit, 0f, 1);
            return;
        }
        var launch = ((IFusedLstmSequenceTraining)backend).StatefulLstmLauncher;
        int epsBits = BitConverter.SingleToInt32Bits(epsilon);
        using var rowTerms = backend.AllocateBuffer(rows);
        launch(K.SmxProjResBackwardRows, rows, gradOutput, baseHidden, proposal, gateLogit, gradBaseHidden, gradProposal, rowTerms, rows, dim, epsBits);
        launch(K.SmxProjResBackwardDgate, 1, rowTerms, gateLogit, gradGateLogit, rows);
    }

    private static void ReleaseScratch(IDirectGpuBackend backend, IGpuBuffer buffer)
    {
        if (backend is IQueuedBufferRelease queued)
            queued.ReleaseAfterQueuedWork(buffer);
        else
            buffer.Dispose();
    }

    public static void SpectralCirculantMatrixForward(
        IDirectGpuBackend backend,
        IGpuBuffer diagScale,
        IGpuBuffer spectrumRe,
        IGpuBuffer spectrumIm,
        IGpuBuffer output,
        int batch,
        int dim)
    {
        if (batch < 0) throw new ArgumentOutOfRangeException(nameof(batch));
        if (dim <= 0 || (dim & 1) != 0) throw new ArgumentOutOfRangeException(nameof(dim));
        if (batch == 0) return;
        var launch = ((IFusedLstmSequenceTraining)backend).StatefulLstmLauncher;
        var hTime = backend.AllocateBuffer(checked(batch * dim));
        try
        {
            launch(K.SmxSpecCircH, batch * dim, spectrumRe, spectrumIm, hTime, batch, dim);
            launch(K.SmxSpecCircW, checked(batch * dim * dim), diagScale, hTime, output, batch, dim);
        }
        finally
        {
            ReleaseScratch(backend, hTime);
        }
    }

    public static void SpectralCirculantMatrixBackward(
        IDirectGpuBackend backend,
        IGpuBuffer gradOutput,
        IGpuBuffer diagScale,
        IGpuBuffer spectrumRe,
        IGpuBuffer spectrumIm,
        IGpuBuffer gradDiagScale,
        IGpuBuffer gradSpectrumRe,
        IGpuBuffer gradSpectrumIm,
        int batch,
        int dim)
    {
        if (batch < 0) throw new ArgumentOutOfRangeException(nameof(batch));
        if (dim <= 0 || (dim & 1) != 0) throw new ArgumentOutOfRangeException(nameof(dim));
        if (batch == 0) return;
        var launch = ((IFusedLstmSequenceTraining)backend).StatefulLstmLauncher;
        int nf = dim / 2 + 1;
        var hTime = backend.AllocateBuffer(checked(batch * dim));
        var dhTime = backend.AllocateBuffer(checked(batch * dim));
        try
        {
            launch(K.SmxSpecCircH, batch * dim, spectrumRe, spectrumIm, hTime, batch, dim);
            launch(K.SmxSpecCircBackwardDhDd, batch * dim, gradOutput, diagScale, hTime, gradDiagScale, dhTime, batch, dim);
            launch(K.SmxSpecCircBackwardDspec, checked(batch * nf), dhTime, gradSpectrumRe, gradSpectrumIm, batch, dim);
        }
        finally
        {
            ReleaseScratch(backend, hTime);
            ReleaseScratch(backend, dhTime);
        }
    }

    public static void SigmoidTopKExpertMixForward(
        IDirectGpuBackend backend,
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
        float biasUpdateRate)
    {
        if (rows < 0) throw new ArgumentOutOfRangeException(nameof(rows));
        if (numExperts <= 0) throw new ArgumentOutOfRangeException(nameof(numExperts));
        if (topK <= 0 || topK > numExperts) throw new ArgumentOutOfRangeException(nameof(topK));
        if (channels <= 0) throw new ArgumentOutOfRangeException(nameof(channels));
        if (rows == 0) return;
        var launch = ((IFusedLstmSequenceTraining)backend).StatefulLstmLauncher;
        int epsBits = BitConverter.SingleToInt32Bits(epsilon);
        launch(K.SmxSigTopKMixForward, rows, logits, expertTable, routingBias, output, savedInfo, rows, numExperts, topK, channels, epsBits);
        if (biasUpdateRate != 0f)
        {
            int rateBits = BitConverter.SingleToInt32Bits(biasUpdateRate);
            launch(K.SmxSigTopKBiasUpdate, numExperts, savedInfo, routingBias, rows, numExperts, topK, rateBits);
        }
    }

    public static void SigmoidTopKExpertMixBackward(
        IDirectGpuBackend backend,
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
        float epsilon)
    {
        if (rows < 0) throw new ArgumentOutOfRangeException(nameof(rows));
        if (numExperts <= 0) throw new ArgumentOutOfRangeException(nameof(numExperts));
        if (topK <= 0 || topK > numExperts) throw new ArgumentOutOfRangeException(nameof(topK));
        if (channels <= 0) throw new ArgumentOutOfRangeException(nameof(channels));
        if (rows == 0)
        {
            backend.Fill(gradExpertTable, 0f, checked(numExperts * channels));
            return;
        }
        var launch = ((IFusedLstmSequenceTraining)backend).StatefulLstmLauncher;
        int epsBits = BitConverter.SingleToInt32Bits(epsilon);
        launch(K.SmxSigTopKMixBackwardDlogits, rows, gradOutput, logits, expertTable, savedInfo, gradLogits, rows, numExperts, topK, channels, epsBits);
        launch(K.SmxSigTopKMixBackwardDtable, checked(numExperts * channels), gradOutput, savedInfo, gradExpertTable, rows, numExperts, topK, channels);
    }

    public static void DualModulatedSwiGluForward(
        IDirectGpuBackend backend,
        IGpuBuffer gateUpProj,
        IGpuBuffer gateUpGainDelta,
        IGpuBuffer upBase,
        IGpuBuffer output,
        int rows,
        int hiddenDim)
    {
        if (rows < 0) throw new ArgumentOutOfRangeException(nameof(rows));
        if (hiddenDim <= 0) throw new ArgumentOutOfRangeException(nameof(hiddenDim));
        int total = checked(rows * hiddenDim);
        if (total == 0) return;
        var launch = ((IFusedLstmSequenceTraining)backend).StatefulLstmLauncher;
        launch(K.SmxDualSwiGluForward, total, gateUpProj, gateUpGainDelta, upBase, output, rows, hiddenDim);
    }

    public static void DualModulatedSwiGluBackward(
        IDirectGpuBackend backend,
        IGpuBuffer gradOutput,
        IGpuBuffer gateUpProj,
        IGpuBuffer gateUpGainDelta,
        IGpuBuffer upBase,
        IGpuBuffer gradGateUpProj,
        IGpuBuffer gradGateUpGainDelta,
        IGpuBuffer gradUpBase,
        int rows,
        int hiddenDim)
    {
        if (rows < 0) throw new ArgumentOutOfRangeException(nameof(rows));
        if (hiddenDim <= 0) throw new ArgumentOutOfRangeException(nameof(hiddenDim));
        int total = checked(rows * hiddenDim);
        if (total == 0)
        {
            backend.Fill(gradUpBase, 0f, hiddenDim);
            return;
        }
        var launch = ((IFusedLstmSequenceTraining)backend).StatefulLstmLauncher;
        launch(K.SmxDualSwiGluBackwardElem, total, gradOutput, gateUpProj, gateUpGainDelta, upBase, gradGateUpProj, gradGateUpGainDelta, rows, hiddenDim);
        launch(K.SmxDualSwiGluBackwardDbase, hiddenDim, gradOutput, gateUpProj, gateUpGainDelta, gradUpBase, rows, hiddenDim);
    }

    public static void BornRatioNormalizeForward(
        IDirectGpuBackend backend,
        IGpuBuffer scanOutput,
        IGpuBuffer sinkLogAlphaOrDummy,
        IGpuBuffer output,
        int totalRows,
        int numHeads,
        int headDim,
        int scanDim,
        int hasSink,
        float epsilon)
    {
        if (totalRows < 0) throw new ArgumentOutOfRangeException(nameof(totalRows));
        if (numHeads <= 0) throw new ArgumentOutOfRangeException(nameof(numHeads));
        if (headDim <= 0 || scanDim < headDim + 1) throw new ArgumentOutOfRangeException(nameof(scanDim));
        int total = checked(totalRows * headDim);
        if (total == 0) return;
        var launch = ((IFusedLstmSequenceTraining)backend).StatefulLstmLauncher;
        int epsBits = BitConverter.SingleToInt32Bits(epsilon);
        launch(K.SmxBornRatioForward, total, scanOutput, sinkLogAlphaOrDummy, output, totalRows, numHeads, headDim, scanDim, hasSink, epsBits);
    }

    public static void BornRatioNormalizeBackward(
        IDirectGpuBackend backend,
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
        float epsilon)
    {
        if (totalRows < 0) throw new ArgumentOutOfRangeException(nameof(totalRows));
        if (numHeads <= 0) throw new ArgumentOutOfRangeException(nameof(numHeads));
        if (headDim <= 0 || scanDim < headDim + 1) throw new ArgumentOutOfRangeException(nameof(scanDim));
        if (totalRows == 0)
        {
            if (hasSink != 0) backend.Fill(gradSinkOrDummy, 0f, numHeads);
            return;
        }
        var launch = ((IFusedLstmSequenceTraining)backend).StatefulLstmLauncher;
        int epsBits = BitConverter.SingleToInt32Bits(epsilon);
        launch(K.SmxBornRatioBackwardDscan, totalRows, gradOutput, scanOutput, sinkLogAlphaOrDummy, gradScanOutput, totalRows, numHeads, headDim, scanDim, hasSink, epsBits);
        if (hasSink != 0)
            launch(K.SmxBornRatioBackwardDsink, numHeads, gradOutput, scanOutput, sinkLogAlphaOrDummy, gradSinkOrDummy, totalRows, numHeads, headDim, scanDim, epsBits);
    }

    public static void BoundedSigmoidGateForward(
        IDirectGpuBackend backend,
        IGpuBuffer logits,
        IGpuBuffer bias,
        IGpuBuffer output,
        int rows,
        int cols,
        float floor)
    {
        if (rows < 0) throw new ArgumentOutOfRangeException(nameof(rows));
        if (cols <= 0) throw new ArgumentOutOfRangeException(nameof(cols));
        int total = checked(rows * cols);
        if (total == 0) return;
        var launch = ((IFusedLstmSequenceTraining)backend).StatefulLstmLauncher;
        int floorBits = BitConverter.SingleToInt32Bits(floor);
        launch(K.SmxBoundSigForward, total, logits, bias, output, rows, cols, floorBits);
    }

    public static void BoundedSigmoidGateBackward(
        IDirectGpuBackend backend,
        IGpuBuffer gradOutput,
        IGpuBuffer logits,
        IGpuBuffer bias,
        IGpuBuffer gradLogits,
        IGpuBuffer gradBias,
        int rows,
        int cols,
        float floor)
    {
        if (rows < 0) throw new ArgumentOutOfRangeException(nameof(rows));
        if (cols <= 0) throw new ArgumentOutOfRangeException(nameof(cols));
        int total = checked(rows * cols);
        if (total == 0)
        {
            backend.Fill(gradBias, 0f, cols);
            return;
        }
        var launch = ((IFusedLstmSequenceTraining)backend).StatefulLstmLauncher;
        int floorBits = BitConverter.SingleToInt32Bits(floor);
        launch(K.SmxBoundSigBackwardDlog, total, gradOutput, logits, bias, gradLogits, rows, cols, floorBits);
        launch(K.SmxBoundSigBackwardDbias, cols, gradOutput, logits, bias, gradBias, rows, cols, floorBits);
    }
}
