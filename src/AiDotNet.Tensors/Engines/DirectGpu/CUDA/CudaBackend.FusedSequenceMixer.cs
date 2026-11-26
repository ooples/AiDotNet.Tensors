// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

public sealed partial class CudaBackend : IFusedSequenceMixerOps
{
    internal static bool UseFusedSequenceMixer = Environment.GetEnvironmentVariable("AIDOTNET_CUDA_FUSED_SEQ_MIXER") != "0";
    internal bool HasSequenceMixerKernel(string kernelName) => _kernelCache.ContainsKey(kernelName);

    void IFusedSequenceMixerOps.SplitHalfGivensRotationForward(
        IGpuBuffer input, IGpuBuffer angles, IGpuBuffer output, int rows, int halfDim)
    {
        if (!TrySplitHalfGivensRotationForward(input, angles, output, rows, halfDim))
            PortableSequenceMixerOps.SplitHalfGivensRotationForward(this, input, angles, output, rows, halfDim);
    }

    void IFusedSequenceMixerOps.SplitHalfGivensRotationBackward(
        IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer angles, IGpuBuffer gradInput, IGpuBuffer gradAngles, int rows, int halfDim)
    {
        if (!TrySplitHalfGivensRotationBackward(gradOutput, input, angles, gradInput, gradAngles, rows, halfDim))
            PortableSequenceMixerOps.SplitHalfGivensRotationBackward(this, gradOutput, input, angles, gradInput, gradAngles, rows, halfDim);
    }

    void IFusedSequenceMixerOps.DepthwiseCausalConv1d2TapForward(
        IGpuBuffer input, IGpuBuffer weight0, IGpuBuffer weight1, IGpuBuffer output, int batch, int seqLen, int channels)
    {
        if (!TryDepthwiseCausalConv1d2TapForward(input, weight0, weight1, output, batch, seqLen, channels))
            PortableSequenceMixerOps.DepthwiseCausalConv1d2TapForward(this, input, weight0, weight1, output, batch, seqLen, channels);
    }

    void IFusedSequenceMixerOps.DepthwiseCausalConv1d2TapBackward(
        IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer weight0, IGpuBuffer weight1,
        IGpuBuffer gradInput, IGpuBuffer gradWeight0, IGpuBuffer gradWeight1, int batch, int seqLen, int channels)
    {
        if (!TryDepthwiseCausalConv1d2TapBackward(gradOutput, input, weight0, weight1, gradInput, gradWeight0, gradWeight1, batch, seqLen, channels))
            PortableSequenceMixerOps.DepthwiseCausalConv1d2TapBackward(this, gradOutput, input, weight0, weight1, gradInput, gradWeight0, gradWeight1, batch, seqLen, channels);
    }

    void IFusedSequenceMixerOps.RopeL2NormalizeHeadsForward(
        IGpuBuffer input, IGpuBuffer cos, IGpuBuffer sin, IGpuBuffer output,
        int batch, int seqLen, int numHeads, int headDim, int startPosition, float epsilon, float postScale)
    {
        if (!TryRopeL2NormalizeHeadsForward(input, cos, sin, output, batch, seqLen, numHeads, headDim, startPosition, epsilon, postScale))
            PortableSequenceMixerOps.RopeL2NormalizeHeadsForward(this, input, cos, sin, output, batch, seqLen, numHeads, headDim, startPosition, epsilon, postScale);
    }

    void IFusedSequenceMixerOps.RopeL2NormalizeHeadsBackward(
        IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer cos, IGpuBuffer sin, IGpuBuffer gradInput,
        int batch, int seqLen, int numHeads, int headDim, int startPosition, float epsilon, float postScale)
    {
        if (!TryRopeL2NormalizeHeadsBackward(gradOutput, input, cos, sin, gradInput, batch, seqLen, numHeads, headDim, startPosition, epsilon, postScale))
            PortableSequenceMixerOps.RopeL2NormalizeHeadsBackward(this, gradOutput, input, cos, sin, gradInput, batch, seqLen, numHeads, headDim, startPosition, epsilon, postScale);
    }

    void IFusedSequenceMixerOps.RopeQuadraticOuterFeaturesForward(
        IGpuBuffer input, IGpuBuffer cos, IGpuBuffer sin, IGpuBuffer output,
        int batch, int seqLen, int numHeads, int featureDim, int outputHeadDim, int startPosition)
    {
        if (!TryRopeQuadraticOuterFeaturesForward(input, cos, sin, output, batch, seqLen, numHeads, featureDim, outputHeadDim, startPosition))
            PortableSequenceMixerOps.RopeQuadraticOuterFeaturesForward(this, input, cos, sin, output, batch, seqLen, numHeads, featureDim, outputHeadDim, startPosition);
    }

    void IFusedSequenceMixerOps.RopeQuadraticOuterFeaturesBackward(
        IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer cos, IGpuBuffer sin, IGpuBuffer gradInput,
        int batch, int seqLen, int numHeads, int featureDim, int outputHeadDim, int startPosition)
    {
        if (!TryRopeQuadraticOuterFeaturesBackward(gradOutput, input, cos, sin, gradInput, batch, seqLen, numHeads, featureDim, outputHeadDim, startPosition))
            PortableSequenceMixerOps.RopeQuadraticOuterFeaturesBackward(this, gradOutput, input, cos, sin, gradInput, batch, seqLen, numHeads, featureDim, outputHeadDim, startPosition);
    }

    void IFusedSequenceMixerOps.ProjectiveGatedResidualForward(
        IGpuBuffer baseHidden, IGpuBuffer proposal, IGpuBuffer gateLogit, IGpuBuffer output, int rows, int dim, float epsilon)
    {
        if (!TryProjectiveGatedResidualForward(baseHidden, proposal, gateLogit, output, rows, dim, epsilon))
            PortableSequenceMixerOps.ProjectiveGatedResidualForward(this, baseHidden, proposal, gateLogit, output, rows, dim, epsilon);
    }

    void IFusedSequenceMixerOps.ProjectiveGatedResidualBackward(
        IGpuBuffer gradOutput, IGpuBuffer baseHidden, IGpuBuffer proposal, IGpuBuffer gateLogit,
        IGpuBuffer gradBaseHidden, IGpuBuffer gradProposal, IGpuBuffer gradGateLogit, int rows, int dim, float epsilon)
    {
        if (!TryProjectiveGatedResidualBackward(gradOutput, baseHidden, proposal, gateLogit, gradBaseHidden, gradProposal, gradGateLogit, rows, dim, epsilon))
            PortableSequenceMixerOps.ProjectiveGatedResidualBackward(this, gradOutput, baseHidden, proposal, gateLogit, gradBaseHidden, gradProposal, gradGateLogit, rows, dim, epsilon);
    }

    void IFusedSequenceMixerOps.SpectralCirculantMatrixForward(
        IGpuBuffer diagScale, IGpuBuffer spectrumRe, IGpuBuffer spectrumIm, IGpuBuffer output, int batch, int dim)
    {
        if (!TrySpectralCirculantMatrixForward(diagScale, spectrumRe, spectrumIm, output, batch, dim))
            PortableSequenceMixerOps.SpectralCirculantMatrixForward(this, diagScale, spectrumRe, spectrumIm, output, batch, dim);
    }

    void IFusedSequenceMixerOps.SpectralCirculantMatrixBackward(
        IGpuBuffer gradOutput, IGpuBuffer diagScale, IGpuBuffer spectrumRe, IGpuBuffer spectrumIm,
        IGpuBuffer gradDiagScale, IGpuBuffer gradSpectrumRe, IGpuBuffer gradSpectrumIm, int batch, int dim)
    {
        if (!TrySpectralCirculantMatrixBackward(gradOutput, diagScale, spectrumRe, spectrumIm, gradDiagScale, gradSpectrumRe, gradSpectrumIm, batch, dim))
            PortableSequenceMixerOps.SpectralCirculantMatrixBackward(this, gradOutput, diagScale, spectrumRe, spectrumIm, gradDiagScale, gradSpectrumRe, gradSpectrumIm, batch, dim);
    }

    void IFusedSequenceMixerOps.SigmoidTopKExpertMixForward(
        IGpuBuffer logits, IGpuBuffer expertTable, IGpuBuffer routingBias, IGpuBuffer output, IGpuBuffer savedInfo,
        int rows, int numExperts, int topK, int channels, float epsilon, float biasUpdateRate)
    {
        if (!TrySigmoidTopKExpertMixForward(logits, expertTable, routingBias, output, savedInfo, rows, numExperts, topK, channels, epsilon, biasUpdateRate))
            PortableSequenceMixerOps.SigmoidTopKExpertMixForward(this, logits, expertTable, routingBias, output, savedInfo, rows, numExperts, topK, channels, epsilon, biasUpdateRate);
    }

    void IFusedSequenceMixerOps.SigmoidTopKExpertMixBackward(
        IGpuBuffer gradOutput, IGpuBuffer logits, IGpuBuffer expertTable, IGpuBuffer savedInfo,
        IGpuBuffer gradLogits, IGpuBuffer gradExpertTable, int rows, int numExperts, int topK, int channels, float epsilon)
    {
        if (!TrySigmoidTopKExpertMixBackward(gradOutput, logits, expertTable, savedInfo, gradLogits, gradExpertTable, rows, numExperts, topK, channels, epsilon))
            PortableSequenceMixerOps.SigmoidTopKExpertMixBackward(this, gradOutput, logits, expertTable, savedInfo, gradLogits, gradExpertTable, rows, numExperts, topK, channels, epsilon);
    }

    void IFusedSequenceMixerOps.DualModulatedSwiGluForward(
        IGpuBuffer gateUpProj, IGpuBuffer gateUpGainDelta, IGpuBuffer upBase, IGpuBuffer output, int rows, int hiddenDim)
    {
        if (!TryDualModulatedSwiGluForward(gateUpProj, gateUpGainDelta, upBase, output, rows, hiddenDim))
            PortableSequenceMixerOps.DualModulatedSwiGluForward(this, gateUpProj, gateUpGainDelta, upBase, output, rows, hiddenDim);
    }

    void IFusedSequenceMixerOps.DualModulatedSwiGluBackward(
        IGpuBuffer gradOutput, IGpuBuffer gateUpProj, IGpuBuffer gateUpGainDelta, IGpuBuffer upBase,
        IGpuBuffer gradGateUpProj, IGpuBuffer gradGateUpGainDelta, IGpuBuffer gradUpBase, int rows, int hiddenDim)
    {
        if (!TryDualModulatedSwiGluBackward(gradOutput, gateUpProj, gateUpGainDelta, upBase, gradGateUpProj, gradGateUpGainDelta, gradUpBase, rows, hiddenDim))
            PortableSequenceMixerOps.DualModulatedSwiGluBackward(this, gradOutput, gateUpProj, gateUpGainDelta, upBase, gradGateUpProj, gradGateUpGainDelta, gradUpBase, rows, hiddenDim);
    }

    void IFusedSequenceMixerOps.BornRatioNormalizeForward(
        IGpuBuffer scanOutput, IGpuBuffer sinkLogAlphaOrDummy, IGpuBuffer output,
        int totalRows, int numHeads, int headDim, int scanDim, int hasSink, float epsilon)
    {
        if (!TryBornRatioNormalizeForward(scanOutput, sinkLogAlphaOrDummy, output, totalRows, numHeads, headDim, scanDim, hasSink, epsilon))
            PortableSequenceMixerOps.BornRatioNormalizeForward(this, scanOutput, sinkLogAlphaOrDummy, output, totalRows, numHeads, headDim, scanDim, hasSink, epsilon);
    }

    void IFusedSequenceMixerOps.BornRatioNormalizeBackward(
        IGpuBuffer gradOutput, IGpuBuffer scanOutput, IGpuBuffer sinkLogAlphaOrDummy,
        IGpuBuffer gradScanOutput, IGpuBuffer gradSinkOrDummy,
        int totalRows, int numHeads, int headDim, int scanDim, int hasSink, float epsilon)
    {
        if (!TryBornRatioNormalizeBackward(gradOutput, scanOutput, sinkLogAlphaOrDummy, gradScanOutput, gradSinkOrDummy, totalRows, numHeads, headDim, scanDim, hasSink, epsilon))
            PortableSequenceMixerOps.BornRatioNormalizeBackward(this, gradOutput, scanOutput, sinkLogAlphaOrDummy, gradScanOutput, gradSinkOrDummy, totalRows, numHeads, headDim, scanDim, hasSink, epsilon);
    }

    void IFusedSequenceMixerOps.BoundedSigmoidGateForward(
        IGpuBuffer logits, IGpuBuffer bias, IGpuBuffer output, int rows, int cols, float floor)
    {
        if (!TryBoundedSigmoidGateForward(logits, bias, output, rows, cols, floor))
            PortableSequenceMixerOps.BoundedSigmoidGateForward(this, logits, bias, output, rows, cols, floor);
    }

    void IFusedSequenceMixerOps.BoundedSigmoidGateBackward(
        IGpuBuffer gradOutput, IGpuBuffer logits, IGpuBuffer bias, IGpuBuffer gradLogits, IGpuBuffer gradBias, int rows, int cols, float floor)
    {
        if (!TryBoundedSigmoidGateBackward(gradOutput, logits, bias, gradLogits, gradBias, rows, cols, floor))
            PortableSequenceMixerOps.BoundedSigmoidGateBackward(this, gradOutput, logits, bias, gradLogits, gradBias, rows, cols, floor);
    }

    private unsafe bool TrySplitHalfGivensRotationForward(
        IGpuBuffer input, IGpuBuffer angles, IGpuBuffer output, int rows, int halfDim)
    {
        if (rows == 0) return true;
        if (!UseFusedSequenceMixer || rows < 0 || halfDim <= 0) return false;
        if (!_kernelCache.TryGetValue("smx_cuda_givens_fwd", out var kernel)) return false;

        using var _ = PushContext();
        int totalHalf = checked(rows * halfDim);
        uint blocks = (uint)Math.Min(Math.Max(1, (totalHalf + CudaSequenceMixerKernels.ElemBlockThreads - 1) / CudaSequenceMixerKernels.ElemBlockThreads), CudaSequenceMixerKernels.ElemMaxGridBlocks);
        IntPtr xPtr = input.Handle, aPtr = angles.Handle, yPtr = output.Handle;
        void** args = stackalloc void*[5];
        args[0] = &xPtr; args[1] = &aPtr; args[2] = &yPtr; args[3] = &totalHalf; args[4] = &halfDim;
        LaunchKernel(kernel, blocks, CudaSequenceMixerKernels.ElemBlockThreads, args);
        return true;
    }

    private unsafe bool TrySplitHalfGivensRotationBackward(
        IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer angles, IGpuBuffer gradInput, IGpuBuffer gradAngles, int rows, int halfDim)
    {
        if (rows == 0)
        {
            Fill(gradAngles, 0f, halfDim);
            return true;
        }
        if (!UseFusedSequenceMixer || rows < 0 || halfDim <= 0 || halfDim > CudaSequenceMixerKernels.MaxSingleBlockPerChannel) return false;
        if (!_kernelCache.TryGetValue("smx_cuda_givens_bwd_dx", out var dxKernel)
            || !_kernelCache.TryGetValue("smx_cuda_givens_bwd_dphi", out var dphiKernel))
            return false;

        using var _ = PushContext();
        int totalHalf = checked(rows * halfDim);
        uint dxBlocks = (uint)Math.Min(Math.Max(1, (totalHalf + CudaSequenceMixerKernels.ElemBlockThreads - 1) / CudaSequenceMixerKernels.ElemBlockThreads), CudaSequenceMixerKernels.ElemMaxGridBlocks);
        IntPtr dyPtr = gradOutput.Handle, xPtr = input.Handle, aPtr = angles.Handle, dxPtr = gradInput.Handle, daPtr = gradAngles.Handle;
        void** dxArgs = stackalloc void*[5];
        dxArgs[0] = &dyPtr; dxArgs[1] = &aPtr; dxArgs[2] = &dxPtr; dxArgs[3] = &totalHalf; dxArgs[4] = &halfDim;
        LaunchKernel(dxKernel, dxBlocks, CudaSequenceMixerKernels.ElemBlockThreads, dxArgs);

        void** daArgs = stackalloc void*[6];
        daArgs[0] = &dyPtr; daArgs[1] = &xPtr; daArgs[2] = &aPtr; daArgs[3] = &daPtr; daArgs[4] = &rows; daArgs[5] = &halfDim;
        LaunchKernel(dphiKernel, (uint)halfDim, CudaSequenceMixerKernels.WarpBlockThreads, daArgs);
        return true;
    }

    private unsafe bool TryDepthwiseCausalConv1d2TapForward(
        IGpuBuffer input, IGpuBuffer weight0, IGpuBuffer weight1, IGpuBuffer output, int batch, int seqLen, int channels)
    {
        int total = checked(batch * seqLen * channels);
        if (total == 0) return true;
        if (!UseFusedSequenceMixer || batch < 0 || seqLen < 0 || channels <= 0) return false;
        if (!_kernelCache.TryGetValue("smx_cuda_conv2_fwd", out var kernel)) return false;

        using var _ = PushContext();
        uint blocks = (uint)Math.Min(Math.Max(1, (total + CudaSequenceMixerKernels.ElemBlockThreads - 1) / CudaSequenceMixerKernels.ElemBlockThreads), CudaSequenceMixerKernels.ElemMaxGridBlocks);
        IntPtr xPtr = input.Handle, w0Ptr = weight0.Handle, w1Ptr = weight1.Handle, yPtr = output.Handle;
        void** args = stackalloc void*[7];
        args[0] = &xPtr; args[1] = &w0Ptr; args[2] = &w1Ptr; args[3] = &yPtr; args[4] = &total; args[5] = &seqLen; args[6] = &channels;
        LaunchKernel(kernel, blocks, CudaSequenceMixerKernels.ElemBlockThreads, args);
        return true;
    }

    private unsafe bool TryDepthwiseCausalConv1d2TapBackward(
        IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer weight0, IGpuBuffer weight1,
        IGpuBuffer gradInput, IGpuBuffer gradWeight0, IGpuBuffer gradWeight1, int batch, int seqLen, int channels)
    {
        int total = checked(batch * seqLen * channels);
        if (total == 0)
        {
            Fill(gradWeight0, 0f, channels);
            Fill(gradWeight1, 0f, channels);
            return true;
        }
        if (!UseFusedSequenceMixer || batch < 0 || seqLen < 0 || channels <= 0 || channels > CudaSequenceMixerKernels.MaxSingleBlockPerChannel) return false;
        if (!_kernelCache.TryGetValue("smx_cuda_conv2_bwd_dx", out var dxKernel)
            || !_kernelCache.TryGetValue("smx_cuda_conv2_bwd_dw", out var dwKernel))
            return false;

        using var _ = PushContext();
        uint dxBlocks = (uint)Math.Min(Math.Max(1, (total + CudaSequenceMixerKernels.ElemBlockThreads - 1) / CudaSequenceMixerKernels.ElemBlockThreads), CudaSequenceMixerKernels.ElemMaxGridBlocks);
        IntPtr dyPtr = gradOutput.Handle, xPtr = input.Handle, w0Ptr = weight0.Handle, w1Ptr = weight1.Handle;
        IntPtr dxPtr = gradInput.Handle, dw0Ptr = gradWeight0.Handle, dw1Ptr = gradWeight1.Handle;
        void** dxArgs = stackalloc void*[7];
        dxArgs[0] = &dyPtr; dxArgs[1] = &w0Ptr; dxArgs[2] = &w1Ptr; dxArgs[3] = &dxPtr; dxArgs[4] = &total; dxArgs[5] = &seqLen; dxArgs[6] = &channels;
        LaunchKernel(dxKernel, dxBlocks, CudaSequenceMixerKernels.ElemBlockThreads, dxArgs);

        void** dwArgs = stackalloc void*[7];
        dwArgs[0] = &dyPtr; dwArgs[1] = &xPtr; dwArgs[2] = &dw0Ptr; dwArgs[3] = &dw1Ptr; dwArgs[4] = &batch; dwArgs[5] = &seqLen; dwArgs[6] = &channels;
        LaunchKernel(dwKernel, (uint)channels, CudaSequenceMixerKernels.WarpBlockThreads, dwArgs);
        return true;
    }

    private unsafe bool TryRopeL2NormalizeHeadsForward(
        IGpuBuffer input, IGpuBuffer cos, IGpuBuffer sin, IGpuBuffer output,
        int batch, int seqLen, int numHeads, int headDim, int startPosition, float epsilon, float postScale)
    {
        int totalHeads = checked(batch * seqLen * numHeads);
        if (totalHeads == 0) return true;
        if (!UseFusedSequenceMixer || headDim <= 0 || (headDim & 1) != 0 || headDim > CudaSequenceMixerKernels.MaxRopeL2HeadDim)
            return false;
        if (!_kernelCache.TryGetValue("smx_cuda_ropel2_fwd", out var kernel)) return false;

        using var _ = PushContext();
        uint blocks = (uint)Math.Min(Math.Max(1, (totalHeads + CudaSequenceMixerKernels.WarpsPerBlock - 1) / CudaSequenceMixerKernels.WarpsPerBlock), CudaSequenceMixerKernels.WarpHeadMaxGridBlocks);
        IntPtr xPtr = input.Handle, cPtr = cos.Handle, sPtr = sin.Handle, yPtr = output.Handle;
        void** args = stackalloc void*[11];
        args[0] = &xPtr; args[1] = &cPtr; args[2] = &sPtr; args[3] = &yPtr;
        args[4] = &totalHeads; args[5] = &seqLen; args[6] = &numHeads; args[7] = &headDim;
        args[8] = &startPosition; args[9] = &epsilon; args[10] = &postScale;
        LaunchKernel(kernel, blocks, CudaSequenceMixerKernels.WarpBlockThreads, args);
        return true;
    }

    private unsafe bool TryRopeL2NormalizeHeadsBackward(
        IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer cos, IGpuBuffer sin, IGpuBuffer gradInput,
        int batch, int seqLen, int numHeads, int headDim, int startPosition, float epsilon, float postScale)
    {
        int totalHeads = checked(batch * seqLen * numHeads);
        if (totalHeads == 0) return true;
        if (!UseFusedSequenceMixer || headDim <= 0 || (headDim & 1) != 0 || headDim > CudaSequenceMixerKernels.MaxRopeL2HeadDim)
            return false;
        if (!_kernelCache.TryGetValue("smx_cuda_ropel2_bwd", out var kernel)) return false;

        using var _ = PushContext();
        uint blocks = (uint)Math.Min(Math.Max(1, (totalHeads + CudaSequenceMixerKernels.WarpsPerBlock - 1) / CudaSequenceMixerKernels.WarpsPerBlock), CudaSequenceMixerKernels.WarpHeadMaxGridBlocks);
        IntPtr dyPtr = gradOutput.Handle, xPtr = input.Handle, cPtr = cos.Handle, sPtr = sin.Handle, dxPtr = gradInput.Handle;
        void** args = stackalloc void*[12];
        args[0] = &dyPtr; args[1] = &xPtr; args[2] = &cPtr; args[3] = &sPtr; args[4] = &dxPtr;
        args[5] = &totalHeads; args[6] = &seqLen; args[7] = &numHeads; args[8] = &headDim;
        args[9] = &startPosition; args[10] = &epsilon; args[11] = &postScale;
        LaunchKernel(kernel, blocks, CudaSequenceMixerKernels.WarpBlockThreads, args);
        return true;
    }

    private unsafe bool TryRopeQuadraticOuterFeaturesForward(
        IGpuBuffer input, IGpuBuffer cos, IGpuBuffer sin, IGpuBuffer output,
        int batch, int seqLen, int numHeads, int featureDim, int outputHeadDim, int startPosition)
    {
        int totalHeads = checked(batch * seqLen * numHeads);
        if (totalHeads == 0) return true;
        if (!UseFusedSequenceMixer || featureDim <= 0 || (featureDim & 1) != 0 || featureDim > CudaSequenceMixerKernels.MaxRopeQuadFeatureDim)
            return false;
        if (!_kernelCache.TryGetValue("smx_cuda_ropequad_fwd", out var kernel)) return false;

        using var _ = PushContext();
        uint blocks = (uint)Math.Min(Math.Max(1, (totalHeads + CudaSequenceMixerKernels.WarpsPerBlock - 1) / CudaSequenceMixerKernels.WarpsPerBlock), CudaSequenceMixerKernels.WarpHeadMaxGridBlocks);
        IntPtr xPtr = input.Handle, cPtr = cos.Handle, sPtr = sin.Handle, yPtr = output.Handle;
        void** args = stackalloc void*[10];
        args[0] = &xPtr; args[1] = &cPtr; args[2] = &sPtr; args[3] = &yPtr;
        args[4] = &totalHeads; args[5] = &seqLen; args[6] = &numHeads; args[7] = &featureDim;
        args[8] = &outputHeadDim; args[9] = &startPosition;
        LaunchKernel(kernel, blocks, CudaSequenceMixerKernels.WarpBlockThreads, args);
        return true;
    }

    private unsafe bool TryRopeQuadraticOuterFeaturesBackward(
        IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer cos, IGpuBuffer sin, IGpuBuffer gradInput,
        int batch, int seqLen, int numHeads, int featureDim, int outputHeadDim, int startPosition)
    {
        int totalHeads = checked(batch * seqLen * numHeads);
        if (totalHeads == 0) return true;
        if (!UseFusedSequenceMixer || featureDim <= 0 || (featureDim & 1) != 0 || featureDim > CudaSequenceMixerKernels.MaxRopeQuadFeatureDim)
            return false;
        if (!_kernelCache.TryGetValue("smx_cuda_ropequad_bwd", out var kernel)) return false;

        using var _ = PushContext();
        uint blocks = (uint)Math.Min(Math.Max(1, (totalHeads + CudaSequenceMixerKernels.WarpsPerBlock - 1) / CudaSequenceMixerKernels.WarpsPerBlock), CudaSequenceMixerKernels.WarpHeadMaxGridBlocks);
        IntPtr dyPtr = gradOutput.Handle, xPtr = input.Handle, cPtr = cos.Handle, sPtr = sin.Handle, dxPtr = gradInput.Handle;
        void** args = stackalloc void*[11];
        args[0] = &dyPtr; args[1] = &xPtr; args[2] = &cPtr; args[3] = &sPtr; args[4] = &dxPtr;
        args[5] = &totalHeads; args[6] = &seqLen; args[7] = &numHeads; args[8] = &featureDim;
        args[9] = &outputHeadDim; args[10] = &startPosition;
        LaunchKernel(kernel, blocks, CudaSequenceMixerKernels.WarpBlockThreads, args);
        return true;
    }

    private unsafe bool TryProjectiveGatedResidualForward(
        IGpuBuffer baseHidden, IGpuBuffer proposal, IGpuBuffer gateLogit, IGpuBuffer output, int rows, int dim, float epsilon)
    {
        if (rows == 0) return true;
        if (!UseFusedSequenceMixer || rows < 0 || dim <= 0 || dim > CudaSequenceMixerKernels.MaxProjResDim) return false;
        if (!_kernelCache.TryGetValue("smx_cuda_projres_fwd", out var kernel)) return false;

        using var _ = PushContext();
        uint blocks = (uint)Math.Min(Math.Max(1, (rows + CudaSequenceMixerKernels.WarpsPerBlock - 1) / CudaSequenceMixerKernels.WarpsPerBlock), CudaSequenceMixerKernels.ProjResForwardMaxBlocks);
        IntPtr hPtr = baseHidden.Handle, uPtr = proposal.Handle, gPtr = gateLogit.Handle, yPtr = output.Handle;
        void** args = stackalloc void*[7];
        args[0] = &hPtr; args[1] = &uPtr; args[2] = &gPtr; args[3] = &yPtr;
        args[4] = &rows; args[5] = &dim; args[6] = &epsilon;
        LaunchKernel(kernel, blocks, CudaSequenceMixerKernels.WarpBlockThreads, args);
        return true;
    }

    private unsafe bool TryProjectiveGatedResidualBackward(
        IGpuBuffer gradOutput, IGpuBuffer baseHidden, IGpuBuffer proposal, IGpuBuffer gateLogit,
        IGpuBuffer gradBaseHidden, IGpuBuffer gradProposal, IGpuBuffer gradGateLogit, int rows, int dim, float epsilon)
    {
        if (rows == 0)
        {
            Fill(gradGateLogit, 0f, 1);
            return true;
        }
        if (!UseFusedSequenceMixer || rows < 0 || dim <= 0 || dim > CudaSequenceMixerKernels.MaxProjResDim) return false;
        if (!_kernelCache.TryGetValue("smx_cuda_projres_bwd_rows", out var rowsKernel)
            || !_kernelCache.TryGetValue("smx_cuda_projres_bwd_dgate", out var dgateKernel))
            return false;

        using var _ = PushContext();
        int numBlocks = Math.Min(Math.Max(1, (rows + CudaSequenceMixerKernels.WarpsPerBlock - 1) / CudaSequenceMixerKernels.WarpsPerBlock), CudaSequenceMixerKernels.ProjResBackwardMaxBlocks);
        using var partials = AllocateBuffer(numBlocks);

        IntPtr dyPtr = gradOutput.Handle, hPtr = baseHidden.Handle, uPtr = proposal.Handle, gPtr = gateLogit.Handle;
        IntPtr dhPtr = gradBaseHidden.Handle, duPtr = gradProposal.Handle, pPtr = partials.Handle, dgPtr = gradGateLogit.Handle;

        void** rowArgs = stackalloc void*[10];
        rowArgs[0] = &dyPtr; rowArgs[1] = &hPtr; rowArgs[2] = &uPtr; rowArgs[3] = &gPtr;
        rowArgs[4] = &dhPtr; rowArgs[5] = &duPtr; rowArgs[6] = &pPtr;
        rowArgs[7] = &rows; rowArgs[8] = &dim; rowArgs[9] = &epsilon;
        LaunchKernel(rowsKernel, (uint)numBlocks, CudaSequenceMixerKernels.WarpBlockThreads, rowArgs);

        void** redArgs = stackalloc void*[4];
        redArgs[0] = &pPtr; redArgs[1] = &gPtr; redArgs[2] = &dgPtr; redArgs[3] = &numBlocks;
        LaunchKernel(dgateKernel, 1u, CudaSequenceMixerKernels.ElemBlockThreads, redArgs);
        return true;
    }

    private unsafe bool TrySpectralCirculantMatrixForward(
        IGpuBuffer diagScale, IGpuBuffer spectrumRe, IGpuBuffer spectrumIm, IGpuBuffer output, int batch, int dim)
    {
        if (batch == 0) return true;
        if (!UseFusedSequenceMixer || batch < 0 || dim <= 0 || (dim & 1) != 0 || dim > CudaSequenceMixerKernels.MaxSpecCircDim || batch > CudaSequenceMixerKernels.MaxSingleBlockPerChannel)
            return false;
        if (!_kernelCache.TryGetValue("smx_cuda_speccirc_fwd", out var kernel)) return false;

        using var _ = PushContext();
        IntPtr dPtr = diagScale.Handle, rePtr = spectrumRe.Handle, imPtr = spectrumIm.Handle, oPtr = output.Handle;
        void** args = stackalloc void*[6];
        args[0] = &dPtr; args[1] = &rePtr; args[2] = &imPtr; args[3] = &oPtr; args[4] = &batch; args[5] = &dim;
        LaunchKernel(kernel, (uint)batch, CudaSequenceMixerKernels.ElemBlockThreads, args);
        return true;
    }

    private unsafe bool TrySpectralCirculantMatrixBackward(
        IGpuBuffer gradOutput, IGpuBuffer diagScale, IGpuBuffer spectrumRe, IGpuBuffer spectrumIm,
        IGpuBuffer gradDiagScale, IGpuBuffer gradSpectrumRe, IGpuBuffer gradSpectrumIm, int batch, int dim)
    {
        if (batch == 0) return true;
        if (!UseFusedSequenceMixer || batch < 0 || dim <= 0 || (dim & 1) != 0 || dim > CudaSequenceMixerKernels.MaxSpecCircDim || batch > CudaSequenceMixerKernels.MaxSingleBlockPerChannel)
            return false;
        if (!_kernelCache.TryGetValue("smx_cuda_speccirc_bwd", out var kernel)) return false;

        using var _ = PushContext();
        IntPtr dyPtr = gradOutput.Handle, dPtr = diagScale.Handle, rePtr = spectrumRe.Handle, imPtr = spectrumIm.Handle;
        IntPtr ddPtr = gradDiagScale.Handle, drePtr = gradSpectrumRe.Handle, dimPtr = gradSpectrumIm.Handle;
        void** args = stackalloc void*[9];
        args[0] = &dyPtr; args[1] = &dPtr; args[2] = &rePtr; args[3] = &imPtr;
        args[4] = &ddPtr; args[5] = &drePtr; args[6] = &dimPtr; args[7] = &batch; args[8] = &dim;
        LaunchKernel(kernel, (uint)batch, CudaSequenceMixerKernels.ElemBlockThreads, args);
        return true;
    }

    private unsafe bool TrySigmoidTopKExpertMixForward(
        IGpuBuffer logits, IGpuBuffer expertTable, IGpuBuffer routingBias, IGpuBuffer output, IGpuBuffer savedInfo,
        int rows, int numExperts, int topK, int channels, float epsilon, float biasUpdateRate)
    {
        if (rows == 0) return true;
        if (!UseFusedSequenceMixer
            || rows < 0
            || numExperts <= 0 || numExperts > CudaSequenceMixerKernels.MaxTopKExperts
            || topK <= 0 || topK > numExperts || topK > CudaSequenceMixerKernels.MaxTopK
            || channels <= 0)
            return false;
        if (!_kernelCache.TryGetValue("smx_cuda_sigtopk_mix_fwd", out var fwdKernel)) return false;
        IntPtr biasUpdKernel = IntPtr.Zero;
        if (biasUpdateRate != 0f && !_kernelCache.TryGetValue("smx_cuda_sigtopk_bias_upd", out biasUpdKernel))
            return false;

        using var _ = PushContext();
        IGpuBuffer? countsBuf = null;
        try
        {
            IntPtr countsPtr = IntPtr.Zero;
            if (biasUpdateRate != 0f)
            {
                countsBuf = AllocateIntBuffer(new int[numExperts]);
                countsPtr = countsBuf.Handle;
            }

            uint blocks = (uint)Math.Min(Math.Max(1, (rows + CudaSequenceMixerKernels.WarpsPerBlock - 1) / CudaSequenceMixerKernels.WarpsPerBlock), CudaSequenceMixerKernels.TopKMaxGridBlocks);
            IntPtr logPtr = logits.Handle, tblPtr = expertTable.Handle, biasPtr = routingBias.Handle;
            IntPtr yPtr = output.Handle, infoPtr = savedInfo.Handle;
            void** args = stackalloc void*[11];
            args[0] = &logPtr; args[1] = &tblPtr; args[2] = &biasPtr;
            args[3] = &yPtr; args[4] = &infoPtr; args[5] = &countsPtr;
            args[6] = &rows; args[7] = &numExperts; args[8] = &topK; args[9] = &channels; args[10] = &epsilon;
            LaunchKernel(fwdKernel, blocks, CudaSequenceMixerKernels.WarpBlockThreads, args);

            if (biasUpdateRate != 0f)
            {
                uint updBlocks = (uint)((numExperts + CudaSequenceMixerKernels.ElemBlockThreads - 1) / CudaSequenceMixerKernels.ElemBlockThreads);
                void** uArgs = stackalloc void*[6];
                uArgs[0] = &countsPtr; uArgs[1] = &biasPtr;
                uArgs[2] = &rows; uArgs[3] = &numExperts; uArgs[4] = &topK; uArgs[5] = &biasUpdateRate;
                LaunchKernel(biasUpdKernel, updBlocks, CudaSequenceMixerKernels.ElemBlockThreads, uArgs);
            }
            return true;
        }
        finally
        {
            if (countsBuf is not null)
                ((IQueuedBufferRelease)this).ReleaseAfterQueuedWork(countsBuf);
        }
    }

    private unsafe bool TrySigmoidTopKExpertMixBackward(
        IGpuBuffer gradOutput, IGpuBuffer logits, IGpuBuffer expertTable, IGpuBuffer savedInfo,
        IGpuBuffer gradLogits, IGpuBuffer gradExpertTable, int rows, int numExperts, int topK, int channels, float epsilon)
    {
        if (rows == 0)
        {
            Fill(gradExpertTable, 0f, checked(numExperts * channels));
            return true;
        }
        if (!UseFusedSequenceMixer
            || rows < 0
            || numExperts <= 0 || numExperts > CudaSequenceMixerKernels.MaxTopKExperts
            || topK <= 0 || topK > numExperts || topK > CudaSequenceMixerKernels.MaxTopK
            || channels <= 0)
            return false;
        if (!_kernelCache.TryGetValue("smx_cuda_sigtopk_mix_bwd_dlog", out var dlogKernel)
            || !_kernelCache.TryGetValue("smx_cuda_sigtopk_mix_bwd_dtbl", out var dtblKernel))
            return false;

        using var _ = PushContext();
        uint logBlocks = (uint)Math.Min(Math.Max(1, (rows + CudaSequenceMixerKernels.WarpsPerBlock - 1) / CudaSequenceMixerKernels.WarpsPerBlock), CudaSequenceMixerKernels.TopKMaxGridBlocks);
        IntPtr dyPtr = gradOutput.Handle, logPtr = logits.Handle, tblPtr = expertTable.Handle, infoPtr = savedInfo.Handle;
        IntPtr dlogPtr = gradLogits.Handle, dtblPtr = gradExpertTable.Handle;
        void** lArgs = stackalloc void*[10];
        lArgs[0] = &dyPtr; lArgs[1] = &logPtr; lArgs[2] = &tblPtr; lArgs[3] = &infoPtr; lArgs[4] = &dlogPtr;
        lArgs[5] = &rows; lArgs[6] = &numExperts; lArgs[7] = &topK; lArgs[8] = &channels; lArgs[9] = &epsilon;
        LaunchKernel(dlogKernel, logBlocks, CudaSequenceMixerKernels.WarpBlockThreads, lArgs);

        int cells = checked(numExperts * channels);
        uint tblBlocks = (uint)Math.Min(Math.Max(1, (cells + CudaSequenceMixerKernels.WarpsPerBlock - 1) / CudaSequenceMixerKernels.WarpsPerBlock), CudaSequenceMixerKernels.TopKMaxGridBlocks);
        void** tArgs = stackalloc void*[7];
        tArgs[0] = &dyPtr; tArgs[1] = &infoPtr; tArgs[2] = &dtblPtr;
        tArgs[3] = &rows; tArgs[4] = &numExperts; tArgs[5] = &topK; tArgs[6] = &channels;
        LaunchKernel(dtblKernel, tblBlocks, CudaSequenceMixerKernels.WarpBlockThreads, tArgs);
        return true;
    }

    private unsafe bool TryDualModulatedSwiGluForward(
        IGpuBuffer gateUpProj, IGpuBuffer gateUpGainDelta, IGpuBuffer upBase, IGpuBuffer output, int rows, int hiddenDim)
    {
        int total = checked(rows * hiddenDim);
        if (total == 0) return true;
        if (!UseFusedSequenceMixer || rows < 0 || hiddenDim <= 0) return false;
        if (!_kernelCache.TryGetValue("smx_cuda_dualswiglu_fwd", out var kernel)) return false;

        using var _ = PushContext();
        uint blocks = (uint)Math.Min(Math.Max(1, (total + CudaSequenceMixerKernels.ElemBlockThreads - 1) / CudaSequenceMixerKernels.ElemBlockThreads), CudaSequenceMixerKernels.ElemMaxGridBlocks);
        IntPtr guPtr = gateUpProj.Handle, ggPtr = gateUpGainDelta.Handle, bPtr = upBase.Handle, yPtr = output.Handle;
        void** args = stackalloc void*[6];
        args[0] = &guPtr; args[1] = &ggPtr; args[2] = &bPtr; args[3] = &yPtr; args[4] = &total; args[5] = &hiddenDim;
        LaunchKernel(kernel, blocks, CudaSequenceMixerKernels.ElemBlockThreads, args);
        return true;
    }

    private unsafe bool TryDualModulatedSwiGluBackward(
        IGpuBuffer gradOutput, IGpuBuffer gateUpProj, IGpuBuffer gateUpGainDelta, IGpuBuffer upBase,
        IGpuBuffer gradGateUpProj, IGpuBuffer gradGateUpGainDelta, IGpuBuffer gradUpBase, int rows, int hiddenDim)
    {
        int total = checked(rows * hiddenDim);
        if (total == 0)
        {
            Fill(gradUpBase, 0f, hiddenDim);
            return true;
        }
        if (!UseFusedSequenceMixer || rows < 0 || hiddenDim <= 0 || hiddenDim > CudaSequenceMixerKernels.MaxSingleBlockPerChannel) return false;
        if (!_kernelCache.TryGetValue("smx_cuda_dualswiglu_bwd_elem", out var elemKernel)
            || !_kernelCache.TryGetValue("smx_cuda_dualswiglu_bwd_dbase", out var dbaseKernel))
            return false;

        using var _ = PushContext();
        uint blocks = (uint)Math.Min(Math.Max(1, (total + CudaSequenceMixerKernels.ElemBlockThreads - 1) / CudaSequenceMixerKernels.ElemBlockThreads), CudaSequenceMixerKernels.ElemMaxGridBlocks);
        IntPtr dyPtr = gradOutput.Handle, guPtr = gateUpProj.Handle, ggPtr = gateUpGainDelta.Handle, bPtr = upBase.Handle;
        IntPtr dguPtr = gradGateUpProj.Handle, dggPtr = gradGateUpGainDelta.Handle, dbPtr = gradUpBase.Handle;
        void** eArgs = stackalloc void*[8];
        eArgs[0] = &dyPtr; eArgs[1] = &guPtr; eArgs[2] = &ggPtr; eArgs[3] = &bPtr;
        eArgs[4] = &dguPtr; eArgs[5] = &dggPtr; eArgs[6] = &total; eArgs[7] = &hiddenDim;
        LaunchKernel(elemKernel, blocks, CudaSequenceMixerKernels.ElemBlockThreads, eArgs);

        void** bArgs = stackalloc void*[6];
        bArgs[0] = &dyPtr; bArgs[1] = &guPtr; bArgs[2] = &ggPtr; bArgs[3] = &dbPtr; bArgs[4] = &rows; bArgs[5] = &hiddenDim;
        LaunchKernel(dbaseKernel, (uint)hiddenDim, CudaSequenceMixerKernels.WarpBlockThreads, bArgs);
        return true;
    }

    private unsafe bool TryBornRatioNormalizeForward(
        IGpuBuffer scanOutput, IGpuBuffer sinkLogAlphaOrDummy, IGpuBuffer output,
        int totalRows, int numHeads, int headDim, int scanDim, int hasSink, float epsilon)
    {
        int totalOut = checked(totalRows * headDim);
        if (totalOut == 0) return true;
        if (!UseFusedSequenceMixer || totalRows < 0 || numHeads <= 0 || headDim <= 0 || headDim >= scanDim)
            return false;
        if (!_kernelCache.TryGetValue("smx_cuda_bornratio_fwd", out var kernel)) return false;

        using var _ = PushContext();
        uint blocks = (uint)Math.Min(Math.Max(1, (totalOut + CudaSequenceMixerKernels.ElemBlockThreads - 1) / CudaSequenceMixerKernels.ElemBlockThreads), CudaSequenceMixerKernels.ElemMaxGridBlocks);
        IntPtr sPtr = scanOutput.Handle, sinkPtr = sinkLogAlphaOrDummy.Handle, yPtr = output.Handle;
        void** args = stackalloc void*[9];
        args[0] = &sPtr; args[1] = &sinkPtr; args[2] = &yPtr;
        args[3] = &totalOut; args[4] = &numHeads; args[5] = &headDim; args[6] = &scanDim; args[7] = &hasSink; args[8] = &epsilon;
        LaunchKernel(kernel, blocks, CudaSequenceMixerKernels.ElemBlockThreads, args);
        return true;
    }

    private unsafe bool TryBornRatioNormalizeBackward(
        IGpuBuffer gradOutput, IGpuBuffer scanOutput, IGpuBuffer sinkLogAlphaOrDummy,
        IGpuBuffer gradScanOutput, IGpuBuffer gradSinkOrDummy,
        int totalRows, int numHeads, int headDim, int scanDim, int hasSink, float epsilon)
    {
        if (totalRows == 0)
        {
            if (hasSink != 0) Fill(gradSinkOrDummy, 0f, numHeads);
            return true;
        }
        if (!UseFusedSequenceMixer || totalRows < 0 || numHeads <= 0 || numHeads > CudaSequenceMixerKernels.MaxSingleBlockPerChannel || headDim <= 0 || headDim >= scanDim)
            return false;
        if (!_kernelCache.TryGetValue("smx_cuda_bornratio_bwd_dscan", out var dscanKernel)) return false;
        IntPtr dsinkKernel = IntPtr.Zero;
        if (hasSink != 0 && !_kernelCache.TryGetValue("smx_cuda_bornratio_bwd_dsink", out dsinkKernel))
            return false;

        using var _ = PushContext();
        uint scanBlocks = (uint)Math.Min(Math.Max(1, (totalRows + CudaSequenceMixerKernels.WarpsPerBlock - 1) / CudaSequenceMixerKernels.WarpsPerBlock), CudaSequenceMixerKernels.WarpHeadMaxGridBlocks);
        IntPtr dyPtr = gradOutput.Handle, sPtr = scanOutput.Handle, sinkPtr = sinkLogAlphaOrDummy.Handle;
        IntPtr dsPtr = gradScanOutput.Handle, dsinkPtr = gradSinkOrDummy.Handle;
        void** sArgs = stackalloc void*[10];
        sArgs[0] = &dyPtr; sArgs[1] = &sPtr; sArgs[2] = &sinkPtr; sArgs[3] = &dsPtr;
        sArgs[4] = &totalRows; sArgs[5] = &numHeads; sArgs[6] = &headDim; sArgs[7] = &scanDim; sArgs[8] = &hasSink; sArgs[9] = &epsilon;
        LaunchKernel(dscanKernel, scanBlocks, CudaSequenceMixerKernels.WarpBlockThreads, sArgs);

        if (hasSink != 0)
        {
            void** kArgs = stackalloc void*[9];
            kArgs[0] = &dyPtr; kArgs[1] = &sPtr; kArgs[2] = &sinkPtr; kArgs[3] = &dsinkPtr;
            kArgs[4] = &totalRows; kArgs[5] = &numHeads; kArgs[6] = &headDim; kArgs[7] = &scanDim; kArgs[8] = &epsilon;
            LaunchKernel(dsinkKernel, (uint)numHeads, CudaSequenceMixerKernels.WarpBlockThreads, kArgs);
        }
        return true;
    }

    private unsafe bool TryBoundedSigmoidGateForward(
        IGpuBuffer logits, IGpuBuffer bias, IGpuBuffer output, int rows, int cols, float floor)
    {
        int total = checked(rows * cols);
        if (total == 0) return true;
        if (!UseFusedSequenceMixer || rows < 0 || cols <= 0) return false;
        if (!_kernelCache.TryGetValue("smx_cuda_boundsig_fwd", out var kernel)) return false;

        using var _ = PushContext();
        uint blocks = (uint)Math.Min(Math.Max(1, (total + CudaSequenceMixerKernels.ElemBlockThreads - 1) / CudaSequenceMixerKernels.ElemBlockThreads), CudaSequenceMixerKernels.ElemMaxGridBlocks);
        IntPtr lPtr = logits.Handle, bPtr = bias.Handle, yPtr = output.Handle;
        void** args = stackalloc void*[6];
        args[0] = &lPtr; args[1] = &bPtr; args[2] = &yPtr; args[3] = &total; args[4] = &cols; args[5] = &floor;
        LaunchKernel(kernel, blocks, CudaSequenceMixerKernels.ElemBlockThreads, args);
        return true;
    }

    private unsafe bool TryBoundedSigmoidGateBackward(
        IGpuBuffer gradOutput, IGpuBuffer logits, IGpuBuffer bias, IGpuBuffer gradLogits, IGpuBuffer gradBias, int rows, int cols, float floor)
    {
        int total = checked(rows * cols);
        if (total == 0)
        {
            Fill(gradBias, 0f, cols);
            return true;
        }
        if (!UseFusedSequenceMixer || rows < 0 || cols <= 0 || cols > CudaSequenceMixerKernels.MaxSingleBlockPerChannel) return false;
        if (!_kernelCache.TryGetValue("smx_cuda_boundsig_bwd_dlog", out var dlogKernel)
            || !_kernelCache.TryGetValue("smx_cuda_boundsig_bwd_dbias", out var dbiasKernel))
            return false;

        using var _ = PushContext();
        uint blocks = (uint)Math.Min(Math.Max(1, (total + CudaSequenceMixerKernels.ElemBlockThreads - 1) / CudaSequenceMixerKernels.ElemBlockThreads), CudaSequenceMixerKernels.ElemMaxGridBlocks);
        IntPtr dyPtr = gradOutput.Handle, lPtr = logits.Handle, bPtr = bias.Handle, dlPtr = gradLogits.Handle, dbPtr = gradBias.Handle;
        void** lArgs = stackalloc void*[7];
        lArgs[0] = &dyPtr; lArgs[1] = &lPtr; lArgs[2] = &bPtr; lArgs[3] = &dlPtr; lArgs[4] = &total; lArgs[5] = &cols; lArgs[6] = &floor;
        LaunchKernel(dlogKernel, blocks, CudaSequenceMixerKernels.ElemBlockThreads, lArgs);

        void** bArgs = stackalloc void*[7];
        bArgs[0] = &dyPtr; bArgs[1] = &lPtr; bArgs[2] = &bPtr; bArgs[3] = &dbPtr; bArgs[4] = &rows; bArgs[5] = &cols; bArgs[6] = &floor;
        LaunchKernel(dbiasKernel, (uint)cols, CudaSequenceMixerKernels.WarpBlockThreads, bArgs);
        return true;
    }
}
