// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

public partial class DirectGpuTensorEngine
{
    public override Tensor<T> SplitHalfGivensRotation<T>(Tensor<T> input, Tensor<T> angles)
    {
        using var __gpuOp = EnterGpuOp();
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (angles is null) throw new ArgumentNullException(nameof(angles));
        if (input.Rank < 1) throw new ArgumentException("Input must have rank >= 1.", nameof(input));
        int d = input.Shape[input.Rank - 1];
        if (d <= 0 || (d & 1) != 0)
            throw new ArgumentException($"Expected positive even last dimension, got {d}.", nameof(input));
        int half = d / 2;
        if (angles.Length != half)
            throw new ArgumentException($"Expected angles length {half}, got {angles.Length}.", nameof(angles));

        int rows = input.Length / d;
        if (rows == 0 || GraphMode.IsActive || typeof(T) != typeof(float)
            || !TryGetBackend(out var backend) || backend is not IFusedSequenceMixerOps fused)
        {
            return base.SplitHalfGivensRotation(input, angles);
        }

        try
        {
            var srcIn = input.IsContiguous ? input : input.Contiguous();
            var srcAng = angles.IsContiguous ? angles : angles.Contiguous();
            using var inB = GetOrAllocateBuffer(backend, srcIn);
            using var angB = GetOrAllocateBuffer(backend, srcAng);
            using var outB = AllocateFullyWrittenOutputBuffer(backend, checked(rows * d));

            fused.SplitHalfGivensRotationForward(inB.Buffer, angB.Buffer, outB.Buffer, rows, half);
            var result = DeferTensorResult<T>(backend, outB.Buffer, rows * d, (int[])input._shape.Clone());
            outB.RelinquishOwnership();

            DifferentiableOps.RecordIfActive<T>(
                "SplitHalfGivensRotation",
                result,
                new[] { input, angles },
                static (gradOutput, inputs, output, savedState, engine, grads) =>
                {
                    int h = (int)savedState[0];
                    if (!DifferentiableOps._isBackwardCreateGraph
                        && engine.TryGet<DirectGpuTensorEngine>(out var gpu)
                        && gpu.TrySplitHalfGivensRotationBackwardGpu(gradOutput, inputs[0], inputs[1], h) is { } g)
                    {
                        DifferentiableOps.AccumulateGrad(grads, inputs[0], g.GradInput, engine);
                        DifferentiableOps.AccumulateGrad(grads, inputs[1], g.GradAngles, engine);
                        return;
                    }
                    CpuEngine.SplitHalfGivensRotationBackward(gradOutput, inputs, output, savedState, engine, grads);
                },
                savedState: new object[] { half });
            return result;
        }
        catch (NotSupportedException capabilityGap)
        {
            GpuLaunchProbe.OnFallback("SplitHalfGivensRotation", capabilityGap);
            return base.SplitHalfGivensRotation(input, angles);
        }
    }

    private (Tensor<T> GradInput, Tensor<T> GradAngles)? TrySplitHalfGivensRotationBackwardGpu<T>(
        Tensor<T> gradOutput, Tensor<T> input, Tensor<T> angles, int half)
    {
        if (typeof(T) != typeof(float) || !TryGetBackend(out var backend) || backend is not IFusedSequenceMixerOps fused)
            return null;
        int d = 2 * half;
        int rows = input.Length / d;
        if (rows <= 0 || gradOutput.Length != rows * d || angles.Length != half) return null;

        try
        {
            var srcDy = gradOutput.IsContiguous ? gradOutput : gradOutput.Contiguous();
            var srcIn = input.IsContiguous ? input : input.Contiguous();
            var srcAng = angles.IsContiguous ? angles : angles.Contiguous();
            using var dyB = GetOrAllocateBuffer(backend, srcDy);
            using var inB = GetOrAllocateBuffer(backend, srcIn);
            using var angB = GetOrAllocateBuffer(backend, srcAng);
            using var dxB = AllocateFullyWrittenOutputBuffer(backend, rows * d);
            using var daB = AllocateFullyWrittenOutputBuffer(backend, half);

            fused.SplitHalfGivensRotationBackward(dyB.Buffer, inB.Buffer, angB.Buffer, dxB.Buffer, daB.Buffer, rows, half);
            var gIn = DeferTensorResult<T>(backend, dxB.Buffer, rows * d, (int[])input._shape.Clone());
            dxB.RelinquishOwnership();
            var gAng = DeferTensorResult<T>(backend, daB.Buffer, half, (int[])angles._shape.Clone());
            daB.RelinquishOwnership();
            return (gIn, gAng);
        }
        catch (NotSupportedException capabilityGap)
        {
            GpuLaunchProbe.OnFallback("SplitHalfGivensRotationBackward", capabilityGap);
            return null;
        }
    }

    public override Tensor<T> DepthwiseCausalConv1d2Tap<T>(Tensor<T> input, Tensor<T> weight0, Tensor<T> weight1)
    {
        using var __gpuOp = EnterGpuOp();
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (weight0 is null) throw new ArgumentNullException(nameof(weight0));
        if (weight1 is null) throw new ArgumentNullException(nameof(weight1));
        if (input.Rank != 2 && input.Rank != 3)
            throw new ArgumentException($"Input must have rank 2 [S, C] or rank 3 [B, S, C], got rank {input.Rank}.", nameof(input));

        int batch = input.Rank == 3 ? input.Shape[0] : 1;
        int seqLen = input.Shape[input.Rank - 2];
        int channels = input.Shape[input.Rank - 1];
        if (batch < 0 || seqLen < 0 || channels <= 0)
            throw new ArgumentException("Invalid input dimensions.", nameof(input));
        if (weight0.Length != channels)
            throw new ArgumentException($"Expected weight0 length {channels}, got {weight0.Length}.", nameof(weight0));
        if (weight1.Length != channels)
            throw new ArgumentException($"Expected weight1 length {channels}, got {weight1.Length}.", nameof(weight1));

        int total = checked(batch * seqLen * channels);
        if (total == 0 || GraphMode.IsActive || typeof(T) != typeof(float)
            || !TryGetBackend(out var backend) || backend is not IFusedSequenceMixerOps fused)
        {
            return base.DepthwiseCausalConv1d2Tap(input, weight0, weight1);
        }

        try
        {
            var srcIn = input.IsContiguous ? input : input.Contiguous();
            var srcW0 = weight0.IsContiguous ? weight0 : weight0.Contiguous();
            var srcW1 = weight1.IsContiguous ? weight1 : weight1.Contiguous();
            using var inB = GetOrAllocateBuffer(backend, srcIn);
            using var w0B = GetOrAllocateBuffer(backend, srcW0);
            using var w1B = GetOrAllocateBuffer(backend, srcW1);
            using var outB = AllocateFullyWrittenOutputBuffer(backend, total);

            fused.DepthwiseCausalConv1d2TapForward(inB.Buffer, w0B.Buffer, w1B.Buffer, outB.Buffer, batch, seqLen, channels);
            var result = DeferTensorResult<T>(backend, outB.Buffer, total, (int[])input._shape.Clone());
            outB.RelinquishOwnership();

            DifferentiableOps.RecordIfActive<T>(
                "DepthwiseCausalConv1d2Tap",
                result,
                new[] { input, weight0, weight1 },
                static (gradOutput, inputs, output, savedState, engine, grads) =>
                {
                    int b = (int)savedState[0];
                    int s = (int)savedState[1];
                    int c = (int)savedState[2];
                    if (!DifferentiableOps._isBackwardCreateGraph
                        && engine.TryGet<DirectGpuTensorEngine>(out var gpu)
                        && gpu.TryDepthwiseCausalConv1d2TapBackwardGpu(gradOutput, inputs[0], inputs[1], inputs[2], b, s, c) is { } g)
                    {
                        DifferentiableOps.AccumulateGrad(grads, inputs[0], g.GradInput, engine);
                        DifferentiableOps.AccumulateGrad(grads, inputs[1], g.GradW0, engine);
                        DifferentiableOps.AccumulateGrad(grads, inputs[2], g.GradW1, engine);
                        return;
                    }
                    CpuEngine.DepthwiseCausalConv1d2TapBackward(gradOutput, inputs, output, savedState, engine, grads);
                },
                savedState: new object[] { batch, seqLen, channels });
            return result;
        }
        catch (NotSupportedException capabilityGap)
        {
            GpuLaunchProbe.OnFallback("DepthwiseCausalConv1d2Tap", capabilityGap);
            return base.DepthwiseCausalConv1d2Tap(input, weight0, weight1);
        }
    }

    private (Tensor<T> GradInput, Tensor<T> GradW0, Tensor<T> GradW1)? TryDepthwiseCausalConv1d2TapBackwardGpu<T>(
        Tensor<T> gradOutput, Tensor<T> input, Tensor<T> weight0, Tensor<T> weight1, int batch, int seqLen, int channels)
    {
        if (typeof(T) != typeof(float) || !TryGetBackend(out var backend) || backend is not IFusedSequenceMixerOps fused)
            return null;
        int total = batch * seqLen * channels;
        if (total <= 0 || gradOutput.Length != total) return null;

        try
        {
            var srcDy = gradOutput.IsContiguous ? gradOutput : gradOutput.Contiguous();
            var srcIn = input.IsContiguous ? input : input.Contiguous();
            var srcW0 = weight0.IsContiguous ? weight0 : weight0.Contiguous();
            var srcW1 = weight1.IsContiguous ? weight1 : weight1.Contiguous();
            using var dyB = GetOrAllocateBuffer(backend, srcDy);
            using var inB = GetOrAllocateBuffer(backend, srcIn);
            using var w0B = GetOrAllocateBuffer(backend, srcW0);
            using var w1B = GetOrAllocateBuffer(backend, srcW1);
            using var dxB = AllocateFullyWrittenOutputBuffer(backend, total);
            using var dw0B = AllocateFullyWrittenOutputBuffer(backend, channels);
            using var dw1B = AllocateFullyWrittenOutputBuffer(backend, channels);

            fused.DepthwiseCausalConv1d2TapBackward(dyB.Buffer, inB.Buffer, w0B.Buffer, w1B.Buffer, dxB.Buffer, dw0B.Buffer, dw1B.Buffer, batch, seqLen, channels);
            var gIn = DeferTensorResult<T>(backend, dxB.Buffer, total, (int[])input._shape.Clone());
            dxB.RelinquishOwnership();
            var gW0 = DeferTensorResult<T>(backend, dw0B.Buffer, channels, (int[])weight0._shape.Clone());
            dw0B.RelinquishOwnership();
            var gW1 = DeferTensorResult<T>(backend, dw1B.Buffer, channels, (int[])weight1._shape.Clone());
            dw1B.RelinquishOwnership();
            return (gIn, gW0, gW1);
        }
        catch (NotSupportedException capabilityGap)
        {
            GpuLaunchProbe.OnFallback("DepthwiseCausalConv1d2TapBackward", capabilityGap);
            return null;
        }
    }

    public override Tensor<T> RopeL2NormalizeHeads<T>(
        Tensor<T> input,
        Tensor<T> cos,
        Tensor<T> sin,
        int numHeads,
        double epsilon = 1e-6,
        double postScale = 1.0,
        int startPosition = 0)
    {
        using var __gpuOp = EnterGpuOp();
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (cos is null) throw new ArgumentNullException(nameof(cos));
        if (sin is null) throw new ArgumentNullException(nameof(sin));
        if (input.Rank != 2 && input.Rank != 3)
            throw new ArgumentException($"Input must have rank 2 [S, H*Dh] or rank 3 [B, S, H*Dh], got rank {input.Rank}.", nameof(input));
        if (numHeads <= 0) throw new ArgumentOutOfRangeException(nameof(numHeads));

        int batch = input.Rank == 3 ? input.Shape[0] : 1;
        int seqLen = input.Shape[input.Rank - 2];
        int totalDim = input.Shape[input.Rank - 1];
        if (totalDim <= 0 || (totalDim % numHeads) != 0)
            throw new ArgumentException($"Last dimension {totalDim} must be divisible by numHeads ({numHeads}).", nameof(input));
        int headDim = totalDim / numHeads;
        if ((headDim & 1) != 0)
            throw new ArgumentException($"Head dimension ({headDim}) must be even.", nameof(input));
        int half = headDim / 2;
        if (startPosition < 0) throw new ArgumentOutOfRangeException(nameof(startPosition));
        if (cos.Rank != 2 || sin.Rank != 2 || cos.Shape[1] != half || sin.Shape[1] != half
            || cos.Shape[0] < startPosition + seqLen || sin.Shape[0] < startPosition + seqLen)
            throw new ArgumentException($"cos/sin must have shape [>= {startPosition + seqLen}, {half}].", nameof(cos));
        if (!(epsilon > 0.0)) throw new ArgumentOutOfRangeException(nameof(epsilon));

        int total = checked(batch * seqLen * totalDim);
        if (total == 0 || GraphMode.IsActive || typeof(T) != typeof(float)
            || !TryGetBackend(out var backend) || backend is not IFusedSequenceMixerOps fused)
        {
            return base.RopeL2NormalizeHeads(input, cos, sin, numHeads, epsilon, postScale, startPosition);
        }

        try
        {
            var srcIn = input.IsContiguous ? input : input.Contiguous();
            var srcCos = cos.IsContiguous ? cos : cos.Contiguous();
            var srcSin = sin.IsContiguous ? sin : sin.Contiguous();
            using var inB = GetOrAllocateBuffer(backend, srcIn);
            using var cosB = GetOrAllocateBuffer(backend, srcCos);
            using var sinB = GetOrAllocateBuffer(backend, srcSin);
            using var outB = AllocateFullyWrittenOutputBuffer(backend, total);

            fused.RopeL2NormalizeHeadsForward(
                inB.Buffer, cosB.Buffer, sinB.Buffer, outB.Buffer,
                batch, seqLen, numHeads, headDim, startPosition, (float)epsilon, (float)postScale);
            var result = DeferTensorResult<T>(backend, outB.Buffer, total, (int[])input._shape.Clone());
            outB.RelinquishOwnership();

            DifferentiableOps.RecordIfActive<T>(
                "RopeL2NormalizeHeads",
                result,
                new[] { input },
                static (gradOutput, inputs, output, savedState, engine, grads) =>
                {
                    var c = (Tensor<T>)savedState[0];
                    var s = (Tensor<T>)savedState[1];
                    int b = (int)savedState[2];
                    int sl = (int)savedState[3];
                    int nh = (int)savedState[4];
                    int dh = (int)savedState[5];
                    double eps = (double)savedState[6];
                    double sc = (double)savedState[7];
                    int sp = (int)savedState[8];
                    if (!DifferentiableOps._isBackwardCreateGraph
                        && engine.TryGet<DirectGpuTensorEngine>(out var gpu)
                        && gpu.TryRopeL2NormalizeHeadsBackwardGpu(gradOutput, inputs[0], c, s, b, sl, nh, dh, sp, eps, sc) is { } gIn)
                    {
                        DifferentiableOps.AccumulateGrad(grads, inputs[0], gIn, engine);
                        return;
                    }
                    CpuEngine.RopeL2NormalizeHeadsBackward(gradOutput, inputs, output, savedState, engine, grads);
                },
                savedState: new object[] { cos, sin, batch, seqLen, numHeads, headDim, epsilon, postScale, startPosition });
            return result;
        }
        catch (NotSupportedException capabilityGap)
        {
            GpuLaunchProbe.OnFallback("RopeL2NormalizeHeads", capabilityGap);
            return base.RopeL2NormalizeHeads(input, cos, sin, numHeads, epsilon, postScale, startPosition);
        }
    }

    private Tensor<T>? TryRopeL2NormalizeHeadsBackwardGpu<T>(
        Tensor<T> gradOutput, Tensor<T> input, Tensor<T> cos, Tensor<T> sin,
        int batch, int seqLen, int numHeads, int headDim, int startPosition, double epsilon, double postScale)
    {
        if (typeof(T) != typeof(float) || !TryGetBackend(out var backend) || backend is not IFusedSequenceMixerOps fused)
            return null;
        int total = batch * seqLen * numHeads * headDim;
        if (total <= 0 || gradOutput.Length != total) return null;

        try
        {
            var srcDy = gradOutput.IsContiguous ? gradOutput : gradOutput.Contiguous();
            var srcIn = input.IsContiguous ? input : input.Contiguous();
            var srcCos = cos.IsContiguous ? cos : cos.Contiguous();
            var srcSin = sin.IsContiguous ? sin : sin.Contiguous();
            using var dyB = GetOrAllocateBuffer(backend, srcDy);
            using var inB = GetOrAllocateBuffer(backend, srcIn);
            using var cosB = GetOrAllocateBuffer(backend, srcCos);
            using var sinB = GetOrAllocateBuffer(backend, srcSin);
            using var dxB = AllocateFullyWrittenOutputBuffer(backend, total);

            fused.RopeL2NormalizeHeadsBackward(
                dyB.Buffer, inB.Buffer, cosB.Buffer, sinB.Buffer, dxB.Buffer,
                batch, seqLen, numHeads, headDim, startPosition, (float)epsilon, (float)postScale);
            var gIn = DeferTensorResult<T>(backend, dxB.Buffer, total, (int[])input._shape.Clone());
            dxB.RelinquishOwnership();
            return gIn;
        }
        catch (NotSupportedException capabilityGap)
        {
            GpuLaunchProbe.OnFallback("RopeL2NormalizeHeadsBackward", capabilityGap);
            return null;
        }
    }

    public override Tensor<T> RopeQuadraticOuterFeatures<T>(
        Tensor<T> input,
        Tensor<T> cos,
        Tensor<T> sin,
        int numHeads,
        int outputHeadDim = 0,
        int startPosition = 0)
    {
        using var __gpuOp = EnterGpuOp();
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (cos is null) throw new ArgumentNullException(nameof(cos));
        if (sin is null) throw new ArgumentNullException(nameof(sin));
        if (input.Rank != 2 && input.Rank != 3)
            throw new ArgumentException($"Input must have rank 2 [S, H*Mb] or rank 3 [B, S, H*Mb], got rank {input.Rank}.", nameof(input));
        if (numHeads <= 0) throw new ArgumentOutOfRangeException(nameof(numHeads));

        int batch = input.Rank == 3 ? input.Shape[0] : 1;
        int seqLen = input.Shape[input.Rank - 2];
        int totalIn = input.Shape[input.Rank - 1];
        if (totalIn <= 0 || (totalIn % numHeads) != 0)
            throw new ArgumentException($"Last dimension {totalIn} must be divisible by numHeads ({numHeads}).", nameof(input));
        int mb = totalIn / numHeads;
        if ((mb & 1) != 0)
            throw new ArgumentException($"Feature dimension per head ({mb}) must be even.", nameof(input));
        int mb2 = checked(mb * mb);
        int effOutDim = outputHeadDim > 0 ? outputHeadDim : mb2;
        if (effOutDim < mb2)
            throw new ArgumentOutOfRangeException(nameof(outputHeadDim), $"outputHeadDim ({effOutDim}) must be >= featureDim^2 ({mb2}).");

        int half = mb / 2;
        if (startPosition < 0) throw new ArgumentOutOfRangeException(nameof(startPosition));
        if (cos.Rank != 2 || sin.Rank != 2 || cos.Shape[1] != half || sin.Shape[1] != half
            || cos.Shape[0] < startPosition + seqLen || sin.Shape[0] < startPosition + seqLen)
            throw new ArgumentException($"cos/sin must have shape [>= {startPosition + seqLen}, {half}].", nameof(cos));

        int totalOut = checked(batch * seqLen * numHeads * effOutDim);
        if (totalOut == 0 || GraphMode.IsActive || typeof(T) != typeof(float)
            || !TryGetBackend(out var backend) || backend is not IFusedSequenceMixerOps fused)
        {
            return base.RopeQuadraticOuterFeatures(input, cos, sin, numHeads, outputHeadDim, startPosition);
        }

        int[] outShape = input.Rank == 3
            ? new[] { batch, seqLen, checked(numHeads * effOutDim) }
            : new[] { seqLen, checked(numHeads * effOutDim) };

        try
        {
            var srcIn = input.IsContiguous ? input : input.Contiguous();
            var srcCos = cos.IsContiguous ? cos : cos.Contiguous();
            var srcSin = sin.IsContiguous ? sin : sin.Contiguous();
            using var inB = GetOrAllocateBuffer(backend, srcIn);
            using var cosB = GetOrAllocateBuffer(backend, srcCos);
            using var sinB = GetOrAllocateBuffer(backend, srcSin);
            using var outB = AllocateFullyWrittenOutputBuffer(backend, totalOut);

            fused.RopeQuadraticOuterFeaturesForward(
                inB.Buffer, cosB.Buffer, sinB.Buffer, outB.Buffer,
                batch, seqLen, numHeads, mb, effOutDim, startPosition);
            var result = DeferTensorResult<T>(backend, outB.Buffer, totalOut, outShape);
            outB.RelinquishOwnership();

            DifferentiableOps.RecordIfActive<T>(
                "RopeQuadraticOuterFeatures",
                result,
                new[] { input },
                static (gradOutput, inputs, output, savedState, engine, grads) =>
                {
                    var c = (Tensor<T>)savedState[0];
                    var s = (Tensor<T>)savedState[1];
                    int b = (int)savedState[2];
                    int sl = (int)savedState[3];
                    int nh = (int)savedState[4];
                    int m = (int)savedState[5];
                    int od = (int)savedState[6];
                    int sp = (int)savedState[7];
                    if (!DifferentiableOps._isBackwardCreateGraph
                        && engine.TryGet<DirectGpuTensorEngine>(out var gpu)
                        && gpu.TryRopeQuadraticOuterFeaturesBackwardGpu(gradOutput, inputs[0], c, s, b, sl, nh, m, od, sp) is { } gIn)
                    {
                        DifferentiableOps.AccumulateGrad(grads, inputs[0], gIn, engine);
                        return;
                    }
                    CpuEngine.RopeQuadraticOuterFeaturesBackward(gradOutput, inputs, output, savedState, engine, grads);
                },
                savedState: new object[] { cos, sin, batch, seqLen, numHeads, mb, effOutDim, startPosition, outShape });
            return result;
        }
        catch (NotSupportedException capabilityGap)
        {
            GpuLaunchProbe.OnFallback("RopeQuadraticOuterFeatures", capabilityGap);
            return base.RopeQuadraticOuterFeatures(input, cos, sin, numHeads, outputHeadDim, startPosition);
        }
    }

    private Tensor<T>? TryRopeQuadraticOuterFeaturesBackwardGpu<T>(
        Tensor<T> gradOutput, Tensor<T> input, Tensor<T> cos, Tensor<T> sin,
        int batch, int seqLen, int numHeads, int mb, int effOutDim, int startPosition)
    {
        if (typeof(T) != typeof(float) || !TryGetBackend(out var backend) || backend is not IFusedSequenceMixerOps fused)
            return null;
        int totalIn = batch * seqLen * numHeads * mb;
        int totalOut = batch * seqLen * numHeads * effOutDim;
        if (totalIn <= 0 || gradOutput.Length != totalOut) return null;

        try
        {
            var srcDy = gradOutput.IsContiguous ? gradOutput : gradOutput.Contiguous();
            var srcIn = input.IsContiguous ? input : input.Contiguous();
            var srcCos = cos.IsContiguous ? cos : cos.Contiguous();
            var srcSin = sin.IsContiguous ? sin : sin.Contiguous();
            using var dyB = GetOrAllocateBuffer(backend, srcDy);
            using var inB = GetOrAllocateBuffer(backend, srcIn);
            using var cosB = GetOrAllocateBuffer(backend, srcCos);
            using var sinB = GetOrAllocateBuffer(backend, srcSin);
            using var dxB = AllocateFullyWrittenOutputBuffer(backend, totalIn);

            fused.RopeQuadraticOuterFeaturesBackward(
                dyB.Buffer, inB.Buffer, cosB.Buffer, sinB.Buffer, dxB.Buffer,
                batch, seqLen, numHeads, mb, effOutDim, startPosition);
            var gIn = DeferTensorResult<T>(backend, dxB.Buffer, totalIn, (int[])input._shape.Clone());
            dxB.RelinquishOwnership();
            return gIn;
        }
        catch (NotSupportedException capabilityGap)
        {
            GpuLaunchProbe.OnFallback("RopeQuadraticOuterFeaturesBackward", capabilityGap);
            return null;
        }
    }

    public override Tensor<T> ProjectiveGatedResidual<T>(
        Tensor<T> baseHidden,
        Tensor<T> proposal,
        Tensor<T> parallelGateLogit,
        double epsilon = 1e-6)
    {
        using var __gpuOp = EnterGpuOp();
        if (baseHidden is null) throw new ArgumentNullException(nameof(baseHidden));
        if (proposal is null) throw new ArgumentNullException(nameof(proposal));
        if (parallelGateLogit is null) throw new ArgumentNullException(nameof(parallelGateLogit));
        if (baseHidden.Rank < 1 || proposal.Rank != baseHidden.Rank)
            throw new ArgumentException("baseHidden and proposal must have matching rank >= 1.", nameof(proposal));
        for (int i = 0; i < baseHidden.Rank; i++)
        {
            if (baseHidden.Shape[i] != proposal.Shape[i])
                throw new ArgumentException("baseHidden and proposal shapes must match.", nameof(proposal));
        }
        int dim = baseHidden.Shape[baseHidden.Rank - 1];
        if (dim <= 0) throw new ArgumentException("Last dimension must be positive.", nameof(baseHidden));
        if (parallelGateLogit.Length != 1)
            throw new ArgumentException($"parallelGateLogit must have length 1, got {parallelGateLogit.Length}.", nameof(parallelGateLogit));
        if (!(epsilon > 0.0)) throw new ArgumentOutOfRangeException(nameof(epsilon));

        int rows = baseHidden.Length / dim;
        if (rows == 0 || GraphMode.IsActive || typeof(T) != typeof(float)
            || !TryGetBackend(out var backend) || backend is not IFusedSequenceMixerOps fused)
        {
            return base.ProjectiveGatedResidual(baseHidden, proposal, parallelGateLogit, epsilon);
        }

        try
        {
            var srcH = baseHidden.IsContiguous ? baseHidden : baseHidden.Contiguous();
            var srcU = proposal.IsContiguous ? proposal : proposal.Contiguous();
            var srcG = parallelGateLogit.IsContiguous ? parallelGateLogit : parallelGateLogit.Contiguous();
            using var hB = GetOrAllocateBuffer(backend, srcH);
            using var uB = GetOrAllocateBuffer(backend, srcU);
            using var gB = GetOrAllocateBuffer(backend, srcG);
            using var outB = AllocateFullyWrittenOutputBuffer(backend, rows * dim);

            fused.ProjectiveGatedResidualForward(hB.Buffer, uB.Buffer, gB.Buffer, outB.Buffer, rows, dim, (float)epsilon);
            var result = DeferTensorResult<T>(backend, outB.Buffer, rows * dim, (int[])baseHidden._shape.Clone());
            outB.RelinquishOwnership();

            DifferentiableOps.RecordIfActive<T>(
                "ProjectiveGatedResidual",
                result,
                new[] { baseHidden, proposal, parallelGateLogit },
                static (gradOutput, inputs, output, savedState, engine, grads) =>
                {
                    int d = (int)savedState[0];
                    double eps = (double)savedState[1];
                    if (!DifferentiableOps._isBackwardCreateGraph
                        && engine.TryGet<DirectGpuTensorEngine>(out var gpu)
                        && gpu.TryProjectiveGatedResidualBackwardGpu(gradOutput, inputs[0], inputs[1], inputs[2], d, eps) is { } g)
                    {
                        DifferentiableOps.AccumulateGrad(grads, inputs[0], g.GradH, engine);
                        DifferentiableOps.AccumulateGrad(grads, inputs[1], g.GradU, engine);
                        DifferentiableOps.AccumulateGrad(grads, inputs[2], g.GradGate, engine);
                        return;
                    }
                    CpuEngine.ProjectiveGatedResidualBackward(gradOutput, inputs, output, savedState, engine, grads);
                },
                savedState: new object[] { dim, epsilon });
            return result;
        }
        catch (NotSupportedException capabilityGap)
        {
            GpuLaunchProbe.OnFallback("ProjectiveGatedResidual", capabilityGap);
            return base.ProjectiveGatedResidual(baseHidden, proposal, parallelGateLogit, epsilon);
        }
    }

    private (Tensor<T> GradH, Tensor<T> GradU, Tensor<T> GradGate)? TryProjectiveGatedResidualBackwardGpu<T>(
        Tensor<T> gradOutput, Tensor<T> baseHidden, Tensor<T> proposal, Tensor<T> gateLogit, int dim, double epsilon)
    {
        if (typeof(T) != typeof(float) || !TryGetBackend(out var backend) || backend is not IFusedSequenceMixerOps fused)
            return null;
        int rows = baseHidden.Length / dim;
        if (rows <= 0 || gradOutput.Length != rows * dim || gateLogit.Length != 1) return null;

        try
        {
            var srcDy = gradOutput.IsContiguous ? gradOutput : gradOutput.Contiguous();
            var srcH = baseHidden.IsContiguous ? baseHidden : baseHidden.Contiguous();
            var srcU = proposal.IsContiguous ? proposal : proposal.Contiguous();
            var srcG = gateLogit.IsContiguous ? gateLogit : gateLogit.Contiguous();
            using var dyB = GetOrAllocateBuffer(backend, srcDy);
            using var hB = GetOrAllocateBuffer(backend, srcH);
            using var uB = GetOrAllocateBuffer(backend, srcU);
            using var gB = GetOrAllocateBuffer(backend, srcG);
            using var dhB = AllocateFullyWrittenOutputBuffer(backend, rows * dim);
            using var duB = AllocateFullyWrittenOutputBuffer(backend, rows * dim);
            using var dgB = AllocateFullyWrittenOutputBuffer(backend, 1);

            fused.ProjectiveGatedResidualBackward(
                dyB.Buffer, hB.Buffer, uB.Buffer, gB.Buffer, dhB.Buffer, duB.Buffer, dgB.Buffer,
                rows, dim, (float)epsilon);
            var gH = DeferTensorResult<T>(backend, dhB.Buffer, rows * dim, (int[])baseHidden._shape.Clone());
            dhB.RelinquishOwnership();
            var gU = DeferTensorResult<T>(backend, duB.Buffer, rows * dim, (int[])proposal._shape.Clone());
            duB.RelinquishOwnership();
            var gG = DeferTensorResult<T>(backend, dgB.Buffer, 1, (int[])gateLogit._shape.Clone());
            dgB.RelinquishOwnership();
            return (gH, gU, gG);
        }
        catch (NotSupportedException capabilityGap)
        {
            GpuLaunchProbe.OnFallback("ProjectiveGatedResidualBackward", capabilityGap);
            return null;
        }
    }

    public override Tensor<T> SpectralCirculantMatrix<T>(Tensor<T> diagScale, Tensor<T> spectrumRe, Tensor<T> spectrumIm)
    {
        using var __gpuOp = EnterGpuOp();
        if (diagScale is null) throw new ArgumentNullException(nameof(diagScale));
        if (spectrumRe is null) throw new ArgumentNullException(nameof(spectrumRe));
        if (spectrumIm is null) throw new ArgumentNullException(nameof(spectrumIm));
        if (diagScale.Rank != 1 && diagScale.Rank != 2)
            throw new ArgumentException($"Expected diagScale rank 1 [D] or 2 [B, D], got {diagScale.Rank}.", nameof(diagScale));
        int batch = diagScale.Rank == 2 ? diagScale.Shape[0] : 1;
        int dim = diagScale.Shape[diagScale.Rank - 1];
        if (batch < 0 || dim <= 0 || (dim & 1) != 0)
            throw new ArgumentException($"Expected positive even dim, got batch={batch}, dim={dim}.", nameof(diagScale));
        int nf = dim / 2 + 1;
        if (spectrumRe.Length != checked(batch * nf))
            throw new ArgumentException($"Expected spectrumRe length {batch * nf}, got {spectrumRe.Length}.", nameof(spectrumRe));
        if (spectrumIm.Length != batch * nf)
            throw new ArgumentException($"Expected spectrumIm length {batch * nf}, got {spectrumIm.Length}.", nameof(spectrumIm));

        int totalOut = checked(batch * dim * dim);
        if (totalOut == 0 || GraphMode.IsActive || typeof(T) != typeof(float)
            || !TryGetBackend(out var backend) || backend is not IFusedSequenceMixerOps fused)
        {
            return base.SpectralCirculantMatrix(diagScale, spectrumRe, spectrumIm);
        }

        try
        {
            var sD = diagScale.IsContiguous ? diagScale : diagScale.Contiguous();
            var sRe = spectrumRe.IsContiguous ? spectrumRe : spectrumRe.Contiguous();
            var sIm = spectrumIm.IsContiguous ? spectrumIm : spectrumIm.Contiguous();
            using var dB = GetOrAllocateBuffer(backend, sD);
            using var reB = GetOrAllocateBuffer(backend, sRe);
            using var imB = GetOrAllocateBuffer(backend, sIm);
            using var outB = AllocateFullyWrittenOutputBuffer(backend, totalOut);

            fused.SpectralCirculantMatrixForward(dB.Buffer, reB.Buffer, imB.Buffer, outB.Buffer, batch, dim);
            int[] outShape = diagScale.Rank == 2 ? new[] { batch, dim, dim } : new[] { dim, dim };
            var result = DeferTensorResult<T>(backend, outB.Buffer, totalOut, outShape);
            outB.RelinquishOwnership();

            DifferentiableOps.RecordIfActive<T>(
                "SpectralCirculantMatrix",
                result,
                new[] { diagScale, spectrumRe, spectrumIm },
                static (gradOutput, inputs, output, savedState, engine, grads) =>
                {
                    int b = (int)savedState[0];
                    int d = (int)savedState[1];
                    if (!DifferentiableOps._isBackwardCreateGraph
                        && engine.TryGet<DirectGpuTensorEngine>(out var gpu)
                        && gpu.TrySpectralCirculantMatrixBackwardGpu(gradOutput, inputs[0], inputs[1], inputs[2], b, d) is { } g)
                    {
                        DifferentiableOps.AccumulateGrad(grads, inputs[0], g.GradDiag, engine);
                        DifferentiableOps.AccumulateGrad(grads, inputs[1], g.GradRe, engine);
                        DifferentiableOps.AccumulateGrad(grads, inputs[2], g.GradIm, engine);
                        return;
                    }
                    CpuEngine.SpectralCirculantMatrixBackward(gradOutput, inputs, output, savedState, engine, grads);
                },
                savedState: new object[] { batch, dim });
            return result;
        }
        catch (NotSupportedException capabilityGap)
        {
            GpuLaunchProbe.OnFallback("SpectralCirculantMatrix", capabilityGap);
            return base.SpectralCirculantMatrix(diagScale, spectrumRe, spectrumIm);
        }
    }

    private (Tensor<T> GradDiag, Tensor<T> GradRe, Tensor<T> GradIm)? TrySpectralCirculantMatrixBackwardGpu<T>(
        Tensor<T> gradOutput, Tensor<T> diagScale, Tensor<T> spectrumRe, Tensor<T> spectrumIm, int batch, int dim)
    {
        if (typeof(T) != typeof(float) || !TryGetBackend(out var backend) || backend is not IFusedSequenceMixerOps fused)
            return null;
        int nf = dim / 2 + 1;
        if (batch <= 0 || gradOutput.Length != batch * dim * dim || diagScale.Length != batch * dim)
            return null;

        try
        {
            var sDy = gradOutput.IsContiguous ? gradOutput : gradOutput.Contiguous();
            var sD = diagScale.IsContiguous ? diagScale : diagScale.Contiguous();
            var sRe = spectrumRe.IsContiguous ? spectrumRe : spectrumRe.Contiguous();
            var sIm = spectrumIm.IsContiguous ? spectrumIm : spectrumIm.Contiguous();
            using var dyB = GetOrAllocateBuffer(backend, sDy);
            using var dB = GetOrAllocateBuffer(backend, sD);
            using var reB = GetOrAllocateBuffer(backend, sRe);
            using var imB = GetOrAllocateBuffer(backend, sIm);
            using var gdB = AllocateFullyWrittenOutputBuffer(backend, batch * dim);
            using var greB = AllocateFullyWrittenOutputBuffer(backend, batch * nf);
            using var gimB = AllocateFullyWrittenOutputBuffer(backend, batch * nf);

            fused.SpectralCirculantMatrixBackward(
                dyB.Buffer, dB.Buffer, reB.Buffer, imB.Buffer, gdB.Buffer, greB.Buffer, gimB.Buffer, batch, dim);
            var gDiag = DeferTensorResult<T>(backend, gdB.Buffer, batch * dim, (int[])diagScale._shape.Clone());
            gdB.RelinquishOwnership();
            var gRe = DeferTensorResult<T>(backend, greB.Buffer, batch * nf, (int[])spectrumRe._shape.Clone());
            greB.RelinquishOwnership();
            var gIm = DeferTensorResult<T>(backend, gimB.Buffer, batch * nf, (int[])spectrumIm._shape.Clone());
            gimB.RelinquishOwnership();
            return (gDiag, gRe, gIm);
        }
        catch (NotSupportedException capabilityGap)
        {
            GpuLaunchProbe.OnFallback("SpectralCirculantMatrixBackward", capabilityGap);
            return null;
        }
    }

    public override Tensor<T> SigmoidTopKExpertMix<T>(
        Tensor<T> logits, Tensor<T> expertTable, Tensor<T> routingBias, int topK, double epsilon = 1e-12, double biasUpdateRate = 0.0)
    {
        using var __gpuOp = EnterGpuOp();
        if (logits is null) throw new ArgumentNullException(nameof(logits));
        if (expertTable is null) throw new ArgumentNullException(nameof(expertTable));
        if (routingBias is null) throw new ArgumentNullException(nameof(routingBias));
        if (logits.Rank != 2 || expertTable.Rank != 2)
            throw new ArgumentException("logits [N, G] and expertTable [G, C] must be rank 2.");
        int rows = logits.Shape[0], numExperts = logits.Shape[1], channels = expertTable.Shape[1];
        if (expertTable.Shape[0] != numExperts)
            throw new ArgumentException($"expertTable rows {expertTable.Shape[0]} != numExperts {numExperts}.", nameof(expertTable));
        if (routingBias.Length != numExperts)
            throw new ArgumentException($"routingBias length {routingBias.Length} != numExperts {numExperts}.", nameof(routingBias));
        if (topK < 1 || topK > numExperts)
            throw new ArgumentOutOfRangeException(nameof(topK));

        double effRate = NoGradScope<T>.IsSuppressed ? 0.0 : biasUpdateRate;
        int totalOut = checked(rows * channels);
        if (totalOut == 0 || GraphMode.IsActive || typeof(T) != typeof(float)
            || !TryGetBackend(out var backend) || backend is not IFusedSequenceMixerOps fused)
        {
            return base.SigmoidTopKExpertMix(logits, expertTable, routingBias, topK, epsilon, biasUpdateRate);
        }

        try
        {
            var sLog = logits.IsContiguous ? logits : logits.Contiguous();
            var sTbl = expertTable.IsContiguous ? expertTable : expertTable.Contiguous();
            using var logB = GetOrAllocateBuffer(backend, sLog);
            using var tblB = GetOrAllocateBuffer(backend, sTbl);

            IGpuBuffer? persistentBiasBuf = null;
            OwnedBuffer? tempBiasLease = null;
            IGpuBuffer activeBiasBuf;
            if (effRate != 0.0)
            {
                var biasFloat = (Tensor<float>)(object)routingBias;
                biasFloat.EnsureOwnedForDeviceWrite();
                persistentBiasBuf = AcquireParameterBuffer(biasFloat, out _);
                if (persistentBiasBuf is not null)
                {
                    activeBiasBuf = persistentBiasBuf;
                }
                else
                {
                    return base.SigmoidTopKExpertMix(logits, expertTable, routingBias, topK, epsilon, biasUpdateRate);
                }
            }
            else
            {
                var sBias = routingBias.IsContiguous ? routingBias : routingBias.Contiguous();
                var lease = GetOrAllocateBuffer(backend, sBias);
                tempBiasLease = lease;
                activeBiasBuf = lease.Buffer;
            }

            try
            {
                int infoLen = checked(rows * (2 * topK + 1));
                using var infoB = AllocateFullyWrittenOutputBuffer(backend, infoLen);
                using var outB = AllocateFullyWrittenOutputBuffer(backend, totalOut);

                fused.SigmoidTopKExpertMixForward(
                    logB.Buffer, tblB.Buffer, activeBiasBuf, outB.Buffer, infoB.Buffer,
                    rows, numExperts, topK, channels, (float)epsilon, (float)effRate);

                if (persistentBiasBuf is not null)
                {
                    CommitParameterUpdatedOnDevice((Tensor<float>)(object)routingBias, persistentBiasBuf, backend, syncPoint: null);
                }

                var result = DeferTensorResult<T>(backend, outB.Buffer, totalOut, new[] { rows, channels });
                outB.RelinquishOwnership();
                if (!IsTapeActive<T>())
                    return result;

                var savedInfoTensor = DeferTensorResult<float>(backend, infoB.Buffer, infoLen, new[] { rows, 2 * topK + 1 });
                infoB.RelinquishOwnership();

                DifferentiableOps.RecordIfActive<T>(
                    "SigmoidTopKExpertMix",
                    result,
                    new[] { logits, expertTable },
                    static (gradOutput, inputs, output, savedState, engine, grads) =>
                    {
                        int r = (int)savedState[0], g = (int)savedState[1], k = (int)savedState[2], c = (int)savedState[3];
                        double eps = (double)savedState[4];
                        var infoTensor = (Tensor<float>)savedState[5];
                        if (!DifferentiableOps._isBackwardCreateGraph
                            && engine.TryGet<DirectGpuTensorEngine>(out var gpu)
                            && gpu.TrySigmoidTopKExpertMixBackwardGpu(gradOutput, inputs[0], inputs[1], infoTensor, r, g, k, c, eps) is { } res)
                        {
                            DifferentiableOps.AccumulateGrad(grads, inputs[0], res.GradLogits, engine);
                            DifferentiableOps.AccumulateGrad(grads, inputs[1], res.GradTable, engine);
                            return;
                        }
                        var hostState = new object[] { r, g, k, c, eps, infoTensor.AsSpan().ToArray() };
                        CpuEngine.SigmoidTopKExpertMixBackward(gradOutput, inputs, output, hostState, engine, grads);
                    },
                    savedState: new object[] { rows, numExperts, topK, channels, epsilon, savedInfoTensor });
                return result;
            }
            finally
            {
                tempBiasLease?.Dispose();
            }
        }
        catch (NotSupportedException capabilityGap)
        {
            GpuLaunchProbe.OnFallback("SigmoidTopKExpertMix", capabilityGap);
            return base.SigmoidTopKExpertMix(logits, expertTable, routingBias, topK, epsilon, biasUpdateRate);
        }
    }

    private (Tensor<T> GradLogits, Tensor<T> GradTable)? TrySigmoidTopKExpertMixBackwardGpu<T>(
        Tensor<T> gradOutput, Tensor<T> logits, Tensor<T> expertTable, Tensor<float> savedInfo,
        int rows, int numExperts, int topK, int channels, double epsilon)
    {
        if (typeof(T) != typeof(float) || !TryGetBackend(out var backend) || backend is not IFusedSequenceMixerOps fused)
            return null;
        if (rows <= 0 || gradOutput.Length != rows * channels) return null;

        try
        {
            var sDy = gradOutput.IsContiguous ? gradOutput : gradOutput.Contiguous();
            var sLog = logits.IsContiguous ? logits : logits.Contiguous();
            var sTbl = expertTable.IsContiguous ? expertTable : expertTable.Contiguous();
            var sInfo = savedInfo.IsContiguous ? savedInfo : savedInfo.Contiguous();
            using var dyB = GetOrAllocateBuffer(backend, sDy);
            using var logB = GetOrAllocateBuffer(backend, sLog);
            using var tblB = GetOrAllocateBuffer(backend, sTbl);
            using var infoB = GetOrAllocateBuffer(backend, sInfo);
            using var dlogB = AllocateFullyWrittenOutputBuffer(backend, rows * numExperts);
            using var dtblB = AllocateFullyWrittenOutputBuffer(backend, numExperts * channels);

            fused.SigmoidTopKExpertMixBackward(
                dyB.Buffer, logB.Buffer, tblB.Buffer, infoB.Buffer, dlogB.Buffer, dtblB.Buffer,
                rows, numExperts, topK, channels, (float)epsilon);
            var gLog = DeferTensorResult<T>(backend, dlogB.Buffer, rows * numExperts, (int[])logits._shape.Clone());
            dlogB.RelinquishOwnership();
            var gTbl = DeferTensorResult<T>(backend, dtblB.Buffer, numExperts * channels, (int[])expertTable._shape.Clone());
            dtblB.RelinquishOwnership();
            return (gLog, gTbl);
        }
        catch (NotSupportedException capabilityGap)
        {
            GpuLaunchProbe.OnFallback("SigmoidTopKExpertMixBackward", capabilityGap);
            return null;
        }
    }

    public override Tensor<T> DualModulatedSwiGlu<T>(Tensor<T> gateUpProj, Tensor<T> gateUpGainDelta, Tensor<T> upBase)
    {
        using var __gpuOp = EnterGpuOp();
        if (gateUpProj is null) throw new ArgumentNullException(nameof(gateUpProj));
        if (gateUpGainDelta is null) throw new ArgumentNullException(nameof(gateUpGainDelta));
        if (upBase is null) throw new ArgumentNullException(nameof(upBase));
        if (gateUpProj.Rank != 2 || gateUpGainDelta.Rank != 2)
            throw new ArgumentException("gateUpProj and gateUpGainDelta must be rank 2 [N, 2*H].");
        int rows = gateUpProj.Shape[0], cols2 = gateUpProj.Shape[1];
        if (cols2 <= 0 || (cols2 & 1) != 0)
            throw new ArgumentException($"Expected positive even 2*H columns, got {cols2}.", nameof(gateUpProj));
        int h = cols2 / 2;
        if (gateUpGainDelta.Shape[0] != rows || gateUpGainDelta.Shape[1] != cols2)
            throw new ArgumentException("gateUpGainDelta shape mismatch.", nameof(gateUpGainDelta));
        if (upBase.Length != h)
            throw new ArgumentException($"upBase length {upBase.Length} != {h}.", nameof(upBase));

        int totalOut = checked(rows * h);
        if (totalOut == 0 || GraphMode.IsActive || typeof(T) != typeof(float)
            || !TryGetBackend(out var backend) || backend is not IFusedSequenceMixerOps fused)
        {
            return base.DualModulatedSwiGlu(gateUpProj, gateUpGainDelta, upBase);
        }

        try
        {
            var sGu = gateUpProj.IsContiguous ? gateUpProj : gateUpProj.Contiguous();
            var sGg = gateUpGainDelta.IsContiguous ? gateUpGainDelta : gateUpGainDelta.Contiguous();
            var sB = upBase.IsContiguous ? upBase : upBase.Contiguous();
            using var guB = GetOrAllocateBuffer(backend, sGu);
            using var ggB = GetOrAllocateBuffer(backend, sGg);
            using var bB = GetOrAllocateBuffer(backend, sB);
            using var outB = AllocateFullyWrittenOutputBuffer(backend, totalOut);

            fused.DualModulatedSwiGluForward(guB.Buffer, ggB.Buffer, bB.Buffer, outB.Buffer, rows, h);
            var result = DeferTensorResult<T>(backend, outB.Buffer, totalOut, new[] { rows, h });
            outB.RelinquishOwnership();

            DifferentiableOps.RecordIfActive<T>(
                "DualModulatedSwiGlu",
                result,
                new[] { gateUpProj, gateUpGainDelta, upBase },
                static (gradOutput, inputs, output, savedState, engine, grads) =>
                {
                    int r = (int)savedState[0], hd = (int)savedState[1];
                    if (!DifferentiableOps._isBackwardCreateGraph
                        && engine.TryGet<DirectGpuTensorEngine>(out var gpu)
                        && gpu.TryDualModulatedSwiGluBackwardGpu(gradOutput, inputs[0], inputs[1], inputs[2], r, hd) is { } g)
                    {
                        DifferentiableOps.AccumulateGrad(grads, inputs[0], g.GradGu, engine);
                        DifferentiableOps.AccumulateGrad(grads, inputs[1], g.GradGg, engine);
                        DifferentiableOps.AccumulateGrad(grads, inputs[2], g.GradBase, engine);
                        return;
                    }
                    CpuEngine.DualModulatedSwiGluBackward(gradOutput, inputs, output, savedState, engine, grads);
                },
                savedState: new object[] { rows, h });
            return result;
        }
        catch (NotSupportedException capabilityGap)
        {
            GpuLaunchProbe.OnFallback("DualModulatedSwiGlu", capabilityGap);
            return base.DualModulatedSwiGlu(gateUpProj, gateUpGainDelta, upBase);
        }
    }

    private (Tensor<T> GradGu, Tensor<T> GradGg, Tensor<T> GradBase)? TryDualModulatedSwiGluBackwardGpu<T>(
        Tensor<T> gradOutput, Tensor<T> guProj, Tensor<T> guGain, Tensor<T> upBase, int rows, int h)
    {
        if (typeof(T) != typeof(float) || !TryGetBackend(out var backend) || backend is not IFusedSequenceMixerOps fused)
            return null;
        if (rows <= 0 || gradOutput.Length != rows * h) return null;

        try
        {
            var sDy = gradOutput.IsContiguous ? gradOutput : gradOutput.Contiguous();
            var sGu = guProj.IsContiguous ? guProj : guProj.Contiguous();
            var sGg = guGain.IsContiguous ? guGain : guGain.Contiguous();
            var sB = upBase.IsContiguous ? upBase : upBase.Contiguous();
            using var dyB = GetOrAllocateBuffer(backend, sDy);
            using var guB = GetOrAllocateBuffer(backend, sGu);
            using var ggB = GetOrAllocateBuffer(backend, sGg);
            using var bB = GetOrAllocateBuffer(backend, sB);
            using var dguB = AllocateFullyWrittenOutputBuffer(backend, rows * 2 * h);
            using var dggB = AllocateFullyWrittenOutputBuffer(backend, rows * 2 * h);
            using var dbB = AllocateFullyWrittenOutputBuffer(backend, h);

            fused.DualModulatedSwiGluBackward(
                dyB.Buffer, guB.Buffer, ggB.Buffer, bB.Buffer, dguB.Buffer, dggB.Buffer, dbB.Buffer, rows, h);
            var gGu = DeferTensorResult<T>(backend, dguB.Buffer, rows * 2 * h, (int[])guProj._shape.Clone());
            dguB.RelinquishOwnership();
            var gGg = DeferTensorResult<T>(backend, dggB.Buffer, rows * 2 * h, (int[])guGain._shape.Clone());
            dggB.RelinquishOwnership();
            var gB = DeferTensorResult<T>(backend, dbB.Buffer, h, (int[])upBase._shape.Clone());
            dbB.RelinquishOwnership();
            return (gGu, gGg, gB);
        }
        catch (NotSupportedException capabilityGap)
        {
            GpuLaunchProbe.OnFallback("DualModulatedSwiGluBackward", capabilityGap);
            return null;
        }
    }

    public override Tensor<T> BornRatioNormalize<T>(Tensor<T> scanOutput, Tensor<T>? sinkLogAlpha, int headDim, double epsilon = 1e-4)
    {
        using var __gpuOp = EnterGpuOp();
        if (scanOutput is null) throw new ArgumentNullException(nameof(scanOutput));
        if (scanOutput.Rank != 4)
            throw new ArgumentException($"Expected scanOutput rank 4 [B, S, H, scanDim], got {scanOutput.Rank}.", nameof(scanOutput));
        int b = scanOutput.Shape[0], s = scanOutput.Shape[1], nh = scanOutput.Shape[2], scanDim = scanOutput.Shape[3];
        if (headDim <= 0 || scanDim < headDim + 1)
            throw new ArgumentOutOfRangeException(nameof(headDim));
        if (sinkLogAlpha is not null && sinkLogAlpha.Length != nh)
            throw new ArgumentException($"Expected sinkLogAlpha length {nh}, got {sinkLogAlpha.Length}.", nameof(sinkLogAlpha));

        int totalRows = checked(b * s * nh);
        int totalOut = checked(totalRows * headDim);
        if (totalOut == 0 || GraphMode.IsActive || typeof(T) != typeof(float)
            || !TryGetBackend(out var backend) || backend is not IFusedSequenceMixerOps fused)
        {
            return base.BornRatioNormalize(scanOutput, sinkLogAlpha, headDim, epsilon);
        }

        try
        {
            var sScan = scanOutput.IsContiguous ? scanOutput : scanOutput.Contiguous();
            using var scanB = GetOrAllocateBuffer(backend, sScan);
            OwnedBuffer? sinkLease = null;
            if (sinkLogAlpha is not null)
            {
                var sSink = sinkLogAlpha.IsContiguous ? sinkLogAlpha : sinkLogAlpha.Contiguous();
                sinkLease = GetOrAllocateBuffer(backend, sSink);
            }

            try
            {
                using var outB = AllocateFullyWrittenOutputBuffer(backend, totalOut);
                var sinkBuf = sinkLease is { } sl ? sl.Buffer : scanB.Buffer;
                fused.BornRatioNormalizeForward(
                    scanB.Buffer, sinkBuf, outB.Buffer, totalRows, nh, headDim, scanDim, sinkLogAlpha is not null ? 1 : 0, (float)epsilon);

                var result = DeferTensorResult<T>(backend, outB.Buffer, totalOut, new[] { b, s, nh * headDim });
                outB.RelinquishOwnership();

                var inputs = sinkLogAlpha is null ? new[] { scanOutput } : new[] { scanOutput, sinkLogAlpha };
                DifferentiableOps.RecordIfActive<T>(
                    "BornRatioNormalize",
                    result,
                    inputs,
                    static (gradOutput, ins, output, savedState, engine, grads) =>
                    {
                        int bb = (int)savedState[0], ss = (int)savedState[1], hCnt = (int)savedState[2];
                        int hd = (int)savedState[3], sd = (int)savedState[4];
                        double eps = (double)savedState[5];
                        bool hasS = (bool)savedState[6];
                        Tensor<T>? sinkIn = hasS ? ins[1] : null;
                        if (!DifferentiableOps._isBackwardCreateGraph
                            && engine.TryGet<DirectGpuTensorEngine>(out var gpu)
                            && gpu.TryBornRatioNormalizeBackwardGpu(gradOutput, ins[0], sinkIn, bb * ss * hCnt, hCnt, hd, sd, eps) is { } g)
                        {
                            DifferentiableOps.AccumulateGrad(grads, ins[0], g.GradScan, engine);
                            if (sinkIn is not null && g.GradSink is not null)
                                DifferentiableOps.AccumulateGrad(grads, sinkIn, g.GradSink, engine);
                            return;
                        }
                        CpuEngine.BornRatioNormalizeBackward(gradOutput, ins, output, savedState, engine, grads);
                    },
                    savedState: new object[] { b, s, nh, headDim, scanDim, epsilon, sinkLogAlpha is not null });
                return result;
            }
            finally
            {
                sinkLease?.Dispose();
            }
        }
        catch (NotSupportedException capabilityGap)
        {
            GpuLaunchProbe.OnFallback("BornRatioNormalize", capabilityGap);
            return base.BornRatioNormalize(scanOutput, sinkLogAlpha, headDim, epsilon);
        }
    }

    private (Tensor<T> GradScan, Tensor<T>? GradSink)? TryBornRatioNormalizeBackwardGpu<T>(
        Tensor<T> gradOutput, Tensor<T> scanOutput, Tensor<T>? sinkLogAlpha,
        int totalRows, int nh, int headDim, int scanDim, double epsilon)
    {
        if (typeof(T) != typeof(float) || !TryGetBackend(out var backend) || backend is not IFusedSequenceMixerOps fused)
            return null;
        if (totalRows <= 0 || gradOutput.Length != totalRows * headDim) return null;

        try
        {
            var sDy = gradOutput.IsContiguous ? gradOutput : gradOutput.Contiguous();
            var sScan = scanOutput.IsContiguous ? scanOutput : scanOutput.Contiguous();
            using var dyB = GetOrAllocateBuffer(backend, sDy);
            using var scanB = GetOrAllocateBuffer(backend, sScan);
            OwnedBuffer? sinkLease = null;
            OwnedBuffer? dSinkOut = null;
            if (sinkLogAlpha is not null)
            {
                var sSink = sinkLogAlpha.IsContiguous ? sinkLogAlpha : sinkLogAlpha.Contiguous();
                sinkLease = GetOrAllocateBuffer(backend, sSink);
                dSinkOut = AllocateFullyWrittenOutputBuffer(backend, nh);
            }

            try
            {
                using var dScanB = AllocateFullyWrittenOutputBuffer(backend, totalRows * scanDim);
                var sinkBuf = sinkLease is { } sl ? sl.Buffer : scanB.Buffer;
                var dSinkBuf = dSinkOut is { } ds ? ds.Buffer : dScanB.Buffer;

                fused.BornRatioNormalizeBackward(
                    dyB.Buffer, scanB.Buffer, sinkBuf, dScanB.Buffer, dSinkBuf,
                    totalRows, nh, headDim, scanDim, sinkLogAlpha is not null ? 1 : 0, (float)epsilon);

                var gScan = DeferTensorResult<T>(backend, dScanB.Buffer, totalRows * scanDim, (int[])scanOutput._shape.Clone());
                dScanB.RelinquishOwnership();
                Tensor<T>? gSink = null;
                if (sinkLogAlpha is not null && dSinkOut is { } dso)
                {
                    gSink = DeferTensorResult<T>(backend, dso.Buffer, nh, (int[])sinkLogAlpha._shape.Clone());
                    dso.RelinquishOwnership();
                }
                return (gScan, gSink);
            }
            finally
            {
                sinkLease?.Dispose();
                dSinkOut?.Dispose();
            }
        }
        catch (NotSupportedException capabilityGap)
        {
            GpuLaunchProbe.OnFallback("BornRatioNormalizeBackward", capabilityGap);
            return null;
        }
    }

    public override Tensor<T> BoundedSigmoidGate<T>(Tensor<T> logits, Tensor<T> bias, double floor)
    {
        using var __gpuOp = EnterGpuOp();
        if (logits is null) throw new ArgumentNullException(nameof(logits));
        if (bias is null) throw new ArgumentNullException(nameof(bias));
        if (logits.Rank < 1) throw new ArgumentException("logits must have rank >= 1.", nameof(logits));
        int cols = logits.Shape[logits.Rank - 1];
        if (cols <= 0 || bias.Length != cols)
            throw new ArgumentException($"Expected bias length {cols}, got {bias.Length}.", nameof(bias));

        int rows = logits.Length / cols;
        int total = checked(rows * cols);
        if (total == 0 || GraphMode.IsActive || typeof(T) != typeof(float)
            || !TryGetBackend(out var backend) || backend is not IFusedSequenceMixerOps fused)
        {
            return base.BoundedSigmoidGate(logits, bias, floor);
        }

        try
        {
            var sL = logits.IsContiguous ? logits : logits.Contiguous();
            var sB = bias.IsContiguous ? bias : bias.Contiguous();
            using var lB = GetOrAllocateBuffer(backend, sL);
            using var bB = GetOrAllocateBuffer(backend, sB);
            using var outB = AllocateFullyWrittenOutputBuffer(backend, total);

            fused.BoundedSigmoidGateForward(lB.Buffer, bB.Buffer, outB.Buffer, rows, cols, (float)floor);
            var result = DeferTensorResult<T>(backend, outB.Buffer, total, (int[])logits._shape.Clone());
            outB.RelinquishOwnership();

            DifferentiableOps.RecordIfActive<T>(
                "BoundedSigmoidGate",
                result,
                new[] { logits, bias },
                static (gradOutput, inputs, output, savedState, engine, grads) =>
                {
                    int r = (int)savedState[0], c = (int)savedState[1];
                    double flr = (double)savedState[2];
                    if (!DifferentiableOps._isBackwardCreateGraph
                        && engine.TryGet<DirectGpuTensorEngine>(out var gpu)
                        && gpu.TryBoundedSigmoidGateBackwardGpu(gradOutput, inputs[0], inputs[1], r, c, flr) is { } g)
                    {
                        DifferentiableOps.AccumulateGrad(grads, inputs[0], g.GradLogits, engine);
                        DifferentiableOps.AccumulateGrad(grads, inputs[1], g.GradBias, engine);
                        return;
                    }
                    CpuEngine.BoundedSigmoidGateBackward(gradOutput, inputs, output, savedState, engine, grads);
                },
                savedState: new object[] { rows, cols, floor });
            return result;
        }
        catch (NotSupportedException capabilityGap)
        {
            GpuLaunchProbe.OnFallback("BoundedSigmoidGate", capabilityGap);
            return base.BoundedSigmoidGate(logits, bias, floor);
        }
    }

    private (Tensor<T> GradLogits, Tensor<T> GradBias)? TryBoundedSigmoidGateBackwardGpu<T>(
        Tensor<T> gradOutput, Tensor<T> logits, Tensor<T> bias, int rows, int cols, double floor)
    {
        if (typeof(T) != typeof(float) || !TryGetBackend(out var backend) || backend is not IFusedSequenceMixerOps fused)
            return null;
        if (rows <= 0 || gradOutput.Length != rows * cols) return null;

        try
        {
            var sDy = gradOutput.IsContiguous ? gradOutput : gradOutput.Contiguous();
            var sL = logits.IsContiguous ? logits : logits.Contiguous();
            var sB = bias.IsContiguous ? bias : bias.Contiguous();
            using var dyB = GetOrAllocateBuffer(backend, sDy);
            using var lB = GetOrAllocateBuffer(backend, sL);
            using var bB = GetOrAllocateBuffer(backend, sB);
            using var dlB = AllocateFullyWrittenOutputBuffer(backend, rows * cols);
            using var dbB = AllocateFullyWrittenOutputBuffer(backend, cols);

            fused.BoundedSigmoidGateBackward(dyB.Buffer, lB.Buffer, bB.Buffer, dlB.Buffer, dbB.Buffer, rows, cols, (float)floor);
            var gL = DeferTensorResult<T>(backend, dlB.Buffer, rows * cols, (int[])logits._shape.Clone());
            dlB.RelinquishOwnership();
            var gB = DeferTensorResult<T>(backend, dbB.Buffer, cols, (int[])bias._shape.Clone());
            dbB.RelinquishOwnership();
            return (gL, gB);
        }
        catch (NotSupportedException capabilityGap)
        {
            GpuLaunchProbe.OnFallback("BoundedSigmoidGateBackward", capabilityGap);
            return null;
        }
    }
}
