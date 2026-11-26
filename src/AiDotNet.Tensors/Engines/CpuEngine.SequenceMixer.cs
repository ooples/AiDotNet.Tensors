// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

public partial class CpuEngine
{
    public virtual Tensor<T> SplitHalfGivensRotation<T>(Tensor<T> input, Tensor<T> angles)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (angles is null) throw new ArgumentNullException(nameof(angles));
        if (input.Rank < 1) throw new ArgumentException("Input must have rank >= 1.", nameof(input));
        int d = input.Shape[input.Rank - 1];
        if (d <= 0 || (d & 1) != 0)
            throw new ArgumentException($"Expected positive even last dimension, got {d}.", nameof(input));
        int half = d / 2;
        if (angles.Length != half)
            throw new ArgumentException($"Expected angles length {half}, got {angles.Length}.", nameof(angles));

        if (GraphMode.IsActive || (typeof(T) != typeof(float) && typeof(T) != typeof(double)))
        {
            return SplitHalfGivensRotationComposed(this, input, angles, half);
        }

        int rows = input.Length / d;
        var output = new Tensor<T>((int[])input._shape.Clone());
        if (rows > 0)
        {
            var srcIn = input.IsContiguous ? input : input.Contiguous();
            var srcAng = angles.IsContiguous ? angles : angles.Contiguous();
            if (typeof(T) == typeof(float))
            {
                SplitHalfGivensForwardFloat(
                    ((Tensor<float>)(object)srcIn).AsSpan(),
                    ((Tensor<float>)(object)srcAng).AsSpan(),
                    ((Tensor<float>)(object)output).AsWritableSpan(),
                    rows, half);
            }
            else
            {
                SplitHalfGivensForwardDouble(
                    ((Tensor<double>)(object)srcIn).AsSpan(),
                    ((Tensor<double>)(object)srcAng).AsSpan(),
                    ((Tensor<double>)(object)output).AsWritableSpan(),
                    rows, half);
            }
        }

        DifferentiableOps.RecordIfActive<T>(
            "SplitHalfGivensRotation",
            output,
            new[] { input, angles },
            SplitHalfGivensRotationBackward<T>,
            savedState: new object[] { half });
        return output;
    }

    internal static Tensor<T> SplitHalfGivensRotationComposed<T>(IEngine engine, Tensor<T> input, Tensor<T> angles, int half)
    {
        int rows = input.Length / (2 * half);
        var x3 = engine.Reshape(input, new[] { rows, 1, 2 * half });
        var ang3 = engine.Reshape(angles, new[] { 1, 1, half });
        var c = engine.TensorCos(ang3);
        var s = engine.TensorSin(ang3);
        var h1 = engine.TensorNarrow(x3, 2, 0, half);
        var h2 = engine.TensorNarrow(x3, 2, half, half);
        var r1 = engine.TensorSubtract(engine.TensorMultiply(h1, c), engine.TensorMultiply(h2, s));
        var r2 = engine.TensorAdd(engine.TensorMultiply(h1, s), engine.TensorMultiply(h2, c));
        return engine.Reshape(engine.TensorConcatenate(new[] { r1, r2 }, axis: 2), (int[])input._shape.Clone());
    }

    internal static void SplitHalfGivensRotationBackward<T>(
        Tensor<T> gradOutput,
        Tensor<T>[] inputs,
        Tensor<T> output,
        object[] savedState,
        IEngine engine,
        Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        int half = (int)savedState[0];
        var input = inputs[0];
        var angles = inputs[1];
        if (DifferentiableOps._isBackwardCreateGraph || (typeof(T) != typeof(float) && typeof(T) != typeof(double)))
        {
            // The tape this create-graph backward records on. The nested tape below is disposed on return, so its
            // entries move to this one and the gradients accumulate on it; otherwise a higher order sees no graph.
            var outerTape = GradientTape<T>.Current;
            using var tape = new GradientTape<T>(new GradientTapeOptions { Persistent = false });
            var y = SplitHalfGivensRotationComposed(engine, input, angles, half);
            var loss = engine.ReduceSum(engine.TensorMultiply(y, gradOutput), null);
            var sub = tape.ComputeGradients(loss, new[] { input, angles }, createGraph: DifferentiableOps._isBackwardCreateGraph);
            if (DifferentiableOps._isBackwardCreateGraph && outerTape is not null) tape.TransferEntriesTo(outerTape);
            using var recordOnOuter = GradientTape<T>.RecordOn(outerTape);
            if (sub.TryGetValue(input, out var gi)) DifferentiableOps.AccumulateGrad(grads, input, gi, engine);
            if (sub.TryGetValue(angles, out var ga)) DifferentiableOps.AccumulateGrad(grads, angles, ga, engine);
            return;
        }

        int rows = input.Length / (2 * half);
        var gradIn = new Tensor<T>((int[])input._shape.Clone());
        var gradAng = new Tensor<T>((int[])angles._shape.Clone());
        if (rows > 0)
        {
            var srcDy = gradOutput.IsContiguous ? gradOutput : gradOutput.Contiguous();
            var srcIn = input.IsContiguous ? input : input.Contiguous();
            var srcAng = angles.IsContiguous ? angles : angles.Contiguous();
            if (typeof(T) == typeof(float))
            {
                SplitHalfGivensBackwardFloat(
                    ((Tensor<float>)(object)srcDy).AsSpan(),
                    ((Tensor<float>)(object)srcIn).AsSpan(),
                    ((Tensor<float>)(object)srcAng).AsSpan(),
                    ((Tensor<float>)(object)gradIn).AsWritableSpan(),
                    ((Tensor<float>)(object)gradAng).AsWritableSpan(),
                    rows, half);
            }
            else
            {
                SplitHalfGivensBackwardDouble(
                    ((Tensor<double>)(object)srcDy).AsSpan(),
                    ((Tensor<double>)(object)srcIn).AsSpan(),
                    ((Tensor<double>)(object)srcAng).AsSpan(),
                    ((Tensor<double>)(object)gradIn).AsWritableSpan(),
                    ((Tensor<double>)(object)gradAng).AsWritableSpan(),
                    rows, half);
            }
        }
        DifferentiableOps.AccumulateGrad(grads, input, gradIn, engine);
        DifferentiableOps.AccumulateGrad(grads, angles, gradAng, engine);
    }

    private static void SplitHalfGivensForwardFloat(ReadOnlySpan<float> x, ReadOnlySpan<float> ang, Span<float> y, int rows, int half)
    {
        int dim = 2 * half;
        for (int m = 0; m < rows; m++)
        {
            int off = m * dim;
            for (int i = 0; i < half; i++)
            {
                float c = MathF.Cos(ang[i]);
                float s = MathF.Sin(ang[i]);
                float x0 = x[off + i];
                float x1 = x[off + half + i];
                y[off + i] = x0 * c - x1 * s;
                y[off + half + i] = x0 * s + x1 * c;
            }
        }
    }

    private static void SplitHalfGivensForwardDouble(ReadOnlySpan<double> x, ReadOnlySpan<double> ang, Span<double> y, int rows, int half)
    {
        int dim = 2 * half;
        for (int m = 0; m < rows; m++)
        {
            int off = m * dim;
            for (int i = 0; i < half; i++)
            {
                double c = Math.Cos(ang[i]);
                double s = Math.Sin(ang[i]);
                double x0 = x[off + i];
                double x1 = x[off + half + i];
                y[off + i] = x0 * c - x1 * s;
                y[off + half + i] = x0 * s + x1 * c;
            }
        }
    }

    private static void SplitHalfGivensBackwardFloat(ReadOnlySpan<float> dy, ReadOnlySpan<float> x, ReadOnlySpan<float> ang, Span<float> dx, Span<float> dAng, int rows, int half)
    {
        int dim = 2 * half;
        dAng.Slice(0, half).Clear();
        for (int m = 0; m < rows; m++)
        {
            int off = m * dim;
            for (int i = 0; i < half; i++)
            {
                float c = MathF.Cos(ang[i]);
                float s = MathF.Sin(ang[i]);
                float x0 = x[off + i], x1 = x[off + half + i];
                float dy0 = dy[off + i], dy1 = dy[off + half + i];
                dx[off + i] = dy0 * c + dy1 * s;
                dx[off + half + i] = -dy0 * s + dy1 * c;
                dAng[i] += dy0 * (-x0 * s - x1 * c) + dy1 * (x0 * c - x1 * s);
            }
        }
    }

    private static void SplitHalfGivensBackwardDouble(ReadOnlySpan<double> dy, ReadOnlySpan<double> x, ReadOnlySpan<double> ang, Span<double> dx, Span<double> dAng, int rows, int half)
    {
        int dim = 2 * half;
        dAng.Slice(0, half).Clear();
        for (int m = 0; m < rows; m++)
        {
            int off = m * dim;
            for (int i = 0; i < half; i++)
            {
                double c = Math.Cos(ang[i]);
                double s = Math.Sin(ang[i]);
                double x0 = x[off + i], x1 = x[off + half + i];
                double dy0 = dy[off + i], dy1 = dy[off + half + i];
                dx[off + i] = dy0 * c + dy1 * s;
                dx[off + half + i] = -dy0 * s + dy1 * c;
                dAng[i] += dy0 * (-x0 * s - x1 * c) + dy1 * (x0 * c - x1 * s);
            }
        }
    }

    public virtual Tensor<T> DepthwiseCausalConv1d2Tap<T>(Tensor<T> input, Tensor<T> weight0, Tensor<T> weight1)
    {
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

        if (GraphMode.IsActive || (typeof(T) != typeof(float) && typeof(T) != typeof(double)))
        {
            return DepthwiseCausalConv1d2TapComposed(this, input, weight0, weight1, batch, seqLen, channels);
        }

        var output = new Tensor<T>((int[])input._shape.Clone());
        if (input.Length > 0)
        {
            var srcIn = input.IsContiguous ? input : input.Contiguous();
            var srcW0 = weight0.IsContiguous ? weight0 : weight0.Contiguous();
            var srcW1 = weight1.IsContiguous ? weight1 : weight1.Contiguous();
            if (typeof(T) == typeof(float))
            {
                Conv2TapForwardFloat(
                    ((Tensor<float>)(object)srcIn).AsSpan(),
                    ((Tensor<float>)(object)srcW0).AsSpan(),
                    ((Tensor<float>)(object)srcW1).AsSpan(),
                    ((Tensor<float>)(object)output).AsWritableSpan(),
                    batch, seqLen, channels);
            }
            else
            {
                Conv2TapForwardDouble(
                    ((Tensor<double>)(object)srcIn).AsSpan(),
                    ((Tensor<double>)(object)srcW0).AsSpan(),
                    ((Tensor<double>)(object)srcW1).AsSpan(),
                    ((Tensor<double>)(object)output).AsWritableSpan(),
                    batch, seqLen, channels);
            }
        }

        DifferentiableOps.RecordIfActive<T>(
            "DepthwiseCausalConv1d2Tap",
            output,
            new[] { input, weight0, weight1 },
            DepthwiseCausalConv1d2TapBackward<T>,
            savedState: new object[] { batch, seqLen, channels });
        return output;
    }

    internal static Tensor<T> DepthwiseCausalConv1d2TapComposed<T>(
        IEngine engine, Tensor<T> input, Tensor<T> weight0, Tensor<T> weight1, int batch, int seqLen, int channels)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var x3 = input.Rank == 3 ? input : engine.Reshape(input, new[] { batch, seqLen, channels });
        var w0 = engine.Reshape(weight0, new[] { 1, 1, channels });
        var w1 = engine.Reshape(weight1, new[] { 1, 1, channels });
        var term0 = engine.TensorMultiply(x3, w0);
        if (seqLen <= 1)
            return input.Rank == 3 ? term0 : engine.Reshape(term0, (int[])input._shape.Clone());

        var pad = TensorScope.Detach(new Tensor<T>(new[] { batch, 1, channels }));
        var kept = engine.TensorNarrow(x3, 1, 0, seqLen - 1);
        var weightedPrev = engine.TensorMultiply(kept, w1);
        var term1 = engine.TensorConcatenate(new[] { pad, weightedPrev }, axis: 1);
        var y3 = engine.TensorAdd(term0, term1);
        return input.Rank == 3 ? y3 : engine.Reshape(y3, (int[])input._shape.Clone());
    }

    internal static void DepthwiseCausalConv1d2TapBackward<T>(
        Tensor<T> gradOutput,
        Tensor<T>[] inputs,
        Tensor<T> output,
        object[] savedState,
        IEngine engine,
        Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        int batch = (int)savedState[0];
        int seqLen = (int)savedState[1];
        int channels = (int)savedState[2];
        var input = inputs[0];
        var weight0 = inputs[1];
        var weight1 = inputs[2];
        if (DifferentiableOps._isBackwardCreateGraph || (typeof(T) != typeof(float) && typeof(T) != typeof(double)))
        {
            // The tape this create-graph backward records on. The nested tape below is disposed on return, so its
            // entries move to this one and the gradients accumulate on it; otherwise a higher order sees no graph.
            var outerTape = GradientTape<T>.Current;
            using var tape = new GradientTape<T>(new GradientTapeOptions { Persistent = false });
            var y = DepthwiseCausalConv1d2TapComposed(engine, input, weight0, weight1, batch, seqLen, channels);
            var loss = engine.ReduceSum(engine.TensorMultiply(y, gradOutput), null);
            var sub = tape.ComputeGradients(loss, new[] { input, weight0, weight1 }, createGraph: DifferentiableOps._isBackwardCreateGraph);
            if (DifferentiableOps._isBackwardCreateGraph && outerTape is not null) tape.TransferEntriesTo(outerTape);
            using var recordOnOuter = GradientTape<T>.RecordOn(outerTape);
            if (sub.TryGetValue(input, out var gi)) DifferentiableOps.AccumulateGrad(grads, input, gi, engine);
            if (sub.TryGetValue(weight0, out var gw0)) DifferentiableOps.AccumulateGrad(grads, weight0, gw0, engine);
            if (sub.TryGetValue(weight1, out var gw1)) DifferentiableOps.AccumulateGrad(grads, weight1, gw1, engine);
            return;
        }

        var gradIn = new Tensor<T>((int[])input._shape.Clone());
        var gradW0 = new Tensor<T>((int[])weight0._shape.Clone());
        var gradW1 = new Tensor<T>((int[])weight1._shape.Clone());
        if (input.Length > 0)
        {
            var srcDy = gradOutput.IsContiguous ? gradOutput : gradOutput.Contiguous();
            var srcIn = input.IsContiguous ? input : input.Contiguous();
            var srcW0 = weight0.IsContiguous ? weight0 : weight0.Contiguous();
            var srcW1 = weight1.IsContiguous ? weight1 : weight1.Contiguous();
            if (typeof(T) == typeof(float))
            {
                Conv2TapBackwardFloat(
                    ((Tensor<float>)(object)srcDy).AsSpan(),
                    ((Tensor<float>)(object)srcIn).AsSpan(),
                    ((Tensor<float>)(object)srcW0).AsSpan(),
                    ((Tensor<float>)(object)srcW1).AsSpan(),
                    ((Tensor<float>)(object)gradIn).AsWritableSpan(),
                    ((Tensor<float>)(object)gradW0).AsWritableSpan(),
                    ((Tensor<float>)(object)gradW1).AsWritableSpan(),
                    batch, seqLen, channels);
            }
            else
            {
                Conv2TapBackwardDouble(
                    ((Tensor<double>)(object)srcDy).AsSpan(),
                    ((Tensor<double>)(object)srcIn).AsSpan(),
                    ((Tensor<double>)(object)srcW0).AsSpan(),
                    ((Tensor<double>)(object)srcW1).AsSpan(),
                    ((Tensor<double>)(object)gradIn).AsWritableSpan(),
                    ((Tensor<double>)(object)gradW0).AsWritableSpan(),
                    ((Tensor<double>)(object)gradW1).AsWritableSpan(),
                    batch, seqLen, channels);
            }
        }
        DifferentiableOps.AccumulateGrad(grads, input, gradIn, engine);
        DifferentiableOps.AccumulateGrad(grads, weight0, gradW0, engine);
        DifferentiableOps.AccumulateGrad(grads, weight1, gradW1, engine);
    }

    private static void Conv2TapForwardFloat(ReadOnlySpan<float> x, ReadOnlySpan<float> w0, ReadOnlySpan<float> w1, Span<float> y, int batch, int seqLen, int channels)
    {
        for (int b = 0; b < batch; b++)
        {
            int bOff = b * seqLen * channels;
            for (int t = 0; t < seqLen; t++)
            {
                int row = bOff + t * channels;
                if (t == 0)
                {
                    for (int c = 0; c < channels; c++) y[row + c] = w0[c] * x[row + c];
                }
                else
                {
                    int prev = row - channels;
                    for (int c = 0; c < channels; c++) y[row + c] = w0[c] * x[row + c] + w1[c] * x[prev + c];
                }
            }
        }
    }

    private static void Conv2TapForwardDouble(ReadOnlySpan<double> x, ReadOnlySpan<double> w0, ReadOnlySpan<double> w1, Span<double> y, int batch, int seqLen, int channels)
    {
        for (int b = 0; b < batch; b++)
        {
            int bOff = b * seqLen * channels;
            for (int t = 0; t < seqLen; t++)
            {
                int row = bOff + t * channels;
                if (t == 0)
                {
                    for (int c = 0; c < channels; c++) y[row + c] = w0[c] * x[row + c];
                }
                else
                {
                    int prev = row - channels;
                    for (int c = 0; c < channels; c++) y[row + c] = w0[c] * x[row + c] + w1[c] * x[prev + c];
                }
            }
        }
    }

    private static void Conv2TapBackwardFloat(
        ReadOnlySpan<float> dy, ReadOnlySpan<float> x, ReadOnlySpan<float> w0, ReadOnlySpan<float> w1,
        Span<float> dx, Span<float> dw0, Span<float> dw1, int batch, int seqLen, int channels)
    {
        dw0.Slice(0, channels).Clear();
        dw1.Slice(0, channels).Clear();
        for (int b = 0; b < batch; b++)
        {
            int bOff = b * seqLen * channels;
            for (int t = 0; t < seqLen; t++)
            {
                int row = bOff + t * channels;
                bool hasNext = (t + 1) < seqLen;
                int next = row + channels;
                int prev = row - channels;
                for (int c = 0; c < channels; c++)
                {
                    float dyv = dy[row + c];
                    dx[row + c] = w0[c] * dyv + (hasNext ? w1[c] * dy[next + c] : 0f);
                    dw0[c] += dyv * x[row + c];
                    if (t > 0) dw1[c] += dyv * x[prev + c];
                }
            }
        }
    }

    private static void Conv2TapBackwardDouble(
        ReadOnlySpan<double> dy, ReadOnlySpan<double> x, ReadOnlySpan<double> w0, ReadOnlySpan<double> w1,
        Span<double> dx, Span<double> dw0, Span<double> dw1, int batch, int seqLen, int channels)
    {
        dw0.Slice(0, channels).Clear();
        dw1.Slice(0, channels).Clear();
        for (int b = 0; b < batch; b++)
        {
            int bOff = b * seqLen * channels;
            for (int t = 0; t < seqLen; t++)
            {
                int row = bOff + t * channels;
                bool hasNext = (t + 1) < seqLen;
                int next = row + channels;
                int prev = row - channels;
                for (int c = 0; c < channels; c++)
                {
                    double dyv = dy[row + c];
                    dx[row + c] = w0[c] * dyv + (hasNext ? w1[c] * dy[next + c] : 0.0);
                    dw0[c] += dyv * x[row + c];
                    if (t > 0) dw1[c] += dyv * x[prev + c];
                }
            }
        }
    }

    public virtual Tensor<T> RopeL2NormalizeHeads<T>(
        Tensor<T> input,
        Tensor<T> cos,
        Tensor<T> sin,
        int numHeads,
        double epsilon = 1e-6,
        double postScale = 1.0,
        int startPosition = 0)
    {
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
        long reqRows = (long)startPosition + seqLen;
        if (cos.Rank != 2 || sin.Rank != 2 || cos.Shape[1] != half || sin.Shape[1] != half
            || cos.Shape[0] < reqRows || sin.Shape[0] < reqRows)
            throw new ArgumentException($"cos/sin must have shape [>= {reqRows}, {half}].", nameof(cos));
        if (!(epsilon > 0.0)) throw new ArgumentOutOfRangeException(nameof(epsilon));

        if (GraphMode.IsActive || (typeof(T) != typeof(float) && typeof(T) != typeof(double)))
        {
            return RopeL2NormalizeHeadsComposed(this, input, cos, sin, batch, seqLen, numHeads, headDim, epsilon, postScale, startPosition);
        }

        var output = new Tensor<T>((int[])input._shape.Clone());
        Tensor<T>? srcCos = null;
        Tensor<T>? srcSin = null;
        if (input.Length > 0)
        {
            var srcIn = input.IsContiguous ? input : input.Contiguous();
            srcCos = cos.IsContiguous ? cos : cos.Contiguous();
            srcSin = sin.IsContiguous ? sin : sin.Contiguous();
            if (typeof(T) == typeof(float))
            {
                RopeL2ForwardFloat(
                    ((Tensor<float>)(object)srcIn).AsSpan(),
                    ((Tensor<float>)(object)srcCos).AsSpan(),
                    ((Tensor<float>)(object)srcSin).AsSpan(),
                    ((Tensor<float>)(object)output).AsWritableSpan(),
                    batch * seqLen * numHeads, seqLen, numHeads, headDim, startPosition, (float)epsilon, (float)postScale);
            }
            else
            {
                RopeL2ForwardDouble(
                    ((Tensor<double>)(object)srcIn).AsSpan(),
                    ((Tensor<double>)(object)srcCos).AsSpan(),
                    ((Tensor<double>)(object)srcSin).AsSpan(),
                    ((Tensor<double>)(object)output).AsWritableSpan(),
                    batch * seqLen * numHeads, seqLen, numHeads, headDim, startPosition, epsilon, postScale);
            }
        }

        if (GradientTape<T>.Current is not null && !NoGradScope<T>.IsSuppressed)
        {
            var savedCos = SnapshotRopeWindow(srcCos ?? (cos.IsContiguous ? cos : cos.Contiguous()), startPosition, seqLen, half);
            var savedSin = SnapshotRopeWindow(srcSin ?? (sin.IsContiguous ? sin : sin.Contiguous()), startPosition, seqLen, half);
            DifferentiableOps.RecordIfActive<T>(
                "RopeL2NormalizeHeads",
                output,
                new[] { input },
                RopeL2NormalizeHeadsBackward<T>,
                savedState: new object[] { savedCos, savedSin, batch, seqLen, numHeads, headDim, epsilon, postScale, 0 });
        }
        return output;
    }

    private static Tensor<T> SnapshotRopeWindow<T>(Tensor<T> contiguousTable, int startPosition, int seqLen, int half)
    {
        int len = checked(seqLen * half);
        var arr = new T[len];
        if (len > 0)
            contiguousTable.AsSpan().Slice(checked(startPosition * half), len).CopyTo(arr);
        return TensorScope.Detach(new Tensor<T>(new[] { seqLen, half }, new Vector<T>(arr)));
    }

    internal static Tensor<T> RopeL2NormalizeHeadsComposed<T>(
        IEngine engine, Tensor<T> input, Tensor<T> cos, Tensor<T> sin,
        int batch, int seqLen, int numHeads, int headDim, double epsilon, double postScale, int startPosition)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var z = engine.TensorPermute(engine.Reshape(input, new[] { batch, seqLen, numHeads, headDim }), new[] { 0, 2, 1, 3 });
        var zRot = engine.ApplyRoPEInterleaved(z, cos, sin, startPosition);
        var normSq = engine.ReduceSum(engine.TensorMultiply(zRot, zRot), new[] { 3 }, keepDims: true);
        var denom = engine.TensorSqrt(engine.TensorAddScalar(normSq, ops.FromDouble(epsilon)));
        var zUnit = engine.TensorDivide(zRot, denom);
        if (postScale != 1.0)
            zUnit = engine.TensorMultiplyScalar(zUnit, ops.FromDouble(postScale));
        return engine.Reshape(engine.TensorPermute(zUnit, new[] { 0, 2, 1, 3 }), (int[])input._shape.Clone());
    }

    internal static void RopeL2NormalizeHeadsBackward<T>(
        Tensor<T> gradOutput,
        Tensor<T>[] inputs,
        Tensor<T> output,
        object[] savedState,
        IEngine engine,
        Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var input = inputs[0];
        var cos = (Tensor<T>)savedState[0];
        var sin = (Tensor<T>)savedState[1];
        int batch = (int)savedState[2];
        int seqLen = (int)savedState[3];
        int numHeads = (int)savedState[4];
        int headDim = (int)savedState[5];
        double epsilon = (double)savedState[6];
        double postScale = (double)savedState[7];
        int startPosition = (int)savedState[8];

        if (DifferentiableOps._isBackwardCreateGraph || (typeof(T) != typeof(float) && typeof(T) != typeof(double)))
        {
            // The tape this create-graph backward records on. The nested tape below is disposed on return, so its
            // entries move to this one and the gradients accumulate on it; otherwise a higher order sees no graph.
            var outerTape = GradientTape<T>.Current;
            using var tape = new GradientTape<T>(new GradientTapeOptions { Persistent = false });
            var y = RopeL2NormalizeHeadsComposed(engine, input, cos, sin, batch, seqLen, numHeads, headDim, epsilon, postScale, startPosition);
            var loss = engine.ReduceSum(engine.TensorMultiply(y, gradOutput), null);
            var sub = tape.ComputeGradients(loss, new[] { input }, createGraph: DifferentiableOps._isBackwardCreateGraph);
            if (DifferentiableOps._isBackwardCreateGraph && outerTape is not null) tape.TransferEntriesTo(outerTape);
            using var recordOnOuter = GradientTape<T>.RecordOn(outerTape);
            if (sub.TryGetValue(input, out var gi)) DifferentiableOps.AccumulateGrad(grads, input, gi, engine);
            return;
        }

        var gradIn = new Tensor<T>((int[])input._shape.Clone());
        if (input.Length > 0)
        {
            var srcDy = gradOutput.IsContiguous ? gradOutput : gradOutput.Contiguous();
            var srcIn = input.IsContiguous ? input : input.Contiguous();
            var srcCos = cos.IsContiguous ? cos : cos.Contiguous();
            var srcSin = sin.IsContiguous ? sin : sin.Contiguous();
            if (typeof(T) == typeof(float))
            {
                RopeL2BackwardFloat(
                    ((Tensor<float>)(object)srcDy).AsSpan(),
                    ((Tensor<float>)(object)srcIn).AsSpan(),
                    ((Tensor<float>)(object)srcCos).AsSpan(),
                    ((Tensor<float>)(object)srcSin).AsSpan(),
                    ((Tensor<float>)(object)gradIn).AsWritableSpan(),
                    batch * seqLen * numHeads, seqLen, numHeads, headDim, startPosition, (float)epsilon, (float)postScale);
            }
            else
            {
                RopeL2BackwardDouble(
                    ((Tensor<double>)(object)srcDy).AsSpan(),
                    ((Tensor<double>)(object)srcIn).AsSpan(),
                    ((Tensor<double>)(object)srcCos).AsSpan(),
                    ((Tensor<double>)(object)srcSin).AsSpan(),
                    ((Tensor<double>)(object)gradIn).AsWritableSpan(),
                    batch * seqLen * numHeads, seqLen, numHeads, headDim, startPosition, epsilon, postScale);
            }
        }
        DifferentiableOps.AccumulateGrad(grads, input, gradIn, engine);
    }

    private static void RopeL2ForwardFloat(
        ReadOnlySpan<float> x, ReadOnlySpan<float> cos, ReadOnlySpan<float> sin, Span<float> y,
        int totalHeads, int seqLen, int numHeads, int headDim, int startPos, float eps, float postScale)
    {
        int half = headDim / 2;
        for (int r = 0; r < totalHeads; r++)
        {
            int t = (r / numHeads) % seqLen;
            int cOff = (startPos + t) * half;
            int xOff = r * headDim;
            float normSq = 0f;
            for (int p = 0; p < half; p++)
            {
                float x0 = x[xOff + 2 * p], x1 = x[xOff + 2 * p + 1];
                float c = cos[cOff + p], s = sin[cOff + p];
                float r0 = x0 * c - x1 * s, r1 = x0 * s + x1 * c;
                normSq += r0 * r0 + r1 * r1;
            }
            float sc = postScale / MathF.Sqrt(normSq + eps);
            for (int p = 0; p < half; p++)
            {
                float x0 = x[xOff + 2 * p], x1 = x[xOff + 2 * p + 1];
                float c = cos[cOff + p], s = sin[cOff + p];
                y[xOff + 2 * p] = (x0 * c - x1 * s) * sc;
                y[xOff + 2 * p + 1] = (x0 * s + x1 * c) * sc;
            }
        }
    }

    private static void RopeL2ForwardDouble(
        ReadOnlySpan<double> x, ReadOnlySpan<double> cos, ReadOnlySpan<double> sin, Span<double> y,
        int totalHeads, int seqLen, int numHeads, int headDim, int startPos, double eps, double postScale)
    {
        int half = headDim / 2;
        for (int r = 0; r < totalHeads; r++)
        {
            int t = (r / numHeads) % seqLen;
            int cOff = (startPos + t) * half;
            int xOff = r * headDim;
            double normSq = 0.0;
            for (int p = 0; p < half; p++)
            {
                double x0 = x[xOff + 2 * p], x1 = x[xOff + 2 * p + 1];
                double c = cos[cOff + p], s = sin[cOff + p];
                double r0 = x0 * c - x1 * s, r1 = x0 * s + x1 * c;
                normSq += r0 * r0 + r1 * r1;
            }
            double sc = postScale / Math.Sqrt(normSq + eps);
            for (int p = 0; p < half; p++)
            {
                double x0 = x[xOff + 2 * p], x1 = x[xOff + 2 * p + 1];
                double c = cos[cOff + p], s = sin[cOff + p];
                y[xOff + 2 * p] = (x0 * c - x1 * s) * sc;
                y[xOff + 2 * p + 1] = (x0 * s + x1 * c) * sc;
            }
        }
    }

    private static void RopeL2BackwardFloat(
        ReadOnlySpan<float> dy, ReadOnlySpan<float> x, ReadOnlySpan<float> cos, ReadOnlySpan<float> sin, Span<float> dx,
        int totalHeads, int seqLen, int numHeads, int headDim, int startPos, float eps, float postScale)
    {
        int half = headDim / 2;
        for (int r = 0; r < totalHeads; r++)
        {
            int t = (r / numHeads) % seqLen;
            int cOff = (startPos + t) * half;
            int xOff = r * headDim;
            float normSq = 0f;
            for (int p = 0; p < half; p++)
            {
                float x0 = x[xOff + 2 * p], x1 = x[xOff + 2 * p + 1];
                float c = cos[cOff + p], s = sin[cOff + p];
                float r0 = x0 * c - x1 * s, r1 = x0 * s + x1 * c;
                normSq += r0 * r0 + r1 * r1;
            }
            float invNorm = 1f / MathF.Sqrt(normSq + eps);
            float dot = 0f;
            for (int p = 0; p < half; p++)
            {
                float x0 = x[xOff + 2 * p], x1 = x[xOff + 2 * p + 1];
                float c = cos[cOff + p], s = sin[cOff + p];
                float u0 = (x0 * c - x1 * s) * invNorm, u1 = (x0 * s + x1 * c) * invNorm;
                dot += dy[xOff + 2 * p] * u0 + dy[xOff + 2 * p + 1] * u1;
            }
            float sc = postScale * invNorm;
            for (int p = 0; p < half; p++)
            {
                float x0 = x[xOff + 2 * p], x1 = x[xOff + 2 * p + 1];
                float c = cos[cOff + p], s = sin[cOff + p];
                float u0 = (x0 * c - x1 * s) * invNorm, u1 = (x0 * s + x1 * c) * invNorm;
                float dr0 = sc * (dy[xOff + 2 * p] - u0 * dot);
                float dr1 = sc * (dy[xOff + 2 * p + 1] - u1 * dot);
                dx[xOff + 2 * p] = dr0 * c + dr1 * s;
                dx[xOff + 2 * p + 1] = -dr0 * s + dr1 * c;
            }
        }
    }

    private static void RopeL2BackwardDouble(
        ReadOnlySpan<double> dy, ReadOnlySpan<double> x, ReadOnlySpan<double> cos, ReadOnlySpan<double> sin, Span<double> dx,
        int totalHeads, int seqLen, int numHeads, int headDim, int startPos, double eps, double postScale)
    {
        int half = headDim / 2;
        for (int r = 0; r < totalHeads; r++)
        {
            int t = (r / numHeads) % seqLen;
            int cOff = (startPos + t) * half;
            int xOff = r * headDim;
            double normSq = 0.0;
            for (int p = 0; p < half; p++)
            {
                double x0 = x[xOff + 2 * p], x1 = x[xOff + 2 * p + 1];
                double c = cos[cOff + p], s = sin[cOff + p];
                double r0 = x0 * c - x1 * s, r1 = x0 * s + x1 * c;
                normSq += r0 * r0 + r1 * r1;
            }
            double invNorm = 1.0 / Math.Sqrt(normSq + eps);
            double dot = 0.0;
            for (int p = 0; p < half; p++)
            {
                double x0 = x[xOff + 2 * p], x1 = x[xOff + 2 * p + 1];
                double c = cos[cOff + p], s = sin[cOff + p];
                double u0 = (x0 * c - x1 * s) * invNorm, u1 = (x0 * s + x1 * c) * invNorm;
                dot += dy[xOff + 2 * p] * u0 + dy[xOff + 2 * p + 1] * u1;
            }
            double sc = postScale * invNorm;
            for (int p = 0; p < half; p++)
            {
                double x0 = x[xOff + 2 * p], x1 = x[xOff + 2 * p + 1];
                double c = cos[cOff + p], s = sin[cOff + p];
                double u0 = (x0 * c - x1 * s) * invNorm, u1 = (x0 * s + x1 * c) * invNorm;
                double dr0 = sc * (dy[xOff + 2 * p] - u0 * dot);
                double dr1 = sc * (dy[xOff + 2 * p + 1] - u1 * dot);
                dx[xOff + 2 * p] = dr0 * c + dr1 * s;
                dx[xOff + 2 * p + 1] = -dr0 * s + dr1 * c;
            }
        }
    }

    public virtual Tensor<T> RopeQuadraticOuterFeatures<T>(
        Tensor<T> input,
        Tensor<T> cos,
        Tensor<T> sin,
        int numHeads,
        int outputHeadDim = 0,
        int startPosition = 0)
    {
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
        long reqRows = (long)startPosition + seqLen;
        if (cos.Rank != 2 || sin.Rank != 2 || cos.Shape[1] != half || sin.Shape[1] != half
            || cos.Shape[0] < reqRows || sin.Shape[0] < reqRows)
            throw new ArgumentException($"cos/sin must have shape [>= {reqRows}, {half}].", nameof(cos));

        int[] outShape = input.Rank == 3
            ? new[] { batch, seqLen, checked(numHeads * effOutDim) }
            : new[] { seqLen, checked(numHeads * effOutDim) };

        if (GraphMode.IsActive || (typeof(T) != typeof(float) && typeof(T) != typeof(double)))
        {
            return RopeQuadraticOuterFeaturesComposed(this, input, cos, sin, batch, seqLen, numHeads, mb, effOutDim, startPosition, outShape);
        }

        var output = new Tensor<T>(outShape);
        Tensor<T>? srcCos = null;
        Tensor<T>? srcSin = null;
        if (input.Length > 0)
        {
            var srcIn = input.IsContiguous ? input : input.Contiguous();
            srcCos = cos.IsContiguous ? cos : cos.Contiguous();
            srcSin = sin.IsContiguous ? sin : sin.Contiguous();
            if (typeof(T) == typeof(float))
            {
                RopeQuadForwardFloat(
                    ((Tensor<float>)(object)srcIn).AsSpan(),
                    ((Tensor<float>)(object)srcCos).AsSpan(),
                    ((Tensor<float>)(object)srcSin).AsSpan(),
                    ((Tensor<float>)(object)output).AsWritableSpan(),
                    batch * seqLen * numHeads, seqLen, numHeads, mb, effOutDim, startPosition);
            }
            else
            {
                RopeQuadForwardDouble(
                    ((Tensor<double>)(object)srcIn).AsSpan(),
                    ((Tensor<double>)(object)srcCos).AsSpan(),
                    ((Tensor<double>)(object)srcSin).AsSpan(),
                    ((Tensor<double>)(object)output).AsWritableSpan(),
                    batch * seqLen * numHeads, seqLen, numHeads, mb, effOutDim, startPosition);
            }
        }

        if (GradientTape<T>.Current is not null && !NoGradScope<T>.IsSuppressed)
        {
            var savedCos = SnapshotRopeWindow(srcCos ?? (cos.IsContiguous ? cos : cos.Contiguous()), startPosition, seqLen, half);
            var savedSin = SnapshotRopeWindow(srcSin ?? (sin.IsContiguous ? sin : sin.Contiguous()), startPosition, seqLen, half);
            DifferentiableOps.RecordIfActive<T>(
                "RopeQuadraticOuterFeatures",
                output,
                new[] { input },
                RopeQuadraticOuterFeaturesBackward<T>,
                savedState: new object[] { savedCos, savedSin, batch, seqLen, numHeads, mb, effOutDim, 0, outShape });
        }
        return output;
    }

    internal static Tensor<T> RopeQuadraticOuterFeaturesComposed<T>(
        IEngine engine, Tensor<T> input, Tensor<T> cos, Tensor<T> sin,
        int batch, int seqLen, int numHeads, int mb, int effOutDim, int startPosition, int[] outShape)
    {
        int mb2 = mb * mb;
        var z = engine.TensorPermute(engine.Reshape(input, new[] { batch, seqLen, numHeads, mb }), new[] { 0, 2, 1, 3 });
        var zRot = engine.ApplyRoPEInterleaved(z, cos, sin, startPosition);
        var outer = engine.TensorMultiply(
            engine.Reshape(zRot, new[] { batch, numHeads, seqLen, mb, 1 }),
            engine.Reshape(zRot, new[] { batch, numHeads, seqLen, 1, mb }));
        var phi = engine.TensorPermute(engine.Reshape(outer, new[] { batch, numHeads, seqLen, mb2 }), new[] { 0, 2, 1, 3 });
        if (effOutDim > mb2)
        {
            int rows = batch * seqLen * numHeads;
            var pad = TensorScope.Detach(new Tensor<T>(new[] { rows, effOutDim - mb2 }));
            var phi2d = engine.TensorConcatenate(new[] { engine.Reshape(phi, new[] { rows, mb2 }), pad }, axis: 1);
            return engine.Reshape(phi2d, outShape);
        }
        return engine.Reshape(phi, outShape);
    }

    internal static void RopeQuadraticOuterFeaturesBackward<T>(
        Tensor<T> gradOutput,
        Tensor<T>[] inputs,
        Tensor<T> output,
        object[] savedState,
        IEngine engine,
        Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var input = inputs[0];
        var cos = (Tensor<T>)savedState[0];
        var sin = (Tensor<T>)savedState[1];
        int batch = (int)savedState[2];
        int seqLen = (int)savedState[3];
        int numHeads = (int)savedState[4];
        int mb = (int)savedState[5];
        int effOutDim = (int)savedState[6];
        int startPosition = (int)savedState[7];
        int[] outShape = (int[])savedState[8];

        if (DifferentiableOps._isBackwardCreateGraph || (typeof(T) != typeof(float) && typeof(T) != typeof(double)))
        {
            // The tape this create-graph backward records on. The nested tape below is disposed on return, so its
            // entries move to this one and the gradients accumulate on it; otherwise a higher order sees no graph.
            var outerTape = GradientTape<T>.Current;
            using var tape = new GradientTape<T>(new GradientTapeOptions { Persistent = false });
            var y = RopeQuadraticOuterFeaturesComposed(engine, input, cos, sin, batch, seqLen, numHeads, mb, effOutDim, startPosition, outShape);
            var loss = engine.ReduceSum(engine.TensorMultiply(y, gradOutput), null);
            var sub = tape.ComputeGradients(loss, new[] { input }, createGraph: DifferentiableOps._isBackwardCreateGraph);
            if (DifferentiableOps._isBackwardCreateGraph && outerTape is not null) tape.TransferEntriesTo(outerTape);
            using var recordOnOuter = GradientTape<T>.RecordOn(outerTape);
            if (sub.TryGetValue(input, out var gi)) DifferentiableOps.AccumulateGrad(grads, input, gi, engine);
            return;
        }

        var gradIn = new Tensor<T>((int[])input._shape.Clone());
        if (input.Length > 0)
        {
            var srcDy = gradOutput.IsContiguous ? gradOutput : gradOutput.Contiguous();
            var srcIn = input.IsContiguous ? input : input.Contiguous();
            var srcCos = cos.IsContiguous ? cos : cos.Contiguous();
            var srcSin = sin.IsContiguous ? sin : sin.Contiguous();
            if (typeof(T) == typeof(float))
            {
                RopeQuadBackwardFloat(
                    ((Tensor<float>)(object)srcDy).AsSpan(),
                    ((Tensor<float>)(object)srcIn).AsSpan(),
                    ((Tensor<float>)(object)srcCos).AsSpan(),
                    ((Tensor<float>)(object)srcSin).AsSpan(),
                    ((Tensor<float>)(object)gradIn).AsWritableSpan(),
                    batch * seqLen * numHeads, seqLen, numHeads, mb, effOutDim, startPosition);
            }
            else
            {
                RopeQuadBackwardDouble(
                    ((Tensor<double>)(object)srcDy).AsSpan(),
                    ((Tensor<double>)(object)srcIn).AsSpan(),
                    ((Tensor<double>)(object)srcCos).AsSpan(),
                    ((Tensor<double>)(object)srcSin).AsSpan(),
                    ((Tensor<double>)(object)gradIn).AsWritableSpan(),
                    batch * seqLen * numHeads, seqLen, numHeads, mb, effOutDim, startPosition);
            }
        }
        DifferentiableOps.AccumulateGrad(grads, input, gradIn, engine);
    }

    private static void RopeQuadForwardFloat(
        ReadOnlySpan<float> x, ReadOnlySpan<float> cos, ReadOnlySpan<float> sin, Span<float> y,
        int totalHeads, int seqLen, int numHeads, int mb, int outDim, int startPos)
    {
        int half = mb / 2;
        int mb2 = mb * mb;
        Span<float> rBuf = mb <= 128 ? stackalloc float[mb] : new float[mb];
        for (int r = 0; r < totalHeads; r++)
        {
            int t = (r / numHeads) % seqLen;
            int cOff = (startPos + t) * half;
            int xOff = r * mb;
            int yOff = r * outDim;
            for (int p = 0; p < half; p++)
            {
                float x0 = x[xOff + 2 * p], x1 = x[xOff + 2 * p + 1];
                float c = cos[cOff + p], s = sin[cOff + p];
                rBuf[2 * p] = x0 * c - x1 * s;
                rBuf[2 * p + 1] = x0 * s + x1 * c;
            }
            for (int u = 0; u < mb; u++)
            {
                float ru = rBuf[u];
                for (int v = 0; v < mb; v++)
                    y[yOff + u * mb + v] = ru * rBuf[v];
            }
            if (outDim > mb2)
                y.Slice(yOff + mb2, outDim - mb2).Clear();
        }
    }

    private static void RopeQuadForwardDouble(
        ReadOnlySpan<double> x, ReadOnlySpan<double> cos, ReadOnlySpan<double> sin, Span<double> y,
        int totalHeads, int seqLen, int numHeads, int mb, int outDim, int startPos)
    {
        int half = mb / 2;
        int mb2 = mb * mb;
        Span<double> rBuf = mb <= 128 ? stackalloc double[mb] : new double[mb];
        for (int r = 0; r < totalHeads; r++)
        {
            int t = (r / numHeads) % seqLen;
            int cOff = (startPos + t) * half;
            int xOff = r * mb;
            int yOff = r * outDim;
            for (int p = 0; p < half; p++)
            {
                double x0 = x[xOff + 2 * p], x1 = x[xOff + 2 * p + 1];
                double c = cos[cOff + p], s = sin[cOff + p];
                rBuf[2 * p] = x0 * c - x1 * s;
                rBuf[2 * p + 1] = x0 * s + x1 * c;
            }
            for (int u = 0; u < mb; u++)
            {
                double ru = rBuf[u];
                for (int v = 0; v < mb; v++)
                    y[yOff + u * mb + v] = ru * rBuf[v];
            }
            if (outDim > mb2)
                y.Slice(yOff + mb2, outDim - mb2).Clear();
        }
    }

    private static void RopeQuadBackwardFloat(
        ReadOnlySpan<float> dy, ReadOnlySpan<float> x, ReadOnlySpan<float> cos, ReadOnlySpan<float> sin, Span<float> dx,
        int totalHeads, int seqLen, int numHeads, int mb, int outDim, int startPos)
    {
        int half = mb / 2;
        Span<float> rBuf = mb <= 128 ? stackalloc float[mb] : new float[mb];
        Span<float> drBuf = mb <= 128 ? stackalloc float[mb] : new float[mb];
        for (int r = 0; r < totalHeads; r++)
        {
            int t = (r / numHeads) % seqLen;
            int cOff = (startPos + t) * half;
            int xOff = r * mb;
            int yOff = r * outDim;
            for (int p = 0; p < half; p++)
            {
                float x0 = x[xOff + 2 * p], x1 = x[xOff + 2 * p + 1];
                float c = cos[cOff + p], s = sin[cOff + p];
                rBuf[2 * p] = x0 * c - x1 * s;
                rBuf[2 * p + 1] = x0 * s + x1 * c;
            }
            for (int u = 0; u < mb; u++)
            {
                float acc = 0f;
                for (int v = 0; v < mb; v++)
                    acc += (dy[yOff + u * mb + v] + dy[yOff + v * mb + u]) * rBuf[v];
                drBuf[u] = acc;
            }
            for (int p = 0; p < half; p++)
            {
                float c = cos[cOff + p], s = sin[cOff + p];
                float dr0 = drBuf[2 * p], dr1 = drBuf[2 * p + 1];
                dx[xOff + 2 * p] = dr0 * c + dr1 * s;
                dx[xOff + 2 * p + 1] = -dr0 * s + dr1 * c;
            }
        }
    }

    private static void RopeQuadBackwardDouble(
        ReadOnlySpan<double> dy, ReadOnlySpan<double> x, ReadOnlySpan<double> cos, ReadOnlySpan<double> sin, Span<double> dx,
        int totalHeads, int seqLen, int numHeads, int mb, int outDim, int startPos)
    {
        int half = mb / 2;
        Span<double> rBuf = mb <= 128 ? stackalloc double[mb] : new double[mb];
        Span<double> drBuf = mb <= 128 ? stackalloc double[mb] : new double[mb];
        for (int r = 0; r < totalHeads; r++)
        {
            int t = (r / numHeads) % seqLen;
            int cOff = (startPos + t) * half;
            int xOff = r * mb;
            int yOff = r * outDim;
            for (int p = 0; p < half; p++)
            {
                double x0 = x[xOff + 2 * p], x1 = x[xOff + 2 * p + 1];
                double c = cos[cOff + p], s = sin[cOff + p];
                rBuf[2 * p] = x0 * c - x1 * s;
                rBuf[2 * p + 1] = x0 * s + x1 * c;
            }
            for (int u = 0; u < mb; u++)
            {
                double acc = 0.0;
                for (int v = 0; v < mb; v++)
                    acc += (dy[yOff + u * mb + v] + dy[yOff + v * mb + u]) * rBuf[v];
                drBuf[u] = acc;
            }
            for (int p = 0; p < half; p++)
            {
                double c = cos[cOff + p], s = sin[cOff + p];
                double dr0 = drBuf[2 * p], dr1 = drBuf[2 * p + 1];
                dx[xOff + 2 * p] = dr0 * c + dr1 * s;
                dx[xOff + 2 * p + 1] = -dr0 * s + dr1 * c;
            }
        }
    }

    public virtual Tensor<T> ProjectiveGatedResidual<T>(
        Tensor<T> baseHidden,
        Tensor<T> proposal,
        Tensor<T> parallelGateLogit,
        double epsilon = 1e-6)
    {
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

        if (GraphMode.IsActive || (typeof(T) != typeof(float) && typeof(T) != typeof(double)))
        {
            return ProjectiveGatedResidualComposed(this, baseHidden, proposal, parallelGateLogit, dim, epsilon);
        }

        int rows = baseHidden.Length / dim;
        var output = new Tensor<T>((int[])baseHidden._shape.Clone());
        if (rows > 0)
        {
            var srcH = baseHidden.IsContiguous ? baseHidden : baseHidden.Contiguous();
            var srcU = proposal.IsContiguous ? proposal : proposal.Contiguous();
            var srcG = parallelGateLogit.IsContiguous ? parallelGateLogit : parallelGateLogit.Contiguous();
            if (typeof(T) == typeof(float))
            {
                ProjResForwardFloat(
                    ((Tensor<float>)(object)srcH).AsSpan(),
                    ((Tensor<float>)(object)srcU).AsSpan(),
                    ((Tensor<float>)(object)srcG).AsSpan(),
                    ((Tensor<float>)(object)output).AsWritableSpan(),
                    rows, dim, (float)epsilon);
            }
            else
            {
                ProjResForwardDouble(
                    ((Tensor<double>)(object)srcH).AsSpan(),
                    ((Tensor<double>)(object)srcU).AsSpan(),
                    ((Tensor<double>)(object)srcG).AsSpan(),
                    ((Tensor<double>)(object)output).AsWritableSpan(),
                    rows, dim, epsilon);
            }
        }

        DifferentiableOps.RecordIfActive<T>(
            "ProjectiveGatedResidual",
            output,
            new[] { baseHidden, proposal, parallelGateLogit },
            ProjectiveGatedResidualBackward<T>,
            savedState: new object[] { dim, epsilon });
        return output;
    }

    internal static Tensor<T> ProjectiveGatedResidualComposed<T>(
        IEngine engine, Tensor<T> baseHidden, Tensor<T> proposal, Tensor<T> parallelGateLogit, int dim, double epsilon)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int rows = baseHidden.Length / dim;
        var h2d = baseHidden.Rank == 2 ? baseHidden : engine.Reshape(baseHidden, new[] { rows, dim });
        var u2d = proposal.Rank == 2 ? proposal : engine.Reshape(proposal, new[] { rows, dim });
        var dot = engine.ReduceSum(engine.TensorMultiply(u2d, h2d), new[] { 1 }, keepDims: true);
        var normSq = engine.TensorAddScalar(
            engine.ReduceSum(engine.TensorMultiply(h2d, h2d), new[] { 1 }, keepDims: true),
            ops.FromDouble(epsilon));
        var coeff = engine.TensorDivide(dot, normSq);
        var parallel = engine.TensorMultiply(h2d, engine.TensorBroadcastTo(coeff, new[] { rows, dim }));
        var deltaPerp = engine.TensorSubtract(u2d, parallel);
        var parGate = engine.TensorBroadcastTo(engine.Reshape(engine.Sigmoid(parallelGateLogit), new[] { 1, 1 }), new[] { rows, dim });
        var sum2d = engine.TensorAdd(h2d, engine.TensorAdd(deltaPerp, engine.TensorMultiply(parallel, parGate)));
        return baseHidden.Rank == 2 ? sum2d : engine.Reshape(sum2d, (int[])baseHidden._shape.Clone());
    }

    internal static void ProjectiveGatedResidualBackward<T>(
        Tensor<T> gradOutput,
        Tensor<T>[] inputs,
        Tensor<T> output,
        object[] savedState,
        IEngine engine,
        Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        int dim = (int)savedState[0];
        double epsilon = (double)savedState[1];
        var baseHidden = inputs[0];
        var proposal = inputs[1];
        var gateLogit = inputs[2];

        if (DifferentiableOps._isBackwardCreateGraph || (typeof(T) != typeof(float) && typeof(T) != typeof(double)))
        {
            // The tape this create-graph backward records on. The nested tape below is disposed on return, so its
            // entries move to this one and the gradients accumulate on it; otherwise a higher order sees no graph.
            var outerTape = GradientTape<T>.Current;
            using var tape = new GradientTape<T>(new GradientTapeOptions { Persistent = false });
            var y = ProjectiveGatedResidualComposed(engine, baseHidden, proposal, gateLogit, dim, epsilon);
            var loss = engine.ReduceSum(engine.TensorMultiply(y, gradOutput), null);
            var sub = tape.ComputeGradients(loss, new[] { baseHidden, proposal, gateLogit }, createGraph: DifferentiableOps._isBackwardCreateGraph);
            if (DifferentiableOps._isBackwardCreateGraph && outerTape is not null) tape.TransferEntriesTo(outerTape);
            using var recordOnOuter = GradientTape<T>.RecordOn(outerTape);
            if (sub.TryGetValue(baseHidden, out var gh)) DifferentiableOps.AccumulateGrad(grads, baseHidden, gh, engine);
            if (sub.TryGetValue(proposal, out var gu)) DifferentiableOps.AccumulateGrad(grads, proposal, gu, engine);
            if (sub.TryGetValue(gateLogit, out var gg)) DifferentiableOps.AccumulateGrad(grads, gateLogit, gg, engine);
            return;
        }

        int rows = baseHidden.Length / dim;
        var gradH = new Tensor<T>((int[])baseHidden._shape.Clone());
        var gradU = new Tensor<T>((int[])proposal._shape.Clone());
        var gradG = new Tensor<T>((int[])gateLogit._shape.Clone());
        if (rows > 0)
        {
            var srcDy = gradOutput.IsContiguous ? gradOutput : gradOutput.Contiguous();
            var srcH = baseHidden.IsContiguous ? baseHidden : baseHidden.Contiguous();
            var srcU = proposal.IsContiguous ? proposal : proposal.Contiguous();
            var srcG = gateLogit.IsContiguous ? gateLogit : gateLogit.Contiguous();
            if (typeof(T) == typeof(float))
            {
                ProjResBackwardFloat(
                    ((Tensor<float>)(object)srcDy).AsSpan(),
                    ((Tensor<float>)(object)srcH).AsSpan(),
                    ((Tensor<float>)(object)srcU).AsSpan(),
                    ((Tensor<float>)(object)srcG).AsSpan(),
                    ((Tensor<float>)(object)gradH).AsWritableSpan(),
                    ((Tensor<float>)(object)gradU).AsWritableSpan(),
                    ((Tensor<float>)(object)gradG).AsWritableSpan(),
                    rows, dim, (float)epsilon);
            }
            else
            {
                ProjResBackwardDouble(
                    ((Tensor<double>)(object)srcDy).AsSpan(),
                    ((Tensor<double>)(object)srcH).AsSpan(),
                    ((Tensor<double>)(object)srcU).AsSpan(),
                    ((Tensor<double>)(object)srcG).AsSpan(),
                    ((Tensor<double>)(object)gradH).AsWritableSpan(),
                    ((Tensor<double>)(object)gradU).AsWritableSpan(),
                    ((Tensor<double>)(object)gradG).AsWritableSpan(),
                    rows, dim, epsilon);
            }
        }
        DifferentiableOps.AccumulateGrad(grads, baseHidden, gradH, engine);
        DifferentiableOps.AccumulateGrad(grads, proposal, gradU, engine);
        DifferentiableOps.AccumulateGrad(grads, gateLogit, gradG, engine);
    }

    private static void ProjResForwardFloat(
        ReadOnlySpan<float> h, ReadOnlySpan<float> u, ReadOnlySpan<float> gateLogit, Span<float> y,
        int rows, int dim, float eps)
    {
        float g = 1f / (1f + MathF.Exp(-gateLogit[0]));
        float beta = 1f - g;
        for (int m = 0; m < rows; m++)
        {
            int off = m * dim;
            float dot = 0f, normSq = 0f;
            for (int j = 0; j < dim; j++)
            {
                float hj = h[off + j];
                dot += u[off + j] * hj;
                normSq += hj * hj;
            }
            float scaleH = 1f - beta * (dot / (normSq + eps));
            for (int j = 0; j < dim; j++)
                y[off + j] = u[off + j] + scaleH * h[off + j];
        }
    }

    private static void ProjResForwardDouble(
        ReadOnlySpan<double> h, ReadOnlySpan<double> u, ReadOnlySpan<double> gateLogit, Span<double> y,
        int rows, int dim, double eps)
    {
        double g = 1.0 / (1.0 + Math.Exp(-gateLogit[0]));
        double beta = 1.0 - g;
        for (int m = 0; m < rows; m++)
        {
            int off = m * dim;
            double dot = 0.0, normSq = 0.0;
            for (int j = 0; j < dim; j++)
            {
                double hj = h[off + j];
                dot += u[off + j] * hj;
                normSq += hj * hj;
            }
            double scaleH = 1.0 - beta * (dot / (normSq + eps));
            for (int j = 0; j < dim; j++)
                y[off + j] = u[off + j] + scaleH * h[off + j];
        }
    }

    private static void ProjResBackwardFloat(
        ReadOnlySpan<float> dy, ReadOnlySpan<float> h, ReadOnlySpan<float> u, ReadOnlySpan<float> gateLogit,
        Span<float> dh, Span<float> du, Span<float> dg, int rows, int dim, float eps)
    {
        float g = 1f / (1f + MathF.Exp(-gateLogit[0]));
        float beta = 1f - g;
        float gateSum = 0f;
        for (int m = 0; m < rows; m++)
        {
            int off = m * dim;
            float dot = 0f, normSq = 0f, pm = 0f;
            for (int j = 0; j < dim; j++)
            {
                float hj = h[off + j], uj = u[off + j], dyj = dy[off + j];
                dot += uj * hj;
                normSq += hj * hj;
                pm += dyj * hj;
            }
            float denom = normSq + eps;
            float coeff = dot / denom;
            float scaleH = 1f - beta * coeff;
            float kFactor = (beta * pm) / denom;
            gateSum += coeff * pm;
            for (int j = 0; j < dim; j++)
            {
                float hj = h[off + j], uj = u[off + j], dyj = dy[off + j];
                du[off + j] = dyj - kFactor * hj;
                dh[off + j] = scaleH * dyj - kFactor * (uj - 2f * coeff * hj);
            }
        }
        dg[0] = g * (1f - g) * gateSum;
    }

    private static void ProjResBackwardDouble(
        ReadOnlySpan<double> dy, ReadOnlySpan<double> h, ReadOnlySpan<double> u, ReadOnlySpan<double> gateLogit,
        Span<double> dh, Span<double> du, Span<double> dg, int rows, int dim, double eps)
    {
        double g = 1.0 / (1.0 + Math.Exp(-gateLogit[0]));
        double beta = 1.0 - g;
        double gateSum = 0.0;
        for (int m = 0; m < rows; m++)
        {
            int off = m * dim;
            double dot = 0.0, normSq = 0.0, pm = 0.0;
            for (int j = 0; j < dim; j++)
            {
                double hj = h[off + j], uj = u[off + j], dyj = dy[off + j];
                dot += uj * hj;
                normSq += hj * hj;
                pm += dyj * hj;
            }
            double denom = normSq + eps;
            double coeff = dot / denom;
            double scaleH = 1.0 - beta * coeff;
            double kFactor = (beta * pm) / denom;
            gateSum += coeff * pm;
            for (int j = 0; j < dim; j++)
            {
                double hj = h[off + j], uj = u[off + j], dyj = dy[off + j];
                du[off + j] = dyj - kFactor * hj;
                dh[off + j] = scaleH * dyj - kFactor * (uj - 2.0 * coeff * hj);
            }
        }
        dg[0] = g * (1.0 - g) * gateSum;
    }

    public virtual Tensor<T> SpectralCirculantMatrix<T>(Tensor<T> diagScale, Tensor<T> spectrumRe, Tensor<T> spectrumIm)
    {
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

        if (GraphMode.IsActive || (typeof(T) != typeof(float) && typeof(T) != typeof(double)))
        {
            return SpectralCirculantMatrixComposed(this, diagScale, spectrumRe, spectrumIm, batch, dim);
        }

        int[] outShape = diagScale.Rank == 2 ? new[] { batch, dim, dim } : new[] { dim, dim };
        var output = new Tensor<T>(outShape);
        if (batch > 0)
        {
            var srcD = diagScale.IsContiguous ? diagScale : diagScale.Contiguous();
            var srcRe = spectrumRe.IsContiguous ? spectrumRe : spectrumRe.Contiguous();
            var srcIm = spectrumIm.IsContiguous ? spectrumIm : spectrumIm.Contiguous();
            if (typeof(T) == typeof(float))
            {
                SpectralCirculantForwardFloat(
                    ((Tensor<float>)(object)srcD).AsSpan(),
                    ((Tensor<float>)(object)srcRe).AsSpan(),
                    ((Tensor<float>)(object)srcIm).AsSpan(),
                    ((Tensor<float>)(object)output).AsWritableSpan(),
                    batch, dim);
            }
            else
            {
                SpectralCirculantForwardDouble(
                    ((Tensor<double>)(object)srcD).AsSpan(),
                    ((Tensor<double>)(object)srcRe).AsSpan(),
                    ((Tensor<double>)(object)srcIm).AsSpan(),
                    ((Tensor<double>)(object)output).AsWritableSpan(),
                    batch, dim);
            }
        }

        DifferentiableOps.RecordIfActive<T>(
            "SpectralCirculantMatrix",
            output,
            new[] { diagScale, spectrumRe, spectrumIm },
            SpectralCirculantMatrixBackward<T>,
            savedState: new object[] { batch, dim });
        return output;
    }

    internal static Tensor<T> SpectralCirculantMatrixComposed<T>(
        IEngine engine, Tensor<T> diagScale, Tensor<T> spectrumRe, Tensor<T> spectrumIm, int batch, int dim)
    {
        if (batch == 0)
            return engine.Reshape(diagScale, new[] { 0, dim, dim });
        var ops = MathHelper.GetNumericOperations<T>();
        int nf = dim / 2 + 1;
        var idArr = new T[dim * dim];
        for (int i = 0; i < dim; i++) idArr[i * dim + i] = ops.One;
        var id2d = TensorScope.Detach(new Tensor<T>(new[] { dim, dim }, new Vector<T>(idArr)));
        var swap = TensorScope.Detach(new Tensor<T>(new[] { 2, 2 }, new Vector<T>(new[] { ops.Zero, ops.One, ops.One, ops.Zero })));
        var onesPair = TensorScope.Detach(new Tensor<T>(new[] { 1, 2 }, new Vector<T>(new[] { ops.One, ops.One })));
        var signPair = TensorScope.Detach(new Tensor<T>(new[] { 1, 2 }, new Vector<T>(new[] { ops.FromDouble(-1.0), ops.One })));

        var d2 = engine.Reshape(diagScale, new[] { batch, dim });
        var re2 = engine.Reshape(spectrumRe, new[] { batch, nf });
        var im2 = engine.Reshape(spectrumIm, new[] { batch, nf });
        var mats = new Tensor<T>[batch];
        for (int b = 0; b < batch; b++)
        {
            var db = engine.TensorNarrow(d2, 0, b, 1);
            var reb = engine.Reshape(engine.TensorNarrow(re2, 0, b, 1), new[] { nf, 1 });
            var imb = engine.Reshape(engine.TensorNarrow(im2, 0, b, 1), new[] { nf, 1 });
            var xScaled = engine.TensorMultiply(id2d, db);
            var z = engine.RFFT(xScaled);
            var sw = engine.Reshape(engine.TensorMatMul(engine.Reshape(z, new[] { dim * nf, 2 }), swap), new[] { dim, 2 * nf });
            var reP = engine.Reshape(engine.TensorMultiply(reb, onesPair), new[] { 1, 2 * nf });
            var imP = engine.Reshape(engine.TensorMultiply(imb, signPair), new[] { 1, 2 * nf });
            var y = engine.IRFFT(engine.TensorAdd(engine.TensorMultiply(z, reP), engine.TensorMultiply(sw, imP)), dim);
            mats[b] = engine.Reshape(y, new[] { 1, dim, dim });
        }
        var stacked = batch == 1 ? mats[0] : engine.TensorConcatenate(mats, axis: 0);
        return diagScale.Rank == 2 ? stacked : engine.Reshape(stacked, new[] { dim, dim });
    }

    internal static void SpectralCirculantMatrixBackward<T>(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output, object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        int batch = (int)savedState[0];
        int dim = (int)savedState[1];
        var dIn = inputs[0];
        var reIn = inputs[1];
        var imIn = inputs[2];
        if (DifferentiableOps._isBackwardCreateGraph || (typeof(T) != typeof(float) && typeof(T) != typeof(double)))
        {
            // The tape this create-graph backward records on. The nested tape below is disposed on return, so its
            // entries move to this one and the gradients accumulate on it; otherwise a higher order sees no graph.
            var outerTape = GradientTape<T>.Current;
            using var tape = new GradientTape<T>(new GradientTapeOptions { Persistent = false });
            var y = SpectralCirculantMatrixComposed(engine, dIn, reIn, imIn, batch, dim);
            var sub = tape.ComputeGradients(engine.ReduceSum(engine.TensorMultiply(y, gradOutput), null), new[] { dIn, reIn, imIn }, createGraph: DifferentiableOps._isBackwardCreateGraph);
            if (DifferentiableOps._isBackwardCreateGraph && outerTape is not null) tape.TransferEntriesTo(outerTape);
            using var recordOnOuter = GradientTape<T>.RecordOn(outerTape);
            if (sub.TryGetValue(dIn, out var gd)) DifferentiableOps.AccumulateGrad(grads, dIn, gd, engine);
            if (sub.TryGetValue(reIn, out var gr)) DifferentiableOps.AccumulateGrad(grads, reIn, gr, engine);
            if (sub.TryGetValue(imIn, out var gi)) DifferentiableOps.AccumulateGrad(grads, imIn, gi, engine);
            return;
        }

        var gDiag = new Tensor<T>((int[])dIn._shape.Clone());
        var gRe = new Tensor<T>((int[])reIn._shape.Clone());
        var gIm = new Tensor<T>((int[])imIn._shape.Clone());
        if (batch > 0)
        {
            var sDy = gradOutput.IsContiguous ? gradOutput : gradOutput.Contiguous();
            var sD = dIn.IsContiguous ? dIn : dIn.Contiguous();
            var sRe = reIn.IsContiguous ? reIn : reIn.Contiguous();
            var sIm = imIn.IsContiguous ? imIn : imIn.Contiguous();
            if (typeof(T) == typeof(float))
            {
                SpectralCirculantBackwardFloat(
                    ((Tensor<float>)(object)sDy).AsSpan(),
                    ((Tensor<float>)(object)sD).AsSpan(),
                    ((Tensor<float>)(object)sRe).AsSpan(),
                    ((Tensor<float>)(object)sIm).AsSpan(),
                    ((Tensor<float>)(object)gDiag).AsWritableSpan(),
                    ((Tensor<float>)(object)gRe).AsWritableSpan(),
                    ((Tensor<float>)(object)gIm).AsWritableSpan(),
                    batch, dim);
            }
            else
            {
                SpectralCirculantBackwardDouble(
                    ((Tensor<double>)(object)sDy).AsSpan(),
                    ((Tensor<double>)(object)sD).AsSpan(),
                    ((Tensor<double>)(object)sRe).AsSpan(),
                    ((Tensor<double>)(object)sIm).AsSpan(),
                    ((Tensor<double>)(object)gDiag).AsWritableSpan(),
                    ((Tensor<double>)(object)gRe).AsWritableSpan(),
                    ((Tensor<double>)(object)gIm).AsWritableSpan(),
                    batch, dim);
            }
        }
        DifferentiableOps.AccumulateGrad(grads, dIn, gDiag, engine);
        DifferentiableOps.AccumulateGrad(grads, reIn, gRe, engine);
        DifferentiableOps.AccumulateGrad(grads, imIn, gIm, engine);
    }

    private static void SpectralCirculantForwardFloat(
        ReadOnlySpan<float> diag, ReadOnlySpan<float> specRe, ReadOnlySpan<float> specIm, Span<float> outMat, int batch, int dim)
    {
        int half = dim / 2, nf = half + 1;
        double twoPiOverD = 2.0 * Math.PI / dim;
        double invD = 1.0 / dim;
        var h = new float[dim];
        for (int b = 0; b < batch; b++)
        {
            int sOff = b * nf;
            for (int m = 0; m < dim; m++)
            {
                double sum = specRe[sOff] + ((m & 1) == 0 ? 1.0 : -1.0) * specRe[sOff + half];
                for (int f = 1; f < half; f++)
                {
                    double ang = twoPiOverD * ((f * m) % dim);
                    sum += 2.0 * (specRe[sOff + f] * Math.Cos(ang) - specIm[sOff + f] * Math.Sin(ang));
                }
                h[m] = (float)(sum * invD);
            }
            int mBase = b * dim * dim, dBase = b * dim;
            for (int r = 0; r < dim; r++)
            {
                float dr = diag[dBase + r];
                for (int c = 0; c < dim; c++)
                {
                    int shift = c - r;
                    if (shift < 0) shift += dim;
                    outMat[mBase + r * dim + c] = dr * h[shift];
                }
            }
        }
    }

    private static void SpectralCirculantForwardDouble(
        ReadOnlySpan<double> diag, ReadOnlySpan<double> specRe, ReadOnlySpan<double> specIm, Span<double> outMat, int batch, int dim)
    {
        int half = dim / 2, nf = half + 1;
        double twoPiOverD = 2.0 * Math.PI / dim;
        double invD = 1.0 / dim;
        var h = new double[dim];
        for (int b = 0; b < batch; b++)
        {
            int sOff = b * nf;
            for (int m = 0; m < dim; m++)
            {
                double sum = specRe[sOff] + ((m & 1) == 0 ? 1.0 : -1.0) * specRe[sOff + half];
                for (int f = 1; f < half; f++)
                {
                    double ang = twoPiOverD * ((f * m) % dim);
                    sum += 2.0 * (specRe[sOff + f] * Math.Cos(ang) - specIm[sOff + f] * Math.Sin(ang));
                }
                h[m] = sum * invD;
            }
            int mBase = b * dim * dim, dBase = b * dim;
            for (int r = 0; r < dim; r++)
            {
                double dr = diag[dBase + r];
                for (int c = 0; c < dim; c++)
                {
                    int shift = c - r;
                    if (shift < 0) shift += dim;
                    outMat[mBase + r * dim + c] = dr * h[shift];
                }
            }
        }
    }

    private static void SpectralCirculantBackwardFloat(
        ReadOnlySpan<float> dy, ReadOnlySpan<float> diag, ReadOnlySpan<float> specRe, ReadOnlySpan<float> specIm,
        Span<float> dDiag, Span<float> dRe, Span<float> dIm, int batch, int dim)
    {
        int half = dim / 2, nf = half + 1;
        double twoPiOverD = 2.0 * Math.PI / dim;
        double invD = 1.0 / dim;
        var h = new float[dim];
        var dh = new double[dim];
        for (int b = 0; b < batch; b++)
        {
            int sOff = b * nf, mBase = b * dim * dim, vBase = b * dim;
            for (int m = 0; m < dim; m++)
            {
                double sum = specRe[sOff] + ((m & 1) == 0 ? 1.0 : -1.0) * specRe[sOff + half];
                for (int f = 1; f < half; f++)
                {
                    double ang = twoPiOverD * ((f * m) % dim);
                    sum += 2.0 * (specRe[sOff + f] * Math.Cos(ang) - specIm[sOff + f] * Math.Sin(ang));
                }
                h[m] = (float)(sum * invD);
            }
            for (int idx = 0; idx < dim; idx++)
            {
                double accD = 0.0, accH = 0.0;
                for (int k = 0; k < dim; k++)
                {
                    int shift = k - idx; if (shift < 0) shift += dim;
                    accD += dy[mBase + idx * dim + k] * h[shift];
                    int col = k + idx; if (col >= dim) col -= dim;
                    accH += dy[mBase + k * dim + col] * diag[vBase + k];
                }
                dDiag[vBase + idx] = (float)accD;
                dh[idx] = accH;
            }
            double s0 = 0.0, sn = 0.0;
            for (int m = 0; m < dim; m++) { s0 += dh[m]; sn += ((m & 1) == 0 ? 1.0 : -1.0) * dh[m]; }
            dRe[sOff] = (float)(s0 * invD); dIm[sOff] = 0f;
            dRe[sOff + half] = (float)(sn * invD); dIm[sOff + half] = 0f;
            for (int f = 1; f < half; f++)
            {
                double ar = 0.0, ai = 0.0;
                for (int m = 0; m < dim; m++)
                {
                    double ang = twoPiOverD * ((f * m) % dim);
                    ar += dh[m] * Math.Cos(ang);
                    ai -= dh[m] * Math.Sin(ang);
                }
                dRe[sOff + f] = (float)(2.0 * invD * ar);
                dIm[sOff + f] = (float)(2.0 * invD * ai);
            }
        }
    }

    private static void SpectralCirculantBackwardDouble(
        ReadOnlySpan<double> dy, ReadOnlySpan<double> diag, ReadOnlySpan<double> specRe, ReadOnlySpan<double> specIm,
        Span<double> dDiag, Span<double> dRe, Span<double> dIm, int batch, int dim)
    {
        int half = dim / 2, nf = half + 1;
        double twoPiOverD = 2.0 * Math.PI / dim;
        double invD = 1.0 / dim;
        var h = new double[dim];
        var dh = new double[dim];
        for (int b = 0; b < batch; b++)
        {
            int sOff = b * nf, mBase = b * dim * dim, vBase = b * dim;
            for (int m = 0; m < dim; m++)
            {
                double sum = specRe[sOff] + ((m & 1) == 0 ? 1.0 : -1.0) * specRe[sOff + half];
                for (int f = 1; f < half; f++)
                {
                    double ang = twoPiOverD * ((f * m) % dim);
                    sum += 2.0 * (specRe[sOff + f] * Math.Cos(ang) - specIm[sOff + f] * Math.Sin(ang));
                }
                h[m] = sum * invD;
            }
            for (int idx = 0; idx < dim; idx++)
            {
                double accD = 0.0, accH = 0.0;
                for (int k = 0; k < dim; k++)
                {
                    int shift = k - idx; if (shift < 0) shift += dim;
                    accD += dy[mBase + idx * dim + k] * h[shift];
                    int col = k + idx; if (col >= dim) col -= dim;
                    accH += dy[mBase + k * dim + col] * diag[vBase + k];
                }
                dDiag[vBase + idx] = accD;
                dh[idx] = accH;
            }
            double s0 = 0.0, sn = 0.0;
            for (int m = 0; m < dim; m++) { s0 += dh[m]; sn += ((m & 1) == 0 ? 1.0 : -1.0) * dh[m]; }
            dRe[sOff] = s0 * invD; dIm[sOff] = 0.0;
            dRe[sOff + half] = sn * invD; dIm[sOff + half] = 0.0;
            for (int f = 1; f < half; f++)
            {
                double ar = 0.0, ai = 0.0;
                for (int m = 0; m < dim; m++)
                {
                    double ang = twoPiOverD * ((f * m) % dim);
                    ar += dh[m] * Math.Cos(ang);
                    ai -= dh[m] * Math.Sin(ang);
                }
                dRe[sOff + f] = 2.0 * invD * ar;
                dIm[sOff + f] = 2.0 * invD * ai;
            }
        }
    }

    public virtual Tensor<T> SigmoidTopKExpertMix<T>(
        Tensor<T> logits, Tensor<T> expertTable, Tensor<T> routingBias, int topK, double epsilon = 1e-12, double biasUpdateRate = 0.0)
    {
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
        if (effRate != 0.0)
        {
            if (ReferenceEquals(logits.DataVector, routingBias.DataVector))
                throw new ArgumentException("routingBias cannot share storage with logits when biasUpdateRate is non-zero.", nameof(routingBias));
            if (!routingBias.IsContiguous)
                throw new ArgumentException("routingBias must be contiguous when biasUpdateRate is non-zero.", nameof(routingBias));
        }
        if (GraphMode.IsActive || (typeof(T) != typeof(float) && typeof(T) != typeof(double)))
        {
            return SigmoidTopKExpertMixComposed(this, logits, expertTable, routingBias, topK, epsilon, effRate);
        }

        var output = new Tensor<T>(new[] { rows, channels });
        var savedIndices = new int[Math.Max(1, rows * topK)];
        int strideGates = topK + 1;
        object savedGatesObj;
        if (typeof(T) == typeof(float))
        {
            var savedGates = new float[Math.Max(1, rows * strideGates)];
            savedGatesObj = savedGates;
            if (rows > 0)
            {
                var sLog = logits.IsContiguous ? logits : logits.Contiguous();
                var sTbl = expertTable.IsContiguous ? expertTable : expertTable.Contiguous();
                var sBias = routingBias.IsContiguous ? routingBias : routingBias.Contiguous();
                SigmoidTopKMixForwardFloat(
                    ((Tensor<float>)(object)sLog).AsSpan(),
                    ((Tensor<float>)(object)sTbl).AsSpan(),
                    ((Tensor<float>)(object)sBias).AsSpan(),
                    ((Tensor<float>)(object)output).AsWritableSpan(),
                    savedIndices,
                    savedGates,
                    effRate != 0.0 ? ((Tensor<float>)(object)routingBias).AsWritableSpan() : Span<float>.Empty,
                    rows, numExperts, topK, channels, (float)epsilon, (float)effRate);
            }
        }
        else
        {
            var savedGates = new double[Math.Max(1, rows * strideGates)];
            savedGatesObj = savedGates;
            if (rows > 0)
            {
                var sLog = logits.IsContiguous ? logits : logits.Contiguous();
                var sTbl = expertTable.IsContiguous ? expertTable : expertTable.Contiguous();
                var sBias = routingBias.IsContiguous ? routingBias : routingBias.Contiguous();
                SigmoidTopKMixForwardDouble(
                    ((Tensor<double>)(object)sLog).AsSpan(),
                    ((Tensor<double>)(object)sTbl).AsSpan(),
                    ((Tensor<double>)(object)sBias).AsSpan(),
                    ((Tensor<double>)(object)output).AsWritableSpan(),
                    savedIndices,
                    savedGates,
                    effRate != 0.0 ? ((Tensor<double>)(object)routingBias).AsWritableSpan() : Span<double>.Empty,
                    rows, numExperts, topK, channels, epsilon, effRate);
            }
        }

        DifferentiableOps.RecordIfActive<T>(
            "SigmoidTopKExpertMix",
            output,
            new[] { logits, expertTable },
            SigmoidTopKExpertMixBackward<T>,
            savedState: new object[] { rows, numExperts, topK, channels, epsilon, savedIndices, savedGatesObj });
        return output;
    }

    internal static Tensor<T> SigmoidTopKExpertMixComposed<T>(
        IEngine engine, Tensor<T> logits, Tensor<T> expertTable, Tensor<T> routingBias, int topK, double epsilon, double effRate)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int rows = logits.Shape[0], g = logits.Shape[1];
        var s = engine.Sigmoid(logits);
        var bias2d = (routingBias.Rank == 2 && routingBias.Shape[0] == 1 && routingBias.Shape[1] == g)
            ? routingBias
            : engine.Reshape(routingBias, new[] { 1, g });
        var biased = engine.StopGradient(engine.TensorAdd(s, bias2d));
        engine.TensorTopK(biased, topK, 1, out Tensor<int> idx);
        var sel = engine.TensorGatherRowValues(s, idx);
        var denom = engine.TensorClampMin(engine.ReduceSum(sel, new[] { 1 }, true), ops.FromDouble(epsilon));
        var gates = engine.TensorDivide(sel, denom);
        if (effRate != 0.0 && rows > 0)
        {
            using var noGrad = new NoGradScope<T>();
            double mean = (double)(rows * topK) / g;
            var counts = engine.Reshape(engine.TensorCountIndices<T>(idx, g), routingBias._shape);
            var diff = engine.ScalarMinusTensor(ops.FromDouble(mean), counts);
            var delta = engine.TensorMultiplyScalar(engine.TensorSign(diff), ops.FromDouble(effRate));
            engine.TensorCopy(engine.TensorAdd(routingBias, delta), routingBias);
        }
        return engine.TopKWeightedRowMix(expertTable, idx, gates);
    }

    internal static void SigmoidTopKExpertMixBackward<T>(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output, object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        int rows = (int)savedState[0], numExperts = (int)savedState[1], topK = (int)savedState[2], channels = (int)savedState[3];
        double eps = (double)savedState[4];
        var savedIndices = savedState[5] as int[];
        var logits = inputs[0];
        var table = inputs[1];
        if (savedIndices is null)
            throw new InvalidOperationException("SigmoidTopKExpertMix backward is missing the selection its forward saved.");

        if (DifferentiableOps._isBackwardCreateGraph || (typeof(T) != typeof(float) && typeof(T) != typeof(double)))
        {
            if (rows == 0)
            {
                DifferentiableOps.AccumulateGrad(grads, logits, new Tensor<T>((int[])logits._shape.Clone()), engine);
                DifferentiableOps.AccumulateGrad(grads, table, new Tensor<T>((int[])table._shape.Clone()), engine);
                return;
            }
            var ops = MathHelper.GetNumericOperations<T>();
            var idxTensor = TensorScope.Detach(new Tensor<int>(new[] { rows, topK }, new Vector<int>((int[])savedIndices.Clone())));
            // The tape this create-graph backward records on. The nested tape below is disposed on return, so its
            // entries move to this one and the gradients accumulate on it; otherwise a higher order sees no graph.
            var outerTape = GradientTape<T>.Current;
            using var tape = new GradientTape<T>(new GradientTapeOptions { Persistent = false });
            var s = engine.Sigmoid(logits);
            var sel = engine.TensorGatherRowValues(s, idxTensor);
            var denom = engine.TensorClampMin(engine.ReduceSum(sel, new[] { 1 }, keepDims: true), ops.FromDouble(eps));
            var gates = engine.TensorDivide(sel, denom);
            var y = engine.TopKWeightedRowMix(table, idxTensor, gates);
            var loss = engine.ReduceSum(engine.TensorMultiply(y, gradOutput), null);
            var sub = tape.ComputeGradients(loss, new[] { logits, table }, createGraph: DifferentiableOps._isBackwardCreateGraph);
            if (DifferentiableOps._isBackwardCreateGraph && outerTape is not null) tape.TransferEntriesTo(outerTape);
            using var recordOnOuter = GradientTape<T>.RecordOn(outerTape);
            if (sub.TryGetValue(logits, out var gl)) DifferentiableOps.AccumulateGrad(grads, logits, gl, engine);
            if (sub.TryGetValue(table, out var gt)) DifferentiableOps.AccumulateGrad(grads, table, gt, engine);
            return;
        }

        var gLog = new Tensor<T>((int[])logits._shape.Clone());
        var gTbl = new Tensor<T>((int[])table._shape.Clone());
        if (rows > 0)
        {
            var sDy = gradOutput.IsContiguous ? gradOutput : gradOutput.Contiguous();
            var sLog = logits.IsContiguous ? logits : logits.Contiguous();
            var sTbl = table.IsContiguous ? table : table.Contiguous();
            if (typeof(T) == typeof(float))
            {
                if (savedState[6] is not float[] savedGatesF) throw new InvalidOperationException("SigmoidTopKExpertMix backward is missing the selection its forward saved.");
                SigmoidTopKMixBackwardFloat(
                    ((Tensor<float>)(object)sDy).AsSpan(),
                    ((Tensor<float>)(object)sLog).AsSpan(),
                    ((Tensor<float>)(object)sTbl).AsSpan(),
                    savedIndices,
                    savedGatesF,
                    ((Tensor<float>)(object)gLog).AsWritableSpan(),
                    ((Tensor<float>)(object)gTbl).AsWritableSpan(),
                    rows, numExperts, topK, channels, (float)eps);
            }
            else
            {
                if (savedState[6] is not double[] savedGatesD) throw new InvalidOperationException("SigmoidTopKExpertMix backward is missing the selection its forward saved.");
                SigmoidTopKMixBackwardDouble(
                    ((Tensor<double>)(object)sDy).AsSpan(),
                    ((Tensor<double>)(object)sLog).AsSpan(),
                    ((Tensor<double>)(object)sTbl).AsSpan(),
                    savedIndices,
                    savedGatesD,
                    ((Tensor<double>)(object)gLog).AsWritableSpan(),
                    ((Tensor<double>)(object)gTbl).AsWritableSpan(),
                    rows, numExperts, topK, channels, eps);
            }
        }
        DifferentiableOps.AccumulateGrad(grads, logits, gLog, engine);
        DifferentiableOps.AccumulateGrad(grads, table, gTbl, engine);
    }

    private static void SigmoidTopKMixForwardFloat(
        ReadOnlySpan<float> logits, ReadOnlySpan<float> table, ReadOnlySpan<float> bias,
        Span<float> y, int[] savedIndices, float[] savedGates, Span<float> mutableBias,
        int rows, int numExperts, int topK, int channels, float eps, float effRate)
    {
        int strideGates = topK + 1;
        var counts = effRate != 0f ? new int[numExperts] : null;
        for (int m = 0; m < rows; m++)
        {
            int lOff = m * numExperts, idxOff = m * topK, gOff = m * strideGates;
            float sumSel = 0f;
            for (int r = 0; r < topK; r++)
            {
                int bestIdx = -1;
                float bestScore = 0f, bestSig = 0f;
                for (int j = 0; j < numExperts; j++)
                {
                    bool used = false;
                    for (int p = 0; p < r; p++) if (savedIndices[idxOff + p] == j) { used = true; break; }
                    if (used) continue;
                    float sj = 1f / (1f + MathF.Exp(-logits[lOff + j]));
                    float bj = sj + bias[j];
                    if (bestIdx < 0 || bj > bestScore) { bestIdx = j; bestScore = bj; bestSig = sj; }
                }
                savedIndices[idxOff + r] = bestIdx;
                savedGates[gOff + r] = bestSig;
                sumSel += bestSig;
                if (counts is not null) counts[bestIdx]++;
            }
            savedGates[gOff + topK] = sumSel;
            float denom = sumSel >= eps ? sumSel : eps;
            for (int r = 0; r < topK; r++) savedGates[gOff + r] /= denom;
            int yOff = m * channels;
            for (int c = 0; c < channels; c++)
            {
                float acc = 0f;
                for (int r = 0; r < topK; r++)
                    acc += savedGates[gOff + r] * table[savedIndices[idxOff + r] * channels + c];
                y[yOff + c] = acc;
            }
        }
        if (counts is not null && !mutableBias.IsEmpty)
        {
            long expectedTimesG = (long)rows * topK;
            for (int e = 0; e < numExperts; e++)
            {
                long actualTimesG = (long)counts[e] * numExperts;
                mutableBias[e] += effRate * (expectedTimesG > actualTimesG ? 1f : expectedTimesG < actualTimesG ? -1f : 0f);
            }
        }
    }

    private static void SigmoidTopKMixForwardDouble(
        ReadOnlySpan<double> logits, ReadOnlySpan<double> table, ReadOnlySpan<double> bias,
        Span<double> y, int[] savedIndices, double[] savedGates, Span<double> mutableBias,
        int rows, int numExperts, int topK, int channels, double eps, double effRate)
    {
        int strideGates = topK + 1;
        var counts = effRate != 0.0 ? new int[numExperts] : null;
        for (int m = 0; m < rows; m++)
        {
            int lOff = m * numExperts, idxOff = m * topK, gOff = m * strideGates;
            double sumSel = 0.0;
            for (int r = 0; r < topK; r++)
            {
                int bestIdx = -1;
                double bestScore = 0.0, bestSig = 0.0;
                for (int j = 0; j < numExperts; j++)
                {
                    bool used = false;
                    for (int p = 0; p < r; p++) if (savedIndices[idxOff + p] == j) { used = true; break; }
                    if (used) continue;
                    double sj = 1.0 / (1.0 + Math.Exp(-logits[lOff + j]));
                    double bj = sj + bias[j];
                    if (bestIdx < 0 || bj > bestScore) { bestIdx = j; bestScore = bj; bestSig = sj; }
                }
                savedIndices[idxOff + r] = bestIdx;
                savedGates[gOff + r] = bestSig;
                sumSel += bestSig;
                if (counts is not null) counts[bestIdx]++;
            }
            savedGates[gOff + topK] = sumSel;
            double denom = sumSel >= eps ? sumSel : eps;
            for (int r = 0; r < topK; r++) savedGates[gOff + r] /= denom;
            int yOff = m * channels;
            for (int c = 0; c < channels; c++)
            {
                double acc = 0.0;
                for (int r = 0; r < topK; r++)
                    acc += savedGates[gOff + r] * table[savedIndices[idxOff + r] * channels + c];
                y[yOff + c] = acc;
            }
        }
        if (counts is not null && !mutableBias.IsEmpty)
        {
            long expectedTimesG = (long)rows * topK;
            for (int e = 0; e < numExperts; e++)
            {
                long actualTimesG = (long)counts[e] * numExperts;
                mutableBias[e] += effRate * (expectedTimesG > actualTimesG ? 1.0 : expectedTimesG < actualTimesG ? -1.0 : 0.0);
            }
        }
    }

    private static void SigmoidTopKMixBackwardFloat(
        ReadOnlySpan<float> dy, ReadOnlySpan<float> logits, ReadOnlySpan<float> table, int[] savedIndices, float[] savedGates,
        Span<float> dLogits, Span<float> dTable, int rows, int numExperts, int topK, int channels, float eps)
    {
        dLogits.Clear();
        dTable.Clear();
        int strideGates = topK + 1;
        for (int m = 0; m < rows; m++)
        {
            int lOff = m * numExperts, idxOff = m * topK, gOff = m * strideGates, yOff = m * channels;
            float sumSel = savedGates[gOff + topK];
            float dotDg = 0f;
            for (int r = 0; r < topK; r++)
            {
                int eIdx = savedIndices[idxOff + r];
                float gateR = savedGates[gOff + r];
                float dGate = 0f;
                for (int c = 0; c < channels; c++)
                {
                    float dyc = dy[yOff + c];
                    dGate += dyc * table[eIdx * channels + c];
                    dTable[eIdx * channels + c] += gateR * dyc;
                }
                dotDg += dGate * gateR;
            }
            for (int r = 0; r < topK; r++)
            {
                int eIdx = savedIndices[idxOff + r];
                float dGate = 0f;
                for (int c = 0; c < channels; c++) dGate += dy[yOff + c] * table[eIdx * channels + c];
                float dSel = sumSel >= eps ? (dGate - dotDg) / sumSel : dGate / eps;
                float sj = 1f / (1f + MathF.Exp(-logits[lOff + eIdx]));
                dLogits[lOff + eIdx] = dSel * sj * (1f - sj);
            }
        }
    }

    private static void SigmoidTopKMixBackwardDouble(
        ReadOnlySpan<double> dy, ReadOnlySpan<double> logits, ReadOnlySpan<double> table, int[] savedIndices, double[] savedGates,
        Span<double> dLogits, Span<double> dTable, int rows, int numExperts, int topK, int channels, double eps)
    {
        dLogits.Clear();
        dTable.Clear();
        int strideGates = topK + 1;
        for (int m = 0; m < rows; m++)
        {
            int lOff = m * numExperts, idxOff = m * topK, gOff = m * strideGates, yOff = m * channels;
            double sumSel = savedGates[gOff + topK];
            double dotDg = 0.0;
            for (int r = 0; r < topK; r++)
            {
                int eIdx = savedIndices[idxOff + r];
                double gateR = savedGates[gOff + r];
                double dGate = 0.0;
                for (int c = 0; c < channels; c++)
                {
                    double dyc = dy[yOff + c];
                    dGate += dyc * table[eIdx * channels + c];
                    dTable[eIdx * channels + c] += gateR * dyc;
                }
                dotDg += dGate * gateR;
            }
            for (int r = 0; r < topK; r++)
            {
                int eIdx = savedIndices[idxOff + r];
                double dGate = 0.0;
                for (int c = 0; c < channels; c++) dGate += dy[yOff + c] * table[eIdx * channels + c];
                double dSel = sumSel >= eps ? (dGate - dotDg) / sumSel : dGate / eps;
                double sj = 1.0 / (1.0 + Math.Exp(-logits[lOff + eIdx]));
                dLogits[lOff + eIdx] = dSel * sj * (1.0 - sj);
            }
        }
    }

    public virtual Tensor<T> DualModulatedSwiGlu<T>(Tensor<T> gateUpProj, Tensor<T> gateUpGainDelta, Tensor<T> upBase)
    {
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

        if (GraphMode.IsActive || (typeof(T) != typeof(float) && typeof(T) != typeof(double)))
        {
            return DualModulatedSwiGluComposed(this, gateUpProj, gateUpGainDelta, upBase, h);
        }

        var output = new Tensor<T>(new[] { rows, h });
        if (rows > 0)
        {
            var sGu = gateUpProj.IsContiguous ? gateUpProj : gateUpProj.Contiguous();
            var sGg = gateUpGainDelta.IsContiguous ? gateUpGainDelta : gateUpGainDelta.Contiguous();
            var sBase = upBase.IsContiguous ? upBase : upBase.Contiguous();
            if (typeof(T) == typeof(float))
            {
                var pGu = ((Tensor<float>)(object)sGu).AsSpan();
                var pGg = ((Tensor<float>)(object)sGg).AsSpan();
                var pB = ((Tensor<float>)(object)sBase).AsSpan();
                var pOut = ((Tensor<float>)(object)output).AsWritableSpan();
                for (int m = 0; m < rows; m++)
                {
                    int off2 = m * cols2, off1 = m * h;
                    for (int j = 0; j < h; j++)
                    {
                        float g = pGu[off2 + j] * (1f + pGg[off2 + j]);
                        float sig = 1f / (1f + MathF.Exp(-g));
                        pOut[off1 + j] = (g * sig * pGu[off2 + h + j]) * (pB[j] + pGg[off2 + h + j]);
                    }
                }
            }
            else
            {
                var pGu = ((Tensor<double>)(object)sGu).AsSpan();
                var pGg = ((Tensor<double>)(object)sGg).AsSpan();
                var pB = ((Tensor<double>)(object)sBase).AsSpan();
                var pOut = ((Tensor<double>)(object)output).AsWritableSpan();
                for (int m = 0; m < rows; m++)
                {
                    int off2 = m * cols2, off1 = m * h;
                    for (int j = 0; j < h; j++)
                    {
                        double g = pGu[off2 + j] * (1.0 + pGg[off2 + j]);
                        double sig = 1.0 / (1.0 + Math.Exp(-g));
                        pOut[off1 + j] = (g * sig * pGu[off2 + h + j]) * (pB[j] + pGg[off2 + h + j]);
                    }
                }
            }
        }

        DifferentiableOps.RecordIfActive<T>(
            "DualModulatedSwiGlu",
            output,
            new[] { gateUpProj, gateUpGainDelta, upBase },
            DualModulatedSwiGluBackward<T>,
            savedState: new object[] { rows, h });
        return output;
    }

    internal static Tensor<T> DualModulatedSwiGluComposed<T>(
        IEngine engine, Tensor<T> gateUpProj, Tensor<T> gateUpGainDelta, Tensor<T> upBase, int h)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var gRaw = engine.TensorNarrow(gateUpProj, 1, 0, h);
        var uRaw = engine.TensorNarrow(gateUpProj, 1, h, h);
        var gGain = engine.TensorAddScalar(engine.TensorNarrow(gateUpGainDelta, 1, 0, h), ops.One);
        var b2 = (upBase.Rank == 2 && upBase.Shape[0] == 1 && upBase.Shape[1] == h)
            ? upBase
            : engine.Reshape(upBase, new[] { 1, h });
        var uGain = engine.TensorAdd(b2, engine.TensorNarrow(gateUpGainDelta, 1, h, h));
        var act = engine.TensorMultiply(engine.TensorSiLU(engine.TensorMultiply(gRaw, gGain)), uRaw);
        return engine.TensorMultiply(uGain, act);
    }

    internal static void DualModulatedSwiGluBackward<T>(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output, object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        int rows = (int)savedState[0], h = (int)savedState[1];
        var guProj = inputs[0]; var guGain = inputs[1]; var upBase = inputs[2];
        if (DifferentiableOps._isBackwardCreateGraph || (typeof(T) != typeof(float) && typeof(T) != typeof(double)))
        {
            // The tape this create-graph backward records on. The nested tape below is disposed on return, so its
            // entries move to this one and the gradients accumulate on it; otherwise a higher order sees no graph.
            var outerTape = GradientTape<T>.Current;
            using var tape = new GradientTape<T>(new GradientTapeOptions { Persistent = false });
            var y = DualModulatedSwiGluComposed(engine, guProj, guGain, upBase, h);
            var sub = tape.ComputeGradients(engine.ReduceSum(engine.TensorMultiply(y, gradOutput), null), new[] { guProj, guGain, upBase }, createGraph: DifferentiableOps._isBackwardCreateGraph);
            if (DifferentiableOps._isBackwardCreateGraph && outerTape is not null) tape.TransferEntriesTo(outerTape);
            using var recordOnOuter = GradientTape<T>.RecordOn(outerTape);
            if (sub.TryGetValue(guProj, out var g0)) DifferentiableOps.AccumulateGrad(grads, guProj, g0, engine);
            if (sub.TryGetValue(guGain, out var g1)) DifferentiableOps.AccumulateGrad(grads, guGain, g1, engine);
            if (sub.TryGetValue(upBase, out var g2)) DifferentiableOps.AccumulateGrad(grads, upBase, g2, engine);
            return;
        }

        var dGu = new Tensor<T>((int[])guProj._shape.Clone());
        var dGg = new Tensor<T>((int[])guGain._shape.Clone());
        var dB = new Tensor<T>((int[])upBase._shape.Clone());
        if (rows > 0)
        {
            var sDy = gradOutput.IsContiguous ? gradOutput : gradOutput.Contiguous();
            var sGu = guProj.IsContiguous ? guProj : guProj.Contiguous();
            var sGg = guGain.IsContiguous ? guGain : guGain.Contiguous();
            var sB = upBase.IsContiguous ? upBase : upBase.Contiguous();
            int cols2 = 2 * h;
            if (typeof(T) == typeof(float))
            {
                var pDy = ((Tensor<float>)(object)sDy).AsSpan();
                var pGu = ((Tensor<float>)(object)sGu).AsSpan();
                var pGg = ((Tensor<float>)(object)sGg).AsSpan();
                var pB = ((Tensor<float>)(object)sB).AsSpan();
                var pdGu = ((Tensor<float>)(object)dGu).AsWritableSpan();
                var pdGg = ((Tensor<float>)(object)dGg).AsWritableSpan();
                var pdB = ((Tensor<float>)(object)dB).AsWritableSpan();
                for (int m = 0; m < rows; m++)
                {
                    int off2 = m * cols2, off1 = m * h;
                    for (int j = 0; j < h; j++)
                    {
                        float dy = pDy[off1 + j];
                        float gRaw = pGu[off2 + j], uRaw = pGu[off2 + h + j];
                        float gGain = 1f + pGg[off2 + j], uGain = pB[j] + pGg[off2 + h + j];
                        float g = gRaw * gGain;
                        float sig = 1f / (1f + MathF.Exp(-g));
                        float silu = g * sig;
                        float hv = silu * uRaw;
                        pdGg[off2 + h + j] = dy * hv;
                        pdB[j] += dy * hv;
                        float dh = dy * uGain;
                        pdGu[off2 + h + j] = dh * silu;
                        float dg = (dh * uRaw) * sig * (1f + g * (1f - sig));
                        pdGu[off2 + j] = dg * gGain;
                        pdGg[off2 + j] = dg * gRaw;
                    }
                }
            }
            else
            {
                var pDy = ((Tensor<double>)(object)sDy).AsSpan();
                var pGu = ((Tensor<double>)(object)sGu).AsSpan();
                var pGg = ((Tensor<double>)(object)sGg).AsSpan();
                var pB = ((Tensor<double>)(object)sB).AsSpan();
                var pdGu = ((Tensor<double>)(object)dGu).AsWritableSpan();
                var pdGg = ((Tensor<double>)(object)dGg).AsWritableSpan();
                var pdB = ((Tensor<double>)(object)dB).AsWritableSpan();
                for (int m = 0; m < rows; m++)
                {
                    int off2 = m * cols2, off1 = m * h;
                    for (int j = 0; j < h; j++)
                    {
                        double dy = pDy[off1 + j];
                        double gRaw = pGu[off2 + j], uRaw = pGu[off2 + h + j];
                        double gGain = 1.0 + pGg[off2 + j], uGain = pB[j] + pGg[off2 + h + j];
                        double g = gRaw * gGain;
                        double sig = 1.0 / (1.0 + Math.Exp(-g));
                        double silu = g * sig;
                        double hv = silu * uRaw;
                        pdGg[off2 + h + j] = dy * hv;
                        pdB[j] += dy * hv;
                        double dh = dy * uGain;
                        pdGu[off2 + h + j] = dh * silu;
                        double dg = (dh * uRaw) * sig * (1.0 + g * (1.0 - sig));
                        pdGu[off2 + j] = dg * gGain;
                        pdGg[off2 + j] = dg * gRaw;
                    }
                }
            }
        }
        DifferentiableOps.AccumulateGrad(grads, guProj, dGu, engine);
        DifferentiableOps.AccumulateGrad(grads, guGain, dGg, engine);
        DifferentiableOps.AccumulateGrad(grads, upBase, dB, engine);
    }

    public virtual Tensor<T> BornRatioNormalize<T>(Tensor<T> scanOutput, Tensor<T>? sinkLogAlpha, int headDim, double epsilon = 1e-4)
    {
        if (scanOutput is null) throw new ArgumentNullException(nameof(scanOutput));
        if (scanOutput.Rank != 4)
            throw new ArgumentException($"Expected scanOutput rank 4 [B, S, H, scanDim], got {scanOutput.Rank}.", nameof(scanOutput));
        int b = scanOutput.Shape[0], s = scanOutput.Shape[1], nh = scanOutput.Shape[2], scanDim = scanOutput.Shape[3];
        if (headDim <= 0 || headDim >= scanDim)
            throw new ArgumentOutOfRangeException(nameof(headDim));
        if (sinkLogAlpha is not null && sinkLogAlpha.Length != nh)
            throw new ArgumentException($"Expected sinkLogAlpha length {nh}, got {sinkLogAlpha.Length}.", nameof(sinkLogAlpha));

        if (GraphMode.IsActive || (typeof(T) != typeof(float) && typeof(T) != typeof(double)))
        {
            return BornRatioNormalizeComposed(this, scanOutput, sinkLogAlpha, b, s, nh, headDim, epsilon);
        }

        int totalRows = checked(b * s * nh);
        var output = new Tensor<T>(new[] { b, s, nh * headDim });
        if (totalRows > 0)
        {
            var sScan = scanOutput.IsContiguous ? scanOutput : scanOutput.Contiguous();
            var sSink = sinkLogAlpha is null ? null : (sinkLogAlpha.IsContiguous ? sinkLogAlpha : sinkLogAlpha.Contiguous());
            if (typeof(T) == typeof(float))
            {
                var pS = ((Tensor<float>)(object)sScan).AsSpan();
                var pK = sSink is null ? ReadOnlySpan<float>.Empty : ((Tensor<float>)(object)sSink).AsSpan();
                var pO = ((Tensor<float>)(object)output).AsWritableSpan();
                float epsF = (float)epsilon;
                for (int row = 0; row < totalRows; row++)
                {
                    int h = row % nh;
                    float den = pS[row * scanDim + headDim] + epsF + (sSink is null ? 0f : MathF.Exp(pK[h]));
                    for (int j = 0; j < headDim; j++) pO[row * headDim + j] = pS[row * scanDim + j] / den;
                }
            }
            else
            {
                var pS = ((Tensor<double>)(object)sScan).AsSpan();
                var pK = sSink is null ? ReadOnlySpan<double>.Empty : ((Tensor<double>)(object)sSink).AsSpan();
                var pO = ((Tensor<double>)(object)output).AsWritableSpan();
                for (int row = 0; row < totalRows; row++)
                {
                    int h = row % nh;
                    double den = pS[row * scanDim + headDim] + epsilon + (sSink is null ? 0.0 : Math.Exp(pK[h]));
                    for (int j = 0; j < headDim; j++) pO[row * headDim + j] = pS[row * scanDim + j] / den;
                }
            }
        }

        var inputs = sinkLogAlpha is null ? new[] { scanOutput } : new[] { scanOutput, sinkLogAlpha };
        DifferentiableOps.RecordIfActive<T>(
            "BornRatioNormalize",
            output,
            inputs,
            BornRatioNormalizeBackward<T>,
            savedState: new object[] { b, s, nh, headDim, scanDim, epsilon, sinkLogAlpha is not null });
        return output;
    }

    internal static Tensor<T> BornRatioNormalizeComposed<T>(
        IEngine engine, Tensor<T> scanOutput, Tensor<T>? sinkLogAlpha, int b, int s, int nh, int headDim, double epsilon)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var num = engine.TensorNarrow(scanOutput, 3, 0, headDim);
        var rawDen = engine.TensorNarrow(scanOutput, 3, headDim, 1);
        Tensor<T> den = sinkLogAlpha is not null
            ? engine.TensorAdd(rawDen, engine.Reshape(engine.TensorAddScalar(engine.TensorExp(sinkLogAlpha), ops.FromDouble(epsilon)), new[] { 1, 1, nh, 1 }))
            : engine.TensorAddScalar(rawDen, ops.FromDouble(epsilon));
        return engine.Reshape(engine.TensorDivide(num, den), new[] { b, s, nh * headDim });
    }

    internal static void BornRatioNormalizeBackward<T>(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output, object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        int b = (int)savedState[0], s = (int)savedState[1], nh = (int)savedState[2], headDim = (int)savedState[3], scanDim = (int)savedState[4];
        double eps = (double)savedState[5];
        bool hasSink = (bool)savedState[6];
        var scanOut = inputs[0];
        Tensor<T>? sink = hasSink ? inputs[1] : null;
        if (DifferentiableOps._isBackwardCreateGraph || (typeof(T) != typeof(float) && typeof(T) != typeof(double)))
        {
            // The tape this create-graph backward records on. The nested tape below is disposed on return, so its
            // entries move to this one and the gradients accumulate on it; otherwise a higher order sees no graph.
            var outerTape = GradientTape<T>.Current;
            using var tape = new GradientTape<T>(new GradientTapeOptions { Persistent = false });
            var y = BornRatioNormalizeComposed(engine, scanOut, sink, b, s, nh, headDim, eps);
            var sub = tape.ComputeGradients(engine.ReduceSum(engine.TensorMultiply(y, gradOutput), null), inputs, createGraph: DifferentiableOps._isBackwardCreateGraph);
            if (DifferentiableOps._isBackwardCreateGraph && outerTape is not null) tape.TransferEntriesTo(outerTape);
            using var recordOnOuter = GradientTape<T>.RecordOn(outerTape);
            if (sub.TryGetValue(scanOut, out var gs)) DifferentiableOps.AccumulateGrad(grads, scanOut, gs, engine);
            if (sink is not null && sub.TryGetValue(sink, out var gk)) DifferentiableOps.AccumulateGrad(grads, sink, gk, engine);
            return;
        }

        int totalRows = b * s * nh;
        var dScan = new Tensor<T>((int[])scanOut._shape.Clone());
        Tensor<T>? dSink = sink is not null ? new Tensor<T>((int[])sink._shape.Clone()) : null;
        if (totalRows > 0)
        {
            var sDy = gradOutput.IsContiguous ? gradOutput : gradOutput.Contiguous();
            var sScan = scanOut.IsContiguous ? scanOut : scanOut.Contiguous();
            var sSink = sink is null ? null : (sink.IsContiguous ? sink : sink.Contiguous());
            if (typeof(T) == typeof(float))
            {
                var pDy = ((Tensor<float>)(object)sDy).AsSpan();
                var pS = ((Tensor<float>)(object)sScan).AsSpan();
                var pK = sSink is null ? ReadOnlySpan<float>.Empty : ((Tensor<float>)(object)sSink).AsSpan();
                var pdS = ((Tensor<float>)(object)dScan).AsWritableSpan();
                var pdK = dSink is null ? Span<float>.Empty : ((Tensor<float>)(object)dSink).AsWritableSpan();
                float epsF = (float)eps;
                for (int row = 0; row < totalRows; row++)
                {
                    int h = row % nh;
                    float sinkTerm = sSink is null ? 0f : MathF.Exp(pK[h]);
                    float invDen = 1f / (pS[row * scanDim + headDim] + epsF + sinkTerm);
                    float dot = 0f;
                    for (int j = 0; j < headDim; j++)
                    {
                        float dyj = pDy[row * headDim + j];
                        dot += dyj * pS[row * scanDim + j];
                        pdS[row * scanDim + j] = dyj * invDen;
                    }
                    float dDen = -dot * invDen * invDen;
                    pdS[row * scanDim + headDim] = dDen;
                    if (dSink is not null) pdK[h] += dDen * sinkTerm;
                }
            }
            else
            {
                var pDy = ((Tensor<double>)(object)sDy).AsSpan();
                var pS = ((Tensor<double>)(object)sScan).AsSpan();
                var pK = sSink is null ? ReadOnlySpan<double>.Empty : ((Tensor<double>)(object)sSink).AsSpan();
                var pdS = ((Tensor<double>)(object)dScan).AsWritableSpan();
                var pdK = dSink is null ? Span<double>.Empty : ((Tensor<double>)(object)dSink).AsWritableSpan();
                for (int row = 0; row < totalRows; row++)
                {
                    int h = row % nh;
                    double sinkTerm = sSink is null ? 0.0 : Math.Exp(pK[h]);
                    double invDen = 1.0 / (pS[row * scanDim + headDim] + eps + sinkTerm);
                    double dot = 0.0;
                    for (int j = 0; j < headDim; j++)
                    {
                        double dyj = pDy[row * headDim + j];
                        dot += dyj * pS[row * scanDim + j];
                        pdS[row * scanDim + j] = dyj * invDen;
                    }
                    double dDen = -dot * invDen * invDen;
                    pdS[row * scanDim + headDim] = dDen;
                    if (dSink is not null) pdK[h] += dDen * sinkTerm;
                }
            }
        }
        DifferentiableOps.AccumulateGrad(grads, scanOut, dScan, engine);
        if (sink is not null && dSink is not null) DifferentiableOps.AccumulateGrad(grads, sink, dSink, engine);
    }

    public virtual Tensor<T> BoundedSigmoidGate<T>(Tensor<T> logits, Tensor<T> bias, double floor)
    {
        if (logits is null) throw new ArgumentNullException(nameof(logits));
        if (bias is null) throw new ArgumentNullException(nameof(bias));
        if (logits.Rank < 1) throw new ArgumentException("logits must have rank >= 1.", nameof(logits));
        int cols = logits.Shape[logits.Rank - 1];
        if (cols <= 0 || bias.Length != cols)
            throw new ArgumentException($"Expected bias length {cols}, got {bias.Length}.", nameof(bias));

        if (GraphMode.IsActive || (typeof(T) != typeof(float) && typeof(T) != typeof(double)))
        {
            return BoundedSigmoidGateComposed(this, logits, bias, floor);
        }

        int rows = logits.Length / cols;
        var output = new Tensor<T>((int[])logits._shape.Clone());
        if (rows > 0)
        {
            var sLog = logits.IsContiguous ? logits : logits.Contiguous();
            var sB = bias.IsContiguous ? bias : bias.Contiguous();
            if (typeof(T) == typeof(float))
            {
                var pL = ((Tensor<float>)(object)sLog).AsSpan();
                var pB = ((Tensor<float>)(object)sB).AsSpan();
                var pO = ((Tensor<float>)(object)output).AsWritableSpan();
                float flr = (float)floor, sc = 1f - flr;
                for (int r = 0; r < rows; r++)
                {
                    int off = r * cols;
                    for (int c = 0; c < cols; c++)
                        pO[off + c] = flr + sc / (1f + MathF.Exp(-(pL[off + c] + pB[c])));
                }
            }
            else
            {
                var pL = ((Tensor<double>)(object)sLog).AsSpan();
                var pB = ((Tensor<double>)(object)sB).AsSpan();
                var pO = ((Tensor<double>)(object)output).AsWritableSpan();
                double sc = 1.0 - floor;
                for (int r = 0; r < rows; r++)
                {
                    int off = r * cols;
                    for (int c = 0; c < cols; c++)
                        pO[off + c] = floor + sc / (1.0 + Math.Exp(-(pL[off + c] + pB[c])));
                }
            }
        }

        DifferentiableOps.RecordIfActive<T>(
            "BoundedSigmoidGate",
            output,
            new[] { logits, bias },
            BoundedSigmoidGateBackward<T>,
            savedState: new object[] { rows, cols, floor });
        return output;
    }

    internal static Tensor<T> BoundedSigmoidGateComposed<T>(IEngine engine, Tensor<T> logits, Tensor<T> bias, double floor)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int cols = logits.Shape[logits.Rank - 1];
        var biasVec = bias.Rank == 1 ? bias : engine.Reshape(bias, new[] { cols });
        var sig = engine.Sigmoid(engine.TensorAdd(logits, biasVec));
        return engine.TensorAddScalar(engine.TensorMultiplyScalar(sig, ops.FromDouble(1.0 - floor)), ops.FromDouble(floor));
    }

    internal static void BoundedSigmoidGateBackward<T>(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output, object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        int rows = (int)savedState[0], cols = (int)savedState[1];
        double floor = (double)savedState[2];
        var logits = inputs[0]; var bias = inputs[1];
        if (DifferentiableOps._isBackwardCreateGraph || (typeof(T) != typeof(float) && typeof(T) != typeof(double)))
        {
            // The tape this create-graph backward records on. The nested tape below is disposed on return, so its
            // entries move to this one and the gradients accumulate on it; otherwise a higher order sees no graph.
            var outerTape = GradientTape<T>.Current;
            using var tape = new GradientTape<T>(new GradientTapeOptions { Persistent = false });
            var y = BoundedSigmoidGateComposed(engine, logits, bias, floor);
            var sub = tape.ComputeGradients(engine.ReduceSum(engine.TensorMultiply(y, gradOutput), null), new[] { logits, bias }, createGraph: DifferentiableOps._isBackwardCreateGraph);
            if (DifferentiableOps._isBackwardCreateGraph && outerTape is not null) tape.TransferEntriesTo(outerTape);
            using var recordOnOuter = GradientTape<T>.RecordOn(outerTape);
            if (sub.TryGetValue(logits, out var gl)) DifferentiableOps.AccumulateGrad(grads, logits, gl, engine);
            if (sub.TryGetValue(bias, out var gb)) DifferentiableOps.AccumulateGrad(grads, bias, gb, engine);
            return;
        }

        var dLog = new Tensor<T>((int[])logits._shape.Clone());
        var dBias = new Tensor<T>((int[])bias._shape.Clone());
        if (rows > 0)
        {
            var sDy = gradOutput.IsContiguous ? gradOutput : gradOutput.Contiguous();
            var sL = logits.IsContiguous ? logits : logits.Contiguous();
            var sB = bias.IsContiguous ? bias : bias.Contiguous();
            if (typeof(T) == typeof(float))
            {
                var pDy = ((Tensor<float>)(object)sDy).AsSpan();
                var pL = ((Tensor<float>)(object)sL).AsSpan();
                var pB = ((Tensor<float>)(object)sB).AsSpan();
                var pdL = ((Tensor<float>)(object)dLog).AsWritableSpan();
                var pdB = ((Tensor<float>)(object)dBias).AsWritableSpan();
                float sc = 1f - (float)floor;
                for (int r = 0; r < rows; r++)
                {
                    int off = r * cols;
                    for (int c = 0; c < cols; c++)
                    {
                        float sig = 1f / (1f + MathF.Exp(-(pL[off + c] + pB[c])));
                        float dl = pDy[off + c] * sc * sig * (1f - sig);
                        pdL[off + c] = dl;
                        pdB[c] += dl;
                    }
                }
            }
            else
            {
                var pDy = ((Tensor<double>)(object)sDy).AsSpan();
                var pL = ((Tensor<double>)(object)sL).AsSpan();
                var pB = ((Tensor<double>)(object)sB).AsSpan();
                var pdL = ((Tensor<double>)(object)dLog).AsWritableSpan();
                var pdB = ((Tensor<double>)(object)dBias).AsWritableSpan();
                double sc = 1.0 - floor;
                for (int r = 0; r < rows; r++)
                {
                    int off = r * cols;
                    for (int c = 0; c < cols; c++)
                    {
                        double sig = 1.0 / (1.0 + Math.Exp(-(pL[off + c] + pB[c])));
                        double dl = pDy[off + c] * sc * sig * (1.0 - sig);
                        pdL[off + c] = dl;
                        pdB[c] += dl;
                    }
                }
            }
        }
        DifferentiableOps.AccumulateGrad(grads, logits, dLog, engine);
        DifferentiableOps.AccumulateGrad(grads, bias, dBias, engine);
    }
}
