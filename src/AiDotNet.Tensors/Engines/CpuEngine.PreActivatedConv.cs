// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;
using System.Runtime.CompilerServices;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

public partial class CpuEngine
{
    /// <inheritdoc/>
    public virtual Tensor<T> BatchNormActivationConv2D<T>(
        Tensor<T> input, Tensor<T> preGamma, Tensor<T> preBeta, Tensor<T> preMean, Tensor<T> preVariance, double preEpsilon,
        Tensor<T> kernel, Tensor<T>? bias,
        int strideH, int strideW, int padH, int padW, int dilationH, int dilationW,
        FusedActivationType activation)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (kernel is null) throw new ArgumentNullException(nameof(kernel));
        var pre = PreActivate(input, preGamma, preBeta, preMean, preVariance, preEpsilon);
        return FusedConv2D(pre, kernel, bias, strideH, strideW, padH, padW, dilationH, dilationW, activation);
    }

    /// <inheritdoc/>
    public virtual Tensor<T> BatchNormActivationConv2DBatchNorm<T>(
        Tensor<T> input, Tensor<T> preGamma, Tensor<T> preBeta, Tensor<T> preMean, Tensor<T> preVariance, double preEpsilon,
        Tensor<T> kernel, Tensor<T>? bias,
        Tensor<T> gamma, Tensor<T> beta, Tensor<T> runningMean, Tensor<T> runningVariance, double epsilon,
        int strideH, int strideW, int padH, int padW, int dilationH, int dilationW,
        FusedActivationType activation)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (kernel is null) throw new ArgumentNullException(nameof(kernel));
        // Recording keeps the unfused, recorded ops (FusedConv2DBatchNorm records the convolution and batch norm).
        if (IsRecordingPreActivatedConv<T>())
            return FusedConv2DBatchNorm(PreActivate(input, preGamma, preBeta, preMean, preVariance, preEpsilon), kernel, bias,
                gamma, beta, runningMean, runningVariance, epsilon, strideH, strideW, padH, padW, dilationH, dilationW,
                activation);
        var (foldedKernel, foldedBias) = FoldConvBatchNormForPreActivatedConv(kernel, bias, gamma, beta, runningMean,
            runningVariance, epsilon);
        return BatchNormActivationConv2D(input, preGamma, preBeta, preMean, preVariance, preEpsilon, foldedKernel,
            foldedBias, strideH, strideW, padH, padW, dilationH, dilationW, activation);
    }

    /// <summary>The fold <see cref="BatchNormActivationConv2DBatchNorm{T}"/> uses; the GPU engine folds on the device when the
    /// parameters live there.</summary>
    private protected virtual (Tensor<T> Kernel, Tensor<T> Bias) FoldConvBatchNormForPreActivatedConv<T>(Tensor<T> kernel,
        Tensor<T>? bias, Tensor<T> gamma, Tensor<T> beta, Tensor<T> mean, Tensor<T> variance, double epsilon)
        => ConvBatchNormFold<T>.Get(kernel, bias, gamma, beta, mean, variance, epsilon, ConvBatchNormFold<T>.FoldOnHost);

    private protected static bool IsRecordingPreActivatedConv<T>()
        => GraphMode.IsActive || (GradientTape<T>.Current is not null && !NoGradScope<T>.IsSuppressed);

    // relu(batchnorm(x)) with the frozen statistics; recorded (BatchNormAffine, the differentiable eval-mode form, and a
    // recorded ReLU) under a tape or graph capture.
    private Tensor<T> PreActivate<T>(Tensor<T> input, Tensor<T> gamma, Tensor<T> beta, Tensor<T> mean, Tensor<T> variance,
        double epsilon)
        => IsRecordingPreActivatedConv<T>()
            ? ApplyActivationRecorded(BatchNormAffine(input, gamma, beta, mean, variance, epsilon), FusedActivationType.ReLU)
            : BatchNormInference(input, gamma, beta, mean, variance, epsilon, FusedActivationType.ReLU);

    /// <summary>
    /// An inference batch norm as a per-channel affine, scale = gamma / sqrt(var + eps) and shift = beta - mean * scale,
    /// computed on the host and cached by the gamma tensor until any of the four parameters changes version or identity
    /// (or epsilon changes): the prologue a pre-activated convolution applies to its input.
    /// </summary>
    internal static class BatchNormAffineFold<T>
    {
        private sealed class Entry
        {
            internal Tensor<T>? Beta, Mean, Variance;
            internal int[] Versions = Array.Empty<int>();
            internal double Epsilon;
            internal Tensor<T> Scale = new(new[] { 0 });
            internal Tensor<T> Shift = new(new[] { 0 });
        }

        private static readonly ConditionalWeakTable<Tensor<T>, Entry> s_entries = new();

        [ScalarKernel("Folds a batch norm's per-channel statistics into a scale and shift once per parameter version; cached.")]
        internal static (Tensor<T> Scale, Tensor<T> Shift) Get(Tensor<T> gamma, Tensor<T> beta, Tensor<T> mean,
            Tensor<T> variance, double epsilon)
        {
            var versions = new[]
            {
                gamma.Version, gamma.GpuCacheVersion, beta.Version, beta.GpuCacheVersion,
                mean.Version, mean.GpuCacheVersion, variance.Version, variance.GpuCacheVersion,
            };
            lock (s_entries)
            {
                if (s_entries.TryGetValue(gamma, out var entry) && ReferenceEquals(entry.Beta, beta)
                    && ReferenceEquals(entry.Mean, mean) && ReferenceEquals(entry.Variance, variance)
                    && entry.Epsilon.Equals(epsilon) && entry.Versions.AsSpan().SequenceEqual(versions))
                    return (entry.Scale, entry.Shift);

                var ops = MathHelper.GetNumericOperations<T>();
                var g = gamma.ToArray();
                var b = beta.ToArray();
                var m = mean.ToArray();
                var v = variance.ToArray();
                var scale = new T[g.Length];
                var shift = new T[g.Length];
                for (int c = 0; c < g.Length; c++)
                {
                    double s = ops.ToDouble(g[c]) / Math.Sqrt(ops.ToDouble(v[c]) + epsilon);
                    scale[c] = ops.FromDouble(s);
                    shift[c] = ops.FromDouble(ops.ToDouble(b[c]) - ops.ToDouble(m[c]) * s);
                }
                var created = new Entry
                {
                    Beta = beta, Mean = mean, Variance = variance, Versions = versions, Epsilon = epsilon,
                    Scale = new Tensor<T>(scale, new[] { g.Length }), Shift = new Tensor<T>(shift, new[] { g.Length }),
                };
                s_entries.Remove(gamma);
                s_entries.Add(gamma, created);
                return (created.Scale, created.Shift);
            }
        }
    }
}
