// Copyright (c) AiDotNet. All rights reserved.
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
    public virtual Tensor<T> FusedConv2DBatchNorm<T>(
        Tensor<T> input, Tensor<T> kernel, Tensor<T>? bias,
        Tensor<T> gamma, Tensor<T> beta, Tensor<T> runningMean, Tensor<T> runningVariance, double epsilon,
        int strideH, int strideW, int padH, int padW, int dilationH, int dilationW,
        FusedActivationType activation)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (kernel is null) throw new ArgumentNullException(nameof(kernel));
        if (gamma is null) throw new ArgumentNullException(nameof(gamma));
        if (beta is null) throw new ArgumentNullException(nameof(beta));
        if (runningMean is null) throw new ArgumentNullException(nameof(runningMean));
        if (runningVariance is null) throw new ArgumentNullException(nameof(runningVariance));
        if (kernel.Rank != 4) throw new ArgumentException($"FusedConv2DBatchNorm requires a rank-4 kernel, got rank {kernel.Rank}.", nameof(kernel));
        int outChannels = kernel.Shape[0];
        foreach (var (name, channel) in new[] { (nameof(gamma), gamma), (nameof(beta), beta),
                     (nameof(runningMean), runningMean), (nameof(runningVariance), runningVariance) })
        {
            if (channel.Length != outChannels)
                throw new ArgumentException($"{name} has {channel.Length} elements; the kernel has {outChannels} output channels.", name);
        }
        if (bias is not null && bias.Length != outChannels)
            throw new ArgumentException($"bias has {bias.Length} elements; the kernel has {outChannels} output channels.", nameof(bias));

        // Recording (a tape or graph capture) keeps the unfused ops, which record their own nodes, so gradients and
        // captured plans are unchanged. Folding is for inference, where the statistics are frozen.
        if (GraphMode.IsActive || (GradientTape<T>.Current is not null && !NoGradScope<T>.IsSuppressed))
        {
            var conv = FusedConv2D(input, kernel, bias, strideH, strideW, padH, padW, dilationH, dilationW, FusedActivationType.None);
            // BatchNormAffine is the differentiable form of inference batch norm (gradients for x, gamma and beta, as
            // PyTorch gives in eval mode); BatchNormInference records nothing.
            return ApplyFusedActivation(BatchNormAffine(conv, gamma, beta, runningMean, runningVariance, epsilon), activation);
        }

        var (foldedKernel, foldedBias) = ConvBatchNormFold<T>.Get(kernel, bias, gamma, beta, runningMean, runningVariance, epsilon,
            ConvBatchNormFold<T>.FoldOnHost);
        return FusedConv2D(input, foldedKernel, foldedBias, strideH, strideW, padH, padW, dilationH, dilationW, activation);
    }

    /// <summary>
    /// Batch norm folded into the convolution before it, per output channel o with s = gamma[o] / sqrt(var[o] + eps):
    /// kernel'[o] = kernel[o] * s and bias'[o] = beta[o] + (bias[o] - mean[o]) * s. Cached by the kernel tensor and
    /// rebuilt when any of the six parameters changes version or identity, or epsilon changes, so inference folds once.
    /// The folded tensors keep one identity across calls, so device weight caches (and transforms derived from the
    /// weights, such as Winograd filters) keyed by it stay valid.
    /// </summary>
    internal static class ConvBatchNormFold<T>
    {
        private sealed class Entry
        {
            internal Tensor<T>? Bias, Gamma, Beta, Mean, Variance;
            internal int[] Versions = Array.Empty<int>();
            internal double Epsilon;
            internal Tensor<T> Kernel = new(new[] { 0 });
            internal Tensor<T> FoldedBias = new(new[] { 0 });
        }

        private static readonly ConditionalWeakTable<Tensor<T>, Entry> s_entries = new();

        private static int[] VersionsOf(Tensor<T> kernel, Tensor<T>? bias, Tensor<T> gamma, Tensor<T> beta,
            Tensor<T> mean, Tensor<T> variance)
            => new[]
            {
                kernel.Version, kernel.GpuCacheVersion, bias?.Version ?? -1, bias?.GpuCacheVersion ?? -1,
                gamma.Version, gamma.GpuCacheVersion, beta.Version, beta.GpuCacheVersion,
                mean.Version, mean.GpuCacheVersion, variance.Version, variance.GpuCacheVersion,
            };

        internal static (Tensor<T> Kernel, Tensor<T> Bias) Get(Tensor<T> kernel, Tensor<T>? bias, Tensor<T> gamma,
            Tensor<T> beta, Tensor<T> mean, Tensor<T> variance, double epsilon,
            Func<Tensor<T>, Tensor<T>?, Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>, double, (Tensor<T>, Tensor<T>)> fold,
            Action<Tensor<T>>? releaseReplaced = null)
        {
            var versions = VersionsOf(kernel, bias, gamma, beta, mean, variance);
            lock (s_entries)
            {
                if (s_entries.TryGetValue(kernel, out var entry)
                    && ReferenceEquals(entry.Bias, bias) && ReferenceEquals(entry.Gamma, gamma)
                    && ReferenceEquals(entry.Beta, beta) && ReferenceEquals(entry.Mean, mean)
                    && ReferenceEquals(entry.Variance, variance) && entry.Epsilon.Equals(epsilon)
                    && entry.Versions.AsSpan().SequenceEqual(versions))
                    return (entry.Kernel, entry.FoldedBias);

                // The folded tensors are a cache, not results of the caller's call: a TensorScope must not free them.
                (Tensor<T> foldedKernel, Tensor<T> foldedBias) folded;
                using (TensorScope.Suspend())
                    folded = fold(kernel, bias, gamma, beta, mean, variance, epsilon);
                var (foldedKernel, foldedBias) = folded;
                var created = new Entry
                {
                    Bias = bias, Gamma = gamma, Beta = beta, Mean = mean, Variance = variance,
                    Versions = versions, Epsilon = epsilon, Kernel = foldedKernel, FoldedBias = foldedBias,
                };
                // A fold made while the weights are training is replaced at every evaluation; its folded tensors are
                // the cache's own, so the engine frees their device memory now rather than at their finalizers.
                if (entry is not null && releaseReplaced is not null)
                {
                    releaseReplaced(entry.Kernel);
                    releaseReplaced(entry.FoldedBias);
                }
                s_entries.Remove(kernel);
                s_entries.Add(kernel, created);
                return (foldedKernel, foldedBias);
            }
        }

        /// <summary>
        /// Forgets the fold cached for <paramref name="kernel"/>, handing its folded tensors to
        /// <paramref name="release"/>; nothing when there is none.
        /// </summary>
        internal static void Forget(Tensor<T> kernel, Action<Tensor<T>> release)
        {
            Entry? entry;
            lock (s_entries)
            {
                if (!s_entries.TryGetValue(kernel, out entry)) return;
                s_entries.Remove(kernel);
            }
            release(entry.Kernel);
            release(entry.FoldedBias);
        }

        internal static (Tensor<T> Kernel, Tensor<T> Bias) FoldOnHost(Tensor<T> kernel, Tensor<T>? bias, Tensor<T> gamma,
            Tensor<T> beta, Tensor<T> mean, Tensor<T> variance, double epsilon)
        {
            var ops = MathHelper.GetNumericOperations<T>();
            int outChannels = kernel.Shape[0];
            int perChannel = kernel.Length / outChannels;
            var w = kernel.ToArray();
            var g = gamma.ToArray();
            var b = beta.ToArray();
            var m = mean.ToArray();
            var v = variance.ToArray();
            var c = bias?.ToArray();
            var foldedW = new T[w.Length];
            var foldedB = new T[outChannels];
            for (int o = 0; o < outChannels; o++)
            {
                double s = ops.ToDouble(g[o]) / Math.Sqrt(ops.ToDouble(v[o]) + epsilon);
                double convBias = c is null ? 0.0 : ops.ToDouble(c[o]);
                foldedB[o] = ops.FromDouble(ops.ToDouble(b[o]) + (convBias - ops.ToDouble(m[o])) * s);
                int start = o * perChannel;
                for (int i = start; i < start + perChannel; i++)
                    foldedW[i] = ops.FromDouble(ops.ToDouble(w[i]) * s);
            }
            return (new Tensor<T>(foldedW, kernel.Shape.ToArray()), new Tensor<T>(foldedB, new[] { outChannels }));
        }
    }
}
