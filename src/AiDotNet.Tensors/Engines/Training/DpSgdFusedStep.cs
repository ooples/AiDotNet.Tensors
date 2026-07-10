using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Training;

/// <summary>
/// Fused per-example DP-SGD (Abadi et al. 2016 §3, Algorithm 1). Runs each of K
/// per-example forward+backward passes through the compiled plan (with the
/// optimizer step's learning rate set to zero so weights don't drift between
/// examples), extracts each example's gradients from the parameter .Grad
/// tensors, clips against the GLOBAL parameter-vector L2 norm, sums, adds a
/// single Gaussian noise draw, averages, then applies the final aggregate
/// update to the parameters.
///
/// <para>Fused benefit over an eager per-example loop: each per-example
/// forward+backward runs the compiled plan (with GPU-resident parameters,
/// fused kernels, no per-op host↔device round-trip) instead of a fresh
/// non-persistent <see cref="GradientTape{T}"/>. The clip / aggregate / noise
/// steps run in host code because their control flow (per-example L2 norm
/// then min-clip against C) doesn't fit the current compiled-plan capture
/// model — but those are O(params) scalar operations, not the compute
/// bottleneck. The forward+backward per example IS the expensive part, and
/// that's what gets fused.</para>
///
/// <para>Correctness contract: the clip-BEFORE-aggregate order is enforced by
/// the class's internal structure — callers can't reverse it. This preserves
/// the L2-sensitivity bound the Abadi 2016 privacy proof requires.</para>
/// </summary>
/// <typeparam name="T">Numeric type (float / double).</typeparam>
public sealed class DpSgdFusedStep<T> : IDisposable
{
    // Persistent per-example slots. Refreshed with each example's data before
    // running plan.Step(). The plan captured these references at trace time.
    private Tensor<T>[]? _persistentSlots;
    private ICompiledTrainingPlan<T>? _plan;
    private int[]? _cachedShapeKey;
    private object?[]? _cachedParamIdentities;
    private Tensor<T>[]? _cachedParameters;
    private bool _disposed;

    /// <summary>Whether the fused-resident DP-SGD path is available on this
    /// thread's current engine.</summary>
    public static bool IsAvailable =>
        typeof(T) == typeof(float)
        && AiDotNetEngine.Current is DirectGpuTensorEngine gpu && gpu.SupportsGpu
        && Optimization.TensorCodecOptions.Current.EnableCompilation;

    /// <summary>
    /// Runs a full DP-SGD training step over <paramref name="batchSize"/> examples.
    /// </summary>
    /// <param name="parameters">Trainable tensors (de-duplicated by reference).
    /// Each gets its own moment buffer.</param>
    /// <param name="perExampleSlotData">Callback returning the fresh slot data
    /// for the given example index. Slot count and shapes must be stable across
    /// example indices (a change triggers a recompile).</param>
    /// <param name="forward">Forward closure that consumes the persistent slots
    /// and returns the predicted output tensor.</param>
    /// <param name="computeLoss">Loss closure — pred + slots → scalar loss.</param>
    /// <param name="batchSize">Number of per-example passes to run.</param>
    /// <param name="clipNorm">Per-example L2 norm clip (Abadi 2016's C).</param>
    /// <param name="noiseMultiplier">Gaussian noise multiplier (Abadi 2016's σ).</param>
    /// <param name="learningRate">Learning rate of the FINAL aggregate update
    /// applied to parameters after clip + noise + average.</param>
    /// <param name="rng">RNG for Gaussian noise draws.</param>
    /// <returns>True when the fused compiled DP-SGD path ran; false to fall
    /// back to eager.</returns>
    public bool TryStep(
        IReadOnlyList<Tensor<T>> parameters,
        Func<int, IReadOnlyList<Tensor<T>>> perExampleSlotData,
        Func<IReadOnlyList<Tensor<T>>, Tensor<T>> forward,
        Func<Tensor<T>, IReadOnlyList<Tensor<T>>, Tensor<T>> computeLoss,
        int batchSize,
        double clipNorm,
        double noiseMultiplier,
        float learningRate,
        Random rng)
    {
        ThrowIfDisposed();

        if (!IsAvailable) return false;
        if (parameters is null || parameters.Count == 0) return false;
        if (perExampleSlotData is null) throw new ArgumentNullException(nameof(perExampleSlotData));
        if (forward is null) throw new ArgumentNullException(nameof(forward));
        if (computeLoss is null) throw new ArgumentNullException(nameof(computeLoss));
        if (batchSize <= 0) throw new ArgumentOutOfRangeException(nameof(batchSize));
        if (clipNorm <= 0) throw new ArgumentOutOfRangeException(nameof(clipNorm));
        if (rng is null) throw new ArgumentNullException(nameof(rng));

        var ops = MathHelper.GetNumericOperations<T>();

        try
        {
            // Get the first example to establish slot shapes (before compile).
            var firstExample = perExampleSlotData(0);
            if (firstExample is null || firstExample.Count == 0) return false;

            int[] shapeKey = ComputeCompositeShapeKey(firstExample);
            bool shapeChanged = _cachedShapeKey is null || !ShapeKeysEqual(shapeKey, _cachedShapeKey);
            bool paramsChanged = ParameterSetChanged(parameters);

            if (shapeChanged || paramsChanged)
            {
                InvalidateCachedPlan();
                AllocatePersistentSlots(firstExample);
                _cachedShapeKey = shapeKey;
                RememberParameterSet(parameters);
                _cachedParameters = new Tensor<T>[parameters.Count];
                for (int i = 0; i < parameters.Count; i++) _cachedParameters[i] = parameters[i];

                // GPU-residency for the parameters
                if (typeof(T) == typeof(float)
                    && Environment.GetEnvironmentVariable("AIDOTNET_GPU_RESIDENT_PARAMS") != "0")
                {
                    foreach (var p in _cachedParameters) p.Gpu();
                }
            }

            if (_persistentSlots is null || _cachedParameters is null) return false;

            // Trace + compile on first Step. Plan runs forward+backward and
            // applies a zero-LR SGD step so weights don't drift between
            // per-example replays. The .Grad on each parameter is populated
            // by the backward pass regardless of the optimizer's LR.
            if (_plan is null)
            {
                CopySlotData(firstExample);
                using var arenaSuspend = TensorArena.Suspend();
                using var scope = GraphMode.Enable();
                var pred = forward(_persistentSlots);
                var loss = computeLoss(pred, _persistentSlots);
                _plan = scope.CompileTraining(_cachedParameters, loss);
                _plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0.0f, beta1: 0.9f, beta2: 0.999f, eps: 1e-8f, weightDecay: 0f);
            }

            // Per-example accumulators for clipped gradients.
            var clippedSums = new Tensor<T>[_cachedParameters.Length];
            for (int p = 0; p < _cachedParameters.Length; p++)
                clippedSums[p] = new Tensor<T>((int[])_cachedParameters[p]._shape.Clone());

            // Per-example forward+backward via the compiled plan. Each example
            // populates parameter .Grad; we snapshot into per-example arrays
            // BEFORE clipping (the plan will overwrite .Grad on the next call).
            for (int example = 0; example < batchSize; example++)
            {
                var exampleData = example == 0 ? firstExample : perExampleSlotData(example);
                if (exampleData.Count != _persistentSlots.Length) return false;
                CopySlotData(exampleData);

                // Run compiled forward+backward. LR=0 means weights don't update
                // but .Grad on every parameter gets populated by the backward.
                _plan.Step();

                // Compute GLOBAL L2 norm across ALL parameter gradients
                // concatenated. Required by Abadi 2016 L2-sensitivity bound —
                // per-parameter norms would break the DP guarantee.
                double normSquared = 0.0;
                for (int p = 0; p < _cachedParameters.Length; p++)
                {
                    var grad = _cachedParameters[p].Grad;
                    if (grad is null) continue;
                    var span = grad.AsSpan();
                    for (int i = 0; i < span.Length; i++)
                    {
                        double v = ops.ToDouble(span[i]);
                        normSquared += v * v;
                    }
                }
                double clipFactor = Math.Min(1.0, clipNorm / Math.Sqrt(normSquared + 1e-12));

                // Accumulate clipped per-example gradient into sums.
                for (int p = 0; p < _cachedParameters.Length; p++)
                {
                    var grad = _cachedParameters[p].Grad;
                    if (grad is null) continue;
                    var sumSpan = clippedSums[p].AsWritableSpan();
                    var gSpan = grad.AsSpan();
                    for (int i = 0; i < gSpan.Length; i++)
                    {
                        double v = ops.ToDouble(sumSpan[i]) + ops.ToDouble(gSpan[i]) * clipFactor;
                        sumSpan[i] = ops.FromDouble(v);
                    }
                }
            }

            // Final update: add Gaussian noise, average by batchSize, apply
            // learningRate * aggregate to each parameter.
            double invBatch = 1.0 / batchSize;
            double noiseStd = clipNorm * noiseMultiplier * invBatch;
            for (int p = 0; p < _cachedParameters.Length; p++)
            {
                var sumSpan = clippedSums[p].AsSpan();
                var paramSpan = _cachedParameters[p].AsWritableSpan();
                for (int i = 0; i < sumSpan.Length; i++)
                {
                    double noise = noiseStd > 0 ? SampleGaussian(rng) * noiseStd : 0.0;
                    double aggregateGrad = ops.ToDouble(sumSpan[i]) * invBatch + noise;
                    double newParam = ops.ToDouble(paramSpan[i]) - learningRate * aggregateGrad;
                    paramSpan[i] = ops.FromDouble(newParam);
                }
            }

            return true;
        }
        catch (NotSupportedException)
        {
            InvalidateCachedPlan();
            return false;
        }
        catch (InvalidOperationException)
        {
            InvalidateCachedPlan();
            return false;
        }
    }

    /// <summary>Invalidates the cached compiled plan and persistent slots.
    /// Call when model structure changes; not needed when only data changes.</summary>
    public void Invalidate() => InvalidateCachedPlan();

    private void InvalidateCachedPlan()
    {
        _plan?.Dispose();
        _plan = null;
        _persistentSlots = null;
        _cachedShapeKey = null;
        _cachedParamIdentities = null;
        _cachedParameters = null;
    }

    private void AllocatePersistentSlots(IReadOnlyList<Tensor<T>> firstExample)
    {
        _persistentSlots = new Tensor<T>[firstExample.Count];
        for (int i = 0; i < firstExample.Count; i++)
        {
            _persistentSlots[i] = new Tensor<T>((int[])firstExample[i]._shape.Clone());
        }
    }

    private void CopySlotData(IReadOnlyList<Tensor<T>> fresh)
    {
        if (_persistentSlots is null) return;
        for (int i = 0; i < _persistentSlots.Length; i++)
        {
            var slot = _persistentSlots[i];
            var src = fresh[i];
            if (src.Length != slot.Length)
                throw new InvalidOperationException(
                    $"DpSgdFusedStep: per-example slot {i} shape changed mid-batch. " +
                    $"Expected {slot.Length} elements, got {src.Length}.");
            src.AsSpan().CopyTo(slot.AsWritableSpan());
        }
    }

    private static int[] ComputeCompositeShapeKey(IReadOnlyList<Tensor<T>> slots)
    {
        int total = slots.Count;
        for (int i = 0; i < slots.Count; i++) total += slots[i]._shape.Length;
        var key = new int[total];
        int idx = 0;
        for (int i = 0; i < slots.Count; i++)
        {
            var s = slots[i]._shape;
            for (int d = 0; d < s.Length; d++) key[idx++] = s[d];
            key[idx++] = -1 - i;
        }
        return key;
    }

    private static bool ShapeKeysEqual(int[] a, int[] b)
    {
        if (a.Length != b.Length) return false;
        for (int i = 0; i < a.Length; i++) if (a[i] != b[i]) return false;
        return true;
    }

    private bool ParameterSetChanged(IReadOnlyList<Tensor<T>> parameters)
    {
        if (_cachedParamIdentities is null) return true;
        if (_cachedParamIdentities.Length != parameters.Count) return true;
        for (int i = 0; i < parameters.Count; i++)
            if (!ReferenceEquals(_cachedParamIdentities[i], parameters[i])) return true;
        return false;
    }

    private void RememberParameterSet(IReadOnlyList<Tensor<T>> parameters)
    {
        _cachedParamIdentities = new object?[parameters.Count];
        for (int i = 0; i < parameters.Count; i++) _cachedParamIdentities[i] = parameters[i];
    }

    private static double SampleGaussian(Random rng)
    {
        double u1;
        do { u1 = rng.NextDouble(); } while (u1 < 1e-300);
        double u2 = rng.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }

    public void Dispose()
    {
        if (_disposed) return;
        InvalidateCachedPlan();
        _disposed = true;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(nameof(DpSgdFusedStep<T>));
    }
}
