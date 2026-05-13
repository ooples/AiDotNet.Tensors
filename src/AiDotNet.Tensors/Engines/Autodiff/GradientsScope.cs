using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Autodiff;

/// <summary>
/// Disposable wrapper around the gradient dictionary returned from
/// <see cref="GradientTape{T}.ComputeGradientsScope"/>. Provides
/// explicit caller-controlled lifetime for the per-source gradient
/// tensors so they can be pooled back to <see cref="AutoTensorCache"/>
/// when the caller is done consuming them.
///
/// <para>Closes the gap that <see cref="GradientTape{T}.ComputeGradients"/>
/// has by design: that overload returns a raw dictionary whose values
/// are owned by the caller, so the library has no safe point to pool
/// the gradient tensors back to the cache. Issue #327 measured this
/// gap at ~320 MB / iter on a 4-layer transformer Train step — the
/// per-step gradient tensors (one per weight matrix per layer) flow
/// through the GC instead of recycling.</para>
///
/// <para><b>Usage</b>:</para>
/// <code>
/// using (var scope = tape.ComputeGradientsScope(loss, weights))
/// {
///     optimizer.Step(scope.Grads, weights);
///     // On scope disposal, every gradient tensor in scope.Grads is
///     // returned to AutoTensorCache for next iter's RentOrAllocate
///     // to reuse.
/// }
/// </code>
///
/// <para><b>Lifetime contract</b>: the caller MUST NOT hold references
/// to <see cref="Grads"/> tensors past the <c>using</c> block —
/// post-disposal those tensors may be reissued by
/// <see cref="AutoTensorCache.RentOrAllocate"/> to a different op and
/// silently overwritten. The <see cref="GradientTape{T}.ComputeGradients"/>
/// non-scope overload remains for callers that need to retain
/// gradients beyond a single training step (gradient accumulation,
/// gradient clipping with delayed application, debugging).</para>
/// </summary>
public sealed class GradientsScope<T> : IDisposable
{
    private Dictionary<Tensor<T>, Tensor<T>>? _grads;
    private bool _disposed;

    /// <summary>The gradient dictionary. Valid until this scope disposes.</summary>
    public Dictionary<Tensor<T>, Tensor<T>> Grads
    {
        get
        {
            if (_disposed) throw new ObjectDisposedException(nameof(GradientsScope<T>));
            return _grads!;
        }
    }

    /// <summary>
    /// Constructs the scope. Internal-only; callers obtain instances
    /// via <see cref="GradientTape{T}.ComputeGradientsScope"/>.
    /// </summary>
    internal GradientsScope(Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        _grads = grads;
    }

    /// <summary>
    /// Returns every gradient tensor in this scope to
    /// <see cref="AutoTensorCache"/> for the next training iteration to
    /// reuse. The dictionary itself is cleared so subsequent access
    /// after disposal throws cleanly. Idempotent — calling Dispose
    /// twice is safe.
    /// </summary>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        if (_grads is null) return;

        // Walk distinct gradient tensors (in-place accumulation can
        // make multiple sources share a value reference). Pool each
        // unique tensor instance exactly once — duplicate Return calls
        // are cheap but would inflate the per-shape pool count past
        // its cap and cause some entries to be silently dropped.
        var seen = new HashSet<Tensor<T>>(ReferenceEqualityComparer<Tensor<T>>.Instance);
        foreach (var kv in _grads)
        {
            if (kv.Value is null) continue;
            if (!seen.Add(kv.Value)) continue;
            AutoTensorCache.Return(kv.Value);
        }
        _grads.Clear();
        _grads = null;
    }
}
