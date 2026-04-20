namespace AiDotNet.Tensors.Engines.Gpu;

/// <summary>
/// Training-loop orchestration wrapper bundling an <see cref="AutocastScope"/>
/// with a <see cref="GradScaler{T}"/>. Lets training loops be written as:
/// <code>
/// using var ctx = new MixedPrecisionContext&lt;float&gt;(
///     policy: LayerPrecisionPolicy.ForFP16(),
///     initialLossScale: 65536.0);
///
/// foreach (var (x, y) in batches)
/// {
///     using (ctx.BeginAutocast())
///     {
///         var logits = model.Forward(x);
///         var loss = lossFn(logits, y);
///         var scaled = ctx.Scaler.ScaleLoss(loss, engine);
///         scaled.Backward();
///     }
///     if (ctx.Scaler.UnscaleGradientsAndCheck(grads))
///         optimizer.Step(grads);
///     ctx.Scaler.Update();
/// }
/// </code>
/// </summary>
/// <typeparam name="T">Compute element type — typically <c>float</c>.</typeparam>
public sealed class MixedPrecisionContext<T> : IDisposable
{
    /// <summary>Per-layer precision policy used by every
    /// <see cref="BeginAutocast"/> scope this context opens.</summary>
    public LayerPrecisionPolicy Policy { get; }

    /// <summary>Default precision for layers not overridden by <see cref="Policy"/>.</summary>
    public PrecisionMode DefaultPrecision { get; }

    /// <summary>Gradient scaler shared across every autocast scope this
    /// context opens. Scaler state (growth streak, overflow flag) is
    /// preserved across iterations — reset via <see cref="GradScaler{T}.Reset"/>
    /// when restarting training.</summary>
    public GradScaler<T> Scaler { get; }

    private bool _disposed;

    /// <summary>
    /// Create a new context.
    /// </summary>
    /// <param name="policy">Per-layer overrides. When null, uses
    /// <see cref="LayerPrecisionPolicy.ForFP16"/> by default.</param>
    /// <param name="defaultPrecision">Scope-wide precision. Default
    /// <see cref="PrecisionMode.Float16"/>.</param>
    /// <param name="initialLossScale">Initial scaler factor. PyTorch
    /// default is <c>65536</c>.</param>
    public MixedPrecisionContext(
        LayerPrecisionPolicy? policy = null,
        PrecisionMode defaultPrecision = PrecisionMode.Float16,
        double initialLossScale = 65536.0)
    {
        Policy = policy ?? LayerPrecisionPolicy.ForFP16();
        DefaultPrecision = defaultPrecision;
        Scaler = new GradScaler<T>(initialLossScale);
    }

    /// <summary>Opens an <see cref="AutocastScope"/> on this context's policy.
    /// Dispose the returned scope at the end of each forward pass to revert
    /// to the outer precision (matches PyTorch's <c>with autocast():</c>
    /// idiom).</summary>
    public AutocastScope BeginAutocast()
    {
        if (_disposed) throw new ObjectDisposedException(nameof(MixedPrecisionContext<T>));
        return new AutocastScope(DefaultPrecision, Policy);
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        // Scaler has no unmanaged state; no-op Dispose kept for API symmetry
        // with future training-loop helpers that may own engine-owned buffers.
    }
}
