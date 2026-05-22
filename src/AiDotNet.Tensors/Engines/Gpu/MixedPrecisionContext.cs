using System;

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
    /// Stage 9 (#415): when true and <typeparamref name="T"/> is
    /// <c>double</c>, the training loop is expected to keep an FP64 master
    /// copy of weights while running forward and backward in FP32. This is
    /// the inverse of the traditional FP32-master / FP16-working pattern:
    /// FP64 stays as the optimizer-precision master, FP32 is the faster
    /// compute precision (2× SIMD throughput on AVX2 vs FP64; ~4× on
    /// AVX-512). Default off; opt-in per-model — accumulated drift over
    /// 250+ training iters needs validation against the user's objective.
    /// </summary>
    public bool Fp32WorkingForDouble { get; }

    /// <summary>
    /// Create a new context.
    /// </summary>
    /// <param name="policy">Per-layer overrides. When null, uses
    /// <see cref="LayerPrecisionPolicy.ForFP16"/> by default.</param>
    /// <param name="defaultPrecision">Scope-wide precision. Default
    /// <see cref="PrecisionMode.Float16"/>.</param>
    /// <param name="initialLossScale">Initial scaler factor. PyTorch
    /// default is <c>65536</c>.</param>
    /// <param name="fp32WorkingForDouble">Stage 9 (#415): opt in to the
    /// FP32-working / FP64-master mode when <typeparamref name="T"/> is
    /// <c>double</c>. Silently ignored for non-double T (stored as false
    /// on the property).</param>
    public MixedPrecisionContext(
        LayerPrecisionPolicy? policy = null,
        PrecisionMode defaultPrecision = PrecisionMode.Float16,
        double initialLossScale = 65536.0,
        bool fp32WorkingForDouble = false)
    {
        Policy = policy ?? LayerPrecisionPolicy.ForFP16();
        DefaultPrecision = defaultPrecision;
        Scaler = new GradScaler<T>(initialLossScale);
        Fp32WorkingForDouble = fp32WorkingForDouble && typeof(T) == typeof(double);
    }

    /// <summary>
    /// Stage 9 (#415): downcast an FP64 buffer to FP32 for the working-
    /// precision pass. Caller owns both buffers. Simple element-wise loop
    /// — cost is dwarfed by the subsequent forward/backward FMAs.
    /// </summary>
    public static void CastDoubleToFloat(ReadOnlySpan<double> src, Span<float> dst)
    {
        if (src.Length != dst.Length)
            throw new ArgumentException($"length mismatch: src={src.Length}, dst={dst.Length}");
        for (int i = 0; i < src.Length; i++) dst[i] = (float)src[i];
    }

    /// <summary>
    /// Stage 9 (#415): upcast an FP32 gradient buffer back to FP64 before
    /// the optimizer step. Caller owns both buffers.
    /// </summary>
    public static void CastFloatToDouble(ReadOnlySpan<float> src, Span<double> dst)
    {
        if (src.Length != dst.Length)
            throw new ArgumentException($"length mismatch: src={src.Length}, dst={dst.Length}");
        for (int i = 0; i < src.Length; i++) dst[i] = src[i];
    }

    /// <summary>
    /// Stage 9 (#415): cosine similarity between FP32-path gradient and
    /// FP64 reference (flat-vector). Returns 1.0 for identical direction;
    /// values &lt; 0.999 indicate the FP32-working mode is producing a
    /// meaningfully different step direction and should not be enabled
    /// for the model. Used by validation harnesses, not the hot path.
    /// </summary>
    public static double GradientCosineSimilarity(ReadOnlySpan<double> a, ReadOnlySpan<double> b)
    {
        if (a.Length != b.Length)
            throw new ArgumentException($"length mismatch: a={a.Length}, b={b.Length}");
        double dot = 0.0, na = 0.0, nb = 0.0;
        for (int i = 0; i < a.Length; i++)
        {
            dot += a[i] * b[i];
            na += a[i] * a[i];
            nb += b[i] * b[i];
        }
        if (na == 0.0 || nb == 0.0) return 0.0;
        return dot / (System.Math.Sqrt(na) * System.Math.Sqrt(nb));
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
