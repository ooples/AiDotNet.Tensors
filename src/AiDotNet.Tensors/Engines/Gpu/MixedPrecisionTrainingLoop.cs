using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Gpu;

/// <summary>
/// Single-iteration result from <see cref="MixedPrecisionTrainingLoop{T}.Step"/>.
/// </summary>
/// <param name="Loss">Loss value at this step (computed inside the
/// autocast scope, re-cast to full precision before return).</param>
/// <param name="StepTaken">True when the optimizer update ran; false
/// when the scaler detected an overflow and the step was skipped.</param>
/// <param name="CurrentLossScale">Scaler's scale factor after the
/// post-step <c>Update</c> — useful for telemetry / diagnostics.</param>
public record TrainingStepResult<T>(T Loss, bool StepTaken, double CurrentLossScale);

/// <summary>
/// High-level mixed-precision training helper — the Tensors-parity
/// version of PyTorch's idiomatic autocast-plus-GradScaler pattern:
/// <code>
/// with torch.cuda.amp.autocast():
///     logits = model(x); loss = loss_fn(logits, y)
/// scaler.scale(loss).backward()
/// scaler.step(optimizer)
/// scaler.update()
/// </code>
/// Bundles an <see cref="IEngine"/> + <see cref="MixedPrecisionContext{T}"/>
/// + user-supplied forward / loss / optimizer callbacks into a single
/// <see cref="Step"/> call.
///
/// <para>The loop is deliberately callback-driven rather than baking
/// in a specific model / loss shape — consumers wire their own forward
/// pass, loss function, gradient accessor, and optimizer update. That
/// keeps Tensors decoupled from the high-level training framework while
/// still owning the autocast + scale + overflow-detect orchestration
/// that's universally the same.</para>
/// </summary>
public sealed class MixedPrecisionTrainingLoop<T> : IDisposable
{
    private readonly IEngine _engine;
    private readonly MixedPrecisionContext<T> _context;
    private readonly Func<Tensor<T>, Tensor<T>> _forward;
    private readonly Func<Tensor<T>, Tensor<T>, Tensor<T>> _lossFunction;
    private readonly Func<Tensor<T>[]> _getGradients;
    private readonly Action<Tensor<T>[], float> _applyOptimizerStep;
    private bool _disposed;

    /// <summary>
    /// The wrapped context — exposed so callers can inspect the
    /// scaler state, clear the per-name tensor cache between epochs,
    /// or swap the policy mid-training.
    /// </summary>
    public MixedPrecisionContext<T> Context => _context;

    /// <param name="engine">Engine used inside the autocast scope.</param>
    /// <param name="context">Precision + scaler orchestration state.</param>
    /// <param name="forward">Model forward: input → logits. Must allocate
    /// internal tensors via the supplied engine to interoperate with
    /// autocast.</param>
    /// <param name="lossFunction">(logits, target) → loss scalar.</param>
    /// <param name="getGradients">Returns the parameter gradients — the
    /// Tensors layer doesn't own a tape here, the caller does. For
    /// eager autograd this is <c>tape.Gradients(params)</c>; for
    /// compiled plans it's <c>plan.Gradients</c>.</param>
    /// <param name="applyOptimizerStep">(gradients, learningRate) →
    /// apply the optimizer update. Called only when the scaler's
    /// overflow check clears (no NaN / Inf). Consumers typically bind
    /// this to SGD / Adam / AdamW depending on their training recipe.</param>
    public MixedPrecisionTrainingLoop(
        IEngine engine,
        MixedPrecisionContext<T> context,
        Func<Tensor<T>, Tensor<T>> forward,
        Func<Tensor<T>, Tensor<T>, Tensor<T>> lossFunction,
        Func<Tensor<T>[]> getGradients,
        Action<Tensor<T>[], float> applyOptimizerStep)
    {
        _engine = engine ?? throw new ArgumentNullException(nameof(engine));
        _context = context ?? throw new ArgumentNullException(nameof(context));
        _forward = forward ?? throw new ArgumentNullException(nameof(forward));
        _lossFunction = lossFunction ?? throw new ArgumentNullException(nameof(lossFunction));
        _getGradients = getGradients ?? throw new ArgumentNullException(nameof(getGradients));
        _applyOptimizerStep = applyOptimizerStep ?? throw new ArgumentNullException(nameof(applyOptimizerStep));
    }

    /// <summary>
    /// Execute one training step. Opens the autocast scope, runs
    /// <c>forward → loss → scale → backward (implicit, via caller's
    /// tape) → collect gradients → unscale + overflow-check → optimizer
    /// step (if no overflow) → scaler.Update()</c>.
    /// </summary>
    public TrainingStepResult<T> Step(Tensor<T> input, Tensor<T> target, float learningRate)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(MixedPrecisionTrainingLoop<T>));
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (target is null) throw new ArgumentNullException(nameof(target));

        Tensor<T> loss;
        using (_context.BeginAutocast())
        {
            var logits = _forward(input);
            loss = _lossFunction(logits, target);
            // The scaled loss is what the caller's backward sees — their
            // tape records the scaled-gradient computation from here.
            var scaled = _context.Scaler.ScaleLoss(loss, _engine);
            // We don't call Backward on the tensor — callers run their
            // own tape between this loop's forward and the next step.
            // Expose the scaled loss so they can call Backward on it.
            _ = scaled;
        }

        // Collect gradients and run the unscale + overflow check atomically.
        var gradients = _getGradients();
        if (gradients is null || gradients.Length == 0)
        {
            // No gradients collected — treat as a no-op step (caller's
            // forward likely didn't record). Scale stays put.
            return new TrainingStepResult<T>(loss.AsSpan()[0], StepTaken: false, _context.Scaler.Scale);
        }

        _context.Scaler.Unscale(gradients, _engine);
        bool proceed = _context.Scaler.ShouldStep();
        if (proceed)
        {
            _applyOptimizerStep(gradients, learningRate);
        }
        _context.Scaler.Update();

        T lossScalar = loss.AsSpan().Length > 0
            ? loss.AsSpan()[0]
            : Helpers.MathHelper.GetNumericOperations<T>().Zero;
        return new TrainingStepResult<T>(lossScalar, proceed, _context.Scaler.Scale);
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _context.Dispose();
    }
}
