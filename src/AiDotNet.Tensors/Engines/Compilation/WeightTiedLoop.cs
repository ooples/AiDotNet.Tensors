using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Compilation;

/// <summary>
/// A weight-tied (recurrent-depth) loop: one block applied several times to a state, the same parameters read on every
/// iteration. It runs the same way eagerly, on a gradient tape, and in a training trace for a compiled plan.
/// </summary>
/// <remarks>
/// <para>
/// A compiled training trace unrolls the loop. Each iteration reads the same parameter tensors, so the plan's backward
/// sums every iteration's contribution into one gradient per parameter, as PyTorch autograd does for a module called
/// several times. A tensor that only iteration <c>i</c> reads (a per-iteration adapter indexed by the iteration number
/// the body receives) gets only that iteration's gradient.
/// </para>
/// <para>
/// With <c>checkpointEachIteration</c> every iteration of the body is one activation-checkpoint segment of the trace
/// (the compiled counterpart of wrapping each iteration in <c>torch.utils.checkpoint</c>): the plan keeps only each
/// iteration's output, releases the rest when the iteration's forward ends and recomputes it inside the captured step
/// just before that iteration's backward, so the activations held across the step do not grow with the iteration count
/// beyond one state per iteration. Eagerly (no training trace) the flag changes nothing: the loop runs on the tape.
/// </para>
/// </remarks>
public static class WeightTiedLoop
{
    /// <summary>
    /// Applies <paramref name="body"/> <paramref name="iterations"/> times: <c>h_i = body(h_{i-1}, i - 1)</c>, from
    /// <c>h_0 = state</c>, and returns <c>h_iterations</c>.
    /// </summary>
    /// <param name="engine">The engine the loop's own ops run on (the body uses whatever it uses).</param>
    /// <param name="state">The loop's initial state.</param>
    /// <param name="iterations">How many times the body runs (at least 1).</param>
    /// <param name="body">One iteration: the state and the zero-based iteration number to the next state, the same shape.</param>
    /// <param name="checkpointEachIteration">Mark each iteration as an activation-checkpoint segment of a training trace.</param>
    public static Tensor<T> Run<T>(IEngine engine, Tensor<T> state, int iterations, Func<Tensor<T>, int, Tensor<T>> body,
        bool checkpointEachIteration = false)
    {
        if (engine is null) throw new ArgumentNullException(nameof(engine));
        if (state is null) throw new ArgumentNullException(nameof(state));
        if (body is null) throw new ArgumentNullException(nameof(body));
        if (iterations < 1) throw new ArgumentOutOfRangeException(nameof(iterations), "A loop runs its body at least once.");
        var h = state;
        for (int i = 0; i < iterations; i++)
            h = Iterate(h, i, body, checkpointEachIteration);
        return h;
    }

    /// <summary>
    /// A loop of at most <paramref name="maxIterations"/> iterations with a learned, per-token exit (entropy-regularised
    /// halting, as in Ouro): every iteration runs for every token, so the step has no host-side branch and a captured
    /// graph replays, and a token that has exited keeps its state through the remaining iterations.
    /// </summary>
    /// <remarks>
    /// <para>
    /// After iteration <c>t</c> (1-based) the halting head gives <c>z_t = haltLogit(h_t)</c>, shape <c>[..., 1]</c> for a
    /// state <c>[..., D]</c>, and <c>lambda_t = sigmoid(z_t)</c>. The exit distribution over the iterations is
    /// <c>p_t = lambda_t * prod_{j &lt; t} (1 - lambda_j)</c> for <c>t &lt; T</c> and the remaining mass
    /// <c>p_T = prod_{j &lt; T} (1 - lambda_j)</c> at the last one (<c>T = maxIterations</c>; the head is not evaluated on
    /// <c>h_T</c>). Its logarithm is formed from <c>log lambda = -softplus(-z)</c> and <c>log(1 - lambda) = -softplus(z)</c>,
    /// so neither underflows.
    /// </para>
    /// <para>
    /// The state update is masked: <c>h_t = h_{t-1} + m_t * (body(h_{t-1}, t - 1) - h_{t-1})</c> with
    /// <c>m_t = 1[CDF_{t-1} &lt; exitThreshold]</c>, <c>CDF_{t-1} = sum_{j &lt; t} p_j</c> (so <c>m_1 = 1</c>). The mask is a
    /// device tensor computed without gradient (a hard exit decision, like the threshold exit at inference); the
    /// gradient reaches the body through the tokens still running and the halting head through <c>p</c>. An
    /// <paramref name="exitThreshold"/> of 1 or more never masks (training on the full expected loss).
    /// </para>
    /// </remarks>
    /// <param name="engine">The engine the loop's own ops run on.</param>
    /// <param name="state">The initial state <c>[..., D]</c>.</param>
    /// <param name="maxIterations">The fixed maximum number of iterations (at least 1).</param>
    /// <param name="body">One iteration: the state and the zero-based iteration number to the next state.</param>
    /// <param name="haltLogit">The halting head: a state to its per-token halting logit, shape <c>[..., 1]</c>.</param>
    /// <param name="exitThreshold">The exit CDF threshold <c>q</c> of the masked update; 1 or more never masks.</param>
    /// <param name="checkpointEachIteration">Mark each iteration's body as an activation-checkpoint segment of a training trace.</param>
    [ScalarKernel("Converts a few per-iteration constants (the iteration number, the exit floor); every element is an engine op.")]
    public static AdaptiveLoopResult<T> RunAdaptive<T>(IEngine engine, Tensor<T> state, int maxIterations,
        Func<Tensor<T>, int, Tensor<T>> body, Func<Tensor<T>, Tensor<T>> haltLogit, double exitThreshold = 1.0,
        bool checkpointEachIteration = false)
    {
        if (engine is null) throw new ArgumentNullException(nameof(engine));
        if (state is null) throw new ArgumentNullException(nameof(state));
        if (body is null) throw new ArgumentNullException(nameof(body));
        if (haltLogit is null) throw new ArgumentNullException(nameof(haltLogit));
        if (maxIterations < 1) throw new ArgumentOutOfRangeException(nameof(maxIterations), "A loop runs its body at least once.");
        if (double.IsNaN(exitThreshold) || exitThreshold <= 0)
            throw new ArgumentOutOfRangeException(nameof(exitThreshold), "The exit threshold is a CDF level in (0, 1], or above 1 for no exit.");
        var ops = MathHelper.GetNumericOperations<T>();
        bool masked = exitThreshold < 1.0;
        // A token keeps running while CDF_{t-1} < q, i.e. while its survival prod (1 - lambda_j) > 1 - q.
        T survivalFloor = ops.FromDouble(1.0 - exitThreshold);

        var states = new Tensor<T>[maxIterations];
        var logP = new Tensor<T>[maxIterations];
        Tensor<T>? logSurvival = null;   // log prod_{j < t} (1 - lambda_j); null = log 1
        Tensor<T>? running = null;       // m_t for the next iteration (no gradient); null = all ones
        Tensor<T>? loopsRun = null;      // per token: sum_t m_t (no gradient)
        var h = state;
        for (int t = 0; t < maxIterations; t++)
        {
            var next = Iterate(h, t, body, checkpointEachIteration);
            if (running is not null)
                next = engine.TensorAdd(h, engine.TensorMultiply(running, engine.TensorSubtract(next, h)));
            h = next;
            states[t] = h;
            if (t == maxIterations - 1)
            {
                logP[t] = logSurvival ?? ZerosLike(engine, haltLogit(h));
                break;
            }
            var z = haltLogit(h);
            var logLambda = engine.TensorNegate(engine.Softplus(engine.TensorNegate(z)));
            var logOneMinus = engine.TensorNegate(engine.Softplus(z));
            logP[t] = logSurvival is null ? logLambda : engine.TensorAdd(logSurvival, logLambda);
            logSurvival = logSurvival is null ? logOneMinus : engine.TensorAdd(logSurvival, logOneMinus);
            if (masked)
                using (new NoGradScope<T>())
                {
                    var survival = engine.TensorExp(logSurvival);
                    var stillRunning = engine.TensorGreaterThan(survival, survivalFloor);
                    running = running is null ? stillRunning : engine.TensorMultiply(running, stillRunning);
                    loopsRun = loopsRun is null ? engine.TensorAddScalar(running, ops.One) : engine.TensorAdd(loopsRun, running);
                }
        }

        var p = new Tensor<T>[maxIterations];
        Tensor<T>? negEntropy = null, expectedLoops = null;
        for (int t = 0; t < maxIterations; t++)
        {
            p[t] = engine.TensorExp(logP[t]);
            var plogp = engine.TensorMultiply(p[t], logP[t]);
            negEntropy = negEntropy is null ? plogp : engine.TensorAdd(negEntropy, plogp);
            var weighted = engine.TensorMultiplyScalar(p[t], ops.FromDouble(t + 1));
            expectedLoops = expectedLoops is null ? weighted : engine.TensorAdd(expectedLoops, weighted);
        }
        int tokens = p[0].Length;
        T inverseTokens = ops.FromDouble(1.0 / tokens);
        Tensor<T> loopsRunMean;
        using (new NoGradScope<T>())
            loopsRunMean = loopsRun is null
                ? engine.TensorAddScalar(engine.TensorMultiplyScalar(engine.ReduceSum(p[0], null), ops.Zero), ops.FromDouble(maxIterations))
                : engine.TensorMultiplyScalar(engine.ReduceSum(loopsRun, null), inverseTokens);
        return new AdaptiveLoopResult<T>(engine, h, states, p,
            engine.TensorMultiplyScalar(engine.ReduceSum(negEntropy!, null), ops.FromDouble(-1.0 / tokens)),
            engine.TensorMultiplyScalar(engine.ReduceSum(expectedLoops!, null), inverseTokens),
            loopsRunMean);
    }

    private static Tensor<T> Iterate<T>(Tensor<T> h, int iteration, Func<Tensor<T>, int, Tensor<T>> body, bool checkpoint)
    {
        Tensor<T> next;
        if (checkpoint)
            using (GraphMode.BeginCheckpointSegment())
                next = body(h, iteration);
        else next = body(h, iteration);
        if (next is null) throw new InvalidOperationException($"The loop body returned null at iteration {iteration}.");
        if (!ShapesEqual(next._shape, h._shape))
            throw new InvalidOperationException(
                $"The loop body must keep the state's shape: iteration {iteration} mapped [{string.Join(", ", h._shape)}] "
                + $"to [{string.Join(", ", next._shape)}].");
        return next;
    }

    private static Tensor<T> ZerosLike<T>(IEngine engine, Tensor<T> like)
    {
        // log 1 for a one-iteration loop, shaped like the halting head's output; no gradient flows through it.
        using (new NoGradScope<T>())
            return engine.TensorMultiplyScalar(like, MathHelper.GetNumericOperations<T>().Zero);
    }

    private static bool ShapesEqual(int[] a, int[] b)
    {
        if (a.Length != b.Length) return false;
        for (int i = 0; i < a.Length; i++) if (a[i] != b[i]) return false;
        return true;
    }
}

/// <summary>The result of <see cref="WeightTiedLoop.RunAdaptive{T}"/>: tape-connected device tensors, no host reads.</summary>
public sealed class AdaptiveLoopResult<T>
{
    private readonly IEngine _engine;

    internal AdaptiveLoopResult(IEngine engine, Tensor<T> output, Tensor<T>[] states, Tensor<T>[] exitProbabilities,
        Tensor<T> entropy, Tensor<T> expectedLoops, Tensor<T> loopsRun)
    {
        _engine = engine;
        Output = output;
        States = states;
        ExitProbabilities = exitProbabilities;
        Entropy = entropy;
        ExpectedLoops = expectedLoops;
        LoopsRun = loopsRun;
    }

    /// <summary>The last state, <c>h_T</c>: with a mask, each token's state at the iteration it exited.</summary>
    public Tensor<T> Output { get; }

    /// <summary>The state after each iteration, <c>h_1 .. h_T</c>.</summary>
    public IReadOnlyList<Tensor<T>> States { get; }

    /// <summary>The exit distribution <c>p_1 .. p_T</c>, each shaped like the halting head's output; they sum to 1 per token.</summary>
    public IReadOnlyList<Tensor<T>> ExitProbabilities { get; }

    /// <summary>The mean over tokens of the exit distribution's entropy <c>-sum_t p_t log p_t</c> (a scalar, shape [1]).</summary>
    public Tensor<T> Entropy { get; }

    /// <summary>The mean over tokens of <c>sum_t t p_t</c> (a scalar).</summary>
    public Tensor<T> ExpectedLoops { get; }

    /// <summary>The mean over tokens of the iterations whose update was applied, <c>sum_t m_t</c> (a scalar, no gradient).</summary>
    public Tensor<T> LoopsRun { get; }

    /// <summary>
    /// The mean over tokens of the expected loss under the exit distribution, <c>sum_t p_t L_t</c>, where
    /// <paramref name="perIterationTokenLoss"/>[t] is the per-token loss of <c>States[t]</c> (as many values as tokens).
    /// The entropy-regularised objective is <c>ExpectedLoss(L) - beta * Entropy</c>.
    /// </summary>
    public Tensor<T> ExpectedLoss(IReadOnlyList<Tensor<T>> perIterationTokenLoss)
    {
        if (perIterationTokenLoss is null) throw new ArgumentNullException(nameof(perIterationTokenLoss));
        if (perIterationTokenLoss.Count != ExitProbabilities.Count)
            throw new ArgumentException(
                $"One loss per iteration: {ExitProbabilities.Count} iterations, {perIterationTokenLoss.Count} losses.",
                nameof(perIterationTokenLoss));
        Tensor<T>? total = null;
        for (int t = 0; t < ExitProbabilities.Count; t++)
        {
            var p = ExitProbabilities[t];
            var loss = perIterationTokenLoss[t];
            if (loss.Length != p.Length)
                throw new ArgumentException(
                    $"Iteration {t}: {loss.Length} loss values for {p.Length} tokens.", nameof(perIterationTokenLoss));
            var term = _engine.TensorMultiply(p, _engine.Reshape(loss, p._shape));
            total = total is null ? term : _engine.TensorAdd(total, term);
        }
        var ops = MathHelper.GetNumericOperations<T>();
        return _engine.TensorMultiplyScalar(_engine.ReduceSum(total!, null), ops.FromDouble(1.0 / ExitProbabilities[0].Length));
    }
}
