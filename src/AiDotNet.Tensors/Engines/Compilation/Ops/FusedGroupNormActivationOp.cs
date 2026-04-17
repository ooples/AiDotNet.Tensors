using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Compilation.Ops;

/// <summary>
/// Activation types supported by the fused GroupNorm+Activation kernel.
/// </summary>
internal enum GroupNormActivation : byte
{
    /// <summary>No activation — equivalent to plain GroupNorm.</summary>
    Identity = 0,

    /// <summary>SiLU / Swish: f(x) = x · σ(x). The diffusion-model default.</summary>
    SiLU = 1,

    /// <summary>ReLU: f(x) = max(0, x).</summary>
    ReLU = 2,
}

/// <summary>
/// IR operation for fused GroupNorm + Activation. Eliminates one full-size
/// intermediate tensor per fusion site by applying the activation in the same
/// pass as the normalization — roughly 41 MB saved per ResBlock at Stable
/// Diffusion 1.5 dimensions.
///
/// <para><b>Pattern:</b> <c>DiffusionResBlock</c> has the sequence
/// <c>GroupNorm(x) → SiLU(norm_out)</c> twice per block. SD15 UNet:
/// ~40 ResBlocks × 2 = 80 fusion opportunities per forward pass.</para>
///
/// <para><b>Forward:</b> for SiLU, delegates to
/// <see cref="IEngine.GroupNormSwishInto{T}"/> which applies both operations
/// in a single kernel. For ReLU and Identity, falls back to GroupNorm +
/// pointwise activation (still fused at the IR level — one CompiledStep
/// instead of two).</para>
///
/// <para><b>Backward:</b> composes the activation gradient (pointwise) with
/// the GroupNorm gradient via chain rule, without materializing the
/// intermediate (the normalized-but-not-activated tensor). Uses the
/// fused op's output to reconstruct the pre-activation values when needed
/// by the activation backward.</para>
/// </summary>
/// <typeparam name="T">The tensor element type.</typeparam>
internal sealed class FusedGroupNormActivationOp<T> : ICompiledOp<T>
{
    /// <summary>Input tensor [N, C, ...].</summary>
    public Tensor<T> Input { get; }

    /// <summary>Number of groups to divide channels into.</summary>
    public int NumGroups { get; }

    /// <summary>Scale parameter (gamma). Shape: [C].</summary>
    public Tensor<T> Gamma { get; }

    /// <summary>Bias parameter (beta). Shape: [C].</summary>
    public Tensor<T> Beta { get; }

    /// <summary>Small constant for numerical stability.</summary>
    public double Epsilon { get; }

    /// <summary>Activation to fuse with the normalization.</summary>
    public GroupNormActivation Activation { get; }

    /// <summary>Mean from forward — needed by backward.</summary>
    public Tensor<T>? Mean { get; set; }

    /// <summary>Variance from forward — needed by backward.</summary>
    public Tensor<T>? Variance { get; set; }

    /// <summary>
    /// Pre-activation tensor (= GroupNorm output, before the fused activation)
    /// from the last forward pass. Needed by SiLU backward since SwishBackward
    /// requires the pre-activation — for Identity/ReLU this is not used.
    /// </summary>
    public Tensor<T>? PreActivation { get; set; }

    public FusedGroupNormActivationOp(
        Tensor<T> input,
        int numGroups,
        Tensor<T> gamma,
        Tensor<T> beta,
        double epsilon = 1e-5,
        GroupNormActivation activation = GroupNormActivation.SiLU)
    {
        Input = input ?? throw new ArgumentNullException(nameof(input));
        Gamma = gamma ?? throw new ArgumentNullException(nameof(gamma));
        Beta = beta ?? throw new ArgumentNullException(nameof(beta));
        if (numGroups <= 0)
            throw new ArgumentOutOfRangeException(nameof(numGroups), "Must be positive.");
        NumGroups = numGroups;
        Epsilon = epsilon;
        Activation = activation;
    }

    // ── ICompiledOp<T> ─────────────────────────────────────────────────

    public OpType OpType => OpType.GroupNorm; // Reuses GroupNorm slot (fused is a superset)
    public string OpName => "FusedGroupNormActivation";
    public Tensor<T>[] Inputs => new[] { Input, Gamma, Beta };
    public int[] OutputShape => (int[])Input._shape.Clone();

    public Action<IEngine, Tensor<T>> BuildForwardClosure()
    {
        // Legacy closure with no savedState backfill. Callers needing backward
        // should go through ToCompiledStep which shares the savedState array
        // with the closure.
        return BuildForwardClosureCore(savedState: null);
    }

    /// <summary>Core forward-closure factory shared by <see cref="BuildForwardClosure"/>
    /// and <see cref="ToCompiledStep"/>. When <paramref name="savedState"/> is non-null,
    /// the closure backfills slots [1], [2], and [5] with fresh mean/variance/pre-activation
    /// so backward reads up-to-date tensors.</summary>
    private Action<IEngine, Tensor<T>> BuildForwardClosureCore(object[]? savedState)
    {
        var input = Input;
        var numGroups = NumGroups;
        var gamma = Gamma;
        var beta = Beta;
        var epsilon = Epsilon;
        var activation = Activation;
        var op = this;

        return (eng, output) =>
        {
            Tensor<T>? preActivation = null;
            Tensor<T>? mean = null;
            Tensor<T>? variance = null;
            Tensor<T> tmpMean, tmpVariance;

            switch (activation)
            {
                case GroupNormActivation.SiLU:
                    // Fused kernel writes post-SiLU directly to output — no
                    // GroupNorm work needed for the forward value itself.
                    // Only pay the extra GroupNorm cost (to capture
                    // mean/variance + pre-activation for backward) when
                    // savedState is non-null, i.e. the compiled-step path
                    // that actually wires backward. Pure inference (savedState
                    // null) skips the second pass entirely.
                    eng.GroupNormSwishInto(output, input, numGroups, gamma, beta, epsilon);
                    if (savedState is not null)
                    {
                        preActivation = eng.GroupNorm(input, numGroups, gamma, beta, epsilon,
                            out tmpMean, out tmpVariance);
                        mean = tmpMean;
                        variance = tmpVariance;
                    }
                    break;

                case GroupNormActivation.ReLU:
                    // GroupNorm is required for the forward value here (no
                    // fused ReLU-into kernel), so the work is unavoidable;
                    // the mean/variance outs come along for free.
                    var gnResult = eng.GroupNorm(input, numGroups, gamma, beta, epsilon,
                        out tmpMean, out tmpVariance);
                    var reluResult = eng.ReLU(gnResult);
                    reluResult.AsSpan().CopyTo(output.AsWritableSpan());
                    mean = tmpMean;
                    variance = tmpVariance;
                    break;

                default: // Identity
                    var gnId = eng.GroupNorm(input, numGroups, gamma, beta, epsilon,
                        out tmpMean, out tmpVariance);
                    gnId.AsSpan().CopyTo(output.AsWritableSpan());
                    mean = tmpMean;
                    variance = tmpVariance;
                    break;
            }

            op.Mean = mean;
            op.Variance = variance;
            op.PreActivation = preActivation;

            if (savedState is not null)
            {
                if (mean is not null) savedState[1] = mean;
                if (variance is not null) savedState[2] = variance;
                if (preActivation is not null) savedState[5] = preActivation;
            }
        };
    }

    public BackwardFunction<T>? GetBackwardFunction()
    {
        // Backward: chain rule through activation then GroupNorm.
        return FusedGroupNormActivationBackward;
    }

    public object[]? BuildSavedState()
    {
        // [numGroups, mean, variance, epsilon, activation, preActivation(SiLU only)]
        // Mean/Variance/PreActivation slots start null at build time — the
        // forward closure built by ToCompiledStep backfills them before
        // backward reads. Leaving PreActivation null for Identity/ReLU is
        // safe because the backward switch never dereferences it for those.
        var savedState = new object[6];
        savedState[0] = NumGroups;
        if (Mean is not null) savedState[1] = Mean;
        if (Variance is not null) savedState[2] = Variance;
        savedState[3] = Epsilon;
        savedState[4] = (int)Activation;
        if (PreActivation is not null) savedState[5] = PreActivation;
        return savedState;
    }

    public CompiledStep<T> ToCompiledStep(Tensor<T> outputBuffer)
    {
        // Share one savedState array between the forward closure (which writes
        // mean/variance/preActivation into it every forward) and the CompiledStep
        // (which hands it to backward). Same pattern as GroupNormOp.
        var sharedSavedState = BuildSavedState();
        if (sharedSavedState is null)
            throw new InvalidOperationException(
                "FusedGroupNormActivationOp.BuildSavedState returned null; backward cannot be wired.");

        return new CompiledStep<T>(
            OpName,
            BuildForwardClosureCore(sharedSavedState),
            outputBuffer,
            Inputs,
            GetBackwardFunction(),
            sharedSavedState);
    }

    // ── Factory ─────────────────────────────────────────────────────────

    internal static FusedGroupNormActivationOp<T>? TryFromStep(CompiledStep<T> step)
    {
        if (step.OpName != "FusedGroupNormActivation") return null;
        if (step.Inputs.Length < 3) return null;

        var state = step.SavedState;
        if (state is null || state.Length < 5) return null;

        int numGroups = state[0] is int ng ? ng : 0;
        double epsilon = state[3] is double e ? e : 1e-5;

        // Reject unknown / version-skewed activation codes rather than
        // silently falling through to Identity at forward time. Note:
        // GroupNormActivation is byte-backed so the int stored via
        // BuildSavedState must fit in a byte AND be a defined member.
        if (state[4] is not int activationCode ||
            activationCode < 0 || activationCode > byte.MaxValue ||
            !Enum.IsDefined(typeof(GroupNormActivation), (byte)activationCode))
            return null;
        var activation = (GroupNormActivation)activationCode;

        if (numGroups <= 0) return null;

        return new FusedGroupNormActivationOp<T>(
            step.Inputs[0], numGroups, step.Inputs[1], step.Inputs[2], epsilon, activation);
    }

    // ── Backward ────────────────────────────────────────────────────────

    private static void FusedGroupNormActivationBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var numGroups = (int)savedState[0];
        var mean = (Tensor<T>)savedState[1];
        var variance = (Tensor<T>)savedState[2];
        var epsilon = (double)savedState[3];
        var activation = (GroupNormActivation)(int)savedState[4];

        // Step 1: undo activation gradient.
        //
        // SiLU's derivative σ(h) + h·σ(h)·(1-σ(h)) depends on the PRE-activation
        // h = GroupNorm(x), not the raw input x. The previous code passed
        // inputs[0] (raw input) to SwishBackward — that produced d/dy[SiLU(y)]
        // evaluated at x, which is the wrong gradient. savedState[5] holds the
        // cached pre-activation from the forward pass (see BuildForwardClosureCore).
        //
        // ReLU's derivative (1 iff input > 0) matches 1 iff output > 0, so
        // ReluBackward can use `output` — no pre-activation needed.
        //
        // Identity: no activation, pass gradOutput through.
        var effectiveGrad = activation switch
        {
            GroupNormActivation.SiLU => engine.SwishBackward(
                gradOutput,
                savedState.Length > 5 && savedState[5] is Tensor<T> preAct
                    ? preAct
                    : throw new InvalidOperationException(
                        "FusedGroupNormActivation SiLU backward requires cached pre-activation; " +
                        "step was not built via ToCompiledStep.")),
            GroupNormActivation.ReLU => engine.ReluBackward(gradOutput, output),
            _ => gradOutput,
        };

        // Step 2: GroupNorm backward
        var gradInput = engine.GroupNormBackward(
            effectiveGrad, inputs[0], numGroups, inputs[1], mean, variance, epsilon,
            out var gradGamma, out var gradBeta);

        DifferentiableOps.AccumulateGrad(grads, inputs[0], gradInput, engine);
        DifferentiableOps.AccumulateGrad(grads, inputs[1], gradGamma, engine);
        DifferentiableOps.AccumulateGrad(grads, inputs[2], gradBeta, engine);
    }
}
