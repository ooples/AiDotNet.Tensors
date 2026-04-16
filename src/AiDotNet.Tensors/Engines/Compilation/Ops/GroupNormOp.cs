using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Compilation.Ops;

/// <summary>
/// IR operation for Group Normalization. Stores the op's full metadata
/// (numGroups, epsilon, scale/gamma, bias/beta) so fusion passes can
/// introspect attributes without type-unsafe SavedState casts.
///
/// <para><b>Forward:</b> delegates to <see cref="IEngine.GroupNorm{T}"/>
/// — no new kernel, just an IR-level representation that the JIT compiler
/// can reason about (fuse with SiLU, reorder with residual-add, etc.).</para>
///
/// <para><b>Backward:</b> uses <see cref="BackwardFunctions{T}.GroupNormBackward"/>
/// which calls <see cref="IEngine.GroupNormBackward{T}"/> for the
/// <c>numGroups != 1</c> general case.</para>
///
/// <para><b>SavedState ordering:</b> <c>[numGroups, mean, variance, epsilon]</c>
/// — matches <see cref="BackwardFunctions{T}.GroupNormBackward"/>'s read order
/// and the <see cref="DifferentiableOps.RecordIfActive"/> recording path.
/// Note: the GraphMode recording path in CpuEngine uses a different order
/// <c>[mean, variance, numGroups, epsilon]</c> — that's a pre-existing
/// divergence documented in issue #178.</para>
/// </summary>
/// <typeparam name="T">The tensor element type.</typeparam>
internal sealed class GroupNormOp<T> : ICompiledOp<T>
{
    /// <summary>Number of groups to divide channels into.</summary>
    public int NumGroups { get; }

    /// <summary>Small constant for numerical stability in the variance denominator.</summary>
    public double Epsilon { get; }

    /// <summary>The input tensor (captured at trace time).</summary>
    public Tensor<T> Input { get; }

    /// <summary>Scale parameter (gamma). Shape: [C] where C is the channel count.</summary>
    public Tensor<T> Gamma { get; }

    /// <summary>Bias parameter (beta). Shape: [C].</summary>
    public Tensor<T> Beta { get; }

    /// <summary>
    /// Mean tensor computed during forward — needed by backward.
    /// Set after the first forward execution; null before that.
    /// </summary>
    public Tensor<T>? Mean { get; set; }

    /// <summary>
    /// Variance tensor computed during forward — needed by backward.
    /// Set after the first forward execution; null before that.
    /// </summary>
    public Tensor<T>? Variance { get; set; }

    public GroupNormOp(Tensor<T> input, int numGroups, Tensor<T> gamma, Tensor<T> beta, double epsilon = 1e-5)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (gamma is null) throw new ArgumentNullException(nameof(gamma));
        if (beta is null) throw new ArgumentNullException(nameof(beta));
        if (numGroups <= 0)
            throw new ArgumentOutOfRangeException(nameof(numGroups), "Number of groups must be positive.");

        Input = input;
        NumGroups = numGroups;
        Gamma = gamma;
        Beta = beta;
        Epsilon = epsilon;
    }

    // ── ICompiledOp<T> ─────────────────────────────────────────────────

    public OpType OpType => OpType.GroupNorm;
    public string OpName => "GroupNorm";
    public Tensor<T>[] Inputs => new[] { Input, Gamma, Beta };
    public int[] OutputShape => (int[])Input._shape.Clone();

    public Action<IEngine, Tensor<T>> BuildForwardClosure()
    {
        // Capture by value so the closure doesn't hold `this` alive.
        var input = Input;
        var numGroups = NumGroups;
        var gamma = Gamma;
        var beta = Beta;
        var epsilon = Epsilon;
        var op = this; // For writing back mean/variance

        return (eng, output) =>
        {
            var result = eng.GroupNorm(input, numGroups, gamma, beta, epsilon, out var mean, out var variance);
            result.AsSpan().CopyTo(output.AsWritableSpan());

            // Store mean/variance for backward pass.
            op.Mean = mean;
            op.Variance = variance;
        };
    }

    public BackwardFunction<T>? GetBackwardFunction()
        => BackwardFunctions<T>.GroupNormBackward;

    public object[]? BuildSavedState()
    {
        // Matches BackwardFunctions<T>.GroupNormBackward's read order:
        // [0] = numGroups (int), [1] = mean (Tensor), [2] = variance (Tensor), [3] = epsilon (double)
        return new object[] { NumGroups, Mean!, Variance!, Epsilon };
    }

    public CompiledStep<T> ToCompiledStep(Tensor<T> outputBuffer)
    {
        return new CompiledStep<T>(
            OpName,
            BuildForwardClosure(),
            outputBuffer,
            Inputs,
            GetBackwardFunction(),
            BuildSavedState());
    }

    // ── Factory: try to extract from an existing CompiledStep ──────────

    /// <summary>
    /// Attempts to extract a <see cref="GroupNormOp{T}"/> from an existing
    /// <see cref="CompiledStep{T}"/>. Returns null if the step isn't a
    /// GroupNorm op or its SavedState can't be parsed. Used by fusion passes
    /// that need typed attribute access.
    /// </summary>
    internal static GroupNormOp<T>? TryFromStep(CompiledStep<T> step)
    {
        if (step.OpType != OpType.GroupNorm) return null;
        if (step.Inputs.Length < 3) return null;

        var savedState = step.SavedState;
        if (savedState is null || savedState.Length < 4) return null;

        // Handle BOTH SavedState orderings (see class xmldoc):
        // DifferentiableOps path: [numGroups, mean, variance, epsilon]
        // GraphMode path (buggy): [mean, variance, numGroups, epsilon]
        int numGroups;
        double epsilon;

        if (savedState[0] is int ng)
        {
            // DifferentiableOps ordering (correct)
            numGroups = ng;
            epsilon = savedState[3] is double e ? e : 1e-5;
        }
        else if (savedState[2] is int ng2)
        {
            // GraphMode ordering (pre-existing divergence)
            numGroups = ng2;
            epsilon = savedState[3] is double e ? e : 1e-5;
        }
        else
        {
            return null; // Unrecognized format
        }

        return new GroupNormOp<T>(step.Inputs[0], numGroups, step.Inputs[1], step.Inputs[2], epsilon);
    }
}
