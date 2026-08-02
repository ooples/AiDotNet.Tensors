// Copyright (c) AiDotNet. All rights reserved.
// A complete, target-independent description of one conv-class kernel:
// an iteration space, tensor bindings with index maps, and a reduce+epilogue body.
//
// This is the unit an emitter consumes. It carries a reference interpreter so a
// spec's semantics can be validated on the CPU -- against the same fp64 oracle the
// hand-written kernels are tested against -- BEFORE any backend exists and without
// needing an idle GPU.

using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

namespace AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;

/// <summary>How the reduction axes are combined.</summary>
public enum CodegenReduceKind
{
    /// <summary>No reduction axes; the body is a pure gather/copy.</summary>
    None,

    /// <summary>Sum of the operand product over the reduction axes.</summary>
    Sum,

    /// <summary>Maximum of the operand product over the reduction axes.</summary>
    Max
}

/// <summary>
/// Elementwise transform applied INSIDE the reduction, to each term before it is combined.
/// </summary>
/// <remarks>
/// The activation is an epilogue: it runs once, on the finished accumulator. Softmax's
/// denominator needs the opposite -- <c>sum(exp(x - max))</c> applies exp to every term
/// BEFORE summing -- and LayerNorm's variance needs a square in the same position. Neither
/// is expressible by sequencing two kernels, because the missing piece is the body's shape
/// rather than the number of passes.
/// </remarks>
public enum CodegenPreReduceOp
{
    /// <summary>Reduce the operand product directly.</summary>
    None,

    /// <summary><c>exp(t)</c> -- softmax's denominator.</summary>
    Exp,

    /// <summary><c>t * t</c> -- LayerNorm's variance, and any sum of squares.</summary>
    Square
}

/// <summary>What an extra output holds.</summary>
public enum CodegenExtraOutputKind
{
    /// <summary>
    /// The position of the winning term of a Max reduction, as an integer.
    /// </summary>
    /// <remarks>
    /// A max-pool's indices buffer. The backward pass routes every gradient by it.
    /// </remarks>
    ArgMaxIndex,

    /// <summary>
    /// <c>Scale * primary + BiasScale * bias</c>, as a float.
    /// </summary>
    /// <remarks>
    /// An optimizer state update: the primary output is the new state, and a parameter is
    /// the old value stepped by it. SGD-with-momentum is exactly this shape --
    /// <c>v' = mu*v + g</c> as the primary, <c>p' = p - lr*v'</c> as the extra.
    ///
    /// Worth knowing before reaching for it: PR #874 measured a hand-written fused SGD
    /// against the EXISTING AiDotNet kernel and found a tie (0.73x-1.05x), because that
    /// kernel is already single-pass. The measured headroom in this family is Adam and
    /// AdamW, which the existing kernels do not fuse -- and those need a third state and a
    /// reciprocal square root, not just this.
    /// </remarks>
    AffineOfPrimary
}

/// <summary>One output beyond the first, written from the same iteration point.</summary>
/// <param name="Binding">Where it is written; addressed by parallel axes.</param>
/// <param name="Kind">What it holds.</param>
/// <param name="IndexExpr">For <see cref="CodegenExtraOutputKind.ArgMaxIndex"/>, the
/// expression evaluated at the winning term.</param>
/// <param name="Scale">For <see cref="CodegenExtraOutputKind.AffineOfPrimary"/>, the
/// multiplier on the primary result.</param>
/// <param name="BiasInput">Optional input added after that multiplier.</param>
/// <param name="BiasScale">Multiplier on that input.</param>
public sealed record CodegenExtraOutput(
    CodegenTensorBinding Binding,
    CodegenExtraOutputKind Kind,
    CodegenAffineExpr? IndexExpr = null,
    double Scale = 1.0,
    int? BiasInput = null,
    double BiasScale = 1.0);

/// <summary>Epilogue activation applied after bias and scale.</summary>
public enum CodegenActivationKind
{
    /// <summary>No activation.</summary>
    None,

    /// <summary><c>max(x, 0)</c>.</summary>
    ReLU,

    /// <summary><c>1 / (1 + exp(-x))</c>.</summary>
    Sigmoid,

    /// <summary>Hyperbolic tangent.</summary>
    Tanh,

    /// <summary>SiLU / swish: <c>x * sigmoid(x)</c>.</summary>
    Swish,

    /// <summary><c>1 / x</c>. Turns a summed denominator into the factor to multiply by.</summary>
    Reciprocal,

    /// <summary><c>1 / sqrt(x)</c>. The normalising factor of RMSNorm and Adam.</summary>
    Rsqrt,

    /// <summary>
    /// Gaussian error linear unit, tanh approximation:
    /// <c>0.5x(1 + tanh(sqrt(2/pi)(x + 0.044715x^3)))</c>.
    /// </summary>
    /// <remarks>
    /// The TANH form, not the erf form. They differ by up to about 1e-3 absolute near
    /// |x| = 2, which is far larger than any floating-point concern, so the choice is part
    /// of the operator's definition rather than an implementation detail. Both the emitter
    /// and the fp64 oracle evaluate this same formula, so the comparison stays honest.
    /// </remarks>
    Gelu
}

/// <summary>
/// One kernel: iteration space + tensor bindings + a reduce-and-epilogue body.
/// </summary>
/// <remarks>
/// <para>The body computes, for every point of the parallel iteration space:</para>
/// <code>
///   acc = REDUCE over reduction axes of ( product of ProductInputs )
///   acc = acc + bias            (when BiasInput is set)
///   acc = acc * scale           (when ScaleInput is set)
///   out = activation(acc)
/// </code>
/// <para>
/// That covers direct/depthwise/transposed convolution, im2col-style gathers
/// (no reduction axes), and the fused bias/scale/ReLU epilogues -- i.e. the
/// families where PyTorch must split the work because Inductor cannot fuse
/// through cuDNN. It deliberately does NOT cover data-dependent indexing
/// (deformable convolution's learned offsets); an emitter must reject those
/// rather than mis-lower them.
/// </para>
/// </remarks>
public sealed class CodegenKernelSpec
{
    private readonly CodegenTensorBinding[] _inputs;
    private readonly int[] _productInputs;
    private readonly CodegenExtraOutput[] _extraOutputs;

    /// <summary>Stable kernel name; becomes the emitted entry-point symbol.</summary>
    public string Name { get; }

    /// <summary>The axes this kernel iterates, and the authority on the launch grid.</summary>
    public CodegenIterationSpace Space { get; }

    /// <summary>Tensors the kernel reads.</summary>
    public IReadOnlyList<CodegenTensorBinding> Inputs => _inputs;

    /// <summary>The tensor the kernel writes.</summary>
    public CodegenTensorBinding Output { get; }

    /// <summary>Indices into <see cref="Inputs"/> whose loads are multiplied inside the reduction.</summary>
    public IReadOnlyList<int> ProductInputs => _productInputs;

    /// <summary>How the reduction axes combine.</summary>
    public CodegenReduceKind Reduce { get; }

    /// <summary>
    /// Outputs beyond the first, all written from the same iteration point.
    /// </summary>
    /// <remarks>
    /// The count is not capped at one. A max-pool writes its values and its argmax indices;
    /// an Adam step writes the two moment states and the updated parameter, which is three.
    /// Splitting those across kernels re-reads every operand once per kernel, which is the
    /// specific cost the whole fusion argument is about.
    /// </remarks>
    public IReadOnlyList<CodegenExtraOutput> ExtraOutputs => _extraOutputs;

    /// <summary>Optional index into <see cref="Inputs"/> added after the reduction.</summary>
    public int? BiasInput { get; }

    /// <summary>Optional index into <see cref="Inputs"/> multiplied after the bias.</summary>
    public int? ScaleInput { get; }

    /// <summary>Epilogue activation.</summary>
    public CodegenActivationKind Activation { get; }

    /// <summary>
    /// Constant multiplied into the reduction result, BEFORE bias and scale. 1.0 means none.
    /// </summary>
    /// <remarks>
    /// This is how a mean is expressed: a sum with <c>ReduceScale = 1/count</c>. A separate
    /// <c>Mean</c> reduce kind was the obvious alternative and is the worse one, because the
    /// same constant also serves a loss normalisation and softmax's <c>1/denominator</c> --
    /// one scalar covers three operators where a reduce kind covers one.
    ///
    /// It applies before the bias because a mean-then-add-bias is the operator people mean;
    /// scaling after the bias would scale the bias too. <see cref="ScaleInput"/> is a
    /// TENSOR and stays where it is, after the bias.
    /// </remarks>
    public double ReduceScale { get; }

    /// <summary>
    /// Optional SECOND output, written alongside the first.
    /// </summary>
    /// <remarks>
    /// Addressed by parallel axes exactly like <see cref="Output"/>, and written with the
    /// value of <see cref="SecondaryIndexExpr"/> evaluated AT THE WINNING TERM of a Max
    /// reduction. That is what a max-pool's indices buffer holds, and its backward pass
    /// cannot run without it -- so dispatching a generated max-pool without this would
    /// silently break training rather than merely lose a speedup.
    /// </remarks>
    public CodegenTensorBinding? SecondaryOutput => FirstArgMax?.Binding;

    /// <summary>
    /// Expression whose value at the argmax is written to <see cref="SecondaryOutput"/>.
    /// </summary>
    /// <remarks>
    /// An affine expression over ALL axes, reduction ones included -- it has to be, because
    /// the whole point is which reduction position won. For a max-pool the expression is
    /// the winning input's SPATIAL index, <c>ih * inWidth + iw</c>, because that is the
    /// convention the existing backward kernel reads: it recovers <c>ih = idx / inWidth</c>
    /// and <c>iw = idx % inWidth</c> and adds the batch and channel offset itself. Writing
    /// a flat input offset or a tap index instead would compile and produce plausible
    /// numbers while corrupting every gradient.
    /// </remarks>
    public CodegenAffineExpr? SecondaryIndexExpr => FirstArgMax?.IndexExpr;

    /// <summary>The first argmax extra, which is what the legacy secondary pair named.</summary>
    private CodegenExtraOutput? FirstArgMax
    {
        get
        {
            for (int i = 0; i < _extraOutputs.Length; i++)
                if (_extraOutputs[i].Kind == CodegenExtraOutputKind.ArgMaxIndex)
                    return _extraOutputs[i];
            return null;
        }
    }

    /// <summary>Elementwise transform applied to each term inside the reduction.</summary>
    public CodegenPreReduceOp PreReduce { get; }

    /// <summary>
    /// The number system the arithmetic is carried out in. An element occupies
    /// <see cref="CodegenAlgebraTables.Components"/> adjacent fp32 values.
    /// </summary>
    /// <remarks>
    /// The shapes and index maps are UNCHANGED by this: a tensor of complex numbers has
    /// exactly the shape it says it has, and the components live inside one element. Only
    /// the product and the accumulator widen.
    /// </remarks>
    public CodegenAlgebra Algebra { get; }

    /// <summary>
    /// Optional index into <see cref="Inputs"/> added to each term BEFORE
    /// <see cref="PreReduce"/>.
    /// </summary>
    /// <remarks>
    /// This is what makes the shift in <c>exp(x - max)</c> and the centring in
    /// <c>(x - mean)^2</c> expressible: both are a broadcast add of a per-row statistic
    /// computed by an earlier pass. It is a separate slot from <see cref="BiasInput"/>,
    /// which lands after the reduction.
    /// </remarks>
    public int? PreBiasInput { get; }

    /// <summary>
    /// Constant the pre-bias is multiplied by before being added. −1 makes it a SUBTRACT.
    /// </summary>
    /// <remarks>
    /// The body multiplies its operands, so a difference of two tensors had no expression
    /// at all: <c>(a - b)^2</c>, which is every squared-error loss, was inexpressible even
    /// with the pre-reduction square. Rather than add a second combine mode to the product,
    /// the pre-bias gains a sign -- <c>product + (-1) * b</c> -- which costs one constant
    /// and reuses the slot that already exists for softmax's shift and LayerNorm's
    /// centring.
    /// </remarks>
    public double PreBiasScale { get; }

    /// <summary>Creates a kernel spec and validates its internal consistency.</summary>
    public CodegenKernelSpec(
        string name,
        CodegenIterationSpace space,
        CodegenTensorBinding[] inputs,
        CodegenTensorBinding output,
        int[] productInputs,
        CodegenReduceKind reduce,
        int? biasInput = null,
        int? scaleInput = null,
        CodegenActivationKind activation = CodegenActivationKind.None,
        double reduceScale = 1.0,
        CodegenPreReduceOp preReduce = CodegenPreReduceOp.None,
        int? preBiasInput = null,
        double preBiasScale = 1.0,
        CodegenAlgebra algebra = CodegenAlgebra.Real,
        CodegenExtraOutput[]? extraOutputs = null,
        CodegenTensorBinding? secondaryOutput = null,
        CodegenAffineExpr? secondaryIndexExpr = null)
    {
        if (string.IsNullOrWhiteSpace(name)) throw new ArgumentException("Kernel needs a name.", nameof(name));
        Name = name;
        Space = space ?? throw new ArgumentNullException(nameof(space));
        _inputs = inputs ?? throw new ArgumentNullException(nameof(inputs));
        Output = output ?? throw new ArgumentNullException(nameof(output));
        _productInputs = productInputs ?? throw new ArgumentNullException(nameof(productInputs));
        Reduce = reduce;
        BiasInput = biasInput;
        ScaleInput = scaleInput;
        Activation = activation;
        ReduceScale = reduceScale;
        PreReduce = preReduce;
        PreBiasInput = preBiasInput;
        PreBiasScale = preBiasScale;
        Algebra = algebra;

        // ONE MECHANISM. The legacy secondaryOutput/secondaryIndexExpr pair is folded into
        // the extras list rather than kept beside it: two ways to express "an extra output"
        // is how one of them ends up unmaintained, and the emitter would have to walk both.
        var extras = new List<CodegenExtraOutput>();
        if (secondaryOutput is not null)
            extras.Add(new CodegenExtraOutput(
                secondaryOutput, CodegenExtraOutputKind.ArgMaxIndex, secondaryIndexExpr));
        if (extraOutputs is not null) extras.AddRange(extraOutputs);
        _extraOutputs = extras.ToArray();

        // Parameter order is the launch ABI. A binding whose ParameterIndex disagrees
        // with its position makes the emitter load a different pointer while the
        // interpreter continues to use the array position, so both can look internally
        // consistent and still compute different operators.
        for (int i = 0; i < _inputs.Length; i++)
            if (_inputs[i].ParameterIndex != i)
                throw new ArgumentException(
                    "Input '" + _inputs[i].Name + "' is at position " + i +
                    " but binds parameter " + _inputs[i].ParameterIndex + ".", nameof(inputs));
        if (output.ParameterIndex != _inputs.Length)
            throw new ArgumentException(
                "Output '" + output.Name + "' must bind parameter " + _inputs.Length +
                ", immediately after the inputs; got " + output.ParameterIndex + ".",
                nameof(output));

        // WHAT A NON-REAL ALGEBRA CANNOT COMBINE WITH. Each of these is refused because it
        // is not defined on the number system, not because it is unimplemented -- and the
        // difference matters: an approximation here would produce a kernel that runs and
        // returns something that is not the operator's value.
        if (algebra != CodegenAlgebra.Real)
        {
            if (reduce == CodegenReduceKind.Max)
                throw new ArgumentException(
                    "There is no order on " + algebra + ", so a Max reduction over it is " +
                    "undefined. Reduce the magnitudes explicitly if that is what is wanted.",
                    nameof(reduce));

            if (secondaryOutput is not null)
                throw new ArgumentException(
                    "An argmax secondary output requires an ordered reduction, which " +
                    algebra + " does not have.", nameof(secondaryOutput));

            if (_extraOutputs.Length > 0)
                throw new ArgumentException(
                    "Extra outputs over " + algebra + " are not defined here: an argmax needs " +
                    "an order the algebra does not have, and an affine-of-primary would have " +
                    "to say which components it scales. Write it as its own kernel.");

            if (activation != CodegenActivationKind.None)
                throw new ArgumentException(
                    "Activation " + activation + " is a real function. Applying it " +
                    "component-wise to a " + algebra + " value is a DIFFERENT operator, not " +
                    "the same one generalised, so it must be written as one.",
                    nameof(activation));

            if (preReduce != CodegenPreReduceOp.None)
                throw new ArgumentException(
                    "Pre-reduction " + preReduce + " is a real transform; over " + algebra +
                    " it is a different operator.", nameof(preReduce));

            for (int i = 0; i < _inputs.Length; i++)
                if (_inputs[i].ElementType != CodegenElementType.Float32)
                    throw new ArgumentException(
                        "A " + algebra + " kernel stores its components as fp32; input '" +
                        _inputs[i].Name + "' is " + _inputs[i].ElementType + ".");

            if (output.ElementType != CodegenElementType.Float32)
                throw new ArgumentException(
                    "A " + algebra + " kernel stores its components as fp32; the output is " +
                    output.ElementType + ".");
        }

        if ((secondaryOutput is null) != (secondaryIndexExpr is null))
            throw new ArgumentException(
                "A secondary output needs an index expression and vice versa; one without " +
                "the other would write an undefined value.", nameof(secondaryOutput));

        var claimed = new HashSet<int> { output.ParameterIndex };
        for (int i = 0; i < _extraOutputs.Length; i++)
        {
            var extra = _extraOutputs[i];

            if (!extra.Binding.IsOutput)
                throw new ArgumentException(
                    "Extra output '" + extra.Binding.Name + "' must be marked IsOutput.");

            // TWO OUTPUTS ON ONE PARAMETER IS A SILENT DATA RACE. Both stores would land on
            // the same buffer with no ordering between them, so the result depends on warp
            // scheduling -- it produces plausible values and a different answer per run.
            if (!claimed.Add(extra.Binding.ParameterIndex))
                throw new ArgumentException(
                    "Extra output '" + extra.Binding.Name + "' binds parameter " +
                    extra.Binding.ParameterIndex + ", which another output already writes. " +
                    "Two outputs on one buffer race with no ordering between them.");

            int expectedParameter = _inputs.Length + 1 + i;
            if (extra.Binding.ParameterIndex != expectedParameter)
                throw new ArgumentException(
                    "Extra output '" + extra.Binding.Name + "' is at output position " +
                    (i + 1) + " but binds parameter " + extra.Binding.ParameterIndex +
                    "; expected " + expectedParameter + ".");

            if (extra.Kind == CodegenExtraOutputKind.ArgMaxIndex)
            {
                if (extra.IndexExpr is null)
                    throw new ArgumentException(
                        "Argmax extra output '" + extra.Binding.Name + "' needs an index " +
                        "expression; without one it would write an undefined value.");

                if (reduce != CodegenReduceKind.Max)
                    throw new ArgumentException(
                        "Extra output '" + extra.Binding.Name + "' is an ARGMAX position, so " +
                        "it requires a Max reduction; got " + reduce + ".");
            }
            else if (extra.BiasInput.HasValue)
            {
                RefuseIndexOperand(extra.BiasInput.Value, "the bias of extra output '" +
                    extra.Binding.Name + "'");
            }

            ValidateIndirection(extra.Binding, "extra output '" + extra.Binding.Name + "'");
        }

        // EVERY INDIRECTION MUST POINT AT A REAL INDEX TENSOR. If it pointed at a float
        // operand the emitter would reinterpret that operand's bit pattern as an integer and
        // address with it -- which does not fault, does not look wrong in the PTX, and
        // produces garbage that varies with the input data.
        // AN INDEX TENSOR IS NEVER AN ARITHMETIC OPERAND. This is checked here, on the spec,
        // rather than only at the load site: the emitter has several load paths -- scalar,
        // vectorised, staged, coarsened -- and a guard on one of them is a guard on none.
        // Reading an int32 buffer as fp32 does not fault and produces arithmetic out of a bit
        // pattern, so it must be impossible to construct rather than merely unlikely.
        for (int i = 0; i < _productInputs.Length; i++) RefuseIndexOperand(_productInputs[i], "a product operand");
        if (biasInput.HasValue) RefuseIndexOperand(biasInput.Value, "the bias");
        if (scaleInput.HasValue) RefuseIndexOperand(scaleInput.Value, "the scale");
        if (preBiasInput.HasValue) RefuseIndexOperand(preBiasInput.Value, "the pre-bias");

        ValidateIndirection(output, "output");
        for (int i = 0; i < _inputs.Length; i++) ValidateIndirection(_inputs[i], "input " + i);
        if (secondaryOutput is not null) ValidateIndirection(secondaryOutput, "secondary output");

        ValidateAxes(output, "output");
        for (int i = 0; i < _inputs.Length; i++) ValidateAxes(_inputs[i], "input " + i);
        for (int i = 0; i < _extraOutputs.Length; i++)
        {
            ValidateAxes(_extraOutputs[i].Binding, "extra output " + i);
            if (_extraOutputs[i].IndexExpr is not null)
                ValidateExpressionAxes(_extraOutputs[i].IndexExpr!,
                    "extra output " + i + " index expression");
        }

        if (double.IsNaN(preBiasScale) || double.IsInfinity(preBiasScale))
            throw new ArgumentException(
                "PreBiasScale must be finite; got " + preBiasScale + ".", nameof(preBiasScale));

        if (double.IsNaN(reduceScale) || double.IsInfinity(reduceScale))
            throw new ArgumentException(
                "ReduceScale must be finite; got " + reduceScale + ".", nameof(reduceScale));

        if (_productInputs.Length == 0)
            throw new ArgumentException("At least one operand must feed the body.", nameof(productInputs));
        foreach (int i in _productInputs) Require(i, nameof(productInputs));
        if (biasInput.HasValue) Require(biasInput.Value, nameof(biasInput));
        if (scaleInput.HasValue) Require(scaleInput.Value, nameof(scaleInput));
        if (preBiasInput.HasValue) Require(preBiasInput.Value, nameof(preBiasInput));
        if (!output.IsOutput)
            throw new ArgumentException("Output binding must be marked IsOutput.", nameof(output));

        bool hasReductionAxis = space.ReductionAxes.Length > 0;
        if (hasReductionAxis && reduce == CodegenReduceKind.None)
            throw new ArgumentException("Reduction axes declared but reduce kind is None.", nameof(reduce));
        if (!hasReductionAxis && reduce != CodegenReduceKind.None)
            throw new ArgumentException("Reduce kind declared but no reduction axis exists.", nameof(reduce));

        // The output must be addressable from parallel axes only: a store that
        // depended on a reduction axis would be written once per loop trip.
        var reductionSet = new HashSet<int>(space.ReductionAxes);
        for (int d = 0; d < output.Map.Count; d++)
            foreach (var term in output.Map[d].Terms)
                if (reductionSet.Contains(term.Axis))
                    throw new ArgumentException(
                        $"Output dimension {d} depends on reduction axis '{space.Axes[term.Axis].Name}'.", nameof(output));

        void ValidateAxes(CodegenTensorBinding binding, string role)
        {
            for (int d = 0; d < binding.Map.Count; d++)
                ValidateExpressionAxes(binding.Map[d],
                    role + " '" + binding.Name + "' dimension " + d);
        }

        void ValidateExpressionAxes(CodegenAffineExpr expression, string role)
        {
            foreach (var term in expression.Terms)
                if (term.Axis < 0 || term.Axis >= space.Axes.Count)
                    throw new ArgumentException(
                        role + " references affine axis " + term.Axis + " but the iteration " +
                        "space has " + space.Axes.Count + " axes.");
        }
    }

    /// <summary>
    /// The reference definition of every activation, in fp64.
    /// </summary>
    /// <remarks>
    /// This is the oracle the emitted PTX is measured against, so the formulas here ARE
    /// the operator's definition. GELU uses the tanh approximation; swapping it for the
    /// erf form would move results by up to about 1e-3 near |x| = 2, which is a different
    /// operator rather than a rounding difference.
    /// </remarks>
    public static double ApplyActivation(CodegenActivationKind kind, double x) => kind switch
    {
        CodegenActivationKind.None => x,
        CodegenActivationKind.ReLU => x < 0.0 ? 0.0 : x,
        CodegenActivationKind.Sigmoid => 1.0 / (1.0 + Math.Exp(-x)),
        CodegenActivationKind.Tanh => Math.Tanh(x),
        CodegenActivationKind.Swish => x / (1.0 + Math.Exp(-x)),
        CodegenActivationKind.Reciprocal => 1.0 / x,
        CodegenActivationKind.Rsqrt => 1.0 / Math.Sqrt(x),
        CodegenActivationKind.Gelu =>
            0.5 * x * (1.0 + Math.Tanh(0.7978845608028654 * (x + 0.044715 * x * x * x))),
        _ => throw new NotSupportedException("Unhandled activation " + kind + "."),
    };

    private void Require(int inputIndex, string paramName)
    {
        if ((uint)inputIndex >= (uint)_inputs.Length)
            throw new ArgumentOutOfRangeException(paramName, $"Input index {inputIndex} is outside [0, {_inputs.Length}).");
    }

    /// <summary>Number of kernel pointer parameters (inputs then output).</summary>
    public int ParameterCount => _inputs.Length + 1 + _extraOutputs.Length;

    /// <summary>
    /// CPU reference execution of the spec, in fp64.
    /// </summary>
    /// <remarks>
    /// This is the semantic definition of the spec. An emitter is correct when its
    /// device output matches this within tolerance, and this can be checked with no
    /// GPU at all -- which is what lets the C#-vs-Rust bake-off proceed while the
    /// device is busy.
    /// </remarks>
    /// <param name="inputData">Buffers for each input binding, in binding order.</param>
    /// <returns>The output buffer, row-major, of <c>Output.ElementCount</c> elements.</returns>
    /// <summary>
    /// CPU reference execution for a complex or quaternion kernel.
    /// </summary>
    /// <remarks>
    /// Written as its own walk rather than folded into the real one for the same reason the
    /// emitter keeps its paths apart: the real interpreter's value is a scalar threaded
    /// through max-tracking and argmax bookkeeping that a non-real algebra refuses at
    /// construction anyway. Both walks decompose the iteration space identically, which is
    /// what the comparison actually depends on.
    /// </remarks>
    private double[] InterpretAlgebraic(IReadOnlyList<double[]> inputData, out double[]? secondary)
    {
        secondary = null;
        int components = Algebra.Components();

        var axes = Space.Axes;
        int[] parallel = Space.ParallelAxes;
        int[] reduction = Space.ReductionAxes;
        var values = new int[axes.Count];
        var output = new double[Output.ElementCount * components];

        var accumulator = new double[components];
        var operand = new double[components];
        var product = new double[components];
        var scratch = new double[components];

        long threads = Space.TotalThreads;
        for (long tid = 0; tid < threads; tid++)
        {
            long rest = tid;
            for (int p = parallel.Length - 1; p >= 0; p--)
            {
                int extent = axes[parallel[p]].Extent;
                values[parallel[p]] = (int)(rest % extent);
                rest /= extent;
            }
            for (int r = 0; r < reduction.Length; r++) values[reduction[r]] = 0;

            for (int c = 0; c < components; c++) accumulator[c] = 0.0;

            long trips = Space.ReductionTripCount;
            for (long t = 0; t < trips; t++)
            {
                long rrest = t;
                for (int r = reduction.Length - 1; r >= 0; r--)
                {
                    int extent = axes[reduction[r]].Extent;
                    values[reduction[r]] = (int)(rrest % extent);
                    rrest /= extent;
                }

                bool outOfBounds = false;
                for (int c = 0; c < components; c++) product[c] = 0.0;

                for (int k = 0; k < _productInputs.Length; k++)
                {
                    var binding = _inputs[_productInputs[k]];
                    long off = binding.ResolveOffset(values, inputData, out bool ok);
                    if (!ok) { outOfBounds = true; break; }

                    var buffer = inputData[_productInputs[k]];
                    for (int c = 0; c < components; c++) operand[c] = buffer[off * components + c];

                    if (k == 0) Array.Copy(operand, product, components);
                    else
                    {
                        // Left-folded, and the ORDER MATTERS: a quaternion product does not
                        // commute, so this must walk ProductInputs in the same direction the
                        // emitter does.
                        Algebra.Multiply(product, operand, scratch);
                        Array.Copy(scratch, product, components);
                    }
                }

                // Zero is the additive identity in all of these algebras, so an out-of-range
                // operand contributes nothing -- the same argument that lets the emitter skip
                // a select, and the reason Max is refused rather than approximated.
                if (outOfBounds) continue;

                for (int c = 0; c < components; c++) accumulator[c] += product[c];
            }

            if (ReduceScale != 1.0)
                for (int c = 0; c < components; c++) accumulator[c] *= ReduceScale;

            if (BiasInput.HasValue)
            {
                var bias = _inputs[BiasInput.Value];
                long off = bias.ResolveOffset(values, inputData, out bool ok);
                if (ok)
                {
                    var buffer = inputData[BiasInput.Value];
                    for (int c = 0; c < components; c++) accumulator[c] += buffer[off * components + c];
                }
            }

            if (ScaleInput.HasValue)
            {
                var scale = _inputs[ScaleInput.Value];
                long off = scale.ResolveOffset(values, inputData, out bool ok);
                if (ok)
                {
                    var buffer = inputData[ScaleInput.Value];
                    for (int c = 0; c < components; c++) operand[c] = buffer[off * components + c];

                    // A full algebra multiply, not a component-wise one: scaling a quaternion
                    // by a quaternion is a rotation.
                    Algebra.Multiply(accumulator, operand, scratch);
                    Array.Copy(scratch, accumulator, components);
                }
            }

            long outOff = Output.ResolveOffset(values, inputData, out bool outOk);
            if (!outOk) continue;

            for (int c = 0; c < components; c++)
            {
                if (Output.NeedsAtomicStore) output[outOff * components + c] += accumulator[c];
                else output[outOff * components + c] = accumulator[c];
            }
        }

        return output;
    }

    /// <summary>Refuses an int32 index tensor used where a value operand is expected.</summary>
    private void RefuseIndexOperand(int input, string role)
    {
        if (input < 0 || input >= _inputs.Length)
            throw new ArgumentException(
                $"Input {input} used as {role}, but the spec has {_inputs.Length} inputs.");

        if (_inputs[input].IsIndexTensor)
            throw new ArgumentException(
                $"'{_inputs[input].Name}' is an int32 index tensor and cannot be {role}. " +
                "Reading it as a value would compute with a bit pattern, which neither " +
                "faults nor looks wrong in the generated PTX.");
    }

    /// <summary>
    /// Checks that every data-dependent index on a binding refers to an int32 index tensor
    /// that actually exists.
    /// </summary>
    private void ValidateIndirection(CodegenTensorBinding binding, string role)
    {
        if (!binding.HasIndirection) return;

        for (int d = 0; d < binding.Indirect.Count; d++)
        {
            var indirect = binding.Indirect[d];
            if (indirect is null) continue;

            if (indirect.IndexInput >= _inputs.Length)
                throw new ArgumentException(
                    $"The {role} binding '{binding.Name}' takes dimension {d} from input " +
                    $"{indirect.IndexInput}, but the spec has only {_inputs.Length} inputs.");

            var source = _inputs[indirect.IndexInput];
            if (!source.IsIndexTensor)
                throw new ArgumentException(
                    $"The {role} binding '{binding.Name}' takes dimension {d} from input " +
                    $"'{source.Name}', which is {source.ElementType} rather than Int32. " +
                    "Addressing with a float tensor would reinterpret its bit pattern as an " +
                    "integer, which neither faults nor looks wrong in the generated PTX.");

            if (_productInputs.Contains(indirect.IndexInput))
                throw new ArgumentException(
                    $"Input '{source.Name}' is used both as an index source and as an " +
                    "arithmetic operand. It cannot be both.");
        }
    }

    public double[] Interpret(IReadOnlyList<double[]> inputData) => Interpret(inputData, out _);

    /// <summary>
    /// CPU reference execution returning EVERY output: the primary, then one buffer per
    /// entry of <see cref="ExtraOutputs"/> in order.
    /// </summary>
    /// <remarks>
    /// The single-secondary overload cannot express a three-output kernel, and quietly
    /// returning only the first extra would let a kernel that writes garbage to its third
    /// buffer pass every check. Callers with more than one extra must use this.
    /// </remarks>
    public double[][] InterpretAll(IReadOnlyList<double[]> inputData)
    {
        var primary = Interpret(inputData, out _, out double[][] extras);
        var all = new double[1 + extras.Length][];
        all[0] = primary;
        Array.Copy(extras, 0, all, 1, extras.Length);
        return all;
    }

    /// <summary>
    /// CPU reference execution, also returning the secondary output when there is one.
    /// </summary>
    /// <param name="inputData">Operand buffers, in parameter order.</param>
    /// <param name="secondary">
    /// The argmax positions, or null when the spec has no secondary output.
    /// </param>
    public double[] Interpret(IReadOnlyList<double[]> inputData, out double[]? secondary) =>
        Interpret(inputData, out secondary, out _);

    /// <summary>CPU reference execution, also returning every extra output buffer.</summary>
    public double[] Interpret(
        IReadOnlyList<double[]> inputData, out double[]? secondary, out double[][] extraData)
    {
        if (inputData is null) throw new ArgumentNullException(nameof(inputData));
        if (inputData.Count != _inputs.Length)
            throw new ArgumentException($"Expected {_inputs.Length} input buffers, got {inputData.Count}.", nameof(inputData));

        if (Algebra != CodegenAlgebra.Real)
        {
            extraData = Array.Empty<double[]>();
            return InterpretAlgebraic(inputData, out secondary);
        }

        extraData = new double[_extraOutputs.Length][];
        for (int e = 0; e < _extraOutputs.Length; e++)
            extraData[e] = new double[_extraOutputs[e].Binding.ElementCount];

        double[]? secondaryData = null;
        for (int e = 0; e < _extraOutputs.Length; e++)
            if (_extraOutputs[e].Kind == CodegenExtraOutputKind.ArgMaxIndex)
            {
                secondaryData = extraData[e];
                break;
            }
        secondary = secondaryData;

        var axes = Space.Axes;
        int axisCount = axes.Count;
        int[] parallel = Space.ParallelAxes;
        int[] reduction = Space.ReductionAxes;
        var values = new int[axisCount];
        var output = new double[Output.ElementCount];

        long threads = Space.TotalThreads;
        for (long tid = 0; tid < threads; tid++)
        {
            // Decompose the flat thread id across parallel axes, last-fastest --
            // exactly the decomposition the emitter must generate.
            long rest = tid;
            for (int p = parallel.Length - 1; p >= 0; p--)
            {
                int extent = axes[parallel[p]].Extent;
                values[parallel[p]] = (int)(rest % extent);
                rest /= extent;
            }
            for (int r = 0; r < reduction.Length; r++) values[reduction[r]] = 0;

            double acc = Reduce == CodegenReduceKind.Max ? double.NegativeInfinity : 0.0;
            long argIndex = 0;
            long trips = Space.ReductionTripCount;

            for (long t = 0; t < trips; t++)
            {
                long rrest = t;
                for (int r = reduction.Length - 1; r >= 0; r--)
                {
                    int extent = axes[reduction[r]].Extent;
                    values[reduction[r]] = (int)(rrest % extent);
                    rrest /= extent;
                }

                double product = 1.0;
                bool anyOutOfBounds = false;
                for (int k = 0; k < _productInputs.Length; k++)
                {
                    var binding = _inputs[_productInputs[k]];
                    long off = binding.ResolveOffset(values, inputData, out bool ok);
                    if (!ok) { anyOutOfBounds = true; break; }
                    product *= inputData[_productInputs[k]][off];
                }
                // An out-of-range tap contributes the identity OF THE REDUCTION, which is
                // not always zero. Zero is the additive identity, but for a maximum it is a
                // real candidate: a padded max-pool over all-negative inputs would return
                // 0.0 instead of the true maximum.
                //
                // This was wrong in the oracle AND in the emitter, in the same direction, so
                // the exact-agreement gate passed it -- the one failure mode a shared oracle
                // cannot catch. It is latent only because maxpool2d_2x2 has no padding.
                if (anyOutOfBounds)
                    product = Reduce == CodegenReduceKind.Max ? double.NegativeInfinity : 0.0;

                // The pre-reduction slot: a broadcast shift, then an elementwise transform,
                // applied to EACH term before it is combined. This is what softmax's
                // sum(exp(x - max)) and LayerNorm's sum((x - mean)^2) need and what an
                // epilogue activation cannot provide.
                if (!anyOutOfBounds)
                {
                    if (PreBiasInput.HasValue)
                    {
                        var pb = _inputs[PreBiasInput.Value];
                        long pbOff = pb.ResolveOffset(values, out bool pbOk);
                        if (pbOk) product += PreBiasScale * inputData[PreBiasInput.Value][pbOff];
                    }
                    product = PreReduce switch
                    {
                        CodegenPreReduceOp.None => product,
                        CodegenPreReduceOp.Exp => Math.Exp(product),
                        CodegenPreReduceOp.Square => product * product,
                        _ => throw new NotSupportedException("Unhandled pre-reduce " + PreReduce + "."),
                    };
                }

                // Which term won, evaluated at the moment it wins. Strictly greater, so
                // the FIRST maximum is kept on a tie -- the same choice the existing CUDA
                // kernel makes, and a tie-break the backward pass depends on.
                if (SecondaryIndexExpr is not null && Reduce == CodegenReduceKind.Max &&
                    product > acc)
                {
                    long candidate = SecondaryIndexExpr.Evaluate(values, out bool indexValid);
                    if (indexValid) argIndex = candidate;
                }

                acc = Reduce switch
                {
                    CodegenReduceKind.Sum => acc + product,
                    CodegenReduceKind.Max => Math.Max(acc, product),
                    _ => product
                };
            }

            if (ReduceScale != 1.0) acc *= ReduceScale;

            if (BiasInput.HasValue)
            {
                var b = _inputs[BiasInput.Value];
                long off = b.ResolveOffset(values, out bool ok);
                if (ok) acc += inputData[BiasInput.Value][off];
            }
            if (ScaleInput.HasValue)
            {
                var s = _inputs[ScaleInput.Value];
                long off = s.ResolveOffset(values, out bool ok);
                if (ok) acc *= inputData[ScaleInput.Value][off];
            }
            acc = ApplyActivation(Activation, acc);

            for (int e = 0; e < _extraOutputs.Length; e++)
            {
                var extra = _extraOutputs[e];
                long extraOff = extra.Binding.ResolveOffset(values, inputData, out bool extraOk);
                if (!extraOk) continue;

                if (extra.Kind == CodegenExtraOutputKind.ArgMaxIndex)
                {
                    extraData[e][extraOff] = argIndex;
                    continue;
                }

                // Scale * primary, then optionally + BiasScale * bias. The primary here is
                // the FINISHED value -- after the epilogue activation -- because an
                // optimizer's parameter update steps by the state just computed.
                double value = ApplyActivation(Activation, acc) * extra.Scale;
                if (extra.BiasInput.HasValue)
                {
                    var biasBinding = _inputs[extra.BiasInput.Value];
                    long biasOff = biasBinding.ResolveOffset(values, inputData, out bool biasOk);
                    if (biasOk) value += extra.BiasScale * inputData[extra.BiasInput.Value][biasOff];
                }
                extraData[e][extraOff] = value;
            }

            long outOff = Output.ResolveOffset(values, inputData, out bool outOk);
            if (outOk)
            {
                // SCATTER ACCUMULATES; a direct write ASSIGNS. Two iterations can reach the
                // same destination through a run-time index -- repeated tokens in an
                // embedding backward are the ordinary case, not the corner case -- and the
                // emitter lowers that to red.global.add.f32. An oracle that assigned would
                // disagree with a CORRECT kernel and look like a kernel bug.
                if (Output.NeedsAtomicStore) output[outOff] += acc;
                else output[outOff] = acc;
            }
        }

        return output;
    }

    /// <summary>Human-readable dump of the whole spec.</summary>
    public string Describe()
    {
        var sb = new StringBuilder();
        sb.Append(Name).Append('\n');
        sb.Append("  ").Append(Space.Describe()).Append('\n');
        for (int i = 0; i < _inputs.Length; i++)
            sb.Append("  in  ").Append(_inputs[i].Describe(Space.Axes))
              .Append(_inputs[i].NeedsBoundsCheck ? "  [guarded]" : "").Append('\n');
        sb.Append("  out ").Append(Output.Describe(Space.Axes)).Append('\n');
        sb.Append("  body ").Append(Reduce.ToString().ToLowerInvariant()).Append('(');
        for (int i = 0; i < _productInputs.Length; i++)
        {
            if (i > 0) sb.Append(" * ");
            sb.Append(_inputs[_productInputs[i]].Name);
        }
        sb.Append(')');
        if (BiasInput.HasValue) sb.Append(" + ").Append(_inputs[BiasInput.Value].Name);
        if (ScaleInput.HasValue) sb.Append(" * ").Append(_inputs[ScaleInput.Value].Name);
        if (Activation != CodegenActivationKind.None) sb.Append(" -> ").Append(Activation.ToString().ToLowerInvariant());
        return sb.Append('\n').ToString();
    }

    /// <summary>
    /// Builds the depthwise Conv2D 3x3 + bias + ReLU spec -- the bake-off target.
    /// Chosen because it has affine gather indexing, a multi-tap reduction and an
    /// epilogue: the exact shape of the conv+epilogue fusion PyTorch must split.
    /// </summary>
    public static CodegenKernelSpec DepthwiseConv2D3x3BiasRelu(int batch, int channels, int height, int width)
    {
        // Parallel axes are declared with the contiguous tensor axis LAST, so
        // consecutive threads address consecutive elements.
        var space = new CodegenIterationSpace(
            CodegenAxis.Parallel("n", batch),
            CodegenAxis.Parallel("c", channels),
            CodegenAxis.Parallel("oh", height),
            CodegenAxis.Parallel("ow", width),
            CodegenAxis.Reduce("kh", 3),
            CodegenAxis.Reduce("kw", 3));
        const int N = 0, C = 1, OH = 2, OW = 3, KH = 4, KW = 5;

        var input = new CodegenTensorBinding(
            0, "input", new[] { batch, channels, height, width },
            new[]
            {
                CodegenAffineExpr.Axis(N),
                CodegenAffineExpr.Axis(C),
                CodegenAffineExpr.Window(OH, KH, stride: 1, padding: 1),
                CodegenAffineExpr.Window(OW, KW, stride: 1, padding: 1)
            });

        var weights = new CodegenTensorBinding(
            1, "weights", new[] { channels, 3, 3 },
            new[] { CodegenAffineExpr.Axis(C), CodegenAffineExpr.Axis(KH), CodegenAffineExpr.Axis(KW) });

        var bias = new CodegenTensorBinding(
            2, "bias", new[] { channels }, new[] { CodegenAffineExpr.Axis(C) });

        var output = new CodegenTensorBinding(
            3, "output", new[] { batch, channels, height, width },
            new[]
            {
                CodegenAffineExpr.Axis(N),
                CodegenAffineExpr.Axis(C),
                CodegenAffineExpr.Axis(OH),
                CodegenAffineExpr.Axis(OW)
            },
            isOutput: true);

        return new CodegenKernelSpec(
            $"aidotnet_gen_dwconv2d3x3_n{batch}_c{channels}_h{height}_w{width}_relu",
            space,
            new[] { input, weights, bias },
            output,
            productInputs: new[] { 0, 1 },
            reduce: CodegenReduceKind.Sum,
            biasInput: 2,
            activation: CodegenActivationKind.ReLU);
    }
}
