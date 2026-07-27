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

    /// <summary>Elementwise transform applied to each term inside the reduction.</summary>
    public CodegenPreReduceOp PreReduce { get; }

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
        int? preBiasInput = null)
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
    public int ParameterCount => _inputs.Length + 1;

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
    public double[] Interpret(IReadOnlyList<double[]> inputData)
    {
        if (inputData is null) throw new ArgumentNullException(nameof(inputData));
        if (inputData.Count != _inputs.Length)
            throw new ArgumentException($"Expected {_inputs.Length} input buffers, got {inputData.Count}.", nameof(inputData));

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
                    long off = binding.ResolveOffset(values, out bool ok);
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
                        if (pbOk) product += inputData[PreBiasInput.Value][pbOff];
                    }
                    product = PreReduce switch
                    {
                        CodegenPreReduceOp.None => product,
                        CodegenPreReduceOp.Exp => Math.Exp(product),
                        CodegenPreReduceOp.Square => product * product,
                        _ => throw new NotSupportedException("Unhandled pre-reduce " + PreReduce + "."),
                    };
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

            long outOff = Output.ResolveOffset(values, out bool outOk);
            if (outOk) output[outOff] = acc;
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
