using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using static AiDotNet.Tensors.Onnx.OnnxAttributes;

namespace AiDotNet.Tensors.Onnx;

/// <summary>
/// Translates one recorded engine operation (a <see cref="CompiledStep{T}"/>: its op name, inputs, output shape and
/// saved state) into ONNX nodes. The inverse of the importer's translators: every engine operation they lower ONNX
/// to has an entry here.
/// </summary>
internal sealed class OnnxStepTranslator<T> where T : unmanaged
{
    private readonly OnnxGraphWriter<T> _w;

    internal OnnxStepTranslator(OnnxGraphWriter<T> writer) => _w = writer;

    private static readonly HashSet<string> Supported = new(StringComparer.Ordinal)
    {
        "TensorAdd", "TensorBroadcastAdd", "TensorSubtract", "TensorBroadcastSubtract", "TensorMultiply",
        "TensorBroadcastMultiply", "TensorDivide", "TensorBroadcastDivide", "TensorMax", "TensorMin", "TensorPowerTensor",
        "TensorMatMul", "TensorBatchMatMul", "BatchMatMul", "Equal",
        "ReLU", "Sigmoid", "Tanh", "GELU", "LeakyReLU", "Softmax",
        "TensorExp", "TensorLog", "TensorSqrt", "TensorAbs", "TensorNegate", "TensorPower", "TensorErfc",
        "TensorSign", "TensorFloor", "TensorCeiling", "TensorRound", "Floor", "Ceiling", "Round", "Sign",
        "TensorReciprocal", "Reciprocal",
        "Reshape", "TensorTranspose", "TensorPermute", "TensorSqueeze", "TensorExpandDims", "Expand",
        "TensorSlice", "TensorSplit", "Concat", "Concatenate", "Gather",
        "Conv2D", "ConvTranspose2D", "MaxPool2D", "AvgPool2D", "Pad",
        "BatchNormInference", "LayerNorm", "LayerNorm[0]", "ReduceSum", "ReduceMean", "FusedLinear",
    };

    /// <summary>Pads folded into the conv or pool that reads them: the padded tensor, its source and the padding.</summary>
    private readonly Dictionary<Tensor<T>, (string Source, int Top, int Left, int Bottom, int Right)> _foldedPads =
        new(OnnxGraphWriter<T>.TensorIdentity.Instance);

    internal bool CanTranslate(CompiledStep<T> step)
    {
        if (!Supported.Contains(step.OpName)) return false;
        // A fused linear translates when its activation has an exact ONNX form without parameters.
        return step.OpName != "FusedLinear" || FusedActivation(step) is
            FusedActivationType.None or FusedActivationType.ReLU or FusedActivationType.Sigmoid or
            FusedActivationType.Tanh or FusedActivationType.GELU or FusedActivationType.Softmax or
            FusedActivationType.Swish;
    }

    internal void Translate(CompiledStep<T> step, string[] inputs, string output)
    {
        var state = step.SavedState;
        var outShape = step.OutputBuffer._shape;
        switch (step.OpName)
        {
            case "TensorAdd":
            case "TensorBroadcastAdd":
                if (TryElideZeroAdd(step, inputs, output)) return;
                Binary("Add", inputs, output);
                return;
            case "TensorSubtract":
            case "TensorBroadcastSubtract": Binary("Sub", inputs, output); return;
            case "TensorMultiply":
            case "TensorBroadcastMultiply": Binary("Mul", inputs, output); return;
            case "TensorDivide":
            case "TensorBroadcastDivide": Binary("Div", inputs, output); return;
            case "TensorMax": Binary("Max", inputs, output); return;
            case "TensorMin": Binary("Min", inputs, output); return;
            case "TensorPowerTensor": Binary("Pow", inputs, output); return;
            case "Equal":
            {
                // The plan holds the comparison as 1/0 in the element type; ONNX Equal yields bool.
                string equal = _w.NewName("equal");
                _w.AddNode("Equal", new[] { inputs[0], inputs[1] }, new[] { equal });
                _w.AddNode("Cast", new[] { equal }, new[] { output }, Int("to", OnnxGraphWriter<T>.ElementType));
                return;
            }
            case "TensorMatMul":
            case "TensorBatchMatMul":
            case "BatchMatMul": Binary("MatMul", inputs, output); return;

            case "ReLU": Unary("Relu", inputs, output); return;
            case "Sigmoid": Unary("Sigmoid", inputs, output); return;
            case "Tanh": Unary("Tanh", inputs, output); return;
            case "TensorExp": Unary("Exp", inputs, output); return;
            case "TensorLog": Unary("Log", inputs, output); return;
            case "TensorSqrt": Unary("Sqrt", inputs, output); return;
            case "TensorAbs": Unary("Abs", inputs, output); return;
            case "TensorNegate": Unary("Neg", inputs, output); return;
            case "TensorSign":
            case "Sign": Unary("Sign", inputs, output); return;
            case "TensorFloor":
            case "Floor": Unary("Floor", inputs, output); return;
            case "TensorCeiling":
            case "Ceiling": Unary("Ceil", inputs, output); return;
            case "TensorRound":
            case "Round": Unary("Round", inputs, output); return;
            case "TensorReciprocal":
            case "Reciprocal": Unary("Reciprocal", inputs, output); return;
            case "LeakyReLU":
                _w.AddNode("LeakyRelu", new[] { inputs[0] }, new[] { output },
                    Float("alpha", StateDouble(step, 0)));
                return;
            case "Softmax":
            {
                int axis = StateInt(step, 0);
                _w.AddNode("Softmax", new[] { inputs[0] }, new[] { output }, Int("axis", axis));
                return;
            }
            case "GELU": Gelu(inputs[0], output); return;
            case "TensorPower":
            {
                // The exponent is saved as the element type.
                if (state is not { Length: > 0 } || state[0] is not T exponent)
                    throw new NotSupportedException("TensorPower step has no saved exponent.");
                double e = MathHelper.GetNumericOperations<T>().ToDouble(exponent);
                _w.AddNode("Pow", new[] { inputs[0], _w.AddScalar(e) }, new[] { output });
                return;
            }
            case "TensorErfc":
            {
                // erfc(x) = 1 - erf(x): ONNX has Erf only.
                string erf = _w.NewName("erf");
                _w.AddNode("Erf", new[] { inputs[0] }, new[] { erf });
                _w.AddNode("Sub", new[] { _w.AddScalar(1.0), erf }, new[] { output });
                return;
            }

            case "Reshape":
            case "TensorSqueeze":
            case "TensorExpandDims":
                _w.AddNode("Reshape", new[] { inputs[0], _w.AddInt64(outShape) }, new[] { output });
                return;
            case "Expand":
                _w.AddNode("Expand", new[] { inputs[0], _w.AddInt64(outShape) }, new[] { output });
                return;
            case "TensorTranspose":
            {
                int rank = step.Inputs[0].Rank;
                var perm = Enumerable.Range(0, rank).ToArray();
                if (rank >= 2) (perm[rank - 1], perm[rank - 2]) = (perm[rank - 2], perm[rank - 1]);
                _w.AddNode("Transpose", new[] { inputs[0] }, new[] { output }, Ints("perm", perm));
                return;
            }
            case "TensorPermute":
                _w.AddNode("Transpose", new[] { inputs[0] }, new[] { output }, Ints("perm", StateInts(step, 0)));
                return;
            case "TensorSlice":
            {
                var start = StateInts(step, 0);
                var ends = new long[start.Length];
                for (int i = 0; i < start.Length; i++) ends[i] = start[i] + outShape[i];
                var axes = Enumerable.Range(0, start.Length).ToArray();
                _w.AddNode("Slice", new[] { inputs[0], _w.AddInt64(start), _w.AddInt64(ends), _w.AddInt64(axes) }, new[] { output });
                return;
            }
            case "TensorSplit":
            {
                // state: numSplits, axis, index of this piece; each piece is an equal share of the axis.
                int axis = Normalize(StateInt(step, 1), step.Inputs[0].Rank);
                int piece = StateInt(step, 2);
                int size = outShape[axis];
                _w.AddNode("Slice", new[]
                {
                    inputs[0], _w.AddInt64(new long[] { (long)piece * size }), _w.AddInt64(new long[] { (long)(piece + 1) * size }),
                    _w.AddInt64(new long[] { axis }),
                }, new[] { output });
                return;
            }
            case "Concat":
            case "Concatenate":
                _w.AddNode("Concat", inputs, new[] { output }, Int("axis", StateInt(step, 0)));
                return;
            case "Gather": Gather(step, inputs, output); return;

            case "Conv2D": Conv(step, inputs, output); return;
            case "ConvTranspose2D": ConvTranspose(step, inputs, output); return;
            case "MaxPool2D": MaxPool(step, inputs, output); return;
            case "AvgPool2D": AvgPool(step, inputs, output); return;
            case "Pad": Pad(step, inputs, output); return;

            case "BatchNormInference":
                _w.AddNode("BatchNormalization", inputs, new[] { output }, Float("epsilon", StateDouble(step, 0)));
                return;
            case "LayerNorm":
            case "LayerNorm[0]":
            {
                // The engine normalizes over the trailing dimensions gamma spans.
                int axis = step.Inputs[0].Rank - step.Inputs[1].Rank;
                // The traced form saves [mean, variance, epsilon]; the inference capture saves [epsilon].
                int epsilonIndex = step.OpName == "LayerNorm" ? 2 : 0;
                if (state is null || state.Length <= epsilonIndex || state[epsilonIndex] is not double epsilon)
                    throw new NotSupportedException($"The {step.OpName} step does not record its epsilon.");
                _w.AddNode("LayerNormalization", inputs, new[] { output }, Int("axis", axis), Float("epsilon", epsilon));
                return;
            }
            case "ReduceSum":
            case "ReduceMean": Reduce(step, inputs, output); return;
            case "FusedLinear": FusedLinear(step, inputs, output); return;
        }
        throw new NotSupportedException($"No ONNX translation for '{step.OpName}'.");
    }

    // ── Helpers ────────────────────────────────────────────────────────────────────────────────────────────────────

    private void Unary(string op, string[] inputs, string output) => _w.AddNode(op, new[] { inputs[0] }, new[] { output });

    private void Binary(string op, string[] inputs, string output) => _w.AddNode(op, new[] { inputs[0], inputs[1] }, new[] { output });

    private static int Normalize(int axis, int rank) => axis < 0 ? axis + rank : axis;

    private static object State(CompiledStep<T> step, int index)
        => step.SavedState is { } s && s.Length > index && s[index] is { } v
            ? v
            : throw new NotSupportedException($"'{step.OpName}' step is missing saved state [{index}].");

    private static int StateInt(CompiledStep<T> step, int index) => State(step, index) switch
    {
        int i => i,
        int[] { Length: > 0 } a => a[0],
        long l => checked((int)l),
        var other => throw new NotSupportedException($"'{step.OpName}' saved state [{index}] is {other.GetType().Name}, not int."),
    };

    private static int[] StateInts(CompiledStep<T> step, int index) => State(step, index) switch
    {
        int[] a => a,
        int i => new[] { i },
        var other => throw new NotSupportedException($"'{step.OpName}' saved state [{index}] is {other.GetType().Name}, not int[]."),
    };

    private static double StateDouble(CompiledStep<T> step, int index) => State(step, index) switch
    {
        double d => d,
        float f => f,
        T t => MathHelper.GetNumericOperations<T>().ToDouble(t),
        var other => throw new NotSupportedException($"'{step.OpName}' saved state [{index}] is {other.GetType().Name}, not a number."),
    };

    /// <summary>The importer binds every graph output through <c>x + 0</c>, and broadcasts through <c>x + zeros</c>:
    /// write those as Identity / Expand rather than an Add against an all-zero initializer.</summary>
    private bool TryElideZeroAdd(CompiledStep<T> step, string[] inputs, string output)
    {
        for (int z = 0; z < 2; z++)
        {
            var zero = step.Inputs[z];
            var other = step.Inputs[1 - z];
            if (!IsAllZeroConstant(zero)) continue;
            if (!SameShape(zero._shape, step.OutputBuffer._shape)) continue;   // the zeros carry the output shape
            if (SameShape(other._shape, step.OutputBuffer._shape))
                _w.AddNode("Identity", new[] { inputs[1 - z] }, new[] { output });
            else
                _w.AddNode("Expand", new[] { inputs[1 - z], _w.AddInt64(step.OutputBuffer._shape) }, new[] { output });
            return true;
        }
        return false;
    }

    private bool IsAllZeroConstant(Tensor<T> t)
    {
        if (_w.IsRuntimeValue(t) || t.Layout != LinearAlgebra.TensorLayout.Nchw || !t.IsContiguous) return false;
        var ops = MathHelper.GetNumericOperations<T>();
        foreach (var v in t.AsSpan())
        {
            // -0 is still zero for x + 0, but it would turn +0 inputs into... +0; equality covers both signs.
            if (ops.ToDouble(v) != 0.0) return false;
        }
        return true;
    }

    private static bool SameShape(int[] a, int[] b)
    {
        if (a.Length != b.Length) return false;
        for (int i = 0; i < a.Length; i++) if (a[i] != b[i]) return false;
        return true;
    }

    private void Gelu(string input, string output)
    {
        if (_w.Opset >= 20)
        {
            _w.AddNode("Gelu", new[] { input }, new[] { output }, Str("approximate", "tanh"));
            return;
        }
        // 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))
        string x3 = _w.NewName("gelu"), inner = _w.NewName("gelu"), sum = _w.NewName("gelu"), scaled = _w.NewName("gelu"),
            th = _w.NewName("gelu"), onePlus = _w.NewName("gelu"), half = _w.NewName("gelu");
        _w.AddNode("Pow", new[] { input, _w.AddScalar(3.0) }, new[] { x3 });
        _w.AddNode("Mul", new[] { x3, _w.AddScalar(0.044715) }, new[] { inner });
        _w.AddNode("Add", new[] { input, inner }, new[] { sum });
        _w.AddNode("Mul", new[] { sum, _w.AddScalar(Math.Sqrt(2.0 / Math.PI)) }, new[] { scaled });
        _w.AddNode("Tanh", new[] { scaled }, new[] { th });
        _w.AddNode("Add", new[] { th, _w.AddScalar(1.0) }, new[] { onePlus });
        _w.AddNode("Mul", new[] { input, _w.AddScalar(0.5) }, new[] { half });
        _w.AddNode("Mul", new[] { half, onePlus }, new[] { output });
    }

    private void Gather(CompiledStep<T> step, string[] inputs, string output)
    {
        int dataRank = step.Inputs[0].Rank;
        if (step.Inputs.Length == 2)
        {
            // The ONNX importer's form: the indices are a tensor input holding integer values; state: the axis.
            int gatherAxis = Normalize(StateInt(step, 0), dataRank);
            var indicesTensor = step.Inputs[1];
            string indicesName;
            if (_w.IsRuntimeValue(indicesTensor))
            {
                indicesName = _w.NewName("indices");
                _w.AddNode("Cast", new[] { inputs[1] }, new[] { indicesName },
                    Int("to", (long)Protos.TensorProto.Types.DataType.Int64));
            }
            else
            {
                var ops = MathHelper.GetNumericOperations<T>();
                var values = indicesTensor.ToArray().Select(v => (long)Math.Round(ops.ToDouble(v))).ToArray();
                indicesName = _w.AddInt64(values, indicesTensor._shape);
            }
            _w.AddNode("Gather", new[] { inputs[0], indicesName }, new[] { output }, Int("axis", gatherAxis));
            return;
        }

        // The engine's form: state holds the indices and the axis. The indices' shape is the output's shape at the
        // gathered axis minus the data's remaining dims.
        int axis = Normalize(StateInt(step, 1), dataRank);
        var indicesState = State(step, 0);
        long[] indices = indicesState switch
        {
            int[] a => a.Select(i => (long)i).ToArray(),
            Tensor<int> t => t.ToArray().Select(i => (long)i).ToArray(),
            _ => throw new NotSupportedException($"Gather saved indices of type {indicesState.GetType().Name}."),
        };
        int indicesRank = step.OutputBuffer.Rank - dataRank + 1;
        var indicesShape = step.OutputBuffer._shape.Skip(axis).Take(indicesRank).ToArray();
        _w.AddNode("Gather", new[] { inputs[0], _w.AddInt64(indices, indicesShape) }, new[] { output }, Int("axis", axis));
    }

    private static FusedActivationType? FusedActivation(CompiledStep<T> step)
        => step.SavedState is { Length: > 0 } state ? state[0] as FusedActivationType? : FusedActivationType.None;

    /// <summary>input @ weights (+ bias), then the activation: MatMul, Add and the activation's ONNX operator.</summary>
    private void FusedLinear(CompiledStep<T> step, string[] inputs, string output)
    {
        var activation = FusedActivation(step) ?? throw new NotSupportedException("FusedLinear saved no activation.");
        bool hasBias = inputs.Length > 2;
        string linear = hasBias || activation != FusedActivationType.None ? _w.NewName("linear") : output;
        _w.AddNode("MatMul", new[] { inputs[0], inputs[1] }, new[] { linear });
        string pre = linear;
        if (hasBias)
        {
            pre = activation == FusedActivationType.None ? output : _w.NewName("biased");
            _w.AddNode("Add", new[] { linear, inputs[2] }, new[] { pre });
        }
        switch (activation)
        {
            case FusedActivationType.None: return;
            case FusedActivationType.ReLU: _w.AddNode("Relu", new[] { pre }, new[] { output }); return;
            case FusedActivationType.Sigmoid: _w.AddNode("Sigmoid", new[] { pre }, new[] { output }); return;
            case FusedActivationType.Tanh: _w.AddNode("Tanh", new[] { pre }, new[] { output }); return;
            case FusedActivationType.GELU: Gelu(pre, output); return;   // the fused epilogue is the tanh form
            case FusedActivationType.Softmax:
                _w.AddNode("Softmax", new[] { pre }, new[] { output }, Int("axis", -1));
                return;
            case FusedActivationType.Swish:
            {
                string sig = _w.NewName("sigmoid");
                _w.AddNode("Sigmoid", new[] { pre }, new[] { sig });
                _w.AddNode("Mul", new[] { pre, sig }, new[] { output });
                return;
            }
        }
        throw new NotSupportedException($"FusedLinear activation {activation} has no ONNX translation.");
    }

    private void Conv(CompiledStep<T> step, string[] inputs, string output)
    {
        var x = step.Inputs[0];
        var w = step.Inputs[1];
        var strides = Pair(StateInts(step, 0));
        var pads = Pair(StateInts(step, 1));
        var dilations = Pair(StateInts(step, 2));
        int group = x._shape[1] / w._shape[1];
        var allPads = WithFoldedPad(step, inputs, pads);
        _w.AddNode("Conv", new[] { inputs[0], inputs[1] }, new[] { output },
            Ints("kernel_shape", new[] { w._shape[2], w._shape[3] }),
            Ints("strides", strides),
            Ints("pads", allPads),
            Ints("dilations", dilations),
            Int("group", group));
    }

    private void ConvTranspose(CompiledStep<T> step, string[] inputs, string output)
    {
        var x = step.Inputs[0];
        var w = step.Inputs[1];   // [in, out, kH, kW], ONNX's layout
        var strides = Pair(StateInts(step, 0));
        var pads = Pair(StateInts(step, 1));
        var outShape = step.OutputBuffer._shape;
        var outputPadding = new int[2];
        for (int d = 0; d < 2; d++)
        {
            int natural = (x._shape[2 + d] - 1) * strides[d] - 2 * pads[d] + w._shape[2 + d];
            outputPadding[d] = outShape[2 + d] - natural;
            if (outputPadding[d] < 0)
                throw new NotSupportedException("ConvTranspose2D output is smaller than its stride/padding/kernel imply.");
        }
        _w.AddNode("ConvTranspose", new[] { inputs[0], inputs[1] }, new[] { output },
            Ints("kernel_shape", new[] { w._shape[2], w._shape[3] }),
            Ints("strides", strides),
            Ints("pads", new[] { pads[0], pads[1], pads[0], pads[1] }),
            Ints("output_padding", outputPadding));
    }

    private void MaxPool(CompiledStep<T> step, string[] inputs, string output)
    {
        // state: [k, k], [s, s], and the padding (int) when non-zero.
        var kernel = Pair(StateInts(step, 0));
        var strides = step.SavedState is { Length: > 1 } ? Pair(StateInts(step, 1)) : kernel;
        int pad = step.SavedState is { Length: > 2 } ? StateInt(step, 2) : 0;
        var allPads = WithFoldedPad(step, inputs, new[] { pad, pad });
        _w.AddNode("MaxPool", new[] { inputs[0] }, new[] { output },
            Ints("kernel_shape", kernel), Ints("strides", strides), Ints("pads", allPads));
    }

    private void AvgPool(CompiledStep<T> step, string[] inputs, string output)
    {
        // state: [k, k], [s, s], [p, p], countIncludePad.
        var kernel = Pair(StateInts(step, 0));
        var strides = Pair(StateInts(step, 1));
        var pads = Pair(StateInts(step, 2));
        bool countIncludePad = step.SavedState is { Length: > 3 } && step.SavedState[3] is bool b && b;
        // A folded zero pad was averaged over as real elements: count_include_pad's semantics.
        if (_foldedPads.ContainsKey(step.Inputs[0])) countIncludePad = true;
        var allPads = WithFoldedPad(step, inputs, pads);
        _w.AddNode("AveragePool", new[] { inputs[0] }, new[] { output },
            Ints("kernel_shape", kernel), Ints("strides", strides),
            Ints("pads", allPads),
            Int("count_include_pad", countIncludePad ? 1 : 0));
    }

    private void Pad(CompiledStep<T> step, string[] inputs, string output)
    {
        // state: top, left, bottom, right, fill value (as double); padding applies to the last two dims.
        if (step.SavedState is not { Length: >= 5 })
            throw new NotSupportedException("Pad step does not record its full padding (bottom, right, value).");
        int top = StateInt(step, 0), left = StateInt(step, 1), bottom = StateInt(step, 2), right = StateInt(step, 3);
        double value = StateDouble(step, 4);
        int rank = step.Inputs[0].Rank;
        // A pad whose only reader is a conv or pool becomes that operator's (possibly asymmetric) pads, the form the
        // importer lowered it from: zero for conv and average pool, -inf for max pool.
        var reader = _w.SoleReader(step.OutputBuffer);
        bool foldable = reader?.OpName switch
        {
            "Conv2D" or "AvgPool2D" => value == 0,
            "MaxPool2D" => double.IsNegativeInfinity(value) || value <= -3.0e38,
            _ => false,
        };
        if (rank == 4 && foldable && reader is not null && ReferenceEquals(reader.Inputs[0], step.OutputBuffer))
        {
            _foldedPads[step.OutputBuffer] = (inputs[0], top, left, bottom, right);
            return;
        }
        var pads = new long[2 * rank];
        pads[rank - 2] = top; pads[rank - 1] = left;
        pads[2 * rank - 2] = bottom; pads[2 * rank - 1] = right;
        _w.AddNode("Pad", new[] { inputs[0], _w.AddInt64(pads), _w.AddScalar(value) }, new[] { output }, Str("mode", "constant"));
    }

    private void Reduce(CompiledStep<T> step, string[] inputs, string output)
    {
        // state: axes (int[]; empty = every axis), keepDims.
        int rank = step.Inputs[0].Rank;
        var axes = StateInts(step, 0).Select(a => Normalize(a, rank)).ToArray();
        if (axes.Length == 0) axes = Enumerable.Range(0, rank).ToArray();
        bool keepDims = step.SavedState is { Length: > 1 } && step.SavedState[1] is bool k && k;
        string op = step.OpName == "ReduceSum" ? "ReduceSum" : "ReduceMean";
        // ReduceSum takes axes as an input from opset 13, ReduceMean from opset 18.
        if (op == "ReduceSum" || _w.Opset >= 18)
            _w.AddNode(op, new[] { inputs[0], _w.AddInt64(axes) }, new[] { output }, Int("keepdims", keepDims ? 1 : 0));
        else
            _w.AddNode(op, new[] { inputs[0] }, new[] { output }, Ints("axes", axes), Int("keepdims", keepDims ? 1 : 0));
    }

    /// <summary>
    /// ONNX pads [top, left, bottom, right] for a conv or pool with symmetric <paramref name="symmetric"/> padding, plus
    /// a pad folded into it; a folded pad's source replaces the step's first input.
    /// </summary>
    private int[] WithFoldedPad(CompiledStep<T> step, string[] inputs, int[] symmetric)
    {
        var pads = new[] { symmetric[0], symmetric[1], symmetric[0], symmetric[1] };
        if (_foldedPads.TryGetValue(step.Inputs[0], out var folded))
        {
            inputs[0] = folded.Source;
            pads[0] += folded.Top; pads[1] += folded.Left; pads[2] += folded.Bottom; pads[3] += folded.Right;
        }
        return pads;
    }

    private static int[] Pair(int[] values) => values.Length switch
    {
        1 => new[] { values[0], values[0] },
        2 => values,
        _ => throw new NotSupportedException($"Expected one or two spatial values, got {values.Length}."),
    };
}
