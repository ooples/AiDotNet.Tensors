using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Onnx.Protos;

namespace AiDotNet.Tensors.Onnx.Operators;

/// <summary>
/// ONNX padding descriptor as four edges [top, left, bottom, right]. Used
/// to translate ONNX's potentially-asymmetric <c>pads</c> / <c>auto_pad</c>
/// into the pair that the in-house engine's Conv / Pool APIs expect, with
/// any leftover asymmetry applied up front via an explicit Pad op.
/// </summary>
internal readonly struct OnnxPadding
{
    public int Top { get; }
    public int Left { get; }
    public int Bottom { get; }
    public int Right { get; }
    public OnnxPadding(int top, int left, int bottom, int right) { Top = top; Left = left; Bottom = bottom; Right = right; }

    public bool IsSymmetric => Top == Bottom && Left == Right;
    public bool IsAllZero => Top == 0 && Left == 0 && Bottom == 0 && Right == 0;
    public int[] SymmetricPair => new[] { Top, Left };
    public override string ToString() => $"[{Top},{Left},{Bottom},{Right}]";
}

/// <summary>
/// Resolves ONNX 2D padding descriptors (<c>pads</c> + <c>auto_pad</c>)
/// into a <see cref="OnnxPadding"/>. Shared by Conv / ConvTranspose /
/// MaxPool / AveragePool translators.
/// </summary>
internal static class OnnxAutoPad
{
    /// <summary>
    /// Resolve a Conv-style padding descriptor to an <see cref="OnnxPadding"/>.
    /// Priority: explicit <c>pads</c> → <c>auto_pad</c> derived → zero. Asymmetric
    /// resolutions are allowed; the caller decides whether to pass them to the
    /// engine (symmetric only) or apply an explicit Pad up front.
    /// </summary>
    internal static OnnxPadding Resolve2D(
        string opName,
        int[]? inputShape, int[] kernelShape, int[] strides,
        string? autoPad, int[]? explicitPads,
        bool ceilMode = false)
    {
        // Explicit pads win if provided (ONNX spec: auto_pad is ignored when
        // pads is set).
        if (explicitPads is not null)
        {
            if (explicitPads.Length != 4)
                throw new InvalidDataException(
                    $"{opName} pads attribute has {explicitPads.Length} values; expected 4 (NCHW: [top, left, bottom, right]).");
            // ONNX pads order is [top, left, bottom, right].
            var p = new OnnxPadding(explicitPads[0], explicitPads[1], explicitPads[2], explicitPads[3]);
            return ceilMode && inputShape is not null
                ? WithCeilExtra(p, inputShape, kernelShape, strides)
                : p;
        }

        // No explicit pads. Handle auto_pad.
        switch (autoPad)
        {
            case null:
            case "":
            case "NOTSET":
            case "VALID":
                var basePadValid = new OnnxPadding(0, 0, 0, 0);
                return ceilMode && inputShape is not null
                    ? WithCeilExtra(basePadValid, inputShape, kernelShape, strides)
                    : basePadValid;

            case "SAME_UPPER":
            case "SAME_LOWER":
                if (inputShape is null || inputShape.Length < 4)
                    throw new InvalidDataException(
                        $"{opName} auto_pad={autoPad} requires known input spatial dims.");
                int totalH = TotalSamePad(inputShape[2], kernelShape[0], strides[0]);
                int totalW = TotalSamePad(inputShape[3], kernelShape[1], strides[1]);
                // SAME_UPPER = extra goes to bottom/right; SAME_LOWER = to top/left.
                int topH, botH, leftW, rightW;
                if (autoPad == "SAME_UPPER")
                {
                    topH = totalH / 2; botH = totalH - topH;
                    leftW = totalW / 2; rightW = totalW - leftW;
                }
                else
                {
                    botH = totalH / 2; topH = totalH - botH;
                    rightW = totalW / 2; leftW = totalW - rightW;
                }
                // Auto-pad SAME already targets the ceil output size, so ceil_mode
                // is a no-op here.
                return new OnnxPadding(topH, leftW, botH, rightW);

            default:
                throw new NotSupportedException(
                    $"{opName} auto_pad={autoPad} is not recognized. Supported: NOTSET, VALID, SAME_UPPER, SAME_LOWER.");
        }
    }

    /// <summary>
    /// ceil_mode=1 in ONNX rounds the output shape UP when
    /// (input + total_pad - kernel) is not divisible by stride. The
    /// equivalent floor-mode computation adds up to (stride - 1) extra
    /// bottom/right padding so the same rounded-up output appears naturally.
    /// The returned padding is asymmetric whenever the original was
    /// symmetric but ceil added a single-side extra — the caller applies
    /// it via an explicit Pad op before the pool.
    /// </summary>
    private static OnnxPadding WithCeilExtra(
        OnnxPadding pad, int[] inputShape, int[] kernelShape, int[] strides)
    {
        int extraH = ExtraPadForCeil(inputShape[2], kernelShape[0], strides[0], pad.Top + pad.Bottom);
        int extraW = ExtraPadForCeil(inputShape[3], kernelShape[1], strides[1], pad.Left + pad.Right);
        return new OnnxPadding(pad.Top, pad.Left, pad.Bottom + extraH, pad.Right + extraW);
    }

    private static int ExtraPadForCeil(int inputSize, int kernelSize, int stride, int totalPad)
    {
        // Floor output under (input + totalPad - kernel) / stride + 1:
        int numerator = inputSize + totalPad - kernelSize;
        if (numerator < 0) return 0;
        int floorOut = numerator / stride + 1;
        int ceilOut = (numerator + stride - 1) / stride + 1;
        if (ceilOut == floorOut) return 0;
        // Need enough bottom/right pad to make numerator divisible by stride,
        // i.e. round the numerator up to the next multiple of stride.
        int rounded = ((numerator + stride - 1) / stride) * stride;
        return rounded - numerator;
    }

    private static int TotalSamePad(int inputSize, int kernelSize, int stride)
    {
        // Output size under SAME = ceil(input / stride) → required total pad.
        int outputSize = (inputSize + stride - 1) / stride;
        int total = Math.Max(0, (outputSize - 1) * stride + kernelSize - inputSize);
        return total;
    }

    /// <summary>
    /// Applies an explicit <see cref="IEngine.Pad{T}"/> op when the resolved
    /// ONNX padding is asymmetric, returning (newInput, symmetricResidualPad).
    /// When the padding is already symmetric this is a no-op — caller can
    /// pass the residual pair directly to Conv / Pool.
    /// </summary>
    internal static (Tensor<T> input, int[] symmetricPad) ApplyAsymmetric<T>(
        Engines.IEngine engine, Tensor<T> input, OnnxPadding pad, T padValue) where T : unmanaged
    {
        if (pad.IsSymmetric) return (input, pad.SymmetricPair);

        // Factor out the symmetric component so the Conv/Pool still gets some
        // built-in padding (cheaper than growing the whole input). residual =
        // max(top, bottom) - symmetric gets absorbed by Conv's padding arg.
        int symH = Math.Min(pad.Top, pad.Bottom);
        int symW = Math.Min(pad.Left, pad.Right);
        int extraTop = pad.Top - symH;
        int extraBottom = pad.Bottom - symH;
        int extraLeft = pad.Left - symW;
        int extraRight = pad.Right - symW;
        var padded = engine.Pad(input, extraTop, extraBottom, extraLeft, extraRight, padValue);
        return (padded, new[] { symH, symW });
    }
}

/// <summary>
/// ONNX convolution / pooling operator translators: Conv, ConvTranspose,
/// MaxPool, AveragePool, GlobalAveragePool.
/// </summary>
internal static class ConvOperators
{
    internal static void Register<T>(OnnxOpTranslatorRegistry<T> r) where T : unmanaged
    {
        r.Register(new Conv<T>());
        r.Register(new ConvTranspose<T>());
        r.Register(new MaxPool<T>());
        r.Register(new AveragePool<T>());
        r.Register(new GlobalAveragePool<T>());
    }

    /// <summary>
    /// ONNX Conv — 2D convolution with attributes {kernel_shape, strides,
    /// pads, dilations, group, auto_pad}. Maps to <c>IEngine.Conv2D(int[],
    /// int[], int[])</c>. Pads are ONNX's 4-element [top, left, bottom,
    /// right]; asymmetric splits get materialized up front via
    /// <c>IEngine.Pad</c>. Grouped conv (group > 1) is handled by slicing
    /// input and kernel per group and concatenating per-group outputs.
    /// </summary>
    internal sealed class Conv<T> : IOnnxOpTranslator<T> where T : unmanaged
    {
        public string OpType => "Conv";
        public string? Domain => null;
        public void Translate(OnnxTranslationContext<T> ctx, NodeProto node)
        {
            var input = ctx.GetTensor(node.Input[0]);
            var kernel = ctx.GetTensor(node.Input[1]);
            var hasBias = node.Input.Count > 2 && ctx.HasTensor(node.Input[2]);
            if (input.Rank != 4)
                throw new NotSupportedException(
                    $"Phase 1 Conv supports 4D input (NCHW); got rank {input.Rank}.");

            int group = ctx.GetIntAttrAsInt(node, "group", 1);
            var strides = ctx.GetIntArrayAttr(node, "strides") ?? new[] { 1, 1 };
            var dilations = ctx.GetIntArrayAttr(node, "dilations") ?? new[] { 1, 1 };
            var kernelSpatial = new[] { kernel._shape[2], kernel._shape[3] };
            var explicitPads = ctx.GetIntArrayAttr(node, "pads");
            var autoPad = ctx.GetStringAttr(node, "auto_pad", null);
            var pad = OnnxAutoPad.Resolve2D("Conv", input._shape, kernelSpatial, strides, autoPad, explicitPads);

            // Apply asymmetric component via explicit Pad (zero-valued);
            // Conv gets the symmetric residual.
            var zero = default(T);
            var (paddedInput, symmetricPad) = OnnxAutoPad.ApplyAsymmetric(ctx.Engine, input, pad, zero);

            Tensor<T> result;
            if (group == 1)
            {
                result = ctx.Engine.Conv2D(paddedInput, kernel,
                    stride: strides, padding: symmetricPad, dilation: dilations);
            }
            else
            {
                result = GroupedConv(ctx, paddedInput, kernel, group, strides, symmetricPad, dilations);
            }

            if (hasBias)
            {
                var bias = ctx.GetTensor(node.Input[2]);
                var reshaped = ctx.Engine.Reshape(bias, new[] { 1, bias._shape[0], 1, 1 });
                result = ctx.Engine.TensorBroadcastAdd(result, reshaped);
            }
            ctx.PutTensor(node.Output[0], result);
        }

        /// <summary>
        /// Grouped convolution: splits <paramref name="input"/> along channel
        /// dim 1 into <paramref name="group"/> equal partitions, runs an
        /// independent Conv2D on each, and concatenates the outputs along the
        /// output-channel dim. For <c>group == Cin</c> this is the depthwise
        /// convolution that MobileNet uses; for other values it's the
        /// ResNeXt / ShuffleNet pattern.
        /// </summary>
        private static Tensor<T> GroupedConv(
            OnnxTranslationContext<T> ctx,
            Tensor<T> input, Tensor<T> kernel,
            int group, int[] strides, int[] symmetricPad, int[] dilations)
        {
            if (input._shape[1] % group != 0)
                throw new InvalidDataException(
                    $"Grouped Conv requires Cin ({input._shape[1]}) divisible by group ({group}).");
            if (kernel._shape[0] % group != 0)
                throw new InvalidDataException(
                    $"Grouped Conv requires Cout ({kernel._shape[0]}) divisible by group ({group}).");

            int cInPerGroup = input._shape[1] / group;
            int cOutPerGroup = kernel._shape[0] / group;
            var outputs = new List<Tensor<T>>(group);
            for (int g = 0; g < group; g++)
            {
                var inputSlice = ctx.Engine.TensorSlice(input,
                    start: new[] { 0, g * cInPerGroup, 0, 0 },
                    length: new[] { input._shape[0], cInPerGroup, input._shape[2], input._shape[3] });
                var kernelSlice = ctx.Engine.TensorSlice(kernel,
                    start: new[] { g * cOutPerGroup, 0, 0, 0 },
                    length: new[] { cOutPerGroup, kernel._shape[1], kernel._shape[2], kernel._shape[3] });
                outputs.Add(ctx.Engine.Conv2D(inputSlice, kernelSlice,
                    stride: strides, padding: symmetricPad, dilation: dilations));
            }
            return ctx.Engine.Concat(outputs, axis: 1);
        }
    }

    /// <summary>
    /// ONNX ConvTranspose — transposed 2D convolution. Same attribute set as
    /// Conv plus output_padding. Maps to <c>IEngine.ConvTranspose2D</c>.
    /// </summary>
    internal sealed class ConvTranspose<T> : IOnnxOpTranslator<T> where T : unmanaged
    {
        public string OpType => "ConvTranspose";
        public string? Domain => null;
        public void Translate(OnnxTranslationContext<T> ctx, NodeProto node)
        {
            var input = ctx.GetTensor(node.Input[0]);
            var kernel = ctx.GetTensor(node.Input[1]);
            var hasBias = node.Input.Count > 2 && ctx.HasTensor(node.Input[2]);
            if (input.Rank != 4)
                throw new NotSupportedException(
                    $"Phase 1 ConvTranspose supports 4D input (NCHW); got rank {input.Rank}.");

            int group = ctx.GetIntAttrAsInt(node, "group", 1);
            if (group != 1)
                throw new NotSupportedException("ConvTranspose group != 1 is not yet supported.");

            var strides = ctx.GetIntArrayAttr(node, "strides") ?? new[] { 1, 1 };
            var outputPadding = ctx.GetIntArrayAttr(node, "output_padding") ?? new[] { 0, 0 };
            var kernelSpatial = new[] { kernel._shape[2], kernel._shape[3] };
            var explicitPads = ctx.GetIntArrayAttr(node, "pads");
            var autoPad = ctx.GetStringAttr(node, "auto_pad", null);
            var pad = OnnxAutoPad.Resolve2D("ConvTranspose", input._shape, kernelSpatial, strides, autoPad, explicitPads);
            if (!pad.IsSymmetric)
                throw new NotSupportedException(
                    $"ConvTranspose asymmetric padding {pad} is not yet supported " +
                    "(transposed conv's geometry differs from Conv — asymmetric is tracked as a follow-up).");

            var result = ctx.Engine.ConvTranspose2D(input, kernel,
                stride: strides, padding: pad.SymmetricPair, outputPadding: outputPadding);

            if (hasBias)
            {
                var bias = ctx.GetTensor(node.Input[2]);
                var reshaped = ctx.Engine.Reshape(bias, new[] { 1, bias._shape[0], 1, 1 });
                result = ctx.Engine.TensorBroadcastAdd(result, reshaped);
            }
            ctx.PutTensor(node.Output[0], result);
        }
    }

    /// <summary>
    /// ONNX MaxPool — spatial max pooling with {kernel_shape, strides, pads,
    /// dilations, ceil_mode, storage_order}.
    /// </summary>
    internal sealed class MaxPool<T> : IOnnxOpTranslator<T> where T : unmanaged
    {
        public string OpType => "MaxPool";
        public string? Domain => null;
        public void Translate(OnnxTranslationContext<T> ctx, NodeProto node)
        {
            var input = ctx.GetTensor(node.Input[0]);
            var kernelShape = ctx.GetIntArrayAttr(node, "kernel_shape")
                ?? throw new InvalidDataException("MaxPool requires kernel_shape.");
            var strides = ctx.GetIntArrayAttr(node, "strides") ?? kernelShape;
            int ceilMode = ctx.GetIntAttrAsInt(node, "ceil_mode", 0);

            var explicitPads = ctx.GetIntArrayAttr(node, "pads");
            var autoPad = ctx.GetStringAttr(node, "auto_pad", null);
            var pad = OnnxAutoPad.Resolve2D("MaxPool", input._shape, kernelShape, strides, autoPad, explicitPads, ceilMode: ceilMode != 0);

            // MaxPool's "pad value" for explicit padding should be -infinity so
            // the pool window never picks a padded position as the max.
            var padValue = MathHelper.GetNumericOperations<T>().FromDouble(double.NegativeInfinity);
            var (paddedInput, symmetricPad) = OnnxAutoPad.ApplyAsymmetric(ctx.Engine, input, pad, padValue);

            if (kernelShape[0] != kernelShape[1])
                throw new NotSupportedException(
                    $"MaxPool non-square kernel [{string.Join(",", kernelShape)}] is a Phase 2 op.");
            if (strides[0] != strides[1])
                throw new NotSupportedException(
                    $"MaxPool non-square stride [{string.Join(",", strides)}] is a Phase 2 op.");
            if (symmetricPad[0] != symmetricPad[1])
                throw new NotSupportedException(
                    $"MaxPool residual non-square padding [{symmetricPad[0]}, {symmetricPad[1]}] is a Phase 2 op.");
            var result = ctx.Engine.MaxPool2D(paddedInput, kernelShape[0], strides[0], symmetricPad[0]);
            ctx.PutTensor(node.Output[0], result);
        }
    }

    /// <summary>
    /// ONNX AveragePool — spatial mean pooling. Same attribute handling as
    /// MaxPool.
    /// </summary>
    internal sealed class AveragePool<T> : IOnnxOpTranslator<T> where T : unmanaged
    {
        public string OpType => "AveragePool";
        public string? Domain => null;
        public void Translate(OnnxTranslationContext<T> ctx, NodeProto node)
        {
            var input = ctx.GetTensor(node.Input[0]);
            var kernelShape = ctx.GetIntArrayAttr(node, "kernel_shape")
                ?? throw new InvalidDataException("AveragePool requires kernel_shape.");
            var strides = ctx.GetIntArrayAttr(node, "strides") ?? kernelShape;
            int ceilMode = ctx.GetIntAttrAsInt(node, "ceil_mode", 0);

            var explicitPads = ctx.GetIntArrayAttr(node, "pads");
            var autoPad = ctx.GetStringAttr(node, "auto_pad", null);
            var pad = OnnxAutoPad.Resolve2D("AveragePool", input._shape, kernelShape, strides, autoPad, explicitPads, ceilMode: ceilMode != 0);

            // AveragePool's padded positions contribute 0 to the sum. ONNX
            // has a count_include_pad attribute that (when 1) divides by the
            // full kernel size regardless of padding — we assume the default
            // (count_include_pad=0) and reject the other form rather than
            // compute a dynamic denominator.
            int countIncludePad = ctx.GetIntAttrAsInt(node, "count_include_pad", 0);
            if (countIncludePad != 0 && !pad.IsAllZero)
                throw new NotSupportedException(
                    "AveragePool count_include_pad=1 with non-zero padding is not yet supported.");
            var zero = default(T);
            var (paddedInput, symmetricPad) = OnnxAutoPad.ApplyAsymmetric(ctx.Engine, input, pad, zero);

            if (kernelShape[0] != kernelShape[1])
                throw new NotSupportedException(
                    $"AveragePool non-square kernel [{string.Join(",", kernelShape)}] is a Phase 2 op.");
            if (strides[0] != strides[1])
                throw new NotSupportedException(
                    $"AveragePool non-square stride [{string.Join(",", strides)}] is a Phase 2 op.");
            if (symmetricPad[0] != symmetricPad[1])
                throw new NotSupportedException(
                    $"AveragePool residual non-square padding [{symmetricPad[0]}, {symmetricPad[1]}] is a Phase 2 op.");
            var result = ctx.Engine.AvgPool2D(paddedInput, kernelShape[0], strides[0], symmetricPad[0]);
            ctx.PutTensor(node.Output[0], result);
        }
    }

    /// <summary>
    /// ONNX GlobalAveragePool — reduce mean over the spatial dims (H, W) of
    /// an NCHW input, keeping dims so the output stays 4D as ONNX requires.
    /// Implemented via <c>ReduceMean(input, axes=[2, 3], keepDims=true)</c>
    /// rather than <c>AvgPool2D</c> with a full-extent kernel because the
    /// int[] AvgPool2D overload has no GraphMode recording branch — using
    /// it under GraphMode would leave the op out of the compiled plan.
    /// </summary>
    internal sealed class GlobalAveragePool<T> : IOnnxOpTranslator<T> where T : unmanaged
    {
        public string OpType => "GlobalAveragePool";
        public string? Domain => null;
        public void Translate(OnnxTranslationContext<T> ctx, NodeProto node)
        {
            var input = ctx.GetTensor(node.Input[0]);
            if (input.Rank != 4)
                throw new NotSupportedException(
                    $"GlobalAveragePool expects 4D input; got rank {input.Rank}.");
            var result = ctx.Engine.ReduceMean(input, new[] { 2, 3 }, keepDims: true);
            ctx.PutTensor(node.Output[0], result);
        }
    }
}
