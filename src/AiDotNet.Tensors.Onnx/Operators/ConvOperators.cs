using AiDotNet.Tensors.Onnx.Protos;

namespace AiDotNet.Tensors.Onnx.Operators;

/// <summary>
/// Resolves an ONNX 2D auto_pad attribute into a concrete symmetric [padH, padW]
/// pair. Shared helpers for Conv / MaxPool / AveragePool translators.
/// </summary>
internal static class OnnxAutoPad
{
    /// <summary>
    /// Resolve a Conv-style padding descriptor.
    /// Priority: explicit <c>pads</c> → <c>auto_pad</c> derived → zero.
    /// Rejects asymmetric resolutions (no engine support yet).
    /// </summary>
    internal static int[] Resolve2D(
        string opName,
        int[]? inputShape, int[] kernelShape, int[] strides,
        string? autoPad, int[]? explicitPads)
    {
        // Explicit pads win if provided (ONNX spec: auto_pad is ignored when
        // pads is set).
        if (explicitPads is not null)
        {
            if (explicitPads.Length != 4)
                throw new InvalidDataException(
                    $"{opName} pads attribute has {explicitPads.Length} values; expected 4 (NCHW: [top, left, bottom, right]).");
            if (explicitPads[0] != explicitPads[2] || explicitPads[1] != explicitPads[3])
                throw new NotSupportedException(
                    $"{opName} asymmetric padding [{string.Join(",", explicitPads)}] is a Phase 2 op.");
            return new[] { explicitPads[0], explicitPads[1] };
        }

        // No explicit pads. Handle auto_pad.
        switch (autoPad)
        {
            case null:
            case "":
            case "NOTSET":
            case "VALID":
                return new[] { 0, 0 };

            case "SAME_UPPER":
            case "SAME_LOWER":
                if (inputShape is null || inputShape.Length < 4)
                    throw new InvalidDataException(
                        $"{opName} auto_pad={autoPad} requires known input spatial dims.");
                int padH = TotalSamePad(inputShape[2], kernelShape[0], strides[0]);
                int padW = TotalSamePad(inputShape[3], kernelShape[1], strides[1]);
                // SAME_UPPER = extra goes to bottom/right; SAME_LOWER = to top/left.
                // Either way the spec permits asymmetric splits; we only support
                // the symmetric (even-total) case.
                if ((padH & 1) != 0 || (padW & 1) != 0)
                    throw new NotSupportedException(
                        $"{opName} auto_pad={autoPad} would require asymmetric padding [{padH}, {padW}] total; Phase 1 supports only symmetric splits.");
                return new[] { padH / 2, padW / 2 };

            default:
                throw new NotSupportedException(
                    $"{opName} auto_pad={autoPad} is not recognized. Supported: NOTSET, VALID, SAME_UPPER, SAME_LOWER.");
        }
    }

    private static int TotalSamePad(int inputSize, int kernelSize, int stride)
    {
        // Output size under SAME = ceil(input / stride) → required total pad.
        int outputSize = (inputSize + stride - 1) / stride;
        int total = Math.Max(0, (outputSize - 1) * stride + kernelSize - inputSize);
        return total;
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
    /// int[], int[])</c>. Pads are ONNX's 4-element [topH, leftW, bottomH,
    /// rightW]; we collapse to symmetric [padH, padW] and reject asymmetric
    /// padding. Group convolution is handled by slicing input/kernel per
    /// group and concatenating outputs; grouped conv is common in MobileNet
    /// but not in ResNet-50 / BERT / ViT.
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
            if (group != 1)
                throw new NotSupportedException(
                    $"Conv with group={group} (grouped / depthwise convolution) is a Phase 2 op. " +
                    "Most ResNet / BERT / ViT layers use group=1.");

            var strides = ctx.GetIntArrayAttr(node, "strides") ?? new[] { 1, 1 };
            var dilations = ctx.GetIntArrayAttr(node, "dilations") ?? new[] { 1, 1 };
            var kernelSpatial = new[] { kernel._shape[2], kernel._shape[3] };
            var explicitPads = ctx.GetIntArrayAttr(node, "pads");
            var autoPad = ctx.GetStringAttr(node, "auto_pad", null);
            var padding = OnnxAutoPad.Resolve2D("Conv", input._shape, kernelSpatial, strides, autoPad, explicitPads);

            var result = ctx.Engine.Conv2D(input, kernel,
                stride: strides,
                padding: padding,
                dilation: dilations);

            if (hasBias)
            {
                // Bias shape is [Cout]; broadcast to [1, Cout, 1, 1] for the
                // add. Reshape is cheap (storage-sharing view).
                var bias = ctx.GetTensor(node.Input[2]);
                var reshaped = ctx.Engine.Reshape(bias, new[] { 1, bias._shape[0], 1, 1 });
                result = ctx.Engine.TensorBroadcastAdd(result, reshaped);
            }
            ctx.PutTensor(node.Output[0], result);
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
                throw new NotSupportedException("ConvTranspose group != 1 is a Phase 2 op.");

            var strides = ctx.GetIntArrayAttr(node, "strides") ?? new[] { 1, 1 };
            var outputPadding = ctx.GetIntArrayAttr(node, "output_padding") ?? new[] { 0, 0 };
            var kernelSpatial = new[] { kernel._shape[2], kernel._shape[3] };
            var explicitPads = ctx.GetIntArrayAttr(node, "pads");
            var autoPad = ctx.GetStringAttr(node, "auto_pad", null);
            var padding = OnnxAutoPad.Resolve2D("ConvTranspose", input._shape, kernelSpatial, strides, autoPad, explicitPads);

            var result = ctx.Engine.ConvTranspose2D(input, kernel,
                stride: strides,
                padding: padding,
                outputPadding: outputPadding);

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
    /// dilations, ceil_mode, storage_order}. The engine supports symmetric
    /// pad and stride; asymmetric pad and ceil_mode=1 throw.
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
            if (ceilMode != 0)
                throw new NotSupportedException("MaxPool ceil_mode=1 is a Phase 2 op.");

            var explicitPads = ctx.GetIntArrayAttr(node, "pads");
            var autoPad = ctx.GetStringAttr(node, "auto_pad", null);
            var padding = OnnxAutoPad.Resolve2D("MaxPool", input._shape, kernelShape, strides, autoPad, explicitPads);

            // Engine API takes scalar poolSize/stride/padding. ResNet-50, BERT
            // and ViT all use square kernels; reject non-square here to fail
            // loudly if we ever see a real asymmetric pool.
            if (kernelShape[0] != kernelShape[1])
                throw new NotSupportedException(
                    $"MaxPool non-square kernel [{string.Join(",", kernelShape)}] is a Phase 2 op.");
            if (strides[0] != strides[1])
                throw new NotSupportedException(
                    $"MaxPool non-square stride [{string.Join(",", strides)}] is a Phase 2 op.");
            if (padding[0] != padding[1])
                throw new NotSupportedException(
                    $"MaxPool non-square padding [{padding[0]}, {padding[1]}] is a Phase 2 op.");
            var result = ctx.Engine.MaxPool2D(input, kernelShape[0], strides[0], padding[0]);
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
            if (ceilMode != 0)
                throw new NotSupportedException("AveragePool ceil_mode=1 is a Phase 2 op.");

            var explicitPads = ctx.GetIntArrayAttr(node, "pads");
            var autoPad = ctx.GetStringAttr(node, "auto_pad", null);
            var padding = OnnxAutoPad.Resolve2D("AveragePool", input._shape, kernelShape, strides, autoPad, explicitPads);

            if (kernelShape[0] != kernelShape[1])
                throw new NotSupportedException(
                    $"AveragePool non-square kernel [{string.Join(",", kernelShape)}] is a Phase 2 op.");
            if (strides[0] != strides[1])
                throw new NotSupportedException(
                    $"AveragePool non-square stride [{string.Join(",", strides)}] is a Phase 2 op.");
            if (padding[0] != padding[1])
                throw new NotSupportedException(
                    $"AveragePool non-square padding [{padding[0]}, {padding[1]}] is a Phase 2 op.");
            var result = ctx.Engine.AvgPool2D(input, kernelShape[0], strides[0], padding[0]);
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
