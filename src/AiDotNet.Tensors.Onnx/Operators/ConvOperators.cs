using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
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
    /// Generic-rank padding descriptor: 2N elements for an N-dim spatial
    /// convolution. Entries are ordered [dim0_begin, dim1_begin, ..., dim0_end,
    /// dim1_end, ...] per ONNX convention — same as <c>pads</c> attribute.
    /// </summary>
    internal readonly struct OnnxPaddingND
    {
        public int[] Begins { get; }
        public int[] Ends { get; }
        public OnnxPaddingND(int[] begins, int[] ends) { Begins = begins; Ends = ends; }
        public bool IsSymmetric
        {
            get
            {
                for (int i = 0; i < Begins.Length; i++)
                    if (Begins[i] != Ends[i]) return false;
                return true;
            }
        }
        public bool IsAllZero
        {
            get
            {
                for (int i = 0; i < Begins.Length; i++)
                    if (Begins[i] != 0 || Ends[i] != 0) return false;
                return true;
            }
        }
    }

    /// <summary>
    /// Generic N-D padding resolver for Conv / Pool over any spatial rank
    /// (2 for 2D, 3 for 3D). ONNX <c>pads</c> is [begin_0, ..., begin_{N-1},
    /// end_0, ..., end_{N-1}]; SAME_UPPER/LOWER auto-pad and ceil_mode
    /// handled symmetrically per spatial dim.
    /// </summary>
    internal static OnnxPaddingND ResolveND(
        string opName,
        int[]? inputShape, int[] kernelShape, int[] strides, int spatialRank,
        string? autoPad, int[]? explicitPads,
        bool ceilMode = false)
    {
        int[] begins = new int[spatialRank];
        int[] ends = new int[spatialRank];

        if (explicitPads is not null)
        {
            if (explicitPads.Length != 2 * spatialRank)
                throw new InvalidDataException(
                    $"{opName} pads attribute has {explicitPads.Length} values; expected {2 * spatialRank} for {spatialRank}D spatial.");
            for (int i = 0; i < spatialRank; i++)
            {
                begins[i] = explicitPads[i];
                ends[i] = explicitPads[spatialRank + i];
            }
            return MaybeAddCeilPad(new OnnxPaddingND(begins, ends), inputShape, kernelShape, strides, spatialRank, ceilMode);
        }

        switch (autoPad)
        {
            case null:
            case "":
            case "NOTSET":
            case "VALID":
                return MaybeAddCeilPad(new OnnxPaddingND(begins, ends), inputShape, kernelShape, strides, spatialRank, ceilMode);

            case "SAME_UPPER":
            case "SAME_LOWER":
                if (inputShape is null || inputShape.Length < 2 + spatialRank)
                    throw new InvalidDataException($"{opName} auto_pad={autoPad} requires known spatial dims.");
                for (int i = 0; i < spatialRank; i++)
                {
                    int total = TotalSamePad(inputShape[2 + i], kernelShape[i], strides[i]);
                    if (autoPad == "SAME_UPPER")
                    {
                        begins[i] = total / 2;
                        ends[i] = total - begins[i];
                    }
                    else
                    {
                        ends[i] = total / 2;
                        begins[i] = total - ends[i];
                    }
                }
                return new OnnxPaddingND(begins, ends);

            default:
                throw new NotSupportedException(
                    $"{opName} auto_pad={autoPad} is not recognized. Supported: NOTSET, VALID, SAME_UPPER, SAME_LOWER.");
        }
    }

    private static OnnxPaddingND MaybeAddCeilPad(
        OnnxPaddingND pad, int[]? inputShape, int[] kernelShape, int[] strides, int spatialRank, bool ceilMode)
    {
        if (!ceilMode || inputShape is null) return pad;
        var begins = (int[])pad.Begins.Clone();
        var ends = (int[])pad.Ends.Clone();
        for (int i = 0; i < spatialRank; i++)
            ends[i] += ExtraPadForCeil(inputShape[2 + i], kernelShape[i], strides[i], begins[i] + ends[i]);
        return new OnnxPaddingND(begins, ends);
    }

    /// <summary>
    /// N-D asymmetric padding applicator. Factors out the symmetric
    /// minimum (pass to engine) and applies the remaining asymmetric
    /// part via <see cref="IEngine.Pad"/> (2D only) or an equivalent
    /// manual pad (3D). Returns (padded input, symmetric residual per-dim).
    /// </summary>
    internal static (Tensor<T> input, int[] symmetricPad) ApplyAsymmetricND<T>(
        Engines.IEngine engine, Tensor<T> input, OnnxPaddingND pad, int spatialRank, T padValue) where T : unmanaged
    {
        if (pad.IsSymmetric)
            return (input, pad.Begins);

        int[] sym = new int[spatialRank];
        int[] extraBegin = new int[spatialRank];
        int[] extraEnd = new int[spatialRank];
        bool anyExtra = false;
        for (int i = 0; i < spatialRank; i++)
        {
            sym[i] = Math.Min(pad.Begins[i], pad.Ends[i]);
            extraBegin[i] = pad.Begins[i] - sym[i];
            extraEnd[i] = pad.Ends[i] - sym[i];
            if (extraBegin[i] != 0 || extraEnd[i] != 0) anyExtra = true;
        }
        if (!anyExtra) return (input, sym);

        if (spatialRank == 2)
        {
            var padded = engine.Pad(input, extraBegin[0], extraEnd[0], extraBegin[1], extraEnd[1], padValue);
            return (padded, sym);
        }
        // 3D: use TensorConstantPad which handles any rank. Engine's
        // TensorConstantPad takes a per-dim pair: [begin_0, end_0, ..., begin_{R-1}, end_{R-1}].
        // Input is NCDHW (rank 5); pad only the spatial dims (D, H, W).
        var fullPad = new int[input.Rank * 2];
        for (int i = 0; i < spatialRank; i++)
        {
            fullPad[(2 + i) * 2] = extraBegin[i];
            fullPad[(2 + i) * 2 + 1] = extraEnd[i];
        }
        var paddedND = engine.TensorConstantPad(input, fullPad, padValue);
        return (paddedND, sym);
    }

    /// <summary>
    /// Pre-pads the input along every spatial axis with the FULL
    /// (asymmetric-aware) amount so the subsequent pool/conv can run with
    /// engine-level padding=0. Used by <c>AveragePool</c> with
    /// <c>count_include_pad=1</c>: moving the padding out of the engine's
    /// pool makes the kernel-window denominator always equal the full
    /// kernel area (the "include-pad" semantic ONNX requires).
    /// </summary>
    internal static Tensor<T> ApplyPadAllAxes<T>(
        Engines.IEngine engine, Tensor<T> input, OnnxPaddingND pad, int spatialRank, T padValue) where T : unmanaged
    {
        bool nonZero = false;
        for (int i = 0; i < spatialRank; i++)
            if (pad.Begins[i] != 0 || pad.Ends[i] != 0) { nonZero = true; break; }
        if (!nonZero) return input;
        if (spatialRank == 2)
            return engine.Pad(input, pad.Begins[0], pad.Ends[0], pad.Begins[1], pad.Ends[1], padValue);
        var fullPad = new int[input.Rank * 2];
        for (int i = 0; i < spatialRank; i++)
        {
            fullPad[(2 + i) * 2] = pad.Begins[i];
            fullPad[(2 + i) * 2 + 1] = pad.Ends[i];
        }
        return engine.TensorConstantPad(input, fullPad, padValue);
    }

    /// <summary>
    /// Legacy 2D wrapper retained for callers that want the 2D-specific
    /// struct shape. New code should use <see cref="ResolveND"/> directly.
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
            // Phase 1 spec (Issue #169): Conv 2D + 3D.
            // NCHW   → rank 4 → 2 spatial dims
            // NCDHW  → rank 5 → 3 spatial dims
            if (input.Rank != 4 && input.Rank != 5)
                throw new NotSupportedException(
                    $"Conv supports 4D (NCHW) or 5D (NCDHW) input; got rank {input.Rank}.");
            if (kernel.Rank != input.Rank)
                throw new InvalidDataException(
                    $"Conv kernel rank {kernel.Rank} must match input rank {input.Rank}.");

            int spatialRank = input.Rank - 2;
            int group = ctx.GetIntAttrAsInt(node, "group", 1);
            var strides = ctx.GetIntArrayAttr(node, "strides") ?? Ones(spatialRank);
            var dilations = ctx.GetIntArrayAttr(node, "dilations") ?? Ones(spatialRank);
            var kernelSpatial = new int[spatialRank];
            for (int i = 0; i < spatialRank; i++) kernelSpatial[i] = kernel._shape[2 + i];
            var explicitPads = ctx.GetIntArrayAttr(node, "pads");
            var autoPad = ctx.GetStringAttr(node, "auto_pad", null);
            var pad = OnnxAutoPad.ResolveND("Conv", input._shape, kernelSpatial, strides, spatialRank, autoPad, explicitPads);

            // Apply asymmetric component via explicit Pad (zero-valued);
            // Conv gets the symmetric residual.
            var zero = default(T);
            var (paddedInput, symmetricPad) = OnnxAutoPad.ApplyAsymmetricND(ctx.Engine, input, pad, spatialRank, zero);

            // Layout policy (C10): auto-promote to NCHWc8 when the shape
            // qualifies and we can keep the whole Conv in the packed kernel.
            // The *with-bias* case needs an NCHWc-aware bias add (coming as
            // ConvBiasBn fusion in C9); until then, only pack when bias is
            // absent so the unfused bias-add doesn't see a packed tensor.
            // Also skipped under GraphMode — the eager ReorderToNchwc call
            // forces the padded-input placeholder to realize before the plan
            // has filled it, baking zeros into the packed buffer.
            if (typeof(T) == typeof(float)
                && spatialRank == 2
                && group == 1
                && !hasBias
                && !GraphMode.IsActive
                && !ctx.Options.DisableNchwcAutoPromote
                && paddedInput.Layout == LinearAlgebra.TensorLayout.Nchw
                && paddedInput._shape[1] % LayoutPlanner.PreferredCBlock == 0
                && kernel._shape[0] % LayoutPlanner.PreferredCBlock == 0)
            {
                paddedInput = ctx.Engine.ReorderToNchwc(paddedInput, LinearAlgebra.TensorLayout.Nchwc8);
            }

            Tensor<T> result;
            if (group == 1)
            {
                result = spatialRank == 2
                    ? ctx.Engine.Conv2D(paddedInput, kernel, stride: strides, padding: symmetricPad, dilation: dilations)
                    : ctx.Engine.Conv3D(paddedInput, kernel, stride: strides, padding: symmetricPad, dilation: dilations);
            }
            else
            {
                result = GroupedConv(ctx, paddedInput, kernel, group, strides, symmetricPad, dilations, spatialRank);
            }

            if (hasBias)
            {
                var bias = ctx.GetTensor(node.Input[2]);
                var biasShape = new int[input.Rank];
                biasShape[0] = 1; biasShape[1] = bias._shape[0];
                for (int i = 2; i < input.Rank; i++) biasShape[i] = 1;
                var reshaped = ctx.Engine.Reshape(bias, biasShape);
                result = ctx.Engine.TensorBroadcastAdd(result, reshaped);
            }
            ctx.PutTensor(node.Output[0], result);
        }

        private static int[] Ones(int n)
        {
            var r = new int[n];
            for (int i = 0; i < n; i++) r[i] = 1;
            return r;
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
            int group, int[] strides, int[] symmetricPad, int[] dilations, int spatialRank)
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
                var inputStart = new int[input.Rank];
                inputStart[1] = g * cInPerGroup;
                var inputLen = (int[])input._shape.Clone();
                inputLen[1] = cInPerGroup;
                var kernelStart = new int[kernel.Rank];
                kernelStart[0] = g * cOutPerGroup;
                var kernelLen = (int[])kernel._shape.Clone();
                kernelLen[0] = cOutPerGroup;

                var inputSlice = ctx.Engine.TensorSlice(input, inputStart, inputLen);
                var kernelSlice = ctx.Engine.TensorSlice(kernel, kernelStart, kernelLen);
                var groupResult = spatialRank == 2
                    ? ctx.Engine.Conv2D(inputSlice, kernelSlice, stride: strides, padding: symmetricPad, dilation: dilations)
                    : ctx.Engine.Conv3D(inputSlice, kernelSlice, stride: strides, padding: symmetricPad, dilation: dilations);
                outputs.Add(groupResult);
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
            if (input.Rank != 4 && input.Rank != 5)
                throw new NotSupportedException(
                    $"ConvTranspose supports 4D (NCHW) or 5D (NCDHW) input; got rank {input.Rank}.");

            int spatialRank = input.Rank - 2;
            int group = ctx.GetIntAttrAsInt(node, "group", 1);

            var strides = ctx.GetIntArrayAttr(node, "strides") ?? new int[spatialRank];
            for (int i = 0; i < strides.Length; i++) if (strides[i] == 0) strides[i] = 1;
            var outputPadding = ctx.GetIntArrayAttr(node, "output_padding") ?? new int[spatialRank];
            var kernelSpatial = new int[spatialRank];
            for (int i = 0; i < spatialRank; i++) kernelSpatial[i] = kernel._shape[2 + i];
            var explicitPads = ctx.GetIntArrayAttr(node, "pads");
            var autoPad = ctx.GetStringAttr(node, "auto_pad", null);
            var pad = OnnxAutoPad.ResolveND("ConvTranspose", input._shape, kernelSpatial, strides, spatialRank, autoPad, explicitPads);

            // For ConvTranspose, "padding" in ONNX crops the output from both
            // ends. The engine's ConvTranspose takes a single "padding" value
            // applied symmetrically (pad on each end). For asymmetric pad we
            // pass the larger (or symmetric residual) to the engine, then
            // slice the result to remove any extra crop.
            int[] symPad = new int[spatialRank];
            int[] extraBegin = new int[spatialRank];
            int[] extraEnd = new int[spatialRank];
            bool anyAsym = false;
            for (int i = 0; i < spatialRank; i++)
            {
                symPad[i] = Math.Min(pad.Begins[i], pad.Ends[i]);
                extraBegin[i] = pad.Begins[i] - symPad[i];
                extraEnd[i] = pad.Ends[i] - symPad[i];
                if (extraBegin[i] != 0 || extraEnd[i] != 0) anyAsym = true;
            }

            Tensor<T> result;
            if (group == 1)
            {
                result = spatialRank == 2
                    ? ctx.Engine.ConvTranspose2D(input, kernel, stride: strides, padding: symPad, outputPadding: outputPadding)
                    : ctx.Engine.ConvTranspose3D(input, kernel, stride: strides, padding: symPad, outputPadding: outputPadding);
            }
            else
            {
                // Grouped/depthwise ConvTranspose: split input along C_in and
                // kernel along the group axis of its first dim (ONNX packs
                // [C_in, C_out/group, kH, kW]). Each group gets a slice of
                // input and a slice of kernel, produce a per-group output,
                // concat along C_out.
                if (input._shape[1] % group != 0)
                    throw new InvalidDataException(
                        $"ConvTranspose Cin ({input._shape[1]}) must be divisible by group ({group}).");
                if (kernel._shape[0] % group != 0)
                    throw new InvalidDataException(
                        $"ConvTranspose kernel first dim ({kernel._shape[0]}) must be divisible by group ({group}).");
                int cInPerGroup = input._shape[1] / group;
                int kFirstPerGroup = kernel._shape[0] / group;
                var parts = new List<Tensor<T>>(group);
                for (int g = 0; g < group; g++)
                {
                    var inSliceStart = new int[input.Rank];
                    var inSliceLen = (int[])input._shape.Clone();
                    inSliceStart[1] = g * cInPerGroup;
                    inSliceLen[1] = cInPerGroup;
                    var inSlice = ctx.Engine.TensorSlice(input, inSliceStart, inSliceLen);

                    var kSliceStart = new int[kernel.Rank];
                    var kSliceLen = (int[])kernel._shape.Clone();
                    kSliceStart[0] = g * kFirstPerGroup;
                    kSliceLen[0] = kFirstPerGroup;
                    var kSlice = ctx.Engine.TensorSlice(kernel, kSliceStart, kSliceLen);

                    var groupOut = spatialRank == 2
                        ? ctx.Engine.ConvTranspose2D(inSlice, kSlice, stride: strides, padding: symPad, outputPadding: outputPadding)
                        : ctx.Engine.ConvTranspose3D(inSlice, kSlice, stride: strides, padding: symPad, outputPadding: outputPadding);
                    parts.Add(groupOut);
                }
                result = ctx.Engine.Concat(parts, axis: 1);
            }

            // Apply asymmetric extra crop as a slice if needed.
            if (anyAsym)
            {
                var cropStart = new int[result.Rank];
                var cropLen = (int[])result._shape.Clone();
                for (int i = 0; i < spatialRank; i++)
                {
                    cropStart[2 + i] = extraBegin[i];
                    cropLen[2 + i] = Math.Max(0, result._shape[2 + i] - extraBegin[i] - extraEnd[i]);
                }
                result = ctx.Engine.TensorSlice(result, cropStart, cropLen);
            }

            if (hasBias)
            {
                var bias = ctx.GetTensor(node.Input[2]);
                var biasShape = new int[input.Rank];
                biasShape[0] = 1; biasShape[1] = bias._shape[0];
                for (int i = 2; i < input.Rank; i++) biasShape[i] = 1;
                var reshaped = ctx.Engine.Reshape(bias, biasShape);
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
            if (input.Rank != 4 && input.Rank != 5)
                throw new NotSupportedException($"MaxPool supports 4D or 5D input; got rank {input.Rank}.");
            int spatialRank = input.Rank - 2;
            var kernelShape = ctx.GetIntArrayAttr(node, "kernel_shape")
                ?? throw new InvalidDataException("MaxPool requires kernel_shape.");
            var strides = ctx.GetIntArrayAttr(node, "strides") ?? kernelShape;
            int ceilMode = ctx.GetIntAttrAsInt(node, "ceil_mode", 0);

            var explicitPads = ctx.GetIntArrayAttr(node, "pads");
            var autoPad = ctx.GetStringAttr(node, "auto_pad", null);
            var pad = OnnxAutoPad.ResolveND("MaxPool", input._shape, kernelShape, strides, spatialRank, autoPad, explicitPads, ceilMode: ceilMode != 0);

            var padValue = MathHelper.GetNumericOperations<T>().FromDouble(double.NegativeInfinity);
            var (paddedInput, symmetricPad) = OnnxAutoPad.ApplyAsymmetricND(ctx.Engine, input, pad, spatialRank, padValue);

            if (spatialRank == 2)
            {
                bool squareKernel = kernelShape[0] == kernelShape[1];
                bool squareStride = strides[0] == strides[1];
                bool squarePad = symmetricPad[0] == symmetricPad[1];
                if (squareKernel && squareStride && squarePad)
                {
                    ctx.PutTensor(node.Output[0], ctx.Engine.MaxPool2D(paddedInput, kernelShape[0], strides[0], symmetricPad[0]));
                }
                else
                {
                    // Non-square 2D → promote to 5D (treat as depth=1 3D) and
                    // call the 3D path which takes int[] per-axis parameters.
                    var in4 = paddedInput._shape; // [N, C, H, W]
                    var promoted = ctx.Engine.Reshape(paddedInput, new[] { in4[0], in4[1], 1, in4[2], in4[3] });
                    var pool3D = ctx.Engine.MaxPool3D(promoted,
                        poolSize: new[] { 1, kernelShape[0], kernelShape[1] },
                        stride: new[] { 1, strides[0], strides[1] },
                        padding: new[] { 0, symmetricPad[0], symmetricPad[1] });
                    var o5 = pool3D._shape; // [N, C, 1, oH, oW]
                    var demoted = ctx.Engine.Reshape(pool3D, new[] { o5[0], o5[1], o5[3], o5[4] });
                    ctx.PutTensor(node.Output[0], demoted);
                }
            }
            else
            {
                ctx.PutTensor(node.Output[0], ctx.Engine.MaxPool3D(paddedInput, poolSize: kernelShape, stride: strides, padding: symmetricPad));
            }
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
            if (input.Rank != 4 && input.Rank != 5)
                throw new NotSupportedException($"AveragePool supports 4D or 5D input; got rank {input.Rank}.");
            int spatialRank = input.Rank - 2;
            var kernelShape = ctx.GetIntArrayAttr(node, "kernel_shape")
                ?? throw new InvalidDataException("AveragePool requires kernel_shape.");
            var strides = ctx.GetIntArrayAttr(node, "strides") ?? kernelShape;
            int ceilMode = ctx.GetIntAttrAsInt(node, "ceil_mode", 0);

            var explicitPads = ctx.GetIntArrayAttr(node, "pads");
            var autoPad = ctx.GetStringAttr(node, "auto_pad", null);
            var pad = OnnxAutoPad.ResolveND("AveragePool", input._shape, kernelShape, strides, spatialRank, autoPad, explicitPads, ceilMode: ceilMode != 0);

            int countIncludePad = ctx.GetIntAttrAsInt(node, "count_include_pad", 0);
            var zero = default(T);
            // count_include_pad=1 makes the average denominator the full
            // kernel area at every output cell (padded zeros count). Engine
            // AvgPool's "padding" param implements the count_include_pad=0
            // semantics (padded cells excluded from denominator), so when
            // count_include_pad=1 and pad is non-zero we pre-pad manually
            // with zeros and drop engine-level padding to 0 — that way the
            // kernel window always sees the full area and divides by it.
            Tensor<T> paddedInput;
            int[] symmetricPad;
            if (countIncludePad != 0 && !pad.IsAllZero)
            {
                paddedInput = OnnxAutoPad.ApplyPadAllAxes(ctx.Engine, input, pad, spatialRank, zero);
                symmetricPad = new int[spatialRank]; // zero-padding from the engine's POV.
            }
            else
            {
                (paddedInput, symmetricPad) = OnnxAutoPad.ApplyAsymmetricND(ctx.Engine, input, pad, spatialRank, zero);
            }

            if (spatialRank == 2)
            {
                bool squareKernel = kernelShape[0] == kernelShape[1];
                bool squareStride = strides[0] == strides[1];
                bool squarePad = symmetricPad[0] == symmetricPad[1];
                if (squareKernel && squareStride && squarePad)
                {
                    ctx.PutTensor(node.Output[0], ctx.Engine.AvgPool2D(paddedInput, kernelShape[0], strides[0], symmetricPad[0]));
                }
                else
                {
                    // Non-square 2D → promote to 5D and use the 3D path.
                    var in4 = paddedInput._shape;
                    var promoted = ctx.Engine.Reshape(paddedInput, new[] { in4[0], in4[1], 1, in4[2], in4[3] });
                    var pool3D = ctx.Engine.AvgPool3D(promoted,
                        poolSize: new[] { 1, kernelShape[0], kernelShape[1] },
                        stride: new[] { 1, strides[0], strides[1] },
                        padding: new[] { 0, symmetricPad[0], symmetricPad[1] });
                    var o5 = pool3D._shape;
                    var demoted = ctx.Engine.Reshape(pool3D, new[] { o5[0], o5[1], o5[3], o5[4] });
                    ctx.PutTensor(node.Output[0], demoted);
                }
            }
            else
            {
                ctx.PutTensor(node.Output[0], ctx.Engine.AvgPool3D(paddedInput, poolSize: kernelShape, stride: strides, padding: symmetricPad));
            }
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
