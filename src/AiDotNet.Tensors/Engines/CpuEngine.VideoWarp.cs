using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

/// <summary>
/// Internal fused backward surface for sharing a forward-splat normalization field across both
/// adjoints without expanding the public <see cref="IEngine"/> contract.
/// </summary>
internal interface IForwardSplatBackwardEngine
{
    Tensor<T> GetForwardSplatNormalizationWeights<T>(Tensor<T> flow);

    Tensor<T> ForwardSplatBackwardInputWithWeights<T>(
        Tensor<T> gradOutput, Tensor<T> input, Tensor<T> flow,
        bool normalize, Tensor<T>? normalizationWeights);

    Tensor<T> ForwardSplatBackwardFlowWithWeights<T>(
        Tensor<T> gradOutput, Tensor<T> input, Tensor<T> flow, Tensor<T>? output,
        bool normalize, Tensor<T>? normalizationWeights);
}

/// <summary>Portable video-warping primitives shared by CPU and all direct GPU backends.</summary>
public partial class CpuEngine : IForwardSplatBackwardEngine
{
    /// <inheritdoc />
    public virtual Tensor<T> PartialCorrelationVolume<T>(
        Tensor<T> first, Tensor<T> second, int radius = 4)
    {
        ValidateFeaturePair(first, second);
        if (radius < 0) throw new ArgumentOutOfRangeException(nameof(radius));
        int batch = first.Shape[0];
        int channels = first.Shape[1];
        int height = first.Shape[2];
        int width = first.Shape[3];
        int diameter = radius * 2 + 1;
        int offsets = diameter * diameter;

        if (GraphMode.IsActive && GraphMode.Current is { } scope)
        {
            var capturedFirst = first;
            var capturedSecond = second;
            return scope.RecordBinary(
                LazyNodeType.Custom,
                "PartialCorrelationVolume",
                first,
                second,
                [batch, offsets, height, width],
                (engine, output) =>
                {
                    var replay = engine.PartialCorrelationVolume(capturedFirst, capturedSecond, radius);
                    DirectGpuTensorEngine.CopyResultInto(engine, replay, output);
                },
                BackwardFunctions<T>.PartialCorrelationVolumeBackward,
                [radius]);
        }

        Tensor<T> result;
        using (GradientTape<T>.NoGrad())
        {
            // Process one displacement at a time. The previous Unfold path materialized
            // [B,C,diameter²,H,W] neighborhoods and an equally large products buffer at once;
            // radius=4 therefore held 162 feature maps before producing the final correlation.
            // Each offset now retains only one shifted map and one product, while concatenate
            // owns exactly the required [B,diameter²,H,W] result. Because this method owns one
            // explicit backward node below, the bounded implementation remains fully tape-connected.
            var paddedSecond = Pad(
                second, radius, radius, radius, radius,
                MathHelper.GetNumericOperations<T>().Zero);
            var planes = new Tensor<T>[offsets];
            int offset = 0;
            for (int offsetY = 0; offsetY < diameter; offsetY++)
            for (int offsetX = 0; offsetX < diameter; offsetX++)
            {
                var shifted = TensorSlice(
                    paddedSecond,
                    [0, 0, offsetY, offsetX],
                    [batch, channels, height, width]);
                var products = TensorMultiply(first, shifted);
                planes[offset++] = ReduceMean(products, [1], keepDims: true);
            }
            result = TensorConcatenate(planes, axis: 1);
        }

        DifferentiableOps.RecordIfActive(
            "PartialCorrelationVolume", result, [first, second],
            BackwardFunctions<T>.PartialCorrelationVolumeBackward, [radius]);
        return result;
    }

    /// <inheritdoc />
    public virtual Tensor<T> ForwardSplat<T>(
        Tensor<T> input, Tensor<T> flow, bool normalize = true)
    {
        ValidateSplatInputs(input, flow);

        if (GraphMode.IsActive && GraphMode.Current is { } scope)
        {
            var capturedInput = input;
            var capturedFlow = flow;
            return scope.RecordBinary(
                LazyNodeType.Custom,
                "ForwardSplat",
                input,
                flow,
                input.Shape.ToArray(),
                (engine, output) =>
                {
                    var replay = engine.ForwardSplat(capturedInput, capturedFlow, normalize);
                    DirectGpuTensorEngine.CopyResultInto(engine, replay, output);
                },
                BackwardFunctions<T>.ForwardSplatBackward,
                [normalize]);
        }

        var numOps = MathHelper.GetNumericOperations<T>();
        int batch = input.Shape[0];
        int channels = input.Shape[1];
        int height = input.Shape[2];
        int width = input.Shape[3];
        var accum = new Tensor<T>(input.Shape.ToArray());
        var weights = new double[batch * height * width];
        Tensor<T> contiguousInput;
        Tensor<T> contiguousFlow;
        using (GradientTape<T>.NoGrad())
        {
            contiguousInput = input.IsContiguous ? input : input.Contiguous();
            contiguousFlow = flow.IsContiguous ? flow : flow.Contiguous();
        }
        ReadOnlyMemory<T> inputMemory = contiguousInput.ReadOnlyData;
        ReadOnlyMemory<T> flowMemory = contiguousFlow.ReadOnlyData;
        Memory<T> accumMemory = accum.Data;
        int spatial = height * width;
        long totalWork = checked((long)batch * channels * spatial * 4L);

        // A batch owns disjoint accumulation and weight ranges, so batch-level partitioning has no
        // scatter conflicts and is deterministic for every thread count. Flat NCHW offsets keep the
        // channel loop contiguous within each plane and avoid the rank-4 indexer's repeated bounds,
        // stride, copy-on-write, and version machinery in the hottest four-neighbor loop.
        CpuParallelSettings.ParallelForOrSerial(0, batch, totalWork, b =>
        {
            var inputValues = inputMemory.Span;
            var flowValues = flowMemory.Span;
            var accumulationValues = accumMemory.Span;
            int batchSpatialOffset = b * spatial;
            int flowBatchOffset = b * 2 * spatial;
            int inputBatchOffset = b * channels * spatial;

            for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++)
            {
                int sourcePixel = y * width + x;
                double destinationX = x + numOps.ToDouble(flowValues[flowBatchOffset + sourcePixel]);
                double destinationY = y + numOps.ToDouble(flowValues[flowBatchOffset + spatial + sourcePixel]);
                int x0 = (int)Math.Floor(destinationX);
                int y0 = (int)Math.Floor(destinationY);
                double fractionX = destinationX - x0;
                double fractionY = destinationY - y0;
                AccumulateForwardSplatPixel(
                    inputValues, accumulationValues, weights, numOps,
                    channels, height, width, spatial, batchSpatialOffset, inputBatchOffset,
                    sourcePixel, x0, y0, (1.0 - fractionX) * (1.0 - fractionY));
                AccumulateForwardSplatPixel(
                    inputValues, accumulationValues, weights, numOps,
                    channels, height, width, spatial, batchSpatialOffset, inputBatchOffset,
                    sourcePixel, x0 + 1, y0, fractionX * (1.0 - fractionY));
                AccumulateForwardSplatPixel(
                    inputValues, accumulationValues, weights, numOps,
                    channels, height, width, spatial, batchSpatialOffset, inputBatchOffset,
                    sourcePixel, x0, y0 + 1, (1.0 - fractionX) * fractionY);
                AccumulateForwardSplatPixel(
                    inputValues, accumulationValues, weights, numOps,
                    channels, height, width, spatial, batchSpatialOffset, inputBatchOffset,
                    sourcePixel, x0 + 1, y0 + 1, fractionX * fractionY);
            }

            if (!normalize) return;
            for (int pixel = 0; pixel < spatial; pixel++)
            {
                double weight = weights[batchSpatialOffset + pixel];
                T denominator = numOps.FromDouble(weight == 0.0 ? 1.0 : weight);
                for (int channel = 0; channel < channels; channel++)
                {
                    int index = inputBatchOffset + channel * spatial + pixel;
                    accumulationValues[index] = numOps.Divide(
                        accumulationValues[index], denominator);
                }
            }
        }, deterministicSafe: true);

        var result = accum;

        DifferentiableOps.RecordIfActive(
            "ForwardSplat", result, [input, flow],
            BackwardFunctions<T>.ForwardSplatBackward, [normalize]);
        return result;
    }

    /// <inheritdoc />
    public virtual Tensor<T> ForwardSplatBackwardInput<T>(
        Tensor<T> gradOutput, Tensor<T> input, Tensor<T> flow,
        bool normalize = true)
        => ForwardSplatBackwardInputWithWeights(
            gradOutput, input, flow, normalize, normalizationWeights: null);

    /// <summary>
    /// Input adjoint using an optional normalization field already computed by the backward owner.
    /// </summary>
    internal virtual Tensor<T> ForwardSplatBackwardInputWithWeights<T>(
        Tensor<T> gradOutput, Tensor<T> input, Tensor<T> flow,
        bool normalize, Tensor<T>? normalizationWeights)
    {
        ValidateSplatInputs(input, flow);
        if (gradOutput.Shape != input.Shape)
            throw new ArgumentException("gradOutput shape must match input shape.", nameof(gradOutput));
        ValidateNormalizationWeights(normalizationWeights, input);
        var numOps = MathHelper.GetNumericOperations<T>();
        int batch = input.Shape[0], channels = input.Shape[1], height = input.Shape[2], width = input.Shape[3];
        double[]? denominators = normalize && normalizationWeights is null
            ? ComputeSplatWeights(flow, height, width, numOps)
            : null;
        var gradient = new Tensor<T>(input.Shape.ToArray());
        for (int b = 0; b < batch; b++)
        for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
        {
            GetDestination(flow, b, y, x, numOps, out int x0, out int y0, out double fx, out double fy);
            Add(x0, y0, (1 - fx) * (1 - fy)); Add(x0 + 1, y0, fx * (1 - fy));
            Add(x0, y0 + 1, (1 - fx) * fy); Add(x0 + 1, y0 + 1, fx * fy);
            void Add(int dx, int dy, double weight)
            {
                if ((uint)dx >= (uint)width || (uint)dy >= (uint)height || weight == 0) return;
                double weightSum = normalizationWeights is null
                    ? denominators?[(b * height + dy) * width + dx] ?? 1.0
                    : numOps.ToDouble(normalizationWeights[b, 0, dy, dx]);
                double divisor = normalize && weightSum != 0.0 ? weightSum : 1.0;
                T factor = numOps.FromDouble(weight / divisor);
                for (int c = 0; c < channels; c++)
                    gradient[b, c, y, x] = numOps.Add(
                        gradient[b, c, y, x], numOps.Multiply(gradOutput[b, c, dy, dx], factor));
            }
        }
        return gradient;
    }

    /// <inheritdoc />
    public virtual Tensor<T> ForwardSplatBackwardFlow<T>(
        Tensor<T> gradOutput, Tensor<T> input, Tensor<T> flow, Tensor<T> output,
        bool normalize = true)
        => ForwardSplatBackwardFlowWithWeights(
            gradOutput, input, flow, output, normalize, normalizationWeights: null);

    /// <summary>
    /// Flow adjoint using an optional normalization field already computed by the backward owner.
    /// </summary>
    internal virtual Tensor<T> ForwardSplatBackwardFlowWithWeights<T>(
        Tensor<T> gradOutput, Tensor<T> input, Tensor<T> flow, Tensor<T>? output,
        bool normalize, Tensor<T>? normalizationWeights)
    {
        ValidateSplatInputs(input, flow);
        if (gradOutput.Shape != input.Shape)
            throw new ArgumentException("gradOutput shape must match input shape.", nameof(gradOutput));
        if (normalize)
        {
            if (output is null) throw new ArgumentNullException(nameof(output));
            if (output.Shape != input.Shape)
                throw new ArgumentException("output shape must match input shape.", nameof(output));
        }
        ValidateNormalizationWeights(normalizationWeights, input);
        var numOps = MathHelper.GetNumericOperations<T>();
        int batch = input.Shape[0], channels = input.Shape[1], height = input.Shape[2], width = input.Shape[3];
        double[]? denominators = normalize && normalizationWeights is null
            ? ComputeSplatWeights(flow, height, width, numOps)
            : null;
        var gradient = new Tensor<T>(flow.Shape.ToArray());
        for (int b = 0; b < batch; b++)
        for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
        {
            GetDestination(flow, b, y, x, numOps, out int x0, out int y0, out double fx, out double fy);
            Add(x0, y0, -(1 - fy), -(1 - fx));
            Add(x0 + 1, y0, 1 - fy, -fx);
            Add(x0, y0 + 1, -fy, 1 - fx);
            Add(x0 + 1, y0 + 1, fy, fx);
            void Add(int dx, int dy, double derivativeX, double derivativeY)
            {
                if ((uint)dx >= (uint)width || (uint)dy >= (uint)height) return;
                double weightSum = normalizationWeights is null
                    ? denominators?[(b * height + dy) * width + dx] ?? 1.0
                    : numOps.ToDouble(normalizationWeights[b, 0, dy, dx]);
                double divisor = normalize && weightSum != 0.0 ? weightSum : 1.0;
                double contribution = 0.0;
                for (int c = 0; c < channels; c++)
                {
                    double source = numOps.ToDouble(input[b, c, y, x]);
                    double normalizedSource = normalize
                        ? source - numOps.ToDouble(output![b, c, dy, dx])
                        : source;
                    contribution += numOps.ToDouble(gradOutput[b, c, dy, dx]) * normalizedSource / divisor;
                }
                gradient[b, 0, y, x] = numOps.Add(
                    gradient[b, 0, y, x], numOps.FromDouble(contribution * derivativeX));
                gradient[b, 1, y, x] = numOps.Add(
                    gradient[b, 1, y, x], numOps.FromDouble(contribution * derivativeY));
            }
        }
        return gradient;
    }

    Tensor<T> IForwardSplatBackwardEngine.GetForwardSplatNormalizationWeights<T>(Tensor<T> flow)
        => GetForwardSplatNormalizationWeights(flow);

    Tensor<T> IForwardSplatBackwardEngine.ForwardSplatBackwardInputWithWeights<T>(
        Tensor<T> gradOutput, Tensor<T> input, Tensor<T> flow,
        bool normalize, Tensor<T>? normalizationWeights)
        => ForwardSplatBackwardInputWithWeights(
            gradOutput, input, flow, normalize, normalizationWeights);

    Tensor<T> IForwardSplatBackwardEngine.ForwardSplatBackwardFlowWithWeights<T>(
        Tensor<T> gradOutput, Tensor<T> input, Tensor<T> flow, Tensor<T>? output,
        bool normalize, Tensor<T>? normalizationWeights)
        => ForwardSplatBackwardFlowWithWeights(
            gradOutput, input, flow, output, normalize, normalizationWeights);

    /// <summary>
    /// Computes the safe [B,1,H,W] normalization field once for both splat adjoints. Empty
    /// destination pixels use one, matching the released average-soft-splat zero fallback.
    /// </summary>
    internal virtual Tensor<T> GetForwardSplatNormalizationWeights<T>(Tensor<T> flow)
    {
        ValidateSplatFlow(flow);
        var numOps = MathHelper.GetNumericOperations<T>();
        int batch = flow.Shape[0], height = flow.Shape[2], width = flow.Shape[3];
        var rawWeights = ComputeSplatWeights(flow, height, width, numOps);
        var result = new Tensor<T>([batch, 1, height, width]);
        var values = result.AsWritableSpan();
        for (int index = 0; index < rawWeights.Length; index++)
            values[index] = numOps.FromDouble(rawWeights[index] == 0.0 ? 1.0 : rawWeights[index]);
        return result;
    }

    private static void AccumulateForwardSplatPixel<T>(
        ReadOnlySpan<T> inputValues,
        Span<T> accumulationValues,
        double[] weights,
        INumericOperations<T> numOps,
        int channels,
        int height,
        int width,
        int spatial,
        int batchSpatialOffset,
        int inputBatchOffset,
        int sourcePixel,
        int destinationPixelX,
        int destinationPixelY,
        double weight)
    {
        if ((uint)destinationPixelX >= (uint)width ||
            (uint)destinationPixelY >= (uint)height || weight == 0.0) return;

        int destinationPixel = destinationPixelY * width + destinationPixelX;
        weights[batchSpatialOffset + destinationPixel] += weight;
        T typedWeight = numOps.FromDouble(weight);
        for (int channel = 0; channel < channels; channel++)
        {
            int sourceIndex = inputBatchOffset + channel * spatial + sourcePixel;
            int destinationIndex = inputBatchOffset + channel * spatial + destinationPixel;
            accumulationValues[destinationIndex] = numOps.Add(
                accumulationValues[destinationIndex],
                numOps.Multiply(inputValues[sourceIndex], typedWeight));
        }
    }

    private static void ValidateFeaturePair<T>(Tensor<T> first, Tensor<T> second)
    {
        if (first.Rank != 4 || second.Rank != 4 || first.Shape != second.Shape)
            throw new ArgumentException("Correlation inputs must have equal [B,C,H,W] shapes.");
    }

    private static void ValidateSplatInputs<T>(Tensor<T> input, Tensor<T> flow)
    {
        if (input.Rank != 4 || flow.Rank != 4 || flow.Shape[0] != input.Shape[0] ||
            flow.Shape[1] != 2 || flow.Shape[2] != input.Shape[2] || flow.Shape[3] != input.Shape[3])
            throw new ArgumentException("ForwardSplat requires input [B,C,H,W] and flow [B,2,H,W].");
    }

    private static void ValidateSplatFlow<T>(Tensor<T> flow)
    {
        if (flow is null) throw new ArgumentNullException(nameof(flow));
        if (flow.Rank != 4 || flow.Shape[1] != 2 || flow.Shape[2] <= 0 || flow.Shape[3] <= 0)
            throw new ArgumentException("ForwardSplat flow must have shape [B,2,H,W].", nameof(flow));
    }

    private static void ValidateNormalizationWeights<T>(
        Tensor<T>? normalizationWeights, Tensor<T> input)
    {
        if (normalizationWeights is null) return;
        if (normalizationWeights.Rank != 4 ||
            normalizationWeights.Shape[0] != input.Shape[0] ||
            normalizationWeights.Shape[1] != 1 ||
            normalizationWeights.Shape[2] != input.Shape[2] ||
            normalizationWeights.Shape[3] != input.Shape[3])
        {
            throw new ArgumentException(
                "normalizationWeights must have shape [B,1,H,W] matching input.",
                nameof(normalizationWeights));
        }
    }

    private static double[] ComputeSplatWeights<T>(
        Tensor<T> flow, int height, int width, INumericOperations<T> numOps)
    {
        int batch = flow.Shape[0];
        var weights = new double[batch * height * width];
        for (int b = 0; b < batch; b++)
        for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
        {
            GetDestination(flow, b, y, x, numOps, out int x0, out int y0, out double fx, out double fy);
            Add(x0, y0, (1 - fx) * (1 - fy)); Add(x0 + 1, y0, fx * (1 - fy));
            Add(x0, y0 + 1, (1 - fx) * fy); Add(x0 + 1, y0 + 1, fx * fy);
            void Add(int dx, int dy, double weight)
            {
                if ((uint)dx < (uint)width && (uint)dy < (uint)height)
                    weights[(b * height + dy) * width + dx] += weight;
            }
        }
        return weights;
    }

    private static void GetDestination<T>(
        Tensor<T> flow, int b, int y, int x, INumericOperations<T> numOps,
        out int x0, out int y0, out double fractionX, out double fractionY)
    {
        double destinationX = x + numOps.ToDouble(flow[b, 0, y, x]);
        double destinationY = y + numOps.ToDouble(flow[b, 1, y, x]);
        x0 = (int)Math.Floor(destinationX); y0 = (int)Math.Floor(destinationY);
        fractionX = destinationX - x0; fractionY = destinationY - y0;
    }
}
