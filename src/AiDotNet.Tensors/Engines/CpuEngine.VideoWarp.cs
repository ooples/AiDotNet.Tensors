using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

/// <summary>Portable video-warping primitives shared by CPU and all direct GPU backends.</summary>
public partial class CpuEngine
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

        // Unfold is implemented by every backend and has a registered backward. Expressing
        // correlation through primitives keeps this operation tape-connected without a
        // model-specific gradient implementation.
        var neighborhoods = Unfold(
            second, [diameter, diameter], stride: [1, 1], padding: [radius, radius]);
        neighborhoods = Reshape(neighborhoods, [batch, channels, offsets, height, width]);
        var reference = Reshape(first, [batch, channels, 1, height, width]);
        var products = TensorMultiply(neighborhoods, reference);
        return ReduceMean(products, [1], keepDims: false);
    }

    /// <inheritdoc />
    public virtual Tensor<T> ForwardSplat<T>(
        Tensor<T> input, Tensor<T> flow, bool normalize = true, double epsilon = 1e-7)
    {
        ValidateSplatInputs(input, flow, epsilon);

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
                    var replay = engine.ForwardSplat(capturedInput, capturedFlow, normalize, epsilon);
                    DirectGpuTensorEngine.CopyResultInto(engine, replay, output);
                },
                BackwardFunctions<T>.ForwardSplatBackward,
                [normalize, epsilon]);
        }

        var numOps = MathHelper.GetNumericOperations<T>();
        int batch = input.Shape[0];
        int channels = input.Shape[1];
        int height = input.Shape[2];
        int width = input.Shape[3];
        var accum = new Tensor<T>(input.Shape.ToArray());
        var weights = new double[batch * height * width];

        for (int b = 0; b < batch; b++)
        for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
        {
            double destinationX = x + numOps.ToDouble(flow[b, 0, y, x]);
            double destinationY = y + numOps.ToDouble(flow[b, 1, y, x]);
            int x0 = (int)Math.Floor(destinationX);
            int y0 = (int)Math.Floor(destinationY);
            double fractionX = destinationX - x0;
            double fractionY = destinationY - y0;
            Accumulate(x0, y0, (1.0 - fractionX) * (1.0 - fractionY));
            Accumulate(x0 + 1, y0, fractionX * (1.0 - fractionY));
            Accumulate(x0, y0 + 1, (1.0 - fractionX) * fractionY);
            Accumulate(x0 + 1, y0 + 1, fractionX * fractionY);

            void Accumulate(int destinationPixelX, int destinationPixelY, double weight)
            {
                if ((uint)destinationPixelX >= (uint)width ||
                    (uint)destinationPixelY >= (uint)height || weight == 0.0) return;
                weights[(b * height + destinationPixelY) * width + destinationPixelX] += weight;
                T typedWeight = numOps.FromDouble(weight);
                for (int channel = 0; channel < channels; channel++)
                    accum[b, channel, destinationPixelY, destinationPixelX] = numOps.Add(
                        accum[b, channel, destinationPixelY, destinationPixelX],
                        numOps.Multiply(input[b, channel, y, x], typedWeight));
            }
        }

        var result = accum;
        if (normalize)
        {
            result = new Tensor<T>(input.Shape.ToArray());
            for (int b = 0; b < batch; b++)
            for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++)
            {
                T denominator = numOps.FromDouble(
                    weights[(b * height + y) * width + x] + epsilon);
                for (int channel = 0; channel < channels; channel++)
                    result[b, channel, y, x] = numOps.Divide(accum[b, channel, y, x], denominator);
            }
        }

        DifferentiableOps.RecordIfActive(
            "ForwardSplat", result, [input, flow],
            BackwardFunctions<T>.ForwardSplatBackward, [normalize, epsilon]);
        return result;
    }

    /// <inheritdoc />
    public virtual Tensor<T> ForwardSplatBackwardInput<T>(
        Tensor<T> gradOutput, Tensor<T> input, Tensor<T> flow,
        bool normalize = true, double epsilon = 1e-7)
    {
        ValidateSplatInputs(input, flow, epsilon);
        if (gradOutput.Shape != input.Shape)
            throw new ArgumentException("gradOutput shape must match input shape.", nameof(gradOutput));
        var numOps = MathHelper.GetNumericOperations<T>();
        int batch = input.Shape[0], channels = input.Shape[1], height = input.Shape[2], width = input.Shape[3];
        double[] denominators = ComputeSplatWeights(flow, height, width, numOps);
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
                double divisor = normalize ? denominators[(b * height + dy) * width + dx] + epsilon : 1.0;
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
        bool normalize = true, double epsilon = 1e-7)
    {
        ValidateSplatInputs(input, flow, epsilon);
        var numOps = MathHelper.GetNumericOperations<T>();
        int batch = input.Shape[0], channels = input.Shape[1], height = input.Shape[2], width = input.Shape[3];
        double[] denominators = ComputeSplatWeights(flow, height, width, numOps);
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
                double divisor = normalize ? denominators[(b * height + dy) * width + dx] + epsilon : 1.0;
                double contribution = 0.0;
                for (int c = 0; c < channels; c++)
                {
                    double source = numOps.ToDouble(input[b, c, y, x]);
                    double normalizedSource = normalize
                        ? source - numOps.ToDouble(output[b, c, dy, dx])
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

    private static void ValidateFeaturePair<T>(Tensor<T> first, Tensor<T> second)
    {
        if (first.Rank != 4 || second.Rank != 4 || first.Shape != second.Shape)
            throw new ArgumentException("Correlation inputs must have equal [B,C,H,W] shapes.");
    }

    private static void ValidateSplatInputs<T>(Tensor<T> input, Tensor<T> flow, double epsilon)
    {
        if (input.Rank != 4 || flow.Rank != 4 || flow.Shape[0] != input.Shape[0] ||
            flow.Shape[1] != 2 || flow.Shape[2] != input.Shape[2] || flow.Shape[3] != input.Shape[3])
            throw new ArgumentException("ForwardSplat requires input [B,C,H,W] and flow [B,2,H,W].");
        if (epsilon <= 0 || double.IsNaN(epsilon)) throw new ArgumentOutOfRangeException(nameof(epsilon));
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
