// Copyright (c) AiDotNet. All rights reserved.

using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

/// <summary>
/// GPU-resident video-warp primitives shared by CUDA, HIP, Metal, OpenCL, Vulkan, and WebGPU.
/// The implementation composes the existing unfold and grid-sample kernels so all six backends
/// use the same numerics and no backend-specific model code is required.
/// </summary>
public partial class DirectGpuTensorEngine
{
    /// <inheritdoc />
    public override Tensor<T> PartialCorrelationVolume<T>(
        Tensor<T> first, Tensor<T> second, int radius = 4)
    {
        // CpuEngine's implementation is a pure composition of virtual Unfold, Reshape,
        // TensorMultiply, and ReduceMean calls. On this instance those dispatch to the resident
        // GPU implementations, while their normal tape nodes provide the exact gradient.
        return base.PartialCorrelationVolume(first, second, radius);
    }

    /// <inheritdoc />
    public override Tensor<T> ForwardSplat<T>(
        Tensor<T> input, Tensor<T> flow, bool normalize = true)
    {
        ValidateForwardSplat(input, flow);
        if (typeof(T) != typeof(float) || Compilation.GraphMode.IsActive || !TryGetBackend(out _))
            return base.ForwardSplat(input, flow, normalize);

        Tensor<T> result;
        using (GradientTape<T>.NoGrad())
            result = ForwardSplatGpu(input, flow, normalize);

        // The composed implementation deliberately runs under NoGrad so the public operation owns
        // one stable tape node rather than leaking its grid-sample implementation details.
        DifferentiableOps.RecordIfActive(
            "ForwardSplat", result, [input, flow],
            BackwardFunctions<T>.ForwardSplatBackward, [normalize]);
        return result;
    }

    /// <inheritdoc />
    public override Tensor<T> ForwardSplatBackwardInput<T>(
        Tensor<T> gradOutput, Tensor<T> input, Tensor<T> flow,
        bool normalize = true)
    {
        ValidateForwardSplat(input, flow);
        ValidateSameShape(gradOutput, input, nameof(gradOutput));
        if (typeof(T) != typeof(float) || !TryGetBackend(out _))
            return base.ForwardSplatBackwardInput(gradOutput, input, flow, normalize);

        using (GradientTape<T>.NoGrad())
        {
            var grid = BuildForwardSplatGrid(flow);
            Tensor<T> sampledGradient = gradOutput;
            if (normalize)
            {
                var denominator = BuildForwardSplatDenominator(flow, grid);
                sampledGradient = TensorBroadcastDivide(gradOutput, denominator);
            }

            // A splat's input adjoint is a bilinear gather from the destination gradient.
            return GridSample(sampledGradient, grid,
                GridSampleMode.Bilinear, GridSamplePadding.Zeros, alignCorners: false);
        }
    }

    /// <inheritdoc />
    public override Tensor<T> ForwardSplatBackwardFlow<T>(
        Tensor<T> gradOutput, Tensor<T> input, Tensor<T> flow, Tensor<T> output,
        bool normalize = true)
    {
        ValidateForwardSplat(input, flow);
        ValidateSameShape(gradOutput, input, nameof(gradOutput));
        ValidateSameShape(output, input, nameof(output));
        if (typeof(T) != typeof(float) || !TryGetBackend(out _))
            return base.ForwardSplatBackwardFlow(gradOutput, input, flow, output, normalize);

        using (GradientTape<T>.NoGrad())
        {
            var grid = BuildForwardSplatGrid(flow);
            Tensor<T> destinationGradient = gradOutput;
            if (normalize)
            {
                var denominator = BuildForwardSplatDenominator(flow, grid);
                destinationGradient = TensorBroadcastDivide(gradOutput, denominator);
            }

            // GridSampleBackwardGrid differentiates a gather. Its adjoint identity gives the splat
            // flow derivative without atomics or a backend-specific kernel:
            //   dL/dp_i = d grid_sample(G, p_i)/dp_i weighted by source x_i.
            var flowGradient = GridSampleBackwardGrid(
                input, destinationGradient, grid,
                GridSampleMode.Bilinear, GridSamplePadding.Zeros, alignCorners: false);

            if (normalize)
            {
                // Quotient rule: subtract d grid_sample(sum_c(G_c * output_c), p_i)/dp_i.
                var weightedOutput = TensorMultiply(destinationGradient, output);
                var correctionField = ReduceSum(weightedOutput, [1], keepDims: true);
                var ones = CreateFilledTensor<T>(
                    [input.Shape[0], 1, input.Shape[2], input.Shape[3]], 1.0);
                var correction = GridSampleBackwardGrid(
                    ones, correctionField, grid,
                    GridSampleMode.Bilinear, GridSamplePadding.Zeros, alignCorners: false);
                flowGradient = TensorSubtract(flowGradient, correction);
            }

            // The grid stores normalized coordinates. Convert dL/d(grid) to dL/d(pixel flow).
            var pixelScale = CreateAxisTensor<T>(
                2.0 / input.Shape[3], 2.0 / input.Shape[2]);
            var scaledGradient = TensorBroadcastMultiply(flowGradient, pixelScale);

            // GridSampleBackwardGrid follows the grid contract and returns NHWC coordinates.
            // ForwardSplat's public flow contract is NCHW, so restore that layout before
            // returning the gradient to the tape. Keeping NHWC here silently attached the
            // wrong shape to every GPU-trained flow tensor.
            return TensorPermute(scaledGradient, [0, 3, 1, 2]).Contiguous();
        }
    }

    private Tensor<T> ForwardSplatGpu<T>(
        Tensor<T> input, Tensor<T> flow, bool normalize)
    {
        var grid = BuildForwardSplatGrid(flow);
        var accumulated = GridSampleBackwardInput(
            input, grid, input.Shape.ToArray(),
            GridSampleMode.Bilinear, GridSamplePadding.Zeros, alignCorners: false);
        if (!normalize) return accumulated;

        var denominator = BuildForwardSplatDenominator(flow, grid);
        return TensorBroadcastDivide(accumulated, denominator);
    }

    private Tensor<T> BuildForwardSplatDenominator<T>(
        Tensor<T> flow, Tensor<T> grid)
    {
        var ones = CreateFilledTensor<T>(
            [flow.Shape[0], 1, flow.Shape[2], flow.Shape[3]], 1.0);
        var weights = GridSampleBackwardInput(
            ones, grid, ones.Shape.ToArray(),
            GridSampleMode.Bilinear, GridSamplePadding.Zeros, alignCorners: false);
        var ops = MathHelper.GetNumericOperations<T>();
        return TensorWhere(TensorGreaterThan(weights, ops.Zero), weights, ones);
    }

    private Tensor<T> BuildForwardSplatGrid<T>(Tensor<T> flow)
    {
        int batch = flow.Shape[0], height = flow.Shape[2], width = flow.Shape[3];
        var baseGrid = new Tensor<T>([batch, height, width, 2]);
        var values = baseGrid.AsWritableSpan();
        var ops = MathHelper.GetNumericOperations<T>();
        for (int b = 0; b < batch; b++)
        for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
        {
            int offset = ((b * height + y) * width + x) * 2;
            values[offset] = ops.FromDouble(2.0 * (x + 0.5) / width - 1.0);
            values[offset + 1] = ops.FromDouble(2.0 * (y + 0.5) / height - 1.0);
        }

        var flowNhwc = TensorPermute(flow, [0, 2, 3, 1]);
        var pixelScale = CreateAxisTensor<T>(2.0 / width, 2.0 / height);
        return TensorAdd(baseGrid, TensorBroadcastMultiply(flowNhwc, pixelScale));
    }

    private static Tensor<T> CreateAxisTensor<T>(double x, double y)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        return new Tensor<T>([ops.FromDouble(x), ops.FromDouble(y)], [1, 1, 1, 2]);
    }

    private static Tensor<T> CreateFilledTensor<T>(int[] shape, double value)
    {
        var tensor = new Tensor<T>(shape);
        tensor.AsWritableSpan().Fill(MathHelper.GetNumericOperations<T>().FromDouble(value));
        return tensor;
    }

    private static void ValidateForwardSplat<T>(Tensor<T> input, Tensor<T> flow)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (flow is null) throw new ArgumentNullException(nameof(flow));
        if (input.Rank != 4 || flow.Rank != 4 || flow.Shape[0] != input.Shape[0] ||
            flow.Shape[1] != 2 || flow.Shape[2] != input.Shape[2] || flow.Shape[3] != input.Shape[3])
            throw new ArgumentException("ForwardSplat requires input [B,C,H,W] and flow [B,2,H,W].");
        if (input.Shape[2] <= 0 || input.Shape[3] <= 0)
            throw new ArgumentException("ForwardSplat spatial dimensions must be positive.", nameof(input));
    }

    private static void ValidateSameShape<T>(Tensor<T> actual, Tensor<T> expected, string parameterName)
    {
        if (actual is null) throw new ArgumentNullException(nameof(actual));
        if (expected is null) throw new ArgumentNullException(nameof(expected));
        if (actual.Shape != expected.Shape)
            throw new ArgumentException("Tensor shape must match the input shape.", parameterName);
    }
}
