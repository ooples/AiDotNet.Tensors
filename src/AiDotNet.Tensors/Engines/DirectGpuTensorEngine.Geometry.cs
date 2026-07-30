// Copyright (c) AiDotNet. All rights reserved.
// GPU dispatch for the geometry / sampling ops added by Issue #217:
// Interpolate, PadNd, GridSample (extended), AffineGrid3D. Falls through
// to CpuEngine via inherited base.* when the backend does not implement
// IGeometryBackend, T != float, or the shape / mode combination isn't on
// the GPU's supported surface (1D/3D Interpolate, rank ≠ 4 PadNd, etc.).

using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

public partial class DirectGpuTensorEngine
{
    /// <inheritdoc/>
    public override Tensor<T> Interpolate<T>(Tensor<T> input, int[] sizes, InterpolateMode mode, bool alignCorners = false)
    {
        // GPU surface: T=float, 4D NCHW, modes Nearest/Bilinear/Bicubic/Area.
        if (typeof(T) == typeof(float)
            && input.Rank == 4
            && sizes.Length == 2
            && IsGpuInterpolateMode(mode))
        {
            try
            {
                if (TryGetBackend(out var backend) && backend is IGeometryBackend geom)
                {
                    int N = input._shape[0], C = input._shape[1];
                    int Hin = input._shape[2], Win = input._shape[3];
                    int Hout = sizes[0], Wout = sizes[1];
                    if (N * C * Hout * Wout == 0)
                        return new Tensor<T>(new[] { N, C, Hout, Wout });
                    using var inBuf = GetOrAllocateBuffer(backend, input);
                    var outBuf = AllocateOutputBuffer(backend, N * C * Hout * Wout);
                    try
                    {
                        geom.Interpolate2D(inBuf.Buffer, outBuf.Buffer,
                            N, C, Hin, Win, Hout, Wout, (int)mode, alignCorners);
                        var arr = FinishGpuOp<T>(backend, outBuf, N * C * Hout * Wout);
                        return new Tensor<T>(arr, new[] { N, C, Hout, Wout });
                    }
                    catch { outBuf.Dispose(); throw; }
                }
            }
            catch { }
        }
        return base.Interpolate(input, sizes, mode, alignCorners);
    }

    /// <inheritdoc/>
    public override Tensor<T> InterpolateByScale<T>(Tensor<T> input, double[] scaleFactors, InterpolateMode mode, bool alignCorners = false)
        => base.InterpolateByScale(input, scaleFactors, mode, alignCorners);

    /// <inheritdoc/>
    public override Tensor<T> PadNd<T>(Tensor<T> input, int[] pad, PadMode mode, T value = default!)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (pad is null) throw new ArgumentNullException(nameof(pad));
        if ((pad.Length & 1) != 0)
            throw new ArgumentException("pad length must be even (before/after pairs).", nameof(pad));
        int padAxes = pad.Length / 2;
        if (padAxes > input.Rank)
            throw new ArgumentException($"pad covers {padAxes} axes but input has only {input.Rank}.", nameof(pad));

        if (typeof(T) != typeof(float) || !input.IsContiguous || IsTapeActive<T>()
            || Compilation.GraphMode.IsActive || !TryGetBackend(out var backend))
            return base.PadNd(input, pad, mode, value);

        var before = new int[input.Rank];
        var after = new int[input.Rank];
        for (int i = 0; i < padAxes; i++)
        {
            int axis = input.Rank - 1 - i;
            before[axis] = pad[i * 2];
            after[axis] = pad[i * 2 + 1];
            if (before[axis] < 0 || after[axis] < 0)
                throw new ArgumentException("pad amounts must be non-negative.", nameof(pad));
        }

        var outputShape = new int[input.Rank];
        int outputLength = 1;
        for (int axis = 0; axis < input.Rank; axis++)
        {
            outputShape[axis] = checked(input._shape[axis] + before[axis] + after[axis]);
            outputLength = checked(outputLength * outputShape[axis]);
        }
        if (outputLength == 0)
            return new Tensor<T>(outputShape);

        float padValue = (float)(object)value!;
        using var inputBuffer = GetOrAllocateBuffer(backend, input);

        if (input.Rank >= 1 && input.Rank <= 4 && backend is IGeometryBackend geometry)
        {
            var dimensions = new[] { 1, 1, 1, 1 };
            var padBefore = new int[4];
            var padAfter = new int[4];
            int axisOffset = 4 - input.Rank;
            for (int axis = 0; axis < input.Rank; axis++)
            {
                dimensions[axisOffset + axis] = input._shape[axis];
                padBefore[axisOffset + axis] = before[axis];
                padAfter[axisOffset + axis] = after[axis];
            }

            return DispatchDeferredGpuOp<T>(backend, outputLength, outputShape, output =>
                geometry.Pad4D(inputBuffer.Buffer, output,
                    dimensions[0], dimensions[1], dimensions[2], dimensions[3],
                    padBefore[0], padAfter[0], padBefore[1], padAfter[1],
                    padBefore[2], padAfter[2], padBefore[3], padAfter[3],
                    (int)mode, padValue));
        }

        var sourceStrides = new int[input.Rank];
        var outputStrides = new int[input.Rank];
        sourceStrides[input.Rank - 1] = 1;
        outputStrides[input.Rank - 1] = 1;
        for (int axis = input.Rank - 2; axis >= 0; axis--)
        {
            sourceStrides[axis] = checked(sourceStrides[axis + 1] * input._shape[axis + 1]);
            outputStrides[axis] = checked(outputStrides[axis + 1] * outputShape[axis + 1]);
        }

        return DispatchDeferredGpuOp<T>(backend, outputLength, outputShape, output =>
        {
            if (mode == PadMode.Constant)
                backend.Fill(output, padValue, outputLength);

            for (int outputOffset = 0; outputOffset < outputLength; outputOffset++)
            {
                int remainder = outputOffset;
                int sourceOffset = 0;
                bool inBounds = true;
                for (int axis = 0; axis < input.Rank; axis++)
                {
                    int outputIndex = remainder / outputStrides[axis];
                    remainder %= outputStrides[axis];
                    int sourceIndex = outputIndex - before[axis];
                    int extent = input._shape[axis];
                    if (sourceIndex < 0 || sourceIndex >= extent)
                    {
                        if (mode == PadMode.Constant)
                        {
                            inBounds = false;
                            break;
                        }
                        sourceIndex = MapPadBoundary(sourceIndex, extent, mode);
                    }
                    sourceOffset += sourceIndex * sourceStrides[axis];
                }
                if (inBounds)
                    backend.Copy(inputBuffer.Buffer, sourceOffset, output, outputOffset, 1);
            }
        });
    }

    private static int MapPadBoundary(int index, int extent, PadMode mode)
    {
        if (extent <= 0) return 0;
        if (mode == PadMode.Replicate)
            return index < 0 ? 0 : (index >= extent ? extent - 1 : index);
        if (mode == PadMode.Reflect)
        {
            if (extent == 1) return 0;
            int period = 2 * (extent - 1);
            int reflected = ((index % period) + period) % period;
            return reflected < extent ? reflected : period - reflected;
        }
        if (mode == PadMode.Circular)
            return ((index % extent) + extent) % extent;
        return 0;
    }

    /// <inheritdoc/>
    public override Tensor<T> GridSample<T>(Tensor<T> input, Tensor<T> grid,
        GridSampleMode mode, GridSamplePadding padding, bool alignCorners)
    {
        if (typeof(T) == typeof(float)
            && input.Rank == 4 && grid.Rank == 4 && grid._shape[3] == 2)
        {
            try
            {
                if (TryGetBackend(out var backend) && backend is IGeometryBackend geom)
                {
                    int N = input._shape[0], C = input._shape[1], H = input._shape[2], W = input._shape[3];
                    int outH = grid._shape[1], outW = grid._shape[2];
                    int outLen = N * outH * outW * C;
                    if (outLen == 0) return new Tensor<T>(new[] { N, C, outH, outW });
                    using var inBuf = GetOrAllocateBuffer(backend, input);
                    using var grBuf = GetOrAllocateBuffer(backend, grid);
                    var outBuf = AllocateOutputBuffer(backend, outLen);
                    try
                    {
                        geom.GridSample2D(inBuf.Buffer, grBuf.Buffer, outBuf.Buffer,
                            N, H, W, C, outH, outW, (int)mode, (int)padding, alignCorners);
                        var arr = FinishGpuOp<T>(backend, outBuf, outLen);
                        return new Tensor<T>(arr, new[] { N, C, outH, outW });
                    }
                    catch { outBuf.Dispose(); throw; }
                }
            }
            catch { }
        }
        return base.GridSample(input, grid, mode, padding, alignCorners);
    }

    /// <inheritdoc/>
    public override Tensor<T> AffineGrid3D<T>(Tensor<T> theta, int outputDepth, int outputHeight, int outputWidth, bool alignCorners = false)
    {
        if (typeof(T) == typeof(float)
            && theta.Rank == 3 && theta._shape[1] == 3 && theta._shape[2] == 4)
        {
            try
            {
                if (TryGetBackend(out var backend) && backend is IGeometryBackend geom)
                {
                    int N = theta._shape[0];
                    int outLen = N * outputDepth * outputHeight * outputWidth * 3;
                    if (outLen == 0) return new Tensor<T>(new[] { N, outputDepth, outputHeight, outputWidth, 3 });
                    using var tBuf = GetOrAllocateBuffer(backend, theta);
                    var gBuf = AllocateOutputBuffer(backend, outLen);
                    try
                    {
                        geom.AffineGrid3D(tBuf.Buffer, gBuf.Buffer,
                            N, outputDepth, outputHeight, outputWidth, alignCorners);
                        var arr = FinishGpuOp<T>(backend, gBuf, outLen);
                        return new Tensor<T>(arr, new[] { N, outputDepth, outputHeight, outputWidth, 3 });
                    }
                    catch { gBuf.Dispose(); throw; }
                }
            }
            catch { }
        }
        return base.AffineGrid3D(theta, outputDepth, outputHeight, outputWidth, alignCorners);
    }

    /// <summary>
    /// GPU Interpolate2D kernel supports nearest, bilinear, bicubic and
    /// area. Linear (1D) and trilinear (3D) fall through to CPU because
    /// they need 3D / 5D tensor handling respectively.
    /// </summary>
    private static bool IsGpuInterpolateMode(InterpolateMode mode)
        => mode == InterpolateMode.Nearest
           || mode == InterpolateMode.Bilinear
           || mode == InterpolateMode.Bicubic
           || mode == InterpolateMode.Area;
}
