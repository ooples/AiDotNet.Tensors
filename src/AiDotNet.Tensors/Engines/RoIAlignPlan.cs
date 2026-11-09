using System;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

/// <summary>
/// The bilinear taps of torchvision's <c>roi_align</c>, computed once per call: for every RoI and output bin, the
/// plane offsets and weights (already divided by the sampling grid area) its value is a weighted sum of.
/// </summary>
/// <remarks>
/// <para>
/// The forward is then a gather, parallel over (RoI, channel), and the backward a scatter, parallel over channels so
/// overlapping RoIs never write the same element from two threads. The previous kernels recomputed every sample
/// position for every channel, read and wrote one element at a time through <c>INumericOperations</c>, and ran on
/// one thread: a FasterRCNN Detect over 1,000 proposals spent most of its RoI time there.
/// </para>
/// <para>
/// The sampling rule is torchvision's exactly: samples at <c>(i + 0.5) / ratio</c> of each bin, a sample outside
/// <c>[-1, size]</c> contributes zero but still counts in the average, coordinates in <c>(-1, 0]</c> clamp to 0, and the
/// last row and column clamp in place. <c>aligned</c> shifts by half a pixel and drops the minimum RoI size of 1.
/// </para>
/// </remarks>
internal sealed class RoIAlignPlan
{
    private RoIAlignPlan(int rois, int bins, int[] batch, int[] tapStart, int[] offsets, double[] weights)
    {
        Rois = rois;
        Bins = bins;
        Batch = batch;
        TapStart = tapStart;
        Offsets = offsets;
        Weights = weights;
    }

    public int Rois { get; }
    public int Bins { get; }

    /// <summary>The image each RoI reads.</summary>
    public int[] Batch { get; }

    /// <summary>Bin <c>r * Bins + bin</c> owns taps <c>[TapStart[i], TapStart[i + 1])</c>.</summary>
    public int[] TapStart { get; }

    /// <summary>Offset of each tap within a channel plane.</summary>
    public int[] Offsets { get; }

    /// <summary>Weight of each tap, including the 1 / (ratioY * ratioX) average.</summary>
    public double[] Weights { get; }

    public static RoIAlignPlan Build(double[] boxes, int images, int height, int width,
        int outputHeight, int outputWidth, float spatialScale, int samplingRatio, bool aligned)
    {
        int rois = boxes.Length / 5;
        int bins = outputHeight * outputWidth;
        var batch = new int[rois];
        var tapStart = new int[(rois * bins) + 1];
        var offsets = new System.Collections.Generic.List<int>(rois * bins * 16);
        var weights = new System.Collections.Generic.List<double>(rois * bins * 16);
        double shift = aligned ? 0.5 : 0.0;

        for (int r = 0; r < rois; r++)
        {
            int image = (int)boxes[r * 5];
            if (image < 0 || image >= images)
                throw new ArgumentException($"RoIAlign batch idx {image} out of range [0, {images}).");
            batch[r] = image;
            double x1 = (boxes[(r * 5) + 1] * spatialScale) - shift;
            double y1 = (boxes[(r * 5) + 2] * spatialScale) - shift;
            double x2 = (boxes[(r * 5) + 3] * spatialScale) - shift;
            double y2 = (boxes[(r * 5) + 4] * spatialScale) - shift;
            double roiW = aligned ? (x2 - x1) : Math.Max(x2 - x1, 1.0);
            double roiH = aligned ? (y2 - y1) : Math.Max(y2 - y1, 1.0);
            double binH = roiH / outputHeight, binW = roiW / outputWidth;
            int ry = Math.Max(1, samplingRatio > 0 ? samplingRatio : (int)Math.Ceiling(roiH / outputHeight));
            int rx = Math.Max(1, samplingRatio > 0 ? samplingRatio : (int)Math.Ceiling(roiW / outputWidth));
            double average = 1.0 / (ry * rx);

            for (int ph = 0; ph < outputHeight; ph++)
            {
                for (int pw = 0; pw < outputWidth; pw++)
                {
                    tapStart[(r * bins) + (ph * outputWidth) + pw] = offsets.Count;
                    for (int iy = 0; iy < ry; iy++)
                    {
                        double sy = y1 + (ph * binH) + ((iy + 0.5) * binH / ry);
                        for (int ix = 0; ix < rx; ix++)
                        {
                            double sx = x1 + (pw * binW) + ((ix + 0.5) * binW / rx);
                            AddBilinear(offsets, weights, sy, sx, height, width, average);
                        }
                    }
                }
            }
        }

        tapStart[rois * bins] = offsets.Count;
        return new RoIAlignPlan(rois, bins, batch, tapStart, offsets.ToArray(), weights.ToArray());
    }

    private static void AddBilinear(System.Collections.Generic.List<int> offsets, System.Collections.Generic.List<double> weights,
        double y, double x, int height, int width, double scale)
    {
        if (y < -1.0 || y > height || x < -1.0 || x > width) return;
        if (y <= 0) y = 0;
        if (x <= 0) x = 0;
        int y0 = (int)y, x0 = (int)x;
        int y1 = y0 + 1 >= height ? height - 1 : y0 + 1;
        int x1 = x0 + 1 >= width ? width - 1 : x0 + 1;
        if (y0 >= height - 1) { y0 = y1 = height - 1; y = y0; }
        if (x0 >= width - 1) { x0 = x1 = width - 1; x = x0; }
        double ly = y - y0, lx = x - x0, hy = 1.0 - ly, hx = 1.0 - lx;
        offsets.Add((y0 * width) + x0); weights.Add(hy * hx * scale);
        offsets.Add((y0 * width) + x1); weights.Add(hy * lx * scale);
        offsets.Add((y1 * width) + x0); weights.Add(ly * hx * scale);
        offsets.Add((y1 * width) + x1); weights.Add(ly * lx * scale);
    }

    /// <summary>The boxes as doubles, read once.</summary>
    public static double[] ReadBoxes<T>(Tensor<T> boxes) => ToDoubles(boxes);

    /// <summary>output[r, c, bin] = sum of weight * input[batch[r], c, offset] over the bin's taps.</summary>
    public double[] Forward(double[] input, int channels, int plane)
    {
        var output = new double[Rois * channels * Bins];
        long work = (long)Rois * channels * Bins * 16;
        CpuParallelSettings.ParallelForOrSerial(0, Rois * channels, work, item =>
        {
            int r = item / channels, c = item % channels;
            int planeBase = ((Batch[r] * channels) + c) * plane;
            int outBase = item * Bins;
            for (int bin = 0; bin < Bins; bin++)
            {
                double sum = 0;
                int end = TapStart[(r * Bins) + bin + 1];
                for (int t = TapStart[(r * Bins) + bin]; t < end; t++) sum += Weights[t] * input[planeBase + Offsets[t]];
                output[outBase + bin] = sum;
            }
        });
        return output;
    }

    /// <summary>gradInput[batch[r], c, offset] += weight * gradOutput[r, c, bin]; one thread per channel.</summary>
    public double[] Backward(double[] gradOutput, int images, int channels, int plane)
    {
        var gradInput = new double[images * channels * plane];
        long work = (long)Rois * channels * Bins * 16;
        CpuParallelSettings.ParallelForOrSerial(0, channels, work, c =>
        {
            for (int r = 0; r < Rois; r++)
            {
                int planeBase = ((Batch[r] * channels) + c) * plane;
                int gradBase = ((r * channels) + c) * Bins;
                for (int bin = 0; bin < Bins; bin++)
                {
                    double g = gradOutput[gradBase + bin];
                    if (g == 0) continue;
                    int end = TapStart[(r * Bins) + bin + 1];
                    for (int t = TapStart[(r * Bins) + bin]; t < end; t++) gradInput[planeBase + Offsets[t]] += Weights[t] * g;
                }
            }
        });
        return gradInput;
    }

    /// <summary>A tensor's values as doubles: a typed span read for float and double.</summary>
    public static double[] ToDoubles<T>(Tensor<T> tensor)
    {
        var result = new double[tensor.Length];
        if (tensor is Tensor<float> floats)
        {
            using var lease = floats.Lease();
            var values = lease.Span;
            for (int i = 0; i < values.Length; i++) result[i] = values[i];
        }
        else if (tensor is Tensor<double> doubles)
        {
            using var lease = doubles.Lease();
            lease.Span.CopyTo(result);
        }
        else
        {
            using var lease = tensor.Lease();
            ConvertGeneric(lease.Span, result);
        }

        return result;
    }

    /// <summary>Writes doubles into a new tensor of the given shape: a typed span write for float and double.</summary>
    public static Tensor<T> FromDoubles<T>(double[] values, int[] shape)
    {
        var tensor = new Tensor<T>(shape);
        if (tensor is Tensor<float> floats)
        {
            using var lease = floats.LeaseWritable();
            var target = lease.Span;
            for (int i = 0; i < target.Length; i++) target[i] = (float)values[i];
        }
        else if (tensor is Tensor<double> doubles)
        {
            using var lease = doubles.LeaseWritable();
            values.AsSpan().CopyTo(lease.Span);
        }
        else
        {
            using var lease = tensor.LeaseWritable();
            ConvertGeneric(values, lease.Span);
        }

        return tensor;
    }

    [ScalarKernel("Element types other than float and double have no typed span view; RoIAlign converts them once per call.")]
    private static void ConvertGeneric<T>(ReadOnlySpan<T> source, double[] target)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        for (int i = 0; i < source.Length; i++) target[i] = ops.ToDouble(source[i]);
    }

    [ScalarKernel("Element types other than float and double have no typed span view; RoIAlign converts them once per call.")]
    private static void ConvertGeneric<T>(double[] source, Span<T> target)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        for (int i = 0; i < target.Length; i++) target[i] = ops.FromDouble(source[i]);
    }
}
