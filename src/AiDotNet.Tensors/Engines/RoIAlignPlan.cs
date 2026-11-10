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
/// last row and column clamp in place. <c>aligned</c> shifts by half a pixel and drops the minimum RoI size of 1, so a
/// degenerate aligned RoI has an empty sampling grid and outputs 0.
/// </para>
/// <para>
/// A plan covers a run of RoIs whose taps fit <see cref="TapBudget"/>; the static <c>Forward</c> and <c>Backward</c>
/// build and run one run at a time. An adaptive sampling grid grows with the RoI, so a table for every RoI at once
/// grew with the total sampled area: about 2 GB for 1,000 200x200 RoIs at scale 1.
/// </para>
/// </remarks>
internal sealed class RoIAlignPlan
{
    /// <summary>
    /// The most taps (12 bytes each) a plan holds, unless one RoI alone needs more: about 48 MB. A RoI's taps are at
    /// most four per sample and its samples about its area in pixels, so a single RoI stays within a few channel planes.
    /// </summary>
    internal const long TapBudget = 1L << 22;

    private RoIAlignPlan(int firstRoi, int rois, int bins, int[] batch, int[] tapStart, int[] offsets, double[] weights)
    {
        FirstRoi = firstRoi;
        Rois = rois;
        Bins = bins;
        Batch = batch;
        TapStart = tapStart;
        Offsets = offsets;
        Weights = weights;
    }

    /// <summary>The index, among all the call's RoIs, of this plan's first RoI.</summary>
    public int FirstRoi { get; }

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

    /// <summary>output[r, c, bin] for every RoI, building and gathering one budgeted run of RoIs at a time.</summary>
    public static double[] Forward(double[] boxes, int images, int height, int width, int outputHeight, int outputWidth,
        float spatialScale, int samplingRatio, bool aligned, double[] input, int channels)
    {
        var output = new double[(boxes.Length / 5) * channels * outputHeight * outputWidth];
        foreach (var plan in Plans(boxes, images, height, width, outputHeight, outputWidth, spatialScale, samplingRatio, aligned))
            plan.Forward(input, channels, height * width, output);
        return output;
    }

    /// <summary>gradInput for every RoI, building and scattering one budgeted run of RoIs at a time.</summary>
    public static double[] Backward(double[] boxes, int images, int height, int width, int outputHeight, int outputWidth,
        float spatialScale, int samplingRatio, bool aligned, double[] gradOutput, int channels)
    {
        var gradInput = new double[images * channels * height * width];
        foreach (var plan in Plans(boxes, images, height, width, outputHeight, outputWidth, spatialScale, samplingRatio, aligned))
            plan.Backward(gradOutput, channels, height * width, gradInput);
        return gradInput;
    }

    /// <summary>Consecutive runs of RoIs whose tap bound fits <see cref="TapBudget"/>; a run holds at least one RoI.</summary>
    internal static System.Collections.Generic.IEnumerable<RoIAlignPlan> Plans(double[] boxes, int images, int height,
        int width, int outputHeight, int outputWidth, float spatialScale, int samplingRatio, bool aligned)
    {
        int rois = boxes.Length / 5;
        long bins = (long)outputHeight * outputWidth;
        int first = 0;
        long taps = 0;
        for (int r = 0; r < rois; r++)
        {
            var grid = Grid(boxes, r, outputHeight, outputWidth, spatialScale, samplingRatio, aligned);
            long roiTaps = 4L * grid.Ry * grid.Rx * bins;
            if (r > first && taps + roiTaps > TapBudget)
            {
                yield return Build(boxes, first, r - first, taps, images, height, width, outputHeight, outputWidth,
                    spatialScale, samplingRatio, aligned);
                first = r;
                taps = 0;
            }

            taps += roiTaps;
        }

        if (rois > first)
            yield return Build(boxes, first, rois - first, taps, images, height, width, outputHeight, outputWidth,
                spatialScale, samplingRatio, aligned);
    }

    // A RoI's sampling grid: its scaled corner, bin size, and samples per bin.
    private readonly struct RoiGrid
    {
        public RoiGrid(double y1, double x1, double binH, double binW, int ry, int rx)
        {
            Y1 = y1;
            X1 = x1;
            BinH = binH;
            BinW = binW;
            Ry = ry;
            Rx = rx;
        }

        public double Y1 { get; }
        public double X1 { get; }
        public double BinH { get; }
        public double BinW { get; }
        public int Ry { get; }
        public int Rx { get; }
    }

    private static RoiGrid Grid(double[] boxes, int r, int outputHeight, int outputWidth, float spatialScale,
        int samplingRatio, bool aligned)
    {
        double shift = aligned ? 0.5 : 0.0;
        double x1 = (boxes[(r * 5) + 1] * spatialScale) - shift;
        double y1 = (boxes[(r * 5) + 2] * spatialScale) - shift;
        double x2 = (boxes[(r * 5) + 3] * spatialScale) - shift;
        double y2 = (boxes[(r * 5) + 4] * spatialScale) - shift;
        double roiW = aligned ? (x2 - x1) : Math.Max(x2 - x1, 1.0);
        double roiH = aligned ? (y2 - y1) : Math.Max(y2 - y1, 1.0);
        // As torchvision: an aligned RoI of zero or negative size has an empty grid (no taps), and its bins
        // average to 0 rather than sampling the box edge.
        int ry = samplingRatio > 0 ? samplingRatio : (int)Math.Ceiling(roiH / outputHeight);
        int rx = samplingRatio > 0 ? samplingRatio : (int)Math.Ceiling(roiW / outputWidth);
        return new RoiGrid(y1, x1, roiH / outputHeight, roiW / outputWidth, Math.Max(ry, 0), Math.Max(rx, 0));
    }

    /// <summary>The plan of RoIs <c>[firstRoi, firstRoi + rois)</c>, whose taps number at most <paramref name="tapBound"/>.</summary>
    private static RoIAlignPlan Build(double[] boxes, int firstRoi, int rois, long tapBound, int images, int height,
        int width, int outputHeight, int outputWidth, float spatialScale, int samplingRatio, bool aligned)
    {
        int bins = outputHeight * outputWidth;
        var batch = new int[rois];
        var tapStart = new int[(rois * bins) + 1];
        int capacity = (int)Math.Min(tapBound, int.MaxValue);
        var offsets = new System.Collections.Generic.List<int>(capacity);
        var weights = new System.Collections.Generic.List<double>(capacity);

        for (int r = 0; r < rois; r++)
        {
            int image = (int)boxes[(firstRoi + r) * 5];
            if (image < 0 || image >= images)
                throw new ArgumentException($"RoIAlign batch idx {image} out of range [0, {images}).");
            batch[r] = image;
            var grid = Grid(boxes, firstRoi + r, outputHeight, outputWidth, spatialScale, samplingRatio, aligned);
            double y1 = grid.Y1, x1 = grid.X1, binH = grid.BinH, binW = grid.BinW;
            int ry = grid.Ry, rx = grid.Rx;
            double average = 1.0 / Math.Max((long)ry * rx, 1);

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
        return new RoIAlignPlan(firstRoi, rois, bins, batch, tapStart, offsets.ToArray(), weights.ToArray());
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

    /// <summary>output[FirstRoi + r, c, bin] = sum of weight * input[batch[r], c, offset] over the bin's taps.</summary>
    private void Forward(double[] input, int channels, int plane, double[] output)
    {
        long work = (long)Rois * channels * Bins * 16;
        CpuParallelSettings.ParallelForOrSerial(0, Rois * channels, work, item =>
        {
            int r = item / channels, c = item % channels;
            int planeBase = ((Batch[r] * channels) + c) * plane;
            int outBase = ((FirstRoi * channels) + item) * Bins;
            for (int bin = 0; bin < Bins; bin++)
            {
                double sum = 0;
                int end = TapStart[(r * Bins) + bin + 1];
                for (int t = TapStart[(r * Bins) + bin]; t < end; t++) sum += Weights[t] * input[planeBase + Offsets[t]];
                output[outBase + bin] = sum;
            }
        });
    }

    /// <summary>gradInput[batch[r], c, offset] += weight * gradOutput[FirstRoi + r, c, bin]; one thread per channel.</summary>
    private void Backward(double[] gradOutput, int channels, int plane, double[] gradInput)
    {
        long work = (long)Rois * channels * Bins * 16;
        CpuParallelSettings.ParallelForOrSerial(0, channels, work, c =>
        {
            for (int r = 0; r < Rois; r++)
            {
                int planeBase = ((Batch[r] * channels) + c) * plane;
                int gradBase = (((FirstRoi + r) * channels) + c) * Bins;
                for (int bin = 0; bin < Bins; bin++)
                {
                    double g = gradOutput[gradBase + bin];
                    if (g == 0) continue;
                    int end = TapStart[(r * Bins) + bin + 1];
                    for (int t = TapStart[(r * Bins) + bin]; t < end; t++) gradInput[planeBase + Offsets[t]] += Weights[t] * g;
                }
            }
        });
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
