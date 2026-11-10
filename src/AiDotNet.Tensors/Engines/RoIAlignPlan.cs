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
/// grew with the total sampled area: about 2 GB for 1,000 200x200 RoIs at scale 1. A single RoI whose taps alone pass
/// the budget gets no table: its plan samples on the fly, adding the same terms in the table's order.
/// </para>
/// <para>
/// Float and double tensors are read and written in place, accumulating in double; other element types go through a
/// double copy. The backward accumulates one channel at a time in a double plane per image, so its temporary memory is
/// a channel's worth per worker rather than a double copy of the whole gradient.
/// </para>
/// </remarks>
internal sealed class RoIAlignPlan
{
    /// <summary>
    /// The most taps (12 bytes each) a plan's table holds: about 48 MB. A RoI whose taps alone pass it is sampled
    /// directly instead (see <see cref="IsDirect"/>).
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

    // A single RoI sampled on the fly.
    private RoIAlignPlan(int roi, int image, RoiGrid grid, int outputHeight, int outputWidth, int height, int width)
        : this(roi, 1, outputHeight * outputWidth, new[] { image }, Array.Empty<int>(), Array.Empty<int>(),
            Array.Empty<double>())
    {
        IsDirect = true;
        _grid = grid;
        _outputWidth = outputWidth;
        _height = height;
        _width = width;
    }

    private readonly RoiGrid _grid;
    private readonly int _outputWidth, _height, _width;

    /// <summary>The index, among all the call's RoIs, of this plan's first RoI.</summary>
    public int FirstRoi { get; }

    public int Rois { get; }
    public int Bins { get; }

    /// <summary>A single RoI too large for a tap table, sampled on the fly; its table arrays are empty.</summary>
    public bool IsDirect { get; }

    /// <summary>The image each RoI reads.</summary>
    public int[] Batch { get; }

    /// <summary>Bin <c>r * Bins + bin</c> owns taps <c>[TapStart[i], TapStart[i + 1])</c>.</summary>
    public int[] TapStart { get; }

    /// <summary>Offset of each tap within a channel plane.</summary>
    public int[] Offsets { get; }

    /// <summary>Weight of each tap, including the 1 / (ratioY * ratioX) average.</summary>
    public double[] Weights { get; }

    /// <summary>RoIAlign of <paramref name="input"/> [N, C, H, W] over <paramref name="boxes"/> [K, 5].</summary>
    public static Tensor<T> Forward<T>(Tensor<T> input, Tensor<T> boxes, int outputHeight, int outputWidth,
        float spatialScale, int samplingRatio, bool aligned)
    {
        int images = input._shape[0], channels = input._shape[1], height = input._shape[2], width = input._shape[3];
        var boxValues = ToDoubles(boxes);
        var shape = new[] { boxValues.Length / 5, channels, outputHeight, outputWidth };
        // The result is computed into an array this call owns and then wrapped, so no write goes through a tensor
        // accessor that could hand back a copy; the input is only read, so a copy-on-write peer stays shared.
        int outputLength = shape[0] * channels * outputHeight * outputWidth;
        if (input is Tensor<float> floats)
        {
            var output = new float[outputLength];
            using (new KeepAliveScope(floats))
                Forward<float, FloatElement>(boxValues, images, height, width, outputHeight, outputWidth, spatialScale,
                    samplingRatio, aligned, floats.GetReadOnlyDataArray(), channels, output, TapBudget);
            return (Tensor<T>)(object)Tensor<float>.FromMemory(new Memory<float>(output), shape);
        }

        if (input is Tensor<double> doubles)
        {
            var output = new double[outputLength];
            using (new KeepAliveScope(doubles))
                Forward<double, DoubleElement>(boxValues, images, height, width, outputHeight, outputWidth, spatialScale,
                    samplingRatio, aligned, doubles.GetReadOnlyDataArray(), channels, output, TapBudget);
            return (Tensor<T>)(object)Tensor<double>.FromMemory(new Memory<double>(output), shape);
        }

        var pooled = Forward(boxValues, images, height, width, outputHeight, outputWidth, spatialScale, samplingRatio,
            aligned, ToDoubles(input), channels);
        return FromDoubles<T>(pooled, shape);
    }

    /// <summary>The gradient, of shape <paramref name="inputShape"/>, of RoIAlign's input.</summary>
    public static Tensor<T> Backward<T>(Tensor<T> gradOutput, Tensor<T> boxes, int[] inputShape, int outputHeight,
        int outputWidth, float spatialScale, int samplingRatio, bool aligned)
    {
        int images = inputShape[0], channels = inputShape[1], height = inputShape[2], width = inputShape[3];
        var boxValues = ToDoubles(boxes);
        int inputLength = images * channels * height * width;
        if (gradOutput is Tensor<float> floats)
        {
            var gradInput = new float[inputLength];
            using (new KeepAliveScope(floats))
                Backward<float, FloatElement>(boxValues, images, height, width, outputHeight, outputWidth, spatialScale,
                    samplingRatio, aligned, floats.GetReadOnlyDataArray(), channels, gradInput, TapBudget);
            return (Tensor<T>)(object)Tensor<float>.FromMemory(new Memory<float>(gradInput), (int[])inputShape.Clone());
        }

        if (gradOutput is Tensor<double> doubles)
        {
            var gradInput = new double[inputLength];
            using (new KeepAliveScope(doubles))
                Backward<double, DoubleElement>(boxValues, images, height, width, outputHeight, outputWidth,
                    spatialScale, samplingRatio, aligned, doubles.GetReadOnlyDataArray(), channels, gradInput, TapBudget);
            return (Tensor<T>)(object)Tensor<double>.FromMemory(new Memory<double>(gradInput), (int[])inputShape.Clone());
        }

        var gradients = Backward(boxValues, images, height, width, outputHeight, outputWidth, spatialScale,
            samplingRatio, aligned, ToDoubles(gradOutput), channels);
        return FromDoubles<T>(gradients, inputShape);
    }

    /// <summary>output[r, c, bin] for every RoI, building and gathering one budgeted run of RoIs at a time.</summary>
    public static double[] Forward(double[] boxes, int images, int height, int width, int outputHeight, int outputWidth,
        float spatialScale, int samplingRatio, bool aligned, double[] input, int channels, long tapBudget = TapBudget)
    {
        var output = new double[(boxes.Length / 5) * channels * outputHeight * outputWidth];
        Forward<double, DoubleElement>(boxes, images, height, width, outputHeight, outputWidth, spatialScale,
            samplingRatio, aligned, input, channels, output, tapBudget);
        return output;
    }

    /// <summary>gradInput for every RoI, building and scattering one budgeted run of RoIs at a time.</summary>
    public static double[] Backward(double[] boxes, int images, int height, int width, int outputHeight, int outputWidth,
        float spatialScale, int samplingRatio, bool aligned, double[] gradOutput, int channels, long tapBudget = TapBudget)
    {
        var gradInput = new double[images * channels * height * width];
        Backward<double, DoubleElement>(boxes, images, height, width, outputHeight, outputWidth, spatialScale,
            samplingRatio, aligned, gradOutput, channels, gradInput, tapBudget);
        return gradInput;
    }

    private static void Forward<TNum, TElement>(double[] boxes, int images, int height, int width, int outputHeight,
        int outputWidth, float spatialScale, int samplingRatio, bool aligned, TNum[] input, int channels, TNum[] output,
        long tapBudget)
        where TElement : struct, IRoiElement<TNum>
    {
        foreach (var plan in Plans(boxes, images, height, width, outputHeight, outputWidth, spatialScale, samplingRatio,
            aligned, tapBudget))
            plan.Gather<TNum, TElement>(input, channels, height * width, output);
    }

    // gradInput must start zeroed: each plan adds its RoIs' contributions.
    private static void Backward<TNum, TElement>(double[] boxes, int images, int height, int width, int outputHeight,
        int outputWidth, float spatialScale, int samplingRatio, bool aligned, TNum[] gradOutput, int channels,
        TNum[] gradInput, long tapBudget)
        where TElement : struct, IRoiElement<TNum>
    {
        foreach (var plan in Plans(boxes, images, height, width, outputHeight, outputWidth, spatialScale, samplingRatio,
            aligned, tapBudget))
            plan.Scatter<TNum, TElement>(gradOutput, channels, height * width, gradInput);
    }

    /// <summary>
    /// Consecutive runs of RoIs whose tap bound fits <paramref name="tapBudget"/>, each holding at least one RoI; a RoI
    /// whose taps alone pass it gets a direct plan of its own.
    /// </summary>
    internal static System.Collections.Generic.IEnumerable<RoIAlignPlan> Plans(double[] boxes, int images, int height,
        int width, int outputHeight, int outputWidth, float spatialScale, int samplingRatio, bool aligned,
        long tapBudget = TapBudget)
    {
        int rois = boxes.Length / 5;
        long bins = (long)outputHeight * outputWidth;
        int first = 0;
        long taps = 0;
        for (int r = 0; r < rois; r++)
        {
            var grid = Grid(boxes, r, outputHeight, outputWidth, spatialScale, samplingRatio, aligned);
            long roiTaps = 4L * grid.Ry * grid.Rx * bins;
            if (r > first && taps + roiTaps > tapBudget)
            {
                yield return Build(boxes, first, r - first, taps, images, height, width, outputHeight, outputWidth,
                    spatialScale, samplingRatio, aligned);
                first = r;
                taps = 0;
            }

            if (roiTaps > tapBudget)
            {
                yield return new RoIAlignPlan(r, ImageOf(boxes, r, images), grid, outputHeight, outputWidth, height, width);
                first = r + 1;
                continue;
            }

            taps += roiTaps;
        }

        if (rois > first)
            yield return Build(boxes, first, rois - first, taps, images, height, width, outputHeight, outputWidth,
                spatialScale, samplingRatio, aligned);
    }

    private static int ImageOf(double[] boxes, int r, int images)
    {
        int image = (int)boxes[r * 5];
        if (image < 0 || image >= images)
            throw new ArgumentException($"RoIAlign batch idx {image} out of range [0, {images}).");
        return image;
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
        // The bound counts samples outside the image, which add no taps, so it sizes the lists only up to the budget.
        int capacity = (int)Math.Min(tapBound, TapBudget);
        var offsets = new System.Collections.Generic.List<int>(capacity);
        var weights = new System.Collections.Generic.List<double>(capacity);

        for (int r = 0; r < rois; r++)
        {
            batch[r] = ImageOf(boxes, firstRoi + r, images);
            var grid = Grid(boxes, firstRoi + r, outputHeight, outputWidth, spatialScale, samplingRatio, aligned);
            double average = Average(grid);

            for (int ph = 0; ph < outputHeight; ph++)
            {
                for (int pw = 0; pw < outputWidth; pw++)
                {
                    tapStart[(r * bins) + (ph * outputWidth) + pw] = offsets.Count;
                    for (int iy = 0; iy < grid.Ry; iy++)
                    {
                        double sy = SampleY(grid, ph, iy);
                        for (int ix = 0; ix < grid.Rx; ix++)
                        {
                            if (!Corners(sy, SampleX(grid, pw, ix), height, width, average, out var tap)) continue;
                            offsets.Add(tap.O0); weights.Add(tap.W0);
                            offsets.Add(tap.O1); weights.Add(tap.W1);
                            offsets.Add(tap.O2); weights.Add(tap.W2);
                            offsets.Add(tap.O3); weights.Add(tap.W3);
                        }
                    }
                }
            }
        }

        tapStart[rois * bins] = offsets.Count;
        return new RoIAlignPlan(firstRoi, rois, bins, batch, tapStart, offsets.ToArray(), weights.ToArray());
    }

    private static double Average(RoiGrid grid) => 1.0 / Math.Max((long)grid.Ry * grid.Rx, 1);

    private static double SampleY(RoiGrid grid, int ph, int iy)
        => grid.Y1 + (ph * grid.BinH) + ((iy + 0.5) * grid.BinH / grid.Ry);

    private static double SampleX(RoiGrid grid, int pw, int ix)
        => grid.X1 + (pw * grid.BinW) + ((ix + 0.5) * grid.BinW / grid.Rx);

    // A sample's four bilinear taps: plane offsets, and weights already times the grid average.
    private readonly struct Taps
    {
        public Taps(int o0, int o1, int o2, int o3, double w0, double w1, double w2, double w3)
        {
            O0 = o0;
            O1 = o1;
            O2 = o2;
            O3 = o3;
            W0 = w0;
            W1 = w1;
            W2 = w2;
            W3 = w3;
        }

        public int O0 { get; }
        public int O1 { get; }
        public int O2 { get; }
        public int O3 { get; }
        public double W0 { get; }
        public double W1 { get; }
        public double W2 { get; }
        public double W3 { get; }
    }

    // False for a sample outside [-1, size], which contributes nothing (but still counts in the average).
    private static bool Corners(double y, double x, int height, int width, double scale, out Taps taps)
    {
        taps = default;
        if (y < -1.0 || y > height || x < -1.0 || x > width) return false;
        if (y <= 0) y = 0;
        if (x <= 0) x = 0;
        int y0 = (int)y, x0 = (int)x;
        int y1 = y0 + 1 >= height ? height - 1 : y0 + 1;
        int x1 = x0 + 1 >= width ? width - 1 : x0 + 1;
        if (y0 >= height - 1) { y0 = y1 = height - 1; y = y0; }
        if (x0 >= width - 1) { x0 = x1 = width - 1; x = x0; }
        double ly = y - y0, lx = x - x0, hy = 1.0 - ly, hx = 1.0 - lx;
        taps = new Taps((y0 * width) + x0, (y0 * width) + x1, (y1 * width) + x0, (y1 * width) + x1,
            hy * hx * scale, hy * lx * scale, ly * hx * scale, ly * lx * scale);
        return true;
    }

    /// <summary>output[FirstRoi + r, c, bin] = sum of weight * input[batch[r], c, offset] over the bin's taps.</summary>
    private void Gather<TNum, TElement>(TNum[] input, int channels, int plane, TNum[] output)
        where TElement : struct, IRoiElement<TNum>
    {
        var element = default(TElement);
        long work = (long)Rois * channels * Bins * TapsPerBin();
        CpuParallelSettings.ParallelForOrSerial(0, Rois * channels, work, item =>
        {
            int r = item / channels, c = item % channels;
            int planeBase = ((Batch[r] * channels) + c) * plane;
            int outBase = ((FirstRoi * channels) + item) * Bins;
            for (int bin = 0; bin < Bins; bin++)
            {
                double sum = 0;
                if (IsDirect)
                {
                    // The table's terms in the table's order, computed instead of stored.
                    int ph = bin / _outputWidth, pw = bin % _outputWidth;
                    double average = Average(_grid);
                    for (int iy = 0; iy < _grid.Ry; iy++)
                    {
                        double sy = SampleY(_grid, ph, iy);
                        for (int ix = 0; ix < _grid.Rx; ix++)
                        {
                            if (!Corners(sy, SampleX(_grid, pw, ix), _height, _width, average, out var tap)) continue;
                            sum += tap.W0 * element.ToDouble(input[planeBase + tap.O0]);
                            sum += tap.W1 * element.ToDouble(input[planeBase + tap.O1]);
                            sum += tap.W2 * element.ToDouble(input[planeBase + tap.O2]);
                            sum += tap.W3 * element.ToDouble(input[planeBase + tap.O3]);
                        }
                    }
                }
                else
                {
                    int end = TapStart[(r * Bins) + bin + 1];
                    for (int t = TapStart[(r * Bins) + bin]; t < end; t++)
                        sum += Weights[t] * element.ToDouble(input[planeBase + Offsets[t]]);
                }

                output[outBase + bin] = element.FromDouble(sum);
            }
        }, deterministicSafe: true);
    }

    /// <summary>
    /// gradInput[batch[r], c, offset] += weight * gradOutput[FirstRoi + r, c, bin]: one channel per worker, summed in a
    /// double plane per image the plan reads, then added to gradInput once.
    /// </summary>
    private void Scatter<TNum, TElement>(TNum[] gradOutput, int channels, int plane, TNum[] gradInput)
        where TElement : struct, IRoiElement<TNum>
    {
        var element = default(TElement);
        var images = new System.Collections.Generic.List<int>();
        var slots = new System.Collections.Generic.Dictionary<int, int>();
        var scratchBase = new int[Rois];
        for (int r = 0; r < Rois; r++)
        {
            if (!slots.TryGetValue(Batch[r], out int slot))
            {
                slot = images.Count;
                slots[Batch[r]] = slot;
                images.Add(Batch[r]);
            }

            scratchBase[r] = slot * plane;
        }

        long work = (long)Rois * channels * Bins * TapsPerBin();
        int scratchLength = images.Count * plane;
        CpuParallelSettings.ParallelForOrSerial(0, channels, work, c =>
        {
            // Pooled: a fresh plane per channel put images * plane doubles on the large object heap for every channel.
            var scratch = System.Buffers.ArrayPool<double>.Shared.Rent(scratchLength);
            try
            {
                Array.Clear(scratch, 0, scratchLength);
                for (int r = 0; r < Rois; r++)
                {
                    int planeBase = scratchBase[r];
                    int gradBase = (((FirstRoi + r) * channels) + c) * Bins;
                    for (int bin = 0; bin < Bins; bin++)
                    {
                        double g = element.ToDouble(gradOutput[gradBase + bin]);
                        if (g == 0) continue;
                        if (IsDirect)
                        {
                            int ph = bin / _outputWidth, pw = bin % _outputWidth;
                            double average = Average(_grid);
                            for (int iy = 0; iy < _grid.Ry; iy++)
                            {
                                double sy = SampleY(_grid, ph, iy);
                                for (int ix = 0; ix < _grid.Rx; ix++)
                                {
                                    if (!Corners(sy, SampleX(_grid, pw, ix), _height, _width, average, out var tap)) continue;
                                    scratch[planeBase + tap.O0] += tap.W0 * g;
                                    scratch[planeBase + tap.O1] += tap.W1 * g;
                                    scratch[planeBase + tap.O2] += tap.W2 * g;
                                    scratch[planeBase + tap.O3] += tap.W3 * g;
                                }
                            }
                        }
                        else
                        {
                            int end = TapStart[(r * Bins) + bin + 1];
                            for (int t = TapStart[(r * Bins) + bin]; t < end; t++)
                                scratch[planeBase + Offsets[t]] += Weights[t] * g;
                        }
                    }
                }

                for (int i = 0; i < images.Count; i++)
                {
                    int target = ((images[i] * channels) + c) * plane, source = i * plane;
                    for (int p = 0; p < plane; p++)
                        gradInput[target + p] = element.FromDouble(element.ToDouble(gradInput[target + p]) + scratch[source + p]);
                }
            }
            finally
            {
                System.Buffers.ArrayPool<double>.Shared.Return(scratch);
            }
        }, deterministicSafe: true);
    }

    // The work per (RoI, channel, bin), in multiply-adds, for the parallel cut-off: a direct plan computes its samples' four
    // taps (plus their positions), a table plan reads its average taps per bin.
    private long TapsPerBin()
        => IsDirect
            ? 8L * Math.Max((long)_grid.Ry * _grid.Rx, 1)
            : Math.Max(Offsets.Length / Math.Max((long)Rois * Bins, 1), 1) + 1;

    /// <summary>Reads and writes one element type as double; a struct per type keeps the kernels typed.</summary>
    internal interface IRoiElement<TNum>
    {
        double ToDouble(TNum value);
        TNum FromDouble(double value);
    }

    private readonly struct FloatElement : IRoiElement<float>
    {
        public double ToDouble(float value) => value;
        public float FromDouble(double value) => (float)value;
    }

    private readonly struct DoubleElement : IRoiElement<double>
    {
        public double ToDouble(double value) => value;
        public double FromDouble(double value) => value;
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
