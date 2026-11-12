using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Autodiff;

/// <summary>
/// Double-backward support for the pooling, convolution and resampling ops (see
/// <c>BackwardFunctions.HigherOrder.cs</c> for the general scheme).
/// </summary>
internal static partial class BackwardFunctions<T>
{
    /// <summary>
    /// A 2-D or 3-D grid sample (PyTorch's <c>grid_sample</c>: unnormalize, clip or reflect, then bilinear/trilinear or
    /// nearest, skipping out-of-range taps) as recorded engine ops, differentiable to any order in both the input and
    /// the grid: each axis' fractional voxel coordinate is <c>slope·g + offset - floor</c> with the piecewise slope and
    /// offset of the padding mode held constant, the tap weights are products of those fractions, and the tap values
    /// are a recorded gather of the input. Bicubic is not supported.
    /// </summary>
    internal static Tensor<T> GridSampleRecorded(IEngine e, Tensor<T> input, Tensor<T> grid, GridSampleMode mode,
        GridSamplePadding padding, bool alignCorners)
    {
        if (mode == GridSampleMode.Bicubic) throw new NotSupportedException("A recorded bicubic grid sample is not supported.");
        var ops = MathHelper.GetNumericOperations<T>();
        int dims = grid._shape[grid.Rank - 1];
        int n = input._shape[0], channels = input._shape[1];
        var size = new int[dims];                                   // the grid's x, y(, z) order: W, H(, D)
        for (int a = 0; a < dims; a++) size[a] = input._shape[input.Rank - 1 - a];
        int volume = ShapeLength(size);
        int points = grid.Length / (n * dims);
        var g = RoIAlignPlan.ToDoubles(grid);
        var pointShape = new int[grid.Rank - 1];
        Array.Copy(grid._shape, pointShape, pointShape.Length);
        var outShape = new int[pointShape.Length + 1];
        outShape[0] = n;
        outShape[1] = channels;
        Array.Copy(pointShape, 1, outShape, 2, pointShape.Length - 1);
        var weightShape = (int[])outShape.Clone();
        weightShape[1] = 1;

        // Per point and axis: the voxel coordinate's slope and offset in the normalized coordinate, and its floor.
        var slope = new double[dims][];
        var offset = new double[dims][];
        var floor = new int[dims][];
        for (int a = 0; a < dims; a++) { slope[a] = new double[n * points]; offset[a] = new double[n * points]; floor[a] = new int[n * points]; }
        for (int p = 0; p < n * points; p++)
            for (int a = 0; a < dims; a++)
            {
                double normalized = g[p * dims + a];
                double coord = GridSourceIndex(normalized, size[a], padding, alignCorners, out double s);
                int f = GridVoxelIndex(mode == GridSampleMode.Nearest ? Math.Round(coord) : Math.Floor(coord));
                slope[a][p] = s;
                offset[a][p] = coord - s * normalized - f;
                floor[a][p] = f;
            }

        var fractions = new Tensor<T>[dims];
        if (mode != GridSampleMode.Nearest)
            for (int a = 0; a < dims; a++)
            {
                var axis = e.TensorSliceAxis(grid, grid.Rank - 1, a);
                fractions[a] = e.TensorAdd(e.TensorMultiply(axis, RoIAlignPlan.FromDoubles<T>(slope[a], pointShape)),
                    RoIAlignPlan.FromDoubles<T>(offset[a], pointShape));
            }

        int corners = mode == GridSampleMode.Nearest ? 1 : 1 << dims;
        Tensor<T>? result = null;
        var one = ops.One;
        for (int corner = 0; corner < corners; corner++)
        {
            var index = new int[n * channels * points];
            for (int b = 0; b < n; b++)
                for (int q = 0; q < points; q++)
                {
                    int p = b * points + q, linear = 0;
                    bool inside = true;
                    for (int a = dims - 1; a >= 0; a--)
                    {
                        int i = floor[a][p] + ((corner >> a) & 1);
                        if (i < 0 || i >= size[a]) inside = false;
                        linear = linear * size[a] + i;
                    }
                    for (int ch = 0; ch < channels; ch++)
                        index[(b * channels + ch) * points + q] = inside ? (b * channels + ch) * volume + linear : -1;
                }
            var inputShape = (int[])input._shape.Clone();
            var gatherShape = (int[])outShape.Clone();
            var values = RecordedLinearMap("GridSampleGather", input,
                v => GatherFlat(v, index, gatherShape), w => ScatterAddFlat(w, index, inputShape));
            if (mode == GridSampleMode.Nearest) { result = values; break; }
            Tensor<T>? weight = null;
            for (int a = 0; a < dims; a++)
            {
                var factor = ((corner >> a) & 1) == 1 ? fractions[a] : e.TensorAddScalar(e.TensorNegate(fractions[a]), one);
                weight = weight is null ? factor : e.TensorMultiply(weight, factor);
            }
            var term = e.TensorMultiply(values, e.Reshape(weight!, weightShape));
            result = result is null ? term : e.TensorAdd(result, term);
        }
        return result!;
    }

    /// <summary>
    /// Forward (bilinear) splatting as recorded engine ops, differentiable to any order in the input and the flow:
    /// each source pixel's value is scatter-added to the four pixels around <c>(x + flowX, y + flowY)</c> with the
    /// bilinear weights, then (with <paramref name="normalize"/>) divided by the splatted weight sum where it is non-zero.
    /// Mirrors <see cref="CpuEngine.ForwardSplat{T}"/>.
    /// </summary>
    internal static Tensor<T> ForwardSplatRecorded(IEngine e, Tensor<T> input, Tensor<T> flow, bool normalize)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int n = input._shape[0], channels = input._shape[1], height = input._shape[2], width = input._shape[3];
        int spatial = height * width;
        var f = RoIAlignPlan.ToDoubles(flow);
        var planeShape = new[] { n, height, width };
        var weightShape = new[] { n, 1, height, width };
        var offsetX = new double[n * spatial];
        var offsetY = new double[n * spatial];
        var x0 = new int[n * spatial];
        var y0 = new int[n * spatial];
        for (int b = 0; b < n; b++)
            for (int y = 0; y < height; y++)
                for (int x = 0; x < width; x++)
                {
                    int p = b * spatial + y * width + x;
                    double dx = x + f[b * 2 * spatial + y * width + x], dy = y + f[b * 2 * spatial + spatial + y * width + x];
                    x0[p] = (int)Math.Floor(dx);
                    y0[p] = (int)Math.Floor(dy);
                    offsetX[p] = x - x0[p];
                    offsetY[p] = y - y0[p];
                }
        // Fractions as recorded functions of the flow: flow + (x - floor(x + flow)).
        var fx = e.TensorAdd(e.TensorSliceAxis(flow, 1, 0), RoIAlignPlan.FromDoubles<T>(offsetX, planeShape));
        var fy = e.TensorAdd(e.TensorSliceAxis(flow, 1, 1), RoIAlignPlan.FromDoubles<T>(offsetY, planeShape));
        var one = ops.One;
        Tensor<T> OneMinusOf(Tensor<T> t) => e.TensorAddScalar(e.TensorNegate(t), one);

        Tensor<T>? accum = null, weightSum = null;
        var inputShape = (int[])input._shape.Clone();
        for (int corner = 0; corner < 4; corner++)
        {
            int ox = corner & 1, oy = corner >> 1;
            var weight = e.TensorMultiply(ox == 1 ? fx : OneMinusOf(fx), oy == 1 ? fy : OneMinusOf(fy));
            var valueIndex = new int[n * channels * spatial];
            var weightIndex = new int[n * spatial];
            for (int b = 0; b < n; b++)
                for (int pixel = 0; pixel < spatial; pixel++)
                {
                    int p = b * spatial + pixel;
                    int tx = x0[p] + ox, ty = y0[p] + oy;
                    bool inside = tx >= 0 && tx < width && ty >= 0 && ty < height;
                    weightIndex[p] = inside ? b * spatial + ty * width + tx : -1;
                    for (int ch = 0; ch < channels; ch++)
                        valueIndex[(b * channels + ch) * spatial + pixel] = inside ? (b * channels + ch) * spatial + ty * width + tx : -1;
                }
            var contribution = e.TensorMultiply(input, e.Reshape(weight, weightShape));
            var splatted = RecordedLinearMap("ForwardSplatScatter", contribution,
                v => ScatterAddFlat(v, valueIndex, inputShape), w => GatherFlat(w, valueIndex, inputShape));
            accum = accum is null ? splatted : e.TensorAdd(accum, splatted);
            if (normalize)
            {
                var spread = RecordedLinearMap("ForwardSplatWeightScatter", weight,
                    v => ScatterAddFlat(v, weightIndex, planeShape), w => GatherFlat(w, weightIndex, planeShape));
                weightSum = weightSum is null ? spread : e.TensorAdd(weightSum, spread);
            }
        }
        if (!normalize) return accum!;
        // Where nothing landed the forward divides by 1: add 1 there (a constant) so the division is a recorded op.
        var sums = RoIAlignPlan.ToDoubles(weightSum!);
        var empty = new double[sums.Length];
        for (int i = 0; i < sums.Length; i++) empty[i] = sums[i] == 0.0 ? 1.0 : 0.0;
        var denominator = e.TensorAdd(weightSum!, RoIAlignPlan.FromDoubles<T>(empty, planeShape));
        return e.TensorDivide(accum!, e.TensorBroadcastTo(e.Reshape(denominator, weightShape), inputShape));
    }

    // PyTorch's grid_sampler_compute_source_index with its derivative (GridSampler3D's rule, per axis).
    private static double GridSourceIndex(double normalized, int size, GridSamplePadding padding, bool align, out double slope)
    {
        double coord = align ? (normalized + 1) / 2 * (size - 1) : ((normalized + 1) * size - 1) / 2;
        slope = align ? (size - 1) / 2.0 : size / 2.0;
        if (padding == GridSamplePadding.Border) return GridClip(coord, size, ref slope);
        if (padding == GridSamplePadding.Reflection)
        {
            coord = align ? GridReflect(coord, 0, 2 * (size - 1), ref slope) : GridReflect(coord, -1, 2 * size - 1, ref slope);
            return GridClip(coord, size, ref slope);
        }
        return coord;
    }

    private static double GridClip(double coord, int size, ref double slope)
    {
        if (coord <= 0) { slope = 0; return 0; }
        if (coord >= size - 1) { slope = 0; return size - 1; }
        return coord;
    }

    private static double GridReflect(double coord, int twiceLow, int twiceHigh, ref double slope)
    {
        if (twiceLow == twiceHigh) { slope = 0; return 0; }
        double min = twiceLow / 2.0, span = (twiceHigh - twiceLow) / 2.0, sign = 1;
        coord -= min;
        if (coord < 0) { sign = -1; coord = -coord; }
        double extra = coord % span, flips = Math.Floor(coord / span);
        if (flips % 2 == 0) { slope *= sign; return extra + min; }
        slope *= -sign;
        return span - extra + min;
    }

    private static int GridVoxelIndex(double v) => double.IsNaN(v) || v <= int.MinValue || v >= int.MaxValue ? -100 : (int)v;

    /// <summary>
    /// A linear map with a known adjoint, both computed by host code, recorded so it is differentiable to any order:
    /// <c>y = apply(x)</c>, whose backward is <c>applyTranspose</c> of the upstream gradient, itself recorded with
    /// <c>apply</c> as its backward. Used for the selections of the max-type pools (gather / scatter-add at the saved
    /// winners), which no engine op expresses.
    /// </summary>
    internal static Tensor<T> RecordedLinearMap(string opName, Tensor<T> x, Func<Tensor<T>, Tensor<T>> apply,
        Func<Tensor<T>, Tensor<T>> applyTranspose)
    {
        Tensor<T> y;
        using (new NoGradScope<T>()) y = apply(x);
        DifferentiableOps.RecordUnary(opName, y, x, (g, ins, _, _, e, grads) =>
            DifferentiableOps.AccumulateGrad(grads, ins[0], RecordedLinearMap(opName + "Adjoint", g, applyTranspose, apply), e));
        return y;
    }

    /// <summary>
    /// The create-graph path of a selection op (a max-type pool): output element k is input element
    /// <paramref name="flatIndex"/>[k] (a global flat index; negative = no input). Its input gradient, the upstream
    /// gradient scatter-added to those indices, is recorded with the gather at the same indices as its adjoint.
    /// Returns false (doing nothing) when no graph is being created.
    /// </summary>
    internal static bool TrySelectionBackward(string opName, Tensor<T> gradOutput, Tensor<T> input, int[] flatIndex,
        IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        if (!CreatesGraph) return false;
        var inputShape = (int[])input._shape.Clone();
        var outputShape = (int[])gradOutput._shape.Clone();
        var grad = RecordedLinearMap(opName, gradOutput,
            g => ScatterAddFlat(g, flatIndex, inputShape),
            w => GatherFlat(w, flatIndex, outputShape));
        DifferentiableOps.AccumulateGrad(grads, input, grad, engine);
        return true;
    }

    [ScalarKernel("An index scatter-add with arbitrary, possibly repeated targets; runs only under create-graph.")]
    private static Tensor<T> ScatterAddFlat(Tensor<T> source, int[] flatIndex, int[] shape)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var src = source.IsContiguous ? source : source.Contiguous();
        var result = new Tensor<T>(shape);
        using var srcLease = src.Lease();
        var s = srcLease.Span;
        using var dstLease = result.LeaseWritable();
        var d = dstLease.Span;
        for (int k = 0; k < flatIndex.Length; k++)
            if (flatIndex[k] >= 0) d[flatIndex[k]] = ops.Add(d[flatIndex[k]], s[k]);
        return result;
    }

    private static Tensor<T> GatherFlat(Tensor<T> source, int[] flatIndex, int[] shape)
    {
        var src = source.IsContiguous ? source : source.Contiguous();
        var result = new Tensor<T>(shape);
        using var srcLease = src.Lease();
        var s = srcLease.Span;
        using var dstLease = result.LeaseWritable();
        var d = dstLease.Span;
        for (int k = 0; k < flatIndex.Length; k++)
            if (flatIndex[k] >= 0) d[k] = s[flatIndex[k]];
        return result;
    }

    /// <summary>
    /// The output padding a transposed convolution used, recovered from its output shape (it is not saved):
    /// out = (in - 1)·stride - 2·pad + k + outputPadding over the trailing spatial axes.
    /// </summary>
    internal static int[] TransposedConvOutputPadding(int[] outputShape, int[] inputShape, int[] kernelShape, int[] stride, int[] padding)
    {
        int spatial = inputShape.Length - 2;
        var outputPadding = new int[spatial];
        for (int d = 0; d < spatial; d++)
            outputPadding[d] = outputShape[2 + d] - ((inputShape[2 + d] - 1) * stride[d] - 2 * padding[d] + kernelShape[2 + d]);
        return outputPadding;
    }

    /// <summary>Global flat input indices of a 2-D max pool's winners, saved as [b, c, oh, ow, (h, w)].</summary>
    internal static int[] MaxPool2DGlobalIndices(int[,,,,] maxIndices, int[] inputShape)
    {
        int batch = maxIndices.GetLength(0), channels = maxIndices.GetLength(1);
        int outH = maxIndices.GetLength(2), outW = maxIndices.GetLength(3);
        int height = inputShape[2], width = inputShape[3];
        var global = new int[batch * channels * outH * outW];
        int k = 0;
        for (int b = 0; b < batch; b++)
            for (int c = 0; c < channels; c++)
                for (int oh = 0; oh < outH; oh++)
                    for (int ow = 0; ow < outW; ow++)
                        global[k++] = ((b * channels + c) * height + maxIndices[b, c, oh, ow, 0]) * width + maxIndices[b, c, oh, ow, 1];
        return global;
    }

    /// <summary>Global flat input indices of a 3-D max pool's winners, saved as [b, c, od, oh, ow, (d, h, w)].</summary>
    internal static int[] MaxPool3DGlobalIndices(int[,,,,,] maxIndices, int[] inputShape)
    {
        int batch = maxIndices.GetLength(0), channels = maxIndices.GetLength(1);
        int outD = maxIndices.GetLength(2), outH = maxIndices.GetLength(3), outW = maxIndices.GetLength(4);
        int depth = inputShape[2], height = inputShape[3], width = inputShape[4];
        var global = new int[batch * channels * outD * outH * outW];
        int k = 0;
        for (int b = 0; b < batch; b++)
            for (int c = 0; c < channels; c++)
                for (int od = 0; od < outD; od++)
                    for (int oh = 0; oh < outH; oh++)
                        for (int ow = 0; ow < outW; ow++)
                            global[k++] = (((b * channels + c) * depth + maxIndices[b, c, od, oh, ow, 0]) * height
                                           + maxIndices[b, c, od, oh, ow, 1]) * width + maxIndices[b, c, od, oh, ow, 2];
        return global;
    }

    /// <summary>
    /// Global flat input indices from a pool's per-plane flat winner indices (a tensor shaped like the output, planes
    /// over the leading axes and <paramref name="spatialRank"/> trailing spatial axes).
    /// </summary>
    internal static int[] PlaneTensorIndicesToGlobal(Tensor<int> perPlane, int[] outputShape, int[] inputShape, int spatialRank)
    {
        int outPlane = 1, planeSize = 1;
        for (int d = outputShape.Length - spatialRank; d < outputShape.Length; d++) outPlane *= outputShape[d];
        for (int d = inputShape.Length - spatialRank; d < inputShape.Length; d++) planeSize *= inputShape[d];
        return PlaneIndicesToGlobal(perPlane.GetFlattenedData(), outPlane, planeSize);
    }

    /// <summary>
    /// Global flat input index of each RoI-pool output's winner (-1 for an empty bin), found as
    /// <see cref="RoIPoolBackward"/> finds it.
    /// </summary>
    [ScalarKernel("Mirrors RoIPoolBackward's per-bin argmax search; runs only under create-graph.")]
    internal static int[] RoIPoolGlobalIndices(Tensor<T> input, Tensor<T> boxes, int outH, int outW, float spatialScale)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        int N = input._shape[0], C = input._shape[1], H = input._shape[2], W = input._shape[3];
        int K = boxes._shape[0];
        var result = new int[K * C * outH * outW];
        for (int i = 0; i < result.Length; i++) result[i] = -1;
        using var srcLease = input.Lease();
        var src = srcLease.Span;
        using var bLease = boxes.Lease();
        var b = bLease.Span;
        for (int k = 0; k < K; k++)
        {
            int n = (int)numOps.ToDouble(b[k * 5]);
            if (n < 0 || n >= N) continue;
            for (int c = 0; c < C; c++)
            {
                int planeBase = (n * C + c) * H * W;
                for (int ph = 0; ph < outH; ph++)
                    for (int pw = 0; pw < outW; pw++)
                    {
                        var (hs, he, ws, we) = CpuEngine.RoIPoolWindow(b, k, spatialScale, outH, outW, ph, pw, H, W);
                        if (hs >= he || ws >= we) continue;
                        double best = double.NegativeInfinity;
                        int bestY = hs, bestX = ws;
                        for (int yy = hs; yy < he; yy++)
                            for (int xx = ws; xx < we; xx++)
                            {
                                double v = numOps.ToDouble(src[planeBase + yy * W + xx]);
                                if (v > best) { best = v; bestY = yy; bestX = xx; }
                            }
                        result[((k * C + c) * outH + ph) * outW + pw] = planeBase + bestY * W + bestX;
                    }
            }
        }
        return result;
    }

    /// <summary>
    /// A window reduction as a recorded linear map: output k = weight[k]·Σ input[taps[k][j]] (global flat indices),
    /// differentiable to any order (its adjoint scatters weight[k]·g[k] back to the taps).
    /// </summary>
    internal static Tensor<T> RecordedWindowSum(string opName, Tensor<T> x, int[][] taps, double[] weight, int[] outputShape)
    {
        var inputShape = (int[])x._shape.Clone();
        return RecordedLinearMap(opName, x,
            v => WindowSumApply(v, taps, weight, outputShape),
            g => WindowSumTranspose(g, taps, weight, inputShape));
    }

    [ScalarKernel("A window gather over arbitrary tap lists; runs only under create-graph.")]
    internal static Tensor<T> WindowSumApply(Tensor<T> source, int[][] taps, double[] weight, int[] outputShape)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var src = source.IsContiguous ? source : source.Contiguous();
        var result = new Tensor<T>(outputShape);
        using var srcLease = src.Lease();
        var s = srcLease.Span;
        using var dstLease = result.LeaseWritable();
        var d = dstLease.Span;
        for (int k = 0; k < taps.Length; k++)
        {
            double acc = 0;
            foreach (int t in taps[k]) acc += ops.ToDouble(s[t]);
            d[k] = ops.FromDouble(acc * weight[k]);
        }
        return result;
    }

    [ScalarKernel("A window scatter-add over arbitrary, overlapping tap lists; runs only under create-graph.")]
    internal static Tensor<T> WindowSumTranspose(Tensor<T> gradient, int[][] taps, double[] weight, int[] inputShape)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var src = gradient.IsContiguous ? gradient : gradient.Contiguous();
        var acc = new double[ShapeLength(inputShape)];
        using (var srcLease = src.Lease())
        {
            var s = srcLease.Span;
            for (int k = 0; k < taps.Length; k++)
            {
                double g = ops.ToDouble(s[k]) * weight[k];
                foreach (int t in taps[k]) acc[t] += g;
            }
        }
        var result = new Tensor<T>(inputShape);
        using var dstLease = result.LeaseWritable();
        var d = dstLease.Span;
        for (int i = 0; i < acc.Length; i++) d[i] = ops.FromDouble(acc[i]);
        return result;
    }


    /// <summary>Global flat input indices from per-plane flat indices (one plane per leading (batch, channel) pair).</summary>
    internal static int[] PlaneIndicesToGlobal(int[] perPlane, int outPlane, int planeSize)
    {
        var global = new int[perPlane.Length];
        for (int k = 0; k < perPlane.Length; k++)
            global[k] = perPlane[k] < 0 ? -1 : (k / outPlane) * planeSize + perPlane[k];
        return global;
    }

    /// <summary>
    /// The create-graph path of an op that is linear in each of its inputs (a convolution in (input, kernel), a pool or
    /// resampling in its input): the op's own fast backward gives each input gradient
    /// <c>G_i = B_i(g, x)</c>, recorded as a node of (g, the other inputs). For a cotangent <c>w</c> on <c>G_i</c>,
    /// <c>⟨w, G_i⟩ = ⟨g, f(x with x_i := w)⟩</c>, so its gradient is <paramref name="forward"/> at
    /// <c>x_i := w</c> for g and the op's backward at <c>x_i := w</c> for every other input — both again recorded under
    /// create-graph, so any order differentiates. <c>G_i</c> does not depend on <c>x_i</c> itself.
    /// </summary>
    /// <param name="forward">The op's forward with the same hyper-parameters, as recorded engine ops.</param>
    /// <param name="linearInputs">The inputs the op is linear in (null: all). A gradient for any other input is
    /// recorded with <paramref name="nonlinearSecondOrder"/>, or left as a constant when that is null.</param>
    /// <param name="nonlinearSecondOrder">Message of the <see cref="NotSupportedException"/> a second backward through a
    /// non-linear input's gradient raises (as PyTorch raises for an unimplemented double backward).</param>
    internal static bool TryMultilinearBackward(string opName, BackwardFunction<T> backward, Tensor<T> gradOutput,
        Tensor<T>[] inputs, Tensor<T> output, object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads,
        Func<IEngine, Tensor<T>[], Tensor<T>> forward, bool[]? linearInputs = null, string? nonlinearSecondOrder = null)
    {
        if (!CreatesGraph) return false;
        Tensor<T>?[] kernel;
        using (new NoGradScope<T>()) kernel = KernelGrads(backward, gradOutput, inputs, output, savedState, engine);
        for (int i = 0; i < inputs.Length; i++)
        {
            var gi = kernel[i];
            if (gi is null) continue;
            if (ReferenceEquals(gi, gradOutput))
            {
                DifferentiableOps.AccumulateGrad(grads, inputs[i], gi, engine);
                continue;
            }
            int index = i;
            var nodeInputs = new List<Tensor<T>>(inputs.Length) { gradOutput };
            for (int j = 0; j < inputs.Length; j++) if (j != i) nodeInputs.Add(inputs[j]);
            bool linear = linearInputs is null || linearInputs[i];
            BackwardFunction<T> nodeBackward;
            if (linear)
            {
                nodeBackward = (w, _, _, _, e, g) =>
                {
                    var modified = (Tensor<T>[])inputs.Clone();
                    modified[index] = w;
                    if (DifferentiableOps.IsGradientRequired(gradOutput))
                        DifferentiableOps.AccumulateGrad(g, gradOutput, forward(e, modified), e);
                    bool others = false;
                    for (int j = 0; j < inputs.Length; j++)
                        if (j != index && DifferentiableOps.IsGradientRequired(inputs[j])) others = true;
                    if (!others) return;
                    var cross = KernelGrads(backward, gradOutput, modified, output, savedState, e);
                    for (int j = 0; j < inputs.Length; j++)
                        if (j != index && cross[j] is { } c && DifferentiableOps.IsGradientRequired(inputs[j]))
                            DifferentiableOps.AccumulateGrad(g, inputs[j], c, e);
                };
            }
            else if (nonlinearSecondOrder is not null)
            {
                string message = nonlinearSecondOrder;
                nodeBackward = (_, _, _, _, _, _) => throw new NotSupportedException(message);
            }
            else
            {
                DifferentiableOps.AccumulateGrad(grads, inputs[i], gi, engine);
                continue;
            }
            DifferentiableOps.RecordIfActive(opName + "Grad" + i, gi, nodeInputs.ToArray(), nodeBackward);
            DifferentiableOps.AccumulateGrad(grads, inputs[i], gi, engine);
        }
        return true;
    }
}
