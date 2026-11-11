using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Autodiff;

/// <summary>
/// Double-backward support for the pooling, attention, deformable-convolution, mesh and graph ops whose fused kernels
/// record nothing (see <c>BackwardFunctions.HigherOrder.cs</c> for the shared mechanism).
/// </summary>
internal static partial class BackwardFunctions<T>
{
    /// <summary>
    /// The upstream gradient of a max-type selection scattered to the saved winners (flat input offsets, one per output
    /// element) as a recorded sparse map, whose own backward is the gather at those winners.
    /// </summary>
    internal static Tensor<T> ArgmaxScatterRecorded(Tensor<T> gradOutput, int[] inputShape, int[] argmax)
    {
        int inputLength = 1;
        foreach (int d in inputShape) inputLength *= d;
        var to = new List<int>(argmax.Length);
        var from = new List<int>(argmax.Length);
        for (int i = 0; i < argmax.Length; i++)
        {
            if (argmax[i] < 0 || argmax[i] >= inputLength) continue;
            to.Add(argmax[i]);
            from.Add(i);
        }
        return SparseMap(gradOutput, (int[])inputShape.Clone(), to.ToArray(), from.ToArray(), null);
    }

    /// <summary>
    /// A backward that differentiates <paramref name="composite"/> (the op's forward as recorded engine ops, from the
    /// op's own input tensors): under create-graph with a graph-keeping nested tape (<see cref="TryCompositeBackward"/>),
    /// otherwise by replaying it under a plain nested tape (<see cref="ReplayUnderTape"/>). For ops with no hand-written
    /// first-order kernel for every input.
    /// </summary>
    internal static BackwardFunction<T> CompositeBackward(Func<IEngine, Tensor<T>[], Tensor<T>> composite)
        => (gradOutput, inputs, output, savedState, engine, grads) =>
        {
            if (TryCompositeBackward(gradOutput, inputs, engine, grads, e => composite(e, inputs))) return;
            ReplayUnderTape(e => composite(e, inputs))(gradOutput, inputs, output, savedState, engine, grads);
        };

    /// <summary>Gathers rows of a [rows, cols]-major tensor: out[i, c] = x[index[i], c], 0 for an invalid index.</summary>
    private static Tensor<T> GatherRowsRecorded(Tensor<T> x, int rows, int cols, int[] index, int[] outputShape, int rowOffset = 0)
    {
        var to = new List<int>(index.Length * cols);
        var from = new List<int>(index.Length * cols);
        for (int i = 0; i < index.Length; i++)
        {
            int n = index[i];
            if (n < 0 || n >= rows) continue;
            for (int c = 0; c < cols; c++)
            {
                to.Add(i * cols + c);
                from.Add((rowOffset + n) * cols + c);
            }
        }
        return SparseMap(x, outputShape, to.ToArray(), from.ToArray(), null);
    }

    /// <summary>x·Wᵀ + b over [rows, in]·[out, in] with b [out], as recorded ops.</summary>
    private static Tensor<T> LinearRowsRecorded(IEngine e, Tensor<T> x, Tensor<T> weights, Tensor<T> bias)
    {
        int rows = x._shape[0], outputs = weights._shape[0];
        var y = e.TensorMatMul(x, e.TensorPermute(weights, new[] { 1, 0 }));
        return e.TensorAdd(y, e.TensorBroadcastTo(e.Reshape(bias, new[] { 1, outputs }), new[] { rows, outputs }));
    }

    /// <summary>Spiral convolution (MeshConvolutionOperations.SpiralConv) as recorded ops.</summary>
    internal static Tensor<T> SpiralConvRecorded(IEngine e, Tensor<T> vertexFeatures, int[] spiralIndices, int spiralLength,
        Tensor<T> weights, Tensor<T> bias)
    {
        int vertices = vertexFeatures._shape[0], channels = vertexFeatures._shape[1];
        var to = new List<int>();
        var from = new List<int>();
        for (int v = 0; v < vertices; v++)
            for (int s = 0; s < spiralLength; s++)
            {
                int n = spiralIndices[v * spiralLength + s];
                if (n < 0 || n >= vertices) continue;
                for (int c = 0; c < channels; c++)
                {
                    to.Add((v * spiralLength + s) * channels + c);
                    from.Add(n * channels + c);
                }
            }
        var gathered = SparseMap(vertexFeatures, new[] { vertices, spiralLength * channels }, to.ToArray(), from.ToArray(), null);
        return LinearRowsRecorded(e, gathered, weights, bias);
    }

    /// <summary>
    /// Diffusion convolution (MeshConvolutionOperations.DiffusionConv: the 4-term Taylor series of exp(−tL) applied to
    /// the features, then a linear map) as recorded ops; differentiable in the features, the Laplacian, the weights and
    /// the bias.
    /// </summary>
    internal static Tensor<T> DiffusionConvRecorded(IEngine e, Tensor<T> x, Tensor<T> laplacian, Tensor<T> weights,
        Tensor<T> bias, double t)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var l1 = e.TensorMatMul(laplacian, x);
        var l2 = e.TensorMatMul(laplacian, l1);
        var l3 = e.TensorMatMul(laplacian, l2);
        var l4 = e.TensorMatMul(laplacian, l3);
        var d = e.TensorSubtract(x, e.TensorMultiplyScalar(l1, ops.FromDouble(t)));
        d = e.TensorAdd(d, e.TensorMultiplyScalar(l2, ops.FromDouble(t * t / 2.0)));
        d = e.TensorSubtract(d, e.TensorMultiplyScalar(l3, ops.FromDouble(t * t * t / 6.0)));
        d = e.TensorAdd(d, e.TensorMultiplyScalar(l4, ops.FromDouble(t * t * t * t / 24.0)));
        return LinearRowsRecorded(e, d, weights, bias);
    }

    /// <summary>
    /// The cotangent mesh Laplacian (MeshConvolutionOperations.ComputeMeshLaplacian, Cotangent or Normalized) as recorded
    /// ops of the vertex positions; the faces (and which weights are degenerate) are constants.
    /// </summary>
    internal static Tensor<T> MeshLaplacianRecorded(IEngine e, Tensor<T> vertices, int[] faces, bool normalized)
    {
        int v = vertices._shape[0], f = faces.Length / 3;
        var corner = new Tensor<T>[3];
        for (int k = 0; k < 3; k++)
        {
            var idx = new int[f];
            for (int i = 0; i < f; i++) idx[i] = faces[i * 3 + k];
            corner[k] = GatherRowsRecorded(vertices, v, 3, idx, new[] { f, 3 });
        }

        Tensor<T> Column(Tensor<T> m, int c) => e.Reshape(e.TensorNarrow(m, 1, c, 1), new[] { f });
        // cot of the angle between a and b: (a·b)/|a×b|, 0 where |a×b| ≤ 1e-10 (the forward's degenerate rule).
        Tensor<T> Cot(Tensor<T> a, Tensor<T> b)
        {
            var dot = e.ReduceSum(e.TensorMultiply(a, b), new[] { 1 }, keepDims: false);
            var cx = e.TensorSubtract(e.TensorMultiply(Column(a, 1), Column(b, 2)), e.TensorMultiply(Column(a, 2), Column(b, 1)));
            var cy = e.TensorSubtract(e.TensorMultiply(Column(a, 2), Column(b, 0)), e.TensorMultiply(Column(a, 0), Column(b, 2)));
            var cz = e.TensorSubtract(e.TensorMultiply(Column(a, 0), Column(b, 1)), e.TensorMultiply(Column(a, 1), Column(b, 0)));
            var squared = e.TensorAdd(e.TensorAdd(e.TensorMultiply(cx, cx), e.TensorMultiply(cy, cy)), e.TensorMultiply(cz, cz));
            var squaredValues = RoIAlignPlan.ToDoubles(squared);
            var keep = new double[f];
            var fill = new double[f];
            for (int i = 0; i < f; i++)
            {
                bool ok = Math.Sqrt(squaredValues[i]) > 1e-10;
                keep[i] = ok ? 1 : 0;
                fill[i] = ok ? 0 : 1;
            }
            var keepT = RoIAlignPlan.FromDoubles<T>(keep, new[] { f });
            var safe = e.TensorAdd(e.TensorMultiply(squared, keepT), RoIAlignPlan.FromDoubles<T>(fill, new[] { f }));
            return e.TensorMultiply(e.TensorDivide(dot, e.TensorSqrt(safe)), keepT);
        }

        var cot0 = Cot(e.TensorSubtract(corner[1], corner[0]), e.TensorSubtract(corner[2], corner[0]));
        var cot1 = Cot(e.TensorSubtract(corner[0], corner[1]), e.TensorSubtract(corner[2], corner[1]));
        var cot2 = Cot(e.TensorSubtract(corner[0], corner[2]), e.TensorSubtract(corner[1], corner[2]));

        // Off-diagonal weights −½·cot, added symmetrically per face (a self-pair only touches the diagonal, which the
        // forward overwrites).
        Tensor<T>? offDiagonal = null;
        void Scatter(Tensor<T> cot, int first, int second)
        {
            var to = new List<int>();
            var from = new List<int>();
            var coef = new List<double>();
            for (int i = 0; i < f; i++)
            {
                int a = faces[i * 3 + first], b = faces[i * 3 + second];
                if (a == b) continue;
                to.Add(a * v + b); from.Add(i); coef.Add(-0.5);
                to.Add(b * v + a); from.Add(i); coef.Add(-0.5);
            }
            var part = SparseMap(cot, new[] { v * v }, to.ToArray(), from.ToArray(), coef.ToArray());
            offDiagonal = offDiagonal is null ? part : e.TensorAdd(offDiagonal, part);
        }
        Scatter(cot0, 1, 2);
        Scatter(cot1, 0, 2);
        Scatter(cot2, 0, 1);

        var rowSums = e.ReduceSum(e.Reshape(offDiagonal!, new[] { v, v }), new[] { 1 }, keepDims: false);
        var diagTo = new int[v];
        var diagFrom = new int[v];
        var minusOne = new double[v];
        for (int i = 0; i < v; i++) { diagTo[i] = i * v + i; diagFrom[i] = i; minusOne[i] = -1; }
        var laplacian = e.Reshape(e.TensorAdd(offDiagonal!, SparseMap(rowSums, new[] { v * v }, diagTo, diagFrom, minusOne)), new[] { v, v });
        if (!normalized) return laplacian;

        // D^(-1/2)·L·D^(-1/2) with D the diagonal of L, 0 where that is not above 1e-10 (the forward's rule).
        var flat = e.Reshape(laplacian, new[] { v * v });
        var d = SparseMap(flat, new[] { v }, diagFrom, diagTo, null);
        var dValues = RoIAlignPlan.ToDoubles(d);
        var keepD = new double[v];
        var fillD = new double[v];
        for (int i = 0; i < v; i++) { bool ok = dValues[i] > 1e-10; keepD[i] = ok ? 1 : 0; fillD[i] = ok ? 0 : 1; }
        var keepDT = RoIAlignPlan.FromDoubles<T>(keepD, new[] { v });
        var safeD = e.TensorAdd(e.TensorMultiply(d, keepDT), RoIAlignPlan.FromDoubles<T>(fillD, new[] { v }));
        var s = e.TensorMultiply(e.TensorDivide(keepDT, e.TensorSqrt(safeD)), keepDT);
        var scale = e.TensorMatMul(e.Reshape(s, new[] { v, 1 }), e.Reshape(s, new[] { 1, v }));
        return e.TensorMultiply(laplacian, scale);
    }

    /// <summary>
    /// Single-head graph attention (CpuEngine.GraphAttention) over [batch, nodes, features] as recorded ops: per-edge
    /// LeakyReLU(a_src·h_src + a_tgt·h_tgt), a softmax over each target's incoming edges, and the weighted sum of source
    /// features into each target. The per-target maximum subtracted inside the exponential is a constant (the softmax is
    /// invariant to it).
    /// </summary>
    internal static Tensor<T> GraphAttentionRecorded(IEngine e, Tensor<T> nodes, int[] sources, int[] targets,
        Tensor<T> attentionSource, Tensor<T> attentionTarget, double leakyReluAlpha)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int batch = nodes._shape[0], n = nodes._shape[1], features = nodes._shape[2], edges = sources.Length;
        var srcIdx = new int[batch * edges];
        var tgtIdx = new int[batch * edges];
        for (int b = 0; b < batch; b++)
            for (int k = 0; k < edges; k++)
            {
                srcIdx[b * edges + k] = sources[k] < 0 || sources[k] >= n ? -1 : b * n + sources[k];
                tgtIdx[b * edges + k] = targets[k] < 0 || targets[k] >= n ? -1 : b * n + targets[k];
            }
        var hs = GatherRowsRecorded(nodes, batch * n, features, srcIdx, new[] { batch * edges, features });
        var ht = GatherRowsRecorded(nodes, batch * n, features, tgtIdx, new[] { batch * edges, features });
        var rowsShape = new[] { batch * edges, features };
        var scores = e.TensorAdd(
            e.ReduceSum(e.TensorMultiply(hs, e.TensorBroadcastTo(e.Reshape(attentionSource, new[] { 1, features }), rowsShape)), new[] { 1 }, keepDims: false),
            e.ReduceSum(e.TensorMultiply(ht, e.TensorBroadcastTo(e.Reshape(attentionTarget, new[] { 1, features }), rowsShape)), new[] { 1 }, keepDims: false));
        scores = e.LeakyReLU(scores, ops.FromDouble(leakyReluAlpha));

        var scoreValues = RoIAlignPlan.ToDoubles(scores);
        var groupMax = new double[batch * n];
        for (int i = 0; i < groupMax.Length; i++) groupMax[i] = double.NegativeInfinity;
        for (int i = 0; i < tgtIdx.Length; i++)
            if (tgtIdx[i] >= 0) groupMax[tgtIdx[i]] = Math.Max(groupMax[tgtIdx[i]], scoreValues[i]);
        var shift = new double[batch * edges];
        for (int i = 0; i < shift.Length; i++) shift[i] = tgtIdx[i] >= 0 ? groupMax[tgtIdx[i]] : 0;
        var numerators = e.TensorExp(e.TensorSubtract(scores, RoIAlignPlan.FromDoubles<T>(shift, new[] { batch * edges })));

        var toNode = new List<int>();
        var fromEdge = new List<int>();
        for (int i = 0; i < tgtIdx.Length; i++)
            if (tgtIdx[i] >= 0) { toNode.Add(tgtIdx[i]); fromEdge.Add(i); }
        var sums = SparseMap(numerators, new[] { batch * n }, toNode.ToArray(), fromEdge.ToArray(), null);
        // Each edge's denominator; an edge with no valid target contributes nothing, so its denominator is 1.
        var invalid = new double[batch * edges];
        for (int i = 0; i < invalid.Length; i++) invalid[i] = tgtIdx[i] >= 0 ? 0 : 1;
        var denominators = e.TensorAdd(SparseMap(sums, new[] { batch * edges }, fromEdge.ToArray(), toNode.ToArray(), null),
            RoIAlignPlan.FromDoubles<T>(invalid, new[] { batch * edges }));
        var weights = e.TensorDivide(numerators, denominators);

        var weighted = e.TensorMultiply(hs, e.TensorBroadcastTo(e.Reshape(weights, new[] { batch * edges, 1 }), rowsShape));
        var to = new List<int>();
        var from = new List<int>();
        for (int i = 0; i < tgtIdx.Length; i++)
        {
            if (tgtIdx[i] < 0 || srcIdx[i] < 0) continue;
            for (int c = 0; c < features; c++) { to.Add(tgtIdx[i] * features + c); from.Add(i * features + c); }
        }
        return SparseMap(weighted, new[] { batch, n, features }, to.ToArray(), from.ToArray(), null);
    }

    /// <summary>
    /// Modulated deformable convolution (CpuEngine.DeformableConv2DGrouped: per deformable group a block of 2·K offset
    /// channels, the K y-offsets then the K x-offsets, and K mask channels) as recorded ops. Each sample is bilinear in
    /// the input with weights linear in its offset inside the sampled cell; the cell (the floor of the position) and
    /// the out-of-image corners are constants fixed by the forward's values. Differentiable in input, kernel, offset and
    /// mask.
    /// </summary>
    internal static Tensor<T> DeformableConvRecorded(IEngine e, Tensor<T> input, Tensor<T> kernel, Tensor<T> offset,
        Tensor<T>? mask, int[] stride, int[] padding, int[] dilation, int groups, int deformGroups)
    {
        int batch = input._shape[0], channels = input._shape[1], height = input._shape[2], width = input._shape[3];
        int outChannels = kernel._shape[0], kh = kernel._shape[2], kw = kernel._shape[3];
        int taps = kh * kw;
        int outH = (height + 2 * padding[0] - (dilation[0] * (kh - 1) + 1)) / stride[0] + 1;
        int outW = (width + 2 * padding[1] - (dilation[1] * (kw - 1) + 1)) / stride[1] + 1;
        int plane = outH * outW;
        int perDeform = channels / deformGroups;
        var offsetValues = RoIAlignPlan.ToDoubles(offset);
        int offChannels = 2 * taps * deformGroups;

        // Per deformable group: the bilinear-weight tensors [batch, 1, taps, outH, outW] and, per corner, the gathered
        // input [batch, perDeform, taps, outH, outW].
        var columns = new List<Tensor<T>>(deformGroups);
        for (int dg = 0; dg < deformGroups; dg++)
        {
            var offY = e.TensorNarrow(offset, 1, dg * 2 * taps, taps);            // [B, taps, outH, outW]
            var offX = e.TensorNarrow(offset, 1, dg * 2 * taps + taps, taps);
            int cells = batch * taps * plane;
            var baseY = new double[cells];
            var baseX = new double[cells];
            var cornerIndex = new int[4][];
            for (int k = 0; k < 4; k++) cornerIndex[k] = new int[cells];
            var outside = new bool[cells];
            for (int b = 0; b < batch; b++)
                for (int t = 0; t < taps; t++)
                    for (int oh = 0; oh < outH; oh++)
                        for (int ow = 0; ow < outW; ow++)
                        {
                            int cell = ((b * taps + t) * outH + oh) * outW + ow;
                            int i = t / kw, j = t % kw;
                            double py = oh * stride[0] - padding[0] + i * dilation[0]
                                + offsetValues[((b * offChannels + dg * 2 * taps + t) * outH + oh) * outW + ow];
                            double px = ow * stride[1] - padding[1] + j * dilation[1]
                                + offsetValues[((b * offChannels + dg * 2 * taps + taps + t) * outH + oh) * outW + ow];
                            outside[cell] = py < -1 || py > height || px < -1 || px > width;
                            int y0 = (int)Math.Floor(py), x0 = (int)Math.Floor(px);
                            // ly = py − y0 = offY − (y0 − base position): the constant part.
                            baseY[cell] = y0 - (oh * stride[0] - padding[0] + i * dilation[0]);
                            baseX[cell] = x0 - (ow * stride[1] - padding[1] + j * dilation[1]);
                            for (int k = 0; k < 4; k++)
                            {
                                int yy = y0 + (k >> 1), xx = x0 + (k & 1);
                                cornerIndex[k][cell] = outside[cell] || yy < 0 || yy >= height || xx < 0 || xx >= width ? -1 : yy * width + xx;
                            }
                        }

            var cellShape = new[] { batch, taps, outH, outW };
            var ly = e.TensorSubtract(offY, RoIAlignPlan.FromDoubles<T>(baseY, cellShape));
            var lx = e.TensorSubtract(offX, RoIAlignPlan.FromDoubles<T>(baseX, cellShape));
            var oneMinusLy = OneMinus(e, ly);
            var oneMinusLx = OneMinus(e, lx);
            var cornerWeights = new[]
            {
                e.TensorMultiply(oneMinusLy, oneMinusLx), e.TensorMultiply(oneMinusLy, lx),
                e.TensorMultiply(ly, oneMinusLx), e.TensorMultiply(ly, lx),
            };
            var colShape = new[] { batch, perDeform, taps, outH, outW };
            var weightShape = new[] { batch, 1, taps, outH, outW };
            Tensor<T>? column = null;
            for (int k = 0; k < 4; k++)
            {
                var to = new List<int>();
                var from = new List<int>();
                for (int b = 0; b < batch; b++)
                    for (int c = 0; c < perDeform; c++)
                        for (int t = 0; t < taps; t++)
                            for (int p = 0; p < plane; p++)
                            {
                                int cell = (b * taps + t) * plane + p;
                                int src = cornerIndex[k][cell];
                                if (src < 0) continue;
                                to.Add(((b * perDeform + c) * taps + t) * plane + p);
                                from.Add((b * channels + dg * perDeform + c) * height * width + src);
                            }
                var gathered = SparseMap(input, colShape, to.ToArray(), from.ToArray(), null);
                var term = e.TensorMultiply(gathered, e.TensorBroadcastTo(e.Reshape(cornerWeights[k], weightShape), colShape));
                column = column is null ? term : e.TensorAdd(column, term);
            }
            if (mask is not null)
            {
                var maskGroup = e.Reshape(e.TensorNarrow(mask, 1, dg * taps, taps), weightShape);
                column = e.TensorMultiply(column!, e.TensorBroadcastTo(maskGroup, colShape));
            }
            columns.Add(column!);
        }
        var allColumns = columns.Count == 1 ? columns[0] : e.TensorConcatenate(columns.ToArray(), 1); // [B, C, taps, outH, outW]

        int inPerGroup = channels / groups, outPerGroup = outChannels / groups;
        var perBatch = new List<Tensor<T>>(batch);
        for (int b = 0; b < batch; b++)
        {
            var colB = RecordedOps<T>.At(e, allColumns, 0, b);                       // [C, taps, outH, outW]
            var groupOutputs = new List<Tensor<T>>(groups);
            for (int g = 0; g < groups; g++)
            {
                var colG = e.Reshape(e.TensorNarrow(colB, 0, g * inPerGroup, inPerGroup), new[] { inPerGroup * taps, plane });
                var kernelG = e.Reshape(e.TensorNarrow(kernel, 0, g * outPerGroup, outPerGroup), new[] { outPerGroup, inPerGroup * taps });
                groupOutputs.Add(e.TensorMatMul(kernelG, colG));                       // [outPerGroup, plane]
            }
            var outB = groupOutputs.Count == 1 ? groupOutputs[0] : e.TensorConcatenate(groupOutputs.ToArray(), 0);
            perBatch.Add(e.Reshape(outB, new[] { outChannels, outH, outW }));
        }
        return RecordedOps<T>.Stack(e, perBatch, 0);
    }

    /// <summary>
    /// The flat input offset of each 1-D max-pool window's first maximum (the forward's tie rule), for a node that
    /// saved no indices.
    /// </summary>
    internal static int[] MaxPool1DArgmax(Tensor<T> input, int kernelSize, int stride)
    {
        var x = RoIAlignPlan.ToDoubles(input);
        int rows = input._shape[0] * input._shape[1], width = input._shape[2];
        int outW = (width - kernelSize) / stride + 1;
        var argmax = new int[rows * outW];
        for (int r = 0; r < rows; r++)
            for (int o = 0; o < outW; o++)
            {
                int best = r * width + o * stride;
                for (int k = 1; k < kernelSize; k++)
                    if (x[r * width + o * stride + k] > x[best]) best = r * width + o * stride + k;
                argmax[r * outW + o] = best;
            }
        return argmax;
    }

    /// <summary>
    /// A constant additive mask (−1e30 where the forward's attention weight is exactly zero, 0 elsewhere): a fused
    /// forward's causal (or padding) masking, read off the weights it saved.
    /// </summary>
    internal static Tensor<T> MaskFromZeroWeights(Tensor<T> weights)
    {
        var values = RoIAlignPlan.ToDoubles(weights);
        var mask = new double[values.Length];
        for (int i = 0; i < values.Length; i++) mask[i] = values[i] == 0.0 ? -1e30 : 0.0;
        return RoIAlignPlan.FromDoubles<T>(mask, (int[])weights._shape.Clone());
    }

    /// <summary>
    /// softmax(q·kᵀ·scale + bias [+ causal mask])·v over [batch, heads, seq, dim] tensors as recorded ops, query head h
    /// reading key/value head h / <paramref name="queriesPerKv"/> (grouped-query attention; 1 for ordinary attention).
    /// The causal mask lets query i see keys j ≤ i + (Sk − Sq), the alignment the fused kernels use for a KV cache.
    /// </summary>
    internal static Tensor<T> AttentionRecorded(IEngine e, Tensor<T> query, Tensor<T> key, Tensor<T> value, double scale,
        bool isCausal, Tensor<T>? bias, int queriesPerKv)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int batch = query._shape[0], heads = query._shape[1], sq = query._shape[2];
        int sk = key._shape[2];
        Tensor<T>? causal = null;
        if (isCausal)
        {
            var mask = new double[batch * sq * sk];
            for (int b = 0; b < batch; b++)
                for (int i = 0; i < sq; i++)
                    for (int j = 0; j < sk; j++)
                        if (j > i + (sk - sq)) mask[(b * sq + i) * sk + j] = -1e30;
            causal = RoIAlignPlan.FromDoubles<T>(mask, new[] { batch, sq, sk });
        }
        var scaleT = ops.FromDouble(scale);

        var outputs = new List<Tensor<T>>(heads);
        for (int h = 0; h < heads; h++)
        {
            var qh = RecordedOps<T>.At(e, query, 1, h);
            var kh = RecordedOps<T>.At(e, key, 1, h / queriesPerKv);
            var vh = RecordedOps<T>.At(e, value, 1, h / queriesPerKv);
            var scores = e.TensorMultiplyScalar(e.BatchMatMul(qh, e.TensorPermute(kh, new[] { 0, 2, 1 })), scaleT);
            if (bias is not null)
            {
                var biasHead = bias.Rank == 4 ? RecordedOps<T>.At(e, bias, 1, bias._shape[1] == 1 ? 0 : h) : bias;
                scores = e.TensorAdd(scores, biasHead.Rank == scores.Rank && ShapesEqual(biasHead._shape, scores._shape)
                    ? biasHead
                    : e.TensorBroadcastTo(biasHead, scores._shape));
            }
            if (causal is not null) scores = e.TensorAdd(scores, causal);
            outputs.Add(e.BatchMatMul(e.Softmax(scores, -1), vh));
        }
        return RecordedOps<T>.Stack(e, outputs, 1);
    }
}
