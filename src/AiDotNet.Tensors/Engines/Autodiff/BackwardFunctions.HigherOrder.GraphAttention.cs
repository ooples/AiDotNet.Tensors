// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Autodiff;

internal static partial class BackwardFunctions<T>
{
    /// <summary>
    /// Sparse graph attention (CpuEngine.GraphAttentionCsr) as recorded ops, for its create-graph backward: the fused
    /// kernels are first order. Gathers and the per-row sums are <see cref="SparseMap"/>s over the CSR pattern. The
    /// softmax's max shift, the LeakyReLU slope pattern and the dropout keep mask enter as constants: the shift cancels in
    /// the softmax, and the slope and mask are piecewise constant in the inputs, so every derivative is exact (a.e.).
    /// </summary>
    internal static Tensor<T> GraphAttentionCsrRecorded(IEngine e, Tensor<T> transformed, Tensor<T> attention,
        CsrGraph graph, double slope, double rate, uint seed)
    {
        int heads = attention._shape[0], features = attention._shape[1] / 2;
        int n = graph.NodeCount, edges = graph.EdgeCount;
        int rows = transformed.Length / (heads * features);
        int batch = rows / n;
        var rowPtr = graph.RowPointers;
        var col = graph.ColumnIndices;
        var rowOf = new int[edges];
        for (int i = 0; i < n; i++)
            for (int k = rowPtr[i]; k < rowPtr[i + 1]; k++) rowOf[k] = i;
        int be = batch * edges;

        // Attention halves [1, H, F] gathered from attention [H, 2F].
        var selfFrom = new int[heads * features];
        var neighFrom = new int[heads * features];
        var identity = new int[heads * features];
        for (int h = 0; h < heads; h++)
            for (int f = 0; f < features; f++)
            {
                int k = h * features + f;
                identity[k] = k;
                selfFrom[k] = h * 2 * features + f;
                neighFrom[k] = h * 2 * features + features + f;
            }
        var hf = new[] { rows, heads, features };
        var wh = e.Reshape(transformed, hf);
        var aSelf = e.TensorBroadcastTo(SparseMap(attention, new[] { 1, heads, features }, identity, selfFrom, null), hf);
        var aNeigh = e.TensorBroadcastTo(SparseMap(attention, new[] { 1, heads, features }, identity, neighFrom, null), hf);
        var sSelf = e.ReduceSum(e.TensorMultiply(wh, aSelf), new[] { 2 }, keepDims: false);   // [rows, H]
        var sNeigh = e.ReduceSum(e.TensorMultiply(wh, aNeigh), new[] { 2 }, keepDims: false);

        // Per edge (b, k) and head: the source row's and the neighbour's score.
        var edgeTo = new int[be * heads];
        var rowFrom = new int[be * heads];
        var colFrom = new int[be * heads];
        for (int b = 0; b < batch; b++)
            for (int k = 0; k < edges; k++)
                for (int h = 0; h < heads; h++)
                {
                    int t = (b * edges + k) * heads + h;
                    edgeTo[t] = t;
                    rowFrom[t] = (b * n + rowOf[k]) * heads + h;
                    colFrom[t] = (b * n + col[k]) * heads + h;
                }
        var eh = new[] { be, heads };
        var z = e.TensorAdd(SparseMap(sSelf, eh, edgeTo, rowFrom, null), SparseMap(sNeigh, eh, edgeTo, colFrom, null));

        // LeakyReLU, the row max shift and the dropout scale as constants of this forward.
        var zv = RoIAlignPlan.ToDoubles(z);
        var leak = new double[zv.Length];
        var ev = new double[zv.Length];
        for (int t = 0; t < zv.Length; t++) { leak[t] = zv[t] > 0 ? 1.0 : slope; ev[t] = zv[t] * leak[t]; }
        var shift = new double[zv.Length];
        var keep = new double[zv.Length];
        double scale = rate > 0 ? 1.0 / (1.0 - rate) : 1.0;
        for (int b = 0; b < batch; b++)
            for (int i = 0; i < n; i++)
                for (int h = 0; h < heads; h++)
                {
                    double max = double.NegativeInfinity;
                    for (int k = rowPtr[i]; k < rowPtr[i + 1]; k++) max = Math.Max(max, ev[(b * edges + k) * heads + h]);
                    for (int k = rowPtr[i]; k < rowPtr[i + 1]; k++)
                    {
                        int t = (b * edges + k) * heads + h;
                        shift[t] = max;
                        keep[t] = rate > 0 && CpuEngine.GraphAttentionUniform(seed, (uint)(((long)b * edges + k) * heads + h)) < rate
                            ? 0.0 : scale;
                    }
                }
        var leaky = e.TensorMultiply(z, RoIAlignPlan.FromDoubles<T>(leak, eh));
        var ex = e.TensorExp(e.TensorSubtract(leaky, RoIAlignPlan.FromDoubles<T>(shift, eh)));
        var rowSum = SparseMap(ex, new[] { rows, heads }, rowFrom, edgeTo, null);
        var alpha = e.TensorDivide(ex, SparseMap(rowSum, eh, edgeTo, rowFrom, null));
        var coef = e.TensorMultiply(alpha, RoIAlignPlan.FromDoubles<T>(keep, eh));

        // Messages: coef[b, k, h] * wh[b, col[k], h, :], summed into row (b, rowOf[k]).
        var msgTo = new int[be * heads * features];
        var msgFrom = new int[msgTo.Length];
        var outTo = new int[msgTo.Length];
        for (int t = 0; t < be * heads; t++)
            for (int f = 0; f < features; f++)
            {
                int m = t * features + f;
                msgTo[m] = m;
                msgFrom[m] = colFrom[t] * features + f;
                outTo[m] = rowFrom[t] * features + f;
            }
        var ehf = new[] { be, heads, features };
        var neighbours = SparseMap(wh, ehf, msgTo, msgFrom, null);
        var weighted = e.TensorMultiply(neighbours, e.TensorBroadcastTo(e.Reshape(coef, new[] { be, heads, 1 }), ehf));
        var output = SparseMap(weighted, hf, outTo, msgTo, null);
        return e.Reshape(output, (int[])transformed._shape.Clone());
    }
}
