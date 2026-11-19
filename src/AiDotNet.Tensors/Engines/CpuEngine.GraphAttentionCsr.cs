using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

// Graph attention (Velickovic et al. 2018) over a sparse neighbourhood: the dense masked formulation's exact result
// computed over the graph's edges only. A row's softmax is over its stored neighbours; entries a dense -1e9 additive
// mask would hide contribute exp(-1e9 - max) = 0 in floating point there, so both agree whenever every row has an
// edge (CsrGraph.EveryRowHasEdge).
public partial class CpuEngine
{
    /// <summary>The seed of one forward's dropout draw, read by its backward (and rewritten by each compiled replay).</summary>
    internal sealed class GraphAttentionDropoutState
    {
        public uint Seed;
        public readonly Random? Random;
        public GraphAttentionDropoutState(Random? random) { Random = random; }
        public uint Draw()
        {
            if (Random is null) return 0u;
            lock (Random) Seed = unchecked((uint)Random.Next() ^ ((uint)Random.Next() << 16));
            return Seed;
        }
    }

    /// <inheritdoc/>
    public virtual Tensor<T> GraphAttentionCsr<T>(
        Tensor<T> transformed, Tensor<T> attention, CsrGraph graph,
        double negativeSlope = 0.2, double dropoutRate = 0.0, Random? dropoutRandom = null)
    {
        if (transformed is null) throw new ArgumentNullException(nameof(transformed));
        if (attention is null) throw new ArgumentNullException(nameof(attention));
        if (graph is null) throw new ArgumentNullException(nameof(graph));
        if (attention.Rank != 2 || attention.Shape[1] % 2 != 0 || attention.Shape[1] == 0)
            throw new ArgumentException($"attention must be [heads, 2 * features]; got [{string.Join(", ", attention._shape)}].", nameof(attention));
        if (double.IsNaN(dropoutRate) || dropoutRate < 0.0 || dropoutRate >= 1.0)
            throw new ArgumentOutOfRangeException(nameof(dropoutRate), "dropoutRate must be in [0, 1).");
        int heads = attention.Shape[0], features = attention.Shape[1] / 2;
        long rowWidth = (long)heads * features;
        if (transformed.Length % (rowWidth * graph.NodeCount) != 0)
            throw new ArgumentException(
                $"transformed has {transformed.Length} elements, not a whole number of [{graph.NodeCount} nodes, {heads} heads, {features} features] graphs.",
                nameof(transformed));

        bool dropout = dropoutRate > 0.0 && dropoutRandom is not null;
        var state = new GraphAttentionDropoutState(dropout ? dropoutRandom : null);
        var saved = new object[] { graph, negativeSlope, dropout ? dropoutRate : 0.0, state };

        var scope = GraphMode.Current;
        if (scope != null)
        {
            scope.BindEngineIfUnset(this);
            var capturedWh = transformed;
            var capturedAttn = attention;
            double capturedSlope = negativeSlope, capturedRate = dropout ? dropoutRate : 0.0;
            return scope.RecordBinary(LazyNodeType.Custom, "GraphAttentionCsr", transformed, attention, transformed._shape,
                (eng, output) =>
                {
                    uint seed = state.Draw();
                    var cpu = eng.As<CpuEngine>() ?? this;
                    var r = cpu.GraphAttentionCsrForward(capturedWh, capturedAttn, graph, capturedSlope, capturedRate, seed);
                    DirectGpuTensorEngine.CopyResultInto(eng, r, output);
                },
                GraphAttentionCsrBackwardFn<T>, saved);
        }

        uint s = state.Draw();
        var result = GraphAttentionCsrForward(transformed, attention, graph, negativeSlope, dropout ? dropoutRate : 0.0, s);
        DifferentiableOps.RecordBinary("GraphAttentionCsr", result, transformed, attention, GraphAttentionCsrBackwardFn<T>, saved);
        return result;
    }

    private static void GraphAttentionCsrBackwardFn<T>(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var graph = (CsrGraph)savedState[0];
        double slope = (double)savedState[1];
        double rate = (double)savedState[2];
        var state = (GraphAttentionDropoutState)savedState[3];
        if (BackwardFunctions<T>.CreatesGraph
            && GraphAttentionCsrCreateGraph(gradOutput, inputs, engine, grads, graph, slope, rate, state.Seed))
            return;
        var cpu = engine.As<CpuEngine>() ?? new CpuEngine();
        var (gWh, gAttn) = cpu.GraphAttentionCsrBackward(gradOutput, inputs[0], inputs[1], graph, slope, rate, state.Seed);
        DifferentiableOps.AccumulateGrad(grads, inputs[0], gWh, engine);
        DifferentiableOps.AccumulateGrad(grads, inputs[1], gAttn, engine);
    }

    // The fused kernels are first order: a create-graph backward differentiates the recorded composite instead.
    private static bool GraphAttentionCsrCreateGraph<T>(Tensor<T> gradOutput, Tensor<T>[] inputs, IEngine engine,
        Dictionary<Tensor<T>, Tensor<T>> grads, CsrGraph graph, double slope, double rate, uint seed)
        => BackwardFunctions<T>.TryCompositeBackward(gradOutput, inputs, engine, grads,
            e => BackwardFunctions<T>.GraphAttentionCsrRecorded(e, inputs[0], inputs[1], graph, slope, rate, seed));

    /// <summary>The device-independent dropout draw both the CPU and the GPU kernels use: keep when u &gt;= rate.</summary>
    internal static float GraphAttentionUniform(uint seed, uint index)
    {
        unchecked
        {
            uint x = index * 0x9E3779B1u + seed;
            x ^= x >> 16; x *= 0x7FEB352Du;
            x ^= x >> 15; x *= 0x846CA68Bu;
            x ^= x >> 16;
            return (x >> 8) * (1.0f / 16777216.0f);
        }
    }

    /// <summary>The forward without tape recording; overridden by the GPU engine.</summary>
    internal virtual Tensor<T> GraphAttentionCsrForward<T>(
        Tensor<T> transformed, Tensor<T> attention, CsrGraph graph, double slope, double rate, uint seed)
    {
        int heads = attention.Shape[0], features = attention.Shape[1] / 2;
        int n = graph.NodeCount, rowWidth = heads * features;
        int batch = transformed.Length / (rowWidth * n);
        var wh = GatToDoubles(transformed);
        var a = GatToDoubles(attention);
        var (sSelf, sNeigh) = GraphAttentionScores(wh, a, batch * n, heads, features);
        var outD = new double[wh.Length];
        int edges = graph.EdgeCount;
        var rowPtr = graph.RowPointers; var col = graph.ColumnIndices;
        double keep = rate > 0 ? 1.0 / (1.0 - rate) : 1.0;
        int maxDegree = MaxDegree(graph);

        Parallel.For(0, batch * n, new ParallelOptions { MaxDegreeOfParallelism = CpuParallelSettings.MaxDegreeOfParallelism }, () => new double[maxDegree],
            (bi, _, scratch) =>
            {
                int b = bi / n, i = bi % n;
                int r0 = rowPtr[i], r1 = rowPtr[i + 1];
                for (int h = 0; h < heads; h++)
                {
                    double self = sSelf[bi * heads + h];
                    double max = double.NegativeInfinity;
                    for (int e = r0; e < r1; e++)
                    {
                        double z = self + sNeigh[(b * n + col[e]) * heads + h];
                        double ev = z > 0 ? z : slope * z;
                        scratch[e - r0] = ev;
                        if (ev > max) max = ev;
                    }
                    double sum = 0;
                    for (int e = r0; e < r1; e++) { double x = Math.Exp(scratch[e - r0] - max); scratch[e - r0] = x; sum += x; }
                    int o = (bi * heads + h) * features;
                    for (int e = r0; e < r1; e++)
                    {
                        double c = scratch[e - r0] / sum;
                        if (rate > 0)
                            c = GraphAttentionUniform(seed, (uint)(((long)b * edges + e) * heads + h)) < rate ? 0.0 : c * keep;
                        if (c == 0.0) continue;
                        int w = ((b * n + col[e]) * heads + h) * features;
                        for (int f = 0; f < features; f++) outD[o + f] += c * wh[w + f];
                    }
                }
                return scratch;
            }, _ => { });

        return GatFromDoubles<T>(outD, transformed._shape);
    }

    /// <summary>The backward's (d transformed, d attention) without tape recording; overridden by the GPU engine.</summary>
    internal virtual (Tensor<T> GradTransformed, Tensor<T> GradAttention) GraphAttentionCsrBackward<T>(
        Tensor<T> gradOutput, Tensor<T> transformed, Tensor<T> attention, CsrGraph graph, double slope, double rate, uint seed)
    {
        int heads = attention.Shape[0], features = attention.Shape[1] / 2;
        int n = graph.NodeCount, rowWidth = heads * features;
        int batch = transformed.Length / (rowWidth * n);
        var wh = GatToDoubles(transformed);
        var a = GatToDoubles(attention);
        var go = GatToDoubles(gradOutput);
        var (sSelf, sNeigh) = GraphAttentionScores(wh, a, batch * n, heads, features);
        int edges = graph.EdgeCount;
        var rowPtr = graph.RowPointers; var col = graph.ColumnIndices;
        double keep = rate > 0 ? 1.0 / (1.0 - rate) : 1.0;
        int maxDegree = MaxDegree(graph);
        var gz = new double[(long)batch * edges * heads];
        var cp = new double[gz.Length];
        var gSelf = new double[(long)batch * n * heads];
        var gNeigh = new double[gSelf.Length];
        var options = new ParallelOptions { MaxDegreeOfParallelism = CpuParallelSettings.MaxDegreeOfParallelism };

        // Rows: recompute the coefficients, then the softmax and LeakyReLU backward per edge.
        Parallel.For(0, batch * n, options, () => new double[maxDegree * 2],
            (bi, _, scratch) =>
            {
                int b = bi / n, i = bi % n;
                int r0 = rowPtr[i], r1 = rowPtr[i + 1], d = r1 - r0;
                for (int h = 0; h < heads; h++)
                {
                    double self = sSelf[bi * heads + h];
                    double max = double.NegativeInfinity;
                    for (int e = r0; e < r1; e++)
                    {
                        double z = self + sNeigh[(b * n + col[e]) * heads + h];
                        double ev = z > 0 ? z : slope * z;
                        scratch[e - r0] = ev;
                        if (ev > max) max = ev;
                    }
                    double sum = 0;
                    for (int e = r0; e < r1; e++) { double x = Math.Exp(scratch[e - r0] - max); scratch[e - r0] = x; sum += x; }
                    int o = (bi * heads + h) * features;
                    double dot = 0;
                    for (int e = r0; e < r1; e++)
                    {
                        double c = scratch[e - r0] / sum;
                        scratch[e - r0] = c;
                        double m = rate > 0
                            ? (GraphAttentionUniform(seed, (uint)(((long)b * edges + e) * heads + h)) < rate ? 0.0 : keep)
                            : 1.0;
                        long slot = ((long)b * edges + e) * heads + h;
                        cp[slot] = c * m;
                        double g = 0;
                        if (m != 0.0)
                        {
                            int w = ((b * n + col[e]) * heads + h) * features;
                            for (int f = 0; f < features; f++) g += go[o + f] * wh[w + f];
                            g *= m;
                        }
                        scratch[d + e - r0] = g;
                        dot += c * g;
                    }
                    double gs = 0;
                    for (int e = r0; e < r1; e++)
                    {
                        double z = self + sNeigh[(b * n + col[e]) * heads + h];
                        double ge = scratch[e - r0] * (scratch[d + e - r0] - dot) * (z > 0 ? 1.0 : slope);
                        gz[((long)b * edges + e) * heads + h] = ge;
                        gs += ge;
                    }
                    gSelf[bi * heads + h] = gs;
                }
                return scratch;
            }, _ => { });

        // Columns: gather the aggregation's transpose and the neighbour-score gradient into each node.
        var colPtr = graph.ColumnPointers; var tRow = graph.TransposeRowIndices; var tEdge = graph.TransposeEdgeIndices;
        int aw = 2 * features;
        var gWh = new double[wh.Length];
        Parallel.For(0, batch * n, options, bj =>
        {
            int b = bj / n, j = bj % n;
            for (int h = 0; h < heads; h++)
            {
                int o = (bj * heads + h) * features;
                double gn = 0;
                for (int k = colPtr[j]; k < colPtr[j + 1]; k++)
                {
                    long slot = ((long)b * edges + tEdge[k]) * heads + h;
                    gn += gz[slot];
                    double c = cp[slot];
                    if (c == 0.0) continue;
                    int gi = ((b * n + tRow[k]) * heads + h) * features;
                    for (int f = 0; f < features; f++) gWh[o + f] += c * go[gi + f];
                }
                gNeigh[bj * heads + h] = gn;
                double gs = gSelf[bj * heads + h];
                for (int f = 0; f < features; f++)
                    gWh[o + f] += gs * a[h * aw + f] + gn * a[h * aw + features + f];
            }
        });

        var gA = new double[a.Length];
        for (int h = 0; h < heads; h++)
        {
            for (int bn = 0; bn < batch * n; bn++)
            {
                double gs = gSelf[bn * heads + h], gn = gNeigh[bn * heads + h];
                if (gs == 0.0 && gn == 0.0) continue;
                int w = (bn * heads + h) * features;
                for (int f = 0; f < features; f++)
                {
                    gA[h * aw + f] += gs * wh[w + f];
                    gA[h * aw + features + f] += gn * wh[w + f];
                }
            }
        }

        return (GatFromDoubles<T>(gWh, transformed._shape), GatFromDoubles<T>(gA, attention._shape));
    }

    private static (double[] Self, double[] Neigh) GraphAttentionScores(double[] wh, double[] a, int rows, int heads, int features)
    {
        var self = new double[(long)rows * heads];
        var neigh = new double[self.Length];
        int aw = 2 * features;
        for (int r = 0; r < rows; r++)
        {
            for (int h = 0; h < heads; h++)
            {
                int w = (r * heads + h) * features;
                double s = 0, t = 0;
                for (int f = 0; f < features; f++)
                {
                    s += wh[w + f] * a[h * aw + f];
                    t += wh[w + f] * a[h * aw + features + f];
                }
                self[r * heads + h] = s;
                neigh[r * heads + h] = t;
            }
        }
        return (self, neigh);
    }

    private static int MaxDegree(CsrGraph graph)
    {
        int max = 1;
        var p = graph.RowPointers;
        for (int i = 0; i < graph.NodeCount; i++) max = Math.Max(max, p[i + 1] - p[i]);
        return max;
    }

    [ScalarKernel("Conversion fallback for element types other than float and double, which take the array paths.")]
    private static double[] GatToDoubles<T>(Tensor<T> tensor)
    {
        var data = tensor.ToArray();
        var result = new double[data.Length];
        if (data is float[] f) { for (int i = 0; i < f.Length; i++) result[i] = f[i]; }
        else if (data is double[] d) { Array.Copy(d, result, d.Length); }
        else
        {
            var ops = MathHelper.GetNumericOperations<T>();
            for (int i = 0; i < data.Length; i++) result[i] = ops.ToDouble(data[i]);
        }
        return result;
    }

    [ScalarKernel("Conversion fallback for element types other than float and double, which take the array paths.")]
    private static Tensor<T> GatFromDoubles<T>(double[] values, int[] shape)
    {
        var data = new T[values.Length];
        if (data is float[] f) { for (int i = 0; i < f.Length; i++) f[i] = (float)values[i]; }
        else if (data is double[] d) { Array.Copy(values, d, values.Length); }
        else
        {
            var ops = MathHelper.GetNumericOperations<T>();
            for (int i = 0; i < data.Length; i++) data[i] = ops.FromDouble(values[i]);
        }
        return new Tensor<T>(data, (int[])shape.Clone());
    }
}
