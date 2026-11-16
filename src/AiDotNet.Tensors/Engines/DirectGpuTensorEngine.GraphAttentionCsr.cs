using System;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

// GraphAttentionCsr on the CUDA graph-attention kernels: the inputs stay resident and the CSR/CSC pattern is uploaded
// once per graph (cached on the CsrGraph). Other backends and element types run CpuEngine's implementation.
public partial class DirectGpuTensorEngine
{
    private sealed class GraphAttentionDeviceGraph
    {
        public IDirectGpuBackend Backend = null!;
        public IGpuBuffer RowPointers = null!, Columns = null!, ColumnPointers = null!, TransposeRows = null!, TransposeEdges = null!;
    }

    private static GraphAttentionDeviceGraph DeviceGraph(IDirectGpuBackend backend, CsrGraph graph)
    {
        if (graph.DeviceCache is GraphAttentionDeviceGraph cached && ReferenceEquals(cached.Backend, backend))
            return cached;
        lock (graph)
        {
            if (graph.DeviceCache is GraphAttentionDeviceGraph again && ReferenceEquals(again.Backend, backend))
                return again;
            var device = new GraphAttentionDeviceGraph
            {
                Backend = backend,
                RowPointers = backend.AllocateIntBuffer(graph.RowPointers),
                Columns = backend.AllocateIntBuffer(graph.EdgeCount > 0 ? graph.ColumnIndices : new int[1]),
                ColumnPointers = backend.AllocateIntBuffer(graph.ColumnPointers),
                TransposeRows = backend.AllocateIntBuffer(graph.EdgeCount > 0 ? graph.TransposeRowIndices : new int[1]),
                TransposeEdges = backend.AllocateIntBuffer(graph.EdgeCount > 0 ? graph.TransposeEdgeIndices : new int[1]),
            };
            graph.DeviceCache = device;
            return device;
        }
    }

    /// <inheritdoc/>
    // The public op owns validation, graph-mode recording and the tape; its forward and backward bodies are the
    // internal virtuals below, which run the CUDA kernels. Overridden here so the GPU coverage is visible on the
    // engine's interface map (BackendCompletenessTests) rather than hidden behind the internal hooks.
    public override Tensor<T> GraphAttentionCsr<T>(
        Tensor<T> transformed, Tensor<T> attention, CsrGraph graph,
        double negativeSlope = 0.2, double dropoutRate = 0.0, Random? dropoutRandom = null)
        => base.GraphAttentionCsr(transformed, attention, graph, negativeSlope, dropoutRate, dropoutRandom);

    internal override Tensor<T> GraphAttentionCsrForward<T>(
        Tensor<T> transformed, Tensor<T> attention, CsrGraph graph, double slope, double rate, uint seed)
    {
        if (typeof(T) != typeof(float))
            return base.GraphAttentionCsrForward(transformed, attention, graph, slope, rate, seed);
        using var __gpuOp = EnterGpuOp();
        if (!TryGetBackend(out var backend) || backend is not IGraphAttentionBackend cuda)
            return base.GraphAttentionCsrForward(transformed, attention, graph, slope, rate, seed);
        try
        {
            int heads = attention.Shape[0], features = attention.Shape[1] / 2;
            int nodes = graph.NodeCount;
            int rows = transformed.Length / (heads * features);
            int batch = rows / nodes;
            var dg = DeviceGraph(backend, graph);
            using var whBuf = GetOrAllocateBuffer(backend, transformed);
            using var aBuf = GetOrAllocateBuffer(backend, attention);
            using var selfBuf = AllocateTransientScratch(backend, rows * heads, fullyWritten: true);
            using var neighBuf = AllocateTransientScratch(backend, rows * heads, fullyWritten: true);
            float keep = rate > 0 ? (float)(1.0 / (1.0 - rate)) : 1f;
            return DispatchDeferredGpuOp<T>(backend, transformed.Length, (int[])transformed._shape.Clone(), output =>
            {
                cuda.GraphAttentionScores(whBuf.Buffer, aBuf.Buffer, selfBuf.Buffer, neighBuf.Buffer, rows, heads, features);
                cuda.GraphAttentionForward(whBuf.Buffer, selfBuf.Buffer, neighBuf.Buffer, dg.RowPointers, dg.Columns, output,
                    batch, nodes, heads, features, graph.EdgeCount, (float)slope, (float)rate, keep, seed);
            }, fullyWritten: true);
        }
        catch (NotSupportedException ex)
        {
            if (ThrowOnGpuKernelFallback) throw;
            GpuLaunchProbe.OnFallback("GraphAttentionCsr", ex);
            return base.GraphAttentionCsrForward(transformed, attention, graph, slope, rate, seed);
        }
    }

    internal override (Tensor<T> GradTransformed, Tensor<T> GradAttention) GraphAttentionCsrBackward<T>(
        Tensor<T> gradOutput, Tensor<T> transformed, Tensor<T> attention, CsrGraph graph, double slope, double rate, uint seed)
    {
        if (typeof(T) != typeof(float))
            return base.GraphAttentionCsrBackward(gradOutput, transformed, attention, graph, slope, rate, seed);
        using var __gpuOp = EnterGpuOp();
        if (!TryGetBackend(out var backend) || backend is not IGraphAttentionBackend cuda)
            return base.GraphAttentionCsrBackward(gradOutput, transformed, attention, graph, slope, rate, seed);
        try
        {
            int heads = attention.Shape[0], features = attention.Shape[1] / 2;
            int nodes = graph.NodeCount, edges = graph.EdgeCount;
            int rows = transformed.Length / (heads * features);
            int batch = rows / nodes;
            int edgeSlots = checked(batch * Math.Max(edges, 1) * heads);
            var dg = DeviceGraph(backend, graph);
            using var whBuf = GetOrAllocateBuffer(backend, transformed);
            using var aBuf = GetOrAllocateBuffer(backend, attention);
            using var goBuf = GetOrAllocateBuffer(backend, gradOutput);
            using var selfBuf = AllocateTransientScratch(backend, rows * heads, fullyWritten: true);
            using var neighBuf = AllocateTransientScratch(backend, rows * heads, fullyWritten: true);
            using var gSelfBuf = AllocateTransientScratch(backend, rows * heads, fullyWritten: true);
            using var gNeighBuf = AllocateTransientScratch(backend, rows * heads, fullyWritten: true);
            using var gzBuf = AllocateTransientScratch(backend, edgeSlots, fullyWritten: true);
            using var cpBuf = AllocateTransientScratch(backend, edgeSlots, fullyWritten: true);
            float keep = rate > 0 ? (float)(1.0 / (1.0 - rate)) : 1f;
            var gWh = DispatchDeferredGpuOp<T>(backend, transformed.Length, (int[])transformed._shape.Clone(), output =>
            {
                cuda.GraphAttentionScores(whBuf.Buffer, aBuf.Buffer, selfBuf.Buffer, neighBuf.Buffer, rows, heads, features);
                cuda.GraphAttentionBackwardRows(whBuf.Buffer, goBuf.Buffer, selfBuf.Buffer, neighBuf.Buffer,
                    dg.RowPointers, dg.Columns, gzBuf.Buffer, cpBuf.Buffer, gSelfBuf.Buffer,
                    batch, nodes, heads, features, edges, (float)slope, (float)rate, keep, seed);
                cuda.GraphAttentionBackwardCols(goBuf.Buffer, aBuf.Buffer, gzBuf.Buffer, cpBuf.Buffer, gSelfBuf.Buffer,
                    dg.ColumnPointers, dg.TransposeRows, dg.TransposeEdges, gNeighBuf.Buffer, output,
                    batch, nodes, heads, features, edges);
            }, fullyWritten: true);
            var gAttn = DispatchDeferredGpuOp<T>(backend, attention.Length, (int[])attention._shape.Clone(), output =>
                cuda.GraphAttentionAttentionGrad(whBuf.Buffer, gSelfBuf.Buffer, gNeighBuf.Buffer, output, rows, heads, features),
                fullyWritten: true);
            return (gWh, gAttn);
        }
        catch (NotSupportedException ex)
        {
            if (ThrowOnGpuKernelFallback) throw;
            GpuLaunchProbe.OnFallback("GraphAttentionCsrBackward", ex);
            return base.GraphAttentionCsrBackward(gradOutput, transformed, attention, graph, slope, rate, seed);
        }
    }
}
