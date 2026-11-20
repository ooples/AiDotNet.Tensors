using System;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

// GraphAttentionCsr keeps one device pattern per graph and engine. An engine's captured programs can continue to
// reference it while another engine uses the same graph.
public partial class DirectGpuTensorEngine
{
    private readonly System.Runtime.CompilerServices.ConditionalWeakTable<CsrGraph, GraphAttentionDeviceGraph> _graphAttentionGraphs = new();
    private readonly System.Collections.Concurrent.ConcurrentBag<WeakReference<GraphAttentionDeviceGraph>> _graphAttentionGraphOwners = new();

    private sealed class GraphAttentionDeviceGraph : IDisposable
    {
        private readonly IDirectGpuBackend _backend;
        private readonly IGpuBuffer[] _buffers = new IGpuBuffer[5];
        public IGpuBuffer RowPointers => _buffers[0];
        public IGpuBuffer Columns => _buffers[1];
        public IGpuBuffer ColumnPointers => _buffers[2];
        public IGpuBuffer TransposeRows => _buffers[3];
        public IGpuBuffer TransposeEdges => _buffers[4];

        internal GraphAttentionDeviceGraph(IDirectGpuBackend backend, CsrGraph graph)
        {
            _backend = backend;
            try
            {
                _buffers[0] = backend.AllocateIntBuffer(graph.RowOffsets);
                _buffers[1] = backend.AllocateIntBuffer(graph.EdgeCount > 0 ? graph.Columns : new int[1]);
                _buffers[2] = backend.AllocateIntBuffer(graph.ColumnPointers);
                _buffers[3] = backend.AllocateIntBuffer(graph.EdgeCount > 0 ? graph.TransposeRowIndices : new int[1]);
                _buffers[4] = backend.AllocateIntBuffer(graph.EdgeCount > 0 ? graph.TransposeEdgeIndices : new int[1]);
            }
            catch
            {
                Dispose();
                throw;
            }
        }

        public void Dispose()
        {
            foreach (var buffer in _buffers)
                if (buffer is not null) DisposeBufferAfterQueuedUse(_backend, buffer);
        }
    }

    private GraphAttentionDeviceGraph DeviceGraph(IDirectGpuBackend backend, CsrGraph graph)
    {
        lock (_graphAttentionGraphs)
        {
            if (_graphAttentionGraphs.TryGetValue(graph, out var cached)) return cached;
            var device = new GraphAttentionDeviceGraph(backend, graph);
            _graphAttentionGraphs.Add(graph, device);
            _graphAttentionGraphOwners.Add(new WeakReference<GraphAttentionDeviceGraph>(device));
            return device;
        }
    }

    private void ReleaseGraphAttentionGraphs()
    {
        while (_graphAttentionGraphOwners.TryTake(out var weak))
            if (weak.TryGetTarget(out var graph)) graph.Dispose();
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
        using var __gpuOp = EnterGpuOp();
        if (!TryGetBackend(out var backend))
            return base.GraphAttentionCsrForward(transformed, attention, graph, slope, rate, seed);
        if (typeof(T) != typeof(float))
            throw new NotSupportedException($"{backend.BackendName} supports GraphAttentionCsr only for Single. Use CpuEngine for {typeof(T).Name}.");
        if (rate > 0 && backend is IGpuGraphCapture capture && capture.IsStreamCapturing())
            throw new NotSupportedException("GraphAttentionCsr dropout requires a fresh seed on each execution.");
        if (backend is not IGraphAttentionBackend cuda)
            throw new NotSupportedException($"{backend.BackendName} does not support GraphAttentionCsr. Use CpuEngine for sparse graph attention.");
        try
        {
            int heads = attention.Shape[0], features = attention.Shape[1] / 2;
            int nodes = graph.NodeCount;
            int rows = transformed.Length / (heads * features);
            int batch = rows / nodes;
            var dg = DeviceGraph(backend, graph);
            using var whBuf = GetOrAllocateBuffer(backend, transformed);
            using var aBuf = GetOrAllocateBuffer(backend, attention);
            using var selfBuf = AllocateTransientScratch(backend, checked(rows * heads * 2), fullyWritten: true);
            using var neighBuf = AllocateTransientScratch(backend, checked(rows * heads * 2), fullyWritten: true);
            float keep = rate > 0 ? (float)(1.0 / (1.0 - rate)) : 1f;
            return DispatchDeferredGpuOp<T>(backend, transformed.Length, (int[])transformed._shape.Clone(), output =>
            {
                cuda.GraphAttentionScores(whBuf.Buffer, aBuf.Buffer, selfBuf.Buffer, neighBuf.Buffer, rows, heads, features);
                cuda.GraphAttentionForward(whBuf.Buffer, selfBuf.Buffer, neighBuf.Buffer, dg.RowPointers, dg.Columns, output,
                    batch, nodes, heads, features, graph.EdgeCount, slope, (uint)Math.Ceiling(rate * 16777216.0), keep, seed);
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
        using var __gpuOp = EnterGpuOp();
        if (!TryGetBackend(out var backend))
            return base.GraphAttentionCsrBackward(gradOutput, transformed, attention, graph, slope, rate, seed);
        if (typeof(T) != typeof(float))
            throw new NotSupportedException($"{backend.BackendName} supports GraphAttentionCsr only for Single. Use CpuEngine for {typeof(T).Name}.");
        if (rate > 0 && backend is IGpuGraphCapture capture && capture.IsStreamCapturing())
            throw new NotSupportedException("GraphAttentionCsr dropout requires a fresh seed on each execution.");
        if (backend is not IGraphAttentionBackend cuda)
            throw new NotSupportedException($"{backend.BackendName} does not support GraphAttentionCsr backward. Use CpuEngine for sparse graph attention.");
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
            using var selfBuf = AllocateTransientScratch(backend, checked(rows * heads * 2), fullyWritten: true);
            using var neighBuf = AllocateTransientScratch(backend, checked(rows * heads * 2), fullyWritten: true);
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
                    batch, nodes, heads, features, edges, slope, (uint)Math.Ceiling(rate * 16777216.0), keep, seed);
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
