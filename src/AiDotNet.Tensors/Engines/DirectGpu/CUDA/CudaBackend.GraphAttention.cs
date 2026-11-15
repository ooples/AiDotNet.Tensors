using System;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

public sealed partial class CudaBackend : IGraphAttentionBackend
{
    private IntPtr _graphAttentionModule;
    private string? _graphAttentionError;
    private readonly object _graphAttentionLock = new object();

    // Compiled on first use: only graph-attention layers dispatch here, so other models pay nothing for it.
    private IntPtr GraphAttentionKernel(string name)
    {
        lock (_graphAttentionLock)
        {
            if (_graphAttentionModule == IntPtr.Zero && _graphAttentionError is null)
            {
                try
                {
                    using var _ = PushContext();
                    _graphAttentionModule = CompileKernelModule(_cuDevice, CudaGraphAttentionKernels.GetSource(),
                        "graph_attention_csr", CudaGraphAttentionKernels.GetKernelNames(), useFastMath: false);
                }
                catch (Exception ex) when (ex is not OutOfMemoryException)
                {
                    _graphAttentionError = ex.Message;
                }
            }
            if (_graphAttentionModule == IntPtr.Zero || !_kernelCache.TryGetValue(name, out var kernel))
                throw new NotSupportedException("CUDA graph-attention kernels are unavailable"
                    + (_graphAttentionError is null ? "." : ": " + _graphAttentionError));
            return kernel;
        }
    }

    private static uint GraphAttentionGrid(int total) => (uint)((total + DefaultBlockSize - 1) / DefaultBlockSize);

    /// <summary>Self and neighbour attention scores per (row, head).</summary>
    unsafe void IGraphAttentionBackend.GraphAttentionScores(IGpuBuffer wh, IGpuBuffer attention, IGpuBuffer selfScore,
        IGpuBuffer neighScore, int rows, int heads, int features)
    {
        int total = rows * heads;
        if (total <= 0) return;
        var kernel = GraphAttentionKernel("gat_csr_scores");
        using var _ = PushContext();
        IntPtr pw = wh.Handle, pa = attention.Handle, ps = selfScore.Handle, pn = neighScore.Handle;
        void** args = stackalloc void*[7];
        args[0] = &pw; args[1] = &pa; args[2] = &ps; args[3] = &pn;
        args[4] = &rows; args[5] = &heads; args[6] = &features;
        LaunchKernel(kernel, GraphAttentionGrid(total), DefaultBlockSize, args);
    }

    unsafe void IGraphAttentionBackend.GraphAttentionForward(IGpuBuffer wh, IGpuBuffer selfScore, IGpuBuffer neighScore,
        IGpuBuffer rowPtr, IGpuBuffer col, IGpuBuffer output,
        int batch, int nodes, int heads, int features, int edges, float slope, float rate, float keep, uint seed)
    {
        int total = batch * nodes * heads;
        if (total <= 0) return;
        var kernel = GraphAttentionKernel("gat_csr_forward");
        using var _ = PushContext();
        IntPtr pw = wh.Handle, ps = selfScore.Handle, pn = neighScore.Handle, pr = rowPtr.Handle, pc = col.Handle, po = output.Handle;
        void** args = stackalloc void*[15];
        args[0] = &pw; args[1] = &ps; args[2] = &pn; args[3] = &pr; args[4] = &pc; args[5] = &po;
        args[6] = &batch; args[7] = &nodes; args[8] = &heads; args[9] = &features; args[10] = &edges;
        args[11] = &slope; args[12] = &rate; args[13] = &keep; args[14] = &seed;
        LaunchKernel(kernel, GraphAttentionGrid(total), DefaultBlockSize, args);
    }

    unsafe void IGraphAttentionBackend.GraphAttentionBackwardRows(IGpuBuffer wh, IGpuBuffer gradOutput, IGpuBuffer selfScore,
        IGpuBuffer neighScore, IGpuBuffer rowPtr, IGpuBuffer col, IGpuBuffer gz, IGpuBuffer cp, IGpuBuffer gSelf,
        int batch, int nodes, int heads, int features, int edges, float slope, float rate, float keep, uint seed)
    {
        int total = batch * nodes * heads;
        if (total <= 0) return;
        var kernel = GraphAttentionKernel("gat_csr_backward_rows");
        using var _ = PushContext();
        IntPtr pw = wh.Handle, pg = gradOutput.Handle, ps = selfScore.Handle, pn = neighScore.Handle;
        IntPtr pr = rowPtr.Handle, pc = col.Handle, pz = gz.Handle, pp = cp.Handle, pgs = gSelf.Handle;
        void** args = stackalloc void*[18];
        args[0] = &pw; args[1] = &pg; args[2] = &ps; args[3] = &pn; args[4] = &pr; args[5] = &pc;
        args[6] = &pz; args[7] = &pp; args[8] = &pgs;
        args[9] = &batch; args[10] = &nodes; args[11] = &heads; args[12] = &features; args[13] = &edges;
        args[14] = &slope; args[15] = &rate; args[16] = &keep; args[17] = &seed;
        LaunchKernel(kernel, GraphAttentionGrid(total), DefaultBlockSize, args);
    }

    unsafe void IGraphAttentionBackend.GraphAttentionBackwardCols(IGpuBuffer gradOutput, IGpuBuffer attention, IGpuBuffer gz,
        IGpuBuffer cp, IGpuBuffer gSelf, IGpuBuffer colPtr, IGpuBuffer tRow, IGpuBuffer tEdge,
        IGpuBuffer gNeigh, IGpuBuffer gradWh, int batch, int nodes, int heads, int features, int edges)
    {
        int total = batch * nodes * heads;
        if (total <= 0) return;
        var kernel = GraphAttentionKernel("gat_csr_backward_cols");
        using var _ = PushContext();
        IntPtr pg = gradOutput.Handle, pa = attention.Handle, pz = gz.Handle, pp = cp.Handle, ps = gSelf.Handle;
        IntPtr pcp = colPtr.Handle, ptr = tRow.Handle, pte = tEdge.Handle, pn = gNeigh.Handle, pw = gradWh.Handle;
        void** args = stackalloc void*[15];
        args[0] = &pg; args[1] = &pa; args[2] = &pz; args[3] = &pp; args[4] = &ps;
        args[5] = &pcp; args[6] = &ptr; args[7] = &pte; args[8] = &pn; args[9] = &pw;
        args[10] = &batch; args[11] = &nodes; args[12] = &heads; args[13] = &features; args[14] = &edges;
        LaunchKernel(kernel, GraphAttentionGrid(total), DefaultBlockSize, args);
    }

    unsafe void IGraphAttentionBackend.GraphAttentionAttentionGrad(IGpuBuffer wh, IGpuBuffer gSelf, IGpuBuffer gNeigh,
        IGpuBuffer gradAttention, int rows, int heads, int features)
    {
        int entries = heads * 2 * features;
        if (entries <= 0) return;
        var kernel = GraphAttentionKernel("gat_csr_attention_grad");
        using var _ = PushContext();
        IntPtr pw = wh.Handle, ps = gSelf.Handle, pn = gNeigh.Handle, pa = gradAttention.Handle;
        void** args = stackalloc void*[7];
        args[0] = &pw; args[1] = &ps; args[2] = &pn; args[3] = &pa;
        args[4] = &rows; args[5] = &heads; args[6] = &features;
        LaunchKernel(kernel, (uint)entries, 256, args);
    }
}
