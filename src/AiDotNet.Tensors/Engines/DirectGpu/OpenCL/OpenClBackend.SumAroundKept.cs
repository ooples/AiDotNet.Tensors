// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL
{
    // ISumAroundKeptBackend on OpenCL: the same two deterministic passes as CUDA's sum_around_kept_partial/_finish. Without
    // it the compiled GPU step reduced a convolution's bias gradient (dY [N, C, H, W] over N, H and W) by permuting the
    // whole gradient so C came first and row-summing it: an activation-sized transpose and buffer per convolution.
    public sealed partial class OpenClBackend : ISumAroundKeptBackend
    {
        private const string SumAroundKeptSource = @"
// Pass 1: work-group (c, chunk) sums channel c over rows [chunk * rowsPerChunk, +rowsPerChunk) of x viewed as [L, C, T],
// each row a run of T contiguous floats, by a fixed local-memory tree, into partials[chunk * C + c].
__kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void sum_around_kept_partial(__global const float* x, __global float* partials, int L, int C, int T, int rowsPerChunk)
{
    __local float s[256];
    int lid = get_local_id(0);
    int c = get_group_id(0), chunk = get_group_id(1);
    int l0 = chunk * rowsPerChunk, l1 = min(L, l0 + rowsPerChunk);
    long span = (long)max(0, l1 - l0) * T;
    float acc = 0.0f;
    for (long e = lid; e < span; e += 256)
    {
        int l = l0 + (int)(e / T), t = (int)(e % T);
        acc += x[((long)l * C + c) * T + t];
    }
    s[lid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int o = 128; o > 0; o >>= 1)
    {
        if (lid < o) s[lid] += s[lid + o];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) partials[(long)chunk * C + c] = s[0];
}

// Pass 2: each channel's partials summed in chunk order, times scale.
__kernel void sum_around_kept_finish(__global const float* partials, __global float* output, int C, int chunks, float scale)
{
    int c = get_global_id(0);
    if (c >= C) return;
    float t = 0.0f;
    for (int k = 0; k < chunks; k++) t += partials[(long)k * C + c];
    output[c] = scale * t;
}
";

        // 0 not tried, 1 compiled, 2 unavailable.
        private int _sumAroundKeptState;
        private readonly object _sumAroundKeptLock = new object();

        private bool SumAroundKeptReady()
        {
            int state = System.Threading.Volatile.Read(ref _sumAroundKeptState);
            if (state != 0) return state == 1;
            lock (_sumAroundKeptLock)
            {
                if (_sumAroundKeptState != 0) return _sumAroundKeptState == 1;
                var context = _context;
                bool ok = false;
                if (context is not null && _maxWorkGroupSize >= 256)
                {
                    try
                    {
                        var program = CompileOrLoadCached(SumAroundKeptSource, OpenClBuildOptions.SafeMathFlags,
                            "sum around a kept axis");
                        _programs.Add(program);
                        _kernelCache["sum_around_kept_partial"] = new DirectOpenClKernel(context, program, "sum_around_kept_partial");
                        _kernelCache["sum_around_kept_finish"] = new DirectOpenClKernel(context, program, "sum_around_kept_finish");
                        ok = true;
                    }
                    catch (Exception ex) when (ex is not OutOfMemoryException)
                    {
                        System.Diagnostics.Trace.TraceWarning($"OpenCL sum-around-kept kernels unavailable: {ex.Message}");
                    }
                }
                System.Threading.Volatile.Write(ref _sumAroundKeptState, ok ? 1 : 2);
                return ok;
            }
        }

        // Chunks of the leading block per kept channel: ~256 work-groups in all, at least one row each (as CUDA).
        private static int SumAroundKeptChunks(int leading, int kept) =>
            Math.Max(1, Math.Min(leading, (256 + kept - 1) / Math.Max(1, kept)));

        int ISumAroundKeptBackend.SumAroundKeptScratchFloats(int leading, int kept, int trailing) =>
            leading <= 0 || kept <= 0 || trailing <= 0 ? 0 : checked(SumAroundKeptChunks(leading, kept) * kept);

        bool ISumAroundKeptBackend.TrySumAroundKept(IGpuBuffer input, IGpuBuffer scratch, IGpuBuffer output,
            int leading, int kept, int trailing, float scale)
        {
            long n = (long)leading * kept * trailing;
            if (leading <= 0 || kept <= 0 || trailing <= 0 || input.Size < n || output.Size < kept
                || BufferHandle(output) == BufferHandle(input) || !SumAroundKeptReady())
                return false;
            int chunks = SumAroundKeptChunks(leading, kept);
            if (scratch.Size < (long)chunks * kept) return false;
            int rowsPerChunk = (leading + chunks - 1) / chunks;

            var partial = _kernelCache["sum_around_kept_partial"];
            partial.SetArg(0, BufferHandle(input));
            partial.SetArg(1, BufferHandle(scratch));
            partial.SetArg(2, leading);
            partial.SetArg(3, kept);
            partial.SetArg(4, trailing);
            partial.SetArg(5, rowsPerChunk);
            partial.Execute2D(kept * 256, chunks, 256, 1);

            var finish = _kernelCache["sum_around_kept_finish"];
            finish.SetArg(0, BufferHandle(scratch));
            finish.SetArg(1, BufferHandle(output));
            finish.SetArg(2, kept);
            finish.SetArg(3, chunks);
            finish.SetArg(4, scale);
            int local = ElementwiseGroupSize(kept);
            finish.Execute1D((kept + local - 1) / local * local, local);
            return true;
        }
    }
}