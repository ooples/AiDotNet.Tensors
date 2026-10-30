// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL
{
    // IShortAttentionForwardBackend on OpenCL: one work item per query row, its 64-wide accumulator in registers, keys
    // and values streamed through local memory 32 at a time with an online softmax (the running maximum and sum are
    // rescaled as each block arrives, so no score row is stored), reading Q, K, V and writing O in the interleaved
    // [batch, seq, heads * headDim] layout.
    public sealed partial class OpenClBackend : IShortAttentionForwardBackend
    {
        private const int ShortAttentionMaxHeadDim = 64;

        private const string ShortAttentionSource = @"
#define SA_ROWS 64
#define SA_KEYS 32
#define SA_DMAX 64
#define SA_D4 (SA_DMAX / 4)
__kernel __attribute__((reqd_work_group_size(SA_ROWS, 1, 1)))
void short_attention_forward(__global const float* q, __global const float* k, __global const float* v, __global float* o,
    int batch, int heads, int seqQ, int seqK, int headDim, float scale, int causal, int causalOffset)
{
    // float4 tiles: each key costs a work item 2 * SA_D4 local reads instead of 2 * SA_DMAX.
    __local float4 ks[SA_KEYS][SA_D4];
    __local float4 vs[SA_KEYS][SA_D4];
    const int lid = get_local_id(0);
    const int bh = get_group_id(1);
    const int b = bh / heads, h = bh - b * heads;
    const int row = get_group_id(0) * SA_ROWS + lid;
    const int width = heads * headDim;
    const bool active = row < seqQ;
    // Fixed-width, unrolled loops keep qr and acc in registers; a loop bound by the runtime head width indexes them
    // dynamically, which spilled both arrays to scratch memory (a ViT-Tiny b32 forward ran 3x slower). Columns past the
    // head width are zero in Q, K and V, so they add nothing.
    float4 qr[SA_D4], acc[SA_D4];
    #pragma unroll
    for (int c = 0; c < SA_D4; c++) { qr[c] = (float4)(0.0f); acc[c] = (float4)(0.0f); }
    if (active)
    {
        __global const float* qp = q + ((long)b * seqQ + row) * width + h * headDim;
        #pragma unroll
        for (int c = 0; c < SA_D4; c++)
        {
            int d = 4 * c;
            qr[c] = (float4)(d < headDim ? qp[d] : 0.0f, d + 1 < headDim ? qp[d + 1] : 0.0f,
                             d + 2 < headDim ? qp[d + 2] : 0.0f, d + 3 < headDim ? qp[d + 3] : 0.0f) * scale;
        }
    }
    float m = -INFINITY, l = 0.0f;
    // The last key this row may see (all of them without a causal mask).
    const int lastKey = causal ? row + causalOffset : seqK - 1;
    __local float* kf = (__local float*)ks;
    __local float* vf = (__local float*)vs;
    for (int k0 = 0; k0 < seqK; k0 += SA_KEYS)
    {
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int e = lid; e < SA_KEYS * SA_DMAX; e += SA_ROWS)
        {
            int j = e / SA_DMAX, d = e - j * SA_DMAX, key = k0 + j;
            bool inside = key < seqK && d < headDim;
            long src = ((long)b * seqK + key) * width + h * headDim + d;
            kf[e] = inside ? k[src] : 0.0f;
            vf[e] = inside ? v[src] : 0.0f;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (!active) continue;
        int count = min(SA_KEYS, min(seqK, lastKey + 1) - k0);
        for (int j = 0; j < count; j++)
        {
            float4 s4 = (float4)(0.0f);
            #pragma unroll
            for (int c = 0; c < SA_D4; c++) s4 = mad(qr[c], ks[j][c], s4);
            float s = s4.x + s4.y + s4.z + s4.w;
            float mNew = fmax(m, s);
            float correction = exp(m - mNew);
            float p = exp(s - mNew);
            l = l * correction + p;
            #pragma unroll
            for (int c = 0; c < SA_D4; c++) acc[c] = mad((float4)(p), vs[j][c], acc[c] * correction);
            m = mNew;
        }
    }
    if (!active) return;
    // A row that may see no key (causal offset below zero) has no softmax: its output is zero, as the composed form's.
    float inv = l > 0.0f ? 1.0f / l : 0.0f;
    __global float* op = o + ((long)b * seqQ + row) * width + h * headDim;
    #pragma unroll
    for (int c = 0; c < SA_D4; c++)
    {
        float4 r = acc[c] * inv;
        int d = 4 * c;
        if (d < headDim) op[d] = r.x;
        if (d + 1 < headDim) op[d + 1] = r.y;
        if (d + 2 < headDim) op[d + 2] = r.z;
        if (d + 3 < headDim) op[d + 3] = r.w;
    }
}
";

        // 0 not tried, 1 compiled, 2 unavailable.
        private int _shortAttentionState;
        private readonly object _shortAttentionLock = new object();

        private bool ShortAttentionReady()
        {
            int state = System.Threading.Volatile.Read(ref _shortAttentionState);
            if (state != 0) return state == 1;
            lock (_shortAttentionLock)
            {
                if (_shortAttentionState != 0) return _shortAttentionState == 1;
                var context = _context;
                bool ok = false;
                if (context is not null && _maxWorkGroupSize >= 64)
                {
                    try
                    {
                        var program = CompileOrLoadCached(ShortAttentionSource, OpenClBuildOptions.SafeMathFlags,
                            "short-sequence attention");
                        _programs.Add(program);
                        _kernelCache["short_attention_forward"] = new DirectOpenClKernel(context, program, "short_attention_forward");
                        ok = true;
                    }
                    catch (Exception ex) when (ex is not OutOfMemoryException)
                    {
                        System.Diagnostics.Trace.TraceWarning($"OpenCL short-sequence attention unavailable: {ex.Message}");
                    }
                }
                System.Threading.Volatile.Write(ref _shortAttentionState, ok ? 1 : 2);
                return ok;
            }
        }

        bool IShortAttentionForwardBackend.SupportsShortAttention(int headDim)
            => headDim > 0 && headDim <= ShortAttentionMaxHeadDim;

        bool IShortAttentionForwardBackend.TryShortAttentionForward(IGpuBuffer query, IGpuBuffer key, IGpuBuffer value,
            IGpuBuffer output, int batch, int heads, int seqQ, int seqK, int headDim, float scale, bool causal, int causalOffset)
        {
            if (batch <= 0 || heads <= 0 || seqQ <= 0 || seqK <= 0 || headDim <= 0 || headDim > ShortAttentionMaxHeadDim
                || (long)batch * heads > 65535 || !ShortAttentionReady())
                return false;
            long width = (long)heads * headDim;
            if (query.Size < batch * seqQ * width || key.Size < batch * seqK * width || value.Size < batch * seqK * width
                || output.Size < batch * seqQ * width)
                return false;
            var kernel = _kernelCache["short_attention_forward"];
            kernel.SetArg(0, BufferHandle(query));
            kernel.SetArg(1, BufferHandle(key));
            kernel.SetArg(2, BufferHandle(value));
            kernel.SetArg(3, BufferHandle(output));
            kernel.SetArg(4, batch);
            kernel.SetArg(5, heads);
            kernel.SetArg(6, seqQ);
            kernel.SetArg(7, seqK);
            kernel.SetArg(8, headDim);
            kernel.SetArg(9, scale);
            kernel.SetArg(10, causal ? 1 : 0);
            kernel.SetArg(11, causalOffset);
            kernel.Execute2D((seqQ + 63) / 64 * 64, batch * heads, 64, 1);
            return true;
        }
    }
}