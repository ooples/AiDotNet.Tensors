// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL
{
    // A batch-1 MLP forward in one work-group: the activations ping-pong through local memory and each layer reads its
    // weights once, coalesced across the group. One launch where the per-layer path made about seven (a GEMV, its reduce
    // and an epilogue per layer); the weights still come from global memory, through one compute unit.
    public sealed partial class OpenClBackend : IRowMlpForwardBackend
    {
        private const int RowMlpMaxLayers = 4;
        private const int RowMlpMaxWidth = 2048;   // two local buffers of this many floats: 16 KB
        private const int RowMlpGroup = 256;

        private const string RowMlpSource = @"
#define RM_MAXW 2048
#define RM_GROUP 256

// One layer: dst[j] = act(sum_k src[k] * w[k * n + j] + b[j]) for j < n. With n >= the group each thread takes whole
// columns; otherwise the group splits K into splits = group / n slices and sums the slices' partials in local memory.
inline void row_mlp_layer(__local const float* src, __local float* dst, __global float* y, int toGlobal,
    __global const float* w, __global const float* b, int hasBias, int k, int n, int relu, __local float* partial)
{
    const int lid = get_local_id(0);
    if (n >= RM_GROUP)
    {
        for (int j = lid; j < n; j += RM_GROUP)
        {
            float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
            int kk = 0;
            for (; kk + 3 < k; kk += 4)
            {
                a0 += src[kk] * w[(long)kk * n + j];
                a1 += src[kk + 1] * w[(long)(kk + 1) * n + j];
                a2 += src[kk + 2] * w[(long)(kk + 2) * n + j];
                a3 += src[kk + 3] * w[(long)(kk + 3) * n + j];
            }
            for (; kk < k; kk++) a0 += src[kk] * w[(long)kk * n + j];
            float v = (a0 + a1) + (a2 + a3);
            if (hasBias) v += b[j];
            if (relu)
            {
                uint bits = as_uint(v) & 0x7fffffffu;
                v = (bits > 0x7f800000u) ? v : fmax(0.0f, v);
            }
            if (toGlobal) y[j] = v; else dst[j] = v;
        }
    }
    else
    {
        const int splits = RM_GROUP / n;
        const int j = lid % n, s = lid / n;
        float acc = 0.0f;
        if (s < splits)
        {
            const int chunk = (k + splits - 1) / splits;
            const int k0 = s * chunk, k1 = min(k, k0 + chunk);
            for (int kk = k0; kk < k1; kk++) acc += src[kk] * w[(long)kk * n + j];
        }
        partial[lid] = acc;
    }
    // Both paths reach this one barrier: POCL's work-group region formation aborts the process on a barrier that an
    // early return lets some paths skip (Kernel.cc createParallelRegionBefore assertion).
    barrier(CLK_LOCAL_MEM_FENCE);
    if (n < RM_GROUP && lid < n)
    {
        const int splits = RM_GROUP / n;
        float v = 0.0f;
        for (int t = 0; t < splits; t++) v += partial[t * n + lid];
        if (hasBias) v += b[lid];
        if (relu)
        {
            uint bits = as_uint(v) & 0x7fffffffu;
            v = (bits > 0x7f800000u) ? v : fmax(0.0f, v);
        }
        if (toGlobal) y[lid] = v; else dst[lid] = v;
    }
}

// A layer whose width is a multiple of 4: work item (cg, s) accumulates four adjacent columns (one vload4 per weight
// row) over K-slice s, unrolled four deep so each thread keeps four loads in flight; the slices' partials are summed in
// local memory. The scalar form above had one 4-byte load in flight per thread, so a 512 x 128 layer took ~100 us.
inline void row_mlp_layer4(__local const float* src, __local float* dst, __global float* y, int toGlobal,
    __global const float* w, __global const float* b, int hasBias, int k, int n, int relu, __local float4* partial4)
{
    const int lid = get_local_id(0);
    const int groups = n >> 2;
    for (int base = 0; base < groups; base += RM_GROUP)
    {
        const int active = min(RM_GROUP, groups - base);
        const int splits = RM_GROUP / active;
        const int cg = base + lid % active, s = lid / active;
        float4 a0 = (float4)(0.0f), a1 = (float4)(0.0f), a2 = (float4)(0.0f), a3 = (float4)(0.0f);
        if (s < splits)
        {
            const int chunk = (k + splits - 1) / splits;
            const int k0 = s * chunk, k1 = min(k, k0 + chunk);
            __global const float* col = w + 4 * cg;
            int kk = k0;
            for (; kk + 3 < k1; kk += 4)
            {
                a0 += src[kk] * vload4(0, col + (long)kk * n);
                a1 += src[kk + 1] * vload4(0, col + (long)(kk + 1) * n);
                a2 += src[kk + 2] * vload4(0, col + (long)(kk + 2) * n);
                a3 += src[kk + 3] * vload4(0, col + (long)(kk + 3) * n);
            }
            for (; kk < k1; kk++) a0 += src[kk] * vload4(0, col + (long)kk * n);
        }
        partial4[lid] = (a0 + a1) + (a2 + a3);
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lid < active)
        {
            float4 v = (float4)(0.0f);
            for (int t = 0; t < splits; t++) v += partial4[t * active + lid];
            const int j = 4 * (base + lid);
            float r[4] = { v.x, v.y, v.z, v.w };
            for (int q = 0; q < 4; q++)
            {
                float o = r[q];
                if (hasBias) o += b[j + q];
                if (relu)
                {
                    uint bits = as_uint(o) & 0x7fffffffu;
                    o = (bits > 0x7f800000u) ? o : fmax(0.0f, o);
                }
                if (toGlobal) y[j + q] = o; else dst[j + q] = o;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

__kernel __attribute__((reqd_work_group_size(RM_GROUP, 1, 1)))
void row_mlp_forward(__global const float* x, __global float* y,
    __global const float* w0, __global const float* b0, __global const float* w1, __global const float* b1,
    __global const float* w2, __global const float* b2, __global const float* w3, __global const float* b3,
    const int layers, const int d0, const int d1, const int d2, const int d3, const int d4,
    const int biasMask, const int hiddenRelu, const int outputRelu)
{
    __local float bufA[RM_MAXW];
    __local float bufB[RM_MAXW];
    __local float partial[RM_GROUP];
    __local float4 partial4[RM_GROUP];
    const int lid = get_local_id(0);
    for (int i = lid; i < d0; i += RM_GROUP) bufA[i] = x[i];
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int l = 0; l < layers; l++)
    {
        __global const float* w = l == 0 ? w0 : l == 1 ? w1 : l == 2 ? w2 : w3;
        __global const float* b = l == 0 ? b0 : l == 1 ? b1 : l == 2 ? b2 : b3;
        const int k = l == 0 ? d0 : l == 1 ? d1 : l == 2 ? d2 : d3;
        const int n = l == 0 ? d1 : l == 1 ? d2 : l == 2 ? d3 : d4;
        const int last = l == layers - 1;
        // Even layers read bufA and write bufB; odd layers the reverse.
        if ((n & 3) == 0)
        {
            if ((l & 1) == 0)
                row_mlp_layer4(bufA, bufB, y, last, w, b, (biasMask >> l) & 1, k, n, last ? outputRelu : hiddenRelu, partial4);
            else
                row_mlp_layer4(bufB, bufA, y, last, w, b, (biasMask >> l) & 1, k, n, last ? outputRelu : hiddenRelu, partial4);
        }
        else if ((l & 1) == 0)
            row_mlp_layer(bufA, bufB, y, last, w, b, (biasMask >> l) & 1, k, n, last ? outputRelu : hiddenRelu, partial);
        else
            row_mlp_layer(bufB, bufA, y, last, w, b, (biasMask >> l) & 1, k, n, last ? outputRelu : hiddenRelu, partial);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}
";

        // 0 not tried, 1 compiled, 2 unavailable.
        private int _rowMlpState;
        private DirectOpenClKernel? _rowMlpKernel;
        private readonly object _rowMlpLock = new object();

        private bool RowMlpReady()
        {
            int state = System.Threading.Volatile.Read(ref _rowMlpState);
            if (state != 0) return state == 1;
            lock (_rowMlpLock)
            {
                if (_rowMlpState != 0) return _rowMlpState == 1;
                var context = _context;
                bool ok = false;
                if (context is not null && _maxWorkGroupSize >= RowMlpGroup)
                {
                    try
                    {
                        var program = CompileOrLoadCached(RowMlpSource, OpenClBuildOptions.SafeMathFlags, "batch-1 MLP forward");
                        _programs.Add(program);
                        _rowMlpKernel = new DirectOpenClKernel(context, program, "row_mlp_forward");
                        _kernelCache["row_mlp_forward"] = _rowMlpKernel;
                        ok = true;
                    }
                    catch (Exception ex) when (ex is not OutOfMemoryException)
                    {
                        System.Diagnostics.Trace.TraceWarning($"OpenCL batch-1 MLP forward unavailable: {ex.Message}");
                    }
                }
                System.Threading.Volatile.Write(ref _rowMlpState, ok ? 1 : 2);
                return ok;
            }
        }

        // Whether the kernel takes this MLP: layer count, activations, widths and every buffer large enough.
        private static bool RowMlpShapeSupported(IGpuBuffer input, IGpuBuffer[] weights, IGpuBuffer?[] biases,
            int[] widths, FusedActivationType hiddenActivation, FusedActivationType outputActivation, IGpuBuffer output)
        {
            int layers = weights.Length;
            if (layers < 1 || layers > RowMlpMaxLayers || biases.Length != layers || widths.Length != layers + 1
                || hiddenActivation is not (FusedActivationType.None or FusedActivationType.ReLU)
                || outputActivation is not (FusedActivationType.None or FusedActivationType.ReLU))
                return false;
            for (int i = 0; i <= layers; i++)
                if (widths[i] <= 0 || widths[i] > RowMlpMaxWidth) return false;
            for (int l = 0; l < layers; l++)
            {
                if (weights[l].Size < (long)widths[l] * widths[l + 1]) return false;
                if (biases[l] is { } b && b.Size < widths[l + 1]) return false;
            }
            return input.Size >= widths[0] && output.Size >= widths[layers];
        }

        bool IRowMlpForwardBackend.TryRowMlpForward(IGpuBuffer input, IGpuBuffer[] weights, IGpuBuffer?[] biases,
            int[] widths, FusedActivationType hiddenActivation, FusedActivationType outputActivation, IGpuBuffer output)
        {
            if (!RowMlpShapeSupported(input, weights, biases, widths, hiddenActivation, outputActivation, output)
                || !RowMlpReady() || _rowMlpKernel is not { } kernel) return false;
            int layers = weights.Length;

            uint arg = 0;
            kernel.SetArg(arg++, BufferHandle(input));
            kernel.SetArg(arg++, BufferHandle(output));
            int biasMask = 0;
            for (int l = 0; l < RowMlpMaxLayers; l++)
            {
                // Unused layer slots, and absent biases, take a real buffer the kernel never reads.
                var w = l < layers ? weights[l] : weights[0];
                var b = l < layers ? biases[l] : null;
                if (b is not null) biasMask |= 1 << l;
                kernel.SetArg(arg++, BufferHandle(w));
                kernel.SetArg(arg++, BufferHandle(b ?? w));
            }
            kernel.SetArg(arg++, layers);
            for (int i = 0; i <= RowMlpMaxLayers; i++) kernel.SetArg(arg++, i <= layers ? widths[i] : 1);
            kernel.SetArg(arg++, biasMask);
            kernel.SetArg(arg++, hiddenActivation == FusedActivationType.ReLU ? 1 : 0);
            kernel.SetArg(arg, outputActivation == FusedActivationType.ReLU ? 1 : 0);
            kernel.Execute1D(RowMlpGroup, RowMlpGroup);
            return true;
        }
    }
}