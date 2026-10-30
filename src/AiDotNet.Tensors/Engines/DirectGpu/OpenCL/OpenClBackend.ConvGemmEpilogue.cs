// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL
{
    // The output step of the im2col + GEMM convolution in one pass: the GEMM leaves tmp [Cout, B * L]; this writes the
    // NCHW output [B, Cout, L] with the bias and the ReLU applied. It replaced a permute, a conv2d_bias_add and a relu
    // pass (three reads and three writes of the output) per convolution. With B = 1 tmp is the output and it runs in
    // place, replacing the two epilogue passes.
    public sealed partial class OpenClBackend
    {
        private const string ConvGemmEpilogueSource = @"
// epi bit 0: + bias[c]; bit 1: ReLU, NaN passing through (the relu kernel's bit test, which fast math cannot fold).
__kernel void conv_gemm_epilogue(__global const float* tmp, __global float* out, __global const float* bias,
    const int cout, const int batch, const int l, const int epi)
{
    const int idx = get_global_id(0);
    const int total = batch * cout * l;
    if (idx >= total) return;
    const int p = idx % l;
    const int bc = idx / l;
    const int c = bc % cout;
    const int b = bc / cout;
    float v = tmp[((long)c * batch + b) * l + p];
    if (epi & 1) v = v + bias[c];
    if (epi & 2)
    {
        uint bits = as_uint(v) & 0x7fffffffu;
        v = (bits > 0x7f800000u) ? v : fmax(0.0f, v);
    }
    out[idx] = v;
}

// The same epilogue in place on an NCHW output [B, Cout, L]: one pass where the bias add and the ReLU took one each.
__kernel void conv_nchw_epilogue(__global float* out, __global const float* bias, const int cout, const int l,
    const int total, const int epi)
{
    const int idx = get_global_id(0);
    if (idx >= total) return;
    float v = out[idx];
    if (epi & 1) v = v + bias[(idx / l) % cout];
    if (epi & 2)
    {
        uint bits = as_uint(v) & 0x7fffffffu;
        v = (bits > 0x7f800000u) ? v : fmax(0.0f, v);
    }
    out[idx] = v;
}
";

        // 0 not tried, 1 compiled, 2 unavailable.
        private int _convGemmEpilogueState;
        // Set once state is 1. The launch reads this rather than _kernelCache: the kernel compiles on first use, not
        // with the backend's modules, so a cache lookup is one the kernel-lookup integrity check cannot see compiled.
        private DirectOpenClKernel? _convGemmEpilogueKernel;
        private DirectOpenClKernel? _convNchwEpilogueKernel;
        private readonly object _convGemmEpilogueLock = new object();

        private bool ConvGemmEpilogueReady()
        {
            int state = System.Threading.Volatile.Read(ref _convGemmEpilogueState);
            if (state != 0) return state == 1;
            lock (_convGemmEpilogueLock)
            {
                if (_convGemmEpilogueState != 0) return _convGemmEpilogueState == 1;
                var context = _context;
                bool ok = false;
                if (context is not null)
                {
                    try
                    {
                        var program = CompileOrLoadCached(ConvGemmEpilogueSource, OpenClBuildOptions.SafeMathFlags,
                            "convolution GEMM epilogue");
                        _programs.Add(program);
                        _convGemmEpilogueKernel = new DirectOpenClKernel(context, program, "conv_gemm_epilogue");
                        _kernelCache["conv_gemm_epilogue"] = _convGemmEpilogueKernel;
                        _convNchwEpilogueKernel = new DirectOpenClKernel(context, program, "conv_nchw_epilogue");
                        _kernelCache["conv_nchw_epilogue"] = _convNchwEpilogueKernel;
                        ok = true;
                    }
                    catch (Exception ex) when (ex is not OutOfMemoryException)
                    {
                        System.Diagnostics.Trace.TraceWarning($"OpenCL convolution GEMM epilogue unavailable: {ex.Message}");
                    }
                }
                System.Threading.Volatile.Write(ref _convGemmEpilogueState, ok ? 1 : 2);
                return ok;
            }
        }

        // tmp [Cout, B * L] -> output [B, Cout, L] with the epilogue; tmp may be the output when B = 1. False when the
        // kernel is unavailable (the caller then permutes and runs the epilogue passes).
        private bool TryConvGemmEpilogue(IGpuBuffer tmp, in TunedConv2DArgs a)
        {
            if (!ConvGemmEpilogueReady() || !ConvEpilogue.Fuses(a.Activation)
                || _convGemmEpilogueKernel is not { } kernel) return false;
            int l = a.OutHeight * a.OutWidth;
            int total = a.OutputCount;
            if (total <= 0) return true;
            kernel.SetArg(0, BufferHandle(tmp));
            kernel.SetArg(1, BufferHandle(a.Output));
            kernel.SetArg(2, ConvEpilogueBias(a));
            kernel.SetArg(3, a.OutChannels);
            kernel.SetArg(4, a.Batch);
            kernel.SetArg(5, l);
            kernel.SetArg(6, ConvEpilogueFlags(a));
            int local = ElementwiseGroupSize(total);
            kernel.Execute1D((total + local - 1) / local * local, local);
            return true;
        }

        // The epilogue of a convolution that wrote its NCHW output directly, in one in-place pass. False when there is
        // nothing to apply or the kernel is unavailable (the caller runs the separate passes).
        private bool TryConvNchwEpilogue(in TunedConv2DArgs a)
        {
            int flags = ConvEpilogueFlags(a);
            if (flags == 0 || !ConvGemmEpilogueReady() || !ConvEpilogue.Fuses(a.Activation)
                || _convNchwEpilogueKernel is not { } kernel) return false;
            int total = a.OutputCount;
            if (total <= 0) return true;
            kernel.SetArg(0, BufferHandle(a.Output));
            kernel.SetArg(1, ConvEpilogueBias(a));
            kernel.SetArg(2, a.OutChannels);
            kernel.SetArg(3, a.OutHeight * a.OutWidth);
            kernel.SetArg(4, total);
            kernel.SetArg(5, flags);
            int local = ElementwiseGroupSize(total);
            kernel.Execute1D((total + local - 1) / local * local, local);
            return true;
        }
    }
}