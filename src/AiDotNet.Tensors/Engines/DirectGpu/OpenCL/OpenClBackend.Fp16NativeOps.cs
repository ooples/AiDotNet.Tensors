// Copyright (c) AiDotNet. All rights reserved.
// FP16-NATIVE elementwise / activation ops for the OpenCL backend (Tensors #558): GELU, ReLU and
// residual add over half-stored activations, computed in FP32 in-register. These are the GPU
// counterpart of the CPU FP16-native emit (MixedPrecisionEmit.Unary/Binary) — keeping the activation
// chain genuinely Half end-to-end so the resident-memory win materializes (convert-based compression
// was proven unable to lower peak VRAM).

using System;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL.Kernels;
using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL
{
    public sealed partial class OpenClBackend
    {
        private readonly object _fp16NativeOpLock = new();
        private bool _fp16NativeOpsTried;
        private bool _fp16NativeOpsAvailable;

        /// <summary>
        /// True once the FP16-native op kernels compile on this device. They use the CORE OpenCL
        /// <c>vload_half</c>/<c>vstore_half</c> built-ins (not the optional <c>cl_khr_fp16</c> extension),
        /// so they are expected on every OpenCL 1.0+ device; the flag still gates on a successful compile.
        /// </summary>
        public bool SupportsFp16NativeOps => EnsureFp16NativeOpKernels();

        /// <summary>GELU over a half buffer: out[i] = gelu(in[i]); half in/out, FP32 compute.</summary>
        public void Fp16Gelu(IGpuBuffer input, IGpuBuffer output, int n)
            => DispatchFp16Unary(Fp16NativeOpKernels.GeluKernelName, input, output, n);

        /// <summary>ReLU over a half buffer: out[i] = max(in[i], 0); half in/out, FP32 compute.</summary>
        public void Fp16Relu(IGpuBuffer input, IGpuBuffer output, int n)
            => DispatchFp16Unary(Fp16NativeOpKernels.ReluKernelName, input, output, n);

        /// <summary>Residual add over half buffers: out[i] = a[i] + b[i]; half in/out, FP32 accumulate.</summary>
        public void Fp16Add(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int n)
        {
            if (a is null) throw new ArgumentNullException(nameof(a));
            if (b is null) throw new ArgumentNullException(nameof(b));
            if (output is null) throw new ArgumentNullException(nameof(output));
            if (n <= 0) throw new ArgumentOutOfRangeException(nameof(n), "Element count must be positive.");
            if (!EnsureFp16NativeOpKernels() || !_kernelCache.TryGetValue(Fp16NativeOpKernels.AddKernelName, out var k))
                throw new NotSupportedException("FP16-native op kernels are not available on this OpenCL device.");

            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)a).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)b).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)output).Buffer.Handle);
            k.SetArg(arg++, n);
            k.Execute1D(n, Math.Min(256, n));
        }

        /// <summary>Row softmax over the last axis of a half buffer: one work-group per row, FP32 max/sum,
        /// half in/out. OpenCL counterpart of the CUDA fp16_softmax_native.</summary>
        public void Fp16Softmax(IGpuBuffer input, IGpuBuffer output, int rows, int cols)
        {
            if (input is null) throw new ArgumentNullException(nameof(input));
            if (output is null) throw new ArgumentNullException(nameof(output));
            if (rows <= 0 || cols <= 0) throw new ArgumentException($"rows/cols must be positive (rows={rows}, cols={cols}).");
            if (!EnsureFp16NativeOpKernels() || !_kernelCache.TryGetValue(Fp16NativeOpKernels.SoftmaxKernelName, out var k))
                throw new NotSupportedException("FP16-native op kernels are not available on this OpenCL device.");

            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)input).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)output).Buffer.Handle);
            k.SetArg(arg++, rows);
            k.SetArg(arg++, cols);
            int local = Fp16NativeOpKernels.RowReduceLocalSize;
            k.Execute1D(rows * local, local); // one work-group per row
        }

        /// <summary>Row layernorm over the last axis of a half buffer with half gamma/beta: one work-group
        /// per row, FP32 mean/var, half in/out; optionally writes per-row FP32 mean/variance. OpenCL
        /// counterpart of the CUDA fp16_layernorm_native (the kernel always writes mean/var, so temporary
        /// buffers are supplied when the caller passes none).</summary>
        public void Fp16LayerNorm(IGpuBuffer input, IGpuBuffer gamma, IGpuBuffer beta, IGpuBuffer output,
            IGpuBuffer meanFp32, IGpuBuffer varFp32, int rows, int cols, float eps)
        {
            if (input is null) throw new ArgumentNullException(nameof(input));
            if (gamma is null) throw new ArgumentNullException(nameof(gamma));
            if (beta is null) throw new ArgumentNullException(nameof(beta));
            if (output is null) throw new ArgumentNullException(nameof(output));
            if (rows <= 0 || cols <= 0) throw new ArgumentException($"rows/cols must be positive (rows={rows}, cols={cols}).");
            if (eps <= 0f || float.IsNaN(eps) || float.IsInfinity(eps))
                throw new ArgumentOutOfRangeException(nameof(eps), eps, "eps must be finite and positive.");
            if (!EnsureFp16NativeOpKernels() || !_kernelCache.TryGetValue(Fp16NativeOpKernels.LayerNormKernelName, out var k))
                throw new NotSupportedException("FP16-native op kernels are not available on this OpenCL device.");

            IGpuBuffer? tmpMean = null, tmpVar = null;
            var meanBuf = meanFp32 ?? (tmpMean = AllocateBuffer(rows));
            var varBuf = varFp32 ?? (tmpVar = AllocateBuffer(rows));
            try
            {
                uint arg = 0;
                k.SetArg(arg++, ((DirectOpenClGpuBuffer)input).Buffer.Handle);
                k.SetArg(arg++, ((DirectOpenClGpuBuffer)gamma).Buffer.Handle);
                k.SetArg(arg++, ((DirectOpenClGpuBuffer)beta).Buffer.Handle);
                k.SetArg(arg++, ((DirectOpenClGpuBuffer)output).Buffer.Handle);
                k.SetArg(arg++, ((DirectOpenClGpuBuffer)meanBuf).Buffer.Handle);
                k.SetArg(arg++, ((DirectOpenClGpuBuffer)varBuf).Buffer.Handle);
                k.SetArg(arg++, rows);
                k.SetArg(arg++, cols);
                k.SetArg(arg++, eps);
                int local = Fp16NativeOpKernels.RowReduceLocalSize;
                k.Execute1D(rows * local, local); // one work-group per row
                if (tmpMean is not null || tmpVar is not null) _context?.Finish(); // ensure writes land before disposing temps
            }
            finally
            {
                tmpMean?.Dispose();
                tmpVar?.Dispose();
            }
        }

        private void DispatchFp16Unary(string kernelName, IGpuBuffer input, IGpuBuffer output, int n)
        {
            if (input is null) throw new ArgumentNullException(nameof(input));
            if (output is null) throw new ArgumentNullException(nameof(output));
            if (n <= 0) throw new ArgumentOutOfRangeException(nameof(n), "Element count must be positive.");
            if (!EnsureFp16NativeOpKernels() || !_kernelCache.TryGetValue(kernelName, out var k))
                throw new NotSupportedException("FP16-native op kernels are not available on this OpenCL device.");

            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)input).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)output).Buffer.Handle);
            k.SetArg(arg++, n);
            k.Execute1D(n, Math.Min(256, n));
        }

        private bool EnsureFp16NativeOpKernels()
        {
            if (_fp16NativeOpsTried)
                return _fp16NativeOpsAvailable;

            lock (_fp16NativeOpLock)
            {
                if (_fp16NativeOpsTried)
                    return _fp16NativeOpsAvailable;

                _fp16NativeOpsTried = true;

                if (_context == null)
                {
                    _fp16NativeOpsAvailable = false;
                    return false;
                }

                try
                {
                    var program = CompileOrLoadCached(
                        Fp16NativeOpKernels.GetSource(),
                        OpenClBuildOptions.OptimizationFlags,
                        "FP16-native op kernels");
                    _programs.Add(program);
                    foreach (var name in Fp16NativeOpKernels.GetKernelNames())
                        _kernelCache[name] = new DirectOpenClKernel(_context, program, name);

                    _fp16NativeOpsAvailable = true;
                }
                catch (Exception ex)
                {
                    // Non-fatal: a driver that rejects the kernel just means FP16-native ops fall back to
                    // the convert path. Don't crash init.
                    WriteDiag($"[OpenClBackend] FP16-native op kernel compilation failed (non-fatal): {ex.Message}");
                    _fp16NativeOpsAvailable = false;
                }

                return _fp16NativeOpsAvailable;
            }
        }
    }
}
