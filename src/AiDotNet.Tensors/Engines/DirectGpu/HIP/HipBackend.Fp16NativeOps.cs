// Copyright (c) AiDotNet. All rights reserved.
// FP16-NATIVE elementwise / activation ops for the HIP backend (Tensors #558): GELU / ReLU / residual
// add over half-stored activations. The kernels (fp16_gelu / fp16_relu / fp16_add in HipFp16Kernels)
// read the activation DIRECTLY from a half buffer, compute in FP32 in-register (packed __half2, 2
// elems/thread), and write half — no FP32 buffer materialized, no convert transient. GPU counterpart of
// the CPU FP16-native emit; the launchers below expose them through the same surface as OpenCL/Vulkan/
// Metal/CUDA.

using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.HIP
{
    public sealed partial class HipBackend
    {
        /// <summary>
        /// True once the FP16 kernel module (which carries fp16_gelu / fp16_relu / fp16_add) is compiled.
        /// </summary>
        public bool SupportsFp16NativeOps =>
            _fp16Module != IntPtr.Zero && _kernelCache.ContainsKey("fp16_gelu");

        /// <summary>GELU over a half buffer: out[i] = gelu(in[i]); half in/out, FP32 math.</summary>
        public unsafe void Fp16Gelu(IGpuBuffer input, IGpuBuffer output, int n)
            => LaunchUnaryFp16Native("fp16_gelu", input, output, n);

        /// <summary>ReLU over a half buffer: out[i] = max(in[i], 0); half in/out, FP32 math.</summary>
        public unsafe void Fp16Relu(IGpuBuffer input, IGpuBuffer output, int n)
            => LaunchUnaryFp16Native("fp16_relu", input, output, n);

        /// <summary>Residual add over half buffers: out[i] = a[i] + b[i]; half in/out, FP32 accumulate.</summary>
        public unsafe void Fp16Add(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int n)
        {
            if (a is null) throw new ArgumentNullException(nameof(a));
            if (b is null) throw new ArgumentNullException(nameof(b));
            if (output is null) throw new ArgumentNullException(nameof(output));
            if (n <= 0) throw new ArgumentOutOfRangeException(nameof(n), "Element count must be positive.");
            if (!_kernelCache.TryGetValue("fp16_add", out var kernel))
                throw new InvalidOperationException("HIP kernel not found: fp16_add");

            // The kernel processes two elements per thread (packed __half2).
            uint grid = (uint)(((n + 1) / 2 + DefaultBlockSize - 1) / DefaultBlockSize);
            IntPtr aP = a.Handle, bP = b.Handle, oP = output.Handle;
            int size = n;
            void** args = stackalloc void*[4];
            args[0] = &aP; args[1] = &bP; args[2] = &oP; args[3] = &size;
            LaunchKernel(kernel, grid, DefaultBlockSize, args);
        }

        /// <summary>Row softmax over the last axis of a half buffer: one block per row, FP32 max/sum
        /// reductions, half in/out. HIP counterpart of the CUDA fp16_softmax_native.</summary>
        public unsafe void Fp16Softmax(IGpuBuffer input, IGpuBuffer output, int rows, int cols)
        {
            if (input is null) throw new ArgumentNullException(nameof(input));
            if (output is null) throw new ArgumentNullException(nameof(output));
            if (rows <= 0 || cols <= 0) throw new ArgumentException($"rows/cols must be positive (rows={rows}, cols={cols}).");
            if (!_kernelCache.TryGetValue("fp16_softmax", out var kernel))
                throw new InvalidOperationException("HIP kernel not found: fp16_softmax");
            IntPtr inP = input.Handle, oP = output.Handle;
            int r = rows, c = cols;
            void** args = stackalloc void*[4];
            args[0] = &inP; args[1] = &oP; args[2] = &r; args[3] = &c;
            uint sharedBytes = (uint)((int)DefaultBlockSize * sizeof(float));
            LaunchKernel(kernel, (uint)rows, DefaultBlockSize, args, sharedBytes);
        }

        /// <summary>Row layernorm over the last axis of a half buffer with half gamma/beta: one block per row,
        /// FP32 mean/var, half in/out; optionally writes per-row FP32 mean/variance. HIP counterpart of the
        /// CUDA fp16_layernorm_native.</summary>
        public unsafe void Fp16LayerNorm(IGpuBuffer input, IGpuBuffer gamma, IGpuBuffer beta, IGpuBuffer output,
            IGpuBuffer meanFp32, IGpuBuffer varFp32, int rows, int cols, float eps)
        {
            if (input is null) throw new ArgumentNullException(nameof(input));
            if (gamma is null) throw new ArgumentNullException(nameof(gamma));
            if (beta is null) throw new ArgumentNullException(nameof(beta));
            if (output is null) throw new ArgumentNullException(nameof(output));
            if (rows <= 0 || cols <= 0) throw new ArgumentException($"rows/cols must be positive (rows={rows}, cols={cols}).");
            if (eps <= 0f || float.IsNaN(eps) || float.IsInfinity(eps))
                throw new ArgumentOutOfRangeException(nameof(eps), eps, "eps must be finite and positive.");
            if (!_kernelCache.TryGetValue("fp16_layernorm", out var kernel))
                throw new InvalidOperationException("HIP kernel not found: fp16_layernorm");
            IntPtr inP = input.Handle, gP = gamma.Handle, bP = beta.Handle, oP = output.Handle;
            IntPtr mP = meanFp32 is null ? IntPtr.Zero : meanFp32.Handle;
            IntPtr vP = varFp32 is null ? IntPtr.Zero : varFp32.Handle;
            int r = rows, c = cols; float e = eps;
            void** args = stackalloc void*[9];
            args[0] = &inP; args[1] = &gP; args[2] = &bP; args[3] = &oP;
            args[4] = &mP; args[5] = &vP; args[6] = &r; args[7] = &c; args[8] = &e;
            uint sharedBytes = (uint)((int)DefaultBlockSize * sizeof(float));
            LaunchKernel(kernel, (uint)rows, DefaultBlockSize, args, sharedBytes);
        }

        private unsafe void LaunchUnaryFp16Native(string kernelName, IGpuBuffer input, IGpuBuffer output, int n)
        {
            if (input is null) throw new ArgumentNullException(nameof(input));
            if (output is null) throw new ArgumentNullException(nameof(output));
            if (n <= 0) throw new ArgumentOutOfRangeException(nameof(n), "Element count must be positive.");
            if (!_kernelCache.TryGetValue(kernelName, out var kernel))
                throw new InvalidOperationException($"HIP kernel not found: {kernelName}");

            uint grid = (uint)(((n + 1) / 2 + DefaultBlockSize - 1) / DefaultBlockSize);
            IntPtr inP = input.Handle, oP = output.Handle;
            int size = n;
            void** args = stackalloc void*[3];
            args[0] = &inP; args[1] = &oP; args[2] = &size;
            LaunchKernel(kernel, grid, DefaultBlockSize, args);
        }
    }
}
