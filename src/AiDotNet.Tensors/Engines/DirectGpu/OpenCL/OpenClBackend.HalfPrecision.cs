// Copyright (c) AiDotNet. All rights reserved.
// IGpuHalfPrecisionBackend implementation for the OpenCL backend (issue #560):
// FP16 GEMM so MatrixMultiply<Half> runs on the GPU instead of dropping to a
// scalar CPU fallback.

using System;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL.Kernels;
using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL
{
    /// <summary>
    /// Half-precision GEMM dispatch for the <see cref="IGpuHalfPrecisionBackend"/>
    /// capability interface. Mirrors the CUDA backend's contract so the
    /// backend-agnostic <c>DirectGpuTensorEngine.MatrixMultiply</c> FP16 path
    /// (issue #560) lights up on OpenCL devices — AMD, Intel, and NVIDIA — with
    /// no extra dispatch logic in the engine.
    /// </summary>
    public sealed partial class OpenClBackend : IGpuHalfPrecisionBackend
    {
        private readonly object _halfGemmLock = new();
        private bool _halfGemmKernelsTried;
        private bool _halfGemmKernelsAvailable;

        /// <inheritdoc/>
        /// <remarks>
        /// True once the FP16 GEMM kernels compile on this device. The kernels
        /// use the CORE OpenCL <c>vload_half</c> built-in (not the optional
        /// <c>cl_khr_fp16</c> extension), so they are expected to be available
        /// on every OpenCL 1.0+ device; the flag still gates on a successful
        /// compile so a broken driver degrades gracefully to the CPU path.
        /// </remarks>
        public bool SupportsHgemm => EnsureHalfGemmKernels();

        /// <inheritdoc/>
        public void Hgemm(IGpuBuffer aFp16, IGpuBuffer bFp16, IGpuBuffer cFp16,
            int m, int n, int k)
            => DispatchHalfGemm(HalfPrecisionGemmKernels.Fp16In16fOutKernelName,
                aFp16, bFp16, cFp16, m, n, k);

        /// <inheritdoc/>
        public void GemmFp16In32fOut(IGpuBuffer aFp16, IGpuBuffer bFp16, IGpuBuffer cFp32,
            int m, int n, int k)
            => DispatchHalfGemm(HalfPrecisionGemmKernels.Fp16In32fOutKernelName,
                aFp16, bFp16, cFp32, m, n, k);

        /// <summary>
        /// Lazily compiles and caches the FP16 GEMM kernels on first use.
        /// Non-fatal on failure (returns false → caller falls back to CPU).
        /// </summary>
        private bool EnsureHalfGemmKernels()
        {
            if (_halfGemmKernelsTried)
                return _halfGemmKernelsAvailable;

            lock (_halfGemmLock)
            {
                if (_halfGemmKernelsTried)
                    return _halfGemmKernelsAvailable;

                _halfGemmKernelsTried = true;

                if (_context == null)
                {
                    _halfGemmKernelsAvailable = false;
                    return false;
                }

                try
                {
                    var program = CompileOrLoadCached(
                        HalfPrecisionGemmKernels.GetSource(),
                        OpenClBuildOptions.OptimizationFlags,
                        "Half-precision GEMM kernels");
                    _programs.Add(program);
                    foreach (var name in HalfPrecisionGemmKernels.GetKernelNames())
                        _kernelCache[name] = new DirectOpenClKernel(_context, program, name);

                    _halfGemmKernelsAvailable = true;
                }
                catch (Exception ex)
                {
                    // Non-fatal: a driver that rejects the kernel just means
                    // Half matmul stays on the CPU path. Don't crash init.
                    WriteDiag($"[OpenClBackend] Half-precision GEMM kernel compilation failed (non-fatal): {ex.Message}");
                    _halfGemmKernelsAvailable = false;
                }

                return _halfGemmKernelsAvailable;
            }
        }

        private void DispatchHalfGemm(string kernelName, IGpuBuffer a, IGpuBuffer b, IGpuBuffer c,
            int m, int n, int k)
        {
            if (a is null) throw new ArgumentNullException(nameof(a));
            if (b is null) throw new ArgumentNullException(nameof(b));
            if (c is null) throw new ArgumentNullException(nameof(c));
            // Report the offending dimension by name (mirrors the CUDA backend)
            // so a bad shape is diagnosable rather than failing opaquely in the kernel.
            if (m <= 0) throw new ArgumentOutOfRangeException(nameof(m), "Dimensions must be positive.");
            if (n <= 0) throw new ArgumentOutOfRangeException(nameof(n), "Dimensions must be positive.");
            if (k <= 0) throw new ArgumentOutOfRangeException(nameof(k), "Dimensions must be positive.");

            if (!EnsureHalfGemmKernels() || !_kernelCache.TryGetValue(kernelName, out var kernel))
                throw new NotSupportedException(
                    "Half-precision GEMM kernels are not available on this OpenCL device.");

            const int ts = HalfPrecisionGemmKernels.TileSize;

            uint arg = 0;
            kernel.SetArg(arg++, ((DirectOpenClGpuBuffer)a).Buffer.Handle);
            kernel.SetArg(arg++, ((DirectOpenClGpuBuffer)b).Buffer.Handle);
            kernel.SetArg(arg++, ((DirectOpenClGpuBuffer)c).Buffer.Handle);
            kernel.SetArg(arg++, m);
            kernel.SetArg(arg++, n);
            kernel.SetArg(arg++, k);

            // Global sizes rounded up to a full tile; the kernel guards the
            // out-of-range lanes (they still participate in the local-memory
            // loads + barriers but skip the final store).
            int globalX = ((n + ts - 1) / ts) * ts;   // N axis (dim 0)
            int globalY = ((m + ts - 1) / ts) * ts;   // M axis (dim 1)
            kernel.Execute2D(globalX, globalY, ts, ts);
        }
    }
}
