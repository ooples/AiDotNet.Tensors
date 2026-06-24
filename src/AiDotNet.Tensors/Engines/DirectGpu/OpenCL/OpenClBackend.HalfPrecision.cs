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
        /// <remarks>The fused backward is two dispatches of the same FP16 GEMM kernel program (the
        /// <c>gemm_fp16_backward</c> transposed variant), so it tracks <see cref="SupportsHgemm"/>.</remarks>
        public bool SupportsFp16FusedBackward => EnsureHalfGemmKernels();

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

        /// <inheritdoc/>
        /// <remarks>
        /// Two dispatches of the transposed <c>gemm_fp16_backward</c> kernel (FP16 in, FP32 accumulate, no
        /// materialized transpose): gradA[M,K] = gradC·Bᵀ (transA=0, transB=1) and gradB[K,N] = Aᵀ·gradC
        /// (transA=1, transB=0).
        /// </remarks>
        public void MatMulBackwardFp16Fused(
            IGpuBuffer gradCFp16, IGpuBuffer aFp16, IGpuBuffer bFp16,
            IGpuBuffer gradAOut, IGpuBuffer gradBOut,
            int m, int n, int k, bool gradOutHalf)
        {
            if (gradCFp16 is null) throw new ArgumentNullException(nameof(gradCFp16));
            if (aFp16 is null) throw new ArgumentNullException(nameof(aFp16));
            if (bFp16 is null) throw new ArgumentNullException(nameof(bFp16));
            if (gradAOut is null) throw new ArgumentNullException(nameof(gradAOut));
            if (gradBOut is null) throw new ArgumentNullException(nameof(gradBOut));
            if (m <= 0) throw new ArgumentOutOfRangeException(nameof(m), "Dimensions must be positive.");
            if (n <= 0) throw new ArgumentOutOfRangeException(nameof(n), "Dimensions must be positive.");
            if (k <= 0) throw new ArgumentOutOfRangeException(nameof(k), "Dimensions must be positive.");

            // gradA[M,K] = gradC[M,N] · Bᵀ : Mo=M, No=K, Kc=N; A=gradC (transA=0), B=Bfwd (transB=1).
            DispatchHalfGemmBackward(gradCFp16, bFp16, gradAOut, m, k, n, transA: 0, transB: 1, gradOutHalf);
            // gradB[K,N] = Aᵀ · gradC[M,N] : Mo=K, No=N, Kc=M; A=Afwd (transA=1), B=gradC (transB=0).
            DispatchHalfGemmBackward(aFp16, gradCFp16, gradBOut, k, n, m, transA: 1, transB: 0, gradOutHalf);
        }

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

        /// <summary>
        /// One dispatch of the transposed <c>gemm_fp16_backward</c> kernel: <c>C[Mo,No] = op(A)·op(B)</c> with
        /// per-operand transpose flags and a selectable FP32/FP16 output dtype. Output buffer is the caller's
        /// raw byte buffer cast inside the kernel per <paramref name="gradOutHalf"/>.
        /// </summary>
        private void DispatchHalfGemmBackward(IGpuBuffer a, IGpuBuffer b, IGpuBuffer c,
            int mo, int no, int kc, int transA, int transB, bool gradOutHalf)
        {
            if (!EnsureHalfGemmKernels() ||
                !_kernelCache.TryGetValue(HalfPrecisionGemmKernels.Fp16BackwardKernelName, out var kernel))
                throw new NotSupportedException(
                    "Half-precision GEMM kernels are not available on this OpenCL device.");

            const int ts = HalfPrecisionGemmKernels.TileSize;

            uint arg = 0;
            kernel.SetArg(arg++, ((DirectOpenClGpuBuffer)a).Buffer.Handle);
            kernel.SetArg(arg++, ((DirectOpenClGpuBuffer)b).Buffer.Handle);
            kernel.SetArg(arg++, ((DirectOpenClGpuBuffer)c).Buffer.Handle);
            kernel.SetArg(arg++, mo);
            kernel.SetArg(arg++, no);
            kernel.SetArg(arg++, kc);
            kernel.SetArg(arg++, transA);
            kernel.SetArg(arg++, transB);
            kernel.SetArg(arg++, gradOutHalf ? 1 : 0);

            int globalX = ((no + ts - 1) / ts) * ts;   // No axis (dim 0)
            int globalY = ((mo + ts - 1) / ts) * ts;   // Mo axis (dim 1)
            kernel.Execute2D(globalX, globalY, ts, ts);
        }
    // #1650/#638 FP16 conv im2col (the industry conv-as-GEMM path): fused im2col + FP32->FP16 into a [K,N] half
    // buffer (im2col_kn_fp16hw in Fp16Kernels — compiled into _kernelCache at init), paired with GemmFp16In32fOut
    // (HalfPrecisionGemmKernels). Available only when that kernel compiled on this device; else FP32 conv.
    /// <inheritdoc/>
    public bool Fp16Im2colAvailable => _kernelCache.ContainsKey("im2col_kn_fp16hw");

    /// <inheritdoc/>
    public void Im2colKNFp16(IGpuBuffer input, IGpuBuffer outputHalf,
        int batch, int channels, int height, int width,
        int kernelH, int kernelW, int strideH, int strideW, int padH, int padW, int dilationH, int dilationW)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (outputHalf is null) throw new ArgumentNullException(nameof(outputHalf));
        if (!_kernelCache.TryGetValue("im2col_kn_fp16hw", out var kernel))
            throw new NotSupportedException("FP16 im2col kernel is not available on this OpenCL device.");

        // #671 review: validate convolution parameters before deriving the work size — a zero stride/dilation
        // divides by zero, and non-positive dims/output extents would flow into Execute1D as invalid global sizes.
        if (batch <= 0) throw new ArgumentOutOfRangeException(nameof(batch), "Dimensions must be positive.");
        if (channels <= 0) throw new ArgumentOutOfRangeException(nameof(channels), "Dimensions must be positive.");
        if (height <= 0) throw new ArgumentOutOfRangeException(nameof(height), "Dimensions must be positive.");
        if (width <= 0) throw new ArgumentOutOfRangeException(nameof(width), "Dimensions must be positive.");
        if (kernelH <= 0) throw new ArgumentOutOfRangeException(nameof(kernelH), "Dimensions must be positive.");
        if (kernelW <= 0) throw new ArgumentOutOfRangeException(nameof(kernelW), "Dimensions must be positive.");
        if (strideH <= 0) throw new ArgumentOutOfRangeException(nameof(strideH), "Stride must be positive.");
        if (strideW <= 0) throw new ArgumentOutOfRangeException(nameof(strideW), "Stride must be positive.");
        if (dilationH <= 0) throw new ArgumentOutOfRangeException(nameof(dilationH), "Dilation must be positive.");
        if (dilationW <= 0) throw new ArgumentOutOfRangeException(nameof(dilationW), "Dilation must be positive.");
        if (padH < 0) throw new ArgumentOutOfRangeException(nameof(padH), "Padding must be non-negative.");
        if (padW < 0) throw new ArgumentOutOfRangeException(nameof(padW), "Padding must be non-negative.");

        int outH = (height + 2 * padH - ((kernelH - 1) * dilationH + 1)) / strideH + 1;
        int outW = (width + 2 * padW - ((kernelW - 1) * dilationW + 1)) / strideW + 1;
        if (outH <= 0 || outW <= 0)
            throw new ArgumentException(
                $"Convolution output size is non-positive (outH={outH}, outW={outW}); check kernel/stride/pad/dilation vs input size.");
        long elems = (long)batch * outH * outW * channels * kernelH * kernelW; // N*K work-items
        if (elems > int.MaxValue)
            throw new NotSupportedException($"FP16 im2col work size {elems} exceeds the 1-D dispatch limit.");

        uint arg = 0;
        kernel.SetArg(arg++, ((DirectOpenClGpuBuffer)input).Buffer.Handle);
        kernel.SetArg(arg++, ((DirectOpenClGpuBuffer)outputHalf).Buffer.Handle);
        kernel.SetArg(arg++, batch);
        kernel.SetArg(arg++, channels);
        kernel.SetArg(arg++, height);
        kernel.SetArg(arg++, width);
        kernel.SetArg(arg++, kernelH);
        kernel.SetArg(arg++, kernelW);
        kernel.SetArg(arg++, strideH);
        kernel.SetArg(arg++, strideW);
        kernel.SetArg(arg++, padH);
        kernel.SetArg(arg++, padW);
        kernel.SetArg(arg++, dilationH);
        kernel.SetArg(arg++, dilationW);
        kernel.SetArg(arg++, outH);
        kernel.SetArg(arg++, outW);

        int global = (int)elems; // Execute1D rounds up to a local multiple; the kernel guards OOB work-items
        kernel.Execute1D(global, CalculateOptimalWorkGroupSize1D(global));
    }

    }
}
