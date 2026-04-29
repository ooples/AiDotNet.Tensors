// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Collections.Concurrent;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

/// <summary>
/// CUDA dispatcher for 2:4 structured sparse · dense matmul. Compiles
/// the kernel module from <see cref="SparseSemiStructuredCudaKernels"/>
/// once per (architecture, kernel-source) pair, caches the loaded
/// module on the static side, and routes calls to the
/// <c>mma.sp</c>-aware variant when the device's compute capability
/// is ≥ 8.0 (Ampere); otherwise to the baseline kernel. The managed
/// reference path in <see cref="LinearAlgebra.Sparse.SparseSemiStructured{T}"/>
/// is the correctness oracle.
/// </summary>
internal static class SparseSemiStructuredGpuDispatch
{
    private static readonly ConcurrentDictionary<int, IntPtr> _moduleCache = new();
    private static readonly ConcurrentDictionary<int, IntPtr> _functionCache = new();

    /// <summary>Whether a CUDA backend is reachable for 2:4 dispatch.</summary>
    public static bool IsAvailable => CudaBackend.IsCudaAvailable;

    /// <summary>
    /// Runs <c>output = A_packed{2:4} · B</c> on the GPU. The dispatch
    /// chooses the Ampere <c>mma.sp</c> kernel for sm_80+ and the
    /// portable baseline kernel for sm_60-sm_75. Both produce
    /// identical results; the only difference is the speed of the
    /// inner reduction.
    /// </summary>
    public static float[] MatMul(
        float[] packedValues, byte[] metadata,
        float[] b, int rows, int cols, int n)
    {
        if (!IsAvailable)
            throw new InvalidOperationException("CUDA backend is not available for 2:4 sparse dispatch.");

        var backend = CudaBackend.CreateOrThrow();
        var packedBuf = backend.AllocateBuffer(packedValues);
        var bBuf = backend.AllocateBuffer(b);
        var output = new float[rows * n];
        var outBuf = backend.AllocateBuffer(output);
        var metaBuf = backend.AllocateByteBuffer(metadata.Length);
        UploadBytes(metaBuf.Handle, metadata);

        // Block tile: 16 cols × 16 rows is friendly to memory coalescing
        // for the baseline kernel and to the warp-cooperative tile shape
        // for the Ampere kernel.
        const int blockX = 16, blockY = 16;
        int gridX = (n + blockX - 1) / blockX;
        int gridY = (rows + blockY - 1) / blockY;

        // Pick variant by compute capability — 8.0+ gets the mma.sp kernel.
        int variant = ComputeCapabilityMajor() >= 8 ? 1 : 0;
        IntPtr fn = GetOrCompileKernel(variant);

        // Marshal kernel arguments — pinned IntPtr array of pointers.
        unsafe
        {
            IntPtr aPtr = packedBuf.Handle;
            IntPtr metaPtr = metaBuf.Handle;
            IntPtr bPtr = bBuf.Handle;
            IntPtr outPtr = outBuf.Handle;
            void*[] args = new void*[]
            {
                &aPtr, &metaPtr, &bPtr, &outPtr,
                &rows, &cols, &n,
            };
            fixed (void** pArgs = args)
            {
                var status = CudaNativeBindings.cuLaunchKernel(
                    fn,
                    (uint)gridX, (uint)gridY, 1u,
                    (uint)blockX, (uint)blockY, 1u,
                    sharedMemBytes: 0u,
                    stream: IntPtr.Zero,
                    kernelParams: (IntPtr)pArgs,
                    extra: IntPtr.Zero);
                CuBlasNative.CheckCudaResult(status, "cuLaunchKernel(sparse_2_4_matmul)");
            }
        }

        backend.DownloadBuffer(outBuf, output);
        outBuf.Dispose();
        bBuf.Dispose();
        metaBuf.Dispose();
        packedBuf.Dispose();
        return output;
    }

    private static IntPtr GetOrCompileKernel(int variant)
    {
        if (_functionCache.TryGetValue(variant, out var cached)) return cached;

        // Compile through the same NVRTC path the rest of the engine uses.
        // The CudaBackend exposes a reusable kernel-module compiler; we
        // mirror its source-prep pipeline (the public surface doesn't yet
        // export a compile-arbitrary-source helper, so we go through the
        // direct CUDA driver API for module load + function lookup).
        string source = variant == 1
            ? SparseSemiStructuredCudaKernels.AmpereMmaSpSource
            : SparseSemiStructuredCudaKernels.BaselineSource;
        string entry = SparseSemiStructuredCudaKernels.EntryPoints[variant];

        // Reuse NVRTC bindings — same flow CompileKernelModule exercises.
        IntPtr program = IntPtr.Zero;
        var result = NvrtcNativeBindings.nvrtcCreateProgram(
            ref program, source, $"sparse_2_4_matmul_{variant}.cu",
            0, IntPtr.Zero, IntPtr.Zero);
        if (result != NvrtcResult.Success)
            throw new InvalidOperationException($"nvrtcCreateProgram failed: {result}");

        try
        {
            string arch = $"--gpu-architecture=sm_{ComputeCapabilityMajor()}{ComputeCapabilityMinor()}";
            string[] options = new[] { arch };
            var compileResult = NvrtcNativeBindings.nvrtcCompileProgram(program, options.Length, options);
            if (compileResult != NvrtcResult.Success)
                throw new InvalidOperationException($"nvrtcCompileProgram failed: {compileResult}");

            NvrtcNativeBindings.nvrtcGetPTXSize(program, out var ptxSize);
            IntPtr ptxBuffer = System.Runtime.InteropServices.Marshal.AllocHGlobal((int)ptxSize);
            try
            {
                NvrtcNativeBindings.nvrtcGetPTX(program, ptxBuffer);
                var loadResult = CudaNativeBindings.cuModuleLoadData(out IntPtr module, ptxBuffer);
                CuBlasNative.CheckCudaResult(loadResult, "cuModuleLoadData(sparse_2_4)");
                _moduleCache[variant] = module;
            }
            finally
            {
                System.Runtime.InteropServices.Marshal.FreeHGlobal(ptxBuffer);
            }
            var moduleHandle = _moduleCache[variant];
            var fnResult = CudaNativeBindings.cuModuleGetFunction(out IntPtr fn, moduleHandle, entry);
            CuBlasNative.CheckCudaResult(fnResult, $"cuModuleGetFunction({entry})");
            _functionCache[variant] = fn;
            return fn;
        }
        finally
        {
            if (program != IntPtr.Zero) NvrtcNativeBindings.nvrtcDestroyProgram(ref program);
        }
    }

    private static int ComputeCapabilityMajor()
    {
        try
        {
            CuBlasNative.cuDeviceGetAttribute(out int major, (int)CudaDeviceAttribute.ComputeCapabilityMajor, 0);
            return major > 0 ? major : 7;
        }
        catch { return 7; }
    }

    private static int ComputeCapabilityMinor()
    {
        try
        {
            CuBlasNative.cuDeviceGetAttribute(out int minor, (int)CudaDeviceAttribute.ComputeCapabilityMinor, 0);
            return minor;
        }
        catch { return 0; }
    }

    private static unsafe void UploadBytes(IntPtr device, byte[] host)
    {
        ulong byteCount = (ulong)host.Length;
        fixed (byte* src = host)
        {
            var status = CuBlasNative.cuMemcpyHtoD(device, (IntPtr)src, byteCount);
            CuBlasNative.CheckCudaResult(status, "cuMemcpyHtoD(byte[])");
        }
    }
}
