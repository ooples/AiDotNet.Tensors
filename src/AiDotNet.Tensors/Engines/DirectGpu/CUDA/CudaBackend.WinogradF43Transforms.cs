// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

// #1432: IWinogradF43Transforms on CUDA's own F(4x4, 3x3) transforms (CudaConvWinogradF43Kernels), which have the shared
// pipeline's layouts; the output transform takes the convolution's epilogue. CUDA's forward and gradient routes keep their
// own drivers (RunWinogradF43), with their own GEMM tiles.
public sealed partial class CudaBackend : IWinogradF43Transforms
{
    private unsafe void LaunchWinogradF43(string name, long threads, void** args)
    {
        if (!EnsureWinogradF43Module())
            throw new GpuCapabilityUnavailableException("The CUDA Winograd F(4x4) module is unavailable.");
        using var _ = PushContext();
        uint block = (uint)DefaultBlockSize;
        LaunchKernel(_kernelCache[name], (uint)((threads + block - 1) / block), block, args);
    }

    unsafe void IWinogradF43Transforms.FilterF43(IGpuBuffer w, IGpuBuffer u, int k, int c)
    {
        IntPtr wPtr = w.Handle, uPtr = u.Handle;
        void** args = stackalloc void*[4];
        args[0] = &wPtr; args[1] = &uPtr; args[2] = &k; args[3] = &c;
        LaunchWinogradF43("wino_f43_filter", (long)k * c, args);
    }

    unsafe void IWinogradF43Transforms.InputF43(IGpuBuffer x, IGpuBuffer v, int n, int c, int h, int w, int pad,
        int tilesH, int tilesW)
    {
        IntPtr xPtr = x.Handle, vPtr = v.Handle;
        void** args = stackalloc void*[9];
        args[0] = &xPtr; args[1] = &vPtr; args[2] = &n; args[3] = &c; args[4] = &h; args[5] = &w; args[6] = &pad;
        args[7] = &tilesH; args[8] = &tilesW;
        LaunchWinogradF43("wino_f43_input", (long)c * n * tilesH * tilesW, args);
    }

    unsafe void IWinogradF43Transforms.OutputF43(IGpuBuffer m, IGpuBuffer y, int n, int k, int outH, int outW,
        int tilesH, int tilesW, IGpuBuffer? bias, int epilogue)
    {
        IntPtr mPtr = m.Handle, yPtr = y.Handle, biasPtr = bias?.Handle ?? IntPtr.Zero;
        void** args = stackalloc void*[10];
        args[0] = &mPtr; args[1] = &yPtr; args[2] = &n; args[3] = &k; args[4] = &outH; args[5] = &outW;
        args[6] = &tilesH; args[7] = &tilesW; args[8] = &biasPtr; args[9] = &epilogue;
        LaunchWinogradF43("wino_f43_output", (long)k * n * tilesH * tilesW, args);
    }
}
