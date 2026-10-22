// Copyright (c) AiDotNet. All rights reserved.
// NOTE: validated by an NVRTC compile of the module only -- the dev box has no AMD GPU. The template's bodies run on
// Vulkan here (GLSL), against the same fp64 reference.
using System;
using AiDotNet.Tensors.Helpers.Autotune.TunedKernels;

namespace AiDotNet.Tensors.Engines.DirectGpu.HIP;

// #1432: Winograd F(4x4, 3x3) forward on HIP, through the shared WinogradF43Pipeline: the tile transforms (emitted from
// WinogradF43KernelTemplate) around BatchedGemm. A forward-slot candidate beside F(2x2, 3x3), served only where it
// measures faster.
public sealed partial class HipBackend : IWinogradF43Transforms
{
    private IntPtr _winogradF43Module;

    private bool WinogradF43KernelsReady => HasTunedKernel(WinogradF43KernelTemplate.Filter)
        && HasTunedKernel(WinogradF43KernelTemplate.Input) && HasTunedKernel(WinogradF43KernelTemplate.Output);

    // Strict math, as the F(2x2) transforms: contracting or reassociating the coefficients would move the result away
    // from the other backends' instances of the same template.
    private void CompileWinogradF43Kernels()
    {
        var failure = CompileKernelModule(WinogradF43KernelTemplate.GetModuleSource(WinogradKernelDialect.Hip),
            "winograd_f43", ref _winogradF43Module, WinogradF43KernelTemplate.GetKernelNames(), useFastMath: false);
        if (failure is not null)
            System.Diagnostics.Trace.TraceWarning($"HIP Winograd F(4x4) kernels unavailable: {failure.Message}");
    }

    // The route itself, for the slot candidate and the tests. False when it declines the shape; nothing is launched then.
    internal bool TryConv2DForwardWinogradF43(in TunedConv2DArgs a)
        => WinogradF43KernelsReady && WinogradF43Pipeline.TryForward(this, this, a);

    private unsafe void LaunchWinogradF43(string name, long threads, IGpuBuffer[] buffers, int[] ints)
    {
        var handles = new IntPtr[buffers.Length];
        for (int i = 0; i < buffers.Length; i++) handles[i] = buffers[i].Handle;
        fixed (IntPtr* hp = handles)
        fixed (int* ip = ints)
        {
            void** args = stackalloc void*[buffers.Length + ints.Length];
            for (int i = 0; i < buffers.Length; i++) args[i] = hp + i;
            for (int i = 0; i < ints.Length; i++) args[buffers.Length + i] = ip + i;
            LaunchKernel(TunedKernel(name), (uint)((threads + 255) / 256), 256, args);
        }
    }

    void IWinogradF43Transforms.FilterF43(IGpuBuffer w, IGpuBuffer u, int k, int c)
        => LaunchWinogradF43(WinogradF43KernelTemplate.Filter, (long)k * c, new[] { w, u }, new[] { k, c });

    void IWinogradF43Transforms.InputF43(IGpuBuffer x, IGpuBuffer v, int n, int c, int h, int w, int pad, int tilesH,
        int tilesW)
        => LaunchWinogradF43(WinogradF43KernelTemplate.Input, (long)c * n * tilesH * tilesW, new[] { x, v },
            new[] { n, c, h, w, pad, tilesH, tilesW });

    void IWinogradF43Transforms.OutputF43(IGpuBuffer m, IGpuBuffer y, int n, int k, int outH, int outW, int tilesH,
        int tilesW, IGpuBuffer? bias, int epilogue)
        => LaunchWinogradF43(WinogradF43KernelTemplate.Output, (long)k * n * tilesH * tilesW, new[] { m, y, bias ?? y },
            new[] { n, k, outH, outW, tilesH, tilesW, epilogue });
}
