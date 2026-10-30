#if NET7_0_OR_GREATER
// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;
using AiDotNet.Tensors.Helpers.Autotune.TunedKernels;

namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

// #1432: Winograd F(4x4, 3x3) forward on WebGPU, through the shared WinogradF43Pipeline: the WGSL tile transforms of
// WebGpuWinogradF43Kernels around BatchedGemm. A forward-slot candidate beside F(2x2, 3x3), served only where it
// measures faster.
public sealed partial class WebGpuBackend : IWinogradF43Transforms
{
    private const string WinogradF43ModulePrefix = "WinogradF43_";
    private bool? _winogradF43Ready;

    private bool WinogradF43KernelsReady
    {
        get
        {
            if (_winogradF43Ready is { } ready) return ready;
            lock (_tunedSlotsLock)
            {
                if (_winogradF43Ready is null)
                {
                    bool ok = IsAvailable;
                    if (ok)
                    {
                        try
                        {
                            foreach (var name in WinogradF43KernelTemplate.GetKernelNames())
                                GetOrCreatePipelineAsync(WinogradF43ModulePrefix + name, WebGpuWinogradF43Kernels.SourceFor(name),
                                    name).GetAwaiter().GetResult();
                        }
                        catch (Exception ex)
                        {
                            System.Diagnostics.Trace.TraceWarning($"WebGPU Winograd F(4x4) kernels unavailable: {ex.Message}");
                            ok = false;
                        }
                    }
                    _winogradF43Ready = ok;
                }
                return _winogradF43Ready.Value;
            }
        }
    }

    // The route itself, for the slot candidate and the tests. False when it declines the shape; nothing is launched then.
    internal bool TryConv2DForwardWinogradF43(in TunedConv2DArgs a)
        => WinogradF43KernelsReady && WinogradF43Pipeline.TryForward(this, this, a);

    // A uniform struct is a 16-byte multiple: the ints, bit-cast, then zero padding.
    private static float[] WinogradF43Uniforms(params int[] ints)
    {
        var uniforms = new float[(ints.Length + 3) / 4 * 4];
        for (int i = 0; i < ints.Length; i++) uniforms[i] = BitConverter.Int32BitsToSingle(ints[i]);
        return uniforms;
    }

    void IWinogradF43Transforms.FilterF43(IGpuBuffer w, IGpuBuffer u, int k, int c)
        => Dispatch2BufferAsync(WinogradF43ModulePrefix + WinogradF43KernelTemplate.Filter, WebGpuWinogradF43Kernels.Filter,
            WinogradF43KernelTemplate.Filter, w, u, WinogradF43Uniforms(k, c), checked(k * c)).GetAwaiter().GetResult();

    void IWinogradF43Transforms.InputF43(IGpuBuffer x, IGpuBuffer v, int n, int c, int h, int w, int pad, int tilesH,
        int tilesW)
        => Dispatch2BufferAsync(WinogradF43ModulePrefix + WinogradF43KernelTemplate.Input, WebGpuWinogradF43Kernels.Input,
            WinogradF43KernelTemplate.Input, x, v, WinogradF43Uniforms(n, c, h, w, pad, tilesH, tilesW),
            checked((int)((long)c * n * tilesH * tilesW))).GetAwaiter().GetResult();

    void IWinogradF43Transforms.OutputF43(IGpuBuffer m, IGpuBuffer y, int n, int k, int outH, int outW, int tilesH,
        int tilesW, IGpuBuffer? bias, int epilogue)
    {
        // WebGPU rejects one buffer bound as both read and read-write storage, so the absent bias gets its own placeholder.
        using var placeholder = bias is null ? AllocateBuffer(1) : null;
        Dispatch3BufferAsync(WinogradF43ModulePrefix + WinogradF43KernelTemplate.Output, WebGpuWinogradF43Kernels.Output,
            WinogradF43KernelTemplate.Output, m, y, bias ?? placeholder ?? y,
            WinogradF43Uniforms(n, k, outH, outW, tilesH, tilesW, epilogue),
            checked((int)((long)k * n * tilesH * tilesW))).GetAwaiter().GetResult();
    }
}
#endif
