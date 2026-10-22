// Copyright (c) AiDotNet. All rights reserved.
// NOTE: validated only on a Metal runner -- the dev box has no Apple GPU. The template's bodies run on Vulkan here
// (GLSL), against the same fp64 reference.
using System;
using AiDotNet.Tensors.Helpers.Autotune.TunedKernels;

namespace AiDotNet.Tensors.Engines.DirectGpu.Metal;

// #1432: Winograd F(4x4, 3x3) forward on Metal, through the shared WinogradF43Pipeline: the tile transforms (Metal
// Shading Language emitted from WinogradF43KernelTemplate) around BatchedGemm, compiled as their own library on first
// need. A forward-slot candidate beside F(2x2, 3x3), served only where it measures faster.
public sealed partial class MetalBackend : IWinogradF43Transforms
{
    private const string WinogradF43LibraryName = "WinogradF43";
    private IntPtr _winogradF43Library;
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
                    try
                    {
                        _winogradF43Library = _shaderLibrary.CompileLibrary(WinogradF43LibraryName,
                            WinogradF43KernelTemplate.GetModuleSource(WinogradKernelDialect.Metal));
                    }
                    catch (Exception ex)
                    {
                        System.Diagnostics.Trace.TraceWarning($"Metal Winograd F(4x4) kernels unavailable: {ex.Message}");
                        _winogradF43Library = IntPtr.Zero;
                    }
                    _winogradF43Ready = _winogradF43Library != IntPtr.Zero;
                }
                return _winogradF43Ready.Value;
            }
        }
    }

    // The route itself, for the slot candidate and the tests. False when it declines the shape; nothing is launched then.
    internal bool TryConv2DForwardWinogradF43(in TunedConv2DArgs a)
        => WinogradF43KernelsReady && WinogradF43Pipeline.TryForward(this, this, a);

    private void RunWinogradF43(string name, long threads, IGpuBuffer[] buffers, params int[] ints)
    {
        if (!TryDispatchConvPoolMetal(_winogradF43Library, WinogradF43LibraryName, name, checked((int)threads), buffers, ints))
            throw new GpuCapabilityUnavailableException($"The Metal '{name}' kernel is unavailable.");
    }

    void IWinogradF43Transforms.FilterF43(IGpuBuffer w, IGpuBuffer u, int k, int c)
        => RunWinogradF43(WinogradF43KernelTemplate.Filter, (long)k * c, new[] { w, u }, k, c);

    void IWinogradF43Transforms.InputF43(IGpuBuffer x, IGpuBuffer v, int n, int c, int h, int w, int pad, int tilesH,
        int tilesW)
        => RunWinogradF43(WinogradF43KernelTemplate.Input, (long)c * n * tilesH * tilesW, new[] { x, v }, n, c, h, w, pad,
            tilesH, tilesW);

    void IWinogradF43Transforms.OutputF43(IGpuBuffer m, IGpuBuffer y, int n, int k, int outH, int outW, int tilesH,
        int tilesW, IGpuBuffer? bias, int epilogue)
        => RunWinogradF43(WinogradF43KernelTemplate.Output, (long)k * n * tilesH * tilesW, new[] { m, y, bias ?? y }, n, k,
            outH, outW, tilesH, tilesW, epilogue);
}
