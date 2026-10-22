// Copyright (c) AiDotNet. All rights reserved.
using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Helpers.Autotune.TunedKernels;

namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

// #1432: Winograd F(4x4, 3x3) forward on Vulkan, through the shared WinogradF43Pipeline: the tile transforms (GLSL
// emitted from WinogradF43KernelTemplate) around BatchedGemm. A forward-slot candidate beside F(2x2, 3x3), served only
// where it measures faster.
public sealed unsafe partial class VulkanBackend : IWinogradF43Transforms
{
    private static readonly Dictionary<string, string> s_winogradF43Shaders = BuildWinogradF43Shaders();

    private static Dictionary<string, string> BuildWinogradF43Shaders()
    {
        var shaders = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (var name in WinogradF43KernelTemplate.GetKernelNames())
            shaders[name] = WinogradF43KernelTemplate.GetGlslSource(name);
        return shaders;
    }

    private bool? _winogradF43Ready;

    // Every pipeline, built once; a shader the driver rejects leaves the candidate unoffered, and says why.
    private bool WinogradF43KernelsReady
    {
        get
        {
            if (_winogradF43Ready is { } ready) return ready;
            lock (_tunedSlotsLock)
            {
                if (_winogradF43Ready is null)
                {
                    bool ok = true;
                    try
                    {
                        foreach (var spec in WinogradF43KernelTemplate.Kernels)
                            if (GetOrCreateGlslPipeline(s_winogradF43Shaders[spec.Name], spec.Buffers.Length,
                                    (uint)(spec.Ints.Length * sizeof(uint))) is null)
                            {
                                System.Diagnostics.Trace.TraceWarning(
                                    $"Vulkan Winograd F(4x4) kernels unavailable: '{spec.Name}' did not build.");
                                ok = false;
                                break;
                            }
                    }
                    catch (Exception ex)
                    {
                        System.Diagnostics.Trace.TraceWarning($"Vulkan Winograd F(4x4) kernels unavailable: {ex.Message}");
                        ok = false;
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

    private void RunWinogradF43(string name, long threads, IGpuBuffer[] buffers, params int[] ints)
    {
        var push = new uint[ints.Length];
        for (int i = 0; i < ints.Length; i++) push[i] = (uint)ints[i];
        if (!TryDispatchConvPoolGlsl(s_winogradF43Shaders[name], push, checked((int)threads), buffers))
            throw new GpuCapabilityUnavailableException($"The Vulkan '{name}' shader is unavailable.");
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
