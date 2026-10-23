// Copyright (c) AiDotNet. All rights reserved.
using System;
using AiDotNet.Tensors.Helpers.Autotune.TunedKernels;

namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

// #1432: the fused Winograd F(2x2, 3x3) forward on Vulkan (WinogradFusedForwardTemplate at an 8-channel step, 32 KB of
// shared memory). A forward-slot candidate beside the transform + GEMM routes, served only where it measures faster.
public sealed unsafe partial class VulkanBackend
{
    private static readonly string s_winogradFusedForwardShader =
        WinogradF23KernelTemplate.EmitGlsl(WinogradFusedForwardTemplate.Spec(8, 16));

    private bool? _winogradFusedForwardReady;

    private bool WinogradFusedForwardReady
    {
        get
        {
            if (_winogradFusedForwardReady is { } ready) return ready;
            lock (_tunedSlotsLock)
            {
                if (_winogradFusedForwardReady is null)
                {
                    bool ok;
                    try
                    {
                        ok = GetOrCreateGlslPipeline(s_winogradFusedForwardShader, 4, 12 * sizeof(uint)) is not null;
                        if (!ok) System.Diagnostics.Trace.TraceWarning("Vulkan fused Winograd forward unavailable: the shader did not build.");
                    }
                    catch (Exception ex)
                    {
                        System.Diagnostics.Trace.TraceWarning($"Vulkan fused Winograd forward unavailable: {ex.Message}");
                        ok = false;
                    }
                    _winogradFusedForwardReady = ok;
                }
                return _winogradFusedForwardReady.Value;
            }
        }
    }

    // The route itself, for the slot candidate and the tests. False when it declines the shape; nothing is launched then.
    internal bool TryConv2DForwardWinogradFused(in TunedConv2DArgs a)
    {
        if (!WinogradFusedForwardReady || !WinogradFusedForwardTemplate.TryPlan(a, out int groups, out int[] ints)) return false;
        var push = new uint[ints.Length];
        for (int i = 0; i < ints.Length; i++) push[i] = unchecked((uint)ints[i]);
        // The shader's local size is the group size, so the thread count dispatches exactly `groups` workgroups.
        if (!TryDispatchConvPoolGlsl(s_winogradFusedForwardShader, push, groups * WinogradFusedForwardTemplate.GroupSize,
                a.Input, a.Kernel, a.Output, a.Bias ?? a.Output))
            throw new GpuCapabilityUnavailableException("The Vulkan fused Winograd forward shader is unavailable.");
        return true;
    }
}
