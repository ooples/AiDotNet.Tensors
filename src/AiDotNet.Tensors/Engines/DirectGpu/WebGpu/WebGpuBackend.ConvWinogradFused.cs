#if NET7_0_OR_GREATER
// Copyright (c) AiDotNet. All rights reserved.
using System;
using AiDotNet.Tensors.Helpers.Autotune.TunedKernels;

namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

// #1432: the fused Winograd F(2x2, 3x3) forward on WebGPU (WebGpuWinogradFusedKernels: a 4-channel step, so 16 KB of
// workgroup storage, WebGPU's guaranteed minimum). A forward-slot candidate beside the transform + GEMM routes, served
// only where it measures faster.
public sealed partial class WebGpuBackend
{
    private const string WinogradFusedForwardModule = "WinogradFusedForward";
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
                    bool ok = IsAvailable;
                    if (ok)
                    {
                        try
                        {
                            GetOrCreatePipelineAsync(WinogradFusedForwardModule, WebGpuWinogradFusedKernels.Source,
                                WebGpuWinogradFusedKernels.Name).GetAwaiter().GetResult();
                        }
                        catch (Exception ex)
                        {
                            System.Diagnostics.Trace.TraceWarning($"WebGPU fused Winograd forward unavailable: {ex.Message}");
                            ok = false;
                        }
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
        // The uniform struct holds the twelve ints, bit-cast (48 bytes, a 16-byte multiple).
        var uniforms = new float[ints.Length];
        for (int i = 0; i < ints.Length; i++) uniforms[i] = BitConverter.Int32BitsToSingle(ints[i]);
        // WebGPU rejects one buffer bound as both read and read-write storage, so the absent bias gets its own placeholder.
        using var placeholder = a.Bias is null ? AllocateBuffer(1) : null;
        DispatchNBufferAsync(WinogradFusedForwardModule, WebGpuWinogradFusedKernels.Source, WebGpuWinogradFusedKernels.Name,
            new[] { a.Input, a.Kernel, a.Output, a.Bias ?? placeholder! }, uniforms,
            groups * WinogradFusedForwardTemplate.GroupSize).GetAwaiter().GetResult();
        return true;
    }
}
#endif
