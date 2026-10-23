// Copyright (c) AiDotNet. All rights reserved.
// NOTE: validated only on a Metal runner -- the dev box has no Apple GPU. The template's body runs on Vulkan here (GLSL),
// against the same fp64 reference.
using System;
using AiDotNet.Tensors.Helpers.Autotune.TunedKernels;
using static AiDotNet.Tensors.Engines.DirectGpu.Metal.MetalNativeBindings;

namespace AiDotNet.Tensors.Engines.DirectGpu.Metal;

// #1432: the fused Winograd F(2x2, 3x3) forward on Metal (WinogradFusedForwardTemplate at an 8-channel step: 32 KB of
// threadgroup memory, Apple GPUs' limit), compiled as its own library on first need. A forward-slot candidate beside the
// transform + GEMM routes, served only where it measures faster.
public sealed partial class MetalBackend
{
    private const string WinogradFusedForwardLibraryName = "WinogradFusedForward";
    private static readonly string s_winogradFusedForwardName = WinogradFusedForwardTemplate.Name(8);
    private IntPtr _winogradFusedForwardLibrary;
    private bool? _winogradFusedForwardReady;

    // The library, compiled once, and the kernel able to run its 256 threads in one threadgroup.
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
                        _winogradFusedForwardLibrary = _shaderLibrary.CompileLibrary(WinogradFusedForwardLibraryName,
                            WinogradF23KernelTemplate.EmitModule(new[] { WinogradFusedForwardTemplate.Spec(8, 16) },
                                WinogradKernelDialect.Metal));
                        ok = _winogradFusedForwardLibrary != IntPtr.Zero
                             && GetPipeline(WinogradFusedForwardLibraryName, _winogradFusedForwardLibrary,
                                 s_winogradFusedForwardName).MaxTotalThreadsPerThreadgroup >= WinogradFusedForwardTemplate.GroupSize;
                    }
                    catch (Exception ex)
                    {
                        System.Diagnostics.Trace.TraceWarning($"Metal fused Winograd forward unavailable: {ex.Message}");
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
        var buffers = new[] { a.Input, a.Kernel, a.Output, a.Bias ?? a.Output };
        var pipeline = GetPipeline(WinogradFusedForwardLibraryName, _winogradFusedForwardLibrary, s_winogradFusedForwardName);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        for (int i = 0; i < buffers.Length; i++) encoder.SetBuffer((MetalGpuBuffer)buffers[i], i);
        for (int i = 0; i < ints.Length; i++) encoder.SetBytes((uint)ints[i], (uint)(buffers.Length + i));
        encoder.DispatchThreadgroups(new MTLSize((ulong)groups, 1, 1),
            new MTLSize((ulong)WinogradFusedForwardTemplate.GroupSize, 1, 1));
        return true;
    }
}
