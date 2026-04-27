// Copyright (c) AiDotNet. All rights reserved.
// Backend dispatch wiring (Phase C.5) — picks the best available
// emitter for a given graph + dtype and produces a runnable kernel.

using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Compilation.Codegen.AvxCs;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;

namespace AiDotNet.Tensors.Engines.Compilation.Codegen;

/// <summary>
/// Central entry point for backend selection across the codegen
/// emitters. Tries CPU emitters in priority order (AVX-512 first
/// where supported, falling back to the LambdaExpression JIT) and
/// returns the first one that successfully emits a kernel.
/// </summary>
/// <remarks>
/// <para><b>Why a dispatcher rather than direct emitter use:</b></para>
/// <para>
/// Production fusion passes shouldn't hard-code which emitter to use
/// — the right answer depends on the running CPU's feature flags
/// (AVX-512 availability) and the graph's content (transcendentals
/// force the JIT path because the AVX-512 emitter declines them).
/// The dispatcher centralises that decision so passes consuming
/// codegen output don't repeat the priority logic.
/// </para>
/// <para><b>Order:</b></para>
/// <list type="number">
/// <item><see cref="CpuAvx512Emitter"/> — guaranteed vectorisation
/// when available, declines transcendentals.</item>
/// <item><see cref="CpuDotNetJitEmitter"/> — RyuJIT-vectorised
/// fallback that handles every supported pointwise op including
/// Math.Exp/Tanh/Sigmoid via scalar calls.</item>
/// </list>
/// GPU emitters (Triton/HIP/MSL/WGSL/GLSL) produce source only —
/// runtime compile-and-launch wiring lives in each backend's
/// <c>IDirectGpuBackend</c> implementation and is dispatched from
/// there, not here.
/// </remarks>
public static class CodegenDispatcher
{
    private static readonly List<IKernelEmitter> _cpuEmitters = new()
    {
        new CpuAvx512Emitter(),
        new CpuDotNetJitEmitter(),
    };

    /// <summary>
    /// Attempts to emit a kernel for <paramref name="graph"/> using
    /// the best available CPU emitter. Returns null if every emitter
    /// declined.
    /// </summary>
    /// <param name="graph">The graph to fuse.</param>
    /// <param name="dtype">Output element type.</param>
    /// <param name="declineReasons">Outputs the per-emitter decline
    /// reasons in the order they were tried — useful for telemetry
    /// and debugging the fusion-pass decision.</param>
    /// <returns>A compiled kernel, or null if no emitter succeeded.</returns>
    public static CodegenKernel? TryEmitCpu(
        CodegenGraph graph,
        CodegenElementType dtype,
        out IReadOnlyList<(CodegenTarget Target, string Reason)> declineReasons)
    {
        if (graph is null) throw new ArgumentNullException(nameof(graph));

        var declines = new List<(CodegenTarget, string)>();
        foreach (var emitter in _cpuEmitters)
        {
            var result = emitter.Emit(graph, dtype);
            if (!result.Declined && result.Kernel is not null)
            {
                declineReasons = declines;
                return result.Kernel;
            }
            declines.Add((emitter.Target, result.DeclineReason ?? "(no reason given)"));
        }
        declineReasons = declines;
        return null;
    }

    /// <summary>
    /// Convenience overload that drops the decline-reason output.
    /// </summary>
    public static CodegenKernel? TryEmitCpu(CodegenGraph graph, CodegenElementType dtype)
        => TryEmitCpu(graph, dtype, out _);

    /// <summary>
    /// Returns the list of CPU emitters in dispatch priority order.
    /// Exposed for tests that need to inspect the configured
    /// dispatcher chain.
    /// </summary>
    internal static IReadOnlyList<IKernelEmitter> CpuEmitters => _cpuEmitters;
}
