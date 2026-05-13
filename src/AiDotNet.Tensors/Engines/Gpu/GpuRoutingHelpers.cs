using System;
using AiDotNet.Tensors.Engines.DirectGpu;

namespace AiDotNet.Tensors.Engines.Gpu;

/// <summary>
/// High-level routing helpers that let consumers (e.g. AiDotNet's
/// <c>NeuralNetworkBase.Train</c>) decide whether to dispatch a training
/// or inference step through the GPU deferred-scope path without having
/// to inspect engine internals at every call site.
/// </summary>
/// <remarks>
/// <para>
/// Background — closes the Tensors-side capability surface for issue #334.
/// Before these helpers, every consumer that wanted to auto-route to GPU
/// had to (a) test <c>engine is DirectGpuTensorEngine</c>, (b) cast,
/// (c) call <c>GetAsyncBackend()</c> and check for null, (d) check
/// <c>SupportsDeferredExecution</c>. The result was that the canonical
/// public training entry point (<c>NeuralNetworkBase.Train</c>) just
/// always ran the CPU tape path even when <see cref="DirectGpuTensorEngine"/>
/// was active, because nothing wired the routing-decision logic.
/// </para>
/// <para>
/// These helpers answer two questions a consumer routing dispatcher needs:
/// 1) "Is this engine actually GPU-active right now?" — engine-level only,
/// not a per-op or per-layer eligibility check (the consumer owns that
/// because Tensors doesn't know about layer / loss / optimizer types).
/// 2) "How do I run my deferred GPU action with a CPU fallback in one
/// call?" — <see cref="TryExecuteOnGpu(Action{IDeferredScope}, Action, GpuExecutionOptions?, out string)"/>.
/// </para>
/// <para>
/// The per-layer / per-loss / per-optimizer eligibility checks (steps 2–4
/// of issue #334's required behavior) stay consumer-side: they require
/// knowledge of <c>IGpuLayer</c>, loss-function GPU paths, and optimizer
/// types that live in the AiDotNet repo, not in AiDotNet.Tensors.
/// </para>
/// </remarks>
public static class GpuRoutingHelpers
{
    /// <summary>
    /// Returns true when <paramref name="engine"/> is a GPU engine with a
    /// live async backend — i.e. dispatching a deferred-scope action via
    /// <see cref="DirectGpuTensorEngine.BeginDeferredScope"/> or
    /// <see cref="AsyncGpuBackendExtensions.ExecuteDeferred{TResult}"/> would
    /// actually run on hardware rather than no-op back to a CPU path.
    /// </summary>
    /// <param name="engine">The engine to inspect. May be null.</param>
    /// <returns>
    /// True iff <paramref name="engine"/> is a <see cref="DirectGpuTensorEngine"/>
    /// reporting <see cref="DirectGpuTensorEngine.SupportsDeferredExecution"/> ==
    /// true. False for null, <see cref="CpuEngine"/>, and GPU engines whose
    /// backend isn't async-capable.
    /// </returns>
    public static bool IsGpuActive(IEngine? engine)
    {
        if (engine is not DirectGpuTensorEngine gpu) return false;
        return gpu.SupportsDeferredExecution;
    }

    /// <summary>
    /// Same as <see cref="IsGpuActive(IEngine)"/> applied to
    /// <see cref="AiDotNetEngine.Current"/>. Convenience for consumer code
    /// that already routes through the process-global engine.
    /// </summary>
    public static bool IsGpuActiveOnCurrentEngine() => IsGpuActive(AiDotNetEngine.Current);

    /// <summary>
    /// Tries to obtain the <see cref="IAsyncGpuBackend"/> handle the
    /// engine routes through for deferred scopes. Returns null when the
    /// engine is CPU-only OR is a GPU engine whose underlying backend
    /// isn't async-capable.
    /// </summary>
    /// <param name="engine">The engine to inspect. May be null.</param>
    /// <returns>
    /// The backend handle, or null when no deferred-execution-capable
    /// backend is reachable from <paramref name="engine"/>.
    /// </returns>
    public static IAsyncGpuBackend? TryGetAsyncBackend(IEngine? engine)
    {
        if (engine is not DirectGpuTensorEngine gpu) return null;
        return gpu.GetAsyncBackend();
    }

    /// <summary>
    /// Convenience dispatcher: when the current engine is GPU-active,
    /// runs <paramref name="gpuAction"/> inside a deferred scope on the
    /// async backend; otherwise runs <paramref name="cpuFallback"/>
    /// without entering any deferred scope.
    /// </summary>
    /// <param name="gpuAction">The GPU-side action to execute inside an
    /// <see cref="IDeferredScope"/>. Called only when the GPU path is
    /// active.</param>
    /// <param name="cpuFallback">The CPU-side fallback action. Called
    /// when no GPU engine is active.</param>
    /// <param name="options">Optional execution options; defaults to
    /// <see cref="GpuExecutionOptions.FromEnvironment"/> with
    /// <see cref="GpuExecutionOptions.EnableGpuResidency"/> = true,
    /// matching the issue #334 required behavior.</param>
    /// <param name="reason">When the GPU path is skipped, a short
    /// human-readable explanation suitable for a debug log; null when
    /// the GPU path ran.</param>
    /// <returns>True when the GPU action ran; false when the CPU
    /// fallback ran.</returns>
    /// <remarks>
    /// <para>
    /// Engine-level eligibility only. The caller is responsible for
    /// per-layer / per-op eligibility checks before deciding to call
    /// this helper at all — if the caller knows a layer's forward op
    /// has no GPU kernel, they should call <paramref name="cpuFallback"/>
    /// directly without going through this dispatcher.
    /// </para>
    /// <para>
    /// Exceptions thrown from <paramref name="gpuAction"/> are NOT
    /// caught — the caller decides whether a runtime GPU failure should
    /// fall back to CPU or surface to the user. This helper is a
    /// routing decision, not an error-recovery wrapper.
    /// </para>
    /// </remarks>
    public static bool TryExecuteOnGpu(
        Action<IDeferredScope> gpuAction,
        Action cpuFallback,
        GpuExecutionOptions? options,
        out string? reason)
    {
        if (gpuAction is null) throw new ArgumentNullException(nameof(gpuAction));
        if (cpuFallback is null) throw new ArgumentNullException(nameof(cpuFallback));

        var engine = AiDotNetEngine.Current;
        if (engine is null)
        {
            reason = "No process engine is set; CPU path runs.";
            cpuFallback();
            return false;
        }
        if (engine is not DirectGpuTensorEngine gpu)
        {
            reason = $"Engine '{engine.GetType().Name}' is not a GPU engine; CPU path runs.";
            cpuFallback();
            return false;
        }
        var backend = gpu.GetAsyncBackend();
        if (backend is null)
        {
            reason = "DirectGpuTensorEngine has no async-capable backend; CPU path runs.";
            cpuFallback();
            return false;
        }

        // Default options preserve issue #334's required behavior:
        // EnableGpuResidency = true keeps activations + weights GPU-resident
        // across the deferred scope, avoiding the H2D/D2H round-trip that
        // dominated the VGG11 profile. EnableGraphCompilation = true (the
        // GpuExecutionOptions default) compiles the recorded graph for
        // repeated training steps.
        backend.ExecuteDeferred(gpuAction, options ?? GpuExecutionOptions.FromEnvironment());
        reason = null;
        return true;
    }

    /// <summary>
    /// Overload that drops the diagnostic <c>reason</c> for callers that
    /// don't need it. Mirrors the <c>out string?</c> overload otherwise.
    /// </summary>
    public static bool TryExecuteOnGpu(
        Action<IDeferredScope> gpuAction,
        Action cpuFallback,
        GpuExecutionOptions? options = null)
        => TryExecuteOnGpu(gpuAction, cpuFallback, options, out _);
}
