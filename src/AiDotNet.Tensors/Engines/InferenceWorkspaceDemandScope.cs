using System;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.DirectGpu;

namespace AiDotNet.Tensors.Engines;

/// <summary>
/// A nondifferentiable execution memory boundary. Predictions may return incompatible dead workspace;
/// detached training passes only reuse idle buffers. Neither mode changes tape suppression, versions or execution mode.
/// </summary>
internal static class InferenceWorkspaceDemandScope
{
    [ThreadStatic] private static DirectGpuTensorEngine? t_engine;
    internal static bool BestFitIdleBuffersEnabled { get; set; } =
        Environment.GetEnvironmentVariable("AIDOTNET_INFERENCE_BEST_FIT_IDLE_BUFFERS") == "1";

    [ThreadStatic] private static long t_scope;
    [ThreadStatic] private static long t_nextScope;
    [ThreadStatic] private static bool t_returning;
    [ThreadStatic] private static bool t_allowWorkspaceReturn;

    internal readonly struct Scope : IDisposable
    {
        private readonly DirectGpuTensorEngine? _previous;
        private readonly long _previousScope;
        private readonly long _scope;
        private readonly int _thread;
        private readonly bool _previousAllowReturn;

        internal Scope(DirectGpuTensorEngine? previous, long previousScope, long scope, bool previousAllowReturn)
        {
            _previous = previous;
            _previousScope = previousScope;
            _scope = scope;
            _thread = Environment.CurrentManagedThreadId;
            _previousAllowReturn = previousAllowReturn;
        }

        public void Dispose()
        {
            if (_scope == 0) return;
            if (_thread != Environment.CurrentManagedThreadId)
                throw new InvalidOperationException("An inference memory scope must end on its execution thread.");
            if (t_scope != _scope) return;
            t_engine = _previous;
            t_scope = _previousScope;
            t_allowWorkspaceReturn = _previousAllowReturn;
        }
    }

    internal static Scope Enter(IEngine? engine)
        => EnterCore(engine, allowWorkspaceReturn: true);

    /// <summary>Borrow compatible idle buffers while preserving workspace for the next training step.</summary>
    internal static Scope EnterIdleBufferReuse(IEngine? engine)
        => EnterCore(engine, allowWorkspaceReturn: false);

    private static Scope EnterCore(IEngine? engine, bool allowWorkspaceReturn)
    {
        var previous = t_engine;
        long previousScope = t_scope;
        bool previousAllowReturn = t_allowWorkspaceReturn;
        long next = ++t_nextScope;
        t_engine = GraphMode.IsActive ? null : engine?.As<DirectGpuTensorEngine>();
        t_scope = next;
        t_allowWorkspaceReturn = allowWorkspaceReturn;
        return new Scope(previous, previousScope, next, previousAllowReturn);
    }

    /// <summary>Called before a backend takes its native/context allocation locks.</summary>
    internal static void BeforeAllocation(IDirectGpuBackend backend, int floatElements)
    {
        if (!t_allowWorkspaceReturn || t_engine is not { } engine || t_returning || floatElements <= 0) return;
        try
        {
            t_returning = true;
            engine.ReturnIncompatibleInferenceWorkspace(backend, checked((long)floatElements * sizeof(float)));
        }
        finally { t_returning = false; }
    }

    /// <summary>A real cache miss may change the working size class; this never touches a live plan allocation.</summary>
    internal static bool TryReturnIncompatibleIdleCache<TBuffer>(IDirectGpuBackend backend,
        GpuBufferPool<TBuffer> pool, int elements, GpuBufferPoolAffinity affinity)
        where TBuffer : class, IGpuBuffer, IPoolableGpuBuffer
    {
        if (!t_allowWorkspaceReturn || t_engine is not { } engine || t_returning || elements <= 0) return false;
        long scope = t_scope;
        try
        {
            t_returning = true;
            if (!engine.IsInferenceIdleCacheDemandEligible(backend)
                || t_scope != scope || !ReferenceEquals(t_engine, engine)) return false;
            return pool.ReturnIncompatibleIdleForLargerRequest(elements, affinity) > 0;
        }
        finally { t_returning = false; }
    }

    internal static bool TryRentLargerIdleBuffer<TBuffer>(IDirectGpuBackend backend,
        GpuBufferPool<TBuffer> pool, int elements, GpuBufferPoolAffinity affinity, out TBuffer? buffer)
        where TBuffer : class, IGpuBuffer, IPoolableGpuBuffer
    {
        buffer = null;
        if (!BestFitIdleBuffersEnabled || t_engine is not { } engine || t_returning || elements <= 0) return false;
        long scope = t_scope;
        try
        {
            t_returning = true;
            if (!engine.IsInferenceIdleCacheDemandEligible(backend)
                || t_scope != scope || !ReferenceEquals(t_engine, engine)) return false;
            return pool.TryRentLargerIdleBuffer(elements, affinity, out buffer);
        }
        finally { t_returning = false; }
    }
}
