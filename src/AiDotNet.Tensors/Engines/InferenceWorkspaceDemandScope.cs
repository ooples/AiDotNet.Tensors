using System;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.DirectGpu;

namespace AiDotNet.Tensors.Engines;

/// <summary>
/// A memory-demand boundary for a public, nondifferentiable prediction. This does not change tape suppression,
/// tensor versions or execution mode. The model owns the exact engine; nested predictions restore their caller.
/// </summary>
internal static class InferenceWorkspaceDemandScope
{
    [ThreadStatic] private static DirectGpuTensorEngine? t_engine;
    [ThreadStatic] private static long t_scope;
    [ThreadStatic] private static long t_nextScope;
    [ThreadStatic] private static bool t_returning;

    internal readonly struct Scope : IDisposable
    {
        private readonly DirectGpuTensorEngine? _previous;
        private readonly long _previousScope;
        private readonly long _scope;
        private readonly int _thread;

        internal Scope(DirectGpuTensorEngine? previous, long previousScope, long scope)
        {
            _previous = previous;
            _previousScope = previousScope;
            _scope = scope;
            _thread = Environment.CurrentManagedThreadId;
        }

        public void Dispose()
        {
            if (_scope == 0) return;
            if (_thread != Environment.CurrentManagedThreadId)
                throw new InvalidOperationException("An inference memory scope must end on its execution thread.");
            if (t_scope != _scope) return;
            t_engine = _previous;
            t_scope = _previousScope;
        }
    }

    internal static Scope Enter(IEngine? engine)
    {
        var previous = t_engine;
        long previousScope = t_scope;
        long next = ++t_nextScope;
        t_engine = GraphMode.IsActive ? null : engine?.As<DirectGpuTensorEngine>();
        t_scope = next;
        return new Scope(previous, previousScope, next);
    }

    /// <summary>Called before a backend takes its native/context allocation locks.</summary>
    internal static void BeforeAllocation(IDirectGpuBackend backend, int floatElements)
    {
        if (t_engine is not { } engine || t_returning || floatElements <= 0) return;
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
        if (t_engine is not { } engine || t_returning || elements <= 0) return false;
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
}
