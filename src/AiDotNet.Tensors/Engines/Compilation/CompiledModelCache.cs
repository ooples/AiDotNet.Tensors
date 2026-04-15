using System.Collections.Concurrent;
using AiDotNet.Tensors.Engines.Optimization;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Compilation;

/// <summary>
/// Caches compiled inference and training plans keyed by input shape.
/// Automatically recompiles when input dimensions change.
///
/// Thread-safe: uses ConcurrentDictionary and locking for safe concurrent access.
/// GraphMode/AutoTracer are thread-local, but this cache may be shared across threads.
///
/// Usage:
///   var cache = new CompiledModelCache&lt;float&gt;();
///   var output = cache.GetOrCompileInference(input._shape, () => model.Forward(input));
/// </summary>
public sealed class CompiledModelCache<T> : IDisposable
{
    private readonly ConcurrentDictionary<long, ICompiledPlan<T>> _inferencePlans = new();
    private readonly ConcurrentDictionary<long, ICompiledTrainingPlan<T>> _trainingPlans = new();
    // Maps shape key → the input tensor captured during tracing, so cache hits
    // can copy new data into the plan's captured tensor.
    private readonly ConcurrentDictionary<long, Tensor<T>> _capturedInputs = new();
    private readonly object _compileLock = new();
    private bool _disposed;

    /// <summary>
    /// Gets a cached inference plan for the given input shape, or compiles a new one.
    /// The forward action is traced once on cache miss and compiled into an optimized plan.
    /// </summary>
    /// <param name="inputShape">Shape of the input tensor (used as cache key).</param>
    /// <param name="forward">The forward pass to trace. Called once during compilation.</param>
    /// <returns>The compiled inference plan.</returns>
    public ICompiledPlan<T> GetOrCompileInference(int[] inputShape, Action forward)
    {
        long key = ComputeShapeKey(inputShape);
        if (_inferencePlans.TryGetValue(key, out var cached) && cached.IsValid(inputShape))
            return cached;

        lock (_compileLock)
        {
            // Double-check after acquiring lock
            if (_inferencePlans.TryGetValue(key, out cached) && cached.IsValid(inputShape))
                return cached;

            // Trace the forward pass under GraphMode and compile.
            // The forward action is called once to trace the graph — on subsequent calls
            // the caller must invoke plan.Execute() with current data in the input tensors.
            using var scope = GraphMode.Enable();
            forward();
            var plan = scope.CompileInference<T>();

            // Dispose old plan if shape changed
            if (_inferencePlans.TryGetValue(key, out var old))
                old.Dispose();

            _inferencePlans[key] = plan;
            return plan;
        }
    }

    /// <summary>
    /// Gets a cached inference plan, rebinding the input tensor on cache hit.
    /// On cache miss, the forward action is traced and the input tensor is captured.
    /// On cache hit, the new input's data is copied into the plan's captured input
    /// tensor so the compiled plan sees the current batch.
    /// </summary>
    /// <param name="input">The input tensor. Its data is copied into the plan on cache hit.</param>
    /// <param name="forward">The forward pass to trace (called once on cache miss).</param>
    /// <returns>The compiled inference plan.</returns>
    public ICompiledPlan<T> GetOrCompileInference(Tensor<T> input, Action forward)
    {
        long key = ComputeShapeKey(input._shape);
        if (_inferencePlans.TryGetValue(key, out var cached) && cached.IsValid(input._shape))
        {
            // Rebind: copy current batch data into the tensor the plan captured during tracing
            if (_capturedInputs.TryGetValue(key, out var capturedInput)
                && capturedInput.Length == input.Length)
            {
                input.AsSpan().CopyTo(capturedInput.AsWritableSpan());
            }
            return cached;
        }

        lock (_compileLock)
        {
            // Double-check after acquiring lock
            if (_inferencePlans.TryGetValue(key, out cached) && cached.IsValid(input._shape))
            {
                if (_capturedInputs.TryGetValue(key, out var ci) && ci.Length == input.Length)
                    input.AsSpan().CopyTo(ci.AsWritableSpan());
                return cached;
            }

            using var scope = GraphMode.Enable();
            forward();
            var plan = scope.CompileInference<T>();

            if (_inferencePlans.TryGetValue(key, out var old))
                old.Dispose();

            _inferencePlans[key] = plan;
            _capturedInputs[key] = input; // Track the tensor captured during tracing
            return plan;
        }
    }

    /// <summary>
    /// Gets a cached training plan for the given input shape, or compiles a new one.
    /// The forward + loss computation is traced once and compiled with backward pass.
    /// </summary>
    /// <param name="inputShape">Shape of the input tensor (used as cache key).</param>
    /// <param name="forwardAndLoss">The forward + loss computation to trace.</param>
    /// <param name="parameters">Trainable parameters for gradient computation.</param>
    /// <returns>The compiled training plan.</returns>
    public ICompiledTrainingPlan<T> GetOrCompileTraining(
        int[] inputShape, Action forwardAndLoss, Tensor<T>[] parameters)
    {
        long key = ComputeShapeKey(inputShape);
        if (_trainingPlans.TryGetValue(key, out var cached))
            return cached;

        lock (_compileLock)
        {
            // Double-check after acquiring lock
            if (_trainingPlans.TryGetValue(key, out cached))
                return cached;

            // Trace forward + loss under GraphMode and compile with backward
            using var scope = GraphMode.Enable();
            forwardAndLoss();
            var plan = scope.CompileTraining(parameters);

            // Dispose old plan if exists
            if (_trainingPlans.TryGetValue(key, out var old))
                old.Dispose();

            _trainingPlans[key] = plan;
            return plan;
        }
    }

    /// <summary>
    /// Invalidates all cached plans. Call when model structure changes
    /// (e.g., adding/removing layers, changing activation functions).
    /// </summary>
    public void Invalidate()
    {
        lock (_compileLock)
        {
            foreach (var plan in _inferencePlans.Values) plan.Dispose();
            foreach (var plan in _trainingPlans.Values) plan.Dispose();
            _inferencePlans.Clear();
            _trainingPlans.Clear();
            _capturedInputs.Clear();
        }
    }

    /// <summary>Number of cached inference plans.</summary>
    public int InferencePlanCount => _inferencePlans.Count;

    /// <summary>Number of cached training plans.</summary>
    public int TrainingPlanCount => _trainingPlans.Count;

    /// <summary>
    /// Gets a cached inference plan using symbolic shape matching.
    /// Plans compiled with one batch size can be reused for different batch sizes
    /// if the symbolic dimensions match.
    /// </summary>
    /// <param name="inputShape">Actual input shape.</param>
    /// <param name="forward">Forward pass to trace on cache miss.</param>
    /// <param name="symbolicShape">Symbolic shape with dynamic dimensions marked.</param>
    /// <returns>Compiled inference plan.</returns>
    public ICompiledPlan<T> GetOrCompileInference(int[] inputShape, Action forward, SymbolicShape symbolicShape)
    {
        // Use symbolic key (ignores dynamic dims like batch size)
        long key = symbolicShape.ComputeKey();
        if (_inferencePlans.TryGetValue(key, out var cached))
            return cached;

        lock (_compileLock)
        {
            // Double-check after acquiring lock
            if (_inferencePlans.TryGetValue(key, out cached))
                return cached;

            // Compile with the current concrete shape
            using var scope = GraphMode.Enable();
            forward();
            var plan = scope.CompileInference<T>();

            if (_inferencePlans.TryGetValue(key, out var old))
                old.Dispose();

            // Store under the symbolic key so future lookups with
            // different batch sizes still hit the cache.
            _inferencePlans[key] = plan;
            return plan;
        }
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        Invalidate();
    }

    private static long ComputeShapeKey(int[] shape)
    {
        long hash = unchecked((long)0xcbf29ce484222325L);
        hash ^= typeof(T).GetHashCode();
        hash *= unchecked((long)0x100000001b3L);
        for (int i = 0; i < shape.Length; i++)
        {
            hash ^= shape[i];
            hash *= unchecked((long)0x100000001b3L);
        }

        // Issue #164: mix in the current determinism state (process-wide value or
        // thread-local override, whichever wins) so plans compiled under one mode
        // are not served under the other. Today this matters mainly for forward
        // compatibility — if a future backend re-introduces determinism-divergent
        // kernels (GPU paths, parallel SimdGemm, etc.) the segregation is already
        // in place. The bool collapses into a single bit XORed in, then mixed.
        hash ^= BlasProvider.IsDeterministicMode ? 1L : 0L;
        hash *= unchecked((long)0x100000001b3L);

        return hash;
    }
}
