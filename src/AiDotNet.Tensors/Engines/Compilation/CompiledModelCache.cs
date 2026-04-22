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
    /// The forward function is traced once on cache miss and compiled into an optimized plan.
    /// </summary>
    /// <param name="inputShape">Shape of the input tensor (used as cache key).</param>
    /// <param name="forward">
    /// The forward pass to trace. Must return the tensor the caller intends as
    /// the model's output. The returned tensor is what <c>plan.Execute()</c>
    /// will yield on every replay — the compile step uses it as the explicit
    /// final output, so a forward ending in a pure-view op
    /// (<c>Reshape</c> on contiguous data, <c>Squeeze</c>, ...) or with
    /// host-side control flow behaves correctly (issue #228).
    /// </param>
    /// <returns>The compiled inference plan.</returns>
    public ICompiledPlan<T> GetOrCompileInference(int[] inputShape, Func<Tensor<T>> forward)
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
            // The forward function is called once to trace the graph — on subsequent calls
            // the caller must invoke plan.Execute() with current data in the input tensors.
            using var scope = GraphMode.Enable();
            var explicitOutput = forward();
            ThrowIfForwardRecordedNothing(scope, explicitOutput);
            var plan = scope.CompileInference<T>(explicitOutput);

            // Dispose old plan if shape changed
            if (_inferencePlans.TryGetValue(key, out var old))
                old.Dispose();

            _inferencePlans[key] = plan;
            return plan;
        }
    }

    /// <summary>
    /// Gets a cached inference plan, rebinding the input tensor on cache hit.
    /// On cache miss, the forward function is traced and the input tensor is captured.
    /// On cache hit, the new input's data is copied into the plan's captured input
    /// tensor so the compiled plan sees the current batch.
    /// </summary>
    /// <param name="input">The input tensor. Its data is copied into the plan on cache hit.</param>
    /// <param name="forward">
    /// The forward pass to trace (called once on cache miss). Must return the
    /// tensor the caller intends as the model's output — see the other
    /// overload for details on issue #228.
    /// </param>
    /// <returns>The compiled inference plan.</returns>
    public ICompiledPlan<T> GetOrCompileInference(Tensor<T> input, Func<Tensor<T>> forward)
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
            var explicitOutput = forward();
            ThrowIfForwardRecordedNothing(scope, explicitOutput);
            var plan = scope.CompileInference<T>(explicitOutput);

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
    /// <param name="forwardAndLoss">
    /// The forward + loss computation to trace. Must return the loss tensor
    /// that backward will be seeded with — the plan uses it as the explicit
    /// loss output, so a loss computation ending in a view op (<c>Reshape</c>
    /// to scalarize, <c>Squeeze</c>) behaves correctly (issue #228).
    /// </param>
    /// <param name="parameters">Trainable parameters for gradient computation.</param>
    /// <returns>The compiled training plan.</returns>
    public ICompiledTrainingPlan<T> GetOrCompileTraining(
        int[] inputShape, Func<Tensor<T>> forwardAndLoss, Tensor<T>[] parameters)
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
            var explicitLoss = forwardAndLoss();
            ThrowIfForwardRecordedNothing(scope, explicitLoss);
            var plan = scope.CompileTraining(parameters, explicitLoss);

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
    public ICompiledPlan<T> GetOrCompileInference(int[] inputShape, Func<Tensor<T>> forward, SymbolicShape symbolicShape)
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
            var explicitOutput = forward();
            ThrowIfForwardRecordedNothing(scope, explicitOutput);
            var plan = scope.CompileInference<T>(explicitOutput);

            if (_inferencePlans.TryGetValue(key, out var old))
                old.Dispose();

            // Store under the symbolic key so future lookups with
            // different batch sizes still hit the cache.
            _inferencePlans[key] = plan;
            return plan;
        }
    }

    /// <summary>
    /// Issue #239 guard: surface a clear, actionable error when the traced
    /// forward lambda doesn't record any tensor operations. A zero-op plan
    /// with zero captured inputs is always a mistake — <see cref="ICompiledPlan{T}.SetInputs"/>
    /// cannot wire user data into a graph that has no nodes, and the
    /// resulting <c>ArgumentException</c> ("This plan was compiled with 0
    /// captured input(s)") gives the user no hint about WHY the plan is
    /// empty. The most common cause: the forward body allocates a fresh
    /// tensor with <c>new Tensor&lt;T&gt;(...)</c> and fills it via indexer
    /// writes, both of which bypass the lazy-graph tracer entirely.
    /// </summary>
    private static void ThrowIfForwardRecordedNothing<TElem>(LazyTensorScope scope, Tensor<TElem>? explicitOutput)
    {
        if (scope.NodeCount > 0)
            return;
        // Allow a truly trivial "return constant" forward only when the
        // caller explicitly returned null — in practice no caller does, but
        // we match the existing CompileInference contract.
        if (explicitOutput is null)
            return;
        throw new ArgumentException(
            "The forward lambda did not record any tensor operations, so the compiled plan " +
            "has zero steps and cannot accept inputs. This typically happens when the forward " +
            "uses `new Tensor<T>(...)` and writes values through the indexer — both bypass the " +
            "lazy-graph tracer. Use engine operations on the input tensor (TensorAdd, " +
            "TensorMultiply, MatMul, activation methods, etc.) so the tracer captures the " +
            "computation graph. See issue #239.",
            "forward");
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
