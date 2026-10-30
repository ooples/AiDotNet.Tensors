using System.Runtime.CompilerServices;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Optimization;

/// <summary>
/// Analyzes tensor lifetimes in the compiled graph and reassigns dead tensor
/// buffers to share storage with future tensors. Reduces peak memory from
/// "one allocation per op" to "one allocation per simultaneously-live set."
///
/// <para><b>Algorithm:</b></para>
/// <list type="number">
/// <item>Walk steps in execution order; for each tensor, record first-use
/// (produced) and last-use (last consumed as input).</item>
/// <item>Build a free-list of dead tensors (past their last-use). When a new
/// output buffer is needed, check if a dead tensor of the same shape exists
/// — if so, rebind it via <see cref="TensorBase{T}.RebindStorageFrom"/>.</item>
/// <item>This is a greedy "first-fit" algorithm, not optimal graph coloring.
/// It's O(n·d) where n = step count and d = max dead-pool size — fast enough
/// for compiled plans which run this pass once at compile time.</item>
/// </list>
///
/// <para><b>SD15 impact:</b> UNet forward allocates &gt;2 GB of intermediates
/// without this pass. With lifetime analysis the peak drops to ~200-300 MB
/// (the simultaneously-live set).</para>
/// </summary>
internal sealed class MemoryPlanningPass : ICpuOptimizationPass
{
    public string Name => "MemoryPlanning";
    public bool IsEnabled => true; // Always beneficial

    public CompiledStep<T>[]? TryOptimize<T>(CompiledStep<T>[] steps, IEngine engine)
        => TryOptimize(steps, engine, protectedOutput: null);

    /// <summary>
    /// Plans buffer reuse while preserving the storage selected as the compiled plan's output.
    /// An explicit output can be an internal branch rather than the final scheduled step, or a
    /// view sharing a producer's storage, so its backing storage must remain live through replay.
    /// </summary>
    internal CompiledStep<T>[]? TryOptimize<T>(
        CompiledStep<T>[] steps,
        IEngine engine,
        Tensor<T>? protectedOutput)
    {
        if (steps.Length < 4) return null; // Not worth the overhead for tiny plans

        // Phase 0: Identify which tensors were produced by a step in this
        // plan. Only those are safe to donate to the dead pool — graph
        // inputs and initializer weights must persist across repeat
        // executions of the plan. Adding them to the pool was the root
        // of the ONNX MNIST re-execute failure: on execute #1 the pool
        // rebound a later step's output to (say) Conv weight W's storage,
        // step N's Execute then clobbered W, and execute #2 fed the model
        // weights that were really leftover activation data.
        // HashSet<T>(int, IEqualityComparer<T>) is net472+; use the no-
        // capacity ctor so net471 compiles.
        var producedHere = new HashSet<Tensor<T>>(
            ReferenceEqualityComparer<Tensor<T>>.Instance);
        for (int i = 0; i < steps.Length; i++)
            producedHere.Add(steps[i].OutputBuffer);

        // Phase 1: Compute tensor lifetimes
        // Map each tensor (by reference identity) to its last-use step index.
        // Must use reference-equality semantics — two distinct tensors may have
        // the same hash code or even compare Equal by value, but they're
        // different buffers and must not share a lastUse slot. The earlier
        // implementation keyed the dictionary on RuntimeHelpers.GetHashCode(t)
        // (an int boxed to object), which collided across tensors and caused
        // the dead-pool to treat live buffers as reclaimable — data corruption
        // under any hash collision.
        var lastUse = new Dictionary<Tensor<T>, int>(
            steps.Length * 2, ReferenceEqualityComparer<Tensor<T>>.Instance);
        for (int i = 0; i < steps.Length; i++)
        {
            var step = steps[i];
            // The output buffer is "produced" at step i. Its "last use" is at
            // least i (it might also be consumed later).
            lastUse[step.OutputBuffer] = i;
            for (int j = 0; j < step.Inputs.Length; j++)
                lastUse[step.Inputs[j]] = i;
        }

        if (protectedOutput is not null)
        {
            // A pure-view output is not necessarily a step output by identity. Protect every
            // produced tensor that shares its TensorStorage so a later operation cannot recycle
            // the selected output through an alias before Execute() returns it.
            for (int i = 0; i < steps.Length; i++)
            {
                Tensor<T> produced = steps[i].OutputBuffer;
                if (ReferenceEquals(produced, protectedOutput) ||
                    ReferenceEquals(produced._storage, protectedOutput._storage))
                {
                    lastUse[produced] = steps.Length - 1;
                }
            }
        }

        // Phase 2: Greedy buffer reuse via dead-pool
        // Walk steps again. After each step, check if any tensor's lifetime
        // just ended (lastUse == current step). If so, add it to the dead
        // pool keyed by (element count, rank). When a future step needs an
        // output buffer, check the pool for a compatible dead tensor.
        // Lifetimes are per STORAGE: a view or reshape shares its base's storage, so the memory is dead only after
        // the last use of every tensor on it, and a group holding a weight or graph input (a reshaped weight) is never
        // donated. Only a tensor alone on its storage is rebound (rebinding one alias would cut it off from the
        // others), and only where a step first writes it: an in-place step's output is its target, which an earlier
        // step already wrote, and rebinding it there moved the target onto a donor's storage between the write and the
        // update, which then read the donor's stale values (measured: MultiHeadAttentionForward +
        // TensorBroadcastAddInPlace of its bias). Groups are decided before any rebind (a rebind replaces _storage).
        var groupOf = new Dictionary<Tensor<T>, object>(ReferenceEqualityComparer<Tensor<T>>.Instance);
        var groupSize = new Dictionary<object, int>(ReferenceEqualityComparer<object>.Instance);
        var groupLastUse = new Dictionary<object, int>(ReferenceEqualityComparer<object>.Instance);
        var groupHoldsExternal = new HashSet<object>(ReferenceEqualityComparer<object>.Instance);
        void Note(Tensor<T> t)
        {
            if (groupOf.ContainsKey(t)) return;
            object g = t._storage;
            groupOf[t] = g;
            groupSize[g] = (groupSize.TryGetValue(g, out var n) ? n : 0) + 1;
            if (!producedHere.Contains(t)) groupHoldsExternal.Add(g);
        }
        for (int i = 0; i < steps.Length; i++)
        {
            Note(steps[i].OutputBuffer);
            for (int j = 0; j < steps[i].Inputs.Length; j++) Note(steps[i].Inputs[j]);
        }
        foreach (var kv in lastUse)
        {
            object g = groupOf[kv.Key];
            if (!groupLastUse.TryGetValue(g, out var lu) || kv.Value > lu) groupLastUse[g] = kv.Value;
        }
        var written = new HashSet<Tensor<T>>(ReferenceEqualityComparer<Tensor<T>>.Instance);
        // A group is donated once (a tensor read twice by its last step, Multiply(g, g), put two later outputs on one buffer).
        var donatedGroups = new HashSet<object>(ReferenceEqualityComparer<object>.Instance);

        var deadPool = new Dictionary<long, List<Tensor<T>>>();
        var rebindMap = new Dictionary<Tensor<T>, Tensor<T>>(
            ReferenceEqualityComparer<Tensor<T>>.Instance);
        int reusedCount = 0;

        void DonateIfDead(Tensor<T> t, int step)
        {
            // Graph inputs and initializer weights must stay intact across repeat Execute() calls; only intermediate
            // activations are pool-eligible.
            object g = groupOf[t];
            if (groupHoldsExternal.Contains(g) || groupLastUse[g] != step || !donatedGroups.Add(g)) return;
            long k = BufferKey(t);
            if (!deadPool.TryGetValue(k, out var p))
            {
                p = new List<Tensor<T>>(4);
                deadPool[k] = p;
            }
            p.Add(t);
        }

        for (int i = 0; i < steps.Length; i++)
        {
            var step = steps[i];
            var output = step.OutputBuffer;

            // Try to reuse a dead buffer for this step's output
            long key = BufferKey(output);
            bool firstWrite = written.Add(output);
            if (firstWrite && groupSize[groupOf[output]] == 1 && deadPool.TryGetValue(key, out var pool) && pool.Count > 0)
            {
                // Rebind: the step's Execute writes to its output parameter, so the OUTPUT is rebound to share the
                // DONOR's storage: the donor's memory (dead) is overwritten and the output now lives in it. The most
                // recently freed compatible donor wins (same element count and rank can still differ in shape).
                for (int d = pool.Count - 1; d >= 0; d--)
                {
                    var donor = pool[d];
                    if (!CanRebind(output, donor)) continue;
                    pool.RemoveAt(d);
                    output.RebindStorageFromGraph(donor);
                    rebindMap[output] = donor;
                    reusedCount++;
                    break;
                }
            }

            // After this step, its inputs' and its output's storage may be dead
            for (int j = 0; j < step.Inputs.Length; j++)
                DonateIfDead(step.Inputs[j], i);
            DonateIfDead(output, i);
        }

        // Return null if no reuse was possible (don't create unnecessary array copies)
        return reusedCount > 0 ? steps : null;
    }

    /// <summary>Buffer compatibility key: element count + rank. Two tensors are
    /// buffer-compatible if they have the same total element count (same backing
    /// array size) and same rank (so the rebind shape check passes).</summary>
    private static long BufferKey<T>(Tensor<T> t)
        => ((long)t.Length << 8) | (long)t.Rank;

    private static bool CanRebind<T>(Tensor<T> target, Tensor<T> donor)
    {
        if (!target.IsContiguous || target._storageOffset != 0) return false;
        if (!donor.IsContiguous || donor._storageOffset != 0) return false;
        if (target._shape.Length != donor._shape.Length) return false;
        for (int i = 0; i < target._shape.Length; i++)
            if (target._shape[i] != donor._shape[i]) return false;
        return true;
    }

    /// <summary>Reference-equality comparer for tensor identity tracking.</summary>
    private sealed class ReferenceEqualityComparer<TItem> : IEqualityComparer<TItem> where TItem : class
    {
        public static readonly ReferenceEqualityComparer<TItem> Instance = new();
        public bool Equals(TItem? x, TItem? y) => ReferenceEquals(x, y);
        public int GetHashCode(TItem obj) => RuntimeHelpers.GetHashCode(obj);
    }
}
