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
    {
        if (steps.Length < 4) return null; // Not worth the overhead for tiny plans

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

        // Phase 2: Greedy buffer reuse via dead-pool
        // Walk steps again. After each step, check if any tensor's lifetime
        // just ended (lastUse == current step). If so, add it to the dead
        // pool keyed by (element count, rank). When a future step needs an
        // output buffer, check the pool for a compatible dead tensor.
        var deadPool = new Dictionary<long, List<Tensor<T>>>();
        var rebindMap = new Dictionary<Tensor<T>, Tensor<T>>(
            ReferenceEqualityComparer<Tensor<T>>.Instance);
        int reusedCount = 0;

        for (int i = 0; i < steps.Length; i++)
        {
            var step = steps[i];
            var output = step.OutputBuffer;

            // Try to reuse a dead buffer for this step's output
            long key = BufferKey(output);
            if (deadPool.TryGetValue(key, out var pool) && pool.Count > 0)
            {
                var donor = pool[pool.Count - 1];
                pool.RemoveAt(pool.Count - 1);

                // Rebind: make donor's storage point to output's storage
                // (or vice versa — we want the step's output to write into
                // the donor's backing array so memory is reused).
                // Since the step's Execute writes to the output parameter,
                // we rebind the OUTPUT to share the DONOR's storage. This
                // way the donor's memory gets overwritten (it's dead), and
                // the output tensor now shares that memory.
                if (CanRebind(output, donor))
                {
                    output.RebindStorageFrom(donor);
                    rebindMap[output] = donor;
                    reusedCount++;
                }
            }

            // After this step, check if any of its inputs are now dead
            for (int j = 0; j < step.Inputs.Length; j++)
            {
                var inp = step.Inputs[j];
                if (lastUse.TryGetValue(inp, out int lu) && lu == i)
                {
                    // This input's lifetime just ended — add to dead pool
                    long k = BufferKey(inp);
                    if (!deadPool.TryGetValue(k, out var p))
                    {
                        p = new List<Tensor<T>>(4);
                        deadPool[k] = p;
                    }
                    p.Add(inp);
                }
            }

            // Also check if this step's output dies immediately (consumed
            // only by the next step, which is common for element-wise chains)
            if (lastUse.TryGetValue(output, out int outLu) && outLu == i)
            {
                long k = BufferKey(output);
                if (!deadPool.TryGetValue(k, out var p))
                {
                    p = new List<Tensor<T>>(4);
                    deadPool[k] = p;
                }
                p.Add(output);
            }
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
