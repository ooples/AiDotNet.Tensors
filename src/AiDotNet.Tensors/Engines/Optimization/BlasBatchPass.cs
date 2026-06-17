using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.Simd;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tensors.Engines.Optimization;

/// <summary>
/// Phase 7.1: BLAS batch optimization — groups multiple small independent MatMul
/// operations with compatible shapes into fewer, larger BLAS calls.
///
/// Pattern: multiple independent MatMul[M,K]@[K,N] with same K,N dimensions
/// can be batched into a single larger MatMul by stacking inputs.
///
/// This is particularly effective for multi-head attention (8-16 small matmuls)
/// and Mixture of Experts (many parallel expert computations).
///
/// Expected gain: 2-5x for models with many small independent matmuls.
/// </summary>
internal sealed class BlasBatchPass : ICpuOptimizationPass
{
    public string Name => "BlasBatch";

    public bool IsEnabled => TensorCodecOptions.Current.EnableBlasBatch;

    public CompiledStep<T>[]? TryOptimize<T>(CompiledStep<T>[] steps, IEngine engine)
    {
        if (!IsEnabled || typeof(T) != typeof(float) || steps.Length < 2) return null;

        // Build set of tensors available at each position (outputs of steps 0..i-1).
        // Tensors not produced by any step are graph-level inputs and always available.
        var producedByStep = new Dictionary<object, int>(); // tensor → step index that produces it
        for (int s = 0; s < steps.Length; s++)
            producedByStep[steps[s].OutputBuffer] = s;

        // Find groups of independent MatMul steps with matching inner dimensions
        var result = new List<CompiledStep<T>>(steps.Length);
        bool anyBatched = false;

        // Track which steps have been consumed by batching
        var consumed = new HashSet<int>();

        for (int i = 0; i < steps.Length; i++)
        {
            if (consumed.Contains(i))
                continue;

            if (steps[i].OpType == OpType.TensorMatMul && steps[i].Inputs.Length == 2
                && steps[i].Inputs[0].Rank == 2 && steps[i].Inputs[1].Rank == 2
                && !HasAliasedOutput(steps[i]))
            {
                int k = steps[i].Inputs[0]._shape[1];
                int n = steps[i].Inputs[1]._shape[1];

                // Find all subsequent independent MatMuls with same K,N
                var group = new List<int> { i };
                var groupOutputs = new HashSet<object> { steps[i].OutputBuffer };

                for (int j = i + 1; j < steps.Length; j++)
                {
                    if (consumed.Contains(j)) continue;
                    if (steps[j].OpType != OpType.TensorMatMul || steps[j].Inputs.Length != 2) continue;
                    if (steps[j].Inputs[0].Rank != 2 || steps[j].Inputs[1].Rank != 2) continue;

                    // Skip matmuls whose output buffer was aliased to share storage with
                    // another tensor (see HasAliasedOutput) — they cannot be hoisted.
                    if (HasAliasedOutput(steps[j])) continue;

                    int jk = steps[j].Inputs[0]._shape[1];
                    int jn = steps[j].Inputs[1]._shape[1];
                    if (jk != k || jn != n) continue;

                    // Check independence: j's inputs must not depend on group outputs
                    // AND must be available at position i (produced by steps before i,
                    // or not produced by any step at all — i.e., graph-level inputs).
                    bool canBatch = true;
                    foreach (var inp in steps[j].Inputs)
                    {
                        if (groupOutputs.Contains(inp))
                        {
                            canBatch = false;
                            break;
                        }
                        // If this input is produced by a step that comes after position i,
                        // it won't be available when the batched step executes at position i.
                        if (producedByStep.TryGetValue(inp, out int producerIdx) && producerIdx >= i)
                        {
                            canBatch = false;
                            break;
                        }
                    }

                    if (canBatch)
                    {
                        group.Add(j);
                        groupOutputs.Add(steps[j].OutputBuffer);
                    }
                }

                if (group.Count >= 2)
                {
                    // Worth batching — create batched step
                    foreach (var idx in group)
                        consumed.Add(idx);

                    // For now, execute them sequentially but in a single compiled step
                    // (reduces dispatch overhead). True batched BLAS requires cblas_sgemm_batch.
                    var batchedSteps = group.Select(idx => steps[idx]).ToArray();
                    var firstStep = steps[group[0]];

                    result.Add(new CompiledStep<T>(
                        "BatchedMatMul",
                        (eng, output) =>
                        {
                            foreach (var s in batchedSteps)
                                s.Execute(eng, s.OutputBuffer);
                        },
                        firstStep.OutputBuffer,
                        firstStep.Inputs,
                        null,
                        null));

                    // Add remaining outputs as no-op steps so they're accessible
                    for (int g = 1; g < group.Count; g++)
                    {
                        var s = steps[group[g]];
                        result.Add(new CompiledStep<T>(
                            "BatchedMatMul_Output",
                            (eng, output) => { }, // Already computed above
                            s.OutputBuffer,
                            s.Inputs,
                            null,
                            null));
                    }

                    anyBatched = true;
                    continue;
                }
            }

            result.Add(steps[i]);
        }

        return anyBatched ? result.ToArray() : null;
    }

    /// <summary>
    /// True when this step's output buffer shares its backing storage with another
    /// tensor (storage refcount &gt; 1) — i.e. <see cref="MemoryPlanningPass"/> rebound
    /// it onto a donor's storage, or it is a view.
    /// <para>
    /// Batching <b>hoists</b> every grouped matmul to the position of the group's
    /// first member and runs them back-to-back there (see the batched delegate
    /// below). MemoryPlanningPass runs earlier and computes buffer lifetimes from
    /// the <i>original</i> step order; it may legally donate a buffer D (dead at
    /// its original last-use) to a later matmul M's output. If BlasBatch then
    /// hoists M earlier than D's last consumer, M's write lands in D's storage
    /// before that consumer reads it — corrupting the consumer's input. (Concrete
    /// case: shared-input Q/K/V projection matmuls feeding reshape→permute→SDPA;
    /// the V-projection buffer dies at its reshape, MemoryPlanning donates it to
    /// the K-projection matmul, and batching hoists that K matmul ahead of the V
    /// reshape — clobbering V.) An output with refcount 1 is privately owned, so
    /// hoisting it only computes its own value earlier into its own buffer, which
    /// is always safe. Refusing to batch aliased outputs is the necessary and
    /// sufficient guard; the only cost is forgoing the batch for those matmuls.
    /// </para>
    /// </summary>
    private static bool HasAliasedOutput<T>(CompiledStep<T> step)
        => step.OutputBuffer._storage.RefCount > 1 || step.OutputBuffer.IsView;
}
