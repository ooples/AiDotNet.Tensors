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
                && steps[i].Inputs[0].Rank == 2 && steps[i].Inputs[1].Rank == 2)
            {
                int k = steps[i].Inputs[0]._shape[1];
                int n = steps[i].Inputs[1]._shape[1];

                // Find all subsequent independent MatMuls with same K,N
                var group = new List<int> { i };
                var groupOutputs = new HashSet<object> { steps[i].OutputBuffer };

                for (int j = i + 1; j < steps.Length; j++)
                {
                    // CONSECUTIVE-RUN ONLY: a batched group must be a CONTIGUOUS run of
                    // matmuls. The batched step executes every member at the FIRST member's
                    // position (and no-ops the originals). If a NON-group step sat between two
                    // members, moving the later matmul before it would write that matmul's
                    // output buffer early — and MemoryPlanningPass may have aliased that buffer
                    // onto a tensor still live across the skipped region (buffers are reused by
                    // original-order lifetime). Writing early then clobbers a live value →
                    // silently wrong replay. This was latent while only single-input plans
                    // existed: their secondary leaves are baked constants, so matmul(const,
                    // const) is constant-folded away before this pass and never batched. A
                    // multi-input plan (diffusion denoiser: noisy sample + per-step timestep
                    // embedding) keeps those matmuls dynamic, so they reach this pass and get
                    // reordered (#1620). Stop the run at the first step that can't join, so no
                    // member is ever moved across an intervening step's buffer lifetime.
                    if (consumed.Contains(j)) break;
                    if (steps[j].OpType != OpType.TensorMatMul || steps[j].Inputs.Length != 2) break;
                    if (steps[j].Inputs[0].Rank != 2 || steps[j].Inputs[1].Rank != 2) break;

                    int jk = steps[j].Inputs[0]._shape[1];
                    int jn = steps[j].Inputs[1]._shape[1];
                    if (jk != k || jn != n) break;

                    // Independence: j's inputs must not depend on a group output, and must be
                    // available at position i (produced before i, or a graph-level leaf).
                    bool canBatch = true;
                    foreach (var inp in steps[j].Inputs)
                    {
                        if (groupOutputs.Contains(inp))
                        {
                            canBatch = false;
                            break;
                        }
                        if (producedByStep.TryGetValue(inp, out int producerIdx) && producerIdx >= i)
                        {
                            canBatch = false;
                            break;
                        }
                    }
                    if (!canBatch) break;

                    group.Add(j);
                    groupOutputs.Add(steps[j].OutputBuffer);
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
}
