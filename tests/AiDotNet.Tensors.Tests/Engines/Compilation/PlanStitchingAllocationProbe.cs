// GC.GetAllocatedBytesForCurrentThread is unavailable on this project's
// net471 profile (the API exists on full-framework 4.6+ but isn't surfaced
// here). Gate the probe to modern targets only — the structural sibling
// test in PlanStitchingTests covers net471 with a step-count assertion.
#if NETCOREAPP3_0_OR_GREATER || NET5_0_OR_GREATER
using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// Empirical check for issue #170 acceptance criterion #2:
/// "Allocation profile shows zero intermediate tensor materialization
/// between a and b in the stitched plan."
///
/// The structural sibling test (<c>Then_StitchedPlan_HasExactlyOneBoundaryStep_NoNewTensorMaterialization</c>)
/// proves there is exactly one boundary step. This file PROVES that step's
/// runtime behaviour: a warmed-up stitched-plan Execute() allocates within a
/// few hundred bytes of zero — definitely not a fresh Tensor (which would
/// allocate thousands of bytes for any non-trivial shape).
/// </summary>
public class PlanStitchingAllocationProbe
{
    private static ICompiledPlan<float> CompileMatMulSigmoid(
        IEngine engine, Tensor<float> input, Tensor<float> weight)
    {
        ICompiledPlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            var product = engine.TensorMatMul(input, weight);
            engine.Sigmoid(product);
            plan = scope.CompileInference<float>();
        }
        return plan;
    }

    [Fact]
    public void StitchedExecute_AfterWarmup_AllocatesZeroBytes()
    {
        var engine = new CpuEngine();

        // Big enough that a real Tensor allocation would be hundreds of bytes
        // minimum (Tensor + shape int[] + Vector + storage T[] etc.) — well
        // above the noise floor of any incidental boxing. Output of A is
        // [16, 32] = 512 floats = 2048 bytes for the storage alone.
        var inputA  = Tensor<float>.CreateRandom([16, 24]);
        var weightA = Tensor<float>.CreateRandom([24, 32]);
        var inputB  = Tensor<float>.CreateRandom([16, 32]);
        var weightB = Tensor<float>.CreateRandom([32, 8]);

        var planA = CompileMatMulSigmoid(engine, inputA, weightA);
        var planB = CompileMatMulSigmoid(engine, inputB, weightB);
        using var stitched = planA.Then(planB);

        // Warm everything up so any first-call JIT/lazy-init costs are
        // amortized. Ten replays is plenty for the steady-state to settle.
        for (int i = 0; i < 10; i++) stitched.Execute();

        // Now measure a single Execute under tight conditions.
        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();
        long before = GC.GetAllocatedBytesForCurrentThread();
        stitched.Execute();
        long after  = GC.GetAllocatedBytesForCurrentThread();
        long delta  = after - before;

        // The acceptance criterion is "zero" but in practice there can be
        // tiny incidental allocations from the runtime (boxing at interface
        // boundaries, allocator bookkeeping). We allow up to 256 bytes —
        // any new Tensor would dwarf that (a Tensor<float>[16,32] is at
        // minimum the 2 KB element backing array plus shape/strides/object
        // overhead, easily 2200+ bytes). So this threshold catches "we
        // accidentally materialize a tensor at the boundary" without false-
        // positiving on incidental small-object plumbing.
        Assert.True(delta < 256,
            $"Stitched Execute() allocated {delta} bytes — must be < 256 to prove no Tensor materialization between A and B. " +
            "Any new Tensor allocation would be thousands of bytes.");

        planA.Dispose();
        planB.Dispose();
    }
}
#endif
