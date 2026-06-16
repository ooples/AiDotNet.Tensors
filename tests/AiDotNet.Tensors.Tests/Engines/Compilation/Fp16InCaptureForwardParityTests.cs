// FP16-in-capture increment (a): verify CompiledTrainingPlan's HETEROGENEOUS forward (RunFp16HeteroForward —
// replay every node's Realize over the captured mixed-dtype order) produces a loss that matches the FP32
// forward. This is the gate that the FP16 activation-storage nodes (LazyNode<Half> + MixedPrecisionCast),
// which the float-only forward steps drop, are correctly executed by the plan. Forward-only here; the
// device-cast backward bridge is the next increment.

#nullable disable
using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

[Collection(MixedPrecisionTestCollection.Name)] // serializes MixedPrecisionEmit.TestOverrideEnabled mutators
public class Fp16InCaptureForwardParityTests
{
    private static Tensor<float> Rand(int[] shape, int seed)
    {
        var rng = new Random(seed);
        int n = 1; foreach (var d in shape) n *= d;
        var data = new float[n];
        for (int i = 0; i < n; i++) data[i] = (float)(rng.NextDouble() * 2 - 1);
        return new Tensor<float>(data, shape);
    }

    [Fact]
    public void Fp16HeteroForward_MatchesFp32_ForwardLoss()
    {
        var engine = new CpuEngine();
        var input = Rand(new[] { 4, 5 }, 1);
        var w1 = Rand(new[] { 5, 6 }, 2);
        var w2 = Rand(new[] { 6, 4 }, 3);

        // FP32 reference loss (eager, no graph, no autocast).
        var h0 = engine.TensorMatMul(input, w1);
        var y0 = engine.TensorMatMul(h0, w2);
        var loss0 = engine.ReduceSum(y0, null);
        float fp32Loss = loss0.GetFlat(0);

        // FP16-activation-storage traced plan: under an FP16 autocast + the storage override, the matmul
        // activations are emitted as LazyNode<Half>; CompiledTrainingPlan (increment 0) captures the
        // heterogeneous order.
        var prev = MixedPrecisionEmit.TestOverrideEnabled;
        MixedPrecisionEmit.TestOverrideEnabled = true;
        CompiledTrainingPlan<float> plan;
        try
        {
            using (var scope = GraphMode.Enable())
            using (new AutocastScope(PrecisionMode.Float16))
            {
                var h = engine.TensorMatMul(input, w1);
                var y = engine.TensorMatMul(h, w2);
                engine.ReduceSum(y, null);
                plan = scope.CompileTraining(new[] { w1, w2 });
            }
        }
        finally { MixedPrecisionEmit.TestOverrideEnabled = prev; GraphMode.SetCurrent(null); }

        try
        {
            Assert.True(plan.HasFp16HeteroForward,
                "FP16 hetero order should be captured (the traced graph contains Half activation nodes).");

            var lossT = plan.RunFp16HeteroForward(engine);
            float fp16Loss = lossT.GetFlat(0);

            Assert.True(float.IsFinite(fp16Loss), $"FP16 hetero forward loss must be finite, got {fp16Loss}.");
            // FP16 activation storage rounds activations to ~3 sig digits; the loss is a sum, so allow a
            // generous relative+absolute band. The point is correctness (matches FP32), not bit-equality.
            float tol = 0.1f * Math.Abs(fp32Loss) + 0.5f;
            Assert.True(Math.Abs(fp16Loss - fp32Loss) < tol,
                $"FP16-activation hetero forward loss {fp16Loss} should match FP32 {fp32Loss} within {tol}.");
        }
        finally { plan.Dispose(); }
    }
}
