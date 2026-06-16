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
        return new Tensor<float>(RandData(shape, seed), shape);
    }

    private static float[] RandData(int[] shape, int seed)
    {
        var rng = new Random(seed);
        int n = 1; foreach (var d in shape) n *= d;
        var data = new float[n];
        for (int i = 0; i < n; i++) data[i] = (float)(rng.NextDouble() * 2 - 1);
        return data;
    }

    private static void AssertGradsClose(float[] fp32, float[] fp16, string what)
    {
        Assert.Equal(fp32.Length, fp16.Length);
        float maxAbs = 0, maxDiff = 0;
        for (int i = 0; i < fp32.Length; i++)
        {
            maxAbs = Math.Max(maxAbs, Math.Abs(fp32[i]));
            maxDiff = Math.Max(maxDiff, Math.Abs(fp32[i] - fp16[i]));
        }
        // FP16 activation storage perturbs gradients more than the forward; allow a generous band. The gate
        // is "correct gradient (matches FP32 direction/magnitude)", not bit-equality.
        float tol = 0.15f * maxAbs + 0.05f;
        Assert.True(maxDiff < tol, $"{what} grad: maxDiff {maxDiff} exceeds tol {tol} (maxAbs {maxAbs}).");
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

    [Fact]
    public void Fp16HeteroBackward_MatchesFp32_ParamGrads()
    {
        var engine = new CpuEngine();
        var input = Rand(new[] { 4, 5 }, 1);
        float[] w1d = RandData(new[] { 5, 6 }, 2), w2d = RandData(new[] { 6, 4 }, 3);

        // FP32 reference grads: bare CompileTraining + one Step (forward+backward, no optimizer configured ⇒
        // grads filled, no weight update, no clip since _maxGradNorm==0). Read the grad buffers.
        var w1a = new Tensor<float>((float[])w1d.Clone(), new[] { 5, 6 });
        var w2a = new Tensor<float>((float[])w2d.Clone(), new[] { 6, 4 });
        float[] g1_32, g2_32;
        CompiledTrainingPlan<float> fp32Plan;
        using (var scope = GraphMode.Enable())
        {
            var h = engine.TensorMatMul(input, w1a);
            var y = engine.TensorMatMul(h, w2a);
            engine.ReduceSum(y, null);
            fp32Plan = scope.CompileTraining(new[] { w1a, w2a });
        }
        try
        {
            fp32Plan.Step();
            g1_32 = fp32Plan.Gradients[0].ToArray();
            g2_32 = fp32Plan.Gradients[1].ToArray();
        }
        finally { fp32Plan.Dispose(); }

        // FP16 grads from the heterogeneous forward+backward (same init params).
        var w1b = new Tensor<float>((float[])w1d.Clone(), new[] { 5, 6 });
        var w2b = new Tensor<float>((float[])w2d.Clone(), new[] { 6, 4 });
        var prev = MixedPrecisionEmit.TestOverrideEnabled;
        MixedPrecisionEmit.TestOverrideEnabled = true;
        CompiledTrainingPlan<float> fp16Plan;
        try
        {
            using (var scope = GraphMode.Enable())
            using (new AutocastScope(PrecisionMode.Float16))
            {
                var h = engine.TensorMatMul(input, w1b);
                var y = engine.TensorMatMul(h, w2b);
                engine.ReduceSum(y, null);
                fp16Plan = scope.CompileTraining(new[] { w1b, w2b });
            }
        }
        finally { MixedPrecisionEmit.TestOverrideEnabled = prev; GraphMode.SetCurrent(null); }

        try
        {
            Assert.True(fp16Plan.HasFp16HeteroForward);
            fp16Plan.RunFp16HeteroForward(engine);
            fp16Plan.RunFp16HeteroBackward(engine);
            AssertGradsClose(g1_32, fp16Plan.Gradients[0].ToArray(), "w1");
            AssertGradsClose(g2_32, fp16Plan.Gradients[1].ToArray(), "w2");
        }
        finally { fp16Plan.Dispose(); }
    }

    [Fact]
    public void Fp16HeteroStep_Wired_LossMatchesFp32()
    {
        // End-to-end: plan.Step() must route through the heterogeneous forward+backward (wired into StepEager)
        // and return a loss matching the FP32 plan's Step loss.
        var engine = new CpuEngine();
        var input = Rand(new[] { 4, 5 }, 1);
        float[] w1d = RandData(new[] { 5, 6 }, 2), w2d = RandData(new[] { 6, 4 }, 3);

        var w1a = new Tensor<float>((float[])w1d.Clone(), new[] { 5, 6 });
        var w2a = new Tensor<float>((float[])w2d.Clone(), new[] { 6, 4 });
        float fp32Loss;
        CompiledTrainingPlan<float> fp32Plan;
        using (var scope = GraphMode.Enable())
        {
            var h = engine.TensorMatMul(input, w1a);
            var y = engine.TensorMatMul(h, w2a);
            engine.ReduceSum(y, null);
            fp32Plan = scope.CompileTraining(new[] { w1a, w2a });
        }
        try { fp32Loss = fp32Plan.Step().GetFlat(0); } finally { fp32Plan.Dispose(); }

        var w1b = new Tensor<float>((float[])w1d.Clone(), new[] { 5, 6 });
        var w2b = new Tensor<float>((float[])w2d.Clone(), new[] { 6, 4 });
        var prev = MixedPrecisionEmit.TestOverrideEnabled;
        MixedPrecisionEmit.TestOverrideEnabled = true;
        CompiledTrainingPlan<float> fp16Plan;
        try
        {
            using (var scope = GraphMode.Enable())
            using (new AutocastScope(PrecisionMode.Float16))
            {
                var h = engine.TensorMatMul(input, w1b);
                var y = engine.TensorMatMul(h, w2b);
                engine.ReduceSum(y, null);
                fp16Plan = scope.CompileTraining(new[] { w1b, w2b });
            }
        }
        finally { MixedPrecisionEmit.TestOverrideEnabled = prev; GraphMode.SetCurrent(null); }

        try
        {
            Assert.True(fp16Plan.HasFp16HeteroForward);
            float fp16Loss = fp16Plan.Step().GetFlat(0);
            Assert.True(float.IsFinite(fp16Loss), $"Step loss must be finite, got {fp16Loss}.");
            float tol = 0.1f * Math.Abs(fp32Loss) + 0.5f;
            Assert.True(Math.Abs(fp16Loss - fp32Loss) < tol,
                $"FP16 hetero Step() loss {fp16Loss} should match FP32 Step() loss {fp32Loss} within {tol}.");
        }
        finally { fp16Plan.Dispose(); }
    }
}
