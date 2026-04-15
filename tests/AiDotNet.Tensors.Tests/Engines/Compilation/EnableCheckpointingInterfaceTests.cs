using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// Verifies that <see cref="ICompiledTrainingPlan{T}.EnableCheckpointing"/> is reachable
/// through the public interface (not just the internal concrete class) and that enabling
/// checkpointing produces numerically equivalent gradients.
///
/// Reference: Issue #165 — expose EnableCheckpointing on ICompiledTrainingPlan interface.
/// </summary>
public class EnableCheckpointingInterfaceTests
{
    [Fact]
    public void EnableCheckpointing_IsCallable_ViaPublicInterface()
    {
        var engine = new CpuEngine();
        var input = Tensor<float>.CreateRandom([4, 3]);
        var weight = Tensor<float>.CreateRandom([3, 2]);

        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            var output = engine.TensorMatMul(input, weight);
            engine.ReduceSum(output, null);
            plan = scope.CompileTraining(new[] { weight });
        }

        // Interface method must be callable without reflection or casts
        plan.EnableCheckpointing(segmentSize: 0);

        // Plan should still be usable after enabling checkpointing
        var loss = plan.Step();
        Assert.False(float.IsNaN(loss[0]), "Loss is NaN after enabling checkpointing");
    }

    [Fact]
    public void EnableCheckpointing_ProducesEquivalentGradients()
    {
        // Acceptance criteria: checkpointed gradients match non-checkpointed gradients
        // within floating-point tolerance.
        var engine = new CpuEngine();

        // Fixed inputs — two independent plans must see the same data to produce
        // comparable gradients.
        var input = Tensor<float>.CreateRandom([4, 3]);
        var weightBaseline = input.Clone(); // placeholder; replaced below
        weightBaseline = Tensor<float>.CreateRandom([3, 2]);
        var weightCheckpointed = weightBaseline.Clone();

        ICompiledTrainingPlan<float> baseline;
        using (var scope = GraphMode.Enable())
        {
            var output = engine.TensorMatMul(input, weightBaseline);
            engine.ReduceSum(output, null);
            baseline = scope.CompileTraining(new[] { weightBaseline });
        }

        ICompiledTrainingPlan<float> checkpointed;
        using (var scope = GraphMode.Enable())
        {
            var output = engine.TensorMatMul(input, weightCheckpointed);
            engine.ReduceSum(output, null);
            checkpointed = scope.CompileTraining(new[] { weightCheckpointed });
        }
        checkpointed.EnableCheckpointing(segmentSize: 1);

        var lossBaseline = baseline.Step();
        var lossCheckpointed = checkpointed.Step();

        // Loss values should match within tolerance
        Assert.Equal(lossBaseline[0], lossCheckpointed[0], precision: 4);

        // Gradient shapes match
        Assert.Equal(baseline.Gradients.Length, checkpointed.Gradients.Length);

        // Each gradient tensor must match element-wise within tolerance
        for (int i = 0; i < baseline.Gradients.Length; i++)
        {
            var gBase = baseline.Gradients[i];
            var gChk = checkpointed.Gradients[i];
            Assert.NotNull(gBase);
            Assert.NotNull(gChk);
            Assert.Equal(gBase!._shape, gChk!._shape);

            var baseData = gBase.GetDataArray();
            var chkData = gChk.GetDataArray();
            Assert.Equal(baseData.Length, chkData.Length);
            for (int k = 0; k < baseData.Length; k++)
            {
                Assert.Equal(baseData[k], chkData[k], precision: 4);
            }
        }
    }

    [Fact]
    public void EnableCheckpointing_ViaCompiledModelCache_Works()
    {
        // Verifies the flow that blocked AiDotNet: consumer receives ICompiledTrainingPlan
        // from CompiledModelCache.GetOrCompileTraining and must call EnableCheckpointing
        // on the interface reference directly.
        var engine = new CpuEngine();
        var input = Tensor<float>.CreateRandom([4, 3]);
        var weight = Tensor<float>.CreateRandom([3, 2]);

        var cache = new CompiledModelCache<float>();
        ICompiledTrainingPlan<float> plan = cache.GetOrCompileTraining(
            input._shape,
            () =>
            {
                var output = engine.TensorMatMul(input, weight);
                engine.ReduceSum(output, null);
            },
            new[] { weight });

        // The core acceptance criterion: this must compile without a cast to the
        // internal concrete class.
        plan.EnableCheckpointing(segmentSize: 0);

        var loss = plan.Step();
        Assert.False(float.IsNaN(loss[0]));
    }

    [Fact]
    public void EnableCheckpointing_DefaultSegmentSize_Works()
    {
        // Default parameter (segmentSize: 0 = auto sqrt(N)) should be usable
        // without specifying the argument explicitly.
        var engine = new CpuEngine();
        var input = Tensor<float>.CreateRandom([4, 3]);
        var weight = Tensor<float>.CreateRandom([3, 2]);

        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            var output = engine.TensorMatMul(input, weight);
            engine.ReduceSum(output, null);
            plan = scope.CompileTraining(new[] { weight });
        }

        // No argument — tests the default parameter value is exposed through the interface
        plan.EnableCheckpointing();

        var loss = plan.Step();
        Assert.False(float.IsNaN(loss[0]));
    }
}
