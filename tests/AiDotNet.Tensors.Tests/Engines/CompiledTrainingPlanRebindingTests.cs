using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// Tests that CompiledTrainingPlan.Step() sees in-place parameter updates.
/// Repro for: compiled plan returning constant loss after SGD updates.
/// </summary>
public class CompiledTrainingPlanRebindingTests
{
    [Fact]
    public void Step_SeesInPlaceParameterUpdates()
    {
        var engine = new CpuEngine();
        var numOps = MathHelper.GetNumericOperations<float>();

        // Simple: input @ weight, sum -> loss
        var input = Tensor<float>.CreateRandom([4, 3]);
        var weight = Tensor<float>.CreateRandom([3, 2]);

        // Compile training plan
        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            var output = engine.TensorMatMul(input, weight);
            engine.ReduceSum(output, null);
            plan = scope.CompileTraining(new[] { weight });
        }

        // Step 1: get initial loss
        var loss1 = plan.Step();
        float lossVal1 = loss1[0];

        // Manually update weight in-place (simulate SGD)
        var grad = plan.Gradients[0];
        if (grad is not null)
        {
            var lr = engine.TensorMultiplyScalar(grad, 0.1f);
            engine.TensorSubtractInPlace(weight, lr);
        }
        else
        {
            // Force a change to verify the mechanism
            var data = weight.GetDataArray();
            for (int i = 0; i < data.Length; i++)
                data[i] += 0.1f;
        }

        // Step 2: loss should be different because weight changed
        var loss2 = plan.Step();
        float lossVal2 = loss2[0];

        Assert.False(float.IsNaN(lossVal1), "Loss 1 is NaN");
        Assert.False(float.IsNaN(lossVal2), "Loss 2 is NaN");
        Assert.NotEqual(lossVal1, lossVal2);
    }

    [Fact]
    public void Step_ViaCompiledModelCache_SeesUpdates()
    {
        var engine = new CpuEngine();

        var input = Tensor<float>.CreateRandom([4, 3]);
        var weight = Tensor<float>.CreateRandom([3, 2]);

        var cache = new CompiledModelCache<float>();
        var plan = cache.GetOrCompileTraining(
            input._shape,
            () =>
            {
                var output = engine.TensorMatMul(input, weight);
                engine.ReduceSum(output, null);
            },
            new[] { weight });

        // Step 1
        var loss1 = plan.Step();
        float val1 = loss1[0];

        // SGD update
        var grad = plan.Gradients[0];
        if (grad is not null)
        {
            var lr = engine.TensorMultiplyScalar(grad, 0.1f);
            engine.TensorSubtractInPlace(weight, lr);
        }

        // Step 2 — should get same plan from cache, but see updated weights
        var plan2 = cache.GetOrCompileTraining(
            input._shape,
            () =>
            {
                var output = engine.TensorMatMul(input, weight);
                engine.ReduceSum(output, null);
            },
            new[] { weight });

        var loss2 = plan2.Step();
        float val2 = loss2[0];

        Assert.NotEqual(val1, val2);
    }
}
