using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// The plan's own gradient array is what its clip, regularization and optimizer read. Gradients used to return that
/// array itself whenever no live-map entry differed, so a caller could rebind an entry the next step then consumed.
/// </summary>
[Collection("CompiledTrainingPlanSerial")]
public class CompiledPlanGradientsArrayTests
{
    private static Tensor<float> Make(int[] shape, int seed)
    {
        int n = shape.Aggregate(1, (a, b) => a * b);
        var data = new float[n];
        for (int i = 0; i < n; i++) data[i] = (float)System.Math.Sin(seed * 31 + i * 0.37) * 0.5f;
        return new Tensor<float>(data, shape);
    }

    [Fact]
    public void Gradients_IsACopy_SoRebindingAnEntryDoesNotReachThePlan()
    {
        var engine = new CpuEngine();
        var x = Make(new[] { 4, 8 }, 1);
        var w1 = Make(new[] { 8, 8 }, 2);
        var w2 = Make(new[] { 8, 8 }, 3);
        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            engine.ReduceSum(engine.TensorMatMul(engine.ReLU(engine.TensorMatMul(x, w1)), w2), null);
            plan = scope.CompileTraining(new[] { w1, w2 });
        }
        plan.Step();

        var first = plan.Gradients;
        var original = first[0];
        var expected = original.AsSpan().ToArray();
        first[0] = new Tensor<float>(new float[original.Length], original.Shape.ToArray());

        var second = plan.Gradients;
        Assert.NotSame(first, second);
        Assert.Same(original, second[0]);
        Assert.Equal(expected, second[0].AsSpan().ToArray());
    }
}
