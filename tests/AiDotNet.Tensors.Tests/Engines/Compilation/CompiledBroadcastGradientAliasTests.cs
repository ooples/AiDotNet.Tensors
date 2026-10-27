using System.Reflection;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

[Collection("CompilationGlobalState")]
public sealed class CompiledBroadcastGradientAliasTests
{
    [Theory]
    [InlineData(true, false, false)]
    [InlineData(false, false, false)]
    [InlineData(true, true, false)]
    [InlineData(false, true, false)]
    [InlineData(true, false, true)]
    public void IdentityGradient_ReusesOnlySingleConsumerNonParameterStorage(
        bool rowBias, bool sharedInput, bool parameterInput)
    {
        var prior = AiDotNetEngine.Current;
        var engine = new CpuEngine();
        AiDotNetEngine.Current = engine;
        try
        {
            var weight = new Tensor<float>(Enumerable.Range(1, 15).Select(i => (float)i).ToArray(), [3, 5]);
            var bias = new Tensor<float>(Enumerable.Repeat(1f, rowBias ? 3 : 5).ToArray(), rowBias ? [3, 1] : [1, 5]);
            var factor = new Tensor<float>(Enumerable.Repeat(2f, 15).ToArray(), [3, 5]);
            Tensor<float> left, output;
            ICompiledTrainingPlan<float> plan;
            using (var scope = GraphMode.EnableTraining(new[] { weight, bias }))
            {
                left = parameterInput ? weight : engine.TensorMultiply(weight, factor);
                output = engine.TensorAdd(left, bias);
                var loss = engine.ReduceSum(engine.TensorMultiply(output, output), null);
                if (sharedInput)
                    loss = engine.TensorAdd(loss, engine.ReduceSum(engine.TensorMultiply(left, left), null));
                plan = scope.CompileTraining(new[] { weight, bias }, loss);
            }
            using (plan)
            {
                var map = (Dictionary<Tensor<float>, Tensor<float>>)typeof(CompiledTrainingPlan<float>)
                    .GetField("_liveGradientMap", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(plan)!;
                Assert.Equal(!sharedInput && !parameterInput, map[left].SharesStorageWith(map[output]));
                Assert.False(map[bias].SharesStorageWith(map[output]));
                for (int step = 0; step < 3; step++)
                {
                    engine.TensorFill(weight, (float)(step + 1));
                    plan.Step();
                    float scale = parameterInput ? 1 : 2;
                    float value = scale * (step + 1);
                    float dy = 2 * (value + 1);
                    float dw = scale * (dy + (sharedInput ? 2 * value : 0));
                    Assert.All(plan.Gradients[0].ToArray(), x => Assert.Equal(dw, x));
                    Assert.All(plan.Gradients[1].ToArray(), x => Assert.Equal(dy * (rowBias ? 5 : 3), x));
                }
            }
        }
        finally { AiDotNetEngine.Current = prior; }
    }
}
