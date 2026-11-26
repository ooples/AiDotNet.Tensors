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

/// <summary>
/// A device engine's plan zeroes its gradients and runs the backward without a write generation, so an operand whose
/// gradient the plan made a view of an add's output gradient reached the in-place add with that same buffer as its
/// contribution, and was counted twice whenever the add's other operand (a parameter, or a broadcast bias) kept the
/// step running. Causal multi-head attention adds its mask as a broadcast bias: its compiled query and key gradients
/// were twice the tape's, three times on the second step. The device engine's CPU fallback has the same plan.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class CompiledDevicePassThroughGradientTests
{
    private static Tensor<float> Seeded(int[] shape, int seed)
    {
        var random = new Random(seed);
        int length = shape.Aggregate(1, (a, b) => a * b);
        return new Tensor<float>(Enumerable.Range(0, length).Select(_ => (float)(random.NextDouble() * 2 - 1)).ToArray(), shape);
    }

    public static IEnumerable<object[]> Cases() => new[] { new object[] { "add-parameter" }, new object[] { "broadcast-bias" },
        new object[] { "causal-attention" } };

    [Theory]
    [MemberData(nameof(Cases))]
    public void CompiledGradient_EqualsTheTape_EveryStep(string name)
    {
        var prior = AiDotNetEngine.Current;
        using var engine = new DirectGpuTensorEngine();
        AiDotNetEngine.Current = engine;
        try
        {
            Tensor<float>[] parameters;
            Func<Tensor<float>> forward;
            switch (name)
            {
                case "add-parameter":
                {
                    var x = Seeded([2, 2, 3], 1);
                    var p = Seeded([2, 2, 3], 2);
                    parameters = [x, p];
                    forward = () => engine.TensorMultiply(engine.TensorAdd(engine.TensorMultiplyScalar(x, 0.7f), p), p);
                    break;
                }
                case "broadcast-bias":
                {
                    var x = Seeded([2, 2, 3], 1);
                    var bias = Seeded([2, 3], 3);
                    parameters = [x, bias];
                    forward = () => engine.TensorMultiply(engine.TensorBroadcastAdd(engine.TensorMultiplyScalar(x, 0.7f), bias), x);
                    break;
                }
                default:
                {
                    var q = Seeded([1, 2, 4], 1);
                    var k = Seeded([1, 3, 4], 2);
                    var v = Seeded([1, 3, 4], 3);
                    parameters = [q, k, v];
                    forward = () => engine.MultiHeadAttentionCore(q, k, v, 2, 0.7, causal: true);
                    break;
                }
            }

            float[][] tape;
            using (var recording = new AiDotNet.Tensors.Engines.Autodiff.GradientTape<float>())
            {
                var grads = recording.ComputeGradients(engine.ReduceSum(forward(), null), parameters);
                tape = parameters.Select(p => grads[p].ToArray()).ToArray();
            }

            using var scope = GraphMode.EnableTraining(parameters);
            engine.ReduceSum(forward(), null);
            using var plan = scope.CompileTraining(parameters);
            for (int step = 0; step < 2; step++)
            {
                plan.Step();
                for (int i = 0; i < parameters.Length; i++)
                {
                    var compiled = plan.Gradients[i].GetFlattenedData().ToArray();
                    for (int j = 0; j < compiled.Length; j++)
                        Assert.True(Math.Abs(compiled[j] - tape[i][j]) <= 1e-5f * Math.Max(1f, Math.Abs(tape[i][j])),
                            $"{name} step {step} input {i}[{j}]: tape {tape[i][j]} compiled {compiled[j]}");
                }
            }
        }
        finally { AiDotNetEngine.Current = prior; }
    }
}
