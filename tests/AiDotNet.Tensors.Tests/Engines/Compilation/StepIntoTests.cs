using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// Tests for issue #199 — <see cref="ICompiledTrainingPlan{T}.StepInto"/>
/// and <see cref="ICompiledTrainingPlan{T}.SetInputs"/>. Same capture-safe
/// contract as the inference plan: caller owns the loss / input buffers,
/// the plan copies in + out, kernels stay compile-time-specialised.
/// </summary>
public class StepIntoTests
{
    private static ICompiledTrainingPlan<float> BuildPlan(CpuEngine engine, Tensor<float> input, Tensor<float> weight)
    {
        using var scope = GraphMode.Enable();
        var output = engine.TensorMatMul(input, weight);
        engine.ReduceSum(output, null);
        return scope.CompileTraining(new[] { weight });
    }

    [Fact]
    public void StepInto_WritesLossIntoCallerBuffer()
    {
        var engine = new CpuEngine();
        var input = Tensor<float>.CreateRandom([4, 3]);
        var weight = Tensor<float>.CreateRandom([3, 2]);
        using var plan = BuildPlan(engine, input, weight);
        // Loss from TensorReduceSum of a [4,2] matmul is a scalar — shape [1].
        var lossBuf = new Tensor<float>([1]);
        plan.StepInto(lossBuf);
        Assert.NotEqual(0f, lossBuf.AsSpan()[0]);
    }

    [Fact]
    public void StepInto_MatchesStep_BitExact()
    {
        var engine = new CpuEngine();
        var input = Tensor<float>.CreateRandom([2, 3]);
        var weight = Tensor<float>.CreateRandom([3, 2]);
        using var planA = BuildPlan(engine, input, weight);
        using var planB = BuildPlan(engine, input, weight);

        var viaStep = planA.Step().AsSpan().ToArray();
        var lossBuf = new Tensor<float>([1]);
        planB.StepInto(lossBuf);
        Assert.Equal(viaStep, lossBuf.AsSpan().ToArray());
    }

    [Fact]
    public void StepInto_NullOutput_Throws()
    {
        var engine = new CpuEngine();
        var input = Tensor<float>.CreateRandom([2, 2]);
        var weight = Tensor<float>.CreateRandom([2, 2]);
        using var plan = BuildPlan(engine, input, weight);
        Assert.Throws<ArgumentNullException>(() => plan.StepInto(null!));
    }

    [Fact]
    public void StepInto_WrongShape_Throws()
    {
        var engine = new CpuEngine();
        var input = Tensor<float>.CreateRandom([2, 2]);
        var weight = Tensor<float>.CreateRandom([2, 2]);
        using var plan = BuildPlan(engine, input, weight);
        Assert.Throws<ArgumentException>(() => plan.StepInto(new Tensor<float>([2, 2])));
    }

    [Fact]
    public void StepInto_AfterDispose_Throws()
    {
        var engine = new CpuEngine();
        var input = Tensor<float>.CreateRandom([2, 2]);
        var weight = Tensor<float>.CreateRandom([2, 2]);
        var plan = BuildPlan(engine, input, weight);
        plan.Dispose();
        Assert.Throws<ObjectDisposedException>(() => plan.StepInto(new Tensor<float>([1])));
    }

    [Fact]
    public void SetInputs_ChangesLossOutput()
    {
        var engine = new CpuEngine();
        var input = Tensor<float>.CreateRandom([2, 3]);
        var weight = Tensor<float>.CreateRandom([3, 2]);
        using var plan = BuildPlan(engine, input, weight);

        var firstLoss = plan.Step().AsSpan().ToArray();

        // Swap inputs to something distinctive — should change the loss.
        var newInput = new Tensor<float>([2, 3]);
        var s = newInput.AsWritableSpan();
        for (int i = 0; i < s.Length; i++) s[i] = i * 0.25f + 1f;
        plan.SetInputs(new[] { newInput });
        var newLoss = plan.Step().AsSpan().ToArray();

        Assert.NotEqual(firstLoss, newLoss);
    }

    [Fact]
    public void SetInputs_WrongCount_Throws()
    {
        var engine = new CpuEngine();
        var input = Tensor<float>.CreateRandom([2, 2]);
        var weight = Tensor<float>.CreateRandom([2, 2]);
        using var plan = BuildPlan(engine, input, weight);
        var two = new[] { new Tensor<float>([2, 2]), new Tensor<float>([2, 2]) };
        Assert.Throws<ArgumentException>(() => plan.SetInputs(two));
    }

    [Fact]
    public void SetInputs_AfterDispose_Throws()
    {
        var engine = new CpuEngine();
        var input = Tensor<float>.CreateRandom([2, 2]);
        var weight = Tensor<float>.CreateRandom([2, 2]);
        var plan = BuildPlan(engine, input, weight);
        plan.Dispose();
        Assert.Throws<ObjectDisposedException>(() =>
            plan.SetInputs(new[] { new Tensor<float>([2, 2]) }));
    }

    [Fact]
    public void ExecuteInto_ReplayStability_3Iterations()
    {
        // Issue #199 acceptance: "verify that BeginCapture + ExecuteInto +
        // EndCapture + Replay produces the same bytes as a fresh Execute
        // on 3+ replays." Without a CUDA host we can only validate the
        // capture-safe CPU path — but the same semantic holds: 3 back-to-
        // back ExecuteInto calls against the same output buffer produce
        // identical bytes iff the plan is stateless between runs.
        var engine = new CpuEngine();
        var input = Tensor<float>.CreateRandom([4, 3]);
        var weight = Tensor<float>.CreateRandom([3, 2]);
        ICompiledPlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            engine.TensorMatMul(input, weight);
            plan = scope.CompileInference<float>();
        }
        using (plan)
        {
            var outBuf = new Tensor<float>([4, 2]);
            plan.ExecuteInto(outBuf);
            var first = outBuf.AsSpan().ToArray();
            plan.ExecuteInto(outBuf);
            var second = outBuf.AsSpan().ToArray();
            plan.ExecuteInto(outBuf);
            var third = outBuf.AsSpan().ToArray();
            Assert.Equal(first, second);
            Assert.Equal(second, third);
        }
    }
}
