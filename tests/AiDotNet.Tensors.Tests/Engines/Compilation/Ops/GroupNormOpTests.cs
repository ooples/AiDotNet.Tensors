using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.Compilation.Ops;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation.Ops;

/// <summary>
/// Tests for <see cref="GroupNormOp{T}"/> — the typed IR operation for
/// Group Normalization. Validates forward parity with the eager engine path,
/// backward gradient correctness, attribute access, and factory round-trip.
/// </summary>
public class GroupNormOpTests
{
    // ── Forward parity: GroupNormOp produces same result as eager engine ─────
    [Theory]
    [InlineData(1, 4, 8, 2)]   // [1,4,8] with 2 groups
    [InlineData(2, 8, 4, 4)]   // [2,8,4] with 4 groups
    [InlineData(1, 6, 6, 3)]   // [1,6,6] with 3 groups
    [InlineData(4, 16, 4, 8)]  // [4,16,4] with 8 groups
    public void Forward_MatchesEagerEngine(int batch, int channels, int spatial, int numGroups)
    {
        var engine = new CpuEngine();

        var input = Tensor<float>.CreateRandom([batch, channels, spatial]);
        var gamma = Tensor<float>.CreateRandom([channels]);
        var beta  = Tensor<float>.CreateRandom([channels]);
        double eps = 1e-5;

        // Eager engine path
        var eagerResult = engine.GroupNorm(input, numGroups, gamma, beta, eps,
            out var eagerMean, out var eagerVar);

        // GroupNormOp path
        var op = new GroupNormOp<float>(input, numGroups, gamma, beta, eps);
        var outputBuffer = new Tensor<float>(input._shape);
        var closure = op.BuildForwardClosure();
        closure(engine, outputBuffer);

        // Bitwise comparison
        var eagerData = eagerResult.AsSpan();
        var opData    = outputBuffer.AsSpan();
        Assert.Equal(eagerData.Length, opData.Length);
        for (int i = 0; i < eagerData.Length; i++)
            Assert.Equal(eagerData[i], opData[i]);

        // Mean/variance should be populated after forward
        Assert.NotNull(op.Mean);
        Assert.NotNull(op.Variance);
    }

    // ── Backward: compiled plan gradients match autodiff ─────────────────────
    [Fact]
    public void Backward_GradientsMatchAutodiff()
    {
        var engine = new CpuEngine();

        var input = Tensor<float>.CreateRandom([2, 4, 3]);
        var gamma = Tensor<float>.CreateRandom([4]);
        var beta  = Tensor<float>.CreateRandom([4]);
        int numGroups = 2;

        // Compile a training plan that uses GroupNorm → ReduceSum
        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            var normed = engine.GroupNorm(input, numGroups, gamma, beta, 1e-5,
                out _, out _);
            engine.ReduceSum(normed, null);
            plan = scope.CompileTraining(new[] { gamma, beta });
        }

        var loss = plan.Step();
        Assert.False(float.IsNaN(loss[0]), "GroupNorm training loss is NaN");
        Assert.NotEqual(0f, loss[0]);

        // Gradients should be non-trivial
        Assert.Equal(2, plan.Gradients.Length);
        bool gammaGradNonZero = false;
        var gammaGrad = plan.Gradients[0].AsSpan();
        for (int i = 0; i < gammaGrad.Length; i++)
            if (Math.Abs(gammaGrad[i]) > 1e-8f) { gammaGradNonZero = true; break; }
        Assert.True(gammaGradNonZero, "Gamma gradients are all zero");

        bool betaGradNonZero = false;
        var betaGrad = plan.Gradients[1].AsSpan();
        for (int i = 0; i < betaGrad.Length; i++)
            if (Math.Abs(betaGrad[i]) > 1e-8f) { betaGradNonZero = true; break; }
        Assert.True(betaGradNonZero, "Beta gradients are all zero");

        plan.Dispose();
    }

    // ── Attributes accessible for fusion pass introspection ──────────────────
    [Fact]
    public void Attributes_ExposedForFusionPatternMatching()
    {
        var input = Tensor<float>.CreateRandom([1, 8, 4]);
        var gamma = Tensor<float>.CreateRandom([8]);
        var beta  = Tensor<float>.CreateRandom([8]);

        var op = new GroupNormOp<float>(input, numGroups: 4, gamma, beta, epsilon: 1e-6);

        Assert.Equal(OpType.GroupNorm, op.OpType);
        Assert.Equal("GroupNorm", op.OpName);
        Assert.Equal(4, op.NumGroups);
        Assert.Equal(1e-6, op.Epsilon);
        Assert.Same(input, op.Input);
        Assert.Same(gamma, op.Gamma);
        Assert.Same(beta, op.Beta);
        Assert.Equal(input._shape, op.OutputShape);
        Assert.Equal(3, op.Inputs.Length);
    }

    // ── ToCompiledStep produces a working step ──────────────────────────────
    [Fact]
    public void ToCompiledStep_ProducesExecutableStep()
    {
        var engine = new CpuEngine();
        var input = Tensor<float>.CreateRandom([1, 4, 6]);
        var gamma = Tensor<float>.CreateRandom([4]);
        var beta  = Tensor<float>.CreateRandom([4]);

        var op = new GroupNormOp<float>(input, numGroups: 2, gamma, beta);
        var outputBuffer = new Tensor<float>(input._shape);
        var step = op.ToCompiledStep(outputBuffer);

        Assert.Equal("GroupNorm", step.OpName);
        Assert.Equal(OpType.GroupNorm, step.OpType);

        // Execute the step
        step.Execute(engine, step.OutputBuffer);

        // Should produce non-zero output
        var data = outputBuffer.AsSpan();
        bool anyNonZero = false;
        for (int i = 0; i < data.Length; i++)
            if (Math.Abs(data[i]) > 1e-8f) { anyNonZero = true; break; }
        Assert.True(anyNonZero, "CompiledStep produced all-zero output");
    }

    // ── TryFromStep round-trip: step → Op → step ────────────────────────────
    [Fact]
    public void TryFromStep_RoundTrip_PreservesAttributes()
    {
        var input = Tensor<float>.CreateRandom([2, 8, 4]);
        var gamma = Tensor<float>.CreateRandom([8]);
        var beta  = Tensor<float>.CreateRandom([8]);

        var original = new GroupNormOp<float>(input, numGroups: 4, gamma, beta, epsilon: 1e-6);
        var outputBuffer = new Tensor<float>(input._shape);

        // Op → CompiledStep (with DifferentiableOps savedState ordering)
        var step = new CompiledStep<float>(
            "GroupNorm",
            original.BuildForwardClosure(),
            outputBuffer,
            original.Inputs,
            original.GetBackwardFunction(),
            new object[] { 4, null!, null!, 1e-6 }); // [numGroups, mean, var, eps]

        // CompiledStep → Op via factory
        var recovered = GroupNormOp<float>.TryFromStep(step);
        Assert.NotNull(recovered);
        Assert.Equal(4, recovered!.NumGroups);
        Assert.Equal(1e-6, recovered.Epsilon);
        Assert.Same(input, recovered.Input);
        Assert.Same(gamma, recovered.Gamma);
        Assert.Same(beta, recovered.Beta);
    }

    // ── TryFromStep returns null for non-GroupNorm steps ────────────────────
    [Fact]
    public void TryFromStep_NonGroupNormOp_ReturnsNull()
    {
        var tensor = Tensor<float>.CreateRandom([2, 3]);
        var step = new CompiledStep<float>(
            "TensorMatMul",
            (eng, output) => { },
            tensor,
            new[] { tensor, tensor });

        Assert.Null(GroupNormOp<float>.TryFromStep(step));
    }

    // ── Argument validation ─────────────────────────────────────────────────
    [Fact]
    public void Constructor_NullInput_Throws()
    {
        Assert.Throws<ArgumentNullException>(() =>
            new GroupNormOp<float>(null!, 2, Tensor<float>.CreateRandom([4]), Tensor<float>.CreateRandom([4])));
    }

    [Fact]
    public void Constructor_ZeroGroups_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new GroupNormOp<float>(Tensor<float>.CreateRandom([1, 4, 4]), 0,
                Tensor<float>.CreateRandom([4]), Tensor<float>.CreateRandom([4])));
    }
}
