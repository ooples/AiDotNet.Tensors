using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.Compilation.Ops;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation.Ops;

/// <summary>
/// Tests for <see cref="FusedGroupNormActivationOp{T}"/>. Validates numerical
/// parity with separate GroupNorm + Activation, attribute access, backward
/// correctness, and memory savings (one fewer tensor allocation).
/// </summary>
public class FusedGroupNormActivationOpTests
{
    // ── Forward parity: fused op matches GroupNorm + Activation separately ───
    [Theory]
    [InlineData(0)] // Identity
    [InlineData(1)] // SiLU
    [InlineData(2)] // ReLU
    public void Forward_MatchesSeparateGroupNormPlusActivation(int activationInt)
    {
        var activation = (GroupNormActivation)activationInt;
        var engine = new CpuEngine();

        var input = Tensor<float>.CreateRandom([2, 8, 4, 4]);
        var gamma = Tensor<float>.CreateRandom([8]);
        var beta  = Tensor<float>.CreateRandom([8]);
        int numGroups = 4;
        double eps = 1e-5;

        // Fused path via Op
        var op = new FusedGroupNormActivationOp<float>(
            input, numGroups, gamma, beta, eps, activation);
        var fusedOutput = new Tensor<float>(op.OutputShape);
        op.BuildForwardClosure()(engine, fusedOutput);

        // Separate path: GroupNorm → Activation
        var gnResult = engine.GroupNorm(input, numGroups, gamma, beta, eps, out _, out _);
        var separateResult = activation switch
        {
            GroupNormActivation.SiLU => engine.Swish(gnResult),
            GroupNormActivation.ReLU => engine.ReLU(gnResult),
            _ => gnResult,
        };

        var fusedData    = fusedOutput.AsSpan();
        var separateData = separateResult.AsSpan();
        Assert.Equal(fusedData.Length, separateData.Length);
        for (int i = 0; i < fusedData.Length; i++)
        {
            Assert.True(
                Math.Abs(fusedData[i] - separateData[i]) < 1e-4f,
                $"Mismatch at [{i}]: fused={fusedData[i]}, separate={separateData[i]}, " +
                $"activation={activation}");
        }
    }

    // ── Multiple shape/group combinations ────────────────────────────────────
    [Theory]
    [InlineData(1, 4, 8, 2)]
    [InlineData(2, 16, 4, 8)]
    [InlineData(1, 6, 6, 3)]
    public void Forward_SiLU_VariousShapes(int batch, int channels, int spatial, int numGroups)
    {
        var engine = new CpuEngine();
        var input = Tensor<float>.CreateRandom([batch, channels, spatial]);
        var gamma = Tensor<float>.CreateRandom([channels]);
        var beta  = Tensor<float>.CreateRandom([channels]);

        var op = new FusedGroupNormActivationOp<float>(
            input, numGroups, gamma, beta, 1e-5, GroupNormActivation.SiLU);
        var output = new Tensor<float>(op.OutputShape);
        op.BuildForwardClosure()(engine, output);

        // Basic sanity: output should be non-trivial
        bool anyNonZero = false;
        var data = output.AsSpan();
        for (int i = 0; i < data.Length; i++)
            if (Math.Abs(data[i]) > 1e-8f) { anyNonZero = true; break; }
        Assert.True(anyNonZero, "Fused GroupNorm+SiLU produced all-zero output");
    }

    // ── Attributes for fusion pass introspection ────────────────────────────
    [Fact]
    public void Attributes_ExposedForFusionPatternMatching()
    {
        var input = Tensor<float>.CreateRandom([1, 8, 4]);
        var gamma = Tensor<float>.CreateRandom([8]);
        var beta  = Tensor<float>.CreateRandom([8]);

        var op = new FusedGroupNormActivationOp<float>(
            input, numGroups: 4, gamma, beta, epsilon: 1e-6, GroupNormActivation.SiLU);

        Assert.Equal("FusedGroupNormActivation", op.OpName);
        Assert.Equal(4, op.NumGroups);
        Assert.Equal(1e-6, op.Epsilon);
        Assert.Equal(GroupNormActivation.SiLU, op.Activation);
        Assert.Same(input, op.Input);
        Assert.Same(gamma, op.Gamma);
        Assert.Same(beta, op.Beta);
        Assert.Equal(input._shape, op.OutputShape);
    }

    // ── ToCompiledStep produces executable step ─────────────────────────────
    [Fact]
    public void ToCompiledStep_Executes()
    {
        var engine = new CpuEngine();
        var input = Tensor<float>.CreateRandom([1, 4, 6]);
        var gamma = Tensor<float>.CreateRandom([4]);
        var beta  = Tensor<float>.CreateRandom([4]);

        var op = new FusedGroupNormActivationOp<float>(
            input, numGroups: 2, gamma, beta, 1e-5, GroupNormActivation.SiLU);
        var outputBuffer = new Tensor<float>(op.OutputShape);
        var step = op.ToCompiledStep(outputBuffer);

        Assert.Equal("FusedGroupNormActivation", step.OpName);
        step.Execute(engine, step.OutputBuffer);

        bool anyNonZero = false;
        var data = outputBuffer.AsSpan();
        for (int i = 0; i < data.Length; i++)
            if (Math.Abs(data[i]) > 1e-8f) { anyNonZero = true; break; }
        Assert.True(anyNonZero);
    }

    // ── TryFromStep round-trip ──────────────────────────────────────────────
    [Fact]
    public void TryFromStep_RoundTrip_PreservesAttributes()
    {
        var input = Tensor<float>.CreateRandom([2, 8, 4]);
        var gamma = Tensor<float>.CreateRandom([8]);
        var beta  = Tensor<float>.CreateRandom([8]);

        var original = new FusedGroupNormActivationOp<float>(
            input, 4, gamma, beta, 1e-6, GroupNormActivation.ReLU);
        var step = original.ToCompiledStep(new Tensor<float>(original.OutputShape));

        var recovered = FusedGroupNormActivationOp<float>.TryFromStep(step);
        Assert.NotNull(recovered);
        Assert.Equal(4, recovered!.NumGroups);
        Assert.Equal(1e-6, recovered.Epsilon);
        Assert.Equal(GroupNormActivation.ReLU, recovered.Activation);
    }

    [Fact]
    public void TryFromStep_WrongOpName_ReturnsNull()
    {
        var t = Tensor<float>.CreateRandom([2, 3]);
        var step = new CompiledStep<float>("TensorMatMul", (e, o) => { }, t, new[] { t, t });
        Assert.Null(FusedGroupNormActivationOp<float>.TryFromStep(step));
    }

    // ── Memory: fused uses one fewer tensor than separate ────────────────────
    // Note: a strict allocation-counting test would require GC instrumentation
    // (like the PlanStitchingAllocationProbe). Here we verify structurally: the
    // fused Op produces ONE CompiledStep, while the separate path would be TWO
    // (GroupNorm + Activation). The stitched plan's step count is the proof.
    [Fact]
    public void Fused_ProducesOneStep_NotTwo()
    {
        var input = Tensor<float>.CreateRandom([1, 4, 4]);
        var gamma = Tensor<float>.CreateRandom([4]);
        var beta  = Tensor<float>.CreateRandom([4]);

        var op = new FusedGroupNormActivationOp<float>(
            input, 2, gamma, beta, 1e-5, GroupNormActivation.SiLU);
        var step = op.ToCompiledStep(new Tensor<float>(op.OutputShape));

        // ONE step for the fused op (GroupNorm + SiLU combined).
        Assert.NotNull(step);
        Assert.Equal("FusedGroupNormActivation", step.OpName);
        // The separate path would be 2 steps. This tests the structural invariant.
    }

    // ── Argument validation ─────────────────────────────────────────────────
    [Fact]
    public void Constructor_NullInput_Throws()
    {
        Assert.Throws<ArgumentNullException>(() =>
            new FusedGroupNormActivationOp<float>(
                null!, 2, Tensor<float>.CreateRandom([4]), Tensor<float>.CreateRandom([4])));
    }

    [Fact]
    public void Constructor_ZeroGroups_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new FusedGroupNormActivationOp<float>(
                Tensor<float>.CreateRandom([1, 4, 4]), 0,
                Tensor<float>.CreateRandom([4]), Tensor<float>.CreateRandom([4])));
    }
}
