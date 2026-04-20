using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.Compilation.Ops;
using AiDotNet.Tensors.Engines.Optimization;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Optimization;

/// <summary>
/// Tests for <see cref="DiffusionFusionPass"/> — patterns 11, 12, and 14
/// for diffusion UNet hot sequences. Validates that the matcher fires on
/// the intended chain, doesn't misfire on non-fusable chains, and the
/// rewritten graph produces identical forward output.
/// </summary>
public class DiffusionFusionPassTests
{
    private readonly DiffusionFusionPass _pass = new();
    private readonly CpuEngine _engine = new();

    // ── Helpers ──────────────────────────────────────────────────────────

    private CompiledStep<float> MakeGroupNormStep(
        Tensor<float> input, int numGroups, Tensor<float> gamma, Tensor<float> beta,
        double eps, out Tensor<float> output)
    {
        output = new Tensor<float>(input._shape);
        var capturedInput = input;
        var capturedGamma = gamma;
        var capturedBeta = beta;
        return new CompiledStep<float>(
            "GroupNorm",
            (eng, o) =>
            {
                var r = eng.GroupNorm(capturedInput, numGroups, capturedGamma, capturedBeta, eps, out _, out _);
                r.AsSpan().CopyTo(o.AsWritableSpan());
            },
            output,
            new[] { input, gamma, beta },
            savedState: new object[] { numGroups, null!, null!, eps });
    }

    private CompiledStep<float> MakeSwishStep(Tensor<float> input, out Tensor<float> output)
    {
        output = new Tensor<float>(input._shape);
        var capturedInput = input;
        return new CompiledStep<float>(
            "Swish",
            (eng, o) => { var r = eng.Swish(capturedInput); r.AsSpan().CopyTo(o.AsWritableSpan()); },
            output,
            new[] { input });
    }

    private CompiledStep<float> MakeConv2DStep(
        Tensor<float> input, Tensor<float> kernel, int stride, int padding, int dilation,
        out Tensor<float> output)
    {
        int n = input._shape[0], cout = kernel._shape[0];
        int hOut = (input._shape[2] + 2 * padding - dilation * (kernel._shape[2] - 1) - 1) / stride + 1;
        int wOut = (input._shape[3] + 2 * padding - dilation * (kernel._shape[3] - 1) - 1) / stride + 1;
        output = new Tensor<float>(new[] { n, cout, hOut, wOut });
        var capturedInput = input; var capturedKernel = kernel;
        int s = stride, p = padding, d = dilation;
        return new CompiledStep<float>(
            "Conv2D",
            (eng, o) => { var r = eng.Conv2D(capturedInput, capturedKernel, s, p, d); r.AsSpan().CopyTo(o.AsWritableSpan()); },
            output,
            new[] { input, kernel },
            savedState: new object[] { new[] { stride, stride }, new[] { padding, padding }, new[] { dilation, dilation } });
    }

    private CompiledStep<float> MakeBroadcastAddStep(Tensor<float> a, Tensor<float> b, out Tensor<float> output)
    {
        output = new Tensor<float>(a._shape);
        var ca = a; var cb = b;
        return new CompiledStep<float>(
            "TensorBroadcastAdd",
            (eng, o) => { var r = eng.TensorBroadcastAdd(ca, cb); r.AsSpan().CopyTo(o.AsWritableSpan()); },
            output,
            new[] { a, b });
    }

    private CompiledStep<float> MakeAddStep(Tensor<float> a, Tensor<float> b, out Tensor<float> output)
    {
        output = new Tensor<float>(a._shape);
        var ca = a; var cb = b;
        return new CompiledStep<float>(
            "TensorAdd",
            (eng, o) => { var r = eng.TensorAdd(ca, cb); r.AsSpan().CopyTo(o.AsWritableSpan()); },
            output,
            new[] { a, b });
    }

    // ── Pattern 11: GroupNorm + Swish → FusedGroupNormActivation{SiLU} ──

    [Fact]
    public void Pattern11_MatchesGroupNormPlusSwish()
    {
        var input = Tensor<float>.CreateRandom([2, 8, 4, 4]);
        var gamma = Tensor<float>.CreateRandom([8]);
        var beta  = Tensor<float>.CreateRandom([8]);

        var gnStep = MakeGroupNormStep(input, 4, gamma, beta, 1e-5, out var gnOut);
        var swishStep = MakeSwishStep(gnOut, out _);

        var steps = new[] { gnStep, swishStep };
        var optimized = _pass.TryOptimize(steps, _engine);

        Assert.NotNull(optimized);
        Assert.Single(optimized!); // 2 steps → 1 fused step
        Assert.Equal("FusedGroupNormActivation", optimized[0].OpName);
    }

    [Fact]
    public void Pattern11_FusedOutputMatchesSeparate()
    {
        var input = Tensor<float>.CreateRandom([1, 4, 6, 6]);
        var gamma = Tensor<float>.CreateRandom([4]);
        var beta  = Tensor<float>.CreateRandom([4]);

        // Separate: GroupNorm → Swish
        var gnStep = MakeGroupNormStep(input, 2, gamma, beta, 1e-5, out var gnOut);
        var swishStep = MakeSwishStep(gnOut, out var separateOut);

        gnStep.Execute(_engine, gnStep.OutputBuffer);
        swishStep.Execute(_engine, swishStep.OutputBuffer);
        var separateData = separateOut.AsSpan().ToArray();

        // Fused
        var steps = new[] { gnStep, swishStep };
        var optimized = _pass.TryOptimize(steps, _engine)!;
        optimized[0].Execute(_engine, optimized[0].OutputBuffer);
        var fusedData = optimized[0].OutputBuffer.AsSpan().ToArray();

        Assert.Equal(separateData.Length, fusedData.Length);
        for (int i = 0; i < separateData.Length; i++)
            Assert.True(Math.Abs(separateData[i] - fusedData[i]) < 1e-4f,
                $"Pattern 11 mismatch at [{i}]: separate={separateData[i]}, fused={fusedData[i]}");
    }

    [Fact]
    public void Pattern11_DoesNotMisfire_WhenActivationIsNotSwish()
    {
        var input = Tensor<float>.CreateRandom([1, 4, 4]);
        var gamma = Tensor<float>.CreateRandom([4]);
        var beta  = Tensor<float>.CreateRandom([4]);

        var gnStep = MakeGroupNormStep(input, 2, gamma, beta, 1e-5, out var gnOut);
        // Follow with ReLU instead of Swish — should NOT match Pattern 11
        var reluStep = new CompiledStep<float>(
            "ReLU",
            (eng, o) => { var r = eng.ReLU(gnOut); r.AsSpan().CopyTo(o.AsWritableSpan()); },
            new Tensor<float>(input._shape),
            new[] { gnOut });

        var optimized = _pass.TryOptimize(new[] { gnStep, reluStep }, _engine);
        // Pattern 11 specifically matches Swish, not ReLU
        Assert.Null(optimized);
    }

    // ── Pattern 12: Conv2D + BroadcastAdd + Swish → FusedConv2DBiasActivation ─

    [Fact]
    public void Pattern12_MatchesConvBiasSwish()
    {
        var input  = Tensor<float>.CreateRandom([1, 1, 8, 8]);
        var kernel = Tensor<float>.CreateRandom([2, 1, 3, 3]);
        var bias   = Tensor<float>.CreateRandom([1, 2, 1, 1]); // Already shaped for broadcast

        var convStep = MakeConv2DStep(input, kernel, 1, 0, 1, out var convOut);
        var addStep  = MakeBroadcastAddStep(convOut, bias, out var addOut);
        var swishStep = MakeSwishStep(addOut, out _);

        var steps = new[] { convStep, addStep, swishStep };
        var optimized = _pass.TryOptimize(steps, _engine);

        Assert.NotNull(optimized);
        Assert.Single(optimized!); // 3 steps → 1 fused
        Assert.Equal("FusedConv2DBiasActivation", optimized[0].OpName);
    }

    [Fact]
    public void Pattern12_DoesNotMisfire_WhenChainIsBroken()
    {
        var input  = Tensor<float>.CreateRandom([1, 1, 8, 8]);
        var kernel = Tensor<float>.CreateRandom([2, 1, 3, 3]);

        var convStep = MakeConv2DStep(input, kernel, 1, 0, 1, out var convOut);
        // Swish directly after Conv (no BroadcastAdd) — should NOT match Pattern 12
        var swishStep = MakeSwishStep(convOut, out _);

        var optimized = _pass.TryOptimize(new[] { convStep, swishStep }, _engine);
        // Pattern 12 requires 3 steps; Pattern 11 doesn't match either (Conv != GroupNorm)
        Assert.Null(optimized);
    }

    // ── Pattern 14: Add + GroupNorm → FusedAddGroupNorm ──────────────────

    [Fact]
    public void Pattern14_MatchesAddPlusGroupNorm()
    {
        var a     = Tensor<float>.CreateRandom([1, 4, 6, 6]);
        var b     = Tensor<float>.CreateRandom([1, 4, 6, 6]);
        var gamma = Tensor<float>.CreateRandom([4]);
        var beta  = Tensor<float>.CreateRandom([4]);

        var addStep = MakeAddStep(a, b, out var addOut);
        var gnStep  = MakeGroupNormStep(addOut, 2, gamma, beta, 1e-5, out _);

        var optimized = _pass.TryOptimize(new[] { addStep, gnStep }, _engine);

        Assert.NotNull(optimized);
        Assert.Single(optimized!);
        Assert.Equal("FusedAddGroupNorm", optimized[0].OpName);
    }

    [Fact]
    public void Pattern14_FusedOutputMatchesSeparate()
    {
        var a     = Tensor<float>.CreateRandom([1, 4, 4, 4]);
        var b     = Tensor<float>.CreateRandom([1, 4, 4, 4]);
        var gamma = Tensor<float>.CreateRandom([4]);
        var beta  = Tensor<float>.CreateRandom([4]);

        // Separate: Add → GroupNorm
        var addStep = MakeAddStep(a, b, out var addOut);
        var gnStep  = MakeGroupNormStep(addOut, 2, gamma, beta, 1e-5, out var gnOut);

        addStep.Execute(_engine, addStep.OutputBuffer);
        gnStep.Execute(_engine, gnStep.OutputBuffer);
        var separateData = gnOut.AsSpan().ToArray();

        // Fused
        var steps = new[] { addStep, gnStep };
        var optimized = _pass.TryOptimize(steps, _engine)!;
        optimized[0].Execute(_engine, optimized[0].OutputBuffer);
        var fusedData = optimized[0].OutputBuffer.AsSpan().ToArray();

        Assert.Equal(separateData.Length, fusedData.Length);
        for (int i = 0; i < separateData.Length; i++)
            Assert.True(Math.Abs(separateData[i] - fusedData[i]) < 1e-4f,
                $"Pattern 14 mismatch at [{i}]: separate={separateData[i]}, fused={fusedData[i]}");
    }

    // ── No fusion when steps are independent ────────────────────────────

    [Fact]
    public void NoFusion_WhenStepsAreIndependent()
    {
        var a = Tensor<float>.CreateRandom([2, 3]);
        var b = Tensor<float>.CreateRandom([2, 3]);

        var step1 = new CompiledStep<float>(
            "TensorAdd", (eng, o) => { }, new Tensor<float>(a._shape), new[] { a, b });
        var step2 = new CompiledStep<float>(
            "TensorMultiply", (eng, o) => { }, new Tensor<float>(a._shape), new[] { a, b });

        var optimized = _pass.TryOptimize(new[] { step1, step2 }, _engine);
        Assert.Null(optimized); // No diffusion patterns in this sequence
    }

    // ── Pass returns null for non-float types ───────────────────────────

    [Fact]
    public void NoFusion_ForDoubleType()
    {
        var a = Tensor<double>.CreateRandom([2, 3]);
        var step = new CompiledStep<double>(
            "GroupNorm", (eng, o) => { }, new Tensor<double>(a._shape), new[] { a });

        var optimized = _pass.TryOptimize(new[] { step }, _engine);
        Assert.Null(optimized);
    }
}
