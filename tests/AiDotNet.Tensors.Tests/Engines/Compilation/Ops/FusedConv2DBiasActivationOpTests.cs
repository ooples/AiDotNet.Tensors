using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.Compilation.Ops;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation.Ops;

/// <summary>
/// Tests for <see cref="FusedConv2DBiasActivationOp{T}"/>. Validates numerical
/// parity with the separate Conv + BroadcastAdd + Activation sequence, attribute
/// exposure for fusion passes, and factory round-trip.
/// </summary>
public class FusedConv2DBiasActivationOpTests
{
    // ── Forward parity: fused op matches Conv + Bias + Activation separately ─
    [Theory]
    [InlineData(FusedActivationType.None)]     // Conv+Bias+Identity
    [InlineData(FusedActivationType.ReLU)]     // Conv+Bias+ReLU
    [InlineData(FusedActivationType.Sigmoid)]  // Conv+Bias+Sigmoid
    [InlineData(FusedActivationType.Swish)]    // Conv+Bias+SiLU (diffusion pattern)
    public void Forward_MatchesSeparateConvBiasActivation(FusedActivationType activation)
    {
        var engine = new CpuEngine();

        // [1, 1, 8, 8] input, [2, 1, 3, 3] kernel, [2] bias
        var input  = Tensor<float>.CreateRandom([1, 1, 8, 8]);
        var kernel = Tensor<float>.CreateRandom([2, 1, 3, 3]);
        var bias   = Tensor<float>.CreateRandom([2]);
        int stride = 1, padding = 0, dilation = 1;

        // Fused path via Op
        var op = new FusedConv2DBiasActivationOp<float>(
            input, kernel, bias,
            stride, stride, padding, padding, dilation, dilation, activation);
        var fusedOutput = new Tensor<float>(op.OutputShape);
        op.BuildForwardClosure()(engine, fusedOutput);

        // Separate path: Conv → BroadcastAdd → Activation
        var convResult = engine.Conv2D(input, kernel, stride, padding, dilation);
        // Reshape bias to [1, Cout, 1, 1] for broadcasting
        var biasReshaped = bias.Reshape(new[] { 1, bias._shape[0], 1, 1 });
        var withBias = engine.TensorBroadcastAdd(convResult, biasReshaped);
        var separateResult = activation switch
        {
            FusedActivationType.None    => withBias,
            FusedActivationType.ReLU    => engine.ReLU(withBias),
            FusedActivationType.Sigmoid => engine.Sigmoid(withBias),
            FusedActivationType.Swish   => engine.Swish(withBias),
            _ => withBias,
        };

        // Compare element-wise — allow small floating-point tolerance since
        // the fused kernel may accumulate differently than 3 separate ops.
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

    // ── Forward without bias ────────────────────────────────────────────────
    [Fact]
    public void Forward_NoBias_ProducesNonZeroOutput()
    {
        var engine = new CpuEngine();
        var input  = Tensor<float>.CreateRandom([1, 1, 6, 6]);
        var kernel = Tensor<float>.CreateRandom([2, 1, 3, 3]);

        var op = new FusedConv2DBiasActivationOp<float>(
            input, kernel, bias: null, 1, 1, 0, 0);
        var output = new Tensor<float>(op.OutputShape);
        op.BuildForwardClosure()(engine, output);

        bool anyNonZero = false;
        var data = output.AsSpan();
        for (int i = 0; i < data.Length; i++)
            if (Math.Abs(data[i]) > 1e-8f) { anyNonZero = true; break; }
        Assert.True(anyNonZero, "No-bias fused conv produced all-zero output");
    }

    // ── Attributes for fusion pass introspection ────────────────────────────
    [Fact]
    public void Attributes_ExposedForFusionPatternMatching()
    {
        var input  = Tensor<float>.CreateRandom([1, 3, 16, 16]);
        var kernel = Tensor<float>.CreateRandom([8, 3, 3, 3]);
        var bias   = Tensor<float>.CreateRandom([8]);

        var op = new FusedConv2DBiasActivationOp<float>(
            input, kernel, bias, 2, 2, 1, 1, 1, 1, FusedActivationType.Swish);

        Assert.Equal("FusedConv2DBiasActivation", op.OpName);
        Assert.Equal(2, op.StrideH);
        Assert.Equal(2, op.StrideW);
        Assert.Equal(1, op.PadH);
        Assert.Equal(1, op.PadW);
        Assert.Equal(1, op.DilationH);
        Assert.Equal(1, op.DilationW);
        Assert.Equal(FusedActivationType.Swish, op.Activation);
        Assert.Same(input, op.Input);
        Assert.Same(kernel, op.Kernel);
        Assert.Same(bias, op.Bias);
        Assert.Equal(3, op.Inputs.Length); // input, kernel, bias
    }

    // ── OutputShape calculation ─────────────────────────────────────────────
    [Theory]
    [InlineData(8, 3, 1, 0, 1, 6)]  // standard 3×3 conv
    [InlineData(8, 3, 2, 1, 1, 4)]  // stride=2, pad=1
    [InlineData(8, 5, 1, 2, 1, 8)]  // 5×5 conv with pad=2 (same)
    public void OutputShape_CorrectForVariousConfigs(
        int inputSize, int kernelSize, int stride, int padding, int dilation, int expectedSize)
    {
        var input  = Tensor<float>.CreateRandom([1, 1, inputSize, inputSize]);
        var kernel = Tensor<float>.CreateRandom([1, 1, kernelSize, kernelSize]);

        var op = new FusedConv2DBiasActivationOp<float>(
            input, kernel, null, stride, stride, padding, padding, dilation, dilation);

        var shape = op.OutputShape;
        Assert.Equal(1, shape[0]); // batch
        Assert.Equal(1, shape[1]); // channels
        Assert.Equal(expectedSize, shape[2]); // H
        Assert.Equal(expectedSize, shape[3]); // W
    }

    // ── ToCompiledStep produces executable step ─────────────────────────────
    [Fact]
    public void ToCompiledStep_Executes()
    {
        var engine = new CpuEngine();
        var input  = Tensor<float>.CreateRandom([1, 1, 6, 6]);
        var kernel = Tensor<float>.CreateRandom([2, 1, 3, 3]);
        var bias   = Tensor<float>.CreateRandom([2]);

        var op = new FusedConv2DBiasActivationOp<float>(
            input, kernel, bias, 1, 1, 0, 0, 1, 1, FusedActivationType.ReLU);
        var outputBuffer = new Tensor<float>(op.OutputShape);
        var step = op.ToCompiledStep(outputBuffer);

        Assert.Equal("FusedConv2DBiasActivation", step.OpName);
        step.Execute(engine, step.OutputBuffer);

        bool anyNonZero = false;
        var data = outputBuffer.AsSpan();
        for (int i = 0; i < data.Length; i++)
            if (Math.Abs(data[i]) > 1e-8f) { anyNonZero = true; break; }
        Assert.True(anyNonZero, "CompiledStep produced all-zero output");
    }

    // ── TryFromStep round-trip ──────────────────────────────────────────────
    [Fact]
    public void TryFromStep_RoundTrip_PreservesAttributes()
    {
        var input  = Tensor<float>.CreateRandom([1, 3, 8, 8]);
        var kernel = Tensor<float>.CreateRandom([4, 3, 3, 3]);
        var bias   = Tensor<float>.CreateRandom([4]);

        var original = new FusedConv2DBiasActivationOp<float>(
            input, kernel, bias, 2, 2, 1, 1, 1, 1, FusedActivationType.Swish);
        var step = original.ToCompiledStep(new Tensor<float>(original.OutputShape));

        var recovered = FusedConv2DBiasActivationOp<float>.TryFromStep(step);
        Assert.NotNull(recovered);
        Assert.Equal(2, recovered!.StrideH);
        Assert.Equal(2, recovered.StrideW);
        Assert.Equal(1, recovered.PadH);
        Assert.Equal(FusedActivationType.Swish, recovered.Activation);
    }

    // ── TryFromStep returns null for non-matching ops ───────────────────────
    [Fact]
    public void TryFromStep_WrongOpName_ReturnsNull()
    {
        var t = Tensor<float>.CreateRandom([2, 3]);
        var step = new CompiledStep<float>("TensorMatMul", (e, o) => { }, t, new[] { t, t });
        Assert.Null(FusedConv2DBiasActivationOp<float>.TryFromStep(step));
    }

    // ── Argument validation ─────────────────────────────────────────────────
    [Fact]
    public void Constructor_NullInput_Throws()
    {
        Assert.Throws<ArgumentNullException>(() =>
            new FusedConv2DBiasActivationOp<float>(
                null!, Tensor<float>.CreateRandom([1, 1, 3, 3]), null, 1, 1, 0, 0));
    }

    [Fact]
    public void Constructor_NullKernel_Throws()
    {
        Assert.Throws<ArgumentNullException>(() =>
            new FusedConv2DBiasActivationOp<float>(
                Tensor<float>.CreateRandom([1, 1, 8, 8]), null!, null, 1, 1, 0, 0));
    }
}
