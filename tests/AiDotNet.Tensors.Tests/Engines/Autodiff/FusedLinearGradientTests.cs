using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// Verifies that fused Linear+Activation ops produce identical gradients
/// to the equivalent unfused (separate MatMul + Add + Activation) ops.
/// </summary>
public class FusedLinearGradientTests : IDisposable
{
    // PR #333's GPU auto-detect ModuleInitializer flips AiDotNetEngine.Current
    // to DirectGpuTensorEngine on GPU machines. The fused-vs-unfused gradient
    // identity exercised here only holds on the CPU engine where the fused
    // kernel and the unfused composition share the same accumulator order.
    // Pin to CPU so this test isn't a host-config flake.
    private readonly IEngine _priorEngine = AiDotNetEngine.Current;
    private readonly IEngine _engine;
    public FusedLinearGradientTests()
    {
        AiDotNetEngine.Current = new CpuEngine();
        _engine = AiDotNetEngine.Current;
    }
    public void Dispose() { AiDotNetEngine.Current = _priorEngine; }

    private static Tensor<float> CreateRandom(int[] shape, int seed)
    {
        var rng = new Random(seed);
        var tensor = new Tensor<float>(shape);
        var span = tensor.AsWritableSpan();
        for (int i = 0; i < span.Length; i++)
            span[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        return tensor;
    }

    private static Tensor<double> CreateRandomD(int[] shape, int seed)
    {
        var rng = new Random(seed);
        var tensor = new Tensor<double>(shape);
        var span = tensor.AsWritableSpan();
        for (int i = 0; i < span.Length; i++)
            span[i] = rng.NextDouble() * 2.0 - 1.0;
        return tensor;
    }

    // fp64 FusedLinear (no activation) backward must take the transposed-GEMM fast
    // path (BlasProvider transA/transB), NOT the engine fallback that materialises
    // full weight/input transposes via TensorTranspose → Contiguous() — the latter
    // OOMs on large in-features (ResNet50/fp64 crashed in ComputeGradientsStreaming
    // there). This verifies the double path produces the SAME gradients as the
    // unfused (MatMul + BroadcastAdd) reference. A wide in-features dim exercises the
    // dimension whose transpose copy is the OOM risk.
    [Fact]
    public void FusedLinear_Double_MatchesUnfusedGradients()
    {
        const int M = 4, K = 512, N = 64;
        var input = CreateRandomD([M, K], 42);
        var weight = CreateRandomD([K, N], 43);
        var bias = CreateRandomD([N], 44);

        System.Collections.Generic.Dictionary<Tensor<double>, Tensor<double>> unfusedGrads, fusedGrads;
        using (var tape = new GradientTape<double>())
        {
            var linear = _engine.TensorMatMul(input, weight);
            var biased = _engine.TensorBroadcastAdd(linear, bias);
            var loss = _engine.ReduceSum(biased, new[] { 0, 1 }, keepDims: false);
            unfusedGrads = tape.ComputeGradients(loss, new[] { input, weight, bias });
        }
        using (var tape = new GradientTape<double>())
        {
            var fused = _engine.FusedLinear(input, weight, bias, FusedActivationType.None);
            var loss = _engine.ReduceSum(fused, new[] { 0, 1 }, keepDims: false);
            fusedGrads = tape.ComputeGradients(loss, new[] { input, weight, bias });
        }

        foreach (var param in new[] { input, weight, bias })
        {
            Assert.True(fusedGrads.ContainsKey(param), "Fused gradient missing for parameter");
            var u = unfusedGrads[param];
            var f = fusedGrads[param];
            Assert.Equal(u.Length, f.Length);
            for (int i = 0; i < u.Length; i++)
                Assert.True(Math.Abs(u[i] - f[i]) < 1e-9,
                    $"double FusedLinear grad mismatch at [{i}]: unfused={u[i]:R} fused={f[i]:R}");
        }
    }

    private void VerifyFusedMatchesUnfused(
        Func<IEngine, Tensor<float>, Tensor<float>, Tensor<float>, Tensor<float>> fusedOp,
        Func<IEngine, Tensor<float>, Tensor<float>> activationOp,
        int batchSize = 4, int inFeatures = 8, int outFeatures = 6)
    {
        var input = CreateRandom([batchSize, inFeatures], 42);
        var weight = CreateRandom([inFeatures, outFeatures], 43);
        var bias = CreateRandom([outFeatures], 44);

        // Unfused: separate ops
        Tensor<float> unfusedOutput;
        Dictionary<Tensor<float>, Tensor<float>> unfusedGrads;
        using (var tape = new GradientTape<float>())
        {
            var linear = _engine.TensorMatMul(input, weight);
            var biased = _engine.TensorBroadcastAdd(linear, bias);
            unfusedOutput = activationOp(_engine, biased);
            var loss = _engine.ReduceSum(unfusedOutput, [0, 1], keepDims: false);
            unfusedGrads = tape.ComputeGradients(loss, [input, weight, bias]);
        }

        // Fused: single op
        Tensor<float> fusedOutput;
        Dictionary<Tensor<float>, Tensor<float>> fusedGrads;
        using (var tape = new GradientTape<float>())
        {
            fusedOutput = fusedOp(_engine, input, weight, bias);
            var loss = _engine.ReduceSum(fusedOutput, [0, 1], keepDims: false);
            fusedGrads = tape.ComputeGradients(loss, [input, weight, bias]);
        }

        // Verify forward output matches within float32 tolerance.
        // BLAS GEMM with different accumulation order (fused vs unfused) can diverge
        // by up to K * machine_epsilon (~32 * 1.2e-7 ≈ 4e-6 for K=32).
        Assert.Equal(unfusedOutput.Length, fusedOutput.Length);
        for (int i = 0; i < unfusedOutput.Length; i++)
        {
            double diff = Math.Abs((double)unfusedOutput[i] - (double)fusedOutput[i]);
            Assert.True(diff < 1e-5,
                $"Forward output [{i}]: unfused={unfusedOutput[i]:R} fused={fusedOutput[i]:R} diff={diff:E3}");
        }

        // Verify gradients match for each parameter
        foreach (var param in new[] { input, weight, bias })
        {
            Assert.True(unfusedGrads.ContainsKey(param), $"Unfused gradient missing for parameter");
            Assert.True(fusedGrads.ContainsKey(param), $"Fused gradient missing for parameter");

            var unfusedGrad = unfusedGrads[param];
            var fusedGrad = fusedGrads[param];
            Assert.Equal(unfusedGrad.Length, fusedGrad.Length);

            // Tolerance of 3 decimal places: fused and unfused paths may use different
            // sigmoid implementations (Padé vs MathF.Exp) causing 1-2 ULP differences
            // that propagate through the activation derivative chain.
            for (int i = 0; i < unfusedGrad.Length; i++)
                Assert.Equal(unfusedGrad[i], fusedGrad[i], 3);
        }
    }

    [Fact]
    public void FusedLinearReLU_ProducesIdenticalGradients()
    {
        VerifyFusedMatchesUnfused(
            (e, i, w, b) => e.FusedLinearReLU(i, w, b),
            (e, x) => e.ReLU(x));
    }

    [Fact]
    public void FusedLinearSigmoid_ProducesIdenticalGradients()
    {
        VerifyFusedMatchesUnfused(
            (e, i, w, b) => e.FusedLinearSigmoid(i, w, b),
            (e, x) => e.TensorSigmoid(x));
    }

    [Fact]
    public void FusedLinearTanh_ProducesIdenticalGradients()
    {
        VerifyFusedMatchesUnfused(
            (e, i, w, b) => e.FusedLinearTanh(i, w, b),
            (e, x) => e.Tanh(x));
    }

    [Fact]
    public void FusedLinearGELU_ProducesIdenticalGradients()
    {
        VerifyFusedMatchesUnfused(
            (e, i, w, b) => e.FusedLinearGELU(i, w, b),
            (e, x) => e.GELU(x));
    }

    [Fact]
    public void FusedLinearSwish_ProducesIdenticalGradients()
    {
        VerifyFusedMatchesUnfused(
            (e, i, w, b) => e.FusedLinearSwish(i, w, b),
            (e, x) => e.Swish(x));
    }

    [Theory]
    [InlineData(1, 4, 3)]
    [InlineData(8, 16, 12)]
    [InlineData(16, 32, 8)]
    public void FusedLinearReLU_DifferentSizes_ProducesIdenticalGradients(int batch, int inF, int outF)
    {
        VerifyFusedMatchesUnfused(
            (e, i, w, b) => e.FusedLinearReLU(i, w, b),
            (e, x) => e.ReLU(x), batch, inF, outF);
    }

    [Theory]
    [InlineData(1, 4, 3)]
    [InlineData(8, 16, 12)]
    public void FusedLinearGELU_DifferentSizes_ProducesIdenticalGradients(int batch, int inF, int outF)
    {
        VerifyFusedMatchesUnfused(
            (e, i, w, b) => e.FusedLinearGELU(i, w, b),
            (e, x) => e.GELU(x), batch, inF, outF);
    }
}
