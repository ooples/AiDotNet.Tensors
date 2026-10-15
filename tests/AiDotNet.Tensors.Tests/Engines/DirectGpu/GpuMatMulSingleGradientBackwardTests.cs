using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// The GPU matmul backward when only ONE operand needs a gradient (a constant basis on the left, a data batch on the
/// right): that gradient as a single transpose-flag GEMM. The generic backward materialized both operands' transposes
/// and formed the gradient nothing reads as well (N-BEATS: 61 such matmuls, 122 transpose_2d launches per step).
/// </summary>
[Collection("VulkanGlobalState")]
public sealed class GpuMatMulSingleGradientBackwardTests : IClassFixture<DirectGpuTensorEngineTestFixture>
{
    private readonly DirectGpuTensorEngineTestFixture _fixture;

    public GpuMatMulSingleGradientBackwardTests(DirectGpuTensorEngineTestFixture fixture) => _fixture = fixture;

    private DirectGpuTensorEngine RequireGpu()
    {
        Skip.IfNot(_fixture.IsAvailable, "No DirectGpu backend (CUDA/OpenCL/HIP) available on this system.");
        return _fixture.Engine ?? throw new InvalidOperationException(
            "The DirectGpu fixture reported available but has no engine.", _fixture.InitializationException);
    }

    private static Tensor<float> Make(int rows, int cols, int seed, float scale = 1f)
    {
        var rng = new Random(seed);
        var data = new float[rows * cols];
        for (int i = 0; i < data.Length; i++) data[i] = (float)((rng.NextDouble() * 2 - 1) * scale);
        return new Tensor<float>(data, [rows, cols]);
    }

    private static void AssertClose(double[] expected, float[] actual, string what, double tolerance)
    {
        Assert.Equal(expected.Length, actual.Length);
        double scale = 1e-12;
        foreach (var v in expected) scale = Math.Max(scale, Math.Abs(v));
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(actual[i] - expected[i]) <= tolerance * scale,
                $"{what}[{i}] gpu={actual[i]:G9} expected={expected[i]:G9}");
    }

    [SkippableTheory]
    [InlineData(10, 3, 1024, false)]   // N-BEATS backcast basis: dTheta = Basisᵀ · dY
    [InlineData(5, 3, 256, false)]     // N-BEATS forecast basis
    [InlineData(512, 10, 1024, true)]  // N-BEATS first layer on the data batch: dW = dY · Xᵀ
    [InlineData(7, 13, 5, true)]       // ragged, below any tile size
    [InlineData(7, 13, 5, false)]
    public void SingleGradient2D_MatchesDoubleReference(int m, int k, int n, bool leftOperand)
    {
        var gpu = RequireGpu();
        var a = Make(m, k, 11);
        var b = Make(k, n, 22);
        var g = Make(m, n, 33);

        Skip.IfNot(gpu.TryMatMulSingleGradient2D(g, a, b, leftOperand, out var gradient),
            "A float backend with transposed GEMMs is required.");
        Assert.NotNull(gradient);
        var got = gradient?.ToArray() ?? Array.Empty<float>();

        var ah = a.ToArray(); var bh = b.ToArray(); var gh = g.ToArray();
        if (leftOperand)
        {
            Assert.Equal(new[] { m, k }, gradient?._shape);
            var expected = new double[m * k];   // dA = G · Bᵀ
            for (int i = 0; i < m; i++)
                for (int j = 0; j < k; j++)
                {
                    double s = 0;
                    for (int q = 0; q < n; q++) s += (double)gh[i * n + q] * bh[j * n + q];
                    expected[i * k + j] = s;
                }
            AssertClose(expected, got, "dA", 1e-4);
        }
        else
        {
            Assert.Equal(new[] { k, n }, gradient?._shape);
            var expected = new double[k * n];   // dB = Aᵀ · G
            for (int i = 0; i < k; i++)
                for (int j = 0; j < n; j++)
                {
                    double s = 0;
                    for (int q = 0; q < m; q++) s += (double)ah[q * k + i] * gh[q * n + j];
                    expected[i * n + j] = s;
                }
            AssertClose(expected, got, "dB", 1e-4);
        }
    }

    // loss = sum((Basis · Theta)^2) + sum((W · X)^2): Basis and X are constants, Theta and W the parameters.
    private static Tensor<float> Loss(IEngine e, Tensor<float> basis, Tensor<float> theta, Tensor<float> w, Tensor<float> x)
    {
        var y1 = e.TensorMatMul(basis, theta);
        var y2 = e.TensorMatMul(w, x);
        return e.TensorAdd(e.ReduceSum(e.TensorMultiply(y1, y1), null), e.ReduceSum(e.TensorMultiply(y2, y2), null));
    }

    private static (float[] dTheta, float[] dW) CpuReference(Tensor<float> basis, Tensor<float> theta, Tensor<float> w, Tensor<float> x)
    {
        using var tape = new GradientTape<float>();
        var grads = tape.ComputeGradients(Loss(new CpuEngine(), basis, theta, w, x), new[] { theta, w });
        return (grads[theta].ToArray(), grads[w].ToArray());
    }

    private static void AssertMatches(float[] expected, float[] actual, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        float scale = 1e-12f;
        foreach (var v in expected) scale = Math.Max(scale, Math.Abs(v));
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= 1e-4f * scale, $"{what}[{i}] cpu={expected[i]} gpu={actual[i]}");
    }

    [SkippableFact]
    public void ConstantOperandTape_OnGpu_MatchesCpu()
    {
        var gpu = RequireGpu();
        var basis = Make(10, 3, 1); var theta = Make(3, 96, 2);
        var w = Make(64, 10, 3, 0.3f); var x = Make(10, 96, 4);
        var (cpuTheta, cpuW) = CpuReference(basis, theta, w, x);

        using var tape = new GradientTape<float>();
        var grads = tape.ComputeGradients(Loss(gpu, basis, theta, w, x), new[] { theta, w });
        AssertMatches(cpuTheta, grads[theta].ToArray(), "dTheta");
        AssertMatches(cpuW, grads[w].ToArray(), "dW");
    }

    [SkippableFact]
    public void ConstantOperandCompiledStep_OnGpu_MatchesCpu()
    {
        var gpu = RequireGpu();
        var basis = Make(10, 3, 5); var x = Make(10, 96, 8);
        var thetaHost = Make(3, 96, 6); var wHost = Make(64, 10, 7, 0.3f);
        var (cpuTheta, cpuW) = CpuReference(basis, thetaHost, wHost, x);

        var previous = AiDotNetEngine.Current;
        AiDotNetEngine.Current = gpu;
        try
        {
            // Parameters on the device, as AiDotNet's fused path places them.
            var theta = thetaHost.Clone().Gpu();
            var w = wHost.Clone().Gpu();
            ICompiledTrainingPlan<float> plan;
            using (var scope = GraphMode.Enable())
            {
                Loss(gpu, basis, theta, w, x);
                plan = scope.CompileTraining(new[] { theta, w });
            }
            using (plan)
            {
                // Twice: the second step replays (captured where the backend supports it).
                for (int step = 0; step < 2; step++)
                {
                    plan.Step();
                    AssertMatches(cpuTheta, plan.Gradients[0].ToArray(), $"step {step} dTheta");
                    AssertMatches(cpuW, plan.Gradients[1].ToArray(), $"step {step} dW");
                }
            }
        }
        finally
        {
            AiDotNetEngine.Current = previous;
        }
    }
}