using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

public class CompiledFp64LiveBackingTests
{
    private const int PooledLength = 262_145;

    [Fact]
    public void CompiledMatMul_UsesLivePoolPaddedFp64Output()
    {
        var engine = new CpuEngine();
        var left = new Tensor<double>(new[] { 2.0 }, new[] { 1, 1 });
        var rightData = new double[PooledLength];
        Fill(rightData, 3.0);
        var right = new Tensor<double>(rightData, new[] { 1, PooledLength });

        Tensor<double> output;
        ICompiledPlan<double> plan;
        using (var scope = GraphMode.Enable())
        {
            output = engine.TensorMatMul(left, right);
            plan = scope.CompileInference<double>(output, left._shape);
        }

        using (plan)
        {
            var replayed = plan.Execute();
            var live = replayed.GetLiveBackingArrayAllowingPaddingOrNull();

            Assert.NotNull(live);
            Assert.True(live!.Length > replayed.Length,
                "The regression requires a genuinely pool-padded output buffer.");
            Assert.Equal(6.0, replayed[0], precision: 12);
            Assert.Equal(6.0, replayed[PooledLength / 2], precision: 12);
            Assert.Equal(6.0, replayed[PooledLength - 1], precision: 12);
        }
    }

    [Fact]
    public void CompiledTraining_PoolPaddedFp64GradientUpdatesLiveParameter()
    {
        var engine = new CpuEngine();
        var input = new Tensor<double>(new[] { 1.0 }, new[] { 1, 1 });
        var weightData = new double[PooledLength];
        Fill(weightData, 0.5);
        var weight = new Tensor<double>(weightData, new[] { 1, PooledLength });
        var parameters = new[] { weight };

        using var scope = GraphMode.EnableTraining(parameters);
        var output = engine.TensorMatMul(input, weight);
        var lossTensor = engine.ReduceSum(output, null);
        using var plan = scope.CompileTraining(parameters, lossTensor);
        plan.ConfigureOptimizer(OptimizerType.SGD, 0.0001f, 0.9f, 0.999f, 1e-8f, 0f);

        var compiledPlan = Assert.IsType<CompiledTrainingPlan<double>>(plan);
        double loss = plan.Step()[0];

        Assert.True(IsFinite(loss));
        Assert.InRange(loss, PooledLength * 0.499, PooledLength * 0.501);
        Assert.NotNull(weight.Grad);
        var liveGradient = weight.Grad!.GetLiveBackingArrayAllowingPaddingOrNull();
        Assert.NotNull(liveGradient);
        Assert.True(liveGradient!.Length > weight.Grad.Length,
            "The regression requires a genuinely pool-padded gradient destination.");
        Assert.Equal(1.0, weight.Grad[0], precision: 12);
        Assert.Equal(1.0, compiledPlan.Gradients[0][0], precision: 12);
        Assert.InRange(weight[0], 0.49989, 0.49991);
        Assert.True(IsFinite(weight[PooledLength - 1]));
    }

    [Fact]
    public unsafe void Fp64SgdKernel_UpdatesLargeArrayIncludingScalarTail()
    {
        var parameter = new double[PooledLength];
        var gradient = new double[PooledLength];
        Fill(parameter, 0.5);
        Fill(gradient, 1.0);

        fixed (double* parameterPtr = parameter)
        fixed (double* gradientPtr = gradient)
        {
            FusedOptimizer.SgdUpdateSimd(parameterPtr, gradientPtr, parameter.Length, 0.0001);
        }

        Assert.InRange(parameter[0], 0.49989, 0.49991);
        Assert.InRange(parameter[PooledLength - 1], 0.49989, 0.49991);
    }

    [Fact]
    public void CompiledMatMul_NonContiguousFp64InputFallsBackWithExactResult()
    {
        var engine = new CpuEngine();
        var storage = new Tensor<double>(new[] { 1.0, 2.0, 3.0, 4.0 }, new[] { 2, 2 });
        var leftView = storage.Transpose(new[] { 1, 0 });
        var right = new Tensor<double>(new[] { 5.0, 6.0, 7.0, 8.0 }, new[] { 2, 2 });

        ICompiledPlan<double> plan;
        Tensor<double> output;
        using (var scope = GraphMode.Enable())
        {
            output = engine.TensorMatMul(leftView, right);
            plan = scope.CompileInference(output, leftView);
        }

        using (plan)
        {
            var actual = plan.Execute().ToArray();
            Assert.Equal(new[] { 26.0, 30.0, 38.0, 44.0 }, actual);
        }
    }

    [Fact]
    public void CompiledFp64GeluChain_PreservesGeluSemantics()
    {
        const int hiddenSize = 128;
        var engine = new CpuEngine();
        var input = new Tensor<double>(new[] { -1.0 }, new[] { 1, 1 });
        var w1Data = new double[hiddenSize];
        var w2Data = new double[hiddenSize];
        Fill(w1Data, 1.0);
        Fill(w2Data, 1.0);
        var w1 = new Tensor<double>(w1Data, new[] { 1, hiddenSize });
        var w2 = new Tensor<double>(w2Data, new[] { hiddenSize, 1 });
        var parameters = new[] { w1, w2 };

        using var scope = GraphMode.EnableTraining(parameters);
        var hidden = engine.TensorMatMul(input, w1);
        var activated = engine.GELU(hidden);
        var output = engine.TensorMatMul(activated, w2);
        var lossTensor = engine.ReduceSum(output, null);
        using var plan = scope.CompileTraining(parameters, lossTensor);
        plan.ConfigureOptimizer(OptimizerType.SGD, 0f, 0.9f, 0.999f, 1e-8f, 0f);

        double loss = plan.Step()[0];
        double expectedPerElement = 0.5 * -1.0 *
            (1.0 + Math.Tanh(Math.Sqrt(2.0 / Math.PI) * (-1.0 + 0.044715 * -1.0)));

        Assert.True(loss < 0.0, "GELU(-1) is negative; ReLU substitution would return zero.");
        Assert.Equal(hiddenSize * expectedPerElement, loss, precision: 8);
    }

    private static void Fill(double[] values, double value)
    {
        for (int i = 0; i < values.Length; i++)
        {
            values[i] = value;
        }
    }

    private static bool IsFinite(double value)
    {
        return !double.IsNaN(value) && !double.IsInfinity(value);
    }
}
