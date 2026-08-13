using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

[Collection("EngineCurrentGlobalState")]
public sealed class RoundGradientContractTests : IDisposable
{
    private readonly IEngine _priorEngine = AiDotNetEngine.Current;
    private readonly CpuEngine _engine = new();

    public RoundGradientContractTests() => AiDotNetEngine.Current = _engine;

    public void Dispose() => AiDotNetEngine.Current = _priorEngine;

    [Fact]
    public async Task EagerTape_UsesMathematicalZeroDerivativeAwayFromDiscontinuities()
    {
        await Task.Yield();

        var input = new Tensor<double>(
            new[] { -1.27, -0.73, -0.18, 0.22, 0.81, 1.34 },
            new[] { 6 });
        var projection = new Tensor<double>(
            new[] { 0.7, -0.5, 0.3, -0.2, 0.4, -0.6 },
            new[] { 6 });

        Tensor<double> gradient;
        using (var tape = new GradientTape<double>())
        {
            var rounded = _engine.TensorRound(input);
            var objective = _engine.ReduceSum(_engine.TensorMultiply(rounded, projection), null);
            gradient = tape.ComputeGradients(objective, new[] { input })[input];
        }

        for (int i = 0; i < gradient.Length; i++)
            Assert.Equal(0.0, gradient[i]);
    }
}
