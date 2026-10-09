using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// A tensor a caller keeps after its tape is disposed - the output, the loss, a gradient - must keep its values
/// while later tapes run. A tape-owned arena recycles the buffers it handed out, so this is what decides whether it
/// can be on by default.
/// </summary>
[Collection("EngineCurrentGlobalState")]
public class TapeOwnedArenaLifetimeTests
{
    [Fact]
    public void TensorsKeptPastTheirTape_SurviveLaterTapes()
    {
        bool before = GradientTape<float>.EnableTapeOwnedArena;
        try
        {
            GradientTape<float>.EnableTapeOwnedArena = true;
            var engine = new CpuEngine();
            var rng = RandomHelper.CreateSeededRandom(5);
            var x = new Tensor<float>(new[] { 128, 512 });
            for (int i = 0; i < x.Length; i++) x[i] = (float)rng.NextDouble();
            var w = new Tensor<float>(new[] { 512, 64 });
            for (int i = 0; i < w.Length; i++) w[i] = (float)(rng.NextDouble() - 0.5);

            Tensor<float> keptOutput, keptGradient;
            using (var tape = new GradientTape<float>())
            {
                keptOutput = engine.ReLU(engine.TensorMatMul(x, w));
                var loss = engine.ReduceSum(engine.TensorMultiply(keptOutput, keptOutput), null);
                keptGradient = tape.ComputeGradients(loss, new[] { w })[w];
            }
            var outputValues = keptOutput.ToArray();
            var gradientValues = keptGradient.ToArray();

            // Later steps on other data, allocating the same shapes.
            for (int step = 0; step < 5; step++)
            {
                var x2 = new Tensor<float>(new[] { 128, 512 });
                for (int i = 0; i < x2.Length; i++) x2[i] = -1f - step;
                using var tape = new GradientTape<float>();
                var y = engine.ReLU(engine.TensorMatMul(x2, w));
                var loss = engine.ReduceSum(engine.TensorMultiply(y, y), null);
                tape.ComputeGradients(loss, new[] { w });
            }

            Assert.Equal(outputValues, keptOutput.ToArray());
            Assert.Equal(gradientValues, keptGradient.ToArray());
        }
        finally
        {
            GradientTape<float>.EnableTapeOwnedArena = before;
        }
    }
}
