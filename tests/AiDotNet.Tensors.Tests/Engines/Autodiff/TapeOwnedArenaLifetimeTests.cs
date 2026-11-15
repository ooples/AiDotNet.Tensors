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
    // rows, inner, cols: [rows, inner] x [inner, cols]. The second keeps a 64Ki-element output and a 128Ki-element
    // gradient, sizes the cross-arena pool always kept; the first sits in the Large Object Heap range below them.
    [Theory]
    [InlineData(128, 512, 64)]
    [InlineData(256, 512, 256)]
    public void TensorsKeptPastTheirTape_SurviveLaterTapes(int rows, int inner, int cols)
    {
        bool before = GradientTape<float>.EnableTapeOwnedArena;
        try
        {
            GradientTape<float>.EnableTapeOwnedArena = true;
            var engine = new CpuEngine();
            var rng = RandomHelper.CreateSeededRandom(5);
            var x = new Tensor<float>(new[] { rows, inner });
            for (int i = 0; i < x.Length; i++) x[i] = (float)rng.NextDouble();
            var w = new Tensor<float>(new[] { inner, cols });
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
                var x2 = new Tensor<float>(new[] { rows, inner });
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

    // The quarantine must not cost the reuse: once nothing references a step's buffers, later tapes rent them from the
    // cross-arena pool instead of allocating.
    [Fact]
    public void BuffersNothingReferences_AreReusedByLaterTapes()
    {
        bool before = GradientTape<float>.EnableTapeOwnedArena;
        try
        {
            GradientTape<float>.EnableTapeOwnedArena = true;
            TensorArena.ClearPersistentPool();
            var engine = new CpuEngine();
            var x = new Tensor<float>(new[] { 256, 512 });
            var w = new Tensor<float>(new[] { 512, 256 });
            for (int i = 0; i < x.Length; i++) x[i] = (i % 7) * 0.1f;
            for (int i = 0; i < w.Length; i++) w[i] = (i % 5) * 0.01f - 0.02f;
            void Step()
            {
                using var tape = new GradientTape<float>();
                var y = engine.ReLU(engine.TensorMatMul(x, w));
                tape.ComputeGradients(engine.ReduceSum(engine.TensorMultiply(y, y), null), new[] { w });
            }
            Step();
            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();
            long hitsBefore = TensorArena.PersistentReuseHits;
            Step();   // sweeps the first step's quarantine into the pool, then rents from it
            Step();
            Assert.True(TensorArena.PersistentReuseHits > hitsBefore,
                $"no cross-arena reuse after the first step's buffers were released (quarantined {TensorArena.QuarantinedCount})");
        }
        finally
        {
            GradientTape<float>.EnableTapeOwnedArena = before;
            TensorArena.ClearPersistentPool();
        }
    }
}
