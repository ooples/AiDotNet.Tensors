using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// Tests for <see cref="GradientTape{T}.ComputeGradientsFromSeed"/> — the public seeded-backward entry
/// point that lets a tape-recorded forward be wrapped as a manual-backward layer (Forward records,
/// Backward(outputGradient) seeds the reverse pass at the layer output).
/// </summary>
public class GradientTapeSeededBackwardTests
{
    private readonly IEngine _engine = AiDotNetEngine.Current;

    private static float At(Tensor<float> t, int i) => t.AsWritableSpan()[i];

    [Fact]
    public void ComputeGradientsFromSeed_AppliesUpstreamGradient()
    {
        using var tape = new GradientTape<float>();
        var w = new Tensor<float>(new float[] { 2f, 3f, 4f }, [3]);
        var x = new Tensor<float>(new float[] { 5f, 6f, 7f }, [3]);
        var y = _engine.TensorMultiply(w, x);                             // y = w ⊙ x, recorded on the tape
        var seed = new Tensor<float>(new float[] { 1f, 10f, 100f }, [3]); // upstream gradient at y

        var grads = tape.ComputeGradientsFromSeed(y, seed, new[] { w, x });

        // Chain rule for y = w⊙x seeded by g: dL/dw = g⊙x, dL/dx = g⊙w.
        var gw = grads[w]; var gx = grads[x];
        Assert.Equal(1f * 5f, At(gw, 0), 3);
        Assert.Equal(10f * 6f, At(gw, 1), 3);
        Assert.Equal(100f * 7f, At(gw, 2), 3);
        Assert.Equal(1f * 2f, At(gx, 0), 3);
        Assert.Equal(10f * 3f, At(gx, 1), 3);
        Assert.Equal(100f * 4f, At(gx, 2), 3);
    }

    [Fact]
    public void ComputeGradientsFromSeed_OnesSeed_MatchesDefaultSumBackward()
    {
        var w = new Tensor<float>(new float[] { 2f, 3f }, [2]);
        var x = new Tensor<float>(new float[] { 5f, 6f }, [2]);

        // Default backward from the scalar loss sum(y): dL/dw = x.
        float d0, d1;
        using (var t1 = new GradientTape<float>())
        {
            var y = _engine.TensorMultiply(w, x);
            var loss = _engine.ReduceSum(y, null, false);
            var g = t1.ComputeGradients(loss, new[] { w });
            d0 = At(g[w], 0); d1 = At(g[w], 1);
        }

        // Seeding the reverse pass with ones at y is equivalent (d(sum(y))/dy = ones).
        float s0, s1;
        using (var t2 = new GradientTape<float>())
        {
            var y = _engine.TensorMultiply(w, x);
            var ones = new Tensor<float>(new float[] { 1f, 1f }, [2]);
            var g = t2.ComputeGradientsFromSeed(y, ones, new[] { w });
            s0 = At(g[w], 0); s1 = At(g[w], 1);
        }

        Assert.Equal(d0, s0, 4);
        Assert.Equal(d1, s1, 4);
    }

    [Fact]
    public void ComputeGradientsFromSeed_ShapeMismatch_Throws()
    {
        using var tape = new GradientTape<float>();
        var w = new Tensor<float>(new float[] { 1f, 2f }, [2]);
        var x = new Tensor<float>(new float[] { 3f, 4f }, [2]);
        var y = _engine.TensorMultiply(w, x);
        var badSeed = new Tensor<float>(new float[] { 1f, 2f, 3f }, [3]);
        Assert.Throws<ArgumentException>(() => tape.ComputeGradientsFromSeed(y, badSeed, new[] { w }));
    }

    [Fact]
    public void ComputeGradientsFromSeed_NullArgs_Throw()
    {
        using var tape = new GradientTape<float>();
        var w = new Tensor<float>(new float[] { 1f }, [1]);
        var y = _engine.TensorMultiplyScalar(w, 2f);
        Assert.Throws<ArgumentNullException>(() => tape.ComputeGradientsFromSeed(null!, y));
        Assert.Throws<ArgumentNullException>(() => tape.ComputeGradientsFromSeed(y, null!));
    }
}
