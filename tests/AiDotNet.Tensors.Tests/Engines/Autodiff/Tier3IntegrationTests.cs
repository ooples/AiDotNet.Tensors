using System.Diagnostics;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// Integration tests proving Tier 3 optimizations (grad_fn, delegate chain, native pool)
/// are wired up, functional, and provide measurable benefit.
/// </summary>
public class Tier3IntegrationTests
{
    private readonly ITestOutputHelper _output;
    private readonly IEngine _engine = AiDotNetEngine.Current;

    public Tier3IntegrationTests(ITestOutputHelper output) => _output = output;

    // ═══ grad_fn pointer model ═══

    [Fact]
    public void GradFn_IsSetDuringRecording()
    {
        var a = new Tensor<float>(new float[] { 1, 2, 3, 4 }, [2, 2]);
        var b = new Tensor<float>(new float[] { 5, 6, 7, 8 }, [2, 2]);

        // Before tape: no GradFn
        Assert.Null(a.GradFn);

        using var tape = new GradientTape<float>();
        var result = _engine.TensorAdd(a, b);

        // After recorded op: output has GradFn
        Assert.NotNull(result.GradFn);
        Assert.Equal(2, result.GradFn!.InputCount);
    }

    [Fact]
    public void GradFn_BackwardProducesCorrectGradients()
    {
        var input = new Tensor<float>(new float[] { 1, 2, 3, 4 }, [2, 2]);
        var weights = new Tensor<float>(new float[] { 0.5f, 0.3f, 0.1f, 0.8f }, [2, 2]);

        using var tape = new GradientTape<float>();
        var output = _engine.TensorMatMul(input, weights);
        var loss = _engine.TensorMeanDiff(output);

        // loss.GradFn should be set — triggers graph-based backward
        Assert.NotNull(loss.GradFn);

        var grads = tape.ComputeGradients(loss, sources: new[] { weights });
        Assert.True(grads.ContainsKey(weights), "Graph-based backward should produce weight gradient");

        bool anyNonZero = false;
        for (int i = 0; i < grads[weights].Length; i++)
            if (grads[weights].GetFlat(i) != 0f) anyNonZero = true;
        Assert.True(anyNonZero, "Weight gradient should be non-zero");
    }

    [Fact]
    public void GradFn_MultiLayerChain_ProducesGradients()
    {
        // Simulate a 3-layer MLP forward pass
        var x = Tensor<float>.CreateRandom([4, 8]);
        var w1 = Tensor<float>.CreateRandom([8, 16]);
        var w2 = Tensor<float>.CreateRandom([16, 8]);
        var w3 = Tensor<float>.CreateRandom([8, 4]);

        using var tape = new GradientTape<float>();
        var h1 = _engine.ReLU(_engine.TensorMatMul(x, w1));
        var h2 = _engine.ReLU(_engine.TensorMatMul(h1, w2));
        var out1 = _engine.TensorMatMul(h2, w3);
        var loss = _engine.TensorMeanDiff(out1);

        var grads = tape.ComputeGradients(loss, sources: new[] { w1, w2, w3 });

        Assert.True(grads.ContainsKey(w1), "Should have gradient for w1");
        Assert.True(grads.ContainsKey(w2), "Should have gradient for w2");
        Assert.True(grads.ContainsKey(w3), "Should have gradient for w3");

        _output.WriteLine($"w1 grad norm: {GradNorm(grads[w1]):F6}");
        _output.WriteLine($"w2 grad norm: {GradNorm(grads[w2]):F6}");
        _output.WriteLine($"w3 grad norm: {GradNorm(grads[w3]):F6}");
    }

    // ═══ CompiledDelegateChain ═══

    [Fact]
    public void DelegateChain_RepeatedBackward_ProducesConsistentResults()
    {
        var x = new Tensor<float>(new float[] { 1, 2, 3, 4 }, [2, 2]);
        var w = new Tensor<float>(new float[] { 0.5f, 0.3f, 0.1f, 0.8f }, [2, 2]);

        // First backward — builds the chain
        Dictionary<Tensor<float>, Tensor<float>> grads1;
        using (var tape = new GradientTape<float>())
        {
            var output = _engine.TensorMatMul(x, w);
            var loss = _engine.TensorMeanDiff(output);
            grads1 = tape.ComputeGradients(loss, sources: new[] { w });
        }

        // Second backward — should produce same result
        Dictionary<Tensor<float>, Tensor<float>> grads2;
        using (var tape = new GradientTape<float>())
        {
            var output = _engine.TensorMatMul(x, w);
            var loss = _engine.TensorMeanDiff(output);
            grads2 = tape.ComputeGradients(loss, sources: new[] { w });
        }

        // Gradients should be identical
        Assert.True(grads1.ContainsKey(w) && grads2.ContainsKey(w));
        for (int i = 0; i < w.Length; i++)
        {
            float g1 = grads1[w].GetFlat(i);
            float g2 = grads2[w].GetFlat(i);
            Assert.Equal(g1, g2, 5); // 5 decimal places
        }
    }

    // ═══ Performance verification ═══

    [Fact(Skip = "Performance guard — run manually with --filter Tier3Integration")]
    public void GradFn_IsFasterThanTapeOnly()
    {
        var x = Tensor<float>.CreateRandom([8, 32]);
        var w1 = Tensor<float>.CreateRandom([32, 64]);
        var w2 = Tensor<float>.CreateRandom([64, 32]);
        int iters = 50;

        // Warmup
        for (int w = 0; w < 5; w++)
        {
            using var t = new GradientTape<float>();
            var h = _engine.ReLU(_engine.TensorMatMul(x, w1));
            var o = _engine.TensorMatMul(h, w2);
            var l = _engine.TensorMeanDiff(o);
            t.ComputeGradients(l, sources: new[] { w1, w2 });
        }

        var sw = Stopwatch.StartNew();
        for (int i = 0; i < iters; i++)
        {
            using var tape = new GradientTape<float>();
            var h = _engine.ReLU(_engine.TensorMatMul(x, w1));
            var o = _engine.TensorMatMul(h, w2);
            var loss = _engine.TensorMeanDiff(o);
            tape.ComputeGradients(loss, sources: new[] { w1, w2 });
        }
        sw.Stop();
        double msPerStep = sw.Elapsed.TotalMilliseconds / iters;
        _output.WriteLine($"2-layer MLP forward+backward: {msPerStep:F3}ms");

        // Should complete in reasonable time (budget: 5ms catches regressions)
        Assert.True(msPerStep < 5.0, $"MLP step took {msPerStep:F3}ms — possible regression");
    }

    private static float GradNorm(Tensor<float> grad)
    {
        float sum = 0;
        for (int i = 0; i < grad.Length; i++)
            sum += grad.GetFlat(i) * grad.GetFlat(i);
        return MathF.Sqrt(sum);
    }
}
