using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// Tests that Tensor instance methods route through the engine for tape recording + BLAS.
/// Covers issue #104: Tensor.MatrixMultiply and other ops not recorded by GradientTape.
/// </summary>
public class TensorInstanceOpsTapeTests
{
    private readonly ITestOutputHelper _output;
    private readonly IEngine _engine = AiDotNetEngine.Current;

    public TensorInstanceOpsTapeTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void MatrixMultiply_RecordsOnTape()
    {
        var a = new Tensor<float>(new float[] { 1, 2, 3, 4 }, [2, 2]);
        var b = new Tensor<float>(new float[] { 5, 6, 7, 8 }, [2, 2]);

        using var tape = new GradientTape<float>();
        var result = a.MatrixMultiply(b);
        var loss = _engine.TensorMeanDiff(result);

        Assert.True(tape.EntryCount > 0, "MatrixMultiply should record on tape");
        var grads = tape.ComputeGradients(loss, sources: new[] { a, b });
        Assert.True(grads.ContainsKey(a), "Should have gradient for a");
        Assert.True(grads.ContainsKey(b), "Should have gradient for b");
    }

    [Fact]
    public void BroadcastAdd_ViaEngine_RecordsOnTape()
    {
        // Broadcast ops must go through engine for tape recording (engine calls tensor.BroadcastAdd internally)
        var a = new Tensor<float>(new float[] { 1, 2, 3, 4, 5, 6 }, [2, 3]);
        var b = new Tensor<float>(new float[] { 10, 20, 30 }, [1, 3]);

        using var tape = new GradientTape<float>();
        var result = _engine.TensorBroadcastAdd(a, b);
        var loss = _engine.TensorMeanDiff(result);

        Assert.True(tape.EntryCount > 0, "BroadcastAdd via engine should record on tape");
        var grads = tape.ComputeGradients(loss, sources: new[] { a, b });
        Assert.True(grads.ContainsKey(a));
        Assert.True(grads.ContainsKey(b));
    }

    [Fact]
    public void PointwiseMultiply_RecordsOnTape()
    {
        var a = new Tensor<float>(new float[] { 1, 2, 3, 4 }, [2, 2]);
        var b = new Tensor<float>(new float[] { 5, 6, 7, 8 }, [2, 2]);

        using var tape = new GradientTape<float>();
        var result = a.PointwiseMultiply(b);
        var loss = _engine.TensorMeanDiff(result);

        Assert.True(tape.EntryCount > 0, "PointwiseMultiply should record on tape");
        var grads = tape.ComputeGradients(loss, sources: new[] { a, b });
        Assert.True(grads.ContainsKey(a));
        Assert.True(grads.ContainsKey(b));
    }

    [Fact]
    public void Add_RecordsOnTape()
    {
        var a = new Tensor<float>(new float[] { 1, 2, 3, 4 }, [2, 2]);
        var b = new Tensor<float>(new float[] { 5, 6, 7, 8 }, [2, 2]);

        using var tape = new GradientTape<float>();
        var result = a.Add(b);
        var loss = _engine.TensorMeanDiff(result);

        Assert.True(tape.EntryCount > 0, "Add should record on tape");
        var grads = tape.ComputeGradients(loss, sources: new[] { a, b });
        Assert.True(grads.ContainsKey(a), "Add should have gradient for a");
        Assert.True(grads.ContainsKey(b), "Add should have gradient for b");
    }

    [Fact]
    public void Subtract_RecordsOnTape()
    {
        var a = new Tensor<float>(new float[] { 1, 2, 3, 4 }, [2, 2]);
        var b = new Tensor<float>(new float[] { 5, 6, 7, 8 }, [2, 2]);

        using var tape = new GradientTape<float>();
        var result = a.Subtract(b);
        var loss = _engine.TensorMeanDiff(result);

        Assert.True(tape.EntryCount > 0, "Subtract should record on tape");
        var grads = tape.ComputeGradients(loss, sources: new[] { a, b });
        Assert.True(grads.ContainsKey(a));
    }

    [Fact]
    public void ElementwiseMultiply_RecordsOnTape()
    {
        var a = new Tensor<float>(new float[] { 1, 2, 3, 4 }, [2, 2]);
        var b = new Tensor<float>(new float[] { 5, 6, 7, 8 }, [2, 2]);

        using var tape = new GradientTape<float>();
        var result = a.ElementwiseMultiply(b);
        var loss = _engine.TensorMeanDiff(result);

        Assert.True(tape.EntryCount > 0, "ElementwiseMultiply should record on tape");
        var grads = tape.ComputeGradients(loss, sources: new[] { a, b });
        Assert.True(grads.ContainsKey(a));
    }

    [Fact]
    public void BroadcastSubtract_ViaEngine_RecordsOnTape()
    {
        var a = new Tensor<float>(new float[] { 1, 2, 3, 4, 5, 6 }, [2, 3]);
        var b = new Tensor<float>(new float[] { 10, 20, 30 }, [1, 3]);

        using var tape = new GradientTape<float>();
        var result = _engine.TensorBroadcastSubtract(a, b);
        var loss = _engine.TensorMeanDiff(result);

        Assert.True(tape.EntryCount > 0, "BroadcastSubtract via engine should record on tape");
        var grads = tape.ComputeGradients(loss, sources: new[] { a, b });
        Assert.True(grads.ContainsKey(a));
    }

    [Fact]
    public void BroadcastMultiply_ViaEngine_RecordsOnTape()
    {
        var a = new Tensor<float>(new float[] { 1, 2, 3, 4, 5, 6 }, [2, 3]);
        var b = new Tensor<float>(new float[] { 2, 3, 4 }, [1, 3]);

        using var tape = new GradientTape<float>();
        var result = _engine.TensorBroadcastMultiply(a, b);
        var loss = _engine.TensorMeanDiff(result);

        Assert.True(tape.EntryCount > 0, "BroadcastMultiply via engine should record on tape");
        var grads = tape.ComputeGradients(loss, sources: new[] { a, b });
        Assert.True(grads.ContainsKey(a));
    }

    [Fact]
    public void FullTrainingStep_UsingTensorInstanceMethods()
    {
        // Simulate what RecurrentLayer does: tensor.MatrixMultiply + tensor.Add
        var input = new Tensor<float>(new float[] { 1, 2, 3, 4 }, [2, 2]);
        var weights = new Tensor<float>(new float[] { 0.5f, 0.3f, 0.1f, 0.8f }, [2, 2]);
        var bias = new Tensor<float>(new float[] { 0.1f, 0.2f, 0.3f, 0.4f }, [2, 2]);

        float w0Before = weights.GetFlat(0);

        using var tape = new GradientTape<float>();
        var hidden = input.MatrixMultiply(weights);
        var biased = hidden.Add(bias);
        var loss = _engine.TensorMeanDiff(biased);
        var grads = tape.ComputeGradients(loss, sources: new[] { weights });

        Assert.True(grads.ContainsKey(weights), "Should have weight gradient from tensor instance methods");

        // SGD update
        var wGrad = grads[weights];
        for (int i = 0; i < weights.Length; i++)
            weights.SetFlat(i, weights.GetFlat(i) - 0.01f * wGrad.GetFlat(i));

        Assert.NotEqual(w0Before, weights.GetFlat(0));
        _output.WriteLine($"Weight[0]: {w0Before:F6} -> {weights.GetFlat(0):F6}");
    }
}
