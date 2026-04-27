using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// Regression tests for issue #255 — TensorEmbeddingLookup was not registered
/// on the GradientTape, so dL/dE silently dropped to zero whenever the
/// embedding table was trainable. Surfaces in AiDotNet#1208 as a Transformer
/// converging to a uniform output across all input tokens.
/// </summary>
public class TensorEmbeddingLookupTapeTests
{
    private readonly IEngine _engine = AiDotNetEngine.Current;

    [Fact]
    public void TensorEmbeddingLookup_IntIndices_ProducesNonZeroEmbeddingGradient()
    {
        const int vocab = 8, dim = 16, batch = 4;
        var E = new Tensor<float>([vocab, dim]);
        var eSpan = E.AsWritableSpan();
        for (int i = 0; i < eSpan.Length; i++) eSpan[i] = 0.01f * (i + 1);

        var indices = new Tensor<int>([batch]);
        indices[0] = 0; indices[1] = 1; indices[2] = 2; indices[3] = 3;

        using var tape = new GradientTape<float>();
        var output = _engine.TensorEmbeddingLookup<float, int>(E, indices);
        Assert.Equal(new[] { batch, dim }, output._shape);

        var loss = _engine.ReduceSum(output, null);
        var grads = tape.ComputeGradients(loss, sources: new[] { E });

        Assert.True(grads.ContainsKey(E), "Expected gradient on embedding table E");
        Assert.Equal(E._shape, grads[E]._shape);

        // Selected rows (0..3) should have grad = 1; un-selected (4..7) stay 0.
        var gSpan = grads[E].AsSpan();
        for (int row = 0; row < vocab; row++)
            for (int col = 0; col < dim; col++)
            {
                float expected = row < batch ? 1f : 0f;
                float actual = gSpan[row * dim + col];
                Assert.Equal(expected, actual, 5);
            }
    }

    [Fact]
    public void TensorEmbeddingLookup_RepeatedIndices_AccumulateGrad()
    {
        const int vocab = 4, dim = 3;
        var E = new Tensor<float>([vocab, dim]);
        var indices = new Tensor<int>([5]);
        indices[0] = 1; indices[1] = 1; indices[2] = 1; indices[3] = 0; indices[4] = 2;

        using var tape = new GradientTape<float>();
        var output = _engine.TensorEmbeddingLookup<float, int>(E, indices);
        var loss = _engine.ReduceSum(output, null);
        var grads = tape.ComputeGradients(loss, sources: new[] { E });

        var gSpan = grads[E].AsSpan();
        // Row 0 chosen 1 time, row 1 chosen 3 times, row 2 chosen 1 time, row 3 not chosen.
        for (int c = 0; c < dim; c++)
        {
            Assert.Equal(1f, gSpan[0 * dim + c], 5);
            Assert.Equal(3f, gSpan[1 * dim + c], 5);
            Assert.Equal(1f, gSpan[2 * dim + c], 5);
            Assert.Equal(0f, gSpan[3 * dim + c], 5);
        }
    }

    [Fact]
    public void TensorEmbeddingLookup_Rank2Indices_PreservesOutputShape()
    {
        const int vocab = 6, dim = 8;
        var E = new Tensor<float>([vocab, dim]);
        var eSpan = E.AsWritableSpan();
        for (int i = 0; i < eSpan.Length; i++) eSpan[i] = 0.1f;

        var indices = new Tensor<int>([2, 3]);
        for (int i = 0; i < 6; i++) indices.AsWritableSpan()[i] = i % vocab;

        using var tape = new GradientTape<float>();
        var output = _engine.TensorEmbeddingLookup<float, int>(E, indices);
        Assert.Equal(new[] { 2, 3, dim }, output._shape);

        var loss = _engine.ReduceSum(output, null);
        var grads = tape.ComputeGradients(loss, sources: new[] { E });

        Assert.Equal(E._shape, grads[E]._shape);
    }

    [Fact]
    public void TensorEmbeddingLookup_DoubleEmbeddings_AlsoTapeAware()
    {
        const int vocab = 5, dim = 4;
        var E = new Tensor<double>([vocab, dim]);
        var indices = new Tensor<int>([3]);
        indices[0] = 0; indices[1] = 2; indices[2] = 4;

        using var tape = new GradientTape<double>();
        var output = _engine.TensorEmbeddingLookup<double, int>(E, indices);
        var loss = _engine.ReduceSum(output, null);
        var grads = tape.ComputeGradients(loss, sources: new[] { E });

        Assert.True(grads.ContainsKey(E));
        Assert.Equal(E._shape, grads[E]._shape);

        var gSpan = grads[E].AsSpan();
        for (int row = 0; row < vocab; row++)
            for (int col = 0; col < dim; col++)
            {
                double expected = (row == 0 || row == 2 || row == 4) ? 1.0 : 0.0;
                Assert.Equal(expected, gSpan[row * dim + col], 9);
            }
    }

    // ─────────────────────────────────────────────────────────────────────
    // #255 audit follow-ups: same "missing tape registration" defect on
    // ops with existing engine backward kernels. Catches regressions if
    // anyone strips the DifferentiableOps.Record* call.
    // ─────────────────────────────────────────────────────────────────────

    [Fact]
    public void AvgPool2D_IntArrayOverload_ProducesNonZeroInputGradient()
    {
        var input = new Tensor<float>([1, 1, 4, 4]);
        var iSpan = input.AsWritableSpan();
        for (int i = 0; i < iSpan.Length; i++) iSpan[i] = i + 1;

        using var tape = new GradientTape<float>();
        var output = _engine.AvgPool2D(input, new[] { 2, 2 }, new[] { 2, 2 });
        Assert.Equal(new[] { 1, 1, 2, 2 }, output._shape);

        var loss = _engine.ReduceSum(output, null);
        var grads = tape.ComputeGradients(loss, sources: new[] { input });

        Assert.True(grads.ContainsKey(input));
        Assert.Equal(input._shape, grads[input]._shape);
        // Each input contributes 1/4 to one output cell under loss = sum.
        var gSpan = grads[input].AsSpan();
        for (int i = 0; i < gSpan.Length; i++)
            Assert.Equal(0.25f, gSpan[i], 5);
    }

    [Fact]
    public void ScatterAdd_FourArgOverload_RoutesGradToInputAndValues()
    {
        var input = new Tensor<float>([4, 3]);
        for (int i = 0; i < input.Length; i++) input.AsWritableSpan()[i] = 0.1f * i;
        var values = new Tensor<float>([2, 3]);
        for (int i = 0; i < values.Length; i++) values.AsWritableSpan()[i] = 1f;
        var indices = new Tensor<int>([2]);
        indices[0] = 0; indices[1] = 2;

        using var tape = new GradientTape<float>();
        var output = _engine.ScatterAdd(input, indices, values, axis: 0);
        var loss = _engine.ReduceSum(output, null);
        var grads = tape.ComputeGradients(loss, sources: new[] { input, values });

        Assert.True(grads.ContainsKey(input));
        Assert.True(grads.ContainsKey(values));
        // dL/dInput = 1 everywhere (passthrough).
        var giSpan = grads[input].AsSpan();
        for (int i = 0; i < giSpan.Length; i++) Assert.Equal(1f, giSpan[i], 5);
        // dL/dValues = 1 everywhere (each value contributes once).
        var gvSpan = grads[values].AsSpan();
        for (int i = 0; i < gvSpan.Length; i++) Assert.Equal(1f, gvSpan[i], 5);
    }

    [Fact]
    public void TensorWhere_BoolCondition_RoutesGradToSelectedBranch()
    {
        var x = new Tensor<float>([4]);
        var y = new Tensor<float>([4]);
        var cond = new Tensor<bool>([4]);
        for (int i = 0; i < 4; i++)
        {
            x.AsWritableSpan()[i] = 1f;
            y.AsWritableSpan()[i] = 2f;
            cond.AsWritableSpan()[i] = (i % 2 == 0);
        }

        using var tape = new GradientTape<float>();
        var output = _engine.TensorWhere(cond, x, y);
        var loss = _engine.ReduceSum(output, null);
        var grads = tape.ComputeGradients(loss, sources: new[] { x, y });

        Assert.True(grads.ContainsKey(x));
        Assert.True(grads.ContainsKey(y));
        var gxSpan = grads[x].AsSpan();
        var gySpan = grads[y].AsSpan();
        for (int i = 0; i < 4; i++)
        {
            // Even indices route to x, odd to y.
            Assert.Equal(i % 2 == 0 ? 1f : 0f, gxSpan[i], 5);
            Assert.Equal(i % 2 == 0 ? 0f : 1f, gySpan[i], 5);
        }
    }
}
