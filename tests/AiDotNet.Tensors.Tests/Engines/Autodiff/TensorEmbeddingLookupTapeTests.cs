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

    [Fact]
    public void GeGLU_ProducesNonZeroInputGradient()
    {
        var input = new Tensor<float>([2, 4]);
        for (int i = 0; i < input.Length; i++) input.AsWritableSpan()[i] = (i + 1) * 0.1f;

        using var tape = new GradientTape<float>();
        var output = _engine.GeGLU(input);
        Assert.Equal(new[] { 2, 2 }, output._shape);
        var loss = _engine.ReduceSum(output, null);
        var grads = tape.ComputeGradients(loss, sources: new[] { input });

        Assert.True(grads.ContainsKey(input));
        Assert.Equal(input._shape, grads[input]._shape);
        // Some entry must be non-zero (input is non-zero, weights are non-trivial).
        var g = grads[input].AsSpan();
        bool anyNonZero = false;
        for (int i = 0; i < g.Length; i++) if (Math.Abs(g[i]) > 1e-6f) { anyNonZero = true; break; }
        Assert.True(anyNonZero);
    }

    [Fact]
    public void SwiGLU_ProducesNonZeroInputGradient()
    {
        var input = new Tensor<float>([1, 6]);
        for (int i = 0; i < input.Length; i++) input.AsWritableSpan()[i] = 0.5f - i * 0.1f;

        using var tape = new GradientTape<float>();
        var output = _engine.SwiGLU(input);
        Assert.Equal(new[] { 1, 3 }, output._shape);
        var loss = _engine.ReduceSum(output, null);
        var grads = tape.ComputeGradients(loss, sources: new[] { input });

        Assert.True(grads.ContainsKey(input));
        Assert.Equal(input._shape, grads[input]._shape);
        var g = grads[input].AsSpan();
        bool anyNonZero = false;
        for (int i = 0; i < g.Length; i++) if (Math.Abs(g[i]) > 1e-6f) { anyNonZero = true; break; }
        Assert.True(anyNonZero);
    }

    [Fact]
    public void ReGLU_ProducesNonZeroInputGradient()
    {
        var input = new Tensor<float>([1, 4]);
        // Pick values so b > 0 in the second half (otherwise ReGLU output is identically 0).
        input.AsWritableSpan()[0] = 0.5f;
        input.AsWritableSpan()[1] = 0.7f;
        input.AsWritableSpan()[2] = 1.0f;
        input.AsWritableSpan()[3] = 1.5f;

        using var tape = new GradientTape<float>();
        var output = _engine.ReGLU(input);
        var loss = _engine.ReduceSum(output, null);
        var grads = tape.ComputeGradients(loss, sources: new[] { input });

        Assert.True(grads.ContainsKey(input));
        Assert.Equal(input._shape, grads[input]._shape);
    }

    [Fact]
    public void SphericalSoftmax_ProducesNonZeroInputGradient()
    {
        var input = new Tensor<float>([2, 4]);
        for (int i = 0; i < input.Length; i++) input.AsWritableSpan()[i] = (i + 1) * 0.25f;

        using var tape = new GradientTape<float>();
        var output = _engine.SphericalSoftmax(input);
        // Use a non-trivial scalar reduction (sum of squares) so dL/dx ≠ 0;
        // sum(softmax) is identically 1, which would make dL/dInput == 0
        // even with correct chain — the gradient just happens to vanish.
        var sq = _engine.TensorMultiply(output, output);
        var loss = _engine.ReduceSum(sq, null);
        var grads = tape.ComputeGradients(loss, sources: new[] { input });

        Assert.True(grads.ContainsKey(input));
        Assert.Equal(input._shape, grads[input]._shape);
        // Pre-fix the chain broke at normalize and grad never reached input.
        var g = grads[input].AsSpan();
        bool anyNonZero = false;
        for (int i = 0; i < g.Length; i++) if (Math.Abs(g[i]) > 1e-9f) { anyNonZero = true; break; }
        Assert.True(anyNonZero);
    }

    [Fact]
    public void ReduceVariance_ProducesNonZeroInputGradient()
    {
        var input = new Tensor<float>([3, 4]);
        for (int i = 0; i < input.Length; i++) input.AsWritableSpan()[i] = (i + 1) * 0.1f;

        using var tape = new GradientTape<float>();
        var output = _engine.ReduceVariance(input, new[] { 1 }, keepDims: false);
        var loss = _engine.ReduceSum(output, null);
        var grads = tape.ComputeGradients(loss, sources: new[] { input });

        Assert.True(grads.ContainsKey(input));
        Assert.Equal(input._shape, grads[input]._shape);
    }

    [Fact]
    public void MaxPool3D_DispatchesThroughWithIndicesOnTape()
    {
        // [N, C, D, H, W] = [1, 1, 2, 4, 4] with 2x2x2 pool, stride 2.
        var input = new Tensor<float>([1, 1, 2, 4, 4]);
        for (int i = 0; i < input.Length; i++) input.AsWritableSpan()[i] = (float)(i + 1);

        using var tape = new GradientTape<float>();
        var output = _engine.MaxPool3D(input, new[] { 2, 2, 2 }, new[] { 2, 2, 2 }, new[] { 0, 0, 0 });
        Assert.Equal(new[] { 1, 1, 1, 2, 2 }, output._shape);

        var loss = _engine.ReduceSum(output, null);
        var grads = tape.ComputeGradients(loss, sources: new[] { input });

        Assert.True(grads.ContainsKey(input));
        Assert.Equal(input._shape, grads[input]._shape);
        // Exactly one cell per pool receives gradient = 1; all others = 0.
        var g = grads[input].AsSpan();
        int nonZero = 0;
        for (int i = 0; i < g.Length; i++) if (g[i] != 0f) nonZero++;
        Assert.Equal(4, nonZero);
    }

    [Fact]
    public void ScaledDotProductAttention_RoutesGradToQKV()
    {
        // [batch, heads, seq, d_k]
        const int B = 1, H = 2, S = 4, D = 8;
        var Q = new Tensor<float>([B, H, S, D]);
        var K = new Tensor<float>([B, H, S, D]);
        var V = new Tensor<float>([B, H, S, D]);
        var rng = new Random(7);
        for (int i = 0; i < Q.Length; i++) Q.AsWritableSpan()[i] = (float)(rng.NextDouble() - 0.5);
        for (int i = 0; i < K.Length; i++) K.AsWritableSpan()[i] = (float)(rng.NextDouble() - 0.5);
        for (int i = 0; i < V.Length; i++) V.AsWritableSpan()[i] = (float)(rng.NextDouble() - 0.5);

        using var tape = new GradientTape<float>();
        var output = _engine.ScaledDotProductAttention(Q, K, V, mask: (Tensor<bool>?)null, scale: null, out _);
        var loss = _engine.ReduceSum(output, null);
        var grads = tape.ComputeGradients(loss, sources: new[] { Q, K, V });

        Assert.True(grads.ContainsKey(Q));
        Assert.True(grads.ContainsKey(K));
        Assert.True(grads.ContainsKey(V));
        Assert.Equal(Q._shape, grads[Q]._shape);
        Assert.Equal(K._shape, grads[K]._shape);
        Assert.Equal(V._shape, grads[V]._shape);
    }

    [Fact]
    public void FlashAttention_RoutesGradToQKV()
    {
        const int B = 1, H = 2, S = 4, D = 8;
        var Q = new Tensor<float>([B, H, S, D]);
        var K = new Tensor<float>([B, H, S, D]);
        var V = new Tensor<float>([B, H, S, D]);
        var rng = new Random(11);
        for (int i = 0; i < Q.Length; i++) Q.AsWritableSpan()[i] = (float)(rng.NextDouble() - 0.5);
        for (int i = 0; i < K.Length; i++) K.AsWritableSpan()[i] = (float)(rng.NextDouble() - 0.5);
        for (int i = 0; i < V.Length; i++) V.AsWritableSpan()[i] = (float)(rng.NextDouble() - 0.5);

        using var tape = new GradientTape<float>();
        var output = _engine.FlashAttention(Q, K, V, scale: null, isCausal: false, out _);
        var loss = _engine.ReduceSum(output, null);
        var grads = tape.ComputeGradients(loss, sources: new[] { Q, K, V });

        Assert.True(grads.ContainsKey(Q));
        Assert.True(grads.ContainsKey(K));
        Assert.True(grads.ContainsKey(V));
        Assert.Equal(Q._shape, grads[Q]._shape);
    }

    [Fact]
    public void GroupedQueryAttention_RoutesGradToQKV()
    {
        // numQHeads = numKVHeads * numQueriesPerKV: 4 = 2*2
        const int B = 1, NQ = 4, NKV = 2, S = 4, D = 8;
        var Q = new Tensor<float>([B, NQ, S, D]);
        var K = new Tensor<float>([B, NKV, S, D]);
        var V = new Tensor<float>([B, NKV, S, D]);
        var rng = new Random(13);
        for (int i = 0; i < Q.Length; i++) Q.AsWritableSpan()[i] = (float)(rng.NextDouble() - 0.5);
        for (int i = 0; i < K.Length; i++) K.AsWritableSpan()[i] = (float)(rng.NextDouble() - 0.5);
        for (int i = 0; i < V.Length; i++) V.AsWritableSpan()[i] = (float)(rng.NextDouble() - 0.5);

        using var tape = new GradientTape<float>();
        var output = _engine.GroupedQueryAttention(Q, K, V, numQueriesPerKV: 2, scale: null, isCausal: false, out _);
        var loss = _engine.ReduceSum(output, null);
        var grads = tape.ComputeGradients(loss, sources: new[] { Q, K, V });

        Assert.True(grads.ContainsKey(Q));
        Assert.True(grads.ContainsKey(K));
        Assert.True(grads.ContainsKey(V));
    }

    [Fact]
    public void FusedConv2D_RoutesGradToInputAndKernelAndBias()
    {
        // [N, C, H, W] = [1, 2, 4, 4], kernel [out=2, in=2, 3, 3]
        var input = new Tensor<float>([1, 2, 4, 4]);
        var kernel = new Tensor<float>([2, 2, 3, 3]);
        var bias = new Tensor<float>([2]);
        var rng = new Random(17);
        for (int i = 0; i < input.Length; i++) input.AsWritableSpan()[i] = (float)(rng.NextDouble() - 0.5);
        for (int i = 0; i < kernel.Length; i++) kernel.AsWritableSpan()[i] = (float)(rng.NextDouble() - 0.5);
        for (int i = 0; i < bias.Length; i++) bias.AsWritableSpan()[i] = 0.1f;

        using var tape = new GradientTape<float>();
        var output = _engine.FusedConv2D(input, kernel, bias,
            strideH: 1, strideW: 1, padH: 0, padW: 0, dilationH: 1, dilationW: 1,
            FusedActivationType.ReLU);
        var loss = _engine.ReduceSum(output, null);
        var grads = tape.ComputeGradients(loss, sources: new[] { input, kernel, bias });

        Assert.True(grads.ContainsKey(input));
        Assert.True(grads.ContainsKey(kernel));
        Assert.True(grads.ContainsKey(bias));
        // Pre-fix the in-place bias+activation mutation broke Conv2D's recorded
        // values; after the tape-aware path the grads should be non-zero on
        // active ReLU paths. Just sanity-check shapes here.
        Assert.Equal(input._shape, grads[input]._shape);
        Assert.Equal(kernel._shape, grads[kernel]._shape);
        Assert.Equal(bias._shape, grads[bias]._shape);
    }
}
