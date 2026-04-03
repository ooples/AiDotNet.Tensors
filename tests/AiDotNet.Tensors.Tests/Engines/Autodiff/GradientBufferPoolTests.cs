using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

public class GradientBufferPoolTests
{
    private readonly IEngine _engine = AiDotNetEngine.Current;

    private static Tensor<float> CreateRandom(int[] shape, int seed)
    {
        var rng = new Random(seed);
        var tensor = new Tensor<float>(shape);
        var span = tensor.AsWritableSpan();
        for (int i = 0; i < span.Length; i++)
            span[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        return tensor;
    }

    [Fact]
    public void PooledGradients_MatchNonPooledGradients()
    {
        var input = CreateRandom(new[] { 4, 8 }, 42);
        var weight = CreateRandom(new[] { 8, 6 }, 43);
        var bias = CreateRandom(new[] { 6 }, 44);

        // Non-pooled baseline
        Dictionary<Tensor<float>, Tensor<float>> baselineGrads;
        using (var tape = new GradientTape<float>())
        {
            var output = _engine.FusedLinearReLU(input, weight, bias);
            var loss = _engine.ReduceSum(output, new[] { 0, 1 }, keepDims: false);
            baselineGrads = tape.ComputeGradients(loss, new[] { input, weight, bias });
        }

        // Pooled
        var pool = new GradientBufferPool<float>();
        pool.Register(input);
        pool.Register(weight);
        pool.Register(bias);

        Dictionary<Tensor<float>, Tensor<float>> pooledGrads;
        using (var tape = new GradientTape<float>())
        {
            var output = _engine.FusedLinearReLU(input, weight, bias);
            var loss = _engine.ReduceSum(output, new[] { 0, 1 }, keepDims: false);
            pooledGrads = tape.ComputeGradients(loss, pool, new[] { input, weight, bias });
        }

        // Verify values match
        foreach (var param in new[] { input, weight, bias })
        {
            Assert.True(baselineGrads.ContainsKey(param));
            Assert.True(pooledGrads.ContainsKey(param));
            var baseline = baselineGrads[param];
            var pooled = pooledGrads[param];
            Assert.Equal(baseline.Length, pooled.Length);
            for (int i = 0; i < baseline.Length; i++)
                Assert.Equal(baseline[i], pooled[i], 4);
        }
    }

    [Fact]
    public void PooledGradients_ReuseBuffersAcrossSteps()
    {
        var weight = CreateRandom(new[] { 4, 3 }, 42);
        var pool = new GradientBufferPool<float>();
        pool.Register(weight);

        // Get buffer reference
        Assert.True(pool.TryGetBuffer(weight, out var buffer1, out _));

        // Step 1
        using (var tape = new GradientTape<float>())
        {
            var input = CreateRandom(new[] { 2, 4 }, 10);
            var output = _engine.TensorMatMul(input, weight);
            var loss = _engine.ReduceSum(output, new[] { 0, 1 }, keepDims: false);
            tape.ComputeGradients(loss, pool, new[] { weight });
        }

        // Step 2 — buffer should be the SAME object
        pool.ZeroAll();
        using (var tape = new GradientTape<float>())
        {
            var input = CreateRandom(new[] { 2, 4 }, 20);
            var output = _engine.TensorMatMul(input, weight);
            var loss = _engine.ReduceSum(output, new[] { 0, 1 }, keepDims: false);
            tape.ComputeGradients(loss, pool, new[] { weight });
        }

        Assert.True(pool.TryGetBuffer(weight, out var buffer2, out _));
        Assert.True(ReferenceEquals(buffer1, buffer2), "Pool should reuse the same buffer object");
    }

    [Fact]
    public void ZeroAll_ClearsAllBuffers()
    {
        var weight = CreateRandom(new[] { 4, 3 }, 42);
        var pool = new GradientBufferPool<float>();
        pool.Register(weight);

        // Write some data
        Assert.True(pool.TryGetBuffer(weight, out var buffer, out _));
        var span = buffer.AsWritableSpan();
        for (int i = 0; i < span.Length; i++)
            span[i] = 1.0f;

        // Zero
        pool.ZeroAll();

        // Verify zeroed
        var readSpan = buffer.AsSpan();
        for (int i = 0; i < readSpan.Length; i++)
            Assert.Equal(0f, readSpan[i]);
    }

    [Fact]
    public void UnregisteredParameter_FallsBackToStandardPath()
    {
        var registered = CreateRandom(new[] { 4, 3 }, 42);
        var unregistered = CreateRandom(new[] { 3, 2 }, 43);
        var pool = new GradientBufferPool<float>();
        pool.Register(registered);
        // unregistered is NOT registered

        using var tape = new GradientTape<float>();
        var mid = _engine.TensorMatMul(registered, unregistered);
        var loss = _engine.ReduceSum(mid, new[] { 0, 1 }, keepDims: false);
        var grads = tape.ComputeGradients(loss, pool, new[] { registered, unregistered });

        // Both should have gradients
        Assert.True(grads.ContainsKey(registered));
        Assert.True(grads.ContainsKey(unregistered));

        // Registered param's gradient should be the pool buffer
        Assert.True(pool.TryGetBuffer(registered, out var poolBuf, out _));
        Assert.True(ReferenceEquals(grads[registered], poolBuf));
    }
}
