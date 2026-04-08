using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// Integration tests for AutoTracer auto-compilation.
/// Verifies that compiled plans produce identical results to eager execution
/// across all wired operations and edge cases.
/// </summary>
public class AutoTracerTests
{
    private readonly CpuEngine E = new();
    private const float Tol = 1e-4f;

    private Tensor<float> Rand(int[] shape, int seed)
    {
        var r = new Random(seed);
        var d = new float[shape.Aggregate(1, (a, b) => a * b)];
        for (int i = 0; i < d.Length; i++) d[i] = (float)(r.NextDouble() * 2 - 1);
        return new Tensor<float>(d, shape);
    }

    private Tensor<float> RandPositive(int[] shape, int seed)
    {
        var r = new Random(seed);
        var d = new float[shape.Aggregate(1, (a, b) => a * b)];
        for (int i = 0; i < d.Length; i++) d[i] = (float)(r.NextDouble() * 0.9 + 0.1);
        return new Tensor<float>(d, shape);
    }

    private void AssertClose(Tensor<float> a, Tensor<float> b, float tol = Tol, string msg = "")
    {
        Assert.Equal(a.Shape.ToArray(), b.Shape.ToArray());
        var ad = a.GetDataArray();
        var bd = b.GetDataArray();
        for (int i = 0; i < a.Length; i++)
            Assert.True(Math.Abs(ad[i] - bd[i]) < tol, $"{msg} [{i}]: {ad[i]} vs {bd[i]}");
    }

    #region Eager vs Compiled correctness — each op

    [Fact]
    public void TensorAdd_EagerMatchesCompiledResult()
    {
        // Run the same op 5 times — after threshold, AutoTracer should compile
        // Verify all results match the first eager result
        var a = Rand(new[] { 1000 }, 1);
        var b = Rand(new[] { 1000 }, 2);

        var results = new Tensor<float>[5];
        for (int i = 0; i < 5; i++)
            results[i] = E.TensorAdd(a, b);

        for (int i = 1; i < 5; i++)
            AssertClose(results[0], results[i], msg: $"TensorAdd iteration {i}");
    }

    [Fact]
    public void TensorSubtract_EagerMatchesCompiledResult()
    {
        var a = Rand(new[] { 1000 }, 3);
        var b = Rand(new[] { 1000 }, 4);

        var results = new Tensor<float>[5];
        for (int i = 0; i < 5; i++)
            results[i] = E.TensorSubtract(a, b);

        for (int i = 1; i < 5; i++)
            AssertClose(results[0], results[i], msg: $"TensorSubtract iteration {i}");
    }

    [Fact]
    public void TensorMultiply_EagerMatchesCompiledResult()
    {
        var a = Rand(new[] { 1000 }, 5);
        var b = Rand(new[] { 1000 }, 6);

        var results = new Tensor<float>[5];
        for (int i = 0; i < 5; i++)
            results[i] = E.TensorMultiply(a, b);

        for (int i = 1; i < 5; i++)
            AssertClose(results[0], results[i], msg: $"TensorMultiply iteration {i}");
    }

    [Fact]
    public void TensorExp_EagerMatchesCompiledResult()
    {
        var a = Rand(new[] { 1000 }, 7);

        var results = new Tensor<float>[5];
        for (int i = 0; i < 5; i++)
            results[i] = E.TensorExp(a);

        for (int i = 1; i < 5; i++)
            AssertClose(results[0], results[i], msg: $"TensorExp iteration {i}");
    }

    [Fact]
    public void ReLU_EagerMatchesCompiledResult()
    {
        var a = Rand(new[] { 1000 }, 8);

        var results = new Tensor<float>[5];
        for (int i = 0; i < 5; i++)
            results[i] = E.ReLU(a);

        for (int i = 1; i < 5; i++)
            AssertClose(results[0], results[i], msg: $"ReLU iteration {i}");
    }

    [Fact]
    public void Sigmoid_EagerMatchesCompiledResult()
    {
        var a = Rand(new[] { 1000 }, 9);

        var results = new Tensor<float>[5];
        for (int i = 0; i < 5; i++)
            results[i] = E.Sigmoid(a);

        for (int i = 1; i < 5; i++)
            AssertClose(results[0], results[i], msg: $"Sigmoid iteration {i}");
    }

    [Fact]
    public void Tanh_EagerMatchesCompiledResult()
    {
        var a = Rand(new[] { 1000 }, 10);

        var results = new Tensor<float>[5];
        for (int i = 0; i < 5; i++)
            results[i] = E.Tanh(a);

        for (int i = 1; i < 5; i++)
            AssertClose(results[0], results[i], msg: $"Tanh iteration {i}");
    }

    [Fact]
    public void GELU_EagerMatchesCompiledResult()
    {
        var a = Rand(new[] { 1000 }, 11);

        var results = new Tensor<float>[5];
        for (int i = 0; i < 5; i++)
            results[i] = E.GELU(a);

        for (int i = 1; i < 5; i++)
            AssertClose(results[0], results[i], msg: $"GELU iteration {i}");
    }

    [Fact]
    public void Softmax_EagerMatchesCompiledResult()
    {
        var a = Rand(new[] { 10, 100 }, 12);

        var results = new Tensor<float>[5];
        for (int i = 0; i < 5; i++)
            results[i] = E.Softmax(a, axis: -1);

        for (int i = 1; i < 5; i++)
            AssertClose(results[0], results[i], msg: $"Softmax iteration {i}");
    }

    [Fact]
    public void TensorMatMul_EagerMatchesCompiledResult()
    {
        var a = Rand(new[] { 32, 64 }, 13);
        var b = Rand(new[] { 64, 16 }, 14);

        var results = new Tensor<float>[5];
        for (int i = 0; i < 5; i++)
            results[i] = E.TensorMatMul(a, b);

        for (int i = 1; i < 5; i++)
            AssertClose(results[0], results[i], tol: 1e-3f, msg: $"TensorMatMul iteration {i}");
    }

    #endregion

    #region Shape change triggers recompilation

    [Fact]
    public void ShapeChange_ProducesCorrectResults()
    {
        // Run with one shape, then switch — must still produce correct results
        var a1 = Rand(new[] { 100 }, 20);
        var b1 = Rand(new[] { 100 }, 21);

        // Warm up with shape [100]
        for (int i = 0; i < 5; i++)
            E.TensorAdd(a1, b1);

        // Now use shape [200] — different shape, must not use stale plan
        var a2 = Rand(new[] { 200 }, 22);
        var b2 = Rand(new[] { 200 }, 23);

        var expected = E.TensorAdd(a2, b2);
        var result2 = E.TensorAdd(a2, b2);

        AssertClose(expected, result2, msg: "Shape change correctness");
        Assert.Equal(200, result2.Length);
    }

    #endregion

    #region Multi-op patterns

    [Fact]
    public void MultiOpPattern_AddThenReLU_ProducesCorrectResults()
    {
        var a = Rand(new[] { 500 }, 30);
        var b = Rand(new[] { 500 }, 31);

        var results = new Tensor<float>[5];
        for (int i = 0; i < 5; i++)
        {
            var sum = E.TensorAdd(a, b);
            results[i] = E.ReLU(sum);
        }

        for (int i = 1; i < 5; i++)
            AssertClose(results[0], results[i], msg: $"Add+ReLU iteration {i}");
    }

    [Fact]
    public void MultiOpPattern_MatMulThenSigmoid_ProducesCorrectResults()
    {
        var a = Rand(new[] { 16, 32 }, 32);
        var b = Rand(new[] { 32, 8 }, 33);

        var results = new Tensor<float>[5];
        for (int i = 0; i < 5; i++)
        {
            var mm = E.TensorMatMul(a, b);
            results[i] = E.Sigmoid(mm);
        }

        for (int i = 1; i < 5; i++)
            AssertClose(results[0], results[i], tol: 1e-3f, msg: $"MatMul+Sigmoid iteration {i}");
    }

    #endregion

    #region Edge cases

    [Fact]
    public void SingleElement_ProducesCorrectResults()
    {
        var a = new Tensor<float>(new float[] { 3.0f }, new[] { 1 });
        var b = new Tensor<float>(new float[] { 2.0f }, new[] { 1 });

        var results = new Tensor<float>[5];
        for (int i = 0; i < 5; i++)
            results[i] = E.TensorAdd(a, b);

        for (int i = 0; i < 5; i++)
        {
            Assert.Equal(1, results[i].Length);
            Assert.Equal(5.0f, results[i].GetDataArray()[0], 4);
        }
    }

    [Fact]
    public void LargeTensor_1M_ProducesCorrectResults()
    {
        var a = Rand(new[] { 1_000_000 }, 40);
        var b = Rand(new[] { 1_000_000 }, 41);

        var r1 = E.TensorAdd(a, b);
        var r2 = E.TensorAdd(a, b);

        AssertClose(r1, r2, msg: "1M element correctness");
    }

    [Fact]
    public void SoftmaxOutputSumsToOne()
    {
        var a = Rand(new[] { 5, 100 }, 50);

        for (int iter = 0; iter < 5; iter++)
        {
            var sm = E.Softmax(a, axis: -1);
            var data = sm.GetDataArray();

            // Each row should sum to ~1.0
            for (int row = 0; row < 5; row++)
            {
                float rowSum = 0;
                for (int col = 0; col < 100; col++)
                    rowSum += data[row * 100 + col];
                Assert.True(Math.Abs(rowSum - 1.0f) < 1e-3f, $"Softmax row {row} sum = {rowSum} on iter {iter}");
            }
        }
    }

    #endregion

    #region OpType enum correctness

    [Fact]
    public void OpTypeParser_ParsesAllKnownOps()
    {
        Assert.Equal(OpType.TensorAdd, OpTypeParser.Parse("TensorAdd"));
        Assert.Equal(OpType.TensorSubtract, OpTypeParser.Parse("TensorSubtract"));
        Assert.Equal(OpType.TensorMultiply, OpTypeParser.Parse("TensorMultiply"));
        Assert.Equal(OpType.TensorDivide, OpTypeParser.Parse("TensorDivide"));
        Assert.Equal(OpType.TensorExp, OpTypeParser.Parse("TensorExp"));
        Assert.Equal(OpType.TensorLog, OpTypeParser.Parse("TensorLog"));
        Assert.Equal(OpType.TensorSqrt, OpTypeParser.Parse("TensorSqrt"));
        Assert.Equal(OpType.TensorAbs, OpTypeParser.Parse("TensorAbs"));
        Assert.Equal(OpType.TensorNegate, OpTypeParser.Parse("TensorNegate"));
        Assert.Equal(OpType.TensorMatMul, OpTypeParser.Parse("TensorMatMul"));
        Assert.Equal(OpType.TensorTranspose, OpTypeParser.Parse("TensorTranspose"));
        Assert.Equal(OpType.ReLU, OpTypeParser.Parse("ReLU"));
        Assert.Equal(OpType.Sigmoid, OpTypeParser.Parse("Sigmoid"));
        Assert.Equal(OpType.Tanh, OpTypeParser.Parse("Tanh"));
        Assert.Equal(OpType.GELU, OpTypeParser.Parse("GELU"));
        Assert.Equal(OpType.Softmax, OpTypeParser.Parse("Softmax"));
        Assert.Equal(OpType.LogSoftmax, OpTypeParser.Parse("LogSoftmax"));
        Assert.Equal(OpType.Swish, OpTypeParser.Parse("Swish"));
        Assert.Equal(OpType.Mish, OpTypeParser.Parse("Mish"));
        Assert.Equal(OpType.ELU, OpTypeParser.Parse("ELU"));
        Assert.Equal(OpType.SELU, OpTypeParser.Parse("SELU"));
        Assert.Equal(OpType.LeakyReLU, OpTypeParser.Parse("LeakyReLU"));
        Assert.Equal(OpType.Conv2D, OpTypeParser.Parse("Conv2D"));
        Assert.Equal(OpType.MSELoss, OpTypeParser.Parse("MSELoss"));
        Assert.Equal(OpType.Mean, OpTypeParser.Parse("Mean"));
        Assert.Equal(OpType.ReduceSum, OpTypeParser.Parse("ReduceSum"));
    }

    [Fact]
    public void OpTypeParser_UnknownReturnsUnknown()
    {
        Assert.Equal(OpType.Unknown, OpTypeParser.Parse("NonExistentOp"));
        Assert.Equal(OpType.Unknown, OpTypeParser.Parse(""));
    }

    #endregion

    #region AutoTracer disable/enable

    [Fact]
    public void DisabledAutoTracer_StillProducesCorrectResults()
    {
        bool wasEnabled = AutoTracer.Enabled;
        try
        {
            AutoTracer.Enabled = false;

            var a = Rand(new[] { 500 }, 60);
            var b = Rand(new[] { 500 }, 61);

            var r1 = E.TensorAdd(a, b);
            var r2 = E.TensorAdd(a, b);

            AssertClose(r1, r2, msg: "Disabled AutoTracer correctness");
        }
        finally
        {
            AutoTracer.Enabled = wasEnabled;
        }
    }

    #endregion

    #region Softmax axis paramHash correctness

    [Fact]
    public void Softmax_DifferentAxes_ProduceCorrectResults()
    {
        // Regression test: without paramHash, Softmax(x, axis=0) would compile
        // a plan that gets reused for axis=1, producing wrong results.
        var a = Rand(new[] { 5, 10 }, 70);

        // Run axis=1 several times (may compile a plan)
        var axis1Results = new Tensor<float>[5];
        for (int i = 0; i < 5; i++)
            axis1Results[i] = E.Softmax(a, axis: 1);

        // Now run axis=0 — must NOT reuse axis=1 plan
        var axis0Result = E.Softmax(a, axis: 0);

        // axis=1: each row sums to 1 (10 elements per row, 5 rows)
        var d1 = axis1Results[0].GetDataArray();
        for (int row = 0; row < 5; row++)
        {
            float sum = 0;
            for (int col = 0; col < 10; col++) sum += d1[row * 10 + col];
            Assert.True(Math.Abs(sum - 1.0f) < 1e-3f, $"axis=1 row {row} sum={sum}");
        }

        // axis=0: each column sums to 1 (5 elements per column, 10 columns)
        var d0 = axis0Result.GetDataArray();
        for (int col = 0; col < 10; col++)
        {
            float sum = 0;
            for (int row = 0; row < 5; row++) sum += d0[row * 10 + col];
            Assert.True(Math.Abs(sum - 1.0f) < 1e-3f, $"axis=0 col {col} sum={sum}");
        }
    }

    #endregion

    #region Thread safety

    [Fact]
    public void ThreadSafety_IndependentPerThread()
    {
        // Each thread should have independent AutoTracer state
        var exceptions = new System.Collections.Concurrent.ConcurrentBag<Exception>();
        var threads = new Thread[4];

        for (int t = 0; t < threads.Length; t++)
        {
            int threadSeed = t * 100;
            threads[t] = new Thread(() =>
            {
                try
                {
                    var engine = new CpuEngine();
                    var a = Rand(new[] { 500 }, threadSeed);
                    var b = Rand(new[] { 500 }, threadSeed + 1);

                    var results = new Tensor<float>[5];
                    for (int i = 0; i < 5; i++)
                        results[i] = engine.TensorAdd(a, b);

                    // All results must match
                    var expected = results[0].GetDataArray();
                    for (int i = 1; i < 5; i++)
                    {
                        var actual = results[i].GetDataArray();
                        for (int j = 0; j < expected.Length; j++)
                            if (Math.Abs(expected[j] - actual[j]) > 1e-4f)
                                throw new Exception($"Thread {threadSeed}: mismatch at [{j}]");
                    }
                }
                catch (Exception ex) { exceptions.Add(ex); }
            });
        }

        foreach (var t in threads) t.Start();
        foreach (var t in threads) t.Join();

        Assert.Empty(exceptions);
    }

    #endregion

    #region Newly wired ops correctness

    [Fact]
    public void TensorSqrt_EagerMatchesRepeated()
    {
        var a = RandPositive(new[] { 1000 }, 80);
        var results = new Tensor<float>[5];
        for (int i = 0; i < 5; i++) results[i] = E.TensorSqrt(a);
        for (int i = 1; i < 5; i++) AssertClose(results[0], results[i], msg: $"Sqrt iter {i}");
    }

    [Fact]
    public void TensorAbs_EagerMatchesRepeated()
    {
        var a = Rand(new[] { 1000 }, 81);
        var results = new Tensor<float>[5];
        for (int i = 0; i < 5; i++) results[i] = E.TensorAbs(a);
        for (int i = 1; i < 5; i++) AssertClose(results[0], results[i], msg: $"Abs iter {i}");
    }

    [Fact]
    public void Sin_EagerMatchesRepeated()
    {
        var a = Rand(new[] { 1000 }, 82);
        var results = new Tensor<float>[5];
        for (int i = 0; i < 5; i++) results[i] = E.TensorSin(a);
        for (int i = 1; i < 5; i++) AssertClose(results[0], results[i], msg: $"Sin iter {i}");
    }

    [Fact]
    public void TensorBroadcastAdd_EagerMatchesRepeated()
    {
        var a = Rand(new[] { 10, 100 }, 83);
        var b = Rand(new[] { 1, 100 }, 84);
        var results = new Tensor<float>[5];
        for (int i = 0; i < 5; i++) results[i] = E.TensorBroadcastAdd(a, b);
        for (int i = 1; i < 5; i++) AssertClose(results[0], results[i], msg: $"BroadcastAdd iter {i}");
    }

    [Fact]
    public void Floor_EagerMatchesRepeated()
    {
        var a = Rand(new[] { 1000 }, 85);
        var results = new Tensor<float>[5];
        for (int i = 0; i < 5; i++) results[i] = E.TensorFloor(a);
        for (int i = 1; i < 5; i++) AssertClose(results[0], results[i], msg: $"Floor iter {i}");
    }

    [Fact]
    public void TensorTranspose_EagerMatchesRepeated()
    {
        var a = Rand(new[] { 32, 64 }, 86);
        var results = new Tensor<float>[5];
        for (int i = 0; i < 5; i++) results[i] = E.TensorTranspose(a);
        for (int i = 1; i < 5; i++) AssertClose(results[0], results[i], msg: $"Transpose iter {i}");
    }

    #endregion
}
