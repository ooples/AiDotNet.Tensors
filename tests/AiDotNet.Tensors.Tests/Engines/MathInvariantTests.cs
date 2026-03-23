using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// Mathematical invariant tests that prove engine operations are correct
/// using known algebraic properties. Every test here encodes a mathematical
/// truth that MUST hold regardless of implementation.
/// </summary>
public class MathInvariantTests
{
    private readonly CpuEngine _engine = new();
    private const float Eps = 1e-4f;

    private Tensor<float> Rand(int[] shape, int seed)
    {
        var rng = new Random(seed);
        var data = new float[shape.Aggregate(1, (a, b) => a * b)];
        for (int i = 0; i < data.Length; i++) data[i] = (float)(rng.NextDouble() * 2 - 1);
        return new Tensor<float>(data, shape);
    }

    private Tensor<float> RandPositive(int[] shape, int seed)
    {
        var rng = new Random(seed);
        var data = new float[shape.Aggregate(1, (a, b) => a * b)];
        for (int i = 0; i < data.Length; i++) data[i] = (float)(rng.NextDouble() * 9.9 + 0.1);
        return new Tensor<float>(data, shape);
    }

    private void AssertClose(Tensor<float> a, Tensor<float> b, float tol = Eps, string msg = "")
    {
        Assert.Equal(a.Shape, b.Shape);
        var ad = a.GetDataArray();
        var bd = b.GetDataArray();
        for (int i = 0; i < a.Length; i++)
            Assert.True(Math.Abs(ad[i] - bd[i]) < tol, $"{msg} Element [{i}]: {ad[i]} vs {bd[i]}, diff={Math.Abs(ad[i] - bd[i])}");
    }

    // =====================================================================
    // ARITHMETIC INVARIANTS
    // =====================================================================

    [Fact] public void Add_Commutative() // a + b = b + a
    {
        var a = Rand(new[] { 64 }, 1); var b = Rand(new[] { 64 }, 2);
        AssertClose(_engine.TensorAdd(a, b), _engine.TensorAdd(b, a), msg: "Add not commutative");
    }

    [Fact] public void Add_Associative() // (a + b) + c = a + (b + c)
    {
        var a = Rand(new[] { 64 }, 1); var b = Rand(new[] { 64 }, 2); var c = Rand(new[] { 64 }, 3);
        AssertClose(_engine.TensorAdd(_engine.TensorAdd(a, b), c), _engine.TensorAdd(a, _engine.TensorAdd(b, c)), 1e-3f, "Add not associative");
    }

    [Fact] public void Add_Identity() // a + 0 = a
    {
        var a = Rand(new[] { 64 }, 1);
        var zero = new Tensor<float>(new float[64], new[] { 64 });
        AssertClose(_engine.TensorAdd(a, zero), a, msg: "Add identity failed");
    }

    [Fact] public void Add_Inverse() // a + (-a) = 0
    {
        var a = Rand(new[] { 64 }, 1);
        var negA = _engine.TensorMultiplyScalar(a, -1f);
        var result = _engine.TensorAdd(a, negA);
        var data = result.GetDataArray();
        for (int i = 0; i < data.Length; i++)
            Assert.True(Math.Abs(data[i]) < Eps, $"a + (-a) != 0 at [{i}]: {data[i]}");
    }

    [Fact] public void Subtract_AntiCommutative() // a - b = -(b - a)
    {
        var a = Rand(new[] { 64 }, 1); var b = Rand(new[] { 64 }, 2);
        var lhs = _engine.TensorSubtract(a, b);
        var rhs = _engine.TensorMultiplyScalar(_engine.TensorSubtract(b, a), -1f);
        AssertClose(lhs, rhs, msg: "Subtract not anti-commutative");
    }

    [Fact] public void Multiply_Commutative() // a * b = b * a
    {
        var a = Rand(new[] { 64 }, 1); var b = Rand(new[] { 64 }, 2);
        AssertClose(_engine.TensorMultiply(a, b), _engine.TensorMultiply(b, a), msg: "Multiply not commutative");
    }

    [Fact] public void Multiply_Identity() // a * 1 = a
    {
        var a = Rand(new[] { 64 }, 1);
        AssertClose(_engine.TensorMultiplyScalar(a, 1f), a, msg: "Multiply by 1 failed");
    }

    [Fact] public void Multiply_Zero() // a * 0 = 0
    {
        var a = Rand(new[] { 64 }, 1);
        var result = _engine.TensorMultiplyScalar(a, 0f);
        var data = result.GetDataArray();
        for (int i = 0; i < data.Length; i++)
            Assert.True(Math.Abs(data[i]) < 1e-7f, $"a * 0 != 0 at [{i}]: {data[i]}");
    }

    [Fact] public void Distributive() // a * (b + c) = a*b + a*c
    {
        var a = Rand(new[] { 64 }, 1); var b = Rand(new[] { 64 }, 2); var c = Rand(new[] { 64 }, 3);
        var lhs = _engine.TensorMultiply(a, _engine.TensorAdd(b, c));
        var rhs = _engine.TensorAdd(_engine.TensorMultiply(a, b), _engine.TensorMultiply(a, c));
        AssertClose(lhs, rhs, 1e-3f, "Distributive law failed");
    }

    // =====================================================================
    // MATRIX MULTIPLY INVARIANTS
    // =====================================================================

    [Fact] public void MatMul_IdentityMatrix() // A @ I = A
    {
        var a = new Matrix<float>(4, 4);
        var identity = new Matrix<float>(4, 4);
        var rng = new Random(10);
        for (int i = 0; i < 4; i++) { identity[i, i] = 1f; for (int j = 0; j < 4; j++) a[i, j] = (float)rng.NextDouble(); }
        var result = _engine.MatrixMultiply(a, identity);
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                Assert.Equal(a[i, j], result[i, j], Eps);
    }

    [Fact] public void MatMul_Associative() // (AB)C = A(BC)
    {
        var a = new Matrix<float>(3, 4); var b = new Matrix<float>(4, 5); var c = new Matrix<float>(5, 2);
        var rng = new Random(11);
        for (int i = 0; i < 3; i++) for (int j = 0; j < 4; j++) a[i, j] = (float)rng.NextDouble();
        for (int i = 0; i < 4; i++) for (int j = 0; j < 5; j++) b[i, j] = (float)rng.NextDouble();
        for (int i = 0; i < 5; i++) for (int j = 0; j < 2; j++) c[i, j] = (float)rng.NextDouble();
        var lhs = _engine.MatrixMultiply(_engine.MatrixMultiply(a, b), c);
        var rhs = _engine.MatrixMultiply(a, _engine.MatrixMultiply(b, c));
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 2; j++)
                Assert.True(Math.Abs(lhs[i, j] - rhs[i, j]) < 1e-2f, $"MatMul not associative at [{i},{j}]");
    }

    [Fact] public void MatMul_TransposeProperty() // (AB)^T = B^T A^T
    {
        var a = new Matrix<float>(3, 4); var b = new Matrix<float>(4, 5);
        var rng = new Random(12);
        for (int i = 0; i < 3; i++) for (int j = 0; j < 4; j++) a[i, j] = (float)rng.NextDouble();
        for (int i = 0; i < 4; i++) for (int j = 0; j < 5; j++) b[i, j] = (float)rng.NextDouble();
        var lhs = _engine.MatrixMultiply(a, b).Transpose();
        var rhs = _engine.MatrixMultiply(b.Transpose(), a.Transpose());
        for (int i = 0; i < 5; i++)
            for (int j = 0; j < 3; j++)
                Assert.True(Math.Abs(lhs[i, j] - rhs[i, j]) < Eps, $"(AB)^T != B^T A^T at [{i},{j}]");
    }

    [Fact] public void BatchMatMul_MatchesPerBatchLoop()
    {
        var rng = new Random(13);
        int batch = 4, m = 6, k = 8, n = 5;
        var aData = new float[batch * m * k]; var bData = new float[batch * k * n];
        for (int i = 0; i < aData.Length; i++) aData[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < bData.Length; i++) bData[i] = (float)(rng.NextDouble() * 2 - 1);
        var a = new Tensor<float>(aData, new[] { batch, m, k });
        var b = new Tensor<float>(bData, new[] { batch, k, n });
        var batched = _engine.TensorBatchMatMul(a, b);
        var batchedData = batched.GetDataArray();
        for (int bi = 0; bi < batch; bi++)
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                {
                    float expected = 0;
                    for (int p = 0; p < k; p++) expected += aData[bi * m * k + i * k + p] * bData[bi * k * n + p * n + j];
                    Assert.True(Math.Abs(batchedData[bi * m * n + i * n + j] - expected) < 1e-2f,
                        $"BatchMatMul wrong at batch={bi}, [{i},{j}]");
                }
    }

    // =====================================================================
    // ACTIVATION INVARIANTS
    // =====================================================================

    [Fact] public void Sigmoid_OutputRange() // sigmoid(x) in (0, 1)
    {
        var x = Rand(new[] { 256 }, 20);
        var data = _engine.TensorSigmoid(x).GetDataArray();
        for (int i = 0; i < data.Length; i++)
            Assert.True(data[i] > 0f && data[i] < 1f, $"Sigmoid out of range at [{i}]: {data[i]}");
    }

    [Fact] public void Sigmoid_Symmetry() // sigmoid(-x) = 1 - sigmoid(x)
    {
        var x = Rand(new[] { 64 }, 21);
        var negX = _engine.TensorMultiplyScalar(x, -1f);
        var sigX = _engine.TensorSigmoid(x).GetDataArray();
        var sigNegX = _engine.TensorSigmoid(negX).GetDataArray();
        for (int i = 0; i < 64; i++)
            Assert.True(Math.Abs(sigNegX[i] - (1f - sigX[i])) < Eps, $"sigmoid(-x) != 1-sigmoid(x) at [{i}]");
    }

    [Fact] public void Sigmoid_KnownValue() // sigmoid(0) = 0.5
    {
        var zero = new Tensor<float>(new float[1], new[] { 1 });
        Assert.Equal(0.5f, _engine.TensorSigmoid(zero).GetDataArray()[0], Eps);
    }

    [Fact] public void Tanh_OutputRange() // tanh(x) in (-1, 1)
    {
        var x = Rand(new[] { 256 }, 22);
        var data = _engine.TensorTanh(x).GetDataArray();
        for (int i = 0; i < data.Length; i++)
            Assert.True(data[i] > -1f && data[i] < 1f, $"Tanh out of range at [{i}]: {data[i]}");
    }

    [Fact] public void Tanh_OddFunction() // tanh(-x) = -tanh(x)
    {
        var x = Rand(new[] { 64 }, 23);
        var negX = _engine.TensorMultiplyScalar(x, -1f);
        var tanhX = _engine.TensorTanh(x).GetDataArray();
        var tanhNegX = _engine.TensorTanh(negX).GetDataArray();
        for (int i = 0; i < 64; i++)
            Assert.True(Math.Abs(tanhNegX[i] + tanhX[i]) < Eps, $"tanh(-x) != -tanh(x) at [{i}]");
    }

    [Fact] public void Tanh_KnownValue() // tanh(0) = 0
    {
        var zero = new Tensor<float>(new float[1], new[] { 1 });
        Assert.True(Math.Abs(_engine.TensorTanh(zero).GetDataArray()[0]) < Eps, "tanh(0) != 0");
    }

    [Fact] public void ReLU_NonNegative() // relu(x) >= 0
    {
        var x = Rand(new[] { 256 }, 24);
        var data = _engine.TensorReLU(x).GetDataArray();
        for (int i = 0; i < data.Length; i++)
            Assert.True(data[i] >= 0f, $"ReLU negative at [{i}]: {data[i]}");
    }

    [Fact] public void ReLU_Identity_ForPositive() // relu(x) = x for x > 0
    {
        var data = new float[] { 1, 2, 3, 0.5f, 10 };
        var x = new Tensor<float>(data, new[] { 5 });
        var result = _engine.TensorReLU(x).GetDataArray();
        for (int i = 0; i < 5; i++) Assert.Equal(data[i], result[i], Eps);
    }

    [Fact] public void ReLU_Zero_ForNegative() // relu(x) = 0 for x < 0
    {
        var data = new float[] { -1, -2, -3, -0.5f, -10 };
        var x = new Tensor<float>(data, new[] { 5 });
        var result = _engine.TensorReLU(x).GetDataArray();
        for (int i = 0; i < 5; i++) Assert.Equal(0f, result[i], Eps);
    }

    [Fact] public void GELU_KnownValues() // GELU(0) = 0, GELU(x) ~ x for large x
    {
        var zero = new Tensor<float>(new float[] { 0f }, new[] { 1 });
        Assert.True(Math.Abs(_engine.TensorGELU(zero).GetDataArray()[0]) < Eps, "GELU(0) != 0");
        var large = new Tensor<float>(new float[] { 10f }, new[] { 1 });
        Assert.True(Math.Abs(_engine.TensorGELU(large).GetDataArray()[0] - 10f) < 0.01f, "GELU(10) != ~10");
    }

    // =====================================================================
    // EXP / LOG INVARIANTS
    // =====================================================================

    [Fact] public void ExpLog_Inverse() // log(exp(x)) = x
    {
        var x = Rand(new[] { 64 }, 30);
        AssertClose(_engine.TensorLog(_engine.TensorExp(x)), x, 1e-3f, "log(exp(x)) != x");
    }

    [Fact] public void LogExp_Inverse() // exp(log(x)) = x for x > 0
    {
        var x = RandPositive(new[] { 64 }, 31);
        AssertClose(_engine.TensorExp(_engine.TensorLog(x)), x, 1e-2f, "exp(log(x)) != x");
    }

    [Fact] public void Exp_KnownValue() // exp(0) = 1
    {
        var zero = new Tensor<float>(new float[] { 0f }, new[] { 1 });
        Assert.Equal(1f, _engine.TensorExp(zero).GetDataArray()[0], Eps);
    }

    [Fact] public void Log_KnownValue() // log(1) = 0
    {
        var one = new Tensor<float>(new float[] { 1f }, new[] { 1 });
        Assert.True(Math.Abs(_engine.TensorLog(one).GetDataArray()[0]) < Eps, "log(1) != 0");
    }

    // =====================================================================
    // SOFTMAX INVARIANTS
    // =====================================================================

    [Fact] public void Softmax_SumToOne() // sum(softmax(x)) = 1 per row
    {
        var x = Rand(new[] { 4, 32 }, 40);
        var sm = _engine.Softmax(x, -1).GetDataArray();
        for (int row = 0; row < 4; row++)
        {
            float sum = 0;
            for (int j = 0; j < 32; j++) sum += sm[row * 32 + j];
            Assert.True(Math.Abs(sum - 1f) < 1e-3f, $"Softmax row {row} sums to {sum}, not 1");
        }
    }

    [Fact] public void Softmax_NonNegative() // softmax(x) >= 0
    {
        var x = Rand(new[] { 4, 32 }, 41);
        var data = _engine.Softmax(x, -1).GetDataArray();
        for (int i = 0; i < data.Length; i++)
            Assert.True(data[i] >= 0f, $"Softmax negative at [{i}]: {data[i]}");
    }

    [Fact] public void Softmax_TranslationInvariant() // softmax(x + c) = softmax(x)
    {
        var x = Rand(new[] { 4, 32 }, 42);
        var offset = _engine.TensorAddScalar(x, 100f);
        AssertClose(_engine.Softmax(x, -1), _engine.Softmax(offset, -1), 1e-3f, "Softmax not translation invariant");
    }

    [Fact] public void LogSoftmax_ConsistentWithLog() // log_softmax(x) = log(softmax(x))
    {
        var x = Rand(new[] { 4, 16 }, 43);
        var logSm = _engine.TensorLogSoftmax(x, -1);
        var logOfSm = _engine.TensorLog(_engine.Softmax(x, -1));
        AssertClose(logSm, logOfSm, 1e-3f, "LogSoftmax != log(softmax)");
    }

    // =====================================================================
    // REDUCTION INVARIANTS
    // =====================================================================

    [Fact] public void Sum_Linearity() // sum(a*x + b*y) = a*sum(x) + b*sum(y)
    {
        var x = Rand(new[] { 256 }, 50); var y = Rand(new[] { 256 }, 51);
        float a = 2.5f, b = -1.3f;
        float lhs = _engine.TensorSum(_engine.TensorAdd(_engine.TensorMultiplyScalar(x, a), _engine.TensorMultiplyScalar(y, b)));
        float rhs = a * _engine.TensorSum(x) + b * _engine.TensorSum(y);
        Assert.True(Math.Abs(lhs - rhs) < 0.1f, $"Sum not linear: {lhs} vs {rhs}");
    }

    [Fact] public void Mean_OfConstant() // mean([c,c,c,...]) = c
    {
        float c = 3.14f;
        var x = new Tensor<float>(Enumerable.Repeat(c, 100).ToArray(), new[] { 100 });
        Assert.Equal(c, _engine.TensorMean(x), Eps);
    }

    [Fact] public void Sum_OfOnes() // sum(ones(n)) = n
    {
        int n = 256;
        var ones = new Tensor<float>(Enumerable.Repeat(1f, n).ToArray(), new[] { n });
        Assert.Equal((float)n, _engine.TensorSum(ones), Eps);
    }

    // =====================================================================
    // NORMALIZATION INVARIANTS
    // =====================================================================

    [Fact] public void BatchNorm_OutputMeanNearZero()
    {
        var x = Rand(new[] { 4, 8, 4, 4 }, 60);
        var gamma = new Tensor<float>(Enumerable.Repeat(1f, 8).ToArray(), new[] { 8 });
        var beta = new Tensor<float>(new float[8], new[] { 8 });
        var result = _engine.BatchNorm(x, gamma, beta, 1e-5, out _, out _);
        // Per-channel mean should be near 0
        var data = result.GetDataArray();
        for (int c = 0; c < 8; c++)
        {
            float sum = 0; int count = 0;
            for (int b = 0; b < 4; b++)
                for (int h = 0; h < 4; h++)
                    for (int w = 0; w < 4; w++)
                    { sum += data[((b * 8 + c) * 4 + h) * 4 + w]; count++; }
            float mean = sum / count;
            Assert.True(Math.Abs(mean) < 0.1f, $"BatchNorm channel {c} mean={mean}, expected ~0");
        }
    }

    // =====================================================================
    // ATTENTION INVARIANTS
    // =====================================================================

    [Fact] public void Attention_WeightsSumToOne()
    {
        var q = Rand(new[] { 1, 1, 4, 8 }, 70);
        var k = Rand(new[] { 1, 1, 4, 8 }, 71);
        var v = Rand(new[] { 1, 1, 4, 8 }, 72);
        _engine.ScaledDotProductAttention(q, k, v, null, 1.0 / Math.Sqrt(8), out var weights);
        // Attention weights are softmax over last dim, so each row sums to 1
        var wData = weights.GetDataArray();
        for (int row = 0; row < 4; row++)
        {
            float sum = 0;
            for (int col = 0; col < 4; col++) sum += wData[row * 4 + col];
            Assert.True(Math.Abs(sum - 1f) < 1e-3f, $"Attention weights row {row} sums to {sum}");
        }
    }

    [Fact] public void AttentionBackward_NumericalGradientCheck()
    {
        var rng = new Random(73);
        int b = 1, h = 1, s = 3, d = 4;
        double scale = 1.0 / Math.Sqrt(d);
        float eps = 1e-3f;
        var shape = new[] { b, h, s, d };
        int total = b * h * s * d;

        var qData = new float[total]; var kData = new float[total]; var vData = new float[total];
        for (int i = 0; i < total; i++) { qData[i] = (float)(rng.NextDouble() * 0.5); kData[i] = (float)(rng.NextDouble() * 0.5); vData[i] = (float)(rng.NextDouble() * 0.5); }
        var Q = new Tensor<float>(qData, shape); var K = new Tensor<float>(kData, shape); var V = new Tensor<float>(vData, shape);
        var grad = new Tensor<float>(Enumerable.Repeat(1f, total).ToArray(), shape);

        _engine.ScaledDotProductAttention(Q, K, V, null, scale, out var attnW);
        _engine.ScaledDotProductAttentionBackward(grad, Q, K, V, attnW, scale, out var dQ, out var dK, out var dV);

        // Check dV, dQ, dK via finite differences
        int pass = 0, checks = 0;
        foreach (var (paramData, analytical, name) in new[] { (vData, dV, "dV"), (qData, dQ, "dQ"), (kData, dK, "dK") })
        {
            var aData = analytical.GetDataArray();
            for (int idx = 0; idx < Math.Min(4, total); idx++)
            {
                float orig = paramData[idx];
                paramData[idx] = orig + eps;
                var plus = new Tensor<float>((float[])paramData.Clone(), shape);
                float lPlus = SumAttention(Q, K, name == "dV" ? plus : V, name == "dQ" ? plus : Q, name == "dK" ? plus : K, scale);
                paramData[idx] = orig - eps;
                var minus = new Tensor<float>((float[])paramData.Clone(), shape);
                float lMinus = SumAttention(Q, K, name == "dV" ? minus : V, name == "dQ" ? minus : Q, name == "dK" ? minus : K, scale);
                paramData[idx] = orig;
                float numerical = (lPlus - lMinus) / (2 * eps);
                float relErr = Math.Abs(numerical) > 1e-5f ? Math.Abs(numerical - aData[idx]) / Math.Abs(numerical) : Math.Abs(numerical - aData[idx]);
                checks++;
                if (relErr < 0.1f) pass++;
            }
        }
        Assert.True(pass >= checks * 0.75, $"Attention gradient check: {pass}/{checks} passed (need 75%)");
    }

    private float SumAttention(Tensor<float> defQ, Tensor<float> defK, Tensor<float> v, Tensor<float> q, Tensor<float> k, double scale)
    {
        var output = _engine.ScaledDotProductAttention(q, k, v, null, scale, out _);
        float sum = 0; var d = output.GetDataArray(); for (int i = 0; i < d.Length; i++) sum += d[i]; return sum;
    }

    // =====================================================================
    // CONVOLUTION INVARIANTS
    // =====================================================================

    [Fact] public void Conv2D_IdentityKernel()
    {
        // 1x1 conv with identity kernel = copy
        var input = Rand(new[] { 1, 1, 4, 4 }, 80);
        // 1x1 kernel with weight=1
        var kernel = new Tensor<float>(new float[] { 1f }, new[] { 1, 1, 1, 1 });
        var result = _engine.Conv2D(input, kernel, 1, 0, 1);
        AssertClose(input, result, 1e-3f, "1x1 identity conv failed");
    }

    // =====================================================================
    // TENSOR SHAPE INVARIANTS
    // =====================================================================

    [Fact] public void Transpose_Involution() // transpose(transpose(A)) = A
    {
        var a = Rand(new[] { 4, 8 }, 90);
        var tt = _engine.TensorTranspose(_engine.TensorTranspose(a));
        AssertClose(tt, a, msg: "Transpose not involutory");
    }

    [Fact] public void Eye_IsIdentity() // I[i,j] = (i==j) ? 1 : 0
    {
        var eye = _engine.TensorEye<float>(5);
        var data = eye.GetDataArray();
        for (int i = 0; i < 5; i++)
            for (int j = 0; j < 5; j++)
                Assert.Equal(i == j ? 1f : 0f, data[i * 5 + j], Eps);
    }

    [Fact] public void DotProduct_CommutativeAndCorrect() // a.b = b.a = sum(a*b)
    {
        var a = new Vector<float>(new float[] { 1, 2, 3, 4 });
        var b = new Vector<float>(new float[] { 5, 6, 7, 8 });
        float ab = _engine.DotProduct(a, b);
        float ba = _engine.DotProduct(b, a);
        Assert.Equal(ab, ba, Eps); // commutative
        Assert.Equal(70f, ab, Eps); // 1*5 + 2*6 + 3*7 + 4*8 = 70
    }
}
