using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// Mathematical invariant tests proving engine operations are correct
/// using known algebraic properties. Every test encodes a mathematical
/// truth that MUST hold regardless of implementation.
/// Covers: arithmetic, matmul, activations, exp/log, trig, softmax,
/// reductions, shapes, normalization, conv, pooling, attention, gated
/// activations, loss functions, and comparison operations.
/// </summary>
public class MathInvariantTests
{
    private readonly CpuEngine E = new();
    private const float Tol = 1e-4f;

    private Tensor<float> R(int[] s, int seed) { var r = new Random(seed); var d = new float[s.Aggregate(1, (a, b) => a * b)]; for (int i = 0; i < d.Length; i++) d[i] = (float)(r.NextDouble() * 2 - 1); return new Tensor<float>(d, s); }
    private Tensor<float> RP(int[] s, int seed) { var r = new Random(seed); var d = new float[s.Aggregate(1, (a, b) => a * b)]; for (int i = 0; i < d.Length; i++) d[i] = (float)(r.NextDouble() * 9.9 + 0.1); return new Tensor<float>(d, s); }
    private Tensor<float> C(float v, int n) => new(Enumerable.Repeat(v, n).ToArray(), new[] { n });
    private void AE(Tensor<float> a, Tensor<float> b, float t = 1e-4f, string m = "") { Assert.Equal(a.Shape, b.Shape); var ad = a.GetDataArray(); var bd = b.GetDataArray(); for (int i = 0; i < a.Length; i++) Assert.True(Math.Abs(ad[i] - bd[i]) < t, $"{m} [{i}]: {ad[i]} vs {bd[i]}"); }
    private void AZ(Tensor<float> a, string m = "") { var d = a.GetDataArray(); for (int i = 0; i < d.Length; i++) Assert.True(Math.Abs(d[i]) < Tol, $"{m} [{i}]={d[i]}"); }
    private void AR(float[] d, float lo, float hi, string m) { for (int i = 0; i < d.Length; i++) Assert.True(d[i] >= lo && d[i] <= hi, $"{m} [{i}]={d[i]}"); }

    // ================================================================
    // ADDITION (8)
    // ================================================================
    [Fact] public void Add_Commutative() => AE(E.TensorAdd(R([64], 1), R([64], 2)), E.TensorAdd(R([64], 2), R([64], 1)));
    [Fact] public void Add_Associative() { var a = R([64], 1); var b = R([64], 2); var c = R([64], 3); AE(E.TensorAdd(E.TensorAdd(a, b), c), E.TensorAdd(a, E.TensorAdd(b, c)), 1e-3f); }
    [Fact] public void Add_Identity() => AE(E.TensorAdd(R([64], 1), C(0, 64)), R([64], 1));
    [Fact] public void Add_Inverse() => AZ(E.TensorAdd(R([64], 1), E.TensorMultiplyScalar(R([64], 1), -1f)));
    [Fact] public void AddScalar_Correct() { var a = R([64], 1); var r = E.TensorAddScalar(a, 5f); var ad = a.GetDataArray(); var rd = r.GetDataArray(); for (int i = 0; i < 64; i++) Assert.Equal(ad[i] + 5f, rd[i], Tol); }
    [Fact] public void Add_LargeArray() => Assert.Equal(10000, E.TensorAdd(R([10000], 1), R([10000], 2)).Length);
    [Fact] public void BroadcastAdd_Shape() => Assert.Equal(new[] { 4, 8 }, E.TensorBroadcastAdd(R([4, 8], 1), R([1, 8], 2)).Shape);
    [Fact] public void AddInPlace_Modifies() { var a = R([64], 1); var b = R([64], 2); var orig = (float[])a.GetDataArray().Clone(); E.TensorAddInPlace(a, b); var ad = a.GetDataArray(); var bd = b.GetDataArray(); for (int i = 0; i < 64; i++) Assert.Equal(orig[i] + bd[i], ad[i], Tol); }

    // ================================================================
    // SUBTRACTION (4)
    // ================================================================
    [Fact] public void Subtract_AntiCommutative() => AE(E.TensorSubtract(R([64], 1), R([64], 2)), E.TensorMultiplyScalar(E.TensorSubtract(R([64], 2), R([64], 1)), -1f));
    [Fact] public void Subtract_Self() => AZ(E.TensorSubtract(R([64], 1), R([64], 1)));
    [Fact] public void SubtractScalar_Correct() { var a = R([64], 1); var r = E.TensorSubtractScalar(a, 3f); var ad = a.GetDataArray(); var rd = r.GetDataArray(); for (int i = 0; i < 64; i++) Assert.Equal(ad[i] - 3f, rd[i], Tol); }
    [Fact] public void Subtract_IsAddNegate() => AE(E.TensorSubtract(R([64], 1), R([64], 2)), E.TensorAdd(R([64], 1), E.TensorNegate(R([64], 2))));

    // ================================================================
    // MULTIPLICATION (6)
    // ================================================================
    [Fact] public void Multiply_Commutative() => AE(E.TensorMultiply(R([64], 1), R([64], 2)), E.TensorMultiply(R([64], 2), R([64], 1)));
    [Fact] public void Multiply_Identity() => AE(E.TensorMultiplyScalar(R([64], 1), 1f), R([64], 1));
    [Fact] public void Multiply_Zero() => AZ(E.TensorMultiplyScalar(R([64], 1), 0f));
    [Fact] public void Multiply_Negate() => AE(E.TensorMultiplyScalar(R([64], 1), -1f), E.TensorNegate(R([64], 1)));
    [Fact] public void Distributive() { var a = R([64], 1); var b = R([64], 2); var c = R([64], 3); AE(E.TensorMultiply(a, E.TensorAdd(b, c)), E.TensorAdd(E.TensorMultiply(a, b), E.TensorMultiply(a, c)), 1e-3f); }
    [Fact] public void BroadcastMultiply_Shape() => Assert.Equal(new[] { 4, 8 }, E.TensorBroadcastMultiply(R([4, 8], 1), R([1, 8], 2)).Shape);

    // ================================================================
    // DIVISION (4)
    // ================================================================
    [Fact] public void Divide_BySelf() { var a = RP([64], 1); var r = E.TensorDivide(a, a).GetDataArray(); for (int i = 0; i < 64; i++) Assert.Equal(1f, r[i], 1e-3f); }
    [Fact] public void DivideScalar_IsMultiplyInverse() => AE(E.TensorDivideScalar(R([64], 1), 2f), E.TensorMultiplyScalar(R([64], 1), 0.5f), 1e-3f);
    [Fact] public void Divide_ByOne() => AE(E.TensorDivideScalar(R([64], 1), 1f), R([64], 1));
    [Fact] public void BroadcastDivide_Shape() => Assert.Equal(new[] { 4, 8 }, E.TensorBroadcastDivide(R([4, 8], 1), RP([1, 8], 2)).Shape);

    // ================================================================
    // UNARY MATH (12)
    // ================================================================
    [Fact] public void Abs_NonNegative() { var d = E.TensorAbs(R([256], 1)).GetDataArray(); for (int i = 0; i < d.Length; i++) Assert.True(d[i] >= 0f); }
    [Fact] public void Abs_Idempotent() => AE(E.TensorAbs(E.TensorAbs(R([64], 1))), E.TensorAbs(R([64], 1)));
    [Fact] public void Negate_DoubleNegate() => AE(E.TensorNegate(E.TensorNegate(R([64], 1))), R([64], 1));
    [Fact] public void Sqrt_Squared() { var x = RP([64], 1); AE(E.TensorMultiply(E.TensorSqrt(x), E.TensorSqrt(x)), x, 1e-3f); }
    [Fact] public void Pow_One() => AE(E.TensorPow(R([64], 1), 1f), R([64], 1));
    [Fact] public void Pow_Zero() { var d = E.TensorPow(RP([64], 1), 0f).GetDataArray(); for (int i = 0; i < d.Length; i++) Assert.Equal(1f, d[i], 1e-3f); }
    [Fact] public void Pow_Two() { var x = R([64], 1); AE(E.TensorPow(x, 2f), E.TensorMultiply(x, x), 1e-3f); }
    [Fact] public void Floor_LessOrEqual() { var x = R([64], 1); var f = E.TensorFloor(x).GetDataArray(); var xd = x.GetDataArray(); for (int i = 0; i < 64; i++) Assert.True(f[i] <= xd[i] + 1e-6f); }
    [Fact] public void Ceiling_GreaterOrEqual() { var x = R([64], 1); var c = E.TensorCeiling(x).GetDataArray(); var xd = x.GetDataArray(); for (int i = 0; i < 64; i++) Assert.True(c[i] >= xd[i] - 1e-6f); }
    [Fact] public void Frac_InRange() { AR(E.TensorFrac(R([64], 1)).GetDataArray(), -1f, 1f, "Frac"); }
    [Fact] public void Clip_InRange() { AR(E.TensorClip(R([256], 1), -0.5f, 0.5f).GetDataArray(), -0.5f, 0.5f, "Clip"); }
    [Fact] public void Norm_Positive() { float n = E.TensorNorm(R([64], 2), 0).GetDataArray()[0]; Assert.True(n > 0f); }

    // ================================================================
    // EXP / LOG (6)
    // ================================================================
    [Fact] public void ExpLog_Inverse() => AE(E.TensorLog(E.TensorExp(R([64], 30))), R([64], 30), 1e-3f);
    [Fact] public void LogExp_Inverse() => AE(E.TensorExp(E.TensorLog(RP([64], 31))), RP([64], 31), 1e-2f);
    [Fact] public void Exp_Zero() => Assert.Equal(1f, E.TensorExp(C(0, 1)).GetDataArray()[0], Tol);
    [Fact] public void Log_One() => Assert.True(Math.Abs(E.TensorLog(C(1, 1)).GetDataArray()[0]) < Tol);
    [Fact] public void Exp_Positive() { var d = E.TensorExp(R([64], 32)).GetDataArray(); for (int i = 0; i < d.Length; i++) Assert.True(d[i] > 0f); }
    [Fact] public void Exp_Additive() { var a = R([64], 33); var b = R([64], 34); AE(E.TensorExp(E.TensorAdd(a, b)), E.TensorMultiply(E.TensorExp(a), E.TensorExp(b)), 1e-2f); }

    // ================================================================
    // TRIGONOMETRIC (6)
    // ================================================================
    [Fact] public void Sin_Range() { AR(E.TensorSin(R([64], 40)).GetDataArray(), -1f, 1f, "Sin"); }
    [Fact] public void Cos_Range() { AR(E.TensorCos(R([64], 41)).GetDataArray(), -1f, 1f, "Cos"); }
    [Fact] public void Sin_Zero() => Assert.True(Math.Abs(E.TensorSin(C(0, 1)).GetDataArray()[0]) < Tol);
    [Fact] public void Cos_Zero() => Assert.Equal(1f, E.TensorCos(C(0, 1)).GetDataArray()[0], Tol);
    [Fact] public void Pythagorean_Identity() { var x = R([64], 42); var s = E.TensorSin(x); var c = E.TensorCos(x); AE(E.TensorAdd(E.TensorMultiply(s, s), E.TensorMultiply(c, c)), C(1, 64), 1e-3f); }
    [Fact] public void SinhCosh_Identity() { var x = R([64], 43); var sh = E.TensorSinh(x); var ch = E.TensorCosh(x); AE(E.TensorSubtract(E.TensorMultiply(ch, ch), E.TensorMultiply(sh, sh)), C(1, 64), 1e-3f); }

    // ================================================================
    // ACTIVATIONS (18)
    // ================================================================
    [Fact] public void Sigmoid_Range() { AR(E.TensorSigmoid(R([256], 20)).GetDataArray(), 0f, 1f, "Sigmoid"); }
    [Fact] public void Sigmoid_Symmetry() { var x = R([64], 21); var s = E.TensorSigmoid(x).GetDataArray(); var sn = E.TensorSigmoid(E.TensorNegate(x)).GetDataArray(); for (int i = 0; i < 64; i++) Assert.True(Math.Abs(sn[i] - (1f - s[i])) < Tol); }
    [Fact] public void Sigmoid_Zero() => Assert.Equal(0.5f, E.TensorSigmoid(C(0, 1)).GetDataArray()[0], Tol);
    [Fact] public void Tanh_Range() { AR(E.TensorTanh(R([256], 22)).GetDataArray(), -1f, 1f, "Tanh"); }
    [Fact] public void Tanh_Odd() { var x = R([64], 23); var t = E.TensorTanh(x).GetDataArray(); var tn = E.TensorTanh(E.TensorNegate(x)).GetDataArray(); for (int i = 0; i < 64; i++) Assert.True(Math.Abs(tn[i] + t[i]) < Tol); }
    [Fact] public void Tanh_Zero() => Assert.True(Math.Abs(E.TensorTanh(C(0, 1)).GetDataArray()[0]) < Tol);
    [Fact] public void ReLU_NonNeg() { var d = E.TensorReLU(R([256], 24)).GetDataArray(); for (int i = 0; i < d.Length; i++) Assert.True(d[i] >= 0f); }
    [Fact] public void ReLU_Positive() => AE(E.TensorReLU(new Tensor<float>(new float[] { 1, 2, 3 }, [3])), new Tensor<float>(new float[] { 1, 2, 3 }, [3]));
    [Fact] public void ReLU_Negative() => AZ(E.TensorReLU(new Tensor<float>(new float[] { -1, -2, -3 }, [3])));
    [Fact] public void ReLU_Idempotent() => AE(E.TensorReLU(E.TensorReLU(R([64], 24))), E.TensorReLU(R([64], 24)));
    [Fact] public void GELU_Zero() => Assert.True(Math.Abs(E.TensorGELU(C(0, 1)).GetDataArray()[0]) < Tol);
    [Fact] public void GELU_LargeIdentity() => Assert.True(Math.Abs(E.TensorGELU(C(10, 1)).GetDataArray()[0] - 10f) < 0.01f);
    [Fact] public void SiLU_Zero() => Assert.True(Math.Abs(E.TensorSiLU(C(0, 1)).GetDataArray()[0]) < Tol);
    [Fact] public void Mish_Zero() => Assert.True(Math.Abs(E.TensorMish(C(0, 1)).GetDataArray()[0]) < Tol);
    [Fact] public void LeakyReLU_Positive() => AE(E.TensorLeakyReLU(new Tensor<float>(new float[] { 1, 2, 3 }, [3]), 0.01f), new Tensor<float>(new float[] { 1, 2, 3 }, [3]));
    [Fact] public void LeakyReLU_Negative() => Assert.Equal(-0.1f, E.TensorLeakyReLU(new Tensor<float>(new float[] { -10f }, [1]), 0.01f).GetDataArray()[0], 1e-3f);
    [Fact] public void Sigmoid_Tanh_Relation() { var x = R([64], 25); var sig = E.TensorSigmoid(x).GetDataArray(); var tanh = E.TensorTanh(E.TensorMultiplyScalar(x, 0.5f)).GetDataArray(); for (int i = 0; i < 64; i++) Assert.True(Math.Abs(sig[i] - (0.5f + 0.5f * tanh[i])) < 1e-3f); }
    [Fact] public void HardSwish_Zero() => Assert.True(Math.Abs(E.TensorHardSwish(C(0, 1)).GetDataArray()[0]) < Tol);

    // ================================================================
    // SOFTMAX (6)
    // ================================================================
    [Fact] public void Softmax_SumToOne() { var d = E.Softmax(R([4, 32], 40), -1).GetDataArray(); for (int r = 0; r < 4; r++) { float s = 0; for (int j = 0; j < 32; j++) s += d[r * 32 + j]; Assert.True(Math.Abs(s - 1f) < 1e-3f); } }
    [Fact] public void Softmax_NonNeg() { var d = E.Softmax(R([4, 32], 41), -1).GetDataArray(); for (int i = 0; i < d.Length; i++) Assert.True(d[i] >= 0f); }
    [Fact] public void Softmax_TranslationInvariant() => AE(E.Softmax(R([4, 32], 42), -1), E.Softmax(E.TensorAddScalar(R([4, 32], 42), 100f), -1), 1e-3f);
    [Fact] public void Softmax_MaxGetsHighest() { var d = E.Softmax(new Tensor<float>(new float[] { 1, 10, 2, 3 }, [1, 4]), -1).GetDataArray(); Assert.True(d[1] > d[0] && d[1] > d[2] && d[1] > d[3]); }
    [Fact] public void LogSoftmax_Consistent() => AE(E.TensorLogSoftmax(R([4, 16], 43), -1), E.TensorLog(E.Softmax(R([4, 16], 43), -1)), 1e-3f);
    [Fact] public void TensorSoftmax_SumToOne() { var d = E.TensorSoftmax(R([4, 16], 44), -1).GetDataArray(); for (int r = 0; r < 4; r++) { float s = 0; for (int j = 0; j < 16; j++) s += d[r * 16 + j]; Assert.True(Math.Abs(s - 1f) < 1e-3f); } }

    // ================================================================
    // REDUCTIONS (10)
    // ================================================================
    [Fact] public void Sum_Linearity() { float a = 2.5f, b = -1.3f; float l = E.TensorSum(E.TensorAdd(E.TensorMultiplyScalar(R([256], 50), a), E.TensorMultiplyScalar(R([256], 51), b))); float r = a * E.TensorSum(R([256], 50)) + b * E.TensorSum(R([256], 51)); Assert.True(Math.Abs(l - r) < 0.5f); }
    [Fact] public void Mean_OfConstant() => Assert.Equal(3.14f, E.TensorMean(C(3.14f, 100)), Tol);
    [Fact] public void Sum_OfOnes() => Assert.Equal(256f, E.TensorSum(C(1, 256)), Tol);
    [Fact] public void Sum_OfZeros() => Assert.True(Math.Abs(E.TensorSum(C(0, 256))) < Tol);
    [Fact] public void Mean_IsSumOverN() { var x = R([64], 52); Assert.True(Math.Abs(E.TensorMean(x) - E.TensorSum(x) / 64f) < 1e-3f); }
    [Fact] public void Max_GreaterAll() { var x = R([64], 53); float mx = E.TensorMaxValue(x); var d = x.GetDataArray(); for (int i = 0; i < 64; i++) Assert.True(mx >= d[i] - Tol); }
    [Fact] public void Min_LessAll() { var x = R([64], 54); float mn = E.TensorMinValue(x); var d = x.GetDataArray(); for (int i = 0; i < 64; i++) Assert.True(mn <= d[i] + Tol); }
    [Fact] public void ArgMax_MatchesMax() { var x = R([64], 55); var idx = E.TensorArgMax(x, 0); Assert.Equal(E.TensorMaxValue(x), x.GetDataArray()[(int)idx.GetDataArray()[0]], Tol); }
    [Fact] public void ReduceSum_Axis() { var x = R([4, 8], 56); var r = E.ReduceSum(x, new[] { 1 }, false); Assert.Equal(new[] { 4 }, r.Shape); var xd = x.GetDataArray(); var rd = r.GetDataArray(); for (int i = 0; i < 4; i++) { float s = 0; for (int j = 0; j < 8; j++) s += xd[i * 8 + j]; Assert.True(Math.Abs(rd[i] - s) < 0.01f); } }
    [Fact] public void SumOfSquares_NonNeg() => Assert.True(E.TensorSumOfSquares(R([64], 57)) >= 0f);

    // ================================================================
    // MATRIX OPS (6)
    // ================================================================
    [Fact] public void MatMul_Identity() { var a = new Matrix<float>(4, 4); var id = new Matrix<float>(4, 4); var rng = new Random(10); for (int i = 0; i < 4; i++) { id[i, i] = 1f; for (int j = 0; j < 4; j++) a[i, j] = (float)rng.NextDouble(); } var r = E.MatrixMultiply(a, id); for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) Assert.Equal(a[i, j], r[i, j], Tol); }
    [Fact] public void MatMul_Associative() { var a = new Matrix<float>(3, 4); var b = new Matrix<float>(4, 5); var c = new Matrix<float>(5, 2); var rng = new Random(11); for (int i = 0; i < 3; i++) for (int j = 0; j < 4; j++) a[i, j] = (float)rng.NextDouble(); for (int i = 0; i < 4; i++) for (int j = 0; j < 5; j++) b[i, j] = (float)rng.NextDouble(); for (int i = 0; i < 5; i++) for (int j = 0; j < 2; j++) c[i, j] = (float)rng.NextDouble(); var l = E.MatrixMultiply(E.MatrixMultiply(a, b), c); var r = E.MatrixMultiply(a, E.MatrixMultiply(b, c)); for (int i = 0; i < 3; i++) for (int j = 0; j < 2; j++) Assert.True(Math.Abs(l[i, j] - r[i, j]) < 1e-2f); }
    [Fact] public void MatMul_Transpose() { var a = new Matrix<float>(3, 4); var b = new Matrix<float>(4, 5); var rng = new Random(12); for (int i = 0; i < 3; i++) for (int j = 0; j < 4; j++) a[i, j] = (float)rng.NextDouble(); for (int i = 0; i < 4; i++) for (int j = 0; j < 5; j++) b[i, j] = (float)rng.NextDouble(); var l = E.MatrixMultiply(a, b).Transpose(); var r = E.MatrixMultiply(b.Transpose(), a.Transpose()); for (int i = 0; i < 5; i++) for (int j = 0; j < 3; j++) Assert.True(Math.Abs(l[i, j] - r[i, j]) < Tol); }
    [Fact] public void BatchMatMul_MatchesLoop() { var rng = new Random(13); int B = 4, m = 6, k = 8, n = 5; var ad = new float[B * m * k]; var bd = new float[B * k * n]; for (int i = 0; i < ad.Length; i++) ad[i] = (float)(rng.NextDouble() * 2 - 1); for (int i = 0; i < bd.Length; i++) bd[i] = (float)(rng.NextDouble() * 2 - 1); var r = E.TensorBatchMatMul(new Tensor<float>(ad, [B, m, k]), new Tensor<float>(bd, [B, k, n])).GetDataArray(); for (int bi = 0; bi < B; bi++) for (int i = 0; i < m; i++) for (int j = 0; j < n; j++) { float s = 0; for (int p = 0; p < k; p++) s += ad[bi * m * k + i * k + p] * bd[bi * k * n + p * n + j]; Assert.True(Math.Abs(r[bi * m * n + i * n + j] - s) < 1e-2f); } }
    [Fact] public void DotProduct_Commutative() { var a = new Vector<float>(new float[] { 1, 2, 3, 4 }); var b = new Vector<float>(new float[] { 5, 6, 7, 8 }); Assert.Equal(E.DotProduct(a, b), E.DotProduct(b, a), Tol); }
    [Fact] public void DotProduct_KnownValue() => Assert.Equal(70f, E.DotProduct(new Vector<float>(new float[] { 1, 2, 3, 4 }), new Vector<float>(new float[] { 5, 6, 7, 8 })), Tol);

    // ================================================================
    // SHAPES (8)
    // ================================================================
    [Fact] public void Transpose_Involution() => AE(E.TensorTranspose(E.TensorTranspose(R([4, 8], 90))), R([4, 8], 90));
    [Fact] public void Eye_Identity() { var d = E.TensorEye<float>(5).GetDataArray(); for (int i = 0; i < 5; i++) for (int j = 0; j < 5; j++) Assert.Equal(i == j ? 1f : 0f, d[i * 5 + j], Tol); }
    [Fact] public void Linspace_Endpoints() { var d = E.TensorLinspace(0f, 10f, 11).GetDataArray(); Assert.Equal(0f, d[0], Tol); Assert.Equal(10f, d[10], Tol); }
    [Fact] public void Linspace_EvenSpacing() { var d = E.TensorLinspace(0f, 1f, 5).GetDataArray(); for (int i = 1; i < 5; i++) Assert.True(Math.Abs((d[i] - d[i - 1]) - 0.25f) < Tol); }
    [Fact] public void Diag_Correct() { var d = E.TensorDiag(new Tensor<float>(new float[] { 1, 2, 3 }, [3])).GetDataArray(); Assert.Equal(1f, d[0], Tol); Assert.Equal(0f, d[1], Tol); Assert.Equal(2f, d[4], Tol); Assert.Equal(3f, d[8], Tol); }
    [Fact] public void CumSum_LastEqualsSum() { var x = R([32], 91); var cs = E.TensorCumSum(x, 0).GetDataArray(); Assert.True(Math.Abs(cs[31] - E.TensorSum(x)) < 1e-2f); }
    [Fact] public void Outer_Correct() { var r = E.TensorOuter(new Tensor<float>(new float[] { 1, 2, 3 }, [3]), new Tensor<float>(new float[] { 4, 5 }, [2])); Assert.Equal(new[] { 3, 2 }, r.Shape); Assert.Equal(4f, r.GetDataArray()[0], Tol); Assert.Equal(10f, r.GetDataArray()[3], Tol); }
    [Fact] public void TriangularMask_Shape() => Assert.Equal(new[] { 4, 4 }, E.TensorTriangularMask<float>(4, true, 0).Shape);

    // ================================================================
    // NORMALIZATION (4)
    // ================================================================
    [Fact] public void BatchNorm_ZeroMean() { var x = R([4, 8, 4, 4], 60); var r = E.BatchNorm(x, new Tensor<float>(Enumerable.Repeat(1f, 8).ToArray(), [8]), new Tensor<float>(new float[8], [8]), 1e-5, out _, out _).GetDataArray(); for (int c = 0; c < 8; c++) { float s = 0; int n = 0; for (int bi = 0; bi < 4; bi++) for (int h = 0; h < 4; h++) for (int w = 0; w < 4; w++) { s += r[((bi * 8 + c) * 4 + h) * 4 + w]; n++; } Assert.True(Math.Abs(s / n) < 0.1f); } }
    [Fact] public void LayerNorm_ZeroMean() { var x = R([2, 16], 61); var r = E.LayerNorm(x, new Tensor<float>(Enumerable.Repeat(1f, 16).ToArray(), [16]), new Tensor<float>(new float[16], [16]), 1e-5, out _, out _).GetDataArray(); for (int i = 0; i < 2; i++) { float s = 0; for (int j = 0; j < 16; j++) s += r[i * 16 + j]; Assert.True(Math.Abs(s / 16) < 0.1f); } }
    [Fact] public void CosineSimilarity_SelfIsOne() => Assert.True(Math.Abs(E.CosineSimilarity(new Vector<float>(new float[] { 1, 2, 3, 4 }), new Vector<float>(new float[] { 1, 2, 3, 4 })) - 1f) < Tol);
    [Fact] public void CosineSimilarity_Orthogonal() => Assert.True(Math.Abs(E.CosineSimilarity(new Vector<float>(new float[] { 1, 0 }), new Vector<float>(new float[] { 0, 1 }))) < Tol);

    // ================================================================
    // CONV / POOLING (4)
    // ================================================================
    [Fact] public void Conv2D_IdentityKernel() => AE(E.Conv2D(R([1, 1, 4, 4], 80), new Tensor<float>(new float[] { 1f }, [1, 1, 1, 1]), 1, 0, 1), R([1, 1, 4, 4], 80), 1e-3f);
    [Fact] public void Conv2D_OutputShape() => Assert.Equal(new[] { 1, 16, 8, 8 }, E.Conv2D(R([1, 3, 8, 8], 81), R([16, 3, 3, 3], 82), 1, 1, 1).Shape);
    [Fact] public void MaxPool_Shape() => Assert.Equal(new[] { 1, 1, 2, 2 }, E.MaxPool2D(R([1, 1, 4, 4], 83), 2, 2, 0).Shape);
    [Fact] public void AvgPool_Shape() => Assert.Equal(new[] { 1, 1, 2, 2 }, E.AvgPool2D(R([1, 1, 4, 4], 84), 2, 2, 0).Shape);

    // ================================================================
    // ATTENTION (4)
    // ================================================================
    [Fact] public void Attention_WeightsSumToOne() { E.ScaledDotProductAttention(R([1, 1, 4, 8], 70), R([1, 1, 4, 8], 71), R([1, 1, 4, 8], 72), null, 1.0 / Math.Sqrt(8), out var w); var wd = w.GetDataArray(); for (int r = 0; r < 4; r++) { float s = 0; for (int c = 0; c < 4; c++) s += wd[r * 4 + c]; Assert.True(Math.Abs(s - 1f) < 1e-3f); } }
    [Fact] public void Attention_WeightsNonNeg() { E.ScaledDotProductAttention(R([1, 1, 4, 8], 70), R([1, 1, 4, 8], 71), R([1, 1, 4, 8], 72), null, 1.0 / Math.Sqrt(8), out var w); var wd = w.GetDataArray(); for (int i = 0; i < wd.Length; i++) Assert.True(wd[i] >= 0f); }
    [Fact] public void AttentionBackward_NonZero() { var sh = new[] { 1, 1, 3, 4 }; var Q = R(sh, 73); var K = R(sh, 74); var V = R(sh, 75); E.ScaledDotProductAttention(Q, K, V, null, 0.5, out var aw); E.ScaledDotProductAttentionBackward(R(sh, 76), Q, K, V, aw, 0.5, out var dQ, out var dK, out var dV); Assert.True(dQ.GetDataArray().Any(x => Math.Abs(x) > 1e-7f)); Assert.True(dK.GetDataArray().Any(x => Math.Abs(x) > 1e-7f)); Assert.True(dV.GetDataArray().Any(x => Math.Abs(x) > 1e-7f)); }
    [Fact] public void AttentionBackward_NumericalGradient() { var sh = new[] { 1, 1, 3, 4 }; float eps = 1e-3f; var vd = new float[12]; var rng = new Random(77); for (int i = 0; i < 12; i++) vd[i] = (float)(rng.NextDouble() * 0.5); var Q = R(sh, 78); var K = R(sh, 79); var grad = new Tensor<float>(Enumerable.Repeat(1f, 12).ToArray(), sh); E.ScaledDotProductAttention(Q, K, new Tensor<float>((float[])vd.Clone(), sh), null, 0.5, out var aw); E.ScaledDotProductAttentionBackward(grad, Q, K, new Tensor<float>((float[])vd.Clone(), sh), aw, 0.5, out _, out _, out var dV); var dvd = dV.GetDataArray(); int p = 0, c = 0; for (int i = 0; i < 8; i++) { float o = vd[i]; vd[i] = o + eps; float lp = SA(Q, K, new Tensor<float>((float[])vd.Clone(), sh)); vd[i] = o - eps; float lm = SA(Q, K, new Tensor<float>((float[])vd.Clone(), sh)); vd[i] = o; float ng = (lp - lm) / (2 * eps); float re = Math.Abs(ng) > 1e-5f ? Math.Abs(ng - dvd[i]) / Math.Abs(ng) : Math.Abs(ng - dvd[i]); c++; if (re < 0.1f) p++; } Assert.True(p >= c * 0.75, $"dV: {p}/{c}"); }
    private float SA(Tensor<float> q, Tensor<float> k, Tensor<float> v) { var o = E.ScaledDotProductAttention(q, k, v, null, 0.5, out _); float s = 0; var d = o.GetDataArray(); for (int i = 0; i < d.Length; i++) s += d[i]; return s; }

    // ================================================================
    // GATED ACTIVATIONS (4)
    // ================================================================
    [Fact] public void GLU_HalfDim() => Assert.Equal(new[] { 4, 8 }, E.GLU(R([4, 16], 100), -1).Shape);
    [Fact] public void GeGLU_HalfDim() => Assert.Equal(new[] { 4, 8 }, E.GeGLU(R([4, 16], 101), -1).Shape);
    [Fact] public void SwiGLU_HalfDim() => Assert.Equal(new[] { 4, 8 }, E.SwiGLU(R([4, 16], 102), -1).Shape);
    [Fact] public void ReGLU_HalfDim() => Assert.Equal(new[] { 4, 8 }, E.ReGLU(R([4, 16], 103), -1).Shape);

    // ================================================================
    // LOSS / COMPARISON (4)
    // ================================================================
    [Fact] public void BCE_LowForCorrect() { var loss = E.TensorBinaryCrossEntropy(new Tensor<float>(new float[] { 0.99f, 0.01f }, [2]), new Tensor<float>(new float[] { 1f, 0f }, [2]), 1e-7f); Assert.True(E.TensorMean(loss) < 0.1f); }
    [Fact] public void BCE_NonNeg() { var d = E.TensorBinaryCrossEntropy(RP([64], 110), new Tensor<float>(Enumerable.Repeat(0.5f, 64).ToArray(), [64]), 1e-7f).GetDataArray(); for (int i = 0; i < d.Length; i++) Assert.True(d[i] >= -Tol); }
    [Fact] public void Equals_Reflexive() => AE(E.TensorEquals(R([64], 120), R([64], 120)), C(1, 64));
    [Fact] public void NotEquals_Complement() { var a = R([64], 121); var b = R([64], 122); var eq = E.TensorEquals(a, b).GetDataArray(); var ne = E.TensorNotEquals(a, b).GetDataArray(); for (int i = 0; i < 64; i++) Assert.True(Math.Abs((eq[i] + ne[i]) - 1f) < Tol); }
}
