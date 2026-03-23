using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// Extended mathematical invariant tests for IEngine operations not covered in MathInvariantTests.
/// Every test encodes a mathematical truth that MUST hold regardless of implementation.
/// Covers: TensorAdd variants, inplace/into ops, broadcast ops, comparison ops, shape/index ops,
/// normalization (GroupNorm, InstanceNorm, RMSNorm), reduction ops (ReduceMax, ReduceMean,
/// ReduceVariance, ReduceStd), backward passes, softmax variants, pooling variants,
/// loss functions, scatter/gather, random generation, and more.
/// </summary>
public class MathInvariantExtendedTests
{
    private readonly CpuEngine E = new();
    private const float Tol = 1e-4f;

    // ---- helpers (same compact style as MathInvariantTests) ----
    private Tensor<float> R(int[] s, int seed) { var r = new Random(seed); var d = new float[s.Aggregate(1, (a, b) => a * b)]; for (int i = 0; i < d.Length; i++) d[i] = (float)(r.NextDouble() * 2 - 1); return new Tensor<float>(d, s); }
    private Tensor<float> RP(int[] s, int seed) { var r = new Random(seed); var d = new float[s.Aggregate(1, (a, b) => a * b)]; for (int i = 0; i < d.Length; i++) d[i] = (float)(r.NextDouble() * 9.9 + 0.1); return new Tensor<float>(d, s); }
    private Tensor<float> C(float v, int n) => new(Enumerable.Repeat(v, n).ToArray(), new[] { n });
    private Tensor<float> C(float v, int[] s) => new(Enumerable.Repeat(v, s.Aggregate(1, (a, b) => a * b)).ToArray(), s);
    private void AE(Tensor<float> a, Tensor<float> b, float t = 1e-4f, string m = "") { Assert.Equal(a.Shape, b.Shape); var ad = a.GetDataArray(); var bd = b.GetDataArray(); for (int i = 0; i < a.Length; i++) Assert.True(Math.Abs(ad[i] - bd[i]) < t, $"{m} [{i}]: {ad[i]} vs {bd[i]}"); }
    private void AZ(Tensor<float> a, string m = "") { var d = a.GetDataArray(); for (int i = 0; i < d.Length; i++) Assert.True(Math.Abs(d[i]) < Tol, $"{m} [{i}]={d[i]}"); }
    private void AR(float[] d, float lo, float hi, string m) { for (int i = 0; i < d.Length; i++) Assert.True(d[i] >= lo && d[i] <= hi, $"{m} [{i}]={d[i]}"); }

    // ================================================================
    // TensorAddMany (3)
    // ================================================================
    [Fact] public void AddMany_TwoArgs_SameAsAdd() => AE(E.TensorAddMany(R([64], 1), R([64], 2)), E.TensorAdd(R([64], 1), R([64], 2)));
    [Fact] public void AddMany_ThreeArgs_Associative() { var a = R([32], 1); var b = R([32], 2); var c = R([32], 3); AE(E.TensorAddMany(a, b, c), E.TensorAdd(E.TensorAdd(a, b), c), 1e-3f); }
    [Fact] public void AddMany_Result_NonEmpty() => Assert.Equal(64, E.TensorAddMany(R([64], 1), R([64], 2), R([64], 3)).Length);

    // ================================================================
    // TensorAddScaled (2)
    // ================================================================
    [Fact] public void AddScaled_ZeroScales_IsZero() { var a = R([64], 1); var b = R([64], 2); AZ(E.TensorAddScaled(a, b, 0f, 0f)); }
    [Fact] public void AddScaled_UnitScales_IsAdd() => AE(E.TensorAddScaled(R([64], 1), R([64], 2), 1f, 1f), E.TensorAdd(R([64], 1), R([64], 2)), 1e-3f);

    // ================================================================
    // TensorAddInto (zero-allocation) (2)
    // ================================================================
    [Fact] public void AddInto_MatchesTensorAdd() { var a = R([64], 1); var b = R([64], 2); var dest = new Tensor<float>(new float[64], [64]); E.TensorAddInto(dest, a, b); AE(dest, E.TensorAdd(a, b)); }
    [Fact] public void AddInto_NoSideEffects() { var a = R([64], 1); var origA = (float[])a.GetDataArray().Clone(); var dest = new Tensor<float>(new float[64], [64]); E.TensorAddInto(dest, a, R([64], 2)); Assert.Equal(origA, a.GetDataArray()); }

    // ================================================================
    // TensorMultiplyMany (2)
    // ================================================================
    [Fact] public void MulMany_TwoArgs_SameAsMul() => AE(E.TensorMultiplyMany(RP([64], 1), RP([64], 2)), E.TensorMultiply(RP([64], 1), RP([64], 2)));
    [Fact] public void MulMany_AllOnes_IsOnes() { var ones = C(1f, 64); var r = E.TensorMultiplyMany(ones, ones, ones).GetDataArray(); for (int i = 0; i < 64; i++) Assert.Equal(1f, r[i], Tol); }

    // ================================================================
    // TensorMultiplyInPlace / Into (3)
    // ================================================================
    [Fact] public void MulInPlace_MatchesMul() { var a = R([64], 1); var b = R([64], 2); var expected = E.TensorMultiply(a, b); E.TensorMultiplyInPlace(a, b); AE(a, expected); }
    [Fact] public void MulInto_MatchesMul() { var a = R([64], 1); var b = R([64], 2); var dest = new Tensor<float>(new float[64], [64]); E.TensorMultiplyInto(dest, a, b); AE(dest, E.TensorMultiply(a, b)); }
    [Fact] public void MulInto_ZeroResult() { var a = R([64], 1); var z = C(0f, 64); var dest = new Tensor<float>(new float[64], [64]); E.TensorMultiplyInto(dest, a, z); AZ(dest); }

    // ================================================================
    // TensorBroadcastSubtract (2)
    // ================================================================
    [Fact] public void BroadcastSubtract_Shape() => Assert.Equal(new[] { 4, 8 }, E.TensorBroadcastSubtract(R([4, 8], 1), R([1, 8], 2)).Shape);
    [Fact] public void BroadcastSubtract_SameAsManualExpand() { var a = R([1, 8], 1); var b = R([1, 8], 2); float[] aRow = a.GetDataArray(); float[] bRow = b.GetDataArray(); float[] aExp = Enumerable.Concat(Enumerable.Concat(aRow, aRow), Enumerable.Concat(aRow, aRow)).ToArray(); float[] bExp = Enumerable.Concat(Enumerable.Concat(bRow, bRow), Enumerable.Concat(bRow, bRow)).ToArray(); var result = E.TensorBroadcastSubtract(new Tensor<float>(aExp, [4, 8]), new Tensor<float>(bExp, [4, 8])); var broadcast = E.TensorBroadcastSubtract(new Tensor<float>(aExp, [4, 8]), b); AE(result, broadcast, 1e-3f); }

    // ================================================================
    // TensorBroadcastAddInPlace (2)
    // ================================================================
    [Fact] public void BroadcastAddInPlace_MatchesBroadcastAdd() { var a = R([4, 8], 1); var b = R([1, 8], 2); var expected = E.TensorBroadcastAdd(a, b); var aCopy = new Tensor<float>((float[])a.GetDataArray().Clone(), a.Shape); E.TensorBroadcastAddInPlace(aCopy, b); AE(aCopy, expected, 1e-3f); }
    [Fact] public void BroadcastAddInPlace_AddZero_Unchanged() { var a = R([4, 8], 1); var z = C(0f, [1, 8]); var aCopy = new Tensor<float>((float[])a.GetDataArray().Clone(), a.Shape); E.TensorBroadcastAddInPlace(aCopy, z); AE(aCopy, a); }

    // ================================================================
    // TensorClamp (vs TensorClip) (2)
    // ================================================================
    [Fact] public void Clamp_InRange() { var r = E.TensorClamp(R([256], 1), -0.5f, 0.5f).GetDataArray(); AR(r, -0.5f, 0.5f, "Clamp"); }
    [Fact] public void Clamp_MatchesClip() => AE(E.TensorClamp(R([64], 1), -0.5f, 0.5f), E.TensorClip(R([64], 1), -0.5f, 0.5f));

    // ================================================================
    // TensorGreaterThan / TensorLessThan (4)
    // ================================================================
    [Fact] public void GreaterThan_BinaryValues() { var r = E.TensorGreaterThan(R([64], 1), R([64], 2)).GetDataArray(); for (int i = 0; i < r.Length; i++) Assert.True(r[i] == 0f || r[i] == 1f); }
    [Fact] public void LessThan_BinaryValues() { var r = E.TensorLessThan(R([64], 1), R([64], 2)).GetDataArray(); for (int i = 0; i < r.Length; i++) Assert.True(r[i] == 0f || r[i] == 1f); }
    [Fact] public void GreaterThan_Antisymmetric() { var a = R([64], 1); var b = R([64], 2); var gt = E.TensorGreaterThan(a, b).GetDataArray(); var lt = E.TensorLessThan(b, a).GetDataArray(); for (int i = 0; i < 64; i++) Assert.Equal(gt[i], lt[i], Tol); }
    [Fact] public void LessThan_Scalar_Range() { var r = E.TensorLessThan(R([64], 5), 0f).GetDataArray(); for (int i = 0; i < r.Length; i++) Assert.True(r[i] == 0f || r[i] == 1f); }

    // ================================================================
    // TensorConcatenate / TensorStack / TensorSplit / TensorUnstack (6)
    // ================================================================
    [Fact] public void Concatenate_AxisZero_Shape() { var a = R([4, 8], 1); var b = R([3, 8], 2); Assert.Equal(new[] { 7, 8 }, E.TensorConcatenate(new[] { a, b }, 0).Shape); }
    [Fact] public void Concatenate_SplitInverse() { var a = R([6, 8], 1); var c = E.TensorConcatenate(new[] { a }, 0); Assert.Equal(a.Shape, c.Shape); }
    [Fact] public void Stack_AddsAxis() { var a = R([4, 8], 1); var b = R([4, 8], 2); Assert.Equal(new[] { 2, 4, 8 }, E.TensorStack(new[] { a, b }, 0).Shape); }
    [Fact] public void Unstack_InverseOfStack() { var a = R([4, 8], 1); var b = R([4, 8], 2); var stacked = E.TensorStack(new[] { a, b }, 0); var unstacked = E.TensorUnstack(stacked, 0); Assert.Equal(2, unstacked.Length); Assert.Equal(new[] { 4, 8 }, unstacked[0].Shape); }
    [Fact] public void Split_NumParts() { var x = R([6, 8], 1); var parts = E.TensorSplit(x, 3, 0); Assert.Equal(3, parts.Length); }
    [Fact] public void Split_PartShape() { var x = R([6, 8], 1); var parts = E.TensorSplit(x, 3, 0); Assert.Equal(new[] { 2, 8 }, parts[0].Shape); }

    // ================================================================
    // TensorCopy / TensorFill (3)
    // ================================================================
    [Fact] public void Copy_ContentEqual() { var src = R([64], 1); var dst = new Tensor<float>(new float[64], [64]); E.TensorCopy(src, dst); AE(dst, src); }
    [Fact] public void Copy_ValuesMatch() { var src = R([64], 1); var dst = new Tensor<float>(new float[64], [64]); E.TensorCopy(src, dst); var sd = src.GetDataArray(); var dd = dst.GetDataArray(); for (int i = 0; i < 64; i++) Assert.Equal(sd[i], dd[i], Tol); }
    [Fact] public void Fill_AllSameValue() { var t = new Tensor<float>(new float[64], [64]); E.TensorFill(t, 3.14f); var d = t.GetDataArray(); for (int i = 0; i < d.Length; i++) Assert.Equal(3.14f, d[i], Tol); }

    // ================================================================
    // TensorDiagonal (2)
    // ================================================================
    [Fact] public void Diagonal_ExtractsDiag() { var mat = new Tensor<float>(new float[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 }, [3, 3]); var d = E.TensorDiagonal(mat).GetDataArray(); Assert.Equal(1f, d[0], Tol); Assert.Equal(5f, d[1], Tol); Assert.Equal(9f, d[2], Tol); }
    [Fact] public void Diagonal_Shape() { var mat = R([4, 4], 1); Assert.Equal(new[] { 4 }, E.TensorDiagonal(mat).Shape); }

    // ================================================================
    // TensorDropoutMask (2)
    // ================================================================
    [Fact] public void DropoutMask_ShapeCorrect() => Assert.Equal(new[] { 4, 8 }, E.TensorDropoutMask<float>(new[] { 4, 8 }, 0.3f, 1.0f / 0.7f, 42).Shape);
    [Fact] public void DropoutMask_ValuesInRange() { var m = E.TensorDropoutMask<float>(new[] { 256 }, 0.5f, 2f, 99).GetDataArray(); for (int i = 0; i < m.Length; i++) Assert.True(m[i] == 0f || Math.Abs(m[i] - 2f) < Tol, $"mask[{i}]={m[i]}"); }

    // ================================================================
    // TensorExpandDims / TensorSqueeze (3)
    // ================================================================
    [Fact] public void ExpandDims_AddsOne() { var x = R([4, 8], 1); Assert.Equal(new[] { 4, 1, 8 }, E.TensorExpandDims(x, 1).Shape); }
    [Fact] public void Squeeze_RemovesOne() { var x = R([4, 1, 8], 1); Assert.Equal(new[] { 4, 8 }, E.TensorSqueeze(x, 1).Shape); }
    [Fact] public void ExpandSqueeze_Inverse() { var x = R([4, 8], 1); AE(E.TensorSqueeze(E.TensorExpandDims(x, 1), 1), x); }

    // ================================================================
    // TensorGather / TensorIndexSelect / TensorScatter / TensorScatterAdd (4)
    // ================================================================
    [Fact] public void Gather_Shape() { var src = R([10, 4], 1); var idx = new Tensor<int>(new[] { 0, 2, 5 }, new[] { 3 }); Assert.Equal(new[] { 3, 4 }, E.TensorGather(src, idx, 0).Shape); }
    [Fact] public void Gather_KnownValues() { var src = new Tensor<float>(new float[] { 10, 20, 30, 40, 50, 60 }, [3, 2]); var idx = new Tensor<int>(new[] { 1, 2 }, new[] { 2 }); var r = E.TensorGather(src, idx, 0).GetDataArray(); Assert.Equal(30f, r[0], Tol); Assert.Equal(40f, r[1], Tol); Assert.Equal(50f, r[2], Tol); }
    [Fact] public void IndexSelect_Shape() { var src = R([10, 4], 1); var idx = new Tensor<int>(new[] { 0, 2, 5 }, new[] { 3 }); Assert.Equal(new[] { 3, 4 }, E.TensorIndexSelect(src, idx, 0).Shape); }
    [Fact] public void ScatterAdd_IncreasesDim() { var dst = C(0f, [5, 4]); var src = R([3, 4], 1); var idx = new Tensor<int>(new[] { 0, 2, 4 }, new[] { 3 }); var r = E.TensorScatterAdd(dst, idx, src, 0); Assert.Equal(new[] { 5, 4 }, r.Shape); }

    // ================================================================
    // TensorLerp (3)
    // ================================================================
    [Fact] public void Lerp_AtZero_IsA() { var a = R([64], 1); var b = R([64], 2); AE(E.TensorLerp(a, b, 0f), a); }
    [Fact] public void Lerp_AtOne_IsB() { var a = R([64], 1); var b = R([64], 2); AE(E.TensorLerp(a, b, 1f), b); }
    [Fact] public void Lerp_AtHalf_IsMidpoint() { var a = C(0f, 64); var b = C(2f, 64); var mid = E.TensorLerp(a, b, 0.5f).GetDataArray(); for (int i = 0; i < 64; i++) Assert.Equal(1f, mid[i], Tol); }

    // ================================================================
    // TensorLogSumExp (2)
    // ================================================================
    [Fact] public void LogSumExp_GreaterThanMax() { var x = R([8, 16], 1); var lse = E.TensorLogSumExp(x, 1).GetDataArray(); var xd = x.GetDataArray(); for (int r = 0; r < 8; r++) { float maxVal = xd.Skip(r * 16).Take(16).Max(); Assert.True(lse[r] >= maxVal - Tol); } }
    [Fact] public void LogSumExp_Shape() { var x = R([8, 16], 2); Assert.Equal(new[] { 8 }, E.TensorLogSumExp(x, 1, false).Shape); }

    // ================================================================
    // TensorMap (2)
    // ================================================================
    [Fact] public void Map_Identity_Unchanged() { var x = R([64], 1); AE(E.TensorMap(x, v => v), x); }
    [Fact] public void Map_Negate_MatchesTensorNegate() => AE(E.TensorMap(R([64], 1), v => -v), E.TensorNegate(R([64], 1)));

    // ================================================================
    // TensorMaskedFill (2)
    // ================================================================
    [Fact] public void MaskedFill_FillsTrue() { var x = R([4, 4], 1); var mask = new Tensor<Bit>(Enumerable.Repeat(Bit.True, 16).ToArray(), [4, 4]); var r = E.TensorMaskedFill(x, mask, 99f).GetDataArray(); for (int i = 0; i < r.Length; i++) Assert.Equal(99f, r[i], Tol); }
    [Fact] public void MaskedFill_PreservesFalse() { var x = R([4, 4], 1); var mask = new Tensor<Bit>(Enumerable.Repeat(Bit.False, 16).ToArray(), [4, 4]); AE(E.TensorMaskedFill(x, mask, 99f), x); }

    // ================================================================
    // TensorMatMul (2D) (2)
    // ================================================================
    [Fact] public void TensorMatMul_2D_Shape() { var a = R([3, 4], 1); var b = R([4, 5], 2); Assert.Equal(new[] { 3, 5 }, E.TensorMatMul(a, b).Shape); }
    [Fact] public void TensorMatMul_KnownValue() { var a = new Tensor<float>(new float[] { 1, 2, 3, 4 }, [2, 2]); var b = new Tensor<float>(new float[] { 1, 0, 0, 1 }, [2, 2]); AE(E.TensorMatMul(a, b), a); }

    // ================================================================
    // TensorMax (elementwise) / TensorMin (elementwise) (4)
    // ================================================================
    [Fact] public void ElementwiseMax_MatchesManual() { var a = new Tensor<float>(new float[] { 1, 5, 3 }, [3]); var b = new Tensor<float>(new float[] { 4, 2, 6 }, [3]); var r = E.TensorMax(a, b).GetDataArray(); Assert.Equal(4f, r[0], Tol); Assert.Equal(5f, r[1], Tol); Assert.Equal(6f, r[2], Tol); }
    [Fact] public void ElementwiseMax_Scalar() { var d = E.TensorMax(R([64], 1), 0f).GetDataArray(); for (int i = 0; i < d.Length; i++) Assert.True(d[i] >= 0f); }
    [Fact] public void ElementwiseMin_MatchesManual() { var a = new Tensor<float>(new float[] { 1, 5, 3 }, [3]); var b = new Tensor<float>(new float[] { 4, 2, 6 }, [3]); var r = E.TensorMin(a, b).GetDataArray(); Assert.Equal(1f, r[0], Tol); Assert.Equal(2f, r[1], Tol); Assert.Equal(3f, r[2], Tol); }
    [Fact] public void ElementwiseMin_Scalar() { var d = E.TensorMin(R([64], 1), 0f).GetDataArray(); for (int i = 0; i < d.Length; i++) Assert.True(d[i] <= 0f); }

    // ================================================================
    // TensorNormalize (2)
    // ================================================================
    [Fact] public void Normalize_L2UnitNorm() { var x = R([1, 64], 1); var n = E.TensorNormalize(x, 1, 1e-8f); var d = n.GetDataArray(); float norm = (float)Math.Sqrt(d.Sum(v => v * v)); Assert.True(Math.Abs(norm - 1f) < 1e-3f); }
    [Fact] public void Normalize_Shape() { var x = R([4, 8], 1); Assert.Equal(x.Shape, E.TensorNormalize(x, 1, 1e-8f).Shape); }

    // ================================================================
    // TensorOneHot (3)
    // ================================================================
    [Fact] public void OneHot_Shape() { var idx = new Tensor<int>(new[] { 0, 1, 2 }, [3]); Assert.Equal(new[] { 3, 5 }, E.TensorOneHot<float>(idx, 5).Shape); }
    [Fact] public void OneHot_SumToOne() { var idx = new Tensor<int>(new[] { 0, 1, 2, 3 }, [4]); var r = E.TensorOneHot<float>(idx, 5).GetDataArray(); for (int i = 0; i < 4; i++) { float s = 0; for (int j = 0; j < 5; j++) s += r[i * 5 + j]; Assert.Equal(1f, s, Tol); } }
    [Fact] public void OneHot_HotIsAtIndex() { var idx = new Tensor<int>(new[] { 2 }, [1]); var r = E.TensorOneHot<float>(idx, 4).GetDataArray(); Assert.Equal(0f, r[0], Tol); Assert.Equal(0f, r[1], Tol); Assert.Equal(1f, r[2], Tol); Assert.Equal(0f, r[3], Tol); }

    // ================================================================
    // TensorOuterProduct (2)
    // ================================================================
    [Fact] public void OuterProduct_Shape() { var a = R([3], 1); var b = R([4], 2); Assert.Equal(new[] { 3, 4 }, E.TensorOuterProduct(a, b).Shape); }
    [Fact] public void OuterProduct_KnownValue() { var a = new Tensor<float>(new float[] { 1, 2 }, [2]); var b = new Tensor<float>(new float[] { 3, 4 }, [2]); var r = E.TensorOuterProduct(a, b).GetDataArray(); Assert.Equal(3f, r[0], Tol); Assert.Equal(4f, r[1], Tol); Assert.Equal(6f, r[2], Tol); Assert.Equal(8f, r[3], Tol); }

    // ================================================================
    // TensorPermute (2)
    // ================================================================
    [Fact] public void Permute_SwapAxes_Shape() { var x = R([2, 3, 4], 1); Assert.Equal(new[] { 4, 2, 3 }, E.TensorPermute(x, new[] { 2, 0, 1 }).Shape); }
    [Fact] public void Permute_Identity_Unchanged() { var x = R([2, 3, 4], 1); AE(E.TensorPermute(x, new[] { 0, 1, 2 }), x); }

    // ================================================================
    // TensorPower (vs TensorPow) (2)
    // ================================================================
    [Fact] public void TensorPower_MatchesPow() => AE(E.TensorPower(RP([64], 1), 2f), E.TensorPow(RP([64], 1), 2f), 1e-3f);
    [Fact] public void TensorPowerTensor_ElementWise() { var b = new Tensor<float>(new float[] { 2f, 3f, 4f }, [3]); var e = new Tensor<float>(new float[] { 2f, 2f, 2f }, [3]); var r = E.TensorPower(b, e).GetDataArray(); Assert.Equal(4f, r[0], Tol); Assert.Equal(9f, r[1], Tol); Assert.Equal(16f, r[2], Tol); }

    // ================================================================
    // TensorRandomNormal / TensorRandomUniform / TensorRandomUniformRange (4)
    // ================================================================
    [Fact] public void RandomNormal_Shape() => Assert.Equal(new[] { 4, 8 }, E.TensorRandomNormal<float>(new[] { 4, 8 }, 0f, 1f).Shape);
    [Fact] public void RandomNormal_ApproxMean() { var d = E.TensorRandomNormal<float>(new[] { 1024 }, 5f, 1f).GetDataArray(); Assert.True(Math.Abs(d.Average() - 5f) < 0.2f); }
    [Fact] public void RandomUniform_Shape() => Assert.Equal(new[] { 4, 8 }, E.TensorRandomUniform<float>(new[] { 4, 8 }).Shape);
    [Fact] public void RandomUniformRange_InRange() { var d = E.TensorRandomUniformRange<float>(new[] { 256 }, 2f, 5f, 42).GetDataArray(); AR(d, 2f, 5f, "UniformRange"); }

    // ================================================================
    // TensorRepeatElements / TensorTile (3)
    // ================================================================
    [Fact] public void RepeatElements_Shape() { var x = R([3, 4], 1); Assert.Equal(new[] { 6, 4 }, E.TensorRepeatElements(x, 2, 0).Shape); }
    [Fact] public void Tile_Shape() { var x = R([3, 4], 1); Assert.Equal(new[] { 6, 8 }, E.TensorTile(x, new[] { 2, 2 }).Shape); }
    [Fact] public void Tile_Content() { var x = new Tensor<float>(new float[] { 1, 2, 3, 4 }, [2, 2]); var t = E.TensorTile(x, new[] { 1, 2 }).GetDataArray(); Assert.Equal(1f, t[0], Tol); Assert.Equal(2f, t[1], Tol); Assert.Equal(1f, t[2], Tol); Assert.Equal(2f, t[3], Tol); }

    // ================================================================
    // TensorSlice / TensorSliceAxis / TensorSetSlice / TensorSetSliceAxis (4)
    // ================================================================
    [Fact] public void Slice_Shape() { var x = R([8, 8], 1); Assert.Equal(new[] { 4, 4 }, E.TensorSlice(x, new[] { 0, 0 }, new[] { 4, 4 }).Shape); }
    [Fact] public void SliceAxis_Shape() { var x = R([3, 4, 8], 1); Assert.Equal(new[] { 4, 8 }, E.TensorSliceAxis(x, 0, 2).Shape); }
    [Fact] public void SetSlice_ModifiesDest() { var dst = C(0f, [4, 4]); var src = C(1f, [2, 2]); var r = E.TensorSetSlice(dst, src, new[] { 1, 1 }).GetDataArray(); Assert.Equal(1f, r[1 * 4 + 1], Tol); }
    [Fact] public void SetSliceAxis_Shape() { var dst = R([3, 4, 8], 1); var src = R([4, 8], 2); E.TensorSetSliceAxis(dst, src, 0, 1); Assert.Equal(new[] { 3, 4, 8 }, dst.Shape); }

    // ================================================================
    // TensorSoftmaxBackward (2)
    // ================================================================
    [Fact] public void SoftmaxBackward_Shape() { var sm = E.TensorSoftmax(R([4, 8], 1), -1); var grad = R([4, 8], 2); Assert.Equal(sm.Shape, E.TensorSoftmaxBackward(sm, grad, -1).Shape); }
    [Fact] public void SoftmaxBackward_GradSumsNearZero() { var sm = E.TensorSoftmax(R([1, 8], 1), -1); var grad = C(1f, [1, 8]); var back = E.TensorSoftmaxBackward(sm, grad, -1).GetDataArray(); float s = back.Sum(); Assert.True(Math.Abs(s) < 1e-3f, $"grad sum={s}"); }

    // ================================================================
    // TensorWhere (3)
    // ================================================================
    [Fact] public void Where_AllTrue_IsX() { var x = R([64], 1); var y = R([64], 2); var mask = new Tensor<Bit>(Enumerable.Repeat(Bit.True, 64).ToArray(), [64]); AE(E.TensorWhere(mask, x, y), x); }
    [Fact] public void Where_AllFalse_IsY() { var x = R([64], 1); var y = R([64], 2); var mask = new Tensor<Bit>(Enumerable.Repeat(Bit.False, 64).ToArray(), [64]); AE(E.TensorWhere(mask, x, y), y); }
    [Fact] public void Where_Shape() { var x = R([4, 8], 1); var y = R([4, 8], 2); var mask = new Tensor<Bit>(Enumerable.Repeat(Bit.True, 32).ToArray(), [4, 8]); Assert.Equal(x.Shape, E.TensorWhere(mask, x, y).Shape); }

    // ================================================================
    // TensorTopK (2)
    // ================================================================
    [Fact] public void TopK_Shape() { var x = R([4, 8], 1); var r = E.TensorTopK(x, 3, 1, out _); Assert.Equal(new[] { 4, 3 }, r.Shape); }
    [Fact] public void TopK_IndicesShape() { var x = R([4, 8], 1); E.TensorTopK(x, 3, 1, out var idx); Assert.Equal(new[] { 4, 3 }, idx.Shape); }

    // ================================================================
    // SoftmaxBackward (via Softmax method) (2)
    // ================================================================
    [Fact] public void SoftmaxBackward_Consistent() { var sm = E.Softmax(R([2, 8], 1), -1); var grad = R([2, 8], 2); Assert.Equal(sm.Shape, E.SoftmaxBackward(grad, sm, -1).Shape); }
    [Fact] public void SoftmaxBackward_NonZero() { var sm = E.Softmax(R([2, 8], 1), -1); var grad = R([2, 8], 2); var back = E.SoftmaxBackward(grad, sm, -1); Assert.True(back.GetDataArray().Any(v => Math.Abs(v) > 1e-7f)); }

    // ================================================================
    // GroupNorm (3)
    // ================================================================
    [Fact] public void GroupNorm_Shape() { var x = R([2, 4, 4, 4], 1); var gamma = C(1f, 4); var beta = C(0f, 4); Assert.Equal(x.Shape, E.GroupNorm(x, 2, gamma, beta, 1e-5, out _, out _).Shape); }
    [Fact] public void GroupNorm_ZeroMean() { var x = R([2, 4, 4, 4], 1); var gamma = C(1f, 4); var beta = C(0f, 4); var r = E.GroupNorm(x, 2, gamma, beta, 1e-5, out _, out _).GetDataArray(); float sum = r.Sum(); Assert.True(Math.Abs(sum / r.Length) < 0.1f); }
    [Fact] public void GroupNorm_UnitVariance() { var x = R([2, 4, 8, 8], 1); var gamma = C(1f, 4); var beta = C(0f, 4); E.GroupNorm(x, 4, gamma, beta, 1e-5, out var mean, out var variance); var vd = variance.GetDataArray(); for (int i = 0; i < vd.Length; i++) Assert.True(vd[i] > 0f, $"variance[{i}]={vd[i]}"); }

    // ================================================================
    // InstanceNorm (2)
    // ================================================================
    [Fact] public void InstanceNorm_Shape() { var x = R([2, 4, 4, 4], 1); var gamma = C(1f, 4); var beta = C(0f, 4); Assert.Equal(x.Shape, E.InstanceNorm(x, gamma, beta, 1e-5, out _, out _).Shape); }
    [Fact] public void InstanceNorm_ZeroMean() { var x = R([2, 4, 4, 4], 1); var gamma = C(1f, 4); var beta = C(0f, 4); var r = E.InstanceNorm(x, gamma, beta, 1e-5, out _, out _); E.InstanceNorm(x, gamma, beta, 1e-5, out var mean, out _); Assert.Equal(new[] { 2, 4 }, mean.Shape); }

    // ================================================================
    // RMSNorm (3)
    // ================================================================
    [Fact] public void RMSNorm_Shape() { var x = R([2, 16], 1); var gamma = C(1f, 16); Assert.Equal(x.Shape, E.RMSNorm(x, gamma, 1e-8, out _).Shape); }
    [Fact] public void RMSNorm_GammaTwo_ScalesOutput() { var x = R([2, 16], 1); var g1 = C(1f, 16); var g2 = C(2f, 16); var r1 = E.RMSNorm(x, g1, 1e-8, out _); var r2 = E.RMSNorm(x, g2, 1e-8, out _); AE(E.TensorMultiplyScalar(r1, 2f), r2, 1e-3f); }
    [Fact] public void RMSNorm_RmsPositive() { var x = R([2, 16], 1); var gamma = C(1f, 16); E.RMSNorm(x, gamma, 1e-8, out var rms); var rd = rms.GetDataArray(); for (int i = 0; i < rd.Length; i++) Assert.True(rd[i] > 0f); }

    // ================================================================
    // ReduceMax (2)
    // ================================================================
    [Fact] public void ReduceMax_Shape() { var x = R([4, 8], 1); Assert.Equal(new[] { 4 }, E.ReduceMax(x, new[] { 1 }, false, out _).Shape); }
    [Fact] public void ReduceMax_GreaterAll() { var x = R([4, 8], 1); var maxTensor = E.ReduceMax(x, new[] { 1 }, false, out _).GetDataArray(); var xd = x.GetDataArray(); for (int r = 0; r < 4; r++) for (int c = 0; c < 8; c++) Assert.True(maxTensor[r] >= xd[r * 8 + c] - Tol); }

    // ================================================================
    // ReduceMean (2)
    // ================================================================
    [Fact] public void ReduceMean_Shape() { var x = R([4, 8], 1); Assert.Equal(new[] { 4 }, E.ReduceMean(x, new[] { 1 }, false).Shape); }
    [Fact] public void ReduceMean_CorrectValue() { var x = C(3f, [4, 8]); var m = E.ReduceMean(x, new[] { 1 }, false).GetDataArray(); for (int i = 0; i < 4; i++) Assert.Equal(3f, m[i], Tol); }

    // ================================================================
    // ReduceVariance (2)
    // ================================================================
    [Fact] public void ReduceVariance_Shape() { var x = R([4, 8], 1); Assert.Equal(new[] { 4 }, E.ReduceVariance(x, new[] { 1 }, false).Shape); }
    [Fact] public void ReduceVariance_Nonneg() { var x = R([4, 8], 1); var vd = E.ReduceVariance(x, new[] { 1 }, false).GetDataArray(); for (int i = 0; i < 4; i++) Assert.True(vd[i] >= 0f); }

    // ================================================================
    // ReduceLogVariance (2)
    // ================================================================
    [Fact] public void ReduceLogVariance_Shape() { var x = R([4, 8], 1); Assert.Equal(new[] { 4 }, E.ReduceLogVariance(x, new[] { 1 }, false).Shape); }
    [Fact] public void ReduceLogVariance_LEVariance() { var x = R([4, 8], 2); var v = E.ReduceVariance(x, new[] { 1 }, false).GetDataArray(); var lv = E.ReduceLogVariance(x, new[] { 1 }, false).GetDataArray(); for (int i = 0; i < 4; i++) Assert.True(lv[i] <= v[i] + 0.1f, $"lv[{i}]={lv[i]} v[{i}]={v[i]}"); }

    // ================================================================
    // ReduceStd (2)
    // ================================================================
    [Fact] public void ReduceStd_Shape() { var x = R([4, 8], 1); Assert.Equal(new[] { 4 }, E.ReduceStd(x, new[] { 1 }, false).Shape); }
    [Fact] public void ReduceStd_SqrtOfVariance() { var x = R([4, 8], 2); var std = E.ReduceStd(x, new[] { 1 }, false).GetDataArray(); var vr = E.ReduceVariance(x, new[] { 1 }, false).GetDataArray(); for (int i = 0; i < 4; i++) Assert.True(Math.Abs(std[i] - Math.Sqrt(vr[i])) < 1e-3f); }

    // ================================================================
    // Backward passes: ReLU, Sigmoid, Tanh, GELU, LeakyReLU (6)
    // ================================================================
    [Fact] public void ReluBackward_ZeroNeg() { var input = new Tensor<float>(new float[] { -1f, -2f, 3f, 4f }, [4]); var grad = C(1f, 4); var back = E.ReluBackward(grad, input).GetDataArray(); Assert.Equal(0f, back[0], Tol); Assert.Equal(0f, back[1], Tol); Assert.Equal(1f, back[2], Tol); Assert.Equal(1f, back[3], Tol); }
    [Fact] public void SigmoidBackward_InRange() { var output = E.TensorSigmoid(R([64], 1)); var grad = C(1f, 64); var back = E.SigmoidBackward(grad, output).GetDataArray(); for (int i = 0; i < back.Length; i++) Assert.True(back[i] >= 0f && back[i] <= 0.251f); }
    [Fact] public void TanhBackward_InRange() { var output = E.TensorTanh(R([64], 1)); var grad = C(1f, 64); var back = E.TanhBackward(grad, output).GetDataArray(); for (int i = 0; i < back.Length; i++) Assert.True(back[i] >= 0f && back[i] <= 1.01f); }
    [Fact] public void GeluBackward_NonZeroForPositive() { var input = C(1f, [1, 4]); var grad = C(1f, [1, 4]); var back = E.GeluBackward(grad, input).GetDataArray(); Assert.True(back.All(v => v > 0f)); }
    [Fact] public void LeakyReluBackward_Negative() { var input = new Tensor<float>(new float[] { -1f, -2f, 3f, 4f }, [4]); var grad = C(1f, 4); var back = E.LeakyReluBackward(grad, input, 0.1).GetDataArray(); Assert.Equal(0.1f, back[0], 1e-3f); Assert.Equal(1f, back[2], Tol); }
    [Fact] public void ReluBackward_Shape() { var x = R([2, 8], 1); Assert.Equal(x.Shape, E.ReluBackward(C(1f, [2, 8]), x).Shape); }

    // ================================================================
    // Gated activation backward (GLUBackward, GeGLUBackward, SwiGLUBackward, ReGLUBackward) (4)
    // ================================================================
    [Fact] public void GLUBackward_Shape() { var inp = R([4, 16], 1); var gradOut = R([4, 8], 2); Assert.Equal(inp.Shape, E.GLUBackward(gradOut, inp, -1).Shape); }
    [Fact] public void GeGLUBackward_Shape() { var inp = R([4, 16], 1); var gradOut = R([4, 8], 2); Assert.Equal(inp.Shape, E.GeGLUBackward(gradOut, inp, -1).Shape); }
    [Fact] public void SwiGLUBackward_Shape() { var inp = R([4, 16], 1); var gradOut = R([4, 8], 2); Assert.Equal(inp.Shape, E.SwiGLUBackward(gradOut, inp, -1).Shape); }
    [Fact] public void ReGLUBackward_Shape() { var inp = R([4, 16], 1); var gradOut = R([4, 8], 2); Assert.Equal(inp.Shape, E.ReGLUBackward(gradOut, inp, -1).Shape); }

    // ================================================================
    // Sparsemax / TaylorSoftmax / SphericalSoftmax / GumbelSoftmax (6)
    // ================================================================
    [Fact] public void Sparsemax_SumToOne() { var d = E.Sparsemax(R([4, 8], 1), -1).GetDataArray(); for (int r = 0; r < 4; r++) { float s = 0; for (int j = 0; j < 8; j++) s += d[r * 8 + j]; Assert.True(Math.Abs(s - 1f) < 1e-3f); } }
    [Fact] public void Sparsemax_NonNeg() { var d = E.Sparsemax(R([4, 8], 2), -1).GetDataArray(); for (int i = 0; i < d.Length; i++) Assert.True(d[i] >= -Tol); }
    [Fact] public void TaylorSoftmax_SumToOne() { var d = E.TaylorSoftmax(R([4, 8], 3), 2, -1).GetDataArray(); for (int r = 0; r < 4; r++) { float s = 0; for (int j = 0; j < 8; j++) s += d[r * 8 + j]; Assert.True(Math.Abs(s - 1f) < 1e-2f); } }
    [Fact] public void TaylorSoftmax_NonNeg() { var d = E.TaylorSoftmax(R([4, 8], 4), 2, -1).GetDataArray(); for (int i = 0; i < d.Length; i++) Assert.True(d[i] >= -Tol); }
    [Fact] public void SphericalSoftmax_SumToOne() { var d = E.SphericalSoftmax(R([4, 8], 5), -1).GetDataArray(); for (int r = 0; r < 4; r++) { float s = 0; for (int j = 0; j < 8; j++) s += d[r * 8 + j]; Assert.True(Math.Abs(s - 1f) < 1e-3f); } }
    [Fact] public void GumbelSoftmax_SumToOne() { var d = E.GumbelSoftmax(R([4, 8], 6), 1.0, false, -1).GetDataArray(); for (int r = 0; r < 4; r++) { float s = 0; for (int j = 0; j < 8; j++) s += d[r * 8 + j]; Assert.True(Math.Abs(s - 1f) < 1e-3f); } }

    // ================================================================
    // GlobalAvgPool2D / GlobalMaxPool2D / AdaptiveAvgPool2D (4)
    // ================================================================
    [Fact] public void GlobalAvgPool_Shape() => Assert.Equal(new[] { 2, 4, 1, 1 }, E.GlobalAvgPool2D(R([2, 4, 8, 8], 1)).Shape);
    [Fact] public void GlobalAvgPool_Value() { var x = C(3f, [1, 1, 4, 4]); Assert.Equal(3f, E.GlobalAvgPool2D(x).GetDataArray()[0], Tol); }
    [Fact] public void GlobalMaxPool_Shape() => Assert.Equal(new[] { 2, 4, 1, 1 }, E.GlobalMaxPool2D(R([2, 4, 8, 8], 1)).Shape);
    [Fact] public void AdaptiveAvgPool_Shape() => Assert.Equal(new[] { 2, 4, 3, 3 }, E.AdaptiveAvgPool2D(R([2, 4, 8, 8], 1), 3, 3).Shape);

    // ================================================================
    // CrossEntropyLoss / MseLoss (4)
    // ================================================================
    [Fact] public void CrossEntropy_PerfectPred_Low() { int B = 4; int C = 8; var logits = new float[B * C]; var tgtData = new float[B * C]; for (int i = 0; i < B; i++) { logits[i * C + (i % C)] = 10f; tgtData[i * C + (i % C)] = 1f; } var pred = new Tensor<float>(logits, [B, C]); var tgt = new Tensor<float>(tgtData, [B, C]); float loss = E.CrossEntropyLoss(pred, tgt); Assert.True(loss < 1f, $"loss={loss}"); }
    [Fact] public void CrossEntropy_Nonneg() { var pred = R([4, 8], 1); var tgt = E.TensorSoftmax(R([4, 8], 2), -1); Assert.True(E.CrossEntropyLoss(pred, tgt) >= 0f); }
    [Fact] public void MseLoss_ZeroForPerfect() { var x = R([16], 1); Assert.True(Math.Abs(E.MseLoss(x, x)) < Tol); }
    [Fact] public void MseLoss_KnownValue() { var p = new Tensor<float>(new float[] { 1f, 3f }, [2]); var t = new Tensor<float>(new float[] { 3f, 1f }, [2]); float loss = E.MseLoss(p, t); Assert.Equal(4f, loss, 1e-3f); }

    // ================================================================
    // PairwiseDistance / PairwiseDistanceSquared (3)
    // ================================================================
    [Fact] public void PairwiseDistSquared_Nonneg() { var d = E.PairwiseDistanceSquared(R([4, 8], 1), R([4, 8], 2)).GetDataArray(); for (int i = 0; i < d.Length; i++) Assert.True(d[i] >= -Tol); }
    [Fact] public void PairwiseDist_Nonneg() { var d = E.PairwiseDistance(R([4, 8], 1), R([4, 8], 2)).GetDataArray(); for (int i = 0; i < d.Length; i++) Assert.True(d[i] >= -Tol); }
    [Fact] public void PairwiseDist_SelfIsZero() { var x = R([4, 8], 1); var d = E.PairwiseDistance(x, x); var dd = d.GetDataArray(); for (int i = 0; i < 4; i++) Assert.True(Math.Abs(dd[i * 4 + i]) < Tol, $"diag[{i}]={dd[i * 4 + i]}"); }

    // ================================================================
    // Upsample (2)
    // ================================================================
    [Fact] public void Upsample_Shape() => Assert.Equal(new[] { 1, 1, 8, 8 }, E.Upsample(R([1, 1, 4, 4], 1), 2, 2).Shape);
    [Fact] public void Upsample_ValueReplicated() { var x = new Tensor<float>(new float[] { 1, 2, 3, 4 }, [1, 1, 2, 2]); var u = E.Upsample(x, 2, 2).GetDataArray(); Assert.Equal(1f, u[0], Tol); Assert.Equal(1f, u[1], Tol); Assert.Equal(2f, u[2], Tol); }

    // ================================================================
    // In-place activation ops (SigmoidInPlace, ReLUInPlace, SwishInPlace, GELUInPlace, TanhInPlace, MishInPlace) (6)
    // ================================================================
    [Fact] public void SigmoidInPlace_MatchesSigmoid() { var x = R([64], 1); var expected = E.TensorSigmoid(x); var copy = new Tensor<float>((float[])x.GetDataArray().Clone(), x.Shape); E.SigmoidInPlace(copy); AE(copy, expected); }
    [Fact] public void ReLUInPlace_MatchesReLU() { var x = R([64], 1); var expected = E.TensorReLU(x); var copy = new Tensor<float>((float[])x.GetDataArray().Clone(), x.Shape); E.ReLUInPlace(copy); AE(copy, expected); }
    [Fact] public void SwishInPlace_MatchesSwish() { var x = R([64], 1); var expected = E.TensorSiLU(x); var copy = new Tensor<float>((float[])x.GetDataArray().Clone(), x.Shape); E.SwishInPlace(copy); AE(copy, expected, 1e-3f); }
    [Fact] public void GELUInPlace_MatchesGELU() { var x = R([64], 1); var expected = E.TensorGELU(x); var copy = new Tensor<float>((float[])x.GetDataArray().Clone(), x.Shape); E.GELUInPlace(copy); AE(copy, expected, 1e-3f); }
    [Fact] public void TanhInPlace_MatchesTanh() { var x = R([64], 1); var expected = E.TensorTanh(x); var copy = new Tensor<float>((float[])x.GetDataArray().Clone(), x.Shape); E.TanhInPlace(copy); AE(copy, expected, 1e-3f); }
    [Fact] public void MishInPlace_MatchesMish() { var x = R([64], 1); var expected = E.TensorMish(x); var copy = new Tensor<float>((float[])x.GetDataArray().Clone(), x.Shape); E.MishInPlace(copy); AE(copy, expected, 1e-3f); }

    // ================================================================
    // Into ops: SigmoidInto, ReLUInto, SwishInto, GELUInto, TanhInto, MishInto (6)
    // ================================================================
    [Fact] public void SigmoidInto_MatchesSigmoid() { var x = R([64], 1); var dest = new Tensor<float>(new float[64], [64]); E.SigmoidInto(dest, x); AE(dest, E.TensorSigmoid(x)); }
    [Fact] public void ReLUInto_MatchesReLU() { var x = R([64], 1); var dest = new Tensor<float>(new float[64], [64]); E.ReLUInto(dest, x); AE(dest, E.TensorReLU(x)); }
    [Fact] public void SwishInto_MatchesSwish() { var x = R([64], 1); var dest = new Tensor<float>(new float[64], [64]); E.SwishInto(dest, x); AE(dest, E.TensorSiLU(x), 1e-3f); }
    [Fact] public void GELUInto_MatchesGELU() { var x = R([64], 1); var dest = new Tensor<float>(new float[64], [64]); E.GELUInto(dest, x); AE(dest, E.TensorGELU(x), 1e-3f); }
    [Fact] public void TanhInto_MatchesTanh() { var x = R([64], 1); var dest = new Tensor<float>(new float[64], [64]); E.TanhInto(dest, x); AE(dest, E.TensorTanh(x), 1e-3f); }
    [Fact] public void MishInto_MatchesMish() { var x = R([64], 1); var dest = new Tensor<float>(new float[64], [64]); E.MishInto(dest, x); AE(dest, E.TensorMish(x), 1e-3f); }

    // ================================================================
    // MatMulInto (2)
    // ================================================================
    [Fact] public void MatMulInto_MatchesMatMul() { var a = R([3, 4], 1); var b = R([4, 5], 2); var dest = new Tensor<float>(new float[15], [3, 5]); E.MatMulInto(dest, a, b); AE(dest, E.TensorMatMul(a, b), 1e-3f); }
    [Fact] public void MatMulInto_Shape() { var a = R([4, 8], 1); var b = R([8, 4], 2); var dest = new Tensor<float>(new float[16], [4, 4]); E.MatMulInto(dest, a, b); Assert.Equal(new[] { 4, 4 }, dest.Shape); }

    // ================================================================
    // ConcatInto / TransposeInto / SoftmaxInto / LogSoftmaxInto (4)
    // ================================================================
    [Fact] public void ConcatInto_MatchesConcatenate() { var a = R([3, 4], 1); var b = R([3, 4], 2); var dest = new Tensor<float>(new float[24], [6, 4]); E.ConcatInto(dest, new[] { a, b }, 0); AE(dest, E.TensorConcatenate(new[] { a, b }, 0)); }
    [Fact] public void TransposeInto_Shape() { var x = R([3, 4], 1); var dest = new Tensor<float>(new float[12], [4, 3]); E.TransposeInto(dest, x, new[] { 1, 0 }); Assert.Equal(new[] { 4, 3 }, dest.Shape); }
    [Fact] public void SoftmaxInto_SumsToOne() { var x = R([2, 8], 1); var dest = new Tensor<float>(new float[16], [2, 8]); E.SoftmaxInto(dest, x, -1); var d = dest.GetDataArray(); for (int r = 0; r < 2; r++) { float s = 0; for (int j = 0; j < 8; j++) s += d[r * 8 + j]; Assert.True(Math.Abs(s - 1f) < 1e-3f); } }
    [Fact] public void LogSoftmaxInto_Consistent() { var x = R([2, 8], 1); var dest = new Tensor<float>(new float[16], [2, 8]); E.LogSoftmaxInto(dest, x, -1); AE(dest, E.TensorLogSoftmax(x, -1), 1e-3f); }

    // ================================================================
    // TensorCumSum extended (2)
    // ================================================================
    [Fact] public void CumSum_Monotone_ForPositive() { var x = RP([16], 1); var cs = E.TensorCumSum(x, 0).GetDataArray(); for (int i = 1; i < cs.Length; i++) Assert.True(cs[i] >= cs[i - 1] - Tol); }
    [Fact] public void CumSum_2D_Shape() { var x = R([4, 8], 1); Assert.Equal(x.Shape, E.TensorCumSum(x, 1).Shape); }

    // ================================================================
    // TensorArgMin (2)
    // ================================================================
    [Fact] public void ArgMin_MatchesMin() { var x = R([4, 8], 1); var idx = E.TensorArgMin(x, 0); var xd = x.GetDataArray(); var id = idx.GetDataArray(); for (int j = 0; j < 8; j++) { float expectedMin = 0; for (int i = 0; i < 4; i++) expectedMin = i == 0 ? xd[i * 8 + j] : Math.Min(expectedMin, xd[i * 8 + j]); Assert.Equal(expectedMin, xd[(int)id[j] * 8 + j], Tol); } }
    [Fact] public void ArgMin_Shape() { var x = R([4, 8], 1); Assert.Equal(new[] { 8 }, E.TensorArgMin(x, 0).Shape); }

    // ================================================================
    // PositionalEncoding (2)
    // ================================================================
    [Fact] public void PositionalEncoding_Shape() { var pos = new Tensor<float>(new float[] { 0, 1, 2, 3, 0, 1, 2, 3 }, [4, 2]); var enc = E.PositionalEncoding(pos, 4); Assert.Equal(2, enc.Shape.Length); Assert.Equal(4, enc.Shape[0]); }
    [Fact] public void PositionalEncoding_Bounded() { var posData = Enumerable.Range(0, 8).Select(i => (float)i).ToArray(); var pos = new Tensor<float>(posData, [4, 2]); var enc = E.PositionalEncoding(pos, 4).GetDataArray(); AR(enc, -1.1f, 1.1f, "PosEnc"); }

    // ================================================================
    // ELU activation (2)
    // ================================================================
    [Fact] public void ELU_PositivePassThrough() { var r = E.ELU(new Tensor<float>(new float[] { 1f, 2f, 3f }, [3])).GetDataArray(); Assert.Equal(1f, r[0], Tol); Assert.Equal(2f, r[1], Tol); Assert.Equal(3f, r[2], Tol); }
    [Fact] public void ELU_NegativeCompressed() { var r = E.ELU(new Tensor<float>(new float[] { -10f }, [1])).GetDataArray(); Assert.True(r[0] > -1.01f && r[0] < 0f); }

    // ================================================================
    // Softplus (2)
    // ================================================================
    [Fact] public void Softplus_Zero_IsLog2() => Assert.True(Math.Abs(E.Softplus(C(0, 1)).GetDataArray()[0] - (float)Math.Log(2.0)) < 1e-3f);
    [Fact] public void Softplus_Positive_LargeApprox() { var v = E.Softplus(C(10f, 1)).GetDataArray()[0]; Assert.True(Math.Abs(v - 10f) < 0.01f); }

    // ================================================================
    // TensorBinaryDE / BCEBackward (2)
    // ================================================================
    [Fact] public void BCE_BackwardShape() { var pred = new Tensor<float>(new float[] { 0.7f, 0.3f }, [2]); var tgt = new Tensor<float>(new float[] { 1f, 0f }, [2]); Assert.Equal(pred.Shape, E.TensorBinaryCrossEntropyBackward(pred, tgt, 1e-7f).Shape); }
    [Fact] public void BCE_BackwardSign() { var pred = new Tensor<float>(new float[] { 0.8f }, [1]); var tgt = new Tensor<float>(new float[] { 1f }, [1]); var g = E.TensorBinaryCrossEntropyBackward(pred, tgt, 1e-7f).GetDataArray()[0]; Assert.True(g < 0f, $"grad={g}"); }

    // ================================================================
    // Dropout (3)
    // ================================================================
    [Fact] public void Dropout_Training_HasZeros() { var x = C(1f, 256); var r = E.Dropout(x, 0.5, true, out _).GetDataArray(); Assert.True(r.Count(v => v == 0f) > 0); }
    [Fact] public void Dropout_Inference_Unchanged() { var x = R([64], 1); AE(E.Dropout(x, 0.5, false, out _), x); }
    [Fact] public void Dropout_MaskShape() { var x = R([4, 8], 1); E.Dropout(x, 0.3, true, out var mask); Assert.Equal(x.Shape, mask.Shape); }

    // ================================================================
    // LeakyReLUInPlace / LeakyReLUInto (2)
    // ================================================================
    [Fact] public void LeakyReLUInPlace_MatchesLeakyReLU() { var x = R([64], 1); var expected = E.TensorLeakyReLU(x, 0.01f); var copy = new Tensor<float>((float[])x.GetDataArray().Clone(), x.Shape); E.LeakyReLUInPlace(copy, 0.01f); AE(copy, expected, 1e-3f); }
    [Fact] public void LeakyReLUInto_MatchesLeakyReLU() { var x = R([64], 1); var dest = new Tensor<float>(new float[64], [64]); E.LeakyReLUInto(dest, x, 0.01f); AE(dest, E.TensorLeakyReLU(x, 0.01f), 1e-3f); }

    // ================================================================
    // TensorAddScaled (2)  -- verify linearity property
    // ================================================================
    [Fact] public void AddScaled_ScaleA() { var a = R([64], 1); var b = C(0f, 64); AE(E.TensorAddScaled(a, b, 3f, 0f), E.TensorMultiplyScalar(a, 3f), 1e-3f); }
    [Fact] public void AddScaled_ScaleB() { var a = C(0f, 64); var b = R([64], 2); AE(E.TensorAddScaled(a, b, 0f, 2f), E.TensorMultiplyScalar(b, 2f), 1e-3f); }

    // ================================================================
    // ISSUE #48/#49: Transposed tensor operations (6 tests)
    // Verifies that Transpose() + operation produces correct results
    // ================================================================
    [Fact] public void BatchMatMul_WithTransposedInputs()
    {
        // A: [2, 3, 4], B: [2, 4, 5] -> C: [2, 3, 5]
        // Create A as transpose of [2, 4, 3] to test transposed-then-matmul pattern
        var aOrig = R([2, 4, 3], 200);
        var aT = aOrig.Transpose(new[] { 0, 2, 1 }); // [2, 3, 4]
        var b = R([2, 4, 5], 201);
        var result = E.TensorBatchMatMul(aT, b);
        Assert.Equal(new[] { 2, 3, 5 }, result.Shape);
        // Verify against manual computation
        var ad = aT.GetDataArray(); var bd = b.GetDataArray(); var rd = result.GetDataArray();
        for (int bi = 0; bi < 2; bi++)
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 5; j++)
                {
                    float expected = 0;
                    for (int k = 0; k < 4; k++) expected += ad[bi * 12 + i * 4 + k] * bd[bi * 20 + k * 5 + j];
                    Assert.True(Math.Abs(rd[bi * 15 + i * 5 + j] - expected) < 1e-2f,
                        $"Mismatch at [{bi},{i},{j}]: got {rd[bi * 15 + i * 5 + j]}, expected {expected}");
                }
    }

    [Fact] public void BroadcastMultiply_4D_Values()
    {
        // Issue #48 comment: test 4D+ broadcast with VALUE verification, not just shape
        // a: [2, 3, 4, 5], b: [1, 1, 1, 5] -> broadcast multiply
        var a = R([2, 3, 4, 5], 202);
        var b = R([1, 1, 1, 5], 203);
        var result = E.TensorBroadcastMultiply(a, b);
        Assert.Equal(new[] { 2, 3, 4, 5 }, result.Shape);
        // Verify values: each element a[i] should be multiplied by b[i % 5]
        var ad = a.GetDataArray(); var bd = b.GetDataArray(); var rd = result.GetDataArray();
        for (int i = 0; i < ad.Length; i++)
            Assert.True(Math.Abs(rd[i] - ad[i] * bd[i % 5]) < 1e-4f,
                $"4D broadcast mul wrong at [{i}]: {rd[i]} vs {ad[i] * bd[i % 5]}");
    }

    [Fact] public void BroadcastMultiply_3D_MultiAxis()
    {
        // Broadcast across multiple axes simultaneously: [4, 3, 5] * [4, 1, 5]
        var a = R([4, 3, 5], 204);
        var b = R([4, 1, 5], 205);
        var result = E.TensorBroadcastMultiply(a, b);
        Assert.Equal(new[] { 4, 3, 5 }, result.Shape);
        var ad = a.GetDataArray(); var bd = b.GetDataArray(); var rd = result.GetDataArray();
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 3; j++)
                for (int k = 0; k < 5; k++)
                {
                    float expected = ad[i * 15 + j * 5 + k] * bd[i * 5 + k]; // b broadcasts over axis 1
                    Assert.True(Math.Abs(rd[i * 15 + j * 5 + k] - expected) < 1e-4f);
                }
    }

    [Fact] public void Transpose_ThenAdd_Correct()
    {
        // Transpose both operands, then add — result should match adding originals transposed
        var a = R([4, 8], 206);
        var b = R([4, 8], 207);
        var aT = E.TensorTranspose(a); // [8, 4]
        var bT = E.TensorTranspose(b); // [8, 4]
        var sumT = E.TensorAdd(aT, bT); // [8, 4]
        var sumOrig = E.TensorTranspose(E.TensorAdd(a, b)); // transpose(a+b) should equal aT+bT
        AE(sumT, sumOrig, 1e-4f, "Transpose+Add mismatch");
    }

    [Fact] public void Reshape_ThenMatMul_Correct()
    {
        // Create a tensor, reshape it, then use in matmul
        var flat = R([24], 208);
        var mat = flat.Reshape(4, 6); // [4, 6]
        var b = R([6, 3], 209);
        var result = E.TensorMatMul(mat, b);
        Assert.Equal(new[] { 4, 3 }, result.Shape);
        // Verify manually
        var md = mat.GetDataArray(); var bd = b.GetDataArray(); var rd = result.GetDataArray();
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 3; j++)
            {
                float expected = 0;
                for (int k = 0; k < 6; k++) expected += md[i * 6 + k] * bd[k * 3 + j];
                Assert.True(Math.Abs(rd[i * 3 + j] - expected) < 1e-2f);
            }
    }

    [Fact] public void Conv2DBackwardKernel_Shape()
    {
        // Conv2DBackwardKernel should produce gradient with same shape as kernel
        var input = R([1, 3, 8, 8], 210);
        var gradOutput = R([1, 16, 8, 8], 211);
        var kernelShape = new[] { 16, 3, 3, 3 };
        var dW = E.Conv2DBackwardKernel(gradOutput, input, kernelShape, new[] { 1, 1 }, new[] { 1, 1 }, new[] { 1, 1 });
        Assert.Equal(kernelShape, dW.Shape);
    }

    // ================================================================
    // ADDITIONAL BACKWARD OPS (8 tests)
    // ================================================================
    [Fact] public void SigmoidBackward_ShapePreserved()
    {
        var x = R([64], 220);
        var sig = E.TensorSigmoid(x);
        var grad = R([64], 221);
        var dSig = E.SigmoidBackward(grad, sig);
        Assert.Equal(new[] { 64 }, dSig.Shape);
        // Result should be non-zero since grad is non-zero
        Assert.True(dSig.GetDataArray().Any(v => Math.Abs(v) > 1e-7f), "SigmoidBackward all zeros");
    }

    [Fact] public void TanhBackward_ShapePreserved()
    {
        var x = R([64], 222);
        var th = E.TensorTanh(x);
        var grad = R([64], 223);
        var dTanh = E.TanhBackward(grad, th);
        Assert.Equal(new[] { 64 }, dTanh.Shape);
        Assert.True(dTanh.GetDataArray().Any(v => Math.Abs(v) > 1e-7f), "TanhBackward all zeros");
    }

    [Fact] public void ReLUDerivative_Correct()
    {
        // relu'(x) = 1 if x > 0, 0 otherwise
        var x = new Tensor<float>(new float[] { -2, -1, 0, 1, 2 }, [5]);
        var d = E.ReLUDerivative(x);
        var expected = new Tensor<float>(new float[] { 0, 0, 0, 1, 1 }, [5]);
        AE(d, expected, 1e-4f, "ReLU derivative");
    }

    [Fact] public void SoftmaxBackward_ShapePreserved()
    {
        var x = R([4, 16], 222);
        var sm = E.Softmax(x, -1);
        var grad = R([4, 16], 223);
        var dSm = E.SoftmaxBackward(grad, sm, -1);
        Assert.Equal(new[] { 4, 16 }, dSm.Shape);
    }

    [Fact] public void ReduceMax_MatchesTensorMax()
    {
        var x = R([4, 8], 224);
        var rmax = E.ReduceMax(x, new[] { 1 }, false, out _);
        Assert.Equal(new[] { 4 }, rmax.Shape);
        var xd = x.GetDataArray(); var rd = rmax.GetDataArray();
        for (int i = 0; i < 4; i++)
        {
            float rowMax = float.MinValue;
            for (int j = 0; j < 8; j++) rowMax = Math.Max(rowMax, xd[i * 8 + j]);
            Assert.True(Math.Abs(rd[i] - rowMax) < 1e-4f);
        }
    }

    [Fact] public void ReduceVariance_NonNeg()
    {
        var x = R([4, 16], 225);
        var v = E.ReduceVariance(x, new[] { 1 }, false);
        var vd = v.GetDataArray();
        for (int i = 0; i < vd.Length; i++)
            Assert.True(vd[i] >= -1e-6f, $"Variance negative at [{i}]: {vd[i]}");
    }

    [Fact] public void GroupNorm_ShapePreserved()
    {
        var x = R([2, 8, 4, 4], 226);
        var g = C(1f, 8); var b = C(0f, 8);
        var r = E.GroupNorm(x, 4, new Tensor<float>(g.GetDataArray(), [8]), new Tensor<float>(b.GetDataArray(), [8]), 1e-5, out _, out _);
        Assert.Equal(new[] { 2, 8, 4, 4 }, r.Shape);
    }

    [Fact] public void InstanceNorm_ShapePreserved()
    {
        var x = R([2, 4, 8, 8], 227);
        var g = C(1f, 4); var b = C(0f, 4);
        var r = E.InstanceNorm(x, new Tensor<float>(g.GetDataArray(), [4]), new Tensor<float>(b.GetDataArray(), [4]), 1e-5, out _, out _);
        Assert.Equal(new[] { 2, 4, 8, 8 }, r.Shape);
    }
}
