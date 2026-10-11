using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// grouped_mm and scaled_mm against torch 2.12 on CPU (parity/torch_mm_ref.py), and the recipes torch only runs on
/// sm90+ GPUs (block scales, the 32x4x4 swizzle, scaled_grouped_mm) against an explicit scale-then-multiply.
/// </summary>
public class GroupedScaledMatMulTests
{
    private readonly IEngine _engine = new CpuEngine();

    private static Tensor<float> T(float[] v, int[] shape) => new Tensor<float>(v, shape);

    private static void AssertClose(Tensor<float> expected, Tensor<float> actual, double tolerance)
    {
        Assert.Equal(expected.Shape.ToArray(), actual.Shape.ToArray());
        var e = expected.ToArray(); var a = actual.ToArray();
        for (int i = 0; i < e.Length; i++)
            Assert.True(Math.Abs(e[i] - a[i]) <= tolerance * Math.Max(1, Math.Abs(e[i])), $"element {i}: expected {e[i]}, got {a[i]}");
    }

    private static Tensor<float> Random(int seed, params int[] shape)
    {
        var rng = new Random(seed);
        var t = new Tensor<float>(shape);
        var s = t.AsWritableSpan();
        for (int i = 0; i < s.Length; i++) s[i] = (float)(rng.NextDouble() * 2 - 1);
        return t;
    }

//@@TORCH_CASES@@

    [Fact]
    public void GroupedMatMul_ABiasOfTheWrongShape_IsRejectedAsTheBias()
    {
        // a [4, 3] jagged into 2 groups by rows; b [2, 3, 5]: the bias is [5] or [2, 5].
        var a = Random(11, 4, 3); var b = Random(12, 2, 3, 5);
        var offs = new Tensor<int>(new[] { 2, 4 }, new[] { 2 });
        Assert.Equal("bias", Assert.Throws<ArgumentException>(() => _engine.TensorGroupedMatMul(a, b, offs, Random(13, 4))).ParamName);
        Assert.Equal("bias", Assert.Throws<ArgumentException>(() => _engine.TensorGroupedMatMul(a, b, offs, Random(14, 3, 5))).ParamName);
        Assert.Equal(new[] { 4, 5 }, _engine.TensorGroupedMatMul(a, b, offs, Random(15, 5)).Shape.ToArray());
        Assert.Equal(new[] { 4, 5 }, _engine.TensorGroupedMatMul(a, b, offs, Random(16, 2, 5)).Shape.ToArray());
    }

    [Fact]
    public void ScaledGroupedMatMul_JaggedK_RejectsBlockAndSwizzledScalesOnEitherOperand()
    {
        // a [4, 32] x b [32, 3] with the shared dimension jagged at 16: per-group row-wise scales are [2, 4] and [2, 3].
        var a = Random(21, 4, 32); var b = Random(22, 32, 3);
        var offs = new Tensor<int>(new[] { 16, 32 }, new[] { 2 });
        var sa = Random(23, 2, 4); var sb = Random(24, 2, 3);
        // A 1x16 block scale for b has ceil(32 / 16) * 3 = 6 values, the same count as the row-wise [2, 3]: it used to be
        // decoded as row-wise.
        Assert.Throws<NotSupportedException>(() => _engine.TensorScaledGroupedMatMul(a, b, sa, ScalingRecipe.RowWise, Random(25, 3, 2), ScalingRecipe.BlockWise1x16, offsets: offs));
        Assert.Throws<NotSupportedException>(() => _engine.TensorScaledGroupedMatMul(a, b, sa, ScalingRecipe.RowWise, sb, ScalingRecipe.RowWise, ScaleSwizzle.Swizzle32x4x4, offsets: offs));
        Assert.Throws<NotSupportedException>(() => _engine.TensorScaledGroupedMatMul(a, b, sa, ScalingRecipe.RowWise, sb, ScalingRecipe.RowWise, swizzleB: ScaleSwizzle.Swizzle32x4x4, offsets: offs));
        Assert.Equal(new[] { 2, 4, 3 }, _engine.TensorScaledGroupedMatMul(a, b, sa, ScalingRecipe.RowWise, sb, ScalingRecipe.RowWise, offsets: offs).Shape.ToArray());
    }


    [Theory]
    [InlineData(ScalingRecipe.BlockWise1x16, 16)]
    [InlineData(ScalingRecipe.BlockWise1x32, 32)]
    [InlineData(ScalingRecipe.BlockWise1x128, 128)]
    public void ScaledMatMul_BlockWise1xN_IsTheProductOfTheScaledOperands(ScalingRecipe recipe, int block)
    {
        int m = 5, k = 2 * block + 7, n = 3;
        var a = Random(1, m, k); var b = Random(2, k, n);
        int kb = (k + block - 1) / block;
        var sa = Random(3, m, kb); var sb = Random(4, n, kb);   // b's scales over its transpose [N, K]
        var da = new Tensor<float>(new[] { m, k }); var db = new Tensor<float>(new[] { k, n });
        for (int i = 0; i < m; i++) for (int j = 0; j < k; j++) da[i, j] = a[i, j] * sa[i, j / block];
        for (int j = 0; j < k; j++) for (int c = 0; c < n; c++) db[j, c] = b[j, c] * sb[c, j / block];
        AssertClose(_engine.TensorMatMul(da, db), _engine.TensorScaledMatMul(a, b, sa, recipe, sb, recipe), 1e-5);
    }

    [Fact]
    public void ScaledMatMul_BlockWise128x128_IsTheProductOfTheScaledOperands()
    {
        int m = 130, k = 260, n = 129;
        var a = Random(5, m, k); var b = Random(6, k, n);
        var sa = Random(7, 2, 3); var sb = Random(8, 2, 3);
        var da = new Tensor<float>(new[] { m, k }); var db = new Tensor<float>(new[] { k, n });
        for (int i = 0; i < m; i++) for (int j = 0; j < k; j++) da[i, j] = a[i, j] * sa[i / 128, j / 128];
        for (int j = 0; j < k; j++) for (int c = 0; c < n; c++) db[j, c] = b[j, c] * sb[c / 128, j / 128];
        AssertClose(_engine.TensorMatMul(da, db), _engine.TensorScaledMatMul(a, b, sa, ScalingRecipe.BlockWise128x128, sb, ScalingRecipe.BlockWise128x128), 1e-4);
    }

    // torchao's to_blocked: pad to (128, 4) multiples, view (nR, 128, nC, 4) -> permute (0, 2, 1, 3) ->
    // reshape (-1, 4, 32, 4) -> transpose (1, 2) -> flatten. Written out independently of the library's decode loop.
    private static Tensor<float> ToBlocked(Tensor<float> scales)
    {
        int rows = scales.Shape[0], cols = scales.Shape[1];
        int nR = (rows + 127) / 128, nC = (cols + 3) / 4;
        var flat = new float[nR * nC * 512];
        int idx = 0;
        for (int br = 0; br < nR; br++)
            for (int bc = 0; bc < nC; bc++)
                for (int r32 = 0; r32 < 32; r32++)       // after transpose(1, 2): [32][4][4]
                    for (int slab = 0; slab < 4; slab++)
                        for (int c4 = 0; c4 < 4; c4++)
                        {
                            int r = br * 128 + slab * 32 + r32, c = bc * 4 + c4;
                            flat[idx++] = r < rows && c < cols ? scales[r, c] : 0f;
                        }
        return new Tensor<float>(flat, new[] { flat.Length });
    }

    [Fact]
    public void ScaledMatMul_Swizzle32x4x4_ReadsTheBlockedLayout()
    {
        int m = 150, k = 32 * 6, n = 7;
        var a = Random(9, m, k); var b = Random(10, k, n);
        var sa = Random(11, m, 6); var sb = Random(12, n, 6);
        var plain = _engine.TensorScaledMatMul(a, b, sa, ScalingRecipe.BlockWise1x32, sb, ScalingRecipe.BlockWise1x32);
        var swizzled = _engine.TensorScaledMatMul(a, b, ToBlocked(sa), ScalingRecipe.BlockWise1x32, ToBlocked(sb), ScalingRecipe.BlockWise1x32,
            ScaleSwizzle.Swizzle32x4x4, ScaleSwizzle.Swizzle32x4x4);
        AssertClose(plain, swizzled, 1e-6);
    }

    [Fact]
    public void GroupedMatMul_Bias_IsAddedToEachGroup()
    {
        var a = Random(13, 12, 8); var b = Random(14, 2, 8, 4); var bias = Random(15, 2, 4);
        var offs = new Tensor<int>(new[] { 4, 12 }, new[] { 2 });
        var plain = _engine.TensorGroupedMatMul(a, b, offs);
        var withBias = _engine.TensorGroupedMatMul(a, b, offs, bias);
        for (int r = 0; r < 12; r++)
            for (int c = 0; c < 4; c++)
                Assert.Equal(plain[r, c] + bias[r < 4 ? 0 : 1, c], withBias[r, c], 5);
    }

    [Fact]
    public void ScaledGroupedMatMul_RowWise_IsGroupedMatMulOfTheScaledOperands()
    {
        var a = Random(16, 12, 8); var b = Random(17, 2, 8, 4);
        var sa = Random(18, 12); var sb = Random(19, 2, 4);     // rows of a; [G, N] for b
        var offs = new Tensor<int>(new[] { 4, 12 }, new[] { 2 });
        var da = new Tensor<float>(new[] { 12, 8 }); var db = new Tensor<float>(new[] { 2, 8, 4 });
        for (int r = 0; r < 12; r++) for (int c = 0; c < 8; c++) da[r, c] = a[r, c] * sa[r];
        for (int g = 0; g < 2; g++) for (int r = 0; r < 8; r++) for (int c = 0; c < 4; c++) db[g, r, c] = b[g, r, c] * sb[g, c];
        AssertClose(_engine.TensorGroupedMatMul(da, db, offs),
            _engine.TensorScaledGroupedMatMul(a, b, sa, ScalingRecipe.RowWise, sb, ScalingRecipe.RowWise, offsets: offs), 1e-5);
    }

    [Fact]
    public void GroupedMatMul_IsDifferentiable()
    {
        var a = Random(20, 6, 4); var b = Random(21, 2, 4, 3);
        var offs = new Tensor<int>(new[] { 2, 6 }, new[] { 2 });
        Tensor<float> ga, gb;
        using (var tape = new GradientTape<float>())
        {
            var loss = _engine.ReduceSum(_engine.TensorGroupedMatMul(a, b, offs), null);
            var grads = tape.ComputeGradients(loss, new[] { a, b });
            ga = grads[a]; gb = grads[b];
        }
        // d sum(out)/d a[r, k] = sum_n b[g(r), k, n]; d/d b[g, k, n] = sum over the group's rows of a[r, k]
        for (int r = 0; r < 6; r++)
            for (int kk = 0; kk < 4; kk++)
            {
                int g = r < 2 ? 0 : 1;
                float expected = 0; for (int n = 0; n < 3; n++) expected += b[g, kk, n];
                Assert.Equal(expected, ga[r, kk], 4);
            }
        for (int g = 0; g < 2; g++)
            for (int kk = 0; kk < 4; kk++)
            {
                float expected = 0; for (int r = g == 0 ? 0 : 2; r < (g == 0 ? 2 : 6); r++) expected += a[r, kk];
                for (int n = 0; n < 3; n++) Assert.Equal(expected, gb[g, kk, n], 4);
            }
    }
}
