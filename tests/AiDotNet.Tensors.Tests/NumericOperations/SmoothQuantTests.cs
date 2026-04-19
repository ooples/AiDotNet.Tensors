using AiDotNet.Tensors.NumericOperations;
using Xunit;

namespace AiDotNet.Tensors.Tests.NumericOperations;

/// <summary>
/// Tests for issue #207 B1 — <see cref="SmoothQuant"/> activation-to-
/// weight scale migration. Locks in the algebraic identity and the
/// magnitude-balancing property.
/// </summary>
public class SmoothQuantTests
{
    [Fact]
    public void ComputeSmoothingFactor_BalancedInput_ReturnsOnes()
    {
        // When activation and weight have identical per-column absmax,
        // the factor should be 1 (identity transform).
        var aMax = new float[] { 1f, 2f, 3f };
        var wMax = new float[] { 1f, 2f, 3f };
        var s = SmoothQuant.ComputeSmoothingFactor(aMax, wMax, alpha: 0.5f);
        foreach (var v in s) Assert.Equal(1f, v, 3);
    }

    [Fact]
    public void ComputeSmoothingFactor_HotActivation_MigratesRangeToWeight()
    {
        // Activation has outlier in col 0 (100×), weight doesn't. Factor
        // should be > 1 there so the post-transform activation shrinks
        // and the post-transform weight grows.
        var aMax = new float[] { 100f, 1f };
        var wMax = new float[] { 1f, 1f };
        var s = SmoothQuant.ComputeSmoothingFactor(aMax, wMax, alpha: 0.5f);
        Assert.True(s[0] > 5f, $"Expected factor > 5 for outlier col, got {s[0]}");
        Assert.Equal(1f, s[1], 3);
    }

    [Fact]
    public void ApplyToWeights_MultipliesEachColumn()
    {
        var w = new float[] { 1f, 1f, 1f, 1f, 1f, 1f }; // 2 × 3 row-major
        var s = new float[] { 2f, 4f, 0.5f };
        SmoothQuant.ApplyToWeights(w, outC: 2, inC: 3, s);
        Assert.Equal(2f, w[0]);
        Assert.Equal(4f, w[1]);
        Assert.Equal(0.5f, w[2]);
        Assert.Equal(2f, w[3]);
        Assert.Equal(4f, w[4]);
        Assert.Equal(0.5f, w[5]);
    }

    [Fact]
    public void ApplyToActivations_DividesEachColumn()
    {
        var a = new float[] { 10f, 10f, 10f, 10f, 10f, 10f }; // 2 × 3
        var s = new float[] { 2f, 5f, 10f };
        SmoothQuant.ApplyToActivations(a, batch: 2, inC: 3, s);
        Assert.Equal(5f, a[0]);
        Assert.Equal(2f, a[1]);
        Assert.Equal(1f, a[2]);
    }

    [Fact]
    public void AlgebraicIdentity_YEqualsXW()
    {
        // Core SmoothQuant claim: Y = X · W = X' · W' where X' = X/s, W' = s·W.
        // Verify by computing both sides and comparing.
        const int B = 2, inC = 3, outC = 2;
        var rng = new Random(7);
        var x = new float[B * inC];
        var w = new float[inC * outC]; // row-major [inC, outC]
        for (int i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < w.Length; i++) w[i] = (float)(rng.NextDouble() * 2 - 1);

        var yOriginal = MatMul(x, w, B, inC, outC);

        // Compute factor from absmax stats, then apply.
        var aMax = new float[inC];
        for (int i = 0; i < B; i++)
            for (int j = 0; j < inC; j++)
                aMax[j] = Math.Max(aMax[j], Math.Abs(x[i * inC + j]));
        var wMax = new float[inC];
        for (int j = 0; j < inC; j++)
            for (int k = 0; k < outC; k++)
                wMax[j] = Math.Max(wMax[j], Math.Abs(w[j * outC + k]));
        var s = SmoothQuant.ComputeSmoothingFactor(aMax, wMax, alpha: 0.5f);

        // Note: ApplyToWeights takes [outC, inC] row-major, but our W
        // is [inC, outC]. Use a manual transform that matches the
        // issue's semantics: W'[j, k] = s[j] × W[j, k].
        var wPrime = (float[])w.Clone();
        for (int j = 0; j < inC; j++)
            for (int k = 0; k < outC; k++)
                wPrime[j * outC + k] *= s[j];

        var xPrime = (float[])x.Clone();
        SmoothQuant.ApplyToActivations(xPrime, B, inC, s);

        var ySmoothed = MatMul(xPrime, wPrime, B, inC, outC);

        for (int i = 0; i < yOriginal.Length; i++)
            Assert.Equal(yOriginal[i], ySmoothed[i], 3);
    }

    [Fact]
    public void PerColumnAbsMax_ComputesMaxPerCol()
    {
        var data = new float[]
        {
             1f, -2f,  3f,
            -4f,  5f, -6f,
             0.1f, 0.2f, 0.3f,
        };
        var result = SmoothQuant.PerColumnAbsMax(data, rows: 3, cols: 3);
        Assert.Equal(4f, result[0]);
        Assert.Equal(5f, result[1]);
        Assert.Equal(6f, result[2]);
    }

    [Fact]
    public void ComputeSmoothingFactor_LengthMismatch_Throws()
    {
        Assert.Throws<ArgumentException>(() =>
            SmoothQuant.ComputeSmoothingFactor(new float[2], new float[3]));
    }

    [Fact]
    public void ComputeSmoothingFactor_AlphaOutOfRange_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            SmoothQuant.ComputeSmoothingFactor(new float[1], new float[1], alpha: -0.1f));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            SmoothQuant.ComputeSmoothingFactor(new float[1], new float[1], alpha: 1.1f));
    }

    private static float[] MatMul(float[] a, float[] b, int m, int k, int n)
    {
        var c = new float[m * n];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
            {
                float acc = 0f;
                for (int p = 0; p < k; p++) acc += a[i * k + p] * b[p * n + j];
                c[i * n + j] = acc;
            }
        return c;
    }
}
