using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// Integration tests for BatchMatMul and ScaledDotProductAttentionBackward
/// verifying mathematical correctness. Tests issues #48 and #49.
/// </summary>
public class BatchMatMulAndAttentionTests
{
    private readonly CpuEngine _engine = new();

    #region Issue #48: BatchMatMul correctness

    [Fact]
    public void BatchMatMul_3Dx3D_Identity_ReturnsOriginal()
    {
        var a = new Tensor<float>(new float[] { 1, 2, 3, 4, 5, 6, 7, 8 }, new[] { 2, 2, 2 });
        var identity = new Tensor<float>(new float[] { 1, 0, 0, 1, 1, 0, 0, 1 }, new[] { 2, 2, 2 });

        var result = _engine.TensorBatchMatMul(a, identity);
        var data = result.GetDataArray();

        Assert.Equal(new[] { 2, 2, 2 }, result.Shape.ToArray());
        Assert.Equal(1f, data[0], 1e-5f);
        Assert.Equal(2f, data[1], 1e-5f);
        Assert.Equal(3f, data[2], 1e-5f);
        Assert.Equal(4f, data[3], 1e-5f);
        Assert.Equal(5f, data[4], 1e-5f);
        Assert.Equal(6f, data[5], 1e-5f);
        Assert.Equal(7f, data[6], 1e-5f);
        Assert.Equal(8f, data[7], 1e-5f);
    }

    [Fact]
    public void BatchMatMul_3Dx3D_ScaleMatrix_CorrectPerBatch()
    {
        // batch 0: [[1,2],[3,4]] @ [[2,0],[0,2]] = [[2,4],[6,8]]
        // batch 1: [[5,6],[7,8]] @ [[3,0],[0,3]] = [[15,18],[21,24]]
        var a = new Tensor<float>(new float[] { 1, 2, 3, 4, 5, 6, 7, 8 }, new[] { 2, 2, 2 });
        var b = new Tensor<float>(new float[] { 2, 0, 0, 2, 3, 0, 0, 3 }, new[] { 2, 2, 2 });

        var result = _engine.TensorBatchMatMul(a, b);
        var data = result.GetDataArray();

        Assert.Equal(2f, data[0], 1e-5f);
        Assert.Equal(4f, data[1], 1e-5f);
        Assert.Equal(6f, data[2], 1e-5f);
        Assert.Equal(8f, data[3], 1e-5f);
        Assert.Equal(15f, data[4], 1e-5f);
        Assert.Equal(18f, data[5], 1e-5f);
        Assert.Equal(21f, data[6], 1e-5f);
        Assert.Equal(24f, data[7], 1e-5f);
    }

    [Fact]
    public void BatchMatMul_3Dx3D_ResultsNotUniform()
    {
        // Issue #48: BatchMatMul produces uniform matrix (all elements same value)
        var rng = new Random(42);
        int batch = 4, m = 8, k = 6, n = 10;
        var aData = new float[batch * m * k];
        var bData = new float[batch * k * n];
        for (int i = 0; i < aData.Length; i++) aData[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < bData.Length; i++) bData[i] = (float)(rng.NextDouble() * 2 - 1);

        var a = new Tensor<float>(aData, new[] { batch, m, k });
        var b = new Tensor<float>(bData, new[] { batch, k, n });
        var result = _engine.TensorBatchMatMul(a, b);
        var data = result.GetDataArray();

        float first = data[0];
        bool allSame = true;
        for (int i = 1; i < data.Length; i++)
        {
            if (Math.Abs(data[i] - first) > 1e-6f) { allSame = false; break; }
        }
        Assert.False(allSame, "BatchMatMul produced uniform output — all elements have the same value");
    }

    [Fact]
    public void BatchMatMul_3Dx3D_MatchesManualPerBatchLoop()
    {
        // Issue #48: Manual per-batch TensorMatMul loop produces correct results
        var rng = new Random(123);
        int batch = 3, m = 4, k = 5, n = 6;
        var aData = new float[batch * m * k];
        var bData = new float[batch * k * n];
        for (int i = 0; i < aData.Length; i++) aData[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < bData.Length; i++) bData[i] = (float)(rng.NextDouble() * 2 - 1);

        var a = new Tensor<float>(aData, new[] { batch, m, k });
        var b = new Tensor<float>(bData, new[] { batch, k, n });

        var batchResult = _engine.TensorBatchMatMul(a, b);

        // Manual per-batch reference
        var manualData = new float[batch * m * n];
        for (int bIdx = 0; bIdx < batch; bIdx++)
        {
            var aSlice = new float[m * k];
            var bSlice = new float[k * n];
            Array.Copy(aData, bIdx * m * k, aSlice, 0, m * k);
            Array.Copy(bData, bIdx * k * n, bSlice, 0, k * n);

            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                {
                    float sum = 0;
                    for (int p = 0; p < k; p++)
                        sum += aSlice[i * k + p] * bSlice[p * n + j];
                    manualData[bIdx * m * n + i * n + j] = sum;
                }
        }

        var batchData = batchResult.GetDataArray();
        for (int i = 0; i < batch * m * n; i++)
        {
            Assert.True(Math.Abs(batchData[i] - manualData[i]) < 1e-3f,
                $"Mismatch at index {i}: BatchMatMul={batchData[i]}, Manual={manualData[i]}");
        }
    }

    [Fact]
    public void BatchMatMul_3Dx2D_BroadcastCorrect()
    {
        var rng = new Random(77);
        var aData = new float[2 * 3 * 4];
        var bData = new float[4 * 2];
        for (int i = 0; i < aData.Length; i++) aData[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < bData.Length; i++) bData[i] = (float)(rng.NextDouble() * 2 - 1);

        var a = new Tensor<float>(aData, new[] { 2, 3, 4 });
        var b = new Tensor<float>(bData, new[] { 4, 2 });
        var result = _engine.TensorBatchMatMul(a, b);

        Assert.Equal(new[] { 2, 3, 2 }, result.Shape.ToArray());

        var resultData = result.GetDataArray();
        for (int batch = 0; batch < 2; batch++)
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 2; j++)
                {
                    float expected = 0;
                    for (int p = 0; p < 4; p++)
                        expected += aData[batch * 12 + i * 4 + p] * bData[p * 2 + j];
                    Assert.Equal(expected, resultData[batch * 6 + i * 2 + j], 1e-4f);
                }
    }

    #endregion

    #region Issue #49: ScaledDotProductAttentionBackward

    [Fact]
    public void ScaledDotProductAttentionBackward_GradientsAreNonUniform()
    {
        // API requires 4D: [batch, heads, seq, d_k]
        var rng = new Random(42);
        int batch = 2, heads = 2, seq = 4, dk = 8;
        double scale = 1.0 / Math.Sqrt(dk);
        int total = batch * heads * seq * dk;

        var qData = new float[total];
        var kData = new float[total];
        var vData = new float[total];
        var gradData = new float[total];
        for (int i = 0; i < total; i++)
        {
            qData[i] = (float)(rng.NextDouble() * 0.5);
            kData[i] = (float)(rng.NextDouble() * 0.5);
            vData[i] = (float)(rng.NextDouble() * 0.5);
            gradData[i] = (float)(rng.NextDouble() * 0.5);
        }

        var shape = new[] { batch, heads, seq, dk };
        var Q = new Tensor<float>(qData, shape);
        var K = new Tensor<float>(kData, shape);
        var V = new Tensor<float>(vData, shape);
        var gradOutput = new Tensor<float>(gradData, shape);

        _engine.ScaledDotProductAttention(Q, K, V, null, scale, out var attnWeights);

        _engine.ScaledDotProductAttentionBackward(
            gradOutput, Q, K, V, attnWeights,
            scale,
            out var dQ, out var dK, out var dV);

        var dQData = dQ.GetDataArray();
        var dKData = dK.GetDataArray();
        var dVData = dV.GetDataArray();

        AssertNotUniform(dQData, "dQ");
        AssertNotUniform(dKData, "dK");
        AssertNotUniform(dVData, "dV");

        Assert.Equal(Q.Shape.ToArray(), dQ.Shape.ToArray());
        Assert.Equal(K.Shape.ToArray(), dK.Shape.ToArray());
        Assert.Equal(V.Shape.ToArray(), dV.Shape.ToArray());

        Assert.True(dQData.Any(x => Math.Abs(x) > 1e-7f), "dQ is all zeros");
        Assert.True(dKData.Any(x => Math.Abs(x) > 1e-7f), "dK is all zeros");
        Assert.True(dVData.Any(x => Math.Abs(x) > 1e-7f), "dV is all zeros");
    }

    [Fact]
    public void ScaledDotProductAttentionBackward_NumericalGradientCheck_dV()
    {
        // Numerical gradient check: dV_ij ~= (Loss(V+eps) - Loss(V-eps)) / (2*eps)
        // API requires 4D: [batch, heads, seq, d_k]
        var rng = new Random(99);
        int batch = 1, heads = 1, seq = 3, dk = 4;
        double scale = 1.0 / Math.Sqrt(dk);
        float eps = 1e-3f;
        int total = batch * heads * seq * dk;
        var shape = new[] { batch, heads, seq, dk };

        var qData = new float[total];
        var kData = new float[total];
        var vData = new float[total];
        for (int i = 0; i < total; i++)
        {
            qData[i] = (float)(rng.NextDouble() * 0.5);
            kData[i] = (float)(rng.NextDouble() * 0.5);
            vData[i] = (float)(rng.NextDouble() * 0.5);
        }

        var Q = new Tensor<float>(qData, shape);
        var K = new Tensor<float>(kData, shape);
        var V = new Tensor<float>(vData, shape);
        var gradOutput = new Tensor<float>(
            Enumerable.Repeat(1f, total).ToArray(), shape);

        _engine.ScaledDotProductAttention(Q, K, V, null, scale, out var attnW);
        _engine.ScaledDotProductAttentionBackward(
            gradOutput, Q, K, V, attnW, scale,
            out _, out _, out var dV);
        var dVData = dV.GetDataArray();

        int checkCount = 0, passCount = 0;
        for (int idx = 0; idx < Math.Min(12, vData.Length); idx++)
        {
            float origVal = vData[idx];

            vData[idx] = origVal + eps;
            var vPlus = new Tensor<float>((float[])vData.Clone(), shape);
            var outPlus = _engine.ScaledDotProductAttention(Q, K, vPlus, null, scale, out _);
            float lossPlus = 0;
            var outPlusData = outPlus.GetDataArray();
            for (int j = 0; j < outPlusData.Length; j++) lossPlus += outPlusData[j];

            vData[idx] = origVal - eps;
            var vMinus = new Tensor<float>((float[])vData.Clone(), shape);
            var outMinus = _engine.ScaledDotProductAttention(Q, K, vMinus, null, scale, out _);
            float lossMinus = 0;
            var outMinusData = outMinus.GetDataArray();
            for (int j = 0; j < outMinusData.Length; j++) lossMinus += outMinusData[j];

            vData[idx] = origVal;

            float numerical = (lossPlus - lossMinus) / (2 * eps);
            float analytical = dVData[idx];
            float relError = Math.Abs(numerical) > 1e-5f
                ? Math.Abs(numerical - analytical) / Math.Abs(numerical)
                : Math.Abs(numerical - analytical);

            checkCount++;
            if (relError < 0.05f) passCount++;
        }

        Assert.True(passCount >= checkCount * 0.8,
            $"Numerical gradient check for dV: {passCount}/{checkCount} passed (need 80%)");
    }

    #endregion

    private static void AssertNotUniform(float[] data, string name)
    {
        if (data.Length <= 1) return;
        float first = data[0];
        bool allSame = data.All(x => Math.Abs(x - first) < 1e-8f);
        Assert.False(allSame, $"{name} is uniform - all {data.Length} elements equal {first}");
    }
}
