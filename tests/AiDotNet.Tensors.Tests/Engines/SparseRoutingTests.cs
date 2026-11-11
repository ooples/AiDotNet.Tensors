using System;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// Unit and autodiff parity tests for <see cref="CpuEngine.TensorGatherRowValues{T}"/>,
/// <see cref="CpuEngine.TopKWeightedRowMix{T}"/>, <see cref="CpuEngine.TensorCountIndices{T}"/>, and
/// readout-detached <see cref="CpuEngine.FusedLinearCrossEntropyWithLogits{T}(Tensor{T}, Tensor{T}, Tensor{T}, Tensor{int}, bool)"/>.
/// </summary>
public sealed class SparseRoutingTests
{
    private static Tensor<float> RandFloat(int[] shape, int seed)
    {
        var rng = new Random(seed);
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = (float)(rng.NextDouble() - 0.5);
        return t;
    }

    private static Tensor<double> RandDouble(int[] shape, int seed)
    {
        var rng = new Random(seed);
        var t = new Tensor<double>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = rng.NextDouble() - 0.5;
        return t;
    }

    private static Tensor<int> RandIndices(int[] shape, int numBins, int seed, bool includeOutOfBounds = false)
    {
        var rng = new Random(seed);
        var t = new Tensor<int>(shape);
        for (int i = 0; i < t.Length; i++)
        {
            if (includeOutOfBounds && i % 11 == 0)
                t[i] = (i % 2 == 0) ? -1 : numBins + 3;
            else
                t[i] = rng.Next(0, numBins);
        }
        return t;
    }

    private static void AssertClose(float[] expected, float[] actual, string name, double rel = 1e-5)
    {
        Assert.Equal(expected.Length, actual.Length);
        double scale = Math.Max(expected.Select(v => (double)Math.Abs(v)).DefaultIfEmpty(0).Max(), 1e-6);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= rel * scale,
                $"{name}[{i}]: expected {expected[i]}, got {actual[i]} (scale {scale})");
    }

    private static void AssertCloseDouble(double[] expected, double[] actual, string name, double tol = 1e-11)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= tol,
                $"{name}[{i}]: expected {expected[i]}, got {actual[i]}");
    }

    // Decomposed reference for TensorGatherRowValues: ReduceSum(OneHot(indices, G) * Unsqueeze(values, -2), -1).
    private static Tensor<float> DecomposedGatherRowValues(CpuEngine e, Tensor<float> values, Tensor<int> indices)
    {
        int g = values.Shape[^1], k = indices.Shape[^1];
        int rows = indices.Length / k;
        var oneHot = e.Reshape(e.TensorOneHot<float>(indices, g), [rows, k, g]);
        var val3 = e.Reshape(values, [rows, 1, g]);
        var gathered2d = e.ReduceSum(e.TensorBroadcastMultiply(oneHot, val3), [2], keepDims: false);
        return e.Reshape(gathered2d, indices.Shape.ToArray());
    }

    // Decomposed reference for TopKWeightedRowMix: (ReduceSum(OneHot(indices, G) * Unsqueeze(weights, -1), -2)) @ table.
    private static Tensor<float> DecomposedTopKMix(CpuEngine e, Tensor<float> table, Tensor<int> indices, Tensor<float> weights)
    {
        int g = table.Shape[0], c = table.Shape[1], k = indices.Shape[^1];
        int rows = indices.Length / k;
        var oneHot = e.Reshape(e.TensorOneHot<float>(indices, g), [rows, k, g]);
        var w3 = e.Reshape(weights, [rows, k, 1]);
        var gamma = e.ReduceSum(e.TensorBroadcastMultiply(oneHot, w3), [1], keepDims: false); // [rows, G]
        var mixed2d = e.TensorMatMul(gamma, table); // [rows, C]
        var outShape = indices.Shape.ToArray();
        outShape[^1] = c;
        return e.Reshape(mixed2d, outShape);
    }

    public static TheoryData<int[], int, int, int> RoutingShapes => new()
    {
        { new[] { 5 }, 16, 4, 8 },             // rank-1 single token
        { new[] { 7, 16 }, 16, 4, 37 },        // rank-2, scalar SIMD tail on channels
        { new[] { 2, 5, 32 }, 32, 6, 300 },    // rank-3, channels > SparseRoutingChannelBlock (256)
        { new[] { 4, 8 }, 8, 1, 13 },          // top-1 routing
    };

    [Theory]
    [MemberData(nameof(RoutingShapes))]
    public void GatherRowValues_TapeGradients_MatchOneHotComposition(int[] valShape, int g, int k, int unusedC)
    {
        _ = unusedC;
        var e = new CpuEngine();
        valShape = valShape.ToArray();
        valShape[^1] = g;
        var idxShape = valShape.ToArray();
        idxShape[^1] = k;

        var values = RandFloat(valShape, 10);
        var indices = RandIndices(idxShape, g, 11, includeOutOfBounds: true);
        var upstream = RandFloat(idxShape, 12);
        var axes = Enumerable.Range(0, idxShape.Length).ToArray();

        (float[] Y, float[] Dv) Run(bool fused)
        {
            using var tape = new GradientTape<float>();
            var y = fused ? e.TensorGatherRowValues(values, indices) : DecomposedGatherRowValues(e, values, indices);
            var loss = e.ReduceSum(e.TensorMultiply(y, upstream), axes, keepDims: false);
            var grads = tape.ComputeGradients(loss, [values]);
            return (y.ToArray(), grads[values].ToArray());
        }

        var refRun = Run(fused: false);
        var fusedRun = Run(fused: true);
        AssertClose(refRun.Y, fusedRun.Y, "gather_y");
        AssertClose(refRun.Dv, fusedRun.Dv, "gather_dv");
    }

    [Theory]
    [MemberData(nameof(RoutingShapes))]
    public void TopKWeightedRowMix_TapeGradients_MatchOneHotMatMulComposition(int[] valShape, int g, int k, int c)
    {
        var e = new CpuEngine();
        var idxShape = valShape.ToArray();
        idxShape[^1] = k;
        var outShape = valShape.ToArray();
        outShape[^1] = c;

        var table = RandFloat([g, c], 20);
        var indices = RandIndices(idxShape, g, 21, includeOutOfBounds: true);
        var weights = RandFloat(idxShape, 22);
        var upstream = RandFloat(outShape, 23);
        var axes = Enumerable.Range(0, outShape.Length).ToArray();

        (float[] Y, float[] Dt, float[] Dw) Run(bool fused)
        {
            using var tape = new GradientTape<float>();
            var y = fused ? e.TopKWeightedRowMix(table, indices, weights) : DecomposedTopKMix(e, table, indices, weights);
            var loss = e.ReduceSum(e.TensorMultiply(y, upstream), axes, keepDims: false);
            var grads = tape.ComputeGradients(loss, [table, weights]);
            return (y.ToArray(), grads[table].ToArray(), grads[weights].ToArray());
        }

        var refRun = Run(fused: false);
        var fusedRun = Run(fused: true);
        AssertClose(refRun.Y, fusedRun.Y, "mix_y");
        AssertClose(refRun.Dt, fusedRun.Dt, "mix_dtable");
        AssertClose(refRun.Dw, fusedRun.Dw, "mix_dweights");
    }

    [Fact]
    public void DoublePath_MatchesExactDefinitions()
    {
        var e = new CpuEngine();
        int rows = 6, g = 11, k = 3, c = 29;
        var values = RandDouble([rows, g], 31);
        var indices = RandIndices([rows, k], g, 32);
        var gGather = RandDouble([rows, k], 33);
        var table = RandDouble([g, c], 34);
        var weights = RandDouble([rows, k], 35);
        var gMix = RandDouble([rows, c], 36);

        var yGather = e.TensorGatherRowValues(values, indices).ToArray();
        var dvGather = e.TensorGatherRowValuesBackward(gGather, indices, g).ToArray();
        var yMix = e.TopKWeightedRowMix(table, indices, weights).ToArray();
        var dwMix = e.TopKWeightedRowMixBackwardWeights(gMix, table, indices).ToArray();
        var dtMix = e.TopKWeightedRowMixBackwardTable(gMix, indices, weights, g).ToArray();

        var vArr = values.ToArray();
        var iArr = indices.ToArray();
        var ggArr = gGather.ToArray();
        var tArr = table.ToArray();
        var wArr = weights.ToArray();
        var gmArr = gMix.ToArray();

        var expectedYGather = new double[rows * k];
        var expectedDv = new double[rows * g];
        var expectedYMix = new double[rows * c];
        var expectedDw = new double[rows * k];
        var expectedDt = new double[g * c];

        for (int r = 0; r < rows; r++)
        {
            for (int j = 0; j < k; j++)
            {
                int col = iArr[r * k + j];
                expectedYGather[r * k + j] = vArr[r * g + col];
                expectedDv[r * g + col] += ggArr[r * k + j];
                double dot = 0.0;
                for (int ch = 0; ch < c; ch++)
                {
                    double wProd = wArr[r * k + j] * tArr[col * c + ch];
                    expectedYMix[r * c + ch] += wProd;
                    dot += gmArr[r * c + ch] * tArr[col * c + ch];
                    expectedDt[col * c + ch] += wArr[r * k + j] * gmArr[r * c + ch];
                }
                expectedDw[r * k + j] = dot;
            }
        }

        AssertCloseDouble(expectedYGather, yGather, "double_gather_y");
        AssertCloseDouble(expectedDv, dvGather, "double_gather_dv");
        AssertCloseDouble(expectedYMix, yMix, "double_mix_y");
        AssertCloseDouble(expectedDw, dwMix, "double_mix_dw");
        AssertCloseDouble(expectedDt, dtMix, "double_mix_dt");
    }

    [Fact]
    public void MutatingIndicesAfterForward_DoesNotCorruptBackward()
    {
        var e = new CpuEngine();
        var table = RandFloat([8, 12], 40);
        var indices = RandIndices([5, 3], 8, 41);
        var weights = RandFloat([5, 3], 42);

        float[] expectedDt, expectedDw;
        using (var tape1 = new GradientTape<float>())
        {
            var y1 = e.TopKWeightedRowMix(table, indices, weights);
            var loss1 = e.ReduceSum(y1, [0, 1], keepDims: false);
            var g1 = tape1.ComputeGradients(loss1, [table, weights]);
            expectedDt = g1[table].ToArray();
            expectedDw = g1[weights].ToArray();
        }

        using var tape2 = new GradientTape<float>();
        var y2 = e.TopKWeightedRowMix(table, indices, weights);
        for (int i = 0; i < indices.Length; i++) indices[i] = 0; // mutate caller tensor before backward
        var loss2 = e.ReduceSum(y2, [0, 1], keepDims: false);
        var g2 = tape2.ComputeGradients(loss2, [table, weights]);

        AssertClose(expectedDt, g2[table].ToArray(), "snapshot_dt");
        AssertClose(expectedDw, g2[weights].ToArray(), "snapshot_dw");
    }

    [Fact]
    public void TensorCountIndices_MatchesOneHotCounts()
    {
        IEngine e = new CpuEngine();
        int numBins = 16;
        var idx = RandIndices([3, 7, 4], numBins, 50, includeOutOfBounds: true);
        var counts = e.TensorCountIndices<float>(idx, numBins).ToArray();
        var intCounts = e.TensorCountIndices<int>(idx, numBins).ToArray();

        var expected = new float[numBins];
        foreach (int v in idx.ToArray())
            if (v >= 0 && v < numBins) expected[v] += 1f;

        AssertClose(expected, counts, "count_float", rel: 0);
        for (int b = 0; b < numBins; b++)
            Assert.Equal((int)expected[b], intCounts[b]);
    }

    [Fact]
    public void FusedLinearCrossEntropy_DetachedReadout_PreservesSharedReadoutGradients()
    {
        IEngine e = new CpuEngine();
        int n = 12, d = 16, vocab = 48;
        var hIntermediate = RandFloat([n, d], 60);
        var hFinal = RandFloat([n, d], 61);
        var sharedWeight = RandFloat([d, vocab], 62);
        var sharedBias = RandFloat([vocab], 63);
        var targets = RandIndices([n], vocab, 64);

        // Reference: compute intermediate loss with a cloned detached copy of sharedWeight/sharedBias, and final loss
        // with the live sharedWeight/sharedBias.
        float[] refDhInter, refDhFinal, refDw, refDb;
        float refLoss;
        using (var tapeRef = new GradientTape<float>())
        {
            var wDetached = new Tensor<float>((float[])sharedWeight.ToArray().Clone(), [d, vocab]);
            var bDetached = new Tensor<float>((float[])sharedBias.ToArray().Clone(), [vocab]);
            var lInter = e.FusedLinearCrossEntropyWithLogits(hIntermediate, wDetached, bDetached, targets);
            var lFinal = e.FusedLinearCrossEntropyWithLogits(hFinal, sharedWeight, sharedBias, targets);
            var total = e.TensorAdd(e.TensorMultiplyScalar(lInter, 0.25f), lFinal);
            refLoss = total[0];
            var grads = tapeRef.ComputeGradients(total, [hIntermediate, hFinal, sharedWeight, sharedBias]);
            refDhInter = grads[hIntermediate].ToArray();
            refDhFinal = grads[hFinal].ToArray();
            refDw = grads[sharedWeight].ToArray();
            refDb = grads[sharedBias].ToArray();
        }

        // Detached-readout overload: passes live sharedWeight/sharedBias directly with computeReadoutGrads: false on
        // the intermediate layer — zero clone, zero dWeight GEMM on the intermediate layer, exact same gradients!
        using var tape = new GradientTape<float>();
        var lossInter = e.FusedLinearCrossEntropyWithLogits(
            hIntermediate, sharedWeight, sharedBias, targets, computeReadoutGrads: false);
        var lossFinal = e.FusedLinearCrossEntropyWithLogits(
            hFinal, sharedWeight, sharedBias, targets, computeReadoutGrads: true);
        var totalLoss = e.TensorAdd(e.TensorMultiplyScalar(lossInter, 0.25f), lossFinal);
        var g = tape.ComputeGradients(totalLoss, [hIntermediate, hFinal, sharedWeight, sharedBias]);

        Assert.True(Math.Abs(refLoss - totalLoss[0]) <= 1e-6f, $"loss: expected {refLoss}, got {totalLoss[0]}");
        AssertClose(refDhInter, g[hIntermediate].ToArray(), "dh_inter");
        AssertClose(refDhFinal, g[hFinal].ToArray(), "dh_final");
        AssertClose(refDw, g[sharedWeight].ToArray(), "dw_shared");
        AssertClose(refDb, g[sharedBias].ToArray(), "db_shared");
    }
}
