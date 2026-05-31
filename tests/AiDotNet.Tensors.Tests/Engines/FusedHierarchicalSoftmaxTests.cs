using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// #499: FusedHierarchicalSoftmax — class probabilities over a balanced binary tree
/// whose per-level gate is sigmoid(input·nodeWeights[level]). The fused op computes
/// the treeDepth shared gate sigmoids ONCE per row then forms each class's path
/// product; this verifies it matches the naive per-class path-probability reference
/// (which recomputes the gate for every class, as the eager layer does).
/// </summary>
public class FusedHierarchicalSoftmaxTests
{
    private static double NaivePathProb(double[] input, double[][] nodeW, int treeDepth, int numClasses, int cls)
    {
        double prob = 1.0; int node = 1;
        for (int depth = 0; depth < treeDepth; depth++)
        {
            double dot = 0; for (int k = 0; k < input.Length; k++) dot += input[k] * nodeW[depth][k];
            double gate = 1.0 / (1.0 + Math.Exp(-dot));
            bool goRight = (cls & (1 << (treeDepth - depth - 1))) != 0;
            prob *= goRight ? gate : (1.0 - gate);
            node = node * 2 + (goRight ? 1 : 0);
            if (node >= numClasses) break;
        }
        return prob;
    }

    [Theory]
    [InlineData(8)]   // power of two — complete tree, probabilities sum to 1
    [InlineData(5)]   // non-power-of-two — incomplete tree (early break path)
    [InlineData(16)]
    public void FusedHierarchicalSoftmax_MatchesNaivePathProbability(int numClasses)
    {
        var engine = new CpuEngine();
        const int rows = 4, d = 9;
        int treeDepth = (int)Math.Ceiling(Math.Log(numClasses, 2)); // net471 has no Math.Log2
        var rng = new Random(20260620);

        var xData = new float[rows * d];
        for (int i = 0; i < xData.Length; i++) xData[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        var x = new Tensor<float>(xData, new[] { rows, d });
        var wData = new float[treeDepth * d];
        for (int i = 0; i < wData.Length; i++) wData[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        var nodeW = new Tensor<float>(wData, new[] { treeDepth, d });

        var outT = engine.FusedHierarchicalSoftmax(x, nodeW, numClasses);
        Assert.Equal(numClasses, outT._shape[outT._shape.Length - 1]);
        Assert.Equal(rows * numClasses, outT.Length);

        var wRows = new double[treeDepth][];
        for (int t = 0; t < treeDepth; t++)
        {
            wRows[t] = new double[d];
            for (int k = 0; k < d; k++) wRows[t][k] = wData[t * d + k];
        }

        for (int r = 0; r < rows; r++)
        {
            var inRow = new double[d];
            for (int k = 0; k < d; k++) inRow[k] = xData[r * d + k];
            double rowSum = 0;
            for (int c = 0; c < numClasses; c++)
            {
                double expected = NaivePathProb(inRow, wRows, treeDepth, numClasses, c);
                double actual = Convert.ToDouble(outT[r * numClasses + c]);
                Assert.True(Math.Abs(expected - actual) < 1e-5, $"HSoftmax[{r},{c}] {actual} != {expected}");
                rowSum += actual;
            }
            // A complete (power-of-two) tree's leaf probabilities form a distribution.
            if ((numClasses & (numClasses - 1)) == 0)
                Assert.True(Math.Abs(rowSum - 1.0) < 1e-4, $"complete-tree row {r} should sum to 1 ({rowSum})");
        }
    }
}
