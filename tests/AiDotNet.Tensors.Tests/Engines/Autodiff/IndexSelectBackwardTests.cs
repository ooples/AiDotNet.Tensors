using System;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>IndexSelect backward on the CPU engine: leading-axis (engine scatter-add) and inner-axis (host loop) paths.</summary>
public sealed class IndexSelectBackwardTests
{
    [Theory]
    [InlineData(0)]   // outer size 1: routed through TensorEmbeddingLookupBackward
    [InlineData(1)]   // outer size > 1: array loop
    public void Gradient_AccumulatesRepeatedIndices(int axis)
    {
        var eng = new CpuEngine();
        const int r = 5, c = 7;
        var rng = new Random(axis + 1);
        var xData = Enumerable.Range(0, r * c).Select(_ => rng.NextDouble()).ToArray();
        int dimSize = axis == 0 ? r : c;
        var idx = new[] { 1, 3, 1, dimSize - 1, 0, 1 };
        int[] outShape = axis == 0 ? new[] { idx.Length, c } : new[] { r, idx.Length };
        var wData = Enumerable.Range(0, outShape[0] * outShape[1]).Select(_ => rng.NextDouble()).ToArray();

        var x = new Tensor<double>(new[] { r, c }, new Vector<double>(xData));
        var w = new Tensor<double>(outShape, new Vector<double>(wData));
        Tensor<double> g;
        using (var tape = new GradientTape<double>())
        {
            var y = eng.TensorIndexSelectDiff(x, new Tensor<int>(idx, new[] { idx.Length }), axis);
            var loss = eng.ReduceSum(eng.TensorMultiply(y, w), new[] { 0, 1 }, keepDims: false);
            g = tape.ComputeGradients(loss, new[] { x })[x];
        }

        var expected = new double[r * c];
        for (int i = 0; i < idx.Length; i++)
            if (axis == 0)
                for (int j = 0; j < c; j++) expected[idx[i] * c + j] += wData[i * c + j];
            else
                for (int j = 0; j < r; j++) expected[j * c + idx[i]] += wData[j * idx.Length + i];

        var actual = g.ToArray();
        Assert.Equal(new[] { r, c }, g.Shape.ToArray());
        for (int j = 0; j < expected.Length; j++) Assert.Equal(expected[j], actual[j], 12);
    }
}
