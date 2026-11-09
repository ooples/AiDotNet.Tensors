using System;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// CPU TensorScatter with torch.scatter semantics on any rank and axis, and its gradient. It used to accept only a 2-D
/// tensor with axis 1, wrote through the per-element indexer, and recorded no backward, so nothing flowed to the
/// destination or the source in training.
/// </summary>
public sealed class ScatterAnyRankTests
{
    public static TheoryData<int[], int, int> Cases => new()
    {
        { new[] { 9 }, 0, 4 },
        { new[] { 3, 8 }, 1, 3 },
        { new[] { 6, 5 }, 0, 2 },
        { new[] { 2, 7, 4 }, 1, 3 },
        { new[] { 2, 3, 6 }, 2, 2 },
        { new[] { 2, 3, 6 }, -1, 2 },
        { new[] { 5, 2, 3 }, 0, 2 },
    };

    // Indices along the axis: distinct within each fibre, so the reference is order-independent.
    private static Tensor<int> Indices(int[] shape, int ax, int count, Random rng)
    {
        var idxShape = (int[])shape.Clone();
        idxShape[ax] = count;
        int outer = shape.Take(ax).Aggregate(1, (a, b) => a * b), inner = shape.Skip(ax + 1).Aggregate(1, (a, b) => a * b);
        var data = new int[outer * count * inner];
        for (int o = 0; o < outer; o++)
            for (int k = 0; k < inner; k++)
            {
                var pick = Enumerable.Range(0, shape[ax]).OrderBy(_ => rng.Next()).Take(count).ToArray();
                for (int j = 0; j < count; j++) data[(o * count + j) * inner + k] = pick[j];
            }
        return new Tensor<int>(data, idxShape);
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void Scatters_along_the_axis_and_back_propagates(int[] shape, int axis, int count)
    {
        var eng = new CpuEngine();
        int ax = axis < 0 ? shape.Length + axis : axis;
        var rng = new Random(shape.Sum() * 10 + ax);
        int n = shape.Aggregate(1, (a, b) => a * b);
        var idx = Indices(shape, ax, count, rng);
        var dstData = Enumerable.Range(0, n).Select(_ => rng.NextDouble()).ToArray();
        var srcData = Enumerable.Range(0, idx.Length).Select(_ => rng.NextDouble()).ToArray();
        var wData = Enumerable.Range(0, n).Select(_ => rng.NextDouble()).ToArray();
        var dst = new Tensor<double>(shape, new Vector<double>(dstData));
        var src = new Tensor<double>(idx.Shape.ToArray(), new Vector<double>(srcData));
        var w = new Tensor<double>(shape, new Vector<double>(wData));

        Tensor<double> y;
        System.Collections.Generic.Dictionary<Tensor<double>, Tensor<double>> g;
        using (var tape = new GradientTape<double>())
        {
            y = eng.TensorScatter(dst, idx, src, axis);
            var loss = eng.ReduceSum(eng.TensorMultiply(y, w), null, keepDims: false);
            g = tape.ComputeGradients(loss, new[] { dst, src });
        }

        int outer = shape.Take(ax).Aggregate(1, (a, b) => a * b), inner = shape.Skip(ax + 1).Aggregate(1, (a, b) => a * b);
        var expected = (double[])dstData.Clone();
        var gDst = (double[])wData.Clone();
        var gSrc = new double[idx.Length];
        var idxData = idx.ToArray();
        for (int o = 0; o < outer; o++)
            for (int j = 0; j < count; j++)
                for (int k = 0; k < inner; k++)
                {
                    int p = (o * count + j) * inner + k, q = (o * shape[ax] + idxData[p]) * inner + k;
                    expected[q] = srcData[p];
                    gDst[q] = 0;
                    gSrc[p] = wData[q];
                }

        Assert.Equal(shape, y.Shape.ToArray());
        Assert.Equal(expected, y.ToArray());
        Assert.Equal(gDst, g[dst].ToArray());
        Assert.Equal(gSrc, g[src].ToArray());
        Assert.Equal(dstData, dst.ToArray()); // the destination is not modified
    }

    [Fact]
    public void Overwritten_source_entries_get_no_gradient()
    {
        // Last write wins: with indices [0, 0] along the axis, only the second source entry reaches the output.
        var eng = new CpuEngine();
        var dst = new Tensor<double>(new[] { 1, 3 }, new Vector<double>(new[] { 1.0, 2.0, 3.0 }));
        var src = new Tensor<double>(new[] { 1, 2 }, new Vector<double>(new[] { 10.0, 20.0 }));
        var idx = new Tensor<int>(new[] { 0, 0 }, new[] { 1, 2 });
        using var tape = new GradientTape<double>();
        var y = eng.TensorScatter(dst, idx, src, 1);
        var g = tape.ComputeGradients(eng.ReduceSum(y, null, keepDims: false), new[] { dst, src });
        Assert.Equal(new[] { 20.0, 2.0, 3.0 }, y.ToArray());
        Assert.Equal(new[] { 0.0, 1.0 }, g[src].ToArray());
        Assert.Equal(new[] { 0.0, 1.0, 1.0 }, g[dst].ToArray());
    }

    [Fact]
    public void Compiled_training_plan_propagates_scatter_gradients()
    {
        var priorEngine = AiDotNetEngine.Current;
        AiDotNetEngine.Current = new CpuEngine();
        try
        {
            var dst = new Tensor<float>(new[] { 2, 3, 4 }, new Vector<float>(Enumerable.Range(0, 24).Select(i => 0.1f * i).ToArray()));
            var src = new Tensor<float>(new[] { 2, 2, 4 }, new Vector<float>(Enumerable.Range(0, 16).Select(i => 1f + i).ToArray()));
            var idx = Indices(new[] { 2, 3, 4 }, 1, 2, new Random(3));
            var w = new Tensor<float>(new[] { 2, 3, 4 }, new Vector<float>(Enumerable.Range(0, 24).Select(i => 1f + (i % 5)).ToArray()));
            Tensor<float> Forward(IEngine e) => e.ReduceSum(e.TensorMultiply(e.TensorScatter(dst, idx, src, 1), w), null);
            System.Collections.Generic.Dictionary<Tensor<float>, Tensor<float>> tapeGrads;
            using (var t = new GradientTape<float>())
                tapeGrads = t.ComputeGradients(Forward(new CpuEngine()), new[] { dst, src });
            ICompiledTrainingPlan<float> plan;
            using (var scope = GraphMode.EnableTraining(new[] { dst, src }))
            {
                Forward(new CpuEngine());
                plan = scope.CompileTraining(new[] { dst, src });
            }
            try
            {
                plan.Step();
                Assert.Equal(tapeGrads[dst].ToArray(), plan.Gradients[0].ToArray());
                Assert.Equal(tapeGrads[src].ToArray(), plan.Gradients[1].ToArray());
            }
            finally { plan.Dispose(); }
        }
        finally { AiDotNetEngine.Current = priorEngine; }
    }

    [Fact]
    public void Index_copy_and_add_reject_values_of_the_wrong_shape()
    {
        // [2, 3, 4] with two indices on axis 1 needs values [2, 2, 4]; [2, 4, 2] has the same length, not the shape.
        var eng = new CpuEngine();
        var input = new Tensor<float>(new[] { 2, 3, 4 });
        var idx = new Tensor<int>(new[] { 0, 2 }, new[] { 2 });
        Assert.Throws<ArgumentException>(() => eng.Scatter(input, idx, new Tensor<float>(new[] { 2, 4, 2 }), 1));
        Assert.Throws<ArgumentException>(() => eng.ScatterAdd(input, idx, new Tensor<float>(new[] { 2, 4, 2 }), 1));
        Assert.Equal(new[] { 2, 3, 4 }, eng.Scatter(input, idx, new Tensor<float>(new[] { 2, 2, 4 }), 1).Shape.ToArray());
    }

    [Fact]
    public void Rejects_bad_indices_and_shapes()
    {
        var eng = new CpuEngine();
        var dst = new Tensor<float>(new[] { 2, 4 });
        Assert.Throws<IndexOutOfRangeException>(() =>
            eng.TensorScatter(dst, new Tensor<int>(new[] { 0, 4, 1, 1 }, new[] { 2, 2 }), new Tensor<float>(new[] { 2, 2 }), 1));
        Assert.Throws<ArgumentException>(() =>
            eng.TensorScatter(dst, new Tensor<int>(new[] { 0, 1 }, new[] { 1, 2 }), new Tensor<float>(new[] { 1, 2 }), 1));
        Assert.Throws<ArgumentException>(() =>
            eng.TensorScatter(dst, new Tensor<int>(new[] { 0, 1, 2, 3 }, new[] { 2, 2 }), new Tensor<float>(new[] { 2, 3 }), 1));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            eng.TensorScatter(dst, new Tensor<int>(new[] { 0, 1 }, new[] { 2, 1 }), new Tensor<float>(new[] { 2, 1 }), 2));
    }

    [Fact]
    public void Repeated_index_keeps_the_last_write_and_only_it_gets_the_gradient()
    {
        var eng = new CpuEngine();
        var dst = new Tensor<double>(new[] { 1, 3 }, new Vector<double>(new[] { 1.0, 2.0, 3.0 }));
        var idx = new Tensor<int>(new[] { 0, 0 }, new[] { 1, 2 });
        var src = new Tensor<double>(new[] { 1, 2 }, new Vector<double>(new[] { 5.0, 7.0 }));
        var w = new Tensor<double>(new[] { 1, 3 }, new Vector<double>(new[] { 2.0, 3.0, 4.0 }));
        Tensor<double> y;
        System.Collections.Generic.Dictionary<Tensor<double>, Tensor<double>> g;
        using (var tape = new GradientTape<double>())
        {
            y = eng.TensorScatter(dst, idx, src, 1);
            g = tape.ComputeGradients(eng.ReduceSum(eng.TensorMultiply(y, w), null, keepDims: false), new[] { dst, src });
        }
        // torch.scatter: the second write to slot 0 wins, so the first source element is overwritten and gets no
        // gradient; slot 0 of the destination is overwritten too.
        Assert.Equal(new[] { 7.0, 2.0, 3.0 }, y.ToArray());
        Assert.Equal(new[] { 0.0, 2.0 }, g[src].ToArray());
        Assert.Equal(new[] { 0.0, 3.0, 4.0 }, g[dst].ToArray());
    }

    [Fact]
    public void TopK_routes_the_gradient_to_the_selected_positions()
    {
        var eng = new CpuEngine();
        var x = new Tensor<double>(new[] { 2, 3 }, new Vector<double>(new[] { 3.0, 1.0, 2.0, 0.5, 4.0, 6.0 }));
        var w = new Tensor<double>(new[] { 2, 2 }, new Vector<double>(new[] { 10.0, 20.0, 30.0, 40.0 }));
        Tensor<double> values;
        System.Collections.Generic.Dictionary<Tensor<double>, Tensor<double>> g;
        using (var tape = new GradientTape<double>())
        {
            (values, _) = eng.TopK(x, 2, axis: 1);
            g = tape.ComputeGradients(eng.ReduceSum(eng.TensorMultiply(values, w), null, keepDims: false), new[] { x });
        }
        Assert.Equal(new[] { 3.0, 2.0, 6.0, 4.0 }, values.ToArray());
        Assert.Equal(new[] { 10.0, 0.0, 20.0, 0.0, 40.0, 30.0 }, g[x].ToArray());
    }

    [Fact]
    public void Scatter_and_scatter_add_reject_values_of_the_wrong_shape()
    {
        var eng = new CpuEngine();
        var input = new Tensor<float>(new[] { 4, 3 });
        var indices = new Tensor<int>(new[] { 0, 2 }, new[] { 2 });
        // Two rows of 3 are expected; [3, 2] has the right element count but the wrong shape.
        Assert.Throws<ArgumentException>(() => eng.Scatter(input, indices, new Tensor<float>(new[] { 3, 2 }), 0));
        Assert.Throws<ArgumentException>(() => eng.ScatterAdd(input, indices, new Tensor<float>(new[] { 3, 2 }), 0));
        Assert.Throws<ArgumentException>(() => eng.Scatter(input, indices, new Tensor<float>(new[] { 6 }), 0));
        Assert.Equal(new[] { 4, 3 }, eng.Scatter(input, indices, new Tensor<float>(new[] { 2, 3 }), 0).Shape.ToArray());
    }

    [Fact]
    public void Set_slice_rejects_a_source_that_is_not_the_slice()
    {
        var eng = new CpuEngine();
        var dst = new Tensor<float>(new[] { 2, 3, 4 });
        // A full-size source used to be copied as a flat prefix, writing the wrong elements on every axis but the last.
        Assert.Throws<ArgumentException>(() => eng.TensorSetSliceAxis(dst, new Tensor<float>(new[] { 2, 3, 4 }), 1, 0));
        Assert.Throws<ArgumentException>(() => eng.TensorSetSliceAxis(dst, new Tensor<float>(new[] { 4, 2 }), 1, 0));
        var slice = new Tensor<float>(Enumerable.Range(1, 8).Select(i => (float)i).ToArray(), new[] { 2, 4 });
        eng.TensorSetSliceAxis(dst, slice, 1, 2);
        // The slice's gradient keeps the axis as size 1; that is the same slice.
        eng.TensorSetSliceAxis(dst, new Tensor<float>(slice.ToArray(), new[] { 2, 1, 4 }), 1, 0);
        var d = dst.ToArray();
        Assert.Equal(new float[] { 1, 2, 3, 4, 0, 0, 0, 0, 1, 2, 3, 4 }, d.Take(12).ToArray());
        Assert.Equal(new float[] { 5, 6, 7, 8, 0, 0, 0, 0, 5, 6, 7, 8 }, d.Skip(12).ToArray());
    }

    [Fact]
    public void Tensor_index_with_a_matrix_of_indices_keeps_the_index_shape()
    {
        var data = Enumerable.Range(0, 24).Select(i => (float)i).ToArray();
        var t = new Tensor<float>(new[] { 3, 2, 4 }, new Vector<float>(data));
        var picked = t[new Tensor<int>(new[] { 2, 0, 1, 2 }, new[] { 2, 2 })];
        Assert.Equal(new[] { 2, 2, 2, 4 }, picked.Shape.ToArray());
        var expected = new[] { 2, 0, 1, 2 }.SelectMany(r => data.Skip(r * 8).Take(8)).ToArray();
        Assert.Equal(expected, picked.ToArray());
    }
}
