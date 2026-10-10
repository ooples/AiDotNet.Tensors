using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// #653 phase 2: the ops that ran on one thread at every size now fan out -- trailing-repeat broadcast add/multiply,
/// 2-D ReduceSum, single-axis ReduceMean and large strided Contiguous() copies. The issue's correctness gate is
/// bit-identical output versus serial, so every case runs at DOP 1 and DOP 16 and the results must match exactly, and
/// against an independent element-by-element reference.
/// </summary>
[Collection("EngineCurrentGlobalState")]
public class CoreSaturationFanOutTests
{
    private static Tensor<T> Filled<T>(int salt, params int[] shape)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var t = new Tensor<T>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = ops.FromDouble(Math.Sin((i + salt) * 0.731) * 3 + (i % 11) * 0.125);
        return t;
    }

    private static Tensor<T> AtDop<T>(int dop, Func<Tensor<T>> run)
    {
        int previous = CpuParallelSettings.MaxDegreeOfParallelism;
        CpuParallelSettings.MaxDegreeOfParallelism = dop;
        try { var r = run(); return r.Contiguous(); }
        finally { CpuParallelSettings.MaxDegreeOfParallelism = previous; }
    }

    private static void AssertBitIdentical<T>(Tensor<T> serial, Tensor<T> parallel, string what)
    {
        Assert.Equal(serial._shape, parallel._shape);
        for (int i = 0; i < serial.Length; i++)
            Assert.True(serial[i]!.Equals(parallel[i]), $"{what}: element {i} differs between DOP 1 ({serial[i]}) and DOP 16 ({parallel[i]})");
    }

    [Theory]
    [InlineData(false, 4096, 96)]
    [InlineData(true, 4096, 96)]
    [InlineData(false, 7, 33)]   // below the grain: the serial path
    public void TrailingRepeatBroadcast_FansOut_BitIdentical(bool multiply, int rows, int cols)
    {
        var engine = new CpuEngine();
        foreach (var useDouble in new[] { false, true })
        {
            if (useDouble) Check<double>(); else Check<float>();
        }

        void Check<T>()
        {
            var ops = MathHelper.GetNumericOperations<T>();
            var x = Filled<T>(1, rows, cols); var bias = Filled<T>(2, cols);
            Func<Tensor<T>> run = () => multiply ? engine.TensorBroadcastMultiply(x, bias) : engine.TensorBroadcastAdd(x, bias);
            var s = AtDop(1, run); var p = AtDop(16, run);
            AssertBitIdentical(s, p, $"{typeof(T).Name} {(multiply ? "mul" : "add")}");
            for (int i = 0; i < rows * cols; i += 997)
            {
                T expected = multiply ? ops.Multiply(x[i], bias[i % cols]) : ops.Add(x[i], bias[i % cols]);
                Assert.Equal(expected, p[i]);
            }
        }
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    public void ReduceSum2D_FansOut_BitIdenticalToSequentialOrder(int axis)
    {
        var engine = new CpuEngine();
        const int rows = 3000, cols = 300;
        var xd = Filled<double>(3, rows, cols); var xf = Filled<float>(3, rows, cols);
        AssertBitIdentical(AtDop(1, () => engine.ReduceSum(xd, new[] { axis })), AtDop(16, () => engine.ReduceSum(xd, new[] { axis })), "double sum");
        var pf = AtDop(16, () => engine.ReduceSum(xf, new[] { axis }));
        AssertBitIdentical(AtDop(1, () => engine.ReduceSum(xf, new[] { axis })), pf, "float sum");
        // Independent reference in the historical sequential order (row-major accumulation).
        int outLen = axis == 0 ? cols : rows;
        for (int k = 0; k < outLen; k++)
        {
            float acc = 0;
            if (axis == 0) for (int r = 0; r < rows; r++) acc += xf[r * cols + k];
            else for (int c = 0; c < cols; c++) acc += xf[k * cols + c];
            Assert.Equal(acc, pf[k]);
        }
    }

    [Theory]
    [InlineData(new[] { 2048, 128 }, 0)]
    [InlineData(new[] { 2048, 128 }, 1)]
    [InlineData(new[] { 16, 64, 96 }, 1)]
    [InlineData(new[] { 16, 64, 96 }, 2)]
    public void ReduceMeanSingleAxis_FansOut_BitIdenticalToOdometerOrder(int[] shape, int axis)
    {
        var engine = new CpuEngine();
        var x = Filled<float>(5, shape);
        var p = AtDop(16, () => engine.ReduceMean(x, new[] { axis }, keepDims: false));
        AssertBitIdentical(AtDop(1, () => engine.ReduceMean(x, new[] { axis }, keepDims: false)), p, "float mean");
        int outer = 1, inner = 1;
        for (int d = 0; d < axis; d++) outer *= shape[d];
        for (int d = axis + 1; d < shape.Length; d++) inner *= shape[d];
        for (int o = 0; o < outer; o++)
            for (int j = 0; j < inner; j++)
            {
                float acc = 0;
                for (int a = 0; a < shape[axis]; a++) acc += x[(o * shape[axis] + a) * inner + j];
                Assert.Equal(acc * (1.0f / shape[axis]), p[o * inner + j]);
            }
    }

    [Theory]
    [InlineData("transpose2d")]
    [InlineData("permute021")]
    [InlineData("permute102")]
    [InlineData("narrowed")]
    public void LargeStridedContiguous_FansOut_MatchesElementwise(string kind)
    {
        var engine = new CpuEngine();
        Tensor<float> View()
        {
            return kind switch
            {
                "transpose2d" => engine.TensorPermute(Filled<float>(7, 520, 300), new[] { 1, 0 }),
                "permute021" => engine.TensorPermute(Filled<float>(7, 8, 120, 90), new[] { 0, 2, 1 }),
                "permute102" => engine.TensorPermute(Filled<float>(7, 120, 8, 90), new[] { 1, 0, 2 }),
                _ => engine.TensorNarrow(Filled<float>(7, 8, 140, 96), 1, 10, 100),
            };
        }
        var view = View();
        Assert.False(view.IsContiguous);
        Assert.True(view.Length >= 64 * 1024, "the case must exercise the parallel copy");
        var p = AtDop(16, () => View());
        AssertBitIdentical(AtDop(1, () => View()), p, kind);
        // Reference: index the view element by element.
        var idx = new int[view.Rank];
        for (int flat = 0; flat < view.Length; flat++)
        {
            int rem = flat;
            for (int d = view.Rank - 1; d >= 0; d--) { idx[d] = rem % view._shape[d]; rem /= view._shape[d]; }
            Assert.Equal(view[idx], p[flat]);
        }
    }
}
