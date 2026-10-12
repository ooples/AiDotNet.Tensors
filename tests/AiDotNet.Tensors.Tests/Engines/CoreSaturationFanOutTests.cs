using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Simd;
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
    [InlineData(false, 4096, 96)]   // below the 1M fan-out floor: the typed serial loop
    [InlineData(false, 10923, 97)]  // ~1.06M elements, ragged row: the fanned-out typed loop
    [InlineData(true, 10923, 97)]
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
    [InlineData(0, 3000, 300)]
    [InlineData(1, 3000, 300)]
    [InlineData(0, 40, 4500)]   // wider than one accumulation range (2048 columns), ragged
    public void ReduceSum2D_FansOut_BitIdenticalToSequentialOrder(int axis, int rows, int cols)
    {
        var engine = new CpuEngine();
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
    [InlineData(new[] { 40, 4500 }, 0)]       // one outer row wider than an accumulation range (2048), ragged
    [InlineData(new[] { 3, 50, 2100 }, 1)]    // several outer rows, each just past one range
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
    [InlineData("transposeThin")]     // [6144, 32] -> [32, 6144]: one row of 32 x 32 tiles, split over inner tiles
    [InlineData("transposeRagged")]   // neither extent a multiple of the tile
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
                "transposeThin" => engine.TensorPermute(Filled<float>(7, 6144, 32), new[] { 1, 0 }),
                "transposeRagged" => engine.TensorPermute(Filled<float>(7, 70, 3001), new[] { 1, 0 }),
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

    // ELU fans out in vector-aligned chunks: equal to one whole-array SIMD pass (the pre-#653 result) at every DOP.
    [Theory]
    [InlineData(4096 * 96)]
    [InlineData(1_000_003)]   // a ragged tail
    [InlineData(1000)]        // below the grain
    public void Elu_FansOut_EqualsWholeArrayPass(int n)
    {
        var engine = new CpuEngine();
        var xf = Filled<float>(5, n);
        var xd = Filled<double>(5, n);
        var refF = new float[n];
        unsafe
        {
            fixed (float* src = xf.GetDataArray())
            fixed (float* dst = refF)
                SimdKernels.ELUUnsafe(src, dst, n, 0.7f);   // the engine's float kernel, one whole-array pass
        }
        var refD = new double[n];
        SimdKernels.ELU(xd.AsSpan(), 0.7, refD);
        foreach (int dop in new[] { 1, 16, 64 })
        {
            var f = AtDop(dop, () => engine.ELU(xf, 0.7));
            var d = AtDop(dop, () => engine.ELU(xd, 0.7));
            for (int i = 0; i < n; i++)
            {
                Assert.True(f[i].Equals(refF[i]), $"float elu[{i}] at DOP {dop}: {f[i]} vs {refF[i]}");
                Assert.True(d[i].Equals(refD[i]), $"double elu[{i}] at DOP {dop}: {d[i]} vs {refD[i]}");
            }
        }
    }

    // Concat copies row blocks in parallel: element-for-element equal to the inputs, any axis, any DOP.
    [Theory]
    [InlineData(1)]
    [InlineData(0)]
    [InlineData(2)]
    public void Concat_FansOut_CopiesEveryBlock(int axis)
    {
        var engine = new CpuEngine();
        var a = Filled<float>(6, 300, 70, 9);
        var b = Filled<float>(7, axis == 0 ? 120 : 300, axis == 1 ? 33 : 70, axis == 2 ? 4 : 9);
        foreach (int dop in new[] { 1, 16 })
        {
            var r = AtDop(dop, () => engine.Concat(new[] { a, b }, axis));
            int[] s = r._shape;
            for (int i = 0; i < r.Length; i += 7)
            {
                int i2 = i % s[2], i1 = i / s[2] % s[1], i0 = i / (s[1] * s[2]);
                int[] c = { i0, i1, i2 };
                var src = c[axis] < a._shape[axis] ? a : b;
                if (src == b) c[axis] -= a._shape[axis];
                Assert.Equal(src[(c[0] * src._shape[1] + c[1]) * src._shape[2] + c[2]], r[i]);
            }
        }
    }

    // Single-axis ReduceMax fans out: values and absolute argmax indices equal the serial semantics (MinValue start,
    // strictly-greater updates so the first maximum wins and NaN never does, -inf below MinValue keeps index -1).
    [Theory]
    [InlineData(1)]
    [InlineData(0)]
    [InlineData(2)]
    public void ReduceMaxSingleAxis_FansOut_MatchesSerialSemantics(int axis)
    {
        var engine = new CpuEngine();
        var x = Filled<float>(8, 600, 5, 200);
        for (int i = 0; i < x.Length; i += 37) x[i] = float.NaN;
        for (int i = 3; i < x.Length; i += 53) x[i] = 2.0f;   // ties
        for (int j = 0; j < 200; j++) for (int a = 0; a < 5; a++) x[(1 * 5 + a) * 200 + j] = float.NegativeInfinity;   // an all -inf slab
        {
            // A tied maximum at two positions along the reduced axis for every cell of the first outer row: the first
            // occurrence must win.
            int[] sh = x._shape;
            int innerT = 1;
            for (int d = axis + 1; d < 3; d++) innerT *= sh[d];
            for (int j = 0; j < innerT; j++)
            {
                x[(0 * sh[axis] + 1) * innerT + j] = 1e6f;
                x[(0 * sh[axis] + sh[axis] - 1) * innerT + j] = 1e6f;
            }
        }
        foreach (int dop in new[] { 1, 16 })
        {
            int[] idx = Array.Empty<int>();
            var r = AtDop(dop, () => engine.ReduceMax(x, new[] { axis }, false, out idx));
            int[] shape = x._shape;
            int outer = 1, inner = 1;
            for (int d = 0; d < axis; d++) outer *= shape[d];
            for (int d = axis + 1; d < 3; d++) inner *= shape[d];
            for (int o = 0; o < outer; o++)
                for (int j = 0; j < inner; j++)
                {
                    float best = float.MinValue; int bi = -1;
                    for (int a = 0; a < shape[axis]; a++)
                    {
                        int flat = (o * shape[axis] + a) * inner + j;
                        if (x[flat] > best) { best = x[flat]; bi = flat; }
                    }
                    Assert.True(best.Equals(r[o * inner + j]), $"max[{o},{j}] at DOP {dop}: {r[o * inner + j]} vs {best}");
                    Assert.Equal(bi, idx[o * inner + j]);
                }
        }
    }
}
