using System;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// #1131: the CPU sort uses PyTorch's total order (NaN after +inf ascending and first descending, -0 equal to +0,
/// ties in original index order). #1134: a 2-D GridSample point with a NaN, infinite or out-of-int-range coordinate
/// samples nothing (0) in every mode and padding, and passes no gradient.
/// </summary>
public class NonFiniteSortAndGridSampleTests
{
    private readonly CpuEngine _cpu = new();

    private static readonly float[] Pool = { float.NaN, float.PositiveInfinity, float.NegativeInfinity, 2f, 2f, -1f, 0f, -0f, 0.5f };

    // torch.sort(stable=True) order of one row: positions into the row.
    private static int[] TorchOrder(float[] row, bool descending) =>
        row.Select((v, i) => (Value: v, Index: i))
            .OrderBy(p => float.IsNaN(p.Value) ? (descending ? 0 : 2) : 1)
            .ThenBy(p => float.IsNaN(p.Value) ? 0f : descending ? -p.Value : p.Value)
            .ThenBy(p => p.Index)
            .Select(p => p.Index)
            .ToArray();

    private static float[] RandomRows(int seed, int count)
    {
        var rng = new Random(seed);
        var data = new float[count];
        for (int i = 0; i < count; i++) data[i] = Pool[rng.Next(Pool.Length)];
        return data;
    }

    // Checks Sort / Argsort along `axis` of a [outer, axisSize, inner] layout against TorchOrder.
    private void CheckSort<T>(Func<float, T> make, Func<T, float> back, int outer, int axisSize, int inner, bool descending)
    {
        var raw = RandomRows(outer * 31 + axisSize * 7 + inner + (descending ? 1 : 0), outer * axisSize * inner);
        var x = new Tensor<T>(raw.Select(make).ToArray(), inner == 1 ? new[] { outer, axisSize } : new[] { outer, axisSize, inner });
        int axis = 1;
        var (values, indices) = _cpu.TensorSort(x, axis, descending);
        var argsort = _cpu.TensorArgsort(x, axis, descending);
        for (int o = 0; o < outer; o++)
            for (int n = 0; n < inner; n++)
            {
                var row = new float[axisSize];
                for (int a = 0; a < axisSize; a++) row[a] = back(x[(o * axisSize + a) * inner + n]);
                var expected = TorchOrder(row, descending);
                for (int a = 0; a < axisSize; a++)
                {
                    int pos = (o * axisSize + a) * inner + n;
                    Assert.True(expected[a] == indices[pos],
                        $"{typeof(T).Name} {(descending ? "desc" : "asc")} [{o}, :, {n}] position {a}: index {indices[pos]}, expected {expected[a]} (row {string.Join(", ", row)})");
                    Assert.Equal(indices[pos], argsort[pos]);
                    // The value is the source element itself (so -0 stays -0 and NaN stays NaN).
                    Assert.Equal(BitConverter.DoubleToInt64Bits(back(x[(o * axisSize + expected[a]) * inner + n])),
                        BitConverter.DoubleToInt64Bits(back(values[pos])));
                }
            }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Sort_Float_InnerAxis_TorchOrder(bool descending) => CheckSort(v => v, v => v, 6, 19, 1, descending);

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Sort_Float_MiddleAxis_TorchOrder(bool descending) => CheckSort(v => v, v => v, 3, 17, 4, descending);

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Sort_Double_TorchOrder(bool descending)
    {
        CheckSort(v => (double)v, v => (float)v, 5, 23, 1, descending);
        CheckSort(v => (double)v, v => (float)v, 2, 11, 3, descending);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Sort_Int_IsStable(bool descending)
    {
        // No NaN in an integer type: the order is the values', ties in original index order.
        var rng = new Random(descending ? 5 : 4);
        var data = Enumerable.Range(0, 4 * 29).Select(_ => rng.Next(-3, 4)).ToArray();
        var x = new Tensor<int>(data, new[] { 4, 29 });
        var (_, indices) = _cpu.TensorSort(x, -1, descending);
        for (int r = 0; r < 4; r++)
        {
            var expected = Enumerable.Range(0, 29)
                .OrderBy(k => descending ? -data[r * 29 + k] : data[r * 29 + k]).ThenBy(k => k).ToArray();
            for (int a = 0; a < 29; a++) Assert.Equal(expected[a], indices[r * 29 + a]);
        }
    }

    // ---------------------------------------------------------------- #1134

    private static readonly (float X, float Y)[] BadPoints =
    {
        (float.NaN, 0.1f), (0.2f, float.NaN), (float.PositiveInfinity, 0.1f), (0.1f, float.NegativeInfinity),
        (1e30f, -0.3f), (-1e30f, 0.4f), (3e9f, 0.0f), (0.0f, -3e9f),
    };

    private static readonly (float X, float Y)[] GoodPoints =
    {
        (0.25f, 0.1f), (-0.9f, 0.6f), (1.4f, -1.2f), (0.0f, 0.0f), (-1.0f, 1.0f),
    };

    private static Tensor<float> Input() =>
        new(Enumerable.Range(1, 2 * 3 * 4).Select(v => (float)v).ToArray(), new[] { 1, 2, 3, 4 });

    // Interleaves bad and good points; returns the grid and which positions are bad.
    internal static (Tensor<float> Grid, bool[] Bad) MixedGrid()
    {
        int n = BadPoints.Length + GoodPoints.Length;
        var data = new float[n * 2];
        var bad = new bool[n];
        for (int i = 0, gi = 0, bi = 0; i < n; i++)
        {
            bool takeBad = (i % 2 == 0 && bi < BadPoints.Length) || gi >= GoodPoints.Length;
            var p = takeBad ? BadPoints[bi++] : GoodPoints[gi++];
            bad[i] = takeBad;
            data[2 * i] = p.X;
            data[2 * i + 1] = p.Y;
        }
        return (new Tensor<float>(data, new[] { 1, 1, n, 2 }), bad);
    }

    // The same grid with every bad point replaced by a good one, so good positions sample the same thing.
    internal static Tensor<float> GoodOnly(Tensor<float> grid, bool[] bad)
    {
        var g = (Tensor<float>)grid.Clone();
        for (int i = 0; i < bad.Length; i++)
            if (bad[i]) { g[2 * i] = 0.5f; g[2 * i + 1] = 0.5f; }
        return g;
    }

    public static TheoryData<GridSampleMode, GridSamplePadding, bool> AllModes()
    {
        var data = new TheoryData<GridSampleMode, GridSamplePadding, bool>();
        foreach (GridSampleMode m in Enum.GetValues(typeof(GridSampleMode)))
            foreach (GridSamplePadding p in Enum.GetValues(typeof(GridSamplePadding)))
                foreach (bool a in new[] { false, true })
                    data.Add(m, p, a);
        return data;
    }

    [Theory]
    [MemberData(nameof(AllModes))]
    public void GridSample_NonFinitePoint_SamplesNothing(GridSampleMode mode, GridSamplePadding padding, bool alignCorners)
    {
        var input = Input();
        var (grid, bad) = MixedGrid();
        var output = _cpu.GridSample(input, grid, mode, padding, alignCorners);
        var reference = _cpu.GridSample(input, GoodOnly(grid, bad), mode, padding, alignCorners);
        for (int c = 0; c < 2; c++)
            for (int i = 0; i < bad.Length; i++)
            {
                float got = output[c * bad.Length + i];
                if (bad[i]) Assert.True(got == 0f, $"{mode}/{padding}/align={alignCorners} channel {c} point {i}: {got}, expected 0");
                else Assert.Equal(reference[c * bad.Length + i], got);
            }
    }

    [Fact]
    public void GridSample_TwoArgument_NonFinitePoint_SamplesNothing_AndPassesNoGradient()
    {
        var input = Input();
        var (grid, bad) = MixedGrid();
        var (output, dInput, dGrid) = BilinearWithGradients(_cpu, input, grid);
        var (refOutput, _, _) = BilinearWithGradients(_cpu, input, GoodOnly(grid, bad));
        for (int c = 0; c < 2; c++)
            for (int i = 0; i < bad.Length; i++)
                if (bad[i]) Assert.Equal(0f, output[c * bad.Length + i]);
                else Assert.Equal(refOutput[c * bad.Length + i], output[c * bad.Length + i]);
        for (int i = 0; i < bad.Length; i++)
        {
            Assert.True(Finite(dGrid[2 * i]) && Finite(dGrid[2 * i + 1]), $"grid gradient of point {i} is not finite");
            if (bad[i]) { Assert.Equal(0f, dGrid[2 * i]); Assert.Equal(0f, dGrid[2 * i + 1]); }
        }
        Assert.All(dInput, v => Assert.True(Finite(v)));
        // The input gradient comes from the good points alone: the same as giving the bad points zero output weight.
        var (_, onlyGood, _) = BilinearWithGradients(_cpu, input, grid, weightFor: i => bad[i] ? 0f : 1f);
        for (int k = 0; k < dInput.Length; k++) Assert.Equal(onlyGood[k], dInput[k]);
    }

    private static bool Finite(float v) => !float.IsNaN(v) && !float.IsInfinity(v);

    // Forward through the 2-argument (bilinear, zeros, align_corners=false) GridSample, then the tape's gradients of
    // sum(weight_i * output) with respect to the input and the grid.
    internal static (float[] Output, float[] DInput, float[] DGrid) BilinearWithGradients(
        IEngine engine, Tensor<float> input, Tensor<float> grid, Func<int, float>? weightFor = null)
    {
        int points = grid._shape[2];
        var weights = new Tensor<float>(new[] { 1, 2, 1, points });
        for (int c = 0; c < 2; c++)
            for (int i = 0; i < points; i++) weights[c * points + i] = (weightFor?.Invoke(i) ?? 1f) * (1 + 0.25f * c);
        var x = (Tensor<float>)input.Clone();
        var g = (Tensor<float>)grid.Clone();
        using var tape = new GradientTape<float>();
        var output = engine.GridSample(x, g);
        var loss = engine.ReduceSum(engine.TensorMultiply(output, weights), null);
        var grads = tape.ComputeGradients(loss, new[] { x, g });
        return (output.ToArray(), grads[x].ToArray(), grads[g].ToArray());
    }
}
