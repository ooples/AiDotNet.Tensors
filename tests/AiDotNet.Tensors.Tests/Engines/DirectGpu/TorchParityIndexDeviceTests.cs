using System;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Tests.TestHelpers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// #1112 tier 6: angle, floor_divide, all / any, diagflat, diagonal_scatter, msort and quantile / nanquantile run on
/// the device. Each case must match the CPU element by element, launch device work, and record no host fallback at
/// all (all / any compose device reductions, msort / quantile the device sort). Also pins the device sort's order to
/// torch's: NaN after +inf ascending and first descending, ties by original index, and padding never leaking into a
/// row whose length is not a power of two.
/// </summary>
public abstract class TorchParityIndexDeviceTestsBase
{
    private readonly IDirectGpuTensorEngineFixture _fixture;
    protected TorchParityIndexDeviceTestsBase(IDirectGpuTensorEngineFixture fixture) => _fixture = fixture;

    private DirectGpuTensorEngine GpuOrSkip()
    {
        Skip.If(!_fixture.IsAvailable, "No DirectGpu backend available.");
        var gpu = _fixture.Engine!;
        Skip.IfNot(gpu.GetBackend() is ITorchParityElementwiseBackend, "This backend has no PyTorch-parity device kernels.");
        return gpu;
    }

    private static long TotalFallbacks() => GpuLaunchProbe.Fallbacks.Sum(entry =>
    {
        int x = entry.IndexOf("x ", StringComparison.Ordinal);
        return x > 0 && long.TryParse(entry.Substring(0, x), out long count) ? count : 0L;
    });

    // Small integers (ties), zeros, signed zeros, infinities and NaNs.
    private static Tensor<float> Values(int salt, params int[] shape)
    {
        var rng = new Random(salt);
        var t = new Tensor<float>(shape);
        float[] special = { float.NaN, float.PositiveInfinity, float.NegativeInfinity, 0f, -0f };
        for (int i = 0; i < t.Length; i++)
            t[i] = rng.Next(10) == 0 ? special[rng.Next(special.Length)] : rng.Next(-6, 7) * 0.5f;
        return t;
    }

    private static Tensor<float> Finite(int salt, params int[] shape)
    {
        var rng = new Random(salt);
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = (float)((rng.NextDouble() - 0.5) * 10);
        return t;
    }

    private static void AssertMatches(string op, Tensor<float> cpu, Tensor<float> gpu)
    {
        Assert.Equal(cpu._shape, gpu._shape);
        for (int i = 0; i < cpu.Length; i++)
        {
            float c = cpu[i], g = gpu[i];
            if (float.IsNaN(c) || float.IsNaN(g)) { Assert.True(float.IsNaN(c) && float.IsNaN(g), $"{op}[{i}]: cpu {c} gpu {g}"); continue; }
            if (float.IsInfinity(c) || float.IsInfinity(g)) { Assert.Equal(c, g); continue; }
            Assert.True(Math.Abs(c - g) <= 1e-6 * Math.Max(1.0, Math.Abs(c)), $"{op}[{i}]: cpu {c:R} gpu {g:R}");
        }
    }

    public static TheoryData<string> Cases() => new()
    {
        "angle", "floorDivide", "allAll", "anyAll", "allAxis", "anyAxisKeep", "allZeroFree",
        "diagflat", "diagflatPos", "diagflatNeg", "scatter2d", "scatter3d", "scatter3dNegDims",
        "msort", "msort1d", "msortInf",
        "quantile", "quantileDim", "quantileKeep", "quantileLower", "quantileHigher", "quantileNearest", "quantileNearestTie", "quantileMidpoint",
        "quantileNaN", "nanquantile", "nanquantileAllNaN",
    };

    [SkippableTheory]
    [MemberData(nameof(Cases))]
    public void Op_MatchesCpu_OnDevice(string op)
    {
        var gpu = GpuOrSkip();
        var cpu = new CpuEngine();
        var v = Values(3, 6, 7, 5);
        var w = Values(4, 6, 7, 5);
        var finite = Finite(5, 6, 9, 4);
        var allNaNRow = Finite(6, 5, 8);
        for (int j = 0; j < 8; j++) allNaNRow[2 * 8 + j] = float.NaN;   // row 2 is all NaN
        allNaNRow[0] = float.NaN;                                        // row 0 holds one NaN
        Func<IEngine, Tensor<float>> f = op switch
        {
            "angle" => e => e.TensorAngle(v),
            "floorDivide" => e => e.TensorFloorDivide(v, w),
            "allAll" => e => e.TensorAll(v),
            "anyAll" => e => e.TensorAny(v),
            "allAxis" => e => e.TensorAll(v, new[] { 2 }),
            "anyAxisKeep" => e => e.TensorAny(v, new[] { 0, 2 }, keepDims: true),
            "allZeroFree" => e => e.TensorAll(finite, new[] { 1 }),
            "diagflat" => e => e.TensorDiagflat(Finite(7, 3, 4)),
            "diagflatPos" => e => e.TensorDiagflat(Finite(7, 5), 2),
            "diagflatNeg" => e => e.TensorDiagflat(Finite(7, 6), -3),
            "scatter2d" => e => e.TensorDiagonalScatter(Finite(8, 5, 7), Finite(9, 5), 2),
            "scatter3d" => e => e.TensorDiagonalScatter(Finite(8, 4, 3, 6), Finite(9, 3, 4), 1, 0, 2),
            "scatter3dNegDims" => e => e.TensorDiagonalScatter(Finite(8, 3, 5, 4), Finite(9, 3, 3), -1, -1, -2),
            "msort" => e => e.TensorMsort(v),
            "msort1d" => e => e.TensorMsort(Values(10, 37)),
            "msortInf" => e => e.TensorMsort(InfAndNaNColumns()),
            "quantile" => e => e.TensorQuantile(finite, 0.3),
            "quantileDim" => e => e.TensorQuantile(finite, 0.75, 1),
            "quantileKeep" => e => e.TensorQuantile(finite, 0.5, -1, keepDim: true),
            "quantileLower" => e => e.TensorQuantile(finite, 0.37, 1, interpolation: QuantileInterpolation.Lower),
            "quantileHigher" => e => e.TensorQuantile(finite, 0.37, 1, interpolation: QuantileInterpolation.Higher),
            "quantileNearest" => e => e.TensorQuantile(finite, 0.5, 2, interpolation: QuantileInterpolation.Nearest),
            // position 0.5 exactly: nearest rounds ties to even (index 0), as Math.Round(..., ToEven) does
            "quantileNearestTie" => e => e.TensorQuantile(Finite(12, 5, 2), 0.5, 1, interpolation: QuantileInterpolation.Nearest),
            "quantileMidpoint" => e => e.TensorQuantile(finite, 0.37, 0, interpolation: QuantileInterpolation.Midpoint),
            "quantileNaN" => e => e.TensorQuantile(allNaNRow, 0.4, 1),
            "nanquantile" => e => e.TensorNanQuantile(allNaNRow, 0.4, 1),
            _ => e => e.TensorNanQuantile(allNaNRow, 0.9, 1, keepDim: true),
        };

        var expected = f(cpu);
        long launches = GpuLaunchProbe.Count;
        long fallbacks = TotalFallbacks();
        var actual = f(gpu);
        _ = actual.Length > 0 ? actual[0] : 0f;
        Assert.True(GpuLaunchProbe.Count > launches, $"{op}: no device work launched");
        Assert.Equal(fallbacks, TotalFallbacks());
        AssertMatches(op, expected, actual);
    }

    // Columns mixing +inf, NaN and -inf with lengths that need padding to a power of two.
    private static Tensor<float> InfAndNaNColumns()
    {
        var t = new Tensor<float>(new[] { 11, 3 });
        float[] col0 = { float.PositiveInfinity, 1, float.NaN, -2, float.PositiveInfinity, 0, float.NaN, -0f, float.NegativeInfinity, 3, 1 };
        for (int k = 0; k < 11; k++) { t[k * 3] = col0[k]; t[k * 3 + 1] = col0[10 - k]; t[k * 3 + 2] = k % 2 == 0 ? float.PositiveInfinity : float.NaN; }
        return t;
    }

    // #1131: the CPU sort now uses the same total order as the device sort, so values and indices agree exactly
    // (NaN, +-inf, +-0 and ties included), on the innermost axis and on a middle axis.
    [SkippableTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void Sort_CpuAndGpuAgree(bool descending)
    {
        var gpu = GpuOrSkip();
        var cpu = new CpuEngine();
        foreach (var shape in new[] { new[] { 5, 19 }, new[] { 3, 17, 4 } })
        {
            var x = new Tensor<float>(shape);
            var rng = new Random(23 + shape.Length);
            float[] pool = { float.NaN, float.PositiveInfinity, float.NegativeInfinity, 2f, 2f, -1f, 0f, -0f, 0.5f };
            for (int i = 0; i < x.Length; i++) x[i] = pool[rng.Next(pool.Length)];
            var (gv, gi) = gpu.TensorSort(x, 1, descending);
            var (cv, ci) = cpu.TensorSort(x, 1, descending);
            for (int i = 0; i < x.Length; i++)
            {
                Assert.True(ci[i] == gi[i], $"shape [{string.Join(",", shape)}] {(descending ? "desc" : "asc")} element {i}: cpu index {ci[i]}, gpu index {gi[i]}");
                Assert.Equal(BitConverter.ToInt32(BitConverter.GetBytes(cv[i]), 0), BitConverter.ToInt32(BitConverter.GetBytes(gv[i]), 0));
            }
        }
    }

    // torch.sort order on rows of non-power-of-two length: NaN after +inf ascending, first descending; equal values in
    // original order; every index a real position (padding never leaks into the output).
    [SkippableTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void Sort_OrdersNaNAndInfinityLikeTorch_AndNeverLeaksPadding(bool descending)
    {
        var gpu = GpuOrSkip();
        const int rows = 4, length = 13;
        var x = new Tensor<float>(new[] { rows, length });
        var rng = new Random(11);
        float[] pool = { float.NaN, float.PositiveInfinity, float.NegativeInfinity, 2f, 2f, -1f, 0f };
        for (int i = 0; i < x.Length; i++) x[i] = pool[rng.Next(pool.Length)];
        var (values, indices) = gpu.TensorSort(x, -1, descending);
        for (int r = 0; r < rows; r++)
        {
            var expected = Enumerable.Range(0, length)
                .Select(k => (Value: x[r * length + k], Index: k))
                .OrderBy(p => float.IsNaN(p.Value) ? (descending ? 0 : 2) : 1)
                .ThenBy(p => float.IsNaN(p.Value) ? 0f : descending ? -p.Value : p.Value)
                .ThenBy(p => p.Index)
                .ToArray();
            for (int k = 0; k < length; k++)
            {
                float got = values[r * length + k];
                Assert.True(float.IsNaN(expected[k].Value) ? float.IsNaN(got) : got == expected[k].Value,
                    $"row {r} position {k}: expected {expected[k].Value} got {got}");
                Assert.Equal(expected[k].Index, indices[r * length + k]);
            }
        }
    }
}

/// <summary>The cases above on the machine's default DirectGpu backend.</summary>
[Collection("VulkanGlobalState")]
public class TorchParityIndexDeviceTests : TorchParityIndexDeviceTestsBase, IClassFixture<DirectGpuTensorEngineTestFixture>
{
    public TorchParityIndexDeviceTests(DirectGpuTensorEngineTestFixture fixture) : base(fixture) { }
}

/// <summary>The cases above on the Vulkan backend (#1127).</summary>
[Collection("VulkanGlobalState")]
public class TorchParityIndexVulkanDeviceTests : TorchParityIndexDeviceTestsBase, IClassFixture<VulkanTensorEngineTestFixture>
{
    public TorchParityIndexVulkanDeviceTests(VulkanTensorEngineTestFixture fixture) : base(fixture) { }
}
