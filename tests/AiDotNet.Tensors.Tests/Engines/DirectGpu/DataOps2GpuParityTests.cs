#if NETCOREAPP
using System;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// The second batch of sweep data ops through <see cref="DirectGpuTensorEngine"/> (whichever backend it picks) against
/// the CPU engine: the device path must be taken (no recorded fallback); the caller's Random must end at the same place;
/// exact ops (thresholds, bytes, casts, rows) must equal the CPU bit for bit, float transcendental and recursive ones
/// (Box-Muller, lfilter) to float rounding.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class DataOps2GpuParityTests : IClassFixture<GpuCpuCorrectnessFixture>
{
    private readonly GpuCpuCorrectnessFixture _fixture;

    public DataOps2GpuParityTests(GpuCpuCorrectnessFixture fixture) => _fixture = fixture;

    private static void AssertNoFallback(string op) =>
        Assert.DoesNotContain(GpuLaunchProbe.Fallbacks, f => f.StartsWith(op, StringComparison.Ordinal));

    private static void AssertBitsEqual(float[] expected, float[] actual, string label)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(BitConverter.SingleToInt32Bits(expected[i]) == BitConverter.SingleToInt32Bits(actual[i]),
                $"{label} [{i}]: cpu {expected[i]:R}, gpu {actual[i]:R}");
    }

    private static void AssertClose(float[] expected, float[] actual, double tolerance, string label)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= tolerance * Math.Max(1.0, Math.Abs(expected[i])),
                $"{label} [{i}]: cpu {expected[i]:R}, gpu {actual[i]:R}");
    }

    private static Tensor<float> HostTensor(int seed, double scale, params int[] shape)
    {
        var rng = RandomHelper.CreateSeededRandom(seed);
        int n = shape.Aggregate(1, (a, b) => a * b);
        return new Tensor<float>(Enumerable.Range(0, n).Select(_ => (float)(scale * (rng.NextDouble() * 2 - 1))).ToArray(), shape);
    }

    [SkippableFact]
    public void BoxMuller_AndTruncatedNormal_MatchTheCpu_AndTheStream()
    {
        Skip.IfNot(_fixture.IsGpuReady, "No GPU backend.");
        var variants = new[]
        {
            new BoxMullerVariant(BoxMullerFirstUniform.OneMinus, 0.0, BoxMullerSecondUniform.Raw, BoxMullerBranch.Cosine),
            new BoxMullerVariant(BoxMullerFirstUniform.Raw, double.Epsilon, BoxMullerSecondUniform.Raw, BoxMullerBranch.Sine),
            new BoxMullerVariant(BoxMullerFirstUniform.RedrawPairWhileZero, 0.0, BoxMullerSecondUniform.OneMinus,
                BoxMullerBranch.Cosine),
        };
        foreach (var variant in variants)
        {
            var a = RandomHelper.CreateSeededRandom(17);
            var b = RandomHelper.CreateSeededRandom(17);
            GpuLaunchProbe.Reset();
            var gpu = _fixture.Gpu.TensorNormalBoxMullerFromRng<float>(a, new[] { 70, 41 }, variant, 0.1, 0.02).ToArray();
            var gpuClamped = _fixture.Gpu.TensorTruncatedNormalFromRng<float>(a, new[] { 999 }, variant, 0.0, 1.0, -1.5, 1.5)
                .ToArray();
            AssertNoFallback("TruncatedNormalFromRng");
            var cpu = _fixture.Cpu.TensorNormalBoxMullerFromRng<float>(b, new[] { 70, 41 }, variant, 0.1, 0.02).ToArray();
            var cpuClamped = _fixture.Cpu.TensorTruncatedNormalFromRng<float>(b, new[] { 999 }, variant, 0.0, 1.0, -1.5, 1.5)
                .ToArray();
            Assert.Equal(b.NextDouble(), a.NextDouble());
            AssertClose(cpu, gpu, 5e-5, $"normal {variant.Branch}");
            AssertClose(cpuClamped, gpuClamped, 5e-5, $"truncated {variant.Branch}");
            Assert.All(gpuClamped, v => Assert.InRange(v, -1.5f, 1.5f));
        }
    }

    [SkippableFact]
    public void BoxMuller_FirstUniformJustBelowOne_KeepsItsRadiusOnTheDevice()
    {
        Skip.IfNot(_fixture.IsGpuReady, "No GPU backend.");
        // Every u1 within 2^-25 of 1 rounds to 1f, so a device log of a float u1 gives r = 0 where the host's
        // sqrt(-2 ln u1) is up to 2.4e-4: 1 - 2^-26 has r = 1.7e-4, 1 - 2^-30 has r = 4.3e-5.
        double[] draws = { 1 - Math.Pow(2, -26), 0.1, 1 - Math.Pow(2, -30), 0.35, 0.5, 0.2, 1 - Math.Pow(2, -40), 0.9 };
        var raw = new BoxMullerVariant(BoxMullerFirstUniform.Raw, 0.0, BoxMullerSecondUniform.Raw, BoxMullerBranch.Cosine);
        GpuLaunchProbe.Reset();
        var gpu = _fixture.Gpu.TensorNormalBoxMullerFromRng<float>(new ReplayRandom(draws), new[] { 4 }, raw).ToArray();
        AssertNoFallback("TruncatedNormalFromRng");
        var cpu = _fixture.Cpu.TensorNormalBoxMullerFromRng<float>(new ReplayRandom(draws), new[] { 4 }, raw).ToArray();
        Assert.True(cpu[0] > 1e-4f, $"the fixture's first radius should be 1.7e-4, got {cpu[0]:R}");
        // Float trig of a float angle on a float radius: a few float ulps of each value.
        AssertClose(cpu, gpu, 2e-6, "u1 near 1");
    }

    [SkippableFact]
    public void Bernoulli_AndRademacher_MatchTheCpuExactly()
    {
        Skip.IfNot(_fixture.IsGpuReady, "No GPU backend.");
        var probabilities = new Tensor<float>(Enumerable.Range(0, 4096).Select(i => (i % 64) / 63f).ToArray(), new[] { 64, 64 });
        var a = RandomHelper.CreateSeededRandom(23);
        var b = RandomHelper.CreateSeededRandom(23);
        GpuLaunchProbe.Reset();
        var gpuPer = _fixture.Gpu.TensorBernoulliFromRng(a, probabilities, UniformDrawComparison.DrawBelowThreshold).ToArray();
        var gpuScalar = _fixture.Gpu.TensorBernoulliFromRng<float>(a, new[] { 5000 }, 0.25,
            UniformDrawComparison.DrawAboveThreshold).ToArray();
        var gpuSigns = _fixture.Gpu.TensorRademacherFromRng<float>(a, new[] { 3000 }, RademacherRule.PositiveWhenDrawAboveHalf)
            .ToArray();
        AssertNoFallback("BernoulliFromRng");
        AssertNoFallback("ThresholdFromRng");
        var cpuPer = _fixture.Cpu.TensorBernoulliFromRng(b, probabilities, UniformDrawComparison.DrawBelowThreshold).ToArray();
        var cpuScalar = _fixture.Cpu.TensorBernoulliFromRng<float>(b, new[] { 5000 }, 0.25,
            UniformDrawComparison.DrawAboveThreshold).ToArray();
        var cpuSigns = _fixture.Cpu.TensorRademacherFromRng<float>(b, new[] { 3000 }, RademacherRule.PositiveWhenDrawAboveHalf)
            .ToArray();
        Assert.Equal(b.NextDouble(), a.NextDouble());
        AssertBitsEqual(cpuPer, gpuPer, "bernoulli per element");
        AssertBitsEqual(cpuScalar, gpuScalar, "bernoulli scalar");
        AssertBitsEqual(cpuSigns, gpuSigns, "rademacher");
    }

    [SkippableFact]
    public void ToBytes_OfADeviceResult_MatchesTheCpu_EveryType()
    {
        Skip.IfNot(_fixture.IsGpuReady, "No GPU backend.");
        var variant = new BoxMullerVariant(BoxMullerFirstUniform.OneMinus, 0.0, BoxMullerSecondUniform.Raw, BoxMullerBranch.Cosine);
        // A device-resident tensor: the GPU draw's result is never downloaded before the encode.
        var resident = _fixture.Gpu.TensorNormalBoxMullerFromRng<float>(RandomHelper.CreateSeededRandom(29), new[] { 3001 },
            variant, 0.0, 300.0);
        var host = new Tensor<float>(resident.ToArray(), new[] { 3001 });
        foreach (NumericStorageType type in Enum.GetValues(typeof(NumericStorageType)))
            foreach (bool littleEndian in new[] { true, false })
            {
                var device = _fixture.Gpu.TensorNormalBoxMullerFromRng<float>(RandomHelper.CreateSeededRandom(29), new[] { 3001 },
                    variant, 0.0, 300.0);
                Assert.True(device.IsGpuResident || device.HasPendingGpuData, "the draw came back on the host");
                GpuLaunchProbe.Reset();
                var gpu = _fixture.Gpu.TensorToBytes(device, type, littleEndian);
                AssertNoFallback("TensorToBytes");
                Assert.Equal(_fixture.Cpu.TensorToBytes(host, type, littleEndian), gpu);
            }
    }

    [SkippableFact]
    public void Cast_OfADeviceTensor_MatchesTheCpu_EveryMode()
    {
        Skip.IfNot(_fixture.IsGpuReady, "No GPU backend.");
        var values = new[] { 0.5f, -0.5f, 1.5f, -2.5f, 2.5f, 3e9f, -3e9f, float.NaN, 255.5f, -0.75f, 127.5f, -128.5f }
            .Concat(Enumerable.Range(0, 2000).Select(i => (float)Math.Sin(i) * 400f)).ToArray();
        foreach (RoundingMode mode in Enum.GetValues(typeof(RoundingMode)))
        {
            var device = _fixture.Gpu.TensorAdd(new Tensor<float>(values, new[] { values.Length }),
                new Tensor<float>(new float[values.Length], new[] { values.Length }));
            GpuLaunchProbe.Reset();
            var gpuInt = _fixture.Gpu.TensorCast<float, int>(device, mode).ToArray();
            var gpuByte = _fixture.Gpu.TensorCast<float, byte>(device, mode).ToArray();
            var gpuShort = _fixture.Gpu.TensorCast<float, short>(device, mode).ToArray();
            AssertNoFallback("TensorCast");
            var host = new Tensor<float>(values, new[] { values.Length });
            Assert.Equal(_fixture.Cpu.TensorCast<float, int>(host, mode).ToArray(), gpuInt);
            Assert.Equal(_fixture.Cpu.TensorCast<float, byte>(host, mode).ToArray(), gpuByte);
            Assert.Equal(_fixture.Cpu.TensorCast<float, short>(host, mode).ToArray(), gpuShort);
        }
    }

    [SkippableFact]
    public void GatherRows_ScatterRows_AndTheGatherGradient_MatchTheCpu()
    {
        Skip.IfNot(_fixture.IsGpuReady, "No GPU backend.");
        var table = HostTensor(31, 5, 64, 24);
        var rng = RandomHelper.CreateSeededRandom(32);
        var ids = new Tensor<long>(Enumerable.Range(0, 120).Select(i => i % 13 == 5 ? 64L + i : i % 17 == 2 ? -1L : rng.Next(64))
            .ToArray(), new[] { 8, 15 });
        GpuLaunchProbe.Reset();
        var gpu = _fixture.Gpu.TensorGatherRows(table, ids);
        AssertNoFallback("GatherRows");
        var cpu = _fixture.Cpu.TensorGatherRows(table, ids);
        Assert.Equal(cpu.Shape.ToArray(), gpu.Shape.ToArray());
        AssertBitsEqual(cpu.ToArray(), gpu.ToArray(), "gather");

        var w = HostTensor(33, 1, 8, 15, 24);
        float[] Grad(IEngine engine)
        {
            var t = new Tensor<float>(table.ToArray(), new[] { 64, 24 });
            using var tape = new AiDotNet.Tensors.Engines.Autodiff.GradientTape<float>();
            var g = engine.TensorGatherRows(t, ids);
            var grads = tape.ComputeGradients(engine.ReduceSum(engine.TensorMultiply(g, w), null), new[] { t });
            return grads[t].ToArray();
        }
        AssertClose(Grad(_fixture.Cpu), Grad(_fixture.Gpu), 1e-5, "gather gradient");

        // Scatter with a repeated id (the last write wins) and out-of-range ids, into equal tables.
        var scatterIds = new Tensor<long>(new[] { 3L, 70L, 3L, 9L, -4L, 63L }, new[] { 6 });
        var source = HostTensor(34, 9, 6, 24);
        var gpuTable = new Tensor<float>(table.ToArray(), new[] { 64, 24 });
        var cpuTable = new Tensor<float>(table.ToArray(), new[] { 64, 24 });
        GpuLaunchProbe.Reset();
        _fixture.Gpu.TensorScatterRowsInPlace(gpuTable, scatterIds, source);
        AssertNoFallback("ScatterRows");
        _fixture.Cpu.TensorScatterRowsInPlace(cpuTable, scatterIds, source);
        AssertBitsEqual(cpuTable.ToArray(), gpuTable.ToArray(), "scatter");
    }

    /// <summary>A Random that replays fixed NextDouble values.</summary>
    private sealed class ReplayRandom : Random
    {
        private readonly double[] _values;
        private int _next;

        // The base seed is irrelevant: every draw the ops make goes through NextDouble below.
        public ReplayRandom(double[] values) : base(0) => _values = values;

        public override double NextDouble() => _values[_next++];

        protected override double Sample() => NextDouble();
    }

    [SkippableFact]
    public void LinearFilter_AndGuarded_MatchTheCpu()
    {
        Skip.IfNot(_fixture.IsGpuReady, "No GPU backend.");
        var x = HostTensor(41, 2, 300, 64);
        var b = new Tensor<float>(new[] { 0.5f, -0.2f, 0.1f }, new[] { 3 });
        var a = new Tensor<float>(new[] { 2.0f, -0.6f, 0.3f, 0.05f }, new[] { 4 });
        var zi = HostTensor(42, 0.5, 300, 3);
        GpuLaunchProbe.Reset();
        var gpu = _fixture.Gpu.TensorLinearFilter(b, a, x, zi).ToArray();
        var gpuGuarded = _fixture.Gpu.TensorLinearFilterGuarded(new Tensor<float>(new[] { 1f }, new[] { 1 }),
            new Tensor<float>(new[] { 1f, -1.5f }, new[] { 2 }), x, null, 4.0).ToArray();
        AssertNoFallback("LinearFilter");
        var cpu = _fixture.Cpu.TensorLinearFilter(b, a, x, zi).ToArray();
        var cpuGuarded = _fixture.Cpu.TensorLinearFilterGuarded(new Tensor<float>(new[] { 1f }, new[] { 1 }),
            new Tensor<float>(new[] { 1f, -1.5f }, new[] { 2 }), x, null, 4.0).ToArray();
        AssertClose(cpu, gpu, 1e-4, "lfilter");
        AssertClose(cpuGuarded, gpuGuarded, 1e-4, "lfilter guarded");
        Assert.All(gpuGuarded, v => Assert.InRange(v, -4f, 4f));
    }
}
#endif
