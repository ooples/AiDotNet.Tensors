#if NETCOREAPP
using System;
using System.Globalization;
using System.Linq;
using System.Text;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// The data-loading ops through <see cref="DirectGpuTensorEngine"/> (whichever backend it picks) against the CPU engine:
/// the device path must be taken (no recorded fallback) and the results must equal the CPU's bit for bit.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class DataOpsGpuParityTests : IClassFixture<GpuCpuCorrectnessFixture>
{
    private readonly GpuCpuCorrectnessFixture _fixture;

    public DataOpsGpuParityTests(GpuCpuCorrectnessFixture fixture) => _fixture = fixture;

    private static void AssertBitsEqual(float[] expected, float[] actual, string label)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            if (float.IsNaN(expected[i])) Assert.True(float.IsNaN(actual[i]), $"{label} [{i}]: expected NaN, got {actual[i]:R}");
            else Assert.True(BitConverter.SingleToInt32Bits(expected[i]) == BitConverter.SingleToInt32Bits(actual[i]),
                $"{label} [{i}]: cpu {expected[i]:R}, gpu {actual[i]:R}");
        }
    }

    private static void AssertNoFallback(string op) =>
        Assert.DoesNotContain(GpuLaunchProbe.Fallbacks, f => f.StartsWith(op, StringComparison.Ordinal));

    [SkippableFact]
    public void FromBytes_EveryType_MatchesTheCpu_OnTheDevice()
    {
        Skip.IfNot(_fixture.IsGpuReady, "No GPU backend.");
        var rng = RandomHelper.CreateSeededRandom(31);
        foreach (NumericStorageType type in Enum.GetValues(typeof(NumericStorageType)))
        {
            var bytes = new byte[type.GetByteWidth() * 3001];
            rng.NextBytes(bytes);
            foreach (bool littleEndian in new[] { true, false })
            {
                GpuLaunchProbe.Reset();
                var gpu = _fixture.Gpu.TensorFromBytes<float>(bytes, type, new[] { 3001 }, littleEndian).ToArray();
                AssertNoFallback("TensorFromBytes");
                var cpu = _fixture.Cpu.TensorFromBytes<float>(bytes, type, new[] { 3001 }, littleEndian).ToArray();
                AssertBitsEqual(cpu, gpu, $"{type} le={littleEndian}");
            }
        }
    }

    [SkippableFact]
    public void Parse_Csv_MatchesTheCpu_IncludingFieldsTheHostFinishes()
    {
        Skip.IfNot(_fixture.IsGpuReady, "No GPU backend.");
        var corpus = DataOpsTests.NumericCorpus(12_000, seed: 41);
        var sb = new StringBuilder();
        for (int i = 0; i < corpus.Count; i++)
        {
            sb.Append(corpus[i]);
            sb.Append(i % 6 == 5 ? "\r\n" : ",");   // no field of the corpus holds a comma's meaning; a few hold ',' text
        }
        var text = Encoding.UTF8.GetBytes(sb.ToString());
        var bounds = CpuEngine.SplitFieldBounds(text, (byte)',', out int fields);
        Assert.True(bounds.Length == 2 * fields);
        GpuLaunchProbe.Reset();
        var gpu = _fixture.Gpu.TensorParse<float>(text, (byte)',', new[] { fields }, NumberStyles.Float,
            CultureInfo.InvariantCulture, -7f).ToArray();
        AssertNoFallback("TensorParse");
        var cpu = _fixture.Cpu.TensorParse<float>(text, (byte)',', new[] { fields }, NumberStyles.Float,
            CultureInfo.InvariantCulture, -7f).ToArray();
        AssertBitsEqual(cpu, gpu, "csv");
    }

    [SkippableFact]
    public void Parse_AllDecided_StaysOnTheDevice()
    {
        Skip.IfNot(_fixture.IsGpuReady, "No GPU backend.");
        var rng = RandomHelper.CreateSeededRandom(42);
        var values = Enumerable.Range(0, 5000).Select(_ => ((float)(rng.NextDouble() * 200 - 100)).ToString("R", CultureInfo.InvariantCulture));
        var text = Encoding.UTF8.GetBytes(string.Join("\n", values) + "\n");
        GpuLaunchProbe.Reset();
        var gpu = _fixture.Gpu.TensorParse<float>(text, (byte)',', new[] { 50, 100 }, NumberStyles.Float,
            CultureInfo.InvariantCulture, float.NaN);
        AssertNoFallback("TensorParse");
        Assert.Equal(new[] { 50, 100 }, gpu.Shape.ToArray());
        var cpu = _fixture.Cpu.TensorParse<float>(text, (byte)',', new[] { 50, 100 }, NumberStyles.Float,
            CultureInfo.InvariantCulture, float.NaN);
        AssertBitsEqual(cpu.ToArray(), gpu.ToArray(), "decided");
    }

    [SkippableFact]
    public void RandomUniform_WithTheCallersRandom_MatchesTheCpuAndTheStream()
    {
        Skip.IfNot(_fixture.IsGpuReady, "No GPU backend.");
        var a = RandomHelper.CreateSeededRandom(77);
        var b = RandomHelper.CreateSeededRandom(77);
        var gpu = _fixture.Gpu.TensorRandomUniformRange<float>(a, new[] { 64, 33 }, -0.5, 2.0).ToArray();
        var cpu = _fixture.Cpu.TensorRandomUniformRange<float>(b, new[] { 64, 33 }, -0.5, 2.0).ToArray();
        AssertBitsEqual(cpu, gpu, "uniform");
        Assert.Equal(b.NextDouble(), a.NextDouble());
    }
}
#endif
