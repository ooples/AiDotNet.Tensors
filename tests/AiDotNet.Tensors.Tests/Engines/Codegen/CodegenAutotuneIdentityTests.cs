// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Codegen;

[CollectionDefinition(Name, DisableParallelization = true)]
public sealed class CodegenAutotuneCacheCollection
{
    public const string Name = nameof(CodegenAutotuneCacheCollection);
}

[Collection(CodegenAutotuneCacheCollection.Name)]
public class CodegenAutotuneIdentityTests
{
    [Fact]
    public void Cache_DefaultPathHonorsEnvironmentAndFallsBack()
    {
        const string variable = "AIDOTNET_CODEGEN_AUTOTUNE_CACHE";
        string? previous = Environment.GetEnvironmentVariable(variable);
        string configured = Path.Combine(
            Path.GetTempPath(), "configured-codegen-autotune.tsv");

        try
        {
            Environment.SetEnvironmentVariable(variable, configured);
            Assert.Equal(configured, CodegenAutotuneCache.ResolveDefaultCachePath());

            Environment.SetEnvironmentVariable(variable, null);
            Assert.Equal(
                Path.Combine("artifacts", "autotune.tsv"),
                CodegenAutotuneCache.ResolveDefaultCachePath());

            Environment.SetEnvironmentVariable(variable, "   ");
            Assert.Equal(
                Path.Combine("artifacts", "autotune.tsv"),
                CodegenAutotuneCache.ResolveDefaultCachePath());
        }
        finally
        {
            Environment.SetEnvironmentVariable(variable, previous);
        }
    }

    [Fact]
    public void ChunkedSplitFactors_AreOrderedAndShared()
    {
        Assert.Equal(new[] { 2, 4, 7, 14 }, CodegenAutotuneIdentity.ChunkedSplitFactors);
        Assert.True(CodegenAutotuneIdentity.IsChunkedSplitFactor(7));
        Assert.False(CodegenAutotuneIdentity.IsChunkedSplitFactor(8));
    }

    [Fact]
    public void Protocol_RequiresDirectCandidateToClearMeasuredUncertainty()
    {
        Assert.Equal(
            CodegenMeasurementProtocol.AutotuneGainNoiseFloor,
            CodegenMeasurementProtocol.RequiredDirectCandidateGain(0));
        Assert.Equal(
            1.04,
            CodegenMeasurementProtocol.RequiredDirectCandidateGain(0.04),
            precision: 12);
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            CodegenMeasurementProtocol.RequiredDirectCandidateGain(-0.01));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            CodegenMeasurementProtocol.RequiredDirectCandidateGain(double.NaN));
    }

    [Fact]
    public void Protocol_PairedBatchCalibrationEqualizesExposureWithoutShortening()
    {
        // The real parity-transpose shape: preserve the 196 ms baseline batch and
        // lengthen the 23 us candidate from 46 ms to the same exposure window.
        Assert.Equal(
            (A: 2_000, B: 8_522),
            CodegenMeasurementProtocol.CalibratePairIterations(
                2_000, 98.0, 2_000, 23.0));

        // Very short pairs receive the 100 ms floor, while iteration growth is bounded.
        Assert.Equal(
            (A: 10_000, B: 10_000),
            CodegenMeasurementProtocol.CalibratePairIterations(
                100, 10.0, 100, 10.0));
        Assert.Equal(
            (A: 20_000, B: 20_000),
            CodegenMeasurementProtocol.CalibratePairIterations(
                2_000, 0.1, 2_000, 0.1));

        // A long existing batch is never shortened, even when the bounded target is lower.
        Assert.Equal(
            (A: 5, B: 250),
            CodegenMeasurementProtocol.CalibratePairIterations(
                5, 100_000.0, 5, 1_000.0));
        Assert.Equal(
            (A: 50_000, B: 50_000),
            CodegenMeasurementProtocol.CalibratePairIterations(
                50_000, 1.0, 50_000, 1.0));
    }

    [Fact]
    public async Task Identity_IsMemoizedForTheSameInputs_AndIsolatedBySpecAndDevice()
    {
        var first = CodegenKernelSpec.DepthwiseConv2D3x3BiasRelu(1, 32, 16, 16);
        var changed = CodegenKernelSpec.DepthwiseConv2D3x3BiasRelu(1, 64, 16, 16);

        var a = CodegenAutotuneIdentity.Create(first, "gpu-a-sm86-drv1", 8, 6);
        var b = CodegenAutotuneIdentity.Create(first, "gpu-a-sm86-drv1", 8, 6);
        var c = CodegenAutotuneIdentity.Create(changed, "gpu-a-sm86-drv1", 8, 6);
        var d = CodegenAutotuneIdentity.Create(first, "gpu-b-sm86-drv1", 8, 6);
        var e = CodegenAutotuneIdentity.Create(first, "gpu-a-sm86-drv1", 9, 0);
        Task<CodegenAutotuneIdentity>[] concurrent = Enumerable.Range(0, 8)
            .Select(_ => Task.Run(() => CodegenAutotuneIdentity.Create(
                first, "gpu-concurrent-sm86-drv1", 8, 6)))
            .ToArray();
        CodegenAutotuneIdentity[] concurrentResults = await Task.WhenAll(concurrent);

        Assert.Same(a, b);
        Assert.NotSame(a, c);
        Assert.NotSame(a, d);
        Assert.NotSame(a, e);
        Assert.Equal("sm90", e.Target);
        Assert.All(concurrentResults, identity => Assert.Same(concurrentResults[0], identity));
        Assert.Equal(a, b);
        Assert.NotEqual(a.SpecFingerprint, c.SpecFingerprint);
        Assert.Equal("sm86", a.Target);
        Assert.StartsWith("sha256-", a.SpecFingerprint);
        Assert.StartsWith("ptxset-sha256-", a.EmitterFingerprint);
    }

    [Fact]
    public void Cache_RequiresTheExactIdentity_AndRejectsLegacyRows()
    {
        string previous = CodegenAutotuneCache.CachePath;
        string directory = Path.Combine(Path.GetTempPath(), "aidotnet-autotune-" + Guid.NewGuid().ToString("N"));
        string path = Path.Combine(directory, "autotune.tsv");
        string secondPath = Path.Combine(directory, "autotune-second.tsv");
        string lockedPath = Path.Combine(directory, "autotune-locked.tsv");
        Directory.CreateDirectory(directory);

        try
        {
            var spec = CodegenKernelSpec.DepthwiseConv2D3x3BiasRelu(1, 32, 16, 16);
            var identity = CodegenAutotuneIdentity.Create(spec, "gpu-a-sm86-drv1", 8, 6);
            string row = string.Join("\t",
                "depthwise", "no-tile", "10.0", "20.0", "2.0",
                CodegenMeasurementProtocol.Tag,
                identity.DeviceFingerprint, identity.Target,
                identity.SpecFingerprint, identity.EmitterFingerprint, "full");
            string probeRow = row.Replace("depthwise\t", "probe\t");
            probeRow = probeRow.Substring(0, probeRow.Length - "full".Length) + "probe";
            string staleProtocolRow = row
                .Replace("depthwise\t", "stale-protocol\t")
                .Replace("\t" + CodegenMeasurementProtocol.Tag + "\t", "\tp13\t");
            File.WriteAllText(path,
                "kernel\twinner\tbest_us\tmodelled_us\tgain\tprotocol\tdevice\ttarget\tspec\temitter\tscope\n" +
                "legacy\tno-tile\t10.0\t20.0\t2.0\t" + CodegenMeasurementProtocol.Tag + "\n" +
                probeRow + "\n" +
                staleProtocolRow + "\n" +
                row + "\n");
            File.WriteAllText(secondPath,
                "kernel\twinner\tbest_us\tmodelled_us\tgain\tprotocol\tdevice\ttarget\tspec\temitter\tscope\n" +
                row.Replace("\tno-tile\t", "\tlanes4\t") + "\n");

            CodegenAutotuneCache.CachePath = path;
            CodegenAutotuneCache.Invalidate();

            Assert.Equal("no-tile", CodegenAutotuneCache.WinnerFor("depthwise", identity));
            Assert.Null(CodegenAutotuneCache.WinnerFor(
                "depthwise", identity with { Target = "sm90" }));
            Assert.Null(CodegenAutotuneCache.WinnerFor("legacy", identity));
            Assert.Null(CodegenAutotuneCache.WinnerFor("probe", identity));
            Assert.Null(CodegenAutotuneCache.WinnerFor("stale-protocol", identity));

            // Assigning another path must invalidate automatically; requiring callers to
            // remember Invalidate made tests and tools silently serve the previous file.
            CodegenAutotuneCache.CachePath = secondPath;
            Assert.Equal("lanes4", CodegenAutotuneCache.WinnerFor("depthwise", identity));

            File.WriteAllText(lockedPath, row);
            using (var locked = new FileStream(
                       lockedPath, FileMode.Open, FileAccess.Read, FileShare.None))
            {
                CodegenAutotuneCache.CachePath = lockedPath;
                Assert.Null(CodegenAutotuneCache.WinnerFor("depthwise", identity));
            }
        }
        finally
        {
            CodegenAutotuneCache.CachePath = previous;
            CodegenAutotuneCache.Invalidate();
            if (Directory.Exists(directory)) Directory.Delete(directory, recursive: true);
        }
    }
}
