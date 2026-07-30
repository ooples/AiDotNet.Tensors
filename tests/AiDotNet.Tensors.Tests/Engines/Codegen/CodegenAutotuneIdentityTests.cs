// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.IO;
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
    public void ChunkedSplitFactors_AreOrderedAndShared()
    {
        Assert.Equal(new[] { 2, 4, 7, 14 }, CodegenAutotuneIdentity.ChunkedSplitFactors);
        Assert.True(CodegenAutotuneIdentity.IsChunkedSplitFactor(7));
        Assert.False(CodegenAutotuneIdentity.IsChunkedSplitFactor(8));
    }

    [Fact]
    public void Identity_IsStableForTheSameInputs_AndChangesWithTheSpec()
    {
        var first = CodegenKernelSpec.DepthwiseConv2D3x3BiasRelu(1, 32, 16, 16);
        var changed = CodegenKernelSpec.DepthwiseConv2D3x3BiasRelu(1, 64, 16, 16);

        var a = CodegenAutotuneIdentity.Create(first, "gpu-a-sm86-drv1", 8, 6);
        var b = CodegenAutotuneIdentity.Create(first, "gpu-a-sm86-drv1", 8, 6);
        var c = CodegenAutotuneIdentity.Create(changed, "gpu-a-sm86-drv1", 8, 6);

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
                identity.SpecFingerprint, identity.EmitterFingerprint);
            File.WriteAllText(path,
                "kernel\twinner\tbest_us\tmodelled_us\tgain\tprotocol\tdevice\ttarget\tspec\temitter\n" +
                "legacy\tno-tile\t10.0\t20.0\t2.0\t" + CodegenMeasurementProtocol.Tag + "\n" +
                row + "\n");
            File.WriteAllText(secondPath,
                "kernel\twinner\tbest_us\tmodelled_us\tgain\tprotocol\tdevice\ttarget\tspec\temitter\n" +
                row.Replace("\tno-tile\t", "\tlanes4\t") + "\n");

            CodegenAutotuneCache.CachePath = path;
            CodegenAutotuneCache.Invalidate();

            Assert.Equal("no-tile", CodegenAutotuneCache.WinnerFor("depthwise", identity));
            Assert.Null(CodegenAutotuneCache.WinnerFor(
                "depthwise", identity with { Target = "sm90" }));
            Assert.Null(CodegenAutotuneCache.WinnerFor("legacy", identity));

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
