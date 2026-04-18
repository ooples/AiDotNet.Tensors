using AiDotNet.Tensors.Engines;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Onnx.Tests;

/// <summary>
/// Diagnostic test: import ResNet-50 and report which ops (if any) are
/// unsupported, and the resulting plan's step count. Doesn't run inference
/// — that's <see cref="EndToEndModelTests.ResNet50_MatchesOnnxRuntime_Across100Samples"/>.
/// Useful when the full end-to-end test crashes or runs too slowly —
/// this quickly tells you whether the problem is at import time or
/// execute time.
/// </summary>
public class ResNet50ImportOnlyTest
{
    private readonly ITestOutputHelper _output;
    public ResNet50ImportOnlyTest(ITestOutputHelper output) { _output = output; }

    [SkippableFact]
    public void ResNet50_ImportSucceeds_ReportsOps()
    {
        var path = Path.Combine(
            Environment.GetEnvironmentVariable("AIDOTNET_ONNX_MODELS")
                ?? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".aidotnet", "onnx-models"),
            "resnet50-v1-7.onnx");
        Skip.IfNot(File.Exists(path), $"Stage resnet50-v1-7.onnx into {Path.GetDirectoryName(path)} to run this test.");

        using var stream = File.OpenRead(path);
        var engine = new CpuEngine();
        var options = new OnnxImportOptions
        {
            DimensionOverrides = new Dictionary<string, int> { ["N"] = 1 },
        };
        var result = OnnxImporter.Import<float>(stream, engine, options);

        _output.WriteLine($"ProducerName: {result.ProducerName}");
        _output.WriteLine($"IrVersion: {result.IrVersion}");
        _output.WriteLine($"NamedInputs:  {string.Join(", ", result.NamedInputs.Keys)}");
        _output.WriteLine($"NamedOutputs: {string.Join(", ", result.NamedOutputs.Keys)}");
        _output.WriteLine($"Unsupported operators: {string.Join(", ", result.UnsupportedOperators)}");
        if (result.Plan is not null)
            _output.WriteLine($"Plan step count: {result.Plan.StepCount}");

        // Importer must at least recognize every op; a non-empty unsupported
        // list is an actionable TODO but not a crash.
        Assert.Empty(result.UnsupportedOperators);
        Assert.NotNull(result.Plan);
    }

    [SkippableFact]
    public void ResNet50_SingleInference_MatchesOnnxRuntime()
    {
        // One-sample variant of ResNet50_MatchesOnnxRuntime_Across100Samples
        // — isolates whether execution works at all before running the full
        // 100-sample loop. ResNet-50 is a large plan (717 steps) so if there's
        // a per-step allocation leak or a memory-intensive op, the single-
        // sample run surfaces it faster.
        var path = Path.Combine(
            Environment.GetEnvironmentVariable("AIDOTNET_ONNX_MODELS")
                ?? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".aidotnet", "onnx-models"),
            "resnet50-v1-7.onnx");
        Skip.IfNot(File.Exists(path) && OnnxRuntimeReference.IsOrtAvailable,
            $"Stage resnet50-v1-7.onnx into {Path.GetDirectoryName(path)} to run this test.");

        var modelBytes = File.ReadAllBytes(path);
        using var stream = new MemoryStream(modelBytes);
        var engine = new CpuEngine();
        var options = new OnnxImportOptions
        {
            DimensionOverrides = new Dictionary<string, int> { ["N"] = 1 },
        };
        var result = OnnxImporter.Import<float>(stream, engine, options);
        Assert.Empty(result.UnsupportedOperators);
        Assert.NotNull(result.Plan);
        _output.WriteLine($"Plan step count: {result.Plan.StepCount}");

        var inputShape = new[] { 1, 3, 224, 224 };
        int total = 1; foreach (var d in inputShape) total *= d;
        var sample = new float[total];
        var rng = new Random(42);
        for (int i = 0; i < total; i++) sample[i] = (float)(rng.NextDouble() * 2 - 1);

        var ortOut = OnnxRuntimeReference.RunSingleOutput(modelBytes,
            ("data", inputShape, sample));
        _output.WriteLine($"ORT output length: {ortOut.Length}");

        sample.AsSpan().CopyTo(result.Inputs["data"].AsWritableSpan());
        var ourOut = result.Plan.Execute();
        var ourFlat = new float[ourOut.AsSpan().Length];
        ourOut.AsSpan().CopyTo(ourFlat);
        _output.WriteLine($"Our output length: {ourFlat.Length}");

        Assert.Equal(ortOut.Length, ourFlat.Length);
        int mismatches = 0;
        float maxDiff = 0f;
        for (int i = 0; i < ortOut.Length; i++)
        {
            float d = Math.Abs(ortOut[i] - ourFlat[i]);
            float scale = Math.Max(Math.Abs(ortOut[i]), 1f);
            if (d > 1e-4f * scale) { mismatches++; if (d > maxDiff) maxDiff = d; }
        }
        _output.WriteLine($"Mismatches: {mismatches}/{ortOut.Length}. Max diff: {maxDiff}");
        Assert.Equal(0, mismatches);
    }
}
