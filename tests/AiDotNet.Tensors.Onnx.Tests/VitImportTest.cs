using AiDotNet.Tensors.Engines;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Onnx.Tests;

/// <summary>
/// Acceptance test for ViT-B/16 (Issue #169 Phase 1). Uses the Xenova
/// HuggingFace export of google/vit-base-patch16-224-in21k (343 MB). Runs
/// single-sample parity against ONNX Runtime.
/// </summary>
public class VitImportTest
{
    private readonly ITestOutputHelper _output;
    public VitImportTest(ITestOutputHelper output) { _output = output; }

    [SkippableFact]
    public void VitBase_ImportSucceeds_ReportsOps()
    {
        var path = Path.Combine(
            Environment.GetEnvironmentVariable("AIDOTNET_ONNX_MODELS")
                ?? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".aidotnet", "onnx-models"),
            "vit-base.onnx");
        Skip.IfNot(File.Exists(path),
            $"Stage vit-base.onnx into {Path.GetDirectoryName(path)}.");

        using var stream = File.OpenRead(path);
        var engine = new CpuEngine();
        var options = new OnnxImportOptions
        {
            DimensionOverrides = new Dictionary<string, int>
            {
                ["batch_size"] = 1, ["batch"] = 1,
                ["num_channels"] = 3, ["height"] = 224, ["width"] = 224,
            },
            DefaultParametricDim = 1,
        };
        var result = OnnxImporter.Import<float>(stream, engine, options);

        _output.WriteLine($"ProducerName: {result.ProducerName}");
        _output.WriteLine($"IrVersion: {result.IrVersion}");
        _output.WriteLine($"NamedInputs:  {string.Join(", ", result.NamedInputs.Keys)}");
        _output.WriteLine($"NamedOutputs: {string.Join(", ", result.NamedOutputs.Keys)}");
        _output.WriteLine($"Unsupported operators ({result.UnsupportedOperators.Count}):");
        foreach (var op in result.UnsupportedOperators) _output.WriteLine($"  - {op}");
        if (result.Plan is not null)
            _output.WriteLine($"Plan step count: {result.Plan.StepCount}");
        // Import-success contract for this acceptance gate: no unsupported
        // operators AND a non-null executable plan. Without these assertions
        // the test would silently pass on importer regressions.
        Assert.Empty(result.UnsupportedOperators);
        Assert.NotNull(result.Plan);
    }

    [SkippableFact]
    public void VitBase_SingleInference_MatchesOnnxRuntime()
    {
        var path = Path.Combine(
            Environment.GetEnvironmentVariable("AIDOTNET_ONNX_MODELS")
                ?? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".aidotnet", "onnx-models"),
            "vit-base.onnx");
        Skip.IfNot(File.Exists(path) && OnnxRuntimeReference.IsOrtAvailable,
            $"Stage vit-base.onnx into {Path.GetDirectoryName(path)}.");

        var modelBytes = File.ReadAllBytes(path);
        using var stream = new MemoryStream(modelBytes);
        var engine = new CpuEngine();
        var options = new OnnxImportOptions
        {
            DimensionOverrides = new Dictionary<string, int>
            {
                ["batch_size"] = 1, ["batch"] = 1,
                ["num_channels"] = 3, ["height"] = 224, ["width"] = 224,
            },
            DefaultParametricDim = 1,
        };
        var result = OnnxImporter.Import<float>(stream, engine, options);
        Assert.Empty(result.UnsupportedOperators);
        Assert.NotNull(result.Plan);
        _output.WriteLine($"Plan step count: {result.Plan.StepCount}");

        var inputName = result.NamedInputs.Keys.First();
        var inputShape = result.NamedInputs[inputName].ToShapeArray();
        int total = 1; foreach (var d in inputShape) total *= d;
        var sample = new float[total];
        var rng = new Random(42);
        for (int i = 0; i < total; i++) sample[i] = (float)(rng.NextDouble() * 2 - 1);

        using var session = new InferenceSession(modelBytes);
        var ortFeeds = new[]
        {
            NamedOnnxValue.CreateFromTensor(inputName, new DenseTensor<float>(sample, inputShape)),
        };
        using var ortResults = session.Run(ortFeeds);
        var ortByName = new Dictionary<string, float[]>();
        foreach (var r in ortResults)
            ortByName[r.Name] = r.AsTensor<float>().ToArray();

        sample.AsSpan().CopyTo(result.Inputs[inputName].AsWritableSpan());
        result.Plan!.Execute();

        int totalMismatches = 0;
        float maxDiff = 0f;
        foreach (var kv in ortByName)
        {
            Assert.True(result.Outputs.TryGetValue(kv.Key, out var ours),
                $"AiDotNet plan did not produce output '{kv.Key}' that ORT did.");
            var oursSpan = ours!.AsSpan();
            Assert.True(oursSpan.Length == kv.Value.Length,
                $"Output '{kv.Key}' length mismatch: AiDotNet {oursSpan.Length}, ORT {kv.Value.Length}.");
            for (int i = 0; i < kv.Value.Length; i++)
            {
                float d = Math.Abs(kv.Value[i] - oursSpan[i]);
                float scale = Math.Max(Math.Abs(kv.Value[i]), 1f);
                if (d > 1e-4f * scale) { totalMismatches++; if (d > maxDiff) maxDiff = d; }
            }
            _output.WriteLine($"  '{kv.Key}': len={kv.Value.Length}, divergent so far={totalMismatches}, max diff {maxDiff}");
        }
        _output.WriteLine($"ViT-B/16 single inference: {totalMismatches} divergences, max diff {maxDiff}");
        Assert.True(totalMismatches == 0,
            $"ViT: {totalMismatches} elements diverged > 1e-4. Max diff: {maxDiff}.");
    }
}
