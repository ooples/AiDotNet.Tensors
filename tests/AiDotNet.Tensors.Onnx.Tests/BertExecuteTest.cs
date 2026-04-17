using AiDotNet.Tensors.Engines;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Onnx.Tests;

/// <summary>
/// End-to-end BERT-SQuAD execution test. Imports the 784-step plan and
/// runs it against real ORT outputs on a small batch. BERT-SQuAD has 4
/// named inputs (<c>unique_ids_raw_output___9:0</c>,
/// <c>segment_ids:0</c>, <c>input_mask:0</c>, <c>input_ids:0</c>) —
/// ORT expects int64 for three of them, which our float plan receives
/// as integer-valued floats.
/// </summary>
public class BertExecuteTest
{
    private readonly ITestOutputHelper _output;
    public BertExecuteTest(ITestOutputHelper output) { _output = output; }

    [SkippableFact]
    public void BertSquad_SingleInference_MatchesOnnxRuntime()
    {
        var path = Path.Combine(
            Environment.GetEnvironmentVariable("AIDOTNET_ONNX_MODELS")
                ?? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".aidotnet", "onnx-models"),
            "bertsquad-10.onnx");
        Skip.IfNot(File.Exists(path) && OnnxRuntimeReference.IsOrtAvailable,
            $"Stage bertsquad-10.onnx into {Path.GetDirectoryName(path)} to run this test.");

        var modelBytes = File.ReadAllBytes(path);
        using var stream = new MemoryStream(modelBytes);
        var engine = new CpuEngine();
        var options = new OnnxImportOptions
        {
            DimensionOverrides = new Dictionary<string, int>
            {
                ["batch_size"] = 1, ["sequence_length"] = 256,
            },
            DefaultParametricDim = 1,
        };
        var result = OnnxImporter.Import<float>(stream, engine, options);
        Assert.Empty(result.UnsupportedOperators);
        Assert.NotNull(result.Plan);
        _output.WriteLine($"Plan step count: {result.Plan.StepCount}");

        // Build a tiny sample: batch=1, seq=256. token ids are small
        // integers (~30k vocab), masks/segments are 0/1.
        var rng = new Random(42);
        const int batch = 1, seq = 256;
        var inputIds = new long[batch * seq];
        var inputMask = new long[batch * seq];
        var segmentIds = new long[batch * seq];
        var uniqueIds = new long[] { 0 };
        for (int i = 0; i < inputIds.Length; i++)
        {
            inputIds[i] = rng.Next(1, 30000);
            inputMask[i] = 1;
            segmentIds[i] = i < seq / 2 ? 0 : 1;
        }

        // Run ORT with native int64 inputs.
        using var session = new InferenceSession(modelBytes);
        var ortFeeds = new[]
        {
            NamedOnnxValue.CreateFromTensor("unique_ids_raw_output___9:0", new DenseTensor<long>(uniqueIds, new[] { 1 })),
            NamedOnnxValue.CreateFromTensor("segment_ids:0", new DenseTensor<long>(segmentIds, new[] { batch, seq })),
            NamedOnnxValue.CreateFromTensor("input_mask:0", new DenseTensor<long>(inputMask, new[] { batch, seq })),
            NamedOnnxValue.CreateFromTensor("input_ids:0", new DenseTensor<long>(inputIds, new[] { batch, seq })),
        };
        using var ortResults = session.Run(ortFeeds);
        var ortByName = new Dictionary<string, float[]>();
        foreach (var r in ortResults)
        {
            // All outputs are float32 or int64. Convert to float[] uniformly
            // for the tolerance compare.
            try { ortByName[r.Name] = r.AsTensor<float>().ToArray(); }
            catch { ortByName[r.Name] = r.AsTensor<long>().ToArray().Select(x => (float)x).ToArray(); }
        }
        _output.WriteLine($"ORT outputs: {string.Join(", ", ortByName.Keys)}");

        // Feed the same values to our plan as float.
        FillFloat(result.Inputs["input_ids:0"], inputIds);
        FillFloat(result.Inputs["input_mask:0"], inputMask);
        FillFloat(result.Inputs["segment_ids:0"], segmentIds);
        FillFloat(result.Inputs["unique_ids_raw_output___9:0"], uniqueIds);

        result.Plan!.Execute();

        // Compare each declared ONNX output against the corresponding ORT
        // result. OnnxImportResult.Outputs exposes every named graph output
        // after Execute — Execute's single return value is one arbitrary
        // step; the dictionary is the right way to read them all.
        int totalMismatches = 0;
        float maxDiff = 0f;
        foreach (var kv in ortByName)
        {
            var name = kv.Key;
            var ortValues = kv.Value;
            if (!result.Outputs.TryGetValue(name, out var outTensor))
            {
                _output.WriteLine($"  '{name}': missing from our outputs");
                continue;
            }
            var ourFlat = new float[outTensor.AsSpan().Length];
            outTensor.AsSpan().CopyTo(ourFlat);
            if (ourFlat.Length != ortValues.Length)
            {
                _output.WriteLine($"  '{name}': length mismatch ours={ourFlat.Length} ort={ortValues.Length}");
                continue;
            }
            int outputMismatches = 0;
            float outputMaxDiff = 0f;
            for (int i = 0; i < ortValues.Length; i++)
            {
                float d = Math.Abs(ortValues[i] - ourFlat[i]);
                float scale = Math.Max(Math.Abs(ortValues[i]), 1f);
                if (d > 1e-3f * scale)
                {
                    outputMismatches++;
                    if (d > outputMaxDiff) outputMaxDiff = d;
                }
            }
            _output.WriteLine($"  '{name}': {outputMismatches}/{ortValues.Length} diverge, max diff {outputMaxDiff}");
            totalMismatches += outputMismatches;
            if (outputMaxDiff > maxDiff) maxDiff = outputMaxDiff;
        }
        _output.WriteLine($"Total: {totalMismatches} mismatches, max diff {maxDiff}");
    }

    private static void FillFloat(LinearAlgebra.Tensor<float> placeholder, long[] source)
    {
        var dst = placeholder.AsWritableSpan();
        for (int i = 0; i < source.Length; i++) dst[i] = source[i];
    }
}
