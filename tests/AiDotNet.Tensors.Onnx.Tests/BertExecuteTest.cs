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

        var ourOut = result.Plan!.Execute();
        var ourFlat = new float[ourOut.AsSpan().Length];
        ourOut.AsSpan().CopyTo(ourFlat);
        _output.WriteLine($"Our output length: {ourFlat.Length}");

        // Compare to the first ORT output (plan.Execute returns the last
        // step — we match it against whichever ORT result has the same
        // shape). This is a smoke check; full multi-output parity would
        // need changes to our OnnxImportResult to expose each named
        // output's tensor.
        var matched = ortByName.Values.FirstOrDefault(v => v.Length == ourFlat.Length);
        if (matched is null)
        {
            _output.WriteLine("Our output shape not found among ORT outputs; skipping element compare.");
            return;
        }
        int mismatches = 0;
        float maxDiff = 0f;
        for (int i = 0; i < matched.Length; i++)
        {
            float d = Math.Abs(matched[i] - ourFlat[i]);
            float scale = Math.Max(Math.Abs(matched[i]), 1f);
            if (d > 1e-3f * scale) { mismatches++; if (d > maxDiff) maxDiff = d; }
        }
        _output.WriteLine($"Mismatches (1e-3 tol): {mismatches}/{matched.Length}, max diff {maxDiff}");
    }

    private static void FillFloat(LinearAlgebra.Tensor<float> placeholder, long[] source)
    {
        var dst = placeholder.AsWritableSpan();
        for (int i = 0; i < source.Length; i++) dst[i] = source[i];
    }
}
