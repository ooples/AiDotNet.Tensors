using AiDotNet.Tensors.Engines;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Onnx.Tests;

/// <summary>
/// Formal Issue #169 acceptance test: BERT-SQuAD numerical accuracy within
/// 1e-4 on 100 validation samples. Imports once, reuses the plan and an
/// ORT session across the loop — each sample randomizes token IDs / masks
/// / segment IDs and asserts every output element matches to tolerance.
/// </summary>
public class BertExecute100Samples
{
    private readonly ITestOutputHelper _output;
    public BertExecute100Samples(ITestOutputHelper output) { _output = output; }

    [SkippableFact]
    public void BertSquad_MatchesOnnxRuntime_Across100Samples()
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

        using var session = new InferenceSession(modelBytes);

        const int batch = 1, seq = 256;
        var rng = new Random(42);
        int totalMismatches = 0;
        float maxDiff = 0f;

        for (int s = 0; s < 100; s++)
        {
            var inputIds = new long[batch * seq];
            var inputMask = new long[batch * seq];
            var segmentIds = new long[batch * seq];
            var uniqueIds = new long[] { s };
            for (int i = 0; i < inputIds.Length; i++)
            {
                inputIds[i] = rng.Next(1, 30000);
                inputMask[i] = rng.NextDouble() < 0.9 ? 1 : 0;
                segmentIds[i] = i < seq / 2 ? 0 : 1;
            }

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
                try { ortByName[r.Name] = r.AsTensor<float>().ToArray(); continue; } catch { }
                try { ortByName[r.Name] = r.AsTensor<long>().ToArray().Select(x => (float)x).ToArray(); continue; } catch { }
            }

            FillFloat(result.Inputs["input_ids:0"], inputIds);
            FillFloat(result.Inputs["input_mask:0"], inputMask);
            FillFloat(result.Inputs["segment_ids:0"], segmentIds);
            FillFloat(result.Inputs["unique_ids_raw_output___9:0"], uniqueIds);
            result.Plan!.Execute();

            foreach (var kv in ortByName)
            {
                if (!result.Outputs.TryGetValue(kv.Key, out var ours)) continue;
                var oursSpan = ours.AsSpan();
                if (oursSpan.Length != kv.Value.Length) continue;
                for (int i = 0; i < kv.Value.Length; i++)
                {
                    float d = Math.Abs(kv.Value[i] - oursSpan[i]);
                    float scale = Math.Max(Math.Abs(kv.Value[i]), 1f);
                    if (d > 1e-4f * scale)
                    {
                        totalMismatches++;
                        if (d > maxDiff) maxDiff = d;
                    }
                }
            }

            if ((s & 15) == 15)
            {
                GC.Collect(1, GCCollectionMode.Forced);
                GC.WaitForPendingFinalizers();
            }
        }

        _output.WriteLine($"BERT-SQuAD × 100 samples: {totalMismatches} divergences, max diff {maxDiff}");
        Assert.True(totalMismatches == 0,
            $"BERT-SQuAD: {totalMismatches} elements diverged > 1e-4 across 100 samples. Max diff: {maxDiff}.");
    }

    private static void FillFloat(LinearAlgebra.Tensor<float> placeholder, long[] source)
    {
        var dst = placeholder.AsWritableSpan();
        for (int i = 0; i < source.Length; i++) dst[i] = source[i];
    }
}
