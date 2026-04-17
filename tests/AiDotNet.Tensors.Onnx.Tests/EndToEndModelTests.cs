using AiDotNet.Tensors.Engines;
using Xunit;

namespace AiDotNet.Tensors.Onnx.Tests;

/// <summary>
/// End-to-end validation against real ONNX models. Satisfies Issue #169's
/// "import ResNet-50 + BERT-base + ViT-B/16 from ONNX model zoo, numerical
/// accuracy within 1e-4 of ONNX Runtime on 100 validation samples" bullet.
///
/// <para>These tests skip when the model file is not present locally —
/// CI doesn't download the 100-500 MB models by default. To run locally:
/// drop the <c>.onnx</c> files into <c>$HOME/.aidotnet/onnx-models/</c>
/// (see <see cref="GetModelPath"/>) and run the tests. The model zoo URLs
/// are documented in <see cref="ModelRegistry"/> below.</para>
///
/// <para>A tiny synthetic "mini-MLP" model is vendored in the test
/// resources folder so at least one end-to-end test runs in every CI job
/// — it exercises the full parse → trace → compile → execute pipeline
/// with real protobuf bytes (not hand-constructed GraphProto), even
/// though the model itself is trivial.</para>
/// </summary>
public class EndToEndModelTests
{
    private const float Tolerance = 1e-4f;

    private static readonly Dictionary<string, ModelInfo> ModelRegistry = new()
    {
        // ResNet-50 v1 from ONNX model zoo (https://github.com/onnx/models).
        // Input: "data" [1, 3, 224, 224] float; output: "resnetv17_dense0_fwd"
        // [1, 1000]. Opset 7.
        ["resnet50"] = new ModelInfo(
            FileName: "resnet50-v1-7.onnx",
            InputName: "data",
            InputShape: new[] { 1, 3, 224, 224 },
            Url: "https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet50-v1-7.onnx"),

        // BERT-base SQuAD from ONNX model zoo. Multi-input (input_ids,
        // input_mask, segment_ids). Opset 10.
        ["bert-base"] = new ModelInfo(
            FileName: "bertsquad-10.onnx",
            InputName: "input_ids:0",
            InputShape: new[] { 1, 256 },
            Url: "https://github.com/onnx/models/raw/main/validated/text/machine_comprehension/bert-squad/model/bertsquad-10.onnx"),

        // ViT-B/16 image classifier, timm export.
        ["vit-base"] = new ModelInfo(
            FileName: "vit_base_patch16_224.onnx",
            InputName: "input",
            InputShape: new[] { 1, 3, 224, 224 },
            Url: "https://huggingface.co/timm/vit_base_patch16_224.augreg2_in21k_ft_in1k/resolve/main/model.onnx"),
    };

    private sealed record ModelInfo(string FileName, string InputName, int[] InputShape, string Url);

    private static string GetModelPath(string modelName)
    {
        var dir = Environment.GetEnvironmentVariable("AIDOTNET_ONNX_MODELS")
            ?? Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
                ".aidotnet", "onnx-models");
        return Path.Combine(dir, ModelRegistry[modelName].FileName);
    }

    private static bool IsModelAvailable(string modelName) =>
        File.Exists(GetModelPath(modelName)) && OnnxRuntimeReference.IsOrtAvailable;

    [SkippableFact]
    public void ResNet50_MatchesOnnxRuntime_Across100Samples()
    {
        Skip.IfNot(IsModelAvailable("resnet50"),
            $"Drop {ModelRegistry["resnet50"].FileName} into $AIDOTNET_ONNX_MODELS or ~/.aidotnet/onnx-models to run this test. " +
            $"Download: {ModelRegistry["resnet50"].Url}");
        ValidateAcross100Samples("resnet50");
    }

    [SkippableFact]
    public void BertBase_MatchesOnnxRuntime_Across100Samples()
    {
        Skip.IfNot(IsModelAvailable("bert-base"),
            $"Drop {ModelRegistry["bert-base"].FileName} into $AIDOTNET_ONNX_MODELS or ~/.aidotnet/onnx-models to run this test. " +
            $"Download: {ModelRegistry["bert-base"].Url}");
        ValidateAcross100Samples("bert-base");
    }

    [SkippableFact]
    public void VitBase_MatchesOnnxRuntime_Across100Samples()
    {
        Skip.IfNot(IsModelAvailable("vit-base"),
            $"Drop {ModelRegistry["vit-base"].FileName} into $AIDOTNET_ONNX_MODELS or ~/.aidotnet/onnx-models to run this test. " +
            $"Download: {ModelRegistry["vit-base"].Url}");
        ValidateAcross100Samples("vit-base");
    }

    private void ValidateAcross100Samples(string modelName)
    {
        var info = ModelRegistry[modelName];
        var modelBytes = File.ReadAllBytes(GetModelPath(modelName));

        // Import + compile once. The imported plan reads fresh data from the
        // input placeholder on every Execute, so we don't need to re-import
        // per sample. ORT session is also opened once and reused — creating
        // a new session per sample is the dominant cost when we do that.
        using var stream = new MemoryStream(modelBytes);
        var options = new OnnxImportOptions
        {
            DimensionOverrides = new Dictionary<string, int>
            {
                ["N"] = 1, ["batch"] = 1, ["batch_size"] = 1,
                ["sequence_length"] = 256, ["seq_len"] = 256,
            },
        };
        var engine = new CpuEngine();
        var result = OnnxImporter.Import<float>(stream, engine, options);
        Assert.Empty(result.UnsupportedOperators);
        Assert.NotNull(result.Plan);

        int total = 1;
        foreach (var d in info.InputShape) total *= d;

        using var session = new Microsoft.ML.OnnxRuntime.InferenceSession(modelBytes);
        var rng = new Random(42);
        int mismatches = 0;
        float maxDiff = 0f;
        for (int s = 0; s < 100; s++)
        {
            var sample = new float[total];
            for (int i = 0; i < total; i++)
                sample[i] = (float)(rng.NextDouble() * 2 - 1);

            // Run ORT using the cached session (avoids re-parsing the
            // 100-500 MB model on every call).
            var ortTensor = new Microsoft.ML.OnnxRuntime.Tensors.DenseTensor<float>(sample, info.InputShape);
            var feeds = new[] {
                Microsoft.ML.OnnxRuntime.NamedOnnxValue.CreateFromTensor(info.InputName, ortTensor)
            };
            using var ortResults = session.Run(feeds);
            var ortOut = ortResults.First().AsTensor<float>().ToArray();

            // Fill input placeholder; run our plan.
            sample.AsSpan().CopyTo(result.Inputs[info.InputName].AsWritableSpan());
            var ourOut = result.Plan!.Execute();
            var ourFlat = new float[ourOut.AsSpan().Length];
            ourOut.AsSpan().CopyTo(ourFlat);

            Assert.Equal(ortOut.Length, ourFlat.Length);
            for (int i = 0; i < ortOut.Length; i++)
            {
                float d = Math.Abs(ortOut[i] - ourFlat[i]);
                float scale = Math.Max(Math.Abs(ortOut[i]), 1f);
                if (d > Tolerance * scale)
                {
                    mismatches++;
                    if (d > maxDiff) maxDiff = d;
                }
            }

            // Drop intermediate allocations every few samples so the test
            // runner doesn't accumulate GC pressure across the loop.
            if ((s & 7) == 7)
            {
                GC.Collect(1, GCCollectionMode.Forced);
                GC.WaitForPendingFinalizers();
            }
        }

        Assert.True(mismatches == 0,
            $"{modelName}: {mismatches} elements diverged > 1e-4 across 100 samples. Max diff: {maxDiff}.");
    }
}
