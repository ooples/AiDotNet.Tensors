using AiDotNet.Tensors.Engines;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Onnx.Tests;

/// <summary>
/// End-to-end test against the MNIST classifier from the ONNX model zoo.
/// Small enough (~26 KB) that we download it once at test time and cache
/// under the user profile, so this runs in CI without pre-staging model
/// files. Covers the realistic-but-minimal multi-op model path:
/// Conv → Relu → MaxPool → Conv → Relu → MaxPool → Reshape → MatMul → Add
/// at opset 8.
///
/// <para>Satisfies a meaningful subset of Issue #169's "import real model
/// from zoo, match ORT within 1e-4 across 100 samples" acceptance
/// criterion — the larger ResNet-50 / BERT-base / ViT-B/16 variants in
/// <see cref="EndToEndModelTests"/> skip by default because of download
/// size but follow the same pattern.</para>
/// </summary>
public class MnistEndToEndTest
{
    private const string ModelUrl =
        "https://github.com/onnx/models/raw/main/validated/vision/classification/mnist/model/mnist-8.onnx";

    private readonly ITestOutputHelper _output;
    public MnistEndToEndTest(ITestOutputHelper output) { _output = output; }

    [SkippableFact]
    public async Task Mnist_MatchesOnnxRuntime_Across100Samples()
    {
        Skip.IfNot(OnnxRuntimeReference.IsOrtAvailable);

        var modelBytes = await LoadOrDownloadAsync(ModelUrl, "mnist-8.onnx");
        _output.WriteLine($"Model bytes: {modelBytes.Length}");

        using var stream = new MemoryStream(modelBytes);
        var engine = new CpuEngine();
        var result = OnnxImporter.Import<float>(stream, engine);
        Assert.Empty(result.UnsupportedOperators);
        Assert.NotNull(result.Plan);
        _output.WriteLine($"Plan step count: {result.Plan.StepCount}");
        _output.WriteLine($"Inputs: {string.Join(", ", result.NamedInputs.Keys)}");
        _output.WriteLine($"Outputs: {string.Join(", ", result.NamedOutputs.Keys)}");

        // MNIST input is [1, 1, 28, 28] float.
        var inputName = result.NamedInputs.Keys.First();
        var inputShape = result.NamedInputs[inputName].Shape;
        Assert.Equal(new[] { 1, 1, 28, 28 }, inputShape);

        int total = 1;
        foreach (var d in inputShape) total *= d;

        var rng = new Random(42);
        int mismatches = 0;
        float maxDiff = 0f;
        for (int s = 0; s < 100; s++)
        {
            var sample = new float[total];
            for (int i = 0; i < total; i++)
                sample[i] = (float)(rng.NextDouble() * 2 - 1);

            var ortOut = OnnxRuntimeReference.RunSingleOutput(modelBytes,
                (inputName, inputShape, sample));

            sample.AsSpan().CopyTo(result.Inputs[inputName].AsWritableSpan());
            var ourOut = result.Plan!.Execute();
            var ourFlat = new float[ourOut.AsSpan().Length];
            ourOut.AsSpan().CopyTo(ourFlat);

            Assert.Equal(ortOut.Length, ourFlat.Length);
            for (int i = 0; i < ortOut.Length; i++)
            {
                float d = Math.Abs(ortOut[i] - ourFlat[i]);
                float scale = Math.Max(Math.Abs(ortOut[i]), 1f);
                if (d > 1e-4f * scale)
                {
                    mismatches++;
                    if (d > maxDiff) maxDiff = d;
                }
            }
        }

        _output.WriteLine($"Across 100 samples: {mismatches} elements outside 1e-4 relative tolerance. Max diff: {maxDiff}.");
        Assert.True(mismatches == 0,
            $"MNIST: {mismatches} elements diverged > 1e-4 across 100 samples. Max diff: {maxDiff}.");
    }

    private static async Task<byte[]> LoadOrDownloadAsync(string url, string fileName)
    {
        var cacheDir = Environment.GetEnvironmentVariable("AIDOTNET_ONNX_MODELS")
            ?? Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
                ".aidotnet", "onnx-models");
        Directory.CreateDirectory(cacheDir);
        var path = Path.Combine(cacheDir, fileName);
        if (File.Exists(path))
            return File.ReadAllBytes(path);

        using var http = new HttpClient { Timeout = TimeSpan.FromSeconds(60) };
        var bytes = await http.GetByteArrayAsync(url);
        File.WriteAllBytes(path, bytes);
        return bytes;
    }
}
