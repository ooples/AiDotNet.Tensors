using System.Text.Json;
using Google.Protobuf;
using AiDotNet.Tensors.Engines;
using Xunit;
using static AiDotNet.Tensors.Onnx.Tests.OnnxTestHelpers;

namespace AiDotNet.Tensors.Onnx.Tests;

/// <summary>
/// #1574: models PyTorch exported (<c>tools/parity/export_torch_onnx.py</c>) are imported into Tensors, exported back
/// to ONNX, and the re-export run in ONNX Runtime. Torch's own output, the Tensors import and the re-export in ORT must
/// agree. The parity lane writes the models and sets <c>AIDOTNET_TORCH_ONNX_DIR</c>; the test skips when it is unset.
/// </summary>
public class OnnxTorchExportTests
{
    [SkippableTheory]
    [InlineData("resnet18")]
    [InlineData("bert_encoder")]
    public void TorchExport_ImportedAndReexported_MatchesTorchAndRunsInOnnxRuntime(string name)
    {
        var dir = Environment.GetEnvironmentVariable("AIDOTNET_TORCH_ONNX_DIR");
        Skip.If(string.IsNullOrWhiteSpace(dir), "AIDOTNET_TORCH_ONNX_DIR is not set (run tools/parity/export_torch_onnx.py).");
        Skip.IfNot(OnnxRuntimeReference.IsOrtAvailable);

        using var meta = JsonDocument.Parse(File.ReadAllText(Path.Combine(dir!, name + ".json")));
        string inputName = meta.RootElement.GetProperty("input").GetString()!;
        int[] inputShape = meta.RootElement.GetProperty("inputShape").EnumerateArray().Select(e => e.GetInt32()).ToArray();
        float[] input = ReadFloats(Path.Combine(dir!, name + ".input.bin"));
        float[] torch = ReadFloats(Path.Combine(dir!, name + ".torch.bin"));
        byte[] model = File.ReadAllBytes(Path.Combine(dir!, name + ".onnx"));

        using var stream = new MemoryStream(model);
        var imported = OnnxImporter.Import<float>(stream, new CpuEngine());
        Assert.Empty(imported.UnsupportedOperators);
        var feeds = new[] { new RoundTripCases.Feed(inputName, inputShape, input) };
        float[] tensors = RoundTrip.Execute(imported, feeds);
        AssertClose(torch, tensors, 1e-3f);

        byte[] exported = OnnxExporter.Export(imported).ToByteArray();
        float[] ort = OnnxRuntimeReference.RunSingleOutput(exported, new[] { (inputName, inputShape, input) });
        AssertClose(tensors, ort, 1e-3f);
        AssertClose(tensors, RoundTrip.Run(exported, feeds), 1e-4f);
    }

    private static float[] ReadFloats(string path)
    {
        var bytes = File.ReadAllBytes(path);
        var values = new float[bytes.Length / sizeof(float)];
        Buffer.BlockCopy(bytes, 0, values, 0, bytes.Length);
        return values;
    }
}
