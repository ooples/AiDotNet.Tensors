using AiDotNet.Tensors.Engines;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Onnx.Tests;

/// <summary>
/// Fast diagnostic: import BERT-SQuAD and print what operators the plan
/// builds out of. Isolates "did our translator catalog miss an op" from
/// "did inference produce wrong numbers" — the full 100-sample loop
/// takes a long time on BERT-SQuAD (345 MB, ~3000 graph nodes).
/// </summary>
public class BertImportOnlyTest
{
    private readonly ITestOutputHelper _output;
    public BertImportOnlyTest(ITestOutputHelper output) { _output = output; }

    [SkippableFact]
    public void BertSquad_ImportSucceeds_ReportsOps()
    {
        var path = Path.Combine(
            Environment.GetEnvironmentVariable("AIDOTNET_ONNX_MODELS")
                ?? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".aidotnet", "onnx-models"),
            "bertsquad-10.onnx");
        Skip.IfNot(File.Exists(path),
            $"Stage bertsquad-10.onnx into {Path.GetDirectoryName(path)} to run this test.");

        using var stream = File.OpenRead(path);
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

        _output.WriteLine($"ProducerName: {result.ProducerName}");
        _output.WriteLine($"IrVersion: {result.IrVersion}");
        _output.WriteLine($"NamedInputs:  {string.Join(", ", result.NamedInputs.Keys)}");
        _output.WriteLine($"NamedOutputs: {string.Join(", ", result.NamedOutputs.Keys)}");
        _output.WriteLine($"Unsupported operators ({result.UnsupportedOperators.Count}):");
        foreach (var op in result.UnsupportedOperators)
            _output.WriteLine($"  - {op}");
        if (result.Plan is not null)
            _output.WriteLine($"Plan step count: {result.Plan.StepCount}");
    }
}
