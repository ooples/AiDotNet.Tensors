using AiDotNet.Tensors.Onnx.Protos;
using Google.Protobuf;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Onnx.Tests;

/// <summary>
/// Prints every node in BERT-SQuAD that transitively depends on
/// input_mask:0 within the first N forward edges, so we know the exact
/// op chain that transforms the mask before it reaches attention.
/// </summary>
public class BertMaskPath
{
    private readonly ITestOutputHelper _output;
    public BertMaskPath(ITestOutputHelper output) { _output = output; }

    [SkippableFact]
    public void PrintMaskPath()
    {
        var path = Path.Combine(
            Environment.GetEnvironmentVariable("AIDOTNET_ONNX_MODELS")
                ?? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".aidotnet", "onnx-models"),
            "bertsquad-10.onnx");
        Skip.IfNot(File.Exists(path), $"Need {path}");

        var bytes = File.ReadAllBytes(path);
        var model = ModelProto.Parser.ParseFrom(bytes);
        var graph = model.Graph!;

        var reached = new HashSet<string>(StringComparer.Ordinal) { "input_mask:0" };
        var nodes = new List<NodeProto>(graph.Node);

        // 10 BFS hops is more than enough to cover mask-prep chain.
        for (int hop = 0; hop < 15; hop++)
        {
            bool grew = false;
            foreach (var n in nodes)
            {
                if (n.Input.Any(i => reached.Contains(i)))
                {
                    foreach (var o in n.Output)
                        if (!string.IsNullOrEmpty(o) && reached.Add(o)) grew = true;
                }
            }
            if (!grew) break;
        }

        _output.WriteLine($"Nodes transitively dependent on input_mask:0:");
        int count = 0;
        foreach (var n in nodes)
        {
            if (n.Input.Any(i => reached.Contains(i)))
            {
                count++;
                if (count > 60) { _output.WriteLine("  ... (truncated)"); break; }
                _output.WriteLine($"  [{n.OpType}] inputs=({string.Join(", ", n.Input)}) -> outputs=({string.Join(", ", n.Output)})");
            }
        }
        _output.WriteLine($"Total dependent nodes: {count}");
    }
}
