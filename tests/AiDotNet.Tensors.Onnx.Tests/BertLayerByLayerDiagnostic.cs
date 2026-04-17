using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Onnx.Protos;
using Google.Protobuf;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Onnx.Tests;

/// <summary>
/// Layer-by-layer diagnostic: takes BERT-SQuAD, promotes EVERY node's
/// output to a graph output, runs both ORT and our importer on the
/// modified model, and reports the FIRST node whose output diverges —
/// pinpointing the specific translator / engine-op bug rather than
/// guessing from final-output comparisons.
/// </summary>
public class BertLayerByLayerDiagnostic
{
    private readonly ITestOutputHelper _output;
    public BertLayerByLayerDiagnostic(ITestOutputHelper output) { _output = output; }

    [SkippableFact]
    public void BertSquad_FirstDivergentNode_IsReported()
    {
        var path = Path.Combine(
            Environment.GetEnvironmentVariable("AIDOTNET_ONNX_MODELS")
                ?? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".aidotnet", "onnx-models"),
            "bertsquad-10.onnx");
        Skip.IfNot(File.Exists(path) && OnnxRuntimeReference.IsOrtAvailable,
            $"Stage bertsquad-10.onnx into {Path.GetDirectoryName(path)} to run this test.");

        var originalBytes = File.ReadAllBytes(path);
        var model = ModelProto.Parser.ParseFrom(originalBytes);
        var graph = model.Graph!;

        // Snapshot the names originally declared as graph outputs — we'll
        // keep them first in the topological comparison.
        var originalOutputs = new HashSet<string>(graph.Output.Select(o => o.Name), StringComparer.Ordinal);

        // Promote every internal tensor to a graph output so ORT returns it
        // at Run time. The model becomes enormous output-wise but otherwise
        // unchanged. Skip outputs that are already declared or empty.
        var graphInputs = new HashSet<string>(graph.Input.Select(i => i.Name), StringComparer.Ordinal);
        var promoted = new List<string>();
        foreach (var node in graph.Node)
        {
            foreach (var outName in node.Output)
            {
                if (string.IsNullOrEmpty(outName)) continue;
                if (originalOutputs.Contains(outName)) continue;
                // ORT rejects duplicate output names; once is enough.
                if (promoted.Contains(outName)) continue;
                promoted.Add(outName);
                var vi = new ValueInfoProto { Name = outName };
                // We don't know the exact shape/type statically; ORT only
                // needs the name to emit the tensor.
                graph.Output.Add(vi);
            }
        }
        _output.WriteLine($"Promoted {promoted.Count} intermediate tensors to graph outputs.");

        var modifiedBytes = model.ToByteArray();

        // Inputs: same as BertExecuteTest — batch=1, seq=256.
        const int batch = 1, seq = 256;
        var rng = new Random(42);
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
        using var session = new InferenceSession(modifiedBytes);
        var ortFeeds = new[]
        {
            NamedOnnxValue.CreateFromTensor("unique_ids_raw_output___9:0", new DenseTensor<long>(uniqueIds, new[] { 1 })),
            NamedOnnxValue.CreateFromTensor("segment_ids:0", new DenseTensor<long>(segmentIds, new[] { batch, seq })),
            NamedOnnxValue.CreateFromTensor("input_mask:0", new DenseTensor<long>(inputMask, new[] { batch, seq })),
            NamedOnnxValue.CreateFromTensor("input_ids:0", new DenseTensor<long>(inputIds, new[] { batch, seq })),
        };
        using var ortResults = session.Run(ortFeeds);

        // Materialize ORT outputs as float[].
        var ortByName = new Dictionary<string, float[]>(ortResults.Count);
        foreach (var r in ortResults)
        {
            try { ortByName[r.Name] = r.AsTensor<float>().ToArray(); continue; } catch { }
            try { ortByName[r.Name] = r.AsTensor<long>().ToArray().Select(x => (float)x).ToArray(); continue; } catch { }
            try { ortByName[r.Name] = r.AsTensor<int>().ToArray().Select(x => (float)x).ToArray(); continue; } catch { }
            // bool / other — skip for now
        }
        _output.WriteLine($"ORT returned {ortByName.Count} comparable outputs.");

        // Import the modified model + run.
        using var stream = new MemoryStream(modifiedBytes);
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
        _output.WriteLine($"Imported, {result.UnsupportedOperators.Count} unsupported ops, plan {result.Plan?.StepCount} steps.");
        Assert.Empty(result.UnsupportedOperators);

        foreach (var kv in result.Inputs)
            _output.WriteLine($"  Input '{kv.Key}': shape=[{string.Join(",", kv.Value._shape)}], len={kv.Value.AsWritableSpan().Length}");

        FillFloat(result.Inputs["input_ids:0"], inputIds);
        FillFloat(result.Inputs["input_mask:0"], inputMask);
        FillFloat(result.Inputs["segment_ids:0"], segmentIds);
        FillFloat(result.Inputs["unique_ids_raw_output___9:0"], uniqueIds);

        // Sanity: after fill, log the boundary region of segment_ids.
        var segSpan = result.Inputs["segment_ids:0"].AsWritableSpan();
        _output.WriteLine($"  segment_ids[126..131] after fill: {segSpan[126]} {segSpan[127]} {segSpan[128]} {segSpan[129]} {segSpan[130]}");
        result.Plan!.Execute();

        // Walk the graph in topological order (same order as our importer's
        // sorted nodes — source order is topological enough for tf2onnx
        // exports) and find the first node whose output diverges.
        int checkedNodes = 0, matched = 0;
        string? firstDivergentNode = null;
        float firstDivergentMaxDiff = 0f;
        string? firstDivergentOpType = null;

        foreach (var node in graph.Node)
        {
            foreach (var outName in node.Output)
            {
                if (string.IsNullOrEmpty(outName)) continue;
                if (!ortByName.TryGetValue(outName, out var ort)) continue;
                if (!result.Outputs.TryGetValue(outName, out var ours)) continue;

                var oursSpan = ours.AsSpan();
                if (oursSpan.Length != ort.Length) continue;

                checkedNodes++;
                float maxDiff = 0f;
                for (int i = 0; i < ort.Length; i++)
                {
                    float d = Math.Abs(ort[i] - oursSpan[i]);
                    float scale = Math.Max(Math.Abs(ort[i]), 1f);
                    if (d > 1e-4f * scale && d > maxDiff) maxDiff = d;
                }

                if (maxDiff < 1e-4f)
                {
                    matched++;
                }
                else if (firstDivergentNode is null)
                {
                    firstDivergentNode = outName;
                    firstDivergentMaxDiff = maxDiff;
                    firstDivergentOpType = node.OpType;

                    // Dump the first divergent indices for visual inspection.
                    _output.WriteLine($"--- First divergent node: {node.OpType} '{outName}' ---");
                    _output.WriteLine($"  Our shape: [{string.Join(",", ours._shape)}] len {oursSpan.Length}");
                    _output.WriteLine($"  ORT len: {ort.Length}");
                    _output.WriteLine($"  Node inputs: [{string.Join(", ", node.Input)}]");
                    int shown = 0;
                    for (int i = 0; i < ort.Length && shown < 15; i++)
                    {
                        float d = Math.Abs(ort[i] - oursSpan[i]);
                        if (d <= 1e-6f) continue;
                        _output.WriteLine($"    i={i}: ort={ort[i]:F6} ours={oursSpan[i]:F6} diff={d:F6}");
                        shown++;
                    }

                    // Dump inputs to this Reshape (for diagnosing Reshape
                    // specifically — typically segment_ids + shape-tensor).
                    foreach (var inName in node.Input)
                    {
                        if (result.Outputs.TryGetValue(inName, out var inputOurs))
                        {
                            var inSpan = inputOurs.AsSpan();
                            _output.WriteLine($"    input '{inName}' in outputs: shape=[{string.Join(",", inputOurs._shape)}] len={inSpan.Length}, first few: {(inSpan.Length > 0 ? inSpan[0].ToString() : "")} {(inSpan.Length > 1 ? inSpan[1].ToString() : "")} {(inSpan.Length > 2 ? inSpan[2].ToString() : "")}");
                            if (ortByName.TryGetValue(inName, out var inputOrt))
                            {
                                _output.WriteLine($"    ORT's '{inName}': len={inputOrt.Length}, first few: {(inputOrt.Length > 0 ? inputOrt[0] : 0)} {(inputOrt.Length > 1 ? inputOrt[1] : 0)} {(inputOrt.Length > 2 ? inputOrt[2] : 0)}");
                            }
                        }
                        else if (result.Inputs.TryGetValue(inName, out var inPlaceholder))
                        {
                            var s = inPlaceholder.AsWritableSpan();
                            _output.WriteLine($"    input '{inName}' from graph-input placeholder: shape=[{string.Join(",", inPlaceholder._shape)}] len={s.Length}, [0..5]={s[0]} {s[1]} {s[2]} {s[3]} {s[4]}, [125..130]={s[125]} {s[126]} {s[127]} {s[128]} {s[129]}");
                        }
                    }
                }
            }
        }

        _output.WriteLine($"Checked {checkedNodes} intermediate outputs; {matched} bit-exact, {checkedNodes - matched} divergent.");
        if (firstDivergentNode is not null)
        {
            _output.WriteLine($"FIRST DIVERGENT NODE: op={firstDivergentOpType}, output='{firstDivergentNode}', max diff {firstDivergentMaxDiff}");
        }
        else
        {
            _output.WriteLine("All outputs match. (No divergence found.)");
        }
    }

    private static void FillFloat(LinearAlgebra.Tensor<float> placeholder, long[] source)
    {
        var dst = placeholder.AsWritableSpan();
        for (int i = 0; i < source.Length; i++) dst[i] = source[i];
    }
}
