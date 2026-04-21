using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Onnx.Protos;
using Google.Protobuf;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Onnx.Tests;

/// <summary>
/// Layer-by-layer diagnostic for the multi-execute bug. Promotes every
/// intermediate to a graph output. Imports once. Runs s=0 + captures every
/// intermediate. Runs s=1 + compares every intermediate's value to s=0's
/// (frozen = bug) and to ORT's s=1 (divergent = downstream of the freeze).
/// Reports the first node whose output is identical to s=0 but should have
/// changed — that's the node whose step closure is silently reading stale
/// data.
/// </summary>
public class BertLayerMultiExecute
{
    private readonly ITestOutputHelper _output;
    public BertLayerMultiExecute(ITestOutputHelper output) { _output = output; }

    [SkippableFact]
    public void FindFirstFrozenNode()
    {
        var path = Path.Combine(
            Environment.GetEnvironmentVariable("AIDOTNET_ONNX_MODELS")
                ?? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".aidotnet", "onnx-models"),
            "bertsquad-10.onnx");
        Skip.IfNot(File.Exists(path) && OnnxRuntimeReference.IsOrtAvailable, $"Need {path}");

        var originalBytes = File.ReadAllBytes(path);
        var model = ModelProto.Parser.ParseFrom(originalBytes);
        var graph = model.Graph!;

        var originalOutputs = new HashSet<string>(graph.Output.Select(o => o.Name), StringComparer.Ordinal);
        var graphInputs = new HashSet<string>(graph.Input.Select(i => i.Name), StringComparer.Ordinal);
        var promoted = new List<string>();
        foreach (var node in graph.Node)
        {
            foreach (var outName in node.Output)
            {
                if (string.IsNullOrEmpty(outName)) continue;
                if (originalOutputs.Contains(outName)) continue;
                if (promoted.Contains(outName)) continue;
                promoted.Add(outName);
                graph.Output.Add(new ValueInfoProto { Name = outName });
            }
        }
        _output.WriteLine($"Promoted {promoted.Count} intermediates.");

        var modifiedBytes = model.ToByteArray();
        using var session = new InferenceSession(modifiedBytes);

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
        Assert.Empty(result.UnsupportedOperators);
        Assert.NotNull(result.Plan);

        // Helper to build feeds for a given sample.
        (long[] ids, long[] mask, long[] seg, long[] uid) Sample(int s)
        {
            const int batch = 1, seq = 256;
            var rng = new Random(42);
            long[] ids = null!, mask = null!, seg = null!;
            long[] uid = new long[] { 0 };
            for (int k = 0; k <= s; k++)
            {
                ids = new long[batch * seq];
                mask = new long[batch * seq];
                seg = new long[batch * seq];
                uid = new long[] { k };
                for (int i = 0; i < ids.Length; i++)
                {
                    ids[i] = rng.Next(1, 30000);
                    mask[i] = rng.NextDouble() < 0.9 ? 1 : 0;
                    seg[i] = i < seq / 2 ? 0 : 1;
                }
            }
            return (ids, mask, seg, uid);
        }

        // Run s=0 on our plan, capture every output tensor.
        var (ids0, mask0, seg0, uid0) = Sample(0);
        FillFloat(result.Inputs["input_ids:0"], ids0);
        FillFloat(result.Inputs["input_mask:0"], mask0);
        FillFloat(result.Inputs["segment_ids:0"], seg0);
        FillFloat(result.Inputs["unique_ids_raw_output___9:0"], uid0);
        result.Plan!.Execute();
        var s0Ours = new Dictionary<string, float[]>();
        foreach (var kv in result.Outputs) s0Ours[kv.Key] = kv.Value.AsSpan().ToArray();
        _output.WriteLine($"Captured {s0Ours.Count} s=0 intermediates.");

        // Run s=1 on our plan.
        var (ids1, mask1, seg1, uid1) = Sample(1);
        FillFloat(result.Inputs["input_ids:0"], ids1);
        FillFloat(result.Inputs["input_mask:0"], mask1);
        FillFloat(result.Inputs["segment_ids:0"], seg1);
        FillFloat(result.Inputs["unique_ids_raw_output___9:0"], uid1);
        result.Plan!.Execute();

        // Collect ORT s=1.
        var ortFeeds = new[]
        {
            NamedOnnxValue.CreateFromTensor("unique_ids_raw_output___9:0", new DenseTensor<long>(uid1, new[] { 1 })),
            NamedOnnxValue.CreateFromTensor("segment_ids:0", new DenseTensor<long>(seg1, new[] { 1, 256 })),
            NamedOnnxValue.CreateFromTensor("input_mask:0", new DenseTensor<long>(mask1, new[] { 1, 256 })),
            NamedOnnxValue.CreateFromTensor("input_ids:0", new DenseTensor<long>(ids1, new[] { 1, 256 })),
        };
        using var ortResults = session.Run(ortFeeds);
        var ortS1 = new Dictionary<string, float[]>();
        foreach (var r in ortResults)
        {
            try { ortS1[r.Name] = r.AsTensor<float>().ToArray(); continue; } catch { }
            try { ortS1[r.Name] = r.AsTensor<long>().ToArray().Select(x => (float)x).ToArray(); continue; } catch { }
        }

        // Walk topo order, find first node where our s=1 output is IDENTICAL
        // to our s=0 output BUT ORT says it should differ from s=0.
        int checkedNodes = 0, frozen = 0, correctlyChanged = 0;
        string? firstFrozen = null;
        int firstFrozenIdx = -1;
        int nodeIdx = 0;
        var frozenByOpType = new Dictionary<string, int>();
        foreach (var node in graph.Node)
        {
            foreach (var outName in node.Output)
            {
                if (string.IsNullOrEmpty(outName)) continue;
                if (!s0Ours.TryGetValue(outName, out var s0)) continue;
                if (!ortS1.TryGetValue(outName, out var ortValues)) continue;
                if (!result.Outputs.TryGetValue(outName, out var ours1)) continue;
                var ours1Arr = ours1.AsSpan().ToArray();
                if (ours1Arr.Length != s0.Length || ours1Arr.Length != ortValues.Length) continue;
                if (ours1Arr.Length == 0) continue;

                checkedNodes++;
                bool ortDiffersFromS0 = false;
                bool oursIdenticalToS0 = true;
                for (int i = 0; i < ours1Arr.Length; i++)
                {
                    if (ours1Arr[i] != s0[i]) oursIdenticalToS0 = false;
                    if (Math.Abs(ortValues[i] - s0[i]) > 1e-4f) ortDiffersFromS0 = true;
                    if (!oursIdenticalToS0 && ortDiffersFromS0) break;
                }

                if (oursIdenticalToS0 && ortDiffersFromS0)
                {
                    frozen++;
                    frozenByOpType[node.OpType] = (frozenByOpType.TryGetValue(node.OpType, out var prev) ? prev : 0) + 1;
                    if (firstFrozen == null)
                    {
                        firstFrozen = outName;
                        firstFrozenIdx = nodeIdx;
                        _output.WriteLine($"FIRST FROZEN NODE: [{node.OpType}] '{outName}' (len={ours1Arr.Length})");
                        _output.WriteLine($"  node #{nodeIdx}, inputs=[{string.Join(", ", node.Input)}]");
                        int n = Math.Min(3, ours1Arr.Length);
                        string s0s = string.Join(" ", Enumerable.Range(0, n).Select(i => s0[i].ToString("F4")));
                        string ourss = string.Join(" ", Enumerable.Range(0, n).Select(i => ours1Arr[i].ToString("F4")));
                        string orts = string.Join(" ", Enumerable.Range(0, n).Select(i => ortValues[i].ToString("F4")));
                        _output.WriteLine($"  our-s0[0..{n}]: {s0s}");
                        _output.WriteLine($"  our-s1[0..{n}]: {ourss}");
                        _output.WriteLine($"  ort-s1[0..{n}]: {orts}");
                    }
                }
                else if (!oursIdenticalToS0)
                {
                    correctlyChanged++;
                    if (correctlyChanged <= 5)
                    {
                        _output.WriteLine($"CHANGED: [{node.OpType}] '{outName}' (len={ours1Arr.Length})");
                        int n = Math.Min(3, ours1Arr.Length);
                        string s0s = string.Join(" ", Enumerable.Range(0, n).Select(i => s0[i].ToString("F4")));
                        string ourss = string.Join(" ", Enumerable.Range(0, n).Select(i => ours1Arr[i].ToString("F4")));
                        string orts = string.Join(" ", Enumerable.Range(0, n).Select(i => ortValues[i].ToString("F4")));
                        _output.WriteLine($"  s0={s0s} ours1={ourss} ort1={orts}");
                    }
                }
            }
            nodeIdx++;
        }

        _output.WriteLine($"Checked {checkedNodes}, frozen {frozen}, correctlyChanged {correctlyChanged}");
        foreach (var kv in frozenByOpType.OrderByDescending(k => k.Value))
            _output.WriteLine($"  frozen by op: {kv.Key} × {kv.Value}");
    }

    private static void FillFloat(LinearAlgebra.Tensor<float> placeholder, long[] source)
    {
        var dst = placeholder.AsWritableSpan();
        for (int i = 0; i < source.Length; i++) dst[i] = source[i];
    }
}
