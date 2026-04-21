using AiDotNet.Tensors.Engines;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Onnx.Tests;

/// <summary>
/// Checks whether the graph-input placeholders for BERT get their
/// buffers trampled between executes. If a memory-planner or op closure
/// reuses an input placeholder's storage, the second execute would start
/// with corrupted inputs — matching the observed multi-execute bug.
/// </summary>
public class BertInputStability
{
    private readonly ITestOutputHelper _output;
    public BertInputStability(ITestOutputHelper output) { _output = output; }

    [SkippableFact]
    public void InputBuffers_NotOverwrittenByExecute()
    {
        var path = Path.Combine(
            Environment.GetEnvironmentVariable("AIDOTNET_ONNX_MODELS")
                ?? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".aidotnet", "onnx-models"),
            "bertsquad-10.onnx");
        Skip.IfNot(File.Exists(path), $"Need {path}");

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
        Assert.NotNull(result.Plan);

        const int batch = 1, seq = 256;
        var rng = new Random(42);
        var inputIds = new long[batch * seq];
        var inputMask = new long[batch * seq];
        var segmentIds = new long[batch * seq];
        var uniqueIds = new long[] { 0 };
        for (int i = 0; i < inputIds.Length; i++)
        {
            inputIds[i] = rng.Next(1, 30000);
            inputMask[i] = rng.NextDouble() < 0.9 ? 1 : 0;
            segmentIds[i] = i < seq / 2 ? 0 : 1;
        }

        FillFloat(result.Inputs["input_ids:0"], inputIds);
        FillFloat(result.Inputs["input_mask:0"], inputMask);
        FillFloat(result.Inputs["segment_ids:0"], segmentIds);
        FillFloat(result.Inputs["unique_ids_raw_output___9:0"], uniqueIds);

        // Snapshot BEFORE execute.
        var beforeIds = result.Inputs["input_ids:0"].AsSpan().ToArray();
        var beforeMask = result.Inputs["input_mask:0"].AsSpan().ToArray();
        var beforeSeg = result.Inputs["segment_ids:0"].AsSpan().ToArray();
        var beforeUid = result.Inputs["unique_ids_raw_output___9:0"].AsSpan().ToArray();

        result.Plan!.Execute();

        // Snapshot AFTER execute — if any placeholder buffer was trampled,
        // these will differ from the 'before' snapshot.
        var afterIds = result.Inputs["input_ids:0"].AsSpan().ToArray();
        var afterMask = result.Inputs["input_mask:0"].AsSpan().ToArray();
        var afterSeg = result.Inputs["segment_ids:0"].AsSpan().ToArray();
        var afterUid = result.Inputs["unique_ids_raw_output___9:0"].AsSpan().ToArray();

        int idsDiff = Count(beforeIds, afterIds);
        int maskDiff = Count(beforeMask, afterMask);
        int segDiff = Count(beforeSeg, afterSeg);
        int uidDiff = Count(beforeUid, afterUid);

        _output.WriteLine($"input_ids after Execute: {idsDiff}/{beforeIds.Length} positions differ");
        _output.WriteLine($"input_mask after Execute: {maskDiff}/{beforeMask.Length} positions differ");
        _output.WriteLine($"segment_ids after Execute: {segDiff}/{beforeSeg.Length} positions differ");
        _output.WriteLine($"unique_ids after Execute: {uidDiff}/{beforeUid.Length} positions differ");

        if (maskDiff > 0)
        {
            _output.WriteLine("Sample mask-before-vs-after:");
            for (int i = 0; i < Math.Min(20, beforeMask.Length); i++)
            {
                if (beforeMask[i] != afterMask[i])
                    _output.WriteLine($"  [{i}] before={beforeMask[i]} after={afterMask[i]}");
            }
        }

        Assert.Equal(0, idsDiff);
        Assert.Equal(0, maskDiff);
        Assert.Equal(0, segDiff);
        Assert.Equal(0, uidDiff);
    }

    private static int Count(float[] a, float[] b)
    {
        int c = 0;
        for (int i = 0; i < a.Length; i++) if (a[i] != b[i]) c++;
        return c;
    }

    private static void FillFloat(LinearAlgebra.Tensor<float> placeholder, long[] source)
    {
        var dst = placeholder.AsWritableSpan();
        for (int i = 0; i < source.Length; i++) dst[i] = source[i];
    }
}
