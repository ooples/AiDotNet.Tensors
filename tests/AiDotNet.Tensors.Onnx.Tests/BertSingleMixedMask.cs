using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Optimization;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Onnx.Tests;

/// <summary>
/// Isolates whether the BERT × 100 failure is (a) amplified float drift
/// from realistic mask inputs on a single execute, or (b) multi-execute
/// state bleed. Test A runs s=0 exactly. Test B runs it twice and
/// diffs the two outputs against each other.
/// </summary>
public class BertSingleMixedMask
{
    private readonly ITestOutputHelper _output;
    public BertSingleMixedMask(ITestOutputHelper output) { _output = output; }

    [SkippableFact]
    public void SingleSample_s0_MatchesOnnxRuntime()
    {
        var path = ModelPath();
        Skip.IfNot(File.Exists(path) && OnnxRuntimeReference.IsOrtAvailable, $"Need {path}");

        var (result, session) = ImportBoth(path);
        var (inputIds, inputMask, segmentIds, uniqueIds) = SampleAtIndex(0);

        _output.WriteLine($"Mask zeros: {inputMask.Count(m => m == 0)}/{inputMask.Length}");

        var ortByName = RunOrt(session, inputIds, inputMask, segmentIds, uniqueIds);

        FillFloat(result.Inputs["input_ids:0"], inputIds);
        FillFloat(result.Inputs["input_mask:0"], inputMask);
        FillFloat(result.Inputs["segment_ids:0"], segmentIds);
        FillFloat(result.Inputs["unique_ids_raw_output___9:0"], uniqueIds);
        result.Plan!.Execute();

        ReportOutputDivergence(ortByName, result);
        session.Dispose();
    }

    [SkippableFact]
    public void MultiExecute_DifferentInputs_AllMatchOrt_NoOpt()
    {
        var path = ModelPath();
        Skip.IfNot(File.Exists(path) && OnnxRuntimeReference.IsOrtAvailable, $"Need {path}");

        // Turn off every post-compile optimization pass to see whether the
        // multi-execute state bleed is caused by a pass (ConstantFolding,
        // CSE, Fusion, etc.).
        var opts = new TensorCodecOptions
        {
            EnableConstantFolding = false,
            EnableForwardCSE = false,
            EnableDataflowFusion = false,
            EnableConvBnFusion = false,
            EnableAttentionFusion = false,
            EnablePointwiseFusion = false,
            EnableBlasBatch = false,
            EnableAlgebraicBackward = false,
        };
        TensorCodecOptions.SetCurrent(opts);

        // Guarantee we restore the thread-local options even when an assertion
        // or exception fires before the explicit SetCurrent(null) below. Without
        // this finally, a failed assertion in this test would leave every
        // optimization pass disabled for the rest of the test-class / fixture
        // lifetime on this worker thread.
        try
        {
        var (result, session) = ImportBoth(path);
        int totalMismatches = 0;
        float maxDiff = 0f;

        for (int s = 0; s < 5; s++)
        {
            var (inputIds, inputMask, segmentIds, uniqueIds) = SampleAtIndex(s);
            var ortByName = RunOrt(session, inputIds, inputMask, segmentIds, uniqueIds);
            FillFloat(result.Inputs["input_ids:0"], inputIds);
            FillFloat(result.Inputs["input_mask:0"], inputMask);
            FillFloat(result.Inputs["segment_ids:0"], segmentIds);
            FillFloat(result.Inputs["unique_ids_raw_output___9:0"], uniqueIds);
            result.Plan!.Execute();

            int sampleMismatches = 0;
            float sampleMaxDiff = 0f;
            foreach (var kv in ortByName)
            {
                if (!result.Outputs.TryGetValue(kv.Key, out var ours)) continue;
                var oursSpan = ours.AsSpan();
                if (oursSpan.Length != kv.Value.Length) continue;
                for (int i = 0; i < kv.Value.Length; i++)
                {
                    float d = Math.Abs(kv.Value[i] - oursSpan[i]);
                    float scale = Math.Max(Math.Abs(kv.Value[i]), 1f);
                    if (d > 1e-4f * scale) { sampleMismatches++; if (d > sampleMaxDiff) sampleMaxDiff = d; }
                }
            }
            _output.WriteLine($"  s={s} mask0={inputMask.Count(m => m == 0)}: {sampleMismatches} diverge, max={sampleMaxDiff}");
            totalMismatches += sampleMismatches;
            if (sampleMaxDiff > maxDiff) maxDiff = sampleMaxDiff;
        }

        _output.WriteLine($"5 samples, different inputs, same plan instance: {totalMismatches} diverge, max {maxDiff}");
        Assert.True(totalMismatches == 0,
            $"Multi-execute-different-inputs (no-opt): {totalMismatches} elements diverged. State bleed.");
        session.Dispose();
        }
        finally
        {
            TensorCodecOptions.SetCurrent(null);
        }
    }

    [SkippableFact]
    public void DiagnoseGather_SeesNewIndicesOnRerun()
    {
        var path = ModelPath();
        Skip.IfNot(File.Exists(path), $"Need {path}");

        GatherDebug.Enabled = true;
        GatherDebug.Clear();
        try
        {
            var (result, _) = ImportBoth(path, withSession: false);
            var importLogs = GatherDebug.Snapshot();
            _output.WriteLine($"Import gather count: {importLogs.Length}");
            for (int i = 0; i < Math.Min(2, importLogs.Length); i++)
                _output.WriteLine($"  IMP[{i}]: {importLogs[i]}");
            GatherDebug.Clear();
            var (ids0, mask0, seg0, uid0) = SampleAtIndex(0);
            FillFloat(result.Inputs["input_ids:0"], ids0);
            FillFloat(result.Inputs["input_mask:0"], mask0);
            FillFloat(result.Inputs["segment_ids:0"], seg0);
            FillFloat(result.Inputs["unique_ids_raw_output___9:0"], uid0);
            _output.WriteLine($"s=0 input_ids[0..3]={ids0[0]},{ids0[1]},{ids0[2]}");
            result.Plan!.Execute();
            var run1Logs = GatherDebug.Snapshot();
            GatherDebug.Clear();

            var (ids1, mask1, seg1, uid1) = SampleAtIndex(1);
            FillFloat(result.Inputs["input_ids:0"], ids1);
            FillFloat(result.Inputs["input_mask:0"], mask1);
            FillFloat(result.Inputs["segment_ids:0"], seg1);
            FillFloat(result.Inputs["unique_ids_raw_output___9:0"], uid1);
            _output.WriteLine($"s=1 input_ids[0..3]={ids1[0]},{ids1[1]},{ids1[2]}");
            result.Plan!.Execute();
            var run2Logs = GatherDebug.Snapshot();

            _output.WriteLine($"Run1 gather count: {run1Logs.Length}");
            for (int i = 0; i < run1Logs.Length; i++)
                _output.WriteLine($"  R1[{i}]: {run1Logs[i]}");
            _output.WriteLine($"Run2 gather count: {run2Logs.Length}");
            for (int i = 0; i < run2Logs.Length; i++)
                _output.WriteLine($"  R2[{i}]: {run2Logs[i]}");
        }
        finally { GatherDebug.Enabled = false; GatherDebug.Clear(); }
    }

    [SkippableFact]
    public void DiagnoseS1Output_IsItSameAsS0OrDifferent()
    {
        var path = ModelPath();
        Skip.IfNot(File.Exists(path) && OnnxRuntimeReference.IsOrtAvailable, $"Need {path}");

        var (result, session) = ImportBoth(path);

        // Run s=0 and capture OURS output.
        var (ids0, mask0, seg0, uid0) = SampleAtIndex(0);
        FillFloat(result.Inputs["input_ids:0"], ids0);
        FillFloat(result.Inputs["input_mask:0"], mask0);
        FillFloat(result.Inputs["segment_ids:0"], seg0);
        FillFloat(result.Inputs["unique_ids_raw_output___9:0"], uid0);
        result.Plan!.Execute();
        var s0Ours = new Dictionary<string, float[]>();
        foreach (var kv in result.Outputs) s0Ours[kv.Key] = kv.Value.AsSpan().ToArray();

        // Get s=1 ORT output.
        var (ids1, mask1, seg1, uid1) = SampleAtIndex(1);
        var s1Ort = RunOrt(session, ids1, mask1, seg1, uid1);

        // Run s=1 on OUR plan (same instance).
        FillFloat(result.Inputs["input_ids:0"], ids1);
        FillFloat(result.Inputs["input_mask:0"], mask1);
        FillFloat(result.Inputs["segment_ids:0"], seg1);
        FillFloat(result.Inputs["unique_ids_raw_output___9:0"], uid1);
        result.Plan!.Execute();

        foreach (var name in new[] { "unstack:0", "unstack:1", "unique_ids:0" })
        {
            if (!result.Outputs.TryGetValue(name, out var ourS1)) continue;
            var ourS1Arr = ourS1.AsSpan().ToArray();
            var s0 = s0Ours[name];
            int matchS0 = 0, matchOrt = 0;
            s1Ort.TryGetValue(name, out var ort);
            for (int i = 0; i < ourS1Arr.Length; i++)
            {
                if (ourS1Arr[i] == s0[i]) matchS0++;
                if (ort != null && Math.Abs(ourS1Arr[i] - ort[i]) < 1e-4f) matchOrt++;
            }
            _output.WriteLine($"  '{name}': our-s1 matches our-s0 in {matchS0}/{ourS1Arr.Length} positions; matches ORT-s1 in {matchOrt}/{ourS1Arr.Length}");
            _output.WriteLine($"    our-s0[0..3]: {s0[0]:F4} {s0[1]:F4} {s0[2]:F4}");
            _output.WriteLine($"    our-s1[0..3]: {ourS1Arr[0]:F4} {ourS1Arr[1]:F4} {ourS1Arr[2]:F4}");
            if (ort != null)
                _output.WriteLine($"    ort-s1[0..3]: {ort[0]:F4} {ort[1]:F4} {ort[2]:F4}");
        }
        session.Dispose();
    }

    [SkippableFact]
    public void MultiExecute_SameInputs_ProducesSameOutputs()
    {
        var path = ModelPath();
        Skip.IfNot(File.Exists(path), $"Need {path}");

        var (result, _) = ImportBoth(path, withSession: false);
        var (inputIds, inputMask, segmentIds, uniqueIds) = SampleAtIndex(0);

        FillFloat(result.Inputs["input_ids:0"], inputIds);
        FillFloat(result.Inputs["input_mask:0"], inputMask);
        FillFloat(result.Inputs["segment_ids:0"], segmentIds);
        FillFloat(result.Inputs["unique_ids_raw_output___9:0"], uniqueIds);
        result.Plan!.Execute();
        var run1 = new Dictionary<string, float[]>();
        foreach (var kv in result.Outputs)
            run1[kv.Key] = kv.Value.AsSpan().ToArray();

        // Same inputs — re-execute, compare to run1.
        FillFloat(result.Inputs["input_ids:0"], inputIds);
        FillFloat(result.Inputs["input_mask:0"], inputMask);
        FillFloat(result.Inputs["segment_ids:0"], segmentIds);
        FillFloat(result.Inputs["unique_ids_raw_output___9:0"], uniqueIds);
        result.Plan!.Execute();

        int totalMismatches = 0;
        float maxDiff = 0f;
        foreach (var kv in run1)
        {
            if (!result.Outputs.TryGetValue(kv.Key, out var ours)) continue;
            var oursSpan = ours.AsSpan();
            if (oursSpan.Length != kv.Value.Length) continue;
            int mismatches = 0;
            for (int i = 0; i < kv.Value.Length; i++)
            {
                float d = Math.Abs(kv.Value[i] - oursSpan[i]);
                if (d > 0f) { mismatches++; if (d > maxDiff) maxDiff = d; }
            }
            _output.WriteLine($"  '{kv.Key}': {mismatches}/{kv.Value.Length} differ between runs, max={maxDiff}");
            totalMismatches += mismatches;
        }
        _output.WriteLine($"Multi-execute determinism: {totalMismatches} differ, max diff {maxDiff}");
        Assert.True(totalMismatches == 0,
            $"Determinism: two executes with same inputs diverged ({totalMismatches} elements, max {maxDiff}). State bleed.");
    }

    private static string ModelPath() => Path.Combine(
        Environment.GetEnvironmentVariable("AIDOTNET_ONNX_MODELS")
            ?? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".aidotnet", "onnx-models"),
        "bertsquad-10.onnx");

    private static (long[] inputIds, long[] inputMask, long[] segmentIds, long[] uniqueIds) SampleAtIndex(int s)
    {
        // Reproduce the RNG sequence of BertExecute100Samples for sample index s.
        const int batch = 1, seq = 256;
        var rng = new Random(42);
        long[] inputIds = null!, inputMask = null!, segmentIds = null!;
        long[] uniqueIds = new long[] { 0 };
        for (int k = 0; k <= s; k++)
        {
            inputIds = new long[batch * seq];
            inputMask = new long[batch * seq];
            segmentIds = new long[batch * seq];
            uniqueIds = new long[] { k };
            for (int i = 0; i < inputIds.Length; i++)
            {
                inputIds[i] = rng.Next(1, 30000);
                inputMask[i] = rng.NextDouble() < 0.9 ? 1 : 0;
                segmentIds[i] = i < seq / 2 ? 0 : 1;
            }
        }
        return (inputIds, inputMask, segmentIds, uniqueIds);
    }

    private static (OnnxImportResult<float> result, InferenceSession session) ImportBoth(string path, bool withSession = true)
    {
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
        var session = withSession ? new InferenceSession(modelBytes) : null!;
        return (result, session!);
    }

    private static Dictionary<string, float[]> RunOrt(InferenceSession session, long[] inputIds, long[] inputMask, long[] segmentIds, long[] uniqueIds)
    {
        const int batch = 1, seq = 256;
        var feeds = new[]
        {
            NamedOnnxValue.CreateFromTensor("unique_ids_raw_output___9:0", new DenseTensor<long>(uniqueIds, new[] { 1 })),
            NamedOnnxValue.CreateFromTensor("segment_ids:0", new DenseTensor<long>(segmentIds, new[] { batch, seq })),
            NamedOnnxValue.CreateFromTensor("input_mask:0", new DenseTensor<long>(inputMask, new[] { batch, seq })),
            NamedOnnxValue.CreateFromTensor("input_ids:0", new DenseTensor<long>(inputIds, new[] { batch, seq })),
        };
        using var ortResults = session.Run(feeds);
        var outs = new Dictionary<string, float[]>();
        foreach (var r in ortResults)
        {
            // Fail explicitly if ORT returns a dtype we don't know how to
            // convert — previously the blanket catch let the parity test go
            // green with missing outputs.
            bool converted = false;
            try { outs[r.Name] = r.AsTensor<float>().ToArray(); converted = true; } catch { }
            if (!converted)
            {
                try { outs[r.Name] = r.AsTensor<long>().ToArray().Select(x => (float)x).ToArray(); converted = true; } catch { }
            }
            Assert.True(converted,
                $"ORT output '{r.Name}' has an unsupported element type; extend RunOrt's conversion list.");
        }
        return outs;
    }

    private void ReportOutputDivergence(Dictionary<string, float[]> ortByName, OnnxImportResult<float> result)
    {
        int totalMismatches = 0;
        float maxDiff = 0f;
        foreach (var kv in ortByName)
        {
            Assert.True(result.Outputs.TryGetValue(kv.Key, out var ours),
                $"AiDotNet plan did not produce output '{kv.Key}' that ORT did.");
            var oursSpan = ours!.AsSpan();
            Assert.True(oursSpan.Length == kv.Value.Length,
                $"Output '{kv.Key}' length mismatch: AiDotNet {oursSpan.Length}, ORT {kv.Value.Length}.");
            int mismatches = 0;
            for (int i = 0; i < kv.Value.Length; i++)
            {
                float d = Math.Abs(kv.Value[i] - oursSpan[i]);
                float scale = Math.Max(Math.Abs(kv.Value[i]), 1f);
                if (d > 1e-4f * scale) { mismatches++; if (d > maxDiff) maxDiff = d; }
            }
            _output.WriteLine($"  '{kv.Key}': {mismatches}/{kv.Value.Length} diverge, max diff {maxDiff}");
            totalMismatches += mismatches;
        }
        _output.WriteLine($"Single s=0 vs ORT: {totalMismatches} diverge, max diff {maxDiff}");
        Assert.True(totalMismatches == 0,
            $"BERT SingleMixedMask: {totalMismatches} output elements diverged beyond 1e-4 rel tol. Max: {maxDiff}.");
    }

    private static void FillFloat(LinearAlgebra.Tensor<float> placeholder, long[] source)
    {
        var dst = placeholder.AsWritableSpan();
        for (int i = 0; i < source.Length; i++) dst[i] = source[i];
    }
}
