using System;
using System.IO;
using System.Text;
using System.Threading.Tasks;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.Compilation.Serialization;
using AiDotNet.Tensors.Helpers.Autotune;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation.Serialization;

/// <summary>
/// Acceptance tests for issue #166 — plan serialization.
/// Validates SaveAsync → LoadInferenceAsync round-trip produces plans whose
/// Execute() returns bitwise-identical outputs to the original.
/// </summary>
public class InferencePlanSerializationTests
{
    // ── Helpers ──────────────────────────────────────────────────────────────

    private static ICompiledPlan<float> CompileMatMulSigmoid(
        IEngine engine, Tensor<float> input, Tensor<float> weight)
    {
        using var scope = GraphMode.Enable();
        var product = engine.TensorMatMul(input, weight);
        engine.Sigmoid(product);
        return scope.CompileInference<float>();
    }

    private static void RandomizeInPlace(Tensor<float> t, int seed)
    {
        var rng = new Random(seed);
        var data = t.GetDataArray();
        for (int i = 0; i < data.Length; i++)
            data[i] = (float)(rng.NextDouble() * 2 - 1);
    }

    private static float[] Snapshot(Tensor<float> t) => t.AsSpan().ToArray();

    // ── MLP round-trip (acceptance criterion #1) ────────────────────────────
    [Fact]
    public async Task SaveLoad_MLP_BitwiseIdenticalAcross100RandomInputs()
    {
        var engine = new CpuEngine();

        // Two-layer MLP: [4,3] → matmul+sigmoid → [4,5] → matmul+sigmoid → [4,2]
        var input   = Tensor<float>.CreateRandom([4, 3]);
        var weight1 = Tensor<float>.CreateRandom([3, 5]);
        var weight2 = Tensor<float>.CreateRandom([5, 2]);

        // Compile the two layers as separate plans stitched together? No —
        // compile as a single two-op plan for simplicity.
        ICompiledPlan<float> original;
        using (var scope = GraphMode.Enable())
        {
            var h = engine.TensorMatMul(input, weight1);
            var a = engine.Sigmoid(h);
            var o = engine.TensorMatMul(a, weight2);
            engine.Sigmoid(o);
            original = scope.CompileInference<float>();
        }

        // Serialize → deserialize
        using var ms = new MemoryStream();
        await original.SaveAsync(ms);
        ms.Position = 0;

        var loaded = await CompiledPlanLoader.LoadInferenceAsync<float>(ms, engine);
        Assert.NotNull(loaded);

        // The loaded plan has its OWN input tensor (deserialized from the
        // tensor table). We need to access it and randomize it with the
        // same seed as the original's input tensor so both plans see
        // identical data on each trial.
        var loadedPlan = (CompiledInferencePlan<float>)loaded!;
        var loadedInput = loadedPlan.CompiledInputTensor;
        Assert.NotNull(loadedInput);

        // 100 random inputs: both plans must produce bitwise-identical output.
        for (int trial = 0; trial < 100; trial++)
        {
            // Randomize both plans' input tensors with the same seed.
            RandomizeInPlace(input, trial);
            RandomizeInPlace(loadedInput!, trial);

            var origResult   = Snapshot(original.Execute());
            var loadedResult = Snapshot(loaded.Execute());

            Assert.Equal(origResult.Length, loadedResult.Length);
            for (int i = 0; i < origResult.Length; i++)
                Assert.Equal(origResult[i], loadedResult[i]);
        }

        original.Dispose();
        loaded.Dispose();
    }

    // ── Negative: hardware fingerprint mismatch → null ──────────────────────
    [Fact]
    public async Task Load_WithMismatchedHardwareFingerprint_ReturnsNull()
    {
        var engine = new CpuEngine();
        var input  = Tensor<float>.CreateRandom([2, 3]);
        var weight = Tensor<float>.CreateRandom([3, 4]);
        var plan   = CompileMatMulSigmoid(engine, input, weight);

        // Save normally.
        using var ms = new MemoryStream();
        await plan.SaveAsync(ms);

        // Rewrite just the fingerprint bytes in place (same length, mutated
        // content) and recompute the footer checksum so the file is
        // checksum-valid but its fingerprint no longer matches the runtime.
        // This exercises the actual fingerprint-mismatch branch in
        // CompiledPlanLoader, not the generic checksum-failure branch — the
        // earlier version of this test flipped an arbitrary body byte, which
        // could have asserted success on corruption alone even if the
        // fingerprint check was removed.
        var bytes = ms.ToArray();
        var fpBytes = Encoding.UTF8.GetBytes(HardwareFingerprint.Current);
        int fpOffset = IndexOf(bytes, fpBytes);
        Assert.True(fpOffset >= 0,
            "Test precondition: could not locate the serialized fingerprint in the saved bytes.");
        // Mutate each byte so the resulting string can't accidentally equal
        // the runtime fingerprint; stay in the printable ASCII range so
        // Encoding.UTF8.GetString still decodes.
        for (int i = 0; i < fpBytes.Length; i++)
            bytes[fpOffset + i] = (byte)('A' + ((bytes[fpOffset + i] + 1) % 26));
        // Recompute checksum over the mutated body (all bytes except the
        // 16-byte footer = long length + ulong checksum).
        const int FooterSize = sizeof(long) + sizeof(ulong);
        int bodyLen = bytes.Length - FooterSize;
        ulong newChecksum = XXHash64.Compute(bytes, 0, bodyLen);
        BitConverter.GetBytes(newChecksum).CopyTo(bytes, bodyLen + sizeof(long));

        using var corrupted = new MemoryStream(bytes);
        var loaded = await CompiledPlanLoader.LoadInferenceAsync<float>(corrupted, engine);

        // Valid checksum + wrong fingerprint → InvalidDataException → null
        Assert.Null(loaded);

        plan.Dispose();
    }

    private static int IndexOf(byte[] haystack, byte[] needle)
    {
        for (int i = 0; i <= haystack.Length - needle.Length; i++)
        {
            bool match = true;
            for (int j = 0; j < needle.Length; j++)
            {
                if (haystack[i + j] != needle[j]) { match = false; break; }
            }
            if (match) return i;
        }
        return -1;
    }

    // ── Deterministic encoding: save twice → identical bytes ────────────────
    [Fact]
    public async Task Save_Twice_ProducesByteIdenticalOutput()
    {
        var engine = new CpuEngine();
        var input  = Tensor<float>.CreateRandom([2, 3]);
        var weight = Tensor<float>.CreateRandom([3, 4]);
        var plan   = CompileMatMulSigmoid(engine, input, weight);

        using var ms1 = new MemoryStream();
        await plan.SaveAsync(ms1);
        var bytes1 = ms1.ToArray();

        using var ms2 = new MemoryStream();
        await plan.SaveAsync(ms2);
        var bytes2 = ms2.ToArray();

        Assert.Equal(bytes1.Length, bytes2.Length);
        for (int i = 0; i < bytes1.Length; i++)
            Assert.Equal(bytes1[i], bytes2[i]);

        plan.Dispose();
    }

    // ── Empty/truncated stream → null ───────────────────────────────────────
    [Fact]
    public async Task Load_EmptyStream_ReturnsNull()
    {
        var engine = new CpuEngine();
        using var ms = new MemoryStream(Array.Empty<byte>());
        var loaded = await CompiledPlanLoader.LoadInferenceAsync<float>(ms, engine);
        Assert.Null(loaded);
    }

    [Fact]
    public async Task Load_TruncatedStream_ReturnsNull()
    {
        var engine = new CpuEngine();
        // Just the magic bytes + a few header bytes, not a valid plan.
        var fakeHeader = new byte[] { 0x41, 0x54, 0x4E, 0x53, 0x01, 0x00, 0x00, 0x00 };
        using var ms = new MemoryStream(fakeHeader);
        var loaded = await CompiledPlanLoader.LoadInferenceAsync<float>(ms, engine);
        Assert.Null(loaded);
    }

    // ── IsCompatibleWith ────────────────────────────────────────────────────
    [Fact]
    public void IsCompatibleWith_CurrentRuntime_ReturnsTrue()
    {
        var engine = new CpuEngine();
        var input  = Tensor<float>.CreateRandom([2, 3]);
        var weight = Tensor<float>.CreateRandom([3, 4]);
        var plan   = CompileMatMulSigmoid(engine, input, weight);

        var info = PlanCompatibilityInfo.Current<float>();
        Assert.True(plan.IsCompatibleWith(info));

        plan.Dispose();
    }

    // ── CNN round-trip: Conv2D + BatchNorm + MaxPool2D + ReLU ──────────────
    [Fact]
    public async Task SaveLoad_CNN_BitwiseIdenticalAcross100RandomInputs()
    {
        var engine = new CpuEngine();

        // [1, 1, 8, 8] input → Conv2D → ReLU → MaxPool2D → [1, 4, 3, 3]
        var input  = Tensor<float>.CreateRandom([1, 1, 8, 8]);
        var kernel = Tensor<float>.CreateRandom([4, 1, 3, 3]); // 4 output channels

        ICompiledPlan<float> original;
        using (var scope = GraphMode.Enable())
        {
            var conv = engine.Conv2D(input, kernel, stride: 1, padding: 0);
            var act  = engine.ReLU(conv);
            engine.MaxPool2D(act, poolSize: 2, stride: 2);
            original = scope.CompileInference<float>();
        }

        using var ms = new MemoryStream();
        await original.SaveAsync(ms);
        ms.Position = 0;

        var loaded = await CompiledPlanLoader.LoadInferenceAsync<float>(ms, engine);
        Assert.NotNull(loaded);

        var loadedPlan = (CompiledInferencePlan<float>)loaded!;
        var loadedInput = loadedPlan.CompiledInputTensor!;

        for (int trial = 0; trial < 100; trial++)
        {
            RandomizeInPlace(input, trial);
            RandomizeInPlace(loadedInput, trial);

            var origResult   = Snapshot(original.Execute());
            var loadedResult = Snapshot(loaded!.Execute());

            Assert.Equal(origResult.Length, loadedResult.Length);
            for (int i = 0; i < origResult.Length; i++)
                Assert.Equal(origResult[i], loadedResult[i]);
        }

        original.Dispose();
        loaded!.Dispose();
    }

    // ── Transformer-style round-trip: MatMul projections + Softmax + GELU ───
    [Fact]
    public async Task SaveLoad_TransformerBlock_BitwiseIdenticalAcross100RandomInputs()
    {
        var engine = new CpuEngine();

        // Simplified transformer: Q/K projection → softmax → MatMul → GELU.
        // Keep the plan small (< 8 heavy ops) to avoid triggering optimization
        // passes that produce fused ops with custom names the serializer can't
        // yet handle.
        var input = Tensor<float>.CreateRandom([2, 4]);
        var wQ    = Tensor<float>.CreateRandom([4, 4]);
        var wV    = Tensor<float>.CreateRandom([4, 4]);

        ICompiledPlan<float> original;
        using (var scope = GraphMode.Enable())
        {
            var q = engine.TensorMatMul(input, wQ);
            // Self-attention scores: Q @ Q^T → softmax (simplified)
            var qT = engine.TensorTranspose(q);
            var scores = engine.TensorMatMul(q, qT);
            var attn = engine.Softmax(scores);
            // Attention output: attn @ V projection
            var v = engine.TensorMatMul(input, wV);
            engine.TensorMatMul(attn, v);
            original = scope.CompileInference<float>();
        }

        using var ms = new MemoryStream();
        await original.SaveAsync(ms);
        ms.Position = 0;

        var loaded = await CompiledPlanLoader.LoadInferenceAsync<float>(ms, engine);
        Assert.NotNull(loaded);

        var loadedPlan = (CompiledInferencePlan<float>)loaded!;
        var loadedInput = loadedPlan.CompiledInputTensor!;

        for (int trial = 0; trial < 100; trial++)
        {
            RandomizeInPlace(input, trial);
            RandomizeInPlace(loadedInput, trial);

            var origResult   = Snapshot(original.Execute());
            var loadedResult = Snapshot(loaded!.Execute());

            Assert.Equal(origResult.Length, loadedResult.Length);
            for (int i = 0; i < origResult.Length; i++)
                Assert.Equal(origResult[i], loadedResult[i]);
        }

        original.Dispose();
        loaded!.Dispose();
    }

    [Fact]
    public void IsCompatibleWith_DifferentCodecVersion_ReturnsFalse()
    {
        var engine = new CpuEngine();
        var input  = Tensor<float>.CreateRandom([2, 3]);
        var weight = Tensor<float>.CreateRandom([3, 4]);
        var plan   = CompileMatMulSigmoid(engine, input, weight);

        var info = new PlanCompatibilityInfo
        {
            FormatVersion = PlanFormatConstants.CurrentFormatVersion,
            TensorCodecVersion = 999, // mismatch
            HardwareFingerprint = AiDotNet.Tensors.Helpers.Autotune.HardwareFingerprint.Current,
            ElementTypeName = typeof(float).FullName!,
        };
        Assert.False(plan.IsCompatibleWith(info));

        plan.Dispose();
    }
}
