// Copyright (c) AiDotNet. All rights reserved.

#nullable disable

using System.Collections.Generic;
using System.IO;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Licensing;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Serialization.HuggingFace;
using AiDotNet.Tensors.Serialization.Safetensors;
using Xunit;

namespace AiDotNet.Tensors.Tests.Serialization.HuggingFace;

/// <summary>
/// Acceptance test for issue #218: a state-dict written under HF
/// naming conventions and loaded back through
/// <see cref="HFStateDictMapper"/> must reproduce identical
/// numerical outputs from a forward pass. The full chain (write → HF
/// names → read → reverse-map → forward pass) covers every component
/// the issue's "load HF model and reproduce reference logits" gate
/// touches without bundling a real HF checkpoint into the test
/// project. A real HF tiny-model integration test is gated to a
/// separate, weight-bundling suite (see issue #218 deferral note).
/// </summary>
[Collection("PersistenceGuard")]
public class HFSyntheticModelParityTests
{
    private static IDisposable IsolatedTrial()
    {
        var path = Path.Combine(Path.GetTempPath(), "aidotnet-hfparity-trial-" + System.Guid.NewGuid().ToString("N") + ".json");
        return PersistenceGuard.SetTestTrialFilePathOverride(path);
    }

    /// <summary>
    /// Builds a deterministic 2-layer MLP weight set. Same seed →
    /// same numbers → bit-exact comparison after round-trip.
    /// </summary>
    private static Dictionary<string, Tensor<float>> BuildSyntheticState()
    {
        // hidden = 8, in = 4, out = 3 — small enough to verify by
        // eye if anything goes wrong, large enough to exercise rank-2
        // matmul.
        const int InDim = 4, Hidden = 8, OutDim = 3;
        var rng = new System.Random(0xC0FFEE);
        float[] w1 = new float[Hidden * InDim];
        float[] b1 = new float[Hidden];
        float[] w2 = new float[OutDim * Hidden];
        float[] b2 = new float[OutDim];
        for (int i = 0; i < w1.Length; i++) w1[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < b1.Length; i++) b1[i] = (float)(rng.NextDouble() * 0.1);
        for (int i = 0; i < w2.Length; i++) w2[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < b2.Length; i++) b2[i] = (float)(rng.NextDouble() * 0.1);

        return new Dictionary<string, Tensor<float>>
        {
            // AiDotNet-style naming after LLAMA preset's MapForward —
            // "layers[0].mlp.up.weight" etc. We write the file under
            // HF naming, then verify the reader+mapper recovers these
            // canonical AiDotNet names.
            ["layers[0].mlp.up.weight"] = new Tensor<float>(w1, new[] { Hidden, InDim }),
            ["layers[0].mlp.up.bias"] = new Tensor<float>(b1, new[] { Hidden }),
            ["layers[1].mlp.down.weight"] = new Tensor<float>(w2, new[] { OutDim, Hidden }),
            ["layers[1].mlp.down.bias"] = new Tensor<float>(b2, new[] { OutDim }),
        };
    }

    /// <summary>
    /// y = max(0, x · Wᵀ + b). Forward pass for one Linear+ReLU
    /// stage, computed manually so the test doesn't depend on layer
    /// classes from the wider stack — keeps the parity assertion
    /// isolated to load/save behaviour.
    /// </summary>
    private static float[] LinearReluForward(float[] x, Tensor<float> w, Tensor<float> b)
    {
        int outDim = w.Shape[0], inDim = w.Shape[1];
        Assert.Equal(inDim, x.Length);
        Assert.Equal(outDim, b.Length);
        var wSpan = w.AsSpan();
        var bSpan = b.AsSpan();
        var y = new float[outDim];
        for (int o = 0; o < outDim; o++)
        {
            float acc = bSpan[o];
            for (int i = 0; i < inDim; i++) acc += wSpan[o * inDim + i] * x[i];
            y[o] = acc > 0 ? acc : 0;
        }
        return y;
    }

    [Fact]
    public void StateDict_RoundTrip_HFLlamaNames_RecoversBitExactWeights()
    {
        using var _ = IsolatedTrial();
        var preset = HFStateDictMapper.Get("LLAMA");
        var src = BuildSyntheticState();

        // Map AiDotNet names → HF names so we can write under HF
        // naming, then verify the reverse map recovers what we
        // started with on read.
        var hfNamed = new Dictionary<string, Tensor<float>>();
        foreach (var kv in src)
        {
            // The LLAMA preset's MLP rules use `up_proj` / `down_proj`
            // forward and `mlp.up` / `mlp.down` reverse. We ship
            // canonical HF-style keys for the synthetic state so the
            // load path applies a real rule, not a no-op.
            string hfName = preset.MapReverse(kv.Key);
            Assert.NotEqual(kv.Key, hfName); // a real rewrite, not identity
            hfNamed[hfName] = kv.Value;
        }

        string path = Path.Combine(Path.GetTempPath(), "hf-parity-" + System.Guid.NewGuid().ToString("N") + ".safetensors");
        try
        {
            using (var w = SafetensorsWriter.Create(path))
            {
                foreach (var kv in hfNamed) w.Add(kv.Key, kv.Value);
                w.Save();
            }

            // Load: read every entry, apply forward map, recover the
            // canonical AiDotNet names + bytes.
            using var r = SafetensorsReader.Open(path);
            var recovered = new Dictionary<string, Tensor<float>>();
            foreach (var name in r.Names)
            {
                string aidnName = preset.MapForward(name);
                recovered[aidnName] = r.ReadTensor<float>(name);
            }

            // Same set of names round-trips.
            Assert.Equal(src.Count, recovered.Count);
            foreach (var key in src.Keys) Assert.True(recovered.ContainsKey(key), $"missing {key}");

            // Bit-exact byte equality on every weight tensor.
            foreach (var key in src.Keys)
            {
                var a = src[key].AsSpan();
                var b = recovered[key].AsSpan();
                Assert.Equal(a.Length, b.Length);
                for (int i = 0; i < a.Length; i++) Assert.Equal(a[i], b[i]);
            }
        }
        finally { try { File.Delete(path); } catch { /* best-effort */ } }
    }

    [Fact]
    public void Forward_ReLU_Bf_OnLoadedWeights_MatchesSource()
    {
        using var _ = IsolatedTrial();
        var preset = HFStateDictMapper.Get("LLAMA");
        var src = BuildSyntheticState();

        // Round-trip via HF-named safetensors.
        string path = Path.Combine(Path.GetTempPath(), "hf-parity-fwd-" + System.Guid.NewGuid().ToString("N") + ".safetensors");
        try
        {
            using (var w = SafetensorsWriter.Create(path))
            {
                foreach (var kv in src) w.Add(preset.MapReverse(kv.Key), kv.Value);
                w.Save();
            }

            var loaded = new Dictionary<string, Tensor<float>>();
            using (var r = SafetensorsReader.Open(path))
                foreach (var name in r.Names)
                    loaded[preset.MapForward(name)] = r.ReadTensor<float>(name);

            // Forward pass with a deterministic input on both source
            // and loaded weights — outputs must match bit-for-bit
            // because round-trip is byte-exact.
            float[] x = { 0.5f, -0.25f, 1.0f, -0.75f };

            float[] hSrc = LinearReluForward(x, src["layers[0].mlp.up.weight"], src["layers[0].mlp.up.bias"]);
            float[] hLoaded = LinearReluForward(x, loaded["layers[0].mlp.up.weight"], loaded["layers[0].mlp.up.bias"]);
            Assert.Equal(hSrc.Length, hLoaded.Length);
            for (int i = 0; i < hSrc.Length; i++) Assert.Equal(hSrc[i], hLoaded[i]);

            float[] ySrc = LinearReluForward(hSrc, src["layers[1].mlp.down.weight"], src["layers[1].mlp.down.bias"]);
            float[] yLoaded = LinearReluForward(hLoaded, loaded["layers[1].mlp.down.weight"], loaded["layers[1].mlp.down.bias"]);
            Assert.Equal(ySrc.Length, yLoaded.Length);
            for (int i = 0; i < ySrc.Length; i++) Assert.Equal(ySrc[i], yLoaded[i]);
        }
        finally { try { File.Delete(path); } catch { /* best-effort */ } }
    }

    [Fact]
    public void Sharded_HFLlama_RoundTrip_PreservesWeights()
    {
        using var _ = IsolatedTrial();
        var preset = HFStateDictMapper.Get("LLAMA");
        var src = BuildSyntheticState();

        string dir = Path.Combine(Path.GetTempPath(), "hf-sharded-" + System.Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            // Force at least 2 shards so the multi-shard reader path
            // is exercised on the HF-named load.
            var w = new ShardedSafetensorsWriter(dir, "model", shardSizeBytes: 64);
            foreach (var kv in src)
            {
                int rank = kv.Value.Shape.Length;
                long[] longShape = new long[rank];
                for (int i = 0; i < rank; i++) longShape[i] = kv.Value.Shape[i];
                float[] arr = kv.Value.AsSpan().ToArray();
                byte[] bytes = System.Runtime.InteropServices.MemoryMarshal.AsBytes(arr.AsSpan()).ToArray();
                w.AddRaw(preset.MapReverse(kv.Key), SafetensorsDtype.F32, longShape, bytes);
            }
            int shards = w.Save();
            // Tighter assertion — the test exists to exercise the
            // multi-shard reader path. Accepting "≥ 1" silently
            // passes if a regression collapses everything into a
            // single shard (which a non-sharded reader can also
            // handle).
            Assert.True(shards >= 2,
                $"expected the synthetic checkpoint to span multiple shards; got {shards}.");

            using var rd = ShardedSafetensorsReader.Open(Path.Combine(dir, "model.safetensors.index.json"));
            foreach (var key in src.Keys)
            {
                string hfName = preset.MapReverse(key);
                Assert.True(rd.Entries.ContainsKey(hfName), $"sharded reader missing {hfName}");
                var loaded = rd.ReadTensor<float>(hfName);
                var srcSpan = src[key].AsSpan();
                var loadedSpan = loaded.AsSpan();
                Assert.Equal(srcSpan.Length, loadedSpan.Length);
                for (int i = 0; i < srcSpan.Length; i++) Assert.Equal(srcSpan[i], loadedSpan[i]);
            }
        }
        finally { try { Directory.Delete(dir, recursive: true); } catch { /* best-effort */ } }
    }
}
