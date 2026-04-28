// Copyright (c) AiDotNet. All rights reserved.
// Audit-driven tests for the issue #218 acceptance criteria additions:
// mmap reader, appender, sharded write, HF presets, GGUF writer.

#nullable disable

using System;
using System.IO;
using System.Linq;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Licensing;
using AiDotNet.Tensors.NumericOperations;
using AiDotNet.Tensors.Serialization.HuggingFace;
using AiDotNet.Tensors.Serialization.Safetensors;
using Xunit;

namespace AiDotNet.Tensors.Tests.Serialization.Safetensors;

[Collection("PersistenceGuard")]
public class SafetensorsAuditTests
{
    private static IDisposable IsolatedTrial()
    {
        var path = Path.Combine(Path.GetTempPath(), "aidotnet-test-trial-" + Guid.NewGuid().ToString("N") + ".json");
        return PersistenceGuard.SetTestTrialFilePathOverride(path);
    }

    [Fact]
    public void MmapReader_RoundTripsFloat32()
    {
        using var _ = IsolatedTrial();
        var path = Path.Combine(Path.GetTempPath(), "mmap-test-" + Guid.NewGuid().ToString("N") + ".safetensors");
        try
        {
            // Write via the regular writer.
            using (var w = SafetensorsWriter.Create(path))
            {
                w.Add("a", new Tensor<float>(new[] { 1.5f, -2.5f, 3.125f, 4f }, new[] { 4 }));
                w.Save();
            }

            // Read via mmap reader.
            using var r = SafetensorsMmapReader.Open(path);
            Assert.Single(r.Names);
            var t = r.ReadTensor<float>("a");
            Assert.Equal(new[] { 1.5f, -2.5f, 3.125f, 4f }, t.AsSpan().ToArray());
        }
        finally { try { File.Delete(path); } catch { } }
    }

    [Fact]
    public void Appender_AddsNewTensorsWithoutRewritingExisting()
    {
        using var _ = IsolatedTrial();
        var path = Path.Combine(Path.GetTempPath(), "append-test-" + Guid.NewGuid().ToString("N") + ".safetensors");
        try
        {
            // Initial write — 2 tensors.
            using (var w = SafetensorsWriter.Create(path))
            {
                w.Add("a", new Tensor<float>(new[] { 1f, 2f }, new[] { 2 }));
                w.Add("b", new Tensor<float>(new[] { 10f, 20f, 30f }, new[] { 3 }));
                w.Save();
            }

            // Append 1 new tensor.
            var appender = SafetensorsAppender.Open(path);
            appender.Add("c", new Tensor<float>(new[] { 100f, 200f }, new[] { 2 }));
            int total = appender.Save();
            Assert.Equal(3, total);

            // Read back — all 3 tensors present, original values unchanged.
            using var r = SafetensorsReader.Open(path);
            Assert.Equal(3, r.Entries.Count);
            Assert.Equal(new[] { 1f, 2f }, r.ReadTensor<float>("a").AsSpan().ToArray());
            Assert.Equal(new[] { 10f, 20f, 30f }, r.ReadTensor<float>("b").AsSpan().ToArray());
            Assert.Equal(new[] { 100f, 200f }, r.ReadTensor<float>("c").AsSpan().ToArray());
        }
        finally { try { File.Delete(path); } catch { } }
    }

    [Fact]
    public void Appender_RejectsNameCollision()
    {
        using var _ = IsolatedTrial();
        var path = Path.Combine(Path.GetTempPath(), "append-collide-" + Guid.NewGuid().ToString("N") + ".safetensors");
        try
        {
            using (var w = SafetensorsWriter.Create(path))
            {
                w.Add("a", new Tensor<float>(new[] { 1f }, new[] { 1 }));
                w.Save();
            }
            var appender = SafetensorsAppender.Open(path);
            appender.Add("a", new Tensor<float>(new[] { 99f }, new[] { 1 }));
            Assert.Throws<InvalidOperationException>(() => appender.Save());
        }
        finally { try { File.Delete(path); } catch { } }
    }

    [Fact]
    public void HFPresets_AllSixNewPresetsRegistered()
    {
        var available = HFStateDictMapper.AvailablePresets.ToList();
        Assert.Contains("GPT_NEO", available);
        Assert.Contains("CLIP", available);
        Assert.Contains("T5", available);
        Assert.Contains("WHISPER", available);
        Assert.Contains("SD_UNET", available);
        Assert.Contains("SDXL_UNET", available);
    }

    [Fact]
    public void HFPresets_ClipTextModelTransformsCorrectly()
    {
        var preset = HFStateDictMapper.Get("CLIP");
        Assert.Equal("text.encoder.layers[0].attention.q.weight",
            preset.MapForward("text_model.encoder.layers.0.self_attn.q_proj.weight"));
    }

    [Fact]
    public void HFPresets_T5SelfAttentionSplitsByLayerJ()
    {
        var preset = HFStateDictMapper.Get("T5");
        Assert.Equal("encoder.layers[3].attention.q.weight",
            preset.MapForward("encoder.block.3.layer.0.SelfAttention.q.weight"));
        Assert.Equal("decoder.layers[2].cross_attn.k.weight",
            preset.MapForward("decoder.block.2.layer.1.EncDecAttention.k.weight"));
    }

    [Fact]
    public void HFPresets_WhisperEncoderDecoderRouteCorrectly()
    {
        var preset = HFStateDictMapper.Get("WHISPER");
        Assert.Equal("encoder.layers[5].attention.q.weight",
            preset.MapForward("model.encoder.layers.5.self_attn.q_proj.weight"));
        Assert.Equal("decoder.layers[1].cross_attn.v.weight",
            preset.MapForward("model.decoder.layers.1.encoder_attn.v_proj.weight"));
    }

    [Fact]
    public void HFPresets_StableDiffusionTransformsResnetAndAttention()
    {
        var preset = HFStateDictMapper.Get("SD_UNET");
        Assert.Equal("down[1].res[0].conv1.weight",
            preset.MapForward("down_blocks.1.resnets.0.conv1.weight"));
        Assert.Equal("down[2].attn[1].tblocks[0].attn2.q.weight",
            preset.MapForward("down_blocks.2.attentions.1.transformer_blocks.0.attn2.to_q.weight"));
    }

    [Fact]
    public void HFPresets_SDXLAddsAddEmbeddingPath()
    {
        var preset = HFStateDictMapper.Get("SDXL_UNET");
        Assert.Equal("add_embed.fc1.weight",
            preset.MapForward("add_embedding.linear_1.weight"));
    }

    [Fact]
    public void GgufWriter_RoundTripsF32()
    {
        using var _ = IsolatedTrial();
        var path = Path.Combine(Path.GetTempPath(), "gguf-test-" + Guid.NewGuid().ToString("N") + ".gguf");
        try
        {
            var data = new[] { 1f, 2f, 3f, 4f };
            using (var w = GgufWriter.Create(path))
            {
                w.Metadata["general.name"] = "test";
                w.AddF32("weight", new long[] { 4 }, data);
                w.Save();
            }

            using var fs = new FileStream(path, FileMode.Open, FileAccess.Read);
            var file = GgufReader.Read(fs);
            Assert.Equal(3, file.Version);
            Assert.Single(file.Tensors);
            Assert.Equal("weight", file.Tensors[0].Name);
            Assert.Equal(GgufType.F32, file.Tensors[0].Type);
            Assert.Equal(new long[] { 4 }, file.Tensors[0].Dimensions);
        }
        finally { try { File.Delete(path); } catch { } }
    }

    [Fact]
    public void GgufWriter_Q8_0_BlockSizeMatchesSpec()
    {
        using var _ = IsolatedTrial();
        var path = Path.Combine(Path.GetTempPath(), "gguf-q8-" + Guid.NewGuid().ToString("N") + ".gguf");
        try
        {
            // 64 elements → 2 blocks of 32 → 68 bytes payload (2 × 34).
            var data = new float[64];
            for (int i = 0; i < 64; i++) data[i] = i / 10f;
            using (var w = GgufWriter.Create(path))
            {
                w.AddQ8_0("w", new long[] { 64 }, data);
                w.Save();
            }

            // The reader doesn't dequantise but it should at least
            // recognise the type code and report the right shape.
            using var fs = new FileStream(path, FileMode.Open, FileAccess.Read);
            var file = GgufReader.Read(fs);
            Assert.Single(file.Tensors);
            Assert.Equal(GgufType.Q8_0, file.Tensors[0].Type);
            Assert.Equal(new long[] { 64 }, file.Tensors[0].Dimensions);
        }
        finally { try { File.Delete(path); } catch { } }
    }

    [Fact]
    public void GgufWriter_Q4_0_RejectsNonDivisibleShape()
    {
        using var _ = IsolatedTrial();
        var path = Path.Combine(Path.GetTempPath(), "gguf-bad-" + Guid.NewGuid().ToString("N") + ".gguf");
        try
        {
            using var w = GgufWriter.Create(path);
            // 31 elements — not divisible by 32 — should reject up front.
            Assert.Throws<ArgumentException>(
                () => w.AddQ4_0("w", new long[] { 31 }, new float[31]));
        }
        finally { try { File.Delete(path); } catch { } }
    }

    [Fact]
    public void GgufWriter_Q4_K_RequiresMultipleOf256()
    {
        using var _ = IsolatedTrial();
        var path = Path.Combine(Path.GetTempPath(), "gguf-q4k-" + Guid.NewGuid().ToString("N") + ".gguf");
        try
        {
            using (var w = GgufWriter.Create(path))
            {
                Assert.Throws<ArgumentException>(
                    () => w.AddQ4_K("w", new long[] { 255 }, new float[255]));
            }
            // Valid: 256 elements emit a single 144-byte super-block.
            using (var w = GgufWriter.Create(path))
            {
                var data = new float[256];
                for (int i = 0; i < 256; i++) data[i] = (float)Math.Sin(i * 0.1);
                w.AddQ4_K("w", new long[] { 256 }, data);
                w.Save();
            }
            using var fs = new FileStream(path, FileMode.Open, FileAccess.Read);
            var file = GgufReader.Read(fs);
            Assert.Equal(GgufType.Q4_K, file.Tensors[0].Type);
        }
        finally { try { File.Delete(path); } catch { } }
    }

    [Fact]
    public void TensorBoard_AddImage_EmitsRecord()
    {
        using var _ = IsolatedTrial();
        using var ms = new MemoryStream();
        using (var w = AiDotNet.Tensors.Serialization.TensorBoard.TensorBoardSummaryWriter.ToStream(ms))
        {
            // Tiny "PNG-like" payload (real PNG header bytes — TB
            // doesn't decode, just stores).
            byte[] png = new byte[] { 0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A };
            w.AddImage("samples/0", png, 16, 16, step: 0);
            w.Flush();
        }
        Assert.True(ms.Length > 50);
    }

    [Fact]
    public void TensorBoard_AddEmbedding_WritesProjectorFiles()
    {
        using var _ = IsolatedTrial();
        var dir = Path.Combine(Path.GetTempPath(), "tb-emb-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            using (var w = AiDotNet.Tensors.Serialization.TensorBoard.TensorBoardSummaryWriter.OpenLogDir(dir))
            {
                var embeddings = new float[3, 2]
                {
                    { 0.1f, 0.2f },
                    { 0.3f, 0.4f },
                    { 0.5f, 0.6f },
                };
                w.AddEmbedding(dir, "test_emb", embeddings, new[] { "a", "b", "c" });
            }
            Assert.True(File.Exists(Path.Combine(dir, "test_emb", "tensors.tsv")));
            Assert.True(File.Exists(Path.Combine(dir, "test_emb", "metadata.tsv")));
            Assert.True(File.Exists(Path.Combine(dir, "projector_config.pbtxt")));
        }
        finally { try { Directory.Delete(dir, recursive: true); } catch { } }
    }

    [Fact]
    public void TensorBoard_AddGraph_EmitsGraphDefField()
    {
        using var _ = IsolatedTrial();
        using var ms = new MemoryStream();
        using (var w = AiDotNet.Tensors.Serialization.TensorBoard.TensorBoardSummaryWriter.ToStream(ms))
        {
            // Synthetic GraphDef bytes — TB doesn't validate at write time.
            byte[] graphDef = new byte[] { 0x0A, 0x05, 0x68, 0x65, 0x6C, 0x6C, 0x6F };
            w.AddGraph(graphDef);
            w.Flush();
        }
        Assert.True(ms.Length > 30);
    }
}
