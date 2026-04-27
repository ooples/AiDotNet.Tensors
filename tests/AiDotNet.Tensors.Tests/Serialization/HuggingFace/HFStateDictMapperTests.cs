// Copyright (c) AiDotNet. All rights reserved.

#nullable disable

using System.Linq;
using AiDotNet.Tensors.Serialization.HuggingFace;
using Xunit;

namespace AiDotNet.Tensors.Tests.Serialization.HuggingFace;

public class HFStateDictMapperTests
{
    [Fact]
    public void AvailablePresets_ContainsAllShipped()
    {
        var presets = HFStateDictMapper.AvailablePresets.ToList();
        Assert.Contains("BERT", presets);
        Assert.Contains("GPT2", presets);
        Assert.Contains("LLAMA", presets);
        Assert.Contains("VIT", presets);
    }

    [Fact]
    public void Get_UnknownPreset_Throws()
    {
        Assert.Throws<System.Collections.Generic.KeyNotFoundException>(
            () => HFStateDictMapper.Get("DOES_NOT_EXIST"));
    }

    [Fact]
    public void BERT_QueryWeight_RewrittenToQkvQ()
    {
        var preset = HFStateDictMapper.Get("BERT");
        Assert.Equal(
            "encoder.layers[0].attention.qkv.q.weight",
            preset.MapForward("bert.encoder.layer.0.attention.self.query.weight"));
    }

    [Fact]
    public void BERT_LayerNormBias_RewrittenToNormBias()
    {
        var preset = HFStateDictMapper.Get("BERT");
        Assert.Equal(
            "encoder.layers[3].attention.norm.bias",
            preset.MapForward("bert.encoder.layer.3.attention.output.LayerNorm.bias"));
    }

    [Fact]
    public void Llama_LayerWeights_RewriteCorrectly()
    {
        var preset = HFStateDictMapper.Get("LLAMA");
        Assert.Equal(
            "layers[0].attention.q.weight",
            preset.MapForward("model.layers.0.self_attn.q_proj.weight"));
        Assert.Equal(
            "layers[5].mlp.gate.weight",
            preset.MapForward("model.layers.5.mlp.gate_proj.weight"));
        Assert.Equal(
            "embedding.weight",
            preset.MapForward("model.embed_tokens.weight"));
    }

    [Fact]
    public void GPT2_FusedQkv_PreservedAsFused()
    {
        var preset = HFStateDictMapper.Get("GPT2");
        // GPT-2's attention is fused; we don't try to split it in
        // the mapper — keep the fused name so callers route the bytes
        // intact.
        Assert.Equal(
            "blocks[0].attention.qkv_fused.weight",
            preset.MapForward("transformer.h.0.attn.c_attn.weight"));
    }

    [Fact]
    public void ViT_PatchEmbed_RewriteCorrectly()
    {
        var preset = HFStateDictMapper.Get("VIT");
        Assert.Equal(
            "patch_embed.weight",
            preset.MapForward("vit.embeddings.patch_embeddings.projection.weight"));
        Assert.Equal(
            "encoder.layers[2].attention.q.weight",
            preset.MapForward("vit.encoder.layer.2.attention.attention.query.weight"));
    }

    [Fact]
    public void Unknown_NamePassesThroughUnchanged()
    {
        var preset = HFStateDictMapper.Get("BERT");
        // No rule matches → return as-is.
        var result = preset.MapForward("custom.unknown.tensor");
        Assert.Equal("custom.unknown.tensor", result);
    }

    [Fact]
    public void Register_CustomPreset_RoundTrips()
    {
        var custom = new HFNamingPreset("CUSTOM_TEST",
            new[]
            {
                new HFRule(@"^old\.", "new."),
            });
        HFStateDictMapper.Register("CUSTOM_TEST", custom);
        var got = HFStateDictMapper.Get("CUSTOM_TEST");
        Assert.Equal("new.layer.weight", got.MapForward("old.layer.weight"));
    }
}
