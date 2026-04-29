// Copyright (c) AiDotNet. All rights reserved.

#nullable disable

using System.Collections.Generic;
using System.Linq;
using AiDotNet.Tensors.Serialization.HuggingFace;
using Xunit;

namespace AiDotNet.Tensors.Tests.Serialization.HuggingFace;

/// <summary>
/// Per-preset coverage sweep — every shipped HF preset is exercised
/// against a curated list of representative tensor names drawn from
/// the preset's documented architecture family. The contract checked
/// is forward-then-reverse idempotence: <c>MapReverse(MapForward(x))</c>
/// must recover <c>x</c> for every supported HF name. Without this
/// sweep, a regression in any single preset's rules can silently
/// corrupt a state-dict load that <see cref="HFStateDictMapperTests"/>
/// happens not to hit because that suite only spot-checks one or two
/// names per preset.
/// </summary>
public class HFPresetBijectionSweepTests
{
    public static IEnumerable<object[]> AllPresetSamples()
    {
        // Each entry is (preset_name, hf_tensor_name) covering at
        // least one rule per category in the preset's rule set.
        var samples = new (string Preset, string Hf)[]
        {
            // BERT — encoder block, embeddings, prediction head
            ("BERT", "bert.encoder.layer.0.attention.self.query.weight"),
            ("BERT", "bert.encoder.layer.7.attention.self.key.bias"),
            ("BERT", "bert.encoder.layer.3.attention.self.value.weight"),
            ("BERT", "bert.encoder.layer.2.attention.output.dense.weight"),
            ("BERT", "bert.encoder.layer.5.attention.output.LayerNorm.bias"),
            ("BERT", "bert.embeddings.word_embeddings.weight"),
            ("BERT", "cls.predictions.transform.dense.weight"),

            // GPT2 — block, fused qkv, mlp
            ("GPT2", "transformer.h.0.attn.c_attn.weight"),
            ("GPT2", "transformer.h.4.attn.c_proj.bias"),
            ("GPT2", "transformer.h.11.mlp.c_fc.weight"),
            ("GPT2", "transformer.h.11.mlp.c_proj.weight"),
            ("GPT2", "transformer.wte.weight"),
            ("GPT2", "transformer.wpe.weight"),
            ("GPT2", "transformer.ln_f.bias"),

            // LLAMA — qkv split, mlp split, layernorms, lm_head
            ("LLAMA", "model.layers.0.self_attn.q_proj.weight"),
            ("LLAMA", "model.layers.31.self_attn.k_proj.weight"),
            ("LLAMA", "model.layers.7.self_attn.v_proj.weight"),
            ("LLAMA", "model.layers.7.self_attn.o_proj.weight"),
            ("LLAMA", "model.layers.5.mlp.gate_proj.weight"),
            ("LLAMA", "model.layers.5.mlp.up_proj.weight"),
            ("LLAMA", "model.layers.5.mlp.down_proj.weight"),
            ("LLAMA", "model.layers.0.input_layernorm.weight"),
            ("LLAMA", "model.layers.0.post_attention_layernorm.weight"),
            ("LLAMA", "model.embed_tokens.weight"),
            ("LLAMA", "model.norm.weight"),
            ("LLAMA", "lm_head.weight"),

            // VIT — patch embed, encoder block, layernorm, pos embed
            ("VIT", "vit.embeddings.patch_embeddings.projection.weight"),
            ("VIT", "vit.embeddings.position_embeddings"),
            ("VIT", "vit.embeddings.cls_token"),
            ("VIT", "vit.encoder.layer.2.attention.attention.query.weight"),
            ("VIT", "vit.encoder.layer.2.attention.attention.key.bias"),
            ("VIT", "vit.encoder.layer.2.attention.attention.value.weight"),
            ("VIT", "vit.encoder.layer.2.attention.output.dense.weight"),
            ("VIT", "vit.encoder.layer.5.layernorm_before.weight"),
            ("VIT", "vit.encoder.layer.5.layernorm_after.weight"),
            ("VIT", "vit.encoder.layer.5.intermediate.dense.weight"),
            ("VIT", "vit.encoder.layer.5.output.dense.bias"),
            ("VIT", "vit.layernorm.weight"),

            // GPT_NEO — double-attention prefix, ln_1/ln_2
            ("GPT_NEO", "transformer.h.0.attn.attention.q_proj.weight"),
            ("GPT_NEO", "transformer.h.0.attn.attention.k_proj.weight"),
            ("GPT_NEO", "transformer.h.0.attn.attention.v_proj.weight"),
            ("GPT_NEO", "transformer.h.0.attn.attention.out_proj.bias"),
            ("GPT_NEO", "transformer.h.3.ln_1.weight"),
            ("GPT_NEO", "transformer.h.3.ln_2.bias"),
            ("GPT_NEO", "transformer.h.7.mlp.c_fc.weight"),
            ("GPT_NEO", "transformer.h.7.mlp.c_proj.weight"),
            ("GPT_NEO", "transformer.wte.weight"),

            // CLIP — text + vision sub-towers, projections
            ("CLIP", "text_model.encoder.layers.0.self_attn.q_proj.weight"),
            ("CLIP", "text_model.encoder.layers.5.self_attn.k_proj.bias"),
            ("CLIP", "text_model.encoder.layers.5.layer_norm1.weight"),
            ("CLIP", "text_model.encoder.layers.5.layer_norm2.weight"),
            ("CLIP", "text_model.encoder.layers.5.mlp.fc1.weight"),
            ("CLIP", "text_model.encoder.layers.5.mlp.fc2.bias"),
            ("CLIP", "text_model.embeddings.token_embedding.weight"),
            ("CLIP", "text_model.embeddings.position_embedding.weight"),
            ("CLIP", "text_model.final_layer_norm.weight"),
            ("CLIP", "vision_model.encoder.layers.0.self_attn.v_proj.weight"),
            ("CLIP", "vision_model.embeddings.patch_embedding.weight"),
            ("CLIP", "vision_model.embeddings.class_embedding"),
            ("CLIP", "text_projection.weight"),
            ("CLIP", "visual_projection.weight"),

            // T5 — encoder/decoder split, self/cross/enc_dec attention
            ("T5", "encoder.block.0.layer.0.SelfAttention.q.weight"),
            ("T5", "encoder.block.5.layer.0.SelfAttention.k.weight"),
            ("T5", "encoder.block.5.layer.0.SelfAttention.v.weight"),
            ("T5", "encoder.block.5.layer.0.SelfAttention.o.weight"),
            ("T5", "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"),
            ("T5", "encoder.block.5.layer.0.layer_norm.weight"),
            ("T5", "encoder.block.5.layer.1.DenseReluDense.wi.weight"),
            ("T5", "encoder.block.5.layer.1.DenseReluDense.wo.weight"),
            ("T5", "encoder.block.5.layer.1.layer_norm.weight"),
            ("T5", "decoder.block.0.layer.0.SelfAttention.q.weight"),
            ("T5", "decoder.block.0.layer.1.EncDecAttention.q.weight"),
            ("T5", "decoder.block.0.layer.1.EncDecAttention.k.weight"),
            ("T5", "decoder.block.0.layer.2.DenseReluDense.wi.weight"),
            ("T5", "decoder.block.0.layer.2.DenseReluDense.wo.weight"),
            ("T5", "encoder.final_layer_norm.weight"),
            ("T5", "decoder.final_layer_norm.weight"),
            ("T5", "shared.weight"),
            ("T5", "lm_head.weight"),

            // WHISPER — encoder conv-stem + decoder
            ("WHISPER", "model.encoder.layers.0.self_attn.q_proj.weight"),
            ("WHISPER", "model.encoder.layers.5.self_attn.k_proj.bias"),
            ("WHISPER", "model.encoder.layers.5.self_attn.v_proj.weight"),
            ("WHISPER", "model.encoder.layers.5.self_attn.out_proj.weight"),
            ("WHISPER", "model.encoder.layers.5.self_attn_layer_norm.weight"),
            ("WHISPER", "model.encoder.layers.5.fc1.weight"),
            ("WHISPER", "model.encoder.layers.5.fc2.weight"),
            ("WHISPER", "model.encoder.layers.5.final_layer_norm.weight"),
            ("WHISPER", "model.encoder.conv1.weight"),
            ("WHISPER", "model.encoder.conv2.weight"),
            ("WHISPER", "model.encoder.embed_positions.weight"),
            ("WHISPER", "model.encoder.layer_norm.weight"),
            ("WHISPER", "model.decoder.layers.0.encoder_attn.q_proj.weight"),
            ("WHISPER", "model.decoder.layers.0.encoder_attn.k_proj.weight"),
            ("WHISPER", "model.decoder.layers.0.encoder_attn_layer_norm.weight"),
            ("WHISPER", "model.decoder.embed_tokens.weight"),
            ("WHISPER", "model.decoder.embed_positions.weight"),
            ("WHISPER", "model.decoder.layer_norm.weight"),
            ("WHISPER", "proj_out.weight"),

            // SD_UNET — down/up/mid blocks, transformer blocks, time embed
            ("SD_UNET", "down_blocks.0.resnets.1.norm1.weight"),
            ("SD_UNET", "down_blocks.0.resnets.1.norm2.weight"),
            ("SD_UNET", "down_blocks.0.resnets.1.conv1.weight"),
            ("SD_UNET", "down_blocks.0.resnets.1.conv2.weight"),
            ("SD_UNET", "down_blocks.0.resnets.1.time_emb_proj.weight"),
            ("SD_UNET", "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.weight"),
            ("SD_UNET", "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_k.weight"),
            ("SD_UNET", "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_v.weight"),
            ("SD_UNET", "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_out.0.weight"),
            ("SD_UNET", "down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_q.weight"),
            ("SD_UNET", "down_blocks.0.attentions.0.transformer_blocks.0.norm1.weight"),
            ("SD_UNET", "down_blocks.0.attentions.0.transformer_blocks.0.norm2.weight"),
            ("SD_UNET", "down_blocks.0.attentions.0.transformer_blocks.0.norm3.weight"),
            ("SD_UNET", "down_blocks.0.attentions.0.proj_in.weight"),
            ("SD_UNET", "down_blocks.0.attentions.0.proj_out.weight"),
            ("SD_UNET", "down_blocks.0.downsamplers.0.conv.weight"),
            ("SD_UNET", "up_blocks.1.resnets.0.conv1.weight"),
            ("SD_UNET", "up_blocks.1.upsamplers.0.conv.weight"),
            ("SD_UNET", "mid_block.resnets.0.conv1.weight"),
            ("SD_UNET", "mid_block.attentions.0.proj_in.weight"),
            ("SD_UNET", "time_embedding.linear_1.weight"),
            ("SD_UNET", "time_embedding.linear_2.weight"),
            ("SD_UNET", "conv_in.weight"),
            ("SD_UNET", "conv_out.weight"),
            ("SD_UNET", "conv_norm_out.weight"),

            // SDXL_UNET — adds add_embedding plus inherits SD_UNET
            ("SDXL_UNET", "add_embedding.linear_1.weight"),
            ("SDXL_UNET", "add_embedding.linear_2.bias"),
            ("SDXL_UNET", "down_blocks.0.resnets.1.conv1.weight"),
            ("SDXL_UNET", "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.weight"),
            ("SDXL_UNET", "time_embedding.linear_1.weight"),
            ("SDXL_UNET", "conv_in.weight"),
        };

        return samples.Select(s => new object[] { s.Preset, s.Hf });
    }

    /// <summary>
    /// For every supported HF tensor name, the forward-then-reverse
    /// chain must recover the original. Failure means a regex
    /// asymmetry between the forward rule and the inverse derivation
    /// — usually a forward replacement that introduces a character
    /// the inverse derivation can't reconstruct.
    /// </summary>
    [Theory]
    [MemberData(nameof(AllPresetSamples))]
    public void ForwardThenReverse_RoundTripsToOriginal(string presetName, string hfName)
    {
        // Only the forward-then-reverse round-trip is asserted. Some
        // shipped rules are deliberately identity rewrites (e.g.
        // SD_UNET's `^conv_in\.` → `conv_in.`) — they exist as anchors
        // for the inverse derivation rather than to change the name.
        // Asserting forward != input would reject those legitimate
        // cases and turn this sweep into a churn-detector.
        var preset = HFStateDictMapper.Get(presetName);
        string forward = preset.MapForward(hfName);
        string roundTrip = preset.MapReverse(forward);
        Assert.Equal(hfName, roundTrip);
    }

    /// <summary>
    /// All 10 presets ship — guarantees every preset documented in the
    /// issue-218 acceptance gate is registered. New presets must be
    /// added here so they don't ship undocumented.
    /// </summary>
    [Fact]
    public void AllShippedPresets_AreRegistered()
    {
        // Exact-match assertion (not subset) so a new SHIPPED preset
        // (one that lands in src/) without being documented here
        // fails the test. Test-only registrations (suffix "_TEST")
        // are filtered out — those are added by other tests and
        // would otherwise pollute the registry under shared-process
        // xunit collection runs.
        var registered = HFStateDictMapper.AvailablePresets
            .Where(name => !name.EndsWith("_TEST"))
            .OrderBy(x => x)
            .ToList();
        var expected = new[] { "BERT", "GPT2", "LLAMA", "VIT", "GPT_NEO", "CLIP", "T5", "WHISPER", "SD_UNET", "SDXL_UNET" }
            .OrderBy(x => x).ToList();
        Assert.Equal(expected, registered);
    }
}
