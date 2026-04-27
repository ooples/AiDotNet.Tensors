// Copyright (c) AiDotNet. All rights reserved.

#nullable disable

using AiDotNet.Tensors.Serialization.HuggingFace;
using Xunit;

namespace AiDotNet.Tensors.Tests.Serialization.HuggingFace;

public class HfConfigTests
{
    [Fact]
    public void Parses_BertBaseConfig()
    {
        // Snippet of a real bert-base-uncased config.json
        string json = """
        {
          "model_type": "bert",
          "architectures": ["BertForMaskedLM"],
          "hidden_size": 768,
          "num_hidden_layers": 12,
          "num_attention_heads": 12,
          "vocab_size": 30522,
          "intermediate_size": 3072,
          "max_position_embeddings": 512,
          "layer_norm_eps": 1e-12
        }
        """;
        var c = HfConfig.Parse(json);
        Assert.Equal("bert", c.ModelType);
        Assert.Equal(new[] { "BertForMaskedLM" }, c.Architectures);
        Assert.Equal(768, c.HiddenSize);
        Assert.Equal(12, c.NumHiddenLayers);
        Assert.Equal(12, c.NumAttentionHeads);
        Assert.Equal(30522, c.VocabSize);
        Assert.Equal(3072, c.IntermediateSize);
        Assert.Equal(512, c.MaxPositionEmbeddings);
        Assert.Equal(1e-12, c.LayerNormEps);
    }

    [Fact]
    public void Parses_GPT2_AlternateKeyNames()
    {
        // GPT-2 uses n_embd / n_layer / n_head / n_inner / n_ctx
        string json = """
        {
          "model_type": "gpt2",
          "n_embd": 768,
          "n_layer": 12,
          "n_head": 12,
          "n_inner": 3072,
          "n_ctx": 1024,
          "layer_norm_epsilon": 1e-5,
          "vocab_size": 50257
        }
        """;
        var c = HfConfig.Parse(json);
        Assert.Equal("gpt2", c.ModelType);
        Assert.Equal(768, c.HiddenSize);                  // via n_embd
        Assert.Equal(12, c.NumHiddenLayers);              // via n_layer
        Assert.Equal(12, c.NumAttentionHeads);            // via n_head
        Assert.Equal(3072, c.IntermediateSize);           // via n_inner
        Assert.Equal(1024, c.MaxPositionEmbeddings);      // via n_ctx
        Assert.Equal(1e-5, c.LayerNormEps);               // via layer_norm_epsilon
    }

    [Fact]
    public void Parses_LlamaWithGqa()
    {
        string json = """
        {
          "model_type": "llama",
          "hidden_size": 4096,
          "num_hidden_layers": 32,
          "num_attention_heads": 32,
          "num_key_value_heads": 8,
          "intermediate_size": 11008,
          "rms_norm_eps": 1e-5,
          "vocab_size": 32000
        }
        """;
        var c = HfConfig.Parse(json);
        Assert.Equal(32, c.NumAttentionHeads);
        Assert.Equal(8, c.NumKeyValueHeads);
        Assert.Equal(1e-5, c.LayerNormEps);  // via rms_norm_eps
    }

    [Fact]
    public void Raw_PreservesUnknownFields()
    {
        string json = """
        {
          "model_type": "custom",
          "rope_theta": 500000.0,
          "tie_word_embeddings": false,
          "torch_dtype": "bfloat16"
        }
        """;
        var c = HfConfig.Parse(json);
        Assert.True(c.Raw.ContainsKey("rope_theta"));
        Assert.True(c.Raw.ContainsKey("tie_word_embeddings"));
        Assert.True(c.Raw.ContainsKey("torch_dtype"));
        // Typed fields not present return null.
        Assert.Null(c.HiddenSize);
        Assert.Null(c.VocabSize);
    }

    [Fact]
    public void RejectsNonObjectRoot()
    {
        Assert.Throws<System.IO.InvalidDataException>(() => HfConfig.Parse("[1, 2, 3]"));
    }
}
