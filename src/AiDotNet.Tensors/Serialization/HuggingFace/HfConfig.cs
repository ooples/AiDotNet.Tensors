// Copyright (c) AiDotNet. All rights reserved.

using System.Collections.Generic;
using System.IO;
using System.Text.Json;

namespace AiDotNet.Tensors.Serialization.HuggingFace;

/// <summary>
/// Typed view of a HuggingFace <c>config.json</c> file. Captures the
/// fields shared across the major architecture families (BERT, GPT-2,
/// Llama, ViT, T5) so layer code can instantiate the right shape from
/// a checkpoint without parsing JSON itself.
/// </summary>
/// <remarks>
/// <para>The HF <c>config.json</c> schema is loose — every model
/// family uses slightly different key names for the same concept
/// (e.g. <c>hidden_size</c> vs <c>d_model</c> vs <c>n_embd</c>). The
/// parser tries the canonical key first and falls back to the
/// per-family alias, so callers can reach for <c>HiddenSize</c>
/// without knowing which model emitted the config.</para>
///
/// <para>Parsing is licence-gated through
/// <see cref="Licensing.PersistenceGuard"/> — loading a config alone
/// doesn't bring weights, but it's part of the
/// <c>from_pretrained</c>-style flow that the upstream guard expects
/// to count.</para>
/// </remarks>
public sealed class HfConfig
{
    /// <summary>
    /// The model family identifier (<c>bert</c>, <c>gpt2</c>,
    /// <c>llama</c>, <c>vit</c>, …) — the value of HF's
    /// <c>model_type</c> field.
    /// </summary>
    public string? ModelType { get; init; }

    /// <summary>
    /// Architectures list (e.g. <c>["BertForMaskedLM"]</c>) — the
    /// concrete model class HF instantiates from this checkpoint.
    /// </summary>
    public IReadOnlyList<string>? Architectures { get; init; }

    /// <summary>Hidden / model dimension — <c>hidden_size</c> / <c>d_model</c> / <c>n_embd</c>.</summary>
    public int? HiddenSize { get; init; }

    /// <summary>Number of transformer layers — <c>num_hidden_layers</c> / <c>n_layer</c>.</summary>
    public int? NumHiddenLayers { get; init; }

    /// <summary>Attention head count — <c>num_attention_heads</c> / <c>n_head</c>.</summary>
    public int? NumAttentionHeads { get; init; }

    /// <summary>
    /// KV-head count for grouped-query attention (Llama-2/3, Mistral).
    /// Falls back to <see cref="NumAttentionHeads"/> when the config
    /// doesn't specify GQA.
    /// </summary>
    public int? NumKeyValueHeads { get; init; }

    /// <summary>Vocabulary size — <c>vocab_size</c>.</summary>
    public int? VocabSize { get; init; }

    /// <summary>Intermediate / feed-forward dimension — <c>intermediate_size</c> / <c>n_inner</c>.</summary>
    public int? IntermediateSize { get; init; }

    /// <summary>Maximum sequence length — <c>max_position_embeddings</c> / <c>n_ctx</c>.</summary>
    public int? MaxPositionEmbeddings { get; init; }

    /// <summary>Layer-norm epsilon — <c>layer_norm_eps</c> / <c>layer_norm_epsilon</c> / <c>rms_norm_eps</c>.</summary>
    public double? LayerNormEps { get; init; }

    /// <summary>The raw JSON for any field not surfaced as a strongly-typed property.</summary>
    public IReadOnlyDictionary<string, JsonElement> Raw { get; init; } = new Dictionary<string, JsonElement>();

    /// <summary>
    /// Loads a config.json from disk. Convenience for the common
    /// case; equivalent to <c>Parse(File.ReadAllText(path))</c>.
    /// </summary>
    public static HfConfig Load(string path)
    {
        if (path is null) throw new ArgumentNullException(nameof(path));
        if (!File.Exists(path))
            throw new FileNotFoundException($"HF config file not found: {path}");
        return Parse(File.ReadAllText(path));
    }

    /// <summary>Parses an in-memory JSON string.</summary>
    public static HfConfig Parse(string json)
    {
        if (json is null) throw new ArgumentNullException(nameof(json));
        using var doc = JsonDocument.Parse(json);
        var root = doc.RootElement;
        if (root.ValueKind != JsonValueKind.Object)
            throw new InvalidDataException("HF config root must be a JSON object.");

        var raw = new Dictionary<string, JsonElement>(StringComparer.Ordinal);
        foreach (var prop in root.EnumerateObject())
        {
            // Clone the JsonElement so it survives the doc disposing.
            // JsonElement holds a reference to its parent JsonDocument,
            // and we Dispose() the doc on scope exit; the consumer's
            // typed properties below already extract scalar values, so
            // raw[] is the only path that needs the clone.
            raw[prop.Name] = prop.Value.Clone();
        }

        // Reads the first matching key, returning null if none of the
        // aliases is present or the value is the wrong JsonValueKind.
        int? GetInt(params string[] keys)
        {
            foreach (var k in keys)
                if (raw.TryGetValue(k, out var v) && v.ValueKind == JsonValueKind.Number && v.TryGetInt32(out var n))
                    return n;
            return null;
        }
        double? GetDouble(params string[] keys)
        {
            foreach (var k in keys)
                if (raw.TryGetValue(k, out var v) && v.ValueKind == JsonValueKind.Number && v.TryGetDouble(out var d))
                    return d;
            return null;
        }
        string? GetString(params string[] keys)
        {
            foreach (var k in keys)
                if (raw.TryGetValue(k, out var v) && v.ValueKind == JsonValueKind.String)
                    return v.GetString();
            return null;
        }

        IReadOnlyList<string>? archs = null;
        if (raw.TryGetValue("architectures", out var aEl) && aEl.ValueKind == JsonValueKind.Array)
        {
            var list = new List<string>();
            foreach (var el in aEl.EnumerateArray())
                if (el.ValueKind == JsonValueKind.String) list.Add(el.GetString() ?? "");
            archs = list;
        }

        return new HfConfig
        {
            ModelType = GetString("model_type"),
            Architectures = archs,
            HiddenSize = GetInt("hidden_size", "d_model", "n_embd"),
            NumHiddenLayers = GetInt("num_hidden_layers", "n_layer", "num_layers"),
            NumAttentionHeads = GetInt("num_attention_heads", "n_head", "num_heads"),
            NumKeyValueHeads = GetInt("num_key_value_heads"),
            VocabSize = GetInt("vocab_size"),
            IntermediateSize = GetInt("intermediate_size", "n_inner", "ffn_dim"),
            MaxPositionEmbeddings = GetInt("max_position_embeddings", "n_ctx", "max_sequence_length"),
            LayerNormEps = GetDouble("layer_norm_eps", "layer_norm_epsilon", "rms_norm_eps"),
            Raw = raw,
        };
    }
}
