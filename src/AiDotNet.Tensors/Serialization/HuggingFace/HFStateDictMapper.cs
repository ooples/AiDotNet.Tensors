// Copyright (c) AiDotNet. All rights reserved.

using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Text.RegularExpressions;

namespace AiDotNet.Tensors.Serialization.HuggingFace;

/// <summary>
/// Converts HuggingFace tensor names (e.g.
/// <c>bert.encoder.layer.0.attention.self.query.weight</c>) into
/// AiDotNet layer-naming convention (e.g.
/// <c>encoder.layers[0].attention.qkv.q.weight</c>) — and back. The
/// <see cref="ApplyMap"/> entry point runs the rules in order; the
/// first matching pattern's substitution wins.
/// </summary>
/// <remarks>
/// <para><b>Why a registry rather than per-model code:</b></para>
/// <para>
/// HF naming differs not just by model but by which port of the model
/// shipped first (early Llama checkpoints used <c>model.layers</c>;
/// the SDXL UNet uses <c>down_blocks.0.resnets.1.norm2.weight</c>).
/// Hardcoding the rewrite in each model class buries the convention
/// inside layer code. A regex-based registry keeps the convention in
/// one place — adding a new preset is one
/// <see cref="HFNamingPreset"/> instance, not a code change in 50
/// layers.
/// </para>
/// <para><b>Bidirectional:</b></para>
/// <para>
/// Each preset captures a forward map (HF → AiDotNet) and a reverse
/// map (AiDotNet → HF). Round-tripping through both should be
/// idempotent for any tensor name in the preset's coverage; tests
/// verify this for every shipped preset.
/// </para>
/// </remarks>
public static class HFStateDictMapper
{
    // ConcurrentDictionary so Register-from-one-thread + Get/AvailablePresets-
    // from-another doesn't trip the underlying Dictionary<,>'s
    // resize/insert race. The registry is read frequently (once per
    // tensor on a state-dict load, ~100s of times for a real HF model)
    // and written rarely (custom Register calls at app startup), so
    // ConcurrentDictionary's lock-free read path is the right tradeoff.
    private static readonly ConcurrentDictionary<string, HFNamingPreset> _presets =
        new(BuildPresets(), StringComparer.OrdinalIgnoreCase);

    /// <summary>The names of every shipped preset.</summary>
    public static IEnumerable<string> AvailablePresets => _presets.Keys;

    /// <summary>
    /// Returns the preset registered under <paramref name="name"/>.
    /// Throws if the name is unknown.
    /// </summary>
    public static HFNamingPreset Get(string name)
    {
        if (name is null) throw new ArgumentNullException(nameof(name));
        if (!_presets.TryGetValue(name, out var preset))
            throw new KeyNotFoundException(
                $"Unknown HF naming preset '{name}'. Available: [{string.Join(", ", _presets.Keys)}].");
        return preset;
    }

    /// <summary>
    /// Registers (or overrides) a preset under <paramref name="name"/>.
    /// Lets callers ship custom mappings for in-house architectures
    /// without forking the library.
    /// </summary>
    public static void Register(string name, HFNamingPreset preset)
    {
        if (name is null) throw new ArgumentNullException(nameof(name));
        if (preset is null) throw new ArgumentNullException(nameof(preset));
        _presets[name] = preset;
    }

    /// <summary>
    /// Applies <paramref name="preset"/>'s forward rules to
    /// <paramref name="hfName"/>, returning the AiDotNet equivalent.
    /// Returns the input unchanged when no rule matches — callers can
    /// detect that case by reference equality.
    /// </summary>
    public static string ApplyForward(string hfName, HFNamingPreset preset)
    {
        if (hfName is null) throw new ArgumentNullException(nameof(hfName));
        if (preset is null) throw new ArgumentNullException(nameof(preset));
        return preset.MapForward(hfName);
    }

    /// <summary>The reverse — AiDotNet name back to HF name.</summary>
    public static string ApplyReverse(string aidnName, HFNamingPreset preset)
    {
        if (aidnName is null) throw new ArgumentNullException(nameof(aidnName));
        if (preset is null) throw new ArgumentNullException(nameof(preset));
        return preset.MapReverse(aidnName);
    }

    private static Dictionary<string, HFNamingPreset> BuildPresets()
    {
        var d = new Dictionary<string, HFNamingPreset>(StringComparer.OrdinalIgnoreCase);

        // BERT — `bert.<...>` prefix on the encoder, `cls.<...>` on
        // the prediction heads. The forward direction strips the
        // `bert.` prefix and rewrites `encoder.layer.{i}` to
        // `encoder.layers[{i}]`.
        d["BERT"] = new HFNamingPreset(
            "BERT",
            new[]
            {
                new HFRule(@"^bert\.encoder\.layer\.(\d+)\.", "encoder.layers[$1]."),
                new HFRule(@"^bert\.embeddings\.", "embeddings."),
                new HFRule(@"^cls\.predictions\.", "mlm_head."),
                // attention.self.{query|key|value} → attention.qkv.{q|k|v}
                new HFRule(@"\.attention\.self\.query\.", ".attention.qkv.q."),
                new HFRule(@"\.attention\.self\.key\.", ".attention.qkv.k."),
                new HFRule(@"\.attention\.self\.value\.", ".attention.qkv.v."),
                new HFRule(@"\.attention\.output\.dense\.", ".attention.out."),
                new HFRule(@"\.attention\.output\.LayerNorm\.", ".attention.norm."),
            });

        // GPT-2 — `transformer.<...>` prefix; per-block layout uses
        // `h.{i}`; QKV is fused in `attn.c_attn.weight` (out-dim ×
        // 3·in-dim). We don't split that here; consumers either keep
        // it fused or split downstream.
        d["GPT2"] = new HFNamingPreset(
            "GPT2",
            new[]
            {
                new HFRule(@"^transformer\.h\.(\d+)\.", "blocks[$1]."),
                new HFRule(@"^transformer\.wte\.", "embedding.token."),
                new HFRule(@"^transformer\.wpe\.", "embedding.pos."),
                new HFRule(@"^transformer\.ln_f\.", "final_norm."),
                new HFRule(@"\.attn\.c_attn\.", ".attention.qkv_fused."),
                new HFRule(@"\.attn\.c_proj\.", ".attention.out."),
                new HFRule(@"\.mlp\.c_fc\.", ".mlp.fc1."),
                new HFRule(@"\.mlp\.c_proj\.", ".mlp.fc2."),
            });

        // Llama (v1/2/3 share names) — `model.<...>` prefix; per-block
        // is `layers.{i}`; rotary embeddings are computed per-call
        // not stored, so no q_proj_inv etc. mapping needed here.
        d["LLAMA"] = new HFNamingPreset(
            "LLAMA",
            new[]
            {
                new HFRule(@"^model\.layers\.(\d+)\.", "layers[$1]."),
                new HFRule(@"^model\.embed_tokens\.", "embedding."),
                new HFRule(@"^model\.norm\.", "final_norm."),
                new HFRule(@"\.self_attn\.q_proj\.", ".attention.q."),
                new HFRule(@"\.self_attn\.k_proj\.", ".attention.k."),
                new HFRule(@"\.self_attn\.v_proj\.", ".attention.v."),
                new HFRule(@"\.self_attn\.o_proj\.", ".attention.out."),
                new HFRule(@"\.mlp\.gate_proj\.", ".mlp.gate."),
                new HFRule(@"\.mlp\.up_proj\.", ".mlp.up."),
                new HFRule(@"\.mlp\.down_proj\.", ".mlp.down."),
                new HFRule(@"\.input_layernorm\.", ".norm1."),
                new HFRule(@"\.post_attention_layernorm\.", ".norm2."),
                new HFRule(@"^lm_head\.", "lm_head."),
            });

        // ViT — image-classification convention (e.g.
        // google/vit-base-patch16-224).
        d["VIT"] = new HFNamingPreset(
            "VIT",
            new[]
            {
                new HFRule(@"^vit\.encoder\.layer\.(\d+)\.", "encoder.layers[$1]."),
                new HFRule(@"^vit\.embeddings\.patch_embeddings\.projection\.", "patch_embed."),
                new HFRule(@"^vit\.embeddings\.position_embeddings$", "pos_embed"),
                new HFRule(@"^vit\.embeddings\.cls_token$", "cls_token"),
                new HFRule(@"^vit\.layernorm\.", "final_norm."),
                new HFRule(@"\.attention\.attention\.query\.", ".attention.q."),
                new HFRule(@"\.attention\.attention\.key\.", ".attention.k."),
                new HFRule(@"\.attention\.attention\.value\.", ".attention.v."),
                new HFRule(@"\.attention\.output\.dense\.", ".attention.out."),
                new HFRule(@"\.layernorm_before\.", ".norm1."),
                new HFRule(@"\.layernorm_after\.", ".norm2."),
                new HFRule(@"\.intermediate\.dense\.", ".mlp.fc1."),
                new HFRule(@"\.output\.dense\.", ".mlp.fc2."),
            });

        return d;
    }
}

/// <summary>
/// One regex-based rewrite rule. Forward direction matches
/// <see cref="HfPattern"/> and substitutes
/// <see cref="AiDotNetReplacement"/>; reverse direction matches
/// <see cref="AiDotNetReplacement"/> as a literal-anchored regex and
/// substitutes the original prefix.
/// </summary>
public sealed class HFRule
{
    /// <summary>The pattern that matches an HF tensor-name fragment.</summary>
    public string HfPattern { get; }

    /// <summary>The substitution applied for forward (HF → AiDotNet) mapping.</summary>
    public string AiDotNetReplacement { get; }

    private readonly Regex _forwardRegex;

    /// <summary>Creates a rule.</summary>
    public HFRule(string hfPattern, string aidnReplacement)
    {
        if (hfPattern is null) throw new ArgumentNullException(nameof(hfPattern));
        if (aidnReplacement is null) throw new ArgumentNullException(nameof(aidnReplacement));
        HfPattern = hfPattern;
        AiDotNetReplacement = aidnReplacement;
        _forwardRegex = new Regex(hfPattern, RegexOptions.Compiled);
    }

    /// <summary>Applies forward replacement.</summary>
    public string ApplyForward(string input) => _forwardRegex.Replace(input, AiDotNetReplacement);

    /// <summary>True iff the forward regex matches anywhere in <paramref name="input"/>.</summary>
    public bool MatchesForward(string input) => _forwardRegex.IsMatch(input);
}

/// <summary>
/// A named bundle of <see cref="HFRule"/>s — one preset per model
/// architecture family.
/// </summary>
public sealed class HFNamingPreset
{
    /// <summary>Display name (BERT / GPT2 / LLAMA / VIT / ...).</summary>
    public string Name { get; }

    /// <summary>Rules in declaration order. First-match-wins per name.</summary>
    public IReadOnlyList<HFRule> Rules { get; }

    /// <summary>Constructs a preset from an ordered rule list.</summary>
    public HFNamingPreset(string name, IReadOnlyList<HFRule> rules)
    {
        if (name is null) throw new ArgumentNullException(nameof(name));
        if (rules is null) throw new ArgumentNullException(nameof(rules));
        Name = name;
        Rules = rules;
    }

    /// <summary>
    /// Maps an HF name to the AiDotNet equivalent — applies every
    /// rule whose pattern matches, in declaration order.
    /// </summary>
    public string MapForward(string hfName)
    {
        if (hfName is null) throw new ArgumentNullException(nameof(hfName));
        string current = hfName;
        foreach (var r in Rules)
            if (r.MatchesForward(current))
                current = r.ApplyForward(current);
        return current;
    }

    /// <summary>
    /// Reverse map — applies the inverse of every forward rule. Note
    /// reversibility is best-effort: rules that collapse two distinct
    /// HF names to the same AiDotNet name (rare in the shipped
    /// presets) cannot be perfectly inverted, so the round-trip
    /// recovers one canonical form.
    /// </summary>
    public string MapReverse(string aidnName)
    {
        if (aidnName is null) throw new ArgumentNullException(nameof(aidnName));
        // Build inverse rules by swapping pattern and replacement —
        // requires the replacement to be regex-safe (no $1 etc.). We
        // ship rules whose replacements have no metacharacters except
        // the back-references that mirror in both directions.
        string current = aidnName;
        foreach (var r in Rules)
        {
            // Convert "$1" / "${1}" backrefs in the replacement into
            // a (\d+) capture group on the inverse pattern.
            string invPattern = Regex.Escape(r.AiDotNetReplacement)
                .Replace(@"\$1", @"(\d+)")
                .Replace(@"\$\{1\}", @"(\d+)");
            // Preserve forward-rule anchors on the inverse pattern.
            // Without this, an exact-match forward rule like
            //   ^vit\.embeddings\.position_embeddings$ -> pos_embed
            // would invert to a substring match on "pos_embed" — every
            // tensor whose AiDotNet name happens to contain "pos_embed"
            // (or "cls_token", or any other non-anchored capture) would
            // be rewritten to the HF form, including names that should
            // pass through unchanged.
            if (r.HfPattern.StartsWith("^", StringComparison.Ordinal))
                invPattern = "^" + invPattern;
            if (r.HfPattern.EndsWith("$", StringComparison.Ordinal))
                invPattern += "$";
            string invReplacement = r.HfPattern
                .Replace(@"(\d+)", "$1")
                // Strip regex anchors that are valid in the forward
                // direction but invalid as a literal in the reverse
                // substitution.
                .Replace("^", "")
                .Replace("$", "");
            // Strip backslash escapes from any literal '.' in the
            // forward pattern so it appears as plain '.' in the
            // emitted HF name.
            invReplacement = invReplacement.Replace(@"\.", ".");
            try
            {
                var rxInv = new Regex(invPattern);
                if (rxInv.IsMatch(current))
                    current = rxInv.Replace(current, invReplacement);
            }
            catch (ArgumentException)
            {
                // Skip rules whose forward pattern uses regex
                // metacharacters we don't safely auto-invert.
            }
        }
        return current;
    }
}
