// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace AiDotNet.Tensors.Licensing;

/// <summary>
/// The signed claim set carried by an <c>aidn2.</c> asymmetric license token — the Tensors port of the
/// main SDK's <c>LicenseClaims</c>, parsed/serialized with <see cref="System.Text.Json"/> (Tensors does
/// not reference Newtonsoft).
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> A license token is a small block of JSON (these fields) plus a
/// cryptographic signature over that exact JSON. Because the signature can only be produced with the
/// server's private key, nobody can change any field (e.g. bump <c>exp</c> or upgrade <c>tier</c>)
/// without invalidating the signature.</para>
/// <para>Field names are intentionally short (JWT-style) to keep tokens compact.</para>
/// </remarks>
internal sealed class Aidn2Claims
{
    /// <summary>Subject — the customer / license the token was issued to.</summary>
    [JsonPropertyName("sub")]
    public string? Sub { get; set; }

    /// <summary>Tier: <c>community</c>, <c>pro</c>, or <c>enterprise</c>.</summary>
    [JsonPropertyName("tier")]
    public string? Tier { get; set; }

    /// <summary>Number of licensed seats.</summary>
    [JsonPropertyName("seats")]
    public int Seats { get; set; }

    /// <summary>Issued-at, Unix seconds (UTC).</summary>
    [JsonPropertyName("iat")]
    public long Iat { get; set; }

    /// <summary>Expiry, Unix seconds (UTC). Bounds offline use — a leaked token self-expires.</summary>
    [JsonPropertyName("exp")]
    public long Exp { get; set; }

    /// <summary>Key id — selects which embedded public key verifies this token (rotation).</summary>
    [JsonPropertyName("kid")]
    public string? Kid { get; set; }

    /// <summary>
    /// Signature algorithm identifier. Always <c>"EdDSA"</c> for aidn2 tokens (Ed25519). Recorded in the
    /// claims so a future primitive change / rotation is unambiguous — the verifier rejects any token
    /// whose <c>alg</c> it does not implement rather than silently assuming Ed25519.
    /// </summary>
    [JsonPropertyName("alg")]
    public string? Alg { get; set; }

    /// <summary>
    /// Unique token id. The revocation deny-list (CRL) names revoked <c>jti</c> values, so a specific
    /// leaked token can be killed before its <c>exp</c> without revoking the whole signing key.
    /// </summary>
    [JsonPropertyName("jti")]
    public string? Jti { get; set; }

    /// <summary>
    /// Explicit capability grants (e.g. <c>tensors:save</c>, <c>tensors:load</c>). Authoritative for
    /// offline capability gating: the persistence guard checks these rather than assuming any
    /// <c>Active</c> license unlocks everything.
    /// </summary>
    [JsonPropertyName("caps")]
    public string[]? Caps { get; set; }

    /// <summary>
    /// Optional machine-binding fingerprint (node-lock). When present, the verifier requires it to equal
    /// this machine's id hash, so a leaked customer token is useless on another machine.
    /// </summary>
    [JsonPropertyName("mach")]
    public string? Mach { get; set; }

    /// <summary>
    /// Optional audience/scope binding (e.g. <c>"ci"</c>, <c>"prod"</c>). When present, the verifier
    /// requires it to equal the host's configured expected scope (<c>AIDOTNET_LICENSE_SCOPE</c>).
    /// </summary>
    [JsonPropertyName("scope")]
    public string? Scope { get; set; }

    // Case-INSENSITIVE off (default): the token uses the exact short lowercase names above. Nulls are
    // omitted so absent optional claims don't clutter the compact canonical JSON that gets signed.
    private static readonly JsonSerializerOptions s_options = new()
    {
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
    };

    /// <summary>
    /// Serializes the claims to compact JSON. Used only by the (server/test) signer to produce the exact
    /// bytes that appear in the token's middle segment. The verifier verifies over the raw decoded segment
    /// bytes as received — it never re-serializes.
    /// </summary>
    internal string ToCanonicalJson() => JsonSerializer.Serialize(this, s_options);

    /// <summary>Parses claims JSON. Returns null on malformed input.</summary>
    internal static Aidn2Claims? TryParse(string json)
    {
        try
        {
            return JsonSerializer.Deserialize<Aidn2Claims>(json, s_options);
        }
        catch (JsonException)
        {
            return null;
        }
    }
}
