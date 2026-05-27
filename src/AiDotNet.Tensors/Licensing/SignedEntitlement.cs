// Copyright (c) AiDotNet. All rights reserved.

using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;

namespace AiDotNet.Tensors.Licensing;

/// <summary>
/// Offline, RSA-signed entitlement verification for the runtime persistence
/// guard. This is the cryptographic trust anchor that does NOT depend on the
/// online license server (and therefore cannot be defeated by spoofing /
/// MITM'ing that server — the single cheapest bypass of the online path):
/// the entitlement is a JSON token signed by Ooples' private key, and the
/// runtime verifies it with an embedded RSA public key, fully offline.
///
/// <para><b>Threat model — read this honestly.</b> RSA signature verification
/// means an attacker cannot FORGE an entitlement without Ooples' private key.
/// It does NOT make the guard unbreakable: this runs in managed bytecode the
/// attacker controls, so they can still patch out the verification call or swap
/// the embedded key (see <see cref="LicenseTamperEvidence"/>, which makes the
/// key-swap detectable). Client-side enforcement is a high, audited speed bump,
/// not a vault — genuine protection is this + tamper-evidence + the contractual
/// relationship. Do not advertise it as "uncrackable".</para>
///
/// <para><b>Token format</b> (the outer envelope; only <c>payload</c> is
/// signed, mirroring <c>build/LicenseValidator</c>'s signed-payload-as-source-
/// of-truth rule):</para>
/// <code>
/// {
///   "payload":   "&lt;base64 of the canonical JSON below&gt;",
///   "signature": "&lt;base64 RSA-2048 PKCS#1 v1.5 SHA-256 signature over payload bytes&gt;"
/// }
/// // signed payload:
/// {
///   "marker":       "AiDotNet-Tensors-Entitlement-v1",
///   "tenant":       "ACME Corp.",
///   "expires":      "YYYY-MM-DD",
///   "capabilities": ["tensors:save", "tensors:load"]
/// }
/// </code>
/// </summary>
public static class SignedEntitlement
{
    /// <summary>Marker every valid signed payload must carry.</summary>
    public const string Marker = "AiDotNet-Tensors-Entitlement-v1";

    /// <summary>
    /// Placeholder for the embedded RSA public key. Until Ooples replaces this
    /// with the real public key, every entitlement is rejected (safe by
    /// default) and the runtime falls through to the existing online/trial
    /// path — so shipping this feature with the placeholder changes nothing.
    /// </summary>
    public const string PublicKeyPlaceholderMarker = "REPLACE_WITH_REAL_OOPLES_ENTITLEMENT_PUBLIC_KEY";

    /// <summary>
    /// Embedded RSA public key as <c>{modulusBase64}:{exponentBase64}</c>
    /// (RSAParameters form — chosen over PKCS#1 DER so import works on both
    /// net471 and net10.0 with no BouncyCastle dependency in the runtime
    /// assembly). Replace for production per build/LicenseValidator/README.md;
    /// the PRIVATE key never ships.
    /// </summary>
    public const string EntitlementPublicKey = PublicKeyPlaceholderMarker;

    /// <summary>Test-only override of the embedded key (see the build-time validator's identical hook).</summary>
    internal static string? s_overridePublicKey;

    private static string ResolvePublicKey() => s_overridePublicKey ?? EntitlementPublicKey;

    /// <summary>
    /// Resolves a signed entitlement from the environment / disk and verifies
    /// it offline. Resolution order: <c>AIDOTNET_LICENSE_TOKEN</c> env var
    /// (inline JSON or a path to a token file), then
    /// <c>~/.aidotnet/tensors-entitlement.json</c>. Returns <c>null</c> when
    /// no entitlement is configured (the caller then falls through to the
    /// online/trial path) — only a present-but-INVALID entitlement is a hard
    /// failure the caller should surface.
    /// </summary>
    public static EntitlementResult? TryVerifyConfigured()
    {
        // Feature is INERT until a real embedded key replaces the placeholder:
        // return null (not configured) so the guard falls through to the
        // existing key/trial path. This keeps shipping the placeholder a
        // no-op — a stray token on a dev box can't hard-fail persistence.
        if (string.Equals(ResolvePublicKey(), PublicKeyPlaceholderMarker, StringComparison.Ordinal))
            return null;

        string? raw = ResolveRawToken();
        if (raw is null) return null;
        return Verify(raw);
    }

    private static string? ResolveRawToken()
    {
        try
        {
            string? env = Environment.GetEnvironmentVariable("AIDOTNET_LICENSE_TOKEN");
            if (!string.IsNullOrWhiteSpace(env))
            {
                // Inline JSON if it looks like an object, else treat as a path.
                string t = env!.Trim();
                if (t.StartsWith("{", StringComparison.Ordinal)) return t;
                if (File.Exists(t)) return File.ReadAllText(t);
            }
        }
        catch { /* SecurityException / IO — fall through to file probe */ }

        try
        {
            string home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
            string path = Path.Combine(home, ".aidotnet", "tensors-entitlement.json");
            if (File.Exists(path))
            {
                string content = File.ReadAllText(path).Trim();
                if (!string.IsNullOrWhiteSpace(content)) return content;
            }
        }
        catch { /* IO / SecurityException — no entitlement */ }

        return null;
    }

    /// <summary>
    /// Verifies a raw entitlement-envelope JSON string. Verification order:
    /// envelope parses → has payload+signature → embedded key is not the
    /// placeholder → RSA signature verifies over the payload bytes → signed
    /// payload parses → marker matches → not expired. Every privilege field
    /// (marker, expires, capabilities, tenant) is read from the SIGNED payload
    /// only, never the envelope.
    /// </summary>
    public static EntitlementResult Verify(string envelopeJson)
    {
        if (string.IsNullOrWhiteSpace(envelopeJson))
            return EntitlementResult.Invalid("Entitlement token is empty.");

        string pubKey = ResolvePublicKey();
        if (string.Equals(pubKey, PublicKeyPlaceholderMarker, StringComparison.Ordinal))
            return EntitlementResult.Invalid(
                "Entitlement verification refused: this build ships the placeholder entitlement " +
                "public key. Production builds must replace SignedEntitlement.EntitlementPublicKey.");

        string payloadB64, signatureB64;
        try
        {
            using var envelope = JsonDocument.Parse(envelopeJson);
            var root = envelope.RootElement;
            if (root.ValueKind != JsonValueKind.Object
                || !TryGetString(root, "payload", out payloadB64)
                || !TryGetString(root, "signature", out signatureB64))
                return EntitlementResult.Invalid("Entitlement token missing 'payload' or 'signature'.");
        }
        catch (JsonException ex)
        {
            return EntitlementResult.Invalid("Entitlement token is not valid JSON: " + ex.Message);
        }

        byte[] payloadBytes, signatureBytes;
        try
        {
            payloadBytes = Convert.FromBase64String(payloadB64);
            signatureBytes = Convert.FromBase64String(signatureB64);
        }
        catch (FormatException ex)
        {
            return EntitlementResult.Invalid("Entitlement payload/signature is not valid base64: " + ex.Message);
        }

        bool signatureOk;
        try
        {
            using var rsa = ImportPublicKey(pubKey);
            signatureOk = rsa.VerifyData(payloadBytes, signatureBytes,
                HashAlgorithmName.SHA256, RSASignaturePadding.Pkcs1);
        }
        catch (Exception ex)
        {
            return EntitlementResult.Invalid("Entitlement signature verification errored: " + ex.Message);
        }

        if (!signatureOk)
            return EntitlementResult.Invalid("Entitlement signature is invalid (forged, corrupt, or wrong key).");

        // Signature verified → the payload is authentic. Parse it as the sole
        // source of truth for the gating fields.
        try
        {
            using var payloadDoc = JsonDocument.Parse(new ReadOnlyMemory<byte>(payloadBytes));
            var signed = payloadDoc.RootElement;
            if (signed.ValueKind != JsonValueKind.Object)
                return EntitlementResult.Invalid("Entitlement signed payload is not a JSON object.");

            if (!TryGetString(signed, "marker", out var marker)
                || !string.Equals(marker, Marker, StringComparison.Ordinal))
                return EntitlementResult.Invalid("Entitlement signed payload has a missing/wrong marker.");

            DateTimeOffset? expires = null;
            if (TryGetString(signed, "expires", out var expRaw)
                && DateTimeOffset.TryParse(expRaw, CultureInfo.InvariantCulture,
                    DateTimeStyles.AssumeUniversal | DateTimeStyles.AdjustToUniversal, out var parsed))
            {
                expires = parsed;
                if (parsed.UtcDateTime.Date < DateTimeOffset.UtcNow.UtcDateTime.Date)
                    return EntitlementResult.Invalid($"Entitlement expired on {parsed:yyyy-MM-dd} UTC.");
            }

            var caps = new HashSet<string>(StringComparer.Ordinal);
            if (signed.TryGetProperty("capabilities", out var capsEl) && capsEl.ValueKind == JsonValueKind.Array)
            {
                foreach (var c in capsEl.EnumerateArray())
                {
                    if (c.ValueKind == JsonValueKind.String)
                    {
                        var v = c.GetString();
                        if (!string.IsNullOrWhiteSpace(v)) caps.Add(v!);
                    }
                }
            }

            string? tenant = TryGetString(signed, "tenant", out var t) && !string.IsNullOrWhiteSpace(t) ? t : null;
            return EntitlementResult.Valid(tenant, expires, caps);
        }
        catch (JsonException ex)
        {
            return EntitlementResult.Invalid("Entitlement signed payload is not valid JSON: " + ex.Message);
        }
    }

    private static bool TryGetString(JsonElement obj, string name, out string value)
    {
        if (obj.TryGetProperty(name, out var el) && el.ValueKind == JsonValueKind.String)
        {
            value = el.GetString() ?? string.Empty;
            return value.Length > 0;
        }
        value = string.Empty;
        return false;
    }

    /// <summary>
    /// Imports the embedded <c>{modulusBase64}:{exponentBase64}</c> public key.
    /// Uses <see cref="RSA.ImportParameters"/> (RSAParameters) so it works on
    /// net471 and net10.0 alike — <see cref="RSA.ImportRSAPublicKey(System.ReadOnlySpan{byte}, out int)"/>
    /// is netstandard2.1+ only.
    /// </summary>
    private static RSA ImportPublicKey(string modExpBase64)
    {
        int sep = modExpBase64.IndexOf(':');
        if (sep <= 0 || sep >= modExpBase64.Length - 1)
            throw new FormatException("Embedded entitlement key must be '{modulusBase64}:{exponentBase64}'.");
        var modulus = Convert.FromBase64String(modExpBase64.Substring(0, sep));
        var exponent = Convert.FromBase64String(modExpBase64.Substring(sep + 1));
        var rsa = RSA.Create();
        rsa.ImportParameters(new RSAParameters { Modulus = modulus, Exponent = exponent });
        return rsa;
    }
}

/// <summary>Outcome of verifying a signed entitlement token.</summary>
public sealed class EntitlementResult
{
    /// <summary>True iff the entitlement's RSA signature, marker, and expiry all verified.</summary>
    public bool IsValid { get; }

    /// <summary>Tenant from the signed payload, or null.</summary>
    public string? Tenant { get; }

    /// <summary>Expiry from the signed payload, or null (perpetual).</summary>
    public DateTimeOffset? ExpiresAt { get; }

    /// <summary>Diagnostic message when <see cref="IsValid"/> is false.</summary>
    public string? Message { get; }

    private readonly HashSet<string> _capabilities;

    /// <summary>True iff the signed entitlement grants <paramref name="capability"/>.</summary>
    public bool HasCapability(string capability) => _capabilities.Contains(capability);

    private EntitlementResult(bool valid, string? tenant, DateTimeOffset? expiresAt, string? message, HashSet<string>? caps)
    {
        IsValid = valid;
        Tenant = tenant;
        ExpiresAt = expiresAt;
        Message = message;
        _capabilities = caps ?? new HashSet<string>(StringComparer.Ordinal);
    }

    internal static EntitlementResult Valid(string? tenant, DateTimeOffset? expiresAt, HashSet<string> caps)
        => new(true, tenant, expiresAt, null, caps);

    internal static EntitlementResult Invalid(string message)
        => new(false, null, null, message, null);
}
