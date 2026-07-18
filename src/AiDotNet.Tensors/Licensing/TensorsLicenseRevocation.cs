using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Text.Json;

namespace AiDotNet.Tensors.Licensing;

/// <summary>
/// Offline revocation deny-list (CRL) for signed entitlements. Lets a specific leaked entitlement
/// (<c>jti</c>) be killed BEFORE it expires. The Tensors analogue of AiDotNet's
/// <c>LicenseRevocationProvider</c>, anchored to the same RSA entitlement key via
/// <see cref="SignedEntitlement.VerifyWithEntitlementKey"/>.
/// </summary>
/// <remarks>
/// <para><b>Fail-open by design:</b> revocation is ADDITIVE. No CRL, or a CRL that is expired /
/// unsigned / malformed (including any build that still ships the placeholder entitlement key, so the
/// signature can't be checked), revokes nothing — the entitlement's own expiry still bounds it.</para>
///
/// <para><b>Wire format</b> (same envelope as the entitlement — RSA-SHA256 over the raw payload bytes):</para>
/// <code>
/// { "payload":  "&lt;base64(payload_json)&gt;",   // { "iat":.., "exp":.., "rjti":[..] }  exp/iat = Unix seconds
///   "signature":"&lt;base64(RSA-SHA256 over the decoded payload bytes)&gt;" }
/// </code>
/// </remarks>
internal static class TensorsLicenseRevocation
{
    /// <summary>Embedded manifest resource holding the release-time signed CRL (may be absent).</summary>
    private const string ResourceName = "AiDotNet.Tensors.LicenseRevocation";

    private static readonly object _lock = new();
    private static bool _embeddedLoaded;
    private static Crl? _embedded;
    private static Crl? _fetched;

    /// <summary>
    /// True when a valid (signature-verified, unexpired) CRL revokes the entitlement identified by
    /// <paramref name="jti"/>. Fails open (false) when no valid CRL is present.
    /// </summary>
    internal static bool IsRevoked(string? jti)
    {
        if (string.IsNullOrEmpty(jti)) return false;
        var crl = Effective();
        return crl is not null && crl.RevokedJti.Contains(jti!);
    }

    /// <summary>
    /// Installs a CRL from an ONLINE refresh; verified and expiry-checked here, ignored if it fails to
    /// verify, is expired, or is not newer than the one already installed. Returns true when accepted.
    /// </summary>
    internal static bool TryInstallFetched(string json, DateTimeOffset nowUtc)
    {
        var candidate = ParseAndVerify(json, nowUtc);
        if (candidate is null) return false;
        lock (_lock)
        {
            if (_fetched is not null && candidate.Iat <= _fetched.Iat) return false;
            _fetched = candidate;
            return true;
        }
    }

    /// <summary>TEST-ONLY: set the effective fetched CRL (from JSON) or clear both to a known state.</summary>
    internal static void OverrideForTesting(string? crlJson, DateTimeOffset nowUtc)
    {
        lock (_lock)
        {
            _embeddedLoaded = true;
            _embedded = null;
            _fetched = crlJson is null ? null : ParseAndVerify(crlJson, nowUtc);
        }
    }

    private static Crl? Effective()
    {
        lock (_lock)
        {
            EnsureEmbeddedLoadedNoLock();
            if (_embedded is null) return _fetched;
            if (_fetched is null) return _embedded;
            return _fetched.Iat >= _embedded.Iat ? _fetched : _embedded;
        }
    }

    private static void EnsureEmbeddedLoadedNoLock()
    {
        if (_embeddedLoaded) return;
        _embeddedLoaded = true;
        try
        {
            var assembly = typeof(TensorsLicenseRevocation).Assembly;
            using var stream = assembly.GetManifestResourceStream(ResourceName);
            if (stream is null || stream.Length == 0) return;
            using var reader = new StreamReader(stream, Encoding.UTF8);
            _embedded = ParseAndVerify(reader.ReadToEnd(), DateTimeOffset.UtcNow);
        }
        catch (Exception ex)
        {
            System.Diagnostics.Trace.TraceWarning(
                "TensorsLicenseRevocation: failed to load embedded CRL: " + ex.GetType().Name + ": " + ex.Message);
        }
    }

    /// <summary>
    /// Parses a CRL envelope, verifies its RSA signature against the embedded entitlement key, and
    /// enforces expiry. Returns null on any problem (fail-open).
    /// </summary>
    internal static Crl? ParseAndVerify(string json, DateTimeOffset nowUtc)
    {
        if (string.IsNullOrWhiteSpace(json)) return null;

        string payloadB64, sigB64;
        try
        {
            using var envelope = JsonDocument.Parse(json);
            var root = envelope.RootElement;
            if (root.ValueKind != JsonValueKind.Object
                || !TryGetString(root, "payload", out payloadB64)
                || !TryGetString(root, "signature", out sigB64))
                return null;
        }
        catch (JsonException) { return null; }

        byte[] payloadBytes, sig;
        try
        {
            payloadBytes = Convert.FromBase64String(payloadB64);
            sig = Convert.FromBase64String(sigB64);
        }
        catch (FormatException) { return null; }

        if (payloadBytes.Length == 0 || sig.Length == 0) return null;
        if (!SignedEntitlement.VerifyWithEntitlementKey(payloadBytes, sig)) return null;

        try
        {
            using var payloadDoc = JsonDocument.Parse(new ReadOnlyMemory<byte>(payloadBytes));
            var p = payloadDoc.RootElement;
            if (p.ValueKind != JsonValueKind.Object) return null;

            long iat = p.TryGetProperty("iat", out var iatEl) && iatEl.ValueKind == JsonValueKind.Number
                ? iatEl.GetInt64() : 0;
            long exp = p.TryGetProperty("exp", out var expEl) && expEl.ValueKind == JsonValueKind.Number
                ? expEl.GetInt64() : 0;

            // Ignore a stale (expired) CRL — client should refetch; entitlement expiry still applies.
            if (exp > 0 && DateTimeOffset.FromUnixTimeSeconds(exp) < nowUtc) return null;

            var rjti = new HashSet<string>(StringComparer.Ordinal);
            if (p.TryGetProperty("rjti", out var rjtiEl) && rjtiEl.ValueKind == JsonValueKind.Array)
            {
                foreach (var e in rjtiEl.EnumerateArray())
                {
                    if (e.ValueKind == JsonValueKind.String)
                    {
                        var v = e.GetString();
                        if (!string.IsNullOrEmpty(v)) rjti.Add(v!);
                    }
                }
            }

            return new Crl(iat, rjti);
        }
        catch (JsonException) { return null; }
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

    internal sealed class Crl
    {
        internal long Iat { get; }
        internal HashSet<string> RevokedJti { get; }
        internal Crl(long iat, HashSet<string> rjti) { Iat = iat; RevokedJti = rjti; }
    }
}
