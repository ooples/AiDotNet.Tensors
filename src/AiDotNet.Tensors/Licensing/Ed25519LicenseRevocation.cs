// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Text.Json;
using Org.BouncyCastle.Crypto.Parameters;
using Org.BouncyCastle.Crypto.Signers;

namespace AiDotNet.Tensors.Licensing;

/// <summary>
/// Offline revocation deny-list (CRL) for <c>aidn2.</c> tokens. Lets a specific leaked token (<c>jti</c>)
/// or a compromised signing key (<c>kid</c>) be killed BEFORE its <c>exp</c>. The Tensors port of the main
/// SDK's <c>LicenseRevocationProvider</c>, anchored to the same embedded Ed25519 public key via
/// <see cref="TensorsLicensePublicKeyProvider"/>. Kept SEPARATE from the RSA
/// <see cref="TensorsLicenseRevocation"/>, which anchors to the RSA entitlement key.
/// </summary>
/// <remarks>
/// <para><b>Fail-open by design:</b> revocation is ADDITIVE. A build with no CRL, or a CRL that is
/// expired / unsigned / malformed, revokes nothing — the token's own <c>exp</c> and bindings still bound
/// it. Only a valid (signature-verified, unexpired) CRL can revoke.</para>
///
/// <para><b>Wire format</b> (JWS-style, sign-over-exact-bytes like aidn2):</para>
/// <code>
/// { "kid": "&lt;signing kid&gt;",
///   "payload": "&lt;base64url(payload_json)&gt;",   // { "iat":.., "exp":.., "rkids":[..], "rjti":[..] }
///   "sig": "&lt;base64url(ed25519 sig over the decoded payload bytes)&gt;" }
/// </code>
/// The signature is computed over the RAW bytes recovered by base64url-decoding <c>payload</c>, so there
/// is no JSON canonicalization ambiguity (identical guarantee to <see cref="AsymmetricEntitlementVerifier"/>).
/// </remarks>
internal static class Ed25519LicenseRevocation
{
    /// <summary>Embedded manifest resource holding the release-time CRL (may be absent).</summary>
    private const string ResourceName = "AiDotNet.Tensors.LicenseRevocationEd25519";

    private const int Ed25519SignatureSize = 64;

    /// <summary>Max Unix-seconds (9999-12-31T23:59:59Z) so an out-of-range signed exp can't throw from
    /// <see cref="DateTimeOffset.FromUnixTimeSeconds"/>.</summary>
    private const long MaxUnixSeconds = 253402300799L;

    /// <summary>Clock-skew tolerance for a CRL's <c>iat</c>: an <c>iat</c> more than this far in the future is
    /// rejected (it decides CRL ordering, so a grossly future-dated value must not win), while benign clock
    /// differences are tolerated.</summary>
    private static readonly TimeSpan IatFutureSkew = TimeSpan.FromMinutes(5);

    private static readonly object _lock = new();
    private static bool _embeddedLoaded;
    private static bool _diskCacheLoaded;
    private static Crl? _embedded;   // release-embedded CRL (verified once)
    private static Crl? _fetched;    // newer CRL installed from an online refresh (verified on install)

    /// <summary>
    /// Returns true when a valid (signature-verified, unexpired) CRL revokes this token by its
    /// <paramref name="kid"/> or <paramref name="jti"/>. Fails open (returns false) when no valid CRL
    /// is present.
    /// </summary>
    internal static bool IsRevoked(string? kid, string? jti)
    {
        var crl = Effective();
        if (crl is null) return false;

        if (kid is { Length: > 0 } && crl.RevokedKids.Contains(kid)) return true;
        if (jti is { Length: > 0 } && crl.RevokedJti.Contains(jti)) return true;
        return false;
    }

    /// <summary>
    /// Installs a CRL obtained from an ONLINE refresh. Verified and expiry-checked here; ignored if it
    /// fails to verify, is expired, or is older than the one already installed. Returns true when accepted.
    /// </summary>
    internal static bool TryInstallFetched(string json, DateTimeOffset nowUtc)
    {
        var candidate = ParseAndVerify(json, nowUtc);
        if (candidate is null) return false;

        lock (_lock)
        {
            if (_fetched is not null)
            {
                // A STRICTLY OLDER CRL (replay) must never shrink the deny-list.
                if (candidate.Iat < _fetched.Iat) return false;
                // SAME-SECOND CRL (1s iat precision): merge (union) rather than discard, so a second
                // additive CRL issued in the same second isn't lost. Keep the later expiry.
                if (candidate.Iat == _fetched.Iat)
                {
                    _fetched = Merge(_fetched, candidate);
                    return true;
                }
            }
            _fetched = candidate;
            return true;
        }
    }

    private static Crl Merge(Crl a, Crl b)
    {
        var kids = new HashSet<string>(a.RevokedKids, StringComparer.Ordinal);
        kids.UnionWith(b.RevokedKids);
        var jti = new HashSet<string>(a.RevokedJti, StringComparer.Ordinal);
        jti.UnionWith(b.RevokedJti);
        return new Crl(a.Iat, Math.Max(a.Exp, b.Exp), kids, jti);
    }

    /// <summary>TEST-ONLY: sets the effective fetched CRL (from JSON) or clears both to a known state.</summary>
    internal static void OverrideForTesting(string? crlJson, DateTimeOffset nowUtc)
    {
        lock (_lock)
        {
            _embeddedLoaded = true;
            // Also mark the disk cache as "handled" so a test's explicit CRL isn't silently overridden by a
            // real ~/.aidotnet/revocations.crl left on the machine — tests must be deterministic.
            _diskCacheLoaded = true;
            _embedded = null;
            _fetched = crlJson is null ? null : ParseAndVerify(crlJson, nowUtc);
        }
    }

    /// <summary>The most recent valid CRL among the embedded and fetched copies (or null).</summary>
    private static Crl? Effective()
    {
        lock (_lock)
        {
            EnsureEmbeddedLoadedNoLock();
            EnsureDiskCacheLoadedNoLock();
            var now = DateTimeOffset.UtcNow;
            var emb = IsCurrentlyValid(_embedded, now) ? _embedded : null;
            var fet = IsCurrentlyValid(_fetched, now) ? _fetched : null;
            if (emb is null) return fet;
            if (fet is null) return emb;
            // Equal iat (1s precision): union embedded + fetched deny-lists so source/load ordering can
            // never drop a valid revocation. Otherwise the strictly-newer CRL wins.
            if (fet.Iat == emb.Iat) return Merge(fet, emb);
            return fet.Iat > emb.Iat ? fet : emb;
        }
    }

    /// <summary>A CRL is currently valid when it has no expiry (exp&lt;=0) or its expiry is still in the future,
    /// so a CRL valid when loaded does not stay authoritative past its exp. (exp is bounded at parse time.)</summary>
    private static bool IsCurrentlyValid(Crl? c, DateTimeOffset now) =>
        c is not null && (c.Exp <= 0 || DateTimeOffset.FromUnixTimeSeconds(c.Exp) >= now);

    /// <summary>
    /// Loads the last online-fetched CRL cached on disk (by <see cref="TensorsOnlineLicenseServices"/>) once
    /// per process, so revocation is enforced even on a fully-offline start that never refreshes. A later live
    /// refresh via <see cref="TryInstallFetched"/> supersedes it (newer iat wins). Fail-open on any error.
    /// </summary>
    private static void EnsureDiskCacheLoadedNoLock()
    {
        if (_diskCacheLoaded) return;
        _diskCacheLoaded = true;
        try
        {
            string? json = TensorsOnlineLicenseServices.ReadCachedCrl();
            if (json is null) return;
            var candidate = ParseAndVerify(json, DateTimeOffset.UtcNow);
            if (candidate is null) return;
            // Newer disk-cached CRL wins; an equal-iat one is unioned (same-second additive refresh) so a
            // valid revocation is never dropped by load ordering.
            if (_fetched is null || candidate.Iat > _fetched.Iat) _fetched = candidate;
            else if (candidate.Iat == _fetched.Iat) _fetched = Merge(_fetched, candidate);
        }
        catch (Exception ex)
        {
            System.Diagnostics.Trace.TraceWarning(
                "Ed25519LicenseRevocation: failed to load cached CRL: " + ex.GetType().Name + ": " + ex.Message);
        }
    }

    private static void EnsureEmbeddedLoadedNoLock()
    {
        if (_embeddedLoaded) return;
        _embeddedLoaded = true;
        try
        {
            var assembly = typeof(Ed25519LicenseRevocation).Assembly;
            using var stream = assembly.GetManifestResourceStream(ResourceName);
            if (stream is null || stream.Length == 0) return;
            using var reader = new StreamReader(stream, Encoding.UTF8);
            _embedded = ParseAndVerify(reader.ReadToEnd(), DateTimeOffset.UtcNow);
        }
        catch (Exception ex)
        {
            System.Diagnostics.Trace.TraceWarning(
                "Ed25519LicenseRevocation: failed to load embedded CRL: " + ex.GetType().Name + ": " + ex.Message);
        }
    }

    /// <summary>
    /// Parses a CRL envelope, verifies its Ed25519 signature against the embedded public key selected by
    /// its <c>kid</c>, and enforces expiry. Returns null on any problem (fail-open: not enforced).
    /// </summary>
    internal static Crl? ParseAndVerify(string json, DateTimeOffset nowUtc)
    {
        if (string.IsNullOrWhiteSpace(json)) return null;

        string kid, payloadB64, sigB64;
        try
        {
            using var envelope = JsonDocument.Parse(json);
            var root = envelope.RootElement;
            if (root.ValueKind != JsonValueKind.Object
                || !TryGetString(root, "kid", out kid)
                || !TryGetString(root, "payload", out payloadB64)
                || !TryGetString(root, "sig", out sigB64))
            {
                return null;
            }
        }
        catch (JsonException) { return null; }

        byte[] payloadBytes, sig;
        try
        {
            payloadBytes = Base64UrlHelper.Decode(payloadB64);
            sig = Base64UrlHelper.Decode(sigB64);
        }
        catch (FormatException) { return null; }

        if (payloadBytes.Length == 0 || sig.Length != Ed25519SignatureSize) return null;
        if (!TensorsLicensePublicKeyProvider.TryGetPublicKey(kid, out byte[] publicKey)) return null;

        try
        {
            var verifier = new Ed25519Signer();
            verifier.Init(false, new Ed25519PublicKeyParameters(publicKey, 0));
            verifier.BlockUpdate(payloadBytes, 0, payloadBytes.Length);
            if (!verifier.VerifySignature(sig)) return null;
        }
        catch (Exception ex)
        {
            System.Diagnostics.Trace.TraceWarning(
                "Ed25519LicenseRevocation: CRL signature verification error: " + ex.GetType().Name + ": " + ex.Message);
            return null;
        }

        try
        {
            using var payloadDoc = JsonDocument.Parse(new ReadOnlyMemory<byte>(payloadBytes));
            var p = payloadDoc.RootElement;
            if (p.ValueKind != JsonValueKind.Object) return null;

            // Validate iat BEFORE using it for CRL ordering. Require a JSON number parsed non-throwingly
            // (GetInt64 would throw FormatException, which the outer catch does NOT cover), reject
            // non-positive / out-of-range values, and reject a grossly future-dated iat. No zero fallback:
            // a missing or invalid iat is a malformed CRL, not "iat 0".
            if (!p.TryGetProperty("iat", out var iatEl) || iatEl.ValueKind != JsonValueKind.Number
                || !iatEl.TryGetInt64(out long iat) || iat <= 0 || iat > MaxUnixSeconds)
            {
                return null;
            }
            if (DateTimeOffset.FromUnixTimeSeconds(iat) - nowUtc > IatFutureSkew) return null;

            long exp = 0;
            if (p.TryGetProperty("exp", out var expEl) && expEl.ValueKind == JsonValueKind.Number
                && !expEl.TryGetInt64(out exp))
            {
                return null; // exp is a number but not representable as int64 — reject, don't throw.
            }

            // Reject an out-of-range exp BEFORE converting (FromUnixTimeSeconds throws outside its range).
            if (exp < 0 || exp > MaxUnixSeconds) return null;

            // Ignore an already-expired CRL. The stored exp is ALSO enforced at access time in Effective(),
            // so a CRL valid now does not stay authoritative past its expiry. (exp==0 => no expiry.)
            if (exp > 0 && DateTimeOffset.FromUnixTimeSeconds(exp) < nowUtc) return null;

            // Reject malformed present revocation arrays (non-array, or non-string entries) rather than
            // silently treating them as empty — a malformed newer CRL must not replace an older deny-list.
            if (!TryReadStringSet(p, "rkids", out var rkids) || !TryReadStringSet(p, "rjti", out var rjti))
            {
                return null;
            }

            return new Crl(iat, exp, rkids, rjti);
        }
        catch (Exception ex) when (ex is JsonException or ArgumentException)
        {
            return null;
        }
    }

    /// <summary>
    /// Reads a revocation string set. An ABSENT field yields an empty set (<c>true</c>). A PRESENT field
    /// that is not a JSON array, or that contains any non-string entry, is malformed and yields
    /// <c>false</c> so the caller can reject the whole CRL instead of silently accepting an empty deny-list.
    /// A present, valid empty array yields an empty set (<c>true</c>).
    /// </summary>
    private static bool TryReadStringSet(JsonElement obj, string name, out HashSet<string> set)
    {
        set = new HashSet<string>(StringComparer.Ordinal);
        if (!obj.TryGetProperty(name, out var arr))
        {
            return true; // absent field => empty set is fine.
        }
        if (arr.ValueKind != JsonValueKind.Array)
        {
            return false; // present but not an array => malformed.
        }
        foreach (var e in arr.EnumerateArray())
        {
            if (e.ValueKind != JsonValueKind.String)
            {
                return false; // non-string entry => malformed.
            }
            var v = e.GetString();
            if (!string.IsNullOrEmpty(v)) set.Add(v!);
        }
        return true;
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
        internal long Exp { get; }
        internal HashSet<string> RevokedKids { get; }
        internal HashSet<string> RevokedJti { get; }
        internal Crl(long iat, long exp, HashSet<string> kids, HashSet<string> jti)
        {
            Iat = iat; Exp = exp; RevokedKids = kids; RevokedJti = jti;
        }
    }
}
