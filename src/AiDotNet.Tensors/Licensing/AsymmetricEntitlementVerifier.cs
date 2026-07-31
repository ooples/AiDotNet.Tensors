// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using Org.BouncyCastle.Crypto.Parameters;
using Org.BouncyCastle.Crypto.Signers;

namespace AiDotNet.Tensors.Licensing;

/// <summary>
/// Verifies <c>aidn2.</c> asymmetric license tokens against the embedded PUBLIC signing key(s), fully
/// offline. The Tensors port of the main SDK's <c>AsymmetricLicenseVerifier</c>, returning the shared
/// <see cref="EntitlementResult"/> so it plugs into <see cref="PersistenceGuard"/> uniformly with the RSA
/// <see cref="SignedEntitlement"/> path.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> A token looks like <c>aidn2.&lt;claims&gt;.&lt;signature&gt;</c>. We only
/// check that the signature was produced by the holder of the private key (which lives on the server and
/// never ships). Because verification uses a <b>public</b> key, an attacker who extracts it from the DLL
/// still cannot forge a valid token.</para>
///
/// <para><b>Primitive:</b> Ed25519 (EdDSA over Curve25519, RFC 8032). .NET's
/// <c>System.Security.Cryptography</c> ships no Ed25519 type on any of the Tensors target frameworks
/// (net471/net8/net10), so verification uses the well-audited <c>BouncyCastle.Cryptography</c> library —
/// the first BouncyCastle dependency in the otherwise-BouncyCastle-free Tensors runtime, added solely for
/// this offline verification path.</para>
///
/// <para><b>Token grammar:</b> <c>aidn2.&lt;base64url(claims_json)&gt;.&lt;base64url(signature)&gt;</c>
/// where the signature is computed over the raw UTF-8 bytes of <c>claims_json</c>. Claims carry
/// <c>alg:"EdDSA"</c>.</para>
///
/// <para><b>Enforcement order (all AFTER the signature check, since the fields are signed):</b> signature
/// → CRL revocation (kid/jti) → machine binding (<c>mach</c>) → scope binding
/// (<c>AIDOTNET_LICENSE_SCOPE</c>) → expiry (5-min clock skew). Fails closed on every problem: only an
/// authentic, unrevoked, correctly-bound, unexpired token yields a valid <see cref="EntitlementResult"/>.</para>
/// </remarks>
internal static class AsymmetricEntitlementVerifier
{
    /// <summary>The token version prefix for asymmetric (public-key-signed) licenses.</summary>
    internal const string Prefix = "aidn2";

    /// <summary>The signature algorithm this verifier implements. Tokens must declare this (or omit alg).</summary>
    internal const string Algorithm = "EdDSA";

    /// <summary>Ed25519 signatures are exactly 64 bytes.</summary>
    private const int Ed25519SignatureSize = 64;

    /// <summary>
    /// Upper bound for a Unix-seconds <c>exp</c> claim (9999-12-31T23:59:59Z). Values outside
    /// (0, MaxUnixSeconds] are rejected up front so <see cref="DateTimeOffset.FromUnixTimeSeconds"/> can
    /// never throw.
    /// </summary>
    private const long MaxUnixSeconds = 253402300799L;

    /// <summary>Allowed clock skew when checking expiry (kept small — expiry is the primary offline bound).</summary>
    private static readonly TimeSpan ClockSkew = TimeSpan.FromMinutes(5);

    /// <summary>
    /// Strict UTF-8 decoder (no BOM, throw on invalid bytes). Unlike <see cref="Encoding.UTF8"/>, which
    /// silently replaces malformed byte sequences with U+FFFD, this fails closed so malformed claim bytes
    /// are rejected rather than decoded into replacement characters.
    /// </summary>
    private static readonly UTF8Encoding StrictUtf8 = new UTF8Encoding(encoderShouldEmitUTF8Identifier: false, throwOnInvalidBytes: true);

    /// <summary>
    /// Returns true if <paramref name="key"/> is in the asymmetric token shape:
    /// <c>aidn2.&lt;base64url&gt;.&lt;base64url&gt;</c>. Does NOT verify the signature — a cheap structural
    /// check used for routing.
    /// </summary>
    internal static bool IsAsymmetricKeyFormat(string? key)
    {
        if (key is null) return false;
        var parts = key.Split('.');
        if (parts.Length != 3 || parts[0] != Prefix || parts[1].Length == 0 || parts[2].Length == 0)
        {
            return false;
        }

        for (int i = 0; i < parts[1].Length; i++)
        {
            if (!IsBase64UrlChar(parts[1][i])) return false;
        }

        for (int i = 0; i < parts[2].Length; i++)
        {
            if (!IsBase64UrlChar(parts[2][i])) return false;
        }

        return true;
    }

    private static bool IsBase64UrlChar(char c)
    {
        return (c >= 'A' && c <= 'Z')
            || (c >= 'a' && c <= 'z')
            || (c >= '0' && c <= '9')
            || c == '-'
            || c == '_';
    }

    /// <summary>
    /// Resolves an <c>aidn2.</c> token from the environment / disk and verifies it offline. Resolution
    /// sources (first Active wins): (a) <c>AIDOTNET_LICENSE_KEY</c> when it is itself an <c>aidn2.</c>
    /// token, then (b) the main SDK's cached offline tokens <c>~/.aidotnet/offline-*.token</c> (one token
    /// per file). Returns the verified <see cref="EntitlementResult"/> when a token is Active, or
    /// <see langword="null"/> when none is present or none verifies — this path is ADDITIVE and
    /// positive-grant-only, so absence/failure simply falls through to the RSA/key/trial paths.
    /// </summary>
    internal static EntitlementResult? TryVerifyConfigured()
    {
        var now = DateTimeOffset.UtcNow;

        // (a) AIDOTNET_LICENSE_KEY, only when it is an aidn2. token (a server AIDN-* / aidn. key is handled
        // by the existing online key path — we must not mis-route it here).
        try
        {
            string? envKey = Environment.GetEnvironmentVariable("AIDOTNET_LICENSE_KEY");
            if (envKey is not null && IsAsymmetricKeyFormat(envKey.Trim()))
            {
                var r = Verify(envKey.Trim(), now);
                if (r.IsValid) return r;
            }
        }
        catch (Exception ex)
        {
            System.Diagnostics.Trace.TraceWarning(
                "AsymmetricEntitlementVerifier: unable to read AIDOTNET_LICENSE_KEY: " + ex.Message);
        }

        // (b) Cached offline tokens minted by a prior successful online validation in the main SDK.
        try
        {
            string home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
            string dir = Path.Combine(home, ".aidotnet");
            if (Directory.Exists(dir))
            {
                foreach (string file in Directory.EnumerateFiles(dir, "offline-*.token"))
                {
                    string token;
                    try { token = File.ReadAllText(file).Trim(); }
                    catch { continue; }

                    if (!IsAsymmetricKeyFormat(token)) continue;
                    var r = Verify(token, now);
                    if (r.IsValid) return r;
                }
            }
        }
        catch (Exception ex)
        {
            System.Diagnostics.Trace.TraceWarning(
                "AsymmetricEntitlementVerifier: cached offline token scan failed: " + ex.GetType().Name + ": " + ex.Message);
        }

        return null;
    }

    /// <summary>
    /// Verifies an <c>aidn2.</c> token fully offline: Ed25519 signature against the embedded public key
    /// selected by <c>kid</c>, then CRL revocation, machine binding, scope binding, and expiry — all only
    /// AFTER the signature check. Returns a valid <see cref="EntitlementResult"/> (tenant=<c>sub</c>,
    /// expiry, signed <c>caps</c>) ONLY when the token is authentic and active; every other outcome
    /// (bad signature, unknown kid, revoked, wrong machine, wrong scope, expired, malformed) is a
    /// fail-closed <see cref="EntitlementResult.Invalid"/>.
    /// </summary>
    internal static EntitlementResult Verify(string key, DateTimeOffset nowUtc)
    {
        if (string.IsNullOrWhiteSpace(key))
        {
            return EntitlementResult.Invalid("License token is empty or missing.");
        }

        var parts = key.Split('.');
        if (parts.Length != 3 || parts[0] != Prefix)
        {
            return EntitlementResult.Invalid("License token is not a valid aidn2 token.");
        }

        byte[] claimsBytes;
        byte[] signature;
        try
        {
            claimsBytes = Base64UrlHelper.Decode(parts[1]);
            signature = Base64UrlHelper.Decode(parts[2]);
        }
        catch (FormatException)
        {
            return EntitlementResult.Invalid("License token encoding is malformed.");
        }

        if (claimsBytes.Length == 0 || signature.Length == 0)
        {
            return EntitlementResult.Invalid("License token is missing claims or signature.");
        }

        if (signature.Length != Ed25519SignatureSize)
        {
            return EntitlementResult.Invalid("License signature is not a valid Ed25519 signature.");
        }

        string claimsJson;
        try
        {
            claimsJson = StrictUtf8.GetString(claimsBytes);
        }
        catch (DecoderFallbackException)
        {
            return EntitlementResult.Invalid("License token claims are not valid UTF-8.");
        }
        catch (ArgumentException)
        {
            return EntitlementResult.Invalid("License token claims are not valid UTF-8.");
        }

        Aidn2Claims? claims = Aidn2Claims.TryParse(claimsJson);
        if (claims is null)
        {
            return EntitlementResult.Invalid("License token claims are malformed.");
        }

        // Reject a token that declares a signature algorithm this verifier does not implement. A missing
        // alg is tolerated (implicitly EdDSA for aidn2); a present, mismatched alg fails closed.
        if (claims.Alg is not null && claims.Alg.Length > 0 &&
            !string.Equals(claims.Alg, Algorithm, StringComparison.Ordinal))
        {
            return EntitlementResult.Invalid("License token declares unsupported signature algorithm '" + claims.Alg + "'.");
        }

        string? kid = claims.Kid;
        if (kid is null || kid.Length == 0)
        {
            return EntitlementResult.Invalid("License token does not name a signing key (kid).");
        }

        // A build that embeds no matching public key CANNOT verify a signature. Fail closed.
        if (!TensorsLicensePublicKeyProvider.TryGetPublicKey(kid, out byte[] publicKey))
        {
            return EntitlementResult.Invalid(
                "This build cannot verify the license signature: no embedded public key matches the " +
                "token's key id (kid='" + kid + "').");
        }

        bool signatureValid;
        try
        {
            var verifier = new Ed25519Signer();
            verifier.Init(false, new Ed25519PublicKeyParameters(publicKey, 0));
            verifier.BlockUpdate(claimsBytes, 0, claimsBytes.Length);
            signatureValid = verifier.VerifySignature(signature);
        }
        catch (Exception ex)
        {
            System.Diagnostics.Trace.TraceWarning(
                "AsymmetricEntitlementVerifier: verification error: " + ex.GetType().Name + ": " + ex.Message);
            return EntitlementResult.Invalid("License signature verification failed.");
        }

        if (!signatureValid)
        {
            return EntitlementResult.Invalid("License signature verification failed.");
        }

        // Signature is authentic. Enforce the signed bindings + revocation. Fail closed on every problem.

        // Revocation (CRL deny-list): a leaked token (jti) or a compromised key (kid) can be killed before
        // exp. Absent/expired/unsigned CRL => not enforced (fail-open; token exp still bounds).
        if (Ed25519LicenseRevocation.IsRevoked(kid, claims.Jti))
        {
            return EntitlementResult.Invalid("License token has been revoked.");
        }

        // Machine binding (node-lock): a token carrying `mach` is only valid on the machine whose id hash
        // matches. Reuses the main-SDK-parity machine hash so a token minted there verifies here.
        if (claims.Mach is { Length: > 0 } &&
            !string.Equals(claims.Mach, TensorsMachineFingerprint.GetMachineIdHash(), StringComparison.Ordinal))
        {
            return EntitlementResult.Invalid("License token is bound to a different machine.");
        }

        // Scope/audience binding: a token carrying `scope` is only valid where the host declares the same
        // expected scope (AIDOTNET_LICENSE_SCOPE). A scoped token on an unscoped host is rejected.
        if (claims.Scope is { Length: > 0 } &&
            !string.Equals(claims.Scope, ResolveExpectedScope(), StringComparison.Ordinal))
        {
            return EntitlementResult.Invalid("License token scope '" + claims.Scope + "' does not match this host's expected scope.");
        }

        // Fail CLOSED on a missing / non-positive / out-of-range exp BEFORE converting it: a malformed exp
        // must never be treated as "non-expiring".
        if (claims.Exp <= 0 || claims.Exp > MaxUnixSeconds)
        {
            return EntitlementResult.Invalid("License token has a missing or invalid expiry (exp).");
        }

        DateTimeOffset expiresAt = DateTimeOffset.FromUnixTimeSeconds(claims.Exp);
        // Compare via subtraction rather than `expiresAt + ClockSkew` so an exp at the upper bound
        // (MaxUnixSeconds, near DateTimeOffset.MaxValue) cannot overflow when adding the skew.
        if (nowUtc > expiresAt && nowUtc - expiresAt > ClockSkew)
        {
            return EntitlementResult.Invalid("License token expired on " + expiresAt.UtcDateTime.ToString("u") + ".");
        }

        // Active. Carry the SIGNED capabilities so the guard gates on exactly what this token grants.
        var caps = new HashSet<string>(StringComparer.Ordinal);
        if (claims.Caps is not null)
        {
            foreach (var c in claims.Caps)
            {
                if (!string.IsNullOrWhiteSpace(c)) caps.Add(c);
            }
        }

        string? tenant = string.IsNullOrWhiteSpace(claims.Sub) ? null : claims.Sub;
        return EntitlementResult.Valid(tenant, expiresAt, caps);
    }

    /// <summary>
    /// The scope this host expects a scoped license token to declare, from <c>AIDOTNET_LICENSE_SCOPE</c>
    /// (empty when unset). A token carrying a <c>scope</c> claim is accepted only when it equals this value.
    /// </summary>
    private static string ResolveExpectedScope()
    {
        try
        {
            return Environment.GetEnvironmentVariable("AIDOTNET_LICENSE_SCOPE")?.Trim() ?? string.Empty;
        }
        catch (Exception ex)
        {
            System.Diagnostics.Trace.TraceWarning(
                "AsymmetricEntitlementVerifier: unable to read AIDOTNET_LICENSE_SCOPE: " + ex.Message);
            return string.Empty;
        }
    }
}
