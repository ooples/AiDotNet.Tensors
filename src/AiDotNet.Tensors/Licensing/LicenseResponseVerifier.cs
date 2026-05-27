// Copyright (c) AiDotNet. All rights reserved.

using System.Collections.Generic;
using System.Security.Cryptography;
using System.Text;

namespace AiDotNet.Tensors.Licensing;

/// <summary>
/// Verifies that an online license-server validation response is authentic and
/// fresh, closing the server-spoof / MITM bypass of the online path. The client
/// sends a per-request random <see cref="NewNonce"/>; the server signs a
/// canonical string binding that nonce to the response fields; the client
/// verifies the RSA signature against an embedded server public key and checks
/// the nonce echoes the one it sent (which defeats replay of any previously
/// captured response).
///
/// <para><b>Scope (honest):</b> this only matters when
/// <see cref="AiDotNetTensorsLicenseKey.RequireSignedResponse"/> is enabled AND
/// the server actually signs (see
/// <c>docs/enterprise/license-server-response-signing.md</c>). It hardens the
/// online/standard tier against trial-bypass-by-server-spoofing. High-assurance
/// enterprise should use the fully-offline <see cref="SignedEntitlement"/>,
/// whose trust anchor needs no server. Like all client-side checks, this is
/// bypassable by patching the managed assembly — it is authenticity, not
/// tamper-proofing.</para>
/// </summary>
public static class LicenseResponseVerifier
{
    /// <summary>Placeholder embedded server-response public key — until replaced, signed-response verification is unavailable.</summary>
    public const string PublicKeyPlaceholderMarker = "REPLACE_WITH_REAL_OOPLES_RESPONSE_SIGNING_PUBLIC_KEY";

    /// <summary>
    /// Embedded RSA public key for the response-signing keypair, as
    /// <c>{modulusBase64}:{exponentBase64}</c> (RSAParameters form — cross-TFM,
    /// no BouncyCastle). This is a SEPARATE key from the entitlement key
    /// (<see cref="SignedEntitlement"/>); compromise of one must not affect the
    /// other. The private half lives only on the license server / HSM.
    /// </summary>
    public const string ResponseSigningPublicKey = PublicKeyPlaceholderMarker;

    /// <summary>Test-only override of the embedded key.</summary>
    internal static string? s_overridePublicKey;

    private static string ResolvePublicKey() => s_overridePublicKey ?? ResponseSigningPublicKey;

    /// <summary>True once a real (non-placeholder) response-signing key is embedded or overridden.</summary>
    public static bool IsConfigured =>
        !string.Equals(ResolvePublicKey(), PublicKeyPlaceholderMarker, StringComparison.Ordinal);

    /// <summary>
    /// Generates a fresh, cryptographically-random 256-bit nonce (base64url, no
    /// padding) for one validation request. Uses <see cref="RandomNumberGenerator"/>
    /// — never <c>System.Random</c> — so it can't be predicted.
    /// </summary>
    public static string NewNonce()
    {
        var bytes = new byte[32];
        using (var rng = RandomNumberGenerator.Create())
            rng.GetBytes(bytes);
        return Convert.ToBase64String(bytes).Replace('+', '-').Replace('/', '_').TrimEnd('=');
    }

    /// <summary>
    /// Canonical string the server signs and the client reconstructs. MUST stay
    /// byte-for-byte identical on both sides (see the server-signing spec).
    /// Fields are newline-separated; capabilities are ordinal-sorted and
    /// comma-joined; absent fields are the empty string.
    /// </summary>
    public static string BuildCanonical(
        string nonce, string? status, string? tier, string? expiresAt, IEnumerable<string>? capabilities)
    {
        var caps = new List<string>();
        if (capabilities is not null)
            foreach (var c in capabilities)
                if (!string.IsNullOrEmpty(c)) caps.Add(c);
        caps.Sort(StringComparer.Ordinal);

        var sb = new StringBuilder();
        sb.Append(nonce ?? string.Empty).Append('\n');
        sb.Append(status ?? string.Empty).Append('\n');
        sb.Append(tier ?? string.Empty).Append('\n');
        sb.Append(expiresAt ?? string.Empty).Append('\n');
        sb.Append(string.Join(",", caps));
        return sb.ToString();
    }

    /// <summary>
    /// Verifies <paramref name="signatureBase64"/> over the canonical form of
    /// the response fields, using the embedded server public key. Returns false
    /// if the key isn't configured, the signature is missing/malformed, or it
    /// doesn't verify. The caller is responsible for having generated
    /// <paramref name="nonce"/> and confirming the response echoed it.
    /// </summary>
    public static bool Verify(
        string nonce, string? status, string? tier, string? expiresAt,
        IEnumerable<string>? capabilities, string? signatureBase64)
    {
        if (!IsConfigured || string.IsNullOrWhiteSpace(signatureBase64))
            return false;

        byte[] sig;
        try { sig = Convert.FromBase64String(signatureBase64); }
        catch (FormatException) { return false; }

        byte[] canonical = Encoding.UTF8.GetBytes(BuildCanonical(nonce, status, tier, expiresAt, capabilities));
        try
        {
            using var rsa = ImportPublicKey(ResolvePublicKey());
            return rsa.VerifyData(canonical, sig, HashAlgorithmName.SHA256, RSASignaturePadding.Pkcs1);
        }
        catch
        {
            return false;
        }
    }

    private static RSA ImportPublicKey(string modExpBase64)
    {
        int sep = modExpBase64.IndexOf(':');
        if (sep <= 0 || sep >= modExpBase64.Length - 1)
            throw new FormatException("Embedded response-signing key must be '{modulusBase64}:{exponentBase64}'.");
        var modulus = Convert.FromBase64String(modExpBase64.Substring(0, sep));
        var exponent = Convert.FromBase64String(modExpBase64.Substring(sep + 1));
        var rsa = RSA.Create();
        rsa.ImportParameters(new RSAParameters { Modulus = modulus, Exponent = exponent });
        return rsa;
    }
}
