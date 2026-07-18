// Copyright (c) AiDotNet. All rights reserved.
// Test support for the offline aidn2 (Ed25519) license-token path: an ephemeral test keypair whose PUBLIC
// half is injected into TensorsLicensePublicKeyProvider and whose PRIVATE half signs real test tokens/CRLs,
// so tests exercise the genuine cryptographic verification path.

#nullable disable

using System;
using System.Collections.Generic;
using System.Text;
using System.Text.Json;
using AiDotNet.Tensors.Licensing;
using Org.BouncyCastle.Crypto;
using Org.BouncyCastle.Crypto.Generators;
using Org.BouncyCastle.Crypto.Parameters;
using Org.BouncyCastle.Crypto.Signers;
using Org.BouncyCastle.Security;

namespace AiDotNet.Tensors.Tests.Licensing;

/// <summary>
/// Produces REAL Ed25519-signed <c>aidn2.</c> tokens (and signed CRLs) against an ephemeral test keypair,
/// and injects that keypair's public half into <see cref="TensorsLicensePublicKeyProvider"/> so
/// <see cref="AsymmetricEntitlementVerifier"/> verifies them offline. Nothing here is a real signing key —
/// the keypair is regenerated per test process and never committed. Mirrors the main SDK's
/// <c>LicenseTestSupport</c>.
/// </summary>
internal static class Aidn2TestSupport
{
    /// <summary>Fixed key-id for the ephemeral test signing keypair (distinct from the real prod-2026a kid).</summary>
    internal const string TestKid = "test-kid-2026a";

    private static readonly AsymmetricCipherKeyPair TestKeyPair = CreateEd25519KeyPair();

    private static AsymmetricCipherKeyPair CreateEd25519KeyPair()
    {
        var generator = new Ed25519KeyPairGenerator();
        generator.Init(new Ed25519KeyGenerationParameters(new SecureRandom()));
        return generator.GenerateKeyPair();
    }

    /// <summary>The raw 32-byte Ed25519 PUBLIC key of the ephemeral test signing keypair.</summary>
    internal static byte[] TestPublicKey =>
        ((Ed25519PublicKeyParameters)TestKeyPair.Public).GetEncoded();

    /// <summary>The default embedded public-key set for tests: { TestKid → test public key }.</summary>
    internal static Dictionary<string, byte[]> DefaultTestKeySet() =>
        new(StringComparer.Ordinal) { [TestKid] = TestPublicKey };

    /// <summary>A fresh, unrelated Ed25519 PRIVATE key — signs a token whose signature will NOT verify.</summary>
    internal static AsymmetricKeyParameter CreateForeignSigningKey() =>
        CreateEd25519KeyPair().Private;

    /// <summary>
    /// Produces a valid <c>aidn2.{base64url(claims)}.{base64url(sig)}</c> token signed with the ephemeral
    /// test PRIVATE key — exactly what <see cref="AsymmetricEntitlementVerifier.Verify"/> verifies. The
    /// claims are signed over the EXACT bytes that are then base64url-encoded into the token, so the
    /// verifier (which verifies over the raw decoded segment) always sees the signed bytes.
    /// </summary>
    internal static string SignedTokenV2(
        string sub = "test-customer",
        string tier = "pro",
        int seats = 5,
        DateTimeOffset? iat = null,
        DateTimeOffset? exp = null,
        string kid = null,
        string alg = "EdDSA",
        AsymmetricKeyParameter signingKey = null,
        string jti = null,
        string[] caps = null,
        string mach = null,
        string scope = null)
    {
        var claims = new Aidn2Claims
        {
            Sub = sub,
            Tier = tier,
            Seats = seats,
            Iat = (iat ?? DateTimeOffset.UtcNow).ToUnixTimeSeconds(),
            Exp = (exp ?? DateTimeOffset.UtcNow.AddDays(30)).ToUnixTimeSeconds(),
            Kid = kid ?? TestKid,
            Alg = alg,
            Jti = jti,
            Caps = caps,
            Mach = mach,
            Scope = scope,
        };

        byte[] claimsBytes = Encoding.UTF8.GetBytes(claims.ToCanonicalJson());
        var signer = new Ed25519Signer();
        signer.Init(true, signingKey ?? TestKeyPair.Private);
        signer.BlockUpdate(claimsBytes, 0, claimsBytes.Length);
        byte[] sig = signer.GenerateSignature();

        return AsymmetricEntitlementVerifier.Prefix + "." +
               Base64UrlEncode(claimsBytes) + "." +
               Base64UrlEncode(sig);
    }

    /// <summary>
    /// Produces a signed CRL in the format <see cref="Ed25519LicenseRevocation.ParseAndVerify"/> verifies:
    /// <c>{ kid, payload:base64url({iat,exp,rkids,rjti}), sig:base64url(Ed25519 over payload) }</c>.
    /// </summary>
    internal static string SignedCrlV2(
        string[] revokedJti = null,
        string[] revokedKids = null,
        DateTimeOffset? exp = null,
        string kid = null,
        AsymmetricKeyParameter signingKey = null)
    {
        var payload = new
        {
            iat = DateTimeOffset.UtcNow.ToUnixTimeSeconds(),
            exp = (exp ?? DateTimeOffset.UtcNow.AddDays(1)).ToUnixTimeSeconds(),
            rkids = revokedKids ?? Array.Empty<string>(),
            rjti = revokedJti ?? Array.Empty<string>(),
        };
        byte[] payloadBytes = Encoding.UTF8.GetBytes(JsonSerializer.Serialize(payload));
        var signer = new Ed25519Signer();
        signer.Init(true, signingKey ?? TestKeyPair.Private);
        signer.BlockUpdate(payloadBytes, 0, payloadBytes.Length);
        byte[] sig = signer.GenerateSignature();

        var envelope = new
        {
            kid = kid ?? TestKid,
            payload = Base64UrlEncode(payloadBytes),
            sig = Base64UrlEncode(sig),
        };
        return JsonSerializer.Serialize(envelope);
    }

    /// <summary>
    /// Scopes the embedded license public-key set to <paramref name="keys"/> for the duration of a test,
    /// restoring the previous set on dispose. Pass <see langword="null"/> to simulate a dev/fork build with
    /// NO embedded public key and assert fail-closed behaviour.
    /// </summary>
    internal static IDisposable WithPublicKeys(IReadOnlyDictionary<string, byte[]> keys)
    {
        var previous = TensorsLicensePublicKeyProvider.CurrentSnapshot();
        TensorsLicensePublicKeyProvider.OverrideForTesting(keys);
        return new RestorePublicKeys(previous);
    }

    private sealed class RestorePublicKeys : IDisposable
    {
        private readonly IReadOnlyDictionary<string, byte[]> _previous;
        public RestorePublicKeys(IReadOnlyDictionary<string, byte[]> previous) => _previous = previous;
        public void Dispose() => TensorsLicensePublicKeyProvider.OverrideForTesting(_previous);
    }

    private static string Base64UrlEncode(byte[] bytes) =>
        Convert.ToBase64String(bytes).Replace('+', '-').Replace('/', '_').TrimEnd('=');
}
