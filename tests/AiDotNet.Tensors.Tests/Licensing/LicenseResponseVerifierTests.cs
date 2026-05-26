// Copyright (c) AiDotNet. All rights reserved.
// Tests for signed online-validation-response verification (anti-spoof / anti-replay).

#nullable disable

using System;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using AiDotNet.Tensors.Licensing;
using Xunit;

namespace AiDotNet.Tensors.Tests.Licensing;

/// <summary>
/// Verifies that, when RequireSignedResponse is on, the online validation
/// response must carry a valid RSA signature over the canonical
/// (nonce|status|tier|expires|caps) form — closing the server-spoof / replay
/// bypass — while the default (unsigned) path is unchanged. Serial via the
/// PersistenceGuard collection (mutates the embedded response-key override).
/// </summary>
[Collection("PersistenceGuard")]
public class LicenseResponseVerifierTests
{
    private static (RSA rsa, string embeddedKey) NewKeypair()
    {
        var rsa = RSA.Create();
        if (rsa.KeySize != 2048) rsa.KeySize = 2048;
        var p = rsa.ExportParameters(false);
        return (rsa, Convert.ToBase64String(p.Modulus) + ":" + Convert.ToBase64String(p.Exponent));
    }

    private static string Sign(RSA rsa, string canonical)
        => Convert.ToBase64String(rsa.SignData(Encoding.UTF8.GetBytes(canonical),
            HashAlgorithmName.SHA256, RSASignaturePadding.Pkcs1));

    private static void WithEmbeddedKey(string key, Action body)
    {
        var saved = LicenseResponseVerifier.s_overridePublicKey;
        try { LicenseResponseVerifier.s_overridePublicKey = key; body(); }
        finally { LicenseResponseVerifier.s_overridePublicKey = saved; }
    }

    [Fact]
    public void NewNonce_IsUnique_AndNonEmpty()
    {
        var a = LicenseResponseVerifier.NewNonce();
        var b = LicenseResponseVerifier.NewNonce();
        Assert.False(string.IsNullOrWhiteSpace(a));
        Assert.NotEqual(a, b);
    }

    [Fact]
    public void Verify_ValidSignature_True()
    {
        var (rsa, embedded) = NewKeypair();
        using (rsa)
        WithEmbeddedKey(embedded, () =>
        {
            string nonce = LicenseResponseVerifier.NewNonce();
            var caps = new[] { "tensors:load", "tensors:save" };
            string canonical = LicenseResponseVerifier.BuildCanonical(nonce, "active", "enterprise", "2099-12-31", caps);
            string sig = Sign(rsa, canonical);
            Assert.True(LicenseResponseVerifier.Verify(nonce, "active", "enterprise", "2099-12-31", caps, sig));
        });
    }

    [Fact]
    public void Verify_WrongNonce_False()
    {
        var (rsa, embedded) = NewKeypair();
        using (rsa)
        WithEmbeddedKey(embedded, () =>
        {
            string nonce = LicenseResponseVerifier.NewNonce();
            string canonical = LicenseResponseVerifier.BuildCanonical(nonce, "active", "enterprise", "2099-12-31", null);
            string sig = Sign(rsa, canonical);
            // Replay against a DIFFERENT nonce (what the next request would send).
            Assert.False(LicenseResponseVerifier.Verify(LicenseResponseVerifier.NewNonce(), "active", "enterprise", "2099-12-31", null, sig));
        });
    }

    [Fact]
    public void Verify_ForgedWithOtherKey_False()
    {
        var (rsaReal, embedded) = NewKeypair();
        var (rsaAttacker, _) = NewKeypair();
        using (rsaReal)
        using (rsaAttacker)
        WithEmbeddedKey(embedded, () =>
        {
            string nonce = LicenseResponseVerifier.NewNonce();
            string canonical = LicenseResponseVerifier.BuildCanonical(nonce, "active", null, null, null);
            string sig = Sign(rsaAttacker, canonical);
            Assert.False(LicenseResponseVerifier.Verify(nonce, "active", null, null, null, sig));
        });
    }

    [Fact]
    public void Verify_PlaceholderKey_NotConfigured_False()
    {
        // No override → placeholder embedded key → verification unavailable.
        Assert.False(LicenseResponseVerifier.IsConfigured);
        Assert.False(LicenseResponseVerifier.Verify("n", "active", null, null, null, "AAAA"));
    }

    [Fact]
    public void ParseResponse_RequireSigned_ValidSignature_Active()
    {
        var (rsa, embedded) = NewKeypair();
        using (rsa)
        WithEmbeddedKey(embedded, () =>
        {
            string nonce = LicenseResponseVerifier.NewNonce();
            var caps = new[] { "tensors:save", "tensors:load" };
            string sig = Sign(rsa, LicenseResponseVerifier.BuildCanonical(nonce, "active", "enterprise", "2099-12-31", caps));
            string json = JsonSerializer.Serialize(new
            {
                status = "active",
                tier = "enterprise",
                expires_at = "2099-12-31",
                capabilities = caps,
                signature = sig,
            });
            var r = LicenseValidator.ParseResponse(json, 200, nonce, requireSigned: true);
            Assert.Equal(LicenseKeyStatus.Active, r.Status);
            Assert.True(r.HasCapability("tensors:save"));
        });
    }

    [Fact]
    public void ParseResponse_RequireSigned_MissingSignature_Invalid()
    {
        var (rsa, embedded) = NewKeypair();
        using (rsa)
        WithEmbeddedKey(embedded, () =>
        {
            string nonce = LicenseResponseVerifier.NewNonce();
            // A spoofed server returns "active" with NO signature.
            string json = JsonSerializer.Serialize(new { status = "active", tier = "enterprise" });
            var r = LicenseValidator.ParseResponse(json, 200, nonce, requireSigned: true);
            Assert.Equal(LicenseKeyStatus.Invalid, r.Status);
        });
    }

    [Fact]
    public void ParseResponse_RequireSigned_TamperedStatus_Invalid()
    {
        var (rsa, embedded) = NewKeypair();
        using (rsa)
        WithEmbeddedKey(embedded, () =>
        {
            string nonce = LicenseResponseVerifier.NewNonce();
            // Signature was computed over status="expired"; attacker flips the
            // body to "active" but can't re-sign → canonical mismatch → Invalid.
            string sig = Sign(rsa, LicenseResponseVerifier.BuildCanonical(nonce, "expired", null, null, null));
            string json = JsonSerializer.Serialize(new { status = "active", signature = sig });
            var r = LicenseValidator.ParseResponse(json, 200, nonce, requireSigned: true);
            Assert.Equal(LicenseKeyStatus.Invalid, r.Status);
        });
    }

    [Fact]
    public void ParseResponse_NotRequiringSignature_UnsignedStillParses()
    {
        // Default path (requireSigned: false) is unchanged — unsigned response parses normally.
        string json = JsonSerializer.Serialize(new { status = "active", capabilities = new[] { "tensors:save" } });
        var r = LicenseValidator.ParseResponse(json, 200);
        Assert.Equal(LicenseKeyStatus.Active, r.Status);
        Assert.True(r.HasCapability("tensors:save"));
    }
}
