// Copyright (c) AiDotNet. All rights reserved.
// Tests for offline RSA-signed entitlement verification + its PersistenceGuard wiring.

#nullable disable

using System;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using AiDotNet.Tensors.Licensing;
using Xunit;

namespace AiDotNet.Tensors.Tests.Licensing;

/// <summary>
/// Verifies the offline, RSA-signed entitlement path: a token signed by the
/// (test) private key is accepted; a forged, tampered, expired, wrong-marker,
/// or placeholder-key token is rejected; and a valid entitlement authorises a
/// persistence op with no license key and no trial tick. Serial via the shared
/// PersistenceGuard collection — these mutate process-wide static state
/// (embedded-key override, the AIDOTNET_LICENSE_TOKEN env var, the guard's
/// entitlement cache), all restored in finally.
/// </summary>
[Collection("PersistenceGuard")]
public class SignedEntitlementTests
{
    private static (RSA rsa, string embeddedKey) NewKeypair()
    {
        // RSA.Create(int) is unavailable on net471 (only RSA.Create(string)),
        // so create then set KeySize — works on net471 and net10.0 alike.
        var rsa = RSA.Create();
        if (rsa.KeySize != 2048) rsa.KeySize = 2048;
        var p = rsa.ExportParameters(includePrivateParameters: false);
        string embedded = Convert.ToBase64String(p.Modulus) + ":" + Convert.ToBase64String(p.Exponent);
        return (rsa, embedded);
    }

    private static string SignToken(RSA rsa, string marker, string expires, string[] capabilities)
    {
        var payload = new { marker, tenant = "ACME Corp.", expires, capabilities };
        var payloadBytes = Encoding.UTF8.GetBytes(JsonSerializer.Serialize(payload));
        var sig = rsa.SignData(payloadBytes, HashAlgorithmName.SHA256, RSASignaturePadding.Pkcs1);
        var envelope = new
        {
            payload = Convert.ToBase64String(payloadBytes),
            signature = Convert.ToBase64String(sig),
        };
        return JsonSerializer.Serialize(envelope);
    }

    /// <summary>Runs <paramref name="body"/> with the embedded key overridden to the test key, always restored.</summary>
    private static void WithEmbeddedKey(string embeddedKey, Action body)
    {
        var saved = SignedEntitlement.s_overridePublicKey;
        try { SignedEntitlement.s_overridePublicKey = embeddedKey; body(); }
        finally { SignedEntitlement.s_overridePublicKey = saved; }
    }

    [Fact]
    public void Verify_ValidToken_IsValid_WithCapabilities()
    {
        var (rsa, embedded) = NewKeypair();
        using (rsa)
        WithEmbeddedKey(embedded, () =>
        {
            string token = SignToken(rsa, SignedEntitlement.Marker, "2099-12-31", new[] { "tensors:save", "tensors:load" });
            var r = SignedEntitlement.Verify(token);
            Assert.True(r.IsValid, r.Message);
            Assert.True(r.HasCapability("tensors:save"));
            Assert.True(r.HasCapability("tensors:load"));
            Assert.False(r.HasCapability("tensors:admin"));
        });
    }

    [Fact]
    public void Verify_ForgedWithDifferentKey_Rejected()
    {
        var (rsaReal, embedded) = NewKeypair();
        var (rsaAttacker, _) = NewKeypair();
        using (rsaReal)
        using (rsaAttacker)
        WithEmbeddedKey(embedded, () =>
        {
            // Signed by the attacker's key, verified against the embedded (real) key.
            string forged = SignToken(rsaAttacker, SignedEntitlement.Marker, "2099-12-31", new[] { "tensors:save" });
            var r = SignedEntitlement.Verify(forged);
            Assert.False(r.IsValid);
        });
    }

    [Fact]
    public void Verify_TamperedSignature_Rejected()
    {
        var (rsa, embedded) = NewKeypair();
        using (rsa)
        WithEmbeddedKey(embedded, () =>
        {
            string token = SignToken(rsa, SignedEntitlement.Marker, "2099-12-31", new[] { "tensors:save" });
            // Flip a bit in the base64 signature field.
            using var doc = JsonDocument.Parse(token);
            string sig = doc.RootElement.GetProperty("signature").GetString();
            var sigBytes = Convert.FromBase64String(sig);
            sigBytes[0] ^= 0xFF;
            string payload = doc.RootElement.GetProperty("payload").GetString();
            string tampered = JsonSerializer.Serialize(new { payload, signature = Convert.ToBase64String(sigBytes) });
            var r = SignedEntitlement.Verify(tampered);
            Assert.False(r.IsValid);
        });
    }

    [Fact]
    public void Verify_Expired_Rejected()
    {
        var (rsa, embedded) = NewKeypair();
        using (rsa)
        WithEmbeddedKey(embedded, () =>
        {
            string token = SignToken(rsa, SignedEntitlement.Marker, "2000-01-01", new[] { "tensors:save" });
            var r = SignedEntitlement.Verify(token);
            Assert.False(r.IsValid);
        });
    }

    [Fact]
    public void Verify_WrongMarker_Rejected()
    {
        var (rsa, embedded) = NewKeypair();
        using (rsa)
        WithEmbeddedKey(embedded, () =>
        {
            string token = SignToken(rsa, "Wrong-Marker", "2099-12-31", new[] { "tensors:save" });
            var r = SignedEntitlement.Verify(token);
            Assert.False(r.IsValid);
        });
    }

    [Fact]
    public void Verify_PlaceholderEmbeddedKey_Rejected()
    {
        // With the shipped placeholder key, even a structurally valid token is refused.
        var (rsa, _) = NewKeypair();
        using (rsa)
        {
            string token = SignToken(rsa, SignedEntitlement.Marker, "2099-12-31", new[] { "tensors:save" });
            var r = SignedEntitlement.Verify(token); // no override → placeholder key in effect
            Assert.False(r.IsValid);
        }
    }

    [Fact]
    public void TryVerifyConfigured_PlaceholderKey_IsInert_ReturnsNull()
    {
        // Even with a token present in the env, the placeholder key makes the
        // feature inert (returns null → guard falls through to key/trial path).
        var saved = Environment.GetEnvironmentVariable("AIDOTNET_LICENSE_TOKEN");
        var (rsa, _) = NewKeypair();
        using (rsa)
        try
        {
            Environment.SetEnvironmentVariable("AIDOTNET_LICENSE_TOKEN",
                SignToken(rsa, SignedEntitlement.Marker, "2099-12-31", new[] { "tensors:save" }));
            Assert.Null(SignedEntitlement.TryVerifyConfigured()); // placeholder → inert
        }
        finally { Environment.SetEnvironmentVariable("AIDOTNET_LICENSE_TOKEN", saved); }
    }

    [Fact]
    public void PersistenceGuard_ValidEntitlement_AuthorisesWithNoKeyOrTrial()
    {
        var savedEnv = Environment.GetEnvironmentVariable("AIDOTNET_LICENSE_TOKEN");
        var savedKey = SignedEntitlement.s_overridePublicKey;
        var (rsa, embedded) = NewKeypair();
        using (rsa)
        try
        {
            SignedEntitlement.s_overridePublicKey = embedded;
            Environment.SetEnvironmentVariable("AIDOTNET_LICENSE_TOKEN",
                SignToken(rsa, SignedEntitlement.Marker, "2099-12-31", new[] { "tensors:save", "tensors:load" }));
            PersistenceGuard.ClearValidatorCacheForTesting(); // force re-resolve of the entitlement

            // No license key, no trial override — the signed entitlement alone authorises.
            PersistenceGuard.EnforceBeforeSave();
            PersistenceGuard.EnforceBeforeLoad();
        }
        finally
        {
            Environment.SetEnvironmentVariable("AIDOTNET_LICENSE_TOKEN", savedEnv);
            SignedEntitlement.s_overridePublicKey = savedKey;
            PersistenceGuard.ClearValidatorCacheForTesting();
        }
    }

    [Fact]
    public void PersistenceGuard_EntitlementMissingCapability_Throws()
    {
        var savedEnv = Environment.GetEnvironmentVariable("AIDOTNET_LICENSE_TOKEN");
        var savedKey = SignedEntitlement.s_overridePublicKey;
        var (rsa, embedded) = NewKeypair();
        using (rsa)
        try
        {
            SignedEntitlement.s_overridePublicKey = embedded;
            // Grants only load — save must be refused.
            Environment.SetEnvironmentVariable("AIDOTNET_LICENSE_TOKEN",
                SignToken(rsa, SignedEntitlement.Marker, "2099-12-31", new[] { "tensors:load" }));
            PersistenceGuard.ClearValidatorCacheForTesting();

            Assert.Throws<LicenseRequiredException>(() => PersistenceGuard.EnforceBeforeSave());
        }
        finally
        {
            Environment.SetEnvironmentVariable("AIDOTNET_LICENSE_TOKEN", savedEnv);
            SignedEntitlement.s_overridePublicKey = savedKey;
            PersistenceGuard.ClearValidatorCacheForTesting();
        }
    }

    [Fact]
    public void PersistenceGuard_InvalidEntitlementPresent_Throws_DoesNotFallThroughToTrial()
    {
        var savedEnv = Environment.GetEnvironmentVariable("AIDOTNET_LICENSE_TOKEN");
        var savedKey = SignedEntitlement.s_overridePublicKey;
        var (rsaReal, embedded) = NewKeypair();
        var (rsaAttacker, _) = NewKeypair();
        using (rsaReal)
        using (rsaAttacker)
        try
        {
            SignedEntitlement.s_overridePublicKey = embedded;
            // Forged token (attacker key) — present but invalid. Must hard-fail,
            // NOT silently degrade to the free trial path.
            Environment.SetEnvironmentVariable("AIDOTNET_LICENSE_TOKEN",
                SignToken(rsaAttacker, SignedEntitlement.Marker, "2099-12-31", new[] { "tensors:save" }));
            PersistenceGuard.ClearValidatorCacheForTesting();

            Assert.Throws<LicenseRequiredException>(() => PersistenceGuard.EnforceBeforeSave());
        }
        finally
        {
            Environment.SetEnvironmentVariable("AIDOTNET_LICENSE_TOKEN", savedEnv);
            SignedEntitlement.s_overridePublicKey = savedKey;
            PersistenceGuard.ClearValidatorCacheForTesting();
        }
    }
}
