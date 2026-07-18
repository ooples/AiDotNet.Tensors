// Copyright (c) AiDotNet. All rights reserved.
// Tests for offline RSA-signed entitlement verification + its PersistenceGuard wiring.

#nullable disable

using System;
using System.Collections.Generic;
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

    private static string SignToken(RSA rsa, string marker, string expires, string[] capabilities,
        string scope = null, string jti = null)
    {
        // Dictionary (not an anonymous type) so optional scope/jti claims can be included conditionally; the
        // RSA signature is computed over the EXACT serialized bytes, so claim order/shape doesn't matter.
        var payload = new Dictionary<string, object>
        {
            ["marker"] = marker,
            ["tenant"] = "ACME Corp.",
            ["expires"] = expires,
            ["capabilities"] = capabilities,
        };
        if (scope != null) payload["scope"] = scope;
        if (jti != null) payload["jti"] = jti;

        var payloadBytes = Encoding.UTF8.GetBytes(JsonSerializer.Serialize(payload));
        var sig = rsa.SignData(payloadBytes, HashAlgorithmName.SHA256, RSASignaturePadding.Pkcs1);
        var envelope = new
        {
            payload = Convert.ToBase64String(payloadBytes),
            signature = Convert.ToBase64String(sig),
        };
        return JsonSerializer.Serialize(envelope);
    }

    /// <summary>Signs an RSA CRL envelope (same key/format as the entitlement) revoking <paramref name="rjti"/>.</summary>
    private static string SignCrl(RSA rsa, string[] rjti, DateTimeOffset iat, DateTimeOffset exp)
    {
        var payload = new { iat = iat.ToUnixTimeSeconds(), exp = exp.ToUnixTimeSeconds(), rjti };
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
        // Production now embeds a REAL entitlement key, so this test explicitly
        // pins the placeholder via the override to exercise the inert-build path
        // (rather than relying on the default embedded key still being the placeholder).
        var saved = Environment.GetEnvironmentVariable("AIDOTNET_LICENSE_TOKEN");
        var savedKey = SignedEntitlement.s_overridePublicKey;
        var (rsa, _) = NewKeypair();
        using (rsa)
        try
        {
            SignedEntitlement.s_overridePublicKey = SignedEntitlement.PublicKeyPlaceholderMarker;
            Environment.SetEnvironmentVariable("AIDOTNET_LICENSE_TOKEN",
                SignToken(rsa, SignedEntitlement.Marker, "2099-12-31", new[] { "tensors:save" }));
            Assert.Null(SignedEntitlement.TryVerifyConfigured()); // placeholder → inert
        }
        finally
        {
            Environment.SetEnvironmentVariable("AIDOTNET_LICENSE_TOKEN", saved);
            SignedEntitlement.s_overridePublicKey = savedKey;
        }
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

    // ───────────── RSA scope binding (mirrors the aidn2 scope tests) ─────────────

    [Fact]
    public void Verify_Scope_AcceptedWhenHostScopeMatches()
    {
        var (rsa, embedded) = NewKeypair();
        using (rsa)
        using (new ScopeEnv("ci"))
        WithEmbeddedKey(embedded, () =>
        {
            string token = SignToken(rsa, SignedEntitlement.Marker, "2099-12-31",
                new[] { "tensors:save" }, scope: "ci");
            var r = SignedEntitlement.Verify(token);
            Assert.True(r.IsValid, r.Message);
        });
    }

    [Fact]
    public void Verify_Scope_RejectedWhenHostScopeDiffers()
    {
        var (rsa, embedded) = NewKeypair();
        using (rsa)
        using (new ScopeEnv("prod"))
        WithEmbeddedKey(embedded, () =>
        {
            string token = SignToken(rsa, SignedEntitlement.Marker, "2099-12-31",
                new[] { "tensors:save" }, scope: "ci");
            var r = SignedEntitlement.Verify(token);
            Assert.False(r.IsValid);
            Assert.Contains("scope", r.Message, StringComparison.OrdinalIgnoreCase);
        });
    }

    [Fact]
    public void Verify_Scope_RejectedWhenHostScopeUnset()
    {
        var (rsa, embedded) = NewKeypair();
        using (rsa)
        using (new ScopeEnv(null))
        WithEmbeddedKey(embedded, () =>
        {
            // A scoped entitlement on an unscoped host is rejected (scope "" != "ci").
            string token = SignToken(rsa, SignedEntitlement.Marker, "2099-12-31",
                new[] { "tensors:save" }, scope: "ci");
            var r = SignedEntitlement.Verify(token);
            Assert.False(r.IsValid);
        });
    }

    // ───────────── RSA revocation (mirrors the aidn2 CRL tests) ─────────────

    [Fact]
    public void Verify_RevokedJti_Rejected()
    {
        var (rsa, embedded) = NewKeypair();
        var now = DateTimeOffset.UtcNow;
        using (rsa)
        WithEmbeddedKey(embedded, () =>
        {
            try
            {
                // A CRL signed by the SAME entitlement key (unexpired) revokes this jti.
                TensorsLicenseRevocation.OverrideForTesting(
                    SignCrl(rsa, new[] { "leaked-ent-1" }, now, now.AddDays(1)), now);

                string token = SignToken(rsa, SignedEntitlement.Marker, "2099-12-31",
                    new[] { "tensors:save" }, jti: "leaked-ent-1");
                var r = SignedEntitlement.Verify(token);
                Assert.False(r.IsValid);
                Assert.Contains("revoked", r.Message, StringComparison.OrdinalIgnoreCase);
            }
            finally
            {
                TensorsLicenseRevocation.OverrideForTesting(null, now);
            }
        });
    }

    [Fact]
    public void Verify_NonRevokedJti_IsValid()
    {
        var (rsa, embedded) = NewKeypair();
        var now = DateTimeOffset.UtcNow;
        using (rsa)
        WithEmbeddedKey(embedded, () =>
        {
            try
            {
                TensorsLicenseRevocation.OverrideForTesting(
                    SignCrl(rsa, new[] { "some-other-ent" }, now, now.AddDays(1)), now);

                string token = SignToken(rsa, SignedEntitlement.Marker, "2099-12-31",
                    new[] { "tensors:save" }, jti: "my-ent");
                var r = SignedEntitlement.Verify(token);
                Assert.True(r.IsValid, r.Message);
            }
            finally
            {
                TensorsLicenseRevocation.OverrideForTesting(null, now);
            }
        });
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

    /// <summary>Sets AIDOTNET_LICENSE_SCOPE for the scope tests, restoring the prior value on dispose.</summary>
    private sealed class ScopeEnv : IDisposable
    {
        private readonly string _previous;
        public ScopeEnv(string value)
        {
            _previous = Environment.GetEnvironmentVariable("AIDOTNET_LICENSE_SCOPE");
            Environment.SetEnvironmentVariable("AIDOTNET_LICENSE_SCOPE", value);
        }
        public void Dispose() => Environment.SetEnvironmentVariable("AIDOTNET_LICENSE_SCOPE", _previous);
    }
}
