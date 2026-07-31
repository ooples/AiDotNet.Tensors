// Copyright (c) AiDotNet. All rights reserved.
// Tests for the offline aidn2 (Ed25519) license-token verification path and its PersistenceGuard wiring.

#nullable disable

using System;
using AiDotNet.Tensors.Licensing;
using Xunit;

namespace AiDotNet.Tensors.Tests.Licensing;

/// <summary>
/// Verifies the offline <c>aidn2.</c> (Ed25519) path against the real <see cref="AsymmetricEntitlementVerifier"/>
/// with a fixture-injected test public key: a valid machine-bound token → Active with its signed caps;
/// different-machine, expired, unknown-kid / no-embedded-key, forged, revoked, and scope-mismatch tokens →
/// rejected (fail closed); and a valid token wired through <see cref="PersistenceGuard"/> authorises a
/// persistence op. Serial via the shared PersistenceGuard collection — these mutate process-wide static
/// state (the embedded public-key override, the CRL override, env vars, the guard's aidn2 cache), all
/// restored in finally.
/// </summary>
[Collection("PersistenceGuard")]
public class AsymmetricEntitlementVerifierTests
{
    private static readonly DateTimeOffset Now = DateTimeOffset.UtcNow;

    // ───────────── valid machine-bound token ─────────────

    [Fact]
    public void Verify_ValidMachineBoundToken_IsActive_WithCaps()
    {
        using (Aidn2TestSupport.WithPublicKeys(Aidn2TestSupport.DefaultTestKeySet()))
        {
            string token = Aidn2TestSupport.SignedTokenV2(
                mach: TensorsMachineFingerprint.GetMachineIdHash(),
                caps: new[] { "tensors:save", "tensors:load" });

            var r = AsymmetricEntitlementVerifier.Verify(token, Now);

            Assert.True(r.IsValid, r.Message);
            Assert.True(r.HasCapability("tensors:save"));
            Assert.True(r.HasCapability("tensors:load"));
            Assert.False(r.HasCapability("tensors:admin"));
            Assert.Equal("test-customer", r.Tenant);
        }
    }

    [Fact]
    public void Verify_NoMachineClaim_IsActive_Anywhere()
    {
        using (Aidn2TestSupport.WithPublicKeys(Aidn2TestSupport.DefaultTestKeySet()))
        {
            // A token WITHOUT a mach claim is unbound and valid on any machine.
            var r = AsymmetricEntitlementVerifier.Verify(
                Aidn2TestSupport.SignedTokenV2(caps: new[] { "tensors:save" }), Now);
            Assert.True(r.IsValid, r.Message);
        }
    }

    // ───────────── machine binding ─────────────

    [Fact]
    public void Verify_DifferentMachine_Rejected()
    {
        using (Aidn2TestSupport.WithPublicKeys(Aidn2TestSupport.DefaultTestKeySet()))
        {
            var r = AsymmetricEntitlementVerifier.Verify(
                Aidn2TestSupport.SignedTokenV2(mach: "not-this-machine-hash", caps: new[] { "tensors:save" }), Now);
            Assert.False(r.IsValid);
            Assert.Contains("machine", r.Message, StringComparison.OrdinalIgnoreCase);
        }
    }

    // ───────────── expiry ─────────────

    [Fact]
    public void Verify_Expired_Rejected()
    {
        using (Aidn2TestSupport.WithPublicKeys(Aidn2TestSupport.DefaultTestKeySet()))
        {
            // Expired well beyond the 5-minute clock skew.
            var r = AsymmetricEntitlementVerifier.Verify(
                Aidn2TestSupport.SignedTokenV2(exp: Now.AddHours(-1), caps: new[] { "tensors:save" }), Now);
            Assert.False(r.IsValid);
            Assert.Contains("expired", r.Message, StringComparison.OrdinalIgnoreCase);
        }
    }

    // ───────────── unknown kid / no embedded key ─────────────

    [Fact]
    public void Verify_UnknownKid_Rejected()
    {
        using (Aidn2TestSupport.WithPublicKeys(Aidn2TestSupport.DefaultTestKeySet()))
        {
            // Signed with the test key but declaring a kid the build doesn't embed → cannot verify.
            var r = AsymmetricEntitlementVerifier.Verify(
                Aidn2TestSupport.SignedTokenV2(kid: "unknown-kid-9999", caps: new[] { "tensors:save" }), Now);
            Assert.False(r.IsValid);
            Assert.Contains("kid", r.Message, StringComparison.OrdinalIgnoreCase);
        }
    }

    [Fact]
    public void Verify_NoEmbeddedKey_DevBuild_Rejected()
    {
        // Simulate a dev/fork build with NO embedded public key: even a structurally valid token fails closed.
        using (Aidn2TestSupport.WithPublicKeys(null))
        {
            var r = AsymmetricEntitlementVerifier.Verify(
                Aidn2TestSupport.SignedTokenV2(caps: new[] { "tensors:save" }), Now);
            Assert.False(r.IsValid);
        }
    }

    [Fact]
    public void Verify_ForgedWithForeignKey_Rejected()
    {
        using (Aidn2TestSupport.WithPublicKeys(Aidn2TestSupport.DefaultTestKeySet()))
        {
            // Signed by an unrelated private key; verified against the embedded test public key → bad sig.
            var forged = Aidn2TestSupport.SignedTokenV2(
                signingKey: Aidn2TestSupport.CreateForeignSigningKey(), caps: new[] { "tensors:save" });
            var r = AsymmetricEntitlementVerifier.Verify(forged, Now);
            Assert.False(r.IsValid);
            Assert.Contains("signature", r.Message, StringComparison.OrdinalIgnoreCase);
        }
    }

    // ───────────── scope binding ─────────────

    [Fact]
    public void Verify_Scope_AcceptedWhenHostScopeMatches()
    {
        using (Aidn2TestSupport.WithPublicKeys(Aidn2TestSupport.DefaultTestKeySet()))
        using (new ScopeEnv("ci"))
        {
            var r = AsymmetricEntitlementVerifier.Verify(
                Aidn2TestSupport.SignedTokenV2(scope: "ci", caps: new[] { "tensors:save" }), Now);
            Assert.True(r.IsValid, r.Message);
        }
    }

    [Fact]
    public void Verify_Scope_RejectedWhenHostScopeDiffers()
    {
        using (Aidn2TestSupport.WithPublicKeys(Aidn2TestSupport.DefaultTestKeySet()))
        using (new ScopeEnv("prod"))
        {
            var r = AsymmetricEntitlementVerifier.Verify(
                Aidn2TestSupport.SignedTokenV2(scope: "ci", caps: new[] { "tensors:save" }), Now);
            Assert.False(r.IsValid);
            Assert.Contains("scope", r.Message, StringComparison.OrdinalIgnoreCase);
        }
    }

    [Fact]
    public void Verify_Scope_RejectedWhenHostScopeUnset()
    {
        using (Aidn2TestSupport.WithPublicKeys(Aidn2TestSupport.DefaultTestKeySet()))
        using (new ScopeEnv(null))
        {
            var r = AsymmetricEntitlementVerifier.Verify(
                Aidn2TestSupport.SignedTokenV2(scope: "ci", caps: new[] { "tensors:save" }), Now);
            Assert.False(r.IsValid);
        }
    }

    // ───────────── CRL revocation ─────────────

    [Fact]
    public void Verify_RevokedJti_Rejected()
    {
        using (Aidn2TestSupport.WithPublicKeys(Aidn2TestSupport.DefaultTestKeySet()))
        {
            try
            {
                Ed25519LicenseRevocation.OverrideForTesting(
                    Aidn2TestSupport.SignedCrlV2(revokedJti: new[] { "leaked-token-1" }), Now);

                var r = AsymmetricEntitlementVerifier.Verify(
                    Aidn2TestSupport.SignedTokenV2(jti: "leaked-token-1", caps: new[] { "tensors:save" }), Now);
                Assert.False(r.IsValid);
                Assert.Contains("revoked", r.Message, StringComparison.OrdinalIgnoreCase);
            }
            finally
            {
                Ed25519LicenseRevocation.OverrideForTesting(null, Now);
            }
        }
    }

    [Fact]
    public void Verify_NonRevokedJti_IsActive()
    {
        using (Aidn2TestSupport.WithPublicKeys(Aidn2TestSupport.DefaultTestKeySet()))
        {
            try
            {
                Ed25519LicenseRevocation.OverrideForTesting(
                    Aidn2TestSupport.SignedCrlV2(revokedJti: new[] { "some-other-token" }), Now);

                var r = AsymmetricEntitlementVerifier.Verify(
                    Aidn2TestSupport.SignedTokenV2(jti: "my-token", caps: new[] { "tensors:save" }), Now);
                Assert.True(r.IsValid, r.Message);
            }
            finally
            {
                Ed25519LicenseRevocation.OverrideForTesting(null, Now);
            }
        }
    }

    // ───────────── PersistenceGuard wiring ─────────────

    [Fact]
    public void PersistenceGuard_ValidAidn2TokenInEnv_AuthorisesWithNoKeyOrTrial()
    {
        var savedKeyEnv = Environment.GetEnvironmentVariable("AIDOTNET_LICENSE_KEY");
        using (Aidn2TestSupport.WithPublicKeys(Aidn2TestSupport.DefaultTestKeySet()))
        {
            try
            {
                Environment.SetEnvironmentVariable("AIDOTNET_LICENSE_KEY",
                    Aidn2TestSupport.SignedTokenV2(
                        mach: TensorsMachineFingerprint.GetMachineIdHash(),
                        caps: new[] { "tensors:save", "tensors:load" }));
                PersistenceGuard.ClearValidatorCacheForTesting(); // force re-resolve of the aidn2 token

                // No RSA entitlement, no server key, no trial override — the aidn2 token alone authorises.
                PersistenceGuard.EnforceBeforeSave();
                PersistenceGuard.EnforceBeforeLoad();
            }
            finally
            {
                Environment.SetEnvironmentVariable("AIDOTNET_LICENSE_KEY", savedKeyEnv);
                PersistenceGuard.ClearValidatorCacheForTesting();
            }
        }
    }

    [Fact]
    public void PersistenceGuard_Aidn2TokenMissingCapability_DoesNotAuthorise_FallsThroughToKeyPath()
    {
        var savedKeyEnv = Environment.GetEnvironmentVariable("AIDOTNET_LICENSE_KEY");
        // A pre-existing RSA entitlement token in AIDOTNET_LICENSE_TOKEN could grant 'tensors:save' after the
        // aidn2 result declines it, masking the key-path fall-through this test asserts. Clear it for the
        // duration and restore it in finally.
        var savedTokenEnv = Environment.GetEnvironmentVariable("AIDOTNET_LICENSE_TOKEN");
        using (Aidn2TestSupport.WithPublicKeys(Aidn2TestSupport.DefaultTestKeySet()))
        {
            try
            {
                Environment.SetEnvironmentVariable("AIDOTNET_LICENSE_TOKEN", null);

                // Token grants only load — save is not granted by aidn2, so the guard falls through. With
                // AIDOTNET_LICENSE_KEY set to an aidn2 token (not a valid server-key format), the key path
                // then rejects it → LicenseRequiredException (positive-grant-only aidn2 never hard-fails
                // itself, but it also doesn't mask the existing key-path behaviour).
                Environment.SetEnvironmentVariable("AIDOTNET_LICENSE_KEY",
                    Aidn2TestSupport.SignedTokenV2(
                        mach: TensorsMachineFingerprint.GetMachineIdHash(),
                        caps: new[] { "tensors:load" }));
                PersistenceGuard.ClearValidatorCacheForTesting();

                Assert.Throws<LicenseRequiredException>(() => PersistenceGuard.EnforceBeforeSave());
            }
            finally
            {
                Environment.SetEnvironmentVariable("AIDOTNET_LICENSE_KEY", savedKeyEnv);
                Environment.SetEnvironmentVariable("AIDOTNET_LICENSE_TOKEN", savedTokenEnv);
                PersistenceGuard.ClearValidatorCacheForTesting();
            }
        }
    }

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
