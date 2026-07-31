// Copyright (c) AiDotNet. All rights reserved.
// Tests for the auto-offline client glue (TensorsOnlineLicenseServices): sibling-URL derivation, the shared
// per-key offline-token cache filename scheme, cached-token read → verified EntitlementResult, and CRL disk
// round-trip. All disk I/O is redirected to a temp dir via OverrideCacheDirForTesting so the real user
// profile is never touched.

#nullable disable

using System;
using System.IO;
using System.Security.Cryptography;
using System.Text;
using AiDotNet.Tensors.Licensing;
using Xunit;

namespace AiDotNet.Tensors.Tests.Licensing;

/// <summary>
/// Exercises <see cref="TensorsOnlineLicenseServices"/> — the standalone-Tensors auto-offline glue that
/// mints/caches offline <c>aidn2</c> tokens and the signed CRL off a successful online validation. Serial via
/// the shared PersistenceGuard collection because these mutate process-wide static state (the cache-dir
/// override and the injected public-key set), all restored in <c>finally</c>/<c>using</c>.
/// </summary>
[Collection("PersistenceGuard")]
public class TensorsOnlineLicenseServicesTests
{
    private static readonly DateTimeOffset Now = DateTimeOffset.UtcNow;

    private const string ValidateUrl =
        "https://yfkqwpgjahoamlgckjib.supabase.co/functions/v1/validate-license";

    // An AIDN-* server-validated key (the only key type for which an offline token is minted).
    private const string ServerKey = "AIDN-PROD-PRO-abcdef12";

    // ───────────── sibling-URL derivation ─────────────

    [Fact]
    public void DeriveFunctionUrl_SwapsTrailingSegment()
    {
        Assert.Equal(
            "https://yfkqwpgjahoamlgckjib.supabase.co/functions/v1/get-revocations",
            TensorsOnlineLicenseServices.DeriveFunctionUrl(ValidateUrl, "get-revocations"));

        Assert.Equal(
            "https://yfkqwpgjahoamlgckjib.supabase.co/functions/v1/issue-license",
            TensorsOnlineLicenseServices.DeriveFunctionUrl(ValidateUrl, "issue-license"));
    }

    [Fact]
    public void DeriveFunctionUrl_PreservesQueryTail()
    {
        Assert.Equal(
            "https://host/functions/v1/get-revocations?x=1",
            TensorsOnlineLicenseServices.DeriveFunctionUrl(
                "https://host/functions/v1/validate-license?x=1", "get-revocations"));
    }

    [Fact]
    public void DeriveFunctionUrl_NonMatchingUrl_ReturnsNull()
    {
        Assert.Null(TensorsOnlineLicenseServices.DeriveFunctionUrl("https://host/custom-stub", "get-revocations"));
        Assert.Null(TensorsOnlineLicenseServices.DeriveFunctionUrl("", "get-revocations"));
    }

    // ───────────── shared per-key cache filename scheme (byte-for-byte with the main SDK) ─────────────

    [Fact]
    public void CachedTokenFilename_MatchesMainSdkPerKeyScheme()
    {
        using var dir = new TempCacheDir();
        using (TensorsOnlineLicenseServices.OverrideCacheDirForTesting(dir.Path))
        {
            TensorsOnlineLicenseServices.CacheOfflineTokenForTesting(ServerKey, "aidn2.x.y");

            // Independently reproduce the main SDK's scheme: "offline-" + first 8 bytes of
            // SHA256("license-cache:" + key) as lowercase hex + ".token".
            string expected = "offline-" + MainSdkShortHash(ServerKey) + ".token";
            Assert.True(File.Exists(Path.Combine(dir.Path, expected)),
                "expected cache file '" + expected + "' was not written");
        }
    }

    // ───────────── cached-token read → verified EntitlementResult with caps ─────────────

    [Fact]
    public void TryValidateCachedOfflineToken_ReadsSharedScheme_ReturnsActiveWithCaps()
    {
        using var dir = new TempCacheDir();
        using (Aidn2TestSupport.WithPublicKeys(Aidn2TestSupport.DefaultTestKeySet()))
        using (TensorsOnlineLicenseServices.OverrideCacheDirForTesting(dir.Path))
        {
            // Mint a REAL machine-bound aidn2 token (verifies against the injected test public key on THIS
            // machine) and cache it under the per-key filename the refresh path uses.
            string token = Aidn2TestSupport.SignedTokenV2(
                mach: TensorsMachineFingerprint.GetMachineIdHash(),
                caps: new[] { "tensors:save", "tensors:load" });
            TensorsOnlineLicenseServices.CacheOfflineTokenForTesting(ServerKey, token);

            var r = TensorsOnlineLicenseServices.TryValidateCachedOfflineToken(ServerKey);

            Assert.NotNull(r);
            Assert.True(r.IsValid, r.Message);
            Assert.True(r.HasCapability("tensors:save"));
            Assert.True(r.HasCapability("tensors:load"));
            Assert.False(r.HasCapability("tensors:admin"));
        }
    }

    [Fact]
    public void TryValidateCachedOfflineToken_DifferentMachine_ReturnsNull()
    {
        using var dir = new TempCacheDir();
        using (Aidn2TestSupport.WithPublicKeys(Aidn2TestSupport.DefaultTestKeySet()))
        using (TensorsOnlineLicenseServices.OverrideCacheDirForTesting(dir.Path))
        {
            // A token bound to a DIFFERENT machine must never grant here (fail closed).
            string token = Aidn2TestSupport.SignedTokenV2(
                mach: "not-this-machine-hash", caps: new[] { "tensors:save" });
            TensorsOnlineLicenseServices.CacheOfflineTokenForTesting(ServerKey, token);

            Assert.Null(TensorsOnlineLicenseServices.TryValidateCachedOfflineToken(ServerKey));
        }
    }

    [Fact]
    public void TryValidateCachedOfflineToken_NoFile_ReturnsNull()
    {
        using var dir = new TempCacheDir();
        using (TensorsOnlineLicenseServices.OverrideCacheDirForTesting(dir.Path))
        {
            Assert.Null(TensorsOnlineLicenseServices.TryValidateCachedOfflineToken("AIDN-PROD-PRO-deadbeef"));
        }
    }

    // ───────────── CRL disk round-trip ─────────────

    [Fact]
    public void CrlDiskRoundTrip_WriteThenRead_ReturnsSameBytes()
    {
        using var dir = new TempCacheDir();
        using (TensorsOnlineLicenseServices.OverrideCacheDirForTesting(dir.Path))
        {
            string crl = Aidn2TestSupport.SignedCrlV2(revokedJti: new[] { "leaked-token-1" });
            TensorsOnlineLicenseServices.CacheCrlForTesting(crl);

            Assert.True(File.Exists(Path.Combine(dir.Path, "revocations.crl")));
            Assert.Equal(crl, TensorsOnlineLicenseServices.ReadCachedCrl());
        }
    }

    [Fact]
    public void ReadCachedCrl_NoFile_ReturnsNull()
    {
        using var dir = new TempCacheDir();
        using (TensorsOnlineLicenseServices.OverrideCacheDirForTesting(dir.Path))
        {
            Assert.Null(TensorsOnlineLicenseServices.ReadCachedCrl());
        }
    }

    [Fact]
    public void CachedCrl_IsSignatureVerified_AndInstallable()
    {
        // The cached CRL round-trips AND is a genuine signature-verified CRL that TryInstallFetched accepts.
        using (Aidn2TestSupport.WithPublicKeys(Aidn2TestSupport.DefaultTestKeySet()))
        {
            try
            {
                string crl = Aidn2TestSupport.SignedCrlV2(revokedJti: new[] { "leaked-token-1" });
                Assert.True(Ed25519LicenseRevocation.TryInstallFetched(crl, Now));
            }
            finally
            {
                // Reset shared static state so this test doesn't leak a fetched CRL into others, even if the
                // installation/assertion above throws.
                Ed25519LicenseRevocation.OverrideForTesting(null, Now);
            }
        }
    }

    // ───────────── helpers ─────────────

    /// <summary>Independent reproduction of the main SDK's OnlineLicenseServices.ShortHash.</summary>
    private static string MainSdkShortHash(string value)
    {
        byte[] bytes = Encoding.UTF8.GetBytes("license-cache:" + value);
        using var sha = SHA256.Create();
        byte[] hash = sha.ComputeHash(bytes);
        var sb = new StringBuilder(16);
        for (int i = 0; i < 8; i++)
        {
            sb.Append(hash[i].ToString("x2"));
        }

        return sb.ToString();
    }

    private sealed class TempCacheDir : IDisposable
    {
        public string Path { get; }

        public TempCacheDir()
        {
            Path = System.IO.Path.Combine(
                System.IO.Path.GetTempPath(), "aidn-tensors-online-tests-" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(Path);
        }

        public void Dispose()
        {
            try { Directory.Delete(Path, recursive: true); }
            catch { /* best effort */ }
        }
    }
}
