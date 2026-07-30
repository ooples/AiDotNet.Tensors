// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Text.Json;
#if !NET471
using System.Net.Http;
#endif

namespace AiDotNet.Tensors.Licensing;

/// <summary>
/// Client glue for the v2 hybrid online→offline licensing model in a STANDALONE Tensors app (used without the
/// main <c>AiDotNet</c> SDK): fetches the signed revocation list (CRL) and mints/caches short-lived offline
/// <c>aidn2</c> tokens off a SUCCESSFUL online validation, so an <c>AIDN-*</c> server key keeps working —
/// with the correct capabilities and while staying revocable — after the machine goes offline. The Tensors
/// port of the main SDK's <c>OnlineLicenseServices</c>, using <c>System.Text.Json</c> (no Newtonsoft) and the
/// Tensors verifiers (<see cref="AsymmetricEntitlementVerifier"/>, <see cref="Ed25519LicenseRevocation"/>).
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> An <c>AIDN-*</c> key normally must phone home to the license server on every
/// run. That breaks air-gapped / offline use. This helper closes the gap: right after a successful online
/// check, it (a) downloads the signed revocation list so leaked keys can be denied even offline, and (b) asks
/// the server for a short-lived, machine-locked offline token and caches it. Next time the server is
/// unreachable, the SDK verifies that cached token locally (against the embedded public key) instead of
/// failing — until the token expires and a fresh online check is needed.</para>
///
/// <para><b>Shared cache with the main SDK:</b> the CRL is cached to <c>~/.aidotnet/revocations.crl</c> and
/// offline tokens to <c>~/.aidotnet/offline-&lt;hash&gt;.token</c> using the SAME per-key filename scheme the
/// main SDK's <c>OnlineLicenseServices</c> uses (<c>offline-</c> + first 8 bytes of
/// <c>SHA256("license-cache:" + key)</c> hex + <c>.token</c>), so the two packages never collide and either
/// can consume the other's cached token / CRL.</para>
///
/// <para><b>Fail-open / best-effort:</b> every network + disk operation here is wrapped so a failure NEVER
/// breaks or blocks the validation that triggered it. The refresh runs on a background thread/task (off the
/// hot path) and is throttled so it doesn't hit the network on every validation. The offline fallback only
/// ever GRANTS when a genuinely signature-valid, unexpired, machine-matched token is present — it can't
/// weaken security, only preserve availability.</para>
/// </remarks>
internal static class TensorsOnlineLicenseServices
{
    /// <summary>Minimum spacing between background refreshes (per machine, tracked by cache-file mtime), so a
    /// tight validate loop doesn't hammer the endpoints. Well inside the offline token's exp so it stays fresh.</summary>
    private static readonly TimeSpan RefreshInterval = TimeSpan.FromHours(12);

    private static readonly object FileLock = new();
    private static string? _cacheDirOverrideForTests;

#if !NET471
    private static readonly HttpClient Http = new() { Timeout = TimeSpan.FromSeconds(10) };
#else
    /// <summary>Explicit HTTP timeout for the net471 <see cref="System.Net.HttpWebRequest"/> path (WebClient
    /// has none), matching the modern <see cref="HttpClient"/> 10-second budget.</summary>
    private const int Net471TimeoutMs = 10_000;
#endif

    private static string CacheDir => _cacheDirOverrideForTests ?? Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".aidotnet");

    private static string CrlPath => Path.Combine(CacheDir, "revocations.crl");

    // One cache file per key (keyed by a short hash of the key, never the plaintext key) so multiple keys on
    // one machine don't clobber each other's offline tokens. IDENTICAL scheme to the main SDK's
    // OnlineLicenseServices.TokenPath so either package consumes the other's token.
    private static string TokenPath(string licenseKey) =>
        Path.Combine(CacheDir, "offline-" + ShortHash(licenseKey) + ".token");

    /// <summary>TEST-ONLY: redirects the cache directory so tests don't touch the real user profile.</summary>
    internal static IDisposable OverrideCacheDirForTesting(string dir)
    {
        var previous = _cacheDirOverrideForTests;
        _cacheDirOverrideForTests = dir;
        return new CacheDirScope(previous);
    }

    /// <summary>TEST-ONLY: writes an offline token to the same per-key cache path <see cref="RefreshCore"/> uses.</summary>
    internal static void CacheOfflineTokenForTesting(string licenseKey, string token) =>
        WriteAtomic(TokenPath(licenseKey), token);

    /// <summary>TEST-ONLY: writes a CRL to the same cache path a background refresh would.</summary>
    internal static void CacheCrlForTesting(string crl) => WriteAtomic(CrlPath, crl);

    // ───────────────────────── refresh (background, off the hot path) ─────────────────────────

    /// <summary>
    /// Fire-and-forget refresh triggered after a successful ONLINE validation: pulls the latest signed CRL and
    /// (for a server <c>AIDN-*</c> key) mints + caches a fresh offline token. Never blocks the caller and never
    /// throws. Throttled by <see cref="RefreshInterval"/>.
    /// </summary>
    internal static void RefreshInBackground(string validateUrl, string licenseKey, string machineIdHash)
    {
        // Throttle the CRL and this key's offline token INDEPENDENTLY: a fresh CRL must not suppress minting
        // a missing token for this key, and a fresh token (for this or any other key) must not suppress the
        // CRL refresh. Skip scheduling only when BOTH relevant artifacts are already fresh.
        if (!ShouldRefreshCrl() && !ShouldRefreshToken(licenseKey))
        {
            return;
        }

        try
        {
#if NET471
            var thread = new System.Threading.Thread(() => RefreshCore(validateUrl, licenseKey, machineIdHash))
            {
                IsBackground = true,
                Name = "aidotnet-tensors-license-refresh"
            };
            thread.Start();
#else
            _ = System.Threading.Tasks.Task.Run(() => RefreshCore(validateUrl, licenseKey, machineIdHash));
#endif
        }
        catch (Exception ex)
        {
            // Even scheduling the work is best-effort — a thread-pool/thread failure must not surface.
            System.Diagnostics.Trace.TraceWarning(
                "TensorsOnlineLicenseServices: failed to schedule refresh: " + ex.GetType().Name + ": " + ex.Message);
        }
    }

    private static void RefreshCore(string validateUrl, string licenseKey, string machineIdHash)
    {
        // CRL first: cheapest, benefits every key type (revocation enforcement offline). Guarded by its OWN
        // freshness so a fresh offline token never suppresses a needed CRL refresh.
        try
        {
            if (ShouldRefreshCrl())
            {
                string? crlUrl = DeriveFunctionUrl(validateUrl, "get-revocations");
                if (crlUrl is not null)
                {
                    string? crl = HttpGet(crlUrl);
                    if (!string.IsNullOrWhiteSpace(crl) &&
                        Ed25519LicenseRevocation.TryInstallFetched(crl!, DateTimeOffset.UtcNow))
                    {
                        WriteAtomic(CrlPath, crl!);
                    }
                }
            }
        }
        catch (Exception ex)
        {
            System.Diagnostics.Trace.TraceWarning(
                "TensorsOnlineLicenseServices: CRL refresh failed: " + ex.GetType().Name + ": " + ex.Message);
        }

        // Offline token: only meaningful for a server AIDN-* key (aidn./aidn2. keys already verify offline).
        // Guarded by THIS key's own token freshness so a fresh CRL (or another key's token) never suppresses
        // minting a missing token for this key.
        try
        {
            if (ShouldRefreshToken(licenseKey))
            {
                string? issueUrl = DeriveFunctionUrl(validateUrl, "issue-license");
                if (issueUrl is not null)
                {
                    var payload = new Dictionary<string, string?>
                    {
                        ["license_key"] = licenseKey,
                        ["machine_id_hash"] = machineIdHash,
                        // A standalone Tensors app wants the tensors capability set on the minted token, so it
                        // tags the issue request with its own package name (the main SDK sends "AiDotNet").
                        ["package"] = "AiDotNet.Tensors",
                    };
                    string body = JsonSerializer.Serialize(payload);
                    string? resp = HttpPost(issueUrl, body);
                    string? token = ExtractOfflineToken(resp);
                    // Only cache a token that actually verifies against THIS build's embedded public key — a
                    // token we can't verify is useless offline, and caching it would just fail later.
                    if (token is not null &&
                        AsymmetricEntitlementVerifier.Verify(token, DateTimeOffset.UtcNow).IsValid)
                    {
                        WriteAtomic(TokenPath(licenseKey), token);
                    }
                }
            }
        }
        catch (Exception ex)
        {
            System.Diagnostics.Trace.TraceWarning(
                "TensorsOnlineLicenseServices: offline-token issue failed: " + ex.GetType().Name + ": " + ex.Message);
        }
    }

    // ───────────────────────── offline consumption ─────────────────────────

    /// <summary>
    /// Offline fallback used when the license server is unreachable: returns the cached offline <c>aidn2</c>
    /// token's verified <see cref="EntitlementResult"/> if one exists and verifies (IsValid) against the
    /// embedded public key on THIS machine, else null. Because it delegates to
    /// <see cref="AsymmetricEntitlementVerifier"/>, all v2 bindings (signature, exp, machine-lock, CRL
    /// revocation) are enforced — a leaked or expired cached token yields a non-valid result and is ignored.
    /// Reads the per-key file from the (test-overridable) cache directory.
    /// </summary>
    internal static EntitlementResult? TryValidateCachedOfflineToken(string licenseKey)
    {
        try
        {
            string path = TokenPath(licenseKey);
            string token;
            lock (FileLock)
            {
                if (!File.Exists(path))
                {
                    return null;
                }

                token = File.ReadAllText(path).Trim();
            }

            if (!AsymmetricEntitlementVerifier.IsAsymmetricKeyFormat(token))
            {
                return null;
            }

            var result = AsymmetricEntitlementVerifier.Verify(token, DateTimeOffset.UtcNow);
            return result.IsValid ? result : null;
        }
        catch (Exception ex)
        {
            System.Diagnostics.Trace.TraceWarning(
                "TensorsOnlineLicenseServices: cached offline token read failed: " + ex.GetType().Name + ": " + ex.Message);
            return null;
        }
    }

    /// <summary>Reads the last-fetched CRL from disk (or null). Used by <see cref="Ed25519LicenseRevocation"/>
    /// to enforce the most recent online revocation list even on a fully offline start.</summary>
    internal static string? ReadCachedCrl()
    {
        try
        {
            lock (FileLock)
            {
                return File.Exists(CrlPath) ? File.ReadAllText(CrlPath) : null;
            }
        }
        catch (Exception ex)
        {
            System.Diagnostics.Trace.TraceWarning(
                "TensorsOnlineLicenseServices: cached CRL read failed: " + ex.GetType().Name + ": " + ex.Message);
            return null;
        }
    }

    // ───────────────────────── helpers ─────────────────────────

    /// <summary>
    /// Derives a sibling edge-function URL (e.g. get-revocations, issue-license) from the validate-license URL
    /// by swapping the trailing function segment. Returns null when the URL isn't the expected
    /// <c>…/validate-license</c> shape (e.g. a custom test stub), so callers simply skip the feature.
    /// </summary>
    internal static string? DeriveFunctionUrl(string validateUrl, string functionName)
    {
        if (string.IsNullOrEmpty(validateUrl))
        {
            return null;
        }

        const string marker = "/validate-license";
        int i = validateUrl.LastIndexOf(marker, StringComparison.OrdinalIgnoreCase);
        if (i < 0)
        {
            return null;
        }

        // Preserve anything after the segment (e.g. a query string) though normally there is none.
        string tail = validateUrl.Substring(i + marker.Length);
        return validateUrl.Substring(0, i) + "/" + functionName + tail;
    }

    private static string? ExtractOfflineToken(string? responseJson)
    {
        if (string.IsNullOrWhiteSpace(responseJson))
        {
            return null;
        }

        try
        {
            using var doc = JsonDocument.Parse(responseJson!);
            var root = doc.RootElement;
            if (root.ValueKind != JsonValueKind.Object)
            {
                return null;
            }

            bool valid = root.TryGetProperty("valid", out var v)
                && (v.ValueKind == JsonValueKind.True
                    || (v.ValueKind == JsonValueKind.String && bool.TryParse(v.GetString(), out var b) && b));
            if (!valid)
            {
                return null;
            }

            if (root.TryGetProperty("offline_token", out var t) && t.ValueKind == JsonValueKind.String)
            {
                string? token = t.GetString();
                return string.IsNullOrWhiteSpace(token) ? null : token;
            }

            return null;
        }
        catch (JsonException)
        {
            return null;
        }
    }

    /// <summary>True when the CRL cache is missing or older than <see cref="RefreshInterval"/>.</summary>
    private static bool ShouldRefreshCrl() => IsArtifactStale(CrlPath);

    /// <summary>True when <paramref name="licenseKey"/> is a server <c>AIDN-*</c> key (the only kind for which
    /// an offline token is minted) AND that key's own cached token is missing or older than
    /// <see cref="RefreshInterval"/>. Only THIS key's token file is consulted, so another key's fresh token
    /// can never suppress this key's refresh.</summary>
    private static bool ShouldRefreshToken(string licenseKey) =>
        LicenseValidator.IsServerValidatedKeyFormat(licenseKey) && IsArtifactStale(TokenPath(licenseKey));

    /// <summary>True when a specific cache artifact is missing or older than <see cref="RefreshInterval"/>.</summary>
    private static bool IsArtifactStale(string path)
    {
        try
        {
            lock (FileLock)
            {
                if (!File.Exists(path))
                {
                    return true;
                }

                return DateTime.UtcNow - File.GetLastWriteTimeUtc(path) >= RefreshInterval;
            }
        }
        catch
        {
            // If we can't tell how fresh the artifact is, err toward refreshing (still throttled by the fact
            // that a refresh only runs after a successful online validation).
            return true;
        }
    }

    private static void WriteAtomic(string path, string content)
    {
        lock (FileLock)
        {
            string? dir = Path.GetDirectoryName(path);
            if (!string.IsNullOrEmpty(dir))
            {
                Directory.CreateDirectory(dir);
            }

            // Uniquely named temp file in the SAME directory (same volume, so the swap is a rename), then a
            // SINGLE atomic replace — never delete the destination first. This closes both the missing-file
            // window a concurrent reader could hit and the fixed-"*.tmp"-name race with the main SDK / another
            // process.
            string tmp = path + "." + Guid.NewGuid().ToString("N") + ".tmp";
            try
            {
                File.WriteAllText(tmp, content);
#if NET471
                // .NET Framework has no File.Move overwrite overload. File.Replace is an atomic swap but
                // requires the destination to already exist; fall back to a plain Move when it does not (a
                // best-effort equivalent, matching the modern overwrite Move).
                if (File.Exists(path))
                {
                    File.Replace(tmp, path, destinationBackupFileName: null);
                }
                else
                {
                    File.Move(tmp, path);
                }
#else
                File.Move(tmp, path, overwrite: true);
#endif
            }
            finally
            {
                // Clean up the temp file if the replace didn't consume it (e.g. an exception mid-write).
                try { if (File.Exists(tmp)) File.Delete(tmp); } catch { /* best effort */ }
            }
        }
    }

    private static string ShortHash(string value)
    {
        byte[] bytes = Encoding.UTF8.GetBytes("license-cache:" + value);
#if NET471
        using var sha = System.Security.Cryptography.SHA256.Create();
        byte[] hash = sha.ComputeHash(bytes);
#else
        byte[] hash = System.Security.Cryptography.SHA256.HashData(bytes);
#endif
        var sb = new StringBuilder(16);
        for (int i = 0; i < 8; i++)
        {
            sb.Append(hash[i].ToString("x2"));
        }

        return sb.ToString();
    }

    private static string? HttpGet(string url)
    {
#if NET471
        // WebClient has NO default timeout, so a hung endpoint could pin a background thread far beyond the
        // documented refresh budget. Use HttpWebRequest with an explicit 10s timeout (matching the modern
        // HttpClient path).
        var req = (System.Net.HttpWebRequest)System.Net.WebRequest.Create(url);
        req.Method = "GET";
        req.Timeout = Net471TimeoutMs;
        req.ReadWriteTimeout = Net471TimeoutMs;
        using var resp = (System.Net.HttpWebResponse)req.GetResponse();
        using var reader = new System.IO.StreamReader(resp.GetResponseStream()!);
        return reader.ReadToEnd();
#else
        using var resp = Http.GetAsync(url).ConfigureAwait(false).GetAwaiter().GetResult();
        return resp.Content.ReadAsStringAsync().ConfigureAwait(false).GetAwaiter().GetResult();
#endif
    }

    private static string? HttpPost(string url, string json)
    {
#if NET471
        // Same reasoning as HttpGet: enforce an explicit 10s timeout that WebClient.UploadString lacks.
        var req = (System.Net.HttpWebRequest)System.Net.WebRequest.Create(url);
        req.Method = "POST";
        req.ContentType = "application/json";
        req.Timeout = Net471TimeoutMs;
        req.ReadWriteTimeout = Net471TimeoutMs;

        byte[] payload = Encoding.UTF8.GetBytes(json);
        req.ContentLength = payload.Length;
        using (var reqStream = req.GetRequestStream())
        {
            reqStream.Write(payload, 0, payload.Length);
        }

        try
        {
            using var resp = (System.Net.HttpWebResponse)req.GetResponse();
            using var reader = new System.IO.StreamReader(resp.GetResponseStream()!);
            return reader.ReadToEnd();
        }
        catch (System.Net.WebException ex) when (ex.Response is System.Net.HttpWebResponse http)
        {
            // A 403 (e.g. invalid_key) still carries a JSON body we want to inspect/ignore cleanly.
            using var reader = new System.IO.StreamReader(http.GetResponseStream()!);
            return reader.ReadToEnd();
        }
#else
        using var content = new StringContent(json, Encoding.UTF8, "application/json");
        using var resp = Http.PostAsync(url, content).ConfigureAwait(false).GetAwaiter().GetResult();
        return resp.Content.ReadAsStringAsync().ConfigureAwait(false).GetAwaiter().GetResult();
#endif
    }

    private sealed class CacheDirScope : IDisposable
    {
        private readonly string? _previous;
        public CacheDirScope(string? previous) => _previous = previous;
        public void Dispose() => _cacheDirOverrideForTests = _previous;
    }
}
