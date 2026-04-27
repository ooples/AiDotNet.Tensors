// Copyright (c) AiDotNet. All rights reserved.

using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Net;
using System.Text;
using System.Text.Json;
#if !NET471
using System.Net.Http;
#endif

namespace AiDotNet.Tensors.Licensing;

/// <summary>
/// Tensor-layer license validator. Hits the same Supabase Edge Function
/// the upstream <c>AiDotNet</c> validator uses, sends a
/// <c>package=AiDotNet.Tensors</c> hint so the server attaches the
/// right per-tier capability set, and caches the result with an
/// offline grace window.
/// </summary>
/// <remarks>
/// <para><b>Why a validator in the tensor layer:</b></para>
/// <para>
/// The user explicitly asked for the tensor layer to participate in
/// license enforcement so reading a <c>.gguf</c> / <c>.safetensors</c>
/// file via the public API requires a valid license. The same key
/// works in upstream <c>AiDotNet</c> — both layers hit the same server
/// — but the tensor layer can also stand alone for users who only
/// pull the <c>AiDotNet.Tensors</c> NuGet.
/// </para>
/// <para><b>What this validator does NOT do (delegated to upstream):</b></para>
/// <list type="bullet">
///   <item><description>Offline HMAC verification against an embedded
///   build key. The tensor layer doesn't ship a build secret —
///   offline-only validation always returns
///   <see cref="LicenseKeyStatus.ValidationPending"/> until the server
///   is reachable, then caches per <c>OfflineGracePeriod</c>.</description></item>
///   <item><description>Trial-counter management. That lives in
///   <see cref="TrialState"/> and is consulted by
///   <see cref="PersistenceGuard"/> when no key is configured.</description></item>
/// </list>
/// </remarks>
public sealed class LicenseValidator
{
    /// <summary>
    /// Default endpoint — same Supabase Edge Function the upstream
    /// validator uses so a single key works in both packages.
    /// </summary>
    public const string DefaultServerUrl =
        "https://yfkqwpgjahoamlgckjib.supabase.co/functions/v1/validate-license";

    private readonly AiDotNetTensorsLicenseKey _licenseKey;
    private LicenseValidationResult? _cached;
    private readonly object _cacheLock = new();

#if !NET471
    private static readonly HttpClient SharedHttpClient = new()
    {
        Timeout = TimeSpan.FromSeconds(15)
    };
#endif

    /// <summary>The most recent cached validation result, or null.</summary>
    public LicenseValidationResult? CachedResult
    {
        get { lock (_cacheLock) return _cached; }
    }

    /// <summary>Creates a validator for the given key.</summary>
    public LicenseValidator(AiDotNetTensorsLicenseKey licenseKey)
    {
        _licenseKey = licenseKey ?? throw new ArgumentNullException(nameof(licenseKey));
    }

    /// <summary>
    /// Validates the configured license key. Uses the shared default
    /// server unless <see cref="AiDotNetTensorsLicenseKey.ServerUrl"/>
    /// is set. Caches the result for the configured offline grace
    /// window so subsequent calls don't re-hit the network.
    /// </summary>
    public LicenseValidationResult Validate()
    {
        if (string.IsNullOrWhiteSpace(_licenseKey.Key))
        {
            return Cache(new LicenseValidationResult(
                LicenseKeyStatus.Invalid,
                tier: null, expiresAt: null, validatedAt: DateTimeOffset.UtcNow,
                message: "License key is empty.",
                capabilities: null));
        }

        if (!ValidateKeyFormat(_licenseKey.Key))
        {
            return Cache(new LicenseValidationResult(
                LicenseKeyStatus.Invalid,
                tier: null, expiresAt: null, validatedAt: DateTimeOffset.UtcNow,
                message: "License key format is invalid. Expected aidn.{id}.{signature} or AIDN-*-{hex}.",
                capabilities: null));
        }

        // Explicit offline-only mode (ServerUrl == "")
        bool explicitOfflineOnly =
            _licenseKey.ServerUrl is not null && _licenseKey.ServerUrl.Trim().Length == 0;

        if (explicitOfflineOnly)
        {
            // No build key in the tensor layer → can't HMAC-verify
            // signed keys offline. Surface as ValidationPending so the
            // guard can decide based on its own policy (today: deny).
            return Cache(new LicenseValidationResult(
                LicenseKeyStatus.ValidationPending,
                tier: null, expiresAt: null, validatedAt: DateTimeOffset.UtcNow,
                message: "Tensor-layer offline-only mode is not supported (no embedded build key). " +
                         "Allow online validation or use upstream AiDotNet for offline verification.",
                capabilities: null));
        }

        // Cache hit within grace window — return without hitting the network.
        lock (_cacheLock)
        {
            if (_cached is not null
                && _cached.Status == LicenseKeyStatus.Active
                && _cached.ValidatedAt + _licenseKey.OfflineGracePeriod > DateTimeOffset.UtcNow)
            {
                return _cached;
            }
        }

        try
        {
            var result = ValidateOnline();
            return Cache(result);
        }
        catch (Exception ex)
        {
            Trace.TraceWarning(
                "AiDotNet.Tensors LicenseValidator: online validation failed: "
                + ex.GetType().Name + ": " + ex.Message);

            // Fall back to cached result if still in grace.
            lock (_cacheLock)
            {
                if (_cached is not null
                    && _cached.ValidatedAt + _licenseKey.OfflineGracePeriod > DateTimeOffset.UtcNow)
                {
                    return _cached;
                }
            }

            // Stale cache → expired. No cache → pending.
            lock (_cacheLock)
            {
                if (_cached is not null)
                {
                    var expired = new LicenseValidationResult(
                        LicenseKeyStatus.Expired,
                        tier: _cached.Tier,
                        expiresAt: _cached.ExpiresAt,
                        validatedAt: DateTimeOffset.UtcNow,
                        message: "License server unreachable and grace period exceeded.",
                        capabilities: null);
                    _cached = expired;
                    return expired;
                }
            }

            return Cache(new LicenseValidationResult(
                LicenseKeyStatus.ValidationPending,
                tier: null, expiresAt: null, validatedAt: DateTimeOffset.UtcNow,
                message: "License server unreachable. Initial validation pending.",
                capabilities: null));
        }
    }

    private LicenseValidationResult Cache(LicenseValidationResult r)
    {
        lock (_cacheLock) _cached = r;
        return r;
    }

    /// <summary>
    /// True iff <paramref name="key"/> is in <c>aidn.{id}.{sig}</c>
    /// format (signed offline-eligible) or <c>AIDN-*-{hex}</c> format
    /// (server-validated). Mirrors the upstream check so the same key
    /// strings work in both packages.
    /// </summary>
    public static bool ValidateKeyFormat(string key)
        => IsSignedKeyFormat(key) || IsServerValidatedKeyFormat(key);

    internal static bool IsSignedKeyFormat(string key)
    {
        var parts = key.Split('.');
        if (parts.Length != 3 || parts[0] != "aidn"
            || parts[1].Length == 0 || parts[2].Length == 0)
            return false;
        for (int i = 0; i < parts[1].Length; i++)
            if (!char.IsLetterOrDigit(parts[1][i])) return false;
        // Signature is base64url — letters, digits, plus the URL-safe
        // sigil set ('-', '_') and the legacy non-URL-safe ('+', '/')
        // pair, plus any '=' padding chars at the end. Reject anything
        // outside that set so a malformed key fails offline rather than
        // round-tripping to the server.
        for (int i = 0; i < parts[2].Length; i++)
        {
            char c = parts[2][i];
            bool ok = char.IsLetterOrDigit(c)
                   || c == '-' || c == '_'
                   || c == '+' || c == '/'
                   || c == '=';
            if (!ok) return false;
        }
        return true;
    }

    internal static bool IsServerValidatedKeyFormat(string key)
    {
        var parts = key.Split('-');
        if (parts.Length < 4
            || !parts[0].Equals("AIDN", StringComparison.OrdinalIgnoreCase))
            return false;
        string last = parts[parts.Length - 1];
        if (last.Length < 8) return false;
        for (int i = 0; i < last.Length; i++)
        {
            char c = last[i];
            bool isHex = (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
            if (!isHex) return false;
        }
        return true;
    }

    private LicenseValidationResult ValidateOnline()
    {
        string url = _licenseKey.ServerUrl ?? DefaultServerUrl;

        // Build minimal request body. Mirrors upstream's body keys plus
        // a `package` hint so the server can attach the right capability
        // set for the calling library — `AiDotNet.Tensors` gets the
        // tensors:* capabilities, `AiDotNet` gets the model:* set, and
        // higher tiers get both.
        var body = new Dictionary<string, string?>
        {
            ["license_key"] = _licenseKey.Key,
            ["package"] = "AiDotNet.Tensors",
        };
        if (_licenseKey.Environment is not null)
            body["environment"] = _licenseKey.Environment;
        if (_licenseKey.EnableTelemetry)
        {
            try { body["hostname"] = Environment.MachineName; } catch { }
            try { body["os_description"] = System.Runtime.InteropServices.RuntimeInformation.OSDescription; } catch { }
        }

        string json = JsonSerializer.Serialize(body);

#if NET471
        return PostNet471(url, json);
#else
        return PostModern(url, json);
#endif
    }

#if NET471
    private LicenseValidationResult PostNet471(string url, string json)
    {
        var req = (HttpWebRequest)WebRequest.Create(url);
        req.Method = "POST";
        req.ContentType = "application/json";
        req.Timeout = 15_000;
        var bodyBytes = Encoding.UTF8.GetBytes(json);
        req.ContentLength = bodyBytes.Length;
        using (var rs = req.GetRequestStream()) rs.Write(bodyBytes, 0, bodyBytes.Length);

        using var resp = (HttpWebResponse)req.GetResponse();
        using var stream = resp.GetResponseStream() ?? Stream.Null;
        using var reader = new StreamReader(stream, Encoding.UTF8);
        string responseJson = reader.ReadToEnd();
        return ParseResponse(responseJson, (int)resp.StatusCode);
    }
#else
    private LicenseValidationResult PostModern(string url, string json)
    {
        var content = new StringContent(json, Encoding.UTF8, "application/json");
        var response = SharedHttpClient.PostAsync(url, content).GetAwaiter().GetResult();
        string responseJson = response.Content.ReadAsStringAsync().GetAwaiter().GetResult();
        return ParseResponse(responseJson, (int)response.StatusCode);
    }
#endif

    /// <summary>
    /// Parses the server response into a <see cref="LicenseValidationResult"/>.
    /// Visible to tests so the parser can be exercised without a live
    /// server.
    /// </summary>
    internal static LicenseValidationResult ParseResponse(string responseJson, int statusCode)
    {
        // Network non-2xx → invalid. The Edge Function returns 200
        // for both valid and invalid keys with status in the body, so
        // a non-2xx is a transport failure rather than a key rejection.
        if (statusCode < 200 || statusCode >= 300)
        {
            return new LicenseValidationResult(
                LicenseKeyStatus.Invalid,
                tier: null, expiresAt: null, validatedAt: DateTimeOffset.UtcNow,
                message: $"License server returned HTTP {statusCode}.",
                capabilities: null);
        }

        try
        {
            using var doc = JsonDocument.Parse(responseJson);
            var root = doc.RootElement;

            string? statusStr = root.TryGetProperty("status", out var s) ? s.GetString() : null;
            var status = statusStr switch
            {
                "active" => LicenseKeyStatus.Active,
                "expired" => LicenseKeyStatus.Expired,
                "revoked" => LicenseKeyStatus.Revoked,
                "seat_limit_reached" => LicenseKeyStatus.SeatLimitReached,
                "invalid" => LicenseKeyStatus.Invalid,
                _ => LicenseKeyStatus.Invalid,
            };

            string? tier = root.TryGetProperty("tier", out var t) ? t.GetString() : null;
            DateTimeOffset? expiresAt = null;
            if (root.TryGetProperty("expires_at", out var exp) && exp.ValueKind == JsonValueKind.String)
            {
                if (DateTimeOffset.TryParse(exp.GetString(), out var parsed)) expiresAt = parsed;
            }
            string? message = root.TryGetProperty("message", out var m) ? m.GetString() : null;

            var caps = new List<string>();
            if (root.TryGetProperty("capabilities", out var capsEl)
                && capsEl.ValueKind == JsonValueKind.Array)
            {
                foreach (var cap in capsEl.EnumerateArray())
                {
                    if (cap.ValueKind == JsonValueKind.String)
                    {
                        var v = cap.GetString();
                        if (!string.IsNullOrEmpty(v)) caps.Add(v!);
                    }
                }
            }

            return new LicenseValidationResult(
                status, tier, expiresAt,
                validatedAt: DateTimeOffset.UtcNow,
                message: message,
                capabilities: caps);
        }
        catch (Exception ex)
        {
            return new LicenseValidationResult(
                LicenseKeyStatus.Invalid,
                tier: null, expiresAt: null, validatedAt: DateTimeOffset.UtcNow,
                message: "Failed to parse license server response: " + ex.Message,
                capabilities: null);
        }
    }
}
