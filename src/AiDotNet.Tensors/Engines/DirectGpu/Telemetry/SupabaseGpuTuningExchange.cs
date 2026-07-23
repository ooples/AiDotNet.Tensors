using System;
using System.Collections.Generic;
using System.Globalization;
using System.Net.Http;
using System.Reflection;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using AiDotNet.Tensors.Helpers.Autotune;

namespace AiDotNet.Tensors.Engines.DirectGpu.Telemetry;

/// <summary>
/// Supabase-backed <see cref="IGpuTuningExchange"/> (Phase 2). Publishes locally
/// measured winners and fetches community winners for a hardware model + kernel
/// + shape from the <c>gpu_tuning_profiles</c> PostgREST table.
///
/// <para><b>Opt-in and off by default.</b> Even with embedded credentials, the
/// exchange is inert unless the user sets <c>AIDOTNET_TELEMETRY=true</c>. URL and
/// key resolve from (constructor arg) → (env <c>AIDOTNET_TELEMETRY_URL</c> /
/// <c>AIDOTNET_TELEMETRY_KEY</c>) → (assembly metadata <c>TelemetryUrl</c> /
/// <c>TelemetryKey</c>), so an env var always overrides the embedded default.</para>
///
/// <para><b>Never throws into dispatch.</b> Every network call is wrapped; on any
/// error (offline, timeout, malformed response) Fetch returns empty and Publish
/// no-ops. The publishable key is public-by-design; RLS on the table restricts
/// the anon role to insert/select, and the client-side re-verification in
/// <see cref="CommunityAutotune"/> means a hostile row can never harm a peer.</para>
/// </summary>
public sealed class SupabaseGpuTuningExchange : IGpuTuningExchange, IDisposable
{
    public const string UrlEnvVar = "AIDOTNET_TELEMETRY_URL";
    public const string KeyEnvVar = "AIDOTNET_TELEMETRY_KEY";
    public const string OptInEnvVar = "AIDOTNET_TELEMETRY";
    private const string Table = "gpu_tuning_profiles";

    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
        PropertyNameCaseInsensitive = true
    };

    private readonly HttpClient _http;
    private readonly string _url;
    private readonly string _key;
    private readonly string? _clientHash;
    private readonly string? _version;
    private bool _disposed;

    public bool IsEnabled { get; }

    public SupabaseGpuTuningExchange(
        bool enabled = false,
        string? url = null,
        string? key = null,
        string? clientHash = null,
        HttpClient? httpClient = null)
    {
        _url = TrimTrailingSlash(Resolve(url, UrlEnvVar, "TelemetryUrl"));
        _key = Resolve(key, KeyEnvVar, "TelemetryKey");

        bool optedIn = string.Equals(
            Environment.GetEnvironmentVariable(OptInEnvVar), "true", StringComparison.OrdinalIgnoreCase);

        IsEnabled = enabled
            && optedIn
            && !string.IsNullOrWhiteSpace(_url)
            && !string.IsNullOrWhiteSpace(_key)
            && Uri.TryCreate(_url, UriKind.Absolute, out Uri? uri)
            && uri.Scheme == Uri.UriSchemeHttps;

        _clientHash = clientHash;
        _version = typeof(SupabaseGpuTuningExchange).Assembly.GetName().Version?.ToString();
        _http = httpClient ?? new HttpClient { Timeout = TimeSpan.FromSeconds(8) };
        if (IsEnabled)
        {
            _http.DefaultRequestHeaders.TryAddWithoutValidation("apikey", _key);
            _http.DefaultRequestHeaders.TryAddWithoutValidation("Authorization", "Bearer " + _key);
        }
    }

    public IReadOnlyList<GpuTuningProfile> Fetch(
        string modelKey, string category, string kernelName, string shapeKey)
    {
        if (!IsEnabled) return Array.Empty<GpuTuningProfile>();
        try
        {
            string requestUri = BuildFetchUri(modelKey, category, kernelName, shapeKey, limit: 5);
            using HttpResponseMessage response = _http.GetAsync(requestUri).GetAwaiter().GetResult();
            if (!response.IsSuccessStatusCode) return Array.Empty<GpuTuningProfile>();
            string body = response.Content.ReadAsStringAsync().GetAwaiter().GetResult();
            Row[]? rows = JsonSerializer.Deserialize<Row[]>(body, JsonOptions);
            if (rows is null || rows.Length == 0) return Array.Empty<GpuTuningProfile>();

            var result = new List<GpuTuningProfile>(rows.Length);
            foreach (Row row in rows)
                if (row is not null) result.Add(row.ToProfile());
            return result;
        }
        catch
        {
            return Array.Empty<GpuTuningProfile>();
        }
    }

    public void Publish(GpuTuningProfile profile)
    {
        if (!IsEnabled || profile is null) return;
        try
        {
            Row row = Row.FromProfile(profile, _clientHash, _version);
            string json = JsonSerializer.Serialize(row, JsonOptions);
            using var content = new StringContent(json, Encoding.UTF8, "application/json");
            using var request = new HttpRequestMessage(
                HttpMethod.Post, _url + "/rest/v1/" + Table) { Content = content };
            request.Headers.TryAddWithoutValidation("Prefer", "return=minimal");
            using HttpResponseMessage _ = _http.SendAsync(request).GetAwaiter().GetResult();
        }
        catch
        {
            // Publishing is advisory — never surface a telemetry error to dispatch.
        }
    }

    internal string BuildFetchUri(
        string modelKey, string category, string kernelName, string shapeKey, int limit)
    {
        var sb = new StringBuilder(_url.Length + 256);
        sb.Append(_url).Append("/rest/v1/").Append(Table).Append('?');
        sb.Append("model_key=eq.").Append(Uri.EscapeDataString(modelKey ?? string.Empty));
        sb.Append("&category=eq.").Append(Uri.EscapeDataString(category ?? string.Empty));
        sb.Append("&kernel_name=eq.").Append(Uri.EscapeDataString(kernelName ?? string.Empty));
        sb.Append("&shape_key=eq.").Append(Uri.EscapeDataString(shapeKey ?? string.Empty));
        sb.Append("&order=measured_gflops.desc");
        sb.Append("&limit=").Append(limit.ToString(CultureInfo.InvariantCulture));
        return sb.ToString();
    }

    private static string Resolve(string? explicitValue, string envVar, string metadataKey)
    {
        if (!string.IsNullOrWhiteSpace(explicitValue)) return explicitValue!;
        string? env = Environment.GetEnvironmentVariable(envVar);
        if (!string.IsNullOrWhiteSpace(env)) return env!;
        foreach (AssemblyMetadataAttribute attr in typeof(SupabaseGpuTuningExchange)
                     .Assembly.GetCustomAttributes<AssemblyMetadataAttribute>())
            if (string.Equals(attr.Key, metadataKey, StringComparison.Ordinal))
                return attr.Value ?? string.Empty;
        return string.Empty;
    }

    private static string TrimTrailingSlash(string url) =>
        string.IsNullOrEmpty(url) ? url : url.TrimEnd('/');

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _http.Dispose();
    }

    /// <summary>PostgREST row shape for <c>gpu_tuning_profiles</c> (snake_case columns).</summary>
    internal sealed class Row
    {
        [JsonPropertyName("id")] public long? Id { get; set; }
        [JsonPropertyName("model_key")] public string ModelKey { get; set; } = "";
        [JsonPropertyName("vendor")] public string Vendor { get; set; } = "";
        [JsonPropertyName("model")] public string Model { get; set; } = "";
        [JsonPropertyName("architecture")] public string Architecture { get; set; } = "";
        [JsonPropertyName("driver_version")] public int DriverVersion { get; set; }
        [JsonPropertyName("category")] public string Category { get; set; } = "";
        [JsonPropertyName("kernel_name")] public string KernelName { get; set; } = "";
        [JsonPropertyName("shape_key")] public string ShapeKey { get; set; } = "";
        [JsonPropertyName("variant")] public string Variant { get; set; } = "";
        [JsonPropertyName("parameters")] public Dictionary<string, string> Parameters { get; set; } = new();
        [JsonPropertyName("measured_gflops")] public double MeasuredGflops { get; set; }
        [JsonPropertyName("client_hash")] public string? ClientHash { get; set; }
        [JsonPropertyName("aidotnet_version")] public string? AiDotNetVersion { get; set; }
        [JsonPropertyName("recorded_at")] public string? RecordedAt { get; set; }

        internal GpuTuningProfile ToProfile() => new()
        {
            ModelKey = ModelKey,
            Vendor = Vendor,
            Model = Model,
            Architecture = Architecture,
            DriverVersion = DriverVersion,
            Category = Category,
            KernelName = KernelName,
            ShapeKey = ShapeKey,
            Variant = Variant,
            Parameters = Parameters ?? new Dictionary<string, string>(StringComparer.Ordinal),
            MeasuredGflops = MeasuredGflops,
            ClientHash = ClientHash,
            AiDotNetVersion = AiDotNetVersion
        };

        internal static Row FromProfile(GpuTuningProfile p, string? clientHash, string? version) => new()
        {
            // Id and RecordedAt are server-assigned; left null so they are omitted on insert.
            ModelKey = p.ModelKey,
            Vendor = p.Vendor,
            Model = p.Model,
            Architecture = p.Architecture,
            DriverVersion = p.DriverVersion,
            Category = p.Category,
            KernelName = p.KernelName,
            ShapeKey = p.ShapeKey,
            Variant = p.Variant,
            Parameters = p.Parameters ?? new Dictionary<string, string>(StringComparer.Ordinal),
            MeasuredGflops = p.MeasuredGflops,
            ClientHash = clientHash ?? p.ClientHash,
            AiDotNetVersion = version ?? p.AiDotNetVersion
        };
    }
}
