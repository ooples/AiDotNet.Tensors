#if NET5_0_OR_GREATER
using System.Collections.Generic;
using System.Text.Json;
using System.Text.Json.Serialization;
using AiDotNet.Tensors.Engines.DirectGpu.Telemetry;
using AiDotNet.Tensors.Helpers.Autotune;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu.Telemetry;

/// <summary>
/// Network-free tests for <see cref="SupabaseGpuTuningExchange"/>. The parse test
/// uses the EXACT JSON captured from a live PostgREST round-trip against the
/// provisioned gpu_tuning_profiles table, so the deserialization contract is
/// pinned to the real backend response.
/// </summary>
public sealed class SupabaseGpuTuningExchangeTests
{
    // Verbatim body returned by GET /rest/v1/gpu_tuning_profiles?... during
    // Phase-2 provisioning of project yfkqwpgjahoamlgckjib.
    private const string LiveRestJson =
        "[{\"id\":1,\"model_key\":\"nvidia|SMOKE TEST CARD|sm86|drv12030\",\"vendor\":\"nvidia\"," +
        "\"model\":\"SMOKE TEST CARD\",\"architecture\":\"sm86\",\"driver_version\":12030," +
        "\"category\":\"conv2d\",\"kernel_name\":\"tiled-gemm-1x1-nchw-fp32\",\"shape_key\":\"smoke-shape\"," +
        "\"variant\":\"tile-16\",\"parameters\":{\"Tile\": \"16\"},\"measured_gflops\":123.4," +
        "\"client_hash\":\"smoketest-cf\",\"aidotnet_version\":\"phase2-smoke\"," +
        "\"recorded_at\":\"2026-07-23T12:20:30.251286+00:00\"}]";

    [Fact]
    public void Row_Deserializes_LiveRestResponse()
    {
        SupabaseGpuTuningExchange.Row[]? rows =
            JsonSerializer.Deserialize<SupabaseGpuTuningExchange.Row[]>(LiveRestJson);
        Assert.NotNull(rows);
        Assert.Single(rows!);

        GpuTuningProfile p = rows![0].ToProfile();
        Assert.Equal("nvidia|SMOKE TEST CARD|sm86|drv12030", p.ModelKey);
        Assert.Equal("conv2d", p.Category);
        Assert.Equal("tiled-gemm-1x1-nchw-fp32", p.KernelName);
        Assert.Equal("smoke-shape", p.ShapeKey);
        Assert.Equal("tile-16", p.Variant);
        Assert.Equal("16", p.Parameters["Tile"]);
        Assert.Equal(123.4, p.MeasuredGflops, 3);
        Assert.Equal(12030, p.DriverVersion);

        // And it round-trips into a re-verifiable sweep candidate.
        AutotuneCandidate c = p.ToCandidate();
        Assert.Equal("tile-16", c.Variant);
        Assert.Equal("16", c.Parameters["Tile"]);
    }

    [Fact]
    public void Row_FromProfile_OmitsServerAssignedFields_AndUsesSnakeCase()
    {
        var profile = new GpuTuningProfile
        {
            ModelKey = "nvidia|RTX 3080|sm86|drv12030",
            Vendor = "nvidia",
            Model = "RTX 3080",
            Architecture = "sm86",
            DriverVersion = 12030,
            Category = "conv2d",
            KernelName = "tiled-gemm-1x1-nchw-fp32",
            ShapeKey = "n32-k64",
            Variant = "tile-16",
            Parameters = new Dictionary<string, string> { ["Tile"] = "16" },
            MeasuredGflops = 900.0
        };

        var options = new JsonSerializerOptions { DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull };
        string json = JsonSerializer.Serialize(
            SupabaseGpuTuningExchange.Row.FromProfile(profile, "hash1", "1.2.3"), options);

        Assert.Contains("\"model_key\":\"nvidia|RTX 3080|sm86|drv12030\"", json);
        Assert.Contains("\"measured_gflops\":900", json);
        Assert.Contains("\"parameters\":{\"Tile\":\"16\"}", json);
        Assert.Contains("\"client_hash\":\"hash1\"", json);
        // Server-assigned columns must not be sent on insert.
        Assert.DoesNotContain("\"id\":", json);
        Assert.DoesNotContain("\"recorded_at\":", json);
    }

    [Fact]
    public void BuildFetchUri_EncodesFiltersAndOrdersByGflops()
    {
        using var exchange = new SupabaseGpuTuningExchange(enabled: false, url: "https://example.supabase.co");
        string uri = exchange.BuildFetchUri(
            "nvidia|RTX 3080|sm86|drv12030", "conv2d", "tiled-gemm-1x1-nchw-fp32", "n32-k64", limit: 5);

        Assert.Contains("/rest/v1/gpu_tuning_profiles?", uri);
        Assert.Contains("model_key=eq.nvidia%7CRTX%203080%7Csm86%7Cdrv12030", uri); // '|'->%7C, ' '->%20
        Assert.Contains("category=eq.conv2d", uri);
        Assert.Contains("order=measured_gflops.desc", uri);
        Assert.Contains("limit=5", uri);
    }

    [Fact]
    public void Disabled_Fetch_ReturnsEmpty_AndPublish_DoesNotThrow()
    {
        // No AIDOTNET_TELEMETRY opt-in in the test env -> IsEnabled must be false.
        using var exchange = new SupabaseGpuTuningExchange(
            enabled: true, url: "https://example.supabase.co", key: "sb_publishable_x");
        Assert.False(exchange.IsEnabled);
        Assert.Empty(exchange.Fetch("m", "c", "k", "s"));
        exchange.Publish(new GpuTuningProfile { Variant = "tile-16" }); // must not throw / must not hit network
    }
}
#endif
