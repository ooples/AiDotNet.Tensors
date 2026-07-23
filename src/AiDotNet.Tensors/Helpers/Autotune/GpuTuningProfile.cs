using System;
using System.Collections.Generic;

namespace AiDotNet.Tensors.Helpers.Autotune;

/// <summary>
/// A shareable, opt-in community tuning record (Phase 2). It is the wire/row
/// shape for the Supabase <c>gpu_tuning_profiles</c> table, and it is keyed on
/// the per-<b>model</b> identity (<see cref="GpuDeviceFingerprint.ModelKey"/>),
/// NOT the per-card local key — a winner measured on one RTX 3080 is meant to
/// reach every RTX 3080 on the same driver + architecture.
///
/// <para><b>Trust:</b> <see cref="MeasuredGflops"/> is the reporter's number and
/// is treated as advisory only. A downloaded profile is never launched on faith;
/// it is turned back into an <see cref="AutotuneCandidate"/> via
/// <see cref="ToCandidate"/> and must win the local on-device sweep (see
/// <see cref="CommunityAutotune"/>), so a poisoned or hardware-mismatched row
/// simply loses instead of running.</para>
/// </summary>
public sealed class GpuTuningProfile
{
    /// <summary>Per-model hardware key (<see cref="GpuDeviceFingerprint.ModelKey"/>).</summary>
    public string ModelKey { get; set; } = "";
    public string Vendor { get; set; } = "";
    public string Model { get; set; } = "";
    public string Architecture { get; set; } = "";
    public int DriverVersion { get; set; }

    /// <summary>Kernel category, e.g. <c>conv2d</c> / <c>sdpa</c> / <c>gemm</c>.</summary>
    public string Category { get; set; } = "";

    /// <summary>
    /// Shareable kernel-family name WITHOUT the per-card fingerprint suffix (e.g.
    /// <c>tiled-gemm-1x1-nchw-fp32</c>), so it is portable across cards of a model.
    /// </summary>
    public string KernelName { get; set; } = "";

    /// <summary>The exact-contract shape key (<see cref="ShapeProfile.ToFileStem"/>).</summary>
    public string ShapeKey { get; set; } = "";

    /// <summary>The winning variant id (e.g. <c>tile-16</c>).</summary>
    public string Variant { get; set; } = "";

    /// <summary>Structured launch parameters that reconstruct the config.</summary>
    public Dictionary<string, string> Parameters { get; set; } = new(StringComparer.Ordinal);

    /// <summary>Reporter's measured throughput (GFLOPS) — advisory; re-verified locally before use.</summary>
    public double MeasuredGflops { get; set; }

    /// <summary>Pseudonymous reporter id (no PII); optional.</summary>
    public string? ClientHash { get; set; }

    /// <summary>Reporting library version; optional.</summary>
    public string? AiDotNetVersion { get; set; }

    /// <summary>Turns a downloaded profile back into a sweep candidate to be re-verified on-device.</summary>
    public AutotuneCandidate ToCandidate()
    {
        var copy = new Dictionary<string, string>(
            Parameters?.Count ?? 0, StringComparer.Ordinal);
        if (Parameters is not null)
            foreach (KeyValuePair<string, string> kv in Parameters)
                copy[kv.Key] = kv.Value ?? string.Empty;
        return new AutotuneCandidate(Variant, copy);
    }

    /// <summary>
    /// Builds a shareable profile from a locally-measured winner. The winner's
    /// kernel identity is the LOCAL, fingerprint-suffixed name; the shareable
    /// <see cref="KernelName"/> is <paramref name="shareableKernelName"/> (the
    /// family name without the suffix), and the hardware class comes from the
    /// per-model fields of <paramref name="fingerprint"/>.
    /// </summary>
    public static GpuTuningProfile FromWinner(
        GpuDeviceFingerprint fingerprint,
        string category,
        string shareableKernelName,
        ShapeProfile shape,
        AutotuneResolution winner,
        string? clientHash = null,
        string? aidotnetVersion = null)
    {
        if (string.IsNullOrWhiteSpace(category))
            throw new ArgumentException("A kernel category is required.", nameof(category));
        if (string.IsNullOrWhiteSpace(shareableKernelName))
            throw new ArgumentException("A shareable kernel name is required.", nameof(shareableKernelName));

        var parameters = new Dictionary<string, string>(StringComparer.Ordinal);
        if (winner.Parameters is not null)
            foreach (KeyValuePair<string, string> kv in winner.Parameters)
                parameters[kv.Key] = kv.Value ?? string.Empty;

        return new GpuTuningProfile
        {
            ModelKey = fingerprint.ModelKey,
            Vendor = fingerprint.Vendor,
            Model = fingerprint.Model,
            Architecture = fingerprint.Architecture,
            DriverVersion = fingerprint.DriverVersion,
            Category = category,
            KernelName = shareableKernelName,
            ShapeKey = shape.ToFileStem(),
            Variant = winner.Variant,
            Parameters = parameters,
            MeasuredGflops = winner.MeasuredGflops,
            ClientHash = clientHash,
            AiDotNetVersion = aidotnetVersion
        };
    }
}
