// Copyright (c) AiDotNet. All rights reserved.

namespace AiDotNet.Tensors.Licensing;

/// <summary>
/// Represents a license key for AiDotNet.Tensors persistence operations
/// (safetensors / GGUF / pickle / sharded checkpoint readers and writers).
/// The same key string used by the upstream <c>AiDotNet</c> package works
/// here — the licensing server returns a per-key set of capabilities and
/// the tensor layer accepts the key when those capabilities include
/// <c>tensors:save</c> or <c>tensors:load</c>.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> When you purchase a license for AiDotNet you
/// receive one key string. That same string activates persistence in
/// every package that reads/writes model data — the tensor library
/// (this package) and the upstream model library both consult the same
/// licensing server. Buying the "Tensors-only" tier still produces a
/// key that works here; buying "Pro" or "Enterprise" produces a key
/// that works here AND in upstream <c>AiDotNet</c>.</para>
///
/// <para><b>Default server:</b> the validation endpoint defaults to the
/// shared AiDotNet license server. Override
/// <see cref="ServerUrl"/> only when running an offline-only validation
/// against a self-signed key.</para>
///
/// <para>This type intentionally has the same shape as the upstream
/// <c>AiDotNet.Models.AiDotNetLicenseKey</c> so the upstream
/// <c>AiModelBuilder</c> can copy fields field-by-field and flow a
/// single user-provided key into both layers without round-tripping
/// through serialization.</para>
/// </remarks>
public sealed class AiDotNetTensorsLicenseKey
{
    /// <summary>
    /// Gets the license key string (e.g., <c>aidn.{id}.{secret}</c> for
    /// signed offline-eligible keys, or <c>AIDN-*-{hex}</c> for
    /// server-validated community/CI keys).
    /// </summary>
    public string Key { get; }

    /// <summary>
    /// Gets or sets the URL of the license validation server.
    /// <para>
    /// <list type="bullet">
    /// <item><c>null</c> (default) — uses the built-in AiDotNet license
    /// server.</item>
    /// <item>Empty string — explicit offline-only mode; only signed
    /// <c>aidn.{id}.{sig}</c> keys are accepted.</item>
    /// <item>Custom URL — point at a self-hosted validation endpoint
    /// implementing the same Edge-Function contract.</item>
    /// </list>
    /// </para>
    /// </summary>
    public string? ServerUrl { get; set; }

    /// <summary>
    /// Gets or sets the environment label sent during validation
    /// (e.g., <c>production</c>, <c>staging</c>, <c>development</c>).
    /// </summary>
    public string? Environment { get; set; }

    /// <summary>
    /// Gets or sets the duration that a cached validation result is
    /// trusted when the server is unreachable. Defaults to 7 days.
    /// </summary>
    public TimeSpan OfflineGracePeriod { get; set; } = TimeSpan.FromDays(7);

    /// <summary>
    /// Gets or sets whether advisory machine-ID telemetry is sent to
    /// the license server during validation. Defaults to <c>true</c>.
    /// </summary>
    public bool EnableTelemetry { get; set; } = true;

    /// <summary>
    /// Creates a new license key with the supplied string.
    /// </summary>
    /// <param name="key">The license key string. Must not be null,
    /// empty, or whitespace.</param>
    /// <exception cref="ArgumentNullException">Thrown when
    /// <paramref name="key"/> is null.</exception>
    /// <exception cref="ArgumentException">Thrown when
    /// <paramref name="key"/> is empty or whitespace.</exception>
    public AiDotNetTensorsLicenseKey(string key)
    {
        if (key is null) throw new ArgumentNullException(nameof(key));
        if (string.IsNullOrWhiteSpace(key))
            throw new ArgumentException("License key must not be empty or whitespace.", nameof(key));
        Key = key;
    }
}
