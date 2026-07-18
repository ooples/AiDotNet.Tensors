// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Text.Json;

namespace AiDotNet.Tensors.Licensing;

/// <summary>
/// Provides the PUBLIC license-signing key(s) embedded as a resource in official builds. Used to verify
/// <c>aidn2.</c> asymmetric (Ed25519 / EdDSA) license tokens offline — the Tensors port of the main SDK's
/// <c>LicensePublicKeyProvider</c>, parsed with <see cref="System.Text.Json"/>.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> The <c>aidn2.</c> license format is signed on the AiDotNet license server
/// with a <b>private</b> key that never leaves the server. Every published DLL only embeds the matching
/// <b>public</b> key. A public key can verify a signature but cannot create one, so extracting it from the
/// DLL gives an attacker nothing — they still cannot forge a license without the private key.</para>
///
/// <para>The embedded resource <c>AiDotNet.Tensors.LicensePublicKey</c> is a small JSON document listing
/// one or more Ed25519 public keys, each tagged with a <c>kid</c> (key id). A token names the <c>kid</c>
/// it was signed with, so the signing key can be <b>rotated</b> without invalidating already-issued
/// licenses that reference an older, still-trusted key.</para>
///
/// <para>Resource schema (JWK OKP — RFC 8037; an Ed25519 public key is the 32-byte <c>x</c> value):</para>
/// <code>
/// { "keys": [ { "kty": "OKP", "crv": "Ed25519", "kid": "prod-2026a", "x": "&lt;base64url 32-byte pubkey&gt;" } ] }
/// </code>
/// </remarks>
internal static class TensorsLicensePublicKeyProvider
{
    private const string ResourceName = "AiDotNet.Tensors.LicensePublicKey";

    /// <summary>Ed25519 raw public keys are exactly 32 bytes.</summary>
    internal const int Ed25519PublicKeySize = 32;

    // kid -> raw 32-byte Ed25519 public key.
    private static Dictionary<string, byte[]>? _cachedKeys;
    private static bool _loaded;
    private static readonly object _lock = new();

    /// <summary>
    /// TEST-ONLY: overrides the embedded public key set so tests can inject an ephemeral test keypair's
    /// public half and exercise the real signature-verification path. Pass <see langword="null"/> to
    /// simulate a dev/fork build (no embedded public key) and assert fail-closed offline behaviour.
    /// </summary>
    internal static void OverrideForTesting(IReadOnlyDictionary<string, byte[]>? keys)
    {
        lock (_lock)
        {
            _cachedKeys = keys is { Count: > 0 } ? CloneAll(keys) : null;
            _loaded = true;
        }
    }

    /// <summary>
    /// Snapshots the currently-effective public key set (for scoped test overrides), or null if none.
    /// </summary>
    internal static IReadOnlyDictionary<string, byte[]>? CurrentSnapshot()
    {
        lock (_lock)
        {
            EnsureLoadedNoLock();
            return _cachedKeys is { Count: > 0 } ? CloneAll(_cachedKeys) : null;
        }
    }

    /// <summary>
    /// Gets whether this build embeds at least one license public key (i.e. it can verify <c>aidn2.</c>
    /// tokens offline).
    /// </summary>
    internal static bool HasAnyKey
    {
        get
        {
            lock (_lock)
            {
                EnsureLoadedNoLock();
                return _cachedKeys is { Count: > 0 };
            }
        }
    }

    /// <summary>
    /// Resolves the raw 32-byte Ed25519 public key for a given <c>kid</c>. Returns false if this build
    /// embeds no matching key.
    /// </summary>
    internal static bool TryGetPublicKey(string kid, out byte[] publicKey)
    {
        publicKey = Array.Empty<byte>();
        if (string.IsNullOrEmpty(kid)) return false;

        lock (_lock)
        {
            EnsureLoadedNoLock();
            if (_cachedKeys is not null && _cachedKeys.TryGetValue(kid, out var found))
            {
                publicKey = (byte[])found.Clone();
                return true;
            }
        }

        return false;
    }

    private static void EnsureLoadedNoLock()
    {
        if (_loaded) return;

        try
        {
            var assembly = typeof(TensorsLicensePublicKeyProvider).Assembly;
            using var stream = assembly.GetManifestResourceStream(ResourceName);
            if (stream is null || stream.Length == 0)
            {
                _cachedKeys = null;
                return;
            }

            using var reader = new StreamReader(stream, Encoding.UTF8);
            _cachedKeys = Parse(reader.ReadToEnd());
        }
        catch (Exception ex)
        {
            System.Diagnostics.Trace.TraceWarning(
                "TensorsLicensePublicKeyProvider: failed to load public key(s): " + ex.GetType().Name + ": " + ex.Message);
            _cachedKeys = null;
        }
        finally
        {
            _loaded = true;
        }
    }

    /// <summary>
    /// Parses the JWK OKP public-key manifest. Returns null when no usable Ed25519 key is present.
    /// Non-Ed25519 / malformed / wrong-length entries are skipped (fail closed).
    /// </summary>
    internal static Dictionary<string, byte[]>? Parse(string json)
    {
        if (string.IsNullOrWhiteSpace(json)) return null;

        JsonElement root;
        try
        {
            using var doc = JsonDocument.Parse(json);
            // Clone so the element survives disposal of the JsonDocument.
            root = doc.RootElement.Clone();
        }
        catch (JsonException)
        {
            return null;
        }

        if (root.ValueKind != JsonValueKind.Object
            || !root.TryGetProperty("keys", out var keysEl)
            || keysEl.ValueKind != JsonValueKind.Array)
        {
            return null;
        }

        var result = new Dictionary<string, byte[]>(StringComparer.Ordinal);
        foreach (var entry in keysEl.EnumerateArray())
        {
            if (entry.ValueKind != JsonValueKind.Object) continue;

            string? kty = GetString(entry, "kty");
            string? kid = GetString(entry, "kid");
            string? x = GetString(entry, "x");
            string? crv = GetString(entry, "crv");

            // Enforce the COMPLETE Ed25519 JWK type contract: kty MUST be "OKP" and crv MUST be "Ed25519"
            // (both present and exact), in addition to a non-empty kid and x. A missing/incorrect kty or crv
            // is rejected so an entry of an unknown/unsupported key type can never be trusted as Ed25519.
            if (kid is null || kid.Length == 0 || x is null || x.Length == 0
                || !string.Equals(kty, "OKP", StringComparison.Ordinal)
                || !string.Equals(crv, "Ed25519", StringComparison.Ordinal))
            {
                continue;
            }

            try
            {
                byte[] pub = Base64UrlHelper.Decode(x);
                if (pub.Length != Ed25519PublicKeySize)
                {
                    System.Diagnostics.Trace.TraceWarning(
                        "TensorsLicensePublicKeyProvider: skipping key '" + kid + "' with wrong Ed25519 length " + pub.Length + ".");
                    continue;
                }

                result[kid] = pub;
            }
            catch (FormatException ex)
            {
                System.Diagnostics.Trace.TraceWarning(
                    "TensorsLicensePublicKeyProvider: skipping malformed key '" + kid + "': " + ex.Message);
            }
        }

        return result.Count > 0 ? result : null;
    }

    private static string? GetString(JsonElement obj, string name)
        => obj.TryGetProperty(name, out var el) && el.ValueKind == JsonValueKind.String ? el.GetString() : null;

    private static Dictionary<string, byte[]> CloneAll(IReadOnlyDictionary<string, byte[]> src)
    {
        var copy = new Dictionary<string, byte[]>(StringComparer.Ordinal);
        foreach (var kvp in src)
        {
            copy[kvp.Key] = kvp.Value is null ? Array.Empty<byte>() : (byte[])kvp.Value.Clone();
        }

        return copy;
    }
}
