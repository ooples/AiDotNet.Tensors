// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using System.Collections.Concurrent;
using System.Globalization;
using System.IO;
using System.Security.Cryptography;
using System.Text;

namespace AiDotNet.Tensors.Engines.Compilation.FusionTileIr;

/// <summary>
/// Thread-safe, atomic disk and in-memory artifact cache keyed by SHA-256 of <c>(CudaSource | SmArch | CompilerOptions)</c>.
/// Stores compiled CUBIN / PTX binaries produced by linked <see cref="FusionTileDescriptor"/> graphs so cold process restarts
/// load linked cross-layer kernels in microseconds without invoking NVRTC.
/// </summary>
internal static class FusionTileArtifactCache
{
    private static readonly ConcurrentDictionary<string, byte[]> s_memoryCache = new(StringComparer.Ordinal);
    private static string? s_overrideCacheRoot;

    /// <summary>Allows unit tests to isolate disk artifact persistence inside a temporary directory.</summary>
    internal static string? OverrideCacheDirectory
    {
        get => s_overrideCacheRoot;
        set => s_overrideCacheRoot = value;
    }

    /// <summary>Drops the in-memory tier so tests can prove a lookup is served from the persisted file.</summary>
    internal static void ClearMemoryCache() => s_memoryCache.Clear();

    public static string ComputeArtifactKey(string cudaOrPtxSource, int smMajor, int smMinor, bool fastMath)
    {
        if (cudaOrPtxSource is null) throw new ArgumentNullException(nameof(cudaOrPtxSource));
        string payload = string.Concat(
            "sm_",
            smMajor.ToString(CultureInfo.InvariantCulture),
            smMinor.ToString(CultureInfo.InvariantCulture),
            fastMath ? "|fast|" : "|ieee|",
            cudaOrPtxSource);

        using var sha = SHA256.Create();
        byte[] digest = sha.ComputeHash(Encoding.UTF8.GetBytes(payload));
        var sb = new StringBuilder(64);
        for (int i = 0; i < digest.Length; i++)
        {
            sb.Append(digest[i].ToString("x2", CultureInfo.InvariantCulture));
        }
        return sb.ToString();
    }

    public static bool TryLoadArtifact(string sha256Key, string extension, out byte[] artifactBytes)
    {
        if (string.IsNullOrWhiteSpace(sha256Key))
            throw new ArgumentException("Artifact key cannot be empty.", nameof(sha256Key));

        string memKey = sha256Key + extension;
        if (s_memoryCache.TryGetValue(memKey, out var cached) && cached is not null)
        {
            // Hand out a copy so a caller mutating the returned array cannot corrupt later hits.
            artifactBytes = (byte[])cached.Clone();
            return true;
        }

        string? dir = ResolveCacheDirectory();
        if (dir is not null)
        {
            string path = Path.Combine(dir, sha256Key + extension);
            if (File.Exists(path))
            {
                try
                {
                    byte[] fromDisk = File.ReadAllBytes(path);
                    if (fromDisk.Length > 0)
                    {
                        s_memoryCache[memKey] = fromDisk;
                        artifactBytes = (byte[])fromDisk.Clone();
                        return true;
                    }
                }
                catch (IOException)
                {
                    // Best-effort cache lookup; fall through on concurrent I/O contention.
                }
                catch (UnauthorizedAccessException)
                {
                    // Unreadable artifact (permissions / sandbox): treat as a cache miss.
                }
            }
        }

        artifactBytes = Array.Empty<byte>();
        return false;
    }

    public static void StoreArtifact(string sha256Key, string extension, byte[] artifactBytes)
    {
        if (string.IsNullOrWhiteSpace(sha256Key))
            throw new ArgumentException("Artifact key cannot be empty.", nameof(sha256Key));
        if (artifactBytes is null || artifactBytes.Length == 0)
            return;

        // Keep a private copy so a producer reusing its buffer cannot change what later hits return.
        byte[] owned = (byte[])artifactBytes.Clone();
        string memKey = sha256Key + extension;
        s_memoryCache[memKey] = owned;

        string? dir = ResolveCacheDirectory();
        if (dir is null)
            return;

        string? tempPath = null;
        try
        {
            Directory.CreateDirectory(dir);
            string finalPath = Path.Combine(dir, sha256Key + extension);

            // Content-addressed: a valid (non-empty) existing artifact already holds these bytes.
            var existing = new FileInfo(finalPath);
            if (existing.Exists && existing.Length > 0)
                return;

            tempPath = Path.Combine(dir, sha256Key + "." + Guid.NewGuid().ToString("N") + ".tmp");
            File.WriteAllBytes(tempPath, owned);

            if (existing.Exists)
            {
                // Replace the empty/invalid entry atomically so later processes stop missing on it.
#if NETCOREAPP3_0_OR_GREATER
                File.Move(tempPath, finalPath, overwrite: true);
#else
                File.Replace(tempPath, finalPath, destinationBackupFileName: null);
#endif
            }
            else
            {
                File.Move(tempPath, finalPath);
            }
        }
        catch (IOException)
        {
            // Another process/thread published the identical content-addressed artifact concurrently.
        }
        catch (UnauthorizedAccessException)
        {
            // Read-only sandbox fallback: memory cache remains populated.
        }
        finally
        {
            // Remove the staged file if publication did not consume it; never touch the published artifact.
            if (tempPath is not null)
            {
                try
                {
                    if (File.Exists(tempPath)) File.Delete(tempPath);
                }
                catch (IOException) { }
                catch (UnauthorizedAccessException) { }
            }
        }
    }

    private static string? ResolveCacheDirectory()
    {
        if (!string.IsNullOrWhiteSpace(s_overrideCacheRoot))
            return s_overrideCacheRoot;

        string? envDir = Environment.GetEnvironmentVariable("AIDOTNET_FUSION_TILE_CACHE_DIR");
        if (!string.IsNullOrWhiteSpace(envDir))
            return envDir;

        try
        {
            string localAppData = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
            if (!string.IsNullOrWhiteSpace(localAppData))
            {
                return Path.Combine(localAppData, "AiDotNet", "FusionTileCache");
            }
        }
        catch
        {
            // Ignore environment folder lookup failures in restricted containers.
        }
        return null;
    }
}
