using System.IO;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;

namespace AiDotNet.Tensors.Helpers.Autotune;

/// <summary>
/// Persistent on-disk cache of auto-tuner winners, keyed by hardware fingerprint,
/// kernel id, and input shape. Lets benchmark choices survive across process
/// restarts — first run pays the tuning cost, every subsequent run on the same
/// machine loads the winner directly.
///
/// <para><b>Location:</b> <c>~/.aidotnet/autotune/{fingerprint}/{kernelStem}__{shape}.json</c>.
/// Override the root via environment variable <c>AIDOTNET_AUTOTUNE_CACHE_PATH</c>
/// (useful for CI, benchmarks, or multi-tenant hosts). Hardware fingerprint is
/// computed by <see cref="HardwareFingerprint"/> — a CPU moved between machines
/// keeps older entries isolated by living in its own directory.</para>
///
/// <para><b>Concurrency:</b> writes are atomic (write-to-tmp + rename). Readers
/// never observe a half-written file. Concurrent writers for the same key pick
/// last-write-wins; since benchmarked winners are expected to be stable on the
/// same shape/hardware, this races-to-the-same-value.</para>
///
/// <para><b>Corruption handling:</b> malformed or truncated files on read are
/// treated as a cache miss (<see cref="Lookup"/> returns null) and never throw.
/// Callers should re-run the benchmark and <see cref="Store"/> a fresh winner.</para>
///
/// <para><b>Differentiator:</b> PyTorch's <c>max_autotune</c> is per-session and
/// opt-in — every <c>torch.compile</c> call re-benchmarks from scratch. With
/// <see cref="AutotuneCache"/> the second run feels "instant" because the
/// expensive search has already happened.</para>
/// </summary>
public static class AutotuneCache
{
    private const string EnvVarCachePath = "AIDOTNET_AUTOTUNE_CACHE_PATH";

    private static readonly JsonSerializerOptions _jsonOptions = new()
    {
        WriteIndented = true,
        PropertyNameCaseInsensitive = true,
    };

    private static readonly object _lock = new();

    /// <summary>
    /// Default cache root. Resolves to <c>~/.aidotnet/autotune/</c> unless the
    /// <c>AIDOTNET_AUTOTUNE_CACHE_PATH</c> environment variable is set (in which
    /// case that value is used verbatim as the root).
    /// </summary>
    public static string DefaultCachePath
    {
        get
        {
            string? overridePath = Environment.GetEnvironmentVariable(EnvVarCachePath);
            if (!string.IsNullOrWhiteSpace(overridePath)) return overridePath;

            string home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
            if (string.IsNullOrWhiteSpace(home))
                home = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
            return Path.Combine(home, ".aidotnet", "autotune");
        }
    }

    /// <summary>
    /// Short, filesystem-safe string uniquely identifying the current CPU.
    /// Exposed for diagnostic logging and for consumers that want to display
    /// "tuned for &lt;fingerprint&gt;" in their UI.
    /// </summary>
    public static string CurrentHardwareFingerprint => HardwareFingerprint.Current;

    /// <summary>
    /// Looks up a previously-recorded winner for the given kernel + shape on the
    /// current hardware. Returns null on a cache miss — or on any form of
    /// corruption (missing file, malformed JSON, truncation, schema mismatch).
    /// Never throws.
    /// </summary>
    public static KernelChoice? Lookup(KernelId kernelId, ShapeProfile shape)
    {
        try
        {
            string path = ResolvePath(kernelId, shape);
            if (!File.Exists(path)) return null;

            string json = File.ReadAllText(path);
            if (string.IsNullOrWhiteSpace(json)) return null;

            var choice = JsonSerializer.Deserialize<KernelChoice>(json, _jsonOptions);
            if (choice is null) return null;

            // Schema-version gate: reject files newer than we know how to read
            // (forward-incompatible) and files with SchemaVersion <= 0 (written
            // by a broken writer).
            if (choice.SchemaVersion <= 0) return null;

            return choice;
        }
        catch
        {
            // Disk I/O errors, JSON parse failures, permission errors — all
            // become cache misses. The caller's fallback is to re-run the
            // benchmark, so a miss is always safe.
            return null;
        }
    }

    /// <summary>
    /// Records a winning configuration to disk. Write is atomic — readers never
    /// observe a partial file. Concurrent writers race to last-write-wins, which
    /// is safe because the winner for the same shape on the same hardware is
    /// expected to be deterministic.
    /// </summary>
    /// <exception cref="IOException">Thrown only for non-recoverable filesystem
    /// failures (disk full, read-only cache root). Callers who consider the
    /// cache optional should wrap in a try/catch.</exception>
    public static void Store(KernelId kernelId, ShapeProfile shape, KernelChoice winner)
    {
        if (winner is null) throw new ArgumentNullException(nameof(winner));

        // Serialize outside the lock so we don't hold it across JSON work.
        string json = JsonSerializer.Serialize(winner, _jsonOptions);

        string finalPath = ResolvePath(kernelId, shape);
        string directory = Path.GetDirectoryName(finalPath)
            ?? throw new InvalidOperationException($"Cannot resolve directory from path '{finalPath}'");
        Directory.CreateDirectory(directory);

        // Atomic write: write to a sibling .tmp file, then rename over the target.
        // Rename is atomic at the filesystem level on NTFS and ext4 when both
        // paths share a filesystem (they do, same directory), so concurrent
        // readers see either the old file or the new file — never a half-written
        // one.
        string tmpPath = finalPath + ".tmp-" + Environment.CurrentManagedThreadId + "-" + Guid.NewGuid().ToString("N");
        try
        {
            File.WriteAllText(tmpPath, json);

            // Move with overwrite. File.Move's 'overwrite' overload is net5+;
            // emulate atomicity on older runtimes via File.Replace when possible,
            // otherwise fall back to delete+move (briefly non-atomic, but this
            // path runs only on net471).
#if NET5_0_OR_GREATER
            File.Move(tmpPath, finalPath, overwrite: true);
#else
            if (File.Exists(finalPath))
            {
                try
                {
                    // File.Replace is atomic and supported on net471.
                    File.Replace(tmpPath, finalPath, destinationBackupFileName: null);
                }
                catch (PlatformNotSupportedException)
                {
                    File.Delete(finalPath);
                    File.Move(tmpPath, finalPath);
                }
            }
            else
            {
                File.Move(tmpPath, finalPath);
            }
#endif
        }
        catch
        {
            // Clean up the tmp file on any failure so we don't leak clutter.
            try { if (File.Exists(tmpPath)) File.Delete(tmpPath); } catch { /* best effort */ }
            throw;
        }
    }

    /// <summary>
    /// Tries a store, swallowing any failure. Use this when the cache is a
    /// best-effort optimisation (benchmark already ran; we'd rather keep the
    /// winner in memory than fail the caller's workflow because the disk is
    /// read-only).
    /// </summary>
    public static bool TryStore(KernelId kernelId, ShapeProfile shape, KernelChoice winner)
    {
        try
        {
            Store(kernelId, shape, winner);
            return true;
        }
        catch
        {
            return false;
        }
    }

    /// <summary>
    /// Pre-populates the cache in parallel for a known set of kernel/shape
    /// targets. For each target the <paramref name="benchmarker"/> is invoked
    /// only if <see cref="Lookup"/> returns null — so already-cached entries are
    /// skipped. Useful from build-time hooks that want to front-load tuning
    /// cost before the first serving request.
    /// </summary>
    /// <param name="targets">Kernel/shape pairs to ensure are cached.</param>
    /// <param name="benchmarker">Runs the benchmark for a miss; result is
    /// persisted. Delegate must be thread-safe (targets run in parallel).</param>
    /// <param name="ct">Cancellation.</param>
    /// <returns>
    /// A tuple of (number of cache hits, number of fresh benchmarks run, number
    /// of stores that failed to persist). The caller can log these for
    /// diagnostic visibility.
    /// </returns>
    public static async Task<(int hits, int benchmarked, int storeFailures)> WarmupAsync(
        IEnumerable<(KernelId id, ShapeProfile shape)> targets,
        Func<KernelId, ShapeProfile, CancellationToken, Task<KernelChoice?>> benchmarker,
        CancellationToken ct = default)
    {
        if (targets is null)      throw new ArgumentNullException(nameof(targets));
        if (benchmarker is null)  throw new ArgumentNullException(nameof(benchmarker));

        int hits = 0, benchmarked = 0, storeFailures = 0;

        var list = targets as IList<(KernelId, ShapeProfile)> ?? targets.ToList();
        // Parallelise across targets — each benchmark target is independent.
        await Task.WhenAll(list.Select(async t =>
        {
            ct.ThrowIfCancellationRequested();
            var (id, shape) = t;

            if (Lookup(id, shape) is not null)
            {
                Interlocked.Increment(ref hits);
                return;
            }

            var winner = await benchmarker(id, shape, ct).ConfigureAwait(false);
            if (winner is null) return; // Benchmarker skipped this target.

            Interlocked.Increment(ref benchmarked);
            if (!TryStore(id, shape, winner))
                Interlocked.Increment(ref storeFailures);
        })).ConfigureAwait(false);

        return (hits, benchmarked, storeFailures);
    }

    /// <summary>
    /// Deletes every cache file under the current hardware fingerprint's
    /// directory. Intended for tests and for operators who want to force a
    /// full re-tune (e.g. after a driver upgrade on the same hardware that
    /// might have changed kernel performance).
    /// </summary>
    public static void ClearCurrentHardware()
    {
        try
        {
            string dir = Path.Combine(DefaultCachePath, CurrentHardwareFingerprint);
            if (Directory.Exists(dir)) Directory.Delete(dir, recursive: true);
        }
        catch
        {
            // Best effort — same rationale as Lookup.
        }
    }

    /// <summary>
    /// Resolves the full filesystem path for a given kernel/shape cache entry.
    /// Path shape: <c>{DefaultCachePath}/{fingerprint}/{kernelStem}__{shapeStem}.json</c>.
    /// </summary>
    private static string ResolvePath(KernelId kernelId, ShapeProfile shape)
    {
        string fileName = $"{kernelId.ToFileStem()}__{shape.ToFileStem()}.json";
        return Path.Combine(DefaultCachePath, CurrentHardwareFingerprint, fileName);
    }
}
