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

    /// <summary>
    /// The schema version this binary can read. Must match
    /// <see cref="KernelChoice.SchemaVersion"/>'s default. Files with a higher
    /// version are treated as misses (forward-compat escape hatch for newer
    /// library versions that extend the schema); files with a non-positive
    /// version are treated as corruption.
    /// </summary>
    internal const int CurrentSchemaVersion = 1;

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
            if (choice.SchemaVersion <= 0 || choice.SchemaVersion > CurrentSchemaVersion) return null;

            // Payload-completeness gate: KernelChoice's `Variant` and `Parameters`
            // are non-nullable in the C# type, but System.Text.Json will happily
            // assign null if the JSON explicitly contains `"Variant": null` (the
            // property initializer only runs when the key is absent). A partial
            // or tampered file like `{"Variant":null,"Parameters":null,"SchemaVersion":1}`
            // deserializes without throwing. Callers of Lookup then dereference
            // those fields and crash — so we promote incomplete payloads to a
            // safe cache miss here, and the caller re-runs the benchmark.
            if (string.IsNullOrWhiteSpace(choice.Variant)) return null;
            if (choice.Parameters is null) return null;

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
    /// <exception cref="ArgumentNullException"><paramref name="winner"/> is null.</exception>
    /// <exception cref="ArgumentException">The winner has a missing
    /// <see cref="KernelChoice.Variant"/> or null
    /// <see cref="KernelChoice.Parameters"/> — failing loudly here keeps the
    /// <c>Store</c> → <c>Lookup</c> round-trip contract honest, instead of
    /// writing a file that the reader would immediately reject as corruption.</exception>
    /// <exception cref="IOException">Thrown only for non-recoverable filesystem
    /// failures (disk full, read-only cache root). Callers who consider the
    /// cache optional should wrap in a try/catch.</exception>
    public static void Store(KernelId kernelId, ShapeProfile shape, KernelChoice winner)
    {
        if (winner is null) throw new ArgumentNullException(nameof(winner));

        // Boundary validation: reject the same "incomplete payload" shapes that
        // Lookup's completeness guard refuses to return. Without this, a caller
        // could store `new KernelChoice { Variant = null, Parameters = null }`,
        // succeed, and then see Lookup() report a miss — silently breaking the
        // round-trip contract. Fail loudly at the write boundary instead.
        if (string.IsNullOrWhiteSpace(winner.Variant))
            throw new ArgumentException(
                "KernelChoice.Variant must be a non-empty, non-whitespace string. " +
                "A winner without a variant identifier is not a valid cache entry.",
                nameof(winner));
        if (winner.Parameters is null)
            throw new ArgumentException(
                "KernelChoice.Parameters must not be null (use an empty dictionary " +
                "if no tuned hyperparameters apply).",
                nameof(winner));

        // Persist a shallow copy rather than the caller's instance so we can:
        //   (1) stamp CurrentSchemaVersion authoritatively — callers don't get
        //       to write arbitrary SchemaVersion values that Lookup would then
        //       reject (same round-trip contract as above).
        //   (2) defensively copy Parameters so a later mutation by the caller
        //       doesn't alter what ends up on disk (or doesn't, if we lose a
        //       race with the write).
        //   (3) leave the caller's object untouched — no surprising side effects.
        var persisted = new KernelChoice
        {
            Variant        = winner.Variant,
            Parameters     = new Dictionary<string, string>(winner.Parameters),
            MeasuredGflops = winner.MeasuredGflops,
            MeasuredTimeMs = winner.MeasuredTimeMs,
            RecordedAtUtc  = winner.RecordedAtUtc,
            SchemaVersion  = CurrentSchemaVersion,
        };

        // Serialize outside the lock so we don't hold it across JSON work.
        string json = JsonSerializer.Serialize(persisted, _jsonOptions);

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
