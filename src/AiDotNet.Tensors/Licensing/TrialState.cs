// Copyright (c) AiDotNet. All rights reserved.

using System.IO;
using System.Text.Json;

namespace AiDotNet.Tensors.Licensing;

/// <summary>
/// Persistent state for the AiDotNet.Tensors free-trial mechanism.
/// Persisted to <c>~/.aidotnet/tensors-trial.json</c> by default;
/// tests can override the path via
/// <see cref="PersistenceGuard.SetTestTrialFilePathOverride"/>.
/// </summary>
/// <remarks>
/// <para><b>Why a separate trial file from upstream:</b></para>
/// <para>
/// Upstream <c>AiDotNet</c> writes <c>~/.aidotnet/trial.json</c> for
/// model-level operations. The tensor layer uses a sibling file so
/// the two trials are independently tracked — a user who just
/// installed <c>AiDotNet.Tensors</c> for safetensors I/O gets a fresh
/// trial budget rather than inheriting whatever counter upstream
/// previously decremented. When upstream is also installed, the
/// upstream guard suppresses the tensor guard via
/// <see cref="PersistenceGuard.InternalOperation"/>, so only the
/// upstream counter ticks for save/load issued through the upstream
/// API surface.
/// </para>
/// </remarks>
public sealed class TrialState
{
    /// <summary>
    /// Default budget — 30 days OR 50 persistence operations,
    /// whichever expires first. Matches upstream's "generous trial
    /// for evaluation" pattern.
    /// </summary>
    public const int DefaultMaxOperations = 50;

    /// <summary>Default trial window in days.</summary>
    public const int DefaultMaxDays = 30;

    /// <summary>UTC timestamp the trial first started.</summary>
    public DateTimeOffset StartedAt { get; set; }

    /// <summary>Number of save+load+serialize+deserialize ops counted
    /// against the trial.</summary>
    public int OperationsConsumed { get; set; }

    /// <summary>
    /// Returns <c>true</c> if the trial budget is exhausted — either
    /// the time window is past or the operation count is hit.
    /// </summary>
    public bool IsExpired(DateTimeOffset now)
    {
        if (now - StartedAt >= TimeSpan.FromDays(DefaultMaxDays)) return true;
        if (OperationsConsumed >= DefaultMaxOperations) return true;
        return false;
    }

    /// <summary>
    /// Default location: <c>%USERPROFILE%/.aidotnet/tensors-trial.json</c>
    /// on Windows, <c>~/.aidotnet/tensors-trial.json</c> on Unix.
    /// </summary>
    public static string DefaultPath
    {
        get
        {
            string home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
            return Path.Combine(home, ".aidotnet", "tensors-trial.json");
        }
    }

    /// <summary>
    /// Reads the trial state from <paramref name="path"/>, or returns
    /// a fresh state initialised to "started now, zero operations" if
    /// the file does not exist or fails to parse. The fresh state is
    /// not written to disk by this method — call <see cref="Save"/>
    /// after consuming an operation.
    /// </summary>
    public static TrialState Load(string path)
    {
        try
        {
            if (!File.Exists(path))
                return new TrialState { StartedAt = DateTimeOffset.UtcNow, OperationsConsumed = 0 };
            string json = File.ReadAllText(path);
            using var doc = JsonDocument.Parse(json);
            var root = doc.RootElement;
            var state = new TrialState
            {
                StartedAt = root.TryGetProperty("startedAt", out var s)
                    && DateTimeOffset.TryParse(s.GetString(), out var parsed)
                    ? parsed
                    : DateTimeOffset.UtcNow,
                OperationsConsumed = root.TryGetProperty("operationsConsumed", out var o)
                    && o.ValueKind == JsonValueKind.Number
                    ? o.GetInt32()
                    : 0,
            };
            return state;
        }
        catch
        {
            // Corrupt file → start fresh rather than block the user.
            return new TrialState { StartedAt = DateTimeOffset.UtcNow, OperationsConsumed = 0 };
        }
    }

    /// <summary>
    /// Persists this state to <paramref name="path"/>, creating any
    /// missing parent directories. Failures are swallowed (with a
    /// trace warning) so a user with a read-only home directory can
    /// still use the library — the trial just won't persist between
    /// runs.
    /// </summary>
    public void Save(string path)
    {
        try
        {
            string? dir = Path.GetDirectoryName(path);
            if (!string.IsNullOrEmpty(dir) && !Directory.Exists(dir))
                Directory.CreateDirectory(dir!);
            string json = JsonSerializer.Serialize(new
            {
                startedAt = StartedAt.ToString("o"),
                operationsConsumed = OperationsConsumed,
            });
            File.WriteAllText(path, json);
        }
        catch (Exception ex)
        {
            System.Diagnostics.Trace.TraceWarning(
                "AiDotNet.Tensors trial state could not be persisted: "
                + ex.GetType().Name + ": " + ex.Message);
        }
    }
}
