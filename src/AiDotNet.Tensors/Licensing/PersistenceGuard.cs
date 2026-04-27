// Copyright (c) AiDotNet. All rights reserved.

using System.IO;
using System.Threading;

namespace AiDotNet.Tensors.Licensing;

/// <summary>
/// Centralised enforcement point for AiDotNet.Tensors persistence
/// operations. Every public read/write of a tensor file format
/// (<c>GgufReader</c>, <c>SafetensorsReader/Writer</c>, pickle / .pt
/// reader, sharded checkpoint loader) must call through one of the
/// <c>EnforceBefore*</c> methods so the same gate covers every entry
/// point.
/// </summary>
/// <remarks>
/// <para><b>How licensing flows:</b></para>
/// <list type="number">
///   <item><description>The user provides a key string via env var
///   <c>AIDOTNET_LICENSE_KEY</c>, the file
///   <c>~/.aidotnet/license.key</c>, or
///   <see cref="SetActiveLicenseKey"/>.</description></item>
///   <item><description>On the first persistence op, the guard
///   resolves the key, hits the licensing server (cached for the
///   offline-grace window), and checks the returned capability set
///   includes <c>tensors:save</c> / <c>tensors:load</c>.</description></item>
///   <item><description>If no key is configured, the guard falls back
///   to <see cref="TrialState"/> — the user gets the default trial
///   budget (50 ops or 30 days). After the budget is exhausted, the
///   guard throws <see cref="LicenseRequiredException"/>.</description></item>
///   <item><description>Upstream <c>AiDotNet</c> wraps its own
///   persistence calls in <see cref="InternalOperation"/> so the
///   tensor guard's counter doesn't double-tick when both layers are
///   in the same call stack.</description></item>
/// </list>
/// <para><b>Thread safety:</b></para>
/// <para>
/// Every mutable piece of state — the active key, the
/// internal-operation depth, the test trial-file override — is held
/// in <see cref="AsyncLocal{T}"/> rather than <c>[ThreadStatic]</c>.
/// An <c>await</c> inside a guarded scope frequently resumes on a
/// different thread-pool thread; thread-static state would fall away
/// on the continuation, so the guard would either fire on a call
/// that should have been suppressed or miss a call that should have
/// been counted. <see cref="AsyncLocal{T}"/> flows with the logical
/// call context.
/// </para>
/// </remarks>
public static class PersistenceGuard
{
    // Capability strings used by the guard. The licensing server is
    // expected to attach these to a key when the user's tier covers
    // them.
    public const string CapabilitySave = "tensors:save";
    public const string CapabilityLoad = "tensors:load";
    public const string CapabilitySerialize = "tensors:save";
    public const string CapabilityDeserialize = "tensors:load";

    // ─── Active license key ────────────────────────────────────────

    private static readonly AsyncLocal<AiDotNetTensorsLicenseKey?> _activeKey = new();

    /// <summary>
    /// Sets the active license key for the current logical call
    /// context. Returns an <see cref="IDisposable"/> that restores
    /// the prior key on disposal, so nested scopes compose cleanly.
    /// Upstream <c>AiDotNet</c>'s <c>AiModelBuilder</c> calls this at
    /// the start of <c>BuildAsync</c> to flow its license into the
    /// tensor layer; standalone tensor users call it once at startup
    /// (or rely on the env-var/file fallback).
    /// </summary>
    public static IDisposable SetActiveLicenseKey(AiDotNetTensorsLicenseKey? key)
    {
        var prev = _activeKey.Value;
        _activeKey.Value = key;
        return new ActiveKeyScope(prev);
    }

    private sealed class ActiveKeyScope : IDisposable
    {
        private readonly AiDotNetTensorsLicenseKey? _previous;
        public ActiveKeyScope(AiDotNetTensorsLicenseKey? prev) { _previous = prev; }
        public void Dispose() => _activeKey.Value = _previous;
    }

    // ─── Internal-operation suppression ────────────────────────────

    private static readonly AsyncLocal<int> _internalDepth = new();

    /// <summary>
    /// Increments the internal-operation counter for the current
    /// logical call context. While the returned scope is active,
    /// every <c>EnforceBefore*</c> call is a no-op — used by upstream
    /// <c>AiDotNet</c> when its own <c>ModelPersistenceGuard</c> has
    /// already enforced and the tensor guard would otherwise
    /// double-count. Supports nesting; only the outermost dispose
    /// re-enables enforcement.
    /// </summary>
    public static IDisposable InternalOperation()
    {
        _internalDepth.Value++;
        return new InternalOperationScope();
    }

    private sealed class InternalOperationScope : IDisposable
    {
        private bool _disposed;
        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            _internalDepth.Value = Math.Max(0, _internalDepth.Value - 1);
        }
    }

    // ─── Test-only trial file override ─────────────────────────────

    private static readonly AsyncLocal<string?> _trialFilePathOverride = new();

    /// <summary>
    /// Test hook — overrides the trial-state file path for the
    /// current logical call context so test runs don't mutate the
    /// developer's <c>~/.aidotnet/tensors-trial.json</c>. Pass
    /// <c>null</c> to clear an existing override.
    /// </summary>
    /// <remarks>
    /// <c>internal</c> on purpose — exposed only via the test project's
    /// <c>InternalsVisibleTo</c> entry. Making this <c>public</c> would
    /// hand every consumer a one-line trial-tracking bypass: redirect
    /// the override to a fresh temp file before each operation and the
    /// counter never accumulates against the real trial.json.
    /// </remarks>
    internal static IDisposable SetTestTrialFilePathOverride(string? path)
    {
        var prev = _trialFilePathOverride.Value;
        _trialFilePathOverride.Value = path;
        return new TrialPathScope(prev);
    }

    private sealed class TrialPathScope : IDisposable
    {
        private readonly string? _previous;
        public TrialPathScope(string? prev) { _previous = prev; }
        public void Dispose() => _trialFilePathOverride.Value = _previous;
    }

    // ─── Shared validator cache ────────────────────────────────────
    //
    // Validating the key on every persistence op would re-run online
    // checks (or at minimum re-parse env-var / disk on every call).
    // Cache by key string so repeated saves with the same key reuse
    // the validator instance — which itself caches the network result
    // for the offline-grace window.

    private static readonly System.Collections.Concurrent.ConcurrentDictionary<string, LicenseValidator> _validators = new();
    private static LicenseValidator GetValidator(AiDotNetTensorsLicenseKey key)
        => _validators.GetOrAdd(key.Key, _ => new LicenseValidator(key));

    // Per-key "pending allowance consumed" set. The first
    // ValidationPending result for a key is allowed (lets users on a
    // flaky network bootstrap into the offline-grace window), but
    // subsequent pending results are rejected — the validator caches
    // a pending result for the full grace period, so without this cap
    // a user with persistent network failure would get unlimited free
    // operations for the entire grace window.
    //
    // Cleared whenever validation transitions to Active, so a transient
    // network blip followed by a successful re-check restores the
    // single-allowance budget for the next outage.
    private static readonly System.Collections.Concurrent.ConcurrentDictionary<string, byte> _pendingConsumed = new();

    // Single-process gate around the trial state's read-check-increment-
    // write sequence. Without this, two concurrent EnforceCore calls can
    // each load the same on-disk operationsConsumed value, both see the
    // pre-tick count, both write back count+1, and the second write
    // silently overwrites the first — losing increments and letting
    // users exceed the operation cap.
    //
    // This is process-local. Multi-process safety would need
    // OS-level file locking inside TrialState.Load/Save; persistence
    // I/O is intentionally low-frequency (Save / Load / Serialize /
    // Deserialize entry points) so a per-process lock is the right
    // grain — matches upstream's ModelPersistenceGuard pattern.
    private static readonly object _trialStateGate = new();

    // ─── Enforcement entry points ──────────────────────────────────

    /// <summary>
    /// Enforces licensing before a save operation (write a tensor or
    /// state-dict to a file).
    /// </summary>
    public static void EnforceBeforeSave() => EnforceCore(CapabilitySave, "save");

    /// <summary>
    /// Enforces licensing before a load operation (read a tensor or
    /// state-dict from a file).
    /// </summary>
    public static void EnforceBeforeLoad() => EnforceCore(CapabilityLoad, "load");

    /// <summary>
    /// Enforces licensing before a Serialize call. Same gate as
    /// <see cref="EnforceBeforeSave"/> — kept as a separate method so
    /// callers express intent.
    /// </summary>
    public static void EnforceBeforeSerialize() => EnforceCore(CapabilitySerialize, "serialize");

    /// <summary>
    /// Enforces licensing before a Deserialize call. Same gate as
    /// <see cref="EnforceBeforeLoad"/>.
    /// </summary>
    public static void EnforceBeforeDeserialize() => EnforceCore(CapabilityDeserialize, "deserialize");

    private static void EnforceCore(string requiredCapability, string operationLabel)
    {
        // Suppressed under InternalOperation — upstream AiDotNet
        // already enforced before delegating to us.
        if (_internalDepth.Value > 0) return;

        // Resolve key: AsyncLocal > env var > file. None present →
        // fall through to the trial path.
        var key = ResolveActiveKey();
        if (key is not null)
        {
            var validator = GetValidator(key);
            var result = validator.Validate();

            if (result.Status == LicenseKeyStatus.Active && result.HasCapability(requiredCapability))
            {
                // Active validation clears any prior pending-consumed
                // marker for this key — a transient outage that
                // recovers should restore the single-pending allowance
                // for the next outage.
                _pendingConsumed.TryRemove(key.Key, out _);
                return;
            }
            if (result.Status == LicenseKeyStatus.Active && !result.HasCapability(requiredCapability))
                throw new LicenseRequiredException(
                    $"License is active but does not include the '{requiredCapability}' capability " +
                    $"required for tensor {operationLabel}. Tier: {result.Tier ?? "(unknown)"}.",
                    trialDaysElapsed: null,
                    operationsPerformed: null,
                    keyStatus: LicenseKeyStatus.CapabilityMissing,
                    requiredCapability: requiredCapability);

            if (result.Status == LicenseKeyStatus.ValidationPending)
            {
                // First pending result per key is allowed (bootstraps
                // users on a flaky network). Subsequent pending
                // results — which the validator returns from its
                // cached pending throughout the offline grace window
                // — are rejected, so a user with persistent network
                // failure can't loop through unlimited operations.
                // TryAdd returns false if the key was already
                // present, i.e. we've already consumed the allowance.
                if (_pendingConsumed.TryAdd(key.Key, 0))
                    return;

                throw new LicenseRequiredException(
                    "License server unreachable and the one-call validation-pending allowance " +
                    "has already been consumed for this key. Restore network connectivity to " +
                    "re-validate, or fall back to upstream AiDotNet for offline HMAC verification.",
                    trialDaysElapsed: null,
                    operationsPerformed: null,
                    keyStatus: LicenseKeyStatus.ValidationPending,
                    requiredCapability: requiredCapability);
            }

            // Anything else (Expired / Revoked / Invalid / SeatLimit) → throw.
            throw new LicenseRequiredException(
                $"License key validation failed (status={result.Status}): {result.Message ?? "(no message)"}.",
                trialDaysElapsed: null,
                operationsPerformed: null,
                keyStatus: result.Status,
                requiredCapability: requiredCapability);
        }

        // No key configured → consult the trial state. The
        // load → IsExpired → increment → save sequence runs under
        // _trialStateGate so concurrent EnforceCore calls can't race
        // and lose increments; see _trialStateGate's comment.
        string trialPath = _trialFilePathOverride.Value ?? TrialState.DefaultPath;
        lock (_trialStateGate)
        {
            var trial = TrialState.Load(trialPath);
            var now = DateTimeOffset.UtcNow;
            if (trial.IsExpired(now))
            {
                int daysElapsed = (int)(now - trial.StartedAt).TotalDays;
                throw new LicenseRequiredException(
                    $"Free trial for AiDotNet.Tensors has expired ({trial.OperationsConsumed} operations / {daysElapsed} days). " +
                    "Set AIDOTNET_LICENSE_KEY or call PersistenceGuard.SetActiveLicenseKey to continue.",
                    trialDaysElapsed: daysElapsed,
                    operationsPerformed: trial.OperationsConsumed,
                    keyStatus: null,
                    requiredCapability: requiredCapability);
            }

            // Tick the counter and persist.
            trial.OperationsConsumed++;
            trial.Save(trialPath);
        }
    }

    private static AiDotNetTensorsLicenseKey? ResolveActiveKey()
    {
        if (_activeKey.Value is { } explicitKey) return explicitKey;

        // Env var: AIDOTNET_LICENSE_KEY (same name upstream uses, so
        // setting it once configures both packages).
        try
        {
            string? envKey = Environment.GetEnvironmentVariable("AIDOTNET_LICENSE_KEY");
            if (!string.IsNullOrWhiteSpace(envKey))
                return new AiDotNetTensorsLicenseKey(envKey!);
        }
        catch { /* SecurityException etc. — fall through */ }

        // File: ~/.aidotnet/license.key (also matches upstream).
        try
        {
            string home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
            string path = Path.Combine(home, ".aidotnet", "license.key");
            if (File.Exists(path))
            {
                string content = File.ReadAllText(path).Trim();
                if (!string.IsNullOrWhiteSpace(content))
                    return new AiDotNetTensorsLicenseKey(content);
            }
        }
        catch { /* IOException / SecurityException — fall through */ }

        return null;
    }

    /// <summary>
    /// Test hook — clears the validator cache and per-key
    /// pending-consumed markers so a test can configure a new key
    /// and have the next call hit the validator fresh rather than
    /// seeing the previous test's cached result or its consumed
    /// pending allowance.
    /// </summary>
    internal static void ClearValidatorCacheForTesting()
    {
        _validators.Clear();
        _pendingConsumed.Clear();
    }
}
