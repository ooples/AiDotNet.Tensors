// Copyright (c) AiDotNet. All rights reserved.
// Tests for tensor-layer license enforcement.

#nullable disable

using System;
using System.IO;
using AiDotNet.Tensors.Licensing;
using AiDotNet.Tensors.NumericOperations;
using Xunit;

namespace AiDotNet.Tensors.Tests.Licensing;

/// <summary>
/// Tests run serially via xUnit's collection mechanism — the guard
/// uses process-wide AsyncLocal/static state and parallel test
/// execution would interleave key/trial-path overrides.
/// </summary>
[Collection("PersistenceGuard")]
public class PersistenceGuardTests
{
    /// <summary>
    /// Test fixture that provides an isolated trial-file path per
    /// test, ensuring no test leaks into <c>~/.aidotnet/tensors-trial.json</c>
    /// or affects another test's counter.
    /// </summary>
    private static IDisposable IsolatedTrial(out string path)
    {
        path = Path.Combine(Path.GetTempPath(), "aidotnet-test-trial-" + Guid.NewGuid().ToString("N") + ".json");
        var override_ = PersistenceGuard.SetTestTrialFilePathOverride(path);
        return new TrialPathCleanup(override_, path);
    }

    private sealed class TrialPathCleanup : IDisposable
    {
        private readonly IDisposable _scope;
        private readonly string _path;
        public TrialPathCleanup(IDisposable scope, string path) { _scope = scope; _path = path; }
        public void Dispose()
        {
            _scope.Dispose();
            try { if (File.Exists(_path)) File.Delete(_path); } catch { }
        }
    }

    [Fact]
    public void EnforceBeforeLoad_NoKey_FreshTrial_Allows()
    {
        using var trial = IsolatedTrial(out _);
        // No exception — fresh trial budget allows the call.
        PersistenceGuard.EnforceBeforeLoad();
    }

    [Fact]
    public void EnforceBeforeLoad_NoKey_TrialPersistsAcrossCalls()
    {
        // After 3 calls, the on-disk trial file should reflect 3
        // operations consumed.
        using var trial = IsolatedTrial(out var path);
        PersistenceGuard.EnforceBeforeLoad();
        PersistenceGuard.EnforceBeforeLoad();
        PersistenceGuard.EnforceBeforeLoad();
        var state = TrialState.Load(path);
        Assert.Equal(3, state.OperationsConsumed);
    }

    [Fact]
    public void EnforceBeforeLoad_NoKey_TrialBudgetExhausted_Throws()
    {
        // Pre-populate trial state with the operation cap reached.
        using var trial = IsolatedTrial(out var path);
        var preExhausted = new TrialState
        {
            StartedAt = DateTimeOffset.UtcNow,
            OperationsConsumed = TrialState.DefaultMaxOperations,
        };
        preExhausted.Save(path);

        var ex = Assert.Throws<LicenseRequiredException>(() => PersistenceGuard.EnforceBeforeLoad());
        Assert.NotNull(ex.OperationsPerformed);
        Assert.Equal(TrialState.DefaultMaxOperations, ex.OperationsPerformed);
    }

    [Fact]
    public void EnforceBeforeLoad_NoKey_TrialDaysExhausted_Throws()
    {
        // Pre-populate trial state with start time 31 days ago (over
        // the 30-day window).
        using var trial = IsolatedTrial(out var path);
        var preExhausted = new TrialState
        {
            StartedAt = DateTimeOffset.UtcNow - TimeSpan.FromDays(31),
            OperationsConsumed = 1,
        };
        preExhausted.Save(path);

        var ex = Assert.Throws<LicenseRequiredException>(() => PersistenceGuard.EnforceBeforeLoad());
        Assert.NotNull(ex.TrialDaysElapsed);
        Assert.True(ex.TrialDaysElapsed >= 30);
    }

    [Fact]
    public void InternalOperation_SuppressesEnforcement()
    {
        // Force trial expiration so a top-level call would throw.
        // The InternalOperation scope must suppress the throw and the
        // counter must NOT tick for the suppressed call.
        using var trial = IsolatedTrial(out var path);
        var preExhausted = new TrialState
        {
            StartedAt = DateTimeOffset.UtcNow,
            OperationsConsumed = TrialState.DefaultMaxOperations,
        };
        preExhausted.Save(path);

        using (PersistenceGuard.InternalOperation())
        {
            // No throw despite the trial being exhausted.
            PersistenceGuard.EnforceBeforeLoad();
            PersistenceGuard.EnforceBeforeSave();
        }

        // Counter unchanged — InternalOperation didn't tick it.
        var state = TrialState.Load(path);
        Assert.Equal(TrialState.DefaultMaxOperations, state.OperationsConsumed);

        // Outside the scope, enforcement fires again.
        Assert.Throws<LicenseRequiredException>(() => PersistenceGuard.EnforceBeforeLoad());
    }

    [Fact]
    public void InternalOperation_Nested_OnlyOutermostReEnables()
    {
        using var trial = IsolatedTrial(out var path);
        var preExhausted = new TrialState
        {
            StartedAt = DateTimeOffset.UtcNow,
            OperationsConsumed = TrialState.DefaultMaxOperations,
        };
        preExhausted.Save(path);

        using (PersistenceGuard.InternalOperation())
        {
            using (PersistenceGuard.InternalOperation())
            {
                PersistenceGuard.EnforceBeforeLoad(); // suppressed
            }
            // Still inside the outer scope — still suppressed.
            PersistenceGuard.EnforceBeforeLoad();
        }
        // Outermost disposed → enforcement re-enabled.
        Assert.Throws<LicenseRequiredException>(() => PersistenceGuard.EnforceBeforeLoad());
    }

    [Fact]
    public async System.Threading.Tasks.Task InternalOperation_FlowsAcrossAwait()
    {
        // AsyncLocal must carry the suppression across an await boundary
        // so a continuation on a different thread-pool thread still sees
        // the scope.
        using var trial = IsolatedTrial(out var path);
        var preExhausted = new TrialState
        {
            StartedAt = DateTimeOffset.UtcNow,
            OperationsConsumed = TrialState.DefaultMaxOperations,
        };
        preExhausted.Save(path);

        using (PersistenceGuard.InternalOperation())
        {
            await System.Threading.Tasks.Task.Yield();
            // Same logical context after the yield — must still be suppressed.
            PersistenceGuard.EnforceBeforeLoad();
        }
    }

    [Fact]
    public void GgufReader_Read_FiresEnforcement()
    {
        // End-to-end: GgufReader.Read calls into the guard. Trial
        // exhausted → reading throws LicenseRequiredException
        // BEFORE the stream is consumed, so a corrupt-stream-vs-
        // license-error test ordering can't mask the gate.
        using var trial = IsolatedTrial(out var path);
        var preExhausted = new TrialState
        {
            StartedAt = DateTimeOffset.UtcNow,
            OperationsConsumed = TrialState.DefaultMaxOperations,
        };
        preExhausted.Save(path);

        using var ms = new MemoryStream(new byte[] { 0xFF, 0xFF, 0xFF, 0xFF });
        Assert.Throws<LicenseRequiredException>(() => GgufReader.Read(ms));
    }

    [Fact]
    public void SetActiveLicenseKey_ScopeRestoresPriorKey()
    {
        var inner = new AiDotNetTensorsLicenseKey("aidn.test.scope.AAAAAAAA");
        using (PersistenceGuard.SetActiveLicenseKey(inner))
        {
            // Inner key active.
            // We can't easily verify externally without a server stub,
            // but the scope-restore semantic is the contract under test.
        }
        // After dispose, the previous (null) key is restored — a fresh
        // EnforceBeforeLoad should now hit the trial path.
        using var trial = IsolatedTrial(out _);
        PersistenceGuard.EnforceBeforeLoad();  // no exception, trial allowed
    }

    [Fact]
    public void Capabilities_TensorsLoad_AcceptedBy_HasCapability()
    {
        // Direct test of the capability convention — server returns
        // "tensors:load" and "tensors:save" for a Tensors-tier key.
        var result = new LicenseValidationResult(
            LicenseKeyStatus.Active,
            tier: "tensors-only",
            expiresAt: null,
            validatedAt: DateTimeOffset.UtcNow,
            message: null,
            capabilities: new[] { "tensors:save", "tensors:load" });
        Assert.True(result.HasCapability("tensors:save"));
        Assert.True(result.HasCapability("tensors:load"));
        Assert.False(result.HasCapability("model:save"));
    }

    [Fact]
    public void LicenseValidator_KeyFormatValidation()
    {
        // Signed format
        Assert.True(LicenseValidator.ValidateKeyFormat("aidn.abc123.def456"));
        // Server-validated format (4+ segments, hex tail >= 8 chars)
        Assert.True(LicenseValidator.ValidateKeyFormat("AIDN-FOO-BAR-deadbeef12"));
        // Garbage
        Assert.False(LicenseValidator.ValidateKeyFormat("not-a-valid-key"));
        Assert.False(LicenseValidator.ValidateKeyFormat(""));
        Assert.False(LicenseValidator.ValidateKeyFormat("aidn..."));  // empty id+sig
    }

    [Fact]
    public void LicenseValidator_ParseResponse_HandlesActiveWithCapabilities()
    {
        string json = """
        {
          "status": "active",
          "tier": "pro",
          "capabilities": ["tensors:save", "tensors:load", "model:save", "model:load"],
          "message": "OK"
        }
        """;
        var r = LicenseValidator.ParseResponse(json, 200);
        Assert.Equal(LicenseKeyStatus.Active, r.Status);
        Assert.Equal("pro", r.Tier);
        Assert.True(r.HasCapability("tensors:save"));
        Assert.True(r.HasCapability("model:save"));
    }

    [Fact]
    public void LicenseValidator_ParseResponse_HandlesNon2xx()
    {
        var r = LicenseValidator.ParseResponse("{}", 500);
        Assert.Equal(LicenseKeyStatus.Invalid, r.Status);
        Assert.Contains("HTTP 500", r.Message);
    }

    [Fact]
    public void LicenseValidator_ParseResponse_HandlesMalformedJson()
    {
        var r = LicenseValidator.ParseResponse("not json", 200);
        Assert.Equal(LicenseKeyStatus.Invalid, r.Status);
    }
}

/// <summary>
/// xUnit test collection — forces all tests in
/// <see cref="PersistenceGuardTests"/> to run sequentially. Each test
/// configures process-wide AsyncLocal / static state, and parallel
/// execution would interleave overrides and produce flaky failures.
/// </summary>
[CollectionDefinition("PersistenceGuard", DisableParallelization = true)]
public class PersistenceGuardCollection { }
