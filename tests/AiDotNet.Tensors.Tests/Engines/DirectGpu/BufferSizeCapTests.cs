// Copyright (c) AiDotNet. All rights reserved.
// Issue #285 — buffer size cap + chunked fallback test coverage.
// Pure-logic tests; integration tests gated on real GPU availability live elsewhere.

#nullable disable

using System;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Several tests in this class mutate
/// <see cref="GpuFallbackOptionsHolder.Current"/>, which is process-wide.
/// Without the non-parallel collection an unrelated test running on
/// another thread could observe the mutated value, producing flaky
/// failures. The xUnit collection definition below disables parallel
/// execution for this class so the holder mutation is safe.
/// </summary>
[CollectionDefinition("BufferSizeCapTests", DisableParallelization = true)]
public class BufferSizeCapTestsCollection { }

[Collection("BufferSizeCapTests")]
public class BufferSizeCapTests
{
    // ───────────────────────────────────────────────────────────────────
    // GpuBufferTooLargeException
    // ───────────────────────────────────────────────────────────────────

    [Fact]
    public void Exception_PopulatesAllFields()
    {
        var ex = new GpuBufferTooLargeException(
            backend: "OpenCL",
            requestedBytes: 2_000_000_000,
            deviceMaxAllocBytes: 1_073_741_824,
            deviceName: "AMD Radeon RX 5500 XT");

        Assert.Equal("OpenCL", ex.Backend);
        Assert.Equal(2_000_000_000, ex.RequestedBytes);
        Assert.Equal(1_073_741_824, ex.DeviceMaxAllocBytes);
        Assert.Equal("AMD Radeon RX 5500 XT", ex.DeviceName);
        Assert.IsAssignableFrom<InvalidOperationException>(ex);
    }

    [Fact]
    public void Exception_MessageIsHelpful()
    {
        var ex = new GpuBufferTooLargeException("OpenCL", 2_000_000_000, 1_073_741_824, "TestGpu");
        // Message must mention the backend, the bytes (in human form), and a fix hint.
        Assert.Contains("OpenCL", ex.Message);
        Assert.Contains("TestGpu", ex.Message);
        Assert.Contains("Reduce batch size", ex.Message);
        Assert.Contains("ConfigureGpuAcceleration", ex.Message);
    }

    [Fact]
    public void Exception_HandlesNullBackendAndDevice()
    {
        var ex = new GpuBufferTooLargeException(null, 1024, 512, null);
        Assert.NotNull(ex.Message);
        Assert.NotEmpty(ex.Message);
    }

    [Fact]
    public void Exception_HandlesUnknownCap()
    {
        // cap == 0 means "device cap unknown / not queried"
        var ex = new GpuBufferTooLargeException("CUDA", 1024, 0, "CudaDevice");
        Assert.Contains("unknown", ex.Message);
    }

    // ───────────────────────────────────────────────────────────────────
    // GpuBufferSizeGuard.EnsureFits
    // ───────────────────────────────────────────────────────────────────

    [Fact]
    public void EnsureFits_ThrowsWhenOverCap()
    {
        var ex = Assert.Throws<GpuBufferTooLargeException>(() =>
            GpuBufferSizeGuard.EnsureFits("OpenCL", requestedBytes: 2048, deviceCap:1024, deviceName: "X"));
        Assert.Equal(2048, ex.RequestedBytes);
        Assert.Equal(1024, ex.DeviceMaxAllocBytes);
    }

    [Fact]
    public void EnsureFits_AllowsAtCap()
    {
        // Boundary: requestedBytes == cap is allowed (only > throws).
        GpuBufferSizeGuard.EnsureFits("OpenCL", requestedBytes: 1024, deviceCap:1024, deviceName: "X");
    }

    [Fact]
    public void EnsureFits_AllowsUnderCap()
    {
        GpuBufferSizeGuard.EnsureFits("OpenCL", requestedBytes: 512, deviceCap:1024, deviceName: "X");
    }

    [Fact]
    public void EnsureFits_SkipsWhenCapIsZero()
    {
        // cap == 0 → skip validation (used when device cap query fails).
        GpuBufferSizeGuard.EnsureFits("OpenCL", requestedBytes: long.MaxValue, deviceCap:0, deviceName: "X");
    }

    [Fact]
    public void EnsureFits_SkipsWhenRequestedIsZero()
    {
        // requestedBytes <= 0 is non-sensical; guard skips and lets the
        // backend's native call surface its own error if any.
        GpuBufferSizeGuard.EnsureFits("OpenCL", requestedBytes: 0, deviceCap:1024, deviceName: "X");
        GpuBufferSizeGuard.EnsureFits("OpenCL", requestedBytes: -1, deviceCap:1024, deviceName: "X");
    }

    // ───────────────────────────────────────────────────────────────────
    // GpuFallbackOptions defaults
    // ───────────────────────────────────────────────────────────────────

    [Fact]
    public void Options_Default_UsesIndustryStandardDefaults()
    {
        var opts = GpuFallbackOptions.Default;
        // All three knobs null on the bare default instance.
        Assert.Null(opts.MaxBufferBytes);
        Assert.Null(opts.ChunkingPolicy);
        Assert.Null(opts.MaxChunkCount);

        // Effective values resolve to the documented defaults.
        Assert.Equal(2048L, InvokeEffectiveMaxBufferBytes(opts, deviceCap: 2048));
        Assert.Equal(GpuChunkingPolicy.AutoChunk, InvokeEffectiveChunkingPolicy(opts));
        Assert.Equal(64, InvokeEffectiveMaxChunkCount(opts));
    }

    [Fact]
    public void Options_OverrideOnePropertyAtATime()
    {
        var opts = new GpuFallbackOptions
        {
            MaxBufferBytes = 512_000_000,
            ChunkingPolicy = GpuChunkingPolicy.NeverChunk_FailFast,
            MaxChunkCount = 8,
        };
        Assert.Equal(512_000_000, InvokeEffectiveMaxBufferBytes(opts, deviceCap: 1_000_000_000));
        Assert.Equal(GpuChunkingPolicy.NeverChunk_FailFast, InvokeEffectiveChunkingPolicy(opts));
        Assert.Equal(8, InvokeEffectiveMaxChunkCount(opts));
    }

    [Fact]
    public void Holder_DefaultIsNeverNull()
    {
        var prev = GpuFallbackOptionsHolder.Current;
        try
        {
            GpuFallbackOptionsHolder.Current = null!; // forced null assignment
            Assert.NotNull(GpuFallbackOptionsHolder.Current);
            Assert.Same(GpuFallbackOptions.Default, GpuFallbackOptionsHolder.Current);
        }
        finally
        {
            GpuFallbackOptionsHolder.Current = prev;
        }
    }

    [Fact]
    public void EnsureFits_AppliesUserOverride_LowerCapWins()
    {
        // Regression for review-comment #9: GpuFallbackOptions.MaxBufferBytes
        // override must actually be applied by the size guard. Set the user
        // override to a value LOWER than the device cap and confirm the guard
        // throws when requestedBytes > userOverride but <= deviceCap.
        var prev = GpuFallbackOptionsHolder.Current;
        try
        {
            GpuFallbackOptionsHolder.Current = new GpuFallbackOptions
            {
                MaxBufferBytes = 1024, // user-override cap
            };
            // Device cap = 4096 (large), user override = 1024 → effective cap = 1024.
            // Request 2048 → should throw.
            var ex = Assert.Throws<GpuBufferTooLargeException>(() =>
                GpuBufferSizeGuard.EnsureFits("OpenCL", requestedBytes: 2048, deviceCap:4096, deviceName: "X"));
            Assert.Equal(1024, ex.DeviceMaxAllocBytes); // exception reports the EFFECTIVE cap
            Assert.Equal(2048, ex.RequestedBytes);

            // Same device cap, request 512 → fits under user override.
            GpuBufferSizeGuard.EnsureFits("OpenCL", requestedBytes: 512, deviceCap:4096, deviceName: "X");
        }
        finally
        {
            GpuFallbackOptionsHolder.Current = prev;
        }
    }

    [Fact]
    public void EnsureFits_DeviceCapLowerThanUserOverride_DeviceWins()
    {
        // When user override > device cap, the device cap is the hard limit
        // (we can't allocate past it). Effective cap = min(device, user).
        var prev = GpuFallbackOptionsHolder.Current;
        try
        {
            GpuFallbackOptionsHolder.Current = new GpuFallbackOptions
            {
                MaxBufferBytes = 100_000_000_000, // user wants huge cap
            };
            // Device cap = 1024 wins.
            var ex = Assert.Throws<GpuBufferTooLargeException>(() =>
                GpuBufferSizeGuard.EnsureFits("OpenCL", requestedBytes: 2048, deviceCap:1024, deviceName: "X"));
            Assert.Equal(1024, ex.DeviceMaxAllocBytes);
        }
        finally
        {
            GpuFallbackOptionsHolder.Current = prev;
        }
    }

    [Fact]
    public void EnsureFits_UserOverride_TrustedWhenDeviceCapUnknown()
    {
        // When the device cap query failed (cap == 0) but user supplied an
        // override, the user value is used as the cap.
        var prev = GpuFallbackOptionsHolder.Current;
        try
        {
            GpuFallbackOptionsHolder.Current = new GpuFallbackOptions
            {
                MaxBufferBytes = 256,
            };
            var ex = Assert.Throws<GpuBufferTooLargeException>(() =>
                GpuBufferSizeGuard.EnsureFits("OpenCL", requestedBytes: 1024, deviceCap:0, deviceName: "X"));
            Assert.Equal(256, ex.DeviceMaxAllocBytes);
        }
        finally
        {
            GpuFallbackOptionsHolder.Current = prev;
        }
    }

    [Fact]
    public void Holder_RoundTripsCustomOptions()
    {
        var prev = GpuFallbackOptionsHolder.Current;
        try
        {
            var custom = new GpuFallbackOptions { MaxChunkCount = 16 };
            GpuFallbackOptionsHolder.Current = custom;
            Assert.Same(custom, GpuFallbackOptionsHolder.Current);
        }
        finally
        {
            GpuFallbackOptionsHolder.Current = prev;
        }
    }

    // Thin wrappers around the internal accessors so the call sites read
    // naturally. The test assembly has InternalsVisibleTo so we can call
    // these directly — no reflection (and no coverage-blind spots).
    private static long InvokeEffectiveMaxBufferBytes(GpuFallbackOptions opts, long deviceCap)
        => opts.EffectiveMaxBufferBytes(deviceCap);

    private static GpuChunkingPolicy InvokeEffectiveChunkingPolicy(GpuFallbackOptions opts)
        => opts.EffectiveChunkingPolicy;

    private static int InvokeEffectiveMaxChunkCount(GpuFallbackOptions opts)
        => opts.EffectiveMaxChunkCount;

    // ───────────────────────────────────────────────────────────────────
    // Smoke test: every backend now exposes MaxBufferAllocBytes via the
    // IDirectGpuBackend interface (issue #285 pattern propagation).
    // ───────────────────────────────────────────────────────────────────

    [Fact]
    public void IDirectGpuBackend_ExposesMaxBufferAllocBytes()
    {
        var prop = typeof(IDirectGpuBackend).GetProperty("MaxBufferAllocBytes");
        Assert.NotNull(prop);
        Assert.Equal(typeof(long), prop!.PropertyType);
        Assert.True(prop.CanRead);
    }

    [Fact]
    public void AllBackendImplementations_OverrideMaxBufferAllocBytes()
    {
        // Every concrete IDirectGpuBackend in the assembly must implement
        // MaxBufferAllocBytes (issue #285 — without it, the size guard can't
        // tell what's safe to allocate). Reflection sweep ensures the
        // property is present on every backend, including ones added later.
        var assembly = typeof(IDirectGpuBackend).Assembly;
        var concreteBackends = assembly.GetTypes()
            .Where(t => !t.IsAbstract && !t.IsInterface
                        && typeof(IDirectGpuBackend).IsAssignableFrom(t))
            .ToList();
        Assert.NotEmpty(concreteBackends);
        foreach (var backendType in concreteBackends)
        {
            var prop = backendType.GetProperty("MaxBufferAllocBytes",
                System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Instance);
            Assert.True(prop is not null,
                $"{backendType.FullName} does not expose MaxBufferAllocBytes (issue #285).");
        }
    }
}
