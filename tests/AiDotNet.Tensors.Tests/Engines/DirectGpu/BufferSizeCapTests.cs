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
            InvokeEnsureFits("OpenCL", requestedBytes: 2048, cap: 1024, deviceName: "X"));
        Assert.Equal(2048, ex.RequestedBytes);
        Assert.Equal(1024, ex.DeviceMaxAllocBytes);
    }

    [Fact]
    public void EnsureFits_AllowsAtCap()
    {
        // Boundary: requestedBytes == cap is allowed (only > throws).
        InvokeEnsureFits("OpenCL", requestedBytes: 1024, cap: 1024, deviceName: "X");
    }

    [Fact]
    public void EnsureFits_AllowsUnderCap()
    {
        InvokeEnsureFits("OpenCL", requestedBytes: 512, cap: 1024, deviceName: "X");
    }

    [Fact]
    public void EnsureFits_SkipsWhenCapIsZero()
    {
        // cap == 0 → skip validation (used when device cap query fails).
        InvokeEnsureFits("OpenCL", requestedBytes: long.MaxValue, cap: 0, deviceName: "X");
    }

    [Fact]
    public void EnsureFits_SkipsWhenRequestedIsZero()
    {
        // requestedBytes <= 0 is non-sensical; guard skips and lets the
        // backend's native call surface its own error if any.
        InvokeEnsureFits("OpenCL", requestedBytes: 0, cap: 1024, deviceName: "X");
        InvokeEnsureFits("OpenCL", requestedBytes: -1, cap: 1024, deviceName: "X");
    }

    private static void InvokeEnsureFits(string backend, long requestedBytes, long cap, string deviceName)
    {
        // GpuBufferSizeGuard is internal; access via reflection to test from
        // the test assembly without needing a public surface.
        var type = typeof(GpuBufferTooLargeException).Assembly
            .GetType("AiDotNet.Tensors.Engines.DirectGpu.GpuBufferSizeGuard")!;
        var method = type.GetMethod("EnsureFits",
            System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Static)!;
        try
        {
            method.Invoke(null, new object[] { backend, requestedBytes, cap, deviceName });
        }
        catch (System.Reflection.TargetInvocationException tie)
            when (tie.InnerException is not null)
        {
            // Unwrap so callers see the original exception.
            throw tie.InnerException;
        }
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

    private static long InvokeEffectiveMaxBufferBytes(GpuFallbackOptions opts, long deviceCap)
    {
        var m = typeof(GpuFallbackOptions).GetMethod("EffectiveMaxBufferBytes",
            System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)!;
        return (long)m.Invoke(opts, new object[] { deviceCap })!;
    }

    private static GpuChunkingPolicy InvokeEffectiveChunkingPolicy(GpuFallbackOptions opts)
    {
        var p = typeof(GpuFallbackOptions).GetProperty("EffectiveChunkingPolicy",
            System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)!;
        return (GpuChunkingPolicy)p.GetValue(opts)!;
    }

    private static int InvokeEffectiveMaxChunkCount(GpuFallbackOptions opts)
    {
        var p = typeof(GpuFallbackOptions).GetProperty("EffectiveMaxChunkCount",
            System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)!;
        return (int)p.GetValue(opts)!;
    }

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
