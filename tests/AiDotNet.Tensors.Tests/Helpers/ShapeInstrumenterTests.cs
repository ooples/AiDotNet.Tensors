// Copyright (c) AiDotNet. All rights reserved.

using System;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Helpers;

/// <summary>
/// Issue #403 Phase A.1 — verifies ShapeInstrumenter correctly collects
/// every GEMM call dispatched through BlasProvider.ShapeLogHook, dedupes
/// by (M, N, K, transA, transB), and restores the prior hook on dispose.
/// </summary>
public class ShapeInstrumenterTests
{
    [Fact]
    public void Probe_aggregates_unique_shape_calls()
    {
        using var probe = new ShapeInstrumenter();

        // Fire the hook directly. The kernels are tested via the GEMM
        // call sites; this test isolates the probe's aggregation logic.
        BlasProvider.ShapeLogHook!.Invoke(32, 64, 128, false, false);
        BlasProvider.ShapeLogHook!.Invoke(32, 64, 128, false, false);
        BlasProvider.ShapeLogHook!.Invoke(32, 64, 128, false, false);
        BlasProvider.ShapeLogHook!.Invoke(64, 32, 128, false, false);
        BlasProvider.ShapeLogHook!.Invoke(32, 64, 128, true, false);

        Assert.Equal(5L, probe.TotalCalls);
        Assert.Equal(3, probe.UniqueShapes);

        var catalog = probe.GetCatalog();
        Assert.Equal(3, catalog.Count);

        // Descending count order: (32,64,128,N,N)=3 first, then ties broken
        // by descending FLOPs (both ties have identical FLOPs here, ordering
        // among them is implementation-defined, so just check count).
        Assert.Equal(3L, catalog[0].Calls);
        Assert.Equal(32, catalog[0].M);
        Assert.Equal(64, catalog[0].N);
        Assert.Equal(128, catalog[0].K);
        Assert.False(catalog[0].TransA);
    }

    [Fact]
    public void Probe_restores_previous_hook_on_dispose()
    {
        Assert.Null(BlasProvider.ShapeLogHook);

        int outerCount = 0;
        BlasProvider.ShapeLogHook = (_, _, _, _, _) => outerCount++;

        try
        {
            using (var probe = new ShapeInstrumenter())
            {
                // Inside the scope, the probe owns the hook — outer
                // counter must not advance.
                BlasProvider.ShapeLogHook!.Invoke(16, 16, 16, false, false);
                Assert.Equal(1L, probe.TotalCalls);
                Assert.Equal(0, outerCount);
            }

            // After dispose, the outer counter must take over again.
            BlasProvider.ShapeLogHook!.Invoke(8, 8, 8, false, false);
            Assert.Equal(1, outerCount);
        }
        finally
        {
            BlasProvider.ShapeLogHook = null;
        }
    }

    [Fact]
    public void Concurrent_probe_construction_throws()
    {
        using var first = new ShapeInstrumenter();
        Assert.Throws<InvalidOperationException>(() => new ShapeInstrumenter());
    }

    [Fact]
    public void Format_catalog_renders_markdown_table()
    {
        using var probe = new ShapeInstrumenter();
        BlasProvider.ShapeLogHook!.Invoke(10, 20, 30, false, false);
        BlasProvider.ShapeLogHook!.Invoke(10, 20, 30, false, false);
        BlasProvider.ShapeLogHook!.Invoke(40, 50, 60, true, true);

        string md = probe.FormatCatalog();

        Assert.Contains("Unique shapes: 2", md);
        Assert.Contains("Total calls: 3", md);
        Assert.Contains("| Calls |", md);
        // Spot-check that both shapes appear with their transpose flags.
        Assert.Contains("    10 |    20 |    30 |  N |  N |", md);
        Assert.Contains("    40 |    50 |    60 |  T |  T |", md);
    }
}
