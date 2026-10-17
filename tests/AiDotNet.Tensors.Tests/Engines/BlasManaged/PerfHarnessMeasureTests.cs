using System;
using System.Diagnostics;
using AiDotNet.Tensors.Tests.Engines.BlasManaged.Catalog;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

/// <summary>
/// The perf-bar harness alternates its two arms (#368), so a load change during a measurement shifts both arms
/// alike and leaves their ratio alone. Timing the arms one after the other measured the load as much as the kernels:
/// on a shared machine a catalog shape read 2.10x native that measures 1.17x.
/// </summary>
[Collection("BlasManaged-Perf-Serial")]
public class PerfHarnessMeasureTests
{
    private static readonly Shape Probe = new("probe", 1, 1, 1, false, false, DType.Single, 1, "test");

    private static void Spin(double microseconds)
    {
        var sw = Stopwatch.StartNew();
        while (sw.Elapsed.TotalMilliseconds * 1000 < microseconds) { }
    }

    [Fact]
    public void Ratio_IsTheArmsCostRatio_WhenTheLoadChangesMidMeasurement()
    {
        // The managed arm costs twice the native one. Partway through, everything gets three times slower, the way a
        // neighbouring process's load does. Sequential arms would see the native arm only under the heavier load.
        long calls = 0;
        double load = 1;
        void Tick() { if (++calls == 160) load = 3; }
        void Managed() { Tick(); Spin(20 * load); }
        void Native() { Tick(); Spin(10 * load); }

        var result = PerfHarness.Measure(Probe, "Single", 30, Managed, Native);

        Assert.True(load == 3, "the load never changed during the measurement; the test would be vacuous");
        Assert.InRange(result.RatioBmOverNative, 1.7, 2.3);
    }

    [Fact]
    public void WithoutNative_ReportsNoComparison()
    {
        var result = PerfHarness.Measure(Probe, "Single", 5, () => Spin(5), null);

        Assert.False(result.NativeAvailable);
        Assert.Equal(0, result.RatioBmOverNative);
        Assert.Equal(-1, result.NativeMedianMs);
        Assert.True(result.BlasManagedMedianMs > 0);
    }
}
