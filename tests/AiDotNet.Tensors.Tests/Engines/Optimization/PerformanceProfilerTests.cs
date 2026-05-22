using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Optimization;
using AiDotNet.Tensors.Engines.Profiling;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Optimization;

[Collection("PerformanceProfilerInstance")]
public class PerformanceProfilerTests
{
    [Fact]
    public void Profile_WhenEnabled_RecordsStats()
    {
        var profiler = PerformanceProfiler.Instance;
        string operationName = $"op-{Guid.NewGuid():N}";
        bool wasEnabled = profiler.Enabled;

        profiler.Clear();
        profiler.Enabled = true;

        try
        {
            using (profiler.Profile(operationName))
            {
                _ = 1 + 1;
            }

            var stats = profiler.GetStats(operationName);
            Assert.NotNull(stats);
            Assert.Equal(operationName, stats!.OperationName);
            Assert.Equal(1, stats.CallCount);
            Assert.True(stats.TotalTicks > 0);
        }
        finally
        {
            profiler.Enabled = wasEnabled;
            profiler.Clear();
        }
    }

    [Fact]
    public void Profile_WhenDisabled_ReturnsEmptyDisposable()
    {
        var profiler = PerformanceProfiler.Instance;
        string operationName = $"op-disabled-{Guid.NewGuid():N}";
        bool wasEnabled = profiler.Enabled;

        profiler.Clear();
        profiler.Enabled = false;

        try
        {
            using (profiler.Profile(operationName))
            {
                _ = 1 + 1;
            }

            Assert.Null(profiler.GetStats(operationName));
        }
        finally
        {
            profiler.Enabled = wasEnabled;
            profiler.Clear();
        }
    }

    [Fact]
    public void CpuMatMul_RecordsStats_WhenLegacyProfilerEnabled()
    {
        // Issue #363 regression: setting PerformanceProfiler.Instance.Enabled = true
        // must surface CPU op timings (the documented contract for
        // AiDotNet.Diagnostics.TensorsOperationProfile.Capture). Pre-fix the
        // CPU engine emitted ops only through Profiler.OpScope's ambient
        // ProfilerSession; the legacy singleton stayed empty no matter how
        // many CPU forward passes ran. The bridge in Profiler.OpScope routes
        // op timings into both surfaces when the legacy flag is on.
        var profiler = PerformanceProfiler.Instance;
        bool wasEnabled = profiler.Enabled;
        profiler.Clear();
        profiler.Enabled = true;

        try
        {
            var engine = new CpuEngine();
            var a = new Tensor<float>(new float[] { 1, 2, 3, 4 }, new[] { 2, 2 });
            var b = new Tensor<float>(new float[] { 5, 6, 7, 8 }, new[] { 2, 2 });

            // Run a small batch of CPU MatMul calls. Each one goes through
            // CpuEngine.TensorMatMul -> Profiler.OpScope("TensorMatMul"),
            // which post-#363 also feeds PerformanceProfiler.Instance.
            for (int i = 0; i < 5; i++)
                _ = engine.TensorMatMul(a, b);

            var stats = profiler.GetStats("TensorMatMul");
            Assert.NotNull(stats);
            Assert.Equal(5, stats!.CallCount);
            Assert.True(stats.TotalTicks > 0, "TotalTicks must be positive after 5 MatMul calls.");
        }
        finally
        {
            profiler.Enabled = wasEnabled;
            profiler.Clear();
        }
    }

    [Fact]
    public void CpuMatMul_StaysEmpty_WhenLegacyProfilerDisabled()
    {
        // Counterpart: when Enabled is false (default), CPU MatMul must NOT
        // tick the legacy aggregator. Confirms the bridge is gated, not
        // unconditional — overhead on the disabled hot path must stay zero.
        var profiler = PerformanceProfiler.Instance;
        bool wasEnabled = profiler.Enabled;
        profiler.Clear();
        profiler.Enabled = false;

        try
        {
            var engine = new CpuEngine();
            var a = new Tensor<float>(new float[] { 1, 2, 3, 4 }, new[] { 2, 2 });
            var b = new Tensor<float>(new float[] { 5, 6, 7, 8 }, new[] { 2, 2 });
            for (int i = 0; i < 5; i++)
                _ = engine.TensorMatMul(a, b);

            Assert.Null(profiler.GetStats("TensorMatMul"));
        }
        finally
        {
            profiler.Enabled = wasEnabled;
            profiler.Clear();
        }
    }
}

/// <summary>
/// xunit collection that serializes tests touching the global
/// <see cref="PerformanceProfiler.Instance"/> singleton's Enabled flag /
/// stats. Without this, two tests flipping <c>Enabled</c> in parallel
/// would race: one's "set true / run / assert" interleaves with another's
/// "set false / run / assert null" and both see the wrong state. The
/// fixture itself holds no state — it's just a serialization tag.
/// </summary>
[CollectionDefinition("PerformanceProfilerInstance", DisableParallelization = true)]
public sealed class PerformanceProfilerCollection { }
