using System;
using System.Collections.Concurrent;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Tensors.Engines.DirectGpu;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// The fallback report is read while other threads record fallbacks (parallel tests, multi-threaded engines). Reading
/// it must never throw. It works on a private dictionary, so the process-wide probe that other tests assert on is
/// untouched.
/// </summary>
public class GpuLaunchProbeFallbacksTests
{
    [Fact]
    public void Reading_the_fallback_report_while_another_thread_records_never_throws()
    {
        var fallbacks = new ConcurrentDictionary<string, int>();
        using var stop = new CancellationTokenSource();
        var writer = Task.Run(() =>
        {
            for (int i = 0; !stop.IsCancellationRequested; i++)
            {
                fallbacks.AddOrUpdate("op" + (i % 4096), 1, static (_, count) => count + 1);
                if (i % 4096 == 4095) fallbacks.Clear();
            }
        });

        Exception? failure = null;
        try
        {
            for (int read = 0; read < 20_000 && failure is null; read++)
            {
                try
                {
                    var report = GpuLaunchProbe.FormatFallbacks(fallbacks);
                    foreach (var line in report) Assert.Contains("x op", line);
                }
                catch (Exception ex) when (ex is not Xunit.Sdk.XunitException)
                {
                    failure = ex;
                }
            }
        }
        finally
        {
            stop.Cancel();
            writer.Wait();
        }

        Assert.Null(failure);
    }

    [Fact]
    public void The_fallback_report_is_ordered_by_op_and_counts_each()
    {
        var fallbacks = new ConcurrentDictionary<string, int>();
        fallbacks["b"] = 2;
        fallbacks["a"] = 1;
        Assert.Equal(new[] { "1x a", "2x b" }, GpuLaunchProbe.FormatFallbacks(fallbacks));
    }
}
