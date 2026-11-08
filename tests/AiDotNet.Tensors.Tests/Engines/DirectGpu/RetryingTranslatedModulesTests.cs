// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Tensors.Engines.DirectGpu.WebGpu;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// The WGSL module sets (WebGpuBackend's and the Dawn fixture's) are translated on first use. A translation that times out
/// must not be cached: one regex timeout during a long pause otherwise failed every later request in the process (all
/// 188 Dawn parity tests in one CUDA run).
/// </summary>
public class RetryingTranslatedModulesTests
{
    private static RegexMatchTimeoutException Timeout() => new("input", "pattern", TimeSpan.FromSeconds(60));

    [Fact]
    public void A_translation_that_timed_out_runs_again_on_the_next_request()
    {
        int calls = 0;
        var modules = CudaDialectToWgsl.RetryingModules(() => ++calls == 1 ? throw Timeout() : "modules");

        Assert.Throws<RegexMatchTimeoutException>(() => CudaDialectToWgsl.Modules(modules));
        Assert.Equal("modules", CudaDialectToWgsl.Modules(modules));
        Assert.Equal(2, calls);
    }

    [Fact]
    public async Task A_translation_that_timed_out_after_another_published_uses_the_published_modules()
    {
        int calls = 0;
        using var firstStarted = new ManualResetEventSlim();
        using var secondPublished = new ManualResetEventSlim();
        var modules = CudaDialectToWgsl.RetryingModules(() =>
        {
            if (Interlocked.Increment(ref calls) == 1)
            {
                firstStarted.Set();
                Assert.True(secondPublished.Wait(TimeSpan.FromSeconds(30)), "the second translation never published");
                throw Timeout();
            }
            return "published";
        });

        var first = Task.Run(() => CudaDialectToWgsl.Modules(modules));
        Assert.True(firstStarted.Wait(TimeSpan.FromSeconds(30)), "the first translation never started");
        Assert.Equal("published", CudaDialectToWgsl.Modules(modules));
        secondPublished.Set();

        Assert.Equal("published", await first);
    }

    [Fact]
    public void A_timeout_with_nothing_published_propagates()
    {
        var modules = CudaDialectToWgsl.RetryingModules<string>(() => throw Timeout());

        Assert.Throws<RegexMatchTimeoutException>(() => CudaDialectToWgsl.Modules(modules));
        Assert.False(modules.IsValueCreated);
    }
}
