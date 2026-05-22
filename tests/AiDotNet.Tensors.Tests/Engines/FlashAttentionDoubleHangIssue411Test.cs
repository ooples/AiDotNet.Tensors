// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Diagnostics;
using System.Threading.Tasks;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// Verification regression for ooples/AiDotNet.Tensors#411 — FlashAttention&lt;double&gt;
/// hangs at SD UNet attention shape <c>[B=1, heads=8, seqQ=seqK=1024, headDim=80]</c>
/// because OpenBLAS multi-threaded DGEMM oversubscribes against the outer parallel-for
/// over <c>batch * heads</c> (32-thread BLAS × 8-head outer = 256-thread oversubscription
/// on a 32-core host; pre-fix wall time was &gt;600 s, observed as a process hang).
///
/// <para>Fix landed in PR #410 commit 8bc2ab0f — <c>FlashAttentionDouble</c> now wraps
/// its parallel-for in <c>BlasProvider.ScopeOpenBlasThreads(1)</c> so each per-head GEMM
/// runs single-threaded, no oversubscription. Bench result after fix at this shape: ~30-65 ms.</para>
///
/// <para>This test pins the contract — a hard 30-second budget. If a future change ever
/// removes the OpenBLAS scope or re-introduces the oversubscription path, this test will
/// time out cleanly instead of letting CI hang.</para>
///
/// <para>CI-follow-up: marked with <c>[Collection("BlasGlobalState")]</c> so the test is
/// serialised with other tests that mutate the global OpenBLAS thread count
/// (DeterministicMode/DeterministicByDefault). Without this, an unrelated test running in
/// parallel could set OpenBLAS threads to a high value between this test's
/// <c>ScopeOpenBlasThreads(1)</c> acquisition and the parallel-for entry, defeating the
/// scope and re-introducing the oversubscription cliff. Originally not gated — fixed after
/// PR #426 hit the 30s timeout on Linux CI even though the production fix from #410 was
/// merged.</para>
/// </summary>
[Collection("BlasGlobalState")]
public class FlashAttentionDoubleHangIssue411Test
{
    private readonly ITestOutputHelper _output;

    public FlashAttentionDoubleHangIssue411Test(ITestOutputHelper output)
    {
        _output = output;
    }

    [Fact(Timeout = 30_000)]
    public async Task FlashAttentionDouble_SdUnetShape_CompletesUnderBudget()
    {
        // xUnit Fact(Timeout=...) needs an async signature so the runner can
        // cancel a hanging test cleanly. Body is synchronous; the await + Yield
        // is just to satisfy the async-method requirement.
        await Task.Yield();

        // Exact shape from issue #411's reproducer: SD UNet 32×32 attention level
        // (batch=1, heads=8, seq=1024, headDim=80).
        var engine = new CpuEngine();
        var rng = new Random(0);
        var q = new Tensor<double>(new[] { 1, 8, 1024, 80 });
        var k = new Tensor<double>(new[] { 1, 8, 1024, 80 });
        var v = new Tensor<double>(new[] { 1, 8, 1024, 80 });

        // Fill with small random values, mirroring the reproducer comment.
        for (int i = 0; i < q.Length; i++) q[i] = rng.NextDouble() * 0.1;
        for (int i = 0; i < k.Length; i++) k[i] = rng.NextDouble() * 0.1;
        for (int i = 0; i < v.Length; i++) v[i] = rng.NextDouble() * 0.1;

        // Warmup so OpenBLAS thread cache / JIT / pool effects don't pollute the timing.
        // Single warmup is enough — if the fix is intact the warmup itself completes in
        // ~50 ms (BLAS-backed dgemm). If the fix is missing, the warmup hangs and the
        // Timeout fires before we ever reach the measured loop.
        var resultWarm = engine.FlashAttention<double>(q, k, v, scale: null, isCausal: false, out _);
        Assert.NotNull(resultWarm);

        // Measure 3 iterations averaged. Even on a heavily-loaded CI runner this
        // should be well under 5 seconds per call (the issue's pre-fix observation
        // was >4 minutes per call).
        const int iters = 3;
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < iters; i++)
        {
            var _r = engine.FlashAttention<double>(q, k, v, scale: null, isCausal: false, out _);
        }
        sw.Stop();
        double msPerCall = sw.Elapsed.TotalMilliseconds / iters;
        _output.WriteLine($"FlashAttention<double> [1, 8, 1024, 80]: {msPerCall:F1} ms/call ({iters} iters)");

        // Hard ceiling: 5 seconds per call. Pre-fix this shape took >600 s
        // (process hang). Post-fix bench measured ~33 ms on a 32-core host.
        // 5 s gives generous headroom for slower CI runners while still catching
        // any regression that re-introduces the oversubscription cliff.
        Assert.True(msPerCall < 5000,
            $"FlashAttention<double> at SD UNet shape took {msPerCall:F1} ms/call — " +
            $"a regression of the OpenBLAS thread-scope fix from PR #410 (issue #411). " +
            $"Pre-fix this shape hung >600 s under 32-thread BLAS × 8-head outer parallel-for.");
    }
}
