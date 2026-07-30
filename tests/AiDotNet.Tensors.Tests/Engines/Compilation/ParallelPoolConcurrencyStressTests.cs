using System;
using System.Buffers;
using System.Collections.Concurrent;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// Reproduces the load-dependent crash the full suite hit after the pool gained per-call max-DOP,
/// Execute&lt;TLocal&gt; and flat-Invoke. Hammers the singleton PersistentParallelExecutor from many
/// concurrent threads with a MIX of plain dispatches, thread-local (rented per-participant buffer)
/// dispatches, and flat Invoke — the exact shapes the migrated SpMM / im2col / backward sites use —
/// and detects (a) result corruption on plain output, and (b) buffer ALIASING/use-after-free on the
/// thread-local path (another participant writing a buffer that should be exclusively ours), which is
/// what surfaces as an access-violation crash inside the unsafe GEMM/im2col kernels.
/// </summary>
[Collection("CompilationGlobalState")]
public class ParallelPoolConcurrencyStressTests
{
    [Fact]
    public void MixedDispatch_ManyConcurrentCallers_NoCorruptionNoAliasing()
    {
        int threads = Math.Max(8, Environment.ProcessorCount);
        const int iterationsPerThread = 4000;
        const int chunks = 24;
        const int bufLen = 512;
        var longPool = ArrayPool<long>.Shared;
        var errors = new ConcurrentQueue<string>();

        var workers = new Task[threads];
        for (int t = 0; t < threads; t++)
        {
            int tid = t;
            workers[t] = Task.Run(() =>
            {
                for (int it = 0; it < iterationsPerThread && errors.IsEmpty; it++)
                {
                    switch (it % 3)
                    {
                        case 0:
                        {
                            // Plain dispatch: each chunk writes its own output slot.
                            var output = new int[chunks];
                            CpuParallelSettings.LightweightParallel(chunks, c => output[c] = unchecked(c * 2654435761u).GetHashCode());
                            for (int c = 0; c < chunks; c++)
                            {
                                int expected = unchecked(c * 2654435761u).GetHashCode();
                                if (output[c] != expected) errors.Enqueue($"PLAIN t{tid} it{it} c{c}={output[c]} exp{expected}");
                            }
                            break;
                        }
                        case 1:
                        {
                            // Thread-local dispatch: each PARTICIPANT rents ONE buffer reused across its
                            // strided chunks. Write a unique marker to the whole buffer, widen the window,
                            // then verify it's untouched — catches a buffer handed to two participants or
                            // returned-then-reused while still live.
                            CpuParallelSettings.LightweightParallel<long[]>(
                                chunks, 0,
                                localInit: () => longPool.Rent(bufLen),
                                body: (c, buf) =>
                                {
                                    long marker = ((long)(tid + 1) << 40) ^ ((long)(it + 1) << 12) ^ (c + 1);
                                    for (int j = 0; j < bufLen; j++) buf[j] = marker + j;
                                    Thread.SpinWait(40); // widen the aliasing race window
                                    for (int j = 0; j < bufLen; j++)
                                        if (buf[j] != marker + j)
                                        {
                                            errors.Enqueue($"ALIAS t{tid} it{it} c{c} j{j} got={buf[j]} exp={marker + j}");
                                            break;
                                        }
                                },
                                localFinally: buf => longPool.Return(buf));
                            break;
                        }
                        default:
                        {
                            // Flat Invoke: two independent tasks writing distinct flags.
                            int a = 0, b = 0;
                            CpuParallelSettings.LightweightInvoke(() => a = tid + 1, () => b = it + 1);
                            if (a != tid + 1 || b != it + 1) errors.Enqueue($"INVOKE t{tid} it{it} a={a} b={b}");
                            break;
                        }
                    }
                }
            });
        }

        Task.WaitAll(workers);
        Assert.True(errors.IsEmpty, "Pool concurrency corruption/aliasing:\n  " + string.Join("\n  ", errors.Take(8)));
    }
}
