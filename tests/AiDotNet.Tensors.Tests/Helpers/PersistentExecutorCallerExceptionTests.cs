using System;
using System.Threading;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Helpers;

/// <summary>
/// A plain dispatch whose throwing chunk belongs to the calling thread must still run every other chunk exactly once
/// before Execute rethrows. It used to skip the finish (wait for the workers, run what they had not started) whenever
/// the caller's own set threw, so it rethrew while a worker's chunks were unstarted or still running.
/// </summary>
[Collection("BlasManaged-Stats-Serial")]
public sealed class PersistentExecutorCallerExceptionTests
{
    [Fact]
    public void CallerChunkThrows_EveryOtherChunkRunsBeforeExecuteReturns()
    {
        int saved = CpuParallelSettings.MaxDegreeOfParallelism;
        try
        {
            // Two participants: the caller owns the even chunks, the one worker the odd ones.
            CpuParallelSettings.MaxDegreeOfParallelism = 2;
            const int numChunks = 8;
            int[] visits = new int[numChunks];
            var thrown = Assert.ThrowsAny<Exception>(() =>
                PersistentParallelExecutor.Instance.Execute(numChunks, (long)PersistentParallelExecutor.DefaultSerialGrainSize * 4, c =>
                {
                    if (c % 2 == 1) Thread.Sleep(20);   // the worker's chunks are still running when the caller's throws
                    Interlocked.Increment(ref visits[c]);
                    if (c == 0) throw new InvalidOperationException("planted-exception-from-caller-chunk-0");
                }));
            Assert.Contains("planted-exception-from-caller-chunk-0", thrown.Message);
            for (int c = 0; c < numChunks; c++)
                Assert.True(Volatile.Read(ref visits[c]) == 1, $"chunk {c} ran {visits[c]} times before Execute returned");
        }
        finally
        {
            CpuParallelSettings.MaxDegreeOfParallelism = saved;
        }
    }
}