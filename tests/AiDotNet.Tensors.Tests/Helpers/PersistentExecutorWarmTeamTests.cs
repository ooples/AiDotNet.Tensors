// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using System.Diagnostics;
using System.Threading;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Helpers;

/// <summary>
/// Warm-spinning between dispatches: a worker spins only while the dispatches still wake it, and a hot loop's dispatches
/// give their warm workers the grace period instead of stealing their chunks at once.
/// </summary>
[Collection("CompilationGlobalState")]
public sealed class PersistentExecutorWarmTeamTests
{
    private static PersistentParallelExecutor Pool => PersistentParallelExecutor.Instance;

    // AIDOTNET_COOP_POOL=0 sends every dispatch to Parallel.For, so the persistent workers never run or warm-spin.
    private static bool PoolDisabled => Environment.GetEnvironmentVariable("AIDOTNET_COOP_POOL") == "0";

    // AIDOTNET_PPE_WARM_FULL_WIDTH=0 parks a team that, with the dispatching thread, fills the machine.
    private static bool FullWidthParks => Environment.GetEnvironmentVariable("AIDOTNET_PPE_WARM_FULL_WIDTH") == "0";

    // The widest team (workers plus the dispatching thread) that may warm-spin in the configured mode.
    private static int WidestWarmTeam => FullWidthParks ? Environment.ProcessorCount - 1 : Environment.ProcessorCount;

    [SkippableFact]
    public void WorkersLeftOutOfSmallerDispatches_ParkWhileThePoolStaysBusy()
    {
        Skip.If(PoolDisabled, "AIDOTNET_COOP_POOL=0: dispatches never reach the persistent workers.");
        int dop = Math.Min(8, Math.Min(WidestWarmTeam, Pool.WorkerCount + 1));
        Skip.If(dop < 3, "Needs a team of at least two workers that may warm-spin.");
        int priorDop = CpuParallelSettings.MaxDegreeOfParallelism;
        var priorWarm = PersistentParallelExecutor.WarmWindow;
        try
        {
            CpuParallelSettings.MaxDegreeOfParallelism = dop;
            PersistentParallelExecutor.WarmWindow = TimeSpan.FromSeconds(5);
            int workers = dop - 1;

            // A full-width dispatch whose chunks are long enough that every worker runs; afterwards they warm-spin.
            Pool.Execute(workers * 8, _ => Thread.SpinWait(20_000));
            Assert.True(WaitUntil(() => AllUnparked(workers), 200), "the full-width team should be spinning");

            // A stream of one-worker dispatches keeps the pool recent; the workers it no longer wakes must park.
            var sw = Stopwatch.StartNew();
            bool parked = false;
            while (sw.ElapsedMilliseconds < 1000 && !(parked = AllParked(1, workers)))
                Pool.Execute(2, _ => Thread.SpinWait(50));
            Assert.True(parked, "workers outside the one-worker team kept spinning for a second");
        }
        finally
        {
            CpuParallelSettings.MaxDegreeOfParallelism = priorDop;
            PersistentParallelExecutor.WarmWindow = priorWarm;
        }
    }

    [SkippableFact]
    public void AHotLoopOfDispatches_WaitsOnItsWarmWorkers()
    {
        Skip.If(PoolDisabled, "AIDOTNET_COOP_POOL=0: dispatches never reach the persistent workers.");
        Skip.If(Math.Min(WidestWarmTeam, Pool.WorkerCount + 1) < 4, "Needs a team of three workers that may warm-spin.");
        int priorDop = CpuParallelSettings.MaxDegreeOfParallelism;
        var priorWarm = PersistentParallelExecutor.WarmWindow;
        try
        {
            CpuParallelSettings.MaxDegreeOfParallelism = 4;
            PersistentParallelExecutor.WarmWindow = TimeSpan.FromSeconds(5);
            bool waited = false;
            for (int i = 0; i < 2000 && !waited; i++)
            {
                Pool.Execute(16, _ => Thread.SpinWait(2_000));
                waited = !Pool.LastDispatchStoleEagerly;
            }
            Assert.True(waited, "no dispatch of a hot loop treated its spinning workers as warm");
        }
        finally
        {
            CpuParallelSettings.MaxDegreeOfParallelism = priorDop;
            PersistentParallelExecutor.WarmWindow = priorWarm;
        }
    }

    private static bool AllUnparked(int workers)
    {
        for (int i = 0; i < workers; i++)
            if (Pool.IsParked(i)) return false;
        return true;
    }

    private static bool AllParked(int from, int to)
    {
        for (int i = from; i < to; i++)
            if (!Pool.IsParked(i)) return false;
        return true;
    }

    private static bool WaitUntil(Func<bool> condition, int milliseconds)
    {
        var sw = Stopwatch.StartNew();
        while (!condition())
        {
            if (sw.ElapsedMilliseconds > milliseconds) return false;
            Thread.Sleep(1);
        }
        return true;
    }
}
