using System;
using System.Collections.Concurrent;
using System.Reflection;
using System.Threading;
using System.Threading.Tasks;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

[CollectionDefinition(Name, DisableParallelization = true)]
public sealed class NativeAvailabilityProbeRaceCollection
{
    public const string Name = "NativeAvailabilityProbeRace";
}

/// <summary>
/// The NVRTC and HIP availability probes are process-wide statics. They used to publish "checked" before the probe
/// stored its result, so a caller racing the first probe (xUnit starts test classes in parallel) could read "not
/// available" on a machine where the library loads. These tests reset the statics and hit the probe from many threads
/// at once: every caller must see the same answer as a single, settled probe. The collection is non-parallel because
/// the reset is visible to every other test in the process.
/// </summary>
[Collection(NativeAvailabilityProbeRaceCollection.Name)]
public class NativeAvailabilityProbeRaceTests
{
    private const int Threads = 32;

    [Theory]
    [InlineData("AiDotNet.Tensors.Engines.DirectGpu.CUDA.NvrtcNativeBindings", "_api")]
    [InlineData("AiDotNet.Tensors.Engines.DirectGpu.HIP.HipNativeBindings", "_isAvailable")]
    public void ConcurrentFirstProbes_AllSeeTheSettledResult(string typeName, string resultField)
    {
        var type = typeof(AiDotNet.Tensors.Engines.DirectGpu.IDirectGpuBackend).Assembly.GetType(typeName, throwOnError: true)!;
        const BindingFlags Static = BindingFlags.NonPublic | BindingFlags.Static;
        var checkedField = type.GetField("_availabilityChecked", Static)
            ?? throw new InvalidOperationException($"{typeName}._availabilityChecked not found.");
        var result = type.GetField(resultField, Static)
            ?? throw new InvalidOperationException($"{typeName}.{resultField} not found.");
        var isAvailable = type.GetProperty("IsAvailable", Static | BindingFlags.Public)
            ?? throw new InvalidOperationException($"{typeName}.IsAvailable not found.");

        bool settled = (bool)isAvailable.GetValue(null)!;
        object? settledResult = result.GetValue(null);
        try
        {
            for (int round = 0; round < 8; round++)
            {
                checkedField.SetValue(null, false);
                result.SetValue(null, resultField == "_api" ? null : (object)false);

                var seen = new ConcurrentBag<bool>();
                using var start = new Barrier(Threads);
                var tasks = new Task[Threads];
                for (int t = 0; t < Threads; t++)
                {
                    tasks[t] = Task.Factory.StartNew(() =>
                    {
                        start.SignalAndWait();
                        seen.Add((bool)isAvailable.GetValue(null)!);
                    }, TaskCreationOptions.LongRunning);
                }
                Task.WaitAll(tasks);

                Assert.Equal(Threads, seen.Count);
                Assert.All(seen, s => Assert.Equal(settled, s));
            }
        }
        finally
        {
            result.SetValue(null, settledResult);
            checkedField.SetValue(null, true);
        }
    }
}
