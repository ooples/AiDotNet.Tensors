using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// The float GroupNorm splits its groups into chunks by the CPU degree of parallelism, a process-wide setting another
/// thread can change at any time. A forward that read it once for the chunk count and again for the chunk size covered
/// only part of the groups when it changed in between, leaving their statistics and output unset.
/// </summary>
[Collection("EngineCurrentGlobalState")]
public class GroupNormParallelDegreeTests
{
    [Fact]
    public void EveryGroupIsComputed_WhenTheDegreeOfParallelismChangesMidCall()
    {
        var engine = new CpuEngine();
        const int batch = 2, channels = 16, groups = 8, spatial = 8 * 8;
        var rng = new Random(20261127);
        var data = new float[batch * channels * spatial];
        for (int i = 0; i < data.Length; i++) data[i] = (float)(rng.NextDouble() * 2 - 1);
        var x = new Tensor<float>(data, new[] { batch, channels, 8, 8 });
        var gamma = new Tensor<float>(new[] { channels });
        var beta = new Tensor<float>(new[] { channels });
        gamma.Fill(1f);
        beta.Fill(0f);

        int prior = CpuParallelSettings.MaxDegreeOfParallelism;
        int caller = Environment.CurrentManagedThreadId;
        // One chunk is decided; then, before any chunk runs, the setting widens (as another thread could make it).
        CpuParallelSettings.MaxDegreeOfParallelism = 1;
        CpuEngine.GroupNormDegreeDecidedForTests = () =>
        {
            if (Environment.CurrentManagedThreadId == caller)
                CpuParallelSettings.MaxDegreeOfParallelism = batch * groups;
        };
        Tensor<float> output, variance;
        try
        {
            output = engine.GroupNorm(x, groups, gamma, beta, 1e-5, out _, out variance);
        }
        finally
        {
            CpuEngine.GroupNormDegreeDecidedForTests = null;
            CpuParallelSettings.MaxDegreeOfParallelism = prior;
        }

        var v = variance.GetDataArray();
        Assert.Equal(batch * groups, v.Length);
        for (int g = 0; g < v.Length; g++) Assert.True(v[g] > 0f, $"variance[{g}]={v[g]}");
        // Every group normalizes to unit variance, so none of its output is left unwritten.
        var y = output.GetDataArray();
        int groupSize = channels / groups * spatial;
        for (int g = 0; g < batch * groups; g++)
        {
            double sumSq = 0;
            for (int i = 0; i < groupSize; i++) sumSq += (double)y[g * groupSize + i] * y[g * groupSize + i];
            Assert.True(sumSq / groupSize > 0.5, $"group {g} output mean square {sumSq / groupSize}");
        }
    }
}
