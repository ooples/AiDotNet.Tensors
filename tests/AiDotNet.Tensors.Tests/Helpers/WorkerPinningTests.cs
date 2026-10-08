using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Helpers;

/// <summary>
/// Worker pinning only changes where pool threads run, never what they compute (#653). Results must be
/// bit-identical under every <see cref="WorkerPinning"/> mode, including when the mode flips between
/// parallel ops and the workers re-pin or un-pin on their next wake.
/// </summary>
[Collection("BlasManaged-Stats-Serial")]
public class WorkerPinningTests
{
    [Fact]
    public void Default_IsAuto_UnlessTheEnvironmentOverridesIt()
    {
        string? env = Environment.GetEnvironmentVariable("AIDOTNET_PIN_WORKERS");
        if (env is null)
            Assert.Equal(WorkerPinning.Auto, CpuParallelSettings.WorkerPinning);
    }

    [Fact]
    public void Results_AreBitIdentical_AcrossModesAndModeSwitches()
    {
        var engine = new CpuEngine();
        var rng = RandomHelper.CreateSeededRandom(653);
        var a = new Tensor<float>(new[] { 256, 768 });
        var b = new Tensor<float>(new[] { 768, 768 });
        var scores = new Tensor<float>(new[] { 12, 256, 256 });
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() - 0.5);
        for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() - 0.5);
        for (int i = 0; i < scores.Length; i++) scores[i] = (float)(rng.NextDouble() * 4 - 2);

        var beforeMode = CpuParallelSettings.WorkerPinning;
        int beforeDop = CpuParallelSettings.MaxDegreeOfParallelism;
        try
        {
            CpuParallelSettings.MaxDegreeOfParallelism = Math.Max(2, Math.Min(16, Environment.ProcessorCount));
            CpuParallelSettings.WorkerPinning = WorkerPinning.Never;
            var gemmReference = engine.BatchMatMul(a, b);
            var softmaxReference = engine.Softmax(scores, -1);

            var sequence = new[]
            {
                (WorkerPinning.Always, 0), (WorkerPinning.Never, 0), (WorkerPinning.Auto, 0),
                (WorkerPinning.Auto, Environment.ProcessorCount), (WorkerPinning.Always, 0),
            };
            foreach (var (mode, dop) in sequence)
            {
                CpuParallelSettings.WorkerPinning = mode;
                if (dop > 0) CpuParallelSettings.MaxDegreeOfParallelism = dop;

                var gemm = engine.BatchMatMul(a, b);
                var softmax = engine.Softmax(scores, -1);

                for (int i = 0; i < gemm.Length; i++) Assert.Equal(gemmReference[i], gemm[i]);
                for (int i = 0; i < softmax.Length; i++) Assert.Equal(softmaxReference[i], softmax[i]);
            }
        }
        finally
        {
            CpuParallelSettings.WorkerPinning = beforeMode;
            CpuParallelSettings.MaxDegreeOfParallelism = beforeDop;
        }
    }
}
