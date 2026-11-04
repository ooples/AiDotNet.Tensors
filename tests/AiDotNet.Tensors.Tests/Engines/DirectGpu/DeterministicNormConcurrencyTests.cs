using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Gpu;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

[Collection("DirectGpuSerial")]
public sealed class DeterministicNormConcurrencyTests : IClassFixture<DirectGpuTensorEngineTestFixture>
{
    private readonly DirectGpuTensorEngineTestFixture _fixture;
    public DeterministicNormConcurrencyTests(DirectGpuTensorEngineTestFixture fixture) => _fixture = fixture;

    [SkippableFact]
    public void ConcurrentReductions_KeepEachCallPartialsSeparate()
    {
        Skip.IfNot(_fixture.IsAvailable, "No GPU device.");
        var backend = _fixture.Engine!.GetBackend()!;
        var multi = (IMultiTensorKernels)backend;
        var resources = new List<IGpuBuffer>();
        var buffers = new IGpuBuffer[8];
        var records = new IGpuBuffer[8];
        var sizes = new int[8];
        var expected = new uint[8][];
        try
        {
            for (int call = 0; call < buffers.Length; call++)
            {
                var values = new float[4097 + call * 101];
                for (int i = 0; i < values.Length; i++) values[i] = (call + 1) * (i % 19 - 9) * 0.125f;
                sizes[call] = values.Length;
                expected[call] = DeterministicGradientNorm.Compute(new[] { values }, float.MaxValue, NonFiniteGradientPolicy.LeaveUnscaled);
                resources.Add(buffers[call] = backend.AllocateBuffer(values));
                resources.Add(records[call] = backend.AllocateBuffer(DeterministicGradientNorm.RecordWords));
            }
            Parallel.For(0, buffers.Length, call =>
            {
                for (int repeat = 0; repeat < 8; repeat++)
                {
                    multi.MultiTensorSumOfSquares(new[] { buffers[call] }, new[] { sizes[call] }, records[call]);
                    var actual = MultiTensorArgs.RecordFromSlots(backend.DownloadBuffer(records[call]));
                    Assert.Equal(expected[call][DeterministicGradientNorm.WordSumSquaresLo], actual[DeterministicGradientNorm.WordSumSquaresLo]);
                    Assert.Equal(expected[call][DeterministicGradientNorm.WordSumSquaresHi], actual[DeterministicGradientNorm.WordSumSquaresHi]);
                    Assert.Equal(expected[call][DeterministicGradientNorm.WordNorm], actual[DeterministicGradientNorm.WordNorm]);
                }
            });
        }
        finally
        {
            foreach (var resource in resources) resource.Dispose();
        }
    }
}
