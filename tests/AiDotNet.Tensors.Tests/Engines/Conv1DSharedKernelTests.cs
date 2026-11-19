using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.BlasManaged;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

[Collection("BlasManaged-Stats-Serial")]
public sealed class Conv1DSharedKernelTests
{
    [Theory]
    [InlineData(17, 529, 47)]
    [InlineData(24, 512, 80)]
    public void PackedKernelMatchesIndependentFloat64IncludingKTailAndEdgeRows(int m, int k, int n)
    {
        var kernel = new float[m * k + 2];
        var input = new float[k * n];
        for (int i = 0; i < kernel.Length; i++) kernel[i] = (float)(.02 * Math.Cos(i * .031));
        for (int i = 0; i < input.Length; i++) input[i] = (float)(.2 * Math.Sin(i * .073));
        using var packed = Conv1DColumnPanelKernel.TryCreate(kernel, 1, m, k, n);
        if (packed is null)
        {
            Assert.False(MachineKernelGemm.IsFp32MacroAvailable
                && AiDotNet.Tensors.Engines.BlasManaged.BlasManaged.GetEvolutionTuningTile<float>(PackingMode.ForcePackBoth, true) == (6, 16));
            return;
        }
        int ldc = n + 3;
        var output = new float[m * ldc + 2];
        Array.Fill(output, float.NaN);
        var scratch = new float[packed.WorkerScratchFloats];
        for (int repetition = 0; repetition < 2; repetition++)
        {
            packed.Multiply(input, n, output.AsSpan(1), ldc, scratch);
            for (int row = 0; row < m; row++)
            {
                for (int col = 0; col < n; col++)
                {
                    double expected = 0;
                    for (int inner = 0; inner < k; inner++)
                        expected += (double)kernel[1 + row * k + inner] * input[inner * n + col];
                    Assert.InRange(Math.Abs(output[1 + row * ldc + col] - expected), 0,
                        2e-6 * Math.Max(1, Math.Abs(expected)));
                }
                for (int col = n; col < ldc; col++) Assert.True(float.IsNaN(output[1 + row * ldc + col]));
            }
            Assert.True(float.IsNaN(output[0]));
            Assert.True(float.IsNaN(output[output.Length - 1]));
        }
    }

    [Fact]
    public void MacroDisabledRetainsOffsetConvolutionFallback()
    {
        bool previous = MachineKernelGemm.EnablePanelKernel;
        try
        {
            MachineKernelGemm.EnablePanelKernel = false;
            Assert.Null(Conv1DColumnPanelKernel.TryCreate(new float[24 * 544], 0, 24, 544, 256));
            new Conv1DColumnPanelTests().ForwardPanelsReadAndWriteNonzeroStorageOffsets();
        }
        finally { MachineKernelGemm.EnablePanelKernel = previous; }
    }

    [Fact]
    public void SharedPackRefusesUnboundedKernelScratchBeforeAccessingInput()
    {
        Assert.Null(Conv1DColumnPanelKernel.TryCreate(Array.Empty<float>(), 0, 4096, 8192, 256));
        Assert.Null(Conv1DColumnPanelKernel.TryCreate(Array.Empty<float>(), 0, int.MaxValue, int.MaxValue, 256));
    }

    [Fact]
    public void SharedKernelDoesNotSurviveParameterMutationBetweenConvolutions()
    {
        var engine = new CpuEngine();
        var input = new Tensor<float>(new[] { 1, 32, 1, 2201 });
        var weights = new Tensor<float>(new[] { 24, 32, 1, 17 });
        input.Fill(.2f);
        weights.Fill(.03f);
#if NET5_0_OR_GREATER
        using var suppressDirect = AiDotNet.Tensors.Engines.Simd.DirectConvAvx2.SuppressOnThisThread();
#endif
        var first = engine.Conv2D(input, weights, new[] { 1, 1 }, new[] { 0, 8 }, new[] { 1, 1 }).ToArray();
        weights.Fill(.06f);
        var second = engine.Conv2D(input, weights, new[] { 1, 1 }, new[] { 0, 8 }, new[] { 1, 1 }).ToArray();
        for (int i = 0; i < first.Length; i++) Assert.Equal(first[i] * 2, second[i]);
    }
}
