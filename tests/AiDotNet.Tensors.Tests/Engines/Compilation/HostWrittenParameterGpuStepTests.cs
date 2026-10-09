using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Tests.Engines.DirectGpu;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// A host write to a device-bound parameter between steps must reach the next step's update. On a backend without an
/// in-place host upload (OpenCL, Vulkan, ...) the plan used to drop the binding: the next forward bound a new buffer with
/// the written values while the grouped device optimizer kept updating the old buffer and then made it authoritative,
/// so the write was lost.
/// </summary>
[Collection("VulkanGlobalState")]
public sealed class HostWrittenParameterGpuStepTests : IClassFixture<DirectGpuTensorEngineTestFixture>
{
    private readonly DirectGpuTensorEngineTestFixture _fixture;

    public HostWrittenParameterGpuStepTests(DirectGpuTensorEngineTestFixture fixture) => _fixture = fixture;

    [SkippableFact]
    public void HostWriteBetweenSteps_IsWhatTheNextUpdateStartsFrom()
    {
        Skip.IfNot(_fixture.IsAvailable, "No GPU device.");
        IEngine gpu = _fixture.Engine!;
        var previous = AiDotNetEngine.Current;
        AiDotNetEngine.Current = gpu;
        try
        {
            const int batch = 4, inF = 8, outF = 8;
            const float lr = 0.1f;
            var xData = new float[batch * inF];
            for (int i = 0; i < xData.Length; i++) xData[i] = 0.25f * ((i % 7) - 3);
            var x = new Tensor<float>(xData, new[] { batch, inF });
            var wData = new float[inF * outF];
            for (int i = 0; i < wData.Length; i++) wData[i] = 0.01f * i;
            var w = new Tensor<float>(wData, new[] { inF, outF }).Gpu();

            ICompiledTrainingPlan<float> plan;
            using (var scope = GraphMode.Enable())
            {
                gpu.ReduceSum(gpu.TensorMatMul(x, w), null);
                plan = scope.CompileTraining(new[] { w });
            }
            using (plan)
            {
                plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: lr);
                plan.Step();

                // d(sum(x.W))/dW[i, j] = sum_b x[b, i], whatever W is.
                var written = new float[inF * outF];
                for (int i = 0; i < written.Length; i++) written[i] = 5f - 0.1f * i;
                var span = w.AsWritableSpan();
                for (int i = 0; i < written.Length; i++) span[i] = written[i];
                w.IncrementVersion();

                plan.Step();

                var after = w.ToArray();
                for (int i = 0; i < inF; i++)
                {
                    float colSum = 0f;
                    for (int b = 0; b < batch; b++) colSum += xData[b * inF + i];
                    for (int j = 0; j < outF; j++)
                    {
                        float expected = written[i * outF + j] - lr * colSum;
                        Assert.True(Math.Abs(expected - after[i * outF + j]) <= 1e-4f,
                            $"W[{i},{j}]: expected {expected}, got {after[i * outF + j]}");
                    }
                }
            }
        }
        finally
        {
            AiDotNetEngine.Current = previous;
        }
    }
}
