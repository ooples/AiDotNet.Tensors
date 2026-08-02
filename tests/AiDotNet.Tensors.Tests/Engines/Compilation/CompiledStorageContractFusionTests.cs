using System;
using System.Buffers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.Optimization;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// End-to-end storage-contract coverage for compiled closures that retain raw arrays.
/// A closure must bind live storage or decline the optimization; retaining a logical
/// <c>GetDataArray()</c> materialization silently stale-reads inputs and drops writes
/// for pool-padded tensors and views.
/// </summary>
[Collection("CompilationGlobalState")]
public sealed class CompiledStorageContractFusionTests
{
    [Fact]
    public void DataflowFusion_PoolPaddedOperands_ReplayUsesLiveStorageAndGradients()
    {
        const int m = 2, k = 3, h = 128, n = 3;
        var input = PooledTensor(new[] { m, k }, 1.0f);
        var w1 = PooledTensor(new[] { k, h }, 1.0f);
        var w2 = PooledTensor(new[] { h, n }, 1.0f);
        AssertPoolPadded(input, w1, w2);

        var engine = new CpuEngine();
        var previousEngine = AiDotNetEngine.Current;
        var previousOptions = TensorCodecOptions.Current;
        TensorCodecOptions.SetCurrent(new TensorCodecOptions
        {
            EnableDataflowFusion = true,
            EnableBackwardGradientPooling = false,
        });
        AiDotNetEngine.Current = engine;

        try
        {
            var parameters = new[] { w1, w2 };
            CompiledTrainingPlan<float> plan;
            Tensor<float> output;
            using (var scope = GraphMode.EnableTraining(parameters))
            {
                var hidden = engine.ReLU(engine.TensorMatMul(input, w1));
                output = engine.TensorMatMul(hidden, w2);
                var loss = engine.ReduceSum(output, axes: null, keepDims: false);
                plan = scope.CompileTraining(parameters, loss);
            }

            using (plan)
            {
                Fill(w1, 2.0f); // mutate after closure construction
                var loss = plan.Step();

                Assert.Equal(4608.0f, loss[0], 3);
                Assert.NotNull(output.Grad);
                AssertAllClose(output.Grad!, 1.0f);
                Assert.Same(plan.Gradients[0], w1.Grad);
                Assert.Same(plan.Gradients[1], w2.Grad);
                AssertAllClose(plan.Gradients[0], 6.0f);
                AssertAllClose(plan.Gradients[1], 12.0f);
            }
        }
        finally
        {
            AiDotNetEngine.Current = previousEngine;
            TensorCodecOptions.SetCurrent(previousOptions);
            TensorAllocator.Return(input);
            TensorAllocator.Return(w1);
            TensorAllocator.Return(w2);
        }
    }

    [Fact]
    public void DataflowFusion_OffsetParameterView_DeclinesAndFallbackUsesCorrectSlice()
    {
        const int m = 2, k = 3, h = 128, n = 3;
        var input = new Tensor<float>(new float[m * k], new[] { m, k });
        Fill(input, 1.0f);
        var backing = new float[k * h + 2];
        backing[0] = 12345.0f;
        backing[^1] = -12345.0f;
        var w1 = Tensor<float>.FromMemory(
            new Memory<float>(backing, 1, k * h), new[] { k, h });
        var w2 = new Tensor<float>(new float[h * n], new[] { h, n });
        Fill(w1, 2.0f);
        Fill(w2, 1.0f);
        Assert.Null(w1.GetLiveBackingArrayAllowingPaddingOrNull());

        var engine = new CpuEngine();
        var previousEngine = AiDotNetEngine.Current;
        var previousOptions = TensorCodecOptions.Current;
        TensorCodecOptions.SetCurrent(new TensorCodecOptions
        {
            EnableDataflowFusion = true,
            EnableBackwardGradientPooling = false,
        });
        AiDotNetEngine.Current = engine;

        try
        {
            var parameters = new[] { w1, w2 };
            CompiledTrainingPlan<float> plan;
            using (var scope = GraphMode.EnableTraining(parameters))
            {
                var hidden = engine.ReLU(engine.TensorMatMul(input, w1));
                var output = engine.TensorMatMul(hidden, w2);
                var loss = engine.ReduceSum(output, axes: null, keepDims: false);
                plan = scope.CompileTraining(parameters, loss);
            }

            using (plan)
            {
                Assert.Equal(4608.0f, plan.Step()[0], 3);
                AssertAllClose(plan.Gradients[0], 6.0f);
            }

            Assert.Equal(12345.0f, backing[0]);
            Assert.Equal(-12345.0f, backing[^1]);
        }
        finally
        {
            AiDotNetEngine.Current = previousEngine;
            TensorCodecOptions.SetCurrent(previousOptions);
        }
    }

    [Fact]
    public void AnalyticMatMulLoss_PoolPaddedReplay_WritesLiveLossAndGradients()
    {
        const int m = 3, k = 5, n = 7;
        var input = PooledTensor(new[] { m, k }, 1.0f);
        var weight = PooledTensor(new[] { k, n }, 1.0f);
        AssertPoolPadded(input, weight);

        var engine = new CpuEngine();
        var previousEngine = AiDotNetEngine.Current;
        string? previousAnalytic = Environment.GetEnvironmentVariable("AIDOTNET_ANALYTIC_FORWARD");
        Environment.SetEnvironmentVariable("AIDOTNET_ANALYTIC_FORWARD", "1");
        AiDotNetEngine.Current = engine;

        try
        {
            var parameters = new[] { weight };
            CompiledTrainingPlan<float> plan;
            using (var scope = GraphMode.EnableTraining(parameters))
            {
                var output = engine.TensorMatMul(input, weight);
                var loss = engine.ReduceSum(output, axes: null, keepDims: false);
                plan = scope.CompileTraining(parameters, loss);
            }

            using (plan)
            {
                Fill(weight, 2.0f);
                Assert.Equal(210.0f, plan.Step()[0], 3);
                AssertAllClose(plan.Gradients[0], 3.0f);
            }
        }
        finally
        {
            AiDotNetEngine.Current = previousEngine;
            Environment.SetEnvironmentVariable("AIDOTNET_ANALYTIC_FORWARD", previousAnalytic);
            TensorAllocator.Return(input);
            TensorAllocator.Return(weight);
        }
    }

    [Fact]
    public void SlicePrefixFusion_PoolPaddedReplay_WritesLiveLossAndGradientPrefix()
    {
        const int m = 2, k = 3, nFull = 5, nUsed = 2;
        var input = PooledTensor(new[] { m, k }, 1.0f);
        var weight = PooledTensor(new[] { k, nFull }, 1.0f);
        AssertPoolPadded(input, weight);

        var engine = new CpuEngine();
        var previousEngine = AiDotNetEngine.Current;
        var previousOptions = TensorCodecOptions.Current;
        string? previousSlice = Environment.GetEnvironmentVariable("AIDOTNET_SLICE_PREFIX_FUSION");
        Environment.SetEnvironmentVariable("AIDOTNET_SLICE_PREFIX_FUSION", "1");
        TensorCodecOptions.SetCurrent(new TensorCodecOptions { EnableBackwardGradientPooling = false });
        AiDotNetEngine.Current = engine;

        try
        {
            var parameters = new[] { weight };
            CompiledTrainingPlan<float> plan;
            using (var scope = GraphMode.EnableTraining(parameters))
            {
                var full = engine.TensorMatMul(input, weight);
                var prefix = engine.TensorSlice(full, new[] { 0, 0 }, new[] { m, nUsed });
                var loss = engine.ReduceSum(prefix, axes: null, keepDims: false);
                plan = scope.CompileTraining(parameters, loss);
            }

            using (plan)
            {
                Fill(weight, 2.0f);
                Assert.Equal(24.0f, plan.Step()[0], 3);
                var gradient = plan.Gradients[0].AsSpan();
                for (int row = 0; row < k; row++)
                for (int col = 0; col < nFull; col++)
                    Assert.Equal(col < nUsed ? 2.0f : 0.0f, gradient[row * nFull + col], 3);
            }
        }
        finally
        {
            AiDotNetEngine.Current = previousEngine;
            TensorCodecOptions.SetCurrent(previousOptions);
            Environment.SetEnvironmentVariable("AIDOTNET_SLICE_PREFIX_FUSION", previousSlice);
            TensorAllocator.Return(input);
            TensorAllocator.Return(weight);
        }
    }

    [Fact]
    public void CrossLayerMatMulFusion_PoolPaddedReplay_WritesLiveLossAndGradients()
    {
        const int m = 2, k = 3, h = 4, n = 5;
        var input = PooledTensor(new[] { m, k }, 1.0f);
        var w1 = PooledTensor(new[] { k, h }, 1.0f);
        var w2 = PooledTensor(new[] { h, n }, 1.0f);
        AssertPoolPadded(input, w1, w2);

        var engine = new CpuEngine();
        var previousEngine = AiDotNetEngine.Current;
        var previousOptions = TensorCodecOptions.Current;
        string? previousCross = Environment.GetEnvironmentVariable("AIDOTNET_CROSS_LAYER_FUSION");
        Environment.SetEnvironmentVariable("AIDOTNET_CROSS_LAYER_FUSION", "1");
        TensorCodecOptions.SetCurrent(new TensorCodecOptions
        {
            EnableDataflowFusion = true,
            EnableBackwardGradientPooling = false,
        });
        AiDotNetEngine.Current = engine;

        try
        {
            var parameters = new[] { w1, w2 };
            CompiledTrainingPlan<float> plan;
            using (var scope = GraphMode.EnableTraining(parameters))
            {
                var hidden = engine.TensorMatMul(input, w1);
                var output = engine.TensorMatMul(hidden, w2);
                var loss = engine.ReduceMean(output, axes: null, keepDims: false);
                plan = scope.CompileTraining(parameters, loss);
            }

            using (plan)
            {
                Fill(w1, 2.0f);
                Assert.Equal(24.0f, plan.Step()[0], 3);
                AssertAllClose(plan.Gradients[0], 1.0f);
                AssertAllClose(plan.Gradients[1], 1.2f);
            }
        }
        finally
        {
            AiDotNetEngine.Current = previousEngine;
            TensorCodecOptions.SetCurrent(previousOptions);
            Environment.SetEnvironmentVariable("AIDOTNET_CROSS_LAYER_FUSION", previousCross);
            TensorAllocator.Return(input);
            TensorAllocator.Return(w1);
            TensorAllocator.Return(w2);
        }
    }

    [Fact]
    public void BatchedWeightGradients_PoolPaddedDestinations_WriteLiveStorage_WhenAvailable()
    {
        if (!BlasProvider.IsMklBatchedAvailable) return;

        const int m = 2, k = 3, n = 5;
        var x1 = PooledTensor(new[] { m, k }, 1.0f);
        var x2 = PooledTensor(new[] { m, k }, 2.0f);
        var w1 = PooledTensor(new[] { k, n }, 1.0f);
        var w2 = PooledTensor(new[] { k, n }, 1.0f);
        AssertPoolPadded(x1, x2, w1, w2);

        var engine = new CpuEngine();
        var previousEngine = AiDotNetEngine.Current;
        var previousOptions = TensorCodecOptions.Current;
        string? previousBatch = Environment.GetEnvironmentVariable("AIDOTNET_DW_BATCHING");
        Environment.SetEnvironmentVariable("AIDOTNET_DW_BATCHING", "1");
        TensorCodecOptions.SetCurrent(new TensorCodecOptions
        {
            EnableDataflowFusion = false,
            EnableBackwardGradientPooling = false,
        });
        AiDotNetEngine.Current = engine;

        try
        {
            var parameters = new[] { w1, w2 };
            CompiledTrainingPlan<float> plan;
            using (var scope = GraphMode.EnableTraining(parameters))
            {
                var y1 = engine.TensorMatMul(x1, w1);
                var y2 = engine.TensorMatMul(x2, w2);
                var loss = engine.ReduceSum(engine.TensorAdd(y1, y2), axes: null, keepDims: false);
                plan = scope.CompileTraining(parameters, loss);
            }

            using (plan)
            {
                _ = plan.Step();
                AssertAllClose(plan.Gradients[0], 2.0f);
                AssertAllClose(plan.Gradients[1], 4.0f);
            }
        }
        finally
        {
            AiDotNetEngine.Current = previousEngine;
            TensorCodecOptions.SetCurrent(previousOptions);
            Environment.SetEnvironmentVariable("AIDOTNET_DW_BATCHING", previousBatch);
            TensorAllocator.Return(x1);
            TensorAllocator.Return(x2);
            TensorAllocator.Return(w1);
            TensorAllocator.Return(w2);
        }
    }

    private static Tensor<float> PooledTensor(int[] shape, float value)
    {
        int length = 1;
        for (int i = 0; i < shape.Length; i++) length *= shape[i];
        var backing = ArrayPool<float>.Shared.Rent(length);
        Array.Clear(backing, 0, backing.Length);
        var tensor = Tensor<float>.FromPooledMemory(
            new Memory<float>(backing, 0, length), shape, backing);
        Fill(tensor, value);
        return tensor;
    }

    private static void Fill(Tensor<float> tensor, float value)
    {
        var span = tensor.AsWritableSpan();
        for (int i = 0; i < span.Length; i++) span[i] = value;
    }

    private static void AssertPoolPadded(params Tensor<float>[] tensors)
    {
        foreach (var tensor in tensors)
        {
            var live = tensor.GetLiveBackingArrayAllowingPaddingOrNull();
            Assert.NotNull(live);
            Assert.True(live!.Length > tensor.Length,
                $"Test shape [{string.Join(",", tensor._shape)}] did not receive padded storage.");
        }
    }

    private static void AssertAllClose(Tensor<float> tensor, float expected)
    {
        var span = tensor.AsSpan();
        for (int i = 0; i < span.Length; i++)
            Assert.True(MathF.Abs(span[i] - expected) <= 1e-3f,
                $"Gradient[{i}] was {span[i]:R}; expected {expected:R}.");
    }
}
