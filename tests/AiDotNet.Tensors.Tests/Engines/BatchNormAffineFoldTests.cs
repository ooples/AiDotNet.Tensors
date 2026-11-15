using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

[Collection("DirectGpuSerial")]
public sealed class BatchNormAffineFoldTests
{
    [Fact]
    public void Double_precision_and_cache_invalidation_follow_the_tensor_operands()
    {
        var engine = new CpuEngine();
        var gamma = new Tensor<double>(new[] { 1.0000000001, 2.0000000003 }, new[] { 2 });
        var beta = new Tensor<double>(new[] { 0.1000000001, -0.2000000002 }, new[] { 2 });
        var mean = new Tensor<double>(new[] { 0.3000000003, -0.4000000004 }, new[] { 2 });
        var variance = new Tensor<double>(new[] { 0.5000000005, 0.6000000006 }, new[] { 2 });
        const double epsilon = 1e-12;
        var first = CpuEngine.BatchNormAffineFold<double>.Get(engine, gamma, beta, mean, variance, epsilon);
        var cached = CpuEngine.BatchNormAffineFold<double>.Get(engine, gamma, beta, mean, variance, epsilon);
        Assert.Same(first.Scale, cached.Scale);
        Assert.Same(first.Shift, cached.Shift);
        for (int i = 0; i < 2; i++)
        {
            double expectedScale = gamma[i] / Math.Sqrt(variance[i] + epsilon);
            Assert.Equal(expectedScale, first.Scale[i], 13);
            Assert.Equal(beta[i] - mean[i] * expectedScale, first.Shift[i], 13);
        }
        // A cache is local to its engine, even for the same input identities and versions.
        var other = CpuEngine.BatchNormAffineFold<double>.Get(new CpuEngine(), gamma, beta, mean, variance, epsilon);
        Assert.NotSame(first.Scale, other.Scale);
        variance.AsWritableSpan()[0] = 1.2500000001;
        variance.IncrementVersion();
        var changed = CpuEngine.BatchNormAffineFold<double>.Get(engine, gamma, beta, mean, variance, epsilon);
        Assert.NotSame(first.Scale, changed.Scale);
        Assert.Equal(gamma[0] / Math.Sqrt(variance[0] + epsilon), changed.Scale[0], 13);
        var changedEpsilon = CpuEngine.BatchNormAffineFold<double>.Get(engine, gamma, beta, mean, variance, 0.1);
        Assert.NotSame(changed.Scale, changedEpsilon.Scale);
        Assert.Equal(gamma[0] / Math.Sqrt(variance[0] + 0.1), changedEpsilon.Scale[0], 13);
    }

    [SkippableFact]
    public void Device_operands_stay_on_device_and_cached_results_outlive_the_callers_scope()
    {
        DirectGpuTensorEngine? gpu = null;
        try { gpu = new DirectGpuTensorEngine(); }
        catch (Exception ex) when (ex is PlatformNotSupportedException or DllNotFoundException) { }
        using var device = gpu;
        Skip.If(gpu is null || !gpu.IsGpuAvailable, "No direct GPU backend available.");
        var prior = AiDotNetEngine.Current;
        try
        {
            AiDotNetEngine.Current = gpu!;
            var source = new Tensor<float>(new[] { 1f, 2f }, new[] { 2 });
            var gamma = gpu!.TensorAddScalar(source, 1f);
            var beta = gpu.TensorAddScalar(source, 2f);
            var mean = gpu.TensorAddScalar(source, 3f);
            var variance = gpu.TensorAddScalar(source, 4f);
            var operands = new[] { gamma, beta, mean, variance };
            foreach (var operand in operands) Assert.True(HostSync.IsPending(operand.DataVector));
            (Tensor<float> Scale, Tensor<float> Shift) folded;
            using (TensorScope.Begin())
                folded = CpuEngine.BatchNormAffineFold<float>.Get(gpu, gamma, beta, mean, variance, 1e-5);
            foreach (var operand in operands) Assert.True(HostSync.IsPending(operand.DataVector));
            var cached = CpuEngine.BatchNormAffineFold<float>.Get(gpu, gamma, beta, mean, variance, 1e-5);
            Assert.Same(folded.Scale, cached.Scale);
            Assert.Same(folded.Shift, cached.Shift);
            var scales = folded.Scale.ToArray();
            var shifts = folded.Shift.ToArray();
            for (int i = 0; i < 2; i++)
            {
                float expectedScale = (i + 2f) / (float)Math.Sqrt(i + 5f + 1e-5f);
                Assert.True(Math.Abs(scales[i] - expectedScale) < 1e-6f);
                Assert.True(Math.Abs(shifts[i] - ((i + 3f) - (i + 4f) * expectedScale)) < 1e-6f);
            }
        }
        finally { AiDotNetEngine.Current = prior; }
    }
}
