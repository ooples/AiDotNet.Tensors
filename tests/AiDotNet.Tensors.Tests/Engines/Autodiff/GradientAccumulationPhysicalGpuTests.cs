using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

[Collection("EngineCurrentGlobalState")]
public sealed class GradientAccumulationPhysicalGpuTests
{
    private readonly ITestOutputHelper _output;
    public GradientAccumulationPhysicalGpuTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void Float32Accumulation_PreservesSmallResidualOnPhysicalGpu()
    {
        using var gpu = new DirectGpuTensorEngine();
        bool required = Environment.GetEnvironmentVariable("AIDOTNET_REQUIRE_GPU_TESTS") == "1";
        if (!gpu.IsGpuAvailable)
        {
            Assert.False(required, "A physical GPU was required, but none initialized.");
            return;
        }
        if (gpu.TestBackend is not IGpuFp16ElementwiseBackend { SupportsFp16NativeOps: true })
        {
            Assert.False(required, "Native FP16 elementwise support is required for this precision comparison.");
            return;
        }

        var previousEngine = AiDotNetEngine.Current;
        bool previousStrict = DirectGpuTensorEngine.ThrowOnGpuKernelFallback;
        AiDotNetEngine.Current = gpu;
        DirectGpuTensorEngine.ThrowOnGpuKernelFallback = true;
        try
        {
            var inherited = Accumulate(gpu, GradientAccumulationPrecision.InheritBackwardPrecision);
            var wide = Accumulate(gpu, GradientAccumulationPrecision.Float32);
            Assert.Equal(GpuExecutionRoute.Gpu, inherited.Plan.Route);
            Assert.Equal(GpuScalarType.Float16, inherited.Plan.InputStorage);
            Assert.Equal(GpuScalarType.Float16, inherited.Plan.OutputStorage);
            Assert.Equal(GpuExecutionRoute.Gpu, wide.Plan.Route);
            Assert.Equal(GpuScalarType.Float32, wide.Plan.InputStorage);
            Assert.Equal(GpuScalarType.Float32, wide.Plan.AccumulatorType);
            Assert.Equal(GpuScalarType.Float32, wide.Plan.OutputStorage);
            const double expected = 1.0005;
            double inheritedError = Math.Abs(inherited.Value - expected);
            double wideError = Math.Abs(wide.Value - expected);
            _output.WriteLine($"Backend={wide.Plan.Backend}; expected={expected:R}; inherited={inherited.Value:R} (error={inheritedError:R}); FP32={wide.Value:R} (error={wideError:R})");
            Assert.InRange(wideError, 0, 1e-6);
            Assert.True(wideError < inheritedError / 100,
                "FP32 gradient accumulation must retain the small residual substantially more accurately than FP16.");
        }
        finally
        {
            DirectGpuTensorEngine.ThrowOnGpuKernelFallback = previousStrict;
            AiDotNetEngine.Current = previousEngine;
        }
    }

    private static (float Value, GpuComputePlan Plan) Accumulate(
        DirectGpuTensorEngine gpu, GradientAccumulationPrecision precision)
    {
        using var autocast = new AutocastScope(PrecisionMode.Float16);
        using var tape = new GradientTape<float>(new GradientTapeOptions
        {
            Persistent = false,
            GradientAccumulationPrecision = precision,
        });
        var x = new Tensor<float>(new[] { 1f }, new[] { 1 });
        var main = gpu.TensorMultiplyScalar(x, 1f);
        var residual = gpu.TensorMultiplyScalar(x, 0.0005f);
        var sum = gpu.TensorAdd(main, residual);
        var loss = gpu.ReduceSum(sum, new[] { 0 }, false);
        var additions = new List<GpuComputePlan>();
        void Capture(GpuComputePlan plan)
        {
            if (plan.OperationKind == GpuPrecisionOperation.Add) additions.Add(plan);
        }
        GpuPrecisionDiagnostics.PlanExecuted += Capture;
        try
        {
            // Out-of-place accumulation is required to exercise native FP16 on all backends;
            // the optional in-place FP16 shortcut is CUDA-specific.
            var gradient = tape.ComputeGradients(loss, new[] { x }, createGraph: true)[x];
            return (gradient.ToArray()[0], Assert.Single(additions));
        }
        finally { GpuPrecisionDiagnostics.PlanExecuted -= Capture; }
    }
}
