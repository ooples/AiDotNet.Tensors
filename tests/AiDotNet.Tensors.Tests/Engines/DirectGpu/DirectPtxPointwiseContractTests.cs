#if NET5_0_OR_GREATER
using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

public sealed class DirectPtxPointwiseContractTests
{
    [Fact]
    public void CoverageManifest_IsCompleteUniqueAndFailClosed()
    {
        Assert.Equal(74, DirectPtxPointwiseCoverageManifest.All.Count);
        Assert.Equal(
            DirectPtxPointwiseCoverageManifest.All.Count,
            DirectPtxPointwiseCoverageManifest.All
                .Select(cell => cell.Api)
                .Distinct(StringComparer.Ordinal)
                .Count());
        Assert.Equal(
            3,
            DirectPtxPointwiseCoverageManifest.All.Count(
                cell => cell.Status == DirectPtxPointwiseCoverageStatus.ExperimentalDirectPtx));
        Assert.DoesNotContain(
            DirectPtxPointwiseCoverageManifest.All,
            cell => cell.Status == DirectPtxPointwiseCoverageStatus.PromotedDirectPtx);

        string api = DirectPtxPointwiseCoverageManifest.All[0].Api;
        Assert.Equal(api, DirectPtxPointwiseCoverageManifest.Get(api).Api);
        Assert.Throws<ArgumentException>(() => DirectPtxPointwiseCoverageManifest.Get(" "));
        Assert.Throws<KeyNotFoundException>(() =>
            DirectPtxPointwiseCoverageManifest.Get("CudaBackend.NotARealPointwiseRoute"));
    }

    [Fact]
    public void SwiGluEmitter_HasPointerOnlyVectorizedAbiAndNoPromotedShape()
    {
        string ptx = PtxFusedSwiGluF32Kernel.EmitPtx(8, 6, 1, 4096);

        Assert.Contains(".target sm_86", ptx);
        Assert.Contains($".visible .entry {PtxFusedSwiGluF32Kernel.EntryPoint}", ptx);
        Assert.Equal(2, Count(ptx, ".param .u64"));
        Assert.DoesNotContain(".param .u32", ptx);
        Assert.Equal(2, Count(ptx, "ld.global.nc.v4.f32"));
        Assert.Equal(1, Count(ptx, "st.global.v4.f32"));
        Assert.Equal(4, Count(ptx, "ex2.approx.f32"));
        Assert.False(PtxFusedSwiGluF32Kernel.IsPromotedShape(1, 4096));
    }

    [Fact]
    public void GeGluEmitter_HasPointerOnlyVectorizedAbiAndNoPromotedShape()
    {
        string ptx = PtxFusedGeGluF32Kernel.EmitPtx(8, 6, 1, 4096);

        Assert.Contains(".target sm_86", ptx);
        Assert.Contains($".visible .entry {PtxFusedGeGluF32Kernel.EntryPoint}", ptx);
        Assert.Equal(2, Count(ptx, ".param .u64"));
        Assert.DoesNotContain(".param .u32", ptx);
        Assert.Equal(2, Count(ptx, "ld.global.nc.v4.f32"));
        Assert.Equal(1, Count(ptx, "st.global.v4.f32"));
        Assert.Equal(4, Count(ptx, "tanh.approx.f32"));
        Assert.False(PtxFusedGeGluF32Kernel.IsPromotedShape(1, 4096));
    }

    [Fact]
    public void GeGluBackwardEmitter_HasPointerOnlyVectorizedAbiAndNoPromotedShape()
    {
        string ptx = PtxFusedGeGluBackwardF32Kernel.EmitPtx(8, 6, 1, 4096);

        Assert.Contains(".target sm_86", ptx);
        Assert.Contains($".visible .entry {PtxFusedGeGluBackwardF32Kernel.EntryPoint}", ptx);
        Assert.Equal(3, Count(ptx, ".param .u64"));
        Assert.DoesNotContain(".param .u32", ptx);
        Assert.Equal(3, Count(ptx, "ld.global.nc.v4.f32"));
        Assert.Equal(2, Count(ptx, "st.global.v4.f32"));
        Assert.Equal(4, Count(ptx, "tanh.approx.f32"));
        Assert.False(PtxFusedGeGluBackwardF32Kernel.IsPromotedShape(1, 4096));
    }

    [SkippableFact]
    public void DriverOnlySwiGluForward_MatchesCpuOracleAndHasNoLocalMemory()
    {
        using var runtime = CreateValidatedGpuRuntime();
        using var kernel = new PtxFusedSwiGluF32Kernel(runtime, 1, 4096);
        float[] input = CreateSplitInput(4096);
        var expected = new float[4096];
        for (int index = 0; index < expected.Length; index++)
        {
            double value = input[index];
            double gate = input[4096 + index];
            expected[index] = (float)(value * gate / (1.0 + Math.Exp(-gate)));
        }

        float[] actual = RunForward(runtime, kernel.Blueprint, input, kernel.Launch);
        AssertKernelAudit(kernel.Audit, minimumActiveBlocks: 6);
        AssertClose(actual, expected, absoluteTolerance: 3e-5f, relativeTolerance: 3e-4f);
    }

    [SkippableFact]
    public void DriverOnlyGeGluForward_MatchesCpuOracleAndHasNoLocalMemory()
    {
        using var runtime = CreateValidatedGpuRuntime();
        using var kernel = new PtxFusedGeGluF32Kernel(runtime, 1, 4096);
        float[] input = CreateSplitInput(4096);
        var expected = new float[4096];
        for (int index = 0; index < expected.Length; index++)
            expected[index] = input[index] * GeluTanh(input[4096 + index]);

        float[] actual = RunForward(runtime, kernel.Blueprint, input, kernel.Launch);
        AssertKernelAudit(kernel.Audit, minimumActiveBlocks: 6);
        AssertClose(actual, expected, absoluteTolerance: 3e-5f, relativeTolerance: 3e-4f);
    }

    [SkippableFact]
    public void DriverOnlyGeGluBackward_MatchesAnalyticOracleAndHasNoLocalMemory()
    {
        using var runtime = CreateValidatedGpuRuntime();
        using var kernel = new PtxFusedGeGluBackwardF32Kernel(runtime, 1, 4096);
        float[] input = CreateSplitInput(4096);
        float[] gradOutput = Enumerable.Range(0, 4096)
            .Select(index => (float)(((index % 29) - 14) / 31.0))
            .ToArray();
        var expected = new float[8192];
        for (int index = 0; index < gradOutput.Length; index++)
        {
            double value = input[index];
            double gate = input[4096 + index];
            double gradient = gradOutput[index];
            expected[index] = (float)(gradient * GeluTanh(gate));
            expected[4096 + index] =
                (float)(gradient * value * GeluTanhDerivative(gate));
        }

        using var gradOutputBuffer =
            runtime.AllocateBytes(kernel.Blueprint.Tensors[0].RequiredBytes);
        using var inputBuffer =
            runtime.AllocateBytes(kernel.Blueprint.Tensors[1].RequiredBytes);
        using var gradInputBuffer =
            runtime.AllocateBytes(kernel.Blueprint.Tensors[2].RequiredBytes);
        gradOutputBuffer.Upload<float>(gradOutput);
        inputBuffer.Upload<float>(input);
        kernel.Launch(
            DirectPtxTensorView.CreateOwned(
                gradOutputBuffer, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(inputBuffer, kernel.Blueprint.Tensors[1]),
            DirectPtxTensorView.CreateOwned(
                gradInputBuffer, kernel.Blueprint.Tensors[2]));
        runtime.Synchronize();

        var actual = new float[expected.Length];
        gradInputBuffer.Download<float>(actual);
        AssertKernelAudit(kernel.Audit, minimumActiveBlocks: 4);
        AssertClose(actual, expected, absoluteTolerance: 5e-5f, relativeTolerance: 5e-4f);
    }

    private static DirectPtxRuntime CreateValidatedGpuRuntime()
    {
        Skip.IfNot(
            DirectPtxRuntime.IsAvailable,
            "Requires an NVIDIA CUDA driver and GPU.");
        var runtime = new DirectPtxRuntime();
        try
        {
            Skip.IfNot(
                DirectPtxArchitecture.HasValidatedGatedGlu(
                    runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor),
                "Requires the validated GA10x/SM86 gated-GLU backend.");
            return runtime;
        }
        catch
        {
            runtime.Dispose();
            throw;
        }
    }

    private static float[] RunForward(
        DirectPtxRuntime runtime,
        DirectPtxKernelBlueprint blueprint,
        float[] input,
        Action<DirectPtxTensorView, DirectPtxTensorView> launch)
    {
        using var inputBuffer = runtime.AllocateBytes(blueprint.Tensors[0].RequiredBytes);
        using var outputBuffer = runtime.AllocateBytes(blueprint.Tensors[1].RequiredBytes);
        inputBuffer.Upload<float>(input);
        launch(
            DirectPtxTensorView.CreateOwned(inputBuffer, blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(outputBuffer, blueprint.Tensors[1]));
        runtime.Synchronize();
        var actual = new float[4096];
        outputBuffer.Download<float>(actual);
        return actual;
    }

    private static float[] CreateSplitInput(int halfDimension)
    {
        var input = new float[2 * halfDimension];
        for (int index = 0; index < halfDimension; index++)
        {
            input[index] = (float)(((index % 23) - 11) / 19.0);
            input[halfDimension + index] = (float)(((index % 31) - 15) / 17.0);
        }
        return input;
    }

    private static float GeluTanh(double value)
    {
        const double coefficient = 0.7978845608028654;
        const double cubic = 0.044715;
        return (float)(0.5 * value *
            (1.0 + Math.Tanh(coefficient * (value + cubic * value * value * value))));
    }

    private static float GeluTanhDerivative(double value)
    {
        const double coefficient = 0.7978845608028654;
        const double cubic = 0.044715;
        double square = value * value;
        double tanh = Math.Tanh(coefficient * (value + cubic * square * value));
        return (float)(0.5 * (1.0 + tanh) +
            0.5 * value * (1.0 - tanh * tanh) *
            coefficient * (1.0 + 3.0 * cubic * square));
    }

    private static void AssertKernelAudit(
        DirectPtxKernelAudit audit,
        int minimumActiveBlocks)
    {
        Assert.Equal(0, audit.Function.LocalBytesPerThread);
        Assert.Equal(0, audit.Function.StaticSharedBytes);
        Assert.True(audit.ActiveBlocksPerMultiprocessor >= minimumActiveBlocks);
    }

    private static void AssertClose(
        float[] actual,
        float[] expected,
        float absoluteTolerance,
        float relativeTolerance)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int index = 0; index < expected.Length; index++)
        {
            float tolerance = absoluteTolerance +
                relativeTolerance * MathF.Abs(expected[index]);
            Assert.True(MathF.Abs(actual[index] - expected[index]) <= tolerance,
                $"index {index}: actual {actual[index]:G9}, expected {expected[index]:G9}, " +
                $"tolerance {tolerance:G9}.");
        }
    }

    private static int Count(string text, string value) =>
        (text.Length - text.Replace(value, string.Empty, StringComparison.Ordinal).Length) /
        value.Length;
}
#endif
