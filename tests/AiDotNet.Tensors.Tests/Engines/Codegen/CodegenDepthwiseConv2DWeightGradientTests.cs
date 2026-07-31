// Copyright (c) AiDotNet. All rights reserved.

using System;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ptx;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Codegen;

public sealed class CodegenDepthwiseConv2DWeightGradientTests
{
    [Fact]
    public void DepthwiseWeightGradient_RecoversCooperativeReductionFromMaps()
    {
        var entry = CodegenKernelCatalog.Find("depthwise_conv2d_3x3_bwd_weights")!;

        Assert.True(CodegenDepthwiseConv2DWeightGradientPlan.TryCreate(
            entry.Verify, out var verify, out string verifyReason), verifyReason);
        Assert.NotNull(verify);
        Assert.Equal((8, 2, 16, 16, 24, 512, 96),
            (verify!.Channels, verify.Batch, verify.Height, verify.Width,
             verify.Blocks, verify.ReductionElements, verify.SharedMemoryBytes));

        Assert.True(CodegenDepthwiseConv2DWeightGradientPlan.TryCreate(
            entry.Bench, out var bench, out string benchReason), benchReason);
        Assert.NotNull(bench);
        Assert.Equal((64, 32, 56, 56, 192, 100352, 96),
            (bench!.Channels, bench.Batch, bench.Height, bench.Width,
             bench.Blocks, bench.ReductionElements, bench.SharedMemoryBytes));
        Assert.Equal(0, bench.GradOutputInput);
        Assert.Equal(1, bench.DataInput);
    }

    [Fact]
    public void DepthwiseWeightGradient_RefusesDifferentSemantics()
    {
        var depthwise = CodegenKernelCatalog.Find("depthwise_conv2d_3x3_bwd_weights")!.Verify;
        var relu = new CodegenKernelSpec(
            depthwise.Name, depthwise.Space,
            new[] { depthwise.Inputs[0], depthwise.Inputs[1] }, depthwise.Output,
            new[] { 0, 1 }, CodegenReduceKind.Sum,
            activation: CodegenActivationKind.ReLU);

        Assert.False(CodegenDepthwiseConv2DWeightGradientPlan.TryCreate(
            relu, out _, out string epilogueReason));
        Assert.Contains("epilogue", epilogueReason);

        var dense = CodegenKernelCatalog.Find("conv2d_3x3_bwd_weights")!.Verify;
        Assert.False(CodegenDepthwiseConv2DWeightGradientPlan.TryCreate(
            dense, out _, out string denseReason));
        Assert.Contains("[channel,kh,kw]", denseReason);

        var partial = CodegenSplitReduction.TryPlan(depthwise)!.Partial;
        Assert.False(CodegenDepthwiseConv2DWeightGradientPlan.TryCreate(
            partial, out _, out string splitReason));
        Assert.Contains("[channel,kh,kw]", splitReason);
    }

    [Fact]
    public void DepthwiseWeightGradient_EmitsCoalescedThreeTapReduction()
    {
        var spec = CodegenKernelCatalog.Find("depthwise_conv2d_3x3_bwd_weights")!.Bench;
        var emitter = new PtxDepthwiseConv2DWeightGradientEmitter();

        string first = emitter.Emit(spec, 8, 6);
        string second = emitter.Emit(spec, 8, 6);

        Assert.Equal(first, second);
        Assert.Equal(192u, emitter.LaunchBlocks);
        Assert.Equal(256, emitter.LaunchBlockThreads);
        Assert.Equal(96, emitter.SharedMemoryBytes);
        Assert.StartsWith(".version 7.1", first, StringComparison.Ordinal);
        Assert.Contains("three kw accumulators share dOut", first);
        Assert.Contains("shfl.sync.down.b32 %r20, %r19, 16, 31, 0xffffffff", first);
        Assert.Contains("bar.sync 0", first);
        Assert.Contains("fma.rn.f32", first);
        Assert.DoesNotContain("atom", first);
        Assert.Throws<NotSupportedException>(() => emitter.Emit(spec, 6, 1));
    }

    [Fact]
    public void ShuffleControl_EncodesClampAndSegmentMask()
    {
        Assert.Equal(0x001f,
            PtxDepthwiseConv2DWeightGradientEmitter.ShuffleControlForWidth(32));
        Assert.Equal(0x100f,
            PtxDepthwiseConv2DWeightGradientEmitter.ShuffleControlForWidth(16));
        Assert.Equal(0x1807,
            PtxDepthwiseConv2DWeightGradientEmitter.ShuffleControlForWidth(8));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            PtxDepthwiseConv2DWeightGradientEmitter.ShuffleControlForWidth(64));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            PtxDepthwiseConv2DWeightGradientEmitter.ShuffleControlForWidth(12));
    }

    [SkippableFact]
    public unsafe void DepthwiseWeightGradient_MatchesInterpreterOnDevice()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Direct PTX runtime is unavailable.");
        var spec = CodegenKernelCatalog.Find("depthwise_conv2d_3x3_bwd_weights")!.Verify;
        double[][] inputs = CreateInputs(spec, out var host);
        double[] expected = spec.Interpret(inputs);

        using var runtime = new DirectPtxRuntime();
        Skip.If(runtime.ComputeCapabilityMajor < 7,
            "The cooperative depthwise weight gradient requires sm_70+.");
        var emitter = new PtxDepthwiseConv2DWeightGradientEmitter();
        string ptx = emitter.Emit(
            spec, runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor);
        using var module = runtime.LoadModule(ptx, allowExperimentalJitFallback: true);
        IntPtr function = module.GetFunction(spec.Name, out _);
        using var first = runtime.AllocateBytes((nuint)(host[0].Length * sizeof(float)));
        using var second = runtime.AllocateBytes((nuint)(host[1].Length * sizeof(float)));
        using var output = runtime.AllocateBytes((nuint)(expected.Length * sizeof(float)));
        first.Upload<float>(host[0]);
        second.Upload<float>(host[1]);
        LaunchThree(module, function, first.Pointer, second.Pointer, output.Pointer,
            emitter.LaunchBlocks, checked((uint)emitter.LaunchBlockThreads));
        runtime.Synchronize();

        var actual = new float[expected.Length];
        output.Download<float>(actual);
        AssertClose(expected, actual, 3e-4, "depthwise weight-gradient verify shape");
    }

    [SkippableFact]
    public unsafe void DepthwiseWeightGradient_MatchesAffineAtBenchShape()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Direct PTX runtime is unavailable.");
        var spec = CodegenKernelCatalog.Find("depthwise_conv2d_3x3_bwd_weights")!.Bench;
        _ = CreateInputs(spec, out var host);

        using var runtime = new DirectPtxRuntime();
        Skip.If(runtime.ComputeCapabilityMajor < 7,
            "The cooperative depthwise weight gradient requires sm_70+.");
        using var first = runtime.AllocateBytes((nuint)(host[0].Length * sizeof(float)));
        using var second = runtime.AllocateBytes((nuint)(host[1].Length * sizeof(float)));
        using var cooperativeOutput = runtime.AllocateBytes(
            (nuint)(spec.Output.ElementCount * sizeof(float)));
        using var affineOutput = runtime.AllocateBytes(
            (nuint)(spec.Output.ElementCount * sizeof(float)));
        first.Upload<float>(host[0]);
        second.Upload<float>(host[1]);

        var cooperative = new PtxDepthwiseConv2DWeightGradientEmitter();
        string cooperativePtx = cooperative.Emit(
            spec, runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor);
        using var cooperativeModule = runtime.LoadModule(
            cooperativePtx, allowExperimentalJitFallback: true);
        IntPtr cooperativeFunction = cooperativeModule.GetFunction(spec.Name, out _);
        LaunchThree(cooperativeModule, cooperativeFunction,
            first.Pointer, second.Pointer, cooperativeOutput.Pointer,
            cooperative.LaunchBlocks, checked((uint)cooperative.LaunchBlockThreads));

        var affine = new PtxAffineEmitter();
        string affinePtx = affine.Emit(
            spec, runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor);
        using var affineModule = runtime.LoadModule(affinePtx, allowExperimentalJitFallback: true);
        IntPtr affineFunction = affineModule.GetFunction(spec.Name, out _);
        LaunchThree(affineModule, affineFunction,
            first.Pointer, second.Pointer, affineOutput.Pointer,
            affine.LaunchBlocks, checked((uint)affine.LaunchBlockX));
        runtime.Synchronize();

        var expected = new float[spec.Output.ElementCount];
        var actual = new float[spec.Output.ElementCount];
        affineOutput.Download<float>(expected);
        cooperativeOutput.Download<float>(actual);
        AssertClose(expected, actual, 2e-3,
            "depthwise weight-gradient benchmark shape", relative: true);
    }

    private static double[][] CreateInputs(CodegenKernelSpec spec, out float[][] host)
    {
        var inputs = new double[spec.Inputs.Count][];
        host = new float[spec.Inputs.Count][];
        for (int i = 0; i < spec.Inputs.Count; i++)
        {
            host[i] = new float[spec.Inputs[i].ElementCount];
            inputs[i] = new double[host[i].Length];
            for (int e = 0; e < host[i].Length; e++)
            {
                host[i][e] = (float)((((e * 37 + i * 101) % 97) - 48) / 64.0);
                inputs[i][e] = host[i][e];
            }
        }
        return inputs;
    }

    private static void AssertClose(
        double[] expected, float[] actual, double tolerance, string label,
        bool relative = false)
    {
        double worst = 0;
        int at = 0;
        for (int i = 0; i < actual.Length; i++)
        {
            double difference = Math.Abs(expected[i] - actual[i]);
            if (relative) difference /= Math.Max(1.0, Math.Abs(expected[i]));
            if (difference > worst) { worst = difference; at = i; }
        }
        Assert.True(worst < tolerance,
            $"{label} differs by {worst:E3} at {at}: " +
            $"expected {expected[at]}, actual {actual[at]}");
    }

    private static void AssertClose(
        float[] expected, float[] actual, double tolerance, string label,
        bool relative = false)
    {
        var widened = new double[expected.Length];
        for (int i = 0; i < expected.Length; i++) widened[i] = expected[i];
        AssertClose(widened, actual, tolerance, label, relative);
    }

    private static unsafe void LaunchThree(
        DirectPtxModule module, IntPtr function,
        IntPtr first, IntPtr second, IntPtr output,
        uint blocks, uint blockX)
    {
        IntPtr p0 = first, p1 = second, p2 = output;
        void** arguments = stackalloc void*[3];
        arguments[0] = &p0;
        arguments[1] = &p1;
        arguments[2] = &p2;
        module.Launch(function, blocks, 1, 1, blockX, 1, 1, 0, arguments);
    }
}
