// Copyright (c) AiDotNet. All rights reserved.

using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ptx;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;
using Xunit;
using static AiDotNet.Tensors.Tests.Engines.Codegen.TiledPtxTestHelper;

namespace AiDotNet.Tensors.Tests.Engines.Codegen;

public sealed class CodegenTiledConv2DTests
{
    [Fact]
    public void DenseForward_RecoversDirectWindowAndEpilogue()
    {
        var entry = CodegenKernelCatalog.Find("conv2d_3x3_bias_relu")!;

        Assert.True(CodegenTiledConv2DPlan.TryCreate(
            entry.Verify, out var verify, out string verifyReason), verifyReason);
        Assert.NotNull(verify);
        Assert.Equal((2, 8, 16, 16, 8),
            (verify!.Batch, verify.M, verify.OutputHeight,
             verify.OutputWidth, verify.ReductionChannels));
        Assert.Equal((8, 4, 1, 4, 32),
            (verify.TileM, verify.TileChannels, verify.ThreadTileM,
             verify.ThreadTileWidth, verify.BlockThreads));
        Assert.Equal((1, -1), (verify.TapSign, verify.WindowConstant));
        Assert.False(verify.MatrixReductionMajor);
        Assert.Equal(entry.Verify.BiasInput, verify.BiasInput);

        Assert.True(CodegenTiledConv2DPlan.TryCreate(
            entry.Bench, out var bench, out string benchReason), benchReason);
        Assert.NotNull(bench);
        Assert.Equal((8, 64, 28, 28, 32),
            (bench!.Batch, bench.M, bench.OutputHeight,
             bench.OutputWidth, bench.ReductionChannels));
        Assert.Equal((32, 4, 2, 4, 112),
            (bench.TileM, bench.TileChannels, bench.ThreadTileM,
             bench.ThreadTileWidth, bench.BlockThreads));
        Assert.Equal(448, bench.Blocks);
        Assert.Equal(2 * (32 * 4 * 9 + 4 * 3 * 28) * sizeof(float),
            bench.SharedMemoryBytes);
    }

    [Fact]
    public void DenseBackwardData_RecoversAdjointWindowAndWeightLayout()
    {
        var entry = CodegenKernelCatalog.Find("conv2d_3x3_bwd_data")!;

        Assert.True(CodegenTiledConv2DPlan.TryCreate(
            entry.Verify, out var verify, out string verifyReason), verifyReason);
        Assert.NotNull(verify);
        Assert.Equal((2, 8, 16, 16, 8),
            (verify!.Batch, verify.M, verify.OutputHeight,
             verify.OutputWidth, verify.ReductionChannels));
        Assert.Equal((-1, 1), (verify.TapSign, verify.WindowConstant));
        Assert.True(verify.MatrixReductionMajor);
        Assert.Null(verify.BiasInput);

        Assert.True(CodegenTiledConv2DPlan.TryCreate(
            entry.Bench, out var bench, out string benchReason), benchReason);
        Assert.NotNull(bench);
        Assert.Equal((8, 32, 28, 28, 64),
            (bench!.Batch, bench.M, bench.OutputHeight,
             bench.OutputWidth, bench.ReductionChannels));
        Assert.Equal((32, 4, 2, 4, 112),
            (bench.TileM, bench.TileChannels, bench.ThreadTileM,
             bench.ThreadTileWidth, bench.BlockThreads));
        Assert.Equal(224, bench.Blocks);
        Assert.True(bench.MatrixReductionMajor);
    }

    [Fact]
    public void DenseForward_RefusesStaticSharedMemoryOverBudget()
    {
        var source = CodegenKernelCatalog.Find("conv2d_3x3_bias_relu")!.Bench;
        Assert.True(CodegenTiledConv2DPlan.TryCreate(
            source, out var sourcePlan, out string sourceReason), sourceReason);

        const int width = 512;
        var axes = CopyAxes(source);
        axes[sourcePlan!.ColumnAxis] = CodegenAxis.Parallel(
            axes[sourcePlan.ColumnAxis].Name, width);
        var inputs = CopyInputs(source);
        inputs[sourcePlan.StreamInput] = WithShapeDimension(
            inputs[sourcePlan.StreamInput], 3, width);
        CodegenTensorBinding output = WithShapeDimension(source.Output, 3, width);
        var widened = new CodegenKernelSpec(
            source.Name, new CodegenIterationSpace(axes), inputs, output,
            CopyProductInputs(source), source.Reduce,
            biasInput: source.BiasInput,
            scaleInput: source.ScaleInput,
            activation: source.Activation,
            reduceScale: source.ReduceScale,
            preReduce: source.PreReduce,
            preBiasInput: source.PreBiasInput,
            preBiasScale: source.PreBiasScale,
            algebra: source.Algebra);

        Assert.False(CodegenTiledConv2DPlan.TryCreate(
            widened, out var plan, out string reason));
        Assert.Null(plan);
        Assert.Equal(
            "58368 bytes of static shared memory exceed the 49152-byte budget",
            reason);
    }

    [Fact]
    public void DenseForward_EmitsAsyncZeroFillAndExactEpilogue()
    {
        var spec = CodegenKernelCatalog.Find("conv2d_3x3_bias_relu")!.Bench;
        var emitter = new PtxTiledConv2DEmitter();

        string ptx = emitter.Emit(spec, 8, 6);

        Assert.Equal(448u, emitter.LaunchBlocks);
        Assert.Equal(112, emitter.LaunchBlockThreads);
        Assert.StartsWith(".version 7.5", ptx, StringComparison.Ordinal);
        Assert.Contains("cp.async.ca.shared.global", ptx);
        Assert.Contains(", 16, %p", ptx);
        Assert.Contains("cp.async.wait_group 0", ptx);
        Assert.Contains("ld.shared.v2.f32", ptx);
        Assert.Contains("ld.shared.v4.f32", ptx);
        Assert.Contains("fma.rn.f32", ptx);
        Assert.Contains("add.rn.f32", ptx);
        Assert.Contains("max.f32", ptx);
        Assert.DoesNotContain("mma.sync", ptx);
    }

    [SkippableTheory]
    [InlineData("conv2d_3x3_bias_relu")]
    [InlineData("conv2d_3x3_bwd_data")]
    public unsafe void DenseWindow_MatchesInterpreterOnDevice(string kernel)
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Direct PTX runtime is unavailable.");
        var spec = CodegenKernelCatalog.Find(kernel)!.Verify;
        var inputs = CreateInputs(spec, out var host);
        double[] expected = spec.Interpret(inputs);

        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(runtime.ComputeCapabilityMajor >= 8, "cp.async requires sm_80 or later.");
        var emitter = new PtxTiledConv2DEmitter();
        string ptx = emitter.Emit(
            spec, runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor);
        using var module = runtime.LoadModule(ptx, allowExperimentalJitFallback: true);
        IntPtr function = module.GetFunction(spec.Name, out _);
        using var first = runtime.AllocateBytes((nuint)(host[0].Length * sizeof(float)));
        using var second = runtime.AllocateBytes((nuint)(host[1].Length * sizeof(float)));
        using var output = runtime.AllocateBytes((nuint)(expected.Length * sizeof(float)));
        first.Upload<float>(host[0]);
        second.Upload<float>(host[1]);

        if (host.Length == 3)
        {
            using var third = runtime.AllocateBytes((nuint)(host[2].Length * sizeof(float)));
            third.Upload<float>(host[2]);
            LaunchFour(module, function, first.Pointer, second.Pointer, third.Pointer,
                output.Pointer, emitter.LaunchBlocks,
                checked((uint)emitter.LaunchBlockThreads), 1);
        }
        else
        {
            LaunchThree(module, function, first.Pointer, second.Pointer,
                output.Pointer, emitter.LaunchBlocks,
                checked((uint)emitter.LaunchBlockThreads), 1);
        }
        runtime.Synchronize();

        var actual = new float[expected.Length];
        output.Download<float>(actual);
        AssertClose(expected, actual, 3e-4, kernel + " tiled 3x3");
    }

    [SkippableTheory]
    [InlineData("conv2d_3x3_bias_relu")]
    [InlineData("conv2d_3x3_bwd_data")]
    public unsafe void DenseWindow_DoubleBufferMatchesAffineAtBenchShape(string kernel)
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Direct PTX runtime is unavailable.");
        var spec = CodegenKernelCatalog.Find(kernel)!.Bench;
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(runtime.ComputeCapabilityMajor >= 8, "cp.async requires sm_80 or later.");
        _ = CreateInputs(spec, out var host);

        using var first = runtime.AllocateBytes((nuint)(host[0].Length * sizeof(float)));
        using var second = runtime.AllocateBytes((nuint)(host[1].Length * sizeof(float)));
        using var tiledOutput = runtime.AllocateBytes(
            (nuint)(spec.Output.ElementCount * sizeof(float)));
        using var affineOutput = runtime.AllocateBytes(
            (nuint)(spec.Output.ElementCount * sizeof(float)));
        first.Upload<float>(host[0]);
        second.Upload<float>(host[1]);

        var tiled = new PtxTiledConv2DEmitter();
        string tiledPtx = tiled.Emit(
            spec, runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor);
        using var tiledModule = runtime.LoadModule(tiledPtx, allowExperimentalJitFallback: true);
        IntPtr tiledFunction = tiledModule.GetFunction(spec.Name, out _);

        var affine = new PtxAffineEmitter();
        string affinePtx = affine.Emit(
            spec, runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor);
        using var affineModule = runtime.LoadModule(affinePtx, allowExperimentalJitFallback: true);
        IntPtr affineFunction = affineModule.GetFunction(spec.Name, out _);

        if (host.Length == 3)
        {
            using var third = runtime.AllocateBytes((nuint)(host[2].Length * sizeof(float)));
            third.Upload<float>(host[2]);
            LaunchFour(tiledModule, tiledFunction, first.Pointer, second.Pointer,
                third.Pointer, tiledOutput.Pointer, tiled.LaunchBlocks,
                checked((uint)tiled.LaunchBlockThreads), 1);
            LaunchFour(affineModule, affineFunction, first.Pointer, second.Pointer,
                third.Pointer, affineOutput.Pointer, affine.LaunchBlocks,
                checked((uint)affine.LaunchBlockX), checked((uint)affine.LaunchBlockY));
        }
        else
        {
            LaunchThree(tiledModule, tiledFunction, first.Pointer, second.Pointer,
                tiledOutput.Pointer, tiled.LaunchBlocks,
                checked((uint)tiled.LaunchBlockThreads), 1);
            LaunchThree(affineModule, affineFunction, first.Pointer, second.Pointer,
                affineOutput.Pointer, affine.LaunchBlocks,
                checked((uint)affine.LaunchBlockX), checked((uint)affine.LaunchBlockY));
        }
        runtime.Synchronize();

        var expected = new float[spec.Output.ElementCount];
        var actual = new float[spec.Output.ElementCount];
        affineOutput.Download<float>(expected);
        tiledOutput.Download<float>(actual);
        AssertClose(expected, actual, 5e-5,
            kernel + " benchmark tiled 3x3", relative: true);
    }

}
