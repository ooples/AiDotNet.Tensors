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
        Assert.Equal((8, 4, 8, 1, 4, 128),
            (verify.TileM, verify.TileRows, verify.TileChannels, verify.ThreadTileM,
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
        Assert.Equal((64, 4, 8, 8, 4, 224),
            (bench.TileM, bench.TileRows, bench.TileChannels, bench.ThreadTileM,
             bench.ThreadTileWidth, bench.BlockThreads));
        Assert.Equal(56, bench.Blocks);
        Assert.Equal(2 * (64 * 8 * 9 + 8 * 6 * 28) * sizeof(float),
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
        Assert.Equal((16, 4, 16, 4, 4, 128),
            (bench.TileM, bench.TileRows, bench.TileChannels, bench.ThreadTileM,
             bench.ThreadTileWidth, bench.BlockThreads));
        Assert.Equal(112, bench.Blocks);
        Assert.Equal(2 * (16 * 16 * 9 + 16 * 6 * 28) * sizeof(float),
            bench.SharedMemoryBytes);
        Assert.True(bench.MatrixReductionMajor);
    }

    [Fact]
    public void ExactSchedule_IsNamedValidatedAndReplayed()
    {
        var spec = CodegenKernelCatalog.Find("conv2d_3x3_bias_relu")!.Bench;
        const string winner = "tiled-conv2d:m16r7c8tm4";
        CodegenTiledConv2DSchedule? schedule =
            CodegenTiledConv2DSchedule.Find(winner);

        Assert.NotNull(schedule);
        Assert.Equal(winner, schedule!.WinnerName);
        Assert.True(CodegenTiledConv2DPlan.TryCreate(
            spec, schedule, out var plan, out string reason), reason);
        Assert.NotNull(plan);
        Assert.Equal((16, 7, 8, 4, 224, 128, 25344),
            (plan!.TileM, plan.TileRows, plan.TileChannels, plan.ThreadTileM,
             plan.BlockThreads, plan.Blocks, plan.SharedMemoryBytes));

        var emitter = new PtxTiledConv2DEmitter(schedule);
        string ptx = emitter.Emit(spec, 8, 6);
        Assert.Equal(128u, emitter.LaunchBlocks);
        Assert.Contains("fma.rn.f32", ptx);

        CodegenTiledConv2DSchedule? forward = CodegenTiledConv2DSchedule.Find(
            "tiled-conv2d:m16r14c8tm8");
        Assert.NotNull(forward);
        Assert.True(CodegenTiledConv2DPlan.TryCreate(
            spec, forward!, out var forwardPlan, out string forwardReason),
            forwardReason);
        Assert.Equal((14, 8, 224, 64, 37888),
            (forwardPlan!.TileRows, forwardPlan.ThreadTileM,
             forwardPlan.BlockThreads, forwardPlan.Blocks,
             forwardPlan.SharedMemoryBytes));

        var invalid = new CodegenTiledConv2DSchedule(32, 3, 8, 4);
        Assert.False(CodegenTiledConv2DPlan.TryCreate(
            spec, invalid, out var rejected, out string rejectedReason));
        Assert.Null(rejected);
        Assert.Equal(
            "the exact schedule must divide M, rows, channels, and its thread tile",
            rejectedReason);
    }

    [Fact]
    public void ExactSplitSchedule_RebuildsBothPromotableKernels()
    {
        var spec = CodegenKernelCatalog.Find("conv2d_3x3_bias_relu")!.Bench;
        const string winner = "tiled-conv2d:m16r7c8tm4:sk2";
        CodegenTiledConv2DSplitSchedule? schedule =
            CodegenTiledConv2DSplitSchedule.Find(winner);

        Assert.NotNull(schedule);
        Assert.Equal(winner, schedule!.WinnerName);
        Assert.True(CodegenTiledConv2DSplitPlan.TryCreate(
            spec, schedule, out var exact, out string reason), reason);
        Assert.NotNull(exact);
        Assert.Equal((2, 16, 32, 256),
            (exact!.PartialPlan.SplitFactor,
             exact.PartialPlan.ReductionChannels,
             exact.PartialPlan.PhysicalReductionChannels,
             exact.PartialPlan.Blocks));
        Assert.Equal(2, exact.Split.Partial.Output.Shape[^1]);

        var partial = new PtxTiledConv2DEmitter(schedule.Tile);
        string partialPtx = partial.Emit(exact.Split.Partial, 8, 6);
        var combine = new PtxAffineEmitter();
        string combinePtx = combine.Emit(exact.Split.Combine, 8, 6);
        Assert.Contains("fma.rn.f32", partialPtx);
        Assert.Contains("ld.global.nc.f32", combinePtx);
        Assert.Contains("add.rn.f32", combinePtx);
        Assert.Contains("max.f32", combinePtx);

        CodegenAutotuneIdentity identity = CodegenAutotuneIdentity.Create(
            spec, "test-device", 8, 6);
        Assert.StartsWith("ptxset-sha256-", identity.EmitterFingerprint,
            StringComparison.Ordinal);
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
            "233472 bytes of static shared memory exceed the 49152-byte budget",
            reason);
    }

    [Fact]
    public void DenseForward_EmitsAsyncZeroFillAndExactEpilogue()
    {
        var spec = CodegenKernelCatalog.Find("conv2d_3x3_bias_relu")!.Bench;
        var emitter = new PtxTiledConv2DEmitter();

        string ptx = emitter.Emit(spec, 8, 6);

        Assert.Equal(56u, emitter.LaunchBlocks);
        Assert.Equal(224, emitter.LaunchBlockThreads);
        Assert.StartsWith(".version 7.5", ptx, StringComparison.Ordinal);
        Assert.Contains("cp.async.ca.shared.global", ptx);
        Assert.Contains(", 16, %p", ptx);
        Assert.Contains("cp.async.wait_group 0", ptx);
        Assert.Contains("ld.shared.v4.f32", ptx);
        Assert.Contains("fma.rn.f32", ptx);
        Assert.Contains("add.rn.f32", ptx);
        Assert.Contains("max.f32", ptx);
        Assert.Contains("st.global.f32", ptx);
        Assert.DoesNotContain("st.global.v4.f32", ptx);
        Assert.DoesNotContain("mma.sync", ptx);
    }

    [Fact]
    public void DenseBackwardData_VectorizesBiasFreeStores()
    {
        var spec = CodegenKernelCatalog.Find("conv2d_3x3_bwd_data")!.Bench;
        var emitter = new PtxTiledConv2DEmitter();

        string ptx = emitter.Emit(spec, 8, 6);

        Assert.Contains("st.global.v4.f32", ptx);
        Assert.DoesNotContain("st.global.f32", ptx);
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
    [InlineData("conv2d_3x3_bias_relu", null)]
    [InlineData("conv2d_3x3_bwd_data", null)]
    [InlineData("conv2d_3x3_bias_relu", "tiled-conv2d:m16r7c8tm4:wh:ds")]
    public unsafe void DenseWindow_DoubleBufferMatchesAffineAtBenchShape(
        string kernel, string? exactWinner)
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

        CodegenTiledConv2DSchedule? exact = exactWinner is null
            ? null
            : CodegenTiledConv2DSchedule.Find(exactWinner);
        Assert.True(exactWinner is null || exact is not null,
            "The requested exact schedule must remain in the measured search space.");
        var tiled = exact is null
            ? new PtxTiledConv2DEmitter()
            : new PtxTiledConv2DEmitter(exact);
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
            kernel + " benchmark tiled 3x3 " + (exactWinner ?? "modelled"),
            relative: true);
    }

}
