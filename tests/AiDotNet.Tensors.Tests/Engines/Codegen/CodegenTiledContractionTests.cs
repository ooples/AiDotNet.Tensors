// Copyright (c) AiDotNet. All rights reserved.

using System.Linq;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ptx;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Codegen;

public sealed class CodegenTiledContractionTests
{
    /// <summary>The matcher follows the derived maps, not the catalog label.</summary>
    [Fact]
    public void BackwardDataPointwise_RecoversWholeTiledContraction()
    {
        var entry = CodegenKernelCatalog.Find("conv2d_1x1_bwd_data")!;

        Assert.True(CodegenTiledContractionPlan.TryCreate(
            entry.Verify, out var verify, out string verifyReason), verifyReason);
        Assert.NotNull(verify);
        Assert.Equal((2, 8, 256, 8), (verify!.Batch, verify.M, verify.N, verify.K));
        Assert.Equal((8, 64, 8), (verify.TileM, verify.TileN, verify.TileK));
        Assert.Equal(128, verify.BlockThreads);
        Assert.True(verify.MatrixReductionMajor);

        Assert.True(CodegenTiledContractionPlan.TryCreate(
            entry.Bench, out var bench, out string benchReason), benchReason);
        Assert.NotNull(bench);
        Assert.Equal((16, 64, 784, 64), (bench!.Batch, bench.M, bench.N, bench.K));
        Assert.Equal((64, 56, 8), (bench.TileM, bench.TileN, bench.TileK));
        Assert.Equal((8, 2), (bench.ThreadTileM, bench.ThreadTileN));
        Assert.Equal(224, bench.BlockThreads);
        Assert.Equal(16 * 1 * 14, bench.Blocks);
        Assert.Equal(2 * 8 * (64 + 56) * sizeof(float), bench.SharedMemoryBytes);
    }

    /// <summary>Forward is recovered from [M,K] weights and its exact fused epilogue.</summary>
    [Fact]
    public void ForwardPointwise_RecoversMmajorContractionWithBiasRelu()
    {
        var entry = CodegenKernelCatalog.Find("conv2d_1x1_bias_relu")!;

        Assert.True(CodegenTiledContractionPlan.TryCreate(
            entry.Verify, out var verify, out string verifyReason), verifyReason);
        Assert.NotNull(verify);
        Assert.Equal((2, 8, 256, 8), (verify!.Batch, verify.M, verify.N, verify.K));
        Assert.False(verify.MatrixReductionMajor);
        Assert.Equal(entry.Verify.BiasInput, verify.BiasInput);

        Assert.True(CodegenTiledContractionPlan.TryCreate(
            entry.Bench, out var bench, out string benchReason), benchReason);
        Assert.NotNull(bench);
        Assert.Equal((16, 64, 784, 64), (bench!.Batch, bench.M, bench.N, bench.K));
        Assert.False(bench.MatrixReductionMajor);
        Assert.Equal((32, 56, 8), (bench.TileM, bench.TileN, bench.TileK));
    }

    /// <summary>The same semantic tile accepts a per-M post-bias scale.</summary>
    [Fact]
    public void DeepEpilogue_RecoversBiasScaleReluContraction()
    {
        var entry = CodegenKernelCatalog.Find("conv2d_1x1_deep_epilogue")!;

        Assert.True(CodegenTiledContractionPlan.TryCreate(
            entry.Bench, out var plan, out string reason), reason);
        Assert.NotNull(plan);
        Assert.Equal(entry.Bench.BiasInput, plan!.BiasInput);
        Assert.Equal(entry.Bench.ScaleInput, plan.ScaleInput);
        Assert.Equal((32, 56, 8), (plan.TileM, plan.TileN, plan.TileK));
    }

    /// <summary>A depthwise stencil is not silently reinterpreted as a dense matrix product.</summary>
    [Fact]
    public void DepthwiseStencil_IsRefusedWithReason()
    {
        var entry = CodegenKernelCatalog.Find("depthwise_conv2d_3x3")!;

        Assert.False(CodegenTiledContractionPlan.TryCreate(
            entry.Bench, out var plan, out string reason));
        Assert.Null(plan);
        Assert.Contains("one contraction axis", reason);
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void BindingShapeThatDisagreesWithMappedAxis_IsRefused(bool matrix)
    {
        var original = CodegenKernelCatalog.Find("conv2d_1x1_bwd_data")!.Verify;
        Assert.True(CodegenTiledContractionPlan.TryCreate(
            original, out var valid, out string validReason), validReason);
        int input = matrix ? valid!.MatrixInput : valid!.StreamInput;
        var malformed = WithShortenedInput(original, input);

        Assert.False(CodegenTiledContractionPlan.TryCreate(
            malformed, out var plan, out string reason));
        Assert.Null(plan);
        Assert.Contains(matrix ? "plain rank-2 matrix" : "output layout", reason);
    }

    [Theory]
    [InlineData("conv2d_1x1_bias_relu")]
    [InlineData("conv2d_1x1_bwd_data")]
    [InlineData("conv2d_1x1_deep_epilogue")]
    public void SelectedTiles_AreWholeAndCpAsyncAligned(string kernel)
    {
        var entry = CodegenKernelCatalog.Find(kernel)!;
        foreach (var spec in new[] { entry.Verify, entry.Bench })
        {
            Assert.True(CodegenTiledContractionPlan.TryCreate(
                spec, out var plan, out string reason), reason);
            Assert.Equal(0, plan!.M % plan.TileM);
            Assert.Equal(0, plan.N % plan.TileN);
            Assert.Equal(0, plan.K % plan.TileK);
            Assert.Equal(0, plan.TileN % 4);
            Assert.Equal(0,
                (plan.MatrixReductionMajor ? plan.TileM : plan.TileK) % 4);
            Assert.Equal(0, plan.StageBytes % 16);
        }
    }

    /// <summary>Measured schedules have stable names and preserve whole-copy invariants.</summary>
    [Fact]
    public void ExactSchedule_RebuildsTheNamedTile()
    {
        const string winner = "tiled-contraction:m64n56k16tm8tn2";
        CodegenTiledContractionSchedule? schedule =
            CodegenTiledContractionSchedule.Find(winner);
        Assert.NotNull(schedule);
        var spec = CodegenKernelCatalog.Find("conv2d_1x1_bias_relu")!.Bench;

        Assert.True(CodegenTiledContractionPlan.TryCreate(
            spec, schedule, out var plan, out string reason), reason);
        Assert.NotNull(plan);
        Assert.Equal((64, 56, 16, 8, 2),
            (plan!.TileM, plan.TileN, plan.TileK, plan.ThreadTileM, plan.ThreadTileN));

        var emitter = new PtxTiledContractionEmitter(schedule!);
        string ptx = emitter.Emit(spec, 8, 6);
        Assert.Contains("tile 64x56x16, thread tile 8x2", ptx);
        Assert.Equal((uint)plan.Blocks, emitter.LaunchBlocks);
    }

    [Fact]
    public void RegisterPrefetchSchedule_LoadsTwoKFragmentsBeforeFirstFma()
    {
        const string winner = "tiled-contraction:m64n56k16tm8tn2:rp";
        CodegenTiledContractionSchedule? schedule =
            CodegenTiledContractionSchedule.Find(winner);
        Assert.NotNull(schedule);
        Assert.True(schedule!.RegisterPrefetch);
        Assert.Equal(winner, schedule.WinnerName);

        var spec = CodegenKernelCatalog.Find("conv2d_1x1_bwd_data")!.Bench;
        var ordinary = new PtxTiledContractionEmitter(
            new CodegenTiledContractionSchedule(64, 56, 16, 8, 2));
        var prefetched = new PtxTiledContractionEmitter(schedule);
        string ordinaryPtx = ordinary.Emit(spec, 8, 6);
        string prefetchedPtx = prefetched.Emit(spec, 8, 6);

        Assert.True(prefetched.Plan!.RegisterPrefetch);
        Assert.Contains("register-prefetched", prefetchedPtx);
        Assert.Equal(
            2 * SharedLoadsBeforeFirstFma(ordinaryPtx),
            SharedLoadsBeforeFirstFma(prefetchedPtx));
    }

    [Fact]
    public void ExactSchedule_RefusesNonDivisibleShape()
    {
        var schedule = new CodegenTiledContractionSchedule(64, 112, 8, 8, 4);
        var spec = CodegenKernelCatalog.Find("conv2d_1x1_bias_relu")!.Verify;

        Assert.False(CodegenTiledContractionPlan.TryCreate(
            spec, schedule, out var plan, out string reason));
        Assert.Null(plan);
        Assert.Contains("whole, 16-byte-aligned", reason);
    }

    [Fact]
    public void ExactSchedule_RefusesOverBudgetSharedMemory()
    {
        var schedule = new CodegenTiledContractionSchedule(64, 112, 64, 8, 4);
        var spec = CodegenKernelCatalog.Find("conv2d_1x1_bias_relu")!.Bench;

        Assert.False(CodegenTiledContractionPlan.TryCreate(
            spec, schedule, out var plan, out string reason));
        Assert.Null(plan);
        Assert.Contains("90112 bytes", reason);
        Assert.Contains("49152-byte budget", reason);
    }

    [Fact]
    public void UnsupportedActivation_IsRefused()
    {
        var spec = SimpleContraction(8, CodegenActivationKind.Sigmoid);

        Assert.False(CodegenTiledContractionPlan.TryCreate(
            spec, out var plan, out string reason));
        Assert.Null(plan);
        Assert.Contains("optional M bias/scale and ReLU", reason);
    }

    [Fact]
    public void ExtentWithoutFourWideWholeTile_IsRefused()
    {
        var spec = SimpleContraction(6, CodegenActivationKind.None);

        Assert.False(CodegenTiledContractionPlan.TryCreate(
            spec, out var plan, out string reason));
        Assert.Null(plan);
        Assert.Contains("no supported whole tile", reason);
    }

    /// <summary>The emitted candidate stages both operands and uses only SIMT FP32 FMA.</summary>
    [Fact]
    public void BackwardDataPointwise_EmitsDoubleBufferedTrueFp32Ptx()
    {
        var spec = CodegenKernelCatalog.Find("conv2d_1x1_bwd_data")!.Bench;
        var emitter = new PtxTiledContractionEmitter();

        string ptx = emitter.Emit(spec, 8, 6);

        Assert.Equal(224u, emitter.LaunchBlocks);
        Assert.Equal(224, emitter.LaunchBlockThreads);
        Assert.Contains("cp.async.ca.shared.global", ptx);
        Assert.Contains("cp.async.wait_group 0", ptx);
        Assert.Contains("fma.rn.f32", ptx);
        Assert.Contains("ld.shared.v4.f32", ptx);
        Assert.Contains("ld.shared.v2.f32", ptx);
        Assert.DoesNotContain("mma.sync", ptx);
        Assert.DoesNotContain("wmma", ptx);
    }

    /// <summary>The M-major copy retains the catalog's fused bias and ReLU.</summary>
    [Fact]
    public void ForwardPointwise_EmitsMmajorCopyAndExactEpilogue()
    {
        var spec = CodegenKernelCatalog.Find("conv2d_1x1_bias_relu")!.Bench;
        var emitter = new PtxTiledContractionEmitter();

        string ptx = emitter.Emit(spec, 8, 6);

        Assert.Equal(448u, emitter.LaunchBlocks);
        Assert.Equal(224, emitter.LaunchBlockThreads);
        Assert.Contains("ld.param.u64 %rd4, [p2]", ptx);
        Assert.Contains("ld.global.f32", ptx);
        Assert.Contains("add.rn.f32", ptx);
        Assert.Contains("max.f32", ptx);
        Assert.Contains("cp.async.ca.shared.global", ptx);
        Assert.Contains("ld.shared.v2.f32", ptx);
    }

    /// <summary>The tiled epilogue preserves bias, then scale, then activation ordering.</summary>
    [Fact]
    public void DeepEpilogue_EmitsBiasThenScaleThenRelu()
    {
        var spec = CodegenKernelCatalog.Find("conv2d_1x1_deep_epilogue")!.Bench;
        var emitter = new PtxTiledContractionEmitter();

        string ptx = emitter.Emit(spec, 8, 6);

        Assert.Contains("ld.param.u64 %rd4, [p2]", ptx);
        Assert.Contains("ld.param.u64 %rd5, [p3]", ptx);
        int bias = ptx.LastIndexOf("add.rn.f32", System.StringComparison.Ordinal);
        int scale = ptx.LastIndexOf("mul.rn.f32", System.StringComparison.Ordinal);
        int relu = ptx.LastIndexOf("max.f32", System.StringComparison.Ordinal);
        Assert.True(bias >= 0 && bias < scale && scale < relu);
        string rowPattern = ", %r7, " + emitter.Plan!.ThreadTileM + ",";
        Assert.Equal(emitter.Plan.ThreadTileM,
            ptx.Split('\n').Count(line =>
                line.Contains("mad.lo.u32", System.StringComparison.Ordinal) &&
                line.Contains(rowPattern, System.StringComparison.Ordinal)));
    }

    [SkippableFact]
    public unsafe void DeepEpilogue_MatchesInterpreterOnDevice()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Direct PTX runtime is unavailable.");
        var spec = CodegenKernelCatalog.Find("conv2d_1x1_deep_epilogue")!.Verify;
        double[][] inputs = TiledPtxTestHelper.CreateInputs(spec, out float[][] host);
        double[] expected = spec.Interpret(inputs);

        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(runtime.ComputeCapabilityMajor >= 8, "cp.async requires sm_80 or later.");
        using var first = runtime.AllocateBytes((nuint)(host[0].Length * sizeof(float)));
        using var second = runtime.AllocateBytes((nuint)(host[1].Length * sizeof(float)));
        using var bias = runtime.AllocateBytes((nuint)(host[2].Length * sizeof(float)));
        using var scale = runtime.AllocateBytes((nuint)(host[3].Length * sizeof(float)));
        using var output = runtime.AllocateBytes((nuint)(expected.Length * sizeof(float)));
        first.Upload<float>(host[0]);
        second.Upload<float>(host[1]);
        bias.Upload<float>(host[2]);
        scale.Upload<float>(host[3]);

        var emitter = new PtxTiledContractionEmitter();
        string ptx = emitter.Emit(
            spec, runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor);
        using var module = runtime.LoadModule(ptx, allowExperimentalJitFallback: true);
        IntPtr function = module.GetFunction(spec.Name, out _);
        TiledPtxTestHelper.LaunchFive(module, function,
            first.Pointer, second.Pointer, bias.Pointer, scale.Pointer, output.Pointer,
            emitter.LaunchBlocks, checked((uint)emitter.LaunchBlockThreads), 1);
        runtime.Synchronize();

        var actual = new float[expected.Length];
        output.Download<float>(actual);
        TiledPtxTestHelper.AssertClose(
            expected, actual, 2e-4, "deep tiled epilogue verify shape");
    }

    /// <summary>The assembled device program agrees with the spec interpreter.</summary>
    [SkippableFact]
    public unsafe void BackwardDataPointwise_MatchesInterpreterOnDevice()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Direct PTX runtime is unavailable.");
        var spec = CodegenKernelCatalog.Find("conv2d_1x1_bwd_data")!.Verify;
        var inputs = new double[spec.Inputs.Count][];
        var host = new float[spec.Inputs.Count][];
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
        double[] expected = spec.Interpret(inputs);

        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(runtime.ComputeCapabilityMajor >= 8, "cp.async requires sm_80 or later.");
        var emitter = new PtxTiledContractionEmitter();
        string ptx = emitter.Emit(
            spec, runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor);
        using var module = runtime.LoadModule(ptx, allowExperimentalJitFallback: true);
        IntPtr function = module.GetFunction(spec.Name, out _);
        using var first = runtime.AllocateBytes((nuint)(host[0].Length * sizeof(float)));
        using var second = runtime.AllocateBytes((nuint)(host[1].Length * sizeof(float)));
        using var output = runtime.AllocateBytes((nuint)(expected.Length * sizeof(float)));
        first.Upload<float>(host[0]);
        second.Upload<float>(host[1]);

        IntPtr p0 = first.Pointer, p1 = second.Pointer, p2 = output.Pointer;
        void** arguments = stackalloc void*[3];
        arguments[0] = &p0;
        arguments[1] = &p1;
        arguments[2] = &p2;
        module.Launch(function, emitter.LaunchBlocks, 1, 1,
            checked((uint)emitter.LaunchBlockThreads), 1, 1, 0, arguments);
        runtime.Synchronize();

        var actual = new float[expected.Length];
        output.Download<float>(actual);
        double worst = 0;
        int at = 0;
        for (int i = 0; i < actual.Length; i++)
        {
            double difference = System.Math.Abs(expected[i] - actual[i]);
            if (difference > worst) { worst = difference; at = i; }
        }
        Assert.True(worst < 2e-4,
            $"tiled contraction deviates by {worst:E3} at {at}: " +
            $"expected {expected[at]}, actual {actual[at]}");
    }

    /// <summary>The [M,K] staging and fused epilogue agree with the fp64 oracle.</summary>
    [SkippableFact]
    public unsafe void ForwardPointwise_MatchesInterpreterOnDevice()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Direct PTX runtime is unavailable.");
        var spec = CodegenKernelCatalog.Find("conv2d_1x1_bias_relu")!.Verify;
        var inputs = new double[spec.Inputs.Count][];
        var host = new float[spec.Inputs.Count][];
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
        double[] expected = spec.Interpret(inputs);

        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(runtime.ComputeCapabilityMajor >= 8, "cp.async requires sm_80 or later.");
        var emitter = new PtxTiledContractionEmitter();
        string ptx = emitter.Emit(
            spec, runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor);
        using var module = runtime.LoadModule(ptx, allowExperimentalJitFallback: true);
        IntPtr function = module.GetFunction(spec.Name, out _);
        using var input = runtime.AllocateBytes((nuint)(host[0].Length * sizeof(float)));
        using var weights = runtime.AllocateBytes((nuint)(host[1].Length * sizeof(float)));
        using var bias = runtime.AllocateBytes((nuint)(host[2].Length * sizeof(float)));
        using var output = runtime.AllocateBytes((nuint)(expected.Length * sizeof(float)));
        input.Upload<float>(host[0]);
        weights.Upload<float>(host[1]);
        bias.Upload<float>(host[2]);

        IntPtr p0 = input.Pointer, p1 = weights.Pointer, p2 = bias.Pointer, p3 = output.Pointer;
        void** arguments = stackalloc void*[4];
        arguments[0] = &p0;
        arguments[1] = &p1;
        arguments[2] = &p2;
        arguments[3] = &p3;
        module.Launch(function, emitter.LaunchBlocks, 1, 1,
            checked((uint)emitter.LaunchBlockThreads), 1, 1, 0, arguments);
        runtime.Synchronize();

        var actual = new float[expected.Length];
        output.Download<float>(actual);
        double worst = 0;
        int at = 0;
        for (int i = 0; i < actual.Length; i++)
        {
            double difference = System.Math.Abs(expected[i] - actual[i]);
            if (difference > worst) { worst = difference; at = i; }
        }
        Assert.True(worst < 2e-4,
            $"tiled forward deviates by {worst:E3} at {at}: " +
            $"expected {expected[at]}, actual {actual[at]}");
    }

    /// <summary>The eight-stage benchmark program agrees with the established affine path.</summary>
    [SkippableFact]
    public unsafe void BackwardDataPointwise_DoubleBufferMatchesAffineAtBenchShape()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Direct PTX runtime is unavailable.");
        var spec = CodegenKernelCatalog.Find("conv2d_1x1_bwd_data")!.Bench;
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(runtime.ComputeCapabilityMajor >= 8, "cp.async requires sm_80 or later.");

        var host = new float[spec.Inputs.Count][];
        for (int i = 0; i < spec.Inputs.Count; i++)
        {
            host[i] = new float[spec.Inputs[i].ElementCount];
            for (int e = 0; e < host[i].Length; e++)
                host[i][e] = (float)((((e * 37 + i * 101) % 97) - 48) / 64.0);
        }

        using var first = runtime.AllocateBytes((nuint)(host[0].Length * sizeof(float)));
        using var second = runtime.AllocateBytes((nuint)(host[1].Length * sizeof(float)));
        using var tiledOutput = runtime.AllocateBytes(
            (nuint)(spec.Output.ElementCount * sizeof(float)));
        using var affineOutput = runtime.AllocateBytes(
            (nuint)(spec.Output.ElementCount * sizeof(float)));
        first.Upload<float>(host[0]);
        second.Upload<float>(host[1]);

        var schedule = CodegenTiledContractionSchedule.Find(
            "tiled-contraction:m64n56k16tm8tn2:rp");
        Assert.NotNull(schedule);
        var tiled = new PtxTiledContractionEmitter(schedule!);
        string tiledPtx = tiled.Emit(
            spec, runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor);
        using var tiledModule = runtime.LoadModule(tiledPtx, allowExperimentalJitFallback: true);
        IntPtr tiledFunction = tiledModule.GetFunction(spec.Name, out _);
        LaunchThree(tiledModule, tiledFunction, first.Pointer, second.Pointer,
            tiledOutput.Pointer, tiled.LaunchBlocks,
            checked((uint)tiled.LaunchBlockThreads), 1);

        var affine = new PtxAffineEmitter();
        string affinePtx = affine.Emit(
            spec, runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor);
        using var affineModule = runtime.LoadModule(affinePtx, allowExperimentalJitFallback: true);
        IntPtr affineFunction = affineModule.GetFunction(spec.Name, out _);
        LaunchThree(affineModule, affineFunction, first.Pointer, second.Pointer,
            affineOutput.Pointer, affine.LaunchBlocks,
            checked((uint)affine.LaunchBlockX), checked((uint)affine.LaunchBlockY));
        runtime.Synchronize();

        var expected = new float[spec.Output.ElementCount];
        var actual = new float[spec.Output.ElementCount];
        affineOutput.Download<float>(expected);
        tiledOutput.Download<float>(actual);
        double worst = 0;
        int at = 0;
        for (int i = 0; i < actual.Length; i++)
        {
            double scale = System.Math.Max(1.0, System.Math.Abs(expected[i]));
            double difference = System.Math.Abs(expected[i] - actual[i]) / scale;
            if (difference > worst) { worst = difference; at = i; }
        }
        Assert.True(worst < 2e-5,
            $"double-buffered contraction differs by {worst:E3} relative at {at}: " +
            $"affine {expected[at]}, tiled {actual[at]}");
    }

    /// <summary>The forward epilogue stays correct across every benchmark output tile.</summary>
    [SkippableFact]
    public unsafe void ForwardPointwise_DoubleBufferMatchesAffineAtBenchShape()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Direct PTX runtime is unavailable.");
        var spec = CodegenKernelCatalog.Find("conv2d_1x1_bias_relu")!.Bench;
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(runtime.ComputeCapabilityMajor >= 8, "cp.async requires sm_80 or later.");

        var host = new float[spec.Inputs.Count][];
        for (int i = 0; i < spec.Inputs.Count; i++)
        {
            host[i] = new float[spec.Inputs[i].ElementCount];
            for (int e = 0; e < host[i].Length; e++)
                host[i][e] = (float)((((e * 37 + i * 101) % 97) - 48) / 64.0);
        }

        using var input = runtime.AllocateBytes((nuint)(host[0].Length * sizeof(float)));
        using var weights = runtime.AllocateBytes((nuint)(host[1].Length * sizeof(float)));
        using var bias = runtime.AllocateBytes((nuint)(host[2].Length * sizeof(float)));
        using var tiledOutput = runtime.AllocateBytes(
            (nuint)(spec.Output.ElementCount * sizeof(float)));
        using var affineOutput = runtime.AllocateBytes(
            (nuint)(spec.Output.ElementCount * sizeof(float)));
        input.Upload<float>(host[0]);
        weights.Upload<float>(host[1]);
        bias.Upload<float>(host[2]);

        var tiled = new PtxTiledContractionEmitter();
        string tiledPtx = tiled.Emit(
            spec, runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor);
        using var tiledModule = runtime.LoadModule(tiledPtx, allowExperimentalJitFallback: true);
        IntPtr tiledFunction = tiledModule.GetFunction(spec.Name, out _);
        LaunchFour(tiledModule, tiledFunction, input.Pointer, weights.Pointer, bias.Pointer,
            tiledOutput.Pointer, tiled.LaunchBlocks,
            checked((uint)tiled.LaunchBlockThreads), 1);

        var affine = new PtxAffineEmitter();
        string affinePtx = affine.Emit(
            spec, runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor);
        using var affineModule = runtime.LoadModule(affinePtx, allowExperimentalJitFallback: true);
        IntPtr affineFunction = affineModule.GetFunction(spec.Name, out _);
        LaunchFour(affineModule, affineFunction, input.Pointer, weights.Pointer, bias.Pointer,
            affineOutput.Pointer, affine.LaunchBlocks,
            checked((uint)affine.LaunchBlockX), checked((uint)affine.LaunchBlockY));
        runtime.Synchronize();

        var expected = new float[spec.Output.ElementCount];
        var actual = new float[spec.Output.ElementCount];
        affineOutput.Download<float>(expected);
        tiledOutput.Download<float>(actual);
        double worst = 0;
        int at = 0;
        for (int i = 0; i < actual.Length; i++)
        {
            double scale = System.Math.Max(1.0, System.Math.Abs(expected[i]));
            double difference = System.Math.Abs(expected[i] - actual[i]) / scale;
            if (difference > worst) { worst = difference; at = i; }
        }
        Assert.True(worst < 2e-5,
            $"double-buffered forward differs by {worst:E3} relative at {at}: " +
            $"affine {expected[at]}, tiled {actual[at]}");
    }

    private static CodegenKernelSpec WithShortenedInput(
        CodegenKernelSpec spec, int inputIndex)
    {
        var inputs = spec.Inputs.ToArray();
        CodegenTensorBinding original = inputs[inputIndex];
        int[] shape = original.Shape.ToArray();
        shape[0]--;
        inputs[inputIndex] = new CodegenTensorBinding(
            original.ParameterIndex, original.Name, shape, original.Map.ToArray(),
            elementType: original.ElementType, indirect: original.Indirect.ToArray());
        return new CodegenKernelSpec(
            spec.Name + "_malformed", spec.Space, inputs, spec.Output,
            spec.ProductInputs.ToArray(), spec.Reduce,
            spec.BiasInput, spec.ScaleInput, spec.Activation, spec.ReduceScale,
            spec.PreReduce, spec.PreBiasInput, spec.PreBiasScale, spec.Algebra,
            spec.ExtraOutputs.ToArray());
    }

    private static int SharedLoadsBeforeFirstFma(string ptx)
    {
        int firstFma = ptx.IndexOf("fma.rn.f32", System.StringComparison.Ordinal);
        Assert.True(firstFma >= 0);
        return ptx.Substring(0, firstFma).Split('\n').Count(line =>
            line.Contains("ld.shared", System.StringComparison.Ordinal));
    }

    private static CodegenKernelSpec SimpleContraction(
        int m, CodegenActivationKind activation)
    {
        var space = new CodegenIterationSpace(
            CodegenAxis.Parallel("batch", 2),
            CodegenAxis.Parallel("m", m),
            CodegenAxis.Parallel("n", 8),
            CodegenAxis.Reduce("k", 4));
        var matrix = new CodegenTensorBinding(0, "matrix", new[] { m, 4 },
            new[] { CodegenAffineExpr.Axis(1), CodegenAffineExpr.Axis(3) });
        var stream = new CodegenTensorBinding(1, "stream", new[] { 2, 4, 8 },
            new[]
            {
                CodegenAffineExpr.Axis(0), CodegenAffineExpr.Axis(3),
                CodegenAffineExpr.Axis(2),
            });
        var output = new CodegenTensorBinding(2, "output", new[] { 2, m, 8 },
            new[]
            {
                CodegenAffineExpr.Axis(0), CodegenAffineExpr.Axis(1),
                CodegenAffineExpr.Axis(2),
            }, isOutput: true);
        return new CodegenKernelSpec(
            "simple_contraction", space, new[] { matrix, stream }, output,
            new[] { 0, 1 }, CodegenReduceKind.Sum, activation: activation);
    }

    private static unsafe void LaunchThree(
        DirectPtxModule module, IntPtr function,
        IntPtr first, IntPtr second, IntPtr output,
        uint blocks, uint blockX, uint blockY)
    {
        IntPtr p0 = first, p1 = second, p2 = output;
        void** arguments = stackalloc void*[3];
        arguments[0] = &p0;
        arguments[1] = &p1;
        arguments[2] = &p2;
        module.Launch(function, blocks, 1, 1, blockX, blockY, 1, 0, arguments);
    }

    private static unsafe void LaunchFour(
        DirectPtxModule module, IntPtr function,
        IntPtr first, IntPtr second, IntPtr third, IntPtr output,
        uint blocks, uint blockX, uint blockY)
    {
        IntPtr p0 = first, p1 = second, p2 = third, p3 = output;
        void** arguments = stackalloc void*[4];
        arguments[0] = &p0;
        arguments[1] = &p1;
        arguments[2] = &p2;
        arguments[3] = &p3;
        module.Launch(function, blocks, 1, 1, blockX, blockY, 1, 0, arguments);
    }
}
