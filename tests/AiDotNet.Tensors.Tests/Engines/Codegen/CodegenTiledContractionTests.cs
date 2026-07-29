// Copyright (c) AiDotNet. All rights reserved.

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
        Assert.Equal((32, 56, 8), (bench.TileM, bench.TileN, bench.TileK));
        Assert.Equal((4, 2), (bench.ThreadTileM, bench.ThreadTileN));
        Assert.Equal(224, bench.BlockThreads);
        Assert.Equal(16 * 2 * 14, bench.Blocks);
        Assert.Equal(2 * 8 * (32 + 56) * sizeof(float), bench.SharedMemoryBytes);
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

    /// <summary>The emitted candidate stages both operands and uses only SIMT FP32 FMA.</summary>
    [Fact]
    public void BackwardDataPointwise_EmitsDoubleBufferedTrueFp32Ptx()
    {
        var spec = CodegenKernelCatalog.Find("conv2d_1x1_bwd_data")!.Bench;
        var emitter = new PtxTiledContractionEmitter();

        string ptx = emitter.Emit(spec, 8, 6);

        Assert.Equal(448u, emitter.LaunchBlocks);
        Assert.Equal(224, emitter.LaunchBlockThreads);
        Assert.Contains("cp.async.ca.shared.global", ptx);
        Assert.Contains("cp.async.wait_group 0", ptx);
        Assert.Contains("fma.rn.f32", ptx);
        Assert.DoesNotContain("mma.sync", ptx);
        Assert.DoesNotContain("wmma", ptx);
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

        var tiled = new PtxTiledContractionEmitter();
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
}
