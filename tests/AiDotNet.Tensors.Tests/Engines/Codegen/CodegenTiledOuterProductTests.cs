// Copyright (c) AiDotNet. All rights reserved.

using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ptx;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Codegen;

public sealed class CodegenTiledOuterProductTests
{
    /// <summary>The matcher sees the split maps, not the weight-gradient catalog name.</summary>
    [Fact]
    public void PointwiseWeightGradientSplit_RecoversOuterProductTiles()
    {
        var entry = CodegenKernelCatalog.Find("conv2d_1x1_bwd_weights")!;
        var verifySplit = CodegenSplitReduction.TryPlan(entry.Verify)!;
        var benchSplit = CodegenSplitReduction.TryPlan(entry.Bench)!;

        Assert.True(CodegenTiledOuterProductPlan.TryCreate(
            verifySplit.Partial, out var verify, out string verifyReason), verifyReason);
        Assert.NotNull(verify);
        Assert.Equal((8, 8, 16, 2, 16),
            (verify!.M, verify.N, verify.Batch, verify.OuterReduction, verify.InnerReduction));
        Assert.Equal((8, 8, 1, 1, 64),
            (verify.TileM, verify.TileN, verify.ThreadTileM,
             verify.ThreadTileN, verify.BlockThreads));
        Assert.Equal(16, verify.Blocks);

        Assert.True(CodegenTiledOuterProductPlan.TryCreate(
            benchSplit.Partial, out var bench, out string benchReason), benchReason);
        Assert.NotNull(bench);
        Assert.Equal((64, 64, 28, 16, 28),
            (bench!.M, bench.N, bench.Batch, bench.OuterReduction, bench.InnerReduction));
        Assert.Equal((16, 16, 2, 2, 64),
            (bench.TileM, bench.TileN, bench.ThreadTileM,
             bench.ThreadTileN, bench.BlockThreads));
        Assert.Equal(448, bench.Blocks);
        Assert.Equal(2 * 28 * (16 + 16) * sizeof(float), bench.SharedMemoryBytes);
    }

    /// <summary>The direct unsplit gradient cannot masquerade as this split layout.</summary>
    [Fact]
    public void UnsplitWeightGradient_IsRefused()
    {
        var spec = CodegenKernelCatalog.Find("conv2d_1x1_bwd_weights")!.Bench;

        Assert.False(CodegenTiledOuterProductPlan.TryCreate(
            spec, out var plan, out string reason));
        Assert.Null(plan);
        Assert.Contains("two surviving reduction axes", reason);
    }

    [Fact]
    public void PointwiseWeightGradientSplit_EmitsCooperativeTrueFp32Ptx()
    {
        var spec = CodegenSplitReduction.TryPlan(
            CodegenKernelCatalog.Find("conv2d_1x1_bwd_weights")!.Bench)!.Partial;
        var emitter = new PtxTiledOuterProductEmitter();

        string ptx = emitter.Emit(spec, 8, 6);

        Assert.Equal(448u, emitter.LaunchBlocks);
        Assert.Equal(64, emitter.LaunchBlockThreads);
        Assert.Contains("cp.async.ca.shared.global", ptx);
        Assert.Contains("fma.rn.f32", ptx);
        Assert.DoesNotContain("mma.sync", ptx);
        Assert.DoesNotContain("atom", ptx);
    }

    [SkippableFact]
    public unsafe void PointwiseWeightGradientSplit_MatchesInterpreterOnDevice()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Direct PTX runtime is unavailable.");
        var original = CodegenKernelCatalog.Find("conv2d_1x1_bwd_weights")!.Verify;
        var spec = CodegenSplitReduction.TryPlan(original)!.Partial;
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
        var emitter = new PtxTiledOuterProductEmitter();
        string ptx = emitter.Emit(spec, runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor);
        using var module = runtime.LoadModule(ptx, allowExperimentalJitFallback: true);
        IntPtr function = module.GetFunction(spec.Name, out _);
        using var left = runtime.AllocateBytes((nuint)(host[0].Length * sizeof(float)));
        using var right = runtime.AllocateBytes((nuint)(host[1].Length * sizeof(float)));
        using var output = runtime.AllocateBytes((nuint)(expected.Length * sizeof(float)));
        left.Upload<float>(host[0]);
        right.Upload<float>(host[1]);
        LaunchThree(module, function, left.Pointer, right.Pointer, output.Pointer,
            emitter.LaunchBlocks, checked((uint)emitter.LaunchBlockThreads), 1);
        runtime.Synchronize();

        var actual = new float[expected.Length];
        output.Download<float>(actual);
        AssertClose(expected, actual, 2e-4, "tiled split partial");
    }

    [SkippableFact]
    public unsafe void PointwiseWeightGradientSplit_MatchesAffineAtBenchShape()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Direct PTX runtime is unavailable.");
        var original = CodegenKernelCatalog.Find("conv2d_1x1_bwd_weights")!.Bench;
        var spec = CodegenSplitReduction.TryPlan(original)!.Partial;
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(runtime.ComputeCapabilityMajor >= 8, "cp.async requires sm_80 or later.");

        var host = new float[spec.Inputs.Count][];
        for (int i = 0; i < spec.Inputs.Count; i++)
        {
            host[i] = new float[spec.Inputs[i].ElementCount];
            for (int e = 0; e < host[i].Length; e++)
                host[i][e] = (float)((((e * 37 + i * 101) % 97) - 48) / 64.0);
        }
        using var left = runtime.AllocateBytes((nuint)(host[0].Length * sizeof(float)));
        using var right = runtime.AllocateBytes((nuint)(host[1].Length * sizeof(float)));
        using var tiledOutput = runtime.AllocateBytes(
            (nuint)(spec.Output.ElementCount * sizeof(float)));
        using var affineOutput = runtime.AllocateBytes(
            (nuint)(spec.Output.ElementCount * sizeof(float)));
        left.Upload<float>(host[0]);
        right.Upload<float>(host[1]);

        var tiled = new PtxTiledOuterProductEmitter();
        string tiledPtx = tiled.Emit(
            spec, runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor);
        using var tiledModule = runtime.LoadModule(tiledPtx, allowExperimentalJitFallback: true);
        IntPtr tiledFunction = tiledModule.GetFunction(spec.Name, out _);
        LaunchThree(tiledModule, tiledFunction, left.Pointer, right.Pointer,
            tiledOutput.Pointer, tiled.LaunchBlocks,
            checked((uint)tiled.LaunchBlockThreads), 1);

        var affine = new PtxAffineEmitter();
        string affinePtx = affine.Emit(
            spec, runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor);
        using var affineModule = runtime.LoadModule(affinePtx, allowExperimentalJitFallback: true);
        IntPtr affineFunction = affineModule.GetFunction(spec.Name, out _);
        LaunchThree(affineModule, affineFunction, left.Pointer, right.Pointer,
            affineOutput.Pointer, affine.LaunchBlocks,
            checked((uint)affine.LaunchBlockX), checked((uint)affine.LaunchBlockY));
        runtime.Synchronize();

        var expected = new float[spec.Output.ElementCount];
        var actual = new float[spec.Output.ElementCount];
        affineOutput.Download<float>(expected);
        tiledOutput.Download<float>(actual);
        var expectedDouble = new double[expected.Length];
        for (int i = 0; i < expected.Length; i++) expectedDouble[i] = expected[i];
        AssertClose(expectedDouble, actual, 2e-5, "benchmark tiled split partial", relative: true);
    }

    private static void AssertClose(
        double[] expected, float[] actual, double tolerance, string label, bool relative = false)
    {
        double worst = 0;
        int at = 0;
        for (int i = 0; i < actual.Length; i++)
        {
            double difference = System.Math.Abs(expected[i] - actual[i]);
            if (relative)
                difference /= System.Math.Max(1.0, System.Math.Abs(expected[i]));
            if (difference > worst) { worst = difference; at = i; }
        }
        Assert.True(worst < tolerance,
            $"{label} differs by {worst:E3} at {at}: expected {expected[at]}, actual {actual[at]}");
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
