// Copyright (c) AiDotNet. All rights reserved.

using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ptx;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Codegen;

public sealed class CodegenTiledConv2DOuterProductTests
{
    [Fact]
    public void DenseWeightGradientSplit_RecoversRowOuterProductTiles()
    {
        var entry = CodegenKernelCatalog.Find("conv2d_3x3_bwd_weights")!;
        var verifySplit = CodegenSplitReduction.TryPlan(entry.Verify)!;
        var benchSplit = CodegenSplitReduction.TryPlan(entry.Bench)!;

        Assert.True(CodegenTiledConv2DOuterProductPlan.TryCreate(
            verifySplit.Partial, out var verify, out string verifyReason), verifyReason);
        Assert.NotNull(verify);
        Assert.Equal((8, 8, 16, 2, 16),
            (verify!.M, verify.N, verify.Batch,
             verify.OuterReduction, verify.InnerReduction));
        Assert.Equal((8, 8, 1, 1, 64),
            (verify.TileM, verify.TileN, verify.ThreadTileM,
             verify.ThreadTileN, verify.BlockThreads));
        Assert.Equal(144, verify.Blocks);

        Assert.True(CodegenTiledConv2DOuterProductPlan.TryCreate(
            benchSplit.Partial, out var bench, out string benchReason), benchReason);
        Assert.NotNull(bench);
        Assert.Equal((64, 32, 28, 8, 28),
            (bench!.M, bench.N, bench.Batch,
             bench.OuterReduction, bench.InnerReduction));
        Assert.Equal((32, 16, 2, 2, 128),
            (bench.TileM, bench.TileN, bench.ThreadTileM,
             bench.ThreadTileN, bench.BlockThreads));
        Assert.Equal(1008, bench.Blocks);
        Assert.Equal(2 * 28 * (32 + 16) * sizeof(float), bench.SharedMemoryBytes);
    }

    [Fact]
    public void DenseWeightGradientSplit_EmitsDoubleBufferedTrueFp32Ptx()
    {
        var spec = CodegenSplitReduction.TryPlan(
            CodegenKernelCatalog.Find("conv2d_3x3_bwd_weights")!.Bench)!.Partial;
        var emitter = new PtxTiledConv2DOuterProductEmitter();

        string ptx = emitter.Emit(spec, 8, 6);

        Assert.Equal(1008u, emitter.LaunchBlocks);
        Assert.Equal(128, emitter.LaunchBlockThreads);
        Assert.StartsWith(".version 7.5", ptx, StringComparison.Ordinal);
        Assert.Contains("cp.async.ca.shared.global", ptx);
        Assert.Contains("cp.async.wait_group 0", ptx);
        Assert.Contains("TILED_DW_REDUCE:", ptx);
        Assert.Contains("fma.rn.f32", ptx);
        Assert.DoesNotContain("mma.sync", ptx);
        Assert.DoesNotContain("atom", ptx);
    }

    [SkippableFact]
    public unsafe void DenseWeightGradientSplit_MatchesInterpreterOnDevice()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Direct PTX runtime is unavailable.");
        var original = CodegenKernelCatalog.Find("conv2d_3x3_bwd_weights")!.Verify;
        var spec = CodegenSplitReduction.TryPlan(original)!.Partial;
        double[][] inputs = CreateInputs(spec, out var host);
        double[] expected = spec.Interpret(inputs);

        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(runtime.ComputeCapabilityMajor >= 8, "cp.async requires sm_80 or later.");
        var emitter = new PtxTiledConv2DOuterProductEmitter();
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
            emitter.LaunchBlocks, checked((uint)emitter.LaunchBlockThreads), 1);
        runtime.Synchronize();

        var actual = new float[expected.Length];
        output.Download<float>(actual);
        AssertClose(expected, actual, 3e-4, "tiled dense 3x3 split partial");
    }

    [SkippableFact]
    public unsafe void DenseWeightGradientSplit_MatchesAffineAtBenchShape()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Direct PTX runtime is unavailable.");
        var original = CodegenKernelCatalog.Find("conv2d_3x3_bwd_weights")!.Bench;
        var spec = CodegenSplitReduction.TryPlan(original)!.Partial;
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

        var tiled = new PtxTiledConv2DOuterProductEmitter();
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
        AssertClose(expected, actual, 5e-5,
            "benchmark tiled dense 3x3 split partial", relative: true);
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
            double difference = System.Math.Abs(expected[i] - actual[i]);
            if (relative)
                difference /= System.Math.Max(1.0, System.Math.Abs(expected[i]));
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
