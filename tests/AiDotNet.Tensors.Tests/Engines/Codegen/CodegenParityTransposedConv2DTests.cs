// Copyright (c) AiDotNet. All rights reserved.

using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ptx;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;
using Xunit;
using static AiDotNet.Tensors.Tests.Engines.Codegen.TiledPtxTestHelper;

namespace AiDotNet.Tensors.Tests.Engines.Codegen;

public sealed class CodegenParityTransposedConv2DTests
{
    [Fact]
    public void Stride2DepthwiseTranspose_RecoversParityTiles()
    {
        var entry = CodegenKernelCatalog.Find("conv_transpose2d_3x3_stride2")!;

        Assert.True(CodegenParityTransposedConv2DPlan.TryCreate(
            entry.Verify, out var verify, out string verifyReason), verifyReason);
        Assert.NotNull(verify);
        Assert.Equal((2, 8, 16, 16, 31, 31, 4096, 16),
            (verify!.Batch, verify.Channels, verify.InputHeight, verify.InputWidth,
             verify.OutputHeight, verify.OutputWidth, verify.InputElements, verify.Blocks));

        Assert.True(CodegenParityTransposedConv2DPlan.TryCreate(
            entry.Bench, out var bench, out string benchReason), benchReason);
        Assert.NotNull(bench);
        Assert.Equal((16, 64, 28, 28, 55, 55, 802816, 3136),
            (bench!.Batch, bench.Channels, bench.InputHeight, bench.InputWidth,
             bench.OutputHeight, bench.OutputWidth, bench.InputElements, bench.Blocks));
    }

    [Fact]
    public void DirectDepthwiseWindow_IsRefused()
    {
        var spec = CodegenKernelCatalog.Find("depthwise_conv2d_3x3")!.Bench;

        Assert.False(CodegenParityTransposedConv2DPlan.TryCreate(
            spec, out var plan, out string reason));
        Assert.Null(plan);
        Assert.Contains("transposed window", reason);
    }

    [Fact]
    public void Stride2DepthwiseTranspose_EmitsGuardFreeParityPtx()
    {
        var spec = CodegenKernelCatalog.Find("conv_transpose2d_3x3_stride2")!.Bench;
        var emitter = new PtxParityTransposedConv2DEmitter();

        string ptx = emitter.Emit(spec, 8, 6);

        Assert.Equal(3136u, emitter.LaunchBlocks);
        Assert.Equal(256, emitter.LaunchBlockThreads);
        Assert.Contains("deterministic 2x2 output parity tile", ptx);
        Assert.Contains("@%p3 st.global.f32", ptx);
        Assert.DoesNotContain("rem.s32", ptx);
        Assert.DoesNotContain("div.s32", ptx);
        Assert.Equal(9, ptx.Split("fma.rn.f32").Length - 1);
    }

    [SkippableFact]
    public unsafe void Stride2DepthwiseTranspose_MatchesInterpreterOnDevice()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Direct PTX runtime is unavailable.");
        var spec = CodegenKernelCatalog.Find("conv_transpose2d_3x3_stride2")!.Verify;
        double[][] inputs = CreateInputs(spec, out var host);
        double[] expected = spec.Interpret(inputs);

        using var runtime = new DirectPtxRuntime();
        var emitter = new PtxParityTransposedConv2DEmitter();
        string ptx = emitter.Emit(
            spec, runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor);
        using var module = runtime.LoadModule(ptx, allowExperimentalJitFallback: true);
        IntPtr function = module.GetFunction(spec.Name, out _);
        using var input = runtime.AllocateBytes((nuint)(host[0].Length * sizeof(float)));
        using var weights = runtime.AllocateBytes((nuint)(host[1].Length * sizeof(float)));
        using var output = runtime.AllocateBytes((nuint)(expected.Length * sizeof(float)));
        input.Upload<float>(host[0]);
        weights.Upload<float>(host[1]);
        LaunchThree(module, function, input.Pointer, weights.Pointer, output.Pointer,
            emitter.LaunchBlocks, checked((uint)emitter.LaunchBlockThreads), 1);
        runtime.Synchronize();

        var actual = new float[expected.Length];
        output.Download<float>(actual);
        AssertClose(expected, actual, 3e-5, "parity-transposed verify shape");
    }

    [SkippableFact]
    public unsafe void Stride2DepthwiseTranspose_MatchesAffineAtBenchShape()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Direct PTX runtime is unavailable.");
        var spec = CodegenKernelCatalog.Find("conv_transpose2d_3x3_stride2")!.Bench;
        _ = CreateInputs(spec, out var host);
        using var runtime = new DirectPtxRuntime();

        using var input = runtime.AllocateBytes((nuint)(host[0].Length * sizeof(float)));
        using var weights = runtime.AllocateBytes((nuint)(host[1].Length * sizeof(float)));
        using var parityOutput = runtime.AllocateBytes(
            (nuint)(spec.Output.ElementCount * sizeof(float)));
        using var affineOutput = runtime.AllocateBytes(
            (nuint)(spec.Output.ElementCount * sizeof(float)));
        input.Upload<float>(host[0]);
        weights.Upload<float>(host[1]);

        var parity = new PtxParityTransposedConv2DEmitter();
        string parityPtx = parity.Emit(
            spec, runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor);
        using var parityModule = runtime.LoadModule(
            parityPtx, allowExperimentalJitFallback: true);
        IntPtr parityFunction = parityModule.GetFunction(spec.Name, out _);
        LaunchThree(parityModule, parityFunction, input.Pointer, weights.Pointer,
            parityOutput.Pointer, parity.LaunchBlocks,
            checked((uint)parity.LaunchBlockThreads), 1);

        var affine = new PtxAffineEmitter();
        string affinePtx = affine.Emit(
            spec, runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor);
        using var affineModule = runtime.LoadModule(
            affinePtx, allowExperimentalJitFallback: true);
        IntPtr affineFunction = affineModule.GetFunction(spec.Name, out _);
        LaunchThree(affineModule, affineFunction, input.Pointer, weights.Pointer,
            affineOutput.Pointer, affine.LaunchBlocks,
            checked((uint)affine.LaunchBlockX), checked((uint)affine.LaunchBlockY));
        runtime.Synchronize();

        var expected = new float[spec.Output.ElementCount];
        var actual = new float[spec.Output.ElementCount];
        affineOutput.Download<float>(expected);
        parityOutput.Download<float>(actual);
        AssertClose(expected, actual, 3e-5,
            "benchmark parity-transposed", relative: true);
    }
}
