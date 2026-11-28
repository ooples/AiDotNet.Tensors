// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

public sealed class NvrtcGraphEmitterTests
{
    private static CodegenGraph SwishPlusY(int[] xShape, int[] yShape)
    {
        var graph = new CodegenGraph();
        int x = graph.AddNode(new CodegenNode(CodegenOpKind.LoadInput, Array.Empty<int>(), CodegenElementType.Float32, xShape, 0));
        int y = graph.AddNode(new CodegenNode(CodegenOpKind.LoadInput, Array.Empty<int>(), CodegenElementType.Float32, yShape, 1));
        int swish = graph.AddNode(new CodegenNode(CodegenOpKind.Swish, new[] { x }, CodegenElementType.Float32, xShape));
        int add = graph.AddNode(new CodegenNode(CodegenOpKind.Add, new[] { swish, y }, CodegenElementType.Float32, xShape));
        graph.AddNode(new CodegenNode(CodegenOpKind.StoreOutput, new[] { add }, CodegenElementType.Float32, xShape, 0));
        return graph;
    }

    private static NvrtcCodegenKernel Emit(CodegenGraph graph)
    {
        var result = new NvrtcGraphEmitter().Emit(graph, CodegenElementType.Float32);
        Assert.False(result.Declined, result.DeclineReason);
        return Assert.IsType<NvrtcCodegenKernel>(result.Kernel);
    }

    [Fact]
    public void EntryPoint_IsKeyedByEmittedSource_SoDenseAndBroadcastLayoutsNeverShareAFunction()
    {
        var dense = Emit(SwishPlusY(new[] { 2, 3 }, new[] { 2, 3 }));
        var broadcast = Emit(SwishPlusY(new[] { 2, 3 }, new[] { 3 }));
        Assert.False(dense.UsesTrailingBroadcast);
        Assert.True(broadcast.UsesTrailingBroadcast);
        Assert.NotEqual(dense.EntryPoint, broadcast.EntryPoint);
        Assert.Contains(dense.EntryPoint, dense.CudaSource);
        Assert.DoesNotContain("__AIDN_NVRTC_ENTRY__", dense.CudaSource);

        // Identical source still reuses the same entry point.
        Assert.Equal(dense.EntryPoint, Emit(SwishPlusY(new[] { 2, 3 }, new[] { 2, 3 })).EntryPoint);
    }

    [Fact]
    public void BindingMetadata_IsFrozenAtEmission()
    {
        var graph = SwishPlusY(new[] { 2, 3 }, new[] { 2, 3 });
        var kernel = Emit(graph);
        graph.AddNode(new CodegenNode(CodegenOpKind.LoadInput, Array.Empty<int>(), CodegenElementType.Float32, new[] { 2, 3 }, 2));
        Assert.Equal(3, kernel.InputCount);          // the live graph changed...
        Assert.Equal(2, kernel.BoundInputCount);     // ...the compiled signature did not
        Assert.Equal(new long[] { 24, 24 }, kernel.InputRequiredBytes);
    }

    [Theory]
    [InlineData(float.NaN)]
    [InlineData(float.PositiveInfinity)]
    [InlineData(float.NegativeInfinity)]
    public void NonFiniteConstant_IsDeclined_NamingTheNodeAndValue(float value)
    {
        var graph = new CodegenGraph();
        int x = graph.AddNode(new CodegenNode(CodegenOpKind.LoadInput, Array.Empty<int>(), CodegenElementType.Float32, new[] { 8 }, 0));
        int c = graph.AddNode(new CodegenNode(CodegenOpKind.Constant, Array.Empty<int>(), CodegenElementType.Float32, new[] { 8 }, value));
        int mul = graph.AddNode(new CodegenNode(CodegenOpKind.Mul, new[] { x, c }, CodegenElementType.Float32, new[] { 8 }));
        graph.AddNode(new CodegenNode(CodegenOpKind.StoreOutput, new[] { mul }, CodegenElementType.Float32, new[] { 8 }, 0));

        var result = new NvrtcGraphEmitter().Emit(graph, CodegenElementType.Float32);
        Assert.True(result.Declined);
        Assert.Contains($"Node {c}", result.DeclineReason);
        Assert.Contains(value.ToString("R", System.Globalization.CultureInfo.InvariantCulture), result.DeclineReason);
    }

    [Fact]
    public void EmittedIndexArithmetic_Is64Bit()
    {
        var dense = Emit(SwishPlusY(new[] { 2, 3 }, new[] { 2, 3 }));
        var broadcast = Emit(SwishPlusY(new[] { 2, 3 }, new[] { 3 }));
        foreach (var k in new[] { dense, broadcast })
        {
            Assert.Contains("long long tid", k.CudaSource);
            Assert.Contains("long long stride", k.CudaSource);
            Assert.DoesNotContain("int idx", k.CudaSource);
        }
    }

    [SkippableFact]
    public void CudaLaunch_RejectsMismatchedElementCount_AndRunsDenseThenBroadcastCorrectly()
    {
        CudaBackend backend;
        try { backend = new CudaBackend(); }
        catch (Exception ex) { Skip.If(true, "CUDA backend unavailable: " + ex.Message); return; }
        using (backend)
        {
            Skip.IfNot(backend.IsAvailable, "CUDA is not available.");
            var xs = new float[] { -1f, 0f, 1f, 2f, -2f, 0.5f };
            using var x = backend.AllocateBuffer(xs);
            using var yDense = backend.AllocateBuffer(new float[] { 1f, 2f, 3f, 4f, 5f, 6f });
            using var yRow = backend.AllocateBuffer(new float[] { 10f, 20f, 30f });
            using var output = backend.AllocateBuffer(8);

            var dense = Emit(SwishPlusY(new[] { 2, 3 }, new[] { 2, 3 }));
            Assert.Throws<ArgumentException>(() => backend.ExecuteCodegenKernelAsync(
                dense, new IGpuBuffer[] { x, yDense }, new IGpuBuffer[] { output }, 8).GetAwaiter().GetResult());
            Assert.Throws<ArgumentException>(() => backend.ExecuteCodegenKernelAsync(
                dense, new IGpuBuffer[] { x, yDense }, new IGpuBuffer[] { output }, 4).GetAwaiter().GetResult());

            backend.ExecuteCodegenKernelAsync(dense, new IGpuBuffer[] { x, yDense }, new IGpuBuffer[] { output }, 6)
                .GetAwaiter().GetResult();
            var got = backend.DownloadBuffer(output);
            for (int i = 0; i < 6; i++)
                Assert.Equal((double)(Swish(xs[i]) + (i + 1)), got[i], 4);

            var broadcast = Emit(SwishPlusY(new[] { 2, 3 }, new[] { 3 }));
            backend.ExecuteCodegenKernelAsync(broadcast, new IGpuBuffer[] { x, yRow }, new IGpuBuffer[] { output }, 6)
                .GetAwaiter().GetResult();
            got = backend.DownloadBuffer(output);
            for (int i = 0; i < 6; i++)
                Assert.Equal((double)(Swish(xs[i]) + 10f * (i % 3 + 1)), got[i], 4);
        }
    }

    private static float Swish(float v) => (float)(v / (1.0 + Math.Exp(-v)));
}
