// Copyright (c) AiDotNet. All rights reserved.
// Phase C of issue #225: verify that each GPU source-emitter
// produces syntactically valid source containing the expected
// identifiers and op expressions. Actual GPU execution validation
// is a Phase C.5 follow-up where each emitter's output is wired
// into the corresponding backend's compilation surface.

#nullable disable

using System;
using AiDotNet.Tensors.Engines.Compilation.Codegen;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Glsl;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Hip;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Msl;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Triton;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Wgsl;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation.Codegen;

public class GpuEmitterTests
{
    // ─── Triton ──────────────────────────────────────────────────────

    [Fact]
    public void Triton_UnaryRelu_ContainsCorrectOp()
    {
        var g = CodegenLowering.LowerUnaryPointwise<float>(CodegenOpKind.ReLU, new[] { 1024 });
        var r = new TritonEmitter().Emit(g, CodegenElementType.Float32);
        Assert.False(r.Declined);
        Assert.Contains("@triton.jit", r.Source);
        Assert.Contains("pointwise_kernel", r.Source);
        Assert.Contains("tl.load", r.Source);
        Assert.Contains("tl.store", r.Source);
        Assert.Contains("tl.maximum", r.Source);     // ReLU → max(x, 0)
        Assert.Contains("BLOCK_SIZE: tl.constexpr", r.Source);
    }

    [Fact]
    public void Triton_BinaryAdd_BuildsTwoInputKernel()
    {
        var g = CodegenLowering.LowerBinaryPointwise<float>(CodegenOpKind.Add, new[] { 256 });
        var r = new TritonEmitter().Emit(g, CodegenElementType.Float32);
        Assert.Contains("in_0_ptr", r.Source);
        Assert.Contains("in_1_ptr", r.Source);
        Assert.Contains("out_0_ptr", r.Source);
        Assert.Contains("v0 + v1", r.Source); // Add expression
    }

    [Fact]
    public void Triton_DeclinesReduction()
    {
        var g = new CodegenGraph();
        int a = g.AddNode(new CodegenNode(CodegenOpKind.LoadInput, Array.Empty<int>(),
            CodegenElementType.Float32, new[] { 4 }, 0));
        int red = g.AddNode(new CodegenNode(CodegenOpKind.ReduceSum, new[] { a },
            CodegenElementType.Float32, new[] { 4 }));
        g.AddNode(new CodegenNode(CodegenOpKind.StoreOutput, new[] { red },
            CodegenElementType.Float32, new[] { 4 }, 0));
        var r = new TritonEmitter().Emit(g, CodegenElementType.Float32);
        Assert.True(r.Declined);
    }

    // ─── HIP ─────────────────────────────────────────────────────────

    [Fact]
    public void Hip_Sigmoid_UsesExpf()
    {
        var g = CodegenLowering.LowerUnaryPointwise<float>(CodegenOpKind.Sigmoid, new[] { 64 });
        var r = new HipEmitter().Emit(g, CodegenElementType.Float32);
        Assert.False(r.Declined);
        Assert.Contains("#include <hip/hip_runtime.h>", r.Source);
        Assert.Contains("extern \"C\" __global__ void pointwise_kernel", r.Source);
        Assert.Contains("blockIdx.x * blockDim.x + threadIdx.x", r.Source);
        Assert.Contains("expf", r.Source); // float-precision sigmoid uses expf
        Assert.Contains("1.0f / (1.0f + expf(-v0))", r.Source);
    }

    [Fact]
    public void Hip_Double_UsesUnsuffixedIntrinsics()
    {
        var g = CodegenLowering.LowerUnaryPointwise<double>(CodegenOpKind.Exp, new[] { 32 });
        var r = new HipEmitter().Emit(g, CodegenElementType.Float64);
        Assert.False(r.Declined);
        Assert.Contains("const double*", r.Source);
        Assert.Contains("double v1 = exp(v0);", r.Source); // double uses exp, not expf
    }

    [Fact]
    public void Hip_ChainedOps_EmitsInOrder()
    {
        var g = CodegenLowering.LowerUnaryChain<float>(
            new[] { CodegenOpKind.Negate, CodegenOpKind.Exp, CodegenOpKind.Sqrt },
            new[] { 16 });
        var r = new HipEmitter().Emit(g, CodegenElementType.Float32);
        Assert.False(r.Declined);
        int negIdx = r.Source.IndexOf("-v0");
        int expIdx = r.Source.IndexOf("expf(v1)");
        int sqrtIdx = r.Source.IndexOf("sqrtf(v2)");
        Assert.InRange(negIdx, 0, int.MaxValue);
        Assert.InRange(expIdx, negIdx, int.MaxValue);
        Assert.InRange(sqrtIdx, expIdx, int.MaxValue);
    }

    // ─── MSL ─────────────────────────────────────────────────────────

    [Fact]
    public void Msl_UnaryTanh_UsesMetalStdLib()
    {
        var g = CodegenLowering.LowerUnaryPointwise<float>(CodegenOpKind.Tanh, new[] { 128 });
        var r = new MslEmitter().Emit(g, CodegenElementType.Float32);
        Assert.False(r.Declined);
        Assert.Contains("#include <metal_stdlib>", r.Source);
        Assert.Contains("using namespace metal;", r.Source);
        Assert.Contains("kernel void pointwise_kernel", r.Source);
        Assert.Contains("device const float*", r.Source);
        Assert.Contains("[[thread_position_in_grid]]", r.Source);
        Assert.Contains("tanh(v0)", r.Source);
    }

    [Fact]
    public void Msl_DeclinesDouble()
    {
        var g = CodegenLowering.LowerUnaryPointwise<double>(CodegenOpKind.Exp, new[] { 4 });
        var r = new MslEmitter().Emit(g, CodegenElementType.Float64);
        Assert.True(r.Declined);
    }

    // ─── WGSL ────────────────────────────────────────────────────────

    [Fact]
    public void Wgsl_UnarySqrt_FollowsWebGpuConvention()
    {
        var g = CodegenLowering.LowerUnaryPointwise<float>(CodegenOpKind.Sqrt, new[] { 512 });
        var r = new WgslEmitter().Emit(g, CodegenElementType.Float32);
        Assert.False(r.Declined);
        Assert.Contains("@group(0) @binding(0) var<storage, read>       in_0 : array<f32>;", r.Source);
        Assert.Contains("@group(0) @binding(1) var<storage, read_write> out_0 : array<f32>;", r.Source);
        Assert.Contains("@compute @workgroup_size(256) fn main", r.Source);
        Assert.Contains("let gid : u32 = id.x;", r.Source);
        Assert.Contains("sqrt(v0)", r.Source);
    }

    [Fact]
    public void Wgsl_DeclinesDouble_WgslCoreLacksF64()
    {
        var g = CodegenLowering.LowerUnaryPointwise<double>(CodegenOpKind.Exp, new[] { 8 });
        var r = new WgslEmitter().Emit(g, CodegenElementType.Float64);
        Assert.True(r.Declined);
    }

    // ─── GLSL ────────────────────────────────────────────────────────

    [Fact]
    public void Glsl_UnaryReLU_ProducesVulkanCompatibleShader()
    {
        var g = CodegenLowering.LowerUnaryPointwise<float>(CodegenOpKind.ReLU, new[] { 256 });
        var r = new GlslEmitter().Emit(g, CodegenElementType.Float32);
        Assert.False(r.Declined);
        Assert.Contains("#version 450", r.Source);
        Assert.Contains("layout(local_size_x = 256) in;", r.Source);
        Assert.Contains("layout(set = 0, binding = 0) readonly buffer InBuf0", r.Source);
        Assert.Contains("layout(set = 0, binding = 1) writeonly buffer OutBuf0", r.Source);
        Assert.Contains("layout(push_constant) uniform P", r.Source);
        Assert.Contains("gl_GlobalInvocationID.x", r.Source);
        Assert.Contains("max(v0, 0.0)", r.Source);
    }

    [Fact]
    public void Glsl_BinaryMul_EmitsBothInputs()
    {
        var g = CodegenLowering.LowerBinaryPointwise<float>(CodegenOpKind.Mul, new[] { 100 });
        var r = new GlslEmitter().Emit(g, CodegenElementType.Float32);
        Assert.False(r.Declined);
        Assert.Contains("InBuf0", r.Source);
        Assert.Contains("InBuf1", r.Source);
        Assert.Contains("v0 * v1", r.Source);
    }

    // ─── Cross-emitter consistency ───────────────────────────────────

    [Fact]
    public void AllEmitters_DeclineOpaqueGraphs()
    {
        var g = new CodegenGraph();
        int a = g.AddNode(new CodegenNode(CodegenOpKind.LoadInput, Array.Empty<int>(),
            CodegenElementType.Float32, new[] { 4 }, 0));
        int op = g.AddNode(new CodegenNode(CodegenOpKind.Opaque, new[] { a },
            CodegenElementType.Float32, new[] { 4 }));
        g.AddNode(new CodegenNode(CodegenOpKind.StoreOutput, new[] { op },
            CodegenElementType.Float32, new[] { 4 }, 0));

        Assert.True(new TritonEmitter().Emit(g, CodegenElementType.Float32).Declined);
        Assert.True(new HipEmitter().Emit(g, CodegenElementType.Float32).Declined);
        Assert.True(new MslEmitter().Emit(g, CodegenElementType.Float32).Declined);
        Assert.True(new WgslEmitter().Emit(g, CodegenElementType.Float32).Declined);
        Assert.True(new GlslEmitter().Emit(g, CodegenElementType.Float32).Declined);
    }

    [Fact]
    public void AllEmitters_ProduceKernelsWithMatchingPortCounts()
    {
        var g = CodegenLowering.LowerBinaryPointwise<float>(CodegenOpKind.Sub, new[] { 64 });

        foreach (var emit in new IKernelEmitter[]
        {
            new TritonEmitter(), new HipEmitter(),
            new MslEmitter(), new WgslEmitter(), new GlslEmitter(),
        })
        {
            var r = emit.Emit(g, CodegenElementType.Float32);
            Assert.False(r.Declined, $"{emit.Target} declined a supported pointwise fusion.");
            Assert.Equal(2, r.Kernel.InputCount);
            Assert.Equal(1, r.Kernel.OutputCount);
            Assert.Equal(CodegenElementType.Float32, r.Kernel.Dtype);
        }
    }

    [Fact]
    public void AllEmitters_ExecuteThrows_PendingBackendDispatch()
    {
        // Phase C ships the source strings; actual GPU dispatch is a
        // Phase C.5 follow-up. Calling Execute on a GpuSourceKernel
        // must throw loudly so callers route through CpuDotNetJit for
        // local execution and don't silently get wrong results.
        var g = CodegenLowering.LowerUnaryPointwise<float>(CodegenOpKind.ReLU, new[] { 4 });
        foreach (var emit in new IKernelEmitter[]
        {
            new TritonEmitter(), new HipEmitter(),
            new MslEmitter(), new WgslEmitter(), new GlslEmitter(),
        })
        {
            var r = emit.Emit(g, CodegenElementType.Float32);
            Assert.False(r.Declined);
            Assert.Throws<NotSupportedException>(
                () => r.Kernel.Execute<float>(new[] { new float[4] }, new[] { new float[4] }));
        }
    }
}
