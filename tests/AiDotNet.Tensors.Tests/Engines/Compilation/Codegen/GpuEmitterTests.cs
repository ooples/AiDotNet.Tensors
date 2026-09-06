// Copyright (c) AiDotNet. All rights reserved.
// Phase C of issue #225: verify that each GPU source-emitter
// produces syntactically valid source containing the expected
// identifiers and op expressions. Backend hardware suites separately validate
// native compilation, dispatch, profiling, and artifact replay where supported.

#nullable disable

using System;
using AiDotNet.Tensors.Engines.Compilation.Codegen;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Glsl;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Hip;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Msl;
using AiDotNet.Tensors.Engines.Compilation.Codegen.OpenCl;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Triton;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Wgsl;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.HIP;
using AiDotNet.Tensors.Engines.DirectGpu.Metal;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;
using AiDotNet.Tensors.Engines.DirectGpu.Vulkan;
#if NET7_0_OR_GREATER
using AiDotNet.Tensors.Engines.DirectGpu.WebGpu;
#endif
using AiDotNet.Tensors.Helpers.Autotune;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation.Codegen;

public class GpuEmitterTests
{
    [Fact]
    public void EveryCodegenTarget_HasTypedNativeRuntimeContract()
    {
        var targets = (CodegenTarget[])Enum.GetValues(typeof(CodegenTarget));
        Assert.NotEmpty(targets);

        foreach (CodegenTarget target in targets)
        {
            CodegenTargetRuntime runtime = CodegenTargetRuntime.For(target);
            Assert.True(Enum.IsDefined(typeof(KernelTuningBackend), runtime.Backend));
            Assert.True(Enum.IsDefined(typeof(CodegenNativeCompilationKind), runtime.Compilation));
            Assert.True(Enum.IsDefined(typeof(CodegenNativeArtifactKind), runtime.NativeArtifact));
            Assert.Equal(target is not CodegenTarget.CpuDotNetJit and not CodegenTarget.CpuAvx512,
                runtime.IsGpu);
        }

        Assert.Equal(KernelTuningBackend.Cuda,
            CodegenTargetRuntime.For(CodegenTarget.DirectPtx).Backend);
        Assert.Equal(KernelTuningBackend.Hip,
            CodegenTargetRuntime.For(CodegenTarget.Hip).Backend);
        Assert.Equal(KernelTuningBackend.OpenCl,
            CodegenTargetRuntime.For(CodegenTarget.OpenCl).Backend);
        Assert.Equal(CodegenNativeCompilationKind.OpenClDriverCompiler,
            CodegenTargetRuntime.For(CodegenTarget.OpenCl).Compilation);
        Assert.Equal(CodegenNativeArtifactKind.OpenClDeviceBinary,
            CodegenTargetRuntime.For(CodegenTarget.OpenCl).NativeArtifact);
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            CodegenTargetRuntime.For((CodegenTarget)int.MaxValue));
    }

    [Fact]
    public void CodegenTarget_ExistingSerializedValuesRemainStable()
    {
        Assert.Equal(0, (int)CodegenTarget.CpuDotNetJit);
        Assert.Equal(1, (int)CodegenTarget.CpuAvx512);
        Assert.Equal(2, (int)CodegenTarget.Triton);
        Assert.Equal(3, (int)CodegenTarget.Hip);
        Assert.Equal(4, (int)CodegenTarget.Msl);
        Assert.Equal(5, (int)CodegenTarget.Wgsl);
        Assert.Equal(6, (int)CodegenTarget.Glsl);
        Assert.Equal(7, (int)CodegenTarget.DirectPtx);
        Assert.Equal(8, (int)CodegenTarget.OpenCl);
    }

    [Fact]
    public void EveryGpuRuntime_HasItsOwnNativeCompilerAndArtifactContract()
    {
        var expected = new[]
        {
            (CodegenTarget.Triton, KernelTuningBackend.Cuda,
                CodegenNativeCompilationKind.TritonCudaJit,
                CodegenNativeArtifactKind.CudaDriverModule),
            (CodegenTarget.DirectPtx, KernelTuningBackend.Cuda,
                CodegenNativeCompilationKind.CudaPtxModuleLoader,
                CodegenNativeArtifactKind.CudaDriverModule),
            (CodegenTarget.Hip, KernelTuningBackend.Hip,
                CodegenNativeCompilationKind.HipRtc,
                CodegenNativeArtifactKind.HipCodeObject),
            (CodegenTarget.OpenCl, KernelTuningBackend.OpenCl,
                CodegenNativeCompilationKind.OpenClDriverCompiler,
                CodegenNativeArtifactKind.OpenClDeviceBinary),
            (CodegenTarget.Msl, KernelTuningBackend.Metal,
                CodegenNativeCompilationKind.MetalLibraryCompiler,
                CodegenNativeArtifactKind.MetalComputePipeline),
            (CodegenTarget.Glsl, KernelTuningBackend.Vulkan,
                CodegenNativeCompilationKind.ShadercToVulkanPipeline,
                CodegenNativeArtifactKind.VulkanComputePipeline),
            (CodegenTarget.Wgsl, KernelTuningBackend.WebGpu,
                CodegenNativeCompilationKind.WebGpuImplementationCompiler,
                CodegenNativeArtifactKind.WebGpuComputePipeline)
        };

        foreach (var item in expected)
        {
            CodegenTargetRuntime runtime = CodegenTargetRuntime.For(item.Item1);
            Assert.Equal(item.Item2, runtime.Backend);
            Assert.Equal(item.Item3, runtime.Compilation);
            Assert.Equal(item.Item4, runtime.NativeArtifact);
            Assert.True(runtime.IsGpu);
        }
    }

    [Fact]
    public void EveryPhysicalGpuBackend_ImplementsNativeCodegenExecution()
    {
        var backendTypes = new List<Type>
        {
            typeof(CudaBackend),
            typeof(HipBackend),
            typeof(OpenClBackend),
            typeof(MetalBackend),
            typeof(VulkanBackend)
        };
#if NET7_0_OR_GREATER
        backendTypes.Add(typeof(WebGpuBackend));
#endif

        Assert.All(backendTypes,
            backendType => Assert.True(
                typeof(INativeGpuCodegenExecutor).IsAssignableFrom(backendType),
                $"{backendType.Name} does not implement native fused codegen execution."));
    }

    [Fact]
    public async Task BackendOwnedNativeCodegenOrchestration_ExecutesOnlyMatchingSuccessfulEmission()
    {
        CodegenGraph graph = CodegenLowering.LowerUnaryPointwise<float>(
            CodegenOpKind.ReLU, new[] { 32 });
        CodegenEmitResult emitted = new OpenClEmitter().Emit(
            graph, CodegenElementType.Float32);

        var matching = new RecordingNativeExecutor(
            KernelTuningBackend.OpenCl, emitted, canExecute: true);
        CodegenEmitResult result = await matching.EmitAndExecuteCodegenKernelAsync(
            graph,
            CodegenElementType.Float32,
            Array.Empty<IGpuBuffer>(),
            Array.Empty<IGpuBuffer>(),
            32);
        Assert.False(result.Declined);
        Assert.Equal(1, matching.ExecutionCount);

        var declined = new RecordingNativeExecutor(
            KernelTuningBackend.OpenCl,
            CodegenEmitResult.Decline("unsupported"),
            canExecute: false);
        result = await declined.EmitAndExecuteCodegenKernelAsync(
            graph,
            CodegenElementType.Float32,
            Array.Empty<IGpuBuffer>(),
            Array.Empty<IGpuBuffer>(),
            32);
        Assert.True(result.Declined);
        Assert.Equal(0, declined.ExecutionCount);

        var mismatched = new RecordingNativeExecutor(
            KernelTuningBackend.Cuda, emitted, canExecute: true);
        await Assert.ThrowsAsync<InvalidOperationException>(async () =>
            await mismatched.EmitAndExecuteCodegenKernelAsync(
                graph,
                CodegenElementType.Float32,
                Array.Empty<IGpuBuffer>(),
                Array.Empty<IGpuBuffer>(),
                32));
        Assert.Equal(0, mismatched.ExecutionCount);
    }

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
        Assert.DoesNotContain("#include <math.h>", r.Source);
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

    // ─── OpenCL ─────────────────────────────────────────────────────

    [Fact]
    public void OpenCl_ChainedOps_ProducesOneNativeKernelEntryPoint()
    {
        var g = CodegenLowering.LowerUnaryChain<float>(
            new[] { CodegenOpKind.Negate, CodegenOpKind.Exp, CodegenOpKind.Sqrt },
            new[] { 1024 });

        var r = new OpenClEmitter().Emit(g, CodegenElementType.Float32);

        Assert.False(r.Declined);
        Assert.Equal(CodegenTarget.OpenCl, r.Kernel.Target);
        Assert.Contains("__kernel void pointwise_kernel", r.Source);
        Assert.Contains("get_global_id(0)", r.Source);
        Assert.Contains("-v0", r.Source);
        Assert.Contains("exp(v1)", r.Source);
        Assert.Contains("sqrt(v2)", r.Source);
        Assert.Equal(1, CountOccurrences(r.Source, "__kernel void"));
    }

    [Fact]
    public void GpuEmitters_DeclineInvalidOrOverflowingElementCounts()
    {
        CodegenEmitResult empty = new OpenClEmitter().Emit(
            CodegenLowering.LowerUnaryPointwise<float>(CodegenOpKind.ReLU, new[] { 0 }),
            CodegenElementType.Float32);
        Assert.True(empty.Declined);
        Assert.Contains("positive", empty.DeclineReason, StringComparison.OrdinalIgnoreCase);

        var graph = new CodegenGraph();
        int input = graph.AddNode(new CodegenNode(
            CodegenOpKind.LoadInput, Array.Empty<int>(), CodegenElementType.Float32,
            new[] { int.MaxValue, int.MaxValue, int.MaxValue }, 0));
        graph.AddNode(new CodegenNode(
            CodegenOpKind.StoreOutput, new[] { input }, CodegenElementType.Float32,
            new[] { int.MaxValue, int.MaxValue, int.MaxValue }, 0));

        CodegenEmitResult overflow = new OpenClEmitter().Emit(graph, CodegenElementType.Float32);
        Assert.True(overflow.Declined);
        Assert.Contains("Int64", overflow.DeclineReason, StringComparison.Ordinal);
    }

    [Fact]
    public void GpuSourceLaunch_RequiresTheExactEmittedExtent()
    {
        CodegenGraph graph = CodegenLowering.LowerUnaryPointwise<float>(
            CodegenOpKind.ReLU, new[] { 4, 8 });
        CodegenEmitResult result = new OpenClEmitter().Emit(graph, CodegenElementType.Float32);
        GpuSourceKernel kernel = Assert.IsType<GpuSourceKernel>(result.Kernel);

        GpuEmitterCommon.ValidateLaunchElementCount(kernel, 32);
        Assert.Throws<ArgumentException>(() =>
            GpuEmitterCommon.ValidateLaunchElementCount(kernel, 31));
        Assert.Throws<ArgumentException>(() =>
            GpuEmitterCommon.ValidateLaunchElementCount(kernel, 33));
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

    [Fact]
    public void EmbeddedWorkgroupGeometry_IsTypedAndMatchesShaderSource()
    {
        CodegenGraph graph = CodegenLowering.LowerUnaryPointwise<float>(
            CodegenOpKind.ReLU, new[] { 1024 });

        CodegenEmitResult glsl = new GlslEmitter { WorkgroupSize = 64 }
            .Emit(graph, CodegenElementType.Float32);
        GpuSourceKernel glslKernel = Assert.IsType<GpuSourceKernel>(glsl.Kernel);
        Assert.Equal(64, glslKernel.DeclaredWorkgroupSize);
        Assert.Contains("local_size_x = 64", glsl.Source, StringComparison.Ordinal);

        CodegenEmitResult wgsl = new WgslEmitter { WorkgroupSize = 128 }
            .Emit(graph, CodegenElementType.Float32);
        GpuSourceKernel wgslKernel = Assert.IsType<GpuSourceKernel>(wgsl.Kernel);
        Assert.Equal(128, wgslKernel.DeclaredWorkgroupSize);
        Assert.Contains("@workgroup_size(128)", wgsl.Source, StringComparison.Ordinal);
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
        Assert.True(new OpenClEmitter().Emit(g, CodegenElementType.Float32).Declined);
        Assert.True(new MslEmitter().Emit(g, CodegenElementType.Float32).Declined);
        Assert.True(new WgslEmitter().Emit(g, CodegenElementType.Float32).Declined);
        Assert.True(new GlslEmitter().Emit(g, CodegenElementType.Float32).Declined);
    }

    [Fact]
    public void AllEmitters_DeclineUnimplementedPointwiseAndMovementOpsWithoutThrowing()
    {
        foreach (CodegenOpKind unsupported in new[]
        {
            CodegenOpKind.GELU,
            CodegenOpKind.Constant,
            CodegenOpKind.Transpose
        })
        {
            var graph = new CodegenGraph();
            int input = graph.AddNode(new CodegenNode(
                CodegenOpKind.LoadInput, Array.Empty<int>(), CodegenElementType.Float32,
                new[] { 4 }, 0));
            int operation = graph.AddNode(new CodegenNode(
                unsupported,
                unsupported == CodegenOpKind.Constant ? Array.Empty<int>() : new[] { input },
                CodegenElementType.Float32,
                new[] { 4 },
                unsupported == CodegenOpKind.Constant ? 1.0f : null));
            graph.AddNode(new CodegenNode(
                CodegenOpKind.StoreOutput, new[] { operation }, CodegenElementType.Float32,
                new[] { 4 }, 0));

            foreach (IKernelEmitter emitter in new IKernelEmitter[]
            {
                new TritonEmitter(), new HipEmitter(), new OpenClEmitter(),
                new MslEmitter(), new WgslEmitter(), new GlslEmitter()
            })
            {
                CodegenEmitResult result = emitter.Emit(graph, CodegenElementType.Float32);
                Assert.True(result.Declined, $"{emitter.Target} accepted unsupported {unsupported}.");
                Assert.Contains(unsupported.ToString(), result.DeclineReason, StringComparison.Ordinal);
            }
        }
    }

    [Fact]
    public void AllEmitters_DeclineMalformedPointwiseArityWithoutThrowing()
    {
        var graph = new CodegenGraph();
        int input = graph.AddNode(new CodegenNode(
            CodegenOpKind.LoadInput, Array.Empty<int>(), CodegenElementType.Float32,
            new[] { 4 }, 0));
        int malformed = graph.AddNode(new CodegenNode(
            CodegenOpKind.Add, new[] { input }, CodegenElementType.Float32,
            new[] { 4 }));
        graph.AddNode(new CodegenNode(
            CodegenOpKind.StoreOutput, new[] { malformed }, CodegenElementType.Float32,
            new[] { 4 }, 0));

        foreach (IKernelEmitter emitter in new IKernelEmitter[]
        {
            new TritonEmitter(), new HipEmitter(), new OpenClEmitter(),
            new MslEmitter(), new WgslEmitter(), new GlslEmitter()
        })
        {
            CodegenEmitResult result = emitter.Emit(graph, CodegenElementType.Float32);
            Assert.True(result.Declined, $"{emitter.Target} accepted malformed Add arity.");
            Assert.Contains("requires 2 input", result.DeclineReason, StringComparison.Ordinal);
        }
    }

    [Fact]
    public void AllEmitters_ProduceKernelsWithMatchingPortCounts()
    {
        var g = CodegenLowering.LowerBinaryPointwise<float>(CodegenOpKind.Sub, new[] { 64 });

        foreach (var emit in new IKernelEmitter[]
        {
            new TritonEmitter(), new HipEmitter(), new OpenClEmitter(),
            new MslEmitter(), new WgslEmitter(), new GlslEmitter(),
        })
        {
            var r = emit.Emit(g, CodegenElementType.Float32);
            Assert.False(r.Declined, $"{emit.Target} declined a supported pointwise fusion.");
            Assert.Equal(2, r.Kernel.InputCount);
            Assert.Equal(1, r.Kernel.OutputCount);
            Assert.Equal(CodegenElementType.Float32, r.Kernel.Dtype);
            Assert.Equal(CodegenTargetRuntime.For(emit.Target), r.Kernel.RequiredRuntime);
        }
    }

    [Fact]
    public void GpuSourceKernels_RejectManagedArrayExecution()
    {
        // Native GPU execution requires backend-owned device buffers, compilation, and a queue.
        // Calling the managed-array API must throw even for targets such as OpenCL whose native
        // backend dispatch is available, so no hidden host copies or CPU fallbacks are introduced.
        var g = CodegenLowering.LowerUnaryPointwise<float>(CodegenOpKind.ReLU, new[] { 4 });
        foreach (var emit in new IKernelEmitter[]
        {
            new TritonEmitter(), new HipEmitter(), new OpenClEmitter(),
            new MslEmitter(), new WgslEmitter(), new GlslEmitter(),
        })
        {
            var r = emit.Emit(g, CodegenElementType.Float32);
            Assert.False(r.Declined);
            Assert.Throws<NotSupportedException>(
                () => r.Kernel.Execute<float>(new[] { new float[4] }, new[] { new float[4] }));
        }
    }

    private static int CountOccurrences(string value, string search)
    {
        int count = 0;
        int offset = 0;
        while ((offset = value.IndexOf(search, offset, StringComparison.Ordinal)) >= 0)
        {
            count++;
            offset += search.Length;
        }
        return count;
    }

    private sealed class RecordingNativeExecutor : INativeGpuCodegenExecutor
    {
        private readonly CodegenEmitResult _result;
        private readonly bool _canExecute;

        internal RecordingNativeExecutor(
            KernelTuningBackend backend,
            CodegenEmitResult result,
            bool canExecute)
        {
            NativeCodegenBackend = backend;
            _result = result;
            _canExecute = canExecute;
        }

        public KernelTuningBackend NativeCodegenBackend { get; }

        internal int ExecutionCount { get; private set; }

        public CodegenEmitResult EmitCodegenKernel(CodegenGraph graph, CodegenElementType dtype) =>
            _result;

        public bool CanExecuteCodegenKernel(CodegenKernel kernel) => _canExecute;

        public ValueTask ExecuteCodegenKernelAsync(
            CodegenKernel kernel,
            IReadOnlyList<IGpuBuffer> inputs,
            IReadOnlyList<IGpuBuffer> outputs,
            int elementCount,
            CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested();
            ExecutionCount++;
            return default;
        }
    }

    // ─── Sub-byte LoadInput paths (Triton) ────────────────────────────

    [Fact]
    public void Triton_NF4Input_EmitsLookupTableUnpack()
    {
        // Build a graph: NF4 input → Negate → float32 output.
        // The Triton emitter must produce a NF4_LUT_* tensor and a
        // gather-based dequantisation prologue before Negate.
        var g = new CodegenGraph();
        int packed = g.AddNode(new CodegenNode(CodegenOpKind.LoadInput, Array.Empty<int>(),
            CodegenElementType.NF4, new[] { 64 }, 0));
        int neg = g.AddNode(new CodegenNode(CodegenOpKind.Negate, new[] { packed },
            CodegenElementType.Float32, new[] { 64 }));
        g.AddNode(new CodegenNode(CodegenOpKind.StoreOutput, new[] { neg },
            CodegenElementType.Float32, new[] { 64 }, 0));

        var r = new TritonEmitter().Emit(g, CodegenElementType.Float32);
        Assert.False(r.Declined);
        Assert.Contains("NF4_LUT_", r.Source);
        Assert.Contains("tl.gather", r.Source);
        Assert.Contains("packed_off_", r.Source);
    }

    [Fact]
    public void Triton_FP4Input_EmitsCanonicalLut()
    {
        var g = new CodegenGraph();
        int packed = g.AddNode(new CodegenNode(CodegenOpKind.LoadInput, Array.Empty<int>(),
            CodegenElementType.FP4, new[] { 64 }, 0));
        g.AddNode(new CodegenNode(CodegenOpKind.StoreOutput, new[] { packed },
            CodegenElementType.Float32, new[] { 64 }, 0));

        var r = new TritonEmitter().Emit(g, CodegenElementType.Float32);
        Assert.False(r.Declined);
        Assert.Contains("FP4_LUT_", r.Source);
    }

    [Fact]
    public void Triton_Int1Input_EmitsBitNetConvention()
    {
        // BitNet 1-bit weights: 0 → -1, 1 → +1.
        var g = new CodegenGraph();
        int packed = g.AddNode(new CodegenNode(CodegenOpKind.LoadInput, Array.Empty<int>(),
            CodegenElementType.Int1, new[] { 256 }, 0));
        g.AddNode(new CodegenNode(CodegenOpKind.StoreOutput, new[] { packed },
            CodegenElementType.Float32, new[] { 256 }, 0));

        var r = new TritonEmitter().Emit(g, CodegenElementType.Float32);
        Assert.False(r.Declined);
        Assert.Contains("offsets // 8", r.Source);
        Assert.Contains("* 2.0 - 1.0", r.Source);
    }

    [Fact]
    public void Triton_Int2Input_PacksFourPerByte()
    {
        var g = new CodegenGraph();
        int packed = g.AddNode(new CodegenNode(CodegenOpKind.LoadInput, Array.Empty<int>(),
            CodegenElementType.Int2, new[] { 128 }, 0));
        g.AddNode(new CodegenNode(CodegenOpKind.StoreOutput, new[] { packed },
            CodegenElementType.Float32, new[] { 128 }, 0));

        var r = new TritonEmitter().Emit(g, CodegenElementType.Float32);
        Assert.False(r.Declined);
        Assert.Contains("offsets // 4", r.Source);
        Assert.Contains("0x3", r.Source);
    }

    [Fact]
    public void Triton_Int3Input_NibbleLayoutWithSlackBit()
    {
        var g = new CodegenGraph();
        int packed = g.AddNode(new CodegenNode(CodegenOpKind.LoadInput, Array.Empty<int>(),
            CodegenElementType.Int3, new[] { 64 }, 0));
        g.AddNode(new CodegenNode(CodegenOpKind.StoreOutput, new[] { packed },
            CodegenElementType.Float32, new[] { 64 }, 0));

        var r = new TritonEmitter().Emit(g, CodegenElementType.Float32);
        Assert.False(r.Declined);
        Assert.Contains("0x7", r.Source);
    }

    [Fact]
    public void Triton_DeclinesSubByteOutput()
    {
        // Sub-byte StoreOutput requires atomic byte updates — out of scope.
        var g = new CodegenGraph();
        int x = g.AddNode(new CodegenNode(CodegenOpKind.LoadInput, Array.Empty<int>(),
            CodegenElementType.Float32, new[] { 64 }, 0));
        g.AddNode(new CodegenNode(CodegenOpKind.StoreOutput, new[] { x },
            CodegenElementType.NF4, new[] { 64 }, 0));

        var r = new TritonEmitter().Emit(g, CodegenElementType.Float32);
        Assert.True(r.Declined);
        Assert.Contains("StoreOutput", r.DeclineReason);
    }
}
