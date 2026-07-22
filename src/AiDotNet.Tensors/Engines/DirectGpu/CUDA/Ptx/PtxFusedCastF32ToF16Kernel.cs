#if NET5_0_OR_GREATER
using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Exact contiguous FP32-to-FP16 dtype conversion (issue #845). Each thread
/// loads one FP32x4 vector, applies four round-to-nearest-even
/// <c>cvt.rn.f16.f32</c> conversions, and commits one packed FP16x4 vector.
/// There are no shared-memory, local-memory, global-intermediate,
/// temporary-allocation, division, remainder, stride, or scalar shape
/// parameters — only two tensor pointers reach the launch ABI. The
/// specialization stays disabled by default and fails closed until three clean
/// promotion runs clear the release gate.
/// </summary>
internal sealed class PtxFusedCastF32ToF16Kernel : IDisposable
{
    internal const string EntryPoint = "aidotnet_fused_cast_f32_to_f16";
    internal const int DefaultBlockThreads = 256;
    internal const int ElementsPerThread = 4;

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int Size { get; }
    internal int BlockThreads { get; }
    internal int ElementsPerBlock => BlockThreads * ElementsPerThread;
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxFusedCastF32ToF16Kernel(
        DirectPtxRuntime runtime,
        int size,
        int blockThreads = DefaultBlockThreads)
    {
        ArgumentNullException.ThrowIfNull(runtime);
        if (runtime.ArchitectureFamily != DirectPtxArchitectureFamily.Ampere)
            throw new PlatformNotSupportedException(
                "The checked-in FP32->FP16 cast specialization is validated only on Ampere.");
        Validate(size);
        ValidateBlockThreads(size, blockThreads);
        Size = size;
        BlockThreads = blockThreads;
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, size, blockThreads);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor, size, blockThreads);
        _module = runtime.LoadModule(Ptx);
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo info);
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, BlockThreads);
        Blueprint.ResourceBudget.Validate(EntryPoint, info, BlockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Ptx, info, BlockThreads, activeBlocks, _module.JitInfoLog);
    }

    internal unsafe void Launch(
        DirectPtxTensorView input,
        DirectPtxTensorView output)
    {
        Require(input, Blueprint.Tensors[0], nameof(input));
        Require(output, Blueprint.Tensors[1], nameof(output));

        IntPtr inputPointer = input.Pointer;
        IntPtr outputPointer = output.Pointer;
        void** arguments = stackalloc void*[2];
        arguments[0] = &inputPointer;
        arguments[1] = &outputPointer;
        _module.Launch(
            _function,
            checked((uint)(Size / ElementsPerBlock)),
            1,
            1,
            checked((uint)BlockThreads),
            1,
            1,
            0,
            arguments);
    }

    public void Dispose() => _module.Dispose();

    internal static string EmitPtx(
        int ccMajor,
        int ccMinor,
        int size,
        int blockThreads = DefaultBlockThreads)
    {
        Validate(size);
        ValidateBlockThreads(size, blockThreads);
        var ptx = new StringBuilder(2_048);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine(
            $"// exact-shape size={size} block={blockThreads} elems-per-thread={ElementsPerThread} strategy=linear-vec4 op=cast-f32-f16");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 input_ptr,");
        ptx.AppendLine("    .param .u64 output_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine($".maxntid {blockThreads}, 1, 1");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .b16 %rs<4>;");
        ptx.AppendLine("    .reg .b32 %r<3>;");
        ptx.AppendLine("    .reg .b64 %rd<6>;");
        ptx.AppendLine("    .reg .f32 %f<4>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [input_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [output_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %ctaid.x;");
        ptx.AppendLine("    mov.u32 %r1, %tid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r0, {blockThreads}, %r1;");
        ptx.AppendLine($"    mul.wide.u32 %rd2, %r2, {ElementsPerThread * sizeof(float)};");
        ptx.AppendLine($"    mul.wide.u32 %rd3, %r2, {ElementsPerThread * 2};");
        ptx.AppendLine("    add.u64 %rd4, %rd0, %rd2;");
        ptx.AppendLine("    add.u64 %rd5, %rd1, %rd3;");
        ptx.AppendLine("    ld.global.ca.v4.f32 {%f0, %f1, %f2, %f3}, [%rd4];");
        ptx.AppendLine("    cvt.rn.f16.f32 %rs0, %f0;");
        ptx.AppendLine("    cvt.rn.f16.f32 %rs1, %f1;");
        ptx.AppendLine("    cvt.rn.f16.f32 %rs2, %f2;");
        ptx.AppendLine("    cvt.rn.f16.f32 %rs3, %f3;");
        ptx.AppendLine("    st.global.v4.u16 [%rd5], {%rs0, %rs1, %rs2, %rs3};");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture,
        int size,
        int blockThreads)
    {
        var extent = new DirectPtxExtent(size);
        return new DirectPtxKernelBlueprint(
            Operation: "cast-f32-to-f16",
            Version: 1,
            Architecture: architecture,
            Variant: $"linear-vec4-b{blockThreads}-n{size}",
            Tensors:
            [
                new("input", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    extent, extent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("output", DirectPtxPhysicalType.Float16, DirectPtxPhysicalLayout.Vector,
                    extent, extent, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 16,
                MaxStaticSharedBytes: 0,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 1536 / blockThreads),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["formula"] = "output[i] = (fp16)round_nearest_even(input[i])",
                ["mode"] = "inference-forward-cvt-rn-f16",
                ["input"] = "fp32",
                ["output"] = "fp16",
                ["elements-per-thread"] = ElementsPerThread.ToString(),
                ["global-input-reads"] = "one-vector-per-thread",
                ["global-output-writes"] = "one-vector-per-thread",
                ["lane-vector-transaction"] = "aligned-fp32x4-in-fp16x4-out",
                ["shared-intermediate"] = "none",
                ["global-intermediates"] = "none",
                ["temporary-device-allocation"] = "none",
                ["stride-parameters"] = "none",
                ["rounding"] = "round-to-nearest-even",
                ["byte-offset"] = "zero-entire-allocation-view",
                ["padding"] = "none-logical-equals-physical"
            });
    }

    internal static bool IsSupportedShape(int size) =>
        size is 65_536 or 262_144 or 1_048_576 or 4_194_304;

    internal static bool IsPromotedShape(int size) => false;

    private static void Validate(int size)
    {
        if (!IsSupportedShape(size))
            throw new ArgumentOutOfRangeException(nameof(size),
                "The first FP32->FP16 cast family supports exact sizes " +
                "65536, 262144, 1048576, and 4194304.");
    }

    private static void ValidateBlockThreads(int size, int blockThreads)
    {
        if (blockThreads is not (128 or 256 or 512) ||
            size % (blockThreads * ElementsPerThread) != 0)
            throw new ArgumentOutOfRangeException(nameof(blockThreads),
                "Cast block threads must be 128, 256, or 512 and evenly tile the element count.");
    }

    private static void Require(
        DirectPtxTensorView view,
        DirectPtxTensorContract contract,
        string parameter)
    {
        if (view.Pointer == IntPtr.Zero || view.PhysicalType != contract.PhysicalType ||
            view.Layout != contract.Layout || view.LogicalExtent != contract.LogicalExtent ||
            view.PhysicalExtent != contract.PhysicalExtent ||
            view.ByteLength != contract.RequiredBytes ||
            view.AllocationByteLength != contract.RequiredBytes)
            throw new ArgumentException(
                $"{parameter} does not satisfy physical ABI '{contract.Name}'.", parameter);
    }
}
#endif
