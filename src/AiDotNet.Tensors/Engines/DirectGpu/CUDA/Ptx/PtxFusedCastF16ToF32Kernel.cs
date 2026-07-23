using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Exact contiguous FP16-to-FP32 dtype conversion (issue #845). This is the
/// widening mirror of <see cref="PtxFusedCastF32ToF16Kernel"/>: each thread
/// loads one packed FP16x4 vector, applies four <c>cvt.f32.f16</c> widenings,
/// and commits one FP32x4 vector. There are no shared-memory, local-memory,
/// global-intermediate, temporary-allocation, division, remainder, stride, or
/// scalar shape parameters — only two tensor pointers reach the launch ABI.
///
/// Widening FP16 to FP32 is exact: every FP16 value (including subnormals,
/// infinities, and NaN payloads) is representable in FP32, so no rounding mode
/// applies and the conversion is bit-exact against a double-precision oracle.
///
/// The specialization stays disabled by default and fails closed until three
/// clean promotion runs clear the release gate.
/// </summary>
internal sealed class PtxFusedCastF16ToF32Kernel : IDisposable
{
    internal const string EntryPoint = "aidotnet_fused_cast_f16_to_f32";
    internal const int DefaultBlockThreads = 256;
    /// <summary>
    /// Independent vectors each thread moves. Two rather than one so both loads
    /// are issued before either is consumed: a pure streaming cast is limited by
    /// how many bytes a thread keeps in flight, not by arithmetic.
    /// </summary>
    internal const int VectorsPerThread = 2;

    internal const int ElementsPerVector = 4;
    internal const int ElementsPerThread = VectorsPerThread * ElementsPerVector;

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int Size { get; }
    internal int BlockThreads { get; }
    internal int ElementsPerBlock => BlockThreads * ElementsPerThread;
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxFusedCastF16ToF32Kernel(
        DirectPtxRuntime runtime,
        int size,
        int blockThreads = DefaultBlockThreads)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasValidatedCastFp32(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The checked-in FP16->FP32 cast specialization is admitted only on SM86.");
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
            $"// exact-shape size={size} block={blockThreads} elems-per-thread={ElementsPerThread} strategy=linear-vec4x2 op=cast-f16-f32");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 input_ptr,");
        ptx.AppendLine("    .param .u64 output_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine($".maxntid {blockThreads}, 1, 1");
        ptx.AppendLine("{");
        ptx.AppendLine($"    .reg .b16 %rs<{ElementsPerThread}>;");
        ptx.AppendLine("    .reg .b32 %r<3>;");
        ptx.AppendLine("    .reg .b64 %rd<6>;");
        ptx.AppendLine($"    .reg .f32 %f<{ElementsPerThread}>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [input_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [output_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %ctaid.x;");
        ptx.AppendLine("    mov.u32 %r1, %tid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r0, {blockThreads}, %r1;");
        // Input is 2 bytes/element, output is 4 — the byte strides are the exact
        // mirror of the narrowing kernel's.
        ptx.AppendLine($"    mul.wide.u32 %rd2, %r2, {ElementsPerThread * 2};");
        ptx.AppendLine($"    mul.wide.u32 %rd3, %r2, {ElementsPerThread * sizeof(float)};");
        ptx.AppendLine("    add.u64 %rd4, %rd0, %rd2;");
        ptx.AppendLine("    add.u64 %rd5, %rd1, %rd3;");
        // Issue every load before consuming any of them, so the two vectors are
        // in flight together. The input is streamed once and never revisited, so
        // it goes through the read-only data cache rather than polluting L1.
        for (int v = 0; v < VectorsPerThread; v++)
        {
            int b = v * ElementsPerVector;
            int offset = v * ElementsPerVector * 2;
            ptx.AppendLine(
                $"    ld.global.nc.v4.u16 {{%rs{b}, %rs{b + 1}, %rs{b + 2}, %rs{b + 3}}}, [%rd4+{offset}];");
        }
        // FP16 -> FP32 is exact for every input, so no rounding modifier applies.
        for (int i = 0; i < ElementsPerThread; i++)
            ptx.AppendLine($"    cvt.f32.f16 %f{i}, %rs{i};");
        for (int v = 0; v < VectorsPerThread; v++)
        {
            int b = v * ElementsPerVector;
            int offset = v * ElementsPerVector * sizeof(float);
            ptx.AppendLine(
                $"    st.global.v4.f32 [%rd5+{offset}], {{%f{b}, %f{b + 1}, %f{b + 2}, %f{b + 3}}};");
        }
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
            Operation: "cast-f16-to-f32",
            Version: 1,
            Architecture: architecture,
            Variant: $"linear-vec4-b{blockThreads}-n{size}",
            Tensors:
            [
                new("input", DirectPtxPhysicalType.Float16, DirectPtxPhysicalLayout.Vector,
                    extent, extent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    extent, extent, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 16,
                MaxStaticSharedBytes: 0,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 1536 / blockThreads),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["formula"] = "output[i] = (fp32)input[i]",
                ["mode"] = "inference-forward-cvt-f32-f16",
                ["input"] = "fp16",
                ["output"] = "fp32",
                ["elements-per-thread"] = ElementsPerThread.ToString(),
                ["global-input-reads"] = "one-vector-per-thread",
                ["global-output-writes"] = "one-vector-per-thread",
                ["lane-vector-transaction"] = "aligned-fp16x4-in-fp32x4-out",
                ["shared-intermediate"] = "none",
                ["global-intermediates"] = "none",
                ["temporary-device-allocation"] = "none",
                ["stride-parameters"] = "none",
                ["rounding"] = "exact-widening-no-rounding",
                ["byte-offset"] = "zero-entire-allocation-view",
                ["padding"] = "none-logical-equals-physical"
            });
    }

    /// <summary>The widening family mirrors the narrowing family's exact shapes.</summary>
    internal static bool IsSupportedShape(int size) =>
        size is 65_536 or 262_144 or 1_048_576 or 4_194_304;

    internal static bool IsPromotedShape(int size) => false;

    private static void Validate(int size)
    {
        if (!IsSupportedShape(size))
            throw new ArgumentOutOfRangeException(nameof(size),
                "The first FP16->FP32 cast family supports exact sizes " +
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
