using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Exact-shape FP32 dropout using the versioned Philox4x32-10 counter ABI.
/// Each thread produces four adjacent random words, reads one float4, and
/// writes the required inverted-dropout mask and result in a single launch.
/// The public mask is materialized because backward consumes it; no private
/// random-mask or scaled-input intermediate reaches global memory.
/// </summary>
internal sealed class PtxFusedPhiloxDropoutF32Kernel : IDisposable
{
    internal const int ThreadsPerBlock = 128;
    internal const int ValuesPerThread = 4;
    internal const string EntryPoint = "aidotnet_philox_dropout_f32";

    internal const uint PhiloxM0 = 0xD2511F53u;
    internal const uint PhiloxM1 = 0xCD9E8D57u;
    internal const uint PhiloxW0 = 0x9E3779B9u;
    internal const uint PhiloxW1 = 0xBB67AE85u;

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int ElementCount { get; }
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxFusedPhiloxDropoutF32Kernel(DirectPtxRuntime runtime, int elementCount)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasExperimentalRngDropout(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The experimental Philox dropout specialization targets SM86 only.");
        ValidateElementCount(elementCount);

        ElementCount = elementCount;
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, elementCount);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor, elementCount);
        _module = runtime.LoadModule(Ptx);
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo info);
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, ThreadsPerBlock);
        Blueprint.ResourceBudget.Validate(EntryPoint, info, ThreadsPerBlock, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Ptx, info,
            ThreadsPerBlock, activeBlocks, _module.JitInfoLog);
    }

    internal unsafe void Launch(
        DirectPtxTensorView input,
        DirectPtxTensorView output,
        DirectPtxTensorView mask,
        ulong seed,
        ulong subsequence,
        ulong counterOffset,
        uint keepThreshold,
        float inverseKeep)
    {
        Require(input, Blueprint.Tensors[0], nameof(input));
        Require(output, Blueprint.Tensors[1], nameof(output));
        Require(mask, Blueprint.Tensors[2], nameof(mask));
        if (keepThreshold == 0 || !PtxCompat.IsFinite(inverseKeep) || inverseKeep <= 1.0f)
            throw new ArgumentOutOfRangeException(
                nameof(keepThreshold), "Dropout requires a non-empty keep interval and finite inverse keep scale > 1.");
        if (RangesOverlap(input, output) || RangesOverlap(input, mask) || RangesOverlap(output, mask))
            throw new ArgumentException("Direct PTX dropout input, output, and mask may not alias.");

        IntPtr inputPointer = input.Pointer;
        IntPtr outputPointer = output.Pointer;
        IntPtr maskPointer = mask.Pointer;
        void** arguments = stackalloc void*[8];
        arguments[0] = &inputPointer;
        arguments[1] = &outputPointer;
        arguments[2] = &maskPointer;
        arguments[3] = &seed;
        arguments[4] = &subsequence;
        arguments[5] = &counterOffset;
        arguments[6] = &keepThreshold;
        arguments[7] = &inverseKeep;
        uint groups = checked((uint)(ElementCount / ValuesPerThread));
        _module.Launch(
            _function,
            groups / ThreadsPerBlock, 1, 1,
            ThreadsPerBlock, 1, 1, 0, arguments);
    }

    public void Dispose() => _module.Dispose();

    internal static bool IsSupportedElementCount(int elementCount) =>
        elementCount is 4_096 or 65_536 or 1_048_576;

    internal static string EmitPtx(int ccMajor, int ccMinor, int elementCount)
    {
        if (!DirectPtxArchitecture.HasExperimentalRngDropout(ccMajor, ccMinor))
            throw new PlatformNotSupportedException(
                "The experimental Philox dropout emitter targets SM86 only.");
        ValidateElementCount(elementCount);
        int groups = elementCount / ValuesPerThread;
        int blocks = groups / ThreadsPerBlock;
        var ptx = new StringBuilder(9_000);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 input_ptr,");
        ptx.AppendLine("    .param .u64 output_ptr,");
        ptx.AppendLine("    .param .u64 mask_ptr,");
        ptx.AppendLine("    .param .u64 seed,");
        ptx.AppendLine("    .param .u64 subsequence,");
        ptx.AppendLine("    .param .u64 counter_offset,");
        ptx.AppendLine("    .param .u32 keep_threshold,");
        ptx.AppendLine("    .param .f32 inverse_keep");
        ptx.AppendLine(")");
        ptx.AppendLine($".maxntid {ThreadsPerBlock}, 1, 1");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<4>;");
        ptx.AppendLine("    .reg .b32 %r<32>;");
        ptx.AppendLine("    .reg .b64 %rd<12>;");
        ptx.AppendLine("    .reg .f32 %f<16>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [input_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [output_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [mask_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd3, [seed];");
        ptx.AppendLine("    ld.param.u64 %rd4, [subsequence];");
        ptx.AppendLine("    ld.param.u64 %rd5, [counter_offset];");
        ptx.AppendLine("    ld.param.u32 %r22, [keep_threshold];");
        ptx.AppendLine("    ld.param.f32 %f0, [inverse_keep];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {ThreadsPerBlock}, %r0;");
        ptx.AppendLine("    shl.b32 %r3, %r2, 2;");
        ptx.AppendLine("    mov.b64 {%r4,%r5}, %rd3;");
        ptx.AppendLine("    mov.b64 {%r6,%r7}, %rd4;");
        ptx.AppendLine("    mov.b64 {%r8,%r9}, %rd5;");
        ptx.AppendLine("    add.cc.u32 %r10, %r8, %r2;");
        ptx.AppendLine("    addc.u32 %r11, %r9, 0;");
        ptx.AppendLine("    mov.u32 %r12, %r6;");
        ptx.AppendLine("    mov.u32 %r13, %r7;");
        AppendPhiloxRounds(ptx);
        ptx.AppendLine("    setp.lt.u32 %p0, %r10, %r22;");
        ptx.AppendLine("    setp.lt.u32 %p1, %r11, %r22;");
        ptx.AppendLine("    setp.lt.u32 %p2, %r12, %r22;");
        ptx.AppendLine("    setp.lt.u32 %p3, %r13, %r22;");
        ptx.AppendLine("    selp.f32 %f1, %f0, 0f00000000, %p0;");
        ptx.AppendLine("    selp.f32 %f2, %f0, 0f00000000, %p1;");
        ptx.AppendLine("    selp.f32 %f3, %f0, 0f00000000, %p2;");
        ptx.AppendLine("    selp.f32 %f4, %f0, 0f00000000, %p3;");
        ptx.AppendLine("    mul.wide.u32 %rd6, %r3, 4;");
        ptx.AppendLine("    add.u64 %rd7, %rd0, %rd6;");
        ptx.AppendLine("    add.u64 %rd8, %rd1, %rd6;");
        ptx.AppendLine("    add.u64 %rd9, %rd2, %rd6;");
        ptx.AppendLine("    ld.global.v4.f32 {%f5,%f6,%f7,%f8}, [%rd7];");
        ptx.AppendLine("    mul.rn.f32 %f9, %f5, %f1;");
        ptx.AppendLine("    mul.rn.f32 %f10, %f6, %f2;");
        ptx.AppendLine("    mul.rn.f32 %f11, %f7, %f3;");
        ptx.AppendLine("    mul.rn.f32 %f12, %f8, %f4;");
        ptx.AppendLine("    st.global.v4.f32 [%rd8], {%f9,%f10,%f11,%f12};");
        ptx.AppendLine("    st.global.v4.f32 [%rd9], {%f1,%f2,%f3,%f4};");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        ptx.AppendLine($"// exact_elements={elementCount}; exact_blocks={blocks}; no_tail_branch=1");
        return ptx.ToString();
    }

    /// <summary>CPU reference for the versioned device counter ABI.</summary>
    internal static (uint X0, uint X1, uint X2, uint X3) GenerateUInt4(
        ulong seed, ulong subsequence, ulong counter)
    {
        uint c0 = (uint)counter;
        uint c1 = (uint)(counter >> 32);
        uint c2 = (uint)subsequence;
        uint c3 = (uint)(subsequence >> 32);
        uint k0 = (uint)seed;
        uint k1 = (uint)(seed >> 32);
        for (int round = 0; round < 10; round++)
        {
            ulong p0 = (ulong)PhiloxM0 * c0;
            ulong p1 = (ulong)PhiloxM1 * c2;
            (c0, c1, c2, c3) =
                ((uint)(p1 >> 32) ^ c1 ^ k0, (uint)p1,
                 (uint)(p0 >> 32) ^ c3 ^ k1, (uint)p0);
            if (round != 9)
            {
                k0 = unchecked(k0 + PhiloxW0);
                k1 = unchecked(k1 + PhiloxW1);
            }
        }
        return (c0, c1, c2, c3);
    }

    /// <summary>
    /// Emits the shared Philox4x32-10 core. Callers initialize counter words
    /// r10-r13 and key words r4-r5; the generated words replace r10-r13.
    /// Scratch registers r14-r21 are clobbered.
    /// </summary>
    internal static void AppendPhiloxRounds(StringBuilder ptx)
    {
        for (int round = 0; round < 10; round++)
        {
            ptx.AppendLine($"    // Philox4x32-10 round {round}");
            ptx.AppendLine($"    mul.hi.u32 %r14, %r10, 0x{PhiloxM0:X8};");
            ptx.AppendLine($"    mul.lo.u32 %r15, %r10, 0x{PhiloxM0:X8};");
            ptx.AppendLine($"    mul.hi.u32 %r16, %r12, 0x{PhiloxM1:X8};");
            ptx.AppendLine($"    mul.lo.u32 %r17, %r12, 0x{PhiloxM1:X8};");
            ptx.AppendLine("    xor.b32 %r18, %r16, %r11;");
            ptx.AppendLine("    xor.b32 %r18, %r18, %r4;");
            ptx.AppendLine("    mov.u32 %r19, %r17;");
            ptx.AppendLine("    xor.b32 %r20, %r14, %r13;");
            ptx.AppendLine("    xor.b32 %r20, %r20, %r5;");
            ptx.AppendLine("    mov.u32 %r21, %r15;");
            ptx.AppendLine("    mov.u32 %r10, %r18;");
            ptx.AppendLine("    mov.u32 %r11, %r19;");
            ptx.AppendLine("    mov.u32 %r12, %r20;");
            ptx.AppendLine("    mov.u32 %r13, %r21;");
            if (round != 9)
            {
                ptx.AppendLine($"    add.u32 %r4, %r4, 0x{PhiloxW0:X8};");
                ptx.AppendLine($"    add.u32 %r5, %r5, 0x{PhiloxW1:X8};");
            }
        }
    }

    private static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture,
        int elementCount)
    {
        var extent = new DirectPtxExtent(elementCount);
        return new DirectPtxKernelBlueprint(
            Operation: "philox-dropout-f32",
            Version: 1,
            Architecture: architecture,
            Variant: $"exact-n{elementCount}-float4-t{ThreadsPerBlock}",
            Tensors:
            [
                new("input", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    extent, extent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    extent, extent, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact),
                new("mask", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    extent, extent, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 56,
                MaxStaticSharedBytes: 0,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 8),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["rng-abi"] = "philox4x32-10-v1",
                ["key"] = "seed-low32,seed-high32",
                ["counter"] = "counterOffset+float4-group in low64; subsequence in high64",
                ["word-mapping"] = "element i consumes word i%4 of counter floor(i/4)",
                ["uniform"] = "unsigned-word comparison against floor(keepProbability*2^32)",
                ["mask"] = "kept=1/keepProbability; dropped=0",
                ["output"] = "input*mask",
                ["global-reads"] = "one float4 input transaction per thread",
                ["global-writes"] = "one float4 output plus required public mask per thread",
                ["global-intermediates"] = "none",
                ["temporary-device-allocation"] = "none",
                ["stride-and-tail-parameters"] = "none"
            });
    }

    private static void ValidateElementCount(int elementCount)
    {
        if (!IsSupportedElementCount(elementCount))
            throw new ArgumentOutOfRangeException(
                nameof(elementCount),
                "Supported exact element buckets are 4096, 65536, and 1048576.");
    }

    private static void Require(
        DirectPtxTensorView view,
        DirectPtxTensorContract contract,
        string parameter)
    {
        if (view.Pointer == IntPtr.Zero || view.PhysicalType != contract.PhysicalType ||
            view.Layout != contract.Layout || view.LogicalExtent != contract.LogicalExtent ||
            view.PhysicalExtent != contract.PhysicalExtent || view.ByteLength != contract.RequiredBytes ||
            view.AllocationByteLength != contract.RequiredBytes)
            throw new ArgumentException(
                $"{parameter} does not satisfy exact physical ABI '{contract.Name}'.", parameter);
    }

    internal static bool RangesOverlap(DirectPtxTensorView left, DirectPtxTensorView right)
    {
        nuint leftStart = PtxCompat.ToNuint(left.Pointer);
        nuint rightStart = PtxCompat.ToNuint(right.Pointer);
        nuint leftEnd = checked(leftStart + left.ByteLength);
        nuint rightEnd = checked(rightStart + right.ByteLength);
        return leftStart < rightEnd && rightStart < leftEnd;
    }
}
