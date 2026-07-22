using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

internal enum DirectPtxPhiloxFillKind
{
    Uniform,
    Normal,
    BernoulliMask,
    DropThresholdMask
}

/// <summary>
/// Exact-shape standalone Philox fills shared by uniform initialization,
/// normal/noise generation, Bernoulli sampling, and public mask APIs.
/// Four adjacent values are produced per thread with no tail branch.
/// </summary>
internal sealed class PtxPhiloxFillF32Kernel : IDisposable
{
    internal const int ThreadsPerBlock = 128;
    internal const int ValuesPerThread = 4;

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int ElementCount { get; }
    internal DirectPtxPhiloxFillKind Kind { get; }
    internal string EntryPoint { get; }
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxPhiloxFillF32Kernel(
        DirectPtxRuntime runtime,
        DirectPtxPhiloxFillKind kind,
        int elementCount)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasExperimentalRngDropout(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The experimental Philox fill family targets SM86 only.");
        ValidateElementCount(elementCount);
        Kind = kind;
        ElementCount = elementCount;
        EntryPoint = GetEntryPoint(kind);
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, kind, elementCount);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor, kind, elementCount);
        _module = runtime.LoadModule(Ptx);
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo info);
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, ThreadsPerBlock);
        Blueprint.ResourceBudget.Validate(EntryPoint, info, ThreadsPerBlock, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Ptx, info,
            ThreadsPerBlock, activeBlocks, _module.JitInfoLog);
    }

    internal unsafe void LaunchRange(
        DirectPtxTensorView output,
        ulong seed,
        ulong subsequence,
        ulong counterOffset,
        float first,
        float second)
    {
        if (Kind is DirectPtxPhiloxFillKind.BernoulliMask or
            DirectPtxPhiloxFillKind.DropThresholdMask)
            throw new InvalidOperationException("Use LaunchMask for a Bernoulli fill.");
        Require(output);
        if (!PtxCompat.IsFinite(first) || !PtxCompat.IsFinite(second) ||
            (Kind == DirectPtxPhiloxFillKind.Uniform && second <= first) ||
            (Kind == DirectPtxPhiloxFillKind.Normal && second < 0f))
            throw new ArgumentOutOfRangeException(nameof(first),
                "Uniform requires finite max > min; normal requires finite stddev >= 0.");
        IntPtr outputPointer = output.Pointer;
        void** arguments = stackalloc void*[6];
        arguments[0] = &outputPointer;
        arguments[1] = &seed;
        arguments[2] = &subsequence;
        arguments[3] = &counterOffset;
        arguments[4] = &first;
        arguments[5] = &second;
        Launch(arguments);
    }

    internal unsafe void LaunchMask(
        DirectPtxTensorView output,
        ulong seed,
        ulong subsequence,
        ulong counterOffset,
        uint threshold,
        float scale)
    {
        if (Kind is not (DirectPtxPhiloxFillKind.BernoulliMask or
                         DirectPtxPhiloxFillKind.DropThresholdMask))
            throw new InvalidOperationException("LaunchMask requires a Bernoulli fill module.");
        Require(output);
        if (threshold == 0 || !PtxCompat.IsFinite(scale) || scale < 0f)
            throw new ArgumentOutOfRangeException(nameof(threshold));
        IntPtr outputPointer = output.Pointer;
        void** arguments = stackalloc void*[6];
        arguments[0] = &outputPointer;
        arguments[1] = &seed;
        arguments[2] = &subsequence;
        arguments[3] = &counterOffset;
        arguments[4] = &threshold;
        arguments[5] = &scale;
        Launch(arguments);
    }

    private unsafe void Launch(void** arguments)
    {
        uint groups = checked((uint)(ElementCount / ValuesPerThread));
        _module.Launch(
            _function,
            groups / ThreadsPerBlock, 1, 1,
            ThreadsPerBlock, 1, 1, 0, arguments);
    }

    public void Dispose() => _module.Dispose();

    internal static bool IsSupportedElementCount(int elementCount) =>
        elementCount is 4_096 or 65_536 or 1_048_576;

    internal static string EmitPtx(
        int ccMajor,
        int ccMinor,
        DirectPtxPhiloxFillKind kind,
        int elementCount)
    {
        if (!DirectPtxArchitecture.HasExperimentalRngDropout(ccMajor, ccMinor))
            throw new PlatformNotSupportedException(
                "The experimental Philox fill emitter targets SM86 only.");
        ValidateElementCount(elementCount);
        string entryPoint = GetEntryPoint(kind);
        var ptx = new StringBuilder(10_000);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {entryPoint}(");
        ptx.AppendLine("    .param .u64 output_ptr,");
        ptx.AppendLine("    .param .u64 seed,");
        ptx.AppendLine("    .param .u64 subsequence,");
        ptx.AppendLine("    .param .u64 counter_offset,");
        if (kind is DirectPtxPhiloxFillKind.BernoulliMask or
            DirectPtxPhiloxFillKind.DropThresholdMask)
        {
            ptx.AppendLine("    .param .u32 threshold,");
            ptx.AppendLine("    .param .f32 scale");
        }
        else
        {
            ptx.AppendLine("    .param .f32 first,");
            ptx.AppendLine("    .param .f32 second");
        }
        ptx.AppendLine(")");
        ptx.AppendLine($".maxntid {ThreadsPerBlock}, 1, 1");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<4>;");
        ptx.AppendLine("    .reg .b32 %r<32>;");
        ptx.AppendLine("    .reg .b64 %rd<10>;");
        ptx.AppendLine("    .reg .f32 %f<24>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [output_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [seed];");
        ptx.AppendLine("    ld.param.u64 %rd2, [subsequence];");
        ptx.AppendLine("    ld.param.u64 %rd3, [counter_offset];");
        if (kind is DirectPtxPhiloxFillKind.BernoulliMask or
                    DirectPtxPhiloxFillKind.DropThresholdMask)
        {
            ptx.AppendLine("    ld.param.u32 %r22, [threshold];");
            ptx.AppendLine("    ld.param.f32 %f8, [scale];");
        }
        else
        {
            ptx.AppendLine("    ld.param.f32 %f8, [first];");
            ptx.AppendLine("    ld.param.f32 %f9, [second];");
        }
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {ThreadsPerBlock}, %r0;");
        ptx.AppendLine("    shl.b32 %r3, %r2, 2;");
        ptx.AppendLine("    mov.b64 {%r4,%r5}, %rd1;");
        ptx.AppendLine("    mov.b64 {%r6,%r7}, %rd2;");
        ptx.AppendLine("    mov.b64 {%r8,%r9}, %rd3;");
        ptx.AppendLine("    add.cc.u32 %r10, %r8, %r2;");
        ptx.AppendLine("    addc.u32 %r11, %r9, 0;");
        ptx.AppendLine("    mov.u32 %r12, %r6;");
        ptx.AppendLine("    mov.u32 %r13, %r7;");
        PtxFusedPhiloxDropoutF32Kernel.AppendPhiloxRounds(ptx);

        switch (kind)
        {
            case DirectPtxPhiloxFillKind.Uniform:
                AppendUniformOutput(ptx);
                break;
            case DirectPtxPhiloxFillKind.Normal:
                AppendNormalOutput(ptx);
                break;
            case DirectPtxPhiloxFillKind.BernoulliMask:
                AppendMaskOutput(ptx, keepWhenLess: true);
                break;
            case DirectPtxPhiloxFillKind.DropThresholdMask:
                AppendMaskOutput(ptx, keepWhenLess: false);
                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(kind));
        }

        ptx.AppendLine("    mul.wide.u32 %rd4, %r3, 4;");
        ptx.AppendLine("    add.u64 %rd5, %rd0, %rd4;");
        ptx.AppendLine("    st.global.v4.f32 [%rd5], {%f0,%f1,%f2,%f3};");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        ptx.AppendLine($"// exact_elements={elementCount}; values_per_thread=4; no_tail_branch=1");
        return ptx.ToString();
    }

    private static void AppendUniformOutput(StringBuilder ptx)
    {
        ptx.AppendLine("    cvt.rn.f32.u32 %f0, %r10;");
        ptx.AppendLine("    cvt.rn.f32.u32 %f1, %r11;");
        ptx.AppendLine("    cvt.rn.f32.u32 %f2, %r12;");
        ptx.AppendLine("    cvt.rn.f32.u32 %f3, %r13;");
        for (int i = 0; i < 4; i++)
        {
            ptx.AppendLine($"    mul.rn.f32 %f{i}, %f{i}, 0f2F800000;");
            ptx.AppendLine($"    min.f32 %f{i}, %f{i}, 0f3F7FFFFF;");
        }
        ptx.AppendLine("    sub.rn.f32 %f10, %f9, %f8;");
        for (int i = 0; i < 4; i++)
            ptx.AppendLine($"    fma.rn.f32 %f{i}, %f{i}, %f10, %f8;");
    }

    private static void AppendMaskOutput(StringBuilder ptx, bool keepWhenLess)
    {
        for (int i = 0; i < 4; i++)
        {
            ptx.AppendLine($"    setp.lt.u32 %p{i}, %r{10 + i}, %r22;");
            ptx.AppendLine(keepWhenLess
                ? $"    selp.f32 %f{i}, %f8, 0f00000000, %p{i};"
                : $"    selp.f32 %f{i}, 0f00000000, %f8, %p{i};");
        }
    }

    private static void AppendNormalOutput(StringBuilder ptx)
    {
        for (int i = 0; i < 4; i++)
        {
            ptx.AppendLine($"    cvt.rn.f32.u32 %f{i}, %r{10 + i};");
            ptx.AppendLine($"    mul.rn.f32 %f{i}, %f{i}, 0f2F800000;");
            ptx.AppendLine($"    max.f32 %f{i}, %f{i}, 0f2F800000;");
            ptx.AppendLine($"    min.f32 %f{i}, %f{i}, 0f3F7FFFFF;");
        }
        AppendBoxMullerPair(ptx, 0, 1, 10, 11);
        AppendBoxMullerPair(ptx, 2, 3, 12, 13);
        for (int i = 0; i < 4; i++)
            ptx.AppendLine($"    fma.rn.f32 %f{i}, %f{i}, %f9, %f8;");
    }

    private static void AppendBoxMullerPair(
        StringBuilder ptx,
        int uniform0,
        int uniform1,
        int scratch,
        int secondScratch)
    {
        ptx.AppendLine($"    lg2.approx.f32 %f{scratch}, %f{uniform0};");
        ptx.AppendLine($"    mul.rn.f32 %f{scratch}, %f{scratch}, 0fBFB17218;");
        ptx.AppendLine($"    sqrt.approx.f32 %f{scratch}, %f{scratch};");
        ptx.AppendLine($"    mul.rn.f32 %f{secondScratch}, %f{uniform1}, 0f40C90FDB;");
        ptx.AppendLine($"    cos.approx.f32 %f{uniform0}, %f{secondScratch};");
        ptx.AppendLine($"    sin.approx.f32 %f{uniform1}, %f{secondScratch};");
        ptx.AppendLine($"    mul.rn.f32 %f{uniform0}, %f{uniform0}, %f{scratch};");
        ptx.AppendLine($"    mul.rn.f32 %f{uniform1}, %f{uniform1}, %f{scratch};");
    }

    private static string GetEntryPoint(DirectPtxPhiloxFillKind kind) => kind switch
    {
        DirectPtxPhiloxFillKind.Uniform => "aidotnet_philox_uniform_f32",
        DirectPtxPhiloxFillKind.Normal => "aidotnet_philox_normal_f32",
        DirectPtxPhiloxFillKind.BernoulliMask => "aidotnet_philox_bernoulli_mask_f32",
        DirectPtxPhiloxFillKind.DropThresholdMask => "aidotnet_philox_drop_threshold_mask_f32",
        _ => throw new ArgumentOutOfRangeException(nameof(kind))
    };

    private static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture,
        DirectPtxPhiloxFillKind kind,
        int elementCount)
    {
        var extent = new DirectPtxExtent(elementCount);
        return new DirectPtxKernelBlueprint(
            Operation: $"philox-{kind.ToString().ToLowerInvariant()}-f32",
            Version: 1,
            Architecture: architecture,
            Variant: $"exact-n{elementCount}-float4-t{ThreadsPerBlock}",
            Tensors:
            [
                new("output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
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
                ["counter"] = "counterOffset+float4-group; explicit subsequence",
                ["word-mapping"] = "element i consumes word i%4 of counter floor(i/4)",
                ["distribution"] = kind.ToString(),
                ["global-reads"] = "none",
                ["global-writes"] = "one-fp32x4-per-thread",
                ["global-intermediates"] = "none",
                ["temporary-device-allocation"] = "none",
                ["stride-and-tail-parameters"] = "none"
            });
    }

    private void Require(DirectPtxTensorView output)
    {
        DirectPtxTensorContract contract = Blueprint.Tensors[0];
        if (output.Pointer == IntPtr.Zero || output.PhysicalType != contract.PhysicalType ||
            output.Layout != contract.Layout || output.LogicalExtent != contract.LogicalExtent ||
            output.PhysicalExtent != contract.PhysicalExtent ||
            output.ByteLength != contract.RequiredBytes ||
            output.AllocationByteLength != contract.RequiredBytes)
            throw new ArgumentException("Output does not satisfy the exact Philox fill ABI.", nameof(output));
    }

    private static void ValidateElementCount(int elementCount)
    {
        if (!IsSupportedElementCount(elementCount))
            throw new ArgumentOutOfRangeException(nameof(elementCount),
                "Supported exact element buckets are 4096, 65536, and 1048576.");
    }
}
