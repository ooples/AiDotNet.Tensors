using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Exact row-major [rows,256] bias-add plus Philox inverted dropout. Bias,
/// random mask generation, scaling, and the add/multiply epilogue are one PTX
/// launch; only the backward-required public mask is materialized.
/// </summary>
internal sealed class PtxFusedBiasPhiloxDropout256F32Kernel : IDisposable
{
    internal const string EntryPoint = "aidotnet_bias_philox_dropout256_f32";
    internal const int Columns = 256;
    internal const int ThreadsPerBlock = 128;
    internal const int ValuesPerThread = 4;

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int Rows { get; }
    internal int ElementCount => checked(Rows * Columns);
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxFusedBiasPhiloxDropout256F32Kernel(DirectPtxRuntime runtime, int rows)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasExperimentalRngDropout(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The experimental bias-dropout specialization targets SM86 only.");
        ValidateRows(rows);
        Rows = rows;
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, rows);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor, rows);
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
        DirectPtxTensorView bias,
        DirectPtxTensorView output,
        DirectPtxTensorView mask,
        ulong seed,
        ulong subsequence,
        ulong counterOffset,
        uint keepThreshold,
        float inverseKeep)
    {
        Require(input, Blueprint.Tensors[0], nameof(input));
        Require(bias, Blueprint.Tensors[1], nameof(bias));
        Require(output, Blueprint.Tensors[2], nameof(output));
        Require(mask, Blueprint.Tensors[3], nameof(mask));
        if (keepThreshold == 0 || !PtxCompat.IsFinite(inverseKeep) || inverseKeep <= 1f)
            throw new ArgumentOutOfRangeException(nameof(keepThreshold));
        if (PtxFusedPhiloxDropoutF32Kernel.RangesOverlap(input, output) ||
            PtxFusedPhiloxDropoutF32Kernel.RangesOverlap(input, mask) ||
            PtxFusedPhiloxDropoutF32Kernel.RangesOverlap(bias, output) ||
            PtxFusedPhiloxDropoutF32Kernel.RangesOverlap(bias, mask) ||
            PtxFusedPhiloxDropoutF32Kernel.RangesOverlap(output, mask))
            throw new ArgumentException("Bias-dropout writable tensors may not alias inputs or each other.");

        IntPtr inputPointer = input.Pointer;
        IntPtr biasPointer = bias.Pointer;
        IntPtr outputPointer = output.Pointer;
        IntPtr maskPointer = mask.Pointer;
        void** arguments = stackalloc void*[9];
        arguments[0] = &inputPointer;
        arguments[1] = &biasPointer;
        arguments[2] = &outputPointer;
        arguments[3] = &maskPointer;
        arguments[4] = &seed;
        arguments[5] = &subsequence;
        arguments[6] = &counterOffset;
        arguments[7] = &keepThreshold;
        arguments[8] = &inverseKeep;
        uint groups = checked((uint)(ElementCount / ValuesPerThread));
        _module.Launch(
            _function,
            groups / ThreadsPerBlock, 1, 1,
            ThreadsPerBlock, 1, 1, 0, arguments);
    }

    public void Dispose() => _module.Dispose();

    internal static bool IsSupportedShape(int rows, int cols) =>
        cols == Columns && rows is 16 or 256 or 4_096;

    internal static string EmitPtx(int ccMajor, int ccMinor, int rows)
    {
        if (!DirectPtxArchitecture.HasExperimentalRngDropout(ccMajor, ccMinor))
            throw new PlatformNotSupportedException(
                "The experimental bias-dropout emitter targets SM86 only.");
        ValidateRows(rows);
        int elements = checked(rows * Columns);
        int groups = elements / ValuesPerThread;
        int blocks = groups / ThreadsPerBlock;
        var ptx = new StringBuilder(10_000);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 input_ptr,");
        ptx.AppendLine("    .param .u64 bias_ptr,");
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
        ptx.AppendLine("    .reg .b64 %rd<16>;");
        ptx.AppendLine("    .reg .f32 %f<24>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [input_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [bias_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [output_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd3, [mask_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd4, [seed];");
        ptx.AppendLine("    ld.param.u64 %rd5, [subsequence];");
        ptx.AppendLine("    ld.param.u64 %rd6, [counter_offset];");
        ptx.AppendLine("    ld.param.u32 %r22, [keep_threshold];");
        ptx.AppendLine("    ld.param.f32 %f0, [inverse_keep];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {ThreadsPerBlock}, %r0;");
        ptx.AppendLine("    shl.b32 %r3, %r2, 2;");
        ptx.AppendLine("    mov.b64 {%r4,%r5}, %rd4;");
        ptx.AppendLine("    mov.b64 {%r6,%r7}, %rd5;");
        ptx.AppendLine("    mov.b64 {%r8,%r9}, %rd6;");
        ptx.AppendLine("    add.cc.u32 %r10, %r8, %r2;");
        ptx.AppendLine("    addc.u32 %r11, %r9, 0;");
        ptx.AppendLine("    mov.u32 %r12, %r6;");
        ptx.AppendLine("    mov.u32 %r13, %r7;");
        PtxFusedPhiloxDropoutF32Kernel.AppendPhiloxRounds(ptx);
        for (int i = 0; i < 4; i++)
        {
            ptx.AppendLine($"    setp.lt.u32 %p{i}, %r{10 + i}, %r22;");
            ptx.AppendLine($"    selp.f32 %f{1 + i}, %f0, 0f00000000, %p{i};");
        }
        ptx.AppendLine("    mul.wide.u32 %rd7, %r3, 4;");
        ptx.AppendLine("    and.b32 %r23, %r2, 63;");
        ptx.AppendLine("    mul.wide.u32 %rd8, %r23, 16;");
        ptx.AppendLine("    add.u64 %rd9, %rd0, %rd7;");
        ptx.AppendLine("    add.u64 %rd10, %rd1, %rd8;");
        ptx.AppendLine("    add.u64 %rd11, %rd2, %rd7;");
        ptx.AppendLine("    add.u64 %rd12, %rd3, %rd7;");
        ptx.AppendLine("    ld.global.v4.f32 {%f5,%f6,%f7,%f8}, [%rd9];");
        ptx.AppendLine("    ld.global.v4.f32 {%f9,%f10,%f11,%f12}, [%rd10];");
        for (int i = 0; i < 4; i++)
        {
            ptx.AppendLine($"    add.rn.f32 %f{13 + i}, %f{5 + i}, %f{9 + i};");
            ptx.AppendLine($"    mul.rn.f32 %f{17 + i}, %f{13 + i}, %f{1 + i};");
        }
        ptx.AppendLine("    st.global.v4.f32 [%rd11], {%f17,%f18,%f19,%f20};");
        ptx.AppendLine("    st.global.v4.f32 [%rd12], {%f1,%f2,%f3,%f4};");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        ptx.AppendLine($"// exact_shape=[{rows},256]; exact_blocks={blocks}; no_tail_branch=1");
        return ptx.ToString();
    }

    private static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture,
        int rows)
    {
        var matrix = new DirectPtxExtent(rows, Columns);
        var bias = new DirectPtxExtent(Columns);
        return new DirectPtxKernelBlueprint(
            Operation: "bias-philox-dropout256-f32",
            Version: 1,
            Architecture: architecture,
            Variant: $"exact-r{rows}-c256-float4-t{ThreadsPerBlock}",
            Tensors:
            [
                new("input", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    matrix, matrix, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("bias", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    bias, bias, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    matrix, matrix, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact),
                new("saved-mask", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    matrix, matrix, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 64,
                MaxStaticSharedBytes: 0,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 8),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["rng-abi"] = "philox4x32-10-v1",
                ["counter"] = "counterOffset+float4-group; subsequence in high64",
                ["output"] = "(input+bias[column])*savedMask",
                ["mask"] = "kept=1/keepProbability; dropped=0; materialized for public API/backward",
                ["global-reads"] = "one input float4 plus one bias float4 per thread",
                ["global-writes"] = "one output float4 plus one required saved-mask float4 per thread",
                ["register-resident"] = "Philox words, bias sum, mask, scaled result",
                ["global-intermediates"] = "none",
                ["temporary-device-allocation"] = "none",
                ["stride-and-tail-parameters"] = "none"
            });
    }

    private static void ValidateRows(int rows)
    {
        if (!IsSupportedShape(rows, Columns))
            throw new ArgumentOutOfRangeException(
                nameof(rows), "Supported exact [rows,256] layouts use 16, 256, or 4096 rows.");
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
}
