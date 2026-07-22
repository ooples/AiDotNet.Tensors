#if NET5_0_OR_GREATER
using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// One-warp categorical sampling from exactly 32 contiguous probabilities.
/// The normalized CDF scan, Philox draw, and one-hot materialization stay in a
/// single warp and launch; no CDF or sampled-index intermediate reaches VRAM.
/// </summary>
internal sealed class PtxPhiloxCategorical32F32Kernel : IDisposable
{
    internal const string EntryPoint = "aidotnet_philox_categorical32_f32";
    internal const int Classes = 32;
    internal const int ThreadsPerBlock = Classes;

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int Rows { get; }
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxPhiloxCategorical32F32Kernel(DirectPtxRuntime runtime, int rows)
    {
        ArgumentNullException.ThrowIfNull(runtime);
        if (!DirectPtxArchitecture.HasExperimentalRngDropout(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The experimental categorical specialization targets SM86 only.");
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
        DirectPtxTensorView probabilities,
        DirectPtxTensorView oneHot,
        ulong seed,
        ulong subsequence,
        ulong counterOffset)
    {
        Require(probabilities, Blueprint.Tensors[0], nameof(probabilities));
        Require(oneHot, Blueprint.Tensors[1], nameof(oneHot));
        if (PtxFusedPhiloxDropoutF32Kernel.RangesOverlap(probabilities, oneHot))
            throw new ArgumentException("Categorical input and output may not alias.");
        IntPtr probabilityPointer = probabilities.Pointer;
        IntPtr outputPointer = oneHot.Pointer;
        void** arguments = stackalloc void*[5];
        arguments[0] = &probabilityPointer;
        arguments[1] = &outputPointer;
        arguments[2] = &seed;
        arguments[3] = &subsequence;
        arguments[4] = &counterOffset;
        _module.Launch(
            _function,
            checked((uint)Rows), 1, 1,
            ThreadsPerBlock, 1, 1, 0, arguments);
    }

    public void Dispose() => _module.Dispose();

    internal static bool IsSupportedShape(int rows, int classes) =>
        classes == Classes && rows is 128 or 2_048 or 32_768;

    internal static string EmitPtx(int ccMajor, int ccMinor, int rows)
    {
        if (!DirectPtxArchitecture.HasExperimentalRngDropout(ccMajor, ccMinor))
            throw new PlatformNotSupportedException(
                "The experimental categorical emitter targets SM86 only.");
        ValidateRows(rows);
        var ptx = new StringBuilder(9_000);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 probabilities_ptr,");
        ptx.AppendLine("    .param .u64 one_hot_ptr,");
        ptx.AppendLine("    .param .u64 seed,");
        ptx.AppendLine("    .param .u64 subsequence,");
        ptx.AppendLine("    .param .u64 counter_offset");
        ptx.AppendLine(")");
        ptx.AppendLine($".maxntid {ThreadsPerBlock}, 1, 1");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<8>;");
        ptx.AppendLine("    .reg .b32 %r<32>;");
        ptx.AppendLine("    .reg .b64 %rd<12>;");
        ptx.AppendLine("    .reg .f32 %f<16>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [probabilities_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [one_hot_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [seed];");
        ptx.AppendLine("    ld.param.u64 %rd3, [subsequence];");
        ptx.AppendLine("    ld.param.u64 %rd4, [counter_offset];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine("    mad.lo.u32 %r2, %r1, 32, %r0;");
        ptx.AppendLine("    mul.wide.u32 %rd5, %r2, 4;");
        ptx.AppendLine("    add.u64 %rd6, %rd0, %rd5;");
        ptx.AppendLine("    add.u64 %rd7, %rd1, %rd5;");
        ptx.AppendLine("    ld.global.f32 %f0, [%rd6];");
        ptx.AppendLine("    max.f32 %f0, %f0, 0f00000000;");
        ptx.AppendLine("    mov.f32 %f1, %f0;");
        foreach (int offset in new[] { 1, 2, 4, 8, 16 })
        {
            ptx.AppendLine("    mov.b32 %r23, %f1;");
            ptx.AppendLine($"    shfl.sync.up.b32 %r24, %r23, {offset}, 0, 0xffffffff;");
            ptx.AppendLine("    mov.b32 %f2, %r24;");
            ptx.AppendLine($"    setp.ge.u32 %p0, %r0, {offset};");
            ptx.AppendLine("    selp.f32 %f2, %f2, 0f00000000, %p0;");
            ptx.AppendLine("    add.rn.f32 %f1, %f1, %f2;");
        }
        ptx.AppendLine("    mov.b32 %r23, %f1;");
        ptx.AppendLine("    shfl.sync.idx.b32 %r24, %r23, 31, 0x1f, 0xffffffff;");
        ptx.AppendLine("    mov.b32 %f3, %r24;");
        ptx.AppendLine("    mov.b64 {%r4,%r5}, %rd2;");
        ptx.AppendLine("    mov.b64 {%r6,%r7}, %rd3;");
        ptx.AppendLine("    mov.b64 {%r8,%r9}, %rd4;");
        ptx.AppendLine("    add.cc.u32 %r10, %r8, %r1;");
        ptx.AppendLine("    addc.u32 %r11, %r9, 0;");
        ptx.AppendLine("    mov.u32 %r12, %r6;");
        ptx.AppendLine("    mov.u32 %r13, %r7;");
        PtxFusedPhiloxDropoutF32Kernel.AppendPhiloxRounds(ptx);
        ptx.AppendLine("    cvt.rn.f32.u32 %f4, %r10;");
        ptx.AppendLine("    mul.rn.f32 %f4, %f4, 0f2F800000;");
        ptx.AppendLine("    min.f32 %f4, %f4, 0f3F7FFFFF;");
        ptx.AppendLine("    mul.rn.f32 %f5, %f4, %f3;");
        ptx.AppendLine("    sub.rn.f32 %f6, %f1, %f0;");
        ptx.AppendLine("    setp.ge.f32 %p1, %f5, %f6;");
        ptx.AppendLine("    setp.lt.f32 %p2, %f5, %f1;");
        ptx.AppendLine("    and.pred %p3, %p1, %p2;");
        ptx.AppendLine("    setp.le.f32 %p4, %f3, 0f2EDBE6FF;");
        ptx.AppendLine("    setp.eq.u32 %p5, %r0, 0;");
        ptx.AppendLine("    and.pred %p6, %p4, %p5;");
        ptx.AppendLine("    or.pred %p3, %p3, %p6;");
        ptx.AppendLine("    selp.f32 %f7, 0f3F800000, 0f00000000, %p3;");
        ptx.AppendLine("    st.global.f32 [%rd7], %f7;");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        ptx.AppendLine($"// exact_shape=[{rows},32]; one_warp_per_row=1; no_tail_branch=1");
        return ptx.ToString();
    }

    private static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture,
        int rows)
    {
        var extent = new DirectPtxExtent(rows, Classes);
        return new DirectPtxKernelBlueprint(
            Operation: "philox-categorical32-f32",
            Version: 1,
            Architecture: architecture,
            Variant: $"exact-r{rows}-c32-warp-onehot",
            Tensors:
            [
                new("probabilities", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    extent, extent, 4, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("one-hot", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    extent, extent, 4, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 56,
                MaxStaticSharedBytes: 0,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 16),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["rng-abi"] = "philox4x32-10-v1",
                ["counter"] = "counterOffset+row; subsequence in high64",
                ["probabilities"] = "finite non-negative row; normalization is implicit via target=uniform*rowSum",
                ["zero-sum"] = "select class zero deterministically",
                ["output"] = "one-hot FP32 sample",
                ["global-reads"] = "one scalar probability per lane",
                ["global-writes"] = "one final one-hot scalar per lane",
                ["register-resident"] = "probability, CDF prefix, row sum, target, selection predicate",
                ["global-intermediates"] = "none",
                ["temporary-device-allocation"] = "none",
                ["stride-and-tail-parameters"] = "none"
            });
    }

    private static void ValidateRows(int rows)
    {
        if (!IsSupportedShape(rows, Classes))
            throw new ArgumentOutOfRangeException(
                nameof(rows), "Supported exact row counts are 128, 2048, and 32768 for 32 classes.");
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
#endif
