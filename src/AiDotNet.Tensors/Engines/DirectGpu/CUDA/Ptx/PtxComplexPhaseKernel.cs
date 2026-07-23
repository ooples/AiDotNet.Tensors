using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Elementwise complex phase <c>phase[i] = atan2(imag[i], real[i])</c> from separate
/// real/imag buffers (issue #854). PTX has no atan2 primitive, so this reconstructs the angle
/// from a minimax polynomial approximation of <c>atan</c> on [0,1] (≈1e-4) plus octant/quadrant
/// sign folding, matching <c>atan2f</c> to a modest tolerance. One thread owns one element —
/// no shared memory, no reduction.
///
/// 256 threads/block, grid = count/256 (positive multiple of 256).
/// </summary>
internal sealed class PtxComplexPhaseKernel : IDisposable
{
    internal const int BlockThreads = 256;
    internal const int MaxCount = 2048 * 4096;
    internal const string EntryPoint = "aidotnet_complex_phase";

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int Count { get; }
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxComplexPhaseKernel(DirectPtxRuntime runtime, int count)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasValidatedScientific(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The checked-in complex-phase specialization is measured only on GA10x/SM86.");
        ValidateShape(count);
        Count = count;
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, count);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor, count);
        _module = runtime.LoadModule(Ptx);
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo info);
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, BlockThreads);
        Blueprint.ResourceBudget.Validate(EntryPoint, info, BlockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Ptx, info, BlockThreads, activeBlocks, _module.JitInfoLog);
    }

    internal unsafe void Launch(DirectPtxTensorView real, DirectPtxTensorView imag, DirectPtxTensorView phase)
    {
        Require(real, Blueprint.Tensors[0], nameof(real));
        Require(imag, Blueprint.Tensors[1], nameof(imag));
        Require(phase, Blueprint.Tensors[2], nameof(phase));

        IntPtr realPointer = real.Pointer;
        IntPtr imagPointer = imag.Pointer;
        IntPtr phasePointer = phase.Pointer;
        void** arguments = stackalloc void*[3];
        arguments[0] = &realPointer;
        arguments[1] = &imagPointer;
        arguments[2] = &phasePointer;
        _module.Launch(_function, (uint)(Count / BlockThreads), 1, 1, BlockThreads, 1, 1, 0, arguments);
    }

    public void Dispose() => _module.Dispose();

    private static string Hex(float value) => "0f" + BitConverter.ToInt32(BitConverter.GetBytes(value), 0).ToString("X8");

    internal static string EmitPtx(int ccMajor, int ccMinor, int count)
    {
        ValidateShape(count);
        // Minimax atan(a) ~ a*(c0 + t(c1 + t(c2 + t(c3 + t c4)))), t = a^2, a in [0,1].
        string c0 = Hex(0.9998660f), c1 = Hex(-0.3302995f), c2 = Hex(0.1801410f),
               c3 = Hex(-0.0851330f), c4 = Hex(0.0208351f);
        string pi = Hex((float)Math.PI), halfPi = Hex((float)(Math.PI / 2.0)), tiny = Hex(1e-20f);
        const string NegOne = "0fBF800000";

        var ptx = new StringBuilder(5_000);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine($"// complex-phase count={count}");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 real_ptr,");
        ptx.AppendLine("    .param .u64 imag_ptr,");
        ptx.AppendLine("    .param .u64 phase_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine($".maxntid {BlockThreads}, 1, 1");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<6>;");
        ptx.AppendLine("    .reg .b32 %r<8>;");
        ptx.AppendLine("    .reg .b64 %rd<12>;");
        ptx.AppendLine("    .reg .f32 %f<24>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [real_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [imag_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [phase_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {BlockThreads}, %r0;");
        ptx.AppendLine("    mul.wide.u32 %rd3, %r2, 4;");
        ptx.AppendLine("    add.u64 %rd4, %rd0, %rd3;");
        ptx.AppendLine("    add.u64 %rd5, %rd1, %rd3;");
        ptx.AppendLine("    add.u64 %rd6, %rd2, %rd3;");
        ptx.AppendLine("    ld.global.nc.f32 %f0, [%rd4];");   // re
        ptx.AppendLine("    ld.global.nc.f32 %f1, [%rd5];");   // im
        ptx.AppendLine("    abs.f32 %f2, %f0;");               // ax
        ptx.AppendLine("    abs.f32 %f3, %f1;");               // ay
        ptx.AppendLine("    max.f32 %f4, %f2, %f3;");          // mx
        ptx.AppendLine("    min.f32 %f5, %f2, %f3;");          // mn
        // a = mn / mx (guarded); base r = a*poly(a^2).
        ptx.AppendLine($"    setp.lt.f32 %p0, %f4, {tiny};");   // mx ~ 0
        ptx.AppendLine("    rcp.approx.f32 %f6, %f4;");
        ptx.AppendLine("    mul.rn.f32 %f7, %f5, %f6;");        // a
        ptx.AppendLine("    mul.rn.f32 %f8, %f7, %f7;");        // t = a^2
        ptx.AppendLine($"    mov.f32 %f9, {c4};");
        ptx.AppendLine($"    fma.rn.f32 %f9, %f9, %f8, {c3};");
        ptx.AppendLine($"    fma.rn.f32 %f9, %f9, %f8, {c2};");
        ptx.AppendLine($"    fma.rn.f32 %f9, %f9, %f8, {c1};");
        ptx.AppendLine($"    fma.rn.f32 %f9, %f9, %f8, {c0};");
        ptx.AppendLine("    mul.rn.f32 %f10, %f7, %f9;");       // r = atan(a) in [0,pi/4]
        // fq = (ay > ax) ? (pi/2 - r) : r
        ptx.AppendLine($"    mul.rn.f32 %f11, %f10, {NegOne};");
        ptx.AppendLine($"    add.rn.f32 %f11, %f11, {halfPi};"); // pi/2 - r
        ptx.AppendLine("    setp.gt.f32 %p1, %f3, %f2;");        // ay > ax
        ptx.AppendLine("    selp.f32 %f12, %f11, %f10, %p1;");   // fq in [0,pi/2]
        // ang = (re < 0) ? (pi - fq) : fq
        ptx.AppendLine($"    mul.rn.f32 %f13, %f12, {NegOne};");
        ptx.AppendLine($"    add.rn.f32 %f13, %f13, {pi};");     // pi - fq
        ptx.AppendLine("    setp.lt.f32 %p2, %f0, 0f00000000;"); // re < 0
        ptx.AppendLine("    selp.f32 %f14, %f13, %f12, %p2;");
        // ang = (im < 0) ? -ang : ang
        ptx.AppendLine("    neg.f32 %f15, %f14;");
        ptx.AppendLine("    setp.lt.f32 %p3, %f1, 0f00000000;"); // im < 0
        ptx.AppendLine("    selp.f32 %f14, %f15, %f14, %p3;");
        // degenerate (both ~0) -> 0
        ptx.AppendLine("    selp.f32 %f14, 0f00000000, %f14, %p0;");
        ptx.AppendLine("    st.global.f32 [%rd6], %f14;");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static DirectPtxKernelBlueprint CreateBlueprint(DirectPtxArchitectureFamily architecture, int count)
    {
        var extent = new DirectPtxExtent(count);
        return new DirectPtxKernelBlueprint(
            Operation: "complex-phase",
            Version: 1,
            Architecture: architecture,
            Variant: $"fp32-count{count}",
            Tensors:
            [
                new("real", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    extent, extent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("imag", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    extent, extent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("phase", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    extent, extent, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 24,
                MaxStaticSharedBytes: 0,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 1),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["formula"] = "phase[i] = atan2(imag[i], real[i]) via minimax atan + quadrant folding",
                ["approximation"] = "atan-minimax-poly-on-[0,1]-approx-1e-4",
                ["global-intermediates"] = "none",
                ["temporary-device-allocation"] = "none",
                ["stride-parameters"] = "none"
            });
    }

    internal static bool IsSupportedCount(int count) =>
        count > 0 && count % BlockThreads == 0 && count <= MaxCount;

    internal static bool IsPromotedCount(int count) => false;

    private static void ValidateShape(int count)
    {
        if (!IsSupportedCount(count))
            throw new ArgumentOutOfRangeException(
                nameof(count),
                $"Complex phase supports a positive element count that is a multiple of {BlockThreads} up to {MaxCount}.");
    }

    private static void Require(DirectPtxTensorView view, DirectPtxTensorContract contract, string parameter)
    {
        if (view.Pointer == IntPtr.Zero || view.PhysicalType != contract.PhysicalType ||
            view.Layout != contract.Layout || view.LogicalExtent != contract.LogicalExtent ||
            view.PhysicalExtent != contract.PhysicalExtent || view.ByteLength != contract.RequiredBytes)
            throw new ArgumentException(
                $"{parameter} does not satisfy physical ABI '{contract.Name}'.", parameter);
    }
}
