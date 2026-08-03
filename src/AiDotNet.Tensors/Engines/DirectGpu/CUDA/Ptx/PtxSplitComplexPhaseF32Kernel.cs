using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Split real/imag FP32 complex phase for issue #850: <c>outPhase[i] = atan2(inImag[i], inReal[i])</c>.
/// One thread owns one element. PTX has no <c>atan2</c> primitive, so the angle is reconstructed from a
/// minimax <c>atan</c> polynomial on [0,1] (~1e-4) plus quadrant folding, matching <c>atan2f</c> to a
/// modest tolerance - unlike the other split-complex operators, this one is therefore covered by a
/// tolerance-based parity spec rather than a bit-exact one. No shared memory, no reduction, no branch;
/// the launch covers exactly the element count. Three pointers reach the launch ABI.
///
/// The specialization stays disabled by default and fails closed until three clean promotion runs clear
/// the release gate.
/// </summary>
internal sealed class PtxSplitComplexPhaseF32Kernel : IDisposable
{
    internal const int DefaultBlockThreads = 256;
    internal const string EntryPoint = "aidotnet_split_complex_phase_f32";

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int Count { get; }
    internal int BlockThreads { get; }
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxSplitComplexPhaseF32Kernel(
        DirectPtxRuntime runtime, int count, int blockThreads = DefaultBlockThreads)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasValidatedComplexUnary(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The checked-in split-complex phase specialization is admitted only on SM86.");
        Validate(count);
        ValidateBlockThreads(count, blockThreads);
        Count = count;
        BlockThreads = blockThreads;
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, count, blockThreads);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor, count, blockThreads);
        _module = runtime.LoadModule(Ptx);
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo info);
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, BlockThreads);
        Blueprint.ResourceBudget.Validate(EntryPoint, info, BlockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Ptx, info, BlockThreads, activeBlocks, _module.JitInfoLog);
    }

    internal unsafe void Launch(DirectPtxTensorView inReal, DirectPtxTensorView inImag, DirectPtxTensorView outPhase)
    {
        Require(inReal, Blueprint.Tensors[0], nameof(inReal));
        Require(inImag, Blueprint.Tensors[1], nameof(inImag));
        Require(outPhase, Blueprint.Tensors[2], nameof(outPhase));

        IntPtr inRealPointer = inReal.Pointer, inImagPointer = inImag.Pointer, outPointer = outPhase.Pointer;
        void** arguments = stackalloc void*[3];
        arguments[0] = &inRealPointer;
        arguments[1] = &inImagPointer;
        arguments[2] = &outPointer;
        _module.Launch(
            _function,
            checked((uint)(Count / BlockThreads)), 1, 1,
            checked((uint)BlockThreads), 1, 1,
            0,
            arguments);
    }

    public void Dispose() => _module.Dispose();

    private static string Hex(float value) => "0f" + BitConverter.ToInt32(BitConverter.GetBytes(value), 0).ToString("X8");

    internal static string EmitPtx(int ccMajor, int ccMinor, int count, int blockThreads = DefaultBlockThreads)
    {
        Validate(count);
        ValidateBlockThreads(count, blockThreads);
        string c0 = Hex(0.9998660f), c1 = Hex(-0.3302995f), c2 = Hex(0.1801410f),
               c3 = Hex(-0.0851330f), c4 = Hex(0.0208351f);
        string pi = Hex((float)Math.PI), halfPi = Hex((float)(Math.PI / 2.0)), tiny = Hex(1e-20f);
        const string negOne = "0fBF800000";

        var ptx = new StringBuilder(4_000);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine($"// exact-shape count={count} block={blockThreads} layout=split-complex-fp32 op=split-complex-phase");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 in_real_ptr,");
        ptx.AppendLine("    .param .u64 in_imag_ptr,");
        ptx.AppendLine("    .param .u64 out_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine($".maxntid {blockThreads}, 1, 1");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<6>;");
        ptx.AppendLine("    .reg .b32 %r<3>;");
        ptx.AppendLine("    .reg .b64 %rd<10>;");
        ptx.AppendLine("    .reg .f32 %f<24>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [in_real_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [in_imag_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [out_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {blockThreads}, %r0;");
        ptx.AppendLine("    mul.wide.u32 %rd3, %r2, 4;");
        ptx.AppendLine("    add.u64 %rd4, %rd0, %rd3;");
        ptx.AppendLine("    add.u64 %rd5, %rd1, %rd3;");
        ptx.AppendLine("    ld.global.nc.f32 %f0, [%rd4];");   // re
        ptx.AppendLine("    ld.global.nc.f32 %f1, [%rd5];");   // im
        ptx.AppendLine("    abs.f32 %f2, %f0;");               // ax
        ptx.AppendLine("    abs.f32 %f3, %f1;");               // ay
        ptx.AppendLine("    max.f32 %f4, %f2, %f3;");          // mx
        ptx.AppendLine("    min.f32 %f5, %f2, %f3;");          // mn
        ptx.AppendLine($"    setp.lt.f32 %p0, %f4, {tiny};");
        ptx.AppendLine("    rcp.approx.f32 %f6, %f4;");
        ptx.AppendLine("    mul.rn.f32 %f7, %f5, %f6;");        // a = mn/mx
        ptx.AppendLine("    mul.rn.f32 %f8, %f7, %f7;");        // t = a^2
        ptx.AppendLine($"    mov.f32 %f9, {c4};");
        ptx.AppendLine($"    fma.rn.f32 %f9, %f9, %f8, {c3};");
        ptx.AppendLine($"    fma.rn.f32 %f9, %f9, %f8, {c2};");
        ptx.AppendLine($"    fma.rn.f32 %f9, %f9, %f8, {c1};");
        ptx.AppendLine($"    fma.rn.f32 %f9, %f9, %f8, {c0};");
        ptx.AppendLine("    mul.rn.f32 %f10, %f7, %f9;");       // r = atan(a) in [0,pi/4]
        ptx.AppendLine($"    mul.rn.f32 %f11, %f10, {negOne};");
        ptx.AppendLine($"    add.rn.f32 %f11, %f11, {halfPi};");
        ptx.AppendLine("    setp.gt.f32 %p1, %f3, %f2;");        // ay > ax
        ptx.AppendLine("    selp.f32 %f12, %f11, %f10, %p1;");
        ptx.AppendLine($"    mul.rn.f32 %f13, %f12, {negOne};");
        ptx.AppendLine($"    add.rn.f32 %f13, %f13, {pi};");
        ptx.AppendLine("    setp.lt.f32 %p2, %f0, 0f00000000;");  // re < 0
        ptx.AppendLine("    selp.f32 %f14, %f13, %f12, %p2;");
        ptx.AppendLine("    neg.f32 %f15, %f14;");
        ptx.AppendLine("    setp.lt.f32 %p3, %f1, 0f00000000;");  // im < 0
        ptx.AppendLine("    selp.f32 %f14, %f15, %f14, %p3;");
        ptx.AppendLine("    selp.f32 %f14, 0f00000000, %f14, %p0;");
        ptx.AppendLine("    add.u64 %rd6, %rd2, %rd3;");
        ptx.AppendLine("    st.global.f32 [%rd6], %f14;");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture, int count, int blockThreads)
    {
        var extent = new DirectPtxExtent(count);
        return new DirectPtxKernelBlueprint(
            Operation: "split-complex-phase-f32",
            Version: 1,
            Architecture: architecture,
            Variant: $"split-b{blockThreads}-n{count}",
            Tensors:
            [
                new("inReal", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    extent, extent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("inImag", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    extent, extent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("outPhase", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    extent, extent, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 24,
                MaxStaticSharedBytes: 0,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 1536 / blockThreads),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["formula"] = "outPhase[i] = atan2(inImag[i], inReal[i])",
                ["mode"] = "inference-forward-split-complex-phase",
                ["approximation"] = "minimax atan on [0,1] (~1e-4) + quadrant folding; tolerance-based parity, not bit-exact",
                ["input-layout"] = "canonical-split-real-imag-contiguous",
                ["output-layout"] = "contiguous-scalar",
                ["global-input-reads"] = "two-fp32-per-element",
                ["global-output-writes"] = "one-fp32-per-element",
                ["bounds-check"] = "none - the launch covers exactly the element count",
                ["shared-intermediate"] = "none",
                ["global-intermediates"] = "none",
                ["temporary-device-allocation"] = "none",
                ["stride-parameters"] = "none",
                ["byte-offset"] = "zero-entire-allocation-view",
                ["padding"] = "none-logical-equals-physical"
            });
    }

    internal static bool IsSupportedShape(int count) =>
        count is 65_536 or 262_144 or 1_048_576 or 4_194_304;

    internal static bool IsPromotedShape(int count) => false;

    private static void Validate(int count)
    {
        if (!IsSupportedShape(count))
            throw new ArgumentOutOfRangeException(nameof(count),
                "The first split-complex phase family supports exact counts 65536, 262144, 1048576, and 4194304.");
    }

    private static void ValidateBlockThreads(int count, int blockThreads)
    {
        if (blockThreads is not (128 or 256 or 512) || count % blockThreads != 0)
            throw new ArgumentOutOfRangeException(nameof(blockThreads),
                "Split complex phase block threads must be 128, 256, or 512 and evenly tile the element count.");
    }

    private static void Require(DirectPtxTensorView view, DirectPtxTensorContract contract, string parameter)
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
