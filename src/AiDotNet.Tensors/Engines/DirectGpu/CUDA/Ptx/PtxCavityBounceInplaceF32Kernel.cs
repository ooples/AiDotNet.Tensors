using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Cavity-bounce nonlinearity for issue #850, matching the NVRTC <c>cavity_bounce_inplace</c> kernel: the
/// per-bounce post-IFFT step of a spectral cavity model - <c>workReal[i] = tanh(clamp(workReal[i]*invN, -20, 20))</c>
/// and <c>workImag[i] = 0</c>, fusing the IFFT normalization, the saturating nonlinearity, and the
/// imaginary-zeroing. One thread owns one element; the clamp is <c>min</c>/<c>max</c> and the nonlinearity
/// is <c>tanh.approx</c>, so the spec is TOLERANCE-based. <c>invN</c> is a per-launch <c>.param .f32</c>
/// (the IFFT reciprocal length); <c>total</c> is baked; the launch rounds up and a single guard drops the
/// tail lanes. Two pointers plus one f32 scalar reach the launch ABI.
///
/// The specialization stays disabled by default and fails closed until three clean promotion runs clear
/// the release gate.
/// </summary>
internal sealed class PtxCavityBounceInplaceF32Kernel : IDisposable
{
    internal const int DefaultBlockThreads = 256;
    internal const string EntryPoint = "aidotnet_cavity_bounce_inplace_f32";

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int Total { get; }
    internal int BlockThreads { get; }
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxCavityBounceInplaceF32Kernel(DirectPtxRuntime runtime, int total, int blockThreads = DefaultBlockThreads)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasValidatedComplexUnary(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The checked-in cavity-bounce specialization is admitted only on SM86.");
        Validate(total);
        ValidateBlockThreads(blockThreads);
        Total = total;
        BlockThreads = blockThreads;
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, total, blockThreads);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor, total, blockThreads);
        _module = runtime.LoadModule(Ptx);
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo info);
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, BlockThreads);
        Blueprint.ResourceBudget.Validate(EntryPoint, info, BlockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Ptx, info, BlockThreads, activeBlocks, _module.JitInfoLog);
    }

    internal unsafe void Launch(DirectPtxTensorView workReal, DirectPtxTensorView workImag, float invN)
    {
        Require(workReal, Blueprint.Tensors[0], nameof(workReal));
        Require(workImag, Blueprint.Tensors[1], nameof(workImag));

        IntPtr workRealPointer = workReal.Pointer, workImagPointer = workImag.Pointer;
        float invNArg = invN;
        void** arguments = stackalloc void*[3];
        arguments[0] = &workRealPointer;
        arguments[1] = &workImagPointer;
        arguments[2] = &invNArg;
        _module.Launch(
            _function,
            (uint)((Total + BlockThreads - 1) / BlockThreads), 1, 1,
            checked((uint)BlockThreads), 1, 1,
            0,
            arguments);
    }

    public void Dispose() => _module.Dispose();

    private static string Hex(float value) => "0f" + BitConverter.ToInt32(BitConverter.GetBytes(value), 0).ToString("X8");

    internal static string EmitPtx(int ccMajor, int ccMinor, int total, int blockThreads = DefaultBlockThreads)
    {
        Validate(total);
        ValidateBlockThreads(blockThreads);
        string cap = Hex(20f), neg = Hex(-20f);

        var ptx = new StringBuilder(1_920);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine($"// exact-shape total={total} block={blockThreads} op=cavity-bounce-inplace");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 workreal_ptr,");
        ptx.AppendLine("    .param .u64 workimag_ptr,");
        ptx.AppendLine("    .param .f32 inv_n");
        ptx.AppendLine(")");
        ptx.AppendLine($".maxntid {blockThreads}, 1, 1");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<2>;");
        ptx.AppendLine("    .reg .b32 %r<3>;");
        ptx.AppendLine("    .reg .b64 %rd<8>;");
        ptx.AppendLine("    .reg .f32 %f<3>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [workreal_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [workimag_ptr];");
        ptx.AppendLine("    ld.param.f32 %f1, [inv_n];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {blockThreads}, %r0;");
        ptx.AppendLine($"    setp.ge.u32 %p0, %r2, {total};");
        ptx.AppendLine("    @%p0 bra $CB_RET;");
        ptx.AppendLine("    mul.wide.u32 %rd2, %r2, 4;");
        ptx.AppendLine("    add.u64 %rd3, %rd0, %rd2;");
        ptx.AppendLine("    ld.global.f32 %f0, [%rd3];");
        ptx.AppendLine("    mul.rn.f32 %f0, %f0, %f1;");        // r = workReal * invN
        ptx.AppendLine($"    max.f32 %f0, %f0, {neg};");       // clamp lower
        ptx.AppendLine($"    min.f32 %f0, %f0, {cap};");       // clamp upper
        ptx.AppendLine("    tanh.approx.f32 %f0, %f0;");
        ptx.AppendLine("    st.global.f32 [%rd3], %f0;");
        ptx.AppendLine("    add.u64 %rd4, %rd1, %rd2;");
        ptx.AppendLine("    st.global.f32 [%rd4], 0f00000000;");   // workImag = 0
        ptx.AppendLine("$CB_RET:");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture, int total, int blockThreads)
    {
        var extent = new DirectPtxExtent(total);
        return new DirectPtxKernelBlueprint(
            Operation: "cavity-bounce-inplace-f32",
            Version: 1,
            Architecture: architecture,
            Variant: $"b{blockThreads}-n{total}",
            Tensors:
            [
                new("workReal", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    extent, extent, 16, DirectPtxTensorAccess.ReadWrite, DirectPtxExtentMode.Exact),
                new("workImag", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    extent, extent, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 16,
                MaxStaticSharedBytes: 0,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 1536 / blockThreads),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["formula"] = "workReal[i] = tanh(clamp(workReal[i]*invN, -20, 20)); workImag[i] = 0",
                ["mode"] = "inference-forward-cavity-bounce",
                ["arithmetic"] = "mul by invN, min/max clamp, tanh.approx; tolerance-based parity",
                ["scalar"] = "invN is a per-launch .param .f32 (IFFT reciprocal length)",
                ["bounds-check"] = "single guard drops lanes past total",
                ["global-intermediates"] = "in-place on workReal/workImag",
                ["temporary-device-allocation"] = "none",
                ["byte-offset"] = "zero-entire-allocation-view",
                ["padding"] = "none-logical-equals-physical"
            });
    }

    internal static bool IsSupportedShape(int total) => total >= 1 && total <= (1 << 26);

    internal static bool IsPromotedShape(int total) => false;

    private static void Validate(int total)
    {
        if (!IsSupportedShape(total))
            throw new ArgumentOutOfRangeException(nameof(total),
                "The cavity-bounce family supports totals in [1, 2^26].");
    }

    private static void ValidateBlockThreads(int blockThreads)
    {
        if (blockThreads is not (128 or 256 or 512))
            throw new ArgumentOutOfRangeException(nameof(blockThreads),
                "Cavity-bounce block threads must be 128, 256, or 512.");
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
