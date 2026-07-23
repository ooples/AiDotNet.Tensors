using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// RFFT positive-frequency extraction for issue #850, matching the NVRTC <c>rfft_postprocess</c> kernel:
/// after a full length-<c>n</c> complex FFT of real input, copy the first <c>n/2 + 1</c> bins (the
/// non-redundant positive-frequency half, including DC and Nyquist) from the full split spectrum into the
/// packed output. The transform length <c>n</c> (a power of two) is baked; the output length
/// <c>outLen = n/2 + 1</c> is not a power of two, so the launch rounds up and a single bounds guard drops
/// the tail lanes. It is pure bit-exact data movement - both lanes are copied verbatim, so NaN payloads
/// and signed zeros pass through unchanged. Four pointers reach the launch ABI.
///
/// The specialization stays disabled by default and fails closed until three clean promotion runs clear
/// the release gate.
/// </summary>
internal sealed class PtxRfftPostprocessF32Kernel : IDisposable
{
    internal const int DefaultBlockThreads = 256;
    internal const string EntryPoint = "aidotnet_rfft_postprocess_f32";

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int N { get; }
    internal int OutLen { get; }
    internal int BlockThreads { get; }
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxRfftPostprocessF32Kernel(DirectPtxRuntime runtime, int n, int blockThreads = DefaultBlockThreads)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasValidatedComplexUnary(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The checked-in RFFT postprocess specialization is admitted only on SM86.");
        Validate(n);
        ValidateBlockThreads(blockThreads);
        N = n;
        OutLen = n / 2 + 1;
        BlockThreads = blockThreads;
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, n, blockThreads);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor, n, blockThreads);
        _module = runtime.LoadModule(Ptx);
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo info);
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, BlockThreads);
        Blueprint.ResourceBudget.Validate(EntryPoint, info, BlockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Ptx, info, BlockThreads, activeBlocks, _module.JitInfoLog);
    }

    internal unsafe void Launch(
        DirectPtxTensorView fullReal, DirectPtxTensorView fullImag,
        DirectPtxTensorView outReal, DirectPtxTensorView outImag)
    {
        Require(fullReal, Blueprint.Tensors[0], nameof(fullReal));
        Require(fullImag, Blueprint.Tensors[1], nameof(fullImag));
        Require(outReal, Blueprint.Tensors[2], nameof(outReal));
        Require(outImag, Blueprint.Tensors[3], nameof(outImag));

        IntPtr fullRealPointer = fullReal.Pointer, fullImagPointer = fullImag.Pointer;
        IntPtr outRealPointer = outReal.Pointer, outImagPointer = outImag.Pointer;
        void** arguments = stackalloc void*[4];
        arguments[0] = &fullRealPointer;
        arguments[1] = &fullImagPointer;
        arguments[2] = &outRealPointer;
        arguments[3] = &outImagPointer;
        uint grid = (uint)((OutLen + BlockThreads - 1) / BlockThreads);
        _module.Launch(
            _function,
            grid, 1, 1,
            checked((uint)BlockThreads), 1, 1,
            0,
            arguments);
    }

    public void Dispose() => _module.Dispose();

    internal static string EmitPtx(int ccMajor, int ccMinor, int n, int blockThreads = DefaultBlockThreads)
    {
        Validate(n);
        ValidateBlockThreads(blockThreads);
        int outLen = n / 2 + 1;

        var ptx = new StringBuilder(2_304);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine($"// exact-shape n={n} outlen={outLen} block={blockThreads} op=rfft-postprocess");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 full_real_ptr,");
        ptx.AppendLine("    .param .u64 full_imag_ptr,");
        ptx.AppendLine("    .param .u64 out_real_ptr,");
        ptx.AppendLine("    .param .u64 out_imag_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine($".maxntid {blockThreads}, 1, 1");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<2>;");
        ptx.AppendLine("    .reg .b32 %r<3>;");
        ptx.AppendLine("    .reg .b64 %rd<12>;");
        ptx.AppendLine("    .reg .f32 %f<3>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [full_real_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [full_imag_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [out_real_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd3, [out_imag_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {blockThreads}, %r0;");
        ptx.AppendLine($"    setp.ge.u32 %p0, %r2, {outLen};");   // drop tail lanes
        ptx.AppendLine("    @%p0 bra $RFFT_END;");
        ptx.AppendLine("    mul.wide.u32 %rd4, %r2, 4;");
        ptx.AppendLine("    add.u64 %rd6, %rd0, %rd4;");
        ptx.AppendLine("    add.u64 %rd7, %rd1, %rd4;");
        ptx.AppendLine("    ld.global.nc.f32 %f0, [%rd6];");
        ptx.AppendLine("    ld.global.nc.f32 %f1, [%rd7];");
        ptx.AppendLine("    add.u64 %rd8, %rd2, %rd4;");
        ptx.AppendLine("    add.u64 %rd9, %rd3, %rd4;");
        ptx.AppendLine("    st.global.f32 [%rd8], %f0;");
        ptx.AppendLine("    st.global.f32 [%rd9], %f1;");
        ptx.AppendLine("$RFFT_END:");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture, int n, int blockThreads)
    {
        var full = new DirectPtxExtent(n);
        var packed = new DirectPtxExtent(n / 2 + 1);
        return new DirectPtxKernelBlueprint(
            Operation: "rfft-postprocess-f32",
            Version: 1,
            Architecture: architecture,
            Variant: $"b{blockThreads}-n{n}",
            Tensors:
            [
                new("fullReal", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    full, full, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("fullImag", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    full, full, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("outReal", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    packed, packed, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact),
                new("outImag", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    packed, packed, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 16,
                MaxStaticSharedBytes: 0,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 1536 / blockThreads),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["formula"] = "outReal[i]=fullReal[i]; outImag[i]=fullImag[i] for i in [0, n/2]",
                ["mode"] = "inference-forward-rfft-postprocess",
                ["arithmetic"] = "none - pure bit-exact copy; NaN payloads and signed zeros preserved",
                ["output-length"] = "n/2+1 positive-frequency bins including DC and Nyquist",
                ["bounds-check"] = "single guard drops lanes past n/2 (output length is not a power of two)",
                ["global-intermediates"] = "none",
                ["temporary-device-allocation"] = "none",
                ["byte-offset"] = "zero-entire-allocation-view",
                ["padding"] = "none-logical-equals-physical"
            });
    }

    internal static bool IsSupportedShape(int n) =>
        n >= 512 && (n & (n - 1)) == 0 && n <= (1 << 24);

    internal static bool IsPromotedShape(int n) => false;

    private static void Validate(int n)
    {
        if (!IsSupportedShape(n))
            throw new ArgumentOutOfRangeException(nameof(n),
                "The RFFT postprocess family supports power-of-two lengths n>=512 up to 2^24.");
    }

    private static void ValidateBlockThreads(int blockThreads)
    {
        if (blockThreads is not (128 or 256 or 512))
            throw new ArgumentOutOfRangeException(nameof(blockThreads),
                "RFFT postprocess block threads must be 128, 256, or 512.");
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
