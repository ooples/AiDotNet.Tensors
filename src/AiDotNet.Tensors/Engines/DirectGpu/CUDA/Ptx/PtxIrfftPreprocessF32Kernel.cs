using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// IRFFT Hermitian reconstruction for issue #850, matching the NVRTC <c>irfft_preprocess</c> kernel:
/// expand the packed positive-frequency half (<c>inLen = n/2 + 1</c> bins) into a full length-<c>n</c>
/// complex spectrum before the inverse FFT. Bin <c>i</c> in the lower half is copied verbatim; bin
/// <c>i</c> in the upper half is filled by conjugate symmetry from its mirror <c>n - i</c>
/// (<c>fullReal[i] = inReal[n-i]</c>, <c>fullImag[i] = -inImag[n-i]</c>). The transform length <c>n</c>
/// (a power of two) is baked, so the launch covers exactly <c>n</c> threads with no bounds guard; a single
/// branch selects the copy versus mirror path. The imaginary mirror uses <c>neg.f32</c>, a sign-bit flip,
/// so it is bit-exact and preserves NaN payloads and signed zeros. Four pointers reach the launch ABI.
///
/// The specialization stays disabled by default and fails closed until three clean promotion runs clear
/// the release gate.
/// </summary>
internal sealed class PtxIrfftPreprocessF32Kernel : IDisposable
{
    internal const int DefaultBlockThreads = 256;
    internal const string EntryPoint = "aidotnet_irfft_preprocess_f32";

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int N { get; }
    internal int InLen { get; }
    internal int BlockThreads { get; }
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxIrfftPreprocessF32Kernel(DirectPtxRuntime runtime, int n, int blockThreads = DefaultBlockThreads)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasValidatedComplexUnary(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The checked-in IRFFT preprocess specialization is admitted only on SM86.");
        Validate(n);
        ValidateBlockThreads(n, blockThreads);
        N = n;
        InLen = n / 2 + 1;
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
        DirectPtxTensorView inReal, DirectPtxTensorView inImag,
        DirectPtxTensorView fullReal, DirectPtxTensorView fullImag)
    {
        Require(inReal, Blueprint.Tensors[0], nameof(inReal));
        Require(inImag, Blueprint.Tensors[1], nameof(inImag));
        Require(fullReal, Blueprint.Tensors[2], nameof(fullReal));
        Require(fullImag, Blueprint.Tensors[3], nameof(fullImag));

        IntPtr inRealPointer = inReal.Pointer, inImagPointer = inImag.Pointer;
        IntPtr fullRealPointer = fullReal.Pointer, fullImagPointer = fullImag.Pointer;
        void** arguments = stackalloc void*[4];
        arguments[0] = &inRealPointer;
        arguments[1] = &inImagPointer;
        arguments[2] = &fullRealPointer;
        arguments[3] = &fullImagPointer;
        _module.Launch(
            _function,
            checked((uint)(N / BlockThreads)), 1, 1,
            checked((uint)BlockThreads), 1, 1,
            0,
            arguments);
    }

    public void Dispose() => _module.Dispose();

    internal static string EmitPtx(int ccMajor, int ccMinor, int n, int blockThreads = DefaultBlockThreads)
    {
        Validate(n);
        ValidateBlockThreads(n, blockThreads);
        int inLen = n / 2 + 1;

        var ptx = new StringBuilder(2_816);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine($"// exact-shape n={n} inlen={inLen} block={blockThreads} op=irfft-preprocess");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 in_real_ptr,");
        ptx.AppendLine("    .param .u64 in_imag_ptr,");
        ptx.AppendLine("    .param .u64 full_real_ptr,");
        ptx.AppendLine("    .param .u64 full_imag_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine($".maxntid {blockThreads}, 1, 1");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<2>;");
        ptx.AppendLine("    .reg .b32 %r<5>;");
        ptx.AppendLine("    .reg .b64 %rd<14>;");
        ptx.AppendLine("    .reg .f32 %f<3>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [in_real_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [in_imag_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [full_real_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd3, [full_imag_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {blockThreads}, %r0;");   // idx
        ptx.AppendLine("    mul.wide.u32 %rd4, %r2, 4;");
        ptx.AppendLine("    add.u64 %rd5, %rd2, %rd4;");   // &fullReal[idx]
        ptx.AppendLine("    add.u64 %rd6, %rd3, %rd4;");   // &fullImag[idx]
        ptx.AppendLine($"    setp.ge.u32 %p0, %r2, {inLen};");   // upper half -> mirror
        ptx.AppendLine("    @%p0 bra $IRFFT_MIRROR;");
        // copy path: source index = idx
        ptx.AppendLine("    add.u64 %rd7, %rd0, %rd4;");
        ptx.AppendLine("    add.u64 %rd8, %rd1, %rd4;");
        ptx.AppendLine("    ld.global.nc.f32 %f0, [%rd7];");
        ptx.AppendLine("    ld.global.nc.f32 %f1, [%rd8];");
        ptx.AppendLine("    st.global.f32 [%rd5], %f0;");
        ptx.AppendLine("    st.global.f32 [%rd6], %f1;");
        ptx.AppendLine("    bra $IRFFT_END;");
        // mirror path: source index = n - idx, imag negated (conjugate symmetry)
        ptx.AppendLine("$IRFFT_MIRROR:");
        ptx.AppendLine($"    sub.u32 %r3, {n}, %r2;");    // mirrorIdx = n - idx
        ptx.AppendLine("    mul.wide.u32 %rd9, %r3, 4;");
        ptx.AppendLine("    add.u64 %rd10, %rd0, %rd9;");
        ptx.AppendLine("    add.u64 %rd11, %rd1, %rd9;");
        ptx.AppendLine("    ld.global.nc.f32 %f0, [%rd10];");
        ptx.AppendLine("    ld.global.nc.f32 %f1, [%rd11];");
        ptx.AppendLine("    neg.f32 %f1, %f1;");          // conjugate
        ptx.AppendLine("    st.global.f32 [%rd5], %f0;");
        ptx.AppendLine("    st.global.f32 [%rd6], %f1;");
        ptx.AppendLine("$IRFFT_END:");
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
            Operation: "irfft-preprocess-f32",
            Version: 1,
            Architecture: architecture,
            Variant: $"b{blockThreads}-n{n}",
            Tensors:
            [
                new("inReal", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    packed, packed, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("inImag", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    packed, packed, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("fullReal", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    full, full, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact),
                new("fullImag", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    full, full, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 20,
                MaxStaticSharedBytes: 0,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 1536 / blockThreads),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["formula"] = "lower half copied; upper half fullReal[i]=inReal[n-i], fullImag[i]=-inImag[n-i]",
                ["mode"] = "inference-forward-irfft-preprocess",
                ["arithmetic"] = "copy + neg.f32 conjugate (sign-bit flip); NaN payloads and signed zeros preserved",
                ["input-length"] = "n/2+1 packed positive-frequency bins",
                ["bounds-check"] = "none - the launch covers exactly n; one branch selects copy vs mirror",
                ["global-intermediates"] = "none",
                ["temporary-device-allocation"] = "none",
                ["byte-offset"] = "zero-entire-allocation-view",
                ["padding"] = "none-logical-equals-physical"
            });
    }

    internal static bool IsSupportedShape(int n) =>
        n >= 512 && (n & (n - 1)) == 0 && n % DefaultBlockThreads == 0 && n <= (1 << 24);

    internal static bool IsPromotedShape(int n) => false;

    private static void Validate(int n)
    {
        if (!IsSupportedShape(n))
            throw new ArgumentOutOfRangeException(nameof(n),
                "The IRFFT preprocess family supports power-of-two lengths n>=512 that are a multiple of 256, up to 2^24.");
    }

    private static void ValidateBlockThreads(int n, int blockThreads)
    {
        if (blockThreads is not (128 or 256 or 512) || n % blockThreads != 0)
            throw new ArgumentOutOfRangeException(nameof(blockThreads),
                "IRFFT preprocess block threads must be 128, 256, or 512 and evenly tile n.");
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
