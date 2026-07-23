using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// FFT bit-reversal permutation for issue #850, matching the NVRTC <c>bit_reverse_permutation</c>
/// kernel: reorder the split real/imag arrays so that element <c>idx</c> is swapped with the element at
/// its <c>log2(n)</c>-bit-reversed index. The transform length <c>n</c> (a power of two) is baked into
/// the PTX; the reversed index is a single hardware <c>brev.b32</c> shifted right by <c>32 - log2(n)</c>,
/// so no per-bit loop is emitted. One thread owns one element and performs the swap only when
/// <c>reversed &gt; idx</c> (the lower thread of each pair), so the in-place exchange is race-free and
/// bit-exact - pure data movement. Two pointers reach the launch ABI.
///
/// The specialization stays disabled by default and fails closed until three clean promotion runs clear
/// the release gate.
/// </summary>
internal sealed class PtxBitReversePermutationF32Kernel : IDisposable
{
    internal const int DefaultBlockThreads = 256;

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int N { get; }
    internal int Log2N { get; }
    internal int BlockThreads { get; }
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal const string EntryPoint = "aidotnet_fft_bit_reverse_f32";

    internal PtxBitReversePermutationF32Kernel(DirectPtxRuntime runtime, int n, int blockThreads = DefaultBlockThreads)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasValidatedComplexMultiply(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The checked-in FFT bit-reverse specialization is admitted only on SM86.");
        Validate(n);
        ValidateBlockThreads(n, blockThreads);
        N = n;
        Log2N = Log2(n);
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

    internal unsafe void Launch(DirectPtxTensorView real, DirectPtxTensorView imag)
    {
        Require(real, Blueprint.Tensors[0], nameof(real));
        Require(imag, Blueprint.Tensors[1], nameof(imag));

        IntPtr realPointer = real.Pointer, imagPointer = imag.Pointer;
        void** arguments = stackalloc void*[2];
        arguments[0] = &realPointer;
        arguments[1] = &imagPointer;
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
        int shift = 32 - Log2(n);

        var ptx = new StringBuilder(2_560);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine($"// exact-shape n={n} log2n={Log2(n)} block={blockThreads} op=fft-bit-reverse");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 real_ptr,");
        ptx.AppendLine("    .param .u64 imag_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine($".maxntid {blockThreads}, 1, 1");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<2>;");
        ptx.AppendLine("    .reg .b32 %r<6>;");
        ptx.AppendLine("    .reg .b64 %rd<12>;");
        ptx.AppendLine("    .reg .f32 %f<5>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [real_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [imag_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {blockThreads}, %r0;");   // idx
        ptx.AppendLine("    brev.b32 %r3, %r2;");                          // bit-reverse 32 bits
        ptx.AppendLine($"    shr.u32 %r3, %r3, {shift};");                // reversed index (log2n bits)
        ptx.AppendLine("    setp.le.u32 %p0, %r3, %r2;");                  // skip when reversed <= idx
        ptx.AppendLine("    @%p0 bra $BR_END;");
        ptx.AppendLine("    mul.wide.u32 %rd2, %r2, 4;");
        ptx.AppendLine("    mul.wide.u32 %rd3, %r3, 4;");
        ptx.AppendLine("    add.u64 %rd4, %rd0, %rd2;");   // &real[idx]
        ptx.AppendLine("    add.u64 %rd5, %rd0, %rd3;");   // &real[rev]
        ptx.AppendLine("    add.u64 %rd6, %rd1, %rd2;");   // &imag[idx]
        ptx.AppendLine("    add.u64 %rd7, %rd1, %rd3;");   // &imag[rev]
        ptx.AppendLine("    ld.global.f32 %f0, [%rd4];");  // real[idx]
        ptx.AppendLine("    ld.global.f32 %f1, [%rd5];");  // real[rev]
        ptx.AppendLine("    ld.global.f32 %f2, [%rd6];");  // imag[idx]
        ptx.AppendLine("    ld.global.f32 %f3, [%rd7];");  // imag[rev]
        ptx.AppendLine("    st.global.f32 [%rd4], %f1;");  // real[idx] = real[rev]
        ptx.AppendLine("    st.global.f32 [%rd5], %f0;");  // real[rev] = real[idx]
        ptx.AppendLine("    st.global.f32 [%rd6], %f3;");  // imag[idx] = imag[rev]
        ptx.AppendLine("    st.global.f32 [%rd7], %f2;");  // imag[rev] = imag[idx]
        ptx.AppendLine("$BR_END:");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static DirectPtxKernelBlueprint CreateBlueprint(DirectPtxArchitectureFamily architecture, int n, int blockThreads)
    {
        var extent = new DirectPtxExtent(n);
        return new DirectPtxKernelBlueprint(
            Operation: "fft-bit-reverse-f32",
            Version: 1,
            Architecture: architecture,
            Variant: $"b{blockThreads}-n{n}",
            Tensors:
            [
                new("real", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    extent, extent, 16, DirectPtxTensorAccess.ReadWrite, DirectPtxExtentMode.Exact),
                new("imag", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    extent, extent, 16, DirectPtxTensorAccess.ReadWrite, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 16,
                MaxStaticSharedBytes: 0,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 1536 / blockThreads),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["formula"] = "swap element idx with its log2(n)-bit-reversed index when reversed > idx",
                ["mode"] = "inference-forward-fft-bit-reverse",
                ["arithmetic"] = "none - pure bit-exact data movement; index via brev.b32 >> (32-log2n)",
                ["global-intermediates"] = "in-place on real/imag",
                ["temporary-device-allocation"] = "none",
                ["byte-offset"] = "zero-entire-allocation-view",
                ["padding"] = "none-logical-equals-physical"
            });
    }

    private static int Log2(int n)
    {
        int k = 0;
        while ((1 << k) < n) k++;
        return k;
    }

    internal static bool IsSupportedShape(int n) =>
        n >= 256 && (n & (n - 1)) == 0 && n % DefaultBlockThreads == 0 && n <= (1 << 24);

    internal static bool IsPromotedShape(int n) => false;

    private static void Validate(int n)
    {
        if (!IsSupportedShape(n))
            throw new ArgumentOutOfRangeException(nameof(n),
                "The FFT bit-reverse family supports power-of-two lengths n>=256 that are a multiple of 256, up to 2^24.");
    }

    private static void ValidateBlockThreads(int n, int blockThreads)
    {
        if (blockThreads is not (128 or 256 or 512) || n % blockThreads != 0)
            throw new ArgumentOutOfRangeException(nameof(blockThreads),
                "FFT bit-reverse block threads must be 128, 256, or 512 and evenly tile n.");
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
