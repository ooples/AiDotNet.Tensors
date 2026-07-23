using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Batched FFT bit-reversal permutation for issue #850, matching the NVRTC <c>batched_bit_reverse</c>
/// kernel: reorder each of <c>batch</c> contiguous length-<c>n</c> transforms independently. The batch
/// index comes from <c>gridDim.y</c> (<c>%ctaid.y</c>) and offsets into the row at <c>b*n</c>; within a
/// row the reversed index is a single <c>brev.b32</c> shifted right by <c>32 - log2(n)</c>, and the lower
/// thread of each pair performs the in-place swap. It is pure bit-exact data movement, so NaN payloads and
/// signed zeros pass through unchanged. The length <c>n</c> and <c>batch</c> are baked; the launch covers
/// exactly <c>batch * n</c> elements with no bounds guard. Two pointers reach the launch ABI.
///
/// The specialization stays disabled by default and fails closed until three clean promotion runs clear
/// the release gate.
/// </summary>
internal sealed class PtxBatchedBitReverseF32Kernel : IDisposable
{
    internal const int DefaultBlockThreads = 256;
    internal const string EntryPoint = "aidotnet_batched_bit_reverse_f32";

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int N { get; }
    internal int Batch { get; }
    internal int Log2N { get; }
    internal int BlockThreads { get; }
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxBatchedBitReverseF32Kernel(
        DirectPtxRuntime runtime, int n, int batch, int blockThreads = DefaultBlockThreads)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasValidatedComplexMultiply(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The checked-in batched FFT bit-reverse specialization is admitted only on SM86.");
        Validate(n, batch);
        ValidateBlockThreads(n, blockThreads);
        N = n;
        Batch = batch;
        Log2N = Log2(n);
        BlockThreads = blockThreads;
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, n, batch, blockThreads);
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
            checked((uint)(N / BlockThreads)), checked((uint)Batch), 1,
            checked((uint)BlockThreads), 1, 1,
            0,
            arguments);
    }

    public void Dispose() => _module.Dispose();

    internal static string EmitPtx(int ccMajor, int ccMinor, int n, int blockThreads = DefaultBlockThreads)
    {
        ValidateN(n);
        ValidateBlockThreads(n, blockThreads);
        int shift = 32 - Log2(n);

        var ptx = new StringBuilder(2_816);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine($"// exact-shape n={n} log2n={Log2(n)} block={blockThreads} op=batched-fft-bit-reverse (batch=gridDimY)");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 real_ptr,");
        ptx.AppendLine("    .param .u64 imag_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine($".maxntid {blockThreads}, 1, 1");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<2>;");
        ptx.AppendLine("    .reg .b32 %r<8>;");
        ptx.AppendLine("    .reg .b64 %rd<12>;");
        ptx.AppendLine("    .reg .f32 %f<5>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [real_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [imag_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {blockThreads}, %r0;");   // idx within row
        ptx.AppendLine("    brev.b32 %r3, %r2;");
        ptx.AppendLine($"    shr.u32 %r3, %r3, {shift};");                 // reversed within row
        ptx.AppendLine("    setp.le.u32 %p0, %r3, %r2;");
        ptx.AppendLine("    @%p0 bra $BBR_END;");
        ptx.AppendLine("    mov.u32 %r4, %ctaid.y;");                      // batch index
        ptx.AppendLine($"    mul.lo.u32 %r5, %r4, {n};");                 // baseOffset = b*n
        ptx.AppendLine("    add.u32 %r6, %r5, %r2;");                      // baseOffset + idx
        ptx.AppendLine("    add.u32 %r7, %r5, %r3;");                      // baseOffset + reversed
        ptx.AppendLine("    mul.wide.u32 %rd2, %r6, 4;");
        ptx.AppendLine("    mul.wide.u32 %rd3, %r7, 4;");
        ptx.AppendLine("    add.u64 %rd4, %rd0, %rd2;");   // &real[base+idx]
        ptx.AppendLine("    add.u64 %rd5, %rd0, %rd3;");   // &real[base+rev]
        ptx.AppendLine("    add.u64 %rd6, %rd1, %rd2;");   // &imag[base+idx]
        ptx.AppendLine("    add.u64 %rd7, %rd1, %rd3;");   // &imag[base+rev]
        ptx.AppendLine("    ld.global.f32 %f0, [%rd4];");
        ptx.AppendLine("    ld.global.f32 %f1, [%rd5];");
        ptx.AppendLine("    ld.global.f32 %f2, [%rd6];");
        ptx.AppendLine("    ld.global.f32 %f3, [%rd7];");
        ptx.AppendLine("    st.global.f32 [%rd4], %f1;");
        ptx.AppendLine("    st.global.f32 [%rd5], %f0;");
        ptx.AppendLine("    st.global.f32 [%rd6], %f3;");
        ptx.AppendLine("    st.global.f32 [%rd7], %f2;");
        ptx.AppendLine("$BBR_END:");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture, int n, int batch, int blockThreads)
    {
        var extent = new DirectPtxExtent(checked(batch * n));
        return new DirectPtxKernelBlueprint(
            Operation: "batched-fft-bit-reverse-f32",
            Version: 1,
            Architecture: architecture,
            Variant: $"b{blockThreads}-n{n}-batch{batch}",
            Tensors:
            [
                new("real", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    extent, extent, 16, DirectPtxTensorAccess.ReadWrite, DirectPtxExtentMode.Exact),
                new("imag", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    extent, extent, 16, DirectPtxTensorAccess.ReadWrite, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 20,
                MaxStaticSharedBytes: 0,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 1536 / blockThreads),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["formula"] = "per row b: swap element b*n+idx with b*n+reversed(idx) when reversed > idx",
                ["mode"] = "inference-forward-batched-fft-bit-reverse",
                ["arithmetic"] = "none - pure bit-exact data movement; index via brev.b32 >> (32-log2n)",
                ["batch-dimension"] = "gridDim.y; baseOffset = ctaid.y * n",
                ["bounds-check"] = "none - the launch covers exactly batch*n elements",
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

    internal static bool IsSupportedShape(int n, int batch) =>
        batch >= 1 && n >= 512 && (n & (n - 1)) == 0 && n % DefaultBlockThreads == 0 && n <= (1 << 24) &&
        (long)batch * n <= (1L << 27);

    internal static bool IsPromotedShape(int n, int batch) => false;

    private static void Validate(int n, int batch)
    {
        if (!IsSupportedShape(n, batch))
            throw new ArgumentOutOfRangeException(nameof(n),
                "The batched FFT bit-reverse family supports power-of-two n>=512 (multiple of 256, up to 2^24), " +
                "batch>=1, and batch*n<=2^27.");
    }

    private static void ValidateN(int n)
    {
        if (n < 512 || (n & (n - 1)) != 0 || n % DefaultBlockThreads != 0 || n > (1 << 24))
            throw new ArgumentOutOfRangeException(nameof(n),
                "The batched FFT bit-reverse family supports power-of-two n>=512 that are a multiple of 256, up to 2^24.");
    }

    private static void ValidateBlockThreads(int n, int blockThreads)
    {
        if (blockThreads is not (128 or 256 or 512) || n % blockThreads != 0)
            throw new ArgumentOutOfRangeException(nameof(blockThreads),
                "Batched FFT bit-reverse block threads must be 128, 256, or 512 and evenly tile n.");
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
