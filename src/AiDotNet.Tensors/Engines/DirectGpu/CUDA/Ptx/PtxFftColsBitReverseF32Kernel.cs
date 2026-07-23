using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Column-wise FFT bit-reversal for issue #850, matching the NVRTC <c>fft_cols_bit_reverse</c> kernel:
/// permute each column of a row-major <c>height x width</c> split matrix in place along the column
/// (stride <c>width</c>) before the DIT column butterflies. The 2D FFT's row pass is the contiguous
/// batched FFT (<see cref="PtxBatchedBitReverseF32Kernel"/> + <see cref="PtxBatchedFftButterflyF32Kernel"/>);
/// this kernel supplies the strided column pass. Each thread owns one <c>(column, row)</c> cell, flattened
/// as <c>gid = row*width + col</c> is decomposed back to <c>col = gid % width</c>, <c>row = gid / width</c>;
/// the reversed row index is a single <c>brev.b32</c> shifted right by <c>32 - log2(height)</c>, and the
/// lower thread of each pair swaps <c>[row*width+col]</c> with <c>[reversed*width+col]</c>. It is pure
/// bit-exact data movement. <c>width</c> and <c>height</c> are baked; the launch covers exactly
/// <c>width * height</c> cells with no bounds guard. Two pointers reach the launch ABI.
///
/// The specialization stays disabled by default and fails closed until three clean promotion runs clear
/// the release gate.
/// </summary>
internal sealed class PtxFftColsBitReverseF32Kernel : IDisposable
{
    internal const int DefaultBlockThreads = 256;
    internal const string EntryPoint = "aidotnet_fft_cols_bit_reverse_f32";

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int Height { get; }
    internal int Width { get; }
    internal int Log2Height { get; }
    internal int BlockThreads { get; }
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxFftColsBitReverseF32Kernel(
        DirectPtxRuntime runtime, int height, int width, int blockThreads = DefaultBlockThreads)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasValidatedComplexMultiply(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The checked-in column FFT bit-reverse specialization is admitted only on SM86.");
        Validate(height, width);
        ValidateBlockThreads(height, width, blockThreads);
        Height = height;
        Width = width;
        Log2Height = Log2(height);
        BlockThreads = blockThreads;
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, height, width, blockThreads);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor, height, width, blockThreads);
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
            checked((uint)((Width * Height) / BlockThreads)), 1, 1,
            checked((uint)BlockThreads), 1, 1,
            0,
            arguments);
    }

    public void Dispose() => _module.Dispose();

    internal static string EmitPtx(int ccMajor, int ccMinor, int height, int width, int blockThreads = DefaultBlockThreads)
    {
        Validate(height, width);
        ValidateBlockThreads(height, width, blockThreads);
        int shift = 32 - Log2(height);

        var ptx = new StringBuilder(2_816);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine($"// exact-shape height={height} width={width} log2h={Log2(height)} block={blockThreads} op=fft-cols-bit-reverse");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 real_ptr,");
        ptx.AppendLine("    .param .u64 imag_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine($".maxntid {blockThreads}, 1, 1");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<2>;");
        ptx.AppendLine("    .reg .b32 %r<10>;");
        ptx.AppendLine("    .reg .b64 %rd<12>;");
        ptx.AppendLine("    .reg .f32 %f<5>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [real_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [imag_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {blockThreads}, %r0;");   // gid
        ptx.AppendLine($"    rem.u32 %r3, %r2, {width};");                 // col
        ptx.AppendLine($"    div.u32 %r4, %r2, {width};");                 // row
        ptx.AppendLine("    brev.b32 %r5, %r4;");
        ptx.AppendLine($"    shr.u32 %r5, %r5, {shift};");                 // reversed row
        ptx.AppendLine("    setp.le.u32 %p0, %r5, %r4;");
        ptx.AppendLine("    @%p0 bra $FCBR_END;");
        ptx.AppendLine($"    mad.lo.u32 %r6, %r4, {width}, %r3;");        // row*width + col
        ptx.AppendLine($"    mad.lo.u32 %r7, %r5, {width}, %r3;");        // reversed*width + col
        ptx.AppendLine("    mul.wide.u32 %rd2, %r6, 4;");
        ptx.AppendLine("    mul.wide.u32 %rd3, %r7, 4;");
        ptx.AppendLine("    add.u64 %rd4, %rd0, %rd2;");
        ptx.AppendLine("    add.u64 %rd5, %rd0, %rd3;");
        ptx.AppendLine("    add.u64 %rd6, %rd1, %rd2;");
        ptx.AppendLine("    add.u64 %rd7, %rd1, %rd3;");
        ptx.AppendLine("    ld.global.f32 %f0, [%rd4];");
        ptx.AppendLine("    ld.global.f32 %f1, [%rd5];");
        ptx.AppendLine("    ld.global.f32 %f2, [%rd6];");
        ptx.AppendLine("    ld.global.f32 %f3, [%rd7];");
        ptx.AppendLine("    st.global.f32 [%rd4], %f1;");
        ptx.AppendLine("    st.global.f32 [%rd5], %f0;");
        ptx.AppendLine("    st.global.f32 [%rd6], %f3;");
        ptx.AppendLine("    st.global.f32 [%rd7], %f2;");
        ptx.AppendLine("$FCBR_END:");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture, int height, int width, int blockThreads)
    {
        var extent = new DirectPtxExtent(checked(height * width));
        return new DirectPtxKernelBlueprint(
            Operation: "fft-cols-bit-reverse-f32",
            Version: 1,
            Architecture: architecture,
            Variant: $"b{blockThreads}-h{height}-w{width}",
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
                ["formula"] = "per column col: swap [row*width+col] with [reversed(row)*width+col] when reversed > row",
                ["mode"] = "inference-forward-fft-cols-bit-reverse",
                ["arithmetic"] = "none - pure bit-exact data movement; row reversed via brev.b32 >> (32-log2height)",
                ["layout"] = "row-major height x width; column stride = width",
                ["bounds-check"] = "none - the launch covers exactly width*height cells",
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

    internal static bool IsSupportedShape(int height, int width) =>
        height >= 2 && width >= 2 && (height & (height - 1)) == 0 && (width & (width - 1)) == 0 &&
        (long)height * width >= 512 && (height * width) % DefaultBlockThreads == 0 &&
        (long)height * width <= (1L << 24);

    internal static bool IsPromotedShape(int height, int width) => false;

    private static void Validate(int height, int width)
    {
        if (!IsSupportedShape(height, width))
            throw new ArgumentOutOfRangeException(nameof(height),
                "The column FFT bit-reverse family supports power-of-two height,width>=2 with height*width>=512 " +
                "a multiple of 256, up to 2^24.");
    }

    private static void ValidateBlockThreads(int height, int width, int blockThreads)
    {
        if (blockThreads is not (128 or 256 or 512) || (height * width) % blockThreads != 0)
            throw new ArgumentOutOfRangeException(nameof(blockThreads),
                "Column FFT bit-reverse block threads must be 128, 256, or 512 and evenly tile height*width.");
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
