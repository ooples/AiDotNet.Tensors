using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// One column-wise radix-2 DIT FFT butterfly stage for issue #850, matching the NVRTC
/// <c>fft_cols_butterfly</c> kernel: transform each column of a row-major <c>height x width</c> split
/// matrix in place along the column (stride <c>width</c>). The 2D FFT's row pass is the contiguous batched
/// FFT; this kernel supplies the strided column pass. Each thread owns one <c>(column, wing)</c> cell,
/// flattened as <c>gid</c> and decomposed to <c>col = gid % width</c> and a per-column wing index
/// <c>gid / width</c>; the wing selects a top/bottom row pair spanning <c>halfStride</c> rows, addressed
/// with the column stride <c>width</c>. <c>width</c> and <c>height</c> are baked; the per-stage
/// <c>stride</c> and <c>inverse</c> flag are per-launch <c>.param .u32</c> controls. The twiddle uses
/// <c>cos.approx</c>/<c>sin.approx</c> (tolerance-based parity). The launch covers exactly
/// <c>width * (height/2)</c> wings, so no bounds guard is emitted. Two pointers plus two u32 controls reach
/// the launch ABI.
///
/// The specialization stays disabled by default and fails closed until three clean promotion runs clear
/// the release gate.
/// </summary>
internal sealed class PtxFftColsButterflyF32Kernel : IDisposable
{
    internal const int DefaultBlockThreads = 256;
    internal const string EntryPoint = "aidotnet_fft_cols_butterfly_f32";

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int Height { get; }
    internal int Width { get; }
    internal int BlockThreads { get; }
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxFftColsButterflyF32Kernel(
        DirectPtxRuntime runtime, int height, int width, int blockThreads = DefaultBlockThreads)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasValidatedComplexMultiply(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The checked-in column FFT butterfly specialization is admitted only on SM86.");
        Validate(height, width);
        ValidateBlockThreads(height, width, blockThreads);
        Height = height;
        Width = width;
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

    /// <summary>Launches one column stage. <paramref name="stride"/> is the current radix-2 row span
    /// (2,4,...,height); <paramref name="inverse"/> is 1 for the inverse transform, 0 for forward.</summary>
    internal unsafe void Launch(DirectPtxTensorView real, DirectPtxTensorView imag, int stride, int inverse)
    {
        Require(real, Blueprint.Tensors[0], nameof(real));
        Require(imag, Blueprint.Tensors[1], nameof(imag));

        IntPtr realPointer = real.Pointer, imagPointer = imag.Pointer;
        int strideValue = stride, inverseValue = inverse;
        void** arguments = stackalloc void*[4];
        arguments[0] = &realPointer;
        arguments[1] = &imagPointer;
        arguments[2] = &strideValue;
        arguments[3] = &inverseValue;
        _module.Launch(
            _function,
            checked((uint)((Width * (Height / 2)) / BlockThreads)), 1, 1,
            checked((uint)BlockThreads), 1, 1,
            0,
            arguments);
    }

    public void Dispose() => _module.Dispose();

    private static string Hex(float value) => "0f" + BitConverter.ToInt32(BitConverter.GetBytes(value), 0).ToString("X8");

    internal static string EmitPtx(int ccMajor, int ccMinor, int height, int width, int blockThreads = DefaultBlockThreads)
    {
        Validate(height, width);
        ValidateBlockThreads(height, width, blockThreads);
        string twoPi = Hex((float)(2.0 * Math.PI)), negTwoPi = Hex((float)(-2.0 * Math.PI));

        var ptx = new StringBuilder(4_608);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine($"// exact-shape height={height} width={width} block={blockThreads} op=fft-cols-butterfly (stride,inverse are .param)");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 real_ptr,");
        ptx.AppendLine("    .param .u64 imag_ptr,");
        ptx.AppendLine("    .param .u32 stride_val,");
        ptx.AppendLine("    .param .u32 inverse_val");
        ptx.AppendLine(")");
        ptx.AppendLine($".maxntid {blockThreads}, 1, 1");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<2>;");
        ptx.AppendLine("    .reg .b32 %r<20>;");
        ptx.AppendLine("    .reg .b64 %rd<16>;");
        ptx.AppendLine("    .reg .f32 %f<20>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [real_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [imag_ptr];");
        ptx.AppendLine("    ld.param.u32 %r8, [stride_val];");
        ptx.AppendLine("    ld.param.u32 %r9, [inverse_val];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {blockThreads}, %r0;");   // gid
        ptx.AppendLine($"    rem.u32 %r3, %r2, {width};");                 // col
        ptx.AppendLine($"    div.u32 %r4, %r2, {width};");                 // wingLinear (per-column)
        ptx.AppendLine("    shr.u32 %r5, %r8, 1;");                        // halfStride
        ptx.AppendLine("    div.u32 %r6, %r4, %r5;");                      // butterflyId
        ptx.AppendLine("    rem.u32 %r7, %r4, %r5;");                      // wingId
        ptx.AppendLine("    mad.lo.u32 %r10, %r6, %r8, %r7;");            // topRow = butterflyId*stride + wingId
        ptx.AppendLine($"    mad.lo.u32 %r11, %r10, {width}, %r3;");      // topIdx = topRow*width + col
        ptx.AppendLine($"    mad.lo.u32 %r13, %r5, {width}, %r11;");      // botIdx = topIdx + halfStride*width
        // twiddle: angle = (inverse ? +2pi : -2pi) * wingId / stride
        ptx.AppendLine("    cvt.rn.f32.u32 %f0, %r7;");
        ptx.AppendLine("    cvt.rn.f32.u32 %f1, %r8;");
        ptx.AppendLine("    setp.ne.u32 %p0, %r9, 0;");
        ptx.AppendLine($"    selp.f32 %f2, {twoPi}, {negTwoPi}, %p0;");
        ptx.AppendLine("    mul.rn.f32 %f2, %f2, %f0;");
        ptx.AppendLine("    div.rn.f32 %f2, %f2, %f1;");
        ptx.AppendLine("    cos.approx.f32 %f3, %f2;");
        ptx.AppendLine("    sin.approx.f32 %f4, %f2;");
        ptx.AppendLine("    mul.wide.u32 %rd2, %r11, 4;");
        ptx.AppendLine("    mul.wide.u32 %rd3, %r13, 4;");
        ptx.AppendLine("    add.u64 %rd4, %rd0, %rd2;");   // &real[topIdx]
        ptx.AppendLine("    add.u64 %rd5, %rd1, %rd2;");   // &imag[topIdx]
        ptx.AppendLine("    add.u64 %rd6, %rd0, %rd3;");   // &real[botIdx]
        ptx.AppendLine("    add.u64 %rd7, %rd1, %rd3;");   // &imag[botIdx]
        ptx.AppendLine("    ld.global.f32 %f5, [%rd4];");
        ptx.AppendLine("    ld.global.f32 %f6, [%rd5];");
        ptx.AppendLine("    ld.global.f32 %f7, [%rd6];");
        ptx.AppendLine("    ld.global.f32 %f8, [%rd7];");
        ptx.AppendLine("    mul.rn.f32 %f9, %f8, %f4;");
        ptx.AppendLine("    neg.f32 %f9, %f9;");
        ptx.AppendLine("    fma.rn.f32 %f9, %f7, %f3, %f9;");   // twiddledReal
        ptx.AppendLine("    mul.rn.f32 %f10, %f8, %f3;");
        ptx.AppendLine("    fma.rn.f32 %f10, %f7, %f4, %f10;"); // twiddledImag
        ptx.AppendLine("    add.rn.f32 %f11, %f5, %f9;");
        ptx.AppendLine("    add.rn.f32 %f12, %f6, %f10;");
        ptx.AppendLine("    sub.rn.f32 %f13, %f5, %f9;");
        ptx.AppendLine("    sub.rn.f32 %f14, %f6, %f10;");
        ptx.AppendLine("    st.global.f32 [%rd4], %f11;");
        ptx.AppendLine("    st.global.f32 [%rd5], %f12;");
        ptx.AppendLine("    st.global.f32 [%rd6], %f13;");
        ptx.AppendLine("    st.global.f32 [%rd7], %f14;");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture, int height, int width, int blockThreads)
    {
        var extent = new DirectPtxExtent(checked(height * width));
        return new DirectPtxKernelBlueprint(
            Operation: "fft-cols-butterfly-f32",
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
                MaxRegistersPerThread: 32,
                MaxStaticSharedBytes: 0,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 1536 / blockThreads),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["formula"] = "per column col: top'=top+W*bot, bot'=top-W*bot at column stride width, W=exp(+/-2pi i wing/stride)",
                ["mode"] = "inference-forward-fft-cols-butterfly-stage",
                ["approximation"] = "cos.approx/sin.approx twiddle; tolerance-based parity, not bit-exact",
                ["control-params"] = "stride and inverse are per-launch .param .u32 (a 2D FFT launches log2(height) column stages)",
                ["layout"] = "row-major height x width; column stride = width; bottom row = top row + halfStride",
                ["bounds-check"] = "none - the launch covers exactly width*(height/2) wings",
                ["global-intermediates"] = "in-place on real/imag",
                ["temporary-device-allocation"] = "none",
                ["byte-offset"] = "zero-entire-allocation-view",
                ["padding"] = "none-logical-equals-physical"
            });
    }

    internal static bool IsSupportedShape(int height, int width) =>
        height >= 2 && width >= 2 && (height & (height - 1)) == 0 && (width & (width - 1)) == 0 &&
        (long)width * (height / 2) >= 256 && (width * (height / 2)) % DefaultBlockThreads == 0 &&
        (long)height * width <= (1L << 24);

    internal static bool IsPromotedShape(int height, int width) => false;

    private static void Validate(int height, int width)
    {
        if (!IsSupportedShape(height, width))
            throw new ArgumentOutOfRangeException(nameof(height),
                "The column FFT butterfly family supports power-of-two height,width>=2 with width*(height/2)>=256 " +
                "a multiple of 256, up to height*width=2^24.");
    }

    private static void ValidateBlockThreads(int height, int width, int blockThreads)
    {
        if (blockThreads is not (128 or 256 or 512) || (width * (height / 2)) % blockThreads != 0)
            throw new ArgumentOutOfRangeException(nameof(blockThreads),
                "Column FFT butterfly block threads must be 128, 256, or 512 and evenly tile width*(height/2).");
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
