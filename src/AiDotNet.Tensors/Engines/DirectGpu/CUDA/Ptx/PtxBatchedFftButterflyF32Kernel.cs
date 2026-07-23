using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// One batched radix-2 DIT FFT butterfly stage for issue #850, matching the NVRTC
/// <c>batched_fft_butterfly</c> kernel. It transforms each of <c>batch</c> contiguous length-<c>n</c> rows
/// independently: the batch index comes from <c>gridDim.y</c> (<c>%ctaid.y</c>) and offsets into the row
/// at <c>b*n</c>. The transform length <c>n</c> is baked; the per-stage <c>stride</c> and <c>inverse</c>
/// flag are per-launch <c>.param .u32</c> controls (a batched FFT launches this kernel log2(n) times with
/// doubling strides). One thread owns one butterfly wing, computes the twiddle with
/// <c>cos.approx</c>/<c>sin.approx</c> (so this stage is covered by a tolerance-based parity spec), and
/// writes both outputs in place. The launch covers exactly <c>batch * (n/2)</c> wings, so no bounds guard
/// is emitted. Two pointers plus two u32 controls reach the launch ABI.
///
/// The specialization stays disabled by default and fails closed until three clean promotion runs clear
/// the release gate.
/// </summary>
internal sealed class PtxBatchedFftButterflyF32Kernel : IDisposable
{
    internal const int DefaultBlockThreads = 256;
    internal const string EntryPoint = "aidotnet_batched_fft_butterfly_f32";

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int N { get; }
    internal int Batch { get; }
    internal int BlockThreads { get; }
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxBatchedFftButterflyF32Kernel(
        DirectPtxRuntime runtime, int n, int batch, int blockThreads = DefaultBlockThreads)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasValidatedComplexMultiply(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The checked-in batched FFT butterfly specialization is admitted only on SM86.");
        Validate(n, batch);
        ValidateBlockThreads(n, blockThreads);
        N = n;
        Batch = batch;
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

    /// <summary>Launches one stage over all rows. <paramref name="stride"/> is the current radix-2 span
    /// (2,4,...,n); <paramref name="inverse"/> is 1 for the inverse transform, 0 for forward.</summary>
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
            checked((uint)((N / 2) / BlockThreads)), checked((uint)Batch), 1,
            checked((uint)BlockThreads), 1, 1,
            0,
            arguments);
    }

    public void Dispose() => _module.Dispose();

    private static string Hex(float value) => "0f" + BitConverter.ToInt32(BitConverter.GetBytes(value), 0).ToString("X8");

    internal static string EmitPtx(int ccMajor, int ccMinor, int n, int blockThreads = DefaultBlockThreads)
    {
        ValidateN(n);
        ValidateBlockThreads(n, blockThreads);
        string twoPi = Hex((float)(2.0 * Math.PI)), negTwoPi = Hex((float)(-2.0 * Math.PI));

        var ptx = new StringBuilder(4_352);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine($"// exact-shape n={n} block={blockThreads} op=batched-fft-butterfly (batch=gridDimY; stride,inverse are .param)");
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
        ptx.AppendLine("    .reg .b32 %r<18>;");
        ptx.AppendLine("    .reg .b64 %rd<16>;");
        ptx.AppendLine("    .reg .f32 %f<20>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [real_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [imag_ptr];");
        ptx.AppendLine("    ld.param.u32 %r8, [stride_val];");
        ptx.AppendLine("    ld.param.u32 %r9, [inverse_val];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {blockThreads}, %r0;");   // idx within row's wings
        ptx.AppendLine("    shr.u32 %r3, %r8, 1;");                        // halfStride = stride/2
        ptx.AppendLine("    div.u32 %r4, %r2, %r3;");                      // butterflyId
        ptx.AppendLine("    rem.u32 %r5, %r2, %r3;");                      // wingId
        ptx.AppendLine("    mad.lo.u32 %r6, %r4, %r8, %r5;");             // rowTopIdx
        ptx.AppendLine("    add.u32 %r7, %r6, %r3;");                     // rowBotIdx
        ptx.AppendLine("    mov.u32 %r10, %ctaid.y;");                     // batch index
        ptx.AppendLine($"    mul.lo.u32 %r11, %r10, {n};");              // baseOffset = b*n
        ptx.AppendLine("    add.u32 %r6, %r6, %r11;");                    // topIdx = baseOffset + rowTopIdx
        ptx.AppendLine("    add.u32 %r7, %r7, %r11;");                    // botIdx = baseOffset + rowBotIdx
        // twiddle: angle = (inverse ? +2pi : -2pi) * wingId / stride
        ptx.AppendLine("    cvt.rn.f32.u32 %f0, %r5;");
        ptx.AppendLine("    cvt.rn.f32.u32 %f1, %r8;");
        ptx.AppendLine("    setp.ne.u32 %p0, %r9, 0;");
        ptx.AppendLine($"    selp.f32 %f2, {twoPi}, {negTwoPi}, %p0;");
        ptx.AppendLine("    mul.rn.f32 %f2, %f2, %f0;");
        ptx.AppendLine("    div.rn.f32 %f2, %f2, %f1;");
        ptx.AppendLine("    cos.approx.f32 %f3, %f2;");
        ptx.AppendLine("    sin.approx.f32 %f4, %f2;");
        ptx.AppendLine("    mul.wide.u32 %rd2, %r6, 4;");
        ptx.AppendLine("    mul.wide.u32 %rd3, %r7, 4;");
        ptx.AppendLine("    add.u64 %rd4, %rd0, %rd2;");   // &real[topIdx]
        ptx.AppendLine("    add.u64 %rd5, %rd1, %rd2;");   // &imag[topIdx]
        ptx.AppendLine("    add.u64 %rd6, %rd0, %rd3;");   // &real[botIdx]
        ptx.AppendLine("    add.u64 %rd7, %rd1, %rd3;");   // &imag[botIdx]
        ptx.AppendLine("    ld.global.f32 %f5, [%rd4];");  // topReal
        ptx.AppendLine("    ld.global.f32 %f6, [%rd5];");  // topImag
        ptx.AppendLine("    ld.global.f32 %f7, [%rd6];");  // botReal
        ptx.AppendLine("    ld.global.f32 %f8, [%rd7];");  // botImag
        ptx.AppendLine("    mul.rn.f32 %f9, %f8, %f4;");        // botImag*twiddleImag
        ptx.AppendLine("    neg.f32 %f9, %f9;");
        ptx.AppendLine("    fma.rn.f32 %f9, %f7, %f3, %f9;");   // twiddledReal
        ptx.AppendLine("    mul.rn.f32 %f10, %f8, %f3;");       // botImag*twiddleReal
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
        DirectPtxArchitectureFamily architecture, int n, int batch, int blockThreads)
    {
        var extent = new DirectPtxExtent(checked(batch * n));
        return new DirectPtxKernelBlueprint(
            Operation: "batched-fft-butterfly-f32",
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
                MaxRegistersPerThread: 32,
                MaxStaticSharedBytes: 0,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 1536 / blockThreads),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["formula"] = "per row b: top'=top+W*bot, bot'=top-W*bot at b*n offset, W=exp(+/-2pi i wing/stride)",
                ["mode"] = "inference-forward-batched-fft-butterfly-stage",
                ["approximation"] = "cos.approx/sin.approx twiddle; tolerance-based parity, not bit-exact",
                ["control-params"] = "stride and inverse are per-launch .param .u32 (an FFT launches log2(n) stages)",
                ["batch-dimension"] = "gridDim.y; baseOffset = ctaid.y * n",
                ["bounds-check"] = "none - the launch covers exactly batch*(n/2) wings",
                ["global-intermediates"] = "in-place on real/imag",
                ["temporary-device-allocation"] = "none",
                ["byte-offset"] = "zero-entire-allocation-view",
                ["padding"] = "none-logical-equals-physical"
            });
    }

    internal static bool IsSupportedShape(int n, int batch) =>
        batch >= 1 && n >= 512 && (n & (n - 1)) == 0 && (n / 2) % DefaultBlockThreads == 0 && n <= (1 << 24) &&
        (long)batch * n <= (1L << 27);

    internal static bool IsPromotedShape(int n, int batch) => false;

    private static void Validate(int n, int batch)
    {
        if (!IsSupportedShape(n, batch))
            throw new ArgumentOutOfRangeException(nameof(n),
                "The batched FFT butterfly family supports power-of-two n>=512 with (n/2) a multiple of 256 " +
                "(up to 2^24), batch>=1, and batch*n<=2^27.");
    }

    private static void ValidateN(int n)
    {
        if (n < 512 || (n & (n - 1)) != 0 || (n / 2) % DefaultBlockThreads != 0 || n > (1 << 24))
            throw new ArgumentOutOfRangeException(nameof(n),
                "The batched FFT butterfly family supports power-of-two n>=512 with (n/2) a multiple of 256, up to 2^24.");
    }

    private static void ValidateBlockThreads(int n, int blockThreads)
    {
        if (blockThreads is not (128 or 256 or 512) || (n / 2) % blockThreads != 0)
            throw new ArgumentOutOfRangeException(nameof(blockThreads),
                "Batched FFT butterfly block threads must be 128, 256, or 512 and evenly tile n/2.");
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
