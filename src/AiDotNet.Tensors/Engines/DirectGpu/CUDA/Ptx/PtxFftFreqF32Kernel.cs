using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>Which frequency layout <see cref="PtxFftFreqF32Kernel"/> generates.</summary>
internal enum DirectPtxFftFreqOp
{
    /// <summary><c>fftfreq</c>: length <c>n</c>, layout [0,1,…,split-1, split-n,…,-1] * scale.</summary>
    Full,

    /// <summary><c>rfftfreq</c>: length <c>n/2 + 1</c>, layout [0,1,…,n/2] * scale.</summary>
    Real
}

/// <summary>
/// FFT sample-frequency generation for issue #850, backing the <c>Fft.FftFreq</c> / <c>Fft.RFftFreq</c>
/// variants. For <see cref="DirectPtxFftFreqOp.Full"/> it writes the <c>n</c> discrete Fourier transform
/// sample frequencies <c>[0,1,…,(n+1)/2-1, (n+1)/2-n,…,-1] * scale</c>; for
/// <see cref="DirectPtxFftFreqOp.Real"/> it writes the <c>n/2 + 1</c> non-negative frequencies
/// <c>[0,1,…,n/2] * scale</c>. <c>scale = 1/(d*n)</c> is a per-launch <c>.param .f32</c>. Each output is one
/// integer index cast to fp32 and scaled, so the specialization is bit-exact against the reference sample
/// frequencies (the scale itself is computed once on the host). <c>n</c> and the op are baked; the launch
/// rounds up and a single guard drops the tail lanes. One pointer plus one f32 scalar reach the launch ABI.
///
/// The specialization stays disabled by default and fails closed until three clean promotion runs clear
/// the release gate.
/// </summary>
internal sealed class PtxFftFreqF32Kernel : IDisposable
{
    internal const int DefaultBlockThreads = 256;
    internal const string EntryPoint = "aidotnet_fft_freq_f32";

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int N { get; }
    internal DirectPtxFftFreqOp Op { get; }
    internal int OutLen { get; }
    internal int BlockThreads { get; }
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxFftFreqF32Kernel(
        DirectPtxRuntime runtime, int n, DirectPtxFftFreqOp op, int blockThreads = DefaultBlockThreads)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasValidatedComplexUnary(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The checked-in fft-freq specialization is admitted only on SM86.");
        Validate(n);
        ValidateBlockThreads(blockThreads);
        N = n;
        Op = op;
        OutLen = op == DirectPtxFftFreqOp.Full ? n : n / 2 + 1;
        BlockThreads = blockThreads;
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, n, op, blockThreads);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor, n, op, blockThreads);
        _module = runtime.LoadModule(Ptx);
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo info);
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, BlockThreads);
        Blueprint.ResourceBudget.Validate(EntryPoint, info, BlockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Ptx, info, BlockThreads, activeBlocks, _module.JitInfoLog);
    }

    internal unsafe void Launch(DirectPtxTensorView output, float scale)
    {
        Require(output, Blueprint.Tensors[0], nameof(output));

        IntPtr outputPointer = output.Pointer;
        float scaleArg = scale;
        void** arguments = stackalloc void*[2];
        arguments[0] = &outputPointer;
        arguments[1] = &scaleArg;
        _module.Launch(
            _function,
            (uint)((OutLen + BlockThreads - 1) / BlockThreads), 1, 1,
            checked((uint)BlockThreads), 1, 1,
            0,
            arguments);
    }

    public void Dispose() => _module.Dispose();

    internal static string EmitPtx(int ccMajor, int ccMinor, int n, DirectPtxFftFreqOp op, int blockThreads = DefaultBlockThreads)
    {
        Validate(n);
        ValidateBlockThreads(blockThreads);
        bool full = op == DirectPtxFftFreqOp.Full;
        int outLen = full ? n : n / 2 + 1;
        int split = (n + 1) / 2;

        var ptx = new StringBuilder(2_048);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine($"// exact-shape n={n} outlen={outLen} block={blockThreads} op=fft-freq-{(full ? "full" : "real")}");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 output_ptr,");
        ptx.AppendLine("    .param .f32 scale_val");
        ptx.AppendLine(")");
        ptx.AppendLine($".maxntid {blockThreads}, 1, 1");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<3>;");
        ptx.AppendLine("    .reg .b32 %r<5>;");
        ptx.AppendLine("    .reg .b64 %rd<6>;");
        ptx.AppendLine("    .reg .f32 %f<3>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [output_ptr];");
        ptx.AppendLine("    ld.param.f32 %f0, [scale_val];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {blockThreads}, %r0;");   // i
        ptx.AppendLine($"    setp.ge.u32 %p0, %r2, {outLen};");
        ptx.AppendLine("    @%p0 bra $FREQ_RET;");
        if (full)
        {
            // valInt = (i < split) ? i : i - n
            ptx.AppendLine($"    sub.s32 %r3, %r2, {n};");                 // i - n
            ptx.AppendLine($"    setp.lt.u32 %p1, %r2, {split};");
            ptx.AppendLine("    selp.b32 %r4, %r2, %r3, %p1;");
            ptx.AppendLine("    cvt.rn.f32.s32 %f1, %r4;");
        }
        else
        {
            ptx.AppendLine("    cvt.rn.f32.u32 %f1, %r2;");                // i
        }
        ptx.AppendLine("    mul.rn.f32 %f1, %f1, %f0;");                   // * scale
        ptx.AppendLine("    mul.wide.u32 %rd2, %r2, 4;");
        ptx.AppendLine("    add.u64 %rd3, %rd0, %rd2;");
        ptx.AppendLine("    st.global.f32 [%rd3], %f1;");
        ptx.AppendLine("$FREQ_RET:");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture, int n, DirectPtxFftFreqOp op, int blockThreads)
    {
        int outLen = op == DirectPtxFftFreqOp.Full ? n : n / 2 + 1;
        var extent = new DirectPtxExtent(outLen);
        return new DirectPtxKernelBlueprint(
            Operation: "fft-freq-f32",
            Version: 1,
            Architecture: architecture,
            Variant: $"b{blockThreads}-n{n}-{(op == DirectPtxFftFreqOp.Full ? "full" : "real")}",
            Tensors:
            [
                new("output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    extent, extent, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 16,
                MaxStaticSharedBytes: 0,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 1536 / blockThreads),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["formula"] = op == DirectPtxFftFreqOp.Full
                    ? "output[i] = ((i < (n+1)/2) ? i : i - n) * scale"
                    : "output[i] = i * scale for i in [0, n/2]",
                ["mode"] = "inference-forward-fft-freq",
                ["arithmetic"] = "integer index cast to fp32 and scaled; bit-exact against the reference bins",
                ["scalar"] = "scale = 1/(d*n) is a per-launch .param .f32 (computed once on the host)",
                ["bounds-check"] = "single guard drops lanes past the output length",
                ["global-intermediates"] = "none",
                ["temporary-device-allocation"] = "none",
                ["byte-offset"] = "zero-entire-allocation-view",
                ["padding"] = "none-logical-equals-physical"
            });
    }

    internal static bool IsSupportedShape(int n) => n >= 2 && n <= (1 << 26);

    internal static bool IsPromotedShape(int n) => false;

    private static void Validate(int n)
    {
        if (!IsSupportedShape(n))
            throw new ArgumentOutOfRangeException(nameof(n),
                "The fft-freq family supports lengths n in [2, 2^26].");
    }

    private static void ValidateBlockThreads(int blockThreads)
    {
        if (blockThreads is not (128 or 256 or 512))
            throw new ArgumentOutOfRangeException(nameof(blockThreads),
                "Fft-freq block threads must be 128, 256, or 512.");
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
