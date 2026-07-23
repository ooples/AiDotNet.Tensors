using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Magnitude/phase to Hermitian spectrum for issue #850, matching the NVRTC
/// <c>parity210_build_spectrum</c> kernel: reconstruct a full length-<c>nFft</c> complex spectrum for each
/// <c>(batch, frame)</c> from the packed positive-frequency magnitude/phase. One thread owns one
/// <c>(batch, frame)</c> pair; it zeroes the <c>nFft</c> spectrum bins, fills the first <c>numFreqs</c> bins
/// with <c>m*cos(p)</c>/<c>m*sin(p)</c>, then mirrors bins <c>1..numFreqs-2</c> by conjugate symmetry into
/// <c>nFft-k</c>. The polar reconstruction uses <c>cos.approx</c>/<c>sin.approx</c>, so the parity spec is
/// TOLERANCE-based, not bit-exact. <c>batch</c>, <c>numFreqs</c>, <c>numFrames</c>, and <c>nFft</c> are
/// baked; the launch covers exactly <c>batch * numFrames</c> threads with no bounds guard. Four pointers
/// reach the launch ABI.
///
/// The specialization stays disabled by default and fails closed until three clean promotion runs clear
/// the release gate.
/// </summary>
internal sealed class PtxBuildSpectrumF32Kernel : IDisposable
{
    internal const int DefaultBlockThreads = 256;
    internal const string EntryPoint = "aidotnet_build_spectrum_f32";

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int Batch { get; }
    internal int NumFreqs { get; }
    internal int NumFrames { get; }
    internal int NFft { get; }
    internal int BlockThreads { get; }
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxBuildSpectrumF32Kernel(
        DirectPtxRuntime runtime, int batch, int numFreqs, int numFrames, int nFft,
        int blockThreads = DefaultBlockThreads)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasValidatedComplexUnary(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The checked-in build-spectrum specialization is admitted only on SM86.");
        Validate(batch, numFreqs, numFrames, nFft);
        ValidateBlockThreads(batch, numFrames, blockThreads);
        Batch = batch;
        NumFreqs = numFreqs;
        NumFrames = numFrames;
        NFft = nFft;
        BlockThreads = blockThreads;
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, batch, numFreqs, numFrames, nFft, blockThreads);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor,
            batch, numFreqs, numFrames, nFft, blockThreads);
        _module = runtime.LoadModule(Ptx);
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo info);
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, BlockThreads);
        Blueprint.ResourceBudget.Validate(EntryPoint, info, BlockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Ptx, info, BlockThreads, activeBlocks, _module.JitInfoLog);
    }

    internal unsafe void Launch(
        DirectPtxTensorView mag, DirectPtxTensorView phase,
        DirectPtxTensorView specRe, DirectPtxTensorView specIm)
    {
        Require(mag, Blueprint.Tensors[0], nameof(mag));
        Require(phase, Blueprint.Tensors[1], nameof(phase));
        Require(specRe, Blueprint.Tensors[2], nameof(specRe));
        Require(specIm, Blueprint.Tensors[3], nameof(specIm));

        IntPtr magPointer = mag.Pointer, phasePointer = phase.Pointer;
        IntPtr specRePointer = specRe.Pointer, specImPointer = specIm.Pointer;
        void** arguments = stackalloc void*[4];
        arguments[0] = &magPointer;
        arguments[1] = &phasePointer;
        arguments[2] = &specRePointer;
        arguments[3] = &specImPointer;
        int total = Batch * NumFrames;
        _module.Launch(
            _function,
            (uint)((total + BlockThreads - 1) / BlockThreads), 1, 1,
            checked((uint)BlockThreads), 1, 1,
            0,
            arguments);
    }

    public void Dispose() => _module.Dispose();

    internal static string EmitPtx(
        int ccMajor, int ccMinor, int batch, int numFreqs, int numFrames, int nFft,
        int blockThreads = DefaultBlockThreads)
    {
        Validate(batch, numFreqs, numFrames, nFft);
        ValidateBlockThreads(batch, numFrames, blockThreads);
        int magStride = checked(numFreqs * numFrames);
        int mirrorEnd = numFreqs - 1;   // loop3 runs k in [1, numFreqs-1)
        int total = checked(batch * numFrames);

        var ptx = new StringBuilder(3_584);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine($"// exact-shape batch={batch} numFreqs={numFreqs} numFrames={numFrames} nFft={nFft} block={blockThreads} op=build-spectrum");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 mag_ptr,");
        ptx.AppendLine("    .param .u64 phase_ptr,");
        ptx.AppendLine("    .param .u64 specre_ptr,");
        ptx.AppendLine("    .param .u64 specim_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine($".maxntid {blockThreads}, 1, 1");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<2>;");
        ptx.AppendLine("    .reg .b32 %r<12>;");
        ptx.AppendLine("    .reg .b64 %rd<20>;");
        ptx.AppendLine("    .reg .f32 %f<6>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [mag_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [phase_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [specre_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd3, [specim_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {blockThreads}, %r0;");   // idx
        ptx.AppendLine($"    setp.ge.u32 %p0, %r2, {total};");
        ptx.AppendLine("    @%p0 bra $BS_RET;");
        ptx.AppendLine($"    rem.u32 %r3, %r2, {numFrames};");             // frame
        ptx.AppendLine($"    div.u32 %r4, %r2, {numFrames};");             // b
        ptx.AppendLine($"    mul.lo.u32 %r5, %r4, {magStride};");         // magOff = b*numFreqs*numFrames
        ptx.AppendLine($"    mul.wide.u32 %rd4, %r2, {nFft * 4};");       // specOff bytes = idx*nFft*4
        ptx.AppendLine("    add.u64 %rd5, %rd2, %rd4;");                   // specReBase
        ptx.AppendLine("    add.u64 %rd6, %rd3, %rd4;");                   // specImBase
        ptx.AppendLine("    mul.wide.u32 %rd7, %r5, 4;");                  // magOff bytes
        ptx.AppendLine("    add.u64 %rd8, %rd0, %rd7;");                   // magBase
        ptx.AppendLine("    add.u64 %rd9, %rd1, %rd7;");                   // phaseBase
        ptx.AppendLine("    mov.f32 %f5, 0f00000000;");                    // zero
        // Loop 1: zero the nFft spectrum bins.
        ptx.AppendLine("    mov.u32 %r6, 0;");
        ptx.AppendLine("$BS_ZERO:");
        ptx.AppendLine($"    setp.ge.u32 %p0, %r6, {nFft};");
        ptx.AppendLine("    @%p0 bra $BS_FILL;");
        ptx.AppendLine("    mul.wide.u32 %rd10, %r6, 4;");
        ptx.AppendLine("    add.u64 %rd11, %rd5, %rd10;");
        ptx.AppendLine("    add.u64 %rd12, %rd6, %rd10;");
        ptx.AppendLine("    st.global.f32 [%rd11], %f5;");
        ptx.AppendLine("    st.global.f32 [%rd12], %f5;");
        ptx.AppendLine("    add.u32 %r6, %r6, 1;");
        ptx.AppendLine("    bra $BS_ZERO;");
        // Loop 2: fill the first numFreqs bins from polar magnitude/phase.
        ptx.AppendLine("$BS_FILL:");
        ptx.AppendLine("    mov.u32 %r7, 0;");
        ptx.AppendLine("$BS_FILL_LOOP:");
        ptx.AppendLine($"    setp.ge.u32 %p0, %r7, {numFreqs};");
        ptx.AppendLine("    @%p0 bra $BS_MIRROR;");
        ptx.AppendLine($"    mad.lo.u32 %r8, %r7, {numFrames}, %r3;");    // k*numFrames + frame
        ptx.AppendLine("    mul.wide.u32 %rd10, %r8, 4;");
        ptx.AppendLine("    add.u64 %rd13, %rd8, %rd10;");
        ptx.AppendLine("    add.u64 %rd14, %rd9, %rd10;");
        ptx.AppendLine("    ld.global.nc.f32 %f0, [%rd13];");   // m
        ptx.AppendLine("    ld.global.nc.f32 %f1, [%rd14];");   // p
        ptx.AppendLine("    cos.approx.f32 %f2, %f1;");
        ptx.AppendLine("    sin.approx.f32 %f3, %f1;");
        ptx.AppendLine("    mul.rn.f32 %f2, %f0, %f2;");        // m*cos(p)
        ptx.AppendLine("    mul.rn.f32 %f3, %f0, %f3;");        // m*sin(p)
        ptx.AppendLine("    mul.wide.u32 %rd15, %r7, 4;");
        ptx.AppendLine("    add.u64 %rd16, %rd5, %rd15;");
        ptx.AppendLine("    add.u64 %rd17, %rd6, %rd15;");
        ptx.AppendLine("    st.global.f32 [%rd16], %f2;");
        ptx.AppendLine("    st.global.f32 [%rd17], %f3;");
        ptx.AppendLine("    add.u32 %r7, %r7, 1;");
        ptx.AppendLine("    bra $BS_FILL_LOOP;");
        // Loop 3: mirror bins 1..numFreqs-2 by conjugate symmetry into nFft-k.
        ptx.AppendLine("$BS_MIRROR:");
        ptx.AppendLine("    mov.u32 %r9, 1;");
        ptx.AppendLine("$BS_MIRROR_LOOP:");
        ptx.AppendLine($"    setp.ge.u32 %p0, %r9, {mirrorEnd};");
        ptx.AppendLine("    @%p0 bra $BS_RET;");
        ptx.AppendLine($"    sub.u32 %r10, {nFft}, %r9;");               // dst = nFft - k
        ptx.AppendLine("    mul.wide.u32 %rd10, %r9, 4;");
        ptx.AppendLine("    add.u64 %rd13, %rd5, %rd10;");   // &specRe[k]
        ptx.AppendLine("    add.u64 %rd14, %rd6, %rd10;");   // &specIm[k]
        ptx.AppendLine("    ld.global.f32 %f0, [%rd13];");
        ptx.AppendLine("    ld.global.f32 %f1, [%rd14];");
        ptx.AppendLine("    neg.f32 %f1, %f1;");             // conjugate
        ptx.AppendLine("    mul.wide.u32 %rd15, %r10, 4;");
        ptx.AppendLine("    add.u64 %rd16, %rd5, %rd15;");   // &specRe[dst]
        ptx.AppendLine("    add.u64 %rd17, %rd6, %rd15;");   // &specIm[dst]
        ptx.AppendLine("    st.global.f32 [%rd16], %f0;");
        ptx.AppendLine("    st.global.f32 [%rd17], %f1;");
        ptx.AppendLine("    add.u32 %r9, %r9, 1;");
        ptx.AppendLine("    bra $BS_MIRROR_LOOP;");
        ptx.AppendLine("$BS_RET:");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture, int batch, int numFreqs, int numFrames, int nFft, int blockThreads)
    {
        var magExtent = new DirectPtxExtent(checked(batch * numFreqs * numFrames));
        var specExtent = new DirectPtxExtent(checked(batch * numFrames * nFft));
        return new DirectPtxKernelBlueprint(
            Operation: "build-spectrum-f32",
            Version: 1,
            Architecture: architecture,
            Variant: $"b{blockThreads}-bt{batch}-nf{numFreqs}-fr{numFrames}-fft{nFft}",
            Tensors:
            [
                new("mag", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    magExtent, magExtent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("phase", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    magExtent, magExtent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("specRe", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    specExtent, specExtent, 16, DirectPtxTensorAccess.ReadWrite, DirectPtxExtentMode.Exact),
                new("specIm", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    specExtent, specExtent, 16, DirectPtxTensorAccess.ReadWrite, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 32,
                MaxStaticSharedBytes: 0,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 1536 / blockThreads),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["formula"] = "spec[k]=mag*exp(i*phase) for k<numFreqs, conjugate mirror into nFft-k, zeros elsewhere",
                ["mode"] = "inference-forward-build-spectrum",
                ["arithmetic"] = "cos.approx/sin.approx polar reconstruction; tolerance-based parity, not bit-exact",
                ["loops"] = "zero nFft bins, fill numFreqs bins, mirror 1..numFreqs-2 by conjugate symmetry",
                ["bounds-check"] = "none - the launch covers exactly batch*numFrames threads",
                ["global-intermediates"] = "in-place on specRe/specIm",
                ["temporary-device-allocation"] = "none",
                ["byte-offset"] = "zero-entire-allocation-view",
                ["padding"] = "none-logical-equals-physical"
            });
    }

    internal static bool IsSupportedShape(int batch, int numFreqs, int numFrames, int nFft) =>
        batch >= 1 && numFrames >= 1 && nFft >= 4 && numFreqs >= 2 && numFreqs <= nFft / 2 + 1 &&
        (long)batch * numFrames * nFft <= (1L << 26);

    internal static bool IsPromotedShape(int batch, int numFreqs, int numFrames, int nFft) => false;

    private static void Validate(int batch, int numFreqs, int numFrames, int nFft)
    {
        if (!IsSupportedShape(batch, numFreqs, numFrames, nFft))
            throw new ArgumentOutOfRangeException(nameof(nFft),
                "The build-spectrum family requires batch>=1, numFrames>=1, nFft>=4, 2<=numFreqs<=nFft/2+1, " +
                "and batch*numFrames*nFft<=2^26.");
    }

    private static void ValidateBlockThreads(int batch, int numFrames, int blockThreads)
    {
        if (blockThreads is not (128 or 256 or 512))
            throw new ArgumentOutOfRangeException(nameof(blockThreads),
                "Build-spectrum block threads must be 128, 256, or 512.");
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
