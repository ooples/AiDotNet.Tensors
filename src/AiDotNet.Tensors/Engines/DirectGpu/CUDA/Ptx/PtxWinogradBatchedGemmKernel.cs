using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Batched register-blocked GEMM for the 16 Winograd F(2,3) positions:
/// M[b] = U[b] . V[b] for b = 0..15, where U[16,K,C], V[16,C,P], M[16,K,P]
/// (position-major, contiguous [K,C]/[C,P]/[K,P] per position). Each thread block
/// computes one position's BM x BN output tile with a TM x TN register micro-tile
/// and shared-staged A/B — the exact structure of the 1x1 register-blocked kernel
/// that beats cuDNN, batched over the 16 positions via grid.z. This isolates the
/// register-blocked GEMM (no 16-accumulator explosion) from the output transform,
/// which runs as a separate stage.
/// </summary>
internal sealed class PtxWinogradBatchedGemmKernel : IDisposable
{
    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int OutputChannels { get; }   // M dimension (K)
    internal int InputChannels { get; }     // contraction (C)
    internal int Tiles { get; }             // N dimension (P)
    internal int BlockM { get; }
    internal int BlockN { get; }
    internal int BlockK { get; }
    internal int ThreadM { get; }
    internal int ThreadN { get; }

    internal string Ptx { get; }
    internal DirectPtxFunctionInfo FunctionInfo { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal int BlockThreads => (BlockM / ThreadM) * (BlockN / ThreadN);
    internal long UBytes => (long)16 * OutputChannels * InputChannels * sizeof(float);
    internal long VBytes => (long)16 * InputChannels * Tiles * sizeof(float);
    internal long MBytes => (long)16 * OutputChannels * Tiles * sizeof(float);

    internal string EntryPoint => FormattableString.Invariant(
        $"aidotnet_winograd_batched_gemm_k{OutputChannels}_c{InputChannels}_p{Tiles}_{BlockM}x{BlockN}x{BlockK}_{ThreadM}x{ThreadN}");

    internal PtxWinogradBatchedGemmKernel(
        DirectPtxRuntime runtime, int outputChannels, int inputChannels, int tiles,
        int blockM = 64, int blockN = 64, int blockK = 8, int threadM = 4, int threadN = 4)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new NotSupportedException("Batched Winograd GEMM has no experimental non-SM86 specialization.");
        OutputChannels = outputChannels; InputChannels = inputChannels; Tiles = tiles;
        BlockM = blockM; BlockN = blockN; BlockK = blockK; ThreadM = threadM; ThreadN = threadN;
        if (outputChannels % blockM != 0 || tiles % blockN != 0 || inputChannels % blockK != 0)
            throw new ArgumentException("BM|K, BN|P, BK|C required.");
        if (blockM % threadM != 0 || blockN % threadN != 0)
            throw new ArgumentException("BM%TM==0, BN%TN==0 required.");
        if (!IsPow2(blockK) || !IsPow2(blockN)) throw new ArgumentException("BK, BN must be powers of two.");
        int threads = BlockThreads;
        if (threads > 1024 || blockM * blockK % threads != 0 || blockK * blockN % threads != 0)
            throw new ArgumentException("Invalid tiling for cooperative load.");

        Blueprint = CreateBlueprint(runtime.ArchitectureFamily);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor);
        _module = runtime.LoadModule(
            Ptx, allowExperimentalJitFallback: DirectPtxFeatureGate.ConvolutionExperimentOverride);
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo functionInfo);
        FunctionInfo = functionInfo;
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, BlockThreads);
        Blueprint.ResourceBudget.Validate(EntryPoint, functionInfo, BlockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Ptx, functionInfo, BlockThreads, activeBlocks, _module);
    }

    private static bool IsPow2(int v) => v > 0 && (v & (v - 1)) == 0;

    internal DirectPtxKernelBlueprint CreateBlueprint(DirectPtxArchitectureFamily architecture)
    {
        var u = new DirectPtxExtent(16, OutputChannels, InputChannels);
        var v = new DirectPtxExtent(16, InputChannels, Tiles);
        var m = new DirectPtxExtent(16, OutputChannels, Tiles);
        int sharedBytes = (BlockM * BlockK + BlockK * BlockN) * sizeof(float);
        return new DirectPtxKernelBlueprint(
            Operation: "winograd-batched-gemm",
            Version: 1,
            Architecture: architecture,
            Variant: FormattableString.Invariant(
                $"k{OutputChannels}-c{InputChannels}-p{Tiles}-bm{BlockM}-bn{BlockN}-bk{BlockK}-tm{ThreadM}-tn{ThreadN}-fp32"),
            Tensors:
            [
                new("U", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    u, u, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("V", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    v, v, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("M", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    m, m, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: ThreadM * ThreadN + ThreadM + ThreadN + 40,
                MaxStaticSharedBytes: sharedBytes,
                MaxLocalBytesPerThread: 0, MinBlocksPerMultiprocessor: 1),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["equation"] = "M[b] = U[b] . V[b]  (b = 0..15 Winograd positions)",
                ["algorithm"] = "register-blocked-batched-gemm",
                ["input"] = "fp32", ["accumulator"] = "fp32-fma", ["output"] = "fp32",
                ["shape-selection"] = "host-only-exact-contract",
                ["promotion"] = "experimental-pending-gpu-evidence"
            });
    }

    internal unsafe void Launch(DirectPtxTensorView u, DirectPtxTensorView v, DirectPtxTensorView m)
    {
        Require(u, Blueprint.Tensors[0], nameof(u));
        Require(v, Blueprint.Tensors[1], nameof(v));
        Require(m, Blueprint.Tensors[2], nameof(m));
        IntPtr uPtr = u.Pointer, vPtr = v.Pointer, mPtr = m.Pointer;
        void** arguments = stackalloc void*[3];
        arguments[0] = &uPtr; arguments[1] = &vPtr; arguments[2] = &mPtr;
        uint gridX = (uint)(Tiles / BlockN);
        uint gridY = (uint)(OutputChannels / BlockM);
        _module.Launch(_function, gridX, gridY, 16,
            (uint)(BlockN / ThreadN), (uint)(BlockM / ThreadM), 1, 0, arguments);
    }

    private static void Require(DirectPtxTensorView view, DirectPtxTensorContract contract, string parameter)
    {
        if (view.Pointer == IntPtr.Zero || view.PhysicalType != contract.PhysicalType ||
            view.Layout != contract.Layout || view.LogicalExtent != contract.LogicalExtent ||
            view.PhysicalExtent != contract.PhysicalExtent ||
            view.ByteLength != contract.RequiredBytes ||
            view.AllocationByteLength != contract.RequiredBytes)
            throw new ArgumentException($"{parameter} does not satisfy exact physical ABI '{contract.Name}'.", parameter);
    }

    internal string EmitPtx(int major, int minor)
    {
        if (!DirectPtxArchitecture.HasExperimentalConvolution(major, minor))
            throw new NotSupportedException("Only the experimental SM86 batched-GEMM emitter exists.");
        string I(int v) => v.ToString(CultureInfo.InvariantCulture);
        int k = OutputChannels, c = InputChannels, pp = Tiles;
        int bm = BlockM, bn = BlockN, bk = BlockK, tm = ThreadM, tn = ThreadN;
        int threads = BlockThreads;
        int loadsA = bm * bk / threads, loadsB = bk * bn / threads;
        int contractionTiles = c / bk, log2Bk = Log2(bk), log2Bn = Log2(bn);
        int aBytes = bm * bk * 4, bBytes = bk * bn * 4;
        int kc = k * c, cp = c * pp, kp = k * pp;
        string entry = EntryPoint;

        var s = new StringBuilder(32768);
        s.AppendLine(".version 7.1");
        s.AppendLine($".target sm_{major}{minor}");
        s.AppendLine(".address_size 64");
        s.AppendLine();
        s.AppendLine($".visible .entry {entry}(");
        s.AppendLine("    .param .u64 u_ptr,");
        s.AppendLine("    .param .u64 v_ptr,");
        s.AppendLine("    .param .u64 m_ptr");
        s.AppendLine(")");
        s.AppendLine("{");
        s.AppendLine("    .reg .pred %p<2>;");
        s.AppendLine("    .reg .b32 %r<48>;");
        s.AppendLine("    .reg .b64 %rd<40>;");
        s.AppendLine("    .reg .f32 %f<64>;");
        s.AppendLine($"    .shared .align 4 .b8 wbg_a[{I(aBytes)}];");
        s.AppendLine($"    .shared .align 4 .b8 wbg_b[{I(bBytes)}];");
        s.AppendLine("    ld.param.u64 %rd0, [u_ptr];");
        s.AppendLine("    ld.param.u64 %rd1, [v_ptr];");
        s.AppendLine("    ld.param.u64 %rd2, [m_ptr];");
        // position bz = ctaid.z; offset the three pointers by the position slab.
        s.AppendLine("    mov.u32 %r40, %ctaid.z;");
        s.AppendLine($"    mul.wide.u32 %rd30, %r40, {I(kc)};");
        s.AppendLine("    shl.b64 %rd30, %rd30, 2;");
        s.AppendLine("    add.u64 %rd0, %rd0, %rd30;");           // U[bz]
        s.AppendLine($"    mul.wide.u32 %rd31, %r40, {I(cp)};");
        s.AppendLine("    shl.b64 %rd31, %rd31, 2;");
        s.AppendLine("    add.u64 %rd1, %rd1, %rd31;");           // V[bz]
        s.AppendLine($"    mul.wide.u32 %rd32, %r40, {I(kp)};");
        s.AppendLine("    shl.b64 %rd32, %rd32, 2;");
        s.AppendLine("    add.u64 %rd2, %rd2, %rd32;");           // M[bz]
        s.AppendLine("    mov.u32 %r0, %tid.x;");   // tx
        s.AppendLine("    mov.u32 %r1, %tid.y;");   // ty
        s.AppendLine("    mov.u32 %r2, %ctaid.x;"); // col tile
        s.AppendLine("    mov.u32 %r3, %ctaid.y;"); // row tile
        s.AppendLine($"    mul.lo.u32 %r4, %r3, {I(bm)};");        // rowTileBase
        s.AppendLine($"    mad.lo.u32 %r5, %r1, {I(tm)}, %r4;");   // rowBase = + ty*TM
        s.AppendLine($"    mul.lo.u32 %r6, %r2, {I(bn)};");        // colTileBase
        s.AppendLine($"    mad.lo.u32 %r9, %r0, {I(tn)}, %r6;");   // colFirst = colTileBase + tx*TN
        s.AppendLine($"    mad.lo.u32 %r10, %r1, {I(bn / tn)}, %r0;"); // tid
        s.AppendLine("    mov.u64 %rd4, wbg_a;");
        s.AppendLine("    mov.u64 %rd5, wbg_b;");
        s.AppendLine($"    mul.lo.u32 %r11, %r1, {I(tm * bk)};");
        s.AppendLine("    mul.wide.u32 %rd6, %r11, 4;");
        s.AppendLine("    add.u64 %rd6, %rd4, %rd6;");            // aBase
        s.AppendLine($"    mul.lo.u32 %r12, %r0, {I(tn)};");
        s.AppendLine("    mul.wide.u32 %rd7, %r12, 4;");
        s.AppendLine("    add.u64 %rd7, %rd5, %rd7;");            // bBase
        for (int a = 0; a < tm * tn; a++) s.AppendLine($"    mov.f32 %f{I(a)}, 0f00000000;");
        s.AppendLine("    mov.u32 %r13, 0;");
        s.AppendLine("WBG_LOOP:");
        for (int li = 0; li < loadsA; li++)
        {
            s.AppendLine($"    add.u32 %r20, %r10, {I(li * threads)};");
            s.AppendLine($"    shr.u32 %r21, %r20, {I(log2Bk)};");
            s.AppendLine($"    and.b32 %r22, %r20, {I(bk - 1)};");
            s.AppendLine("    add.u32 %r23, %r4, %r21;");
            s.AppendLine($"    mad.lo.u32 %r24, %r13, {I(bk)}, %r22;");
            s.AppendLine($"    mad.lo.u32 %r25, %r23, {I(c)}, %r24;");
            s.AppendLine("    mul.wide.u32 %rd10, %r25, 4;");
            s.AppendLine("    add.u64 %rd10, %rd0, %rd10;");
            s.AppendLine("    ld.global.nc.f32 %f40, [%rd10];");
            s.AppendLine("    mul.wide.u32 %rd11, %r20, 4;");
            s.AppendLine("    add.u64 %rd11, %rd4, %rd11;");
            s.AppendLine("    st.shared.f32 [%rd11], %f40;");
        }
        for (int li = 0; li < loadsB; li++)
        {
            s.AppendLine($"    add.u32 %r20, %r10, {I(li * threads)};");
            s.AppendLine($"    shr.u32 %r21, %r20, {I(log2Bn)};");
            s.AppendLine($"    and.b32 %r22, %r20, {I(bn - 1)};");
            s.AppendLine($"    mad.lo.u32 %r23, %r13, {I(bk)}, %r21;");
            s.AppendLine($"    mad.lo.u32 %r25, %r23, {I(pp)}, %r6;");
            s.AppendLine("    add.u32 %r25, %r25, %r22;");
            s.AppendLine("    mul.wide.u32 %rd10, %r25, 4;");
            s.AppendLine("    add.u64 %rd10, %rd1, %rd10;");
            s.AppendLine("    ld.global.nc.f32 %f40, [%rd10];");
            s.AppendLine("    mul.wide.u32 %rd11, %r20, 4;");
            s.AppendLine("    add.u64 %rd11, %rd5, %rd11;");
            s.AppendLine("    st.shared.f32 [%rd11], %f40;");
        }
        s.AppendLine("    bar.sync 0;");
        for (int kk = 0; kk < bk; kk++)
        {
            for (int m = 0; m < tm; m++)
                s.AppendLine($"    ld.shared.f32 %f{I(16 + m)}, [%rd6+{I((m * bk + kk) * 4)}];");
            for (int nn = 0; nn < tn; nn++)
                s.AppendLine($"    ld.shared.f32 %f{I(24 + nn)}, [%rd7+{I((kk * bn + nn) * 4)}];");
            for (int m = 0; m < tm; m++)
                for (int nn = 0; nn < tn; nn++)
                    s.AppendLine($"    fma.rn.f32 %f{I(m * tn + nn)}, %f{I(16 + m)}, %f{I(24 + nn)}, %f{I(m * tn + nn)};");
        }
        s.AppendLine("    bar.sync 0;");
        s.AppendLine("    add.u32 %r13, %r13, 1;");
        s.AppendLine($"    setp.lt.u32 %p0, %r13, {I(contractionTiles)};");
        s.AppendLine("    @%p0 bra WBG_LOOP;");
        // store M[bz][rowBase+m][colFirst+nn]
        for (int m = 0; m < tm; m++)
        {
            s.AppendLine($"    add.u32 %r30, %r5, {I(m)};");           // row
            s.AppendLine($"    mul.lo.u32 %r31, %r30, {I(pp)};");      // row*P
            s.AppendLine("    add.u32 %r31, %r31, %r9;");              // + colFirst
            for (int nn = 0; nn < tn; nn++)
            {
                s.AppendLine($"    add.u32 %r32, %r31, {I(nn)};");
                s.AppendLine("    mul.wide.u32 %rd13, %r32, 4;");
                s.AppendLine("    add.u64 %rd13, %rd2, %rd13;");
                s.AppendLine($"    st.global.f32 [%rd13], %f{I(m * tn + nn)};");
            }
        }
        s.AppendLine("    ret;");
        s.AppendLine("}");
        return s.ToString();
    }

    private static int Log2(int v) { int r = 0; while ((v >>= 1) != 0) r++; return r; }

    public void Dispose() => _module.Dispose();
}
