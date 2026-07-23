using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Fused Winograd F(2,3) batched-GEMM + output transform. Consumes the
/// position-major filter transform U[16,K,C] and input transform V[16,C,P]
/// (P = N*TH*TW output tiles) and produces the 3x3 same-padded conv+bias+ReLU
/// output directly — never materializing the M = U (x) V intermediate.
///
/// <para>A thread block computes a BM(output channels) x BN(output tiles) tile:
/// it stages U and V sub-tiles into shared memory (the data reuse that the naive
/// per-tile kernel lacks) and accumulates all 16 Winograd positions M[16] per
/// output element in registers over the channel dimension. The epilogue applies
/// A^T M A (2x2), adds bias, ReLUs, and scatters to output[N,K,H,W]. This is the
/// register-blocked-GEMM lesson from the 1x1 winner applied to the 16 Winograd
/// GEMMs, with the output transform fused to avoid the M workspace round-trip.</para>
/// </summary>
internal sealed class PtxWinogradF23FusedGemmKernel : IDisposable
{
    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int Batch { get; }
    internal int InputChannels { get; }
    internal int Height { get; }
    internal int Width { get; }
    internal int OutputChannels { get; }
    internal int BlockM { get; }
    internal int BlockN { get; }
    internal int BlockK { get; }

    internal string Ptx { get; }
    internal DirectPtxFunctionInfo FunctionInfo { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal int TileRows => Height / 2;
    internal int TileCols => Width / 2;
    internal int Tiles => Batch * TileRows * TileCols;    // P
    internal int BlockThreads => BlockM * BlockN;
    internal long UBytes => (long)16 * OutputChannels * InputChannels * sizeof(float);
    internal long VBytes => (long)16 * InputChannels * Tiles * sizeof(float);
    internal long BiasBytes => (long)OutputChannels * sizeof(float);
    internal long OutputBytes => (long)Batch * OutputChannels * Height * Width * sizeof(float);

    internal string EntryPoint => FormattableString.Invariant(
        $"aidotnet_winograd_f23_fused_gemm_n{Batch}_c{InputChannels}_h{Height}_w{Width}_k{OutputChannels}_{BlockM}x{BlockN}x{BlockK}");

    internal PtxWinogradF23FusedGemmKernel(
        DirectPtxRuntime runtime, int batch, int inputChannels, int height, int width, int outputChannels,
        int blockM = 16, int blockN = 16, int blockK = 8)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new NotSupportedException("Fused Winograd GEMM has no experimental non-SM86 specialization.");
        if ((height & 1) != 0 || (width & 1) != 0) throw new ArgumentException("Even H,W required.");
        Batch = batch; InputChannels = inputChannels; Height = height; Width = width; OutputChannels = outputChannels;
        BlockM = blockM; BlockN = blockN; BlockK = blockK;
        if (outputChannels % blockM != 0 || Tiles % blockN != 0 || inputChannels % blockK != 0)
            throw new ArgumentException("BM|K, BN|P, BK|C required.");
        int threads = blockM * blockN;
        if (16 * blockM * blockK % threads != 0 || 16 * blockK * blockN % threads != 0)
            throw new ArgumentException("Shared U/V tiles must divide evenly across threads.");
        if (threads > 1024) throw new ArgumentException("Block exceeds 1024 threads.");

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

    internal DirectPtxKernelBlueprint CreateBlueprint(DirectPtxArchitectureFamily architecture)
    {
        var u = new DirectPtxExtent(16, OutputChannels, InputChannels);
        var v = new DirectPtxExtent(16, InputChannels, Tiles);
        var bias = new DirectPtxExtent(OutputChannels);
        var output = new DirectPtxExtent(Batch, OutputChannels, Height, Width);
        int sharedBytes = 16 * (BlockM * BlockK + BlockK * BlockN) * sizeof(float);
        return new DirectPtxKernelBlueprint(
            Operation: "winograd-f23-fused-gemm",
            Version: 1,
            Architecture: architecture,
            Variant: FormattableString.Invariant(
                $"n{Batch}-c{InputChannels}-h{Height}-w{Width}-k{OutputChannels}-bm{BlockM}-bn{BlockN}-bk{BlockK}-fp32"),
            Tensors:
            [
                new("U", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    u, u, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("V", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    v, v, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("bias", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    bias, bias, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Nchw,
                    output, output, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 96, MaxStaticSharedBytes: sharedBytes,
                MaxLocalBytesPerThread: 0, MinBlocksPerMultiprocessor: 1),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["equation"] = "relu(A^T (sum_c U (x) V) A + bias)",
                ["algorithm"] = "winograd-f23-fused-batched-gemm",
                ["input"] = "fp32", ["accumulator"] = "fp32-fma", ["output"] = "fp32",
                ["intermediate-global-bytes"] = "0",
                ["shape-selection"] = "host-only-exact-contract",
                ["promotion"] = "experimental-pending-gpu-evidence"
            });
    }

    internal unsafe void Launch(
        DirectPtxTensorView u, DirectPtxTensorView v, DirectPtxTensorView bias, DirectPtxTensorView output)
    {
        Require(u, Blueprint.Tensors[0], nameof(u));
        Require(v, Blueprint.Tensors[1], nameof(v));
        Require(bias, Blueprint.Tensors[2], nameof(bias));
        Require(output, Blueprint.Tensors[3], nameof(output));
        IntPtr uPtr = u.Pointer, vPtr = v.Pointer, bPtr = bias.Pointer, oPtr = output.Pointer;
        void** arguments = stackalloc void*[4];
        arguments[0] = &uPtr; arguments[1] = &vPtr; arguments[2] = &bPtr; arguments[3] = &oPtr;
        uint gridX = (uint)(Tiles / BlockN);
        uint gridY = (uint)(OutputChannels / BlockM);
        _module.Launch(_function, gridX, gridY, 1, (uint)BlockN, (uint)BlockM, 1, 0, arguments);
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
            throw new NotSupportedException("Only the experimental SM86 fused Winograd GEMM emitter exists.");
        string I(int v) => v.ToString(CultureInfo.InvariantCulture);
        int c = InputChannels, k = OutputChannels, h = Height, w = Width;
        int th = TileRows, tw = TileCols, p = Tiles;
        int bm = BlockM, bn = BlockN, bk = BlockK, threads = BlockThreads;
        int kc = k * c, cp = c * p;
        int uTileElems = bm * bk, vTileElems = bk * bn;   // per-position shared tile sizes
        int usLoads = 16 * uTileElems / threads, vsLoads = 16 * vTileElems / threads;
        int usBytes = 16 * uTileElems * 4, vsBytes = 16 * vTileElems * 4;
        int contractionTiles = c / bk;
        string entry = EntryPoint;

        var s = new StringBuilder(65536);
        s.AppendLine(".version 7.1");
        s.AppendLine($".target sm_{major}{minor}");
        s.AppendLine(".address_size 64");
        s.AppendLine();
        s.AppendLine($".visible .entry {entry}(");
        s.AppendLine("    .param .u64 u_ptr,");
        s.AppendLine("    .param .u64 v_ptr,");
        s.AppendLine("    .param .u64 bias_ptr,");
        s.AppendLine("    .param .u64 output_ptr");
        s.AppendLine(")");
        s.AppendLine("{");
        s.AppendLine("    .reg .pred %p<2>;");
        s.AppendLine("    .reg .b32 %r<48>;");
        s.AppendLine("    .reg .b64 %rd<32>;");
        s.AppendLine("    .reg .f32 %f<64>;");
        s.AppendLine($"    .shared .align 4 .b8 wino_us[{I(usBytes)}];");
        s.AppendLine($"    .shared .align 4 .b8 wino_vs[{I(vsBytes)}];");
        s.AppendLine("    ld.param.u64 %rd0, [u_ptr];");
        s.AppendLine("    ld.param.u64 %rd1, [v_ptr];");
        s.AppendLine("    ld.param.u64 %rd2, [bias_ptr];");
        s.AppendLine("    ld.param.u64 %rd3, [output_ptr];");
        s.AppendLine("    mov.u32 %r0, %tid.x;");   // tx (p within tile)
        s.AppendLine("    mov.u32 %r1, %tid.y;");   // ty (k within tile)
        s.AppendLine("    mov.u32 %r2, %ctaid.x;"); // p-tile
        s.AppendLine("    mov.u32 %r3, %ctaid.y;"); // k-tile
        s.AppendLine($"    mad.lo.u32 %r4, %r3, {I(bm)}, %r1;");   // k = kBase + ty
        s.AppendLine($"    mad.lo.u32 %r5, %r2, {I(bn)}, %r0;");   // p = pBase + tx
        s.AppendLine($"    mul.lo.u32 %r6, %r3, {I(bm)};");        // kBase
        s.AppendLine($"    mul.lo.u32 %r7, %r2, {I(bn)};");        // pBase
        s.AppendLine($"    mad.lo.u32 %r8, %r1, {I(bn)}, %r0;");   // tid = ty*BN + tx
        s.AppendLine("    mov.u64 %rd4, wino_us;");
        s.AppendLine("    mov.u64 %rd5, wino_vs;");
        // per-thread shared bases for the inner product
        s.AppendLine($"    mul.lo.u32 %r9, %r1, {I(bk)};");        // ty*BK
        s.AppendLine("    mul.wide.u32 %rd6, %r9, 4;");
        s.AppendLine("    add.u64 %rd6, %rd4, %rd6;");             // UsRowBase = us + ty*BK*4
        s.AppendLine("    mul.wide.u32 %rd7, %r0, 4;");
        s.AppendLine("    add.u64 %rd7, %rd5, %rd7;");             // VsColBase = vs + tx*4
        // zero M[0..15]
        for (int i = 0; i < 16; i++) s.AppendLine($"    mov.f32 %f{I(i)}, 0f00000000;");
        s.AppendLine("    mov.u32 %r10, 0;");                     // contraction tile index
        s.AppendLine("WINO_GEMM_LOOP:");
        s.AppendLine($"    mul.lo.u32 %r11, %r10, {I(bk)};");     // c0 = ct*BK
        // cooperative flat load Us: l = tid + it*threads ; xi=l/(BM*BK), i=(l%(BM*BK))/BK, j=l%BK
        int log2UTile = Log2(uTileElems), log2Bk = Log2(bk);
        for (int it = 0; it < usLoads; it++)
        {
            s.AppendLine($"    add.u32 %r12, %r8, {I(it * threads)};");   // l
            s.AppendLine($"    shr.u32 %r13, %r12, {I(log2UTile)};");     // xi
            s.AppendLine($"    and.b32 %r14, %r12, {I(uTileElems - 1)};");// rem
            s.AppendLine($"    shr.u32 %r15, %r14, {I(log2Bk)};");        // i (row in BM)
            s.AppendLine($"    and.b32 %r16, %r14, {I(bk - 1)};");        // j (col in BK)
            // U global: U[xi*K*C + (kBase+i)*C + (c0+j)]
            s.AppendLine($"    mad.lo.u32 %r17, %r13, {I(kc)}, 0;");      // xi*KC
            s.AppendLine("    add.u32 %r18, %r6, %r15;");                 // kBase + i
            s.AppendLine($"    mad.lo.u32 %r17, %r18, {I(c)}, %r17;");    // + (kBase+i)*C
            s.AppendLine("    add.u32 %r17, %r17, %r11;");                // + c0
            s.AppendLine("    add.u32 %r17, %r17, %r16;");                // + j
            s.AppendLine("    mul.wide.u32 %rd10, %r17, 4;");
            s.AppendLine("    add.u64 %rd10, %rd0, %rd10;");
            s.AppendLine("    ld.global.nc.f32 %f16, [%rd10];");
            s.AppendLine("    mul.wide.u32 %rd11, %r12, 4;");
            s.AppendLine("    add.u64 %rd11, %rd4, %rd11;");
            s.AppendLine("    st.shared.f32 [%rd11], %f16;");
        }
        // cooperative flat load Vs: l ; xi=l/(BK*BN), i=(l%(BK*BN))/BN, j=l%BN
        int log2VTile = Log2(vTileElems), log2Bn = Log2(bn);
        for (int it = 0; it < vsLoads; it++)
        {
            s.AppendLine($"    add.u32 %r12, %r8, {I(it * threads)};");
            s.AppendLine($"    shr.u32 %r13, %r12, {I(log2VTile)};");     // xi
            s.AppendLine($"    and.b32 %r14, %r12, {I(vTileElems - 1)};");
            s.AppendLine($"    shr.u32 %r15, %r14, {I(log2Bn)};");        // i (row in BK)
            s.AppendLine($"    and.b32 %r16, %r14, {I(bn - 1)};");        // j (col in BN)
            // V global: V[xi*C*P + (c0+i)*P + (pBase+j)]
            s.AppendLine($"    mul.lo.u32 %r17, %r13, {I(cp)};");         // xi*CP
            s.AppendLine("    add.u32 %r18, %r11, %r15;");                // c0 + i
            s.AppendLine($"    mad.lo.u32 %r17, %r18, {I(p)}, %r17;");    // + (c0+i)*P
            s.AppendLine("    add.u32 %r17, %r17, %r7;");                 // + pBase
            s.AppendLine("    add.u32 %r17, %r17, %r16;");                // + j
            s.AppendLine("    mul.wide.u32 %rd10, %r17, 4;");
            s.AppendLine("    add.u64 %rd10, %rd1, %rd10;");
            s.AppendLine("    ld.global.nc.f32 %f16, [%rd10];");
            s.AppendLine("    mul.wide.u32 %rd11, %r12, 4;");
            s.AppendLine("    add.u64 %rd11, %rd5, %rd11;");
            s.AppendLine("    st.shared.f32 [%rd11], %f16;");
        }
        s.AppendLine("    bar.sync 0;");
        // inner product: for kk, for xi: M[xi] += Us[xi][ty][kk] * Vs[xi][kk][tx]
        for (int kk = 0; kk < bk; kk++)
            for (int xi = 0; xi < 16; xi++)
            {
                int usOff = (xi * uTileElems + kk) * 4;             // UsRowBase + (xi*BM*BK + kk)*4
                int vsOff = (xi * vTileElems + kk * bn) * 4;        // VsColBase + (xi*BK*BN + kk*BN)*4
                s.AppendLine($"    ld.shared.f32 %f16, [%rd6+{I(usOff)}];");
                s.AppendLine($"    ld.shared.f32 %f17, [%rd7+{I(vsOff)}];");
                s.AppendLine($"    fma.rn.f32 %f{I(xi)}, %f16, %f17, %f{I(xi)};");
            }
        s.AppendLine("    bar.sync 0;");
        s.AppendLine("    add.u32 %r10, %r10, 1;");
        s.AppendLine($"    setp.lt.u32 %p0, %r10, {I(contractionTiles)};");
        s.AppendLine("    @%p0 bra WINO_GEMM_LOOP;");
        // output transform Y = A^T M A -> %f40..%f43 (2x2), scratch s in %f48..55
        EmitOutputTransform(s);
        // decode p -> (n, ti, tj): p = %r5
        s.AppendLine($"    rem.u32 %r20, %r5, {I(tw)};");   // tj
        s.AppendLine($"    div.u32 %r21, %r5, {I(tw)};");
        s.AppendLine($"    rem.u32 %r22, %r21, {I(th)};");  // ti
        s.AppendLine($"    div.u32 %r23, %r21, {I(th)};");  // n
        s.AppendLine("    mul.lo.u32 %r24, %r22, 2;");      // oh = 2*ti
        s.AppendLine("    mul.lo.u32 %r25, %r20, 2;");      // ow = 2*tj
        // bias[k]
        s.AppendLine("    mul.wide.u32 %rd12, %r4, 4;");
        s.AppendLine("    add.u64 %rd12, %rd2, %rd12;");
        s.AppendLine("    ld.global.nc.f32 %f44, [%rd12];");
        // output plane base: output + (n*K + k)*H*W*4
        s.AppendLine($"    mad.lo.u32 %r26, %r23, {I(k)}, %r4;");   // n*K + k
        s.AppendLine($"    mul.wide.u32 %rd13, %r26, {I(h * w)};");
        s.AppendLine("    shl.b64 %rd13, %rd13, 2;");
        s.AppendLine("    add.u64 %rd13, %rd3, %rd13;");
        for (int oi = 0; oi < 2; oi++)
            for (int oj = 0; oj < 2; oj++)
            {
                int yreg = 40 + oi * 2 + oj;
                s.AppendLine($"    add.rn.f32 %f{I(yreg)}, %f{I(yreg)}, %f44;");
                s.AppendLine($"    max.f32 %f{I(yreg)}, %f{I(yreg)}, 0f00000000;");
                s.AppendLine($"    add.u32 %r27, %r24, {I(oi)};");     // oh+oi
                s.AppendLine($"    mul.lo.u32 %r27, %r27, {I(w)};");
                s.AppendLine("    add.u32 %r27, %r27, %r25;");         // + ow
                s.AppendLine($"    add.u32 %r27, %r27, {I(oj)};");     // + oj
                s.AppendLine("    mul.wide.u32 %rd14, %r27, 4;");
                s.AppendLine("    add.u64 %rd14, %rd13, %rd14;");
                s.AppendLine($"    st.global.f32 [%rd14], %f{I(yreg)};");
            }
        s.AppendLine("    ret;");
        s.AppendLine("}");
        return s.ToString();
    }

    // Y = A^T M A ; M in %f0..15 (row-major i*4+j); Y in %f40..43 ; scratch s in %f48..55.
    private static void EmitOutputTransform(StringBuilder s)
    {
        string I(int v) => v.ToString(CultureInfo.InvariantCulture);
        int M(int i, int j) => i * 4 + j;
        int S(int i, int j) => 48 + i * 4 + j;
        int Y(int i, int j) => 40 + i * 2 + j;
        for (int j = 0; j < 4; j++)
        {
            s.AppendLine($"    add.rn.f32 %f{I(S(0, j))}, %f{I(M(0, j))}, %f{I(M(1, j))};");
            s.AppendLine($"    add.rn.f32 %f{I(S(0, j))}, %f{I(S(0, j))}, %f{I(M(2, j))};");
            s.AppendLine($"    sub.rn.f32 %f{I(S(1, j))}, %f{I(M(1, j))}, %f{I(M(2, j))};");
            s.AppendLine($"    sub.rn.f32 %f{I(S(1, j))}, %f{I(S(1, j))}, %f{I(M(3, j))};");
        }
        for (int i = 0; i < 2; i++)
        {
            s.AppendLine($"    add.rn.f32 %f{I(Y(i, 0))}, %f{I(S(i, 0))}, %f{I(S(i, 1))};");
            s.AppendLine($"    add.rn.f32 %f{I(Y(i, 0))}, %f{I(Y(i, 0))}, %f{I(S(i, 2))};");
            s.AppendLine($"    sub.rn.f32 %f{I(Y(i, 1))}, %f{I(S(i, 1))}, %f{I(S(i, 2))};");
            s.AppendLine($"    sub.rn.f32 %f{I(Y(i, 1))}, %f{I(Y(i, 1))}, %f{I(S(i, 3))};");
        }
    }

    private static int Log2(int v) { int r = 0; while ((v >>= 1) != 0) r++; return r; }

    public void Dispose() => _module.Dispose();
}
