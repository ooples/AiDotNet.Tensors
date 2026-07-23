using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Register-blocked fused Winograd F(2,3) GEMM + output transform. Like
/// <see cref="PtxWinogradF23FusedGemmKernel"/> (reads U[16,K,C]+V[16,C,P], fuses
/// A^T M A + bias + ReLU with no M workspace), but each thread computes a
/// TM x TN micro-tile of output — so every shared-memory load feeds TM*TN FMAs
/// instead of one, lifting the arithmetic intensity above the shared-bandwidth
/// bound that limits the TM=TN=1 variant. The 16 Winograd positions hold
/// 16*TM*TN accumulators, but fragments are consumed one position at a time so
/// the live register count stays modest (no spills at TM=TN=2). This directly
/// attacks the intermediate-buffer + reuse gap that leaves the batched pipeline
/// at ~0.73x cuDNN.
/// </summary>
internal sealed class PtxWinogradF23FusedRegBlockedKernel : IDisposable
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
    internal int ThreadM { get; }
    internal int ThreadN { get; }

    internal string Ptx { get; }
    internal DirectPtxFunctionInfo FunctionInfo { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal int TileRows => Height / 2;
    internal int TileCols => Width / 2;
    internal int Tiles => Batch * TileRows * TileCols;
    internal int BlockThreads => (BlockM / ThreadM) * (BlockN / ThreadN);
    internal long UBytes => (long)16 * OutputChannels * InputChannels * sizeof(float);
    internal long VBytes => (long)16 * InputChannels * Tiles * sizeof(float);
    internal long BiasBytes => (long)OutputChannels * sizeof(float);
    internal long OutputBytes => (long)Batch * OutputChannels * Height * Width * sizeof(float);

    internal string EntryPoint => FormattableString.Invariant(
        $"aidotnet_winograd_f23_fused_rb_n{Batch}_c{InputChannels}_h{Height}_w{Width}_k{OutputChannels}_{BlockM}x{BlockN}x{BlockK}_{ThreadM}x{ThreadN}");

    internal PtxWinogradF23FusedRegBlockedKernel(
        DirectPtxRuntime runtime, int batch, int inputChannels, int height, int width, int outputChannels,
        int blockM = 32, int blockN = 32, int blockK = 8, int threadM = 2, int threadN = 2)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new NotSupportedException("Register-blocked fused Winograd has no experimental non-SM86 specialization.");
        if ((height & 1) != 0 || (width & 1) != 0) throw new ArgumentException("Even H,W required.");
        Batch = batch; InputChannels = inputChannels; Height = height; Width = width; OutputChannels = outputChannels;
        BlockM = blockM; BlockN = blockN; BlockK = blockK; ThreadM = threadM; ThreadN = threadN;
        if (outputChannels % blockM != 0 || Tiles % blockN != 0 || inputChannels % blockK != 0)
            throw new ArgumentException("BM|K, BN|P, BK|C required.");
        if (blockM % threadM != 0 || blockN % threadN != 0)
            throw new ArgumentException("BM%TM==0, BN%TN==0 required.");
        if (!IsPow2(blockM * blockK) || !IsPow2(blockK) || !IsPow2(blockN))
            throw new ArgumentException("BM*BK, BK, BN must be powers of two (flat cooperative load).");
        int threads = BlockThreads;
        if (threads > 1024 || 16 * blockM * blockK % threads != 0 || 16 * blockK * blockN % threads != 0)
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
        var bias = new DirectPtxExtent(OutputChannels);
        var output = new DirectPtxExtent(Batch, OutputChannels, Height, Width);
        int sharedBytes = 16 * (BlockM * BlockK + BlockK * BlockN) * sizeof(float);
        return new DirectPtxKernelBlueprint(
            Operation: "winograd-f23-fused-regblocked",
            Version: 1,
            Architecture: architecture,
            Variant: FormattableString.Invariant(
                $"n{Batch}-c{InputChannels}-h{Height}-w{Width}-k{OutputChannels}-bm{BlockM}-bn{BlockN}-bk{BlockK}-tm{ThreadM}-tn{ThreadN}-fp32"),
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
                MaxRegistersPerThread: 200,   // heavy: 16*TM*TN accs + unrolled inner/epilogue
                MaxStaticSharedBytes: sharedBytes,
                MaxLocalBytesPerThread: 0, MinBlocksPerMultiprocessor: 1),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["equation"] = "relu(A^T (sum_c U (x) V) A + bias)",
                ["algorithm"] = "winograd-f23-fused-register-blocked",
                ["microtile"] = FormattableString.Invariant($"{ThreadM}x{ThreadN}"),
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
        _module.Launch(_function, gridX, gridY, 1,
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
            throw new NotSupportedException("Only the experimental SM86 register-blocked fused Winograd emitter exists.");
        string I(int v) => v.ToString(CultureInfo.InvariantCulture);
        int c = InputChannels, k = OutputChannels, h = Height, w = Width;
        int th = TileRows, tw = TileCols, pp = Tiles;
        int bm = BlockM, bn = BlockN, bk = BlockK, tm = ThreadM, tn = ThreadN, threads = BlockThreads;
        int kc = k * c, cp = c * pp;
        int uTile = bm * bk, vTile = bk * bn;
        int usLoads = 16 * uTile / threads, vsLoads = 16 * vTile / threads;
        int usBytes = 16 * uTile * 4, vsBytes = 16 * vTile * 4;
        int contractionTiles = c / bk;
        int log2UTile = Log2(uTile), log2Bk = Log2(bk), log2VTile = Log2(vTile), log2Bn = Log2(bn);
        int acc = 16 * tm * tn;              // M accumulators: %f0..acc-1
        int aReg = acc, bReg = acc + tm;     // fragment banks
        int loadTmp = acc + tm + tn;
        // epilogue register banks (above everything above)
        int sBank = loadTmp + 4;             // s[2][4] = 8
        int yBank = sBank + 8;               // Y[2][2] = 4
        int biasReg = yBank + 4;
        int totalF = biasReg + 2;
        string entry = EntryPoint;
        int Midx(int xi, int im, int jn) => xi * (tm * tn) + im * tn + jn;

        var s = new StringBuilder(131072);
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
        s.AppendLine($"    .reg .f32 %f<{I(totalF + 2)}>;");
        s.AppendLine($"    .shared .align 4 .b8 wfr_us[{I(usBytes)}];");
        s.AppendLine($"    .shared .align 4 .b8 wfr_vs[{I(vsBytes)}];");
        s.AppendLine("    ld.param.u64 %rd0, [u_ptr];");
        s.AppendLine("    ld.param.u64 %rd1, [v_ptr];");
        s.AppendLine("    ld.param.u64 %rd2, [bias_ptr];");
        s.AppendLine("    ld.param.u64 %rd3, [output_ptr];");
        s.AppendLine("    mov.u32 %r0, %tid.x;");
        s.AppendLine("    mov.u32 %r1, %tid.y;");
        s.AppendLine("    mov.u32 %r2, %ctaid.x;");
        s.AppendLine("    mov.u32 %r3, %ctaid.y;");
        s.AppendLine($"    mul.lo.u32 %r4, %r3, {I(bm)};");        // kBase
        s.AppendLine($"    mul.lo.u32 %r5, %r2, {I(bn)};");        // pBase
        s.AppendLine($"    mad.lo.u32 %r6, %r1, {I(tm)}, %r4;");   // rowBase = kBase + ty*TM
        s.AppendLine($"    mad.lo.u32 %r7, %r0, {I(tn)}, %r5;");   // colBase = pBase + tx*TN
        s.AppendLine($"    mad.lo.u32 %r8, %r1, {I(bn / tn)}, %r0;"); // tid
        s.AppendLine("    mov.u64 %rd4, wfr_us;");
        s.AppendLine("    mov.u64 %rd5, wfr_vs;");
        s.AppendLine($"    mul.lo.u32 %r9, %r1, {I(tm * bk)};");   // (ty*TM)*BK
        s.AppendLine("    mul.wide.u32 %rd6, %r9, 4;");
        s.AppendLine("    add.u64 %rd6, %rd4, %rd6;");             // UsRowBase
        s.AppendLine($"    mul.lo.u32 %r10, %r0, {I(tn)};");       // tx*TN
        s.AppendLine("    mul.wide.u32 %rd7, %r10, 4;");
        s.AppendLine("    add.u64 %rd7, %rd5, %rd7;");             // VsColBase
        for (int i = 0; i < acc; i++) s.AppendLine($"    mov.f32 %f{I(i)}, 0f00000000;");
        s.AppendLine("    mov.u32 %r13, 0;");
        s.AppendLine("WFR_LOOP:");
        s.AppendLine($"    mul.lo.u32 %r11, %r13, {I(bk)};");      // c0
        for (int it = 0; it < usLoads; it++)
        {
            s.AppendLine($"    add.u32 %r20, %r8, {I(it * threads)};");
            s.AppendLine($"    shr.u32 %r21, %r20, {I(log2UTile)};");
            s.AppendLine($"    and.b32 %r22, %r20, {I(uTile - 1)};");
            s.AppendLine($"    shr.u32 %r23, %r22, {I(log2Bk)};");
            s.AppendLine($"    and.b32 %r24, %r22, {I(bk - 1)};");
            s.AppendLine($"    mad.lo.u32 %r25, %r21, {I(kc)}, 0;");
            s.AppendLine("    add.u32 %r26, %r4, %r23;");
            s.AppendLine($"    mad.lo.u32 %r25, %r26, {I(c)}, %r25;");
            s.AppendLine("    add.u32 %r25, %r25, %r11;");
            s.AppendLine("    add.u32 %r25, %r25, %r24;");
            s.AppendLine("    mul.wide.u32 %rd10, %r25, 4;");
            s.AppendLine("    add.u64 %rd10, %rd0, %rd10;");
            s.AppendLine($"    ld.global.nc.f32 %f{I(loadTmp)}, [%rd10];");
            s.AppendLine("    mul.wide.u32 %rd11, %r20, 4;");
            s.AppendLine("    add.u64 %rd11, %rd4, %rd11;");
            s.AppendLine($"    st.shared.f32 [%rd11], %f{I(loadTmp)};");
        }
        for (int it = 0; it < vsLoads; it++)
        {
            s.AppendLine($"    add.u32 %r20, %r8, {I(it * threads)};");
            s.AppendLine($"    shr.u32 %r21, %r20, {I(log2VTile)};");
            s.AppendLine($"    and.b32 %r22, %r20, {I(vTile - 1)};");
            s.AppendLine($"    shr.u32 %r23, %r22, {I(log2Bn)};");
            s.AppendLine($"    and.b32 %r24, %r22, {I(bn - 1)};");
            s.AppendLine($"    mul.lo.u32 %r25, %r21, {I(cp)};");
            s.AppendLine("    add.u32 %r26, %r11, %r23;");
            s.AppendLine($"    mad.lo.u32 %r25, %r26, {I(pp)}, %r25;");
            s.AppendLine("    add.u32 %r25, %r25, %r5;");
            s.AppendLine("    add.u32 %r25, %r25, %r24;");
            s.AppendLine("    mul.wide.u32 %rd10, %r25, 4;");
            s.AppendLine("    add.u64 %rd10, %rd1, %rd10;");
            s.AppendLine($"    ld.global.nc.f32 %f{I(loadTmp)}, [%rd10];");
            s.AppendLine("    mul.wide.u32 %rd11, %r20, 4;");
            s.AppendLine("    add.u64 %rd11, %rd5, %rd11;");
            s.AppendLine($"    st.shared.f32 [%rd11], %f{I(loadTmp)};");
        }
        s.AppendLine("    bar.sync 0;");
        for (int kk = 0; kk < bk; kk++)
            for (int xi = 0; xi < 16; xi++)
            {
                for (int im = 0; im < tm; im++)
                    s.AppendLine($"    ld.shared.f32 %f{I(aReg + im)}, [%rd6+{I((xi * uTile + im * bk + kk) * 4)}];");
                for (int jn = 0; jn < tn; jn++)
                    s.AppendLine($"    ld.shared.f32 %f{I(bReg + jn)}, [%rd7+{I((xi * vTile + kk * bn + jn) * 4)}];");
                for (int im = 0; im < tm; im++)
                    for (int jn = 0; jn < tn; jn++)
                        s.AppendLine($"    fma.rn.f32 %f{I(Midx(xi, im, jn))}, %f{I(aReg + im)}, %f{I(bReg + jn)}, %f{I(Midx(xi, im, jn))};");
            }
        s.AppendLine("    bar.sync 0;");
        s.AppendLine("    add.u32 %r13, %r13, 1;");
        s.AppendLine($"    setp.lt.u32 %p0, %r13, {I(contractionTiles)};");
        s.AppendLine("    @%p0 bra WFR_LOOP;");
        // epilogue: for each (im,jn) output, output-transform the 16 M[xi], +bias, relu, store.
        for (int im = 0; im < tm; im++)
        {
            // row = rowBase + im ; bias[row] ; outRowBase = (n_row)*... computed per (im): need n from p (jn-dependent)
            s.AppendLine($"    add.u32 %r30, %r6, {I(im)};");   // row (k)
            s.AppendLine("    mul.wide.u32 %rd12, %r30, 4;");
            s.AppendLine("    add.u64 %rd12, %rd2, %rd12;");
            s.AppendLine($"    ld.global.nc.f32 %f{I(biasReg)}, [%rd12];");  // bias[row]
            for (int jn = 0; jn < tn; jn++)
            {
                // s = A^T M ; M[i][j] = %f Midx(i*4+j, im, jn)
                int M(int i, int j) => Midx(i * 4 + j, im, jn);
                int S(int i, int j) => sBank + i * 4 + j;
                int Y(int i, int j) => yBank + i * 2 + j;
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
                // p = colBase + jn ; decode -> n, ti, tj
                s.AppendLine($"    add.u32 %r31, %r7, {I(jn)};");          // p
                s.AppendLine($"    rem.u32 %r32, %r31, {I(tw)};");        // tj
                s.AppendLine($"    div.u32 %r33, %r31, {I(tw)};");
                s.AppendLine($"    rem.u32 %r34, %r33, {I(th)};");        // ti
                s.AppendLine($"    div.u32 %r35, %r33, {I(th)};");        // n
                s.AppendLine("    mul.lo.u32 %r36, %r34, 2;");            // oh
                s.AppendLine("    mul.lo.u32 %r37, %r32, 2;");            // ow
                s.AppendLine($"    mad.lo.u32 %r38, %r35, {I(k)}, %r30;");// n*K + row
                s.AppendLine($"    mul.wide.u32 %rd13, %r38, {I(h * w)};");
                s.AppendLine("    shl.b64 %rd13, %rd13, 2;");
                s.AppendLine("    add.u64 %rd13, %rd3, %rd13;");
                for (int oi = 0; oi < 2; oi++)
                    for (int oj = 0; oj < 2; oj++)
                    {
                        int yr = yBank + oi * 2 + oj;
                        s.AppendLine($"    add.rn.f32 %f{I(yr)}, %f{I(yr)}, %f{I(biasReg)};");
                        s.AppendLine($"    max.f32 %f{I(yr)}, %f{I(yr)}, 0f00000000;");
                        s.AppendLine($"    add.u32 %r39, %r36, {I(oi)};");
                        s.AppendLine($"    mul.lo.u32 %r39, %r39, {I(w)};");
                        s.AppendLine("    add.u32 %r39, %r39, %r37;");
                        s.AppendLine($"    add.u32 %r39, %r39, {I(oj)};");
                        s.AppendLine("    mul.wide.u32 %rd14, %r39, 4;");
                        s.AppendLine("    add.u64 %rd14, %rd13, %rd14;");
                        s.AppendLine($"    st.global.f32 [%rd14], %f{I(yr)};");
                    }
            }
        }
        s.AppendLine("    ret;");
        s.AppendLine("}");
        return s.ToString();
    }

    private static int Log2(int v) { int r = 0; while ((v >>= 1) != 0) r++; return r; }

    public void Dispose() => _module.Dispose();
}
