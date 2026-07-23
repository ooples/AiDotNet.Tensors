using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Software-pipelined fully-fused FP16 Tensor-Core Winograd F(2,3) 3x3 convolution.
/// Same all-K fusion as <see cref="PtxWinogradWmmaFusedAllKKernel"/> (one 16-warp
/// block owns 8 tiles across all K channels, V computed on-chip, M exchanged in
/// shared, A^T M A epilogue) but the contraction C is split into 16-channel chunks
/// and the input transform is double-buffered against the GEMM: each iteration
/// transforms chunk N+1's V into one shared buffer while the Tensor Cores accumulate
/// chunk N from the other, so the transform's (high-latency, ~14%-coalesced) global
/// input loads overlap the mma instead of running in a separate serial phase. This
/// is cuDNN's pipelining trick and targets the last ~1.29x gap. Specialized K &lt;= 64.
/// </summary>
internal sealed class PtxWinogradWmmaPipelinedKernel : IDisposable
{
    internal const int Warps = 16;
    internal const int BlockThreads = Warps * 32;   // 512
    internal const int TileP = 8;

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int Batch { get; }
    internal int InputChannels { get; }
    internal int Height { get; }
    internal int Width { get; }
    internal int OutputChannels { get; }
    internal string Ptx { get; }
    internal DirectPtxFunctionInfo FunctionInfo { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal int TileRows => Height / 2;
    internal int TileCols => Width / 2;
    internal int Tiles => Batch * TileRows * TileCols;
    internal int VBufBytes => 16 * TileP * 16 * sizeof(ushort);        // one chunk (16 ch) V buffer
    internal int MBytes => 16 * OutputChannels * TileP * sizeof(float);
    internal int SharedBytes => Math.Max(2 * VBufBytes, MBytes);
    internal long UBytes => (long)16 * OutputChannels * InputChannels * sizeof(ushort);
    internal long InputBytes => (long)Batch * InputChannels * Height * Width * sizeof(float);
    internal long BiasBytes => (long)OutputChannels * sizeof(float);
    internal long OutputBytes => (long)Batch * OutputChannels * Height * Width * sizeof(float);

    internal string EntryPoint => FormattableString.Invariant(
        $"aidotnet_winograd_wmma_pipe_n{Batch}_c{InputChannels}_h{Height}_w{Width}_k{OutputChannels}");

    internal PtxWinogradWmmaPipelinedKernel(
        DirectPtxRuntime runtime, int batch, int inputChannels, int height, int width, int outputChannels)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new NotSupportedException("Pipelined WMMA Winograd has no experimental non-SM86 specialization.");
        if (runtime.ComputeCapabilityMajor < 7)
            throw new NotSupportedException("WMMA Tensor Core PTX requires compute capability 7.0 or newer.");
        if (batch <= 0 || inputChannels <= 0 || height <= 0 || width <= 0 || outputChannels <= 0)
            throw new ArgumentOutOfRangeException(nameof(batch));
        if ((height & 1) != 0 || (width & 1) != 0)
            throw new ArgumentException("Even H and W (whole 2x2 tiling) required.");
        if ((outputChannels & 15) != 0 || outputChannels > 64)
            throw new ArgumentOutOfRangeException(nameof(outputChannels), "K must be a multiple of 16 and <= 64.");
        if ((inputChannels & 15) != 0)
            throw new ArgumentOutOfRangeException(nameof(inputChannels), "C must be a multiple of 16.");
        Batch = batch; InputChannels = inputChannels; Height = height; Width = width; OutputChannels = outputChannels;
        if ((Tiles % TileP) != 0)
            throw new ArgumentException($"P (tile count) must be a multiple of {TileP}.");

        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, batch, inputChannels, height, width, outputChannels);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor,
            batch, inputChannels, height, width, outputChannels);
        _module = runtime.LoadModule(
            Ptx, allowExperimentalJitFallback: DirectPtxFeatureGate.ConvolutionExperimentOverride);
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo functionInfo);
        FunctionInfo = functionInfo;
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, BlockThreads);
        Blueprint.ResourceBudget.Validate(EntryPoint, functionInfo, BlockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Ptx, functionInfo, BlockThreads, activeBlocks, _module);
    }

    internal DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture, int n, int c, int h, int w, int k)
    {
        var u = new DirectPtxExtent(16, k, c);
        var input = new DirectPtxExtent(n, c, h, w);
        var bias = new DirectPtxExtent(k);
        var output = new DirectPtxExtent(n, k, h, w);
        return new DirectPtxKernelBlueprint(
            Operation: "winograd-wmma-pipelined",
            Version: 1,
            Architecture: architecture,
            Variant: FormattableString.Invariant($"n{n}-c{c}-h{h}-w{w}-k{k}-m16n8k16-f16f32-pipelined"),
            Tensors:
            [
                new("U", DirectPtxPhysicalType.Float16, DirectPtxPhysicalLayout.RowMajor2D,
                    u, u, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("input", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Nchw,
                    input, input, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("bias", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    bias, bias, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Nchw,
                    output, output, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 128, MaxStaticSharedBytes: SharedBytes,
                MaxLocalBytesPerThread: 0, MinBlocksPerMultiprocessor: 1),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["equation"] = "relu(A^T (U * (B^T d B)) A + bias)",
                ["pipeline"] = "double-buffered C-chunks: transform(N+1) overlaps GEMM(N)",
                ["global-intermediates"] = "none",
                ["shape-selection"] = "host-only-exact-contract",
                ["promotion"] = "experimental-pending-gpu-evidence"
            });
    }

    internal unsafe void Launch(
        DirectPtxTensorView u, DirectPtxTensorView input, DirectPtxTensorView bias, DirectPtxTensorView output)
    {
        Require(u, Blueprint.Tensors[0], nameof(u));
        Require(input, Blueprint.Tensors[1], nameof(input));
        Require(bias, Blueprint.Tensors[2], nameof(bias));
        Require(output, Blueprint.Tensors[3], nameof(output));
        IntPtr uPtr = u.Pointer, iPtr = input.Pointer, bPtr = bias.Pointer, oPtr = output.Pointer;
        void** args = stackalloc void*[4];
        args[0] = &uPtr; args[1] = &iPtr; args[2] = &bPtr; args[3] = &oPtr;
        _module.Launch(_function, (uint)(Tiles / TileP), 1, 1, BlockThreads, 1, 1, 0, args);
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

    // Emit the input transform for one 16-channel chunk into shared buffer at byte
    // offset bufOff. 128 items (8 tiles x 16 chunk-channels); thread guarded to tid<128.
    private static void EmitTransformChunk(StringBuilder s, int chunk, int bufOff, int h, int w, int c, int tw, int th, int hw)
    {
        string I(int v) => v.ToString(CultureInfo.InvariantCulture);
        int cc = chunk * 16;
        int vTileStride = 16 * sizeof(ushort);       // buffer tile stride (16 ch)
        int vPosStride = TileP * 16 * sizeof(ushort);
        string done = $"TR_DONE_{chunk}", slow = $"TR_SLOW_{chunk}", skip = $"TR_SKIP_{chunk}";
        s.AppendLine($"    setp.ge.u32 %pr0, %r0, 128;");
        s.AppendLine($"    @%pr0 bra {skip};");
        s.AppendLine($"    rem.u32 %r6, %r0, {I(TileP)};");         // tile
        s.AppendLine($"    div.u32 %r7, %r0, {I(TileP)};");         // c_local (0..15)
        s.AppendLine($"    add.u32 %r50, %r7, {I(cc)};");           // global channel
        s.AppendLine($"    add.u32 %r8, %r3, %r6;");                // global tile p
        s.AppendLine($"    rem.u32 %r9, %r8, {I(tw)};");
        s.AppendLine($"    div.u32 %r10, %r8, {I(tw)};");
        s.AppendLine($"    rem.u32 %r11, %r10, {I(th)};");
        s.AppendLine($"    div.u32 %r12, %r10, {I(th)};");
        s.AppendLine("    mul.lo.s32 %r13, %r11, 2;");
        s.AppendLine("    sub.s32 %r13, %r13, 1;");
        s.AppendLine("    mul.lo.s32 %r14, %r9, 2;");
        s.AppendLine("    sub.s32 %r14, %r14, 1;");
        s.AppendLine($"    mad.lo.u32 %r15, %r12, {I(c)}, %r50;");
        s.AppendLine($"    mul.wide.u32 %rd5, %r15, {I(hw)};");
        s.AppendLine("    shl.b64 %rd5, %rd5, 2;");
        s.AppendLine("    add.u64 %rd5, %rd1, %rd5;");
        // interior fast path
        s.AppendLine("    setp.ge.s32 %pr1, %r13, 0;");
        s.AppendLine($"    setp.le.s32 %pr2, %r13, {I(h - 4)};");
        s.AppendLine("    setp.ge.s32 %pr3, %r14, 0;");
        s.AppendLine($"    setp.le.s32 %pr4, %r14, {I(w - 4)};");
        s.AppendLine("    and.pred %pr1, %pr1, %pr2;");
        s.AppendLine("    and.pred %pr3, %pr3, %pr4;");
        s.AppendLine("    and.pred %pr5, %pr1, %pr3;");
        s.AppendLine($"    @!%pr5 bra {slow};");
        s.AppendLine($"    mad.lo.s32 %r16, %r13, {I(w)}, %r14;");
        s.AppendLine("    mul.wide.s32 %rd6, %r16, 4;");
        s.AppendLine("    add.u64 %rd6, %rd5, %rd6;");
        for (int di = 0; di < 4; di++)
            for (int dj = 0; dj < 4; dj++)
                s.AppendLine($"    ld.global.nc.f32 %d{di * 4 + dj}, [%rd6+{I((di * w + dj) * 4)}];");
        s.AppendLine($"    bra {done};");
        s.AppendLine($"{slow}:");
        for (int di = 0; di < 4; di++)
            for (int dj = 0; dj < 4; dj++)
            {
                int reg = di * 4 + dj;
                s.AppendLine($"    mov.f32 %d{reg}, 0f00000000;");
                s.AppendLine($"    add.s32 %r16, %r13, {I(di)};");
                s.AppendLine($"    add.s32 %r17, %r14, {I(dj)};");
                s.AppendLine("    setp.ge.s32 %pr1, %r16, 0;");
                s.AppendLine($"    setp.lt.s32 %pr2, %r16, {I(h)};");
                s.AppendLine("    setp.ge.s32 %pr3, %r17, 0;");
                s.AppendLine($"    setp.lt.s32 %pr4, %r17, {I(w)};");
                s.AppendLine("    and.pred %pr1, %pr1, %pr2;");
                s.AppendLine("    and.pred %pr3, %pr3, %pr4;");
                s.AppendLine("    and.pred %pr1, %pr1, %pr3;");
                s.AppendLine($"    mul.lo.u32 %r18, %r16, {I(w)};");
                s.AppendLine("    add.u32 %r18, %r18, %r17;");
                s.AppendLine("    mul.wide.s32 %rd6, %r18, 4;");
                s.AppendLine("    add.u64 %rd6, %rd5, %rd6;");
                s.AppendLine($"    @%pr1 ld.global.nc.f32 %d{reg}, [%rd6];");
            }
        s.AppendLine($"{done}:");
        int D(int i, int j) => i * 4 + j;
        int T(int i, int j) => 16 + i * 4 + j;
        int Vv(int i, int j) => 32 + i * 4 + j;
        for (int j = 0; j < 4; j++)
        {
            s.AppendLine($"    sub.rn.f32 %d{T(0, j)}, %d{D(0, j)}, %d{D(2, j)};");
            s.AppendLine($"    add.rn.f32 %d{T(1, j)}, %d{D(1, j)}, %d{D(2, j)};");
            s.AppendLine($"    sub.rn.f32 %d{T(2, j)}, %d{D(2, j)}, %d{D(1, j)};");
            s.AppendLine($"    sub.rn.f32 %d{T(3, j)}, %d{D(1, j)}, %d{D(3, j)};");
        }
        for (int i = 0; i < 4; i++)
        {
            s.AppendLine($"    sub.rn.f32 %d{Vv(i, 0)}, %d{T(i, 0)}, %d{T(i, 2)};");
            s.AppendLine($"    add.rn.f32 %d{Vv(i, 1)}, %d{T(i, 1)}, %d{T(i, 2)};");
            s.AppendLine($"    sub.rn.f32 %d{Vv(i, 2)}, %d{T(i, 2)}, %d{T(i, 1)};");
            s.AppendLine($"    sub.rn.f32 %d{Vv(i, 3)}, %d{T(i, 1)}, %d{T(i, 3)};");
        }
        s.AppendLine($"    mul.lo.u32 %r19, %r6, {I(vTileStride)};");
        s.AppendLine("    shl.b32 %r20, %r7, 1;");                  // c_local*2
        s.AppendLine("    add.u32 %r19, %r19, %r20;");
        s.AppendLine($"    add.u32 %r19, %r19, {I(bufOff)};");
        s.AppendLine("    cvt.u64.u32 %rd7, %r19;");
        s.AppendLine("    add.u64 %rd7, %rd4, %rd7;");
        for (int xi = 0; xi < 16; xi++)
        {
            s.AppendLine($"    cvt.rn.f16.f32 %h0, %d{32 + xi};");
            s.AppendLine($"    st.shared.b16 [%rd7+{I(xi * vPosStride)}], %h0;");
        }
        s.AppendLine($"{skip}:");
    }

    // Emit the GEMM accumulate for one chunk from shared buffer at bufOff.
    private static void EmitGemmChunk(StringBuilder s, int chunk, int bufOff, int c, int kc, int nsubK)
    {
        string I(int v) => v.ToString(CultureInfo.InvariantCulture);
        int cc = chunk * 16;
        int vPosStride = TileP * 16 * sizeof(ushort);
        int vTileStride = 16 * sizeof(ushort);
        int chanRowBytes = 16 * c * sizeof(ushort);
        // B base in buffer: rd10 = smem + bufOff + xi*vPosStride + g*vTileStride + (tig*2)*2
        s.AppendLine($"    mul.lo.u32 %r28, %r22, {I(vPosStride)};");
        s.AppendLine($"    mad.lo.u32 %r28, %r23, {I(vTileStride)}, %r28;");
        s.AppendLine("    shl.b32 %r29, %r25, 1;");
        s.AppendLine("    add.u32 %r28, %r28, %r29;");
        s.AppendLine($"    add.u32 %r28, %r28, {I(bufOff)};");
        s.AppendLine("    cvt.u64.u32 %rd10, %r28;");
        s.AppendLine("    add.u64 %rd10, %rd4, %rd10;");
        s.AppendLine("    ld.shared.b32 %b0, [%rd10];");
        s.AppendLine("    ld.shared.b32 %b1, [%rd10+16];");
        for (int msub = 0; msub < nsubK; msub++)
        {
            int aoff = cc * sizeof(ushort) + msub * chanRowBytes;   // U col = chunk*16, row block msub
            s.AppendLine($"    ld.global.b32 %a0, [%rd8+{I(aoff)}];");
            s.AppendLine($"    ld.global.b32 %a2, [%rd8+{I(aoff + 16)}];");
            s.AppendLine($"    ld.global.b32 %a1, [%rd9+{I(aoff)}];");
            s.AppendLine($"    ld.global.b32 %a3, [%rd9+{I(aoff + 16)}];");
            s.AppendLine("    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 " +
                $"{{%c{msub * 4},%c{msub * 4 + 1},%c{msub * 4 + 2},%c{msub * 4 + 3}}}, " +
                "{%a0,%a1,%a2,%a3}, {%b0,%b1}, " +
                $"{{%c{msub * 4},%c{msub * 4 + 1},%c{msub * 4 + 2},%c{msub * 4 + 3}}};");
        }
    }

    internal string EmitPtx(int major, int minor, int n, int c, int h, int w, int k)
    {
        if (!DirectPtxArchitecture.HasExperimentalConvolution(major, minor))
            throw new NotSupportedException("Only the experimental SM86 pipelined WMMA Winograd emitter exists.");
        if ((k & 15) != 0 || (c & 15) != 0 || k > 64) throw new ArgumentException("K mult 16 <=64, C mult 16.");
        string I(int v) => v.ToString(CultureInfo.InvariantCulture);
        int th = h / 2, tw = w / 2;
        int p = n * th * tw;
        int hw = h * w;
        int kc = k * c;
        int nchunks = c / 16;
        int nsubK = k / 16;
        int vbuf = VBufBytes;
        int mPosStride = k * TileP * sizeof(float);
        string entry = EntryPoint;

        var s = new StringBuilder(131072);
        s.AppendLine(".version 7.1");
        s.AppendLine($".target sm_{major}{minor}");
        s.AppendLine(".address_size 64");
        s.AppendLine();
        s.AppendLine($".visible .entry {entry}(");
        s.AppendLine("    .param .u64 u_ptr,");
        s.AppendLine("    .param .u64 input_ptr,");
        s.AppendLine("    .param .u64 bias_ptr,");
        s.AppendLine("    .param .u64 output_ptr");
        s.AppendLine(")");
        s.AppendLine("{");
        s.AppendLine("    .reg .pred %pr<8>;");
        s.AppendLine("    .reg .b16 %h<2>;");
        s.AppendLine("    .reg .b32 %a<4>;");
        s.AppendLine("    .reg .b32 %b<2>;");
        s.AppendLine("    .reg .f32 %c<16>;");
        s.AppendLine("    .reg .f32 %d<48>;");
        s.AppendLine("    .reg .f32 %f<32>;");
        s.AppendLine("    .reg .b32 %r<56>;");
        s.AppendLine("    .reg .b64 %rd<32>;");
        s.AppendLine($"    .shared .align 16 .b8 smem[{Math.Max(2 * vbuf, mPosStride * 16)}];");
        s.AppendLine("    ld.param.u64 %rd0, [u_ptr];");
        s.AppendLine("    ld.param.u64 %rd1, [input_ptr];");
        s.AppendLine("    ld.param.u64 %rd2, [bias_ptr];");
        s.AppendLine("    ld.param.u64 %rd3, [output_ptr];");
        s.AppendLine("    mov.u32 %r0, %tid.x;");
        s.AppendLine("    mov.u32 %r1, %ctaid.x;");
        s.AppendLine("    mul.lo.u32 %r3, %r1, 8;");                // pbase
        s.AppendLine("    mov.u64 %rd4, smem;");
        // GEMM per-warp bases (constant across chunks)
        s.AppendLine("    and.b32 %r21, %r0, 31;");
        s.AppendLine("    shr.u32 %r22, %r0, 5;");                  // warpId = position
        s.AppendLine("    shr.u32 %r23, %r21, 2;");                 // g
        s.AppendLine("    and.b32 %r24, %r21, 3;");                 // tig
        s.AppendLine("    shl.b32 %r25, %r24, 1;");                 // tig*2
        s.AppendLine($"    mad.lo.u32 %r26, %r23, {I(c)}, %r25;");  // g*C + tig*2
        s.AppendLine($"    mad.lo.u32 %r26, %r22, {I(kc)}, %r26;");
        s.AppendLine("    mul.wide.u32 %rd8, %r26, 2;");
        s.AppendLine("    add.u64 %rd8, %rd0, %rd8;");              // A row g base (channel 0)
        s.AppendLine("    add.u32 %r27, %r23, 8;");
        s.AppendLine($"    mad.lo.u32 %r27, %r27, {I(c)}, %r25;");
        s.AppendLine($"    mad.lo.u32 %r27, %r22, {I(kc)}, %r27;");
        s.AppendLine("    mul.wide.u32 %rd9, %r27, 2;");
        s.AppendLine("    add.u64 %rd9, %rd0, %rd9;");
        for (int i = 0; i < 16; i++)
            s.AppendLine($"    mov.f32 %c{i}, 0f00000000;");
        s.AppendLine();

        // ===== software pipeline over C-chunks =====
        s.AppendLine("    // prologue: transform chunk 0 -> buf0");
        EmitTransformChunk(s, 0, 0, h, w, c, tw, th, hw);
        s.AppendLine("    bar.sync 0;");
        for (int chunk = 0; chunk < nchunks; chunk++)
        {
            int curBuf = (chunk % 2) * vbuf;
            s.AppendLine($"    // ---- iter {chunk} : transform(next) overlaps gemm(cur) ----");
            if (chunk + 1 < nchunks)
            {
                int nxtBuf = ((chunk + 1) % 2) * vbuf;
                EmitTransformChunk(s, chunk + 1, nxtBuf, h, w, c, tw, th, hw);
            }
            EmitGemmChunk(s, chunk, curBuf, c, kc, nsubK);
            s.AppendLine("    bar.sync 0;");
        }
        s.AppendLine();

        // ===== phase 3: M exchange + output transform =====
        s.AppendLine($"    mul.lo.u32 %r30, %r22, {I(mPosStride)};");
        s.AppendLine($"    mad.lo.u32 %r31, %r23, {I(TileP * 4)}, %r30;");
        s.AppendLine("    shl.b32 %r32, %r24, 3;");
        s.AppendLine("    add.u32 %r31, %r31, %r32;");
        s.AppendLine("    cvt.u64.u32 %rd11, %r31;");
        s.AppendLine("    add.u64 %rd11, %rd4, %rd11;");
        for (int msub = 0; msub < nsubK; msub++)
        {
            int cbyte = msub * 16 * TileP * 4;
            s.AppendLine($"    st.shared.f32 [%rd11+{I(cbyte)}], %c{msub * 4};");
            s.AppendLine($"    st.shared.f32 [%rd11+{I(cbyte + 4)}], %c{msub * 4 + 1};");
            s.AppendLine($"    st.shared.f32 [%rd11+{I(cbyte + 8 * TileP * 4)}], %c{msub * 4 + 2};");
            s.AppendLine($"    st.shared.f32 [%rd11+{I(cbyte + 8 * TileP * 4 + 4)}], %c{msub * 4 + 3};");
        }
        s.AppendLine("    bar.sync 0;");
        s.AppendLine($"    setp.ge.u32 %pr0, %r0, {I(k * TileP)};");
        s.AppendLine("    @%pr0 bra DONE;");
        s.AppendLine($"    div.u32 %r33, %r0, {I(TileP)};");        // ch
        s.AppendLine($"    rem.u32 %r34, %r0, {I(TileP)};");        // tile
        s.AppendLine($"    mul.lo.u32 %r35, %r33, {I(TileP * 4)};");
        s.AppendLine("    shl.b32 %r36, %r34, 2;");
        s.AppendLine("    add.u32 %r35, %r35, %r36;");
        s.AppendLine("    cvt.u64.u32 %rd12, %r35;");
        s.AppendLine("    add.u64 %rd12, %rd4, %rd12;");
        for (int xi = 0; xi < 16; xi++)
            s.AppendLine($"    ld.shared.f32 %f{I(xi)}, [%rd12+{I(xi * mPosStride)}];");
        for (int j = 0; j < 4; j++)
        {
            int s0 = 16 + j, s1 = 20 + j;
            s.AppendLine($"    add.rn.f32 %f{s0}, %f{I(0 * 4 + j)}, %f{I(1 * 4 + j)};");
            s.AppendLine($"    add.rn.f32 %f{s0}, %f{s0}, %f{I(2 * 4 + j)};");
            s.AppendLine($"    sub.rn.f32 %f{s1}, %f{I(1 * 4 + j)}, %f{I(2 * 4 + j)};");
            s.AppendLine($"    sub.rn.f32 %f{s1}, %f{s1}, %f{I(3 * 4 + j)};");
        }
        for (int i = 0; i < 2; i++)
        {
            int sBase = 16 + i * 4, y0 = 24 + i * 2, y1 = 25 + i * 2;
            s.AppendLine($"    add.rn.f32 %f{y0}, %f{sBase}, %f{sBase + 1};");
            s.AppendLine($"    add.rn.f32 %f{y0}, %f{y0}, %f{sBase + 2};");
            s.AppendLine($"    sub.rn.f32 %f{y1}, %f{sBase + 1}, %f{sBase + 2};");
            s.AppendLine($"    sub.rn.f32 %f{y1}, %f{y1}, %f{sBase + 3};");
        }
        s.AppendLine("    mov.u32 %r37, %r33;");
        s.AppendLine("    add.u32 %r38, %r3, %r34;");
        s.AppendLine("    mul.wide.u32 %rd13, %r37, 4;");
        s.AppendLine("    add.u64 %rd13, %rd2, %rd13;");
        s.AppendLine("    ld.global.nc.f32 %f28, [%rd13];");
        s.AppendLine($"    rem.u32 %r39, %r38, {I(tw)};");
        s.AppendLine($"    div.u32 %r40, %r38, {I(tw)};");
        s.AppendLine($"    rem.u32 %r41, %r40, {I(th)};");
        s.AppendLine($"    div.u32 %r42, %r40, {I(th)};");
        s.AppendLine("    mul.lo.u32 %r43, %r41, 2;");
        s.AppendLine("    mul.lo.u32 %r44, %r39, 2;");
        s.AppendLine($"    mad.lo.u32 %r45, %r42, {I(k)}, %r37;");
        s.AppendLine($"    mul.wide.u32 %rd14, %r45, {I(hw)};");
        s.AppendLine("    shl.b64 %rd14, %rd14, 2;");
        s.AppendLine("    add.u64 %rd14, %rd3, %rd14;");
        for (int oi = 0; oi < 2; oi++)
            for (int oj = 0; oj < 2; oj++)
            {
                int yreg = 24 + oi * 2 + oj;
                s.AppendLine($"    add.rn.f32 %f{yreg}, %f{yreg}, %f28;");
                s.AppendLine($"    max.f32 %f{yreg}, %f{yreg}, 0f00000000;");
                s.AppendLine($"    add.u32 %r46, %r43, {I(oi)};");
                s.AppendLine($"    mul.lo.u32 %r46, %r46, {I(w)};");
                s.AppendLine("    add.u32 %r46, %r46, %r44;");
                s.AppendLine($"    add.u32 %r46, %r46, {I(oj)};");
                s.AppendLine("    mul.wide.u32 %rd15, %r46, 4;");
                s.AppendLine("    add.u64 %rd15, %rd14, %rd15;");
                s.AppendLine($"    st.global.f32 [%rd15], %f{yreg};");
            }
        s.AppendLine("DONE:");
        s.AppendLine("    ret;");
        s.AppendLine("}");
        return s.ToString();
    }

    public void Dispose() => _module.Dispose();
}
