using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Fully-fused FP16 Tensor-Core Winograd F(2,3) 3x3 convolution: input transform,
/// the 16 position GEMMs, and the output transform all in ONE kernel with no global
/// intermediates. A 16-warp block (one Winograd position per warp) first computes
/// V = B^T d B for its 8 output tiles across all channels straight into shared
/// memory (phase 1) -- so the input transform's stores never hit global, killing the
/// uncoalesced V[16,P,C] write that dominated the two-kernel pipeline AND the 51 MB
/// V round-trip. Each warp then runs its position's <c>mma.sync m16n8k16 f16-&gt;f32</c>
/// GEMM reading B fragments from the shared V and A fragments from the precomputed
/// fp16 U (phase 2), exchanges the 16 position results through shared, and applies
/// Y = A^T M A + bias + ReLU in a combined epilogue (phase 3). This is cuDNN's
/// fused structure: input(fp32) and precomputed U(fp16) in, output(fp32) out, nothing
/// else touches global.
/// </summary>
internal sealed class PtxWinogradWmmaFullyFusedKernel : IDisposable
{
    internal const int Warps = 16;
    internal const int BlockThreads = Warps * 32;   // 512
    internal const int TileK = 16;                  // output channels per block
    internal const int TileP = 8;                   // output tiles per block
    internal const int MSharedBytes = 16 * TileK * TileP * sizeof(float);   // 8192

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int Batch { get; }
    internal int InputChannels { get; }   // C
    internal int Height { get; }
    internal int Width { get; }
    internal int OutputChannels { get; }  // K
    internal string Ptx { get; }
    internal DirectPtxFunctionInfo FunctionInfo { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal int TileRows => Height / 2;
    internal int TileCols => Width / 2;
    internal int Tiles => Batch * TileRows * TileCols;   // P
    internal int VSharedBytes => 16 * TileP * InputChannels * sizeof(ushort);
    internal long UBytes => (long)16 * OutputChannels * InputChannels * sizeof(ushort);
    internal long InputBytes => (long)Batch * InputChannels * Height * Width * sizeof(float);
    internal long BiasBytes => (long)OutputChannels * sizeof(float);
    internal long OutputBytes => (long)Batch * OutputChannels * Height * Width * sizeof(float);

    internal string EntryPoint => FormattableString.Invariant(
        $"aidotnet_winograd_wmma_fullfused_n{Batch}_c{InputChannels}_h{Height}_w{Width}_k{OutputChannels}");

    internal PtxWinogradWmmaFullyFusedKernel(
        DirectPtxRuntime runtime, int batch, int inputChannels, int height, int width, int outputChannels)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new NotSupportedException("Fully-fused WMMA Winograd has no experimental non-SM86 specialization.");
        if (runtime.ComputeCapabilityMajor < 7)
            throw new NotSupportedException("WMMA Tensor Core PTX requires compute capability 7.0 or newer.");
        if (batch <= 0 || inputChannels <= 0 || height <= 0 || width <= 0 || outputChannels <= 0)
            throw new ArgumentOutOfRangeException(nameof(batch));
        if ((height & 1) != 0 || (width & 1) != 0)
            throw new ArgumentException("Even H and W (whole 2x2 tiling) required.");
        if ((outputChannels & 15) != 0)
            throw new ArgumentOutOfRangeException(nameof(outputChannels), "K must be a multiple of 16.");
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
            Operation: "winograd-wmma-fully-fused",
            Version: 1,
            Architecture: architecture,
            Variant: FormattableString.Invariant($"n{n}-c{c}-h{h}-w{w}-k{k}-m16n8k16-f16f32-fullfused"),
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
                MaxRegistersPerThread: 96, MaxStaticSharedBytes: VSharedBytes + MSharedBytes,
                MaxLocalBytesPerThread: 0, MinBlocksPerMultiprocessor: 1),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["equation"] = "relu(A^T (U * (B^T d B)) A + bias)",
                ["fusion"] = "input transform + 16 GEMMs + output transform in one kernel",
                ["global-intermediates"] = "none (V computed straight to shared)",
                ["mma"] = "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32, one position per warp",
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
        _module.Launch(
            _function,
            (uint)(Tiles / TileP), (uint)(OutputChannels / TileK), 1,
            BlockThreads, 1, 1,
            0,
            args);
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

    internal string EmitPtx(int major, int minor, int n, int c, int h, int w, int k)
    {
        if (!DirectPtxArchitecture.HasExperimentalConvolution(major, minor))
            throw new NotSupportedException("Only the experimental SM86 fully-fused WMMA Winograd emitter exists.");
        if ((k & 15) != 0 || (c & 15) != 0) throw new ArgumentException("K and C must be multiples of 16.");
        string I(int v) => v.ToString(CultureInfo.InvariantCulture);
        int th = h / 2, tw = w / 2;
        int p = n * th * tw;                 // P
        int hw = h * w;
        int kc = k * c;                      // U position stride (elements)
        int ksteps = c / 16;
        int vPosStride = TileP * c * sizeof(ushort);   // V_shared position stride
        int vTileStride = c * sizeof(ushort);          // V_shared tile stride
        int items = TileP * c;                          // phase-1 (tile,channel) items
        int vBytes = 16 * TileP * c * sizeof(ushort);
        string entry = EntryPoint;

        var s = new StringBuilder(65536);
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
        s.AppendLine("    .reg .f32 %d<48>;");   // phase-1 patch + transform scratch
        s.AppendLine("    .reg .f32 %f<32>;");   // epilogue scratch
        s.AppendLine("    .reg .b32 %r<48>;");
        s.AppendLine("    .reg .b64 %rd<32>;");
        s.AppendLine($"    .shared .align 16 .b8 smem[{vBytes + MSharedBytes}];");
        s.AppendLine("    ld.param.u64 %rd0, [u_ptr];");
        s.AppendLine("    ld.param.u64 %rd1, [input_ptr];");
        s.AppendLine("    ld.param.u64 %rd2, [bias_ptr];");
        s.AppendLine("    ld.param.u64 %rd3, [output_ptr];");
        s.AppendLine("    mov.u32 %r0, %tid.x;");
        s.AppendLine("    mov.u32 %r1, %ctaid.x;");                 // p-block (x8)
        s.AppendLine("    mov.u32 %r2, %ctaid.y;");                 // k-block (x16)
        s.AppendLine("    mul.lo.u32 %r3, %r1, 8;");                // pbase
        s.AppendLine("    mul.lo.u32 %r4, %r2, 16;");               // kbase
        s.AppendLine("    mov.u64 %rd4, smem;");                    // V_shared base
        s.AppendLine();

        // ================= PHASE 1: input transform -> V_shared =================
        // Each iteration: item covers one (tile, channel). tile = item / C, c = item % C.
        // Load 4x4 patch, V = B^T d B, store 16 positions to V_shared[pos][tile][c].
        s.AppendLine("    // ---- phase 1: on-chip input transform ----");
        s.AppendLine("    mov.u32 %r5, %r0;");                      // item = tid
        s.AppendLine("P1_LOOP:");
        s.AppendLine($"    setp.ge.u32 %pr0, %r5, {I(items)};");
        s.AppendLine("    @%pr0 bra P1_DONE;");
        s.AppendLine($"    div.u32 %r6, %r5, {I(c)};");             // tile (0..7)
        s.AppendLine($"    rem.u32 %r7, %r5, {I(c)};");             // channel c
        s.AppendLine($"    add.u32 %r8, %r3, %r6;");                // global tile p = pbase + tile
        s.AppendLine($"    rem.u32 %r9, %r8, {I(tw)};");            // tj
        s.AppendLine($"    div.u32 %r10, %r8, {I(tw)};");
        s.AppendLine($"    rem.u32 %r11, %r10, {I(th)};");          // ti
        s.AppendLine($"    div.u32 %r12, %r10, {I(th)};");          // nn
        s.AppendLine("    mul.lo.s32 %r13, %r11, 2;");
        s.AppendLine("    sub.s32 %r13, %r13, 1;");                 // ih0 = 2*ti - 1
        s.AppendLine("    mul.lo.s32 %r14, %r9, 2;");
        s.AppendLine("    sub.s32 %r14, %r14, 1;");                 // iw0 = 2*tj - 1
        s.AppendLine($"    mad.lo.u32 %r15, %r12, {I(c)}, %r7;");   // nn*C + c
        s.AppendLine($"    mul.wide.u32 %rd5, %r15, {I(hw)};");
        s.AppendLine("    shl.b64 %rd5, %rd5, 2;");
        s.AppendLine("    add.u64 %rd5, %rd1, %rd5;");              // input channel base
        // load 4x4 patch d -> %d0..15 (zero padded)
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
        // t = B^T d (%d16..31), V = t B (%d32..47)
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
        // store 16 positions to V_shared[pos][tile][c] = smem + pos*vPosStride + tile*vTileStride + c*2
        s.AppendLine($"    mul.lo.u32 %r19, %r6, {I(vTileStride)};");   // tile*vTileStride
        s.AppendLine("    shl.b32 %r20, %r7, 1;");                       // c*2
        s.AppendLine("    add.u32 %r19, %r19, %r20;");
        s.AppendLine("    cvt.u64.u32 %rd7, %r19;");
        s.AppendLine("    add.u64 %rd7, %rd4, %rd7;");                   // &V_shared[0][tile][c]
        for (int xi = 0; xi < 16; xi++)
        {
            s.AppendLine($"    cvt.rn.f16.f32 %h0, %d{32 + xi};");
            s.AppendLine($"    st.shared.b16 [%rd7+{I(xi * vPosStride)}], %h0;");
        }
        s.AppendLine($"    add.u32 %r5, %r5, {I(BlockThreads)};");
        s.AppendLine("    bra P1_LOOP;");
        s.AppendLine("P1_DONE:");
        s.AppendLine("    bar.sync 0;");
        s.AppendLine();

        // ================= PHASE 2: per-warp position GEMM =================
        // warp = position xi; lane -> g (row/tile within block), tig.
        s.AppendLine("    // ---- phase 2: position GEMM (A=U global, B=V shared) ----");
        s.AppendLine("    and.b32 %r21, %r0, 31;");                 // lane
        s.AppendLine("    shr.u32 %r22, %r0, 5;");                  // warpId = position xi
        s.AppendLine("    shr.u32 %r23, %r21, 2;");                 // g (0..7)
        s.AppendLine("    and.b32 %r24, %r21, 3;");                 // tig
        s.AppendLine("    shl.b32 %r25, %r24, 1;");                 // tig*2
        // A (U) fragment bases from global: rd8 = u + (xi*KC + (kbase+g)*C + tig*2)*2 ; rd9 (kbase+g+8)
        s.AppendLine("    add.u32 %r26, %r4, %r23;");               // kbase+g
        s.AppendLine($"    mad.lo.u32 %r26, %r26, {I(c)}, %r25;");
        s.AppendLine($"    mad.lo.u32 %r26, %r22, {I(kc)}, %r26;");
        s.AppendLine("    mul.wide.u32 %rd8, %r26, 2;");
        s.AppendLine("    add.u64 %rd8, %rd0, %rd8;");
        s.AppendLine("    add.u32 %r27, %r4, %r23;");
        s.AppendLine("    add.u32 %r27, %r27, 8;");
        s.AppendLine($"    mad.lo.u32 %r27, %r27, {I(c)}, %r25;");
        s.AppendLine($"    mad.lo.u32 %r27, %r22, {I(kc)}, %r27;");
        s.AppendLine("    mul.wide.u32 %rd9, %r27, 2;");
        s.AppendLine("    add.u64 %rd9, %rd0, %rd9;");
        // B (V) fragment base in shared: rd10 = smem + xi*vPosStride + g*vTileStride + (tig*2)*2
        s.AppendLine($"    mul.lo.u32 %r28, %r22, {I(vPosStride)};");
        s.AppendLine($"    mad.lo.u32 %r28, %r23, {I(vTileStride)}, %r28;");
        s.AppendLine("    shl.b32 %r29, %r25, 1;");                 // (tig*2)*2
        s.AppendLine("    add.u32 %r28, %r28, %r29;");
        s.AppendLine("    cvt.u64.u32 %rd10, %r28;");
        s.AppendLine("    add.u64 %rd10, %rd4, %rd10;");
        for (int i = 0; i < 16; i++)
            s.AppendLine($"    mov.f32 %c{i}, 0f00000000;");
        // NB: only the first mma tile column set (4 accs) is used per warp (16ch x 8tiles).
        for (int ks = 0; ks < ksteps; ks++)
        {
            int off = ks * 32;                 // U global k-step (bytes)
            int voff = ks * 16 * sizeof(ushort);  // V shared k-step (c advances 16) -> 32 bytes
            s.AppendLine($"    ld.global.b32 %a0, [%rd8+{I(off)}];");
            s.AppendLine($"    ld.global.b32 %a2, [%rd8+{I(off + 16)}];");
            s.AppendLine($"    ld.global.b32 %a1, [%rd9+{I(off)}];");
            s.AppendLine($"    ld.global.b32 %a3, [%rd9+{I(off + 16)}];");
            s.AppendLine($"    ld.shared.b32 %b0, [%rd10+{I(voff)}];");
            s.AppendLine($"    ld.shared.b32 %b1, [%rd10+{I(voff + 16)}];");
            s.AppendLine("    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 " +
                "{%c0,%c1,%c2,%c3}, {%a0,%a1,%a2,%a3}, {%b0,%b1}, {%c0,%c1,%c2,%c3};");
        }
        s.AppendLine("    bar.sync 0;");   // V_shared no longer needed; safe to reuse region for M
        s.AppendLine();

        // ================= PHASE 3: M exchange + output transform =================
        s.AppendLine("    // ---- phase 3: exchange M in shared + output transform ----");
        // M_shared at offset vBytes: M_s[pos][ch][tile] = vBytes + pos*512 + ch*32 + tile*4.
        s.AppendLine($"    shl.b32 %r30, %r22, 9;");                // warpId*512
        s.AppendLine("    shl.b32 %r31, %r23, 5;");                 // g*32
        s.AppendLine("    shl.b32 %r32, %r24, 3;");                 // tig*8
        s.AppendLine("    add.u32 %r30, %r30, %r31;");
        s.AppendLine("    add.u32 %r30, %r30, %r32;");
        s.AppendLine($"    add.u32 %r30, %r30, {I(vBytes)};");
        s.AppendLine("    cvt.u64.u32 %rd11, %r30;");
        s.AppendLine("    add.u64 %rd11, %rd4, %rd11;");
        s.AppendLine("    st.shared.f32 [%rd11], %c0;");
        s.AppendLine("    st.shared.f32 [%rd11+4], %c1;");
        s.AppendLine("    st.shared.f32 [%rd11+256], %c2;");
        s.AppendLine("    st.shared.f32 [%rd11+260], %c3;");
        s.AppendLine("    bar.sync 0;");
        // Epilogue: 128 threads (tid<128) combine 16 positions per (ch,tile).
        s.AppendLine("    setp.ge.u32 %pr0, %r0, 128;");
        s.AppendLine("    @%pr0 bra DONE;");
        s.AppendLine("    shr.u32 %r33, %r0, 3;");                  // ch = tid>>3
        s.AppendLine("    and.b32 %r34, %r0, 7;");                  // tile = tid&7
        s.AppendLine("    shl.b32 %r35, %r33, 5;");                 // ch*32
        s.AppendLine("    shl.b32 %r36, %r34, 2;");                 // tile*4
        s.AppendLine("    add.u32 %r35, %r35, %r36;");
        s.AppendLine($"    add.u32 %r35, %r35, {I(vBytes)};");
        s.AppendLine("    cvt.u64.u32 %rd12, %r35;");
        s.AppendLine("    add.u64 %rd12, %rd4, %rd12;");            // &M_s[0][ch][tile]
        for (int xi = 0; xi < 16; xi++)
            s.AppendLine($"    ld.shared.f32 %f{I(xi)}, [%rd12+{I(xi * 512)}];");
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
        s.AppendLine("    add.u32 %r37, %r4, %r33;");                // k = kbase + ch
        s.AppendLine("    add.u32 %r38, %r3, %r34;");                // p = pbase + tile
        s.AppendLine("    mul.wide.u32 %rd13, %r37, 4;");
        s.AppendLine("    add.u64 %rd13, %rd2, %rd13;");
        s.AppendLine("    ld.global.nc.f32 %f28, [%rd13];");         // bias[k]
        s.AppendLine($"    rem.u32 %r39, %r38, {I(tw)};");           // tj
        s.AppendLine($"    div.u32 %r40, %r38, {I(tw)};");
        s.AppendLine($"    rem.u32 %r41, %r40, {I(th)};");           // ti
        s.AppendLine($"    div.u32 %r42, %r40, {I(th)};");           // nn
        s.AppendLine("    mul.lo.u32 %r43, %r41, 2;");               // oh0
        s.AppendLine("    mul.lo.u32 %r44, %r39, 2;");               // ow0
        s.AppendLine($"    mad.lo.u32 %r45, %r42, {I(k)}, %r37;");   // nn*K + k
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
