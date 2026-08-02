using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// cuDNN-class fully-fused FP16 Tensor-Core Winograd F(2,3) 3x3 convolution. A
/// 4-warp block computes a 16(output-channel) x 32(output-tile) block for all 16
/// Winograd positions. Each k-step, all 128 threads stage the U[16,16,16] and
/// V[16,32,16] fp16 operand tiles into shared with coalesced 16-byte
/// <c>cp.async</c> loads (fixing the ~12.5%-coalesced direct-global fragment loads
/// of the first fused kernel), then every warp reads its mma fragments from shared
/// with <c>ld.shared</c> and runs 16 raw <c>mma.sync m16n8k16 f16-&gt;f32</c>
/// accumulations (one per position) into registers. The staged U tile is reused by
/// all four warps. As in the register-only fused kernel, the m16n8k16 D-fragment
/// has a defined layout, so the output transform Y = A^T M A + bias + ReLU runs
/// thread-locally in the epilogue and the 2x2 tiles scatter straight to
/// output[N,K,H,W] -- M never touches global. This is the coalesced-staging +
/// multi-warp escalation of <see cref="PtxWinogradWmmaFusedKernel"/>.
/// </summary>
internal sealed class PtxWinogradWmmaFusedStagedKernel : IDisposable
{
    internal const int Warps = 4;
    internal const int BlockThreads = Warps * 32;   // 128
    internal const int TileK = 16;                  // output channels per block
    internal const int TileP = 32;                  // output tiles per block (4 warps x 8)
    internal const int USharedBytes = 16 * TileK * 16 * sizeof(ushort);   // 8192
    internal const int VSharedBytes = 16 * TileP * 16 * sizeof(ushort);   // 16384

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
    internal long UBytes => (long)16 * OutputChannels * InputChannels * sizeof(ushort);
    internal long VBytes => (long)16 * Tiles * InputChannels * sizeof(ushort);
    internal long BiasBytes => (long)OutputChannels * sizeof(float);
    internal long OutputBytes => (long)Batch * OutputChannels * Height * Width * sizeof(float);

    internal string EntryPoint => FormattableString.Invariant(
        $"aidotnet_winograd_wmma_fused_staged_n{Batch}_c{InputChannels}_h{Height}_w{Width}_k{OutputChannels}");

    internal PtxWinogradWmmaFusedStagedKernel(
        DirectPtxRuntime runtime, int batch, int inputChannels, int height, int width, int outputChannels)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new NotSupportedException("Staged fused WMMA Winograd has no experimental non-SM86 specialization.");
        if (runtime.ComputeCapabilityMajor < 8)
            throw new NotSupportedException("cp.async staged Tensor Core PTX requires compute capability 8.0 or newer.");
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

    internal static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture, int n, int c, int h, int w, int k)
    {
        int tiles = n * (h / 2) * (w / 2);
        var u = new DirectPtxExtent(16, k, c);
        var v = new DirectPtxExtent(16, tiles, c);
        var bias = new DirectPtxExtent(k);
        var output = new DirectPtxExtent(n, k, h, w);
        return new DirectPtxKernelBlueprint(
            Operation: "winograd-wmma-fused-staged",
            Version: 1,
            Architecture: architecture,
            Variant: FormattableString.Invariant($"n{n}-c{c}-h{h}-w{w}-k{k}-m16n8k16-f16f32-cpasync-4warp"),
            Tensors:
            [
                new("U", DirectPtxPhysicalType.Float16, DirectPtxPhysicalLayout.RowMajor2D,
                    u, u, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("V", DirectPtxPhysicalType.Float16, DirectPtxPhysicalLayout.RowMajor2D,
                    v, v, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("bias", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    bias, bias, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Nchw,
                    output, output, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 128, MaxStaticSharedBytes: USharedBytes + VSharedBytes,
                MaxLocalBytesPerThread: 0, MinBlocksPerMultiprocessor: 1),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["equation"] = "relu(A^T (U*V) A + bias)",
                ["mma"] = "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 x16 positions",
                ["staging"] = "coalesced 16B cp.async U[16,16,16]+V[16,32,16] to shared, ld.shared fragments",
                ["fusion"] = "16 position GEMMs + output transform in registers; no M workspace",
                ["global-intermediates"] = "none",
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
        void** args = stackalloc void*[4];
        args[0] = &uPtr; args[1] = &vPtr; args[2] = &bPtr; args[3] = &oPtr;
        // grid: (P/32 tile-blocks, K/16 channel-blocks, 1); 4 warps per block.
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

    internal static string EmitPtx(int major, int minor, int n, int c, int h, int w, int k)
    {
        if (!DirectPtxArchitecture.HasExperimentalConvolution(major, minor))
            throw new NotSupportedException("Only the experimental SM86 staged fused WMMA Winograd emitter exists.");
        if ((k & 15) != 0 || (c & 15) != 0) throw new ArgumentException("K and C must be multiples of 16.");
        string I(int v) => v.ToString(CultureInfo.InvariantCulture);
        int th = h / 2, tw = w / 2;
        int p = n * th * tw;                 // P
        int hw = h * w;
        int kc = k * c;                      // U position stride (elements)
        int pc = p * c;                      // V position stride (elements)
        int ksteps = c / 16;
        int accReg(int xi, int e) => xi * 4 + e;
        string entry = FormattableString.Invariant(
            $"aidotnet_winograd_wmma_fused_staged_n{n}_c{c}_h{h}_w{w}_k{k}");

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
        s.AppendLine("    .reg .b32 %a<4>;");
        s.AppendLine("    .reg .b32 %b<2>;");
        s.AppendLine("    .reg .f32 %c<64>;");
        s.AppendLine("    .reg .f32 %f<16>;");
        s.AppendLine("    .reg .b32 %r<48>;");
        s.AppendLine("    .reg .b64 %rd<32>;");
        s.AppendLine($"    .shared .align 16 .b8 smem[{USharedBytes + VSharedBytes}];");
        s.AppendLine("    ld.param.u64 %rd0, [u_ptr];");
        s.AppendLine("    ld.param.u64 %rd1, [v_ptr];");
        s.AppendLine("    ld.param.u64 %rd2, [bias_ptr];");
        s.AppendLine("    ld.param.u64 %rd3, [output_ptr];");
        s.AppendLine("    mov.u32 %r0, %tid.x;");
        s.AppendLine("    mov.u32 %r1, %ctaid.x;");                 // p-block (x32)
        s.AppendLine("    mov.u32 %r2, %ctaid.y;");                 // k-block (x16)
        s.AppendLine("    and.b32 %r3, %r0, 31;");                  // lane
        s.AppendLine("    shr.u32 %r4, %r0, 5;");                   // warpId (0..3)
        s.AppendLine("    shr.u32 %r5, %r3, 2;");                   // groupID = lane>>2 (0..7)
        s.AppendLine("    and.b32 %r6, %r3, 3;");                   // tig = lane&3 (0..3)
        s.AppendLine("    mul.lo.u32 %r7, %r2, 16;");               // kbase
        s.AppendLine("    mul.lo.u32 %r8, %r1, 32;");               // pblock
        s.AppendLine("    shl.b32 %r9, %r5, 5;");                   // groupID*32
        s.AppendLine("    shl.b32 %r10, %r6, 2;");                  // tig*4
        s.AppendLine("    add.u32 %r11, %r9, %r10;");               // fragOffU = groupID*32 + tig*4
        s.AppendLine("    shl.b32 %r12, %r4, 8;");                  // warpId*256
        s.AppendLine("    add.u32 %r13, %r12, %r11;");              // fragOffV = warpId*256 + fragOffU
        s.AppendLine("    mov.u64 %rd4, smem;");
        // shared fragment base pointers (position stride added per-pos as immediate)
        s.AppendLine("    cvt.u64.u32 %rd5, %r11;");
        s.AppendLine("    add.u64 %rd5, %rd4, %rd5;");              // U_s frag base = smem + fragOffU
        s.AppendLine("    cvt.u64.u32 %rd6, %r13;");
        s.AppendLine($"    add.u64 %rd6, %rd4, %rd6;");
        s.AppendLine($"    add.u64 %rd6, %rd6, {I(USharedBytes)};"); // V_s frag base = smem + Ushared + fragOffV
        s.AppendLine();
        for (int i = 0; i < 64; i++)
            s.AppendLine($"    mov.f32 %c{i}, 0f00000000;");

        for (int ks = 0; ks < ksteps; ks++)
        {
            int cc = ks * 16;
            s.AppendLine();
            s.AppendLine($"    // ==== k-step {ks} (cc={cc}) : coalesced cp.async staging ====");
            // Stage U: 512 16-byte chunks, 4 per thread. idx = chunk*128 + tid.
            //   half=idx&1, kr=(idx>>1)&15, pos=idx>>5.
            //   src = u + (pos*K*C + (kbase+kr)*C + cc + half*8)*2
            //   dst = smem + pos*512 + kr*32 + half*16
            for (int chunk = 0; chunk < 4; chunk++)
            {
                s.AppendLine($"    add.u32 %r14, %r0, {I(chunk * 128)};");   // idx
                s.AppendLine("    and.b32 %r15, %r14, 1;");                  // half
                s.AppendLine("    shr.u32 %r16, %r14, 1;");
                s.AppendLine("    and.b32 %r17, %r16, 15;");                 // kr
                s.AppendLine("    shr.u32 %r18, %r14, 5;");                  // pos
                // src element index = pos*K*C + (kbase+kr)*C + cc + half*8
                s.AppendLine("    add.u32 %r19, %r7, %r17;");                // kbase+kr
                s.AppendLine($"    mad.lo.u32 %r19, %r19, {I(c)}, {I(cc)};");// (kbase+kr)*C + cc
                s.AppendLine($"    mad.lo.u32 %r19, %r18, {I(kc)}, %r19;");  // + pos*K*C
                s.AppendLine("    shl.b32 %r20, %r15, 3;");                  // half*8
                s.AppendLine("    add.u32 %r19, %r19, %r20;");
                s.AppendLine("    mul.wide.u32 %rd7, %r19, 2;");
                s.AppendLine("    add.u64 %rd7, %rd0, %rd7;");               // src
                // dst = smem + pos*512 + kr*32 + half*16
                s.AppendLine("    shl.b32 %r21, %r18, 9;");                  // pos*512
                s.AppendLine("    shl.b32 %r22, %r17, 5;");                  // kr*32
                s.AppendLine("    add.u32 %r21, %r21, %r22;");
                s.AppendLine("    shl.b32 %r22, %r15, 4;");                  // half*16
                s.AppendLine("    add.u32 %r21, %r21, %r22;");
                s.AppendLine("    cvt.u64.u32 %rd8, %r21;");
                s.AppendLine("    add.u64 %rd8, %rd4, %rd8;");               // dst
                s.AppendLine("    cp.async.cg.shared.global [%rd8], [%rd7], 16;");
            }
            // Stage V: 1024 16-byte chunks, 8 per thread. idx = chunk*128 + tid.
            //   half=idx&1, pr=(idx>>1)&31, pos=idx>>6.
            //   src = v + (pos*P*C + (pblock+pr)*C + cc + half*8)*2
            //   dst = smem + Ushared + pos*1024 + pr*32 + half*16
            for (int chunk = 0; chunk < 8; chunk++)
            {
                s.AppendLine($"    add.u32 %r14, %r0, {I(chunk * 128)};");   // idx
                s.AppendLine("    and.b32 %r15, %r14, 1;");                  // half
                s.AppendLine("    shr.u32 %r16, %r14, 1;");
                s.AppendLine("    and.b32 %r17, %r16, 31;");                 // pr
                s.AppendLine("    shr.u32 %r18, %r14, 6;");                  // pos
                s.AppendLine("    add.u32 %r19, %r8, %r17;");                // pblock+pr
                s.AppendLine($"    mad.lo.u32 %r19, %r19, {I(c)}, {I(cc)};");// (pblock+pr)*C + cc
                s.AppendLine($"    mad.lo.u32 %r19, %r18, {I(pc)}, %r19;");  // + pos*P*C
                s.AppendLine("    shl.b32 %r20, %r15, 3;");                  // half*8
                s.AppendLine("    add.u32 %r19, %r19, %r20;");
                s.AppendLine("    mul.wide.u32 %rd7, %r19, 2;");
                s.AppendLine("    add.u64 %rd7, %rd1, %rd7;");               // src
                s.AppendLine("    shl.b32 %r21, %r18, 10;");                 // pos*1024
                s.AppendLine("    shl.b32 %r22, %r17, 5;");                  // pr*32
                s.AppendLine("    add.u32 %r21, %r21, %r22;");
                s.AppendLine("    shl.b32 %r22, %r15, 4;");                  // half*16
                s.AppendLine("    add.u32 %r21, %r21, %r22;");
                s.AppendLine($"    add.u32 %r21, %r21, {I(USharedBytes)};");
                s.AppendLine("    cvt.u64.u32 %rd8, %r21;");
                s.AppendLine("    add.u64 %rd8, %rd4, %rd8;");               // dst
                s.AppendLine("    cp.async.cg.shared.global [%rd8], [%rd7], 16;");
            }
            s.AppendLine("    cp.async.commit_group;");
            s.AppendLine("    cp.async.wait_group 0;");
            s.AppendLine("    bar.sync 0;");
            // mma: per position, ld.shared fragments + accumulate.
            for (int xi = 0; xi < 16; xi++)
            {
                s.AppendLine($"    add.u64 %rd9, %rd5, {I(xi * 512)};");     // U_s frag for pos xi
                s.AppendLine($"    add.u64 %rd10, %rd6, {I(xi * 1024)};");   // V_s frag for pos xi
                s.AppendLine("    ld.shared.b32 %a0, [%rd9];");
                s.AppendLine("    ld.shared.b32 %a2, [%rd9+16];");
                s.AppendLine("    ld.shared.b32 %a1, [%rd9+256];");
                s.AppendLine("    ld.shared.b32 %a3, [%rd9+272];");
                s.AppendLine("    ld.shared.b32 %b0, [%rd10];");
                s.AppendLine("    ld.shared.b32 %b1, [%rd10+16];");
                s.AppendLine("    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 " +
                    $"{{%c{accReg(xi, 0)},%c{accReg(xi, 1)},%c{accReg(xi, 2)},%c{accReg(xi, 3)}}}, " +
                    "{%a0,%a1,%a2,%a3}, {%b0,%b1}, " +
                    $"{{%c{accReg(xi, 0)},%c{accReg(xi, 1)},%c{accReg(xi, 2)},%c{accReg(xi, 3)}}};");
            }
            if (ks + 1 < ksteps)
                s.AppendLine("    bar.sync 0;");   // guard shared before next stage overwrites it
        }

        // Epilogue: each thread owns 4 outputs (e=0..3), same mapping as the register-only
        // fused kernel but p = pblock + warpId*8 + tig*2 + (e&1), k = kbase + {groupID, groupID+8}.
        s.AppendLine();
        s.AppendLine("    mul.lo.u32 %r23, %r4, 8;");               // warpId*8
        s.AppendLine("    add.u32 %r23, %r8, %r23;");               // pWarpBase = pblock + warpId*8
        s.AppendLine("    add.u32 %r24, %r7, %r5;");                // kbase + groupID
        s.AppendLine("    add.u32 %r25, %r24, 8;");                 // kbase + groupID + 8
        s.AppendLine("    shl.b32 %r26, %r6, 1;");                  // tig*2
        for (int e = 0; e < 4; e++)
        {
            s.AppendLine();
            s.AppendLine($"    // ---- epilogue output {e} ----");
            int Macc(int i, int j) => accReg(i * 4 + j, e);
            int Sreg(int i, int j) => i * 4 + j;      // %f0..7
            for (int j = 0; j < 4; j++)
            {
                s.AppendLine($"    add.rn.f32 %f{Sreg(0, j)}, %c{Macc(0, j)}, %c{Macc(1, j)};");
                s.AppendLine($"    add.rn.f32 %f{Sreg(0, j)}, %f{Sreg(0, j)}, %c{Macc(2, j)};");
                s.AppendLine($"    sub.rn.f32 %f{Sreg(1, j)}, %c{Macc(1, j)}, %c{Macc(2, j)};");
                s.AppendLine($"    sub.rn.f32 %f{Sreg(1, j)}, %f{Sreg(1, j)}, %c{Macc(3, j)};");
            }
            for (int i = 0; i < 2; i++)
            {
                int y0 = 8 + i * 2, y1 = 9 + i * 2;
                s.AppendLine($"    add.rn.f32 %f{y0}, %f{Sreg(i, 0)}, %f{Sreg(i, 1)};");
                s.AppendLine($"    add.rn.f32 %f{y0}, %f{y0}, %f{Sreg(i, 2)};");
                s.AppendLine($"    sub.rn.f32 %f{y1}, %f{Sreg(i, 1)}, %f{Sreg(i, 2)};");
                s.AppendLine($"    sub.rn.f32 %f{y1}, %f{y1}, %f{Sreg(i, 3)};");
            }
            string kThis = e < 2 ? "%r24" : "%r25";
            s.AppendLine($"    add.u32 %r27, %r23, %r26;");         // p = pWarpBase + tig*2
            if ((e & 1) == 1) s.AppendLine("    add.u32 %r27, %r27, 1;");
            s.AppendLine($"    mul.wide.u32 %rd11, {kThis}, 4;");
            s.AppendLine("    add.u64 %rd11, %rd2, %rd11;");
            s.AppendLine("    ld.global.nc.f32 %f12, [%rd11];");    // bias[k]
            s.AppendLine($"    rem.u32 %r28, %r27, {I(tw)};");      // tj
            s.AppendLine($"    div.u32 %r29, %r27, {I(tw)};");
            s.AppendLine($"    rem.u32 %r30, %r29, {I(th)};");      // ti
            s.AppendLine($"    div.u32 %r31, %r29, {I(th)};");      // nn
            s.AppendLine("    mul.lo.u32 %r32, %r30, 2;");          // oh0
            s.AppendLine("    mul.lo.u32 %r33, %r28, 2;");          // ow0
            s.AppendLine($"    mad.lo.u32 %r34, %r31, {I(k)}, {kThis};");   // nn*K + k
            s.AppendLine($"    mul.wide.u32 %rd12, %r34, {I(hw)};");
            s.AppendLine("    shl.b64 %rd12, %rd12, 2;");
            s.AppendLine("    add.u64 %rd12, %rd3, %rd12;");        // &out[nn][k][0][0]
            for (int oi = 0; oi < 2; oi++)
                for (int oj = 0; oj < 2; oj++)
                {
                    int yreg = 8 + oi * 2 + oj;
                    s.AppendLine($"    add.rn.f32 %f{yreg}, %f{yreg}, %f12;");
                    s.AppendLine($"    max.f32 %f{yreg}, %f{yreg}, 0f00000000;");
                    s.AppendLine($"    add.u32 %r35, %r32, {I(oi)};");
                    s.AppendLine($"    mul.lo.u32 %r35, %r35, {I(w)};");
                    s.AppendLine("    add.u32 %r35, %r35, %r33;");
                    s.AppendLine($"    add.u32 %r35, %r35, {I(oj)};");
                    s.AppendLine("    mul.wide.u32 %rd13, %r35, 4;");
                    s.AppendLine("    add.u64 %rd13, %rd12, %rd13;");
                    s.AppendLine($"    st.global.f32 [%rd13], %f{yreg};");
                }
        }
        s.AppendLine("    ret;");
        s.AppendLine("}");
        return s.ToString();
    }

    public void Dispose() => _module.Dispose();
}
