using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Cooperative FP16 Tensor-Core Winograd F(2,3) 3x3 convolution that attacks the
/// 16-position occupancy wall directly. A 16-warp block assigns ONE Winograd
/// position to each warp: every warp runs a single raw <c>mma.sync m16n8k16
/// f16-&gt;f32</c> accumulation for its position over a 16(channel) x 8(tile) block,
/// so each thread holds only 4 accumulators -- full occupancy, unlike the
/// single-warp fused kernels that pin all 16 positions (64-128 registers) in one
/// thread and starve the Tensor Cores. The 16 position results M[xi] are then
/// exchanged through shared memory (8 KB, not the ~98 MB global round-trip), and a
/// combined epilogue pass reads all 16 positions for each (channel, tile) and
/// applies Y = A^T M A + bias + ReLU, scattering the 2x2 tiles to output[N,K,H,W].
/// This keeps both the Tensor-Core GEMM throughput and high occupancy.
/// </summary>
internal sealed class PtxWinogradWmmaCoopKernel : IDisposable
{
    internal const int Warps = 16;                 // one per Winograd position
    internal const int BlockThreads = Warps * 32;  // 512
    internal const int TileK = 16;                 // output channels per block
    internal const int TileP = 8;                  // output tiles per block
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
    internal long UBytes => (long)16 * OutputChannels * InputChannels * sizeof(ushort);
    internal long VBytes => (long)16 * Tiles * InputChannels * sizeof(ushort);
    internal long BiasBytes => (long)OutputChannels * sizeof(float);
    internal long OutputBytes => (long)Batch * OutputChannels * Height * Width * sizeof(float);

    internal string EntryPoint => FormattableString.Invariant(
        $"aidotnet_winograd_wmma_coop_n{Batch}_c{InputChannels}_h{Height}_w{Width}_k{OutputChannels}");

    internal PtxWinogradWmmaCoopKernel(
        DirectPtxRuntime runtime, int batch, int inputChannels, int height, int width, int outputChannels)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new NotSupportedException("Cooperative WMMA Winograd has no experimental non-SM86 specialization.");
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

    internal static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture, int n, int c, int h, int w, int k)
    {
        int tiles = n * (h / 2) * (w / 2);
        var u = new DirectPtxExtent(16, k, c);
        var v = new DirectPtxExtent(16, tiles, c);
        var bias = new DirectPtxExtent(k);
        var output = new DirectPtxExtent(n, k, h, w);
        return new DirectPtxKernelBlueprint(
            Operation: "winograd-wmma-coop",
            Version: 1,
            Architecture: architecture,
            Variant: FormattableString.Invariant($"n{n}-c{c}-h{h}-w{w}-k{k}-m16n8k16-f16f32-coop16warp"),
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
                MaxRegistersPerThread: 64, MaxStaticSharedBytes: MSharedBytes,
                MaxLocalBytesPerThread: 0, MinBlocksPerMultiprocessor: 1),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["equation"] = "relu(A^T (U*V) A + bias)",
                ["mma"] = "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32, one position per warp",
                ["exchange"] = "16 position results M[xi] via shared (8KB), combined output transform",
                ["occupancy"] = "4 accumulators/thread -> full occupancy (escapes 16-position register wall)",
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
        // grid: (P/8 tile-blocks, K/16 channel-blocks, 1); 16 warps per block.
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
            throw new NotSupportedException("Only the experimental SM86 cooperative WMMA Winograd emitter exists.");
        if ((k & 15) != 0 || (c & 15) != 0) throw new ArgumentException("K and C must be multiples of 16.");
        string I(int v) => v.ToString(CultureInfo.InvariantCulture);
        int th = h / 2, tw = w / 2;
        int p = n * th * tw;                 // P
        int hw = h * w;
        int kc = k * c;                      // U position stride (elements)
        int pc = p * c;                      // V position stride (elements)
        int ksteps = c / 16;
        string entry = FormattableString.Invariant(
            $"aidotnet_winograd_wmma_coop_n{n}_c{c}_h{h}_w{w}_k{k}");

        var s = new StringBuilder(32768);
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
        s.AppendLine("    .reg .pred %pr<2>;");
        s.AppendLine("    .reg .b32 %a<4>;");
        s.AppendLine("    .reg .b32 %b<2>;");
        s.AppendLine("    .reg .f32 %c<4>;");     // 4 accumulators (one position)
        s.AppendLine("    .reg .f32 %f<32>;");    // epilogue: 16 M + S + Y + bias
        s.AppendLine("    .reg .b32 %r<40>;");
        s.AppendLine("    .reg .b64 %rd<24>;");
        s.AppendLine($"    .shared .align 16 .b8 smem[{MSharedBytes}];");
        s.AppendLine("    ld.param.u64 %rd0, [u_ptr];");
        s.AppendLine("    ld.param.u64 %rd1, [v_ptr];");
        s.AppendLine("    ld.param.u64 %rd2, [bias_ptr];");
        s.AppendLine("    ld.param.u64 %rd3, [output_ptr];");
        s.AppendLine("    mov.u32 %r0, %tid.x;");
        s.AppendLine("    mov.u32 %r1, %ctaid.x;");                 // p-block (x8)
        s.AppendLine("    mov.u32 %r2, %ctaid.y;");                 // k-block (x16)
        s.AppendLine("    and.b32 %r3, %r0, 31;");                  // lane
        s.AppendLine("    shr.u32 %r4, %r0, 5;");                   // warpId = position xi (0..15)
        s.AppendLine("    shr.u32 %r5, %r3, 2;");                   // g = lane>>2 (0..7)
        s.AppendLine("    and.b32 %r6, %r3, 3;");                   // tig = lane&3 (0..3)
        s.AppendLine("    mul.lo.u32 %r7, %r2, 16;");               // kbase
        s.AppendLine("    mul.lo.u32 %r8, %r1, 8;");                // pbase
        s.AppendLine("    shl.b32 %r9, %r6, 1;");                   // tig*2
        s.AppendLine("    mov.u64 %rd4, smem;");
        // U fragment base for (kbase+g): rd5 = u + (warpId*KC + (kbase+g)*C + tig*2)*2
        s.AppendLine("    add.u32 %r10, %r7, %r5;");                // kbase+g
        s.AppendLine($"    mad.lo.u32 %r10, %r10, {I(c)}, %r9;");   // (kbase+g)*C + tig*2
        s.AppendLine($"    mad.lo.u32 %r10, %r4, {I(kc)}, %r10;");  // + warpId*KC
        s.AppendLine("    mul.wide.u32 %rd5, %r10, 2;");
        s.AppendLine("    add.u64 %rd5, %rd0, %rd5;");
        // U fragment base for (kbase+g+8): rd6
        s.AppendLine("    add.u32 %r11, %r7, %r5;");
        s.AppendLine("    add.u32 %r11, %r11, 8;");                 // kbase+g+8
        s.AppendLine($"    mad.lo.u32 %r11, %r11, {I(c)}, %r9;");
        s.AppendLine($"    mad.lo.u32 %r11, %r4, {I(kc)}, %r11;");
        s.AppendLine("    mul.wide.u32 %rd6, %r11, 2;");
        s.AppendLine("    add.u64 %rd6, %rd0, %rd6;");
        // V fragment base for (pbase+g): rd7 = v + (warpId*PC + (pbase+g)*C + tig*2)*2
        s.AppendLine("    add.u32 %r12, %r8, %r5;");                // pbase+g
        s.AppendLine($"    mad.lo.u32 %r12, %r12, {I(c)}, %r9;");
        s.AppendLine($"    mad.lo.u32 %r12, %r4, {I(pc)}, %r12;");
        s.AppendLine("    mul.wide.u32 %rd7, %r12, 2;");
        s.AppendLine("    add.u64 %rd7, %rd1, %rd7;");
        s.AppendLine();
        s.AppendLine("    mov.f32 %c0, 0f00000000;");
        s.AppendLine("    mov.f32 %c1, 0f00000000;");
        s.AppendLine("    mov.f32 %c2, 0f00000000;");
        s.AppendLine("    mov.f32 %c3, 0f00000000;");
        // GEMM: this warp's single position, accumulate over C.
        for (int ks = 0; ks < ksteps; ks++)
        {
            int off = ks * 32;
            s.AppendLine($"    ld.global.b32 %a0, [%rd5+{I(off)}];");
            s.AppendLine($"    ld.global.b32 %a2, [%rd5+{I(off + 16)}];");
            s.AppendLine($"    ld.global.b32 %a1, [%rd6+{I(off)}];");
            s.AppendLine($"    ld.global.b32 %a3, [%rd6+{I(off + 16)}];");
            s.AppendLine($"    ld.global.b32 %b0, [%rd7+{I(off)}];");
            s.AppendLine($"    ld.global.b32 %b1, [%rd7+{I(off + 16)}];");
            s.AppendLine("    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 " +
                "{%c0,%c1,%c2,%c3}, {%a0,%a1,%a2,%a3}, {%b0,%b1}, {%c0,%c1,%c2,%c3};");
        }
        // Write this warp's M[position] to shared: M_s[warpId][ch][tile], layout
        // warpId*512 + ch*32 + tile*4. c0=(g,tig*2) c1=(g,tig*2+1) c2=(g+8,tig*2) c3=(g+8,tig*2+1).
        s.AppendLine($"    shl.b32 %r13, %r4, 9;");                 // warpId*512
        s.AppendLine("    shl.b32 %r14, %r5, 5;");                  // g*32
        s.AppendLine("    shl.b32 %r15, %r6, 3;");                  // tig*8  (= tig*2 * 4)
        s.AppendLine("    add.u32 %r16, %r13, %r14;");
        s.AppendLine("    add.u32 %r16, %r16, %r15;");              // base for (g,tig*2)
        s.AppendLine("    cvt.u64.u32 %rd8, %r16;");
        s.AppendLine("    add.u64 %rd8, %rd4, %rd8;");
        s.AppendLine("    st.shared.f32 [%rd8], %c0;");
        s.AppendLine("    st.shared.f32 [%rd8+4], %c1;");
        s.AppendLine("    st.shared.f32 [%rd8+256], %c2;");         // (g+8)*32 = +256
        s.AppendLine("    st.shared.f32 [%rd8+260], %c3;");
        s.AppendLine("    bar.sync 0;");
        s.AppendLine();
        // Epilogue: 128 threads combine the 16 positions per (ch,tile).
        s.AppendLine("    setp.ge.u32 %pr0, %r0, 128;");
        s.AppendLine("    @%pr0 bra DONE;");
        s.AppendLine("    shr.u32 %r17, %r0, 3;");                  // ch = tid>>3 (0..15)
        s.AppendLine("    and.b32 %r18, %r0, 7;");                  // tile = tid&7 (0..7)
        s.AppendLine("    shl.b32 %r19, %r17, 5;");                 // ch*32
        s.AppendLine("    shl.b32 %r20, %r18, 2;");                 // tile*4
        s.AppendLine("    add.u32 %r19, %r19, %r20;");              // ch*32 + tile*4
        s.AppendLine("    cvt.u64.u32 %rd9, %r19;");
        s.AppendLine("    add.u64 %rd9, %rd4, %rd9;");              // &M_s[0][ch][tile]
        for (int xi = 0; xi < 16; xi++)
            s.AppendLine($"    ld.shared.f32 %f{I(xi)}, [%rd9+{I(xi * 512)}];");
        // Y = A^T M A. M[i][j]=%f(i*4+j). S in %f16..23, Y in %f24..27.
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
        // k = kbase + ch ; p = pbase + tile.
        s.AppendLine("    add.u32 %r21, %r7, %r17;");                // k
        s.AppendLine("    add.u32 %r22, %r8, %r18;");                // p
        s.AppendLine("    mul.wide.u32 %rd10, %r21, 4;");
        s.AppendLine("    add.u64 %rd10, %rd2, %rd10;");
        s.AppendLine("    ld.global.nc.f32 %f28, [%rd10];");         // bias[k]
        s.AppendLine($"    rem.u32 %r23, %r22, {I(tw)};");           // tj
        s.AppendLine($"    div.u32 %r24, %r22, {I(tw)};");
        s.AppendLine($"    rem.u32 %r25, %r24, {I(th)};");           // ti
        s.AppendLine($"    div.u32 %r26, %r24, {I(th)};");           // nn
        s.AppendLine("    mul.lo.u32 %r27, %r25, 2;");               // oh0
        s.AppendLine("    mul.lo.u32 %r28, %r23, 2;");               // ow0
        s.AppendLine($"    mad.lo.u32 %r29, %r26, {I(k)}, %r21;");   // nn*K + k
        s.AppendLine($"    mul.wide.u32 %rd11, %r29, {I(hw)};");
        s.AppendLine("    shl.b64 %rd11, %rd11, 2;");
        s.AppendLine("    add.u64 %rd11, %rd3, %rd11;");             // &out[nn][k][0][0]
        for (int oi = 0; oi < 2; oi++)
            for (int oj = 0; oj < 2; oj++)
            {
                int yreg = 24 + oi * 2 + oj;
                s.AppendLine($"    add.rn.f32 %f{yreg}, %f{yreg}, %f28;");
                s.AppendLine($"    max.f32 %f{yreg}, %f{yreg}, 0f00000000;");
                s.AppendLine($"    add.u32 %r30, %r27, {I(oi)};");
                s.AppendLine($"    mul.lo.u32 %r30, %r30, {I(w)};");
                s.AppendLine("    add.u32 %r30, %r30, %r28;");
                s.AppendLine($"    add.u32 %r30, %r30, {I(oj)};");
                s.AppendLine("    mul.wide.u32 %rd12, %r30, 4;");
                s.AppendLine("    add.u64 %rd12, %rd11, %rd12;");
                s.AppendLine($"    st.global.f32 [%rd12], %f{yreg};");
            }
        s.AppendLine("DONE:");
        s.AppendLine("    ret;");
        s.AppendLine("}");
        return s.ToString();
    }

    public void Dispose() => _module.Dispose();
}
