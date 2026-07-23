using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Cooperative FP16 Tensor-Core Winograd F(2,3) 3x3 convolution with mma-level
/// register blocking. Like <see cref="PtxWinogradWmmaCoopKernel"/> it assigns one
/// Winograd position per warp (4 accumulators-per-mma -> full occupancy) and
/// exchanges the 16 position results through shared memory, but each warp now
/// computes a 16(channel) x 32(tile) block by reusing a single loaded A=U fragment
/// across four N-subtiles (four <c>mma.sync m16n8k16</c> per k-step). SASS analysis
/// of the non-blocked kernels showed ~24 scalar global fragment loads feeding only
/// 4 HMMA -- the Tensor Cores were starved by low arithmetic intensity, NOT by
/// occupancy (the coop kernel is fully occupied and still slow). Reusing the A
/// fragment across the four subtiles amortizes the U loads and roughly halves the
/// loads-per-mma, the same load-amortization the fast fp32 register-blocked GEMM
/// gets from shared staging. Output transform Y = A^T M A + bias + ReLU runs in a
/// combined epilogue over the 16 staged positions; M never touches global.
/// </summary>
internal sealed class PtxWinogradWmmaCoopBlockedKernel : IDisposable
{
    internal const int Warps = 16;                 // one per Winograd position
    internal const int BlockThreads = Warps * 32;  // 512
    internal const int TileK = 16;                 // output channels per block
    internal const int TileP = 32;                 // output tiles per block (4 N-subtiles of 8)
    internal const int NSub = TileP / 8;           // 4
    internal const int MSharedBytes = 16 * TileK * TileP * sizeof(float);   // 32768

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
        $"aidotnet_winograd_wmma_coopblk_n{Batch}_c{InputChannels}_h{Height}_w{Width}_k{OutputChannels}");

    internal PtxWinogradWmmaCoopBlockedKernel(
        DirectPtxRuntime runtime, int batch, int inputChannels, int height, int width, int outputChannels)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new NotSupportedException("Coop-blocked WMMA Winograd has no experimental non-SM86 specialization.");
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
            Operation: "winograd-wmma-coop-blocked",
            Version: 1,
            Architecture: architecture,
            Variant: FormattableString.Invariant($"n{n}-c{c}-h{h}-w{w}-k{k}-m16n8k16-f16f32-coop16warp-wn4"),
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
                MaxRegistersPerThread: 80, MaxStaticSharedBytes: MSharedBytes,
                MaxLocalBytesPerThread: 0, MinBlocksPerMultiprocessor: 1),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["equation"] = "relu(A^T (U*V) A + bias)",
                ["mma"] = "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32, 4 N-subtiles reuse one A fragment",
                ["arithmetic-intensity"] = "A(U) fragment reused across 4 mma -> halved loads-per-mma vs non-blocked",
                ["exchange"] = "16 position results via shared, combined output transform",
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
            throw new NotSupportedException("Only the experimental SM86 coop-blocked WMMA Winograd emitter exists.");
        if ((k & 15) != 0 || (c & 15) != 0) throw new ArgumentException("K and C must be multiples of 16.");
        string I(int v) => v.ToString(CultureInfo.InvariantCulture);
        int th = h / 2, tw = w / 2;
        int p = n * th * tw;                 // P
        int hw = h * w;
        int kc = k * c;                      // U position stride (elements)
        int pc = p * c;                      // V position stride (elements)
        int ksteps = c / 16;
        int accReg(int sub, int e) => sub * 4 + e;   // %c0..15
        string entry = FormattableString.Invariant(
            $"aidotnet_winograd_wmma_coopblk_n{n}_c{c}_h{h}_w{w}_k{k}");

        var s = new StringBuilder(49152);
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
        s.AppendLine("    .reg .f32 %c<16>;");    // 4 subtiles x 4 accumulators
        s.AppendLine("    .reg .f32 %f<32>;");
        s.AppendLine("    .reg .b32 %r<48>;");
        s.AppendLine("    .reg .b64 %rd<32>;");
        s.AppendLine($"    .shared .align 16 .b8 smem[{MSharedBytes}];");
        s.AppendLine("    ld.param.u64 %rd0, [u_ptr];");
        s.AppendLine("    ld.param.u64 %rd1, [v_ptr];");
        s.AppendLine("    ld.param.u64 %rd2, [bias_ptr];");
        s.AppendLine("    ld.param.u64 %rd3, [output_ptr];");
        s.AppendLine("    mov.u32 %r0, %tid.x;");
        s.AppendLine("    mov.u32 %r1, %ctaid.x;");                 // p-block (x32)
        s.AppendLine("    mov.u32 %r2, %ctaid.y;");                 // k-block (x16)
        s.AppendLine("    and.b32 %r3, %r0, 31;");                  // lane
        s.AppendLine("    shr.u32 %r4, %r0, 5;");                   // warpId = position (0..15)
        s.AppendLine("    shr.u32 %r5, %r3, 2;");                   // g = lane>>2 (0..7)
        s.AppendLine("    and.b32 %r6, %r3, 3;");                   // tig = lane&3
        s.AppendLine("    mul.lo.u32 %r7, %r2, 16;");               // kbase
        s.AppendLine("    mul.lo.u32 %r8, %r1, 32;");               // pbase
        s.AppendLine("    shl.b32 %r9, %r6, 1;");                   // tig*2
        s.AppendLine("    mov.u64 %rd4, smem;");
        // A fragment bases: rd5 = u + (warpId*KC + (kbase+g)*C + tig*2)*2 ; rd6 for (kbase+g+8)
        s.AppendLine("    add.u32 %r10, %r7, %r5;");
        s.AppendLine($"    mad.lo.u32 %r10, %r10, {I(c)}, %r9;");
        s.AppendLine($"    mad.lo.u32 %r10, %r4, {I(kc)}, %r10;");
        s.AppendLine("    mul.wide.u32 %rd5, %r10, 2;");
        s.AppendLine("    add.u64 %rd5, %rd0, %rd5;");
        s.AppendLine("    add.u32 %r11, %r7, %r5;");
        s.AppendLine("    add.u32 %r11, %r11, 8;");
        s.AppendLine($"    mad.lo.u32 %r11, %r11, {I(c)}, %r9;");
        s.AppendLine($"    mad.lo.u32 %r11, %r4, {I(kc)}, %r11;");
        s.AppendLine("    mul.wide.u32 %rd6, %r11, 2;");
        s.AppendLine("    add.u64 %rd6, %rd0, %rd6;");
        // V fragment base for subtile 0 row (pbase+g): rd7 = v + (warpId*PC + (pbase+g)*C + tig*2)*2.
        // subtile s adds (s*8)*C*2 to the row -> constant per-subtile offset s*8*C*2.
        s.AppendLine("    add.u32 %r12, %r8, %r5;");
        s.AppendLine($"    mad.lo.u32 %r12, %r12, {I(c)}, %r9;");
        s.AppendLine($"    mad.lo.u32 %r12, %r4, {I(pc)}, %r12;");
        s.AppendLine("    mul.wide.u32 %rd7, %r12, 2;");
        s.AppendLine("    add.u64 %rd7, %rd1, %rd7;");
        s.AppendLine();
        for (int i = 0; i < 16; i++)
            s.AppendLine($"    mov.f32 %c{i}, 0f00000000;");
        int subRowBytes = 8 * c * sizeof(ushort);   // V row stride between subtiles

        for (int ks = 0; ks < ksteps; ks++)
        {
            int off = ks * 32;
            // Load A fragment once (reused across 4 subtiles).
            s.AppendLine($"    ld.global.b32 %a0, [%rd5+{I(off)}];");
            s.AppendLine($"    ld.global.b32 %a2, [%rd5+{I(off + 16)}];");
            s.AppendLine($"    ld.global.b32 %a1, [%rd6+{I(off)}];");
            s.AppendLine($"    ld.global.b32 %a3, [%rd6+{I(off + 16)}];");
            for (int sub = 0; sub < NSub; sub++)
            {
                int voff = off + sub * subRowBytes;
                s.AppendLine($"    ld.global.b32 %b0, [%rd7+{I(voff)}];");
                s.AppendLine($"    ld.global.b32 %b1, [%rd7+{I(voff + 16)}];");
                s.AppendLine("    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 " +
                    $"{{%c{accReg(sub, 0)},%c{accReg(sub, 1)},%c{accReg(sub, 2)},%c{accReg(sub, 3)}}}, " +
                    "{%a0,%a1,%a2,%a3}, {%b0,%b1}, " +
                    $"{{%c{accReg(sub, 0)},%c{accReg(sub, 1)},%c{accReg(sub, 2)},%c{accReg(sub, 3)}}};");
            }
        }
        // Write M[position] to shared: M_s[warpId][ch][tile], warpId*2048 + ch*128 + tile*4.
        // subtile s: tiles s*8 + {tig*2, tig*2+1}; ch {g, g+8}.
        s.AppendLine($"    shl.b32 %r13, %r4, 11;");                // warpId*2048
        s.AppendLine("    shl.b32 %r14, %r5, 7;");                  // g*128
        s.AppendLine("    shl.b32 %r15, %r6, 3;");                  // tig*2*4 = tig*8
        s.AppendLine("    add.u32 %r16, %r13, %r14;");
        s.AppendLine("    add.u32 %r16, %r16, %r15;");              // base (g, tile=tig*2)
        s.AppendLine("    cvt.u64.u32 %rd8, %r16;");
        s.AppendLine("    add.u64 %rd8, %rd4, %rd8;");
        for (int sub = 0; sub < NSub; sub++)
        {
            int tbyte = sub * 8 * 4;    // tile s*8 -> +s*32 bytes
            s.AppendLine($"    st.shared.f32 [%rd8+{I(tbyte)}], %c{accReg(sub, 0)};");
            s.AppendLine($"    st.shared.f32 [%rd8+{I(tbyte + 4)}], %c{accReg(sub, 1)};");
            s.AppendLine($"    st.shared.f32 [%rd8+{I(tbyte + 1024)}], %c{accReg(sub, 2)};");   // (g+8)*128 = +1024
            s.AppendLine($"    st.shared.f32 [%rd8+{I(tbyte + 1028)}], %c{accReg(sub, 3)};");
        }
        s.AppendLine("    bar.sync 0;");
        s.AppendLine();
        // Epilogue: all 512 threads. tid -> ch=tid>>5 (0..15), tile=tid&31 (0..31).
        s.AppendLine("    shr.u32 %r17, %r0, 5;");                  // ch
        s.AppendLine("    and.b32 %r18, %r0, 31;");                 // tile
        s.AppendLine("    shl.b32 %r19, %r17, 7;");                 // ch*128
        s.AppendLine("    shl.b32 %r20, %r18, 2;");                 // tile*4
        s.AppendLine("    add.u32 %r19, %r19, %r20;");
        s.AppendLine("    cvt.u64.u32 %rd9, %r19;");
        s.AppendLine("    add.u64 %rd9, %rd4, %rd9;");              // &M_s[0][ch][tile]
        for (int xi = 0; xi < 16; xi++)
            s.AppendLine($"    ld.shared.f32 %f{I(xi)}, [%rd9+{I(xi * 2048)}];");
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
        s.AppendLine("    add.u32 %r21, %r7, %r17;");                // k = kbase + ch
        s.AppendLine("    add.u32 %r22, %r8, %r18;");                // p = pbase + tile
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
        s.AppendLine("    add.u64 %rd11, %rd3, %rd11;");
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
        s.AppendLine("    ret;");
        s.AppendLine("}");
        return s.ToString();
    }

    public void Dispose() => _module.Dispose();
}
