using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Fully-fused FP16 Tensor-Core Winograd F(2,3) 3x3 convolution (option C, the
/// cuDNN-style fused path). One warp computes a 16(output-channel) x 8(output-tile)
/// block for all 16 Winograd positions at once: it runs 16 raw
/// <c>mma.sync m16n8k16 f16-&gt;f32</c> accumulations (one per position) directly
/// from the fp16 U[16,K,C] / V[16,P,C] operands in global, keeping every position
/// result M[xi] in the accumulator registers. Because the m16n8k16 D-fragment has
/// a *defined* layout, each thread's four accumulators sit at known (k, tile)
/// coordinates, so the output transform Y = A^T M A + bias + ReLU is applied
/// thread-locally in the epilogue and the 2x2 tiles are scattered straight to
/// output[N,K,H,W]. There is NO M workspace and NO global round-trip between the
/// GEMM and the output transform -- the memory wall that sinks the batched
/// pipelines -- and fusing all 16 positions into one block amortizes the operand
/// traffic over 16x the Tensor-Core work, fixing the small-C underutilization of
/// the batched WMMA GEMM.
/// </summary>
internal sealed class PtxWinogradWmmaFusedKernel : IDisposable
{
    internal const int WarpSize = 32;

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
        $"aidotnet_winograd_wmma_fused_n{Batch}_c{InputChannels}_h{Height}_w{Width}_k{OutputChannels}");

    internal PtxWinogradWmmaFusedKernel(
        DirectPtxRuntime runtime, int batch, int inputChannels, int height, int width, int outputChannels)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new NotSupportedException("Fused WMMA Winograd has no experimental non-SM86 specialization.");
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
        if ((Tiles & 7) != 0)
            throw new ArgumentException("P (tile count) must be a multiple of 8.");

        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, batch, inputChannels, height, width, outputChannels);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor,
            batch, inputChannels, height, width, outputChannels);
        _module = runtime.LoadModule(
            Ptx, allowExperimentalJitFallback: DirectPtxFeatureGate.ConvolutionExperimentOverride);
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo functionInfo);
        FunctionInfo = functionInfo;
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, WarpSize);
        Blueprint.ResourceBudget.Validate(EntryPoint, functionInfo, WarpSize, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Ptx, functionInfo, WarpSize, activeBlocks, _module);
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
            Operation: "winograd-wmma-fused",
            Version: 1,
            Architecture: architecture,
            Variant: FormattableString.Invariant($"n{n}-c{c}-h{h}-w{w}-k{k}-m16n8k16-f16f32-fused"),
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
                MaxRegistersPerThread: 128, MaxStaticSharedBytes: 0,
                MaxLocalBytesPerThread: 0, MinBlocksPerMultiprocessor: 1),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["equation"] = "relu(A^T (U*V) A + bias)",
                ["mma"] = "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 x16 positions",
                ["fusion"] = "16 position GEMMs + output transform in registers; no M workspace",
                ["accumulator"] = "mma-m16n8k16-fp32-register-fragment x16",
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
        // grid: (P/8 tile-blocks, K/16 channel-blocks, 1); one warp per block.
        _module.Launch(
            _function,
            (uint)(Tiles / 8), (uint)(OutputChannels / 16), 1,
            WarpSize, 1, 1,
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
            throw new NotSupportedException("Only the experimental SM86 fused WMMA Winograd emitter exists.");
        if ((k & 15) != 0 || (c & 15) != 0) throw new ArgumentException("K and C must be multiples of 16.");
        string I(int v) => v.ToString(CultureInfo.InvariantCulture);
        int th = h / 2, tw = w / 2;
        int p = n * th * tw;                 // P
        int hw = h * w;
        int kc2 = k * c * sizeof(ushort);    // U xi stride (bytes)
        int pc2 = p * c * sizeof(ushort);    // V xi stride (bytes)
        int ksteps = c / 16;
        int accReg(int xi, int e) => xi * 4 + e;
        string entry = FormattableString.Invariant(
            $"aidotnet_winograd_wmma_fused_n{n}_c{c}_h{h}_w{w}_k{k}");

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
        s.AppendLine("    .reg .pred %pr<2>;");
        s.AppendLine("    .reg .b32 %a<4>;");
        s.AppendLine("    .reg .b32 %b<2>;");
        s.AppendLine("    .reg .f32 %c<64>;");   // 16 positions x 4 accumulators
        s.AppendLine("    .reg .f32 %f<16>;");   // epilogue scratch
        s.AppendLine("    .reg .b32 %r<40>;");
        s.AppendLine("    .reg .b64 %rd<24>;");
        s.AppendLine("    ld.param.u64 %rd0, [u_ptr];");
        s.AppendLine("    ld.param.u64 %rd1, [v_ptr];");
        s.AppendLine("    ld.param.u64 %rd2, [bias_ptr];");
        s.AppendLine("    ld.param.u64 %rd3, [output_ptr];");
        s.AppendLine("    mov.u32 %r0, %tid.x;");
        s.AppendLine("    mov.u32 %r1, %ctaid.x;");                 // p-block
        s.AppendLine("    mov.u32 %r2, %ctaid.y;");                 // k-block
        s.AppendLine("    shr.u32 %r3, %r0, 2;");                   // g = lane>>2 (0..7)
        s.AppendLine("    and.b32 %r4, %r0, 3;");                   // tig = lane&3 (0..3)
        s.AppendLine("    mul.lo.u32 %r5, %r2, 16;");               // kbase
        s.AppendLine("    mul.lo.u32 %r6, %r1, 8;");                // pbase
        s.AppendLine("    add.u32 %r7, %r5, %r3;");                 // kbase+g
        s.AppendLine("    add.u32 %r8, %r7, 8;");                   // kbase+g+8
        s.AppendLine("    add.u32 %r9, %r6, %r3;");                 // pbase+g
        s.AppendLine("    shl.b32 %r10, %r4, 1;");                  // tig*2 (column within k-tile)
        // U row base for (kbase+g):  rd4 = u + ((kbase+g)*C + tig*2)*2
        s.AppendLine($"    mad.lo.u32 %r11, %r7, {I(c)}, %r10;");
        s.AppendLine("    mul.wide.u32 %rd4, %r11, 2;");
        s.AppendLine("    add.u64 %rd4, %rd0, %rd4;");
        // U row base for (kbase+g+8): rd5
        s.AppendLine($"    mad.lo.u32 %r12, %r8, {I(c)}, %r10;");
        s.AppendLine("    mul.wide.u32 %rd5, %r12, 2;");
        s.AppendLine("    add.u64 %rd5, %rd0, %rd5;");
        // V row base for (pbase+g):   rd6 = v + ((pbase+g)*C + tig*2)*2
        s.AppendLine($"    mad.lo.u32 %r13, %r9, {I(c)}, %r10;");
        s.AppendLine("    mul.wide.u32 %rd6, %r13, 2;");
        s.AppendLine("    add.u64 %rd6, %rd1, %rd6;");
        s.AppendLine();
        for (int i = 0; i < 64; i++)
            s.AppendLine($"    mov.f32 %c{i}, 0f00000000;");

        // 16 position GEMMs. Advance per-xi base pointers, inner k-step uses const offsets.
        for (int xi = 0; xi < 16; xi++)
        {
            s.AppendLine();
            s.AppendLine($"    // ---- position {xi} ----");
            s.AppendLine($"    add.u64 %rd7, %rd4, {I(xi * kc2)};");   // U(g)  base for xi
            s.AppendLine($"    add.u64 %rd8, %rd5, {I(xi * kc2)};");   // U(g8) base for xi
            s.AppendLine($"    add.u64 %rd9, %rd6, {I(xi * pc2)};");   // V(g)  base for xi
            for (int ks = 0; ks < ksteps; ks++)
            {
                int off = ks * 32;   // 16 f16 columns per k-step
                s.AppendLine($"    ld.global.b32 %a0, [%rd7+{I(off)}];");
                s.AppendLine($"    ld.global.b32 %a2, [%rd7+{I(off + 16)}];");
                s.AppendLine($"    ld.global.b32 %a1, [%rd8+{I(off)}];");
                s.AppendLine($"    ld.global.b32 %a3, [%rd8+{I(off + 16)}];");
                s.AppendLine($"    ld.global.b32 %b0, [%rd9+{I(off)}];");
                s.AppendLine($"    ld.global.b32 %b1, [%rd9+{I(off + 16)}];");
                s.AppendLine("    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 " +
                    $"{{%c{accReg(xi, 0)},%c{accReg(xi, 1)},%c{accReg(xi, 2)},%c{accReg(xi, 3)}}}, " +
                    "{%a0,%a1,%a2,%a3}, {%b0,%b1}, " +
                    $"{{%c{accReg(xi, 0)},%c{accReg(xi, 1)},%c{accReg(xi, 2)},%c{accReg(xi, 3)}}};");
            }
        }

        // Epilogue: each thread owns 4 outputs e=0..3:
        //   e=0 -> (k=kbase+g,   p=pbase+tig*2)   uses acc[xi][0]
        //   e=1 -> (k=kbase+g,   p=pbase+tig*2+1) uses acc[xi][1]
        //   e=2 -> (k=kbase+g+8, p=pbase+tig*2)   uses acc[xi][2]
        //   e=3 -> (k=kbase+g+8, p=pbase+tig*2+1) uses acc[xi][3]
        // %r10 currently holds tig*2.
        for (int e = 0; e < 4; e++)
        {
            s.AppendLine();
            s.AppendLine($"    // ---- epilogue output {e} ----");
            // Y = A^T M A. M[i][j] = acc[i*4+j][e]. S in %f0..7, Y in %f8..11.
            int Macc(int i, int j) => accReg(i * 4 + j, e);
            int Sreg(int i, int j) => i * 4 + j;          // %f0..7
            for (int j = 0; j < 4; j++)
            {
                s.AppendLine($"    add.rn.f32 %f{Sreg(0, j)}, %c{Macc(0, j)}, %c{Macc(1, j)};");
                s.AppendLine($"    add.rn.f32 %f{Sreg(0, j)}, %f{Sreg(0, j)}, %c{Macc(2, j)};");
                s.AppendLine($"    sub.rn.f32 %f{Sreg(1, j)}, %c{Macc(1, j)}, %c{Macc(2, j)};");
                s.AppendLine($"    sub.rn.f32 %f{Sreg(1, j)}, %f{Sreg(1, j)}, %c{Macc(3, j)};");
            }
            // Y[i][0]=S[i][0]+S[i][1]+S[i][2]; Y[i][1]=S[i][1]-S[i][2]-S[i][3]. Y in %f8..11.
            for (int i = 0; i < 2; i++)
            {
                int y0 = 8 + i * 2, y1 = 9 + i * 2;
                s.AppendLine($"    add.rn.f32 %f{y0}, %f{Sreg(i, 0)}, %f{Sreg(i, 1)};");
                s.AppendLine($"    add.rn.f32 %f{y0}, %f{y0}, %f{Sreg(i, 2)};");
                s.AppendLine($"    sub.rn.f32 %f{y1}, %f{Sreg(i, 1)}, %f{Sreg(i, 2)};");
                s.AppendLine($"    sub.rn.f32 %f{y1}, %f{y1}, %f{Sreg(i, 3)};");
            }
            // k for this e, p for this e.
            string kThis = e < 2 ? "%r7" : "%r8";              // kbase+g or kbase+g+8
            // p = pbase + tig*2 + (e&1). %r14 = p.
            s.AppendLine($"    add.u32 %r14, %r6, %r10;");      // pbase + tig*2
            if ((e & 1) == 1) s.AppendLine("    add.u32 %r14, %r14, 1;");
            // bias[k]
            s.AppendLine($"    mul.wide.u32 %rd10, {kThis}, 4;");
            s.AppendLine("    add.u64 %rd10, %rd2, %rd10;");
            s.AppendLine("    ld.global.nc.f32 %f12, [%rd10];");
            // decode p -> nn, ti, tj
            s.AppendLine($"    rem.u32 %r15, %r14, {I(tw)};");   // tj
            s.AppendLine($"    div.u32 %r16, %r14, {I(tw)};");
            s.AppendLine($"    rem.u32 %r17, %r16, {I(th)};");   // ti
            s.AppendLine($"    div.u32 %r18, %r16, {I(th)};");   // nn
            s.AppendLine("    mul.lo.u32 %r19, %r17, 2;");        // oh0
            s.AppendLine("    mul.lo.u32 %r20, %r15, 2;");        // ow0
            // output channel base: out + (nn*K + k)*H*W*4
            s.AppendLine($"    mad.lo.u32 %r21, %r18, {I(k)}, {kThis};");
            s.AppendLine($"    mul.wide.u32 %rd11, %r21, {I(hw)};");
            s.AppendLine("    shl.b64 %rd11, %rd11, 2;");
            s.AppendLine("    add.u64 %rd11, %rd3, %rd11;");      // &out[nn][k][0][0]
            for (int oi = 0; oi < 2; oi++)
                for (int oj = 0; oj < 2; oj++)
                {
                    int yreg = 8 + oi * 2 + oj;
                    s.AppendLine($"    add.rn.f32 %f{yreg}, %f{yreg}, %f12;");
                    s.AppendLine($"    max.f32 %f{yreg}, %f{yreg}, 0f00000000;");
                    s.AppendLine($"    add.u32 %r22, %r19, {I(oi)};");
                    s.AppendLine($"    mul.lo.u32 %r22, %r22, {I(w)};");
                    s.AppendLine("    add.u32 %r22, %r22, %r20;");
                    s.AppendLine($"    add.u32 %r22, %r22, {I(oj)};");
                    s.AppendLine("    mul.wide.u32 %rd12, %r22, 4;");
                    s.AppendLine("    add.u64 %rd12, %rd11, %rd12;");
                    s.AppendLine($"    st.global.f32 [%rd12], %f{yreg};");
                }
        }
        s.AppendLine("    ret;");
        s.AppendLine("}");
        return s.ToString();
    }

    public void Dispose() => _module.Dispose();
}
