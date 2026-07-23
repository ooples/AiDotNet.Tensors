using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Exact shape descriptor for the register-blocked tiled 1x1 convolution GEMM.
/// A 1x1 NCHW convolution is the batched GEMM O[K, N] = W[K, C] . X[C, N] with
/// N = batch*spatial. Each block computes a BM x BN output tile; each thread
/// computes a TM x TN micro-tile held entirely in registers, so every value
/// staged into shared memory is reused TM (for weights) / TN (for inputs) times
/// before leaving registers — that arithmetic-intensity gain is what lets a hand
/// kernel approach/beat cuDNN, unlike the TM=TN=1 tiled variant.
///
/// <para>Issue #841 forbids dynamic-shape PTX, so BM|K, BN|HW, BK|C, BM%TM==0,
/// BN%TN==0 are all required (baked, no boundary predicate), and BK, BN are
/// powers of two so the cooperative-load row/col split is shift/mask.</para>
/// </summary>
internal readonly struct Conv2DRegBlockShape
{
    internal int Batch { get; }
    internal int OutputChannels { get; }
    internal int InputChannels { get; }
    internal int Spatial { get; }
    internal int BlockM { get; }
    internal int BlockN { get; }
    internal int BlockK { get; }
    internal int ThreadM { get; }
    internal int ThreadN { get; }

    internal Conv2DRegBlockShape(
        int batch, int outputChannels, int inputChannels, int spatial,
        int blockM = 64, int blockN = 64, int blockK = 16, int threadM = 4, int threadN = 4)
    {
        if (batch <= 0 || outputChannels <= 0 || inputChannels <= 0 || spatial <= 0)
            throw new ArgumentOutOfRangeException(nameof(batch), "Dimensions must be positive.");
        if (blockM <= 0 || blockN <= 0 || blockK <= 0 || threadM <= 0 || threadN <= 0)
            throw new ArgumentOutOfRangeException(nameof(blockM), "Tile parameters must be positive.");
        if (outputChannels % blockM != 0 || spatial % blockN != 0 || inputChannels % blockK != 0)
            throw new ArgumentException("BM|K, BN|HW, BK|C are required so no boundary predicate is needed.");
        if (blockM % threadM != 0 || blockN % threadN != 0)
            throw new ArgumentException("BM%TM==0 and BN%TN==0 are required.");
        if (!IsPow2(blockK) || !IsPow2(blockN))
            throw new ArgumentException("BK and BN must be powers of two (cooperative-load index split).");
        int threads = (blockM / threadM) * (blockN / threadN);
        if (threads > 1024) throw new ArgumentException("Block would exceed 1024 threads.");
        if (blockM * blockK % threads != 0 || blockK * blockN % threads != 0)
            throw new ArgumentException("Shared tiles must divide evenly across threads for cooperative load.");

        Batch = batch; OutputChannels = outputChannels; InputChannels = inputChannels; Spatial = spatial;
        BlockM = blockM; BlockN = blockN; BlockK = blockK; ThreadM = threadM; ThreadN = threadN;
    }

    private static bool IsPow2(int v) => v > 0 && (v & (v - 1)) == 0;

    internal int ThreadsPerBlock => (BlockM / ThreadM) * (BlockN / ThreadN);
    internal long InputBytes => (long)Batch * InputChannels * Spatial * sizeof(float);
    internal long WeightBytes => (long)OutputChannels * InputChannels * sizeof(float);
    internal long BiasBytes => (long)OutputChannels * sizeof(float);
    internal long OutputBytes => (long)Batch * OutputChannels * Spatial * sizeof(float);

    internal string EntryPoint => FormattableString.Invariant(
        $"aidotnet_conv2d_n{Batch}_c{InputChannels}_hw{Spatial}_k{OutputChannels}_k1_rb{BlockM}x{BlockN}x{BlockK}_{ThreadM}x{ThreadN}_bias_relu");
    internal string Variant => FormattableString.Invariant(
        $"n{Batch}-c{InputChannels}-hw{Spatial}-k{OutputChannels}-r1-s1-p0-bm{BlockM}-bn{BlockN}-bk{BlockK}-tm{ThreadM}-tn{ThreadN}-fp32");
}

/// <summary>
/// Register-blocked shared-memory tiled FP32 NCHW 1x1 convolution GEMM with
/// fused bias and ReLU. Fully baked geometry: the device receives four pointers.
/// Drafted for the #841 GPU measurement window; correctness (&lt;= 2e-4) and the
/// &gt;=1.10x-vs-cuDNN gate are validated on-device.
/// </summary>
internal sealed class PtxConv2DNchwK1RegBlockedKernel : IDisposable
{
    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal Conv2DRegBlockShape Shape { get; }
    internal string Ptx { get; }
    internal DirectPtxFunctionInfo FunctionInfo { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal int BlockThreads => Shape.ThreadsPerBlock;

    internal PtxConv2DNchwK1RegBlockedKernel(DirectPtxRuntime runtime, Conv2DRegBlockShape shape)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new NotSupportedException(
                $"Register-blocked convolution has no experimental SM {runtime.ComputeCapabilityMajor}.{runtime.ComputeCapabilityMinor} specialization.");
        Shape = shape;

        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, shape);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor, shape);
        _module = runtime.LoadModule(
            Ptx, allowExperimentalJitFallback: DirectPtxFeatureGate.ConvolutionExperimentOverride);
        _function = _module.GetFunction(shape.EntryPoint, out DirectPtxFunctionInfo functionInfo);
        FunctionInfo = functionInfo;
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, BlockThreads);
        Blueprint.ResourceBudget.Validate(shape.EntryPoint, functionInfo, BlockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Ptx, functionInfo,
            BlockThreads, activeBlocks, _module);
    }

    internal static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture, Conv2DRegBlockShape shape)
    {
        var input = new DirectPtxExtent(shape.Batch, shape.InputChannels, shape.Spatial);
        var weights = new DirectPtxExtent(shape.OutputChannels, shape.InputChannels);
        var bias = new DirectPtxExtent(shape.OutputChannels);
        var output = new DirectPtxExtent(shape.Batch, shape.OutputChannels, shape.Spatial);
        int sharedBytes = (shape.BlockM * shape.BlockK + shape.BlockK * shape.BlockN) * sizeof(float);
        return new DirectPtxKernelBlueprint(
            Operation: "conv2d-bias-relu-regblocked",
            Version: 1,
            Architecture: architecture,
            Variant: shape.Variant,
            Tensors:
            [
                new("input", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Nchw,
                    input, input, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("weights", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Oihw,
                    weights, weights, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("bias", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    bias, bias, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Nchw,
                    output, output, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                // Config-derived ceiling: TM*TN accumulators + TM+TN fragments +
                // addressing/temp headroom. Catches register blow-ups (spills)
                // while scaling across block/micro-tile configs. Measured 40 for
                // 64x64x16/4x4 on SM86, comfortably under this.
                MaxRegistersPerThread: shape.ThreadM * shape.ThreadN + shape.ThreadM + shape.ThreadN + 32,
                MaxStaticSharedBytes: sharedBytes,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 1),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["equation"] = "relu(conv2d-nchw-1x1(input,oihw-weights)+bias)",
                ["algorithm"] = "register-blocked-shared-memory-tiled-gemm",
                ["block"] = FormattableString.Invariant($"{shape.BlockM}x{shape.BlockN}x{shape.BlockK}"),
                ["microtile"] = FormattableString.Invariant($"{shape.ThreadM}x{shape.ThreadN}"),
                ["input"] = "fp32",
                ["accumulator"] = "fp32-fma",
                ["output"] = "fp32",
                ["layout"] = "nchw/oihw",
                ["intermediate-global-bytes"] = "0",
                ["shape-selection"] = "host-only-exact-contract",
                ["promotion"] = "experimental-pending-gpu-evidence"
            });
    }

    internal unsafe void Launch(
        DirectPtxTensorView input,
        DirectPtxTensorView weights,
        DirectPtxTensorView bias,
        DirectPtxTensorView output)
    {
        Require(input, Blueprint.Tensors[0], nameof(input));
        Require(weights, Blueprint.Tensors[1], nameof(weights));
        Require(bias, Blueprint.Tensors[2], nameof(bias));
        Require(output, Blueprint.Tensors[3], nameof(output));

        IntPtr inputPointer = input.Pointer;
        IntPtr weightPointer = weights.Pointer;
        IntPtr biasPointer = bias.Pointer;
        IntPtr outputPointer = output.Pointer;
        void** arguments = stackalloc void*[4];
        arguments[0] = &inputPointer;
        arguments[1] = &weightPointer;
        arguments[2] = &biasPointer;
        arguments[3] = &outputPointer;

        uint gridX = (uint)(Shape.Batch * Shape.Spatial / Shape.BlockN);
        uint gridY = (uint)(Shape.OutputChannels / Shape.BlockM);
        _module.Launch(
            _function, gridX, gridY, 1,
            (uint)(Shape.BlockN / Shape.ThreadN), (uint)(Shape.BlockM / Shape.ThreadM), 1, 0, arguments);
    }

    private static void Require(
        DirectPtxTensorView view, DirectPtxTensorContract contract, string parameter)
    {
        if (view.Pointer == IntPtr.Zero || view.PhysicalType != contract.PhysicalType ||
            view.Layout != contract.Layout || view.LogicalExtent != contract.LogicalExtent ||
            view.PhysicalExtent != contract.PhysicalExtent ||
            view.ByteLength != contract.RequiredBytes ||
            view.AllocationByteLength != contract.RequiredBytes)
            throw new ArgumentException(
                $"{parameter} does not satisfy exact physical ABI '{contract.Name}'.", parameter);
    }

    internal static string EmitPtx(
        int computeCapabilityMajor, int computeCapabilityMinor, Conv2DRegBlockShape shape)
    {
        if (!DirectPtxArchitecture.HasExperimentalConvolution(
                computeCapabilityMajor, computeCapabilityMinor))
            throw new NotSupportedException("Only the experimental SM86 register-blocked emitter exists.");

        int k = shape.OutputChannels, c = shape.InputChannels, hw = shape.Spatial;
        int bm = shape.BlockM, bn = shape.BlockN, bk = shape.BlockK, tm = shape.ThreadM, tn = shape.ThreadN;
        int threads = shape.ThreadsPerBlock;
        int loadsA = bm * bk / threads;
        int loadsB = bk * bn / threads;
        int contractionTiles = c / bk;
        int log2Bk = Log2(bk);
        int log2Bn = Log2(bn);
        int aBytes = bm * bk * sizeof(float);
        int bBytes = bk * bn * sizeof(float);
        string entry = shape.EntryPoint;
        string I(int v) => v.ToString(CultureInfo.InvariantCulture);

        var ptx = new StringBuilder(32768);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{computeCapabilityMajor}{computeCapabilityMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {entry}(");
        ptx.AppendLine("    .param .u64 input_ptr,");
        ptx.AppendLine("    .param .u64 weights_ptr,");
        ptx.AppendLine("    .param .u64 bias_ptr,");
        ptx.AppendLine("    .param .u64 output_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<2>;");
        ptx.AppendLine("    .reg .b32 %r<48>;");
        ptx.AppendLine("    .reg .b64 %rd<40>;");
        ptx.AppendLine("    .reg .f32 %f<48>;");
        ptx.AppendLine($"    .shared .align 4 .b8 gemm_a[{I(aBytes)}];");
        ptx.AppendLine($"    .shared .align 4 .b8 gemm_b[{I(bBytes)}];");
        ptx.AppendLine("    ld.param.u64 %rd0, [input_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [weights_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [bias_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd3, [output_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");    // tx in [0, BN/TN)
        ptx.AppendLine("    mov.u32 %r1, %tid.y;");    // ty in [0, BM/TM)
        ptx.AppendLine("    mov.u32 %r2, %ctaid.x;");  // column tile
        ptx.AppendLine("    mov.u32 %r3, %ctaid.y;");  // row tile (K)
        ptx.AppendLine($"    mul.lo.u32 %r4, %r3, {I(bm)};");        // rowTileBase = ctaid.y*BM
        ptx.AppendLine($"    mad.lo.u32 %r5, %r1, {I(tm)}, %r4;");   // rowBase = rowTileBase + ty*TM
        ptx.AppendLine($"    mul.lo.u32 %r6, %r2, {I(bn)};");        // colTileBase = ctaid.x*BN
        ptx.AppendLine($"    div.u32 %r7, %r6, {I(hw)};");           // n
        ptx.AppendLine($"    rem.u32 %r8, %r6, {I(hw)};");           // hwBase
        ptx.AppendLine($"    mad.lo.u32 %r9, %r0, {I(tn)}, %r8;");   // hwFirst = hwBase + tx*TN
        ptx.AppendLine($"    mad.lo.u32 %r10, %r1, {I(bn / tn)}, %r0;"); // tid = ty*(BN/TN)+tx
        ptx.AppendLine("    mov.u64 %rd4, gemm_a;");
        ptx.AppendLine("    mov.u64 %rd5, gemm_b;");
        // per-thread shared micro-tile bases (t-independent)
        ptx.AppendLine($"    mul.lo.u32 %r11, %r1, {I(tm * bk)};");  // ty*TM*BK
        ptx.AppendLine("    mul.wide.u32 %rd6, %r11, 4;");
        ptx.AppendLine("    add.u64 %rd6, %rd4, %rd6;");             // aBase = gemm_a + ty*TM*BK*4
        ptx.AppendLine($"    mul.lo.u32 %r12, %r0, {I(tn)};");       // tx*TN
        ptx.AppendLine("    mul.wide.u32 %rd7, %r12, 4;");
        ptx.AppendLine("    add.u64 %rd7, %rd5, %rd7;");             // bBase = gemm_b + tx*TN*4
        // zero accumulators %f0..%f{TM*TN-1}
        for (int a = 0; a < tm * tn; a++) ptx.AppendLine($"    mov.f32 %f{I(a)}, 0f00000000;");
        ptx.AppendLine("    mov.u32 %r13, 0;");                      // t
        ptx.AppendLine("CONV_RB_LOOP:");
        // cooperative load As (loadsA per thread): l = tid + li*THREADS
        for (int li = 0; li < loadsA; li++)
        {
            ptx.AppendLine($"    add.u32 %r20, %r10, {I(li * threads)};");   // l
            ptx.AppendLine($"    shr.u32 %r21, %r20, {I(log2Bk)};");         // a_row = l>>log2BK
            ptx.AppendLine($"    and.b32 %r22, %r20, {I(bk - 1)};");         // a_col = l&(BK-1)
            ptx.AppendLine("    add.u32 %r23, %r4, %r21;");                  // gRow = rowTileBase + a_row
            ptx.AppendLine($"    mad.lo.u32 %r24, %r13, {I(bk)}, %r22;");    // gCol = t*BK + a_col
            ptx.AppendLine($"    mad.lo.u32 %r25, %r23, {I(c)}, %r24;");     // gRow*C + gCol
            ptx.AppendLine("    mul.wide.u32 %rd10, %r25, 4;");
            ptx.AppendLine("    add.u64 %rd10, %rd1, %rd10;");
            ptx.AppendLine("    ld.global.nc.f32 %f40, [%rd10];");
            ptx.AppendLine("    mul.wide.u32 %rd11, %r20, 4;");
            ptx.AppendLine("    add.u64 %rd11, %rd4, %rd11;");
            ptx.AppendLine("    st.shared.f32 [%rd11], %f40;");
        }
        // cooperative load Bs
        for (int li = 0; li < loadsB; li++)
        {
            ptx.AppendLine($"    add.u32 %r20, %r10, {I(li * threads)};");   // l
            ptx.AppendLine($"    shr.u32 %r21, %r20, {I(log2Bn)};");         // b_row = l>>log2BN
            ptx.AppendLine($"    and.b32 %r22, %r20, {I(bn - 1)};");         // b_col = l&(BN-1)
            ptx.AppendLine($"    mad.lo.u32 %r23, %r13, {I(bk)}, %r21;");    // gChan = t*BK + b_row
            ptx.AppendLine($"    mad.lo.u32 %r24, %r7, {I(c)}, %r23;");      // n*C + gChan
            ptx.AppendLine($"    mad.lo.u32 %r25, %r24, {I(hw)}, %r8;");     // *HW + hwBase
            ptx.AppendLine("    add.u32 %r25, %r25, %r22;");                 // + b_col
            ptx.AppendLine("    mul.wide.u32 %rd10, %r25, 4;");
            ptx.AppendLine("    add.u64 %rd10, %rd0, %rd10;");
            ptx.AppendLine("    ld.global.nc.f32 %f40, [%rd10];");
            ptx.AppendLine("    mul.wide.u32 %rd11, %r20, 4;");
            ptx.AppendLine("    add.u64 %rd11, %rd5, %rd11;");
            ptx.AppendLine("    st.shared.f32 [%rd11], %f40;");
        }
        ptx.AppendLine("    bar.sync 0;");
        // compute: unrolled over kk, m (A frag %f16..), nn (B frag %f24..)
        for (int kk = 0; kk < bk; kk++)
        {
            for (int m = 0; m < tm; m++)
                ptx.AppendLine($"    ld.shared.f32 %f{I(16 + m)}, [%rd6+{I((m * bk + kk) * 4)}];");
            for (int nn = 0; nn < tn; nn++)
                ptx.AppendLine($"    ld.shared.f32 %f{I(24 + nn)}, [%rd7+{I((kk * bn + nn) * 4)}];");
            for (int m = 0; m < tm; m++)
                for (int nn = 0; nn < tn; nn++)
                    ptx.AppendLine($"    fma.rn.f32 %f{I(m * tn + nn)}, %f{I(16 + m)}, %f{I(24 + nn)}, %f{I(m * tn + nn)};");
        }
        ptx.AppendLine("    bar.sync 0;");
        ptx.AppendLine("    add.u32 %r13, %r13, 1;");
        ptx.AppendLine($"    setp.lt.u32 %p0, %r13, {I(contractionTiles)};");
        ptx.AppendLine("    @%p0 bra CONV_RB_LOOP;");
        // epilogue: for each micro-tile output, +bias[row], ReLU, store
        for (int m = 0; m < tm; m++)
        {
            ptx.AppendLine($"    add.u32 %r30, %r5, {I(m)};");          // row = rowBase + m
            ptx.AppendLine("    mul.wide.u32 %rd12, %r30, 4;");
            ptx.AppendLine("    add.u64 %rd12, %rd2, %rd12;");
            ptx.AppendLine("    ld.global.nc.f32 %f44, [%rd12];");      // biasVal
            ptx.AppendLine($"    mad.lo.u32 %r31, %r7, {I(k)}, %r30;"); // n*K + row
            ptx.AppendLine($"    mul.lo.u32 %r31, %r31, {I(hw)};");     // (n*K+row)*HW
            ptx.AppendLine("    add.u32 %r31, %r31, %r9;");             // + hwFirst
            for (int nn = 0; nn < tn; nn++)
            {
                ptx.AppendLine($"    add.rn.f32 %f45, %f{I(m * tn + nn)}, %f44;");
                ptx.AppendLine("    max.f32 %f45, %f45, 0f00000000;");
                ptx.AppendLine($"    add.u32 %r32, %r31, {I(nn)};");    // output linear index
                ptx.AppendLine("    mul.wide.u32 %rd13, %r32, 4;");
                ptx.AppendLine("    add.u64 %rd13, %rd3, %rd13;");
                ptx.AppendLine("    st.global.f32 [%rd13], %f45;");
            }
        }
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static int Log2(int v)
    {
        int r = 0;
        while ((v >>= 1) != 0) r++;
        return r;
    }

    public void Dispose() => _module.Dispose();
}
