using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Exact shape descriptor for the Winograd F(2,3) 3x3 stride-1 same-padded
/// convolution. Winograd F(2,3) computes a 2x2 output tile from a 4x4 input tile
/// with 16 element-wise products per (channel) instead of the 36 multiplies a
/// direct 2x2x3x3 would use — the 2.25x multiply reduction cuDNN uses for 3x3,
/// which is why a direct/implicit-GEMM 3x3 cannot beat cuDNN and Winograd can.
///
/// <para>Issue #841 forbids dynamic-shape PTX: N, C, H, W, K are baked, and H, W
/// must be even so the output tiles a whole 2x2 grid with no partial tiles.</para>
/// </summary>
internal readonly struct Conv2DWinogradShape
{
    internal int Batch { get; }
    internal int InputChannels { get; }
    internal int Height { get; }
    internal int Width { get; }
    internal int OutputChannels { get; }

    /// <summary>When true the "weights" tensor is the precomputed filter transform
    /// U[K,C,4,4] (produced by <see cref="PtxWinogradF23FilterTransformKernel"/>),
    /// so the kernel skips the per-tile G g G^T recomputation.</summary>
    internal bool FilterPretransformed { get; }

    internal Conv2DWinogradShape(
        int batch, int inputChannels, int height, int width, int outputChannels,
        bool filterPretransformed = false)
    {
        if (batch <= 0 || inputChannels <= 0 || height <= 0 || width <= 0 || outputChannels <= 0)
            throw new ArgumentOutOfRangeException(nameof(batch), "Dimensions must be positive.");
        if ((height & 1) != 0 || (width & 1) != 0)
            throw new ArgumentException("Winograd F(2,3) bakes an even H and W (whole 2x2 output tiling).");
        Batch = batch; InputChannels = inputChannels; Height = height; Width = width; OutputChannels = outputChannels;
        FilterPretransformed = filterPretransformed;
    }

    internal int TileRows => Height / 2;
    internal int TileCols => Width / 2;
    internal int TotalTiles => Batch * OutputChannels * TileRows * TileCols;
    internal long InputBytes => (long)Batch * InputChannels * Height * Width * sizeof(float);
    internal long WeightBytes =>
        (long)OutputChannels * InputChannels * (FilterPretransformed ? 16 : 9) * sizeof(float);
    internal long BiasBytes => (long)OutputChannels * sizeof(float);
    internal long OutputBytes => (long)Batch * OutputChannels * Height * Width * sizeof(float);

    private string Suffix => FilterPretransformed ? "_pretf" : "";
    internal string EntryPoint => FormattableString.Invariant(
        $"aidotnet_conv2d_n{Batch}_c{InputChannels}_h{Height}_w{Width}_k{OutputChannels}_3x3s1p1_winograd_f23{Suffix}_bias_relu");
    internal string Variant => FormattableString.Invariant(
        $"n{Batch}-c{InputChannels}-h{Height}-w{Width}-k{OutputChannels}-r3-s1-p1-winograd-f23{Suffix}-fp32");
}

/// <summary>
/// Winograd F(2,3) 3x3 stride-1 same-padded convolution with fused bias and ReLU.
/// Correctness-first: one thread computes one 2x2 output tile for one (image,
/// output-channel), looping over input channels and transforming the 4x4 input
/// and 3x3 filter tiles inline. Fully baked geometry; four device pointers.
///
/// NOTE (#841): drafted for the GPU window. Correctness (&lt;= 1e-3 vs the fp64
/// direct-convolution oracle — Winograd rounds slightly differently), the SASS
/// zero-spill gate, and the &gt;=1.10x-vs-cuDNN proof are validated on-device;
/// the inline-filter-transform layout is the correctness baseline that the
/// register-blocked / filter-pretransform optimization then beats.
/// </summary>
internal sealed class PtxConv2DNchw3x3WinogradF23Kernel : IDisposable
{
    internal const int BlockThreads = 128;

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal Conv2DWinogradShape Shape { get; }
    internal string Ptx { get; }
    internal DirectPtxFunctionInfo FunctionInfo { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxConv2DNchw3x3WinogradF23Kernel(DirectPtxRuntime runtime, Conv2DWinogradShape shape)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new NotSupportedException(
                $"Winograd convolution has no experimental SM {runtime.ComputeCapabilityMajor}.{runtime.ComputeCapabilityMinor} specialization.");
        if (shape.TotalTiles % BlockThreads != 0)
            throw new ArgumentException(
                $"Winograd tile count {shape.TotalTiles} must be a multiple of {BlockThreads} (baked exact launch).");
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
        DirectPtxArchitectureFamily architecture, Conv2DWinogradShape shape)
    {
        var input = new DirectPtxExtent(shape.Batch, shape.InputChannels, shape.Height, shape.Width);
        var weights = shape.FilterPretransformed
            ? new DirectPtxExtent(shape.OutputChannels, shape.InputChannels, 4, 4)   // precomputed U
            : new DirectPtxExtent(shape.OutputChannels, shape.InputChannels, 3, 3);  // raw g
        var bias = new DirectPtxExtent(shape.OutputChannels);
        var output = new DirectPtxExtent(shape.Batch, shape.OutputChannels, shape.Height, shape.Width);
        return new DirectPtxKernelBlueprint(
            Operation: "conv2d-3x3-s1p1-bias-relu-winograd-f23",
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
                // Correctness-first inline-transform layout is register-heavy;
                // validated on-device. Tightened once the optimized layout lands.
                MaxRegistersPerThread: 168,
                MaxStaticSharedBytes: 0,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 1),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["equation"] = "relu(conv2d-3x3-same(input,weights)+bias)",
                ["algorithm"] = "winograd-f23",
                ["output-tile"] = "2x2",
                ["input-tile"] = "4x4",
                ["input"] = "fp32",
                ["accumulator"] = "fp32-fma",
                ["output"] = "fp32",
                ["layout"] = "nchw/oihw",
                ["padding"] = "same-1",
                ["shape-selection"] = "host-only-exact-contract",
                ["promotion"] = "experimental-pending-gpu-evidence"
            });
    }

    internal unsafe void Launch(
        DirectPtxTensorView input, DirectPtxTensorView weights,
        DirectPtxTensorView bias, DirectPtxTensorView output)
    {
        Require(input, Blueprint.Tensors[0], nameof(input));
        Require(weights, Blueprint.Tensors[1], nameof(weights));
        Require(bias, Blueprint.Tensors[2], nameof(bias));
        Require(output, Blueprint.Tensors[3], nameof(output));

        IntPtr inputPointer = input.Pointer, weightPointer = weights.Pointer,
               biasPointer = bias.Pointer, outputPointer = output.Pointer;
        void** arguments = stackalloc void*[4];
        arguments[0] = &inputPointer; arguments[1] = &weightPointer;
        arguments[2] = &biasPointer; arguments[3] = &outputPointer;
        _module.Launch(
            _function, (uint)(Shape.TotalTiles / BlockThreads), 1, 1,
            BlockThreads, 1, 1, 0, arguments);
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

    /// <summary>
    /// Emits the F(2,3) kernel. Registers used: %f0..%f15 = M accumulators;
    /// %f16..%f31 = current V (input transform); %f32..%f47 = current U (filter
    /// transform); %f48+ = scratch. All 16 M held across the channel loop.
    /// </summary>
    internal static string EmitPtx(
        int computeCapabilityMajor, int computeCapabilityMinor, Conv2DWinogradShape shape)
    {
        if (!DirectPtxArchitecture.HasExperimentalConvolution(
                computeCapabilityMajor, computeCapabilityMinor))
            throw new NotSupportedException("Only the experimental SM86 Winograd emitter exists.");

        int n = shape.Batch, c = shape.InputChannels, h = shape.Height, w = shape.Width, k = shape.OutputChannels;
        int tw = shape.TileCols, th = shape.TileRows;
        int tilesPerK = th * tw;
        int tilesPerN = k * tilesPerK;
        string entry = shape.EntryPoint;
        string I(int v) => v.ToString(CultureInfo.InvariantCulture);
        bool filterPretransformed = shape.FilterPretransformed;
        int chw = c * h * w;      // input image stride
        int hw = h * w;           // input channel stride
        int wStride = w;          // input/output row stride
        int wc9 = c * (filterPretransformed ? 16 : 9);  // weights (or U) per output channel

        var p = new StringBuilder(65536);
        p.AppendLine(".version 7.1");
        p.AppendLine($".target sm_{computeCapabilityMajor}{computeCapabilityMinor}");
        p.AppendLine(".address_size 64");
        p.AppendLine();
        p.AppendLine($".visible .entry {entry}(");
        p.AppendLine("    .param .u64 input_ptr,");
        p.AppendLine("    .param .u64 weights_ptr,");
        p.AppendLine("    .param .u64 bias_ptr,");
        p.AppendLine("    .param .u64 output_ptr");
        p.AppendLine(")");
        p.AppendLine("{");
        p.AppendLine("    .reg .pred %p<6>;");
        p.AppendLine("    .reg .b32 %r<48>;");
        p.AppendLine("    .reg .b64 %rd<32>;");
        p.AppendLine("    .reg .f32 %f<96>;");
        p.AppendLine("    ld.param.u64 %rd0, [input_ptr];");
        p.AppendLine("    ld.param.u64 %rd1, [weights_ptr];");
        p.AppendLine("    ld.param.u64 %rd2, [bias_ptr];");
        p.AppendLine("    ld.param.u64 %rd3, [output_ptr];");
        p.AppendLine("    mov.u32 %r0, %tid.x;");
        p.AppendLine("    mov.u32 %r1, %ctaid.x;");
        p.AppendLine($"    mad.lo.u32 %r2, %r1, {I(BlockThreads)}, %r0;");   // global tile id
        // decode id -> n, k, ti, tj
        p.AppendLine($"    rem.u32 %r3, %r2, {I(tw)};");                     // tj
        p.AppendLine($"    div.u32 %r4, %r2, {I(tw)};");
        p.AppendLine($"    rem.u32 %r5, %r4, {I(th)};");                     // ti
        p.AppendLine($"    div.u32 %r6, %r4, {I(th)};");
        p.AppendLine($"    rem.u32 %r7, %r6, {I(k)};");                      // k
        p.AppendLine($"    div.u32 %r8, %r6, {I(k)};");                      // n
        // input tile origin (pad 1): ih0 = 2*ti - 1, iw0 = 2*tj - 1  (signed)
        p.AppendLine("    mul.lo.s32 %r9, %r5, 2;");
        p.AppendLine("    sub.s32 %r9, %r9, 1;");                            // ih0
        p.AppendLine("    mul.lo.s32 %r10, %r3, 2;");
        p.AppendLine("    sub.s32 %r10, %r10, 1;");                          // iw0
        // input image base pointer for this n: input + n*C*H*W*4
        p.AppendLine($"    mul.wide.u32 %rd4, %r8, {I(chw)};");
        p.AppendLine("    shl.b64 %rd4, %rd4, 2;");                          // *4 bytes
        p.AppendLine("    add.u64 %rd4, %rd0, %rd4;");                       // input + n image
        // weights base for this k: weights + k*C*9*4
        p.AppendLine($"    mul.wide.u32 %rd5, %r7, {I(wc9)};");
        p.AppendLine("    shl.b64 %rd5, %rd5, 2;");
        p.AppendLine("    add.u64 %rd5, %rd1, %rd5;");                       // weights + k
        // zero M accumulators %f0..%f15
        for (int i = 0; i < 16; i++) p.AppendLine($"    mov.f32 %f{I(i)}, 0f00000000;");
        // channel loop
        p.AppendLine("    mov.u32 %r11, 0;");                               // c index
        p.AppendLine("    mov.u64 %rd6, %rd4;");                            // running input-channel ptr (n image + c*HW)
        p.AppendLine("    mov.u64 %rd7, %rd5;");                            // running weight ptr (k + c*9)
        p.AppendLine("WINO_C_LOOP:");
        EmitLoad4x4Input(p, h, w, wStride);      // loads d into %f48..%f63 (with zero-pad predication)
        EmitInputTransform(p);                    // V -> %f16..%f31 (consumes %f48..%f63, scratch %f64..)
        if (filterPretransformed)
        {
            // Read the 16 precomputed U values for (k,c): U[(k*C+c)*16 + i].
            for (int i = 0; i < 16; i++)
                p.AppendLine($"    ld.global.nc.f32 %f{I(32 + i)}, [%rd7+{I(i * 4)}];");
        }
        else
        {
            EmitLoadFilter3x3(p);                 // g into %f48..%f56
            EmitFilterTransform(p);               // U -> %f32..%f47
        }
        for (int i = 0; i < 16; i++)              // M += U*V
            p.AppendLine($"    fma.rn.f32 %f{I(i)}, %f{I(32 + i)}, %f{I(16 + i)}, %f{I(i)};");
        p.AppendLine($"    add.u64 %rd6, %rd6, {I(hw * 4)};");             // next channel input
        p.AppendLine($"    add.u64 %rd7, %rd7, {I(filterPretransformed ? 64 : 36)};"); // next channel U(16*4) or g(9*4)
        p.AppendLine("    add.u32 %r11, %r11, 1;");
        p.AppendLine($"    setp.lt.u32 %p0, %r11, {I(c)};");
        p.AppendLine("    @%p0 bra WINO_C_LOOP;");
        // output transform Y = A^T M A -> %f64..%f67 (2x2)
        EmitOutputTransform(p);
        // + bias[k], ReLU, store output[n,k, 2*ti+oi, 2*tj+oj]
        p.AppendLine($"    mul.wide.u32 %rd8, %r7, 4;");
        p.AppendLine("    add.u64 %rd8, %rd2, %rd8;");
        p.AppendLine("    ld.global.nc.f32 %f70, [%rd8];");                // bias[k]
        // output base for (n,k): output + (n*K + k)*H*W*4 + (2*ti)*W*4 + (2*tj)*4
        p.AppendLine($"    mad.lo.u32 %r12, %r8, {I(k)}, %r7;");           // n*K + k
        p.AppendLine($"    mul.wide.u32 %rd9, %r12, {I(hw)};");
        p.AppendLine("    shl.b64 %rd9, %rd9, 2;");
        p.AppendLine("    add.u64 %rd9, %rd3, %rd9;");                     // output + (n,k) plane
        p.AppendLine("    mul.lo.u32 %r13, %r5, 2;");                      // oh = 2*ti
        p.AppendLine("    mul.lo.u32 %r14, %r3, 2;");                      // ow = 2*tj
        // store 2x2: (oi,oj)
        for (int oi = 0; oi < 2; oi++)
            for (int oj = 0; oj < 2; oj++)
            {
                int yreg = 64 + oi * 2 + oj;
                p.AppendLine($"    add.rn.f32 %f{I(yreg)}, %f{I(yreg)}, %f70;");
                p.AppendLine($"    max.f32 %f{I(yreg)}, %f{I(yreg)}, 0f00000000;");
                // linear offset = (oh+oi)*W + (ow+oj)
                p.AppendLine($"    add.u32 %r15, %r13, {I(oi)};");
                p.AppendLine($"    mul.lo.u32 %r15, %r15, {I(wStride)};");
                p.AppendLine($"    add.u32 %r15, %r15, %r14;");
                p.AppendLine($"    add.u32 %r15, %r15, {I(oj)};");
                p.AppendLine("    mul.wide.u32 %rd10, %r15, 4;");
                p.AppendLine("    add.u64 %rd10, %rd9, %rd10;");
                p.AppendLine($"    st.global.f32 [%rd10], %f{I(yreg)};");
            }
        p.AppendLine("    ret;");
        p.AppendLine("}");
        return p.ToString();
    }

    // Loads the 4x4 input patch d[di][dj] = input[ n, c, ih0+di, iw0+dj ] into
    // %f48..%f63 (row-major), zero where out of [0,H)x[0,W). Uses %rd6 (channel
    // base), %r9 (ih0), %r10 (iw0). Scratch: %r16..%r19, %rd11..%rd12, %p1..%p4.
    private static void EmitLoad4x4Input(StringBuilder p, int h, int w, int wStride)
    {
        string I(int v) => v.ToString(CultureInfo.InvariantCulture);
        for (int di = 0; di < 4; di++)
            for (int dj = 0; dj < 4; dj++)
            {
                int reg = 48 + di * 4 + dj;
                p.AppendLine($"    mov.f32 %f{I(reg)}, 0f00000000;");
                // ih = ih0 + di ; iw = iw0 + dj
                p.AppendLine($"    add.s32 %r16, %r9, {I(di)};");
                p.AppendLine($"    add.s32 %r17, %r10, {I(dj)};");
                // in bounds: 0 <= ih < H && 0 <= iw < W
                p.AppendLine("    setp.ge.s32 %p1, %r16, 0;");
                p.AppendLine($"    setp.lt.s32 %p2, %r16, {I(h)};");
                p.AppendLine("    setp.ge.s32 %p3, %r17, 0;");
                p.AppendLine($"    setp.lt.s32 %p4, %r17, {I(w)};");
                p.AppendLine("    and.pred %p1, %p1, %p2;");
                p.AppendLine("    and.pred %p3, %p3, %p4;");
                p.AppendLine("    and.pred %p1, %p1, %p3;");
                // addr = channel base + (ih*W + iw)*4
                p.AppendLine($"    mul.lo.u32 %r18, %r16, {I(wStride)};");
                p.AppendLine("    add.u32 %r18, %r18, %r17;");
                p.AppendLine("    mul.wide.s32 %rd11, %r18, 4;");
                p.AppendLine("    add.u64 %rd11, %rd6, %rd11;");
                p.AppendLine($"    @%p1 ld.global.nc.f32 %f{I(reg)}, [%rd11];");
            }
    }

    // V = B^T d B, d in %f48..%f63, result V in %f16..%f31. Scratch t in %f64..%f79.
    private static void EmitInputTransform(StringBuilder p)
    {
        string I(int v) => v.ToString(CultureInfo.InvariantCulture);
        int D(int i, int j) => 48 + i * 4 + j;
        int T(int i, int j) => 64 + i * 4 + j;
        int V(int i, int j) => 16 + i * 4 + j;
        // t = B^T d : rows [d0-d2, d1+d2, d2-d1, d1-d3]
        for (int j = 0; j < 4; j++)
        {
            p.AppendLine($"    sub.rn.f32 %f{I(T(0, j))}, %f{I(D(0, j))}, %f{I(D(2, j))};");
            p.AppendLine($"    add.rn.f32 %f{I(T(1, j))}, %f{I(D(1, j))}, %f{I(D(2, j))};");
            p.AppendLine($"    sub.rn.f32 %f{I(T(2, j))}, %f{I(D(2, j))}, %f{I(D(1, j))};");
            p.AppendLine($"    sub.rn.f32 %f{I(T(3, j))}, %f{I(D(1, j))}, %f{I(D(3, j))};");
        }
        // V = t B : cols [t0-t2, t1+t2, t2-t1, t1-t3]
        for (int i = 0; i < 4; i++)
        {
            p.AppendLine($"    sub.rn.f32 %f{I(V(i, 0))}, %f{I(T(i, 0))}, %f{I(T(i, 2))};");
            p.AppendLine($"    add.rn.f32 %f{I(V(i, 1))}, %f{I(T(i, 1))}, %f{I(T(i, 2))};");
            p.AppendLine($"    sub.rn.f32 %f{I(V(i, 2))}, %f{I(T(i, 2))}, %f{I(T(i, 1))};");
            p.AppendLine($"    sub.rn.f32 %f{I(V(i, 3))}, %f{I(T(i, 1))}, %f{I(T(i, 3))};");
        }
    }

    // Loads g[gi][gj] = weights[k,c,gi,gj] into %f48..%f56 via %rd7 (running weight ptr).
    private static void EmitLoadFilter3x3(StringBuilder p)
    {
        string I(int v) => v.ToString(CultureInfo.InvariantCulture);
        for (int gi = 0; gi < 3; gi++)
            for (int gj = 0; gj < 3; gj++)
            {
                int reg = 48 + gi * 3 + gj;
                int off = (gi * 3 + gj) * 4;
                p.AppendLine($"    ld.global.nc.f32 %f{I(reg)}, [%rd7+{I(off)}];");
            }
    }

    // U = G g G^T, g in %f48..%f56 (3x3), result U in %f32..%f47 (4x4). Scratch u in %f64..%f75 (4x3).
    private static void EmitFilterTransform(StringBuilder p)
    {
        string I(int v) => v.ToString(CultureInfo.InvariantCulture);
        int G(int i, int j) => 48 + i * 3 + j;     // g 3x3
        int U3(int i, int j) => 64 + i * 3 + j;    // u 4x3
        int U(int i, int j) => 32 + i * 4 + j;     // U 4x4
        // u = G g : rows [g0, (g0+g1+g2)/2, (g0-g1+g2)/2, g2]
        for (int j = 0; j < 3; j++)
        {
            p.AppendLine($"    mov.f32 %f{I(U3(0, j))}, %f{I(G(0, j))};");
            p.AppendLine($"    add.rn.f32 %f80, %f{I(G(0, j))}, %f{I(G(1, j))};");
            p.AppendLine($"    add.rn.f32 %f80, %f80, %f{I(G(2, j))};");
            p.AppendLine($"    mul.rn.f32 %f{I(U3(1, j))}, %f80, 0f3F000000;");   // *0.5
            p.AppendLine($"    sub.rn.f32 %f81, %f{I(G(0, j))}, %f{I(G(1, j))};");
            p.AppendLine($"    add.rn.f32 %f81, %f81, %f{I(G(2, j))};");
            p.AppendLine($"    mul.rn.f32 %f{I(U3(2, j))}, %f81, 0f3F000000;");
            p.AppendLine($"    mov.f32 %f{I(U3(3, j))}, %f{I(G(2, j))};");
        }
        // U = u G^T : cols [u0, (u0+u1+u2)/2, (u0-u1+u2)/2, u2]
        for (int i = 0; i < 4; i++)
        {
            p.AppendLine($"    mov.f32 %f{I(U(i, 0))}, %f{I(U3(i, 0))};");
            p.AppendLine($"    add.rn.f32 %f80, %f{I(U3(i, 0))}, %f{I(U3(i, 1))};");
            p.AppendLine($"    add.rn.f32 %f80, %f80, %f{I(U3(i, 2))};");
            p.AppendLine($"    mul.rn.f32 %f{I(U(i, 1))}, %f80, 0f3F000000;");
            p.AppendLine($"    sub.rn.f32 %f81, %f{I(U3(i, 0))}, %f{I(U3(i, 1))};");
            p.AppendLine($"    add.rn.f32 %f81, %f81, %f{I(U3(i, 2))};");
            p.AppendLine($"    mul.rn.f32 %f{I(U(i, 2))}, %f81, 0f3F000000;");
            p.AppendLine($"    mov.f32 %f{I(U(i, 3))}, %f{I(U3(i, 2))};");
        }
    }

    // Y = A^T M A, M in %f0..%f15 (4x4), result Y in %f64..%f67 (2x2). Scratch s in %f80..%f87 (2x4).
    private static void EmitOutputTransform(StringBuilder p)
    {
        string I(int v) => v.ToString(CultureInfo.InvariantCulture);
        int M(int i, int j) => i * 4 + j;          // M 4x4 (%f0..15)
        int S(int i, int j) => 80 + i * 4 + j;     // s 2x4
        int Y(int i, int j) => 64 + i * 2 + j;     // Y 2x2
        // s = A^T M : row0 = M0+M1+M2 ; row1 = M1-M2-M3
        for (int j = 0; j < 4; j++)
        {
            p.AppendLine($"    add.rn.f32 %f{I(S(0, j))}, %f{I(M(0, j))}, %f{I(M(1, j))};");
            p.AppendLine($"    add.rn.f32 %f{I(S(0, j))}, %f{I(S(0, j))}, %f{I(M(2, j))};");
            p.AppendLine($"    sub.rn.f32 %f{I(S(1, j))}, %f{I(M(1, j))}, %f{I(M(2, j))};");
            p.AppendLine($"    sub.rn.f32 %f{I(S(1, j))}, %f{I(S(1, j))}, %f{I(M(3, j))};");
        }
        // Y = s A : col0 = s0+s1+s2 ; col1 = s1-s2-s3
        for (int i = 0; i < 2; i++)
        {
            p.AppendLine($"    add.rn.f32 %f{I(Y(i, 0))}, %f{I(S(i, 0))}, %f{I(S(i, 1))};");
            p.AppendLine($"    add.rn.f32 %f{I(Y(i, 0))}, %f{I(Y(i, 0))}, %f{I(S(i, 2))};");
            p.AppendLine($"    sub.rn.f32 %f{I(Y(i, 1))}, %f{I(S(i, 1))}, %f{I(S(i, 2))};");
            p.AppendLine($"    sub.rn.f32 %f{I(Y(i, 1))}, %f{I(Y(i, 1))}, %f{I(S(i, 3))};");
        }
    }

    public void Dispose() => _module.Dispose();
}
