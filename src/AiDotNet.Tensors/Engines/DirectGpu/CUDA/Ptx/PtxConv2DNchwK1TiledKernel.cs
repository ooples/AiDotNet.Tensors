using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Exact shape descriptor for the shared-memory tiled 1x1 convolution GEMM. A
/// 1x1 NCHW convolution is the batched GEMM O[n,k,hw] = ReLU(bias[k] +
/// sum_c W[k,c] * X[n,c,hw]); this kernel stages a TILE x TILE weight block and
/// a TILE x TILE input block into shared memory per contraction step so each
/// element is fetched from global memory once per block instead of once per
/// output thread. The tile is chosen to divide K, C, and the spatial extent
/// exactly (issue #841 forbids dynamic-shape PTX), which removes every boundary
/// predicate: a column tile never straddles a batch because TILE divides HW.
/// </summary>
internal readonly struct Conv2DTiledShape
{
    internal int Batch { get; }
    internal int InputChannels { get; }
    internal int OutputChannels { get; }
    internal int Spatial { get; }
    internal int Tile { get; }

    internal Conv2DTiledShape(int batch, int outputChannels, int inputChannels, int spatial, int tile)
    {
        if (batch <= 0 || outputChannels <= 0 || inputChannels <= 0 || spatial <= 0 || tile <= 0)
            throw new ArgumentOutOfRangeException(nameof(tile), "All tiled-conv dimensions must be positive.");
        if (outputChannels % tile != 0 || inputChannels % tile != 0 || spatial % tile != 0)
            throw new ArgumentException(
                "The tile must divide K, C, and the spatial extent exactly so no boundary predicate is needed.");
        Batch = batch;
        OutputChannels = outputChannels;
        InputChannels = inputChannels;
        Spatial = spatial;
        Tile = tile;
    }

    internal long InputBytes => (long)Batch * InputChannels * Spatial * sizeof(float);
    internal long WeightBytes => (long)OutputChannels * InputChannels * sizeof(float);
    internal long BiasBytes => (long)OutputChannels * sizeof(float);
    internal long OutputBytes => (long)Batch * OutputChannels * Spatial * sizeof(float);

    internal string EntryPoint => FormattableString.Invariant(
        $"aidotnet_conv2d_n{Batch}_c{InputChannels}_hw{Spatial}_k{OutputChannels}_k1_tiled{Tile}_bias_relu");
    internal string Variant => FormattableString.Invariant(
        $"n{Batch}-c{InputChannels}-hw{Spatial}-k{OutputChannels}-r1-s1-p0-tile{Tile}-fp32");
}

/// <summary>
/// Shared-memory tiled FP32 NCHW 1x1 convolution GEMM with fused bias and ReLU.
/// The full geometry (N, C, K, spatial, tile) is baked into PTX, so the device
/// receives only four pointers and contains no runtime shape, stride, layout,
/// bounds, or epilogue selection.
///
/// NOTE (issue #841 promotion track): this emitter is drafted ahead of the GPU
/// measurement window. Its correctness (&lt;= 2e-4 vs the CPU oracle), register
/// budget, occupancy, tile-size sweep, and the >=1.10x-vs-cuDNN performance gate
/// are all pending verification on the RTX 3080. The tile sizes below are the
/// starting point of that sweep, not a measured optimum.
/// </summary>
internal sealed class PtxConv2DNchwK1TiledKernel : IDisposable
{
    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal Conv2DTiledShape Shape { get; }
    internal string Ptx { get; }
    internal DirectPtxFunctionInfo FunctionInfo { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal int BlockThreads => Shape.Tile * Shape.Tile;

    internal PtxConv2DNchwK1TiledKernel(DirectPtxRuntime runtime, Conv2DTiledShape shape)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new NotSupportedException(
                $"Tiled convolution has no experimental SM {runtime.ComputeCapabilityMajor}.{runtime.ComputeCapabilityMinor} specialization.");
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
        DirectPtxArchitectureFamily architecture, Conv2DTiledShape shape)
    {
        var input = new DirectPtxExtent(shape.Batch, shape.InputChannels, shape.Spatial);
        var weights = new DirectPtxExtent(shape.OutputChannels, shape.InputChannels);
        var bias = new DirectPtxExtent(shape.OutputChannels);
        var output = new DirectPtxExtent(shape.Batch, shape.OutputChannels, shape.Spatial);
        int sharedBytes = 2 * shape.Tile * shape.Tile * sizeof(float);
        return new DirectPtxKernelBlueprint(
            Operation: "conv2d-bias-relu-tiled",
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
                // Starting envelope for the GPU-window tile-size sweep, not a
                // measured optimum. Register/occupancy budgets are validated on
                // the device at construction; these are deliberately generous.
                MaxRegistersPerThread: 48,
                MaxStaticSharedBytes: sharedBytes,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 1),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["equation"] = "relu(conv2d-nchw-1x1(input,oihw-weights)+bias)",
                ["algorithm"] = "shared-memory-tiled-gemm",
                ["tile"] = shape.Tile.ToString(CultureInfo.InvariantCulture),
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

        uint gridX = (uint)(Shape.Batch * Shape.Spatial / Shape.Tile);
        uint gridY = (uint)(Shape.OutputChannels / Shape.Tile);
        _module.Launch(
            _function, gridX, gridY, 1,
            (uint)Shape.Tile, (uint)Shape.Tile, 1, 0, arguments);
    }

    private static void Require(
        DirectPtxTensorView view,
        DirectPtxTensorContract contract,
        string parameter)
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
    /// Emits the tiled GEMM PTX. Each block computes a TILE x TILE output tile
    /// (rows over K, columns over batched HW); the contraction over C is walked
    /// in TILE-wide steps, with the weight and input tiles staged into shared
    /// memory and reused by every thread in the block.
    /// </summary>
    internal static string EmitPtx(
        int computeCapabilityMajor, int computeCapabilityMinor, Conv2DTiledShape shape)
    {
        if (!DirectPtxArchitecture.HasExperimentalConvolution(
                computeCapabilityMajor, computeCapabilityMinor))
            throw new NotSupportedException("Only the experimental SM86 tiled convolution emitter exists.");

        int tile = shape.Tile;
        int k = shape.OutputChannels;
        int c = shape.InputChannels;
        int hw = shape.Spatial;
        int contractionTiles = c / tile;
        int tileBytes = tile * tile * sizeof(float);
        string entry = shape.EntryPoint;

        var ptx = new StringBuilder(8192);
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
        ptx.AppendLine("    .reg .b32 %r<24>;");
        ptx.AppendLine("    .reg .b64 %rd<20>;");
        ptx.AppendLine("    .reg .f32 %f<8>;");
        ptx.AppendLine($"    .shared .align 4 .b8 conv_tile_a[{tileBytes.ToString(CultureInfo.InvariantCulture)}];");
        ptx.AppendLine($"    .shared .align 4 .b8 conv_tile_b[{tileBytes.ToString(CultureInfo.InvariantCulture)}];");
        ptx.AppendLine("    ld.param.u64 %rd0, [input_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [weights_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [bias_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd3, [output_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");    // column within tile
        ptx.AppendLine("    mov.u32 %r1, %tid.y;");    // row within tile
        ptx.AppendLine("    mov.u32 %r2, %ctaid.x;");  // column-tile index
        ptx.AppendLine("    mov.u32 %r3, %ctaid.y;");  // row-tile index
        ptx.AppendLine($"    mad.lo.u32 %r4, %r3, {tile}, %r1;");  // row = K index
        ptx.AppendLine($"    mul.lo.u32 %r5, %r2, {tile};");       // colTileBase
        ptx.AppendLine($"    div.u32 %r6, %r5, {hw};");            // n = colTileBase / HW
        ptx.AppendLine($"    rem.u32 %r7, %r5, {hw};");            // hwBase = colTileBase % HW
        ptx.AppendLine("    add.u32 %r8, %r7, %r0;");              // hw = hwBase + tid.x
        // shared bases + per-thread cooperative-store slot
        ptx.AppendLine("    mov.u64 %rd4, conv_tile_a;");
        ptx.AppendLine("    mov.u64 %rd5, conv_tile_b;");
        ptx.AppendLine($"    mad.lo.u32 %r9, %r1, {tile}, %r0;");  // tid.y*TILE + tid.x
        ptx.AppendLine("    mul.wide.u32 %rd6, %r9, 4;");
        ptx.AppendLine("    add.u64 %rd7, %rd4, %rd6;");           // A store slot
        ptx.AppendLine("    add.u64 %rd8, %rd5, %rd6;");           // B store slot
        // inner-product row/column bases in shared
        ptx.AppendLine($"    mul.lo.u32 %r10, %r1, {tile};");      // tid.y*TILE
        ptx.AppendLine("    mul.wide.u32 %rd9, %r10, 4;");
        ptx.AppendLine("    add.u64 %rd9, %rd4, %rd9;");           // A row base
        ptx.AppendLine("    mul.wide.u32 %rd10, %r0, 4;");         // tid.x*4
        ptx.AppendLine("    add.u64 %rd10, %rd5, %rd10;");         // B column base
        ptx.AppendLine("    mov.f32 %f0, 0f00000000;");           // accumulator
        ptx.AppendLine("    mov.u32 %r11, 0;");                    // t (contraction tile)
        ptx.AppendLine("CONV_TILE_LOOP:");
        // A[tid.y][tid.x] = weights[row*C + t*TILE + tid.x]
        ptx.AppendLine($"    mad.lo.u32 %r12, %r11, {tile}, %r0;");
        ptx.AppendLine($"    mad.lo.u32 %r13, %r4, {c}, %r12;");
        ptx.AppendLine("    mul.wide.u32 %rd11, %r13, 4;");
        ptx.AppendLine("    add.u64 %rd11, %rd1, %rd11;");
        ptx.AppendLine("    ld.global.nc.f32 %f1, [%rd11];");
        ptx.AppendLine("    st.shared.f32 [%rd7], %f1;");
        // B[tid.y][tid.x] = input[(n*C + t*TILE + tid.y)*HW + hw]
        ptx.AppendLine($"    mad.lo.u32 %r14, %r11, {tile}, %r1;");
        ptx.AppendLine($"    mad.lo.u32 %r15, %r6, {c}, %r14;");
        ptx.AppendLine($"    mad.lo.u32 %r16, %r15, {hw}, %r8;");
        ptx.AppendLine("    mul.wide.u32 %rd12, %r16, 4;");
        ptx.AppendLine("    add.u64 %rd12, %rd0, %rd12;");
        ptx.AppendLine("    ld.global.nc.f32 %f2, [%rd12];");
        ptx.AppendLine("    st.shared.f32 [%rd8], %f2;");
        ptx.AppendLine("    bar.sync 0;");
        // inner product over the staged tile (unrolled)
        for (int kk = 0; kk < tile; kk++)
        {
            int aOffset = kk * sizeof(float);
            int bOffset = kk * tile * sizeof(float);
            ptx.AppendLine($"    ld.shared.f32 %f3, [%rd9+{aOffset.ToString(CultureInfo.InvariantCulture)}];");
            ptx.AppendLine($"    ld.shared.f32 %f4, [%rd10+{bOffset.ToString(CultureInfo.InvariantCulture)}];");
            ptx.AppendLine("    fma.rn.f32 %f0, %f3, %f4, %f0;");
        }
        ptx.AppendLine("    bar.sync 0;");
        ptx.AppendLine("    add.u32 %r11, %r11, 1;");
        ptx.AppendLine($"    setp.lt.u32 %p0, %r11, {contractionTiles};");
        ptx.AppendLine("    @%p0 bra CONV_TILE_LOOP;");
        // epilogue: + bias[row], ReLU, store output[(n*K + row)*HW + hw]
        ptx.AppendLine("    mul.wide.u32 %rd13, %r4, 4;");
        ptx.AppendLine("    add.u64 %rd13, %rd2, %rd13;");
        ptx.AppendLine("    ld.global.nc.f32 %f5, [%rd13];");
        ptx.AppendLine("    add.rn.f32 %f0, %f0, %f5;");
        ptx.AppendLine("    max.f32 %f0, %f0, 0f00000000;");
        ptx.AppendLine($"    mad.lo.u32 %r17, %r6, {k}, %r4;");
        ptx.AppendLine($"    mad.lo.u32 %r18, %r17, {hw}, %r8;");
        ptx.AppendLine("    mul.wide.u32 %rd14, %r18, 4;");
        ptx.AppendLine("    add.u64 %rd14, %rd3, %rd14;");
        ptx.AppendLine("    st.global.f32 [%rd14], %f0;");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    public void Dispose() => _module.Dispose();
}
