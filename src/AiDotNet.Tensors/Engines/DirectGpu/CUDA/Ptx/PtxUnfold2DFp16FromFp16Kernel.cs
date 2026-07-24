using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Direct-PTX FP16 im2col from FP16 input (UnfoldKNFp16FromFp16): extracts sliding local
/// patches of an already-FP16 input[N,C,H,W] into FP16 columns[N, C*KH*KW, OH*OW] for a
/// downstream Tensor-Core GEMM. columns[n, c*KH*KW+kh*KW+kw, oh*OW+ow] =
/// input_fp16[n,c,oh*s+kh-pad,ow*s+kw-pad] (0 outside the padded input). One thread per
/// output element; consecutive threads own consecutive output-spatial index so at stride 1
/// the FP16 input reads and FP16 column stores coalesce (a pure half-to-half gather).
/// </summary>
internal sealed class PtxUnfold2DFp16FromFp16Kernel : IDisposable
{
    internal const int BlockThreads = 256;
    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int Batch { get; }
    internal int Channels { get; }
    internal int Height { get; }
    internal int Width { get; }
    internal int KernelH { get; }
    internal int KernelW { get; }
    internal int Stride { get; }
    internal int Padding { get; }
    internal string Ptx { get; }
    internal DirectPtxFunctionInfo FunctionInfo { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal int OutH => (Height + 2 * Padding - KernelH) / Stride + 1;
    internal int OutW => (Width + 2 * Padding - KernelW) / Stride + 1;
    internal int PatchRows => Channels * KernelH * KernelW;
    internal int Columns => OutH * OutW;
    internal long InputBytes => (long)Batch * Channels * Height * Width * sizeof(ushort);
    internal long OutputBytes => (long)Batch * PatchRows * Columns * sizeof(ushort);

    internal Unfold2DShape Shape => new(Batch, Channels, Height, Width, KernelH, KernelW, Stride, Padding);
    internal static string EntryFor(Unfold2DShape s) => FormattableString.Invariant(
        $"aidotnet_unfold_kn_fp16_from_fp16_n{s.Batch}_c{s.Channels}_h{s.Height}_w{s.Width}_kh{s.KernelH}_kw{s.KernelW}_s{s.Stride}_p{s.Padding}");
    internal string EntryPoint => EntryFor(Shape);

    internal PtxUnfold2DFp16FromFp16Kernel(
        DirectPtxRuntime runtime, int batch, int channels, int height, int width, int kernelH, int kernelW, int stride, int padding)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new NotSupportedException("UnfoldKNFp16FromFp16 has no experimental non-SM86 specialization.");
        if (batch <= 0 || channels <= 0 || height <= 0 || width <= 0 || kernelH <= 0 || kernelW <= 0 || stride <= 0 || padding < 0)
            throw new ArgumentOutOfRangeException(nameof(batch));
        Batch = batch; Channels = channels; Height = height; Width = width; KernelH = kernelH; KernelW = kernelW; Stride = stride; Padding = padding;
        if (OutH <= 0 || OutW <= 0) throw new ArgumentException("Non-positive output spatial.");
        if ((long)batch * PatchRows * Columns % BlockThreads != 0)
            throw new ArgumentException($"N*(C*KH*KW)*(OH*OW) must be a multiple of {BlockThreads}.");

        Unfold2DShape shape = Shape;
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, shape);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor, shape);
        _module = runtime.LoadModule(Ptx, allowExperimentalJitFallback: DirectPtxFeatureGate.ConvolutionExperimentOverride);
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo functionInfo);
        FunctionInfo = functionInfo;
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, BlockThreads);
        Blueprint.ResourceBudget.Validate(EntryPoint, functionInfo, BlockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(Blueprint, runtime.DeviceFingerprint, Ptx, functionInfo, BlockThreads, activeBlocks, _module);
    }

    internal static DirectPtxKernelBlueprint CreateBlueprint(DirectPtxArchitectureFamily architecture, Unfold2DShape shape)
    {
        int Batch = shape.Batch, Channels = shape.Channels, Height = shape.Height, Width = shape.Width;
        int KernelH = shape.KernelH, KernelW = shape.KernelW, Stride = shape.Stride, Padding = shape.Padding;
        int PatchRows = shape.PatchRows, Columns = shape.Columns;
        var input = new DirectPtxExtent(Batch, Channels, Height, Width);
        var output = new DirectPtxExtent(Batch, PatchRows, Columns);
        return new DirectPtxKernelBlueprint(
            Operation: "unfold-kn-fp16-from-fp16", Version: 1, Architecture: architecture,
            Variant: FormattableString.Invariant($"n{Batch}-c{Channels}-h{Height}-w{Width}-kh{KernelH}-kw{KernelW}-s{Stride}-p{Padding}-f16"),
            Tensors:
            [
                new("input", DirectPtxPhysicalType.Float16, DirectPtxPhysicalLayout.Nchw, input, input, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("columns", DirectPtxPhysicalType.Float16, DirectPtxPhysicalLayout.RowMajor2D, output, output, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(MaxRegistersPerThread: 32, MaxStaticSharedBytes: 0, MaxLocalBytesPerThread: 0, MinBlocksPerMultiprocessor: 2),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["equation"] = "columns_fp16[n,c*KH*KW+kh*KW+kw,oh*OW+ow] = input_fp16[n,c,oh*s+kh-pad,ow*s+kw-pad]",
                ["shape-selection"] = "host-only-exact-contract", ["promotion"] = "experimental-pending-gpu-evidence"
            });
    }

    internal unsafe void Launch(DirectPtxTensorView input, DirectPtxTensorView columns)
    {
        Require(input, Blueprint.Tensors[0], nameof(input));
        Require(columns, Blueprint.Tensors[1], nameof(columns));
        IntPtr iPtr = input.Pointer, oPtr = columns.Pointer;
        void** arguments = stackalloc void*[2];
        arguments[0] = &iPtr; arguments[1] = &oPtr;
        int total = Batch * PatchRows * Columns;
        _module.Launch(_function, (uint)(total / BlockThreads), 1, 1, BlockThreads, 1, 1, 0, arguments);
    }

    private static void Require(DirectPtxTensorView view, DirectPtxTensorContract contract, string parameter)
    {
        if (view.Pointer == IntPtr.Zero || view.PhysicalType != contract.PhysicalType || view.Layout != contract.Layout ||
            view.LogicalExtent != contract.LogicalExtent || view.PhysicalExtent != contract.PhysicalExtent ||
            view.ByteLength != contract.RequiredBytes || view.AllocationByteLength != contract.RequiredBytes)
            throw new ArgumentException($"{parameter} does not satisfy exact physical ABI '{contract.Name}'.", parameter);
    }

    internal static string EmitPtx(int major, int minor, Unfold2DShape shape)
    {
        if (!DirectPtxArchitecture.HasExperimentalConvolution(major, minor))
            throw new NotSupportedException("Only the experimental SM86 UnfoldKNFp16FromFp16 emitter exists.");
        string I(int v) => v.ToString(CultureInfo.InvariantCulture);
        int Stride = shape.Stride, Padding = shape.Padding, KernelH = shape.KernelH;
        int c = shape.Channels, h = shape.Height, w = shape.Width, kw = shape.KernelW, oww = shape.OutW;
        int khkw = KernelH * kw, hw = h * w, cols = shape.Columns, prc = shape.PatchRows * cols;
        string entry = EntryFor(shape);

        var s = new StringBuilder(12288);
        s.AppendLine(".version 7.1");
        s.AppendLine($".target sm_{major}{minor}");
        s.AppendLine(".address_size 64");
        s.AppendLine();
        s.AppendLine($".visible .entry {entry}(");
        s.AppendLine("    .param .u64 input_ptr,");
        s.AppendLine("    .param .u64 output_ptr");
        s.AppendLine(")");
        s.AppendLine("{");
        s.AppendLine("    .reg .pred %p<6>;");
        s.AppendLine("    .reg .b16 %hf<2>;");
        s.AppendLine("    .reg .b32 %r<28>;");
        s.AppendLine("    .reg .b64 %rd<12>;");
        s.AppendLine("    ld.param.u64 %rd0, [input_ptr];");
        s.AppendLine("    ld.param.u64 %rd1, [output_ptr];");
        s.AppendLine("    mov.u32 %r0, %tid.x;");
        s.AppendLine("    mov.u32 %r1, %ctaid.x;");
        s.AppendLine($"    mad.lo.u32 %r2, %r1, {I(BlockThreads)}, %r0;");  // idx
        s.AppendLine($"    div.u32 %r3, %r2, {I(prc)};");        // n
        s.AppendLine($"    rem.u32 %r4, %r2, {I(prc)};");
        s.AppendLine($"    div.u32 %r5, %r4, {I(cols)};");       // patchIdx
        s.AppendLine($"    rem.u32 %r6, %r4, {I(cols)};");       // spatialIdx
        s.AppendLine($"    div.u32 %r7, %r5, {I(khkw)};");       // c
        s.AppendLine($"    rem.u32 %r8, %r5, {I(khkw)};");
        s.AppendLine($"    div.u32 %r9, %r8, {I(kw)};");         // kh
        s.AppendLine($"    rem.u32 %r10, %r8, {I(kw)};");        // kw
        s.AppendLine($"    div.u32 %r11, %r6, {I(oww)};");       // oh
        s.AppendLine($"    rem.u32 %r12, %r6, {I(oww)};");       // ow
        s.AppendLine($"    mad.lo.u32 %r13, %r11, {I(Stride)}, %r9;");
        s.AppendLine($"    sub.s32 %r13, %r13, {I(Padding)};");  // ih
        s.AppendLine($"    mad.lo.u32 %r14, %r12, {I(Stride)}, %r10;");
        s.AppendLine($"    sub.s32 %r14, %r14, {I(Padding)};");  // iw
        s.AppendLine("    setp.ge.s32 %p0, %r13, 0;");
        s.AppendLine($"    setp.lt.s32 %p1, %r13, {I(h)};");
        s.AppendLine("    setp.ge.s32 %p2, %r14, 0;");
        s.AppendLine($"    setp.lt.s32 %p3, %r14, {I(w)};");
        s.AppendLine("    and.pred %p0, %p0, %p1;");
        s.AppendLine("    and.pred %p2, %p2, %p3;");
        s.AppendLine("    and.pred %p0, %p0, %p2;");
        s.AppendLine($"    mad.lo.u32 %r15, %r3, {I(c)}, %r7;");
        s.AppendLine($"    mad.lo.u32 %r15, %r15, {I(hw)}, 0;");
        s.AppendLine($"    mad.lo.u32 %r15, %r13, {I(w)}, %r15;");
        s.AppendLine("    add.u32 %r15, %r15, %r14;");
        s.AppendLine("    mul.wide.u32 %rd2, %r15, 2;");        // fp16 input
        s.AppendLine("    add.u64 %rd2, %rd0, %rd2;");
        s.AppendLine("    mov.b16 %hf0, 0;");
        s.AppendLine("    @%p0 ld.global.nc.b16 %hf0, [%rd2];");
        s.AppendLine("    mul.wide.u32 %rd3, %r2, 2;");
        s.AppendLine("    add.u64 %rd3, %rd1, %rd3;");
        s.AppendLine("    st.global.b16 [%rd3], %hf0;");
        s.AppendLine("    ret;");
        s.AppendLine("}");
        return s.ToString();
    }

    public void Dispose() => _module.Dispose();
}
