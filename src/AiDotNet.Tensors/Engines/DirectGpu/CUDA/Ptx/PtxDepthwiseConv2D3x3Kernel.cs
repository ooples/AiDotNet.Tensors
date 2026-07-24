using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Direct-PTX depthwise Conv2D forward (channel-multiplier 1), 3x3 stride-1 same-pad,
/// with per-channel bias and optional ReLU: out[n,c,oh,ow] = relu(bias[c] +
/// sum_{r,s} W[c,r,s] * in[n,c,oh+r-1,ow+s-1]). Each output channel depends only on
/// the matching input channel (no channel reduction), so this is memory-bound; one
/// thread per output element with consecutive threads owning consecutive ow (the
/// contiguous NCHW axis) makes the input and output accesses coalesced and the 9
/// depthwise weights broadcast across the warp. Applies the coalescing lesson.
/// </summary>
internal sealed class PtxDepthwiseConv2D3x3Kernel : IDisposable
{
    internal const int BlockThreads = 256;

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int Batch { get; }
    internal int Channels { get; }
    internal int Height { get; }
    internal int Width { get; }
    internal bool Relu { get; }
    internal string Ptx { get; }
    internal DirectPtxFunctionInfo FunctionInfo { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal long InputBytes => (long)Batch * Channels * Height * Width * sizeof(float);
    internal long WeightBytes => (long)Channels * 9 * sizeof(float);
    internal long BiasBytes => (long)Channels * sizeof(float);
    internal long OutputBytes => (long)Batch * Channels * Height * Width * sizeof(float);

    internal string EntryPoint => FormattableString.Invariant(
        $"aidotnet_depthwise_conv2d3x3_n{Batch}_c{Channels}_h{Height}_w{Width}{(Relu ? "_relu" : "")}");

    internal PtxDepthwiseConv2D3x3Kernel(
        DirectPtxRuntime runtime, int batch, int channels, int height, int width, bool relu = true)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new NotSupportedException("Depthwise Conv2D has no experimental non-SM86 specialization.");
        if (batch <= 0 || channels <= 0 || height <= 0 || width <= 0)
            throw new ArgumentOutOfRangeException(nameof(batch));
        Batch = batch; Channels = channels; Height = height; Width = width; Relu = relu;
        if ((long)batch * channels * height * width % BlockThreads != 0)
            throw new ArgumentException($"N*C*H*W must be a multiple of {BlockThreads}.");

        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, batch, channels, height, width, relu);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor, batch, channels, height, width, relu);
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
        DirectPtxArchitectureFamily architecture, int n, int c, int h, int w, bool relu)
    {
        var io = new DirectPtxExtent(n, c, h, w);
        var weight = new DirectPtxExtent(c, 1, 3, 3);
        var bias = new DirectPtxExtent(c);
        return new DirectPtxKernelBlueprint(
            Operation: "depthwise-conv2d-3x3",
            Version: 1,
            Architecture: architecture,
            Variant: FormattableString.Invariant($"n{n}-c{c}-h{h}-w{w}-r3{(relu ? "-relu" : "")}-fp32"),
            Tensors:
            [
                new("input", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Nchw,
                    io, io, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("weights", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Oihw,
                    weight, weight, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("bias", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    bias, bias, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Nchw,
                    io, io, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 40, MaxStaticSharedBytes: 0,
                MaxLocalBytesPerThread: 0, MinBlocksPerMultiprocessor: 2),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["equation"] = "out[n,c,oh,ow] = " + (relu ? "relu(" : "(") + "bias[c] + sum_{r,s} W[c,r,s]*in[n,c,oh+r-1,ow+s-1])",
                ["access"] = "thread-per-output, ow-contiguous -> coalesced in/out; weights broadcast",
                ["padding"] = "same-1 stride-1", ["shape-selection"] = "host-only-exact-contract",
                ["promotion"] = "experimental-pending-gpu-evidence"
            });
    }

    internal unsafe void Launch(DirectPtxTensorView input, DirectPtxTensorView weights, DirectPtxTensorView bias, DirectPtxTensorView output)
    {
        Require(input, Blueprint.Tensors[0], nameof(input));
        Require(weights, Blueprint.Tensors[1], nameof(weights));
        Require(bias, Blueprint.Tensors[2], nameof(bias));
        Require(output, Blueprint.Tensors[3], nameof(output));
        IntPtr iPtr = input.Pointer, wPtr = weights.Pointer, bPtr = bias.Pointer, oPtr = output.Pointer;
        void** arguments = stackalloc void*[4];
        arguments[0] = &iPtr; arguments[1] = &wPtr; arguments[2] = &bPtr; arguments[3] = &oPtr;
        int total = Batch * Channels * Height * Width;
        _module.Launch(_function, (uint)(total / BlockThreads), 1, 1, BlockThreads, 1, 1, 0, arguments);
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

    internal static string EmitPtx(int major, int minor, int n, int c, int h, int w, bool relu)
    {
        if (!DirectPtxArchitecture.HasExperimentalConvolution(major, minor))
            throw new NotSupportedException("Only the experimental SM86 depthwise emitter exists.");
        string I(int v) => v.ToString(CultureInfo.InvariantCulture);
        int hw = h * w;
        int chw = c * hw;
        string entry = FormattableString.Invariant(
            $"aidotnet_depthwise_conv2d3x3_n{n}_c{c}_h{h}_w{w}{(relu ? "_relu" : "")}");

        var s = new StringBuilder(12288);
        s.AppendLine(".version 7.1");
        s.AppendLine($".target sm_{major}{minor}");
        s.AppendLine(".address_size 64");
        s.AppendLine();
        s.AppendLine($".visible .entry {entry}(");
        s.AppendLine("    .param .u64 input_ptr,");
        s.AppendLine("    .param .u64 weight_ptr,");
        s.AppendLine("    .param .u64 bias_ptr,");
        s.AppendLine("    .param .u64 output_ptr");
        s.AppendLine(")");
        s.AppendLine("{");
        s.AppendLine("    .reg .pred %p<6>;");
        s.AppendLine("    .reg .b32 %r<24>;");
        s.AppendLine("    .reg .b64 %rd<16>;");
        s.AppendLine("    .reg .f32 %f<8>;");
        s.AppendLine("    ld.param.u64 %rd0, [input_ptr];");
        s.AppendLine("    ld.param.u64 %rd1, [weight_ptr];");
        s.AppendLine("    ld.param.u64 %rd2, [bias_ptr];");
        s.AppendLine("    ld.param.u64 %rd3, [output_ptr];");
        s.AppendLine("    mov.u32 %r0, %tid.x;");
        s.AppendLine("    mov.u32 %r1, %ctaid.x;");
        s.AppendLine($"    mad.lo.u32 %r2, %r1, {I(BlockThreads)}, %r0;");  // idx = n*C*HW + c*HW + oh*W + ow
        s.AppendLine($"    div.u32 %r3, %r2, {I(chw)};");          // n
        s.AppendLine($"    rem.u32 %r4, %r2, {I(chw)};");
        s.AppendLine($"    div.u32 %r5, %r4, {I(hw)};");           // c
        s.AppendLine($"    rem.u32 %r6, %r4, {I(hw)};");
        s.AppendLine($"    div.u32 %r7, %r6, {I(w)};");            // oh
        s.AppendLine($"    rem.u32 %r8, %r6, {I(w)};");            // ow
        // bias[c]
        s.AppendLine("    mul.wide.u32 %rd4, %r5, 4;");
        s.AppendLine("    add.u64 %rd4, %rd2, %rd4;");
        s.AppendLine("    ld.global.nc.f32 %f0, [%rd4];");        // acc = bias[c]
        // input channel base: input + (n*C + c)*HW  == idx - (oh*W+ow) ... recompute
        s.AppendLine($"    mad.lo.u32 %r9, %r3, {I(c)}, %r5;");   // n*C + c
        s.AppendLine($"    mul.wide.u32 %rd5, %r9, {I(hw)};");
        s.AppendLine("    shl.b64 %rd5, %rd5, 2;");
        s.AppendLine("    add.u64 %rd5, %rd0, %rd5;");            // &input[n][c][0][0]
        // weight channel base: weights + c*9
        s.AppendLine("    mul.lo.u32 %r10, %r5, 9;");
        for (int t = 0; t < 9; t++)
        {
            int r = t / 3, sK = t % 3;
            s.AppendLine($"    add.s32 %r11, %r7, {I(r - 1)};");   // ih = oh + r - 1
            s.AppendLine($"    add.s32 %r12, %r8, {I(sK - 1)};");  // iw = ow + s - 1
            s.AppendLine("    setp.ge.s32 %p0, %r11, 0;");
            s.AppendLine($"    setp.lt.s32 %p1, %r11, {I(h)};");
            s.AppendLine("    setp.ge.s32 %p2, %r12, 0;");
            s.AppendLine($"    setp.lt.s32 %p3, %r12, {I(w)};");
            s.AppendLine("    and.pred %p0, %p0, %p1;");
            s.AppendLine("    and.pred %p2, %p2, %p3;");
            s.AppendLine("    and.pred %p0, %p0, %p2;");
            s.AppendLine($"    mad.lo.u32 %r13, %r11, {I(w)}, %r12;");
            s.AppendLine("    mul.wide.u32 %rd6, %r13, 4;");
            s.AppendLine("    add.u64 %rd6, %rd5, %rd6;");
            s.AppendLine("    mov.f32 %f1, 0f00000000;");
            s.AppendLine($"    @%p0 ld.global.nc.f32 %f1, [%rd6];");
            s.AppendLine($"    add.u32 %r14, %r10, {I(t)};");
            s.AppendLine("    mul.wide.u32 %rd7, %r14, 4;");
            s.AppendLine("    add.u64 %rd7, %rd1, %rd7;");
            s.AppendLine("    ld.global.nc.f32 %f2, [%rd7];");
            s.AppendLine("    fma.rn.f32 %f0, %f1, %f2, %f0;");
        }
        if (relu)
            s.AppendLine("    max.f32 %f0, %f0, 0f00000000;");
        s.AppendLine("    mul.wide.u32 %rd8, %r2, 4;");
        s.AppendLine("    add.u64 %rd8, %rd3, %rd8;");
        s.AppendLine("    st.global.f32 [%rd8], %f0;");
        s.AppendLine("    ret;");
        s.AppendLine("}");
        return s.ToString();
    }

    public void Dispose() => _module.Dispose();
}
