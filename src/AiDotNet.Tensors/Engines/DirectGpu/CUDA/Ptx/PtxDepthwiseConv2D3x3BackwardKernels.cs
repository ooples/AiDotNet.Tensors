using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Direct-PTX depthwise Conv2D backward-input, 3x3 stride-1 same-pad:
/// dX[n,c,ih,iw] = sum_{r,s} W[c,r,s] * gradOut[n,c,ih-r+1,iw-s+1] (per channel, no
/// channel reduction). One thread per input-gradient element, consecutive threads own
/// consecutive iw -> coalesced gradOut reads and dX stores; the 9 depthwise weights
/// broadcast across the warp.
/// </summary>
internal sealed class PtxDepthwiseConv2D3x3BackwardInputKernel : IDisposable
{
    internal const int BlockThreads = 256;
    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int Batch { get; }
    internal int Channels { get; }
    internal int Height { get; }
    internal int Width { get; }
    internal string Ptx { get; }
    internal DirectPtxFunctionInfo FunctionInfo { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal long GradOutputBytes => (long)Batch * Channels * Height * Width * sizeof(float);
    internal long WeightBytes => (long)Channels * 9 * sizeof(float);
    internal long GradInputBytes => (long)Batch * Channels * Height * Width * sizeof(float);

    internal string EntryPoint => FormattableString.Invariant(
        $"aidotnet_depthwise_bwd_input3x3_n{Batch}_c{Channels}_h{Height}_w{Width}");

    internal PtxDepthwiseConv2D3x3BackwardInputKernel(
        DirectPtxRuntime runtime, int batch, int channels, int height, int width)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new NotSupportedException("Depthwise backward-input has no experimental non-SM86 specialization.");
        if (batch <= 0 || channels <= 0 || height <= 0 || width <= 0)
            throw new ArgumentOutOfRangeException(nameof(batch));
        Batch = batch; Channels = channels; Height = height; Width = width;
        if ((long)batch * channels * height * width % BlockThreads != 0)
            throw new ArgumentException($"N*C*H*W must be a multiple of {BlockThreads}.");

        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, batch, channels, height, width);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor, batch, channels, height, width);
        _module = runtime.LoadModule(Ptx, allowExperimentalJitFallback: DirectPtxFeatureGate.ConvolutionExperimentOverride);
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo functionInfo);
        FunctionInfo = functionInfo;
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, BlockThreads);
        Blueprint.ResourceBudget.Validate(EntryPoint, functionInfo, BlockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(Blueprint, runtime.DeviceFingerprint, Ptx, functionInfo, BlockThreads, activeBlocks, _module);
    }

    internal static DirectPtxKernelBlueprint CreateBlueprint(DirectPtxArchitectureFamily architecture, int n, int c, int h, int w)
    {
        var io = new DirectPtxExtent(n, c, h, w);
        var weight = new DirectPtxExtent(c, 1, 3, 3);
        return new DirectPtxKernelBlueprint(
            Operation: "depthwise-conv2d-backward-input-3x3", Version: 1, Architecture: architecture,
            Variant: FormattableString.Invariant($"n{n}-c{c}-h{h}-w{w}-r3-fp32"),
            Tensors:
            [
                new("gradOutput", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Nchw, io, io, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("weights", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Oihw, weight, weight, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("gradInput", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Nchw, io, io, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(MaxRegistersPerThread: 40, MaxStaticSharedBytes: 0, MaxLocalBytesPerThread: 0, MinBlocksPerMultiprocessor: 2),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["equation"] = "dX[n,c,ih,iw] = sum_{r,s} W[c,r,s]*gradOut[n,c,ih-r+1,iw-s+1]",
                ["padding"] = "same-1 stride-1", ["shape-selection"] = "host-only-exact-contract", ["promotion"] = "experimental-pending-gpu-evidence"
            });
    }

    internal unsafe void Launch(DirectPtxTensorView gradOutput, DirectPtxTensorView weights, DirectPtxTensorView gradInput)
    {
        Require(gradOutput, Blueprint.Tensors[0], nameof(gradOutput));
        Require(weights, Blueprint.Tensors[1], nameof(weights));
        Require(gradInput, Blueprint.Tensors[2], nameof(gradInput));
        IntPtr gPtr = gradOutput.Pointer, wPtr = weights.Pointer, xPtr = gradInput.Pointer;
        void** arguments = stackalloc void*[3];
        arguments[0] = &gPtr; arguments[1] = &wPtr; arguments[2] = &xPtr;
        int total = Batch * Channels * Height * Width;
        _module.Launch(_function, (uint)(total / BlockThreads), 1, 1, BlockThreads, 1, 1, 0, arguments);
    }

    private static void Require(DirectPtxTensorView view, DirectPtxTensorContract contract, string parameter)
    {
        if (view.Pointer == IntPtr.Zero || view.PhysicalType != contract.PhysicalType || view.Layout != contract.Layout ||
            view.LogicalExtent != contract.LogicalExtent || view.PhysicalExtent != contract.PhysicalExtent ||
            view.ByteLength != contract.RequiredBytes || view.AllocationByteLength != contract.RequiredBytes)
            throw new ArgumentException($"{parameter} does not satisfy exact physical ABI '{contract.Name}'.", parameter);
    }

    internal static string EmitPtx(int major, int minor, int n, int c, int h, int w)
    {
        if (!DirectPtxArchitecture.HasExperimentalConvolution(major, minor))
            throw new NotSupportedException("Only the experimental SM86 depthwise backward-input emitter exists.");
        string I(int v) => v.ToString(CultureInfo.InvariantCulture);
        int hw = h * w, chw = c * hw;
        string entry = FormattableString.Invariant($"aidotnet_depthwise_bwd_input3x3_n{n}_c{c}_h{h}_w{w}");

        var s = new StringBuilder(12288);
        s.AppendLine(".version 7.1");
        s.AppendLine($".target sm_{major}{minor}");
        s.AppendLine(".address_size 64");
        s.AppendLine();
        s.AppendLine($".visible .entry {entry}(");
        s.AppendLine("    .param .u64 grad_ptr,");
        s.AppendLine("    .param .u64 weight_ptr,");
        s.AppendLine("    .param .u64 dx_ptr");
        s.AppendLine(")");
        s.AppendLine("{");
        s.AppendLine("    .reg .pred %p<6>;");
        s.AppendLine("    .reg .b32 %r<24>;");
        s.AppendLine("    .reg .b64 %rd<16>;");
        s.AppendLine("    .reg .f32 %f<8>;");
        s.AppendLine("    ld.param.u64 %rd0, [grad_ptr];");
        s.AppendLine("    ld.param.u64 %rd1, [weight_ptr];");
        s.AppendLine("    ld.param.u64 %rd2, [dx_ptr];");
        s.AppendLine("    mov.u32 %r0, %tid.x;");
        s.AppendLine("    mov.u32 %r1, %ctaid.x;");
        s.AppendLine($"    mad.lo.u32 %r2, %r1, {I(BlockThreads)}, %r0;");  // idx
        s.AppendLine($"    div.u32 %r3, %r2, {I(chw)};");          // n
        s.AppendLine($"    rem.u32 %r4, %r2, {I(chw)};");
        s.AppendLine($"    div.u32 %r5, %r4, {I(hw)};");           // c
        s.AppendLine($"    rem.u32 %r6, %r4, {I(hw)};");
        s.AppendLine($"    div.u32 %r7, %r6, {I(w)};");            // ih
        s.AppendLine($"    rem.u32 %r8, %r6, {I(w)};");            // iw
        s.AppendLine("    mov.f32 %f0, 0f00000000;");
        // gradOut channel base = grad + (n*C + c)*HW  (same layout as idx minus spatial)
        s.AppendLine($"    sub.u32 %r9, %r2, %r6;");               // (n*C+c)*HW
        s.AppendLine("    mul.wide.u32 %rd3, %r9, 4;");
        s.AppendLine("    add.u64 %rd3, %rd0, %rd3;");             // &gradOut[n][c][0][0]
        s.AppendLine("    mul.lo.u32 %r10, %r5, 9;");             // weight base c*9
        for (int t = 0; t < 9; t++)
        {
            int r = t / 3, sK = t % 3;
            s.AppendLine($"    add.s32 %r11, %r7, {I(1 - r)};");   // oh = ih - r + 1
            s.AppendLine($"    add.s32 %r12, %r8, {I(1 - sK)};");  // ow = iw - s + 1
            s.AppendLine("    setp.ge.s32 %p0, %r11, 0;");
            s.AppendLine($"    setp.lt.s32 %p1, %r11, {I(h)};");
            s.AppendLine("    setp.ge.s32 %p2, %r12, 0;");
            s.AppendLine($"    setp.lt.s32 %p3, %r12, {I(w)};");
            s.AppendLine("    and.pred %p0, %p0, %p1;");
            s.AppendLine("    and.pred %p2, %p2, %p3;");
            s.AppendLine("    and.pred %p0, %p0, %p2;");
            s.AppendLine($"    mad.lo.u32 %r13, %r11, {I(w)}, %r12;");
            s.AppendLine("    mul.wide.u32 %rd4, %r13, 4;");
            s.AppendLine("    add.u64 %rd4, %rd3, %rd4;");
            s.AppendLine("    mov.f32 %f1, 0f00000000;");
            s.AppendLine($"    @%p0 ld.global.nc.f32 %f1, [%rd4];");
            s.AppendLine($"    add.u32 %r14, %r10, {I(t)};");
            s.AppendLine("    mul.wide.u32 %rd5, %r14, 4;");
            s.AppendLine("    add.u64 %rd5, %rd1, %rd5;");
            s.AppendLine("    ld.global.nc.f32 %f2, [%rd5];");
            s.AppendLine("    fma.rn.f32 %f0, %f1, %f2, %f0;");
        }
        s.AppendLine("    mul.wide.u32 %rd6, %r2, 4;");
        s.AppendLine("    add.u64 %rd6, %rd2, %rd6;");
        s.AppendLine("    st.global.f32 [%rd6], %f0;");
        s.AppendLine("    ret;");
        s.AppendLine("}");
        return s.ToString();
    }

    public void Dispose() => _module.Dispose();
}

/// <summary>
/// Direct-PTX depthwise Conv2D backward-weight, 3x3 stride-1 same-pad:
/// dW[c,r,s] = sum_{n,oh,ow} input[n,c,oh+r-1,ow+s-1] * gradOut[n,c,oh,ow] (per
/// channel). One block per channel reduces the N x H x W contraction into the 9 taps
/// with coalesced spatial reads (consecutive threads walk the flattened (n,oh,ow)
/// index) and a shared tree reduction per tap.
/// </summary>
internal sealed class PtxDepthwiseConv2D3x3BackwardWeightKernel : IDisposable
{
    internal const int BlockThreads = 256;
    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int Batch { get; }
    internal int Channels { get; }
    internal int Height { get; }
    internal int Width { get; }
    internal string Ptx { get; }
    internal DirectPtxFunctionInfo FunctionInfo { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal long InputBytes => (long)Batch * Channels * Height * Width * sizeof(float);
    internal long GradOutputBytes => (long)Batch * Channels * Height * Width * sizeof(float);
    internal long GradWeightBytes => (long)Channels * 9 * sizeof(float);

    internal string EntryPoint => FormattableString.Invariant(
        $"aidotnet_depthwise_bwd_weight3x3_n{Batch}_c{Channels}_h{Height}_w{Width}");

    internal PtxDepthwiseConv2D3x3BackwardWeightKernel(
        DirectPtxRuntime runtime, int batch, int channels, int height, int width)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new NotSupportedException("Depthwise backward-weight has no experimental non-SM86 specialization.");
        if (batch <= 0 || channels <= 0 || height <= 0 || width <= 0)
            throw new ArgumentOutOfRangeException(nameof(batch));
        Batch = batch; Channels = channels; Height = height; Width = width;

        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, batch, channels, height, width);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor, batch, channels, height, width);
        _module = runtime.LoadModule(Ptx, allowExperimentalJitFallback: DirectPtxFeatureGate.ConvolutionExperimentOverride);
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo functionInfo);
        FunctionInfo = functionInfo;
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, BlockThreads);
        Blueprint.ResourceBudget.Validate(EntryPoint, functionInfo, BlockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(Blueprint, runtime.DeviceFingerprint, Ptx, functionInfo, BlockThreads, activeBlocks, _module);
    }

    internal static DirectPtxKernelBlueprint CreateBlueprint(DirectPtxArchitectureFamily architecture, int n, int c, int h, int w)
    {
        var io = new DirectPtxExtent(n, c, h, w);
        var dw = new DirectPtxExtent(c, 1, 3, 3);
        return new DirectPtxKernelBlueprint(
            Operation: "depthwise-conv2d-backward-weight-3x3", Version: 1, Architecture: architecture,
            Variant: FormattableString.Invariant($"n{n}-c{c}-h{h}-w{w}-r3-fp32"),
            Tensors:
            [
                new("input", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Nchw, io, io, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("gradOutput", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Nchw, io, io, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("gradWeight", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Oihw, dw, dw, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(MaxRegistersPerThread: 48, MaxStaticSharedBytes: BlockThreads * sizeof(float), MaxLocalBytesPerThread: 0, MinBlocksPerMultiprocessor: 1),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["equation"] = "dW[c,r,s] = sum_{n,oh,ow} input[n,c,oh+r-1,ow+s-1]*gradOut[n,c,oh,ow]",
                ["padding"] = "same-1 stride-1", ["shape-selection"] = "host-only-exact-contract", ["promotion"] = "experimental-pending-gpu-evidence"
            });
    }

    internal unsafe void Launch(DirectPtxTensorView input, DirectPtxTensorView gradOutput, DirectPtxTensorView gradWeight)
    {
        Require(input, Blueprint.Tensors[0], nameof(input));
        Require(gradOutput, Blueprint.Tensors[1], nameof(gradOutput));
        Require(gradWeight, Blueprint.Tensors[2], nameof(gradWeight));
        IntPtr iPtr = input.Pointer, gPtr = gradOutput.Pointer, wPtr = gradWeight.Pointer;
        void** arguments = stackalloc void*[3];
        arguments[0] = &iPtr; arguments[1] = &gPtr; arguments[2] = &wPtr;
        _module.Launch(_function, (uint)Channels, 1, 1, BlockThreads, 1, 1, 0, arguments);
    }

    private static void Require(DirectPtxTensorView view, DirectPtxTensorContract contract, string parameter)
    {
        if (view.Pointer == IntPtr.Zero || view.PhysicalType != contract.PhysicalType || view.Layout != contract.Layout ||
            view.LogicalExtent != contract.LogicalExtent || view.PhysicalExtent != contract.PhysicalExtent ||
            view.ByteLength != contract.RequiredBytes || view.AllocationByteLength != contract.RequiredBytes)
            throw new ArgumentException($"{parameter} does not satisfy exact physical ABI '{contract.Name}'.", parameter);
    }

    internal static string EmitPtx(int major, int minor, int n, int c, int h, int w)
    {
        if (!DirectPtxArchitecture.HasExperimentalConvolution(major, minor))
            throw new NotSupportedException("Only the experimental SM86 depthwise backward-weight emitter exists.");
        string I(int v) => v.ToString(CultureInfo.InvariantCulture);
        int hw = h * w, nhw = n * hw, chw = c * hw;
        string entry = FormattableString.Invariant($"aidotnet_depthwise_bwd_weight3x3_n{n}_c{c}_h{h}_w{w}");

        var s = new StringBuilder(16384);
        s.AppendLine(".version 7.1");
        s.AppendLine($".target sm_{major}{minor}");
        s.AppendLine(".address_size 64");
        s.AppendLine();
        s.AppendLine($".visible .entry {entry}(");
        s.AppendLine("    .param .u64 input_ptr,");
        s.AppendLine("    .param .u64 grad_ptr,");
        s.AppendLine("    .param .u64 dw_ptr");
        s.AppendLine(")");
        s.AppendLine("{");
        s.AppendLine("    .reg .pred %p<6>;");
        s.AppendLine("    .reg .b32 %r<28>;");
        s.AppendLine("    .reg .b64 %rd<16>;");
        s.AppendLine("    .reg .f32 %f<20>;");
        s.AppendLine($"    .shared .align 4 .b32 red[{I(BlockThreads)}];");
        s.AppendLine("    ld.param.u64 %rd0, [input_ptr];");
        s.AppendLine("    ld.param.u64 %rd1, [grad_ptr];");
        s.AppendLine("    ld.param.u64 %rd2, [dw_ptr];");
        s.AppendLine("    mov.u32 %r0, %tid.x;");
        s.AppendLine("    mov.u32 %r1, %ctaid.x;");                 // c
        for (int t = 0; t < 9; t++)
            s.AppendLine($"    mov.f32 %f{t}, 0f00000000;");
        s.AppendLine("    mov.u32 %r4, %r0;");                     // i over N*HW
        s.AppendLine("LOOP:");
        s.AppendLine($"    setp.ge.u32 %p0, %r4, {I(nhw)};");
        s.AppendLine("    @%p0 bra REDUCE;");
        s.AppendLine($"    div.u32 %r5, %r4, {I(hw)};");           // nn
        s.AppendLine($"    rem.u32 %r6, %r4, {I(hw)};");           // spatial s
        s.AppendLine($"    div.u32 %r7, %r6, {I(w)};");            // oh
        s.AppendLine($"    rem.u32 %r8, %r6, {I(w)};");            // ow
        // channel base index for (nn, c) = nn*C*HW + c*HW
        s.AppendLine($"    mad.lo.u32 %r9, %r5, {I(chw)}, %r6;");  // nn*C*HW + s
        s.AppendLine($"    mad.lo.u32 %r9, %r1, {I(hw)}, %r9;");   // + c*HW
        s.AppendLine("    mul.wide.u32 %rd3, %r9, 4;");
        s.AppendLine("    add.u64 %rd4, %rd1, %rd3;");             // &gradOut
        s.AppendLine("    ld.global.nc.f32 %f9, [%rd4];");        // gradOut value
        s.AppendLine("    add.u64 %rd5, %rd0, %rd3;");            // &input[nn][c][oh][ow] (same index base)
        // input channel base (subtract spatial to get channel origin): &input - s + ...
        s.AppendLine("    sub.u32 %r10, %r9, %r6;");              // channel origin index (nn*C*HW + c*HW)
        s.AppendLine("    mul.wide.u32 %rd6, %r10, 4;");
        s.AppendLine("    add.u64 %rd6, %rd0, %rd6;");            // &input[nn][c][0][0]
        for (int t = 0; t < 9; t++)
        {
            int r = t / 3, sK = t % 3;
            s.AppendLine($"    add.s32 %r11, %r7, {I(r - 1)};");   // ih = oh + r - 1
            s.AppendLine($"    add.s32 %r12, %r8, {I(sK - 1)};");  // iw = ow + s - 1
            s.AppendLine("    setp.ge.s32 %p1, %r11, 0;");
            s.AppendLine($"    setp.lt.s32 %p2, %r11, {I(h)};");
            s.AppendLine("    setp.ge.s32 %p3, %r12, 0;");
            s.AppendLine($"    setp.lt.s32 %p4, %r12, {I(w)};");
            s.AppendLine("    and.pred %p1, %p1, %p2;");
            s.AppendLine("    and.pred %p3, %p3, %p4;");
            s.AppendLine("    and.pred %p1, %p1, %p3;");
            s.AppendLine($"    mad.lo.u32 %r13, %r11, {I(w)}, %r12;");
            s.AppendLine("    mul.wide.u32 %rd7, %r13, 4;");
            s.AppendLine("    add.u64 %rd7, %rd6, %rd7;");
            s.AppendLine("    mov.f32 %f10, 0f00000000;");
            s.AppendLine($"    @%p1 ld.global.nc.f32 %f10, [%rd7];");
            s.AppendLine($"    fma.rn.f32 %f{t}, %f10, %f9, %f{t};");
        }
        s.AppendLine($"    add.u32 %r4, %r4, {I(BlockThreads)};");
        s.AppendLine("    bra LOOP;");
        s.AppendLine("REDUCE:");
        s.AppendLine("    mov.u64 %rd8, red;");
        s.AppendLine("    mul.wide.u32 %rd9, %r0, 4;");
        s.AppendLine("    add.u64 %rd9, %rd8, %rd9;");
        s.AppendLine("    mul.lo.u32 %r14, %r1, 9;");             // dW base c*9
        s.AppendLine("    mul.wide.u32 %rd10, %r14, 4;");
        s.AppendLine("    add.u64 %rd10, %rd2, %rd10;");
        for (int t = 0; t < 9; t++)
        {
            s.AppendLine("    bar.sync 0;");
            s.AppendLine($"    st.shared.f32 [%rd9], %f{t};");
            s.AppendLine("    bar.sync 0;");
            for (int offset = BlockThreads / 2; offset > 0; offset >>= 1)
            {
                string skip = $"S_{t}_{offset}";
                s.AppendLine($"    setp.lt.u32 %p5, %r0, {I(offset)};");
                s.AppendLine($"    @!%p5 bra {skip};");
                s.AppendLine("    ld.shared.f32 %f10, [%rd9];");
                s.AppendLine($"    ld.shared.f32 %f11, [%rd9+{I(offset * 4)}];");
                s.AppendLine("    add.rn.f32 %f10, %f10, %f11;");
                s.AppendLine("    st.shared.f32 [%rd9], %f10;");
                s.AppendLine($"{skip}:");
                s.AppendLine("    bar.sync 0;");
            }
            s.AppendLine("    setp.ne.u32 %p5, %r0, 0;");
            s.AppendLine($"    @%p5 bra AFTER_{t};");
            s.AppendLine("    ld.shared.f32 %f10, [%rd8];");
            s.AppendLine($"    st.global.f32 [%rd10+{I(t * 4)}], %f10;");
            s.AppendLine($"AFTER_{t}:");
        }
        s.AppendLine("    ret;");
        s.AppendLine("}");
        return s.ToString();
    }

    public void Dispose() => _module.Dispose();
}
