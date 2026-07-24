using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Direct-PTX Conv2D backward-weight for 3x3 stride-1 same-padded convolution:
/// dW[k,c,r,s] = sum over (n, oh, ow) of input[n,c,oh+r-1,ow+s-1] * gradOutput[n,k,oh,ow].
/// One block per (output-channel k, input-channel c) pair reduces the whole
/// N x H x W contraction into the 9 filter taps: each thread walks a coalesced
/// stride of the flattened (n, oh, ow) index (consecutive threads read consecutive
/// spatial gradOutput -- the contiguous NCHW axis), reads the shared gradOutput
/// value once and the nine overlapping (cached) input taps, and accumulates 9
/// partials; a shared tree reduction combines them per tap. Correlation form of the
/// weight gradient; deterministic. Applies the coalescing lesson from the 3x3 forward.
/// </summary>
internal sealed class PtxConv2DBackwardWeight3x3Kernel : IDisposable
{
    internal const int BlockThreads = 256;

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int Batch { get; }
    internal int OutputChannels { get; }
    internal int InputChannels { get; }
    internal int Height { get; }
    internal int Width { get; }
    internal string Ptx { get; }
    internal DirectPtxFunctionInfo FunctionInfo { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal long InputBytes => (long)Batch * InputChannels * Height * Width * sizeof(float);
    internal long GradOutputBytes => (long)Batch * OutputChannels * Height * Width * sizeof(float);
    internal long GradWeightBytes => (long)OutputChannels * InputChannels * 9 * sizeof(float);

    internal string EntryPoint => FormattableString.Invariant(
        $"aidotnet_conv2d_bwd_weight3x3_n{Batch}_k{OutputChannels}_c{InputChannels}_h{Height}_w{Width}");

    internal PtxConv2DBackwardWeight3x3Kernel(
        DirectPtxRuntime runtime, int batch, int outputChannels, int inputChannels, int height, int width)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new NotSupportedException("Conv2D backward-weight has no experimental non-SM86 specialization.");
        if (batch <= 0 || outputChannels <= 0 || inputChannels <= 0 || height <= 0 || width <= 0)
            throw new ArgumentOutOfRangeException(nameof(batch));
        Batch = batch; OutputChannels = outputChannels; InputChannels = inputChannels; Height = height; Width = width;

        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, batch, outputChannels, inputChannels, height, width);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor,
            batch, outputChannels, inputChannels, height, width);
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
        DirectPtxArchitectureFamily architecture, int n, int k, int c, int h, int w)
    {
        var input = new DirectPtxExtent(n, c, h, w);
        var grad = new DirectPtxExtent(n, k, h, w);
        var dw = new DirectPtxExtent(k, c, 3, 3);
        return new DirectPtxKernelBlueprint(
            Operation: "conv2d-backward-weight-3x3",
            Version: 1,
            Architecture: architecture,
            Variant: FormattableString.Invariant($"n{n}-k{k}-c{c}-h{h}-w{w}-r3-fp32"),
            Tensors:
            [
                new("input", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Nchw,
                    input, input, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("gradOutput", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Nchw,
                    grad, grad, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("gradWeight", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Oihw,
                    dw, dw, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 48, MaxStaticSharedBytes: BlockThreads * sizeof(float),
                MaxLocalBytesPerThread: 0, MinBlocksPerMultiprocessor: 1),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["equation"] = "dW[k,c,r,s] = sum_{n,oh,ow} input[n,c,oh+r-1,ow+s-1] * gradOut[n,k,oh,ow]",
                ["reduction"] = "one block per (k,c), coalesced spatial reads, 9-tap shared reduce",
                ["padding"] = "same-1 stride-1", ["shape-selection"] = "host-only-exact-contract",
                ["promotion"] = "experimental-pending-gpu-evidence"
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
        _module.Launch(_function, (uint)(OutputChannels * InputChannels), 1, 1, BlockThreads, 1, 1, 0, arguments);
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

    internal static string EmitPtx(int major, int minor, int n, int k, int c, int h, int w)
    {
        if (!DirectPtxArchitecture.HasExperimentalConvolution(major, minor))
            throw new NotSupportedException("Only the experimental SM86 backward-weight emitter exists.");
        string I(int v) => v.ToString(CultureInfo.InvariantCulture);
        int hw = h * w;
        int nhw = n * hw;
        int chw = c * hw;                     // input batch stride
        int kkhw = k * hw;                    // gradOut batch stride
        string entry = FormattableString.Invariant(
            $"aidotnet_conv2d_bwd_weight3x3_n{n}_k{k}_c{c}_h{h}_w{w}");

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
        s.AppendLine("    .reg .b32 %r<32>;");
        s.AppendLine("    .reg .b64 %rd<16>;");
        s.AppendLine("    .reg .f32 %f<20>;");
        s.AppendLine($"    .shared .align 4 .b32 red[{I(BlockThreads)}];");
        s.AppendLine("    ld.param.u64 %rd0, [input_ptr];");
        s.AppendLine("    ld.param.u64 %rd1, [grad_ptr];");
        s.AppendLine("    ld.param.u64 %rd2, [dw_ptr];");
        s.AppendLine("    mov.u32 %r0, %tid.x;");
        s.AppendLine("    mov.u32 %r1, %ctaid.x;");                 // block = k*C + c
        s.AppendLine($"    div.u32 %r2, %r1, {I(c)};");            // k
        s.AppendLine($"    rem.u32 %r3, %r1, {I(c)};");            // c
        // 9 tap accumulators %f0..8
        for (int t = 0; t < 9; t++)
            s.AppendLine($"    mov.f32 %f{t}, 0f00000000;");
        // loop over flattened (n, oh, ow): i = tid; i < N*HW; i += 256
        s.AppendLine("    mov.u32 %r4, %r0;");                     // i
        s.AppendLine("LOOP:");
        s.AppendLine($"    setp.ge.u32 %p0, %r4, {I(nhw)};");
        s.AppendLine("    @%p0 bra REDUCE;");
        s.AppendLine($"    div.u32 %r5, %r4, {I(hw)};");           // nn
        s.AppendLine($"    rem.u32 %r6, %r4, {I(hw)};");           // spatial s
        s.AppendLine($"    div.u32 %r7, %r6, {I(w)};");            // oh
        s.AppendLine($"    rem.u32 %r8, %r6, {I(w)};");            // ow
        // gradOut[nn][k][oh][ow] = grad + (nn*K*HW + k*HW + s)
        s.AppendLine($"    mad.lo.u32 %r9, %r5, {I(kkhw)}, %r6;"); // nn*K*HW + s
        s.AppendLine($"    mad.lo.u32 %r9, %r2, {I(hw)}, %r9;");   // + k*HW
        s.AppendLine("    mul.wide.u32 %rd3, %r9, 4;");
        s.AppendLine("    add.u64 %rd3, %rd1, %rd3;");
        s.AppendLine("    ld.global.nc.f32 %f9, [%rd3];");         // gradOut value (reused for 9 taps)
        // input channel base for (nn, c): input + (nn*C + c)*HW
        s.AppendLine($"    mad.lo.u32 %r10, %r5, {I(c)}, %r3;");   // nn*C + c
        s.AppendLine($"    mul.wide.u32 %rd4, %r10, {I(hw)};");
        s.AppendLine("    shl.b64 %rd4, %rd4, 2;");
        s.AppendLine("    add.u64 %rd4, %rd0, %rd4;");             // &input[nn][c][0][0]
        for (int t = 0; t < 9; t++)
        {
            int kr = t / 3, ks = t % 3;
            s.AppendLine($"    add.s32 %r11, %r7, {I(kr - 1)};");  // ih = oh + kr - 1
            s.AppendLine($"    add.s32 %r12, %r8, {I(ks - 1)};");  // iw = ow + ks - 1
            s.AppendLine("    setp.ge.s32 %p1, %r11, 0;");
            s.AppendLine($"    setp.lt.s32 %p2, %r11, {I(h)};");
            s.AppendLine("    setp.ge.s32 %p3, %r12, 0;");
            s.AppendLine($"    setp.lt.s32 %p4, %r12, {I(w)};");
            s.AppendLine("    and.pred %p1, %p1, %p2;");
            s.AppendLine("    and.pred %p3, %p3, %p4;");
            s.AppendLine("    and.pred %p1, %p1, %p3;");
            s.AppendLine($"    mul.lo.u32 %r13, %r11, {I(w)};");
            s.AppendLine("    add.u32 %r13, %r13, %r12;");
            s.AppendLine("    mul.wide.u32 %rd5, %r13, 4;");
            s.AppendLine("    add.u64 %rd5, %rd4, %rd5;");
            s.AppendLine("    mov.f32 %f10, 0f00000000;");
            s.AppendLine($"    @%p1 ld.global.nc.f32 %f10, [%rd5];");
            s.AppendLine($"    fma.rn.f32 %f{t}, %f10, %f9, %f{t};");
        }
        s.AppendLine($"    add.u32 %r4, %r4, {I(BlockThreads)};");
        s.AppendLine("    bra LOOP;");
        s.AppendLine("REDUCE:");
        // reduce each of the 9 taps via shared[256], write dW[k][c][t]
        s.AppendLine("    mov.u64 %rd6, red;");
        s.AppendLine("    mul.wide.u32 %rd7, %r0, 4;");
        s.AppendLine("    add.u64 %rd7, %rd6, %rd7;");            // &red[tid]
        // dW base: dw + (k*C + c)*9
        s.AppendLine($"    mad.lo.u32 %r14, %r2, {I(c)}, %r3;");  // k*C + c
        s.AppendLine("    mul.lo.u32 %r14, %r14, 9;");
        s.AppendLine("    mul.wide.u32 %rd8, %r14, 4;");
        s.AppendLine("    add.u64 %rd8, %rd2, %rd8;");            // &dW[k][c][0]
        for (int t = 0; t < 9; t++)
        {
            s.AppendLine("    bar.sync 0;");
            s.AppendLine($"    st.shared.f32 [%rd7], %f{t};");
            s.AppendLine("    bar.sync 0;");
            for (int offset = BlockThreads / 2; offset > 0; offset >>= 1)
            {
                string skip = $"S_{t}_{offset}";
                s.AppendLine($"    setp.lt.u32 %p5, %r0, {I(offset)};");
                s.AppendLine($"    @!%p5 bra {skip};");
                s.AppendLine("    ld.shared.f32 %f10, [%rd7];");
                s.AppendLine($"    ld.shared.f32 %f11, [%rd7+{I(offset * 4)}];");
                s.AppendLine("    add.rn.f32 %f10, %f10, %f11;");
                s.AppendLine("    st.shared.f32 [%rd7], %f10;");
                s.AppendLine($"{skip}:");
                s.AppendLine("    bar.sync 0;");
            }
            s.AppendLine("    setp.ne.u32 %p5, %r0, 0;");
            s.AppendLine($"    @%p5 bra AFTER_{t};");
            s.AppendLine("    ld.shared.f32 %f10, [%rd6];");
            s.AppendLine($"    st.global.f32 [%rd8+{I(t * 4)}], %f10;");
            s.AppendLine($"AFTER_{t}:");
        }
        s.AppendLine("    ret;");
        s.AppendLine("}");
        return s.ToString();
    }

    public void Dispose() => _module.Dispose();
}
