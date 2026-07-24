using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Direct-PTX Conv2D backward-input for 3x3 stride-1 same-padded convolution:
/// dX[n,c,ih,iw] = sum over (k, r, s) of W[k,c,r,s] * gradOutput[n,k,ih-r+1,iw-s+1]
/// (the transpose/flip of the forward correlation). One thread per input-gradient
/// element; consecutive threads own consecutive iw (the contiguous NCHW axis) so the
/// gradOutput reads (ow = iw - s + 1) and the dX stores are coalesced, and the weight
/// reads broadcast across the warp (same channel c). Deterministic gather. Applies the
/// coalescing lesson from the 3x3 forward kernel.
/// </summary>
internal sealed class PtxConv2DBackwardInput3x3Kernel : IDisposable
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

    internal long GradOutputBytes => (long)Batch * OutputChannels * Height * Width * sizeof(float);
    internal long WeightBytes => (long)OutputChannels * InputChannels * 9 * sizeof(float);
    internal long GradInputBytes => (long)Batch * InputChannels * Height * Width * sizeof(float);

    internal string EntryPoint => FormattableString.Invariant(
        $"aidotnet_conv2d_bwd_input3x3_n{Batch}_k{OutputChannels}_c{InputChannels}_h{Height}_w{Width}");

    internal PtxConv2DBackwardInput3x3Kernel(
        DirectPtxRuntime runtime, int batch, int outputChannels, int inputChannels, int height, int width)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new NotSupportedException("Conv2D backward-input has no experimental non-SM86 specialization.");
        if (batch <= 0 || outputChannels <= 0 || inputChannels <= 0 || height <= 0 || width <= 0)
            throw new ArgumentOutOfRangeException(nameof(batch));
        Batch = batch; OutputChannels = outputChannels; InputChannels = inputChannels; Height = height; Width = width;
        if ((long)batch * inputChannels * height * width % BlockThreads != 0)
            throw new ArgumentException($"N*C*H*W must be a multiple of {BlockThreads}.");

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
        var grad = new DirectPtxExtent(n, k, h, w);
        var weight = new DirectPtxExtent(k, c, 3, 3);
        var dx = new DirectPtxExtent(n, c, h, w);
        return new DirectPtxKernelBlueprint(
            Operation: "conv2d-backward-input-3x3",
            Version: 1,
            Architecture: architecture,
            Variant: FormattableString.Invariant($"n{n}-k{k}-c{c}-h{h}-w{w}-r3-fp32"),
            Tensors:
            [
                new("gradOutput", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Nchw,
                    grad, grad, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("weights", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Oihw,
                    weight, weight, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("gradInput", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Nchw,
                    dx, dx, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 40, MaxStaticSharedBytes: 0,
                MaxLocalBytesPerThread: 0, MinBlocksPerMultiprocessor: 2),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["equation"] = "dX[n,c,ih,iw] = sum_{k,r,s} W[k,c,r,s] * gradOut[n,k,ih-r+1,iw-s+1]",
                ["access"] = "thread-per-output, iw-contiguous -> coalesced gradOut/dX",
                ["padding"] = "same-1 stride-1", ["shape-selection"] = "host-only-exact-contract",
                ["promotion"] = "experimental-pending-gpu-evidence"
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
        int total = Batch * InputChannels * Height * Width;
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

    internal static string EmitPtx(int major, int minor, int n, int k, int c, int h, int w)
    {
        if (!DirectPtxArchitecture.HasExperimentalConvolution(major, minor))
            throw new NotSupportedException("Only the experimental SM86 backward-input emitter exists.");
        string I(int v) => v.ToString(CultureInfo.InvariantCulture);
        int hw = h * w;
        int chw = c * hw;
        int kkhw = k * hw;
        int cc9 = c * 9;                      // weight k-stride
        string entry = FormattableString.Invariant(
            $"aidotnet_conv2d_bwd_input3x3_n{n}_k{k}_c{c}_h{h}_w{w}");

        var s = new StringBuilder(16384);
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
        s.AppendLine("    .reg .b32 %r<32>;");
        s.AppendLine("    .reg .b64 %rd<16>;");
        s.AppendLine("    .reg .f32 %f<8>;");
        s.AppendLine("    ld.param.u64 %rd0, [grad_ptr];");
        s.AppendLine("    ld.param.u64 %rd1, [weight_ptr];");
        s.AppendLine("    ld.param.u64 %rd2, [dx_ptr];");
        s.AppendLine("    mov.u32 %r0, %tid.x;");
        s.AppendLine("    mov.u32 %r1, %ctaid.x;");
        s.AppendLine($"    mad.lo.u32 %r2, %r1, {I(BlockThreads)}, %r0;");  // idx = n*C*HW + c*HW + ih*W + iw
        s.AppendLine($"    div.u32 %r3, %r2, {I(chw)};");          // n
        s.AppendLine($"    rem.u32 %r4, %r2, {I(chw)};");
        s.AppendLine($"    div.u32 %r5, %r4, {I(hw)};");           // c
        s.AppendLine($"    rem.u32 %r6, %r4, {I(hw)};");
        s.AppendLine($"    div.u32 %r7, %r6, {I(w)};");            // ih
        s.AppendLine($"    rem.u32 %r8, %r6, {I(w)};");            // iw
        s.AppendLine("    mov.f32 %f0, 0f00000000;");             // acc
        // gradOut batch base: grad + n*K*HW
        s.AppendLine($"    mul.lo.u32 %r9, %r3, {I(kkhw)};");
        // weight channel offset: c*9 (added to k*C*9)
        s.AppendLine("    mov.u32 %r10, 0;");                     // kk = 0
        s.AppendLine("LOOP_K:");
        // gradOut[n][kk] base index = n*K*HW + kk*HW
        s.AppendLine($"    mad.lo.u32 %r11, %r10, {I(hw)}, %r9;");
        // weight[kk][c] base = weights + (kk*C + c)*9
        s.AppendLine($"    mad.lo.u32 %r12, %r10, {I(cc9)}, 0;");
        s.AppendLine($"    mad.lo.u32 %r12, %r5, 9, %r12;");      // + c*9
        for (int t = 0; t < 9; t++)
        {
            int r = t / 3, sK = t % 3;
            s.AppendLine($"    add.s32 %r13, %r7, {I(1 - r)};");  // oh = ih - r + 1
            s.AppendLine($"    add.s32 %r14, %r8, {I(1 - sK)};"); // ow = iw - s + 1
            s.AppendLine("    setp.ge.s32 %p0, %r13, 0;");
            s.AppendLine($"    setp.lt.s32 %p1, %r13, {I(h)};");
            s.AppendLine("    setp.ge.s32 %p2, %r14, 0;");
            s.AppendLine($"    setp.lt.s32 %p3, %r14, {I(w)};");
            s.AppendLine("    and.pred %p0, %p0, %p1;");
            s.AppendLine("    and.pred %p2, %p2, %p3;");
            s.AppendLine("    and.pred %p0, %p0, %p2;");
            s.AppendLine($"    mad.lo.u32 %r15, %r13, {I(w)}, %r11;");   // + oh*W
            s.AppendLine("    add.u32 %r15, %r15, %r14;");               // + ow
            s.AppendLine("    mul.wide.u32 %rd3, %r15, 4;");
            s.AppendLine("    add.u64 %rd3, %rd0, %rd3;");
            s.AppendLine("    mov.f32 %f1, 0f00000000;");
            s.AppendLine($"    @%p0 ld.global.nc.f32 %f1, [%rd3];");     // gradOut
            s.AppendLine($"    add.u32 %r16, %r12, {I(t)};");            // weight index
            s.AppendLine("    mul.wide.u32 %rd4, %r16, 4;");
            s.AppendLine("    add.u64 %rd4, %rd1, %rd4;");
            s.AppendLine("    ld.global.nc.f32 %f2, [%rd4];");           // W[kk][c][r][s]
            s.AppendLine("    fma.rn.f32 %f0, %f1, %f2, %f0;");
        }
        s.AppendLine("    add.u32 %r10, %r10, 1;");
        s.AppendLine($"    setp.lt.u32 %p0, %r10, {I(k)};");
        s.AppendLine("    @%p0 bra LOOP_K;");
        // store dX[idx]
        s.AppendLine("    mul.wide.u32 %rd5, %r2, 4;");
        s.AppendLine("    add.u64 %rd5, %rd2, %rd5;");
        s.AppendLine("    st.global.f32 [%rd5], %f0;");
        s.AppendLine("    ret;");
        s.AppendLine("}");
        return s.ToString();
    }

    public void Dispose() => _module.Dispose();
}
