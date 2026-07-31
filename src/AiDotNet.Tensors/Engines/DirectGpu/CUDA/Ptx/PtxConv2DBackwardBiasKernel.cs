using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Direct-PTX Conv2D backward-bias: gradBias[k] = sum over batch and spatial of
/// gradOutput[b, k, h, w]. One block per output channel reduces the whole
/// B x H x W slice for that channel; consecutive threads read consecutive spatial
/// elements (the contiguous axis of NCHW) so the reduction loads are coalesced --
/// the same thread-to-memory mapping lesson that drove the 3x3 kernel. A shared
/// tree reduction combines the per-thread partials. This replaces the CPU-download
/// reduction in DirectGpuTensorEngine.Conv2DBackwardBiasGpu.
/// </summary>
internal sealed class PtxConv2DBackwardBiasKernel : IDisposable
{
    internal const int BlockThreads = 256;

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int Batch { get; }
    internal int OutputChannels { get; }
    internal int Height { get; }
    internal int Width { get; }
    internal string Ptx { get; }
    internal DirectPtxFunctionInfo FunctionInfo { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal int Spatial => Height * Width;
    internal long GradOutputBytes => (long)Batch * OutputChannels * Height * Width * sizeof(float);
    internal long GradBiasBytes => (long)OutputChannels * sizeof(float);

    internal string EntryPoint => FormattableString.Invariant(
        $"aidotnet_conv2d_backward_bias_n{Batch}_k{OutputChannels}_h{Height}_w{Width}");

    internal PtxConv2DBackwardBiasKernel(
        DirectPtxRuntime runtime, int batch, int outputChannels, int height, int width)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new NotSupportedException("Conv2D backward-bias has no experimental non-SM86 specialization.");
        if (batch <= 0 || outputChannels <= 0 || height <= 0 || width <= 0)
            throw new ArgumentOutOfRangeException(nameof(batch));
        Batch = batch; OutputChannels = outputChannels; Height = height; Width = width;

        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, batch, outputChannels, height, width);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor, batch, outputChannels, height, width);
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
        DirectPtxArchitectureFamily architecture, int batch, int k, int h, int w)
    {
        var grad = new DirectPtxExtent(batch, k, h, w);
        var bias = new DirectPtxExtent(k);
        return new DirectPtxKernelBlueprint(
            Operation: "conv2d-backward-bias",
            Version: 1,
            Architecture: architecture,
            Variant: FormattableString.Invariant($"n{batch}-k{k}-h{h}-w{w}-fp32"),
            Tensors:
            [
                new("gradOutput", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Nchw,
                    grad, grad, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("gradBias", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    bias, bias, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 32, MaxStaticSharedBytes: BlockThreads * sizeof(float),
                MaxLocalBytesPerThread: 0, MinBlocksPerMultiprocessor: 1),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["equation"] = "gradBias[k] = sum_{b,h,w} gradOutput[b,k,h,w]",
                ["reduction"] = "one block per channel, coalesced spatial reads, shared tree reduce",
                ["shape-selection"] = "host-only-exact-contract",
                ["promotion"] = "experimental-pending-gpu-evidence"
            });
    }

    internal unsafe void Launch(DirectPtxTensorView gradOutput, DirectPtxTensorView gradBias)
    {
        Require(gradOutput, Blueprint.Tensors[0], nameof(gradOutput));
        Require(gradBias, Blueprint.Tensors[1], nameof(gradBias));
        IntPtr gPtr = gradOutput.Pointer, bPtr = gradBias.Pointer;
        void** arguments = stackalloc void*[2];
        arguments[0] = &gPtr; arguments[1] = &bPtr;
        _module.Launch(_function, (uint)OutputChannels, 1, 1, BlockThreads, 1, 1, 0, arguments);
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

    internal static string EmitPtx(int major, int minor, int batch, int k, int h, int w)
    {
        if (!DirectPtxArchitecture.HasExperimentalConvolution(major, minor))
            throw new NotSupportedException("Only the experimental SM86 backward-bias emitter exists.");
        string I(int v) => v.ToString(CultureInfo.InvariantCulture);
        int hw = h * w;
        int khw = k * hw;                     // stride between batches for a fixed channel
        string entry = FormattableString.Invariant(
            $"aidotnet_conv2d_backward_bias_n{batch}_k{k}_h{h}_w{w}");

        var s = new StringBuilder(8192);
        s.AppendLine(".version 7.1");
        s.AppendLine($".target sm_{major}{minor}");
        s.AppendLine(".address_size 64");
        s.AppendLine();
        s.AppendLine($".visible .entry {entry}(");
        s.AppendLine("    .param .u64 grad_ptr,");
        s.AppendLine("    .param .u64 bias_ptr");
        s.AppendLine(")");
        s.AppendLine("{");
        s.AppendLine("    .reg .pred %p<4>;");
        s.AppendLine("    .reg .b32 %r<16>;");
        s.AppendLine("    .reg .b64 %rd<12>;");
        s.AppendLine("    .reg .f32 %f<4>;");
        s.AppendLine($"    .shared .align 4 .b32 red[{I(BlockThreads)}];");
        s.AppendLine("    ld.param.u64 %rd0, [grad_ptr];");
        s.AppendLine("    ld.param.u64 %rd1, [bias_ptr];");
        s.AppendLine("    mov.u32 %r0, %tid.x;");
        s.AppendLine("    mov.u32 %r1, %ctaid.x;");                 // k = channel
        s.AppendLine("    mov.f32 %f0, 0f00000000;");              // partial
        // per-channel base element index for batch 0: k*HW
        s.AppendLine($"    mul.lo.u32 %r2, %r1, {I(hw)};");        // k*HW
        s.AppendLine("    mov.u32 %r3, 0;");                       // b = 0
        s.AppendLine("LOOP_B:");
        // base element index for (b, k) = b*K*HW + k*HW
        s.AppendLine($"    mad.lo.u32 %r4, %r3, {I(khw)}, %r2;");  // b*K*HW + k*HW
        s.AppendLine("    mov.u32 %r5, %r0;");                     // s = tid
        s.AppendLine("LOOP_S:");
        s.AppendLine($"    setp.ge.u32 %p0, %r5, {I(hw)};");
        s.AppendLine("    @%p0 bra DONE_S;");
        s.AppendLine("    add.u32 %r6, %r4, %r5;");                // element index
        s.AppendLine("    mul.wide.u32 %rd2, %r6, 4;");
        s.AppendLine("    add.u64 %rd2, %rd0, %rd2;");
        s.AppendLine("    ld.global.nc.f32 %f1, [%rd2];");
        s.AppendLine("    add.rn.f32 %f0, %f0, %f1;");
        s.AppendLine($"    add.u32 %r5, %r5, {I(BlockThreads)};");
        s.AppendLine("    bra LOOP_S;");
        s.AppendLine("DONE_S:");
        s.AppendLine("    add.u32 %r3, %r3, 1;");
        s.AppendLine($"    setp.lt.u32 %p1, %r3, {I(batch)};");
        s.AppendLine("    @%p1 bra LOOP_B;");
        // write partial to shared
        s.AppendLine("    mov.u64 %rd3, red;");
        s.AppendLine("    mul.wide.u32 %rd4, %r0, 4;");
        s.AppendLine("    add.u64 %rd4, %rd3, %rd4;");
        s.AppendLine("    st.shared.f32 [%rd4], %f0;");
        s.AppendLine("    bar.sync 0;");
        // shared tree reduction
        for (int offset = BlockThreads / 2; offset > 0; offset >>= 1)
        {
            s.AppendLine($"    setp.lt.u32 %p2, %r0, {I(offset)};");
            s.AppendLine("    @!%p2 bra SKIP_" + I(offset) + ";");
            s.AppendLine("    ld.shared.f32 %f1, [%rd4];");
            s.AppendLine($"    ld.shared.f32 %f2, [%rd4+{I(offset * 4)}];");
            s.AppendLine("    add.rn.f32 %f1, %f1, %f2;");
            s.AppendLine("    st.shared.f32 [%rd4], %f1;");
            s.AppendLine("SKIP_" + I(offset) + ":");
            s.AppendLine("    bar.sync 0;");
        }
        // thread 0 writes result
        s.AppendLine("    setp.ne.u32 %p3, %r0, 0;");
        s.AppendLine("    @%p3 bra DONE;");
        s.AppendLine("    ld.shared.f32 %f0, [%rd3];");
        s.AppendLine("    mul.wide.u32 %rd5, %r1, 4;");
        s.AppendLine("    add.u64 %rd5, %rd1, %rd5;");
        s.AppendLine("    st.global.f32 [%rd5], %f0;");
        s.AppendLine("DONE:");
        s.AppendLine("    ret;");
        s.AppendLine("}");
        return s.ToString();
    }

    public void Dispose() => _module.Dispose();
}
