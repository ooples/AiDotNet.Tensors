using System;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Hand-emitted FP32 NCHW locally-connected 3x3 convolution backward-input (stride 1,
/// zero pad 1). The transpose gather of
/// <see cref="PtxLocallyConnectedConv2DNchw3x3F32Kernel"/>: each input element sums the
/// contributions it made to every output position that used it, with that position's
/// unshared filter bank:
/// <c>dX[ci,iy,ix] = sum_co sum_ky sum_kx W[oy,ox,co,ci,ky,kx] * dOut[co,oy,ox]</c>
/// where <c>oy = iy-ky+1, ox = ix-kx+1</c>, over the exact N1/Cin8/H8/W8/Cout8 geometry.
/// One thread owns one input element. Since <c>oy*W+ox = spatial - (dy*W+dx)</c> (with
/// dy=ky-1, dx=kx-1), each tap's output position is a compile-time offset from the
/// thread's own spatial index; the nine taps are unrolled with a per-tap validity branch
/// and each valid tap accumulates over all output channels (thread-private, no atomics,
/// deterministic). The per-position weight tensor is rank-6, so the ABI contract uses
/// byte-exact collapsed rank-4 extents; the emitter owns the true strides.
/// </summary>
internal sealed class PtxLocallyConnectedConv2DNchw3x3BackwardInputF32Kernel : IDisposable
{
    internal const string EntryPoint = "aidotnet_localconn2d_n1_cin8_h8_w8_cout8_r3_s1_p1_bwd_input";
    internal const int BlockThreads = 128;
    internal const int Batch = 1;
    internal const int InputChannels = 8;
    internal const int Height = 8;
    internal const int Width = 8;
    internal const int OutputChannels = 8;
    internal const int OutHeight = Height;   // stride 1, pad 1
    internal const int OutWidth = Width;
    internal const int KernelSize = 3;
    internal const int TapsPerChannel = KernelSize * KernelSize;         // 9
    internal const int SpatialElements = OutHeight * OutWidth;           // 64
    internal const int OutputElements = Batch * OutputChannels * SpatialElements; // 512
    internal const int InputElements = Batch * InputChannels * Height * Width;    // 512
    internal const int WeightElements =
        SpatialElements * OutputChannels * InputChannels * TapsPerChannel; // 36864
    internal const long GradOutputBytes = (long)OutputElements * sizeof(float);
    internal const long WeightBytes = (long)WeightElements * sizeof(float);
    internal const long GradInputBytes = (long)InputElements * sizeof(float);

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal string Ptx { get; }
    internal DirectPtxFunctionInfo FunctionInfo { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxLocallyConnectedConv2DNchw3x3BackwardInputF32Kernel(DirectPtxRuntime runtime)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new NotSupportedException(
                $"Locally-connected Conv2D 3x3 backward-input has no experimental SM {runtime.ComputeCapabilityMajor}.{runtime.ComputeCapabilityMinor} specialization.");

        Blueprint = CreateBlueprint(runtime.ArchitectureFamily);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor);
        _module = runtime.LoadModule(Ptx);
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo functionInfo);
        FunctionInfo = functionInfo;
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, BlockThreads);
        Blueprint.ResourceBudget.Validate(EntryPoint, functionInfo, BlockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Ptx, functionInfo,
            BlockThreads, activeBlocks, _module.JitInfoLog);
    }

    internal static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture)
    {
        // The locally-connected weight tensor is rank-6 [outH*outW, Cout, Cin, kH, kW];
        // collapse to byte-exact rank-4 extents for the ABI contract. The emitter owns
        // the true per-position strides.
        var gradOutput = new DirectPtxExtent(Batch, OutputChannels, OutHeight, OutWidth);
        var weights = new DirectPtxExtent(SpatialElements, OutputChannels, InputChannels, TapsPerChannel);
        var gradInput = new DirectPtxExtent(Batch, InputChannels, Height, Width);
        return new DirectPtxKernelBlueprint(
            Operation: "locally-connected-conv2d-backward-input",
            Version: 1,
            Architecture: architecture,
            Variant: "n1-cin8-h8-w8-cout8-r3-s1-p1-fp32",
            Tensors:
            [
                new("grad_output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Nchw,
                    gradOutput, gradOutput, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("weights", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Oihw,
                    weights, weights, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("grad_input", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Nchw,
                    gradInput, gradInput, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                // 0 = per-thread register ceiling derived from the device register
                // file at validation time; not pinned to a hardcoded literal.
                MaxRegistersPerThread: 0,
                MaxStaticSharedBytes: 0,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 4),
            Semantics: new System.Collections.Generic.Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["equation"] = "dX[ci,iy,ix] = sum_co sum_ky sum_kx W[oy,ox,co,ci,ky,kx] * dOut[co,oy,ox], oy=iy-ky+1, ox=ix-kx+1",
                ["grad-output"] = "fp32",
                ["accumulator"] = "fp32-fma",
                ["grad-input"] = "fp32",
                ["reduction"] = "thread-private over co (deterministic, no atomics)",
                ["weight-sharing"] = "none (per output position)",
                ["layout"] = "nchw grad; per-position [outHW,Cout,Cin,kH,kW] weights (abi extents collapsed to rank-4 byte-exact)",
                ["padding"] = "zero-pad-1 transpose halo-predicated on iy/ix",
                ["intermediate-global-bytes"] = "0",
                ["shape-selection"] = "host-only-exact-contract",
                ["promotion"] = "experimental-pending-gpu-evidence"
            });
    }

    internal unsafe void Launch(
        DirectPtxTensorView gradOutput,
        DirectPtxTensorView weights,
        DirectPtxTensorView gradInput)
    {
        Require(gradOutput, Blueprint.Tensors[0], nameof(gradOutput));
        Require(weights, Blueprint.Tensors[1], nameof(weights));
        Require(gradInput, Blueprint.Tensors[2], nameof(gradInput));

        IntPtr gradOutputPointer = gradOutput.Pointer;
        IntPtr weightPointer = weights.Pointer;
        IntPtr gradInputPointer = gradInput.Pointer;
        void** arguments = stackalloc void*[3];
        arguments[0] = &gradOutputPointer;
        arguments[1] = &weightPointer;
        arguments[2] = &gradInputPointer;
        _module.Launch(
            _function, InputElements / BlockThreads, 1, 1,
            BlockThreads, 1, 1, 0, arguments);
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

    internal static string EmitPtx(int computeCapabilityMajor, int computeCapabilityMinor)
    {
        if (!DirectPtxArchitecture.HasExperimentalConvolution(
                computeCapabilityMajor, computeCapabilityMinor))
            throw new NotSupportedException(
                "Only the experimental SM86 locally-connected Conv2D 3x3 backward-input emitter exists.");

        const int weightPositionStrideElems = OutputChannels * InputChannels * TapsPerChannel; // 576 per output position
        const int weightChannelStrideBytes = InputChannels * TapsPerChannel * sizeof(float);    // 288 per co
        const int gradOutChannelStrideBytes = SpatialElements * sizeof(float);                   // 256 per co
        // Weight index = pos*576 + co*72 + ci*9 + tap_kk. We fold pos*576 + ci*9 via r7 = ci*9,
        // then advance +288 bytes (72 elems) per co inside the tap.

        var ptx = new StringBuilder(24576);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{computeCapabilityMajor}{computeCapabilityMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 grad_output_ptr,");
        ptx.AppendLine("    .param .u64 weights_ptr,");
        ptx.AppendLine("    .param .u64 grad_input_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<6>;");
        ptx.AppendLine("    .reg .b32 %r<12>;");
        ptx.AppendLine("    .reg .b64 %rd<8>;");
        ptx.AppendLine("    .reg .f32 %f<4>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [grad_output_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [weights_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [grad_input_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {BlockThreads}, %r0;");       // global input id = ci*HW + spatial
        ptx.AppendLine($"    and.b32 %r3, %r2, {SpatialElements - 1};");        // spatial = iy*W + ix
        ptx.AppendLine("    shr.u32 %r6, %r2, 6;");                            // ci (64 = H*W)
        ptx.AppendLine("    shr.u32 %r4, %r3, 3;");                            // iy = spatial / W (8)
        ptx.AppendLine($"    and.b32 %r5, %r3, {Width - 1};");                 // ix = spatial % W
        // Four transpose boundary predicates from (iy, ix).
        ptx.AppendLine("    setp.ge.s32 %p0, %r4, 1;");                         // iy >= 1        (for dy>0: oy=iy-1)
        ptx.AppendLine($"    setp.lt.s32 %p1, %r4, {OutHeight - 1};");         // iy <= outH-2   (for dy<0: oy=iy+1)
        ptx.AppendLine("    setp.ge.s32 %p2, %r5, 1;");                         // ix >= 1        (for dx>0: ox=ix-1)
        ptx.AppendLine($"    setp.lt.s32 %p3, %r5, {OutWidth - 1};");          // ix <= outW-2   (for dx<0: ox=ix+1)
        ptx.AppendLine("    mul.lo.u32 %r7, %r6, 9;");                          // ci*9 (weight ci offset within block)
        ptx.AppendLine("    mov.f32 %f0, 0f00000000;");                        // accumulator
        for (int m = 0; m < TapsPerChannel; m++)
        {
            int ky = m / KernelSize;
            int kx = m % KernelSize;
            int dy = ky - 1;
            int dx = kx - 1;
            int tapShift = dy * OutWidth + dx;   // pos_out = spatial - tapShift
            string skipLabel = $"LC_BWDIN_SKIP_{m}";
            bool needY = dy != 0;
            bool needX = dx != 0;
            // Per-tap validity: oy=iy-dy in [0,outH), ox=ix-dx in [0,outW).
            if (needY || needX)
            {
                string yGuard = dy < 0 ? "%p1" : "%p0";
                string xGuard = dx < 0 ? "%p3" : "%p2";
                if (needY && needX)
                {
                    ptx.AppendLine($"    and.pred %p4, {yGuard}, {xGuard};");
                    ptx.AppendLine($"    @!%p4 bra {skipLabel};");
                }
                else
                {
                    ptx.AppendLine($"    @!{(needY ? yGuard : xGuard)} bra {skipLabel};");
                }
            }
            // pos_out = spatial - tapShift.
            if (tapShift == 0)
                ptx.AppendLine("    mov.u32 %r8, %r3;");
            else if (tapShift > 0)
                ptx.AppendLine($"    sub.u32 %r8, %r3, {tapShift};");
            else
                ptx.AppendLine($"    add.u32 %r8, %r3, {-tapShift};");
            // weight index (co=0) = pos_out*576 + ci*9 + tap_kk.
            ptx.AppendLine($"    mad.lo.u32 %r9, %r8, {weightPositionStrideElems}, %r7;");
            if (m > 0)
                ptx.AppendLine($"    add.u32 %r9, %r9, {m};");
            ptx.AppendLine("    mul.wide.u32 %rd3, %r9, 4;");
            ptx.AppendLine("    add.u64 %rd3, %rd1, %rd3;");                    // weight base (co=0)
            ptx.AppendLine("    mul.wide.u32 %rd4, %r8, 4;");
            ptx.AppendLine("    add.u64 %rd4, %rd0, %rd4;");                    // grad_output base (co=0)
            for (int co = 0; co < OutputChannels; co++)
            {
                int wOff = co * weightChannelStrideBytes;
                int dOff = co * gradOutChannelStrideBytes;
                ptx.AppendLine(wOff == 0
                    ? "    ld.global.nc.f32 %f2, [%rd3];"
                    : $"    ld.global.nc.f32 %f2, [%rd3+{wOff}];");
                ptx.AppendLine(dOff == 0
                    ? "    ld.global.nc.f32 %f1, [%rd4];"
                    : $"    ld.global.nc.f32 %f1, [%rd4+{dOff}];");
                ptx.AppendLine("    fma.rn.f32 %f0, %f2, %f1, %f0;");
            }
            if (needY || needX)
                ptx.AppendLine($"{skipLabel}:");
        }
        // grad_input[ci,iy,ix] = grad_input + id*4 (id = ci*HW + spatial = r2).
        ptx.AppendLine("    mul.wide.u32 %rd5, %r2, 4;");
        ptx.AppendLine("    add.u64 %rd5, %rd2, %rd5;");
        ptx.AppendLine("    st.global.f32 [%rd5], %f0;");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    public void Dispose() => _module.Dispose();
}
