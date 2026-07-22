using System;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Hand-emitted FP32 NCHW locally-connected 3x3 convolution backward-weight (stride 1,
/// zero pad 1). Because the weights are unshared, each one is used by exactly one output
/// position, so its gradient is a single product — there is no spatial reduction:
/// <c>dW[oy,ox,co,ci,ky,kx] = dOut[co,oy,ox] * in[ci, oy+ky-1, ox+kx-1]</c>
/// (zero when the input tap falls in the zero pad), over the exact N1/Cin8/H8/W8/Cout8
/// geometry. One thread owns one weight: it decomposes the flat weight id back into
/// (pos_out, co, ci, ky, kx), loads the one gradient-output element and the one
/// halo-predicated input element, multiplies, and stores. Fully independent per weight
/// (deterministic, no atomics). The per-position weight tensor is rank-6, so the ABI
/// contract uses byte-exact collapsed rank-4 extents; the emitter owns the true strides.
/// </summary>
internal sealed class PtxLocallyConnectedConv2DNchw3x3BackwardWeightF32Kernel : IDisposable
{
    internal const string EntryPoint = "aidotnet_localconn2d_n1_cin8_h8_w8_cout8_r3_s1_p1_bwd_weight";
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
    internal const long InputBytes = (long)InputElements * sizeof(float);
    internal const long GradWeightBytes = (long)WeightElements * sizeof(float);

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal string Ptx { get; }
    internal DirectPtxFunctionInfo FunctionInfo { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxLocallyConnectedConv2DNchw3x3BackwardWeightF32Kernel(DirectPtxRuntime runtime)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new NotSupportedException(
                $"Locally-connected Conv2D 3x3 backward-weight has no experimental SM {runtime.ComputeCapabilityMajor}.{runtime.ComputeCapabilityMinor} specialization.");

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
        var input = new DirectPtxExtent(Batch, InputChannels, Height, Width);
        var gradWeight = new DirectPtxExtent(SpatialElements, OutputChannels, InputChannels, TapsPerChannel);
        return new DirectPtxKernelBlueprint(
            Operation: "locally-connected-conv2d-backward-weight",
            Version: 1,
            Architecture: architecture,
            Variant: "n1-cin8-h8-w8-cout8-r3-s1-p1-fp32",
            Tensors:
            [
                new("grad_output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Nchw,
                    gradOutput, gradOutput, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("input", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Nchw,
                    input, input, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("grad_weight", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Oihw,
                    gradWeight, gradWeight, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
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
                ["equation"] = "dW[oy,ox,co,ci,ky,kx] = dOut[co,oy,ox] * in[ci, oy+ky-1, ox+kx-1]",
                ["grad-output"] = "fp32",
                ["accumulator"] = "fp32-single-product",
                ["grad-weight"] = "fp32",
                ["reduction"] = "none (unshared weight = single output usage)",
                ["weight-sharing"] = "none (per output position)",
                ["layout"] = "nchw grad/input; per-position [outHW,Cout,Cin,kH,kW] grad-weight (abi extents collapsed to rank-4 byte-exact)",
                ["padding"] = "zero-pad-1 halo-predicated on iy/ix",
                ["intermediate-global-bytes"] = "0",
                ["shape-selection"] = "host-only-exact-contract",
                ["promotion"] = "experimental-pending-gpu-evidence"
            });
    }

    internal unsafe void Launch(
        DirectPtxTensorView gradOutput,
        DirectPtxTensorView input,
        DirectPtxTensorView gradWeight)
    {
        Require(gradOutput, Blueprint.Tensors[0], nameof(gradOutput));
        Require(input, Blueprint.Tensors[1], nameof(input));
        Require(gradWeight, Blueprint.Tensors[2], nameof(gradWeight));

        IntPtr gradOutputPointer = gradOutput.Pointer;
        IntPtr inputPointer = input.Pointer;
        IntPtr gradWeightPointer = gradWeight.Pointer;
        void** arguments = stackalloc void*[3];
        arguments[0] = &gradOutputPointer;
        arguments[1] = &inputPointer;
        arguments[2] = &gradWeightPointer;
        _module.Launch(
            _function, WeightElements / BlockThreads, 1, 1,
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
                "Only the experimental SM86 locally-connected Conv2D 3x3 backward-weight emitter exists.");

        var ptx = new StringBuilder(8192);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{computeCapabilityMajor}{computeCapabilityMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 grad_output_ptr,");
        ptx.AppendLine("    .param .u64 input_ptr,");
        ptx.AppendLine("    .param .u64 grad_weight_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<6>;");
        ptx.AppendLine("    .reg .b32 %r<24>;");
        ptx.AppendLine("    .reg .b64 %rd<8>;");
        ptx.AppendLine("    .reg .f32 %f<4>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [grad_output_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [input_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [grad_weight_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {BlockThreads}, %r0;");       // flat weight id
        // Decompose id = pos_out*576 + co*72 + ci*9 + tap_kk.
        ptx.AppendLine("    div.u32 %r3, %r2, 9;");                            // rest = id / 9
        ptx.AppendLine("    mul.lo.u32 %r4, %r3, 9;");
        ptx.AppendLine("    sub.u32 %r5, %r2, %r4;");                          // tap_kk = id - rest*9
        ptx.AppendLine("    and.b32 %r6, %r3, 7;");                            // ci = rest & 7
        ptx.AppendLine("    shr.u32 %r7, %r3, 3;");                            // rest2 = rest >> 3
        ptx.AppendLine("    and.b32 %r8, %r7, 7;");                            // co = rest2 & 7
        ptx.AppendLine("    shr.u32 %r9, %r7, 3;");                            // pos_out = rest2 >> 3
        ptx.AppendLine("    shr.u32 %r10, %r9, 3;");                           // oy = pos_out >> 3
        ptx.AppendLine("    and.b32 %r11, %r9, 7;");                           // ox = pos_out & 7
        ptx.AppendLine("    div.u32 %r12, %r5, 3;");                           // ky = tap_kk / 3
        ptx.AppendLine("    mul.lo.u32 %r13, %r12, 3;");
        ptx.AppendLine("    sub.u32 %r14, %r5, %r13;");                        // kx = tap_kk - ky*3
        ptx.AppendLine("    add.s32 %r15, %r10, %r12;");
        ptx.AppendLine("    sub.s32 %r15, %r15, 1;");                          // iy = oy + ky - 1
        ptx.AppendLine("    add.s32 %r16, %r11, %r14;");
        ptx.AppendLine("    sub.s32 %r16, %r16, 1;");                          // ix = ox + kx - 1
        // Halo predicate: iy in [0,H) and ix in [0,W).
        ptx.AppendLine("    setp.ge.s32 %p0, %r15, 0;");
        ptx.AppendLine($"    setp.lt.s32 %p1, %r15, {Height};");
        ptx.AppendLine("    setp.ge.s32 %p2, %r16, 0;");
        ptx.AppendLine($"    setp.lt.s32 %p3, %r16, {Width};");
        ptx.AppendLine("    and.pred %p4, %p0, %p1;");
        ptx.AppendLine("    and.pred %p4, %p4, %p2;");
        ptx.AppendLine("    and.pred %p4, %p4, %p3;");
        // grad_output index = co*64 + pos_out (always valid).
        ptx.AppendLine($"    mad.lo.u32 %r17, %r8, {SpatialElements}, %r9;");
        ptx.AppendLine("    mul.wide.u32 %rd3, %r17, 4;");
        ptx.AppendLine("    add.u64 %rd3, %rd0, %rd3;");
        ptx.AppendLine("    ld.global.nc.f32 %f2, [%rd3];");                   // dOut[co,oy,ox]
        // input index = ci*64 + iy*W + ix (predicated by halo).
        ptx.AppendLine($"    mad.lo.u32 %r18, %r15, {Width}, %r16;");             // iy*W + ix
        ptx.AppendLine($"    mad.lo.u32 %r18, %r6, {Height * Width}, %r18;");      // ci*(H*W) + iy*W + ix
        ptx.AppendLine("    mul.wide.u32 %rd4, %r18, 4;");
        ptx.AppendLine("    add.u64 %rd4, %rd1, %rd4;");
        ptx.AppendLine("    mov.f32 %f1, 0f00000000;");
        ptx.AppendLine("    @%p4 ld.global.nc.f32 %f1, [%rd4];");             // in[ci,iy,ix] or 0 in pad
        ptx.AppendLine("    mul.f32 %f0, %f2, %f1;");
        // grad_weight[id] = grad_weight + id*4.
        ptx.AppendLine("    mul.wide.u32 %rd5, %r2, 4;");
        ptx.AppendLine("    add.u64 %rd5, %rd2, %rd5;");
        ptx.AppendLine("    st.global.f32 [%rd5], %f0;");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    public void Dispose() => _module.Dispose();
}
