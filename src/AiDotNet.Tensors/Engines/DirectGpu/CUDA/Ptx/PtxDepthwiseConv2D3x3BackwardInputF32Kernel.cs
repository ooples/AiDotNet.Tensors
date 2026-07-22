using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Hand-emitted FP32 NCHW depthwise 3x3 backward-input gradient (stride 1, zero
/// pad 1) — the transpose of <see cref="PtxFusedDepthwiseConv2D3x3F32Kernel"/>.
/// For each input position the gradient is
/// <c>dIn[c,y,x] = sum_{ky,kx} w[c,ky,kx] * dOut[c, y-(ky-1), x-(kx-1)]</c>, i.e.
/// the forward correlation read with the tap offset negated. Geometry is baked
/// into PTX (three pointers, no runtime shape/stride/layout), each thread owns
/// one input pixel, nine FMA taps, zero global intermediates, zero local bytes.
/// </summary>
internal sealed class PtxDepthwiseConv2D3x3BackwardInputF32Kernel : IDisposable
{
    internal const string EntryPoint = "aidotnet_depthwise_conv2d_bwd_input_n1_c64_h16_w16_r3_s1_p1";
    internal const int BlockThreads = 128;
    internal const int Batch = 1;
    internal const int Channels = 64;
    internal const int Height = 16;
    internal const int Width = 16;
    internal const int KernelSize = 3;
    internal const int SpatialElements = Height * Width;
    internal const int OutputElements = Batch * Channels * SpatialElements;
    internal const long GradOutputBytes = (long)Batch * Channels * SpatialElements * sizeof(float);
    internal const long WeightBytes = (long)Channels * KernelSize * KernelSize * sizeof(float);
    internal const long GradInputBytes = (long)OutputElements * sizeof(float);

    internal static readonly DirectPtxConvolutionShape Shape = new(
        Batch, Channels, Height, Width, Channels, Height, Width,
        KernelSize, KernelSize, 1, 1, 1, 1, 1, 1);

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal string Ptx { get; }
    internal DirectPtxFunctionInfo FunctionInfo { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxDepthwiseConv2D3x3BackwardInputF32Kernel(DirectPtxRuntime runtime)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new NotSupportedException(
                $"Depthwise backward-input has no experimental SM {runtime.ComputeCapabilityMajor}.{runtime.ComputeCapabilityMinor} specialization.");

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
        var gradOutput = new DirectPtxExtent(Batch, Channels, Height, Width);
        var weights = new DirectPtxExtent(Channels, 1, KernelSize, KernelSize);
        var gradInput = new DirectPtxExtent(Batch, Channels, Height, Width);
        return new DirectPtxKernelBlueprint(
            Operation: "depthwise-conv2d-bwd-input",
            Version: 1,
            Architecture: architecture,
            Variant: "n1-c64-h16-w16-r3-s1-p1-fp32",
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
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["equation"] = "dIn[c,y,x] = sum_{ky,kx} w[c,ky,kx] * dOut[c,y-(ky-1),x-(kx-1)]",
                ["grad_output"] = "fp32",
                ["accumulator"] = "fp32-fma",
                ["grad_input"] = "fp32",
                ["layout"] = "nchw/oihw-depthwise",
                ["padding"] = "zero-pad-1-halo-predicated",
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
            _function, OutputElements / BlockThreads, 1, 1,
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
                "Only the experimental SM86 depthwise backward-input emitter exists.");

        var ptx = new StringBuilder(16384);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{computeCapabilityMajor}{computeCapabilityMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 gradout_ptr,");
        ptx.AppendLine("    .param .u64 weights_ptr,");
        ptx.AppendLine("    .param .u64 gradin_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<3>;");
        ptx.AppendLine("    .reg .b32 %r<7>;");
        ptx.AppendLine("    .reg .b64 %rd<10>;");
        ptx.AppendLine("    .reg .f32 %f<4>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [gradout_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [weights_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [gradin_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {BlockThreads}, %r0;");       // global input id
        ptx.AppendLine($"    and.b32 %r3, %r2, {SpatialElements - 1};");        // spatial index (y*W + x)
        ptx.AppendLine("    shr.u32 %r4, %r2, 8;");                             // channel c (256 = H*W)
        ptx.AppendLine("    shr.u32 %r5, %r3, 4;");                             // y
        ptx.AppendLine("    and.b32 %r6, %r3, 15;");                            // x
        // Center grad-output address: gradout_ptr + c*(H*W*4) + spatial*4.
        ptx.AppendLine($"    mul.wide.u32 %rd4, %r4, {SpatialElements * sizeof(float)};");
        ptx.AppendLine("    add.u64 %rd4, %rd0, %rd4;");
        ptx.AppendLine("    mul.wide.u32 %rd5, %r3, 4;");
        ptx.AppendLine("    add.u64 %rd4, %rd4, %rd5;");
        // Weight base for this channel: weights_ptr + c*(9*4).
        ptx.AppendLine($"    mul.wide.u32 %rd6, %r4, {KernelSize * KernelSize * sizeof(float)};");
        ptx.AppendLine("    add.u64 %rd6, %rd1, %rd6;");
        ptx.AppendLine("    mov.f32 %f0, 0f00000000;");                         // accumulator
        for (int k = 0; k < KernelSize * KernelSize; k++)
        {
            // Backward-input reads dOut at (y-(ky-1), x-(kx-1)); the tap offset is
            // the forward offset negated. Everything else (predicate, weight index,
            // fma) is the forward emitter unchanged.
            int dy = 1 - k / KernelSize;
            int dx = 1 - k % KernelSize;
            int tapOffsetBytes = (dy * Width + dx) * sizeof(float);
            ptx.AppendLine($"    // tap ky={1 - dy} kx={1 - dx} reads gradOut at (dy={dy}, dx={dx})");
            // Load this tap's weight (always valid): weight base + k*4.
            if (k == 0)
                ptx.AppendLine("    ld.global.nc.f32 %f2, [%rd6];");
            else
                ptx.AppendLine($"    ld.global.nc.f32 %f2, [%rd6+{k * sizeof(float)}];");

            // Halo predicate: emit a bound check only for axes whose offset is non-zero.
            bool needY = dy != 0;
            bool needX = dx != 0;
            if (!needY && !needX)
            {
                // Center tap: always valid.
                ptx.AppendLine("    ld.global.nc.f32 %f1, [%rd4];");
            }
            else
            {
                string guard;
                if (needY)
                    ptx.AppendLine(dy < 0
                        ? "    setp.ge.s32 %p0, %r5, 1;"                        // y >= 1
                        : $"    setp.lt.s32 %p0, %r5, {Height - 1};");          // y <= H-2
                if (needX)
                    ptx.AppendLine(dx < 0
                        ? "    setp.ge.s32 %p1, %r6, 1;"                        // x >= 1
                        : $"    setp.lt.s32 %p1, %r6, {Width - 1};");           // x <= W-2
                if (needY && needX)
                {
                    ptx.AppendLine("    and.pred %p2, %p0, %p1;");
                    guard = "%p2";
                }
                else
                {
                    guard = needY ? "%p0" : "%p1";
                }
                // Tap grad-output address = center +/- constant tap offset.
                if (tapOffsetBytes >= 0)
                    ptx.AppendLine($"    add.u64 %rd7, %rd4, {tapOffsetBytes};");
                else
                    ptx.AppendLine($"    sub.u64 %rd7, %rd4, {-tapOffsetBytes};");
                ptx.AppendLine("    mov.f32 %f1, 0f00000000;");
                ptx.AppendLine($"    @{guard} ld.global.nc.f32 %f1, [%rd7];");
            }
            ptx.AppendLine("    fma.rn.f32 %f0, %f1, %f2, %f0;");
        }
        ptx.AppendLine("    mul.wide.u32 %rd9, %r2, 4;");
        ptx.AppendLine("    add.u64 %rd9, %rd2, %rd9;");
        ptx.AppendLine("    st.global.f32 [%rd9], %f0;");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    public void Dispose() => _module.Dispose();
}
