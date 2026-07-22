#if NET5_0_OR_GREATER
using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Hand-emitted FP32 NCHW depthwise 3x3 convolution (stride 1, zero pad 1). Each
/// channel is convolved with its own 3x3 kernel; the complete geometry is baked
/// into PTX, so the device receives only three pointers and contains no runtime
/// shape, stride, layout, or epilogue selection. Each thread computes one output
/// pixel: it reads at most nine input taps (each tap offset is a compile-time
/// constant off the center, and the halo predicate for a tap is emitted only for
/// the axes whose offset is non-zero) and nine weights, accumulating in one
/// register with fused multiply-add. There is no global intermediate.
/// </summary>
internal sealed class PtxFusedDepthwiseConv2D3x3F32Kernel : IDisposable
{
    internal const string EntryPoint = "aidotnet_depthwise_conv2d_n1_c64_h16_w16_r3_s1_p1";
    internal const int BlockThreads = 128;
    internal const int Batch = 1;
    internal const int Channels = 64;
    internal const int Height = 16;
    internal const int Width = 16;
    internal const int KernelSize = 3;
    internal const int SpatialElements = Height * Width;
    internal const int OutputElements = Batch * Channels * SpatialElements;
    internal const long InputBytes = (long)Batch * Channels * SpatialElements * sizeof(float);
    internal const long WeightBytes = (long)Channels * KernelSize * KernelSize * sizeof(float);
    internal const long OutputBytes = (long)OutputElements * sizeof(float);

    internal static readonly DirectPtxConvolutionShape Shape = new(
        Batch, Channels, Height, Width, Channels, Height, Width,
        KernelSize, KernelSize, 1, 1, 1, 1, 1, 1);

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal string Ptx { get; }
    internal DirectPtxFunctionInfo FunctionInfo { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxFusedDepthwiseConv2D3x3F32Kernel(DirectPtxRuntime runtime)
    {
        ArgumentNullException.ThrowIfNull(runtime);
        if (!DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new NotSupportedException(
                $"Depthwise convolution has no experimental SM {runtime.ComputeCapabilityMajor}.{runtime.ComputeCapabilityMinor} specialization.");

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
        var input = new DirectPtxExtent(Batch, Channels, Height, Width);
        var weights = new DirectPtxExtent(Channels, 1, KernelSize, KernelSize);
        var output = new DirectPtxExtent(Batch, Channels, Height, Width);
        return new DirectPtxKernelBlueprint(
            Operation: "depthwise-conv2d",
            Version: 1,
            Architecture: architecture,
            Variant: "n1-c64-h16-w16-r3-s1-p1-fp32",
            Tensors:
            [
                new("input", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Nchw,
                    input, input, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("weights", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Oihw,
                    weights, weights, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Nchw,
                    output, output, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                // 0 = the per-thread register ceiling is derived from the device
                // register file at validation time (see DirectPtxResourceBudget),
                // so this kernel is not pinned to a hardcoded literal and scales
                // across GPU generations. Occupancy intent below stays explicit.
                MaxRegistersPerThread: 0,
                MaxStaticSharedBytes: 0,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 4),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["equation"] = "depthwise-conv2d-nchw-3x3(input,per-channel-weights), stride1 pad1",
                ["input"] = "fp32",
                ["accumulator"] = "fp32-fma",
                ["output"] = "fp32",
                ["layout"] = "nchw/oihw-depthwise",
                ["padding"] = "zero-pad-1-halo-predicated",
                ["intermediate-global-bytes"] = "0",
                ["shape-selection"] = "host-only-exact-contract",
                ["promotion"] = "experimental-pending-gpu-evidence"
            });
    }

    internal unsafe void Launch(
        DirectPtxTensorView input,
        DirectPtxTensorView weights,
        DirectPtxTensorView output)
    {
        Require(input, Blueprint.Tensors[0], nameof(input));
        Require(weights, Blueprint.Tensors[1], nameof(weights));
        Require(output, Blueprint.Tensors[2], nameof(output));

        IntPtr inputPointer = input.Pointer;
        IntPtr weightPointer = weights.Pointer;
        IntPtr outputPointer = output.Pointer;
        void** arguments = stackalloc void*[3];
        arguments[0] = &inputPointer;
        arguments[1] = &weightPointer;
        arguments[2] = &outputPointer;
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
            throw new NotSupportedException("Only the experimental SM86 depthwise convolution emitter exists.");

        var ptx = new StringBuilder(16384);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{computeCapabilityMajor}{computeCapabilityMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 input_ptr,");
        ptx.AppendLine("    .param .u64 weights_ptr,");
        ptx.AppendLine("    .param .u64 output_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<3>;");
        ptx.AppendLine("    .reg .b32 %r<7>;");
        ptx.AppendLine("    .reg .b64 %rd<10>;");
        ptx.AppendLine("    .reg .f32 %f<4>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [input_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [weights_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [output_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {BlockThreads}, %r0;");       // global output id
        ptx.AppendLine($"    and.b32 %r3, %r2, {SpatialElements - 1};");        // spatial index (y*W + x)
        ptx.AppendLine("    shr.u32 %r4, %r2, 8;");                             // channel c (256 = H*W)
        ptx.AppendLine("    shr.u32 %r5, %r3, 4;");                             // y
        ptx.AppendLine("    and.b32 %r6, %r3, 15;");                            // x
        // Center input address: input_ptr + c*(H*W*4) + spatial*4.
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
            int dy = k / KernelSize - 1;
            int dx = k % KernelSize - 1;
            int tapOffsetBytes = (dy * Width + dx) * sizeof(float);
            ptx.AppendLine($"    // tap ky={dy + 1} kx={dx + 1} (dy={dy}, dx={dx})");
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
                // Tap input address = center +/- constant tap offset.
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
#endif
