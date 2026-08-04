using System;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Hand-emitted FP32 NCHW locally-connected convolution backward-bias (batch reduction).
/// A locally-connected layer carries an unshared per-position, per-channel bias
/// [Cout, outH, outW]; its gradient is the sum of the output gradient over the batch:
/// <c>dBias[co,oy,ox] = sum_n dOut[n,co,oy,ox]</c> over the exact N4/Cout8/outH8/outW8
/// geometry. One thread owns one bias element; since the flat bias id equals the
/// per-batch (channel,spatial) offset, batch element n lives at grad_output[n*512 + id],
/// so the thread simply strides its own lane across the batch (thread-private, no
/// atomics, deterministic). The grad tensors are rank-4 and map to byte-exact extents.
/// </summary>
internal sealed class PtxLocallyConnectedConv2DNchwBackwardBiasF32Kernel : IDisposable
{
    internal const string EntryPoint = "aidotnet_localconn2d_n4_cout8_outh8_outw8_bwd_bias";
    internal const int BlockThreads = 128;
    internal const int Batch = 4;
    internal const int OutputChannels = 8;
    internal const int OutHeight = 8;
    internal const int OutWidth = 8;
    internal const int SpatialElements = OutHeight * OutWidth;            // 64
    internal const int BiasElements = OutputChannels * SpatialElements;   // 512 (per-position per-channel)
    internal const int GradOutputElements = Batch * BiasElements;         // 2048
    internal const long GradOutputBytes = (long)GradOutputElements * sizeof(float);
    internal const long GradBiasBytes = (long)BiasElements * sizeof(float);

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal string Ptx { get; }
    internal DirectPtxFunctionInfo FunctionInfo { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxLocallyConnectedConv2DNchwBackwardBiasF32Kernel(DirectPtxRuntime runtime)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new NotSupportedException(
                $"Locally-connected Conv2D backward-bias has no experimental SM {runtime.ComputeCapabilityMajor}.{runtime.ComputeCapabilityMinor} specialization.");

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
        var gradOutput = new DirectPtxExtent(Batch, OutputChannels, OutHeight, OutWidth);
        var gradBias = new DirectPtxExtent(OutputChannels, OutHeight, OutWidth);
        return new DirectPtxKernelBlueprint(
            Operation: "locally-connected-conv2d-backward-bias",
            Version: 1,
            Architecture: architecture,
            Variant: "n4-cout8-outh8-outw8-fp32",
            Tensors:
            [
                new("grad_output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Nchw,
                    gradOutput, gradOutput, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("grad_bias", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Nchw,
                    gradBias, gradBias, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
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
                ["equation"] = "dBias[co,oy,ox] = sum_n dOut[n,co,oy,ox]",
                ["grad-output"] = "fp32",
                ["accumulator"] = "fp32-add",
                ["grad-bias"] = "fp32",
                ["reduction"] = "batch (thread-private per bias lane, no atomics)",
                ["bias"] = "unshared per-position per-channel [Cout,outH,outW]",
                ["layout"] = "nchw grad-output; [Cout,outH,outW] grad-bias",
                ["intermediate-global-bytes"] = "0",
                ["shape-selection"] = "host-only-exact-contract",
                ["promotion"] = "experimental-pending-gpu-evidence"
            });
    }

    internal unsafe void Launch(
        DirectPtxTensorView gradOutput,
        DirectPtxTensorView gradBias)
    {
        Require(gradOutput, Blueprint.Tensors[0], nameof(gradOutput));
        Require(gradBias, Blueprint.Tensors[1], nameof(gradBias));

        IntPtr gradOutputPointer = gradOutput.Pointer;
        IntPtr gradBiasPointer = gradBias.Pointer;
        void** arguments = stackalloc void*[2];
        arguments[0] = &gradOutputPointer;
        arguments[1] = &gradBiasPointer;
        _module.Launch(
            _function, BiasElements / BlockThreads, 1, 1,
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
                "Only the experimental SM86 locally-connected Conv2D backward-bias emitter exists.");

        const int batchStrideBytes = BiasElements * sizeof(float); // 2048 per batch element

        var ptx = new StringBuilder(4096);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{computeCapabilityMajor}{computeCapabilityMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 grad_output_ptr,");
        ptx.AppendLine("    .param .u64 grad_bias_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .b32 %r<4>;");
        ptx.AppendLine("    .reg .b64 %rd<6>;");
        ptx.AppendLine("    .reg .f32 %f<4>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [grad_output_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [grad_bias_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {BlockThreads}, %r0;");       // bias id = co*HW + spatial
        ptx.AppendLine("    mul.wide.u32 %rd2, %r2, 4;");
        ptx.AppendLine("    add.u64 %rd2, %rd0, %rd2;");                        // grad_output lane base (n=0)
        ptx.AppendLine("    ld.global.nc.f32 %f0, [%rd2];");                   // n = 0
        for (int n = 1; n < Batch; n++)
        {
            ptx.AppendLine($"    ld.global.nc.f32 %f1, [%rd2+{n * batchStrideBytes}];");
            ptx.AppendLine("    add.rn.f32 %f0, %f0, %f1;");                    // + dOut[n]
        }
        ptx.AppendLine("    mul.wide.u32 %rd3, %r2, 4;");
        ptx.AppendLine("    add.u64 %rd3, %rd1, %rd3;");
        ptx.AppendLine("    st.global.f32 [%rd3], %f0;");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    public void Dispose() => _module.Dispose();
}
