using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Direct-PTX native Conv1D forward with per-output-channel bias and optional ReLU:
/// out[n,k,ol] = relu(bias[k] + sum_{c,kl} W[k,c,kl] * in[n,c,ol*stride + kl - pad]).
/// General kernel length, stride, and padding. One thread per output element; at
/// stride 1 consecutive threads own consecutive ol so the input reads (il = ol +
/// kl - pad) and output stores are coalesced (the contiguous NCL axis) -- the same
/// thread-to-memory lesson as the 2D kernels, now on the native 1D layout instead of
/// routing through Conv2D.
/// </summary>
internal sealed class PtxConv1DKernel : IDisposable
{
    internal const int BlockThreads = 256;

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int Batch { get; }
    internal int InputChannels { get; }
    internal int OutputChannels { get; }
    internal int Length { get; }
    internal int KernelLength { get; }
    internal int Stride { get; }
    internal int Padding { get; }
    internal bool Relu { get; }
    internal string Ptx { get; }
    internal DirectPtxFunctionInfo FunctionInfo { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal int OutLength => (Length + 2 * Padding - KernelLength) / Stride + 1;
    internal long InputBytes => (long)Batch * InputChannels * Length * sizeof(float);
    internal long WeightBytes => (long)OutputChannels * InputChannels * KernelLength * sizeof(float);
    internal long BiasBytes => (long)OutputChannels * sizeof(float);
    internal long OutputBytes => (long)Batch * OutputChannels * OutLength * sizeof(float);

    internal string EntryPoint => FormattableString.Invariant(
        $"aidotnet_conv1d_n{Batch}_c{InputChannels}_k{OutputChannels}_l{Length}_kl{KernelLength}_s{Stride}_p{Padding}{(Relu ? "_relu" : "")}");

    internal PtxConv1DKernel(
        DirectPtxRuntime runtime, int batch, int inputChannels, int outputChannels,
        int length, int kernelLength, int stride, int padding, bool relu = true)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new NotSupportedException("Conv1D has no experimental non-SM86 specialization.");
        if (batch <= 0 || inputChannels <= 0 || outputChannels <= 0 || length <= 0 || kernelLength <= 0)
            throw new ArgumentOutOfRangeException(nameof(batch));
        if (stride <= 0 || padding < 0) throw new ArgumentOutOfRangeException(nameof(stride));
        Batch = batch; InputChannels = inputChannels; OutputChannels = outputChannels;
        Length = length; KernelLength = kernelLength; Stride = stride; Padding = padding; Relu = relu;
        if (OutLength <= 0) throw new ArgumentException("Non-positive output length.");
        if ((long)batch * outputChannels * OutLength % BlockThreads != 0)
            throw new ArgumentException($"N*K*OL must be a multiple of {BlockThreads}.");

        Blueprint = CreateBlueprint(runtime.ArchitectureFamily);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor);
        _module = runtime.LoadModule(Ptx, allowExperimentalJitFallback: DirectPtxFeatureGate.ConvolutionExperimentOverride);
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo functionInfo);
        FunctionInfo = functionInfo;
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, BlockThreads);
        Blueprint.ResourceBudget.Validate(EntryPoint, functionInfo, BlockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(Blueprint, runtime.DeviceFingerprint, Ptx, functionInfo, BlockThreads, activeBlocks, _module);
    }

    internal DirectPtxKernelBlueprint CreateBlueprint(DirectPtxArchitectureFamily architecture)
    {
        var input = new DirectPtxExtent(Batch, InputChannels, Length);
        var weight = new DirectPtxExtent(OutputChannels, InputChannels, KernelLength);
        var bias = new DirectPtxExtent(OutputChannels);
        var output = new DirectPtxExtent(Batch, OutputChannels, OutLength);
        return new DirectPtxKernelBlueprint(
            Operation: "conv1d-forward", Version: 1, Architecture: architecture,
            Variant: FormattableString.Invariant($"n{Batch}-c{InputChannels}-k{OutputChannels}-l{Length}-kl{KernelLength}-s{Stride}-p{Padding}{(Relu ? "-relu" : "")}-fp32"),
            Tensors:
            [
                new("input", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D, input, input, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("weights", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D, weight, weight, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("bias", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector, bias, bias, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D, output, output, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(MaxRegistersPerThread: 40, MaxStaticSharedBytes: 0, MaxLocalBytesPerThread: 0, MinBlocksPerMultiprocessor: 2),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["equation"] = "out[n,k,ol] = " + (Relu ? "relu(" : "(") + "bias[k] + sum_{c,kl} W[k,c,kl]*in[n,c,ol*stride+kl-pad])",
                ["access"] = "thread-per-output, ol-contiguous -> coalesced at stride 1",
                ["shape-selection"] = "host-only-exact-contract", ["promotion"] = "experimental-pending-gpu-evidence"
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
        int total = Batch * OutputChannels * OutLength;
        _module.Launch(_function, (uint)(total / BlockThreads), 1, 1, BlockThreads, 1, 1, 0, arguments);
    }

    private static void Require(DirectPtxTensorView view, DirectPtxTensorContract contract, string parameter)
    {
        if (view.Pointer == IntPtr.Zero || view.PhysicalType != contract.PhysicalType || view.Layout != contract.Layout ||
            view.LogicalExtent != contract.LogicalExtent || view.PhysicalExtent != contract.PhysicalExtent ||
            view.ByteLength != contract.RequiredBytes || view.AllocationByteLength != contract.RequiredBytes)
            throw new ArgumentException($"{parameter} does not satisfy exact physical ABI '{contract.Name}'.", parameter);
    }

    internal string EmitPtx(int major, int minor)
    {
        if (!DirectPtxArchitecture.HasExperimentalConvolution(major, minor))
            throw new NotSupportedException("Only the experimental SM86 Conv1D emitter exists.");
        string I(int v) => v.ToString(CultureInfo.InvariantCulture);
        int c = InputChannels, kl = KernelLength, l = Length, ol = OutLength, kk = OutputChannels;
        int kol = kk * ol;                    // output batch stride
        int cl = c * l;                       // input batch stride
        int ckl = c * kl;                     // weight output-channel stride
        string entry = EntryPoint;

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
        s.AppendLine("    .reg .pred %p<4>;");
        s.AppendLine("    .reg .b32 %r<28>;");
        s.AppendLine("    .reg .b64 %rd<16>;");
        s.AppendLine("    .reg .f32 %f<8>;");
        s.AppendLine("    ld.param.u64 %rd0, [input_ptr];");
        s.AppendLine("    ld.param.u64 %rd1, [weight_ptr];");
        s.AppendLine("    ld.param.u64 %rd2, [bias_ptr];");
        s.AppendLine("    ld.param.u64 %rd3, [output_ptr];");
        s.AppendLine("    mov.u32 %r0, %tid.x;");
        s.AppendLine("    mov.u32 %r1, %ctaid.x;");
        s.AppendLine($"    mad.lo.u32 %r2, %r1, {I(BlockThreads)}, %r0;");  // idx = n*K*OL + k*OL + ol
        s.AppendLine($"    div.u32 %r3, %r2, {I(kol)};");         // n
        s.AppendLine($"    rem.u32 %r4, %r2, {I(kol)};");
        s.AppendLine($"    div.u32 %r5, %r4, {I(ol)};");          // k
        s.AppendLine($"    rem.u32 %r6, %r4, {I(ol)};");          // ol
        s.AppendLine("    mul.wide.u32 %rd4, %r5, 4;");
        s.AppendLine("    add.u64 %rd4, %rd2, %rd4;");
        s.AppendLine("    ld.global.nc.f32 %f0, [%rd4];");       // acc = bias[k]
        // il0 = ol*stride - pad
        s.AppendLine($"    mul.lo.u32 %r7, %r6, {I(Stride)};");
        s.AppendLine($"    sub.s32 %r7, %r7, {I(Padding)};");    // il0
        // input batch base: input + n*C*L
        s.AppendLine($"    mul.lo.u32 %r8, %r3, {I(cl)};");
        // weight output-channel base: weights + k*C*KL
        s.AppendLine($"    mul.lo.u32 %r9, %r5, {I(ckl)};");
        s.AppendLine("    mov.u32 %r10, 0;");                    // cc = 0
        s.AppendLine("LOOP_C:");
        // input channel base index = n*C*L + cc*L
        s.AppendLine($"    mad.lo.u32 %r11, %r10, {I(l)}, %r8;");
        // weight (k,cc) base = k*C*KL + cc*KL
        s.AppendLine($"    mad.lo.u32 %r12, %r10, {I(kl)}, %r9;");
        s.AppendLine("    mov.u32 %r13, 0;");                    // klc = 0
        s.AppendLine("LOOP_KL:");
        s.AppendLine("    add.s32 %r14, %r7, %r13;");            // il = il0 + kl
        s.AppendLine("    setp.ge.s32 %p0, %r14, 0;");
        s.AppendLine($"    setp.lt.s32 %p1, %r14, {I(l)};");
        s.AppendLine("    and.pred %p0, %p0, %p1;");
        s.AppendLine("    mov.f32 %f1, 0f00000000;");
        s.AppendLine("    add.u32 %r15, %r11, %r14;");
        s.AppendLine("    mul.wide.u32 %rd5, %r15, 4;");
        s.AppendLine("    add.u64 %rd5, %rd0, %rd5;");
        s.AppendLine("    @%p0 ld.global.nc.f32 %f1, [%rd5];");
        s.AppendLine("    add.u32 %r16, %r12, %r13;");
        s.AppendLine("    mul.wide.u32 %rd6, %r16, 4;");
        s.AppendLine("    add.u64 %rd6, %rd1, %rd6;");
        s.AppendLine("    ld.global.nc.f32 %f2, [%rd6];");
        s.AppendLine("    fma.rn.f32 %f0, %f1, %f2, %f0;");
        s.AppendLine("    add.u32 %r13, %r13, 1;");
        s.AppendLine($"    setp.lt.u32 %p0, %r13, {I(kl)};");
        s.AppendLine("    @%p0 bra LOOP_KL;");
        s.AppendLine("    add.u32 %r10, %r10, 1;");
        s.AppendLine($"    setp.lt.u32 %p0, %r10, {I(c)};");
        s.AppendLine("    @%p0 bra LOOP_C;");
        if (Relu)
            s.AppendLine("    max.f32 %f0, %f0, 0f00000000;");
        s.AppendLine("    mul.wide.u32 %rd7, %r2, 4;");
        s.AppendLine("    add.u64 %rd7, %rd3, %rd7;");
        s.AppendLine("    st.global.f32 [%rd7], %f0;");
        s.AppendLine("    ret;");
        s.AppendLine("}");
        return s.ToString();
    }

    public void Dispose() => _module.Dispose();
}
