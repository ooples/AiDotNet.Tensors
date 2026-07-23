using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Exact-shape fused-linear backward family. Activation derivatives are
/// recomputed in registers independently in each of three output kernels, so
/// no masked-gradient tensor is materialized in global memory.
/// </summary>
internal sealed class PtxFusedLinearBackwardKernel : IDisposable
{
    internal const int BlockThreads = 256;
    internal const string GradInputEntryPoint = "aidotnet_fused_linear_backward_input";
    internal const string GradWeightEntryPoint = "aidotnet_fused_linear_backward_weight";
    internal const string GradBiasEntryPoint = "aidotnet_fused_linear_backward_bias";

    private readonly DirectPtxModule _module;
    private readonly IntPtr _gradInputFunction;
    private readonly IntPtr _gradWeightFunction;
    private readonly IntPtr _gradBiasFunction;

    internal int M { get; }
    internal int K { get; }
    internal int N { get; }
    internal DirectPtxLinearActivation Activation { get; }
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal IReadOnlyList<DirectPtxKernelAudit> Audits { get; }

    internal PtxFusedLinearBackwardKernel(
        DirectPtxRuntime runtime,
        int m,
        int k,
        int n,
        DirectPtxLinearActivation activation)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasValidatedFusedLinear(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The fused-linear backward PTX family is measured only on GA10x/SM86.");
        Validate(m, k, n, activation);

        M = m;
        K = k;
        N = n;
        Activation = activation;
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, m, k, n, activation);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor, m, k, n, activation);
        _module = runtime.LoadModule(Ptx);
        _gradInputFunction = LoadAndAudit(
            runtime, GradInputEntryPoint, out DirectPtxKernelAudit gradInputAudit);
        _gradWeightFunction = LoadAndAudit(
            runtime, GradWeightEntryPoint, out DirectPtxKernelAudit gradWeightAudit);
        _gradBiasFunction = LoadAndAudit(
            runtime, GradBiasEntryPoint, out DirectPtxKernelAudit gradBiasAudit);
        Audits = [gradInputAudit, gradWeightAudit, gradBiasAudit];
    }

    private IntPtr LoadAndAudit(
        DirectPtxRuntime runtime,
        string entryPoint,
        out DirectPtxKernelAudit audit)
    {
        IntPtr function = _module.GetFunction(entryPoint, out DirectPtxFunctionInfo info);
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(function, BlockThreads);
        Blueprint.ResourceBudget.Validate(entryPoint, info, BlockThreads, activeBlocks);
        audit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Ptx, info,
            BlockThreads, activeBlocks, _module);
        return function;
    }

    internal unsafe void Launch(
        DirectPtxTensorView gradOutput,
        DirectPtxTensorView input,
        DirectPtxTensorView weights,
        DirectPtxTensorView saved,
        DirectPtxTensorView gradInput,
        DirectPtxTensorView gradWeight,
        DirectPtxTensorView gradBias)
    {
        Require(gradOutput, Blueprint.Tensors[0], nameof(gradOutput));
        Require(input, Blueprint.Tensors[1], nameof(input));
        Require(weights, Blueprint.Tensors[2], nameof(weights));
        Require(saved, Blueprint.Tensors[3], nameof(saved));
        Require(gradInput, Blueprint.Tensors[4], nameof(gradInput));
        Require(gradWeight, Blueprint.Tensors[5], nameof(gradWeight));
        Require(gradBias, Blueprint.Tensors[6], nameof(gradBias));
        if (Overlaps(gradInput, gradOutput) || Overlaps(gradInput, input) ||
            Overlaps(gradInput, weights) || Overlaps(gradInput, saved) ||
            Overlaps(gradWeight, gradOutput) || Overlaps(gradWeight, input) ||
            Overlaps(gradWeight, weights) || Overlaps(gradWeight, saved) ||
            Overlaps(gradBias, gradOutput) || Overlaps(gradBias, input) ||
            Overlaps(gradBias, weights) || Overlaps(gradBias, saved))
            throw new ArgumentException(
                "Fused-linear gradients may not alias saved forward tensors.");
        if (Overlaps(gradInput, gradWeight) || Overlaps(gradInput, gradBias) ||
            Overlaps(gradWeight, gradBias))
            throw new ArgumentException("Fused-linear gradient outputs must be disjoint.");

        IntPtr go = gradOutput.Pointer;
        IntPtr x = input.Pointer;
        IntPtr w = weights.Pointer;
        IntPtr s = saved.Pointer;
        IntPtr gi = gradInput.Pointer;
        IntPtr gw = gradWeight.Pointer;
        IntPtr gb = gradBias.Pointer;

        void** inputArgs = stackalloc void*[4];
        inputArgs[0] = &go; inputArgs[1] = &w; inputArgs[2] = &s; inputArgs[3] = &gi;
        _module.Launch(
            _gradInputFunction, Grid(checked(M * K)), 1, 1,
            BlockThreads, 1, 1, 0, inputArgs);

        void** weightArgs = stackalloc void*[4];
        weightArgs[0] = &go; weightArgs[1] = &x; weightArgs[2] = &s; weightArgs[3] = &gw;
        _module.Launch(
            _gradWeightFunction, Grid(checked(K * N)), 1, 1,
            BlockThreads, 1, 1, 0, weightArgs);

        void** biasArgs = stackalloc void*[3];
        biasArgs[0] = &go; biasArgs[1] = &s; biasArgs[2] = &gb;
        _module.Launch(
            _gradBiasFunction, Grid(N), 1, 1,
            BlockThreads, 1, 1, 0, biasArgs);
    }

    private static uint Grid(int elements) =>
        checked((uint)(((long)elements + BlockThreads - 1) / BlockThreads));

    public void Dispose() => _module.Dispose();

    internal static string EmitPtx(
        int ccMajor,
        int ccMinor,
        int m,
        int k,
        int n,
        DirectPtxLinearActivation activation)
    {
        Validate(m, k, n, activation);
        var ptx = new StringBuilder(20_000);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine($"// fused-linear backward M={m} K={k} N={n} act={activation}; exact pointer-only ABIs");
        EmitGradInput(ptx, m, k, n, activation);
        EmitGradWeight(ptx, m, k, n, activation);
        EmitGradBias(ptx, m, n, activation);
        return ptx.ToString();
    }

    private static void EmitGradInput(
        StringBuilder ptx, int m, int k, int n, DirectPtxLinearActivation activation)
    {
        EmitHeader(ptx, GradInputEntryPoint, "grad_output_ptr", "weights_ptr", "saved_ptr", "grad_input_ptr");
        EmitRegisters(ptx);
        EmitPointers(ptx, "grad_output_ptr", "weights_ptr", "saved_ptr", "grad_input_ptr");
        EmitIndex(ptx, checked(m * k), k);
        ptx.AppendLine("    mov.f32 %f7, 0f00000000;");
        ptx.AppendLine("    mov.u32 %r4, 0;");
        ptx.AppendLine("GI_LOOP:");
        ptx.AppendLine($"    setp.ge.u32 %p1, %r4, {n};");
        ptx.AppendLine("    @%p1 bra.uni GI_STORE;");
        EmitRowColumnOffset(ptx, "%r2", n, "%r4", "%rd4");
        ptx.AppendLine("    add.u64 %rd5, %rd0, %rd4;");
        ptx.AppendLine("    add.u64 %rd6, %rd2, %rd4;");
        ptx.AppendLine("    ld.global.nc.f32 %f0, [%rd5];");
        ptx.AppendLine("    ld.global.nc.f32 %f1, [%rd6];");
        EmitMaskedGradient(ptx, activation);
        EmitRowColumnOffset(ptx, "%r3", n, "%r4", "%rd7");
        ptx.AppendLine("    add.u64 %rd8, %rd1, %rd7;");
        ptx.AppendLine("    ld.global.nc.f32 %f3, [%rd8];");
        ptx.AppendLine("    fma.rn.f32 %f7, %f2, %f3, %f7;");
        ptx.AppendLine("    add.u32 %r4, %r4, 1;");
        ptx.AppendLine("    bra.uni GI_LOOP;");
        ptx.AppendLine("GI_STORE:");
        ptx.AppendLine("    mul.wide.u32 %rd9, %r1, 4;");
        ptx.AppendLine("    add.u64 %rd10, %rd3, %rd9;");
        ptx.AppendLine("    st.global.f32 [%rd10], %f7;");
        EmitFooter(ptx);
    }

    private static void EmitGradWeight(
        StringBuilder ptx, int m, int k, int n, DirectPtxLinearActivation activation)
    {
        EmitHeader(ptx, GradWeightEntryPoint, "grad_output_ptr", "input_ptr", "saved_ptr", "grad_weight_ptr");
        EmitRegisters(ptx);
        EmitPointers(ptx, "grad_output_ptr", "input_ptr", "saved_ptr", "grad_weight_ptr");
        EmitIndex(ptx, checked(k * n), n); // r2=k index, r3=n index
        ptx.AppendLine("    mov.f32 %f7, 0f00000000;");
        ptx.AppendLine("    mov.u32 %r4, 0;");
        ptx.AppendLine("GW_LOOP:");
        ptx.AppendLine($"    setp.ge.u32 %p1, %r4, {m};");
        ptx.AppendLine("    @%p1 bra.uni GW_STORE;");
        EmitRowColumnOffset(ptx, "%r4", n, "%r3", "%rd4");
        ptx.AppendLine("    add.u64 %rd5, %rd0, %rd4;");
        ptx.AppendLine("    add.u64 %rd6, %rd2, %rd4;");
        ptx.AppendLine("    ld.global.nc.f32 %f0, [%rd5];");
        ptx.AppendLine("    ld.global.nc.f32 %f1, [%rd6];");
        EmitMaskedGradient(ptx, activation);
        EmitRowColumnOffset(ptx, "%r4", k, "%r2", "%rd7");
        ptx.AppendLine("    add.u64 %rd8, %rd1, %rd7;");
        ptx.AppendLine("    ld.global.nc.f32 %f3, [%rd8];");
        ptx.AppendLine("    fma.rn.f32 %f7, %f3, %f2, %f7;");
        ptx.AppendLine("    add.u32 %r4, %r4, 1;");
        ptx.AppendLine("    bra.uni GW_LOOP;");
        ptx.AppendLine("GW_STORE:");
        ptx.AppendLine("    mul.wide.u32 %rd9, %r1, 4;");
        ptx.AppendLine("    add.u64 %rd10, %rd3, %rd9;");
        ptx.AppendLine("    st.global.f32 [%rd10], %f7;");
        EmitFooter(ptx);
    }

    private static void EmitGradBias(
        StringBuilder ptx, int m, int n, DirectPtxLinearActivation activation)
    {
        EmitHeader(ptx, GradBiasEntryPoint, "grad_output_ptr", "saved_ptr", "grad_bias_ptr");
        EmitRegisters(ptx);
        EmitPointers(ptx, "grad_output_ptr", "saved_ptr", "grad_bias_ptr");
        ptx.AppendLine("    mov.u32 %r0, %ctaid.x;");
        ptx.AppendLine("    mov.u32 %r1, %tid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r1, %r0, {BlockThreads}, %r1;");
        ptx.AppendLine($"    setp.ge.u32 %p0, %r1, {n};");
        ptx.AppendLine("    @%p0 bra.uni KERNEL_DONE;");
        ptx.AppendLine("    mov.f32 %f7, 0f00000000;");
        ptx.AppendLine("    mov.u32 %r4, 0;");
        ptx.AppendLine("GB_LOOP:");
        ptx.AppendLine($"    setp.ge.u32 %p1, %r4, {m};");
        ptx.AppendLine("    @%p1 bra.uni GB_STORE;");
        EmitRowColumnOffset(ptx, "%r4", n, "%r1", "%rd4");
        ptx.AppendLine("    add.u64 %rd5, %rd0, %rd4;");
        ptx.AppendLine("    add.u64 %rd6, %rd1, %rd4;");
        ptx.AppendLine("    ld.global.nc.f32 %f0, [%rd5];");
        ptx.AppendLine("    ld.global.nc.f32 %f1, [%rd6];");
        EmitMaskedGradient(ptx, activation);
        ptx.AppendLine("    add.rn.f32 %f7, %f7, %f2;");
        ptx.AppendLine("    add.u32 %r4, %r4, 1;");
        ptx.AppendLine("    bra.uni GB_LOOP;");
        ptx.AppendLine("GB_STORE:");
        ptx.AppendLine("    mul.wide.u32 %rd7, %r1, 4;");
        ptx.AppendLine("    add.u64 %rd8, %rd2, %rd7;");
        ptx.AppendLine("    st.global.f32 [%rd8], %f7;");
        EmitFooter(ptx);
    }

    private static void EmitHeader(StringBuilder ptx, string entry, params string[] parameters)
    {
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {entry}(");
        for (int i = 0; i < parameters.Length; i++)
            ptx.AppendLine($"    .param .u64 {parameters[i]}{(i + 1 == parameters.Length ? string.Empty : ",")}");
        ptx.AppendLine(")");
        ptx.AppendLine($".maxntid {BlockThreads}, 1, 1");
        ptx.AppendLine("{");
    }

    private static void EmitRegisters(StringBuilder ptx)
    {
        ptx.AppendLine("    .reg .pred %p<6>;");
        ptx.AppendLine("    .reg .b32 %r<16>;");
        ptx.AppendLine("    .reg .b64 %rd<16>;");
        ptx.AppendLine("    .reg .f32 %f<16>;");
    }

    private static void EmitPointers(StringBuilder ptx, params string[] names)
    {
        for (int i = 0; i < names.Length; i++)
            ptx.AppendLine($"    ld.param.u64 %rd{i}, [{names[i]}];");
    }

    private static void EmitIndex(StringBuilder ptx, int total, int columns)
    {
        ptx.AppendLine("    mov.u32 %r0, %ctaid.x;");
        ptx.AppendLine("    mov.u32 %r1, %tid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r1, %r0, {BlockThreads}, %r1;");
        ptx.AppendLine($"    setp.ge.u32 %p0, %r1, {total};");
        ptx.AppendLine("    @%p0 bra.uni KERNEL_DONE;");
        ptx.AppendLine($"    div.u32 %r2, %r1, {columns};");
        ptx.AppendLine($"    rem.u32 %r3, %r1, {columns};");
    }

    private static void EmitRowColumnOffset(
        StringBuilder ptx, string row, int columns, string column, string target)
    {
        ptx.AppendLine($"    mul.lo.u32 %r10, {row}, {columns};");
        ptx.AppendLine($"    add.u32 %r10, %r10, {column};");
        ptx.AppendLine($"    mul.wide.u32 {target}, %r10, 4;");
    }

    private static void EmitMaskedGradient(
        StringBuilder ptx, DirectPtxLinearActivation activation)
    {
        // %f0=gradOutput, %f1=saved forward value/preactivation, %f2=result.
        switch (activation)
        {
            case DirectPtxLinearActivation.Relu:
                ptx.AppendLine("    setp.gt.f32 %p2, %f1, 0f00000000;");
                ptx.AppendLine("    selp.f32 %f2, %f0, 0f00000000, %p2;");
                break;
            case DirectPtxLinearActivation.Sigmoid:
                ptx.AppendLine("    sub.rn.f32 %f4, 0f3F800000, %f1;");
                ptx.AppendLine("    mul.rn.f32 %f4, %f1, %f4;");
                ptx.AppendLine("    mul.rn.f32 %f2, %f0, %f4;");
                break;
            case DirectPtxLinearActivation.Tanh:
                ptx.AppendLine("    mul.rn.f32 %f4, %f1, %f1;");
                ptx.AppendLine("    sub.rn.f32 %f4, 0f3F800000, %f4;");
                ptx.AppendLine("    mul.rn.f32 %f2, %f0, %f4;");
                break;
            case DirectPtxLinearActivation.GeluTanh:
                ptx.AppendLine("    mul.rn.f32 %f4, %f1, %f1;");
                ptx.AppendLine("    mul.rn.f32 %f5, %f4, %f1;");
                ptx.AppendLine($"    fma.rn.f32 %f5, %f5, {Literal(0.044715f)}, %f1;");
                ptx.AppendLine($"    mul.rn.f32 %f5, %f5, {Literal(0.7978845608f)};");
                ptx.AppendLine("    tanh.approx.f32 %f6, %f5;");
                ptx.AppendLine($"    fma.rn.f32 %f8, %f4, {Literal(0.134145f)}, 0f3F800000;");
                ptx.AppendLine($"    mul.rn.f32 %f8, %f8, {Literal(0.7978845608f)};");
                ptx.AppendLine("    mul.rn.f32 %f9, %f6, %f6;");
                ptx.AppendLine("    sub.rn.f32 %f9, 0f3F800000, %f9;");
                ptx.AppendLine("    mul.rn.f32 %f8, %f8, %f9;");
                ptx.AppendLine("    add.rn.f32 %f9, 0f3F800000, %f6;");
                ptx.AppendLine("    mul.rn.f32 %f9, %f9, 0f3F000000;");
                ptx.AppendLine("    mul.rn.f32 %f8, %f1, %f8;");
                ptx.AppendLine("    fma.rn.f32 %f8, %f8, 0f3F000000, %f9;");
                ptx.AppendLine("    mul.rn.f32 %f2, %f0, %f8;");
                break;
            case DirectPtxLinearActivation.Swish:
                ptx.AppendLine("    mul.rn.f32 %f4, %f1, 0f3F000000;");
                ptx.AppendLine("    tanh.approx.f32 %f4, %f4;");
                ptx.AppendLine("    fma.rn.f32 %f4, %f4, 0f3F000000, 0f3F000000;");
                ptx.AppendLine("    sub.rn.f32 %f5, 0f3F800000, %f4;");
                ptx.AppendLine("    mul.rn.f32 %f5, %f4, %f5;");
                ptx.AppendLine("    fma.rn.f32 %f5, %f1, %f5, %f4;");
                ptx.AppendLine("    mul.rn.f32 %f2, %f0, %f5;");
                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(activation));
        }
    }

    private static string Literal(float value) =>
        "0f" + PtxCompat.SingleToInt32Bits(value).ToString("X8", CultureInfo.InvariantCulture);

    private static void EmitFooter(StringBuilder ptx)
    {
        ptx.AppendLine("KERNEL_DONE:");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
    }

    internal static bool IsSupportedShape(int m, int k, int n) =>
        m is 64 or 128 or 256 or 512 &&
        k is 256 or 512 or 1024 or 2048 or 4096 &&
        n is 256 or 512 or 1024 or 2048 or 4096;

    private static void Validate(
        int m, int k, int n, DirectPtxLinearActivation activation)
    {
        if (!IsSupportedShape(m, k, n))
            throw new ArgumentOutOfRangeException(nameof(m));
        if (activation is not (DirectPtxLinearActivation.Relu or
            DirectPtxLinearActivation.Sigmoid or DirectPtxLinearActivation.Tanh or
            DirectPtxLinearActivation.GeluTanh or DirectPtxLinearActivation.Swish))
            throw new ArgumentOutOfRangeException(nameof(activation));
    }

    private static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture,
        int m,
        int k,
        int n,
        DirectPtxLinearActivation activation)
    {
        var input = new DirectPtxExtent(m, k);
        var output = new DirectPtxExtent(m, n);
        var weights = new DirectPtxExtent(k, n);
        return new DirectPtxKernelBlueprint(
            Operation: "fused-linear-backward",
            Version: 1,
            Architecture: architecture,
            Variant: $"fp32-m{m}-k{k}-n{n}-{activation}".ToLowerInvariant(),
            Tensors:
            [
                new("gradOutput", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    output, output, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("input", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    input, input, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("weights", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.LinearWeightInputMajor,
                    weights, weights, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("saved", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    output, output, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("gradInput", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    input, input, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact),
                new("gradWeight", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.LinearWeightInputMajor,
                    weights, weights, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact),
                new("gradBias", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    new DirectPtxExtent(n), new DirectPtxExtent(n), 16,
                    DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 48,
                MaxStaticSharedBytes: 0,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 4),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["formula"] = "dX,dW,db for activation(X@W+bias)",
                ["activation"] = activation.ToString(),
                ["masked-gradient"] = "recomputed-in-registers-per-output-kernel",
                ["global-intermediates"] = "none",
                ["temporary-device-allocation"] = "none",
                ["shape-parameters"] = "none",
                ["stride-parameters"] = "none"
            });
    }

    private static void Require(
        DirectPtxTensorView view,
        DirectPtxTensorContract contract,
        string parameter)
    {
        if (view.Pointer == IntPtr.Zero || view.PhysicalType != contract.PhysicalType ||
            view.Layout != contract.Layout || view.LogicalExtent != contract.LogicalExtent ||
            view.PhysicalExtent != contract.PhysicalExtent || view.ByteLength != contract.RequiredBytes)
            throw new ArgumentException(
                $"{parameter} does not satisfy physical ABI '{contract.Name}'.", parameter);
    }

    private static bool Overlaps(DirectPtxTensorView left, DirectPtxTensorView right)
    {
        nuint leftStart = PtxCompat.ToNuint(left.Pointer);
        nuint rightStart = PtxCompat.ToNuint(right.Pointer);
        nuint leftEnd = checked(leftStart + left.ByteLength);
        nuint rightEnd = checked(rightStart + right.ByteLength);
        return leftStart < rightEnd && rightStart < leftEnd;
    }
}
