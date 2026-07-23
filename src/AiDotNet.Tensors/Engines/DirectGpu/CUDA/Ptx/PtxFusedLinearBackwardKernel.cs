using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Exact-shape fused-linear backward family. One launch assigns disjoint CTA
/// ranges to dInput and dWeight. The k=0 dWeight lanes accumulate dBias from
/// the same grad-output/saved-value reads, so no masked-gradient tensor or
/// separate bias-reduction launch is materialized.
/// </summary>
internal sealed class PtxFusedLinearBackwardKernel : IDisposable
{
    internal const int BlockThreads = 256;
    internal const int Tile = 16;
    internal const string EntryPoint = "aidotnet_fused_linear_backward";

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

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
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo info);
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(
            _function, BlockThreads);
        Blueprint.ResourceBudget.Validate(
            EntryPoint, info, BlockThreads, activeBlocks);
        DirectPtxKernelAudit audit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Ptx, info,
            BlockThreads, activeBlocks, _module);
        Audits = [audit];
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

        void** arguments = stackalloc void*[7];
        arguments[0] = &go;
        arguments[1] = &x;
        arguments[2] = &w;
        arguments[3] = &s;
        arguments[4] = &gi;
        arguments[5] = &gw;
        arguments[6] = &gb;
        uint grid = checked(Grid(checked(M * K)) + Grid(checked(K * N)));
        _module.Launch(
            _function, grid, 1, 1,
            BlockThreads, 1, 1, 0, arguments);
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
        var ptx = new StringBuilder(12_000);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine($"// fused-linear backward M={m} K={k} N={n} act={activation}; single exact pointer-only ABI");
        EmitCombined(ptx, m, k, n, activation);
        return ptx.ToString();
    }

    private static void EmitCombined(
        StringBuilder ptx, int m, int k, int n, DirectPtxLinearActivation activation)
    {
        int mTiles = m / Tile;
        int kTiles = k / Tile;
        int nTiles = n / Tile;
        int gradInputBlocks = checked(mTiles * kTiles);
        EmitHeader(
            ptx, EntryPoint,
            "grad_output_ptr", "input_ptr", "weights_ptr", "saved_ptr",
            "grad_input_ptr", "grad_weight_ptr", "grad_bias_ptr");
        EmitRegisters(ptx);
        ptx.AppendLine("    .shared .align 16 .b8 tile_a[1024];");
        ptx.AppendLine("    .shared .align 16 .b8 tile_b[1024];");
        EmitPointers(
            ptx, "grad_output_ptr", "input_ptr", "weights_ptr", "saved_ptr",
            "grad_input_ptr", "grad_weight_ptr", "grad_bias_ptr");
        ptx.AppendLine("    mov.u64 %rd7, tile_a;");
        ptx.AppendLine("    mov.u64 %rd8, tile_b;");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    setp.lt.u32 %p0, %r1, {gradInputBlocks};");
        ptx.AppendLine("    @%p0 bra.uni GRAD_INPUT_PATH;");

        // dWeight = X^T @ maskedGrad. Threads load contiguous X[M,K] and
        // maskedGrad[M,N] panels, then reuse both panels for 16 FMAs. The
        // first K tile also folds each N column into dBias.
        ptx.AppendLine($"    sub.u32 %r1, %r1, {gradInputBlocks};");
        ptx.AppendLine($"    div.u32 %r2, %r1, {nTiles};");
        ptx.AppendLine($"    rem.u32 %r3, %r1, {nTiles};");
        EmitLocalCoordinates(ptx);
        ptx.AppendLine($"    mad.lo.u32 %r6, %r2, {Tile}, %r4;");
        ptx.AppendLine($"    mad.lo.u32 %r7, %r3, {Tile}, %r5;");
        ptx.AppendLine("    setp.eq.u32 %p3, %r2, 0;");
        ptx.AppendLine("    setp.eq.u32 %p4, %r4, 0;");
        ptx.AppendLine("    and.pred %p5, %p3, %p4;");
        ptx.AppendLine("    mov.f32 %f10, 0f00000000;");
        ptx.AppendLine("    mov.f32 %f11, 0f00000000;");
        ptx.AppendLine("    mov.u32 %r8, 0;");
        ptx.AppendLine("GW_TILE_LOOP:");
        ptx.AppendLine($"    setp.ge.u32 %p1, %r8, {m};");
        ptx.AppendLine("    @%p1 bra.uni GW_STORE;");
        ptx.AppendLine("    add.u32 %r9, %r8, %r4;");
        ptx.AppendLine($"    mad.lo.u32 %r10, %r2, {Tile}, %r5;");
        ptx.AppendLine($"    mad.lo.u32 %r12, %r9, {k}, %r10;");
        ptx.AppendLine("    mul.wide.u32 %rd9, %r12, 4;");
        ptx.AppendLine("    add.u64 %rd10, %rd1, %rd9;");
        ptx.AppendLine("    ld.global.nc.f32 %f3, [%rd10];");
        ptx.AppendLine($"    mad.lo.u32 %r10, %r3, {Tile}, %r5;");
        ptx.AppendLine($"    mad.lo.u32 %r12, %r9, {n}, %r10;");
        ptx.AppendLine("    mul.wide.u32 %rd9, %r12, 4;");
        ptx.AppendLine("    add.u64 %rd10, %rd0, %rd9;");
        ptx.AppendLine("    add.u64 %rd11, %rd3, %rd9;");
        ptx.AppendLine("    ld.global.nc.f32 %f0, [%rd10];");
        ptx.AppendLine("    ld.global.nc.f32 %f1, [%rd11];");
        EmitMaskedGradient(ptx, activation);
        EmitStorePanels(ptx, "%f3", "%f2");
        ptx.AppendLine("    bar.sync 0;");
        ptx.AppendLine("    mul.wide.u32 %rd12, %r4, 4;");
        ptx.AppendLine("    add.u64 %rd12, %rd7, %rd12;");
        ptx.AppendLine("    mul.wide.u32 %rd13, %r5, 4;");
        ptx.AppendLine("    add.u64 %rd13, %rd8, %rd13;");
        for (int inner = 0; inner < Tile; inner++)
        {
            string offset = inner == 0 ? string.Empty : $"+{inner * Tile * sizeof(float)}";
            ptx.AppendLine($"    ld.shared.f32 %f0, [%rd12{offset}];");
            ptx.AppendLine($"    ld.shared.f32 %f1, [%rd13{offset}];");
            ptx.AppendLine("    fma.rn.f32 %f10, %f0, %f1, %f10;");
            ptx.AppendLine("    @%p5 add.rn.f32 %f11, %f11, %f1;");
        }
        ptx.AppendLine("    bar.sync 0;");
        ptx.AppendLine($"    add.u32 %r8, %r8, {Tile};");
        ptx.AppendLine("    bra.uni GW_TILE_LOOP;");
        ptx.AppendLine("GW_STORE:");
        ptx.AppendLine($"    mad.lo.u32 %r12, %r6, {n}, %r7;");
        ptx.AppendLine("    mul.wide.u32 %rd9, %r12, 4;");
        ptx.AppendLine("    add.u64 %rd10, %rd5, %rd9;");
        ptx.AppendLine("    st.global.f32 [%rd10], %f10;");
        ptx.AppendLine("    @!%p5 bra KERNEL_DONE;");
        ptx.AppendLine("    mul.wide.u32 %rd9, %r7, 4;");
        ptx.AppendLine("    add.u64 %rd10, %rd6, %rd9;");
        ptx.AppendLine("    st.global.f32 [%rd10], %f11;");
        ptx.AppendLine("    bra.uni KERNEL_DONE;");

        // dInput = maskedGrad @ W. The weight loader transposes a coalesced
        // W[K,N] panel into shared memory so each output thread reads one
        // contiguous shared column during the inner product.
        ptx.AppendLine("GRAD_INPUT_PATH:");
        ptx.AppendLine($"    div.u32 %r2, %r1, {kTiles};");
        ptx.AppendLine($"    rem.u32 %r3, %r1, {kTiles};");
        EmitLocalCoordinates(ptx);
        ptx.AppendLine($"    mad.lo.u32 %r6, %r2, {Tile}, %r4;");
        ptx.AppendLine($"    mad.lo.u32 %r7, %r3, {Tile}, %r5;");
        ptx.AppendLine("    mov.f32 %f10, 0f00000000;");
        ptx.AppendLine("    mov.u32 %r8, 0;");
        ptx.AppendLine("GI_TILE_LOOP:");
        ptx.AppendLine($"    setp.ge.u32 %p1, %r8, {n};");
        ptx.AppendLine("    @%p1 bra.uni GI_STORE;");
        ptx.AppendLine("    add.u32 %r9, %r8, %r5;");
        ptx.AppendLine($"    mad.lo.u32 %r12, %r6, {n}, %r9;");
        ptx.AppendLine("    mul.wide.u32 %rd9, %r12, 4;");
        ptx.AppendLine("    add.u64 %rd10, %rd0, %rd9;");
        ptx.AppendLine("    add.u64 %rd11, %rd3, %rd9;");
        ptx.AppendLine("    ld.global.nc.f32 %f0, [%rd10];");
        ptx.AppendLine("    ld.global.nc.f32 %f1, [%rd11];");
        EmitMaskedGradient(ptx, activation);
        ptx.AppendLine($"    mad.lo.u32 %r10, %r3, {Tile}, %r4;");
        ptx.AppendLine($"    mad.lo.u32 %r12, %r10, {n}, %r9;");
        ptx.AppendLine("    mul.wide.u32 %rd9, %r12, 4;");
        ptx.AppendLine("    add.u64 %rd10, %rd2, %rd9;");
        ptx.AppendLine("    ld.global.nc.f32 %f3, [%rd10];");
        ptx.AppendLine("    mul.wide.u32 %rd9, %r0, 4;");
        ptx.AppendLine("    add.u64 %rd10, %rd7, %rd9;");
        ptx.AppendLine("    st.shared.f32 [%rd10], %f2;");
        ptx.AppendLine($"    mad.lo.u32 %r12, %r5, {Tile}, %r4;");
        ptx.AppendLine("    mul.wide.u32 %rd9, %r12, 4;");
        ptx.AppendLine("    add.u64 %rd10, %rd8, %rd9;");
        ptx.AppendLine("    st.shared.f32 [%rd10], %f3;");
        ptx.AppendLine("    bar.sync 0;");
        ptx.AppendLine($"    mul.wide.u32 %rd12, %r4, {Tile * sizeof(float)};");
        ptx.AppendLine("    add.u64 %rd12, %rd7, %rd12;");
        ptx.AppendLine("    mul.wide.u32 %rd13, %r5, 4;");
        ptx.AppendLine("    add.u64 %rd13, %rd8, %rd13;");
        for (int inner = 0; inner < Tile; inner++)
        {
            string aOffset = inner == 0 ? string.Empty : $"+{inner * sizeof(float)}";
            string bOffset = inner == 0 ? string.Empty : $"+{inner * Tile * sizeof(float)}";
            ptx.AppendLine($"    ld.shared.f32 %f0, [%rd12{aOffset}];");
            ptx.AppendLine($"    ld.shared.f32 %f1, [%rd13{bOffset}];");
            ptx.AppendLine("    fma.rn.f32 %f10, %f0, %f1, %f10;");
        }
        ptx.AppendLine("    bar.sync 0;");
        ptx.AppendLine($"    add.u32 %r8, %r8, {Tile};");
        ptx.AppendLine("    bra.uni GI_TILE_LOOP;");
        ptx.AppendLine("GI_STORE:");
        ptx.AppendLine($"    mad.lo.u32 %r12, %r6, {k}, %r7;");
        ptx.AppendLine("    mul.wide.u32 %rd9, %r12, 4;");
        ptx.AppendLine("    add.u64 %rd10, %rd4, %rd9;");
        ptx.AppendLine("    st.global.f32 [%rd10], %f10;");
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
        ptx.AppendLine("    .reg .pred %p<8>;");
        ptx.AppendLine("    .reg .b32 %r<16>;");
        ptx.AppendLine("    .reg .b64 %rd<20>;");
        ptx.AppendLine("    .reg .f32 %f<16>;");
    }

    private static void EmitPointers(StringBuilder ptx, params string[] names)
    {
        for (int i = 0; i < names.Length; i++)
            ptx.AppendLine($"    ld.param.u64 %rd{i}, [{names[i]}];");
    }

    private static void EmitLocalCoordinates(StringBuilder ptx)
    {
        ptx.AppendLine($"    div.u32 %r4, %r0, {Tile};");
        ptx.AppendLine($"    rem.u32 %r5, %r0, {Tile};");
    }

    private static void EmitStorePanels(
        StringBuilder ptx, string tileAValue, string tileBValue)
    {
        ptx.AppendLine("    mul.wide.u32 %rd9, %r0, 4;");
        ptx.AppendLine("    add.u64 %rd10, %rd7, %rd9;");
        ptx.AppendLine($"    st.shared.f32 [%rd10], {tileAValue};");
        ptx.AppendLine("    add.u64 %rd11, %rd8, %rd9;");
        ptx.AppendLine($"    st.shared.f32 [%rd11], {tileBValue};");
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
            Version: 2,
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
                MaxStaticSharedBytes: 2 * Tile * Tile * sizeof(float),
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 4),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["formula"] = "dX,dW,db for activation(X@W+bias)",
                ["activation"] = activation.ToString(),
                ["launch-topology"] = "single-grid-disjoint-dinput-dweight-cta-ranges",
                ["tiling"] = "16x16-shared-panels-coalesced-global-loads",
                ["masked-gradient"] = "recomputed-in-registers-once-per-shared-panel-load",
                ["bias-gradient"] = "accumulated-by-k0-dweight-tile-from-resident-masked-gradient-panel",
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
