using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Exact FP32 LoRA forward specialization. One block owns one row, stages
/// input@A in shared memory, reuses it for every output column, fuses scaling
/// and the base residual in registers, and commits the final output once.
/// </summary>
internal sealed class PtxFusedLoRAKernel : IDisposable
{
    internal const int BlockThreads = 256;
    internal const string EntryPoint = "aidotnet_fused_lora_forward";

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int Batch { get; }
    internal int InputFeatures { get; }
    internal int Rank { get; }
    internal int OutputFeatures { get; }
    internal float Scaling { get; }
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxFusedLoRAKernel(
        DirectPtxRuntime runtime,
        int batch,
        int inputFeatures,
        int rank,
        int outputFeatures,
        float scaling)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasValidatedFusedLinear(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The fused-LoRA PTX specialization is measured only on GA10x/SM86.");
        Validate(batch, inputFeatures, rank, outputFeatures, scaling);

        Batch = batch;
        InputFeatures = inputFeatures;
        Rank = rank;
        OutputFeatures = outputFeatures;
        Scaling = scaling;
        Blueprint = CreateBlueprint(
            runtime.ArchitectureFamily, batch, inputFeatures, rank, outputFeatures, scaling);
        Ptx = EmitPtx(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor,
            batch, inputFeatures, rank, outputFeatures, scaling);
        _module = runtime.LoadModule(Ptx);
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo info);
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, BlockThreads);
        Blueprint.ResourceBudget.Validate(EntryPoint, info, BlockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Ptx, info,
            BlockThreads, activeBlocks, _module);
    }

    internal unsafe void Launch(
        DirectPtxTensorView input,
        DirectPtxTensorView baseOutput,
        DirectPtxTensorView loraA,
        DirectPtxTensorView loraB,
        DirectPtxTensorView output)
    {
        Require(input, Blueprint.Tensors[0], nameof(input));
        Require(baseOutput, Blueprint.Tensors[1], nameof(baseOutput));
        Require(loraA, Blueprint.Tensors[2], nameof(loraA));
        Require(loraB, Blueprint.Tensors[3], nameof(loraB));
        Require(output, Blueprint.Tensors[4], nameof(output));
        if (Overlaps(output, input) || Overlaps(output, loraA) ||
            Overlaps(output, loraB) ||
            (Overlaps(output, baseOutput) && output.Pointer != baseOutput.Pointer))
            throw new ArgumentException(
                "LoRA output may only alias baseOutput at the exact same pointer.");

        IntPtr inputPointer = input.Pointer;
        IntPtr basePointer = baseOutput.Pointer;
        IntPtr aPointer = loraA.Pointer;
        IntPtr bPointer = loraB.Pointer;
        IntPtr outputPointer = output.Pointer;
        void** arguments = stackalloc void*[5];
        arguments[0] = &inputPointer;
        arguments[1] = &basePointer;
        arguments[2] = &aPointer;
        arguments[3] = &bPointer;
        arguments[4] = &outputPointer;
        _module.Launch(
            _function, (uint)Batch, 1, 1, BlockThreads, 1, 1, 0, arguments);
    }

    public void Dispose() => _module.Dispose();

    internal static string EmitPtx(
        int ccMajor,
        int ccMinor,
        int batch,
        int inputFeatures,
        int rank,
        int outputFeatures,
        float scaling)
    {
        Validate(batch, inputFeatures, rank, outputFeatures, scaling);
        int inputRowBytes = checked(inputFeatures * sizeof(float));
        int outputRowBytes = checked(outputFeatures * sizeof(float));
        int aRowBytes = checked(rank * sizeof(float));
        int bRowBytes = outputRowBytes;
        string scalingLiteral = "0f" +
            PtxCompat.SingleToInt32Bits(scaling).ToString("X8", CultureInfo.InvariantCulture);

        var ptx = new StringBuilder(12_000);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine($"// fused LoRA B={batch} I={inputFeatures} R={rank} O={outputFeatures} scale={scalingLiteral}");
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 input_ptr,");
        ptx.AppendLine("    .param .u64 base_ptr,");
        ptx.AppendLine("    .param .u64 lora_a_ptr,");
        ptx.AppendLine("    .param .u64 lora_b_ptr,");
        ptx.AppendLine("    .param .u64 output_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine($".maxntid {BlockThreads}, 1, 1");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<5>;");
        ptx.AppendLine("    .reg .b32 %r<16>;");
        ptx.AppendLine("    .reg .b64 %rd<24>;");
        ptx.AppendLine("    .reg .f32 %f<8>;");
        ptx.AppendLine($"    .shared .align 16 .b8 projection[{rank * sizeof(float)}];");
        ptx.AppendLine("    .shared .align 16 .b8 projection_partials[256];");
        ptx.AppendLine("    ld.param.u64 %rd0, [input_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [base_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [lora_a_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd3, [lora_b_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd4, [output_ptr];");
        ptx.AppendLine("    mov.u64 %rd5, projection;");
        ptx.AppendLine("    mov.u64 %rd14, projection_partials;");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mul.wide.u32 %rd6, %r1, {inputRowBytes};");
        ptx.AppendLine("    add.u64 %rd7, %rd0, %rd6;");

        // Lanes are arranged as four K contributors for each eight-rank group.
        // All eight warps partition K in steps of 32, so loraA[K,R] is read in
        // contiguous eight-float segments instead of one strided rank column.
        // A 64-float shared fold joins the eight warp partials; lanes 0..7 of
        // warp zero publish the final rank group.
        ptx.AppendLine("    and.b32 %r5, %r0, 31;");
        ptx.AppendLine("    shr.u32 %r6, %r0, 5;");
        ptx.AppendLine("    and.b32 %r7, %r5, 7;");
        ptx.AppendLine("    shr.u32 %r8, %r5, 3;");
        ptx.AppendLine("    mov.u32 %r9, 0;");
        ptx.AppendLine("LORA_PROJECTION_RANK_LOOP:");
        ptx.AppendLine($"    setp.ge.u32 %p4, %r9, {rank};");
        ptx.AppendLine("    @%p4 bra.uni LORA_PROJECTION_DONE;");
        ptx.AppendLine("    add.u32 %r10, %r9, %r7;");
        ptx.AppendLine($"    setp.ge.u32 %p0, %r10, {rank};");
        ptx.AppendLine("    mov.f32 %f0, 0f00000000;");
        ptx.AppendLine("    @%p0 bra LORA_REDUCE_PROJECTION;");
        ptx.AppendLine("    mad.lo.u32 %r2, %r6, 4, %r8;");
        ptx.AppendLine("LORA_INPUT_LOOP:");
        ptx.AppendLine($"    setp.ge.u32 %p1, %r2, {inputFeatures};");
        ptx.AppendLine("    @%p1 bra.uni LORA_REDUCE_PROJECTION;");
        ptx.AppendLine("    mul.wide.u32 %rd8, %r2, 4;");
        ptx.AppendLine("    add.u64 %rd9, %rd7, %rd8;");
        ptx.AppendLine("    ld.global.nc.f32 %f1, [%rd9];");
        ptx.AppendLine($"    mul.wide.u32 %rd10, %r2, {aRowBytes};");
        ptx.AppendLine("    mul.wide.u32 %rd11, %r10, 4;");
        ptx.AppendLine("    add.u64 %rd12, %rd2, %rd10;");
        ptx.AppendLine("    add.u64 %rd12, %rd12, %rd11;");
        ptx.AppendLine("    ld.global.nc.f32 %f2, [%rd12];");
        ptx.AppendLine("    fma.rn.f32 %f0, %f1, %f2, %f0;");
        ptx.AppendLine("    add.u32 %r2, %r2, 32;");
        ptx.AppendLine("    bra.uni LORA_INPUT_LOOP;");
        ptx.AppendLine("LORA_REDUCE_PROJECTION:");
        foreach (int offset in new[] { 8, 16 })
        {
            ptx.AppendLine("    mov.b32 %r12, %f0;");
            ptx.AppendLine(
                $"    shfl.sync.down.b32 %r13, %r12, {offset}, 31, 0xffffffff;");
            ptx.AppendLine("    mov.b32 %f1, %r13;");
            ptx.AppendLine("    add.rn.f32 %f0, %f0, %f1;");
        }
        ptx.AppendLine("    setp.ne.u32 %p2, %r8, 0;");
        ptx.AppendLine("    @%p2 bra LORA_PARTIAL_STORED;");
        ptx.AppendLine("    mad.lo.u32 %r11, %r7, 8, %r6;");
        ptx.AppendLine("    mul.wide.u32 %rd8, %r11, 4;");
        ptx.AppendLine("    add.u64 %rd9, %rd14, %rd8;");
        ptx.AppendLine("    st.shared.f32 [%rd9], %f0;");
        ptx.AppendLine("LORA_PARTIAL_STORED:");
        ptx.AppendLine("    bar.sync 0;");
        ptx.AppendLine("    setp.ne.u32 %p2, %r6, 0;");
        ptx.AppendLine("    @%p2 bra.uni LORA_PROJECTION_GROUP_DONE;");
        ptx.AppendLine("    mad.lo.u32 %r11, %r7, 8, %r8;");
        ptx.AppendLine("    mul.wide.u32 %rd8, %r11, 4;");
        ptx.AppendLine("    add.u64 %rd9, %rd14, %rd8;");
        ptx.AppendLine("    ld.shared.f32 %f0, [%rd9];");
        ptx.AppendLine("    ld.shared.f32 %f1, [%rd9+16];");
        ptx.AppendLine("    add.rn.f32 %f0, %f0, %f1;");
        foreach (int offset in new[] { 8, 16 })
        {
            ptx.AppendLine("    mov.b32 %r12, %f0;");
            ptx.AppendLine(
                $"    shfl.sync.down.b32 %r13, %r12, {offset}, 31, 0xffffffff;");
            ptx.AppendLine("    mov.b32 %f1, %r13;");
            ptx.AppendLine("    add.rn.f32 %f0, %f0, %f1;");
        }
        ptx.AppendLine("    setp.ne.u32 %p2, %r8, 0;");
        ptx.AppendLine("    @%p2 bra LORA_PROJECTION_GROUP_DONE;");
        ptx.AppendLine("    @%p0 bra LORA_PROJECTION_GROUP_DONE;");
        ptx.AppendLine("    mul.wide.u32 %rd11, %r10, 4;");
        ptx.AppendLine("    add.u64 %rd13, %rd5, %rd11;");
        ptx.AppendLine("    st.shared.f32 [%rd13], %f0;");
        ptx.AppendLine("LORA_PROJECTION_GROUP_DONE:");
        ptx.AppendLine("    bar.sync 0;");
        ptx.AppendLine("    add.u32 %r9, %r9, 8;");
        ptx.AppendLine("    bra.uni LORA_PROJECTION_RANK_LOOP;");
        ptx.AppendLine("LORA_PROJECTION_DONE:");
        ptx.AppendLine("    bar.sync 0;");

        // Each thread emits output columns j=tid,tid+256,...
        ptx.AppendLine("    mov.u32 %r3, %r0;");
        ptx.AppendLine($"    mul.wide.u32 %rd14, %r1, {outputRowBytes};");
        ptx.AppendLine("    add.u64 %rd15, %rd1, %rd14;");
        ptx.AppendLine("    add.u64 %rd16, %rd4, %rd14;");
        ptx.AppendLine("LORA_OUTPUT_LOOP:");
        ptx.AppendLine($"    setp.ge.u32 %p2, %r3, {outputFeatures};");
        ptx.AppendLine("    @%p2 bra.uni LORA_DONE;");
        ptx.AppendLine("    mov.f32 %f3, 0f00000000;");
        ptx.AppendLine("    mov.u32 %r4, 0;");
        ptx.AppendLine("LORA_RANK_LOOP:");
        ptx.AppendLine($"    setp.ge.u32 %p3, %r4, {rank};");
        ptx.AppendLine("    @%p3 bra.uni LORA_EPILOGUE;");
        ptx.AppendLine("    mul.wide.u32 %rd17, %r4, 4;");
        ptx.AppendLine("    add.u64 %rd18, %rd5, %rd17;");
        ptx.AppendLine("    ld.shared.f32 %f4, [%rd18];");
        ptx.AppendLine($"    mul.wide.u32 %rd19, %r4, {bRowBytes};");
        ptx.AppendLine("    mul.wide.u32 %rd20, %r3, 4;");
        ptx.AppendLine("    add.u64 %rd21, %rd3, %rd19;");
        ptx.AppendLine("    add.u64 %rd21, %rd21, %rd20;");
        ptx.AppendLine("    ld.global.nc.f32 %f5, [%rd21];");
        ptx.AppendLine("    fma.rn.f32 %f3, %f4, %f5, %f3;");
        ptx.AppendLine("    add.u32 %r4, %r4, 1;");
        ptx.AppendLine("    bra.uni LORA_RANK_LOOP;");
        ptx.AppendLine("LORA_EPILOGUE:");
        ptx.AppendLine("    add.u64 %rd22, %rd15, %rd20;");
        ptx.AppendLine("    ld.global.nc.f32 %f6, [%rd22];");
        ptx.AppendLine($"    fma.rn.f32 %f7, %f3, {scalingLiteral}, %f6;");
        ptx.AppendLine("    add.u64 %rd23, %rd16, %rd20;");
        ptx.AppendLine("    st.global.f32 [%rd23], %f7;");
        ptx.AppendLine($"    add.u32 %r3, %r3, {BlockThreads};");
        ptx.AppendLine("    bra.uni LORA_OUTPUT_LOOP;");
        ptx.AppendLine("LORA_DONE:");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    internal static bool IsSupportedShape(int batch, int inputFeatures, int rank, int outputFeatures) =>
        batch is 1 or 8 or 16 or 32 or 64 or 128 &&
        inputFeatures is 256 or 512 or 1024 or 2048 or 4096 &&
        rank is 4 or 8 or 16 or 32 or 64 &&
        outputFeatures is 256 or 512 or 1024 or 2048 or 4096;

    private static void Validate(
        int batch,
        int inputFeatures,
        int rank,
        int outputFeatures,
        float scaling)
    {
        if (!IsSupportedShape(batch, inputFeatures, rank, outputFeatures))
            throw new ArgumentOutOfRangeException(
                nameof(batch), "Unsupported exact LoRA specialization shape.");
        if (float.IsNaN(scaling) || float.IsInfinity(scaling))
            throw new ArgumentOutOfRangeException(nameof(scaling));
        _ = checked(batch * inputFeatures);
        _ = checked(inputFeatures * rank);
        _ = checked(rank * outputFeatures);
        _ = checked(batch * outputFeatures);
    }

    private static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture,
        int batch,
        int inputFeatures,
        int rank,
        int outputFeatures,
        float scaling)
    {
        var input = new DirectPtxExtent(batch, inputFeatures);
        var output = new DirectPtxExtent(batch, outputFeatures);
        var a = new DirectPtxExtent(inputFeatures, rank);
        var b = new DirectPtxExtent(rank, outputFeatures);
        return new DirectPtxKernelBlueprint(
            Operation: "fused-lora-forward",
            Version: 2,
            Architecture: architecture,
            Variant: $"fp32-b{batch}-i{inputFeatures}-r{rank}-o{outputFeatures}-s{PtxCompat.SingleToInt32Bits(scaling):x8}",
            Tensors:
            [
                new("input", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    input, input, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("baseOutput", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    output, output, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("loraA", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    a, a, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("loraB", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    b, b, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    output, output, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 40,
                MaxStaticSharedBytes: rank * sizeof(float) + 256,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 2),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["formula"] = "baseOutput + scaling * ((input @ loraA) @ loraB)",
                ["scaling-bits"] = PtxCompat.SingleToInt32Bits(scaling).ToString("x8", CultureInfo.InvariantCulture),
                ["intermediate"] = "warp-reduced-shared-projection-per-row",
                ["projection"] = "eight-warp-coalesced-kxr-loads-two-stage-shuffle-reduction",
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
