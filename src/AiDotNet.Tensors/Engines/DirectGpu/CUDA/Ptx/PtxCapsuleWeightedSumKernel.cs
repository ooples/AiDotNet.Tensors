using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Dynamic-routing weighted sum (issue #854), matching the NVRTC <c>capsule_weighted_sum</c> kernel:
/// <c>output[b,c,d] = sum_i coupling[b,i,c] * predictions[b,i,c,d]</c>. Where divisibility permits,
/// one thread owns four or eight contiguous dimensions so index work and coupling loads are shared
/// across multiple accumulators. Other supported shapes retain the scalar path. All paths walk the
/// input-capsule axis serially in registers, without shared memory or cross-thread reduction.
///
/// Logical dims are (batchSize B, inputCapsules I, outputCapsules C, capsuleDim D). Coupling is
/// <c>[B,I,C]</c>, predictions <c>[B,I,C,D]</c>, output <c>[B,C,D]</c> row-major (so the flat output
/// ordering is preserved). Shape is baked into the PTX; the launch takes buffer pointers only.
/// The selected output-group count divides the 256-thread launch exactly, with no bounds guard.
/// </summary>
internal sealed class PtxCapsuleWeightedSumKernel : IDisposable
{
    internal const int BlockThreads = 256;
    internal const int MaxOutputs = 2048 * 4096;
    internal const int MaxInputCapsules = 4096;
    internal const string EntryPoint = "aidotnet_capsule_weighted_sum";

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int BatchSize { get; }
    internal int InputCapsules { get; }
    internal int OutputCapsules { get; }
    internal int CapsuleDim { get; }
    internal int OutputsPerThread { get; }
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxCapsuleWeightedSumKernel(
        DirectPtxRuntime runtime, int batchSize, int inputCapsules, int outputCapsules, int capsuleDim)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasValidatedScientific(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The checked-in capsule-weighted-sum specialization is measured only on GA10x/SM86.");
        ValidateShape(batchSize, inputCapsules, outputCapsules, capsuleDim);
        BatchSize = batchSize;
        InputCapsules = inputCapsules;
        OutputCapsules = outputCapsules;
        CapsuleDim = capsuleDim;
        OutputsPerThread = GetOutputsPerThread(batchSize, outputCapsules, capsuleDim);
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, batchSize, inputCapsules, outputCapsules, capsuleDim);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor, batchSize, inputCapsules, outputCapsules, capsuleDim);
        _module = runtime.LoadModule(Ptx);
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo info);
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, BlockThreads);
        Blueprint.ResourceBudget.Validate(EntryPoint, info, BlockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Ptx, info, BlockThreads, activeBlocks, _module);
    }

    internal unsafe void Launch(DirectPtxTensorView coupling, DirectPtxTensorView predictions, DirectPtxTensorView output)
    {
        DirectPtxAbi.Require(coupling, Blueprint.Tensors[0], nameof(coupling));
        DirectPtxAbi.Require(predictions, Blueprint.Tensors[1], nameof(predictions));
        DirectPtxAbi.Require(output, Blueprint.Tensors[2], nameof(output));

        IntPtr couplingPointer = coupling.Pointer;
        IntPtr predictionsPointer = predictions.Pointer;
        IntPtr outputPointer = output.Pointer;
        void** arguments = stackalloc void*[3];
        arguments[0] = &couplingPointer;
        arguments[1] = &predictionsPointer;
        arguments[2] = &outputPointer;
        uint grid = (uint)((BatchSize * OutputCapsules * CapsuleDim) / (BlockThreads * OutputsPerThread));
        _module.Launch(_function, grid, 1, 1, BlockThreads, 1, 1, 0, arguments);
    }

    public void Dispose() => _module.Dispose();

    internal static string EmitPtx(int ccMajor, int ccMinor, int batchSize, int inputCapsules, int outputCapsules, int capsuleDim)
    {
        ValidateShape(batchSize, inputCapsules, outputCapsules, capsuleDim);
        int couplingStrideB = inputCapsules * outputCapsules;             // coupling [B,I,C]
        int predStrideB = inputCapsules * outputCapsules * capsuleDim;    // predictions [B,I,C,D]
        int predStrideI = outputCapsules * capsuleDim;                    // i stride in predictions
        int outputsPerThread = GetOutputsPerThread(batchSize, outputCapsules, capsuleDim);
        // coupling i stride = outputCapsules (C)

        var ptx = new StringBuilder(4_000);
        DirectPtxPtxText.AppendModuleHeader(ptx, ccMajor, ccMinor, disableLoopUnrolling: true);
        ptx.AppendLine($"// capsule-weighted-sum B={batchSize} I={inputCapsules} C={outputCapsules} D={capsuleDim} x{outputsPerThread}");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 coupling_ptr,");
        ptx.AppendLine("    .param .u64 pred_ptr,");
        ptx.AppendLine("    .param .u64 out_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine($".maxntid {BlockThreads}, 1, 1");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<2>;");
        ptx.AppendLine("    .reg .b32 %r<14>;");
        ptx.AppendLine("    .reg .b64 %rd<16>;");
        ptx.AppendLine($"    .reg .f32 %f<{(outputsPerThread > 1 ? outputsPerThread * 2 + 2 : 4)}>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [coupling_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [pred_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [out_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {BlockThreads}, %r0;");   // idx
        int dimensionGroups = capsuleDim / outputsPerThread;
        ptx.AppendLine($"    rem.u32 %r3, %r2, {dimensionGroups};");       // d or d-group
        ptx.AppendLine($"    div.u32 %r4, %r2, {dimensionGroups};");       // t = idx/dimensionGroups
        ptx.AppendLine($"    rem.u32 %r5, %r4, {outputCapsules};");        // c
        ptx.AppendLine($"    div.u32 %r6, %r4, {outputCapsules};");        // b
        // couplingBaseElem = b*couplingStrideB + c
        ptx.AppendLine($"    mad.lo.u32 %r7, %r6, {couplingStrideB}, %r5;");
        ptx.AppendLine("    mul.wide.u32 %rd3, %r7, 4;");
        ptx.AppendLine("    add.u64 %rd6, %rd0, %rd3;");                   // &coupling[b,0,c]
        // predBaseElem = b*predStrideB + c*capsuleDim + d
        ptx.AppendLine($"    mul.lo.u32 %r8, %r6, {predStrideB};");
        ptx.AppendLine($"    mad.lo.u32 %r8, %r5, {capsuleDim}, %r8;");
        if (outputsPerThread > 1)
            ptx.AppendLine($"    mad.lo.u32 %r8, %r3, {outputsPerThread}, %r8;");
        else
            ptx.AppendLine("    add.u32 %r8, %r8, %r3;");
        ptx.AppendLine("    mul.wide.u32 %rd4, %r8, 4;");
        ptx.AppendLine("    add.u64 %rd7, %rd1, %rd4;");                   // &predictions[b,0,c,d]
        ptx.AppendLine("    mov.f32 %f0, 0f00000000;");                   // sum
        for (int outputIndex = 1; outputIndex < outputsPerThread; outputIndex++)
            ptx.AppendLine($"    mov.f32 %f{outputIndex}, 0f00000000;");
        ptx.AppendLine("    mov.u32 %r9, 0;");                            // i = 0
        ptx.AppendLine("$CWS_I_LOOP:");
        if (outputsPerThread > 1)
        {
            int couplingRegister = outputsPerThread;
            int predictionRegister = couplingRegister + 1;
            ptx.AppendLine($"    ld.global.nc.f32 %f{couplingRegister}, [%rd6];");
            for (int outputIndex = 0; outputIndex < outputsPerThread; outputIndex += 4)
            {
                int value = predictionRegister + outputIndex;
                int byteOffset = outputIndex * sizeof(float);
                ptx.AppendLine($"    ld.global.nc.v4.f32 {{%f{value}, %f{value + 1}, %f{value + 2}, %f{value + 3}}}, [%rd7+{byteOffset}];");
            }
            for (int outputIndex = 0; outputIndex < outputsPerThread; outputIndex++)
                ptx.AppendLine($"    fma.rn.f32 %f{outputIndex}, %f{couplingRegister}, %f{predictionRegister + outputIndex}, %f{outputIndex};");
        }
        else
        {
            ptx.AppendLine("    ld.global.nc.f32 %f1, [%rd6];");
            ptx.AppendLine("    ld.global.nc.f32 %f2, [%rd7];");
            ptx.AppendLine("    fma.rn.f32 %f0, %f1, %f2, %f0;");
        }
        ptx.AppendLine($"    add.u64 %rd6, %rd6, {outputCapsules * 4};"); // coupling i stride = C elems
        ptx.AppendLine($"    add.u64 %rd7, %rd7, {predStrideI * 4};");    // pred i stride = C*D elems
        ptx.AppendLine("    add.u32 %r9, %r9, 1;");
        ptx.AppendLine($"    setp.lt.u32 %p0, %r9, {inputCapsules};");
        ptx.AppendLine("    @%p0 bra $CWS_I_LOOP;");
        ptx.AppendLine($"    mul.wide.u32 %rd8, %r2, {outputsPerThread * 4};");
        ptx.AppendLine("    add.u64 %rd9, %rd2, %rd8;");
        if (outputsPerThread > 1)
        {
            for (int outputIndex = 0; outputIndex < outputsPerThread; outputIndex += 4)
            {
                int byteOffset = outputIndex * sizeof(float);
                ptx.AppendLine($"    st.global.v4.f32 [%rd9+{byteOffset}], {{%f{outputIndex}, %f{outputIndex + 1}, %f{outputIndex + 2}, %f{outputIndex + 3}}};");
            }
        }
        else
            ptx.AppendLine("    st.global.f32 [%rd9], %f0;");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture, int batchSize, int inputCapsules, int outputCapsules, int capsuleDim)
    {
        var couplingExtent = new DirectPtxExtent(batchSize * inputCapsules * outputCapsules);
        var predExtent = new DirectPtxExtent(batchSize * inputCapsules * outputCapsules * capsuleDim);
        var outExtent = new DirectPtxExtent(batchSize * outputCapsules * capsuleDim);
        return new DirectPtxKernelBlueprint(
            Operation: "capsule-weighted-sum",
            Version: 2,
            Architecture: architecture,
            Variant: $"fp32-x{GetOutputsPerThread(batchSize, outputCapsules, capsuleDim)}-b{batchSize}-i{inputCapsules}-c{outputCapsules}-d{capsuleDim}",
            Tensors:
            [
                new("coupling", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    couplingExtent, couplingExtent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("predictions", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    predExtent, predExtent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    outExtent, outExtent, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: DirectPtxResourceBudget.FromDriverMeasurement(
                measuredRegistersPerThread: GetOutputsPerThread(batchSize, outputCapsules, capsuleDim) switch
                {
                    8 => 28,
                    4 => 18,
                    _ => 26
                },
                maxStaticSharedBytes: 0,
                maxLocalBytesPerThread: 0,
                minBlocksPerMultiprocessor: 1),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["formula"] = "output[b,c,d] = sum_i coupling[b,i,c] * predictions[b,i,c,d]",
                ["global-intermediates"] = "none",
                ["temporary-device-allocation"] = "none",
                ["stride-parameters"] = "none",
                ["outputs-per-thread"] = GetOutputsPerThread(batchSize, outputCapsules, capsuleDim)
                    .ToString(System.Globalization.CultureInfo.InvariantCulture)
            });
    }

    private static int GetOutputsPerThread(int batchSize, int outputCapsules, int capsuleDim)
    {
        long outputs = (long)batchSize * outputCapsules * capsuleDim;
        if (capsuleDim % 8 == 0 && outputs % (BlockThreads * 8L) == 0) return 8;
        return capsuleDim % 4 == 0 && outputs % (BlockThreads * 4L) == 0 ? 4 : 1;
    }

    internal static bool IsSupportedShape(int batchSize, int inputCapsules, int outputCapsules, int capsuleDim)
    {
        if (batchSize <= 0 || inputCapsules <= 0 || outputCapsules <= 0 || capsuleDim <= 0) return false;
        if (inputCapsules > MaxInputCapsules) return false;
        long outputs = (long)batchSize * outputCapsules * capsuleDim;
        return outputs > 0 && outputs % BlockThreads == 0 && outputs <= MaxOutputs;
    }

    internal static bool IsPromotedShape(int batchSize, int inputCapsules, int outputCapsules, int capsuleDim) => false;

    private static void ValidateShape(int batchSize, int inputCapsules, int outputCapsules, int capsuleDim)
    {
        if (!IsSupportedShape(batchSize, inputCapsules, outputCapsules, capsuleDim))
            throw new ArgumentOutOfRangeException(
                nameof(batchSize),
                $"Capsule weighted sum requires positive dims with inputCapsules<={MaxInputCapsules} and (B*C*D) a multiple of {BlockThreads} up to {MaxOutputs}.");
    }

}
