using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Dynamic-routing agreement update (issue #854), matching the NVRTC <c>capsule_agreement</c> kernel:
/// <c>agreement[b,i,c] = sum_d predictions[b,i,c,d] * output[b,c,d]</c>. One thread owns one agreement
/// element and walks the capsule-dim axis <c>d = 0..capsuleDim-1</c> serially in registers (both
/// operands are contiguous along <c>d</c>) — no shared memory, no reduction.
///
/// Logical dims are (batchSize B, inputCapsules I, outputCapsules C, capsuleDim D). Predictions is
/// <c>[B,I,C,D]</c>, output <c>[B,C,D]</c>, agreement <c>[B,I,C]</c> row-major (so the flat agreement
/// index is the thread index). Shape is baked into the PTX; the launch takes buffer pointers only.
/// 256 threads/block, grid = (B*I*C)/256, required to divide evenly (no divergent bounds guard).
/// </summary>
internal sealed class PtxCapsuleAgreementKernel : IDisposable
{
    internal const int BlockThreads = 256;
    internal const int MaxOutputs = 2048 * 4096;
    internal const int MaxCapsuleDim = 4096;
    internal const string EntryPoint = "aidotnet_capsule_agreement";

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int BatchSize { get; }
    internal int InputCapsules { get; }
    internal int OutputCapsules { get; }
    internal int CapsuleDim { get; }
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxCapsuleAgreementKernel(
        DirectPtxRuntime runtime, int batchSize, int inputCapsules, int outputCapsules, int capsuleDim)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasValidatedScientific(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The checked-in capsule-agreement specialization is measured only on GA10x/SM86.");
        ValidateShape(batchSize, inputCapsules, outputCapsules, capsuleDim);
        BatchSize = batchSize;
        InputCapsules = inputCapsules;
        OutputCapsules = outputCapsules;
        CapsuleDim = capsuleDim;
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, batchSize, inputCapsules, outputCapsules, capsuleDim);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor, batchSize, inputCapsules, outputCapsules, capsuleDim);
        _module = runtime.LoadModule(Ptx);
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo info);
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, BlockThreads);
        Blueprint.ResourceBudget.Validate(EntryPoint, info, BlockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Ptx, info, BlockThreads, activeBlocks, _module.JitInfoLog);
    }

    internal unsafe void Launch(DirectPtxTensorView predictions, DirectPtxTensorView output, DirectPtxTensorView agreement)
    {
        Require(predictions, Blueprint.Tensors[0], nameof(predictions));
        Require(output, Blueprint.Tensors[1], nameof(output));
        Require(agreement, Blueprint.Tensors[2], nameof(agreement));

        IntPtr predictionsPointer = predictions.Pointer;
        IntPtr outputPointer = output.Pointer;
        IntPtr agreementPointer = agreement.Pointer;
        void** arguments = stackalloc void*[3];
        arguments[0] = &predictionsPointer;
        arguments[1] = &outputPointer;
        arguments[2] = &agreementPointer;
        uint grid = (uint)((BatchSize * InputCapsules * OutputCapsules) / BlockThreads);
        _module.Launch(_function, grid, 1, 1, BlockThreads, 1, 1, 0, arguments);
    }

    public void Dispose() => _module.Dispose();

    internal static string EmitPtx(int ccMajor, int ccMinor, int batchSize, int inputCapsules, int outputCapsules, int capsuleDim)
    {
        ValidateShape(batchSize, inputCapsules, outputCapsules, capsuleDim);
        int predStrideB = inputCapsules * outputCapsules * capsuleDim;   // predictions [B,I,C,D]
        int predStrideI = outputCapsules * capsuleDim;
        int outStrideB = outputCapsules * capsuleDim;                    // output [B,C,D]

        var ptx = new StringBuilder(4_000);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine($"// capsule-agreement B={batchSize} I={inputCapsules} C={outputCapsules} D={capsuleDim}");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 pred_ptr,");
        ptx.AppendLine("    .param .u64 out_ptr,");
        ptx.AppendLine("    .param .u64 agree_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine($".maxntid {BlockThreads}, 1, 1");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<2>;");
        ptx.AppendLine("    .reg .b32 %r<14>;");
        ptx.AppendLine("    .reg .b64 %rd<16>;");
        ptx.AppendLine("    .reg .f32 %f<4>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [pred_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [out_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [agree_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {BlockThreads}, %r0;");   // idx
        ptx.AppendLine($"    rem.u32 %r3, %r2, {outputCapsules};");        // c
        ptx.AppendLine($"    div.u32 %r4, %r2, {outputCapsules};");        // t = idx/C
        ptx.AppendLine($"    rem.u32 %r5, %r4, {inputCapsules};");         // i
        ptx.AppendLine($"    div.u32 %r6, %r4, {inputCapsules};");         // b
        // predBaseElem = b*predStrideB + i*predStrideI + c*capsuleDim
        ptx.AppendLine($"    mul.lo.u32 %r7, %r6, {predStrideB};");
        ptx.AppendLine($"    mad.lo.u32 %r7, %r5, {predStrideI}, %r7;");
        ptx.AppendLine($"    mad.lo.u32 %r7, %r3, {capsuleDim}, %r7;");
        ptx.AppendLine("    mul.wide.u32 %rd3, %r7, 4;");
        ptx.AppendLine("    add.u64 %rd6, %rd0, %rd3;");                   // &predictions[b,i,c,0]
        // outBaseElem = b*outStrideB + c*capsuleDim
        ptx.AppendLine($"    mul.lo.u32 %r8, %r6, {outStrideB};");
        ptx.AppendLine($"    mad.lo.u32 %r8, %r3, {capsuleDim}, %r8;");
        ptx.AppendLine("    mul.wide.u32 %rd4, %r8, 4;");
        ptx.AppendLine("    add.u64 %rd7, %rd1, %rd4;");                   // &output[b,c,0]
        ptx.AppendLine("    mov.f32 %f0, 0f00000000;");                   // sum
        ptx.AppendLine("    mov.u32 %r9, 0;");                            // d = 0
        ptx.AppendLine("$CAG_D_LOOP:");
        ptx.AppendLine("    ld.global.nc.f32 %f1, [%rd6];");             // predictions[b,i,c,d]
        ptx.AppendLine("    ld.global.nc.f32 %f2, [%rd7];");             // output[b,c,d]
        ptx.AppendLine("    fma.rn.f32 %f0, %f1, %f2, %f0;");
        ptx.AppendLine("    add.u64 %rd6, %rd6, 4;");                    // both stride 1 elem
        ptx.AppendLine("    add.u64 %rd7, %rd7, 4;");
        ptx.AppendLine("    add.u32 %r9, %r9, 1;");
        ptx.AppendLine($"    setp.lt.u32 %p0, %r9, {capsuleDim};");
        ptx.AppendLine("    @%p0 bra $CAG_D_LOOP;");
        ptx.AppendLine("    mul.wide.u32 %rd8, %r2, 4;");
        ptx.AppendLine("    add.u64 %rd9, %rd2, %rd8;");
        ptx.AppendLine("    st.global.f32 [%rd9], %f0;");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture, int batchSize, int inputCapsules, int outputCapsules, int capsuleDim)
    {
        var predExtent = new DirectPtxExtent(batchSize * inputCapsules * outputCapsules * capsuleDim);
        var outExtent = new DirectPtxExtent(batchSize * outputCapsules * capsuleDim);
        var agreeExtent = new DirectPtxExtent(batchSize * inputCapsules * outputCapsules);
        return new DirectPtxKernelBlueprint(
            Operation: "capsule-agreement",
            Version: 1,
            Architecture: architecture,
            Variant: $"fp32-b{batchSize}-i{inputCapsules}-c{outputCapsules}-d{capsuleDim}",
            Tensors:
            [
                new("predictions", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    predExtent, predExtent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    outExtent, outExtent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("agreement", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    agreeExtent, agreeExtent, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 24,
                MaxStaticSharedBytes: 0,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 1),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["formula"] = "agreement[b,i,c] = sum_d predictions[b,i,c,d] * output[b,c,d]",
                ["global-intermediates"] = "none",
                ["temporary-device-allocation"] = "none",
                ["stride-parameters"] = "none"
            });
    }

    internal static bool IsSupportedShape(int batchSize, int inputCapsules, int outputCapsules, int capsuleDim)
    {
        if (batchSize <= 0 || inputCapsules <= 0 || outputCapsules <= 0 || capsuleDim <= 0) return false;
        if (capsuleDim > MaxCapsuleDim) return false;
        long outputs = (long)batchSize * inputCapsules * outputCapsules;
        return outputs > 0 && outputs % BlockThreads == 0 && outputs <= MaxOutputs;
    }

    internal static bool IsPromotedShape(int batchSize, int inputCapsules, int outputCapsules, int capsuleDim) => false;

    private static void ValidateShape(int batchSize, int inputCapsules, int outputCapsules, int capsuleDim)
    {
        if (!IsSupportedShape(batchSize, inputCapsules, outputCapsules, capsuleDim))
            throw new ArgumentOutOfRangeException(
                nameof(batchSize),
                $"Capsule agreement requires positive dims with capsuleDim<={MaxCapsuleDim} and (B*I*C) a multiple of {BlockThreads} up to {MaxOutputs}.");
    }

    private static void Require(DirectPtxTensorView view, DirectPtxTensorContract contract, string parameter)
    {
        if (view.Pointer == IntPtr.Zero || view.PhysicalType != contract.PhysicalType ||
            view.Layout != contract.Layout || view.LogicalExtent != contract.LogicalExtent ||
            view.PhysicalExtent != contract.PhysicalExtent || view.ByteLength != contract.RequiredBytes)
            throw new ArgumentException(
                $"{parameter} does not satisfy physical ABI '{contract.Name}'.", parameter);
    }
}
