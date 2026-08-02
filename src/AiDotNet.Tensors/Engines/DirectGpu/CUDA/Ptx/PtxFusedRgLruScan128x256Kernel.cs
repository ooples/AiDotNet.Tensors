using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Exact-shape FP32 RG-LRU forward scan. One SM86 block owns all 256
/// independent channels. Every thread carries its recurrent state across all
/// 128 steps in a register; a four-timestep software pipeline overlaps the
/// independent gate preparation before committing the sequential recurrence.
/// Warp-wide scalar instructions form coalesced 128-byte global transactions.
/// There is no state/intermediate workspace.
/// </summary>
internal sealed class PtxFusedRgLruScan128x256Kernel : IDisposable
{
    internal const string EntryPoint = "aidotnet_rglru_scan_b1_s128_d256";
    internal const int Batch = 1;
    internal const int SequenceLength = 128;
    internal const int RecurrentDimension = 256;
    internal const int BlockThreads = 256;
    internal const int GridBlocks = RecurrentDimension / BlockThreads;

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxFunctionInfo FunctionInfo { get; }
    internal DirectPtxKernelAudit Audit { get; }
    internal string JitInfoLog => _module.JitInfoLog;

    internal PtxFusedRgLruScan128x256Kernel(DirectPtxRuntime runtime)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasValidatedRgLruScan(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new NotSupportedException(
                $"RG-LRU has no issue-#846 prototype for SM " +
                $"{runtime.ComputeCapabilityMajor}.{runtime.ComputeCapabilityMinor}.");

        Blueprint = CreateBlueprint(runtime.ArchitectureFamily);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor);
        DirectPtxModule module = runtime.LoadModule(Ptx);
        try
        {
            _function = module.GetFunction(EntryPoint, out DirectPtxFunctionInfo functionInfo);
            FunctionInfo = functionInfo;
            int activeBlocks = module.GetActiveBlocksPerMultiprocessor(_function, BlockThreads);
            Blueprint.ResourceBudget.Validate(EntryPoint, functionInfo, BlockThreads, activeBlocks);
            Audit = DirectPtxKernelAudit.Create(
                Blueprint, runtime.DeviceFingerprint, Ptx, functionInfo,
                BlockThreads, activeBlocks, module.JitInfoLog);
        }
        catch
        {
            try
            {
                module.Dispose();
            }
            catch
            {
                // Initialization cleanup must not hide the original failure.
            }
            throw;
        }
        _module = module;
    }

    internal static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture)
    {
        var sequence = new DirectPtxExtent(Batch, SequenceLength, RecurrentDimension);
        var channel = new DirectPtxExtent(RecurrentDimension);
        return new DirectPtxKernelBlueprint(
            Operation: "rglru-scan-forward-b1-s128-d256",
            Version: 1,
            Architecture: architecture,
            Variant: "sm86-single-block-channel-persistent-four-step-pipeline",
            Tensors:
            [
                new("value", DirectPtxPhysicalType.Float32,
                    DirectPtxPhysicalLayout.BatchSequenceFeature,
                    sequence, sequence, 128, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("recurrence_gate", DirectPtxPhysicalType.Float32,
                    DirectPtxPhysicalLayout.BatchSequenceFeature,
                    sequence, sequence, 128, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("input_gate", DirectPtxPhysicalType.Float32,
                    DirectPtxPhysicalLayout.BatchSequenceFeature,
                    sequence, sequence, 128, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("decay", DirectPtxPhysicalType.Float32,
                    DirectPtxPhysicalLayout.Vector,
                    channel, channel, 128, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("output", DirectPtxPhysicalType.Float32,
                    DirectPtxPhysicalLayout.BatchSequenceFeature,
                    sequence, sequence, 128, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 32,
                MaxStaticSharedBytes: 0,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 2),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["equation"] = "base=sigmoid(-decay); a=r*base; s=sqrt(max(0,1-a*a)); h=a*h+s*(i*v)",
                ["input"] = "fp32",
                ["accumulator"] = "fp32",
                ["output"] = "fp32",
                ["layout"] = "dense-batch-sequence-feature",
                ["shape"] = "[1,128,256]",
                ["initial-state"] = "zero",
                ["intermediate-global-bytes"] = "0",
                ["temporary-device-bytes"] = "0",
                ["global-reads"] = "decay once/channel; value/recurrence_gate/input_gate once/element",
                ["global-writes"] = "output once/element",
                ["coalescing"] = "32 adjacent channels per warp = one aligned 128-byte span",
                ["time-pipeline"] = "four timesteps prepared independently before sequential state commits",
                ["shared-memory"] = "none; no bank-conflict surface",
                ["promotion"] = "validated-exact-sm86-opt-in"
            });
    }

    internal unsafe void Launch(
        DirectPtxTensorView value,
        DirectPtxTensorView recurrenceGate,
        DirectPtxTensorView inputGate,
        DirectPtxTensorView decay,
        DirectPtxTensorView output)
    {
        Require(value, Blueprint.Tensors[0], nameof(value));
        Require(recurrenceGate, Blueprint.Tensors[1], nameof(recurrenceGate));
        Require(inputGate, Blueprint.Tensors[2], nameof(inputGate));
        Require(decay, Blueprint.Tensors[3], nameof(decay));
        Require(output, Blueprint.Tensors[4], nameof(output));

        IntPtr valuePointer = value.Pointer;
        IntPtr recurrenceGatePointer = recurrenceGate.Pointer;
        IntPtr inputGatePointer = inputGate.Pointer;
        IntPtr decayPointer = decay.Pointer;
        IntPtr outputPointer = output.Pointer;
        void** arguments = stackalloc void*[5];
        arguments[0] = &valuePointer;
        arguments[1] = &recurrenceGatePointer;
        arguments[2] = &inputGatePointer;
        arguments[3] = &decayPointer;
        arguments[4] = &outputPointer;
        _module.Launch(
            _function, GridBlocks, 1, 1, BlockThreads, 1, 1, 0, arguments);
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

    internal static string EmitPtx(int ccMajor, int ccMinor)
    {
        if (!DirectPtxArchitecture.HasValidatedRgLruScan(ccMajor, ccMinor))
            throw new NotSupportedException($"RG-LRU PTX emission is not implemented for SM {ccMajor}.{ccMinor}.");

        var ptx = new StringBuilder(96 * 1024);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 value_ptr,");
        ptx.AppendLine("    .param .u64 recurrence_gate_ptr,");
        ptx.AppendLine("    .param .u64 input_gate_ptr,");
        ptx.AppendLine("    .param .u64 decay_ptr,");
        ptx.AppendLine("    .param .u64 output_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine($".maxntid {BlockThreads}, 1, 1");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<5>;");
        ptx.AppendLine("    .reg .b32 %r<3>;");
        ptx.AppendLine("    .reg .b64 %rd<12>;");
        ptx.AppendLine("    .reg .f32 %f<16>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [value_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [recurrence_gate_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [input_gate_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd3, [decay_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd4, [output_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r2, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r0, %r2, {BlockThreads}, %r0;");
        ptx.AppendLine("    mul.wide.u32 %rd5, %r0, 4;");
        ptx.AppendLine("    add.u64 %rd6, %rd0, %rd5;");
        ptx.AppendLine("    add.u64 %rd7, %rd1, %rd5;");
        ptx.AppendLine("    add.u64 %rd8, %rd2, %rd5;");
        ptx.AppendLine("    add.u64 %rd9, %rd3, %rd5;");
        ptx.AppendLine("    add.u64 %rd10, %rd4, %rd5;");
        ptx.AppendLine("    ld.global.f32 %f0, [%rd9];");
        ptx.AppendLine("    mul.rn.ftz.f32 %f1, %f0, 0f3FB8AA3B;");
        ptx.AppendLine("    ex2.approx.ftz.f32 %f1, %f1;");
        ptx.AppendLine("    add.rn.ftz.f32 %f1, %f1, 0f3F800000;");
        ptx.AppendLine("    rcp.approx.ftz.f32 %f2, %f1;");
        ptx.AppendLine("    mov.f32 %f3, 0f00000000;");
        ptx.AppendLine("    mov.u32 %r1, 0;");
        ptx.AppendLine("$RGLRU_LOOP:");
        ptx.AppendLine("    .pragma \"nounroll\";");
        for (int lane = 0; lane < 4; lane++)
        {
            int offset = lane * RecurrentDimension * sizeof(float);
            ptx.AppendLine($"    ld.global.f32 %f{4 + lane}, [%rd6+{offset}];");
            ptx.AppendLine($"    ld.global.f32 %f{8 + lane}, [%rd7+{offset}];");
            ptx.AppendLine($"    ld.global.f32 %f{12 + lane}, [%rd8+{offset}];");
        }
        ptx.AppendLine("    add.u64 %rd6, %rd6, 4096;");
        ptx.AppendLine("    add.u64 %rd7, %rd7, 4096;");
        ptx.AppendLine("    add.u64 %rd8, %rd8, 4096;");
        for (int lane = 0; lane < 4; lane++)
        {
            int value = 4 + lane;
            int a = 8 + lane;
            int term = 12 + lane;
            ptx.AppendLine($"    mul.rn.ftz.f32 %f{a}, %f{a}, %f2;");
            ptx.AppendLine($"    mul.rn.ftz.f32 %f{term}, %f{term}, %f{value};");
            ptx.AppendLine($"    mul.rn.ftz.f32 %f{value}, %f{a}, %f{a};");
            ptx.AppendLine($"    sub.rn.ftz.f32 %f{value}, 0f3F800000, %f{value};");
            ptx.AppendLine($"    setp.gt.ftz.f32 %p{lane}, %f{value}, 0f00000000;");
            ptx.AppendLine($"    selp.f32 %f{value}, %f{value}, 0f00000000, %p{lane};");
            ptx.AppendLine($"    sqrt.approx.ftz.f32 %f{value}, %f{value};");
            ptx.AppendLine($"    mul.rn.ftz.f32 %f{term}, %f{value}, %f{term};");
        }
        for (int lane = 0; lane < 4; lane++)
        {
            int offset = lane * RecurrentDimension * sizeof(float);
            ptx.AppendLine($"    fma.rn.ftz.f32 %f3, %f{8 + lane}, %f3, %f{12 + lane};");
            ptx.AppendLine($"    st.global.f32 [%rd10+{offset}], %f3;");
        }
        ptx.AppendLine("    add.u64 %rd10, %rd10, 4096;");
        ptx.AppendLine("    add.u32 %r1, %r1, 4;");
        ptx.AppendLine($"    setp.lt.u32 %p4, %r1, {SequenceLength};");
        ptx.AppendLine("    @%p4 bra $RGLRU_LOOP;");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    public void Dispose() => _module.Dispose();
}
