using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Quantum measurement probabilities from complex amplitudes:
/// <c>probabilities[i] = real[i]^2 + imag[i]^2</c> (issue #854), matching the NVRTC
/// <c>quantum_measurement</c> kernel. One thread owns one amplitude — no shared memory, no
/// reduction. (Normalization is a separate reduction op and is not fused here, matching the
/// NVRTC split between <c>quantum_measurement</c> and <c>normalize_probabilities</c>.)
///
/// 256 threads/block, grid = count/256 (a positive multiple of 256), so there is no divergent
/// bounds guard. <c>count = batchSize * stateSize</c>.
/// </summary>
internal sealed class PtxQuantumMeasurementKernel : IDisposable
{
    internal const int BlockThreads = 256;
    internal const int MaxCount = 2048 * 4096;
    internal const string EntryPoint = "aidotnet_quantum_measurement";

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int Count { get; }
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxQuantumMeasurementKernel(DirectPtxRuntime runtime, int count)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasValidatedScientific(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The checked-in quantum-measurement specialization is measured only on GA10x/SM86.");
        ValidateShape(count);
        Count = count;
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, count);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor, count);
        _module = runtime.LoadModule(Ptx);
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo info);
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, BlockThreads);
        Blueprint.ResourceBudget.Validate(EntryPoint, info, BlockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Ptx, info, BlockThreads, activeBlocks, _module.JitInfoLog);
    }

    internal unsafe void Launch(DirectPtxTensorView real, DirectPtxTensorView imag, DirectPtxTensorView probabilities)
    {
        Require(real, Blueprint.Tensors[0], nameof(real));
        Require(imag, Blueprint.Tensors[1], nameof(imag));
        Require(probabilities, Blueprint.Tensors[2], nameof(probabilities));

        IntPtr realPointer = real.Pointer;
        IntPtr imagPointer = imag.Pointer;
        IntPtr probPointer = probabilities.Pointer;
        void** arguments = stackalloc void*[3];
        arguments[0] = &realPointer;
        arguments[1] = &imagPointer;
        arguments[2] = &probPointer;
        _module.Launch(_function, (uint)(Count / BlockThreads), 1, 1, BlockThreads, 1, 1, 0, arguments);
    }

    public void Dispose() => _module.Dispose();

    internal static string EmitPtx(int ccMajor, int ccMinor, int count)
    {
        ValidateShape(count);

        var ptx = new StringBuilder(2_500);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine($"// quantum-measurement count={count}");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 real_ptr,");
        ptx.AppendLine("    .param .u64 imag_ptr,");
        ptx.AppendLine("    .param .u64 prob_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine($".maxntid {BlockThreads}, 1, 1");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .b32 %r<4>;");
        ptx.AppendLine("    .reg .b64 %rd<10>;");
        ptx.AppendLine("    .reg .f32 %f<4>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [real_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [imag_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [prob_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {BlockThreads}, %r0;");
        ptx.AppendLine("    mul.wide.u32 %rd3, %r2, 4;");
        ptx.AppendLine("    add.u64 %rd4, %rd0, %rd3;");
        ptx.AppendLine("    add.u64 %rd5, %rd1, %rd3;");
        ptx.AppendLine("    add.u64 %rd6, %rd2, %rd3;");
        ptx.AppendLine("    ld.global.nc.f32 %f0, [%rd4];");   // re
        ptx.AppendLine("    ld.global.nc.f32 %f1, [%rd5];");   // im
        ptx.AppendLine("    mul.rn.f32 %f2, %f0, %f0;");       // re*re
        ptx.AppendLine("    fma.rn.f32 %f2, %f1, %f1, %f2;");  // + im*im
        ptx.AppendLine("    st.global.f32 [%rd6], %f2;");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static DirectPtxKernelBlueprint CreateBlueprint(DirectPtxArchitectureFamily architecture, int count)
    {
        var extent = new DirectPtxExtent(count);
        return new DirectPtxKernelBlueprint(
            Operation: "quantum-measurement",
            Version: 1,
            Architecture: architecture,
            Variant: $"fp32-count{count}",
            Tensors:
            [
                new("real", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    extent, extent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("imag", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    extent, extent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("probabilities", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    extent, extent, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 16,
                MaxStaticSharedBytes: 0,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 1),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["formula"] = "probabilities[i] = real[i]^2 + imag[i]^2",
                ["note"] = "unnormalized; normalization is a separate reduction op (normalize_probabilities)",
                ["global-intermediates"] = "none",
                ["temporary-device-allocation"] = "none",
                ["stride-parameters"] = "none"
            });
    }

    internal static bool IsSupportedCount(int count) =>
        count > 0 && count % BlockThreads == 0 && count <= MaxCount;

    internal static bool IsPromotedCount(int count) => false;

    private static void ValidateShape(int count)
    {
        if (!IsSupportedCount(count))
            throw new ArgumentOutOfRangeException(
                nameof(count),
                $"Quantum measurement supports a positive element count that is a multiple of {BlockThreads} up to {MaxCount}.");
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
