using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Exact-shape deterministic DDIM update matching the currently advertised
/// AiDotNet API. Schedule coefficients are collapsed once on the host in
/// double precision; the kernel is two float4 reads and four FMAs.
/// </summary>
internal sealed class PtxFusedDdimStepF32Kernel : IDisposable
{
    internal const string EntryPoint = "aidotnet_fused_ddim_step_f32";
    internal const int ThreadsPerBlock = 128;
    internal const int ValuesPerThread = 4;

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int ElementCount { get; }
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxFusedDdimStepF32Kernel(DirectPtxRuntime runtime, int elementCount)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasExperimentalRngDropout(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The experimental DDIM specialization targets SM86 only.");
        ValidateElementCount(elementCount);
        ElementCount = elementCount;
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, elementCount);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor, elementCount);
        _module = runtime.LoadModule(Ptx);
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo info);
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, ThreadsPerBlock);
        Blueprint.ResourceBudget.Validate(EntryPoint, info, ThreadsPerBlock, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Ptx, info,
            ThreadsPerBlock, activeBlocks, _module.JitInfoLog);
    }

    internal unsafe void Launch(
        DirectPtxTensorView xT,
        DirectPtxTensorView epsilon,
        DirectPtxTensorView output,
        float xCoefficient,
        float epsilonCoefficient)
    {
        Require(xT, Blueprint.Tensors[0], nameof(xT));
        Require(epsilon, Blueprint.Tensors[1], nameof(epsilon));
        Require(output, Blueprint.Tensors[2], nameof(output));
        if (!PtxCompat.IsFinite(xCoefficient) || !PtxCompat.IsFinite(epsilonCoefficient))
            throw new ArgumentOutOfRangeException(nameof(xCoefficient));
        if (PtxFusedPhiloxDropoutF32Kernel.RangesOverlap(xT, output) ||
            PtxFusedPhiloxDropoutF32Kernel.RangesOverlap(epsilon, output))
            throw new ArgumentException("DDIM output may not alias either input.");

        IntPtr xPointer = xT.Pointer;
        IntPtr epsilonPointer = epsilon.Pointer;
        IntPtr outputPointer = output.Pointer;
        void** arguments = stackalloc void*[5];
        arguments[0] = &xPointer;
        arguments[1] = &epsilonPointer;
        arguments[2] = &outputPointer;
        arguments[3] = &xCoefficient;
        arguments[4] = &epsilonCoefficient;
        uint groups = checked((uint)(ElementCount / ValuesPerThread));
        _module.Launch(
            _function,
            groups / ThreadsPerBlock, 1, 1,
            ThreadsPerBlock, 1, 1, 0, arguments);
    }

    public void Dispose() => _module.Dispose();

    internal static bool IsSupportedElementCount(int elementCount) =>
        PtxFusedPhiloxDropoutF32Kernel.IsSupportedElementCount(elementCount);

    internal static string EmitPtx(int ccMajor, int ccMinor, int elementCount)
    {
        if (!DirectPtxArchitecture.HasExperimentalRngDropout(ccMajor, ccMinor))
            throw new PlatformNotSupportedException(
                "The experimental DDIM emitter targets SM86 only.");
        ValidateElementCount(elementCount);
        int groups = elementCount / ValuesPerThread;
        int blocks = groups / ThreadsPerBlock;
        var ptx = new StringBuilder(2_500);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 x_t_ptr,");
        ptx.AppendLine("    .param .u64 epsilon_ptr,");
        ptx.AppendLine("    .param .u64 output_ptr,");
        ptx.AppendLine("    .param .f32 x_coefficient,");
        ptx.AppendLine("    .param .f32 epsilon_coefficient");
        ptx.AppendLine(")");
        ptx.AppendLine($".maxntid {ThreadsPerBlock}, 1, 1");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .b32 %r<4>;");
        ptx.AppendLine("    .reg .b64 %rd<8>;");
        ptx.AppendLine("    .reg .f32 %f<14>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [x_t_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [epsilon_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [output_ptr];");
        ptx.AppendLine("    ld.param.f32 %f0, [x_coefficient];");
        ptx.AppendLine("    ld.param.f32 %f1, [epsilon_coefficient];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {ThreadsPerBlock}, %r0;");
        ptx.AppendLine("    mul.wide.u32 %rd3, %r2, 16;");
        ptx.AppendLine("    add.u64 %rd4, %rd0, %rd3;");
        ptx.AppendLine("    add.u64 %rd5, %rd1, %rd3;");
        ptx.AppendLine("    add.u64 %rd6, %rd2, %rd3;");
        ptx.AppendLine("    ld.global.v4.f32 {%f2,%f3,%f4,%f5}, [%rd4];");
        ptx.AppendLine("    ld.global.v4.f32 {%f6,%f7,%f8,%f9}, [%rd5];");
        for (int i = 0; i < 4; i++)
        {
            ptx.AppendLine($"    mul.rn.f32 %f{10 + i}, %f{2 + i}, %f0;");
            ptx.AppendLine($"    fma.rn.f32 %f{10 + i}, %f{6 + i}, %f1, %f{10 + i};");
        }
        ptx.AppendLine("    st.global.v4.f32 [%rd6], {%f10,%f11,%f12,%f13};");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        ptx.AppendLine($"// exact_elements={elementCount}; exact_blocks={blocks}; no_tail_branch=1");
        return ptx.ToString();
    }

    private static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture,
        int elementCount)
    {
        var extent = new DirectPtxExtent(elementCount);
        return new DirectPtxKernelBlueprint(
            Operation: "fused-ddim-step-f32",
            Version: 1,
            Architecture: architecture,
            Variant: $"exact-n{elementCount}-float4-t{ThreadsPerBlock}",
            Tensors:
            [
                new("x-t", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    extent, extent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("epsilon-theta", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    extent, extent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("x-t-minus-one", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    extent, extent, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 24,
                MaxStaticSharedBytes: 0,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 12),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["schedule"] = "host computes cXt and cEpsilon in FP64 once from validated alpha-bar inputs",
                ["output"] = "cXt*xT+cEpsilon*epsilonTheta",
                ["stochastic-mode"] = "not advertised by current AiDotNet FusedDDIMStep API; no eta/noise parameter exists",
                ["global-reads"] = "one xT float4 plus one epsilon float4 per thread",
                ["global-writes"] = "one output float4 per thread",
                ["global-intermediates"] = "none",
                ["temporary-device-allocation"] = "none",
                ["stride-and-tail-parameters"] = "none"
            });
    }

    private static void ValidateElementCount(int elementCount)
    {
        if (!IsSupportedElementCount(elementCount))
            throw new ArgumentOutOfRangeException(
                nameof(elementCount), "Supported exact element buckets are 4096, 65536, and 1048576.");
    }

    private static void Require(
        DirectPtxTensorView view,
        DirectPtxTensorContract contract,
        string parameter)
    {
        if (view.Pointer == IntPtr.Zero || view.PhysicalType != contract.PhysicalType ||
            view.Layout != contract.Layout || view.LogicalExtent != contract.LogicalExtent ||
            view.PhysicalExtent != contract.PhysicalExtent || view.ByteLength != contract.RequiredBytes ||
            view.AllocationByteLength != contract.RequiredBytes)
            throw new ArgumentException(
                $"{parameter} does not satisfy exact physical ABI '{contract.Name}'.", parameter);
    }
}
