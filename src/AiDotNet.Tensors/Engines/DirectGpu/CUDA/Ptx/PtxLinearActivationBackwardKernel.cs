using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Elementwise activation-gradient stage of the FusedLinear*Backward family:
/// <c>dZ[M,N] = dY[M,N] * activation'(preact[M,N])</c>, where <c>preact</c> is the
/// saved forward preactivation (X @ transpose(W) + bias). This is the only
/// activation-specific kernel in the backward pass; the downstream gradients are
/// activation-agnostic — <c>dX = dZ @ W</c> (plain GEMM tile), <c>dW = transpose(dZ)
/// @ X</c> (contract-M GEMM tile) and <c>dBias = colsum(dZ)</c> (reduction). One
/// kernel serves all five activations, differing only in the emitted derivative from
/// <see cref="PtxLinearActivationBackwardEmit"/>.
///
/// One thread per element, 256 threads/block, grid = (M*N)/256. Supported shapes keep
/// M and N multiples of 64, so M*N is always a multiple of 256 (no bounds guard).
/// </summary>
internal sealed class PtxLinearActivationBackwardKernel : IDisposable
{
    internal const int BlockThreads = 256;
    internal const string EntryPoint = "aidotnet_linear_activation_backward";

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int M { get; }
    internal int N { get; }
    internal DirectPtxLinearActivation Activation { get; }
    internal bool DerivativeFromOutput { get; }
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxLinearActivationBackwardKernel(
        DirectPtxRuntime runtime, int m, int n, DirectPtxLinearActivation activation,
        bool derivativeFromOutput = false)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasValidatedFusedLinear(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The checked-in activation-backward specialization is measured only on GA10x/SM86.");
        ValidateShape(m, n);
        M = m;
        N = n;
        Activation = activation;
        DerivativeFromOutput = derivativeFromOutput;
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, m, n, activation);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor, m, n, activation, derivativeFromOutput);
        _module = runtime.LoadModule(Ptx);
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo info);
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, BlockThreads);
        Blueprint.ResourceBudget.Validate(EntryPoint, info, BlockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Ptx, info, BlockThreads, activeBlocks, _module.JitInfoLog);
    }

    internal unsafe void Launch(
        DirectPtxTensorView preact, DirectPtxTensorView dy, DirectPtxTensorView dz)
    {
        Require(preact, Blueprint.Tensors[0], nameof(preact));
        Require(dy, Blueprint.Tensors[1], nameof(dy));
        Require(dz, Blueprint.Tensors[2], nameof(dz));

        IntPtr preactPointer = preact.Pointer;
        IntPtr dyPointer = dy.Pointer;
        IntPtr dzPointer = dz.Pointer;
        void** arguments = stackalloc void*[3];
        arguments[0] = &preactPointer;
        arguments[1] = &dyPointer;
        arguments[2] = &dzPointer;
        _module.Launch(_function, (uint)(M * N / BlockThreads), 1, 1, BlockThreads, 1, 1, 0, arguments);
    }

    public void Dispose() => _module.Dispose();

    internal static string EmitPtx(
        int ccMajor, int ccMinor, int m, int n, DirectPtxLinearActivation activation,
        bool derivativeFromOutput = false)
    {
        ValidateShape(m, n);
        var ptx = new StringBuilder(8_000);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine(
            $"// linear-activation-backward M={m} N={n} act={activation} " +
            $"form={(derivativeFromOutput ? "output" : "preact")}");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 preact_ptr,");
        ptx.AppendLine("    .param .u64 dy_ptr,");
        ptx.AppendLine("    .param .u64 dz_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine($".maxntid {BlockThreads}, 1, 1");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<4>;");
        ptx.AppendLine("    .reg .b32 %r<8>;");
        ptx.AppendLine("    .reg .b64 %rd<12>;");
        ptx.AppendLine("    .reg .f32 %f<16>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [preact_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [dy_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [dz_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {BlockThreads}, %r0;");    // global element index
        ptx.AppendLine("    mul.wide.u32 %rd3, %r2, 4;");
        ptx.AppendLine("    add.u64 %rd4, %rd0, %rd3;");
        ptx.AppendLine("    add.u64 %rd5, %rd1, %rd3;");
        ptx.AppendLine("    add.u64 %rd6, %rd2, %rd3;");
        ptx.AppendLine("    ld.global.nc.f32 %f0, [%rd4];");                 // z (preact) or y (output)
        ptx.AppendLine("    ld.global.nc.f32 %f1, [%rd5];");                 // dy
        if (derivativeFromOutput)
            PtxLinearActivationBackwardEmit.EmitFromOutput(ptx, activation, "%f0", "%f1", "%f2");
        else
            PtxLinearActivationBackwardEmit.Emit(ptx, activation, "%f0", "%f1", "%f2");
        ptx.AppendLine("    st.global.f32 [%rd6], %f2;");                    // dz
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture, int m, int n, DirectPtxLinearActivation activation)
    {
        var extent = new DirectPtxExtent(m, n);
        return new DirectPtxKernelBlueprint(
            Operation: "linear-activation-backward",
            Version: 1,
            Architecture: architecture,
            Variant: $"fp32-m{m}-n{n}-{activation}".ToLowerInvariant(),
            Tensors:
            [
                new("preact", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    extent, extent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("dy", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    extent, extent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("dz", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    extent, extent, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 32,
                MaxStaticSharedBytes: 0,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 1),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["formula"] = "dZ[M,N] = dY[M,N] * activation'(preact[M,N])",
                ["activation"] = activation.ToString(),
                ["role"] = "activation-specific-stage-of-fused-linear-backward",
                ["downstream"] = "dX=dZ@W; dW=transpose(dZ)@X; dBias=colsum(dZ)",
                ["global-intermediates"] = "none",
                ["temporary-device-allocation"] = "none",
                ["stride-parameters"] = "none"
            });
    }

    internal static bool IsSupportedShape(int m, int n) =>
        m > 0 && m % 64 == 0 &&
        n > 0 && n % 64 == 0 &&
        m is 64 or 128 or 256 or 512 or 1024 or 2048 &&
        n is 256 or 512 or 1024 or 2048 or 4096;

    internal static bool IsPromotedShape(int m, int n) => false;

    private static void ValidateShape(int m, int n)
    {
        if (!IsSupportedShape(m, n))
            throw new ArgumentOutOfRangeException(
                nameof(m),
                "Activation-backward supports M in {64,128,256,512,1024,2048}, N in {256,512,1024,2048,4096}.");
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
