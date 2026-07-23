using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Projects each vector onto the Poincaré ball (issue #854), matching the established NVRTC
/// kernel exactly: with <c>maxNorm = 1/√c − ε</c>, a row is rescaled by <c>maxNorm/‖x‖</c>
/// when <c>‖x‖² ≥ maxNorm²</c> and copied through otherwise. Thread-per-vector (one thread
/// owns one row and streams its <c>dim</c> lanes twice — a norm pass and a write pass) — no
/// shared memory, no reduction, no global intermediate.
///
/// 256 threads/block, grid = batch/256 (batch a positive multiple of 256); dim in
/// {32,64,128}.
/// </summary>
internal sealed class PtxPoincareProjectKernel : IDisposable
{
    internal const int BlockThreads = 256;
    internal static readonly int[] SupportedDims = { 32, 64, 128 };
    internal const string EntryPoint = "aidotnet_poincare_project";

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int Batch { get; }
    internal int Dim { get; }
    internal float Curvature { get; }
    internal float Epsilon { get; }
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxPoincareProjectKernel(DirectPtxRuntime runtime, int batch, int dim, float curvature, float epsilon)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasValidatedScientific(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The checked-in poincare-project specialization is measured only on GA10x/SM86.");
        ValidateShape(batch, dim);
        Batch = batch;
        Dim = dim;
        Curvature = curvature;
        Epsilon = epsilon;
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, batch, dim);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor, dim);
        _module = runtime.LoadModule(Ptx);
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo info);
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, BlockThreads);
        Blueprint.ResourceBudget.Validate(EntryPoint, info, BlockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Ptx, info, BlockThreads, activeBlocks, _module.JitInfoLog);
    }

    internal unsafe void Launch(DirectPtxTensorView input, DirectPtxTensorView output)
    {
        Require(input, Blueprint.Tensors[0], nameof(input));
        Require(output, Blueprint.Tensors[1], nameof(output));

        IntPtr inputPointer = input.Pointer;
        IntPtr outputPointer = output.Pointer;
        float curvature = Curvature;
        float epsilon = Epsilon;
        void** arguments = stackalloc void*[4];
        arguments[0] = &inputPointer;
        arguments[1] = &outputPointer;
        arguments[2] = &curvature;
        arguments[3] = &epsilon;
        _module.Launch(_function, (uint)(Batch / BlockThreads), 1, 1, BlockThreads, 1, 1, 0, arguments);
    }

    public void Dispose() => _module.Dispose();

    internal static string EmitPtx(int ccMajor, int ccMinor, int dim)
    {
        if (Array.IndexOf(SupportedDims, dim) < 0)
            throw new ArgumentOutOfRangeException(nameof(dim));
        int rowBytes = checked(dim * sizeof(float));
        const string One = "0f3F800000";

        var ptx = new StringBuilder(6_000);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine($"// poincare-project dim={dim}");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 input_ptr,");
        ptx.AppendLine("    .param .u64 output_ptr,");
        ptx.AppendLine("    .param .f32 curvature,");
        ptx.AppendLine("    .param .f32 epsilon");
        ptx.AppendLine(")");
        ptx.AppendLine($".maxntid {BlockThreads}, 1, 1");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<4>;");
        ptx.AppendLine("    .reg .b32 %r<10>;");
        ptx.AppendLine("    .reg .b64 %rd<16>;");
        ptx.AppendLine("    .reg .f32 %f<16>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [input_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [output_ptr];");
        ptx.AppendLine("    ld.param.f32 %f0, [curvature];");
        ptx.AppendLine("    ld.param.f32 %f1, [epsilon];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {BlockThreads}, %r0;");   // row
        ptx.AppendLine($"    mul.wide.u32 %rd2, %r2, {rowBytes};");
        ptx.AppendLine("    add.u64 %rd3, %rd0, %rd2;");                    // &input[row]
        ptx.AppendLine("    add.u64 %rd4, %rd1, %rd2;");                    // &output[row]

        // ---- Pass 1: sqNorm = sum x[i]^2 ----
        ptx.AppendLine("    mov.f32 %f2, 0f00000000;");
        ptx.AppendLine("    mov.u32 %r3, 0;");
        ptx.AppendLine("NORM_LOOP:");
        ptx.AppendLine($"    setp.ge.u32 %p0, %r3, {dim};");
        ptx.AppendLine("    @%p0 bra.uni NORM_DONE;");
        ptx.AppendLine("    mul.wide.u32 %rd5, %r3, 4;");
        ptx.AppendLine("    add.u64 %rd6, %rd3, %rd5;");
        ptx.AppendLine("    ld.global.nc.f32 %f3, [%rd6];");
        ptx.AppendLine("    fma.rn.f32 %f2, %f3, %f3, %f2;");
        ptx.AppendLine("    add.u32 %r3, %r3, 1;");
        ptx.AppendLine("    bra.uni NORM_LOOP;");
        ptx.AppendLine("NORM_DONE:");

        // maxNorm = 1/sqrt(c) - epsilon ; scale = (sqNorm >= maxNorm^2) ? maxNorm/sqrt(sqNorm) : 1.
        ptx.AppendLine("    sqrt.rn.f32 %f4, %f0;");                        // sqrt(c)
        ptx.AppendLine("    rcp.approx.f32 %f4, %f4;");                     // 1/sqrt(c)
        ptx.AppendLine("    sub.rn.f32 %f4, %f4, %f1;");                    // maxNorm
        ptx.AppendLine("    mul.rn.f32 %f5, %f4, %f4;");                    // maxNorm^2
        ptx.AppendLine("    sqrt.rn.f32 %f6, %f2;");                        // sqrt(sqNorm)
        ptx.AppendLine("    rcp.approx.f32 %f6, %f6;");                     // 1/sqrt(sqNorm)
        ptx.AppendLine("    mul.rn.f32 %f7, %f4, %f6;");                    // maxNorm/sqrt(sqNorm)
        ptx.AppendLine("    setp.ge.f32 %p1, %f2, %f5;");                   // sqNorm >= maxNorm^2
        ptx.AppendLine($"    selp.f32 %f8, %f7, {One}, %p1;");             // scale

        // ---- Pass 2: y[i] = x[i] * scale ----
        ptx.AppendLine("    mov.u32 %r3, 0;");
        ptx.AppendLine("WRITE_LOOP:");
        ptx.AppendLine($"    setp.ge.u32 %p0, %r3, {dim};");
        ptx.AppendLine("    @%p0 bra.uni WRITE_DONE;");
        ptx.AppendLine("    mul.wide.u32 %rd5, %r3, 4;");
        ptx.AppendLine("    add.u64 %rd6, %rd3, %rd5;");
        ptx.AppendLine("    ld.global.nc.f32 %f3, [%rd6];");
        ptx.AppendLine("    mul.rn.f32 %f3, %f3, %f8;");
        ptx.AppendLine("    add.u64 %rd7, %rd4, %rd5;");
        ptx.AppendLine("    st.global.f32 [%rd7], %f3;");
        ptx.AppendLine("    add.u32 %r3, %r3, 1;");
        ptx.AppendLine("    bra.uni WRITE_LOOP;");
        ptx.AppendLine("WRITE_DONE:");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static DirectPtxKernelBlueprint CreateBlueprint(DirectPtxArchitectureFamily architecture, int batch, int dim)
    {
        var extent = new DirectPtxExtent(batch, dim);
        return new DirectPtxKernelBlueprint(
            Operation: "poincare-project",
            Version: 1,
            Architecture: architecture,
            Variant: $"fp32-batch{batch}-dim{dim}",
            Tensors:
            [
                new("input", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    extent, extent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    extent, extent, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 24,
                MaxStaticSharedBytes: 0,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 1),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["formula"] = "y = ||x||^2 >= (1/sqrt(c)-eps)^2 ? x*(1/sqrt(c)-eps)/||x|| : x",
                ["model"] = "thread-per-vector-serial-dim",
                ["global-intermediates"] = "none",
                ["temporary-device-allocation"] = "none",
                ["stride-parameters"] = "none"
            });
    }

    internal static bool IsSupportedShape(int batch, int dim) =>
        batch > 0 && batch % BlockThreads == 0 && Array.IndexOf(SupportedDims, dim) >= 0;

    internal static bool IsPromotedShape(int batch, int dim) => false;

    private static void ValidateShape(int batch, int dim)
    {
        if (!IsSupportedShape(batch, dim))
            throw new ArgumentOutOfRangeException(
                nameof(dim), $"Poincare project supports dim in {{32,64,128}} and batch a multiple of {BlockThreads}.");
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
