using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Bias-gradient reduction <c>dBias[N] = sum_m dZ[m,N]</c>, the <c>dBias</c> stage of
/// the FusedLinear*Backward family. One thread owns one output column and reduces the
/// M rows of that column in a register accumulator, so there is no shared memory and no
/// global intermediate. 256 threads/block, grid = N/256 (supported N are multiples of
/// 256).
/// </summary>
internal sealed class PtxBiasGradientKernel : IDisposable
{
    internal const int BlockThreads = 256;
    internal const string EntryPoint = "aidotnet_dbias_colsum";

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int M { get; }
    internal int N { get; }
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxBiasGradientKernel(DirectPtxRuntime runtime, int m, int n)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasValidatedFusedLinear(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The checked-in dBias specialization is measured only on GA10x/SM86.");
        ValidateShape(m, n);
        M = m;
        N = n;
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, m, n);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor, m, n);
        _module = runtime.LoadModule(Ptx);
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo info);
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, BlockThreads);
        Blueprint.ResourceBudget.Validate(EntryPoint, info, BlockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Ptx, info, BlockThreads, activeBlocks, _module.JitInfoLog);
    }

    internal unsafe void Launch(DirectPtxTensorView dz, DirectPtxTensorView dbias)
    {
        Require(dz, Blueprint.Tensors[0], nameof(dz));
        Require(dbias, Blueprint.Tensors[1], nameof(dbias));

        IntPtr dzPointer = dz.Pointer;
        IntPtr dbiasPointer = dbias.Pointer;
        void** arguments = stackalloc void*[2];
        arguments[0] = &dzPointer;
        arguments[1] = &dbiasPointer;
        _module.Launch(_function, (uint)(N / BlockThreads), 1, 1, BlockThreads, 1, 1, 0, arguments);
    }

    public void Dispose() => _module.Dispose();

    internal static string EmitPtx(int ccMajor, int ccMinor, int m, int n)
    {
        ValidateShape(m, n);
        int nBytes = checked(n * sizeof(float)); // dZ row stride

        var ptx = new StringBuilder(6_000);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine($"// dbias colsum M={m} N={n}");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 dz_ptr,");
        ptx.AppendLine("    .param .u64 dbias_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine($".maxntid {BlockThreads}, 1, 1");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<4>;");
        ptx.AppendLine("    .reg .b32 %r<12>;");
        ptx.AppendLine("    .reg .b64 %rd<12>;");
        ptx.AppendLine("    .reg .f32 %f<4>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [dz_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [dbias_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {BlockThreads}, %r0;");    // output column n
        ptx.AppendLine("    mul.wide.u32 %rd2, %r2, 4;");
        ptx.AppendLine("    add.u64 %rd3, %rd0, %rd2;");                     // &dZ[0, n]
        ptx.AppendLine("    mov.f32 %f0, 0f00000000;");                     // accumulator
        ptx.AppendLine("    mov.u32 %r3, 0;");                              // m counter
        ptx.AppendLine("    mov.u64 %rd4, %rd3;");                          // running &dZ[m, n]
        ptx.AppendLine("COLSUM_LOOP:");
        ptx.AppendLine("    ld.global.nc.f32 %f1, [%rd4];");
        ptx.AppendLine("    add.rn.f32 %f0, %f0, %f1;");
        ptx.AppendLine($"    add.u64 %rd4, %rd4, {nBytes};");               // advance one M row
        ptx.AppendLine("    add.u32 %r3, %r3, 1;");
        ptx.AppendLine($"    setp.lt.u32 %p0, %r3, {m};");
        ptx.AppendLine("    @%p0 bra.uni COLSUM_LOOP;");
        ptx.AppendLine("    add.u64 %rd5, %rd1, %rd2;");                     // &dBias[n]
        ptx.AppendLine("    st.global.f32 [%rd5], %f0;");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture, int m, int n)
    {
        var dz = new DirectPtxExtent(m, n);
        var dbias = new DirectPtxExtent(n);
        return new DirectPtxKernelBlueprint(
            Operation: "dbias-colsum",
            Version: 1,
            Architecture: architecture,
            Variant: $"fp32-m{m}-n{n}",
            Tensors:
            [
                new("dz", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    dz, dz, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("dbias", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    dbias, dbias, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 24,
                MaxStaticSharedBytes: 0,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 1),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["formula"] = "dBias[N] = sum_m dZ[m,N]",
                ["reduction"] = "one-thread-per-column-register-accumulate",
                ["global-intermediates"] = "none",
                ["temporary-device-allocation"] = "none",
                ["stride-parameters"] = "none"
            });
    }

    internal static bool IsSupportedShape(int m, int n) =>
        m > 0 && m % 64 == 0 &&
        n > 0 && n % BlockThreads == 0 &&
        m is 64 or 128 or 256 or 512 or 1024 or 2048 &&
        n is 256 or 512 or 1024 or 2048 or 4096;

    internal static bool IsPromotedShape(int m, int n) => false;

    private static void ValidateShape(int m, int n)
    {
        if (!IsSupportedShape(m, n))
            throw new ArgumentOutOfRangeException(
                nameof(m),
                "dBias supports M in {64,128,256,512,1024,2048}, N in {256,512,1024,2048,4096}.");
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
