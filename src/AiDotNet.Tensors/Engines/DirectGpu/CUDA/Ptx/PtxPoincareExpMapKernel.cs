using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Poincaré exponential map <c>exp_x(v)</c> per vector (issue #854), matching the established
/// NVRTC kernel exactly. Thread-per-vector: one thread reduces <c>|x|²</c>, <c>|v|²</c> and
/// <c>⟨x,v⟩</c> serially, forms the tangent scale <c>tanh(√c·|v|·(1−c|x|²)/2)/(√c·|v|)</c>,
/// then writes the Möbius combination
/// <c>(num_x·x + num_sv·(scale·v)) / denom</c> where the coefficients derive from the scaled
/// tangent. A zero tangent (<c>|v| &lt; 1e-10</c>) copies the base point. No shared memory,
/// no block reduction, no global intermediate.
///
/// 256 threads/block, grid = batch/256 (batch a positive multiple of 256); dim in {32,64,128}.
/// </summary>
internal sealed class PtxPoincareExpMapKernel : IDisposable
{
    internal const int BlockThreads = 256;
    internal static readonly int[] SupportedDims = { 32, 64, 128 };
    internal const string EntryPoint = "aidotnet_poincare_exp_map";

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int Batch { get; }
    internal int Dim { get; }
    internal float Curvature { get; }
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxPoincareExpMapKernel(DirectPtxRuntime runtime, int batch, int dim, float curvature)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasValidatedScientific(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The checked-in poincare-exp-map specialization is measured only on GA10x/SM86.");
        ValidateShape(batch, dim);
        Batch = batch;
        Dim = dim;
        Curvature = curvature;
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, batch, dim);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor, dim);
        _module = runtime.LoadModule(Ptx);
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo info);
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, BlockThreads);
        Blueprint.ResourceBudget.Validate(EntryPoint, info, BlockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Ptx, info, BlockThreads, activeBlocks, _module.JitInfoLog);
    }

    internal unsafe void Launch(DirectPtxTensorView basePoint, DirectPtxTensorView tangent, DirectPtxTensorView output)
    {
        Require(basePoint, Blueprint.Tensors[0], nameof(basePoint));
        Require(tangent, Blueprint.Tensors[1], nameof(tangent));
        Require(output, Blueprint.Tensors[2], nameof(output));

        IntPtr basePointer = basePoint.Pointer;
        IntPtr tangentPointer = tangent.Pointer;
        IntPtr outputPointer = output.Pointer;
        float curvature = Curvature;
        void** arguments = stackalloc void*[4];
        arguments[0] = &basePointer;
        arguments[1] = &tangentPointer;
        arguments[2] = &outputPointer;
        arguments[3] = &curvature;
        _module.Launch(_function, (uint)(Batch / BlockThreads), 1, 1, BlockThreads, 1, 1, 0, arguments);
    }

    public void Dispose() => _module.Dispose();

    internal static string EmitPtx(int ccMajor, int ccMinor, int dim)
    {
        if (Array.IndexOf(SupportedDims, dim) < 0)
            throw new ArgumentOutOfRangeException(nameof(dim));
        int rowBytes = checked(dim * sizeof(float));
        string tiny = "0f" + BitConverter.ToInt32(BitConverter.GetBytes(1e-10f), 0).ToString("X8");
        const string One = "0f3F800000";
        const string Half = "0f3F000000";
        const string Two = "0f40000000";

        var ptx = new StringBuilder(9_000);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine($"// poincare-exp-map dim={dim}");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 base_ptr,");
        ptx.AppendLine("    .param .u64 tangent_ptr,");
        ptx.AppendLine("    .param .u64 output_ptr,");
        ptx.AppendLine("    .param .f32 curvature");
        ptx.AppendLine(")");
        ptx.AppendLine($".maxntid {BlockThreads}, 1, 1");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<4>;");
        ptx.AppendLine("    .reg .b32 %r<10>;");
        ptx.AppendLine("    .reg .b64 %rd<16>;");
        ptx.AppendLine("    .reg .f32 %f<28>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [base_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [tangent_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [output_ptr];");
        ptx.AppendLine("    ld.param.f32 %f0, [curvature];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {BlockThreads}, %r0;");
        ptx.AppendLine($"    mul.wide.u32 %rd3, %r2, {rowBytes};");
        ptx.AppendLine("    add.u64 %rd4, %rd0, %rd3;");     // &x[row]
        ptx.AppendLine("    add.u64 %rd5, %rd1, %rd3;");     // &v[row]
        ptx.AppendLine("    add.u64 %rd6, %rd2, %rd3;");     // &out[row]

        // ---- Pass 1: xNormSq(%f2), vNormSq(%f3), xvDot(%f4) ----
        ptx.AppendLine("    mov.f32 %f2, 0f00000000;");
        ptx.AppendLine("    mov.f32 %f3, 0f00000000;");
        ptx.AppendLine("    mov.f32 %f4, 0f00000000;");
        ptx.AppendLine("    mov.u32 %r3, 0;");
        ptx.AppendLine("NORM_LOOP:");
        ptx.AppendLine($"    setp.ge.u32 %p0, %r3, {dim};");
        ptx.AppendLine("    @%p0 bra.uni NORM_DONE;");
        ptx.AppendLine("    mul.wide.u32 %rd7, %r3, 4;");
        ptx.AppendLine("    add.u64 %rd8, %rd4, %rd7;");
        ptx.AppendLine("    add.u64 %rd9, %rd5, %rd7;");
        ptx.AppendLine("    ld.global.nc.f32 %f20, [%rd8];");    // x_i
        ptx.AppendLine("    ld.global.nc.f32 %f21, [%rd9];");    // v_i
        ptx.AppendLine("    fma.rn.f32 %f2, %f20, %f20, %f2;");
        ptx.AppendLine("    fma.rn.f32 %f3, %f21, %f21, %f3;");
        ptx.AppendLine("    fma.rn.f32 %f4, %f20, %f21, %f4;");
        ptx.AppendLine("    add.u32 %r3, %r3, 1;");
        ptx.AppendLine("    bra.uni NORM_LOOP;");
        ptx.AppendLine("NORM_DONE:");
        ptx.AppendLine("    sqrt.rn.f32 %f5, %f3;");            // vNorm

        // Zero tangent -> copy base point.
        ptx.AppendLine($"    setp.lt.f32 %p1, %f5, {tiny};");
        ptx.AppendLine("    @%p1 bra.uni COPY_BASE;");

        // scale = tanh(sqrtC*vNorm*(1-c*xNormSq)/2) / (sqrtC*vNorm)
        ptx.AppendLine("    sqrt.rn.f32 %f6, %f0;");            // sqrtC
        ptx.AppendLine("    mul.rn.f32 %f7, %f0, %f2;");        // c*xNormSq
        ptx.AppendLine($"    sub.rn.f32 %f7, {One}, %f7;");     // conformalFactor = 1 - c*xNormSq (= num_sv)
        ptx.AppendLine($"    mul.rn.f32 %f8, %f7, {Half};");    // lambda
        ptx.AppendLine("    mul.rn.f32 %f9, %f6, %f5;");        // sqrtC*vNorm
        ptx.AppendLine("    mul.rn.f32 %f10, %f9, %f8;");       // arg
        ptx.AppendLine("    tanh.approx.f32 %f10, %f10;");
        ptx.AppendLine("    rcp.approx.f32 %f11, %f9;");        // 1/(sqrtC*vNorm)
        ptx.AppendLine("    mul.rn.f32 %f11, %f10, %f11;");     // scale
        // svNormSq = scale^2 * vNormSq ; xsvDot = scale * xvDot
        ptx.AppendLine("    mul.rn.f32 %f12, %f11, %f11;");
        ptx.AppendLine("    mul.rn.f32 %f12, %f12, %f3;");      // svNormSq
        ptx.AppendLine("    mul.rn.f32 %f13, %f11, %f4;");      // xsvDot
        // num_coef_x = 1 + 2c*xsvDot + c*svNormSq
        ptx.AppendLine("    mul.rn.f32 %f16, %f0, %f13;");      // c*xsvDot
        ptx.AppendLine($"    mul.rn.f32 %f16, %f16, {Two};");   // 2c*xsvDot
        ptx.AppendLine("    fma.rn.f32 %f14, %f0, %f12, %f16;");// 2c*xsvDot + c*svNormSq
        ptx.AppendLine($"    add.rn.f32 %f14, %f14, {One};");   // num_coef_x
        // denom = 1 + 2c*xsvDot + c^2*xNormSq*svNormSq ; clamp |denom|<1e-10 -> 1e-10
        ptx.AppendLine("    mul.rn.f32 %f17, %f0, %f0;");       // c^2
        ptx.AppendLine("    mul.rn.f32 %f18, %f2, %f12;");      // xNormSq*svNormSq
        ptx.AppendLine("    mul.rn.f32 %f17, %f17, %f18;");
        ptx.AppendLine($"    add.rn.f32 %f15, %f16, {One};");   // 1 + 2c*xsvDot
        ptx.AppendLine("    add.rn.f32 %f15, %f15, %f17;");     // denom
        ptx.AppendLine("    abs.f32 %f19, %f15;");
        ptx.AppendLine($"    setp.lt.f32 %p2, %f19, {tiny};");
        ptx.AppendLine($"    selp.f32 %f15, {tiny}, %f15, %p2;");
        ptx.AppendLine("    rcp.approx.f32 %f16, %f15;");       // inv

        // ---- Pass 2: out[i] = (num_x*x_i + num_sv*(scale*v_i)) * inv ----
        ptx.AppendLine("    mov.u32 %r3, 0;");
        ptx.AppendLine("WRITE_LOOP:");
        ptx.AppendLine($"    setp.ge.u32 %p0, %r3, {dim};");
        ptx.AppendLine("    @%p0 bra.uni EXP_END;");
        ptx.AppendLine("    mul.wide.u32 %rd7, %r3, 4;");
        ptx.AppendLine("    add.u64 %rd8, %rd4, %rd7;");
        ptx.AppendLine("    add.u64 %rd9, %rd5, %rd7;");
        ptx.AppendLine("    ld.global.nc.f32 %f20, [%rd8];");   // x_i
        ptx.AppendLine("    ld.global.nc.f32 %f21, [%rd9];");   // v_i
        ptx.AppendLine("    mul.rn.f32 %f21, %f21, %f11;");     // scaledV = v_i*scale
        ptx.AppendLine("    mul.rn.f32 %f22, %f14, %f20;");     // num_x*x_i
        ptx.AppendLine("    fma.rn.f32 %f22, %f7, %f21, %f22;");// + num_sv*scaledV
        ptx.AppendLine("    mul.rn.f32 %f22, %f22, %f16;");     // * inv
        ptx.AppendLine("    add.u64 %rd10, %rd6, %rd7;");
        ptx.AppendLine("    st.global.f32 [%rd10], %f22;");
        ptx.AppendLine("    add.u32 %r3, %r3, 1;");
        ptx.AppendLine("    bra.uni WRITE_LOOP;");

        // ---- Zero tangent: copy base point ----
        ptx.AppendLine("COPY_BASE:");
        ptx.AppendLine("    mov.u32 %r3, 0;");
        ptx.AppendLine("COPY_LOOP:");
        ptx.AppendLine($"    setp.ge.u32 %p0, %r3, {dim};");
        ptx.AppendLine("    @%p0 bra.uni EXP_END;");
        ptx.AppendLine("    mul.wide.u32 %rd7, %r3, 4;");
        ptx.AppendLine("    add.u64 %rd8, %rd4, %rd7;");
        ptx.AppendLine("    ld.global.nc.f32 %f20, [%rd8];");
        ptx.AppendLine("    add.u64 %rd10, %rd6, %rd7;");
        ptx.AppendLine("    st.global.f32 [%rd10], %f20;");
        ptx.AppendLine("    add.u32 %r3, %r3, 1;");
        ptx.AppendLine("    bra.uni COPY_LOOP;");
        ptx.AppendLine("EXP_END:");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static DirectPtxKernelBlueprint CreateBlueprint(DirectPtxArchitectureFamily architecture, int batch, int dim)
    {
        var extent = new DirectPtxExtent(batch, dim);
        return new DirectPtxKernelBlueprint(
            Operation: "poincare-exp-map",
            Version: 1,
            Architecture: architecture,
            Variant: $"fp32-batch{batch}-dim{dim}",
            Tensors:
            [
                new("base", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    extent, extent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("tangent", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    extent, extent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    extent, extent, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 40,
                MaxStaticSharedBytes: 0,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 1),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["formula"] = "exp_x(v) = mobius(x, scale*v), scale = tanh(sqrtC*|v|*(1-c|x|^2)/2)/(sqrtC*|v|)",
                ["zero-tangent"] = "copies base point when |v| < 1e-10",
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
                nameof(dim), $"Poincare exp map supports dim in {{32,64,128}} and batch a multiple of {BlockThreads}.");
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
