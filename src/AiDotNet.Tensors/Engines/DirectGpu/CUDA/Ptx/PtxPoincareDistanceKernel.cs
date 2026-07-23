using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Poincaré geodesic distance <c>d_c(x,y) = (2/√c)·arctanh(√c·‖(−x) ⊕_c y‖)</c> per vector
/// (issue #854), matching the established NVRTC formula. One block owns one vector (128
/// lanes, dim 32/64/128): it tree-reduces <c>|x|²</c>, <c>|y|²</c>, <c>⟨x,y⟩</c> to form the
/// Möbius coefficients of <c>(−x) ⊕ y</c>, computes each difference lane in registers,
/// tree-reduces the difference norm, then thread 0 writes the scalar distance
/// <c>(1/√c)·ln((1+arg)/(1−arg))</c>, <c>arg = min(√c·‖diff‖, 1)</c>. Output is [batch].
/// </summary>
internal sealed class PtxPoincareDistanceKernel : IDisposable
{
    internal const int BlockThreads = 128;
    internal static readonly int[] SupportedDims = { 32, 64, 128 };
    internal const int MaxBatch = 1 << 20;
    internal const string EntryPoint = "aidotnet_poincare_distance";

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int Batch { get; }
    internal int Dim { get; }
    internal float Curvature { get; }
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxPoincareDistanceKernel(DirectPtxRuntime runtime, int batch, int dim, float curvature)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasValidatedScientific(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The checked-in poincare-distance specialization is measured only on GA10x/SM86.");
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

    internal unsafe void Launch(DirectPtxTensorView x, DirectPtxTensorView y, DirectPtxTensorView output)
    {
        Require(x, Blueprint.Tensors[0], nameof(x));
        Require(y, Blueprint.Tensors[1], nameof(y));
        Require(output, Blueprint.Tensors[2], nameof(output));

        IntPtr xPointer = x.Pointer;
        IntPtr yPointer = y.Pointer;
        IntPtr outputPointer = output.Pointer;
        float curvature = Curvature;
        void** arguments = stackalloc void*[4];
        arguments[0] = &xPointer;
        arguments[1] = &yPointer;
        arguments[2] = &outputPointer;
        arguments[3] = &curvature;
        _module.Launch(_function, (uint)Batch, 1, 1, BlockThreads, 1, 1, 0, arguments);
    }

    public void Dispose() => _module.Dispose();

    internal static string EmitPtx(int ccMajor, int ccMinor, int dim)
    {
        if (Array.IndexOf(SupportedDims, dim) < 0)
            throw new ArgumentOutOfRangeException(nameof(dim));
        int rowBytes = checked(dim * sizeof(float));
        string eps = "0f" + BitConverter.ToInt32(BitConverter.GetBytes(1e-15f), 0).ToString("X8");
        const string One = "0f3F800000";     // 1.0
        const string NegOne = "0fBF800000";   // -1.0
        const string Two = "0f40000000";     // 2.0
        const string Ln2 = "0f3F317218";     // 0.6931471805599453

        var ptx = new StringBuilder(9_000);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine($"// poincare-distance dim={dim}");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 x_ptr,");
        ptx.AppendLine("    .param .u64 y_ptr,");
        ptx.AppendLine("    .param .u64 output_ptr,");
        ptx.AppendLine("    .param .f32 curvature");
        ptx.AppendLine(")");
        ptx.AppendLine($".maxntid {BlockThreads}, 1, 1");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<4>;");
        ptx.AppendLine("    .reg .b32 %r<10>;");
        ptx.AppendLine("    .reg .b64 %rd<20>;");
        ptx.AppendLine("    .reg .f32 %f<32>;");
        ptx.AppendLine($"    .shared .align 16 .b8 red[{BlockThreads * sizeof(float)}];");
        ptx.AppendLine("    ld.param.u64 %rd0, [x_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [y_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [output_ptr];");
        ptx.AppendLine("    ld.param.f32 %f5, [curvature];");
        ptx.AppendLine("    mov.u64 %rd6, red;");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mul.wide.u32 %rd7, %r1, {rowBytes};");
        ptx.AppendLine("    add.u64 %rd8, %rd0, %rd7;");
        ptx.AppendLine("    add.u64 %rd9, %rd1, %rd7;");
        ptx.AppendLine("    mul.wide.u32 %rd11, %r0, 4;");
        ptx.AppendLine("    add.u64 %rd12, %rd6, %rd11;");             // &red[tid]

        // load lane (0 for inactive); keep x_i/y_i in registers through the reductions.
        ptx.AppendLine("    mov.f32 %f0, 0f00000000;");
        ptx.AppendLine("    mov.f32 %f1, 0f00000000;");
        ptx.AppendLine($"    setp.lt.u32 %p0, %r0, {dim};");
        ptx.AppendLine("    add.u64 %rd13, %rd8, %rd11;");
        ptx.AppendLine("    add.u64 %rd14, %rd9, %rd11;");
        ptx.AppendLine("    @%p0 ld.global.nc.f32 %f0, [%rd13];");
        ptx.AppendLine("    @%p0 ld.global.nc.f32 %f1, [%rd14];");
        ptx.AppendLine("    mul.rn.f32 %f20, %f0, %f0;");
        ptx.AppendLine("    mul.rn.f32 %f21, %f1, %f1;");
        ptx.AppendLine("    mul.rn.f32 %f22, %f0, %f1;");
        EmitReduce(ptx, "%f20", "%f2");   // xNormSq
        EmitReduce(ptx, "%f21", "%f3");   // yNormSq
        EmitReduce(ptx, "%f22", "%f4");   // xyDot

        // Mobius(-x, y) coefficients: dot(-x,y) = -xyDot.
        ptx.AppendLine("    mul.rn.f32 %f16, %f5, %f4;");             // c*xyDot
        ptx.AppendLine($"    mul.rn.f32 %f16, %f16, {Two};");         // 2c*xyDot
        // coeff1 = 1 - 2c*xyDot + c*yNormSq
        ptx.AppendLine("    mul.rn.f32 %f7, %f5, %f3;");              // c*yNormSq
        ptx.AppendLine($"    add.rn.f32 %f7, %f7, {One};");           // 1 + c*yNormSq
        ptx.AppendLine("    sub.rn.f32 %f7, %f7, %f16;");            // - 2c*xyDot
        // coeff2 = 1 - c*xNormSq
        ptx.AppendLine("    mul.rn.f32 %f8, %f5, %f2;");
        ptx.AppendLine($"    sub.rn.f32 %f8, {One}, %f8;");
        // denom = 1 - 2c*xyDot + c^2*xNormSq*yNormSq
        ptx.AppendLine("    mul.rn.f32 %f17, %f5, %f5;");
        ptx.AppendLine("    mul.rn.f32 %f18, %f2, %f3;");
        ptx.AppendLine("    mul.rn.f32 %f17, %f17, %f18;");          // c^2*xNormSq*yNormSq
        ptx.AppendLine($"    sub.rn.f32 %f6, {One}, %f16;");          // 1 - 2c*xyDot
        ptx.AppendLine("    add.rn.f32 %f6, %f6, %f17;");
        ptx.AppendLine("    abs.f32 %f6, %f6;");
        ptx.AppendLine($"    max.f32 %f6, %f6, {eps};");
        ptx.AppendLine("    rcp.approx.f32 %f9, %f6;");              // 1/denom
        // diff_i = (coeff1*(-x_i) + coeff2*y_i) / denom ; inactive lanes yield 0.
        ptx.AppendLine("    neg.f32 %f12, %f0;");                    // -x_i
        ptx.AppendLine("    mul.rn.f32 %f12, %f7, %f12;");           // coeff1*(-x_i)
        ptx.AppendLine("    fma.rn.f32 %f12, %f8, %f1, %f12;");      // + coeff2*y_i
        ptx.AppendLine("    mul.rn.f32 %f12, %f12, %f9;");           // / denom
        ptx.AppendLine("    mul.rn.f32 %f13, %f12, %f12;");          // diff_i^2
        EmitReduce(ptx, "%f13", "%f14");  // diffNormSq

        // thread 0: scalar distance.
        ptx.AppendLine("    setp.ne.u32 %p2, %r0, 0;");
        ptx.AppendLine("    @%p2 bra.uni PD_END;");
        ptx.AppendLine("    sqrt.rn.f32 %f20, %f14;");               // diffNorm
        ptx.AppendLine("    sqrt.rn.f32 %f21, %f5;");                // sqrtC
        ptx.AppendLine("    mul.rn.f32 %f22, %f21, %f20;");          // arg = sqrtC*diffNorm
        ptx.AppendLine($"    min.f32 %f22, %f22, {One};");           // clamp to <= 1
        ptx.AppendLine($"    add.rn.f32 %f23, %f22, {One};");        // 1 + arg
        ptx.AppendLine("    lg2.approx.f32 %f23, %f23;");
        ptx.AppendLine($"    mul.rn.f32 %f24, %f22, {NegOne};");     // -arg
        ptx.AppendLine($"    add.rn.f32 %f24, %f24, {One};");        // 1 - arg
        ptx.AppendLine("    lg2.approx.f32 %f24, %f24;");
        ptx.AppendLine("    sub.rn.f32 %f23, %f23, %f24;");          // lg2((1+arg)/(1-arg))
        ptx.AppendLine($"    mul.rn.f32 %f23, %f23, {Ln2};");        // ln((1+arg)/(1-arg))
        ptx.AppendLine("    rcp.approx.f32 %f25, %f21;");            // 1/sqrtC
        ptx.AppendLine("    mul.rn.f32 %f23, %f23, %f25;");          // distance
        ptx.AppendLine("    mul.wide.u32 %rd15, %r1, 4;");
        ptx.AppendLine("    add.u64 %rd16, %rd2, %rd15;");
        ptx.AppendLine("    st.global.f32 [%rd16], %f23;");
        ptx.AppendLine("PD_END:");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static void EmitReduce(StringBuilder ptx, string partial, string dst)
    {
        ptx.AppendLine($"    st.shared.f32 [%rd12], {partial};");
        ptx.AppendLine("    bar.sync 0;");
        foreach (int stride in new[] { 64, 32, 16, 8, 4, 2, 1 })
        {
            ptx.AppendLine($"    setp.lt.u32 %p3, %r0, {stride};");
            ptx.AppendLine("    @%p3 ld.shared.f32 %f10, [%rd12];");
            ptx.AppendLine($"    @%p3 ld.shared.f32 %f11, [%rd12+{stride * sizeof(float)}];");
            ptx.AppendLine("    @%p3 add.rn.f32 %f10, %f10, %f11;");
            ptx.AppendLine("    @%p3 st.shared.f32 [%rd12], %f10;");
            ptx.AppendLine("    bar.sync 0;");
        }
        ptx.AppendLine($"    ld.shared.f32 {dst}, [%rd6];");
        ptx.AppendLine("    bar.sync 0;");
    }

    private static DirectPtxKernelBlueprint CreateBlueprint(DirectPtxArchitectureFamily architecture, int batch, int dim)
    {
        var vectors = new DirectPtxExtent(batch, dim);
        var scalars = new DirectPtxExtent(batch);
        return new DirectPtxKernelBlueprint(
            Operation: "poincare-distance",
            Version: 1,
            Architecture: architecture,
            Variant: $"fp32-batch{batch}-dim{dim}",
            Tensors:
            [
                new("x", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    vectors, vectors, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("y", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    vectors, vectors, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    scalars, scalars, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 44,
                MaxStaticSharedBytes: BlockThreads * sizeof(float),
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 1),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["formula"] = "d = (1/sqrt(c)) * ln((1+arg)/(1-arg)), arg = min(sqrt(c)*||(-x)(+)y||, 1)",
                ["reduction"] = "four-in-block-128-lane-tree-reductions-shared",
                ["global-intermediates"] = "none",
                ["temporary-device-allocation"] = "none",
                ["stride-parameters"] = "none"
            });
    }

    internal static bool IsSupportedShape(int batch, int dim) =>
        batch is > 0 and <= MaxBatch && Array.IndexOf(SupportedDims, dim) >= 0;

    internal static bool IsPromotedShape(int batch, int dim) => false;

    private static void ValidateShape(int batch, int dim)
    {
        if (!IsSupportedShape(batch, dim))
            throw new ArgumentOutOfRangeException(
                nameof(dim), "Poincare distance supports dim in {32,64,128} and batch in [1, 2^20].");
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
