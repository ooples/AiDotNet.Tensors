using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Poincaré geodesic distance <c>d_c(x,y) = (2/√c)·arctanh(√c·‖(−x) ⊕_c y‖)</c> per vector
/// (issue #854), matching the established NVRTC formula. Each 128-lane block owns four vectors,
/// one per warp. The lanes retain their strided x/y elements in registers and use warp shuffles
/// to reduce <c>|x|²</c>, <c>|y|²</c>, <c>⟨x,y⟩</c>, and the final difference norm without shared
/// memory or block barriers. Lane 0 of each warp writes the scalar distance
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
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor, batch, dim);
        var loaded = DirectPtxResourceInitialization.Complete(
            runtime.LoadModule(Ptx),
            module =>
            {
                IntPtr function = module.GetFunction(EntryPoint, out DirectPtxFunctionInfo info);
                int activeBlocks = module.GetActiveBlocksPerMultiprocessor(function, BlockThreads);
                Blueprint.ResourceBudget.Validate(EntryPoint, info, BlockThreads, activeBlocks);
                var audit = DirectPtxKernelAudit.Create(
                    Blueprint, runtime.DeviceFingerprint, Ptx, info, BlockThreads, activeBlocks,
                    module);
                return (Function: function, Audit: audit);
            });
        _module = loaded.Resource;
        _function = loaded.Value.Function;
        Audit = loaded.Value.Audit;
    }

    internal unsafe void Launch(DirectPtxTensorView x, DirectPtxTensorView y, DirectPtxTensorView output)
    {
        DirectPtxAbi.Require(x, Blueprint.Tensors[0], nameof(x));
        DirectPtxAbi.Require(y, Blueprint.Tensors[1], nameof(y));
        DirectPtxAbi.Require(output, Blueprint.Tensors[2], nameof(output));

        IntPtr xPointer = x.Pointer;
        IntPtr yPointer = y.Pointer;
        IntPtr outputPointer = output.Pointer;
        float curvature = Curvature;
        void** arguments = stackalloc void*[4];
        arguments[0] = &xPointer;
        arguments[1] = &yPointer;
        arguments[2] = &outputPointer;
        arguments[3] = &curvature;
        _module.Launch(_function, (uint)((Batch + 3) / 4), 1, 1, BlockThreads, 1, 1, 0, arguments);
    }

    public void Dispose() => _module.Dispose();

    internal static string EmitPtx(int ccMajor, int ccMinor, int batch, int dim)
    {
        if (batch <= 0 || batch > MaxBatch)
            throw new ArgumentOutOfRangeException(nameof(batch));
        if (Array.IndexOf(SupportedDims, dim) < 0)
            throw new ArgumentOutOfRangeException(nameof(dim));
        int rowBytes = checked(dim * sizeof(float));
        int elementsPerLane = dim / 32;
        string eps = "0f" + BitConverter.ToInt32(BitConverter.GetBytes(1e-15f), 0).ToString("X8");
        const string One = "0f3F800000";     // 1.0
        const string NegOne = "0fBF800000";   // -1.0
        const string Two = "0f40000000";     // 2.0
        const string Ln2 = "0f3F317218";     // 0.6931471805599453

        var ptx = new StringBuilder(9_000);
        DirectPtxPtxText.AppendModuleHeader(ptx, ccMajor, ccMinor);
        ptx.AppendLine($"// poincare-distance batch={batch} dim={dim}; one vector per warp");
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
        ptx.AppendLine("    .reg .b32 %r<16>;");
        ptx.AppendLine("    .reg .b64 %rd<16>;");
        ptx.AppendLine("    .reg .f32 %f<32>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [x_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [y_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [output_ptr];");
        ptx.AppendLine("    ld.param.f32 %f14, [curvature];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine("    shr.u32 %r2, %r0, 5;");
        ptx.AppendLine("    and.b32 %r3, %r0, 31;");
        ptx.AppendLine("    mad.lo.u32 %r4, %r1, 4, %r2;");
        ptx.AppendLine($"    setp.ge.u32 %p0, %r4, {batch};");
        ptx.AppendLine("    @%p0 bra.uni PD_END;");
        ptx.AppendLine($"    mul.wide.u32 %rd3, %r4, {rowBytes};");
        ptx.AppendLine("    add.u64 %rd4, %rd0, %rd3;");
        ptx.AppendLine("    add.u64 %rd5, %rd1, %rd3;");
        ptx.AppendLine("    mul.wide.u32 %rd6, %r3, 4;");
        ptx.AppendLine("    add.u64 %rd4, %rd4, %rd6;");
        ptx.AppendLine("    add.u64 %rd5, %rd5, %rd6;");

        ptx.AppendLine("    mov.f32 %f8, 0f00000000;");
        ptx.AppendLine("    mov.f32 %f9, 0f00000000;");
        ptx.AppendLine("    mov.f32 %f10, 0f00000000;");
        for (int element = 0; element < elementsPerLane; element++)
        {
            int offset = element * 32 * sizeof(float);
            ptx.AppendLine($"    ld.global.nc.f32 %f{element}, [%rd4+{offset}];");
            ptx.AppendLine($"    ld.global.nc.f32 %f{4 + element}, [%rd5+{offset}];");
            ptx.AppendLine($"    fma.rn.f32 %f8, %f{element}, %f{element}, %f8;");
            ptx.AppendLine($"    fma.rn.f32 %f9, %f{4 + element}, %f{4 + element}, %f9;");
            ptx.AppendLine($"    fma.rn.f32 %f10, %f{element}, %f{4 + element}, %f10;");
        }
        EmitWarpReduce(ptx, "%f8", "%f11");
        EmitWarpReduce(ptx, "%f9", "%f12");
        EmitWarpReduce(ptx, "%f10", "%f13");

        // Mobius(-x, y) coefficients: dot(-x,y) = -xyDot.
        ptx.AppendLine("    mul.rn.f32 %f16, %f14, %f13;");           // c*xyDot
        ptx.AppendLine($"    mul.rn.f32 %f16, %f16, {Two};");         // 2c*xyDot
        // coeff1 = 1 - 2c*xyDot + c*yNormSq
        ptx.AppendLine("    mul.rn.f32 %f17, %f14, %f12;");           // c*yNormSq
        ptx.AppendLine($"    add.rn.f32 %f17, %f17, {One};");         // 1 + c*yNormSq
        ptx.AppendLine("    sub.rn.f32 %f17, %f17, %f16;");          // - 2c*xyDot
        // coeff2 = 1 - c*xNormSq
        ptx.AppendLine("    mul.rn.f32 %f18, %f14, %f11;");
        ptx.AppendLine($"    sub.rn.f32 %f18, {One}, %f18;");
        // denom = 1 - 2c*xyDot + c^2*xNormSq*yNormSq
        ptx.AppendLine("    mul.rn.f32 %f29, %f14, %f14;");
        ptx.AppendLine("    mul.rn.f32 %f20, %f11, %f12;");
        ptx.AppendLine("    mul.rn.f32 %f29, %f29, %f20;");          // c^2*xNormSq*yNormSq
        ptx.AppendLine($"    sub.rn.f32 %f19, {One}, %f16;");        // 1 - 2c*xyDot
        ptx.AppendLine("    add.rn.f32 %f19, %f19, %f29;");
        ptx.AppendLine("    abs.f32 %f19, %f19;");
        ptx.AppendLine($"    max.f32 %f19, %f19, {eps};");
        ptx.AppendLine("    rcp.approx.f32 %f20, %f19;");            // 1/denom
        ptx.AppendLine("    mov.f32 %f21, 0f00000000;");
        for (int element = 0; element < elementsPerLane; element++)
        {
            ptx.AppendLine($"    neg.f32 %f15, %f{element};");
            ptx.AppendLine("    mul.rn.f32 %f15, %f17, %f15;");
            ptx.AppendLine($"    fma.rn.f32 %f15, %f18, %f{4 + element}, %f15;");
            ptx.AppendLine("    mul.rn.f32 %f15, %f15, %f20;");
            ptx.AppendLine("    fma.rn.f32 %f21, %f15, %f15, %f21;");
        }
        EmitWarpReduce(ptx, "%f21", "%f22");

        // thread 0: scalar distance.
        ptx.AppendLine("    setp.ne.u32 %p2, %r3, 0;");
        ptx.AppendLine("    @%p2 bra.uni PD_END;");
        ptx.AppendLine("    sqrt.rn.f32 %f23, %f22;");               // diffNorm
        ptx.AppendLine("    sqrt.rn.f32 %f24, %f14;");               // sqrtC
        ptx.AppendLine("    mul.rn.f32 %f25, %f24, %f23;");          // arg = sqrtC*diffNorm
        ptx.AppendLine($"    min.f32 %f25, %f25, {One};");           // clamp to <= 1
        ptx.AppendLine($"    add.rn.f32 %f26, %f25, {One};");        // 1 + arg
        ptx.AppendLine("    lg2.approx.f32 %f26, %f26;");
        ptx.AppendLine($"    mul.rn.f32 %f27, %f25, {NegOne};");     // -arg
        ptx.AppendLine($"    add.rn.f32 %f27, %f27, {One};");        // 1 - arg
        ptx.AppendLine("    lg2.approx.f32 %f27, %f27;");
        ptx.AppendLine("    sub.rn.f32 %f26, %f26, %f27;");          // lg2((1+arg)/(1-arg))
        ptx.AppendLine($"    mul.rn.f32 %f26, %f26, {Ln2};");        // ln((1+arg)/(1-arg))
        ptx.AppendLine("    rcp.approx.f32 %f28, %f24;");            // 1/sqrtC
        ptx.AppendLine("    mul.rn.f32 %f26, %f26, %f28;");          // distance
        ptx.AppendLine("    mul.wide.u32 %rd7, %r4, 4;");
        ptx.AppendLine("    add.u64 %rd8, %rd2, %rd7;");
        ptx.AppendLine("    st.global.f32 [%rd8], %f26;");
        ptx.AppendLine("PD_END:");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static void EmitWarpReduce(StringBuilder ptx, string partial, string dst)
    {
        foreach (int offset in new[] { 16, 8, 4, 2, 1 })
        {
            ptx.AppendLine($"    mov.b32 %r8, {partial};");
            ptx.AppendLine($"    shfl.sync.down.b32 %r9, %r8, {offset}, 31, 0xffffffff;");
            ptx.AppendLine("    mov.b32 %f15, %r9;");
            ptx.AppendLine($"    add.rn.f32 {partial}, {partial}, %f15;");
        }
        ptx.AppendLine($"    mov.b32 %r8, {partial};");
        ptx.AppendLine("    shfl.sync.idx.b32 %r9, %r8, 0, 31, 0xffffffff;");
        ptx.AppendLine($"    mov.b32 {dst}, %r9;");
    }

    private static DirectPtxKernelBlueprint CreateBlueprint(DirectPtxArchitectureFamily architecture, int batch, int dim)
    {
        var vectors = new DirectPtxExtent(batch, dim);
        var scalars = new DirectPtxExtent(batch);
        return new DirectPtxKernelBlueprint(
            Operation: "poincare-distance",
            Version: 2,
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
            ResourceBudget: DirectPtxResourceBudget.FromDriverMeasurement(
                measuredRegistersPerThread: 24,
                maxStaticSharedBytes: 0,
                maxLocalBytesPerThread: 0,
                minBlocksPerMultiprocessor: 1),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["formula"] = "d = (1/sqrt(c)) * ln((1+arg)/(1-arg)), arg = min(sqrt(c)*||(-x)(+)y||, 1)",
                ["reduction"] = "four-warp-resident-shuffle-reductions-per-block",
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

}
