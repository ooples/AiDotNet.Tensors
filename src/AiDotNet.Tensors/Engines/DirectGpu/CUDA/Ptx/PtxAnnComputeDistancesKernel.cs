using System;
using System.Collections.Generic;
using System.Text;
using AiDotNet.Tensors.Engines.DirectGpu;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Dense query×database ANN distance matrix (issue #854), matching the NVRTC
/// <c>ann_compute_distances</c> kernel: <c>distances[q,j] = metric(query_q, db_j)</c> where the metric
/// is squared-L2 (<see cref="AnnMetric.L2"/>) or inner product (<see cref="AnnMetric.InnerProduct"/>).
/// One thread owns one (query, db) cell and walks the feature axis serially in registers — no shared
/// memory, no reduction. The metric is baked into the PTX.
///
/// Shape (numQueries, numDatabase, dim) and the metric are baked in, so the launch takes buffer
/// pointers only. 256 threads/block, grid = (numQueries*numDatabase)/256, required to divide evenly
/// (no divergent bounds guard).
/// </summary>
internal sealed class PtxAnnComputeDistancesKernel : IDisposable
{
    internal const int BlockThreads = 256;
    internal const int MaxCells = 2048 * 4096;
    internal const int MaxDim = 4096;
    internal const string EntryPoint = "aidotnet_ann_compute_distances";

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal AnnMetric Metric { get; }
    internal int NumQueries { get; }
    internal int NumDatabase { get; }
    internal int Dim { get; }
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxAnnComputeDistancesKernel(DirectPtxRuntime runtime, AnnMetric metric, int numQueries, int numDatabase, int dim)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasValidatedScientific(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The checked-in ann-compute-distances specialization is measured only on GA10x/SM86.");
        ValidateShape(numQueries, numDatabase, dim);
        Metric = metric;
        NumQueries = numQueries;
        NumDatabase = numDatabase;
        Dim = dim;
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, metric, numQueries, numDatabase, dim);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor, metric, numQueries, numDatabase, dim);
        _module = runtime.LoadModule(Ptx);
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo info);
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, BlockThreads);
        Blueprint.ResourceBudget.Validate(EntryPoint, info, BlockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Ptx, info, BlockThreads, activeBlocks, _module.JitInfoLog);
    }

    internal unsafe void Launch(DirectPtxTensorView queries, DirectPtxTensorView database, DirectPtxTensorView distances)
    {
        Require(queries, Blueprint.Tensors[0], nameof(queries));
        Require(database, Blueprint.Tensors[1], nameof(database));
        Require(distances, Blueprint.Tensors[2], nameof(distances));

        IntPtr queriesPointer = queries.Pointer;
        IntPtr databasePointer = database.Pointer;
        IntPtr distancesPointer = distances.Pointer;
        void** arguments = stackalloc void*[3];
        arguments[0] = &queriesPointer;
        arguments[1] = &databasePointer;
        arguments[2] = &distancesPointer;
        uint grid = (uint)((NumQueries * NumDatabase) / BlockThreads);
        _module.Launch(_function, grid, 1, 1, BlockThreads, 1, 1, 0, arguments);
    }

    public void Dispose() => _module.Dispose();

    // Appends the serial metric accumulation loop over `len` elements, walking `%rd6` (a) and `%rd7` (b),
    // accumulating into `%f0`. Shared by the ANN distance-cell kernels.
    internal static void AppendMetricLoop(StringBuilder ptx, AnnMetric metric, int len, string loopLabel)
    {
        ptx.AppendLine("    mov.f32 %f0, 0f00000000;");
        ptx.AppendLine("    mov.u32 %r9, 0;");
        ptx.AppendLine($"{loopLabel}:");
        ptx.AppendLine("    ld.global.nc.f32 %f1, [%rd6];");
        ptx.AppendLine("    ld.global.nc.f32 %f2, [%rd7];");
        if (metric == AnnMetric.InnerProduct)
        {
            ptx.AppendLine("    fma.rn.f32 %f0, %f1, %f2, %f0;");            // sum += a*b
        }
        else
        {
            ptx.AppendLine("    sub.rn.f32 %f3, %f1, %f2;");                 // d = a-b
            ptx.AppendLine("    fma.rn.f32 %f0, %f3, %f3, %f0;");            // sum += d*d
        }
        ptx.AppendLine("    add.u64 %rd6, %rd6, 4;");
        ptx.AppendLine("    add.u64 %rd7, %rd7, 4;");
        ptx.AppendLine("    add.u32 %r9, %r9, 1;");
        ptx.AppendLine($"    setp.lt.u32 %p0, %r9, {len};");
        ptx.AppendLine($"    @%p0 bra {loopLabel};");
    }

    internal static string EmitPtx(int ccMajor, int ccMinor, AnnMetric metric, int numQueries, int numDatabase, int dim)
    {
        ValidateShape(numQueries, numDatabase, dim);

        var ptx = new StringBuilder(3_500);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine($"// ann-compute-distances metric={metric} q={numQueries} db={numDatabase} dim={dim}");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 q_ptr,");
        ptx.AppendLine("    .param .u64 db_ptr,");
        ptx.AppendLine("    .param .u64 dist_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine($".maxntid {BlockThreads}, 1, 1");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<2>;");
        ptx.AppendLine("    .reg .b32 %r<12>;");
        ptx.AppendLine("    .reg .b64 %rd<14>;");
        ptx.AppendLine("    .reg .f32 %f<4>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [q_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [db_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [dist_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {BlockThreads}, %r0;");   // gid
        ptx.AppendLine($"    div.u32 %r3, %r2, {numDatabase};");           // q
        ptx.AppendLine($"    rem.u32 %r4, %r2, {numDatabase};");           // j
        ptx.AppendLine($"    mul.lo.u32 %r5, %r3, {dim};");               // q*dim
        ptx.AppendLine($"    mul.lo.u32 %r6, %r4, {dim};");               // j*dim
        ptx.AppendLine("    mul.wide.u32 %rd4, %r5, 4;");
        ptx.AppendLine("    mul.wide.u32 %rd5, %r6, 4;");
        ptx.AppendLine("    add.u64 %rd6, %rd0, %rd4;");                   // &queries[q,0]
        ptx.AppendLine("    add.u64 %rd7, %rd1, %rd5;");                   // &database[j,0]
        AppendMetricLoop(ptx, metric, dim, "$ANN_CD_LOOP");
        ptx.AppendLine("    mul.wide.u32 %rd8, %r2, 4;");
        ptx.AppendLine("    add.u64 %rd9, %rd2, %rd8;");
        ptx.AppendLine("    st.global.f32 [%rd9], %f0;");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture, AnnMetric metric, int numQueries, int numDatabase, int dim)
    {
        var qExtent = new DirectPtxExtent(numQueries * dim);
        var dbExtent = new DirectPtxExtent(numDatabase * dim);
        var distExtent = new DirectPtxExtent(numQueries * numDatabase);
        return new DirectPtxKernelBlueprint(
            Operation: "ann-compute-distances",
            Version: 1,
            Architecture: architecture,
            Variant: $"fp32-{metric}-q{numQueries}-db{numDatabase}-d{dim}",
            Tensors:
            [
                new("queries", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    qExtent, qExtent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("database", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    dbExtent, dbExtent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("distances", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    distExtent, distExtent, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 20,
                MaxStaticSharedBytes: 0,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 1),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["formula"] = metric == AnnMetric.InnerProduct
                    ? "distances[q,j] = sum_k queries[q,k] * database[j,k]"
                    : "distances[q,j] = sum_k (queries[q,k] - database[j,k])^2",
                ["metric"] = metric.ToString(),
                ["global-intermediates"] = "none",
                ["temporary-device-allocation"] = "none",
                ["stride-parameters"] = "none"
            });
    }

    internal static bool IsSupportedShape(int numQueries, int numDatabase, int dim)
    {
        if (numQueries <= 0 || numDatabase <= 0 || dim <= 0 || dim > MaxDim) return false;
        long cells = (long)numQueries * numDatabase;
        return cells > 0 && cells % BlockThreads == 0 && cells <= MaxCells;
    }

    internal static bool IsPromotedShape(int numQueries, int numDatabase, int dim) => false;

    private static void ValidateShape(int numQueries, int numDatabase, int dim)
    {
        if (!IsSupportedShape(numQueries, numDatabase, dim))
            throw new ArgumentOutOfRangeException(
                nameof(numQueries),
                $"ANN compute distances requires positive dims with dim<={MaxDim} and (numQueries*numDatabase) a multiple of {BlockThreads} up to {MaxCells}.");
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
