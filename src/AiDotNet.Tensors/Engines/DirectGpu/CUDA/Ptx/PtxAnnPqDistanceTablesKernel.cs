using System;
using System.Collections.Generic;
using System.Text;
using AiDotNet.Tensors.Engines.DirectGpu;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Product-quantization asymmetric distance tables (issue #854), matching the NVRTC
/// <c>ann_pq_distance_tables</c> kernel: for each (query, subspace, sub-centroid),
/// <c>tables[q,s,c] = metric(query_subvec, codebook_subcentroid)</c> over the <c>dsub</c>-length
/// subvector. One thread owns each (q, s, c) cell and uses aligned vector loads when the baked
/// subdimension permits, reducing L1 transactions without multiplying the launched warp count.
///
/// Shape (numQueries, m, ksub, dsub) and the metric are baked in, so the launch takes buffer pointers
/// only. 256 threads/block, grid = (numQueries*m*ksub)/256, required to divide evenly.
/// </summary>
internal sealed class PtxAnnPqDistanceTablesKernel : IDisposable
{
    internal const int BlockThreads = 256;
    internal const int MaxCells = 2048 * 4096;
    internal const int MaxSubDim = 4096;
    internal const string EntryPoint = "aidotnet_ann_pq_distance_tables";

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal AnnMetric Metric { get; }
    internal int NumQueries { get; }
    internal int M { get; }
    internal int Ksub { get; }
    internal int Dsub { get; }
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxAnnPqDistanceTablesKernel(DirectPtxRuntime runtime, AnnMetric metric, int numQueries, int m, int ksub, int dsub)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasValidatedScientific(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The checked-in ann-pq-distance-tables specialization is measured only on GA10x/SM86.");
        ValidateShape(numQueries, m, ksub, dsub);
        Metric = metric;
        NumQueries = numQueries;
        M = m;
        Ksub = ksub;
        Dsub = dsub;
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, metric, numQueries, m, ksub, dsub);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor, metric, numQueries, m, ksub, dsub);
        _module = runtime.LoadModule(Ptx);
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo info);
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, BlockThreads);
        Blueprint.ResourceBudget.Validate(EntryPoint, info, BlockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Ptx, info, BlockThreads, activeBlocks, _module);
    }

    internal unsafe void Launch(DirectPtxTensorView queries, DirectPtxTensorView codebooks, DirectPtxTensorView tables)
    {
        DirectPtxAbi.Require(queries, Blueprint.Tensors[0], nameof(queries));
        DirectPtxAbi.Require(codebooks, Blueprint.Tensors[1], nameof(codebooks));
        DirectPtxAbi.Require(tables, Blueprint.Tensors[2], nameof(tables));

        IntPtr queriesPointer = queries.Pointer;
        IntPtr codebooksPointer = codebooks.Pointer;
        IntPtr tablesPointer = tables.Pointer;
        void** arguments = stackalloc void*[3];
        arguments[0] = &queriesPointer;
        arguments[1] = &codebooksPointer;
        arguments[2] = &tablesPointer;
        uint grid = (uint)((NumQueries * M * Ksub) / BlockThreads);
        _module.Launch(_function, grid, 1, 1, BlockThreads, 1, 1, 0, arguments);
    }

    public void Dispose() => _module.Dispose();

    internal static void AppendVectorizedMetricLoop(
        StringBuilder ptx, AnnMetric metric, int len, int vectorWidth)
    {
        if (vectorWidth is not (1 or 2 or 4) || len <= 0 || len % vectorWidth != 0)
            throw new ArgumentOutOfRangeException(nameof(vectorWidth));

        ptx.AppendLine("    mov.f32 %f0, 0f00000000;");
        ptx.AppendLine("    mov.u32 %r9, 0;");
        ptx.AppendLine("$ANN_PQ_LOOP:");
        if (vectorWidth == 4)
        {
            ptx.AppendLine("    ld.global.nc.v4.f32 {%f1, %f2, %f3, %f4}, [%rd6];");
            ptx.AppendLine("    ld.global.nc.v4.f32 {%f5, %f6, %f7, %f8}, [%rd7];");
            for (int lane = 0; lane < 4; lane++)
                AppendMetricTerm(ptx, metric, $"%f{lane + 1}", $"%f{lane + 5}", "%f9");
        }
        else if (vectorWidth == 2)
        {
            ptx.AppendLine("    ld.global.nc.v2.f32 {%f1, %f2}, [%rd6];");
            ptx.AppendLine("    ld.global.nc.v2.f32 {%f3, %f4}, [%rd7];");
            for (int lane = 0; lane < 2; lane++)
                AppendMetricTerm(ptx, metric, $"%f{lane + 1}", $"%f{lane + 3}", "%f5");
        }
        else
        {
            ptx.AppendLine("    ld.global.nc.f32 %f1, [%rd6];");
            ptx.AppendLine("    ld.global.nc.f32 %f2, [%rd7];");
            AppendMetricTerm(ptx, metric, "%f1", "%f2", "%f3");
        }
        int byteStride = vectorWidth * sizeof(float);
        ptx.AppendLine($"    add.u64 %rd6, %rd6, {byteStride};");
        ptx.AppendLine($"    add.u64 %rd7, %rd7, {byteStride};");
        ptx.AppendLine("    add.u32 %r9, %r9, 1;");
        ptx.AppendLine($"    setp.lt.u32 %p0, %r9, {len / vectorWidth};");
        ptx.AppendLine("    @%p0 bra $ANN_PQ_LOOP;");
    }

    private static void AppendMetricTerm(
        StringBuilder ptx, AnnMetric metric, string left, string right, string difference)
    {
        if (metric == AnnMetric.InnerProduct)
        {
            ptx.AppendLine($"    fma.rn.f32 %f0, {left}, {right}, %f0;");
        }
        else
        {
            ptx.AppendLine($"    sub.rn.f32 {difference}, {left}, {right};");
            ptx.AppendLine($"    fma.rn.f32 %f0, {difference}, {difference}, %f0;");
        }
    }

    internal static string EmitPtx(int ccMajor, int ccMinor, AnnMetric metric, int numQueries, int m, int ksub, int dsub)
    {
        ValidateShape(numQueries, m, ksub, dsub);
        int queryStrideQ = m * dsub;            // query [q, m*dsub]
        int cbStrideS = ksub * dsub;            // codebook [s, ksub*dsub]
        int vectorWidth = dsub % 4 == 0 ? 4 : dsub % 2 == 0 ? 2 : 1;

        var ptx = new StringBuilder(3_500);
        DirectPtxPtxText.AppendModuleHeader(ptx, ccMajor, ccMinor, disableLoopUnrolling: true);
        ptx.AppendLine($"// ann-pq-distance-tables metric={metric} q={numQueries} m={m} ksub={ksub} dsub={dsub}; vector-width={vectorWidth}");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 q_ptr,");
        ptx.AppendLine("    .param .u64 cb_ptr,");
        ptx.AppendLine("    .param .u64 tbl_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine($".maxntid {BlockThreads}, 1, 1");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<2>;");
        ptx.AppendLine("    .reg .b32 %r<16>;");
        ptx.AppendLine("    .reg .b64 %rd<14>;");
        ptx.AppendLine("    .reg .f32 %f<10>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [q_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [cb_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [tbl_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {BlockThreads}, %r0;");   // cell
        ptx.AppendLine($"    rem.u32 %r3, %r2, {ksub};");                  // c
        ptx.AppendLine($"    div.u32 %r4, %r2, {ksub};");                  // tmp
        ptx.AppendLine($"    rem.u32 %r5, %r4, {m};");                     // s
        ptx.AppendLine($"    div.u32 %r6, %r4, {m};");                     // q
        // qSubOff = q*(m*dsub) + s*dsub
        ptx.AppendLine($"    mul.lo.u32 %r7, %r6, {queryStrideQ};");
        ptx.AppendLine($"    mad.lo.u32 %r7, %r5, {dsub}, %r7;");
        ptx.AppendLine("    mul.wide.u32 %rd4, %r7, 4;");
        ptx.AppendLine("    add.u64 %rd6, %rd0, %rd4;");                   // &queries[qSubOff]
        // cbOff = s*(ksub*dsub) + c*dsub
        ptx.AppendLine($"    mul.lo.u32 %r8, %r5, {cbStrideS};");
        ptx.AppendLine($"    mad.lo.u32 %r8, %r3, {dsub}, %r8;");
        ptx.AppendLine("    mul.wide.u32 %rd5, %r8, 4;");
        ptx.AppendLine("    add.u64 %rd7, %rd1, %rd5;");                   // &codebooks[cbOff]
        AppendVectorizedMetricLoop(ptx, metric, dsub, vectorWidth);
        ptx.AppendLine("    mul.wide.u32 %rd8, %r2, 4;");
        ptx.AppendLine("    add.u64 %rd9, %rd2, %rd8;");
        ptx.AppendLine("    st.global.f32 [%rd9], %f0;");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture, AnnMetric metric, int numQueries, int m, int ksub, int dsub)
    {
        var qExtent = new DirectPtxExtent(numQueries * m * dsub);
        var cbExtent = new DirectPtxExtent(m * ksub * dsub);
        var tblExtent = new DirectPtxExtent(numQueries * m * ksub);
        return new DirectPtxKernelBlueprint(
            Operation: "ann-pq-distance-tables",
            Version: 2,
            Architecture: architecture,
            Variant: $"fp32-{metric}-q{numQueries}-m{m}-k{ksub}-d{dsub}",
            Tensors:
            [
                new("queries", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    qExtent, qExtent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("codebooks", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    cbExtent, cbExtent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("tables", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    tblExtent, tblExtent, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: DirectPtxResourceBudget.FromDriverMeasurement(
                measuredRegistersPerThread: 20,
                maxStaticSharedBytes: 0,
                maxLocalBytesPerThread: 0,
                minBlocksPerMultiprocessor: 1),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["formula"] = metric == AnnMetric.InnerProduct
                    ? "tables[q,s,c] = sum_k query[q,s,k] * codebook[s,c,k]"
                    : "tables[q,s,c] = sum_k (query[q,s,k] - codebook[s,c,k])^2",
                ["metric"] = metric.ToString(),
                ["global-intermediates"] = "none",
                ["temporary-device-allocation"] = "none",
                ["stride-parameters"] = "none"
            });
    }

    internal static bool IsSupportedShape(int numQueries, int m, int ksub, int dsub)
    {
        if (numQueries <= 0 || m <= 0 || ksub <= 0 || dsub <= 0 || dsub > MaxSubDim) return false;
        long cells = (long)numQueries * m * ksub;
        return cells > 0 && cells % BlockThreads == 0 && cells <= MaxCells;
    }

    internal static bool IsPromotedShape(int numQueries, int m, int ksub, int dsub) => false;

    private static void ValidateShape(int numQueries, int m, int ksub, int dsub)
    {
        if (!IsSupportedShape(numQueries, m, ksub, dsub))
            throw new ArgumentOutOfRangeException(
                nameof(numQueries),
                $"ANN PQ distance tables requires positive dims with dsub<={MaxSubDim} and (numQueries*m*ksub) a multiple of {BlockThreads} up to {MaxCells}.");
    }

}
