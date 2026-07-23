using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Product-quantization ADC (asymmetric distance computation) scan (issue #854), matching the NVRTC
/// <c>ann_pq_adc_scan</c> kernel: <c>distances[q,i] = sum_s tables[q, s, codes[i,s]]</c>, i.e. for each
/// (query, coded vector) sum one precomputed table lookup per subspace, indexed by the uint8 PQ code.
/// One thread owns one (query, code) cell and walks the <c>m</c> subspaces serially in registers,
/// gathering from the query's distance table — no shared memory, no reduction.
///
/// Shape (numQueries, numCodes, m, ksub) is baked into the PTX, so the launch takes buffer pointers
/// only. 256 threads/block, grid = (numQueries*numCodes)/256, required to divide evenly. Codes are a
/// uint8 buffer laid out [code][m].
/// </summary>
internal sealed class PtxAnnPqAdcScanKernel : IDisposable
{
    internal const int BlockThreads = 256;
    internal const int MaxCells = 2048 * 4096;
    internal const int MaxSubspaces = 4096;
    internal const string EntryPoint = "aidotnet_ann_pq_adc_scan";

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int NumQueries { get; }
    internal int NumCodes { get; }
    internal int M { get; }
    internal int Ksub { get; }
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxAnnPqAdcScanKernel(DirectPtxRuntime runtime, int numQueries, int numCodes, int m, int ksub)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasValidatedScientific(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The checked-in ann-pq-adc-scan specialization is measured only on GA10x/SM86.");
        ValidateShape(numQueries, numCodes, m, ksub);
        NumQueries = numQueries;
        NumCodes = numCodes;
        M = m;
        Ksub = ksub;
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, numQueries, numCodes, m, ksub);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor, numQueries, numCodes, m, ksub);
        _module = runtime.LoadModule(Ptx);
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo info);
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, BlockThreads);
        Blueprint.ResourceBudget.Validate(EntryPoint, info, BlockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Ptx, info, BlockThreads, activeBlocks, _module.JitInfoLog);
    }

    internal unsafe void Launch(DirectPtxTensorView codes, DirectPtxTensorView tables, DirectPtxTensorView distances)
    {
        Require(codes, Blueprint.Tensors[0], nameof(codes));
        Require(tables, Blueprint.Tensors[1], nameof(tables));
        Require(distances, Blueprint.Tensors[2], nameof(distances));

        IntPtr codesPointer = codes.Pointer;
        IntPtr tablesPointer = tables.Pointer;
        IntPtr distancesPointer = distances.Pointer;
        void** arguments = stackalloc void*[3];
        arguments[0] = &codesPointer;
        arguments[1] = &tablesPointer;
        arguments[2] = &distancesPointer;
        uint grid = (uint)((NumQueries * NumCodes) / BlockThreads);
        _module.Launch(_function, grid, 1, 1, BlockThreads, 1, 1, 0, arguments);
    }

    public void Dispose() => _module.Dispose();

    internal static string EmitPtx(int ccMajor, int ccMinor, int numQueries, int numCodes, int m, int ksub)
    {
        ValidateShape(numQueries, numCodes, m, ksub);
        int tblQStride = m * ksub;   // per-query table block

        var ptx = new StringBuilder(3_500);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine($"// ann-pq-adc-scan q={numQueries} codes={numCodes} m={m} ksub={ksub}");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 codes_ptr,");
        ptx.AppendLine("    .param .u64 tbl_ptr,");
        ptx.AppendLine("    .param .u64 dist_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine($".maxntid {BlockThreads}, 1, 1");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<2>;");
        ptx.AppendLine("    .reg .b32 %r<14>;");
        ptx.AppendLine("    .reg .b64 %rd<16>;");
        ptx.AppendLine("    .reg .f32 %f<4>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [codes_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [tbl_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [dist_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {BlockThreads}, %r0;");   // gid
        ptx.AppendLine($"    div.u32 %r3, %r2, {numCodes};");             // q
        ptx.AppendLine($"    rem.u32 %r4, %r2, {numCodes};");             // i
        // tblPtr = tables + q*(m*ksub) elems ; codePtr = codes + i*m bytes
        ptx.AppendLine($"    mul.lo.u32 %r5, %r3, {tblQStride};");
        ptx.AppendLine("    mul.wide.u32 %rd3, %r5, 4;");
        ptx.AppendLine("    add.u64 %rd6, %rd1, %rd3;");                   // tblPtr (s=0)
        ptx.AppendLine($"    mul.wide.u32 %rd4, %r4, {m};");             // i*m bytes
        ptx.AppendLine("    add.u64 %rd7, %rd0, %rd4;");                   // codePtr (s=0)
        ptx.AppendLine("    mov.f32 %f0, 0f00000000;");                   // sum
        ptx.AppendLine("    mov.u32 %r6, 0;");                            // s = 0
        ptx.AppendLine("$ADC_S_LOOP:");
        ptx.AppendLine("    ld.global.nc.u8 %r7, [%rd7];");             // code (zero-extended)
        ptx.AppendLine("    mul.wide.u32 %rd8, %r7, 4;");               // code*4 bytes
        ptx.AppendLine("    add.u64 %rd9, %rd6, %rd8;");               // &tables[q, s, code]
        ptx.AppendLine("    ld.global.nc.f32 %f1, [%rd9];");
        ptx.AppendLine("    add.rn.f32 %f0, %f0, %f1;");
        ptx.AppendLine($"    add.u64 %rd6, %rd6, {ksub * 4};");        // next subspace block (+ksub elems)
        ptx.AppendLine("    add.u64 %rd7, %rd7, 1;");                   // next code byte
        ptx.AppendLine("    add.u32 %r6, %r6, 1;");
        ptx.AppendLine($"    setp.lt.u32 %p0, %r6, {m};");
        ptx.AppendLine("    @%p0 bra $ADC_S_LOOP;");
        ptx.AppendLine("    mul.wide.u32 %rd10, %r2, 4;");
        ptx.AppendLine("    add.u64 %rd11, %rd2, %rd10;");
        ptx.AppendLine("    st.global.f32 [%rd11], %f0;");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture, int numQueries, int numCodes, int m, int ksub)
    {
        var codesExtent = new DirectPtxExtent(numCodes * m);
        var tblExtent = new DirectPtxExtent(numQueries * m * ksub);
        var distExtent = new DirectPtxExtent(numQueries * numCodes);
        return new DirectPtxKernelBlueprint(
            Operation: "ann-pq-adc-scan",
            Version: 1,
            Architecture: architecture,
            Variant: $"fp32-q{numQueries}-codes{numCodes}-m{m}-k{ksub}",
            Tensors:
            [
                new("codes", DirectPtxPhysicalType.UInt8, DirectPtxPhysicalLayout.Vector,
                    codesExtent, codesExtent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("tables", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    tblExtent, tblExtent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
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
                ["formula"] = "distances[q,i] = sum_s tables[q, s, codes[i,s]]",
                ["codes-layout"] = "uint8 [code][m]",
                ["global-intermediates"] = "none",
                ["temporary-device-allocation"] = "none",
                ["stride-parameters"] = "none"
            });
    }

    internal static bool IsSupportedShape(int numQueries, int numCodes, int m, int ksub)
    {
        if (numQueries <= 0 || numCodes <= 0 || m <= 0 || m > MaxSubspaces || ksub <= 0 || ksub > 256) return false;
        long cells = (long)numQueries * numCodes;
        return cells > 0 && cells % BlockThreads == 0 && cells <= MaxCells;
    }

    internal static bool IsPromotedShape(int numQueries, int numCodes, int m, int ksub) => false;

    private static void ValidateShape(int numQueries, int numCodes, int m, int ksub)
    {
        if (!IsSupportedShape(numQueries, numCodes, m, ksub))
            throw new ArgumentOutOfRangeException(
                nameof(numQueries),
                $"ANN PQ ADC scan requires positive dims with m<={MaxSubspaces}, ksub<=256 (uint8 codes), and (numQueries*numCodes) a multiple of {BlockThreads} up to {MaxCells}.");
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
