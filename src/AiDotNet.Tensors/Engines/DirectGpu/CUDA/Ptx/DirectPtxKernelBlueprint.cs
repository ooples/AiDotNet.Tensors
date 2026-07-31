using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>Hardware families are deliberately separate tuning domains.</summary>
internal enum DirectPtxArchitectureFamily
{
    Unsupported,
    Ampere,
    Ada,
    Hopper,
    Blackwell
}

internal static class DirectPtxArchitecture
{
    internal static DirectPtxArchitectureFamily Classify(int major, int minor) => (major, minor) switch
    {
        (8, 9) => DirectPtxArchitectureFamily.Ada,
        (8, _) => DirectPtxArchitectureFamily.Ampere,
        (9, _) => DirectPtxArchitectureFamily.Hopper,
        (>= 10, _) => DirectPtxArchitectureFamily.Blackwell,
        _ => DirectPtxArchitectureFamily.Unsupported
    };

    /// <summary>
    /// The checked-in asynchronous attention implementation is an Ampere
    /// specialization. Other families must supply and benchmark their own
    /// implementation instead of silently inheriting Ampere's tuning.
    /// </summary>
    internal static bool HasValidatedOnlineAttention(int major, int minor) =>
        (major, minor) == (8, 6);

    /// <summary>
    /// The first QKV/RoPE/cache specialization is measured and promoted only
    /// on GA102/SM86. Other Ampere variants remain independent tuning domains.
    /// </summary>
    internal static bool HasValidatedQkvRopeCache(int major, int minor) =>
        (major, minor) == (8, 6);

    /// <summary>
    /// The fused-linear + GELU decode specializations are measured and promoted
    /// only on GA10x/SM86. Other Ampere variants (SM80, SM87) are independent
    /// tuning domains and must supply and benchmark their own specialization
    /// rather than silently inheriting SM86's launch geometry.
    /// </summary>
    internal static bool HasValidatedFusedLinear(int major, int minor) =>
        (major, minor) == (8, 6);

    /// <summary>
    /// The mixed-precision (FP16 / W8A8) fused-linear decode specializations are
    /// measured and promoted only on GA10x/SM86. Other Ampere variants (SM80,
    /// SM87) are independent tuning domains and must supply and benchmark their
    /// own specialization rather than silently inheriting SM86's launch geometry.
    /// </summary>
    internal static bool HasValidatedMixedLinear(int major, int minor) =>
        (major, minor) == (8, 6);

    /// <summary>
    /// The fused residual + bias + LayerNorm + GELU decode specialization is
    /// measured and promoted only on GA10x/SM86. Other Ampere variants (SM80,
    /// SM87) are independent tuning domains and must supply and benchmark their
    /// own specialization rather than silently inheriting SM86's launch geometry.
    /// </summary>
    internal static bool HasValidatedResidualLayerNormGelu(int major, int minor) =>
        (major, minor) == (8, 6);

    /// <summary>
    /// The quantized (W8A8) decode-linear specialization is measured only on
    /// GA102/SM86, matching the other fused-linear predicates. Admitting the
    /// whole Ampere family would run PTX that was never validated on SM80/SM87.
    /// </summary>
    internal static bool HasValidatedQuantizedLinear(int major, int minor) =>
        (major, minor) == (8, 6);

    /// <summary>
    /// Issue #841's first convolution emitter deliberately targets one exact
    /// GA102/SM86 tuning domain. It remains experimental until GPU evidence is
    /// attached; other SMs must use the established backend.
    /// </summary>
    internal static bool HasExperimentalConvolution(int major, int minor) =>
        (major, minor) == (8, 6);
}

internal enum DirectPtxExtentMode
{
    AtLeast,
    Exact
}

[Flags]
internal enum DirectPtxTensorAccess
{
    Read = 1,
    Write = 2,
    ReadWrite = Read | Write
}

/// <summary>A fixed-size rank-four extent that does not allocate on hot paths.</summary>
internal readonly record struct DirectPtxExtent
{
    internal int Rank { get; }
    internal int D0 { get; }
    internal int D1 { get; }
    internal int D2 { get; }
    internal int D3 { get; }

    internal DirectPtxExtent(int d0)
        : this(1, d0, 1, 1, 1) { }

    internal DirectPtxExtent(int d0, int d1)
        : this(2, d0, d1, 1, 1) { }

    internal DirectPtxExtent(int d0, int d1, int d2)
        : this(3, d0, d1, d2, 1) { }

    internal DirectPtxExtent(int d0, int d1, int d2, int d3)
        : this(4, d0, d1, d2, d3) { }

    private DirectPtxExtent(int rank, int d0, int d1, int d2, int d3)
    {
        if (rank is < 1 or > 4) throw new ArgumentOutOfRangeException(nameof(rank));
        if (d0 <= 0 || d1 <= 0 || d2 <= 0 || d3 <= 0)
            throw new ArgumentOutOfRangeException(nameof(d0), "Physical extents must be positive.");
        Rank = rank;
        D0 = d0;
        D1 = d1;
        D2 = d2;
        D3 = d3;
    }

    internal long ElementCount => checked((long)D0 * D1 * D2 * D3);

    public override string ToString() => Rank switch
    {
        1 => $"[{D0}]",
        2 => $"[{D0},{D1}]",
        3 => $"[{D0},{D1},{D2}]",
        _ => $"[{D0},{D1},{D2},{D3}]"
    };
}

/// <summary>
/// Immutable physical ABI for one kernel argument. Logical and physical
/// extents are separate so later padded, packed, ragged, and paged layouts do
/// not need to reintroduce dynamic strides inside the kernel.
/// </summary>
internal readonly record struct DirectPtxTensorContract
{
    internal string Name { get; }
    internal DirectPtxPhysicalType PhysicalType { get; }
    internal DirectPtxPhysicalLayout Layout { get; }
    internal DirectPtxExtent LogicalExtent { get; }
    internal DirectPtxExtent PhysicalExtent { get; }
    internal int AlignmentBytes { get; }
    internal DirectPtxTensorAccess Access { get; }
    internal DirectPtxExtentMode ExtentMode { get; }

    internal DirectPtxTensorContract(
        string name,
        DirectPtxPhysicalType physicalType,
        DirectPtxPhysicalLayout layout,
        DirectPtxExtent logicalExtent,
        DirectPtxExtent physicalExtent,
        int alignmentBytes,
        DirectPtxTensorAccess access,
        DirectPtxExtentMode extentMode = DirectPtxExtentMode.AtLeast)
    {
        if (string.IsNullOrWhiteSpace(name)) throw new ArgumentException("A tensor ABI name is required.", nameof(name));
        if (alignmentBytes <= 0 || (alignmentBytes & (alignmentBytes - 1)) != 0)
            throw new ArgumentOutOfRangeException(nameof(alignmentBytes), "Alignment must be a power of two.");
        if (physicalExtent.ElementCount < logicalExtent.ElementCount)
            throw new ArgumentException("Physical extent cannot be smaller than logical extent.", nameof(physicalExtent));
        Name = name;
        PhysicalType = physicalType;
        Layout = layout;
        LogicalExtent = logicalExtent;
        PhysicalExtent = physicalExtent;
        AlignmentBytes = alignmentBytes;
        Access = access;
        ExtentMode = extentMode;
    }

    internal nuint RequiredBytes => checked((nuint)PhysicalExtent.ElementCount * (nuint)ElementBytes);
    internal int ElementBytes => PhysicalType switch
    {
        DirectPtxPhysicalType.Int8 => 1,
        DirectPtxPhysicalType.Float16 or DirectPtxPhysicalType.BFloat16 => 2,
        DirectPtxPhysicalType.Float32 => 4,
        DirectPtxPhysicalType.Int32 => 4,
        _ => throw new ArgumentOutOfRangeException(nameof(PhysicalType))
    };
}

internal readonly record struct DirectPtxResourceBudget(
    int MaxRegistersPerThread,
    int MaxStaticSharedBytes,
    int MaxLocalBytesPerThread,
    int MinBlocksPerMultiprocessor)
{
    internal void Validate(
        string kernelName,
        DirectPtxFunctionInfo info,
        int blockThreads,
        int activeBlocksPerMultiprocessor)
    {
        if (info.LocalBytesPerThread > MaxLocalBytesPerThread)
            throw new InvalidOperationException(
                $"Direct PTX kernel '{kernelName}' has {info.LocalBytesPerThread} local bytes/thread; budget is {MaxLocalBytesPerThread}.");
        if (info.RegistersPerThread > MaxRegistersPerThread)
            throw new InvalidOperationException(
                $"Direct PTX kernel '{kernelName}' uses {info.RegistersPerThread} registers/thread; budget is {MaxRegistersPerThread}.");
        if (info.StaticSharedBytes > MaxStaticSharedBytes)
            throw new InvalidOperationException(
                $"Direct PTX kernel '{kernelName}' uses {info.StaticSharedBytes} static shared bytes; budget is {MaxStaticSharedBytes}.");
        if (blockThreads > info.MaxThreadsPerBlock)
            throw new InvalidOperationException(
                $"Direct PTX kernel '{kernelName}' requests {blockThreads} threads but the JIT limit is {info.MaxThreadsPerBlock}.");
        if (activeBlocksPerMultiprocessor < MinBlocksPerMultiprocessor)
            throw new InvalidOperationException(
                $"Direct PTX kernel '{kernelName}' admits only {activeBlocksPerMultiprocessor} blocks/SM; budget requires {MinBlocksPerMultiprocessor}.");
    }
}

/// <summary>Versioned, inspectable contract shipped with every direct kernel.</summary>
internal sealed record DirectPtxKernelBlueprint(
    string Operation,
    int Version,
    DirectPtxArchitectureFamily Architecture,
    string Variant,
    IReadOnlyList<DirectPtxTensorContract> Tensors,
    DirectPtxResourceBudget ResourceBudget,
    IReadOnlyDictionary<string, string> Semantics)
{
    internal string Id => $"{Operation}-v{Version}-{Architecture}-{Variant}";
}

/// <summary>JIT resource report plus a content identity for the exact PTX.</summary>
internal sealed record DirectPtxKernelAudit(
    string BlueprintId,
    string DeviceFingerprint,
    string PtxSha256,
    DirectPtxFunctionInfo Function,
    int BlockThreads,
    int ActiveBlocksPerMultiprocessor,
    string JitInfoLog,
    DateTime RecordedAtUtc,
    DirectPtxModuleImageKind ImageKind = DirectPtxModuleImageKind.DriverLinkedCubin,
    string CubinSha256 = "",
    string CubinSourceKey = "",
    string? CubinPath = null)
{
    internal static DirectPtxKernelAudit Create(
        DirectPtxKernelBlueprint blueprint,
        string deviceFingerprint,
        string ptx,
        DirectPtxFunctionInfo function,
        int blockThreads,
        int activeBlocksPerMultiprocessor,
        string jitInfoLog)
    {
        string hash = DirectPtxCubinArtifactCache.ComputePtxSha256(ptx);
        return new DirectPtxKernelAudit(
            blueprint.Id, deviceFingerprint, hash, function, blockThreads,
            activeBlocksPerMultiprocessor, jitInfoLog, DateTime.UtcNow);
    }

    internal static DirectPtxKernelAudit Create(
        DirectPtxKernelBlueprint blueprint,
        string deviceFingerprint,
        string ptx,
        DirectPtxFunctionInfo function,
        int blockThreads,
        int activeBlocksPerMultiprocessor,
        DirectPtxModule module)
    {
        PtxCompat.ThrowIfNull(module, nameof(module));
        string hash = DirectPtxCubinArtifactCache.ComputePtxSha256(ptx);
        return new DirectPtxKernelAudit(
            blueprint.Id, deviceFingerprint, hash, function, blockThreads,
            activeBlocksPerMultiprocessor, module.JitInfoLog, DateTime.UtcNow,
            module.ImageKind, module.CubinSha256, module.CubinSourceKey,
            module.CubinPath);
    }

    internal string ToJson() => JsonSerializer.Serialize(this, new JsonSerializerOptions { WriteIndented = true });
}

/// <summary>
/// Optional Nsight Compute evidence. Driver-reported local size remains the
/// runtime gate; release evidence additionally requires executed spill and
/// local-memory counters to be zero.
/// </summary>
internal sealed record DirectPtxProfilerEvidence(
    long RegisterSpillInstructions,
    long LocalLoadInstructions,
    long LocalStoreInstructions,
    int ObservedMetricGroups,
    string Source)
{
    internal bool ProvesZeroExecutedSpills =>
        ObservedMetricGroups == 3 &&
        RegisterSpillInstructions == 0 && LocalLoadInstructions == 0 &&
        LocalStoreInstructions == 0;

    internal static DirectPtxProfilerEvidence FromNcuCsv(string path)
    {
        PtxCompat.ThrowIfNullOrWhiteSpace(path, nameof(path));
        var values = new Dictionary<string, long>(StringComparer.Ordinal);
        string[][] rows = File.ReadLines(path).Select(ParseCsvLine).ToArray();

        // Nsight Compute 2026.2 raw CSV is column-oriented: metric names are
        // headers and every following data row is a kernel launch. Preserve
        // support for the older row-oriented export below as well.
        for (int rowIndex = 0; rowIndex < rows.Length; rowIndex++)
        {
            string[] header = rows[rowIndex];
            int[] metricColumns = Enumerable.Range(0, header.Length)
                .Where(index => IsSpillMetric(header[index].Trim()))
                .ToArray();
            if (metricColumns.Length == 0) continue;

            int nextHeader = rowIndex + 1;
            while (nextHeader < rows.Length &&
                   !rows[nextHeader].Any(cell => IsSpillMetric(cell.Trim())))
            {
                string[] data = rows[nextHeader];
                foreach (int column in metricColumns)
                {
                    if (column >= data.Length || !TryParseCounter(data[column], out long value)) continue;
                    string metric = header[column].Trim();
                    values[metric] = checked(PtxCompat.GetValueOrDefault(values, metric) + value);
                }
                nextHeader++;
            }
            rowIndex = nextHeader - 1;
        }

        foreach (string[] cells in rows)
        {
            for (int i = 0; i < cells.Length; i++)
            {
                string metric = cells[i].Trim();
                if (!IsSpillMetric(metric)) continue;
                for (int valueIndex = cells.Length - 1; valueIndex > i; valueIndex--)
                {
                    if (TryParseCounter(cells[valueIndex], out long value))
                    {
                        values[metric] = checked(PtxCompat.GetValueOrDefault(values, metric) + value);
                        break;
                    }
                }
            }
        }

        static bool Matches(string metric, string baseName) =>
            string.Equals(metric, baseName, StringComparison.Ordinal) ||
            metric.StartsWith(baseName + ".", StringComparison.Ordinal);
        long Get(params string[] names) => values
            .Where(pair => names.Any(name => Matches(pair.Key, name)))
            .Sum(pair => pair.Value);
        bool Contains(string name) => values.Keys.Any(metric => Matches(metric, name));
        int observedGroups = 0;
        if (Contains("sass__inst_executed_register_spilling") ||
            Contains("sass__inst_executed_register_spilling_mem_local") ||
            Contains("smsp__sass_inst_executed_op_local")) observedGroups++;
        if (Contains("sass__inst_executed_local_loads") ||
            Contains("smsp__sass_inst_executed_op_local_ld")) observedGroups++;
        if (Contains("sass__inst_executed_local_stores") ||
            Contains("smsp__sass_inst_executed_op_local_st")) observedGroups++;
        return new DirectPtxProfilerEvidence(
            Get("sass__inst_executed_register_spilling", "sass__inst_executed_register_spilling_mem_local",
                "smsp__sass_inst_executed_op_local"),
            Get("sass__inst_executed_local_loads", "smsp__sass_inst_executed_op_local_ld"),
            Get("sass__inst_executed_local_stores", "smsp__sass_inst_executed_op_local_st"),
            observedGroups,
            Path.GetFullPath(path));
    }

    private static bool IsSpillMetric(string value) =>
        value == "sass__inst_executed_register_spilling" ||
        value.StartsWith("sass__inst_executed_register_spilling.", StringComparison.Ordinal) ||
        value == "sass__inst_executed_register_spilling_mem_local" ||
        value.StartsWith("sass__inst_executed_register_spilling_mem_local.", StringComparison.Ordinal) ||
        value == "sass__inst_executed_local_loads" ||
        value.StartsWith("sass__inst_executed_local_loads.", StringComparison.Ordinal) ||
        value == "sass__inst_executed_local_stores" ||
        value.StartsWith("sass__inst_executed_local_stores.", StringComparison.Ordinal) ||
        value == "smsp__sass_inst_executed_op_local" ||
        value.StartsWith("smsp__sass_inst_executed_op_local.", StringComparison.Ordinal) ||
        value == "smsp__sass_inst_executed_op_local_ld" ||
        value.StartsWith("smsp__sass_inst_executed_op_local_ld.", StringComparison.Ordinal) ||
        value == "smsp__sass_inst_executed_op_local_st" ||
        value.StartsWith("smsp__sass_inst_executed_op_local_st.", StringComparison.Ordinal);

    private static bool TryParseCounter(string value, out long result)
    {
        string normalized = value.Trim().Replace(",", string.Empty);
        return long.TryParse(normalized, NumberStyles.Integer, CultureInfo.InvariantCulture, out result);
    }

    private static string[] ParseCsvLine(string line)
    {
        var result = new List<string>();
        var value = new StringBuilder();
        bool quoted = false;
        foreach (char c in line)
        {
            if (c == '"') quoted = !quoted;
            else if (c == ',' && !quoted) { result.Add(value.ToString()); value.Clear(); }
            else value.Append(c);
        }
        result.Add(value.ToString());
        return result.ToArray();
    }
}
