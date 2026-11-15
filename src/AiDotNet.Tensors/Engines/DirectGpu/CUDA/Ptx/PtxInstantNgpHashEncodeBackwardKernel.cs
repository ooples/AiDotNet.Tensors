using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Instant-NGP hash-grid encoding backward w.r.t. the hash table (issue #854), matching the NVRTC
/// <c>instant_ngp_hash_encode_level_backward</c> kernel: for each (table entry, feature), accumulate
/// the incoming gradient scaled by the trilinear weight over every point whose voxel corner indexes this entry —
/// <c>tableGradient[entry,f] = sum_n sum_corner [index(corner_n) == entry] * grad_n * w_corner</c>.
/// One thread owns one (entry, feature) output and loops over all points serially in registers,
/// recomputing the 8 corner indices (tiny-cuda-nn's grid_index: dense on a level that fits the table, the coherent
/// prime hash otherwise) and interpolation weights and adding under a match predicate — no shared memory, no
/// reduction, no scatter/atomics.
///
/// Shape (numPoints, scale, positionOffset, gridResolution, tableSize, featuresPerLevel, levelOffset, outputStride)
/// is baked into the PTX, so the launch takes buffer pointers only. 256 threads/block,
/// grid = (tableSize*featuresPerLevel)/256, required to divide evenly (no divergent bounds guard).
/// </summary>
internal sealed class PtxInstantNgpHashEncodeBackwardKernel : IDisposable
{
    internal const int BlockThreads = 256;
    internal const int MaxCells = 2048 * 4096;
    internal const string EntryPoint = "aidotnet_instant_ngp_hash_encode_level_backward";

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int NumPoints { get; }
    internal float Scale { get; }
    internal float PositionOffset { get; }
    internal int GridResolution { get; }
    internal int TableSize { get; }
    internal int FeaturesPerLevel { get; }
    internal int LevelOffset { get; }
    internal int OutputStride { get; }
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxInstantNgpHashEncodeBackwardKernel(
        DirectPtxRuntime runtime, int numPoints, float scale, float positionOffset, int gridResolution, int tableSize,
        int featuresPerLevel, int levelOffset, int outputStride)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasValidatedScientific(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The checked-in instant-ngp-hash-encode-backward specialization is measured only on GA10x/SM86.");
        ValidateShape(numPoints, scale, positionOffset, gridResolution, tableSize, featuresPerLevel, levelOffset, outputStride);
        NumPoints = numPoints;
        Scale = scale;
        PositionOffset = positionOffset;
        GridResolution = gridResolution;
        TableSize = tableSize;
        FeaturesPerLevel = featuresPerLevel;
        LevelOffset = levelOffset;
        OutputStride = outputStride;
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, numPoints, scale, positionOffset, gridResolution, tableSize, featuresPerLevel, levelOffset, outputStride);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor,
            numPoints, scale, positionOffset, gridResolution, tableSize, featuresPerLevel, levelOffset, outputStride);
        var loaded = DirectPtxResourceInitialization.Complete(
            runtime.LoadModule(Ptx), module => InitializeModule(runtime, module));
        _module = loaded.Resource;
        _function = loaded.Value.Function;
        Audit = loaded.Value.Audit;

        (IntPtr Function, DirectPtxKernelAudit Audit) InitializeModule(
            DirectPtxRuntime owner, DirectPtxModule module)
        {
            IntPtr function = module.GetFunction(EntryPoint, out DirectPtxFunctionInfo info);
            int activeBlocks = module.GetActiveBlocksPerMultiprocessor(function, BlockThreads);
            Blueprint.ResourceBudget.Validate(EntryPoint, info, BlockThreads, activeBlocks);
            var audit = DirectPtxKernelAudit.Create(
                Blueprint, owner.DeviceFingerprint, Ptx, info, BlockThreads, activeBlocks, module);
            return (function, audit);
        }
    }

    internal unsafe void Launch(DirectPtxTensorView positions, DirectPtxTensorView outputGradient, DirectPtxTensorView tableGradient)
    {
        DirectPtxAbi.Require(positions, Blueprint.Tensors[0], nameof(positions));
        DirectPtxAbi.Require(outputGradient, Blueprint.Tensors[1], nameof(outputGradient));
        DirectPtxAbi.Require(tableGradient, Blueprint.Tensors[2], nameof(tableGradient));

        IntPtr positionsPointer = positions.Pointer;
        IntPtr outputGradientPointer = outputGradient.Pointer;
        IntPtr tableGradientPointer = tableGradient.Pointer;
        void** arguments = stackalloc void*[3];
        arguments[0] = &positionsPointer;
        arguments[1] = &outputGradientPointer;
        arguments[2] = &tableGradientPointer;
        uint grid = (uint)((TableSize * FeaturesPerLevel) / BlockThreads);
        _module.Launch(_function, grid, 1, 1, BlockThreads, 1, 1, 0, arguments);
    }

    public void Dispose() => _module.Dispose();

    // acc(%f16) += [index(coords)==entry(%r3)] * grad(%f21) * (xyProd * zReg)
    private static void AppendBackwardCorner(
        StringBuilder ptx, string xReg, string yReg, string zReg, string xyProd, string zReg2, int gridResolution, int tableSize)
    {
        PtxInstantNgpHashEncodeKernel.AppendGridIndex(ptx, xReg, yReg, zReg, gridResolution, tableSize);
        ptx.AppendLine($"    mul.rn.f32 %f18, {xyProd}, {zReg2};");
        ptx.AppendLine("    setp.eq.u32 %p1, %r14, %r3;");
        ptx.AppendLine("    @%p1 fma.rn.f32 %f16, %f21, %f18, %f16;");
    }

    internal static string EmitPtx(
        int ccMajor, int ccMinor, int numPoints, float scale, float positionOffset, int gridResolution, int tableSize,
        int featuresPerLevel, int levelOffset, int outputStride)
    {
        ValidateShape(numPoints, scale, positionOffset, gridResolution, tableSize, featuresPerLevel, levelOffset, outputStride);
        string sc = DirectPtxPtxText.Hex(scale), off = DirectPtxPtxText.Hex(positionOffset);
        string clampHi = DirectPtxPtxText.Hex(InstantNGPOperations.PositionClampMax);
        const string one = "0f3F800000", zero = "0f00000000";
        bool dense = InstantNGPOperations.UsesDenseHashGridIndex(gridResolution, tableSize);

        var ptx = new StringBuilder(7_000);
        PtxIsa.AppendVersionAndTarget(ptx, ccMajor, ccMinor);
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine($"// instant-ngp-hash-encode-backward points={numPoints} scale={sc} offset={off} res={gridResolution} {(dense ? "dense" : "hashed")} table={tableSize} fpl={featuresPerLevel} loff={levelOffset} ostride={outputStride}");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 pos_ptr,");
        ptx.AppendLine("    .param .u64 grad_ptr,");
        ptx.AppendLine("    .param .u64 tgrad_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine($".maxntid {BlockThreads}, 1, 1");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<3>;");
        ptx.AppendLine("    .reg .b32 %r<24>;");
        ptx.AppendLine("    .reg .b64 %rd<16>;");
        ptx.AppendLine("    .reg .f32 %f<24>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [pos_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [grad_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [tgrad_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {BlockThreads}, %r0;");   // gid
        ptx.AppendLine($"    div.u32 %r3, %r2, {featuresPerLevel};");      // entry
        ptx.AppendLine($"    rem.u32 %r4, %r2, {featuresPerLevel};");      // f
        ptx.AppendLine($"    mov.f32 %f16, {zero};");                     // acc
        ptx.AppendLine("    mov.u32 %r5, 0;");                            // n = 0
        ptx.AppendLine("$NGPB_N_LOOP:");
        // grad = outputGradient[n*outputStride + levelOffset + f]; a zero gradient adds nothing, so skip it
        ptx.AppendLine($"    mad.lo.u32 %r7, %r5, {outputStride}, %r4;");
        ptx.AppendLine($"    add.u32 %r7, %r7, {levelOffset};");
        ptx.AppendLine("    mul.wide.u32 %rd5, %r7, 4;");
        ptx.AppendLine("    add.u64 %rd6, %rd1, %rd5;");
        ptx.AppendLine("    ld.global.nc.f32 %f21, [%rd6];");            // grad
        ptx.AppendLine($"    setp.eq.f32 %p0, %f21, {zero};");
        ptx.AppendLine("    @%p0 bra $NGPB_CONTINUE;");
        // positions[n*3]
        ptx.AppendLine("    mul.lo.u32 %r6, %r5, 3;");
        ptx.AppendLine("    mul.wide.u32 %rd3, %r6, 4;");
        ptx.AppendLine("    add.u64 %rd4, %rd0, %rd3;");
        ptx.AppendLine("    ld.global.nc.f32 %f0, [%rd4];");
        ptx.AppendLine("    ld.global.nc.f32 %f1, [%rd4+4];");
        ptx.AppendLine("    ld.global.nc.f32 %f2, [%rd4+8];");
        foreach ((string src, string g) in new[] { ("%f0", "%f3"), ("%f1", "%f4"), ("%f2", "%f5") })
        {
            ptx.AppendLine($"    max.f32 {src}, {src}, {zero};");
            ptx.AppendLine($"    min.f32 {src}, {src}, {clampHi};");
            ptx.AppendLine($"    fma.rn.f32 {g}, {sc}, {src}, {off};");
        }
        // coords + weights
        PtxInstantNgpHashEncodeKernel.EmitAxis(ptx, "%f3", "%r8", "%r9", "%f6", "%f9", one);     // x0=%r8 x1=%r9 fx=%f6 ix=%f9
        PtxInstantNgpHashEncodeKernel.EmitAxis(ptx, "%f4", "%r16", "%r17", "%f7", "%f10", one);  // y0=%r16 y1=%r17 fy=%f7 iy=%f10
        PtxInstantNgpHashEncodeKernel.EmitAxis(ptx, "%f5", "%r18", "%r19", "%f8", "%f11", one);  // z0=%r18 z1=%r19 fz=%f8 iz=%f11
        ptx.AppendLine("    mul.rn.f32 %f12, %f9, %f10;");   // ixiy
        ptx.AppendLine("    mul.rn.f32 %f13, %f9, %f7;");    // ixfy
        ptx.AppendLine("    mul.rn.f32 %f14, %f6, %f10;");   // fxiy
        ptx.AppendLine("    mul.rn.f32 %f15, %f6, %f7;");    // fxfy
        // 8 corners in tiny-cuda-nn's order (bit d of c = upper vertex on axis d, x fastest), as the NVRTC kernel and the
        // CPU, so two colliding corners of one point add into the entry in the same order.
        for (int c = 0; c < 8; c++)
        {
            bool upperX = (c & 1) != 0, upperY = (c & 2) != 0, upperZ = (c & 4) != 0;
            string xy = upperX ? (upperY ? "%f15" : "%f14") : (upperY ? "%f13" : "%f12");
            AppendBackwardCorner(ptx, upperX ? "%r9" : "%r8", upperY ? "%r17" : "%r16", upperZ ? "%r19" : "%r18", xy,
                upperZ ? "%f8" : "%f11", gridResolution, tableSize);
        }
        ptx.AppendLine("$NGPB_CONTINUE:");
        ptx.AppendLine("    add.u32 %r5, %r5, 1;");
        ptx.AppendLine($"    setp.lt.u32 %p2, %r5, {numPoints};");
        ptx.AppendLine("    @%p2 bra $NGPB_N_LOOP;");
        ptx.AppendLine("    mul.wide.u32 %rd7, %r2, 4;");
        ptx.AppendLine("    add.u64 %rd8, %rd2, %rd7;");
        ptx.AppendLine("    st.global.f32 [%rd8], %f16;");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture, int numPoints, float scale, float positionOffset, int gridResolution,
        int tableSize, int featuresPerLevel, int levelOffset, int outputStride)
    {
        var posExtent = new DirectPtxExtent(numPoints * 3);
        var gradExtent = new DirectPtxExtent(numPoints * outputStride);
        var tblGradExtent = new DirectPtxExtent(tableSize * featuresPerLevel);
        return new DirectPtxKernelBlueprint(
            Operation: "instant-ngp-hash-encode-backward",
            Version: 2,
            Architecture: architecture,
            Variant: $"fp32-p{numPoints}-s{DirectPtxPtxText.Hex(scale)}-o{DirectPtxPtxText.Hex(positionOffset)}-r{gridResolution}-t{tableSize}-f{featuresPerLevel}-lo{levelOffset}-os{outputStride}",
            Tensors:
            [
                new("positions", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    posExtent, posExtent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("outputGradient", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    gradExtent, gradExtent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("tableGradient", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    tblGradExtent, tblGradExtent, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 40,
                MaxStaticSharedBytes: 0,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 1),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["formula"] = "tableGradient[entry,f] = sum_n sum_corner [index(corner_n)==entry] * grad_n * w_corner",
                ["index"] = InstantNGPOperations.UsesDenseHashGridIndex(gridResolution, tableSize)
                    ? "(x + y*R + z*R*R) mod tableSize, dense (tiny-cuda-nn grid_index)"
                    : "(x ^ y*2654435761 ^ z*805459861) mod tableSize (tiny-cuda-nn coherent_prime_hash)",
                ["grad-skip"] = "points with a zero gradient skipped (exact), matching NVRTC",
                ["global-intermediates"] = "none",
                ["temporary-device-allocation"] = "none",
                ["stride-parameters"] = "levelOffset/outputStride baked (constant, not runtime)"
            });
    }

    internal static bool IsSupportedShape(
        int numPoints, float scale, float positionOffset, int gridResolution, int tableSize, int featuresPerLevel,
        int levelOffset, int outputStride)
    {
        if (numPoints <= 0 || gridResolution <= 0 || tableSize <= 0 || featuresPerLevel <= 0) return false;
        if (!PtxInstantNgpHashEncodeKernel.IsSupportedGrid(scale, positionOffset)) return false;
        if (levelOffset < 0 || outputStride <= 0 || levelOffset + featuresPerLevel > outputStride) return false;
        long cells = (long)tableSize * featuresPerLevel;
        return cells > 0 && cells % BlockThreads == 0 && cells <= MaxCells;
    }

    internal static bool IsPromotedShape(
        int numPoints, float scale, float positionOffset, int gridResolution, int tableSize, int featuresPerLevel,
        int levelOffset, int outputStride) => false;

    private static void ValidateShape(
        int numPoints, float scale, float positionOffset, int gridResolution, int tableSize, int featuresPerLevel,
        int levelOffset, int outputStride)
    {
        if (!IsSupportedShape(numPoints, scale, positionOffset, gridResolution, tableSize, featuresPerLevel, levelOffset, outputStride))
            throw new ArgumentOutOfRangeException(
                nameof(tableSize),
                $"Instant-NGP hash encode backward requires positive dims, a scale in [0, {InstantNGPOperations.MaxGridScale}], an offset in [0, 1), levelOffset+featuresPerLevel<=outputStride, and (tableSize*featuresPerLevel) a multiple of {BlockThreads} up to {MaxCells}.");
    }

}
