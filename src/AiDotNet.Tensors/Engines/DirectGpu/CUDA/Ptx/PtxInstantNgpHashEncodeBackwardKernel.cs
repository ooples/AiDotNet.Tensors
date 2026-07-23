using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Instant-NGP hash-grid encoding backward w.r.t. the hash table (issue #854), matching the NVRTC
/// <c>instant_ngp_hash_encode_level_backward</c> kernel: for each (table entry, feature), accumulate
/// the incoming gradient scaled by the trilinear weight over every point whose voxel corner hashes to
/// this entry — <c>tableGradient[entry,f] = sum_n sum_corner [hash(corner_n) == entry] * grad_n * w_corner</c>.
/// One thread owns one (entry, feature) output and loops over all points serially in registers,
/// recomputing the 8 spatial hashes and interpolation weights and adding under a match predicate — no
/// shared memory, no reduction, no scatter/atomics.
///
/// Shape (numPoints, resolution, tableSize, featuresPerLevel, levelOffset, outputStride) is baked into
/// the PTX, so the launch takes buffer pointers only. 256 threads/block,
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
    internal int Resolution { get; }
    internal int TableSize { get; }
    internal int FeaturesPerLevel { get; }
    internal int LevelOffset { get; }
    internal int OutputStride { get; }
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxInstantNgpHashEncodeBackwardKernel(
        DirectPtxRuntime runtime, int numPoints, int resolution, int tableSize,
        int featuresPerLevel, int levelOffset, int outputStride)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasValidatedScientific(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The checked-in instant-ngp-hash-encode-backward specialization is measured only on GA10x/SM86.");
        ValidateShape(numPoints, resolution, tableSize, featuresPerLevel, levelOffset, outputStride);
        NumPoints = numPoints;
        Resolution = resolution;
        TableSize = tableSize;
        FeaturesPerLevel = featuresPerLevel;
        LevelOffset = levelOffset;
        OutputStride = outputStride;
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, numPoints, resolution, tableSize, featuresPerLevel, levelOffset, outputStride);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor,
            numPoints, resolution, tableSize, featuresPerLevel, levelOffset, outputStride);
        _module = runtime.LoadModule(Ptx);
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo info);
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, BlockThreads);
        Blueprint.ResourceBudget.Validate(EntryPoint, info, BlockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Ptx, info, BlockThreads, activeBlocks, _module.JitInfoLog);
    }

    internal unsafe void Launch(DirectPtxTensorView positions, DirectPtxTensorView outputGradient, DirectPtxTensorView tableGradient)
    {
        Require(positions, Blueprint.Tensors[0], nameof(positions));
        Require(outputGradient, Blueprint.Tensors[1], nameof(outputGradient));
        Require(tableGradient, Blueprint.Tensors[2], nameof(tableGradient));

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

    private static string Hex(float value) => "0f" + BitConverter.ToInt32(BitConverter.GetBytes(value), 0).ToString("X8");

    private static void AppendHash(StringBuilder ptx, string xReg, string yReg, string zReg, int tableSize)
    {
        ptx.AppendLine($"    mul.lo.u32 %r11, {xReg}, 73856093;");
        ptx.AppendLine($"    mul.lo.u32 %r12, {yReg}, 19349663;");
        ptx.AppendLine($"    mul.lo.u32 %r13, {zReg}, 83492791;");
        ptx.AppendLine("    xor.b32 %r14, %r11, %r12;");
        ptx.AppendLine("    xor.b32 %r14, %r14, %r13;");
        ptx.AppendLine($"    rem.u32 %r14, %r14, {tableSize};");
    }

    // acc(%f16) += [hash(coords)==entry(%r3)] * grad(%f21) * (xyProd * zReg)
    private static void AppendBackwardCorner(
        StringBuilder ptx, string xReg, string yReg, string zReg, string xyProd, string zReg2, int tableSize)
    {
        AppendHash(ptx, xReg, yReg, zReg, tableSize);
        ptx.AppendLine($"    mul.rn.f32 %f18, {xyProd}, {zReg2};");
        ptx.AppendLine("    setp.eq.u32 %p1, %r14, %r3;");
        ptx.AppendLine("    @%p1 fma.rn.f32 %f16, %f21, %f18, %f16;");
    }

    private static void EmitAxis(StringBuilder ptx, string gReg, string c0Reg, string c1Reg, string fracReg, string invReg, string one)
    {
        ptx.AppendLine($"    cvt.rmi.s32.f32 {c0Reg}, {gReg};");
        ptx.AppendLine($"    add.s32 {c1Reg}, {c0Reg}, 1;");
        ptx.AppendLine($"    cvt.rn.f32.s32 %f20, {c0Reg};");
        ptx.AppendLine($"    sub.rn.f32 {fracReg}, {gReg}, %f20;");
        ptx.AppendLine($"    sub.rn.f32 {invReg}, {one}, {fracReg};");
    }

    internal static string EmitPtx(
        int ccMajor, int ccMinor, int numPoints, int resolution, int tableSize,
        int featuresPerLevel, int levelOffset, int outputStride)
    {
        ValidateShape(numPoints, resolution, tableSize, featuresPerLevel, levelOffset, outputStride);
        string res = Hex(resolution), clampHi = Hex(0.999999f), gradEps = Hex(1e-10f);
        const string one = "0f3F800000", zero = "0f00000000";

        var ptx = new StringBuilder(7_000);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine($"// instant-ngp-hash-encode-backward points={numPoints} res={resolution} table={tableSize} fpl={featuresPerLevel} loff={levelOffset} ostride={outputStride}");
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
        // positions[n*3]
        ptx.AppendLine("    mul.lo.u32 %r6, %r5, 3;");
        ptx.AppendLine("    mul.wide.u32 %rd3, %r6, 4;");
        ptx.AppendLine("    add.u64 %rd4, %rd0, %rd3;");
        ptx.AppendLine("    ld.global.nc.f32 %f0, [%rd4];");
        ptx.AppendLine("    ld.global.nc.f32 %f1, [%rd4+4];");
        ptx.AppendLine("    ld.global.nc.f32 %f2, [%rd4+8];");
        foreach ((string src, string g) in new[] { ("%f0", "%f3"), ("%f1", "%f4"), ("%f2", "%f5") })
        {
            ptx.AppendLine($"    min.f32 {src}, {src}, {clampHi};");
            ptx.AppendLine($"    max.f32 {src}, {src}, {zero};");
            ptx.AppendLine($"    mul.rn.f32 {g}, {src}, {res};");
        }
        // grad = outputGradient[n*outputStride + levelOffset + f]; skip if |grad| < 1e-10
        ptx.AppendLine($"    mad.lo.u32 %r7, %r5, {outputStride}, %r4;");
        ptx.AppendLine($"    add.u32 %r7, %r7, {levelOffset};");
        ptx.AppendLine("    mul.wide.u32 %rd5, %r7, 4;");
        ptx.AppendLine("    add.u64 %rd6, %rd1, %rd5;");
        ptx.AppendLine("    ld.global.nc.f32 %f21, [%rd6];");            // grad
        ptx.AppendLine("    abs.f32 %f22, %f21;");
        ptx.AppendLine($"    setp.lt.f32 %p0, %f22, {gradEps};");
        ptx.AppendLine("    @%p0 bra $NGPB_CONTINUE;");
        // coords + weights
        EmitAxis(ptx, "%f3", "%r8", "%r9", "%f6", "%f9", one);           // x0=%r8 x1=%r9 fx=%f6 ix=%f9
        EmitAxis(ptx, "%f4", "%r16", "%r17", "%f7", "%f10", one);        // y0=%r16 y1=%r17 fy=%f7 iy=%f10
        EmitAxis(ptx, "%f5", "%r18", "%r19", "%f8", "%f11", one);        // z0=%r18 z1=%r19 fz=%f8 iz=%f11
        ptx.AppendLine("    mul.rn.f32 %f12, %f9, %f10;");   // ixiy
        ptx.AppendLine("    mul.rn.f32 %f13, %f9, %f7;");    // ixfy
        ptx.AppendLine("    mul.rn.f32 %f14, %f6, %f10;");   // fxiy
        ptx.AppendLine("    mul.rn.f32 %f15, %f6, %f7;");    // fxfy
        AppendBackwardCorner(ptx, "%r8", "%r16", "%r18", "%f12", "%f11", tableSize);  // 000
        AppendBackwardCorner(ptx, "%r8", "%r16", "%r19", "%f12", "%f8", tableSize);   // 001
        AppendBackwardCorner(ptx, "%r8", "%r17", "%r18", "%f13", "%f11", tableSize);  // 010
        AppendBackwardCorner(ptx, "%r8", "%r17", "%r19", "%f13", "%f8", tableSize);   // 011
        AppendBackwardCorner(ptx, "%r9", "%r16", "%r18", "%f14", "%f11", tableSize);  // 100
        AppendBackwardCorner(ptx, "%r9", "%r16", "%r19", "%f14", "%f8", tableSize);   // 101
        AppendBackwardCorner(ptx, "%r9", "%r17", "%r18", "%f15", "%f11", tableSize);  // 110
        AppendBackwardCorner(ptx, "%r9", "%r17", "%r19", "%f15", "%f8", tableSize);   // 111
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
        DirectPtxArchitectureFamily architecture, int numPoints, int resolution, int tableSize,
        int featuresPerLevel, int levelOffset, int outputStride)
    {
        var posExtent = new DirectPtxExtent(numPoints * 3);
        var gradExtent = new DirectPtxExtent(numPoints * outputStride);
        var tblGradExtent = new DirectPtxExtent(tableSize * featuresPerLevel);
        return new DirectPtxKernelBlueprint(
            Operation: "instant-ngp-hash-encode-backward",
            Version: 1,
            Architecture: architecture,
            Variant: $"fp32-p{numPoints}-r{resolution}-t{tableSize}-f{featuresPerLevel}-lo{levelOffset}-os{outputStride}",
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
                ["formula"] = "tableGradient[entry,f] = sum_n sum_corner [hash(corner_n)==entry] * grad_n * w_corner",
                ["hash"] = "((x*73856093)^(y*19349663)^(z*83492791)) mod tableSize",
                ["grad-skip"] = "points with |grad| < 1e-10 skipped, matching NVRTC",
                ["global-intermediates"] = "none",
                ["temporary-device-allocation"] = "none",
                ["stride-parameters"] = "levelOffset/outputStride baked (constant, not runtime)"
            });
    }

    internal static bool IsSupportedShape(int numPoints, int resolution, int tableSize, int featuresPerLevel, int levelOffset, int outputStride)
    {
        if (numPoints <= 0 || resolution <= 0 || tableSize <= 0 || featuresPerLevel <= 0) return false;
        if (levelOffset < 0 || outputStride <= 0 || levelOffset + featuresPerLevel > outputStride) return false;
        long cells = (long)tableSize * featuresPerLevel;
        return cells > 0 && cells % BlockThreads == 0 && cells <= MaxCells;
    }

    internal static bool IsPromotedShape(int numPoints, int resolution, int tableSize, int featuresPerLevel, int levelOffset, int outputStride) => false;

    private static void ValidateShape(int numPoints, int resolution, int tableSize, int featuresPerLevel, int levelOffset, int outputStride)
    {
        if (!IsSupportedShape(numPoints, resolution, tableSize, featuresPerLevel, levelOffset, outputStride))
            throw new ArgumentOutOfRangeException(
                nameof(tableSize),
                $"Instant-NGP hash encode backward requires positive dims, levelOffset+featuresPerLevel<=outputStride, and (tableSize*featuresPerLevel) a multiple of {BlockThreads} up to {MaxCells}.");
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
