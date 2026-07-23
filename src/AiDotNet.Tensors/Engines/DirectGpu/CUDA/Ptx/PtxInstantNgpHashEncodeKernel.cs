using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Instant-NGP multiresolution hash-grid encoding, one level (issue #854), matching the NVRTC
/// <c>instant_ngp_hash_encode_level</c> kernel: for each (point, feature), trilinearly interpolate the
/// hashed feature at the 8 voxel corners of the clamped, scaled 3-D position. One thread owns one
/// (point, feature) output and computes the 8 spatial-hash lookups and interpolation weights in
/// registers — no shared memory, no reduction. Integer coordinates use <c>cvt.rmi.s32.f32</c> (floor),
/// and the spatial hash uses the reference multiply/xor/mod.
///
/// Shape (numPoints, resolution, tableSize, featuresPerLevel, levelOffset, outputStride) is baked into
/// the PTX, so the launch takes buffer pointers only. 256 threads/block,
/// grid = (numPoints*featuresPerLevel)/256, required to divide evenly (no divergent bounds guard).
/// </summary>
internal sealed class PtxInstantNgpHashEncodeKernel : IDisposable
{
    internal const int BlockThreads = 256;
    internal const int MaxCells = 2048 * 4096;
    internal const string EntryPoint = "aidotnet_instant_ngp_hash_encode_level";

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

    internal PtxInstantNgpHashEncodeKernel(
        DirectPtxRuntime runtime, int numPoints, int resolution, int tableSize,
        int featuresPerLevel, int levelOffset, int outputStride)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasValidatedScientific(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The checked-in instant-ngp-hash-encode specialization is measured only on GA10x/SM86.");
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

    internal unsafe void Launch(DirectPtxTensorView positions, DirectPtxTensorView hashTable, DirectPtxTensorView output)
    {
        Require(positions, Blueprint.Tensors[0], nameof(positions));
        Require(hashTable, Blueprint.Tensors[1], nameof(hashTable));
        Require(output, Blueprint.Tensors[2], nameof(output));

        IntPtr positionsPointer = positions.Pointer;
        IntPtr hashTablePointer = hashTable.Pointer;
        IntPtr outputPointer = output.Pointer;
        void** arguments = stackalloc void*[3];
        arguments[0] = &positionsPointer;
        arguments[1] = &hashTablePointer;
        arguments[2] = &outputPointer;
        uint grid = (uint)((NumPoints * FeaturesPerLevel) / BlockThreads);
        _module.Launch(_function, grid, 1, 1, BlockThreads, 1, 1, 0, arguments);
    }

    public void Dispose() => _module.Dispose();

    private static string Hex(float value) => "0f" + BitConverter.ToInt32(BitConverter.GetBytes(value), 0).ToString("X8");

    // Emits h = ((x*73856093) ^ (y*19349663) ^ (z*83492791)) % tableSize into %r14, using %r11..%r13.
    private static void AppendHash(StringBuilder ptx, string xReg, string yReg, string zReg, int tableSize)
    {
        ptx.AppendLine($"    mul.lo.u32 %r11, {xReg}, 73856093;");
        ptx.AppendLine($"    mul.lo.u32 %r12, {yReg}, 19349663;");
        ptx.AppendLine($"    mul.lo.u32 %r13, {zReg}, 83492791;");
        ptx.AppendLine("    xor.b32 %r14, %r11, %r12;");
        ptx.AppendLine("    xor.b32 %r14, %r14, %r13;");
        ptx.AppendLine($"    rem.u32 %r14, %r14, {tableSize};");
    }

    // Emits: value += weightReg * hashTable[hash(coords)*featuresPerLevel + f]
    private static void AppendCorner(
        StringBuilder ptx, string xReg, string yReg, string zReg, string weightReg, int tableSize, int featuresPerLevel)
    {
        AppendHash(ptx, xReg, yReg, zReg, tableSize);
        ptx.AppendLine($"    mad.lo.u32 %r15, %r14, {featuresPerLevel}, %r4;");   // hash*fpl + f
        ptx.AppendLine("    mul.wide.u32 %rd10, %r15, 4;");
        ptx.AppendLine("    add.u64 %rd11, %rd1, %rd10;");
        ptx.AppendLine("    ld.global.nc.f32 %f17, [%rd11];");
        ptx.AppendLine($"    fma.rn.f32 %f16, {weightReg}, %f17, %f16;");
    }

    internal static string EmitPtx(
        int ccMajor, int ccMinor, int numPoints, int resolution, int tableSize,
        int featuresPerLevel, int levelOffset, int outputStride)
    {
        ValidateShape(numPoints, resolution, tableSize, featuresPerLevel, levelOffset, outputStride);
        string res = Hex(resolution), clampHi = Hex(0.999999f);
        const string one = "0f3F800000", zero = "0f00000000";

        var ptx = new StringBuilder(6_000);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine($"// instant-ngp-hash-encode points={numPoints} res={resolution} table={tableSize} fpl={featuresPerLevel} loff={levelOffset} ostride={outputStride}");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 pos_ptr,");
        ptx.AppendLine("    .param .u64 tbl_ptr,");
        ptx.AppendLine("    .param .u64 out_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine($".maxntid {BlockThreads}, 1, 1");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<2>;");
        ptx.AppendLine("    .reg .b32 %r<20>;");
        ptx.AppendLine("    .reg .b64 %rd<16>;");
        ptx.AppendLine("    .reg .f32 %f<24>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [pos_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [tbl_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [out_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {BlockThreads}, %r0;");   // gid
        ptx.AppendLine($"    div.u32 %r3, %r2, {featuresPerLevel};");      // n (point)
        ptx.AppendLine($"    rem.u32 %r4, %r2, {featuresPerLevel};");      // f (feature)
        ptx.AppendLine("    mul.lo.u32 %r5, %r3, 3;");                     // n*3
        ptx.AppendLine("    mul.wide.u32 %rd3, %r5, 4;");
        ptx.AppendLine("    add.u64 %rd4, %rd0, %rd3;");                   // &positions[n*3]
        ptx.AppendLine("    ld.global.nc.f32 %f0, [%rd4];");
        ptx.AppendLine("    ld.global.nc.f32 %f1, [%rd4+4];");
        ptx.AppendLine("    ld.global.nc.f32 %f2, [%rd4+8];");
        // clamp each to [0, 0.999999] then scale by resolution
        foreach ((string src, string g) in new[] { ("%f0", "%f3"), ("%f1", "%f4"), ("%f2", "%f5") })
        {
            ptx.AppendLine($"    min.f32 {src}, {src}, {clampHi};");
            ptx.AppendLine($"    max.f32 {src}, {src}, {zero};");
            ptx.AppendLine($"    mul.rn.f32 {g}, {src}, {res};");
        }
        // x0/x1, fx, ix (and y, z). floor via cvt.rmi.s32.f32.
        // coords: x0=%r6 x1=%r7 ; y0=%r8 y1=%r9 ; z0=%r16 z1=%r17
        // fx=%f6 fy=%f7 fz=%f8 ; ix=%f9 iy=%f10 iz=%f11
        EmitAxis(ptx, "%f3", "%r6", "%r7", "%f6", "%f9", one);
        EmitAxis(ptx, "%f4", "%r8", "%r9", "%f7", "%f10", one);
        EmitAxis(ptx, "%f5", "%r16", "%r17", "%f8", "%f11", one);
        // xy products: ixiy=%f12 ixfy=%f13 fxiy=%f14 fxfy=%f15
        ptx.AppendLine("    mul.rn.f32 %f12, %f9, %f10;");
        ptx.AppendLine("    mul.rn.f32 %f13, %f9, %f7;");
        ptx.AppendLine("    mul.rn.f32 %f14, %f6, %f10;");
        ptx.AppendLine("    mul.rn.f32 %f15, %f6, %f7;");
        ptx.AppendLine($"    mov.f32 %f16, {zero};");                     // value
        // 8 corners: (x,y,z) with weight = xy_product * (iz|fz).  %f18/%f19 hold corner weights.
        ptx.AppendLine("    mul.rn.f32 %f18, %f12, %f11;"); AppendCorner(ptx, "%r6", "%r8", "%r16", "%f18", tableSize, featuresPerLevel);  // 000 ix iy iz
        ptx.AppendLine("    mul.rn.f32 %f19, %f12, %f8;");  AppendCorner(ptx, "%r6", "%r8", "%r17", "%f19", tableSize, featuresPerLevel);  // 001 ix iy fz
        ptx.AppendLine("    mul.rn.f32 %f18, %f13, %f11;"); AppendCorner(ptx, "%r6", "%r9", "%r16", "%f18", tableSize, featuresPerLevel);  // 010 ix fy iz
        ptx.AppendLine("    mul.rn.f32 %f19, %f13, %f8;");  AppendCorner(ptx, "%r6", "%r9", "%r17", "%f19", tableSize, featuresPerLevel);  // 011 ix fy fz
        ptx.AppendLine("    mul.rn.f32 %f18, %f14, %f11;"); AppendCorner(ptx, "%r7", "%r8", "%r16", "%f18", tableSize, featuresPerLevel);  // 100 fx iy iz
        ptx.AppendLine("    mul.rn.f32 %f19, %f14, %f8;");  AppendCorner(ptx, "%r7", "%r8", "%r17", "%f19", tableSize, featuresPerLevel);  // 101 fx iy fz
        ptx.AppendLine("    mul.rn.f32 %f18, %f15, %f11;"); AppendCorner(ptx, "%r7", "%r9", "%r16", "%f18", tableSize, featuresPerLevel);  // 110 fx fy iz
        ptx.AppendLine("    mul.rn.f32 %f19, %f15, %f8;");  AppendCorner(ptx, "%r7", "%r9", "%r17", "%f19", tableSize, featuresPerLevel);  // 111 fx fy fz
        // output[n*outputStride + levelOffset + f]
        ptx.AppendLine($"    mad.lo.u32 %r18, %r3, {outputStride}, %r4;");
        ptx.AppendLine($"    add.u32 %r18, %r18, {levelOffset};");
        ptx.AppendLine("    mul.wide.u32 %rd12, %r18, 4;");
        ptx.AppendLine("    add.u64 %rd13, %rd2, %rd12;");
        ptx.AppendLine("    st.global.f32 [%rd13], %f16;");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    // For a scaled coordinate g: c0 = floor(g), c1 = c0+1, frac = g - c0, inv = 1 - frac.
    private static void EmitAxis(StringBuilder ptx, string gReg, string c0Reg, string c1Reg, string fracReg, string invReg, string one)
    {
        ptx.AppendLine($"    cvt.rmi.s32.f32 {c0Reg}, {gReg};");       // floor
        ptx.AppendLine($"    add.s32 {c1Reg}, {c0Reg}, 1;");
        ptx.AppendLine($"    cvt.rn.f32.s32 %f20, {c0Reg};");
        ptx.AppendLine($"    sub.rn.f32 {fracReg}, {gReg}, %f20;");     // frac
        ptx.AppendLine($"    sub.rn.f32 {invReg}, {one}, {fracReg};");  // inv
    }

    private static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture, int numPoints, int resolution, int tableSize,
        int featuresPerLevel, int levelOffset, int outputStride)
    {
        var posExtent = new DirectPtxExtent(numPoints * 3);
        var tblExtent = new DirectPtxExtent(tableSize * featuresPerLevel);
        var outExtent = new DirectPtxExtent(numPoints * outputStride);
        return new DirectPtxKernelBlueprint(
            Operation: "instant-ngp-hash-encode",
            Version: 1,
            Architecture: architecture,
            Variant: $"fp32-p{numPoints}-r{resolution}-t{tableSize}-f{featuresPerLevel}-lo{levelOffset}-os{outputStride}",
            Tensors:
            [
                new("positions", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    posExtent, posExtent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("hashTable", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    tblExtent, tblExtent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    outExtent, outExtent, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 40,
                MaxStaticSharedBytes: 0,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 1),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["formula"] = "output[n, levelOffset+f] = trilinear over 8 hashed corners of clamp01(pos)*resolution",
                ["hash"] = "((x*73856093)^(y*19349663)^(z*83492791)) mod tableSize",
                ["global-intermediates"] = "none",
                ["temporary-device-allocation"] = "none",
                ["stride-parameters"] = "levelOffset/outputStride baked (constant, not runtime)"
            });
    }

    internal static bool IsSupportedShape(int numPoints, int resolution, int tableSize, int featuresPerLevel, int levelOffset, int outputStride)
    {
        if (numPoints <= 0 || resolution <= 0 || tableSize <= 0 || featuresPerLevel <= 0) return false;
        if (levelOffset < 0 || outputStride <= 0 || levelOffset + featuresPerLevel > outputStride) return false;
        long cells = (long)numPoints * featuresPerLevel;
        return cells > 0 && cells % BlockThreads == 0 && cells <= MaxCells;
    }

    internal static bool IsPromotedShape(int numPoints, int resolution, int tableSize, int featuresPerLevel, int levelOffset, int outputStride) => false;

    private static void ValidateShape(int numPoints, int resolution, int tableSize, int featuresPerLevel, int levelOffset, int outputStride)
    {
        if (!IsSupportedShape(numPoints, resolution, tableSize, featuresPerLevel, levelOffset, outputStride))
            throw new ArgumentOutOfRangeException(
                nameof(numPoints),
                $"Instant-NGP hash encode requires positive dims, levelOffset+featuresPerLevel<=outputStride, and (numPoints*featuresPerLevel) a multiple of {BlockThreads} up to {MaxCells}.");
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
