using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Uniform (combinatorial) mesh Laplacian assembly (issue #854), matching the NVRTC
/// <c>resident_uniform_mesh_laplacian</c> kernel: build the dense <c>[numVertices, numVertices]</c>
/// Laplacian from a triangle face list — each face's three edges contribute +1 on the incident
/// diagonal entries and -1 on the shared off-diagonal entries. One thread owns one (row, column) cell
/// and scans all faces serially in registers, accumulating the ±1 contributions under equality
/// predicates — no shared memory, no reduction. Faces are an int32 buffer laid out [face][3].
///
/// Shape (numFaces, numVertices) is baked into the PTX, so the launch takes buffer pointers only.
/// 256 threads/block, grid = (numVertices*numVertices)/256, required to divide evenly.
/// </summary>
internal sealed class PtxMeshLaplacianKernel : IDisposable
{
    internal const int BlockThreads = 256;
    internal const int MaxCells = 2048 * 4096;
    internal const int MaxFaces = 1 << 22;
    internal const string EntryPoint = "aidotnet_resident_uniform_mesh_laplacian";

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int NumFaces { get; }
    internal int NumVertices { get; }
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxMeshLaplacianKernel(DirectPtxRuntime runtime, int numFaces, int numVertices)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasValidatedScientific(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The checked-in mesh-laplacian specialization is measured only on GA10x/SM86.");
        ValidateShape(numFaces, numVertices);
        NumFaces = numFaces;
        NumVertices = numVertices;
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, numFaces, numVertices);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor, numFaces, numVertices);
        _module = runtime.LoadModule(Ptx);
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo info);
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, BlockThreads);
        Blueprint.ResourceBudget.Validate(EntryPoint, info, BlockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Ptx, info, BlockThreads, activeBlocks, _module.JitInfoLog);
    }

    internal unsafe void Launch(DirectPtxTensorView faces, DirectPtxTensorView output)
    {
        Require(faces, Blueprint.Tensors[0], nameof(faces));
        Require(output, Blueprint.Tensors[1], nameof(output));

        IntPtr facesPointer = faces.Pointer;
        IntPtr outputPointer = output.Pointer;
        void** arguments = stackalloc void*[2];
        arguments[0] = &facesPointer;
        arguments[1] = &outputPointer;
        uint grid = (uint)((NumVertices * NumVertices) / BlockThreads);
        _module.Launch(_function, grid, 1, 1, BlockThreads, 1, 1, 0, arguments);
    }

    public void Dispose() => _module.Dispose();

    internal static string EmitPtx(int ccMajor, int ccMinor, int numFaces, int numVertices)
    {
        ValidateShape(numFaces, numVertices);
        const string one = "0f3F800000", negOne = "0fBF800000";

        var ptx = new StringBuilder(5_000);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine($"// resident-uniform-mesh-laplacian faces={numFaces} vertices={numVertices}");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 faces_ptr,");
        ptx.AppendLine("    .param .u64 out_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine($".maxntid {BlockThreads}, 1, 1");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<8>;");
        ptx.AppendLine("    .reg .b32 %r<14>;");
        ptx.AppendLine("    .reg .b64 %rd<12>;");
        ptx.AppendLine("    .reg .f32 %f<2>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [faces_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [out_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {BlockThreads}, %r0;");   // gid
        ptx.AppendLine($"    div.u32 %r3, %r2, {numVertices};");           // row
        ptx.AppendLine($"    rem.u32 %r4, %r2, {numVertices};");           // column
        ptx.AppendLine("    mov.f32 %f0, 0f00000000;");                   // value
        ptx.AppendLine("    mov.u64 %rd4, %rd0;");                        // face walker
        ptx.AppendLine("    mov.u32 %r5, 0;");                            // face = 0
        ptx.AppendLine("$MESH_FACE_LOOP:");
        ptx.AppendLine("    ld.global.nc.s32 %r6, [%rd4];");             // v0
        ptx.AppendLine("    ld.global.nc.s32 %r7, [%rd4+4];");           // v1
        ptx.AppendLine("    ld.global.nc.s32 %r8, [%rd4+8];");           // v2
        // row/col equality predicates: rp0..rp2 = %p0..%p2 ; cp0..cp2 = %p3..%p5
        ptx.AppendLine("    setp.eq.s32 %p0, %r3, %r6;");   // row==v0
        ptx.AppendLine("    setp.eq.s32 %p1, %r3, %r7;");   // row==v1
        ptx.AppendLine("    setp.eq.s32 %p2, %r3, %r8;");   // row==v2
        ptx.AppendLine("    setp.eq.s32 %p3, %r4, %r6;");   // col==v0
        ptx.AppendLine("    setp.eq.s32 %p4, %r4, %r7;");   // col==v1
        ptx.AppendLine("    setp.eq.s32 %p5, %r4, %r8;");   // col==v2
        // 12 conditions matching the NVRTC kernel exactly (diagonal duplicates included).
        void Cond(string rp, string cp, string delta)
        {
            ptx.AppendLine($"    and.pred %p6, {rp}, {cp};");
            ptx.AppendLine($"    @%p6 add.rn.f32 %f0, %f0, {delta};");
        }
        Cond("%p0", "%p4", negOne);   // row==v0 && col==v1 : -1
        Cond("%p1", "%p3", negOne);   // row==v1 && col==v0 : -1
        Cond("%p0", "%p3", one);      // row==v0 && col==v0 : +1
        Cond("%p1", "%p4", one);      // row==v1 && col==v1 : +1
        Cond("%p1", "%p5", negOne);   // row==v1 && col==v2 : -1
        Cond("%p2", "%p4", negOne);   // row==v2 && col==v1 : -1
        Cond("%p1", "%p4", one);      // row==v1 && col==v1 : +1 (dup)
        Cond("%p2", "%p5", one);      // row==v2 && col==v2 : +1
        Cond("%p2", "%p3", negOne);   // row==v2 && col==v0 : -1
        Cond("%p0", "%p5", negOne);   // row==v0 && col==v2 : -1
        Cond("%p2", "%p5", one);      // row==v2 && col==v2 : +1 (dup)
        Cond("%p0", "%p3", one);      // row==v0 && col==v0 : +1 (dup)
        ptx.AppendLine("    add.u64 %rd4, %rd4, 12;");                   // next face (3 int32)
        ptx.AppendLine("    add.u32 %r5, %r5, 1;");
        ptx.AppendLine($"    setp.lt.u32 %p7, %r5, {numFaces};");
        ptx.AppendLine("    @%p7 bra $MESH_FACE_LOOP;");
        ptx.AppendLine("    mul.wide.u32 %rd8, %r2, 4;");
        ptx.AppendLine("    add.u64 %rd9, %rd1, %rd8;");
        ptx.AppendLine("    st.global.f32 [%rd9], %f0;");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static DirectPtxKernelBlueprint CreateBlueprint(DirectPtxArchitectureFamily architecture, int numFaces, int numVertices)
    {
        var facesExtent = new DirectPtxExtent(numFaces * 3);
        var outExtent = new DirectPtxExtent(numVertices * numVertices);
        return new DirectPtxKernelBlueprint(
            Operation: "resident-uniform-mesh-laplacian",
            Version: 1,
            Architecture: architecture,
            Variant: $"fp32-faces{numFaces}-verts{numVertices}",
            Tensors:
            [
                new("faces", DirectPtxPhysicalType.Int32, DirectPtxPhysicalLayout.Vector,
                    facesExtent, facesExtent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    outExtent, outExtent, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 20,
                MaxStaticSharedBytes: 0,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 1),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["formula"] = "L[row,col] = sum over faces of edge contributions (+1 diagonal, -1 off-diagonal)",
                ["faces-layout"] = "int32 [face][3] triangle vertex indices",
                ["global-intermediates"] = "none",
                ["temporary-device-allocation"] = "none",
                ["stride-parameters"] = "none"
            });
    }

    internal static bool IsSupportedShape(int numFaces, int numVertices)
    {
        if (numFaces <= 0 || numFaces > MaxFaces || numVertices <= 0) return false;
        long cells = (long)numVertices * numVertices;
        return cells > 0 && cells % BlockThreads == 0 && cells <= MaxCells;
    }

    internal static bool IsPromotedShape(int numFaces, int numVertices) => false;

    private static void ValidateShape(int numFaces, int numVertices)
    {
        if (!IsSupportedShape(numFaces, numVertices))
            throw new ArgumentOutOfRangeException(
                nameof(numVertices),
                $"Mesh laplacian requires positive numFaces<={MaxFaces} and (numVertices*numVertices) a multiple of {BlockThreads} up to {MaxCells}.");
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
