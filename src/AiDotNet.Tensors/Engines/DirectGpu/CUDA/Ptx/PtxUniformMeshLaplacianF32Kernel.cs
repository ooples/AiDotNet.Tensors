#if NET5_0_OR_GREATER
using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Exact deterministic uniform triangle-mesh Laplacian. Each dense matrix cell
/// retains one accumulator across the baked face scan and writes exactly once.
/// </summary>
internal sealed class PtxUniformMeshLaplacianF32Kernel : IDisposable
{
    internal const int Faces = 2048;
    internal const int Vertices = 1024;
    internal const int FaceIndices = Faces * 3;
    internal const int OutputElements = Vertices * Vertices;
    internal const int BlockThreads = 256;
    internal const string EntryPoint = "aidotnet_uniform_mesh_laplacian_f32_f2048_v1024";

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal string Ptx { get; }
    internal DirectPtxFunctionInfo FunctionInfo { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxUniformMeshLaplacianF32Kernel(DirectPtxRuntime runtime)
    {
        ArgumentNullException.ThrowIfNull(runtime);
        if (!DirectPtxArchitecture.HasValidatedSparseGraph(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new NotSupportedException(
                $"Uniform mesh Laplacian has no SM {runtime.ComputeCapabilityMajor}.{runtime.ComputeCapabilityMinor} experimental specialization.");
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor);
        _module = runtime.LoadModule(Ptx);
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo functionInfo);
        FunctionInfo = functionInfo;
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, BlockThreads);
        Blueprint.ResourceBudget.Validate(EntryPoint, functionInfo, BlockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Ptx, functionInfo,
            BlockThreads, activeBlocks, _module.JitInfoLog);
    }

    internal static bool SupportsShape(int faces, int vertices) =>
        faces == Faces && vertices == Vertices;

    internal unsafe void Launch(DirectPtxTensorView faces, DirectPtxTensorView output)
    {
        Require(faces, Blueprint.Tensors[0], nameof(faces));
        Require(output, Blueprint.Tensors[1], nameof(output));
        if (Overlaps(output, faces))
            throw new ArgumentException("Mesh Laplacian output must be disjoint from faces.", nameof(output));
        IntPtr p0 = faces.Pointer;
        IntPtr p1 = output.Pointer;
        void** arguments = stackalloc void*[2];
        arguments[0] = &p0;
        arguments[1] = &p1;
        _module.Launch(_function, OutputElements / BlockThreads, 1, 1,
            BlockThreads, 1, 1, 0, arguments);
    }

    internal static DirectPtxKernelBlueprint CreateBlueprint(DirectPtxArchitectureFamily architecture)
    {
        var faceExtent = new DirectPtxExtent(Faces, 3);
        var outputExtent = new DirectPtxExtent(Vertices, Vertices);
        return new DirectPtxKernelBlueprint(
            Operation: "uniform-mesh-laplacian-f32",
            Version: 1,
            Architecture: architecture,
            Variant: "faces2048-vertices1024-dense",
            Tensors:
            [
                new DirectPtxTensorContract(
                    "faces", DirectPtxPhysicalType.Int32, DirectPtxPhysicalLayout.TriangleFaces,
                    faceExtent, faceExtent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new DirectPtxTensorContract(
                    "output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    outputExtent, outputExtent, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(32, 0, 0, 6),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["topology"] = "three-undirected-edges-per-triangle-with-multiplicity",
                ["reduction-order"] = "ascending-face",
                ["output-write"] = "single-overwrite",
                ["workspace-bytes"] = "0",
                ["intermediate-global-bytes"] = "0"
            });
    }

    internal static string EmitPtx(int ccMajor, int ccMinor)
    {
        if (ccMajor <= 0 || ccMinor < 0) throw new ArgumentOutOfRangeException(nameof(ccMajor));
        var ptx = new StringBuilder(6144);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 faces_ptr,");
        ptx.AppendLine("    .param .u64 output_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<8>;");
        ptx.AppendLine("    .reg .b32 %r<16>;");
        ptx.AppendLine("    .reg .b64 %rd<10>;");
        ptx.AppendLine("    .reg .f32 %f<3>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [faces_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [output_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {BlockThreads}, %r0;");
        ptx.AppendLine("    shr.u32 %r3, %r2, 10;");
        ptx.AppendLine("    and.b32 %r4, %r2, 1023;");
        ptx.AppendLine("    mov.u32 %r5, 0;");
        ptx.AppendLine("    mov.f32 %f0, 0f00000000;");
        ptx.AppendLine("MESH_LAPLACIAN_FACE_LOOP:");
        ptx.AppendLine($"    setp.ge.u32 %p0, %r5, {Faces};");
        ptx.AppendLine("    @%p0 bra MESH_LAPLACIAN_DONE;");
        ptx.AppendLine("    mad.lo.u32 %r6, %r5, 3, 0;");
        ptx.AppendLine("    mul.wide.u32 %rd2, %r6, 4;");
        ptx.AppendLine("    add.u64 %rd3, %rd0, %rd2;");
        ptx.AppendLine("    ld.global.u32 %r7, [%rd3];");
        ptx.AppendLine("    ld.global.u32 %r8, [%rd3+4];");
        ptx.AppendLine("    ld.global.u32 %r9, [%rd3+8];");
        EmitEdge(ptx, "%r7", "%r8");
        EmitEdge(ptx, "%r8", "%r9");
        EmitEdge(ptx, "%r9", "%r7");
        ptx.AppendLine("    add.u32 %r5, %r5, 1;");
        ptx.AppendLine("    bra MESH_LAPLACIAN_FACE_LOOP;");
        ptx.AppendLine("MESH_LAPLACIAN_DONE:");
        ptx.AppendLine("    mul.wide.u32 %rd4, %r2, 4;");
        ptx.AppendLine("    add.u64 %rd5, %rd1, %rd4;");
        ptx.AppendLine("    st.global.f32 [%rd5], %f0;");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static void EmitEdge(StringBuilder ptx, string first, string second)
    {
        EmitContribution(ptx, "%r3", first, "%r4", second, "0fbf800000");
        EmitContribution(ptx, "%r3", second, "%r4", first, "0fbf800000");
        EmitContribution(ptx, "%r3", first, "%r4", first, "0f3f800000");
        EmitContribution(ptx, "%r3", second, "%r4", second, "0f3f800000");
    }

    private static void EmitContribution(
        StringBuilder ptx,
        string left,
        string leftTarget,
        string right,
        string rightTarget,
        string contribution)
    {
        ptx.AppendLine($"    setp.eq.u32 %p1, {left}, {leftTarget};");
        ptx.AppendLine($"    setp.eq.u32 %p2, {right}, {rightTarget};");
        ptx.AppendLine("    and.pred %p3, %p1, %p2;");
        ptx.AppendLine($"    @%p3 add.rn.f32 %f0, %f0, {contribution};");
    }

    private static void Require(
        DirectPtxTensorView view,
        DirectPtxTensorContract contract,
        string parameter)
    {
        if (view.Pointer == IntPtr.Zero || view.PhysicalType != contract.PhysicalType ||
            view.Layout != contract.Layout || view.LogicalExtent != contract.LogicalExtent ||
            view.PhysicalExtent != contract.PhysicalExtent || view.ByteLength != contract.RequiredBytes ||
            view.AllocationByteLength != contract.RequiredBytes)
            throw new ArgumentException($"{parameter} does not satisfy exact physical ABI '{contract.Name}'.", parameter);
    }

    private static bool Overlaps(DirectPtxTensorView left, DirectPtxTensorView right)
    {
        nuint leftStart = (nuint)left.Pointer;
        nuint rightStart = (nuint)right.Pointer;
        nuint leftEnd = checked(leftStart + left.ByteLength);
        nuint rightEnd = checked(rightStart + right.ByteLength);
        return leftStart < rightEnd && rightStart < leftEnd;
    }

    public void Dispose() => _module.Dispose();
}
#endif
