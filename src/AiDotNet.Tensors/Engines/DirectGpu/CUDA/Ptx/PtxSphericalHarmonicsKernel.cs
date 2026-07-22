using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Spherical-harmonics color evaluation (issue #854 / #775): for each (point, channel),
/// <c>output = clamp01(sum_b shCoefficients[i,b,ch] * basis_b(dir_i))</c>, matching the NVRTC
/// <c>spherical_harmonics</c> kernel. One thread owns one (point, channel) output — the normalized
/// view direction and the real SH basis (degree 0..3, up to 16 functions) are computed in registers
/// and the coefficient dot-product is unrolled over <c>basisCount</c>; no shared memory, no reduction.
///
/// Shape (numPoints, basisCount, numChannels, degree, broadcastDir) is baked into the PTX, so the
/// launch takes buffer pointers only. 256 threads/block, grid = (numPoints*numChannels)/256, required
/// to divide evenly (no divergent bounds guard). <c>basisCount</c> may not exceed <c>(degree+1)^2</c>.
/// </summary>
internal sealed class PtxSphericalHarmonicsKernel : IDisposable
{
    internal const int BlockThreads = 256;
    internal const int MaxCount = 2048 * 4096;
    internal const int MaxDegree = 3;
    internal const string EntryPoint = "aidotnet_spherical_harmonics";

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int NumPoints { get; }
    internal int BasisCount { get; }
    internal int NumChannels { get; }
    internal int Degree { get; }
    internal bool BroadcastDir { get; }
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxSphericalHarmonicsKernel(
        DirectPtxRuntime runtime, int numPoints, int basisCount, int numChannels, int degree, bool broadcastDir)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasValidatedScientific(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The checked-in spherical-harmonics specialization is measured only on GA10x/SM86.");
        ValidateShape(numPoints, basisCount, numChannels, degree);
        NumPoints = numPoints;
        BasisCount = basisCount;
        NumChannels = numChannels;
        Degree = degree;
        BroadcastDir = broadcastDir;
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, numPoints, basisCount, numChannels, degree, broadcastDir);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor,
            numPoints, basisCount, numChannels, degree, broadcastDir);
        _module = runtime.LoadModule(Ptx);
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo info);
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, BlockThreads);
        Blueprint.ResourceBudget.Validate(EntryPoint, info, BlockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Ptx, info, BlockThreads, activeBlocks, _module.JitInfoLog);
    }

    internal unsafe void Launch(
        DirectPtxTensorView shCoefficients, DirectPtxTensorView viewDirections, DirectPtxTensorView output)
    {
        Require(shCoefficients, Blueprint.Tensors[0], nameof(shCoefficients));
        Require(viewDirections, Blueprint.Tensors[1], nameof(viewDirections));
        Require(output, Blueprint.Tensors[2], nameof(output));

        IntPtr coeffPointer = shCoefficients.Pointer;
        IntPtr dirPointer = viewDirections.Pointer;
        IntPtr outputPointer = output.Pointer;
        void** arguments = stackalloc void*[3];
        arguments[0] = &coeffPointer;
        arguments[1] = &dirPointer;
        arguments[2] = &outputPointer;
        uint grid = (uint)((NumPoints * NumChannels) / BlockThreads);
        _module.Launch(_function, grid, 1, 1, BlockThreads, 1, 1, 0, arguments);
    }

    public void Dispose() => _module.Dispose();

    private static string Hex(float value) => "0f" + BitConverter.ToInt32(BitConverter.GetBytes(value), 0).ToString("X8");

    internal static int MaxBasisForDegree(int degree) => (degree + 1) * (degree + 1);

    internal static string EmitPtx(
        int ccMajor, int ccMinor, int numPoints, int basisCount, int numChannels, int degree, bool broadcastDir)
    {
        ValidateShape(numPoints, basisCount, numChannels, degree);

        // Real SH constants (Condon-Shortley folded), matching the NVRTC table.
        string c0 = Hex(0.282095f), c1 = Hex(0.488603f), c2 = Hex(1.092548f), c3 = Hex(0.315392f),
               c4 = Hex(0.546274f), c5 = Hex(0.590044f), c6 = Hex(2.890611f), c7 = Hex(0.457046f),
               c8 = Hex(0.373176f), c9 = Hex(1.445306f);
        string f3 = Hex(3.0f), f5 = Hex(5.0f), f1 = Hex(1.0f), zero = "0f00000000";

        var ptx = new StringBuilder(8_000);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine($"// spherical-harmonics points={numPoints} basis={basisCount} channels={numChannels} degree={degree} broadcast={broadcastDir}");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 coeff_ptr,");
        ptx.AppendLine("    .param .u64 dir_ptr,");
        ptx.AppendLine("    .param .u64 out_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine($".maxntid {BlockThreads}, 1, 1");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<4>;");
        ptx.AppendLine("    .reg .b32 %r<12>;");
        ptx.AppendLine("    .reg .b64 %rd<20>;");
        ptx.AppendLine("    .reg .f32 %f<48>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [coeff_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [dir_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [out_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {BlockThreads}, %r0;");   // idx = i*numChannels + ch
        ptx.AppendLine($"    div.u32 %r3, %r2, {numChannels};");           // i (point)
        ptx.AppendLine($"    rem.u32 %r4, %r2, {numChannels};");           // ch

        // dir base element = (broadcast ? 0 : i) * 3
        if (broadcastDir)
        {
            ptx.AppendLine("    mov.u64 %rd4, %rd1;");                     // &viewDir[0]
        }
        else
        {
            ptx.AppendLine("    mul.lo.u32 %r5, %r3, 3;");
            ptx.AppendLine("    mul.wide.u32 %rd3, %r5, 4;");
            ptx.AppendLine("    add.u64 %rd4, %rd1, %rd3;");              // &viewDir[i*3]
        }
        ptx.AppendLine("    ld.global.nc.f32 %f13, [%rd4];");             // dx
        ptx.AppendLine("    ld.global.nc.f32 %f14, [%rd4+4];");           // dy
        ptx.AppendLine("    ld.global.nc.f32 %f15, [%rd4+8];");           // dz
        // normSq
        ptx.AppendLine("    mul.rn.f32 %f0, %f13, %f13;");
        ptx.AppendLine("    fma.rn.f32 %f0, %f14, %f14, %f0;");
        ptx.AppendLine("    fma.rn.f32 %f0, %f15, %f15, %f0;");           // normSq
        ptx.AppendLine("    sqrt.rn.f32 %f1, %f0;");
        ptx.AppendLine($"    div.rn.f32 %f2, {f1}, %f1;");                // 1/norm
        ptx.AppendLine($"    setp.gt.f32 %p0, %f0, {zero};");
        ptx.AppendLine($"    selp.f32 %f2, %f2, {f1}, %p0;");             // inv = norm>0 ? 1/norm : 1
        ptx.AppendLine("    mul.rn.f32 %f13, %f13, %f2;");                // normalized dx
        ptx.AppendLine("    mul.rn.f32 %f14, %f14, %f2;");                // dy
        ptx.AppendLine("    mul.rn.f32 %f15, %f15, %f2;");                // dz

        bool needSquares = basisCount >= 7;
        if (needSquares)
        {
            ptx.AppendLine("    mul.rn.f32 %f3, %f13, %f13;");            // dx2
            ptx.AppendLine("    mul.rn.f32 %f4, %f14, %f14;");            // dy2
            ptx.AppendLine("    mul.rn.f32 %f5, %f15, %f15;");            // dz2
        }

        // basis[b] -> %f{16+b}. Emit only b < basisCount (all within (degree+1)^2).
        void Basis(int b, string body) { if (b < basisCount) ptx.Append(body); }

        Basis(0, $"    mov.f32 %f16, {c0};\n");
        Basis(1, $"    mul.rn.f32 %f17, %f14, {c1};\n");                  // 0.488603*dy
        Basis(2, $"    mul.rn.f32 %f18, %f15, {c1};\n");                  // 0.488603*dz
        Basis(3, $"    mul.rn.f32 %f19, %f13, {c1};\n");                  // 0.488603*dx
        Basis(4, $"    mul.rn.f32 %f6, %f13, %f14;\n    mul.rn.f32 %f20, %f6, {c2};\n");        // 1.092548*dx*dy
        Basis(5, $"    mul.rn.f32 %f6, %f14, %f15;\n    mul.rn.f32 %f21, %f6, {c2};\n");        // 1.092548*dy*dz
        Basis(6, $"    mul.rn.f32 %f6, %f5, {f3};\n    sub.rn.f32 %f6, %f6, {f1};\n    mul.rn.f32 %f22, %f6, {c3};\n"); // 0.315392*(3dz2-1)
        Basis(7, $"    mul.rn.f32 %f6, %f13, %f15;\n    mul.rn.f32 %f23, %f6, {c2};\n");        // 1.092548*dx*dz
        Basis(8, $"    sub.rn.f32 %f6, %f3, %f4;\n    mul.rn.f32 %f24, %f6, {c4};\n");          // 0.546274*(dx2-dy2)
        Basis(9, $"    mul.rn.f32 %f6, %f3, {f3};\n    sub.rn.f32 %f6, %f6, %f4;\n    mul.rn.f32 %f6, %f6, %f14;\n    mul.rn.f32 %f25, %f6, {c5};\n"); // 0.590044*dy*(3dx2-dy2)
        Basis(10, $"    mul.rn.f32 %f6, %f13, %f14;\n    mul.rn.f32 %f6, %f6, %f15;\n    mul.rn.f32 %f26, %f6, {c6};\n"); // 2.890611*dx*dy*dz
        Basis(11, $"    mul.rn.f32 %f6, %f5, {f5};\n    sub.rn.f32 %f6, %f6, {f1};\n    mul.rn.f32 %f6, %f6, %f14;\n    mul.rn.f32 %f27, %f6, {c7};\n"); // 0.457046*dy*(5dz2-1)
        Basis(12, $"    mul.rn.f32 %f6, %f5, {f5};\n    sub.rn.f32 %f6, %f6, {f3};\n    mul.rn.f32 %f6, %f6, %f15;\n    mul.rn.f32 %f28, %f6, {c8};\n"); // 0.373176*dz*(5dz2-3)
        Basis(13, $"    mul.rn.f32 %f6, %f5, {f5};\n    sub.rn.f32 %f6, %f6, {f1};\n    mul.rn.f32 %f6, %f6, %f13;\n    mul.rn.f32 %f29, %f6, {c7};\n"); // 0.457046*dx*(5dz2-1)
        Basis(14, $"    sub.rn.f32 %f6, %f3, %f4;\n    mul.rn.f32 %f6, %f6, %f15;\n    mul.rn.f32 %f30, %f6, {c9};\n"); // 1.445306*dz*(dx2-dy2)
        Basis(15, $"    mul.rn.f32 %f6, %f4, {f3};\n    sub.rn.f32 %f6, %f3, %f6;\n    mul.rn.f32 %f6, %f6, %f13;\n    mul.rn.f32 %f31, %f6, {c5};\n"); // 0.590044*dx*(dx2-3dy2)

        // coeff base element = i*basisCount*numChannels + ch  ; stride over b = numChannels.
        ptx.AppendLine($"    mul.lo.u32 %r6, %r3, {basisCount * numChannels};");
        ptx.AppendLine("    add.u32 %r6, %r6, %r4;");
        ptx.AppendLine("    mul.wide.u32 %rd5, %r6, 4;");
        ptx.AppendLine("    add.u64 %rd6, %rd0, %rd5;");                  // &coeff[i,0,ch]
        ptx.AppendLine($"    mov.f32 %f7, {zero};");                     // color
        int strideBytes = numChannels * 4;
        for (int b = 0; b < basisCount; b++)
        {
            ptx.AppendLine($"    ld.global.nc.f32 %f8, [%rd6+{b * strideBytes}];");
            ptx.AppendLine($"    fma.rn.f32 %f7, %f8, %f{16 + b}, %f7;");
        }
        // clamp01
        ptx.AppendLine($"    max.f32 %f7, %f7, {zero};");
        ptx.AppendLine($"    min.f32 %f7, %f7, {f1};");
        ptx.AppendLine("    mul.wide.u32 %rd7, %r2, 4;");
        ptx.AppendLine("    add.u64 %rd8, %rd2, %rd7;");
        ptx.AppendLine("    st.global.f32 [%rd8], %f7;");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture, int numPoints, int basisCount, int numChannels, int degree, bool broadcastDir)
    {
        var coeffExtent = new DirectPtxExtent(numPoints * basisCount * numChannels);
        var dirExtent = new DirectPtxExtent((broadcastDir ? 1 : numPoints) * 3);
        var outExtent = new DirectPtxExtent(numPoints * numChannels);
        return new DirectPtxKernelBlueprint(
            Operation: "spherical-harmonics",
            Version: 1,
            Architecture: architecture,
            Variant: $"fp32-p{numPoints}-b{basisCount}-c{numChannels}-deg{degree}-bc{(broadcastDir ? 1 : 0)}",
            Tensors:
            [
                new("shCoefficients", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    coeffExtent, coeffExtent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("viewDirections", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    dirExtent, dirExtent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    outExtent, outExtent, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 64,
                MaxStaticSharedBytes: 0,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 1),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["formula"] = "output[i,ch] = clamp01(sum_b shCoefficients[i,b,ch] * basis_b(normalize(dir_i)))",
                ["basis"] = "real spherical harmonics, degree 0..3 (up to 16 functions)",
                ["global-intermediates"] = "none",
                ["temporary-device-allocation"] = "none",
                ["stride-parameters"] = "none"
            });
    }

    internal static bool IsSupportedShape(int numPoints, int basisCount, int numChannels, int degree)
    {
        if (numPoints <= 0 || basisCount <= 0 || numChannels <= 0) return false;
        if (degree < 0 || degree > MaxDegree) return false;
        if (basisCount > MaxBasisForDegree(degree)) return false;
        long count = (long)numPoints * numChannels;
        return count > 0 && count % BlockThreads == 0 && count <= MaxCount;
    }

    internal static bool IsPromotedShape(int numPoints, int basisCount, int numChannels, int degree) => false;

    private static void ValidateShape(int numPoints, int basisCount, int numChannels, int degree)
    {
        if (!IsSupportedShape(numPoints, basisCount, numChannels, degree))
            throw new ArgumentOutOfRangeException(
                nameof(basisCount),
                $"Spherical harmonics requires degree in [0,{MaxDegree}], basisCount<=(degree+1)^2, and (numPoints*numChannels) a multiple of {BlockThreads} up to {MaxCount}.");
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
