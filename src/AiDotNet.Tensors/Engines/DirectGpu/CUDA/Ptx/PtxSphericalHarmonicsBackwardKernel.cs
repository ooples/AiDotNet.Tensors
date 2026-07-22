using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Spherical-harmonics backward w.r.t. coefficients (issue #854 / #775): for each
/// (point, basis, channel), <c>shGrad[i,b,ch] = clampMask * outputGradient[i,ch] * basis_b(dir_i)</c>,
/// where <c>clampMask</c> zeroes the gradient when the forward pre-clamp value fell outside [0,1].
/// Matches the NVRTC <c>spherical_harmonics_backward</c> kernel. One thread owns one (point, basis,
/// channel) element: it recomputes the register-resident SH basis, recomputes the forward pre-clamp
/// dot-product for the mask, selects <c>basis[b]</c> for its own <c>b</c> via a compile-time-bounded
/// <c>selp</c> chain, and writes the masked gradient — no shared memory, no reduction.
///
/// Shape (numPoints, basisCount, numChannels, degree, broadcastDir) is baked into the PTX, so the
/// launch takes buffer pointers only. 256 threads/block, grid = (numPoints*basisCount*numChannels)/256,
/// required to divide evenly (no divergent bounds guard). <c>basisCount</c> may not exceed
/// <c>(degree+1)^2</c>.
/// </summary>
internal sealed class PtxSphericalHarmonicsBackwardKernel : IDisposable
{
    internal const int BlockThreads = 256;
    internal const int MaxCount = 2048 * 4096;
    internal const int MaxDegree = 3;
    internal const string EntryPoint = "aidotnet_spherical_harmonics_backward";

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

    internal PtxSphericalHarmonicsBackwardKernel(
        DirectPtxRuntime runtime, int numPoints, int basisCount, int numChannels, int degree, bool broadcastDir)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasValidatedScientific(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The checked-in spherical-harmonics-backward specialization is measured only on GA10x/SM86.");
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
        DirectPtxTensorView shCoefficients, DirectPtxTensorView viewDirections,
        DirectPtxTensorView outputGradient, DirectPtxTensorView shGrad)
    {
        Require(shCoefficients, Blueprint.Tensors[0], nameof(shCoefficients));
        Require(viewDirections, Blueprint.Tensors[1], nameof(viewDirections));
        Require(outputGradient, Blueprint.Tensors[2], nameof(outputGradient));
        Require(shGrad, Blueprint.Tensors[3], nameof(shGrad));

        IntPtr coeffPointer = shCoefficients.Pointer;
        IntPtr dirPointer = viewDirections.Pointer;
        IntPtr gradPointer = outputGradient.Pointer;
        IntPtr shGradPointer = shGrad.Pointer;
        void** arguments = stackalloc void*[4];
        arguments[0] = &coeffPointer;
        arguments[1] = &dirPointer;
        arguments[2] = &gradPointer;
        arguments[3] = &shGradPointer;
        uint grid = (uint)((NumPoints * BasisCount * NumChannels) / BlockThreads);
        _module.Launch(_function, grid, 1, 1, BlockThreads, 1, 1, 0, arguments);
    }

    public void Dispose() => _module.Dispose();

    private static string Hex(float value) => "0f" + BitConverter.ToInt32(BitConverter.GetBytes(value), 0).ToString("X8");

    internal static string EmitPtx(
        int ccMajor, int ccMinor, int numPoints, int basisCount, int numChannels, int degree, bool broadcastDir)
    {
        ValidateShape(numPoints, basisCount, numChannels, degree);

        string c0 = Hex(0.282095f), c1 = Hex(0.488603f), c2 = Hex(1.092548f), c3 = Hex(0.315392f),
               c4 = Hex(0.546274f), c5 = Hex(0.590044f), c6 = Hex(2.890611f), c7 = Hex(0.457046f),
               c8 = Hex(0.373176f), c9 = Hex(1.445306f);
        string f3 = Hex(3.0f), f5 = Hex(5.0f), f1 = Hex(1.0f), zero = "0f00000000";

        int perPoint = basisCount * numChannels;
        var ptx = new StringBuilder(9_000);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine($"// spherical-harmonics-backward points={numPoints} basis={basisCount} channels={numChannels} degree={degree} broadcast={broadcastDir}");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 coeff_ptr,");
        ptx.AppendLine("    .param .u64 dir_ptr,");
        ptx.AppendLine("    .param .u64 grad_ptr,");
        ptx.AppendLine("    .param .u64 shgrad_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine($".maxntid {BlockThreads}, 1, 1");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<6>;");
        ptx.AppendLine("    .reg .b32 %r<14>;");
        ptx.AppendLine("    .reg .b64 %rd<24>;");
        ptx.AppendLine("    .reg .f32 %f<48>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [coeff_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [dir_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [grad_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd9, [shgrad_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {BlockThreads}, %r0;");   // idx
        ptx.AppendLine($"    rem.u32 %r4, %r2, {numChannels};");           // ch
        ptx.AppendLine($"    div.u32 %r5, %r2, {numChannels};");           // q = i*basisCount + b
        ptx.AppendLine($"    rem.u32 %r6, %r5, {basisCount};");           // b
        ptx.AppendLine($"    div.u32 %r3, %r5, {basisCount};");           // i (point)

        // dir base element = (broadcast ? 0 : i) * 3
        if (broadcastDir)
        {
            ptx.AppendLine("    mov.u64 %rd4, %rd1;");
        }
        else
        {
            ptx.AppendLine("    mul.lo.u32 %r7, %r3, 3;");
            ptx.AppendLine("    mul.wide.u32 %rd3, %r7, 4;");
            ptx.AppendLine("    add.u64 %rd4, %rd1, %rd3;");
        }
        ptx.AppendLine("    ld.global.nc.f32 %f13, [%rd4];");
        ptx.AppendLine("    ld.global.nc.f32 %f14, [%rd4+4];");
        ptx.AppendLine("    ld.global.nc.f32 %f15, [%rd4+8];");
        ptx.AppendLine("    mul.rn.f32 %f0, %f13, %f13;");
        ptx.AppendLine("    fma.rn.f32 %f0, %f14, %f14, %f0;");
        ptx.AppendLine("    fma.rn.f32 %f0, %f15, %f15, %f0;");
        ptx.AppendLine("    sqrt.rn.f32 %f1, %f0;");
        ptx.AppendLine($"    div.rn.f32 %f2, {f1}, %f1;");
        ptx.AppendLine($"    setp.gt.f32 %p0, %f0, {zero};");
        ptx.AppendLine($"    selp.f32 %f2, %f2, {f1}, %p0;");
        ptx.AppendLine("    mul.rn.f32 %f13, %f13, %f2;");
        ptx.AppendLine("    mul.rn.f32 %f14, %f14, %f2;");
        ptx.AppendLine("    mul.rn.f32 %f15, %f15, %f2;");

        bool needSquares = basisCount >= 7;
        if (needSquares)
        {
            ptx.AppendLine("    mul.rn.f32 %f3, %f13, %f13;");            // dx2
            ptx.AppendLine("    mul.rn.f32 %f4, %f14, %f14;");            // dy2
            ptx.AppendLine("    mul.rn.f32 %f5, %f15, %f15;");            // dz2
        }

        void Basis(int b, string body) { if (b < basisCount) ptx.Append(body); }
        Basis(0, $"    mov.f32 %f16, {c0};\n");
        Basis(1, $"    mul.rn.f32 %f17, %f14, {c1};\n");
        Basis(2, $"    mul.rn.f32 %f18, %f15, {c1};\n");
        Basis(3, $"    mul.rn.f32 %f19, %f13, {c1};\n");
        Basis(4, $"    mul.rn.f32 %f6, %f13, %f14;\n    mul.rn.f32 %f20, %f6, {c2};\n");
        Basis(5, $"    mul.rn.f32 %f6, %f14, %f15;\n    mul.rn.f32 %f21, %f6, {c2};\n");
        Basis(6, $"    mul.rn.f32 %f6, %f5, {f3};\n    sub.rn.f32 %f6, %f6, {f1};\n    mul.rn.f32 %f22, %f6, {c3};\n");
        Basis(7, $"    mul.rn.f32 %f6, %f13, %f15;\n    mul.rn.f32 %f23, %f6, {c2};\n");
        Basis(8, $"    sub.rn.f32 %f6, %f3, %f4;\n    mul.rn.f32 %f24, %f6, {c4};\n");
        Basis(9, $"    mul.rn.f32 %f6, %f3, {f3};\n    sub.rn.f32 %f6, %f6, %f4;\n    mul.rn.f32 %f6, %f6, %f14;\n    mul.rn.f32 %f25, %f6, {c5};\n");
        Basis(10, $"    mul.rn.f32 %f6, %f13, %f14;\n    mul.rn.f32 %f6, %f6, %f15;\n    mul.rn.f32 %f26, %f6, {c6};\n");
        Basis(11, $"    mul.rn.f32 %f6, %f5, {f5};\n    sub.rn.f32 %f6, %f6, {f1};\n    mul.rn.f32 %f6, %f6, %f14;\n    mul.rn.f32 %f27, %f6, {c7};\n");
        Basis(12, $"    mul.rn.f32 %f6, %f5, {f5};\n    sub.rn.f32 %f6, %f6, {f3};\n    mul.rn.f32 %f6, %f6, %f15;\n    mul.rn.f32 %f28, %f6, {c8};\n");
        Basis(13, $"    mul.rn.f32 %f6, %f5, {f5};\n    sub.rn.f32 %f6, %f6, {f1};\n    mul.rn.f32 %f6, %f6, %f13;\n    mul.rn.f32 %f29, %f6, {c7};\n");
        Basis(14, $"    sub.rn.f32 %f6, %f3, %f4;\n    mul.rn.f32 %f6, %f6, %f15;\n    mul.rn.f32 %f30, %f6, {c9};\n");
        Basis(15, $"    mul.rn.f32 %f6, %f4, {f3};\n    sub.rn.f32 %f6, %f3, %f6;\n    mul.rn.f32 %f6, %f6, %f13;\n    mul.rn.f32 %f31, %f6, {c5};\n");

        // preclamp = sum_bb coeff[i,bb,ch] * basis[bb]  (coeff base elem = i*perPoint + ch, stride numChannels)
        ptx.AppendLine($"    mul.lo.u32 %r8, %r3, {perPoint};");
        ptx.AppendLine("    add.u32 %r8, %r8, %r4;");
        ptx.AppendLine("    mul.wide.u32 %rd5, %r8, 4;");
        ptx.AppendLine("    add.u64 %rd6, %rd0, %rd5;");                  // &coeff[i,0,ch]
        ptx.AppendLine($"    mov.f32 %f7, {zero};");                     // preclamp
        int strideBytes = numChannels * 4;
        for (int b = 0; b < basisCount; b++)
        {
            ptx.AppendLine($"    ld.global.nc.f32 %f8, [%rd6+{b * strideBytes}];");
            ptx.AppendLine($"    fma.rn.f32 %f7, %f8, %f{16 + b}, %f7;");
        }

        // colorGrad = outputGradient[i*numChannels + ch]; zero it if preclamp out of [0,1].
        ptx.AppendLine($"    mul.lo.u32 %r9, %r3, {numChannels};");
        ptx.AppendLine("    add.u32 %r9, %r9, %r4;");
        ptx.AppendLine("    mul.wide.u32 %rd7, %r9, 4;");
        ptx.AppendLine("    add.u64 %rd8, %rd2, %rd7;");
        ptx.AppendLine("    ld.global.nc.f32 %f9, [%rd8];");             // colorGrad
        ptx.AppendLine($"    setp.lt.f32 %p1, %f7, {zero};");
        ptx.AppendLine($"    setp.gt.f32 %p2, %f7, {f1};");
        ptx.AppendLine("    or.pred %p3, %p1, %p2;");
        ptx.AppendLine($"    selp.f32 %f9, {zero}, %f9, %p3;");          // out-of-range -> 0

        // basisB = basis[b] selected by the per-thread b (selp chain, b in [0,basisCount)).
        ptx.AppendLine("    mov.f32 %f10, %f16;");
        for (int k = 1; k < basisCount; k++)
        {
            ptx.AppendLine($"    setp.eq.u32 %p4, %r6, {k};");
            ptx.AppendLine($"    selp.f32 %f10, %f{16 + k}, %f10, %p4;");
        }

        ptx.AppendLine("    mul.rn.f32 %f9, %f9, %f10;");                // colorGrad * basis[b]
        ptx.AppendLine("    mul.wide.u32 %rd10, %r2, 4;");
        ptx.AppendLine("    add.u64 %rd11, %rd9, %rd10;");
        ptx.AppendLine("    st.global.f32 [%rd11], %f9;");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture, int numPoints, int basisCount, int numChannels, int degree, bool broadcastDir)
    {
        var coeffExtent = new DirectPtxExtent(numPoints * basisCount * numChannels);
        var dirExtent = new DirectPtxExtent((broadcastDir ? 1 : numPoints) * 3);
        var gradExtent = new DirectPtxExtent(numPoints * numChannels);
        var shGradExtent = new DirectPtxExtent(numPoints * basisCount * numChannels);
        return new DirectPtxKernelBlueprint(
            Operation: "spherical-harmonics-backward",
            Version: 1,
            Architecture: architecture,
            Variant: $"fp32-p{numPoints}-b{basisCount}-c{numChannels}-deg{degree}-bc{(broadcastDir ? 1 : 0)}",
            Tensors:
            [
                new("shCoefficients", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    coeffExtent, coeffExtent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("viewDirections", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    dirExtent, dirExtent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("outputGradient", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    gradExtent, gradExtent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("shGrad", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    shGradExtent, shGradExtent, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 64,
                MaxStaticSharedBytes: 0,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 1),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["formula"] = "shGrad[i,b,ch] = clampMask(preclamp) * outputGradient[i,ch] * basis_b(normalize(dir_i))",
                ["clamp-mask"] = "gradient zeroed when the forward pre-clamp value is outside [0,1]",
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
        if (basisCount > (degree + 1) * (degree + 1)) return false;
        long count = (long)numPoints * basisCount * numChannels;
        return count > 0 && count % BlockThreads == 0 && count <= MaxCount;
    }

    internal static bool IsPromotedShape(int numPoints, int basisCount, int numChannels, int degree) => false;

    private static void ValidateShape(int numPoints, int basisCount, int numChannels, int degree)
    {
        if (!IsSupportedShape(numPoints, basisCount, numChannels, degree))
            throw new ArgumentOutOfRangeException(
                nameof(basisCount),
                $"Spherical harmonics backward requires degree in [0,{MaxDegree}], basisCount<=(degree+1)^2, and (numPoints*basisCount*numChannels) a multiple of {BlockThreads} up to {MaxCount}.");
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
