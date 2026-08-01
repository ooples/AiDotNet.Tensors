using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Quantum Ry-rotation layer (issue #854), matching the NVRTC <c>quantum_rotation</c> kernel:
/// starting from the input state, apply an Ry rotation gate to each qubit in turn. One block owns one
/// batch element; the 256 threads first copy the input state into the output, then for each qubit q
/// (0..numQubits-1) apply the 2x2 Ry gate
/// <c>[[cos(t/2), -sin(t/2)], [sin(t/2), cos(t/2)]]</c> to every amplitude pair (idx0, idx0+2^q),
/// with dependency-scoped synchronization between qubit steps. Because <c>numQubits</c> is baked,
/// the qubit loop is fully unrolled; each butterfly step's index pairs are disjoint. One thread per
/// qubit computes the uniform <c>cos</c>/<c>sin</c> pair into a small block cache, avoiding redundant
/// transcendental work in every thread. The first four inter-step dependencies stay within a warp;
/// later steps use block barriers. Eligible state rows are staged in block shared memory so all
/// gates reuse on-chip data and global output is written once. <c>cos</c>/<c>sin</c> use the hardware
/// <c>cos.approx.f32</c>/<c>sin.approx.f32</c> (accurate for the typical rotation-angle range).
///
/// State layout is split real/imag <c>[batch, 2^numQubits]</c>; <c>angles</c> is <c>[numQubits]</c>
/// shared across the batch. Shape is baked into the PTX; the launch takes buffer pointers only.
/// One block per batch element (grid = batchSize), 256 threads/block.
/// </summary>
internal sealed class PtxQuantumRotationKernel : IDisposable
{
    internal const int BlockThreads = 256;
    internal const int MaxBatch = 2048 * 4096;
    internal const int MaxQubits = 15;   // dim = 2^numQubits <= 32768
    internal const int MaxStagedDim = 4096;
    internal const string EntryPoint = "aidotnet_quantum_rotation";

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int NumQubits { get; }
    internal int BatchSize { get; }
    internal int Dim => 1 << NumQubits;
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxQuantumRotationKernel(DirectPtxRuntime runtime, int numQubits, int batchSize)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasValidatedScientific(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The checked-in quantum-rotation specialization is measured only on GA10x/SM86.");
        ValidateShape(numQubits, batchSize);
        NumQubits = numQubits;
        BatchSize = batchSize;
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, numQubits, batchSize);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor, numQubits, batchSize);
        _module = runtime.LoadModule(Ptx);
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo info);
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, BlockThreads);
        Blueprint.ResourceBudget.Validate(EntryPoint, info, BlockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Ptx, info, BlockThreads, activeBlocks, _module);
    }

    internal unsafe void Launch(
        DirectPtxTensorView stateReal, DirectPtxTensorView stateImag,
        DirectPtxTensorView outReal, DirectPtxTensorView outImag, DirectPtxTensorView angles)
    {
        DirectPtxAbi.Require(stateReal, Blueprint.Tensors[0], nameof(stateReal));
        DirectPtxAbi.Require(stateImag, Blueprint.Tensors[1], nameof(stateImag));
        DirectPtxAbi.Require(outReal, Blueprint.Tensors[2], nameof(outReal));
        DirectPtxAbi.Require(outImag, Blueprint.Tensors[3], nameof(outImag));
        DirectPtxAbi.Require(angles, Blueprint.Tensors[4], nameof(angles));

        IntPtr stateRealPointer = stateReal.Pointer;
        IntPtr stateImagPointer = stateImag.Pointer;
        IntPtr outRealPointer = outReal.Pointer;
        IntPtr outImagPointer = outImag.Pointer;
        IntPtr anglesPointer = angles.Pointer;
        void** arguments = stackalloc void*[5];
        arguments[0] = &stateRealPointer;
        arguments[1] = &stateImagPointer;
        arguments[2] = &outRealPointer;
        arguments[3] = &outImagPointer;
        arguments[4] = &anglesPointer;
        _module.Launch(_function, (uint)BatchSize, 1, 1, BlockThreads, 1, 1, 0, arguments);
    }

    public void Dispose() => _module.Dispose();

    internal static string EmitPtx(int ccMajor, int ccMinor, int numQubits, int batchSize)
    {
        ValidateShape(numQubits, batchSize);
        int dim = 1 << numQubits;
        int half = dim / 2;
        bool stageState = dim <= MaxStagedDim;
        const string oneHalf = "0f3F000000";   // 0.5

        var ptx = new StringBuilder(8_000);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine($"// quantum-rotation qubits={numQubits} dim={dim} batch={batchSize}");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 sr_ptr,");
        ptx.AppendLine("    .param .u64 si_ptr,");
        ptx.AppendLine("    .param .u64 or_ptr,");
        ptx.AppendLine("    .param .u64 oi_ptr,");
        ptx.AppendLine("    .param .u64 ang_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine($".maxntid {BlockThreads}, 1, 1");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<4>;");
        ptx.AppendLine("    .reg .b32 %r<16>;");
        ptx.AppendLine("    .reg .b64 %rd<28>;");
        ptx.AppendLine("    .reg .f32 %f<20>;");
        ptx.AppendLine($"    .shared .align 8 .b8 trig[{numQubits * 2 * sizeof(float)}];");
        if (stageState)
        {
            ptx.AppendLine($"    .shared .align 16 .b8 state_real[{dim * sizeof(float)}];");
            ptx.AppendLine($"    .shared .align 16 .b8 state_imag[{dim * sizeof(float)}];");
        }
        ptx.AppendLine("    ld.param.u64 %rd0, [sr_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [si_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [or_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd3, [oi_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd4, [ang_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");                      // b (batch row)
        ptx.AppendLine($"    mul.wide.u32 %rd5, %r1, {dim * sizeof(float)};");
        ptx.AppendLine("    add.u64 %rd6, %rd0, %rd5;");                   // srBase
        ptx.AppendLine("    add.u64 %rd7, %rd1, %rd5;");                   // siBase
        ptx.AppendLine("    add.u64 %rd8, %rd2, %rd5;");                   // orBase
        ptx.AppendLine("    add.u64 %rd9, %rd3, %rd5;");                   // oiBase
        if (stageState)
        {
            ptx.AppendLine("    mov.u64 %rd21, state_real;");
            ptx.AppendLine("    mov.u64 %rd22, state_imag;");
        }

        // Each gate is uniform across the block: compute its trig pair once and cache it.
        for (int q = 0; q < numQubits; q++)
        {
            int trigOffset = q * 2 * sizeof(float);
            ptx.AppendLine($"    setp.eq.u32 %p3, %r0, {q};");
            ptx.AppendLine($"    @%p3 ld.global.nc.f32 %f2, [%rd4+{q * sizeof(float)}];");
            ptx.AppendLine($"    @%p3 mul.rn.f32 %f3, %f2, {oneHalf};");
            ptx.AppendLine("    @%p3 cos.approx.f32 %f4, %f3;");
            ptx.AppendLine("    @%p3 sin.approx.f32 %f5, %f3;");
            ptx.AppendLine($"    @%p3 st.shared.f32 [trig+{trigOffset}], %f4;");
            ptx.AppendLine($"    @%p3 st.shared.f32 [trig+{trigOffset + sizeof(float)}], %f5;");
        }

        // Copy input state into the working row (strided i=tid; i<dim; i+=256).
        ptx.AppendLine("    mov.u32 %r2, %r0;");
        ptx.AppendLine("$QR_INIT:");
        ptx.AppendLine($"    setp.ge.u32 %p0, %r2, {dim};");
        ptx.AppendLine("    @%p0 bra $QR_INIT_END;");
        ptx.AppendLine("    mul.wide.u32 %rd10, %r2, 4;");
        ptx.AppendLine("    add.u64 %rd11, %rd6, %rd10;");
        ptx.AppendLine("    add.u64 %rd12, %rd7, %rd10;");
        ptx.AppendLine(stageState
            ? "    add.u64 %rd13, %rd21, %rd10;"
            : "    add.u64 %rd13, %rd8, %rd10;");
        ptx.AppendLine(stageState
            ? "    add.u64 %rd14, %rd22, %rd10;"
            : "    add.u64 %rd14, %rd9, %rd10;");
        ptx.AppendLine("    ld.global.nc.f32 %f0, [%rd11];");
        ptx.AppendLine("    ld.global.nc.f32 %f1, [%rd12];");
        ptx.AppendLine(stageState
            ? "    st.shared.f32 [%rd13], %f0;"
            : "    st.global.f32 [%rd13], %f0;");
        ptx.AppendLine(stageState
            ? "    st.shared.f32 [%rd14], %f1;"
            : "    st.global.f32 [%rd14], %f1;");
        ptx.AppendLine($"    add.u32 %r2, %r2, {BlockThreads};");
        ptx.AppendLine("    bra.uni $QR_INIT;");
        ptx.AppendLine("$QR_INIT_END:");
        ptx.AppendLine("    bar.sync 0;");

        // Unrolled per-qubit Ry rotation.
        for (int q = 0; q < numQubits; q++)
        {
            int stride = 1 << q;
            int trigOffset = q * 2 * sizeof(float);
            ptx.AppendLine($"    // --- qubit {q} (stride {stride}) ---");
            ptx.AppendLine($"    ld.shared.f32 %f4, [trig+{trigOffset}];");
            ptx.AppendLine($"    ld.shared.f32 %f5, [trig+{trigOffset + sizeof(float)}];");
            ptx.AppendLine("    mov.u32 %r2, %r0;");                                    // i = tid
            ptx.AppendLine($"$QR_STEP{q}:");
            ptx.AppendLine($"    setp.ge.u32 %p1, %r2, {half};");
            ptx.AppendLine($"    @%p1 bra $QR_STEP{q}_END;");
            // idx0 = ((i >> q) << (q+1)) | (i & (stride-1)) ; idx1 = idx0 + stride
            ptx.AppendLine($"    shr.u32 %r3, %r2, {q};");
            ptx.AppendLine($"    shl.b32 %r4, %r3, {q + 1};");
            ptx.AppendLine($"    and.b32 %r5, %r2, {stride - 1};");
            ptx.AppendLine("    add.u32 %r4, %r4, %r5;");                               // idx0
            ptx.AppendLine($"    add.u32 %r6, %r4, {stride};");                         // idx1
            ptx.AppendLine("    mul.wide.u32 %rd15, %r4, 4;");
            ptx.AppendLine("    mul.wide.u32 %rd16, %r6, 4;");
            ptx.AppendLine(stageState ? "    add.u64 %rd17, %rd21, %rd15;" : "    add.u64 %rd17, %rd8, %rd15;");
            ptx.AppendLine(stageState ? "    add.u64 %rd18, %rd22, %rd15;" : "    add.u64 %rd18, %rd9, %rd15;");
            ptx.AppendLine(stageState ? "    add.u64 %rd19, %rd21, %rd16;" : "    add.u64 %rd19, %rd8, %rd16;");
            ptx.AppendLine(stageState ? "    add.u64 %rd20, %rd22, %rd16;" : "    add.u64 %rd20, %rd9, %rd16;");
            string workingLoad = stageState ? "ld.shared.f32" : "ld.global.f32";
            ptx.AppendLine($"    {workingLoad} %f6, [%rd17];");  // r0
            ptx.AppendLine($"    {workingLoad} %f7, [%rd18];");  // i0
            ptx.AppendLine($"    {workingLoad} %f8, [%rd19];");  // r1
            ptx.AppendLine($"    {workingLoad} %f9, [%rd20];");  // i1
            // nor0 = cosHalf*r0 - sinHalf*r1
            ptx.AppendLine("    mul.rn.f32 %f10, %f5, %f8;");
            ptx.AppendLine("    mul.rn.f32 %f11, %f4, %f6;");
            ptx.AppendLine("    sub.rn.f32 %f11, %f11, %f10;");
            // noi0 = cosHalf*i0 - sinHalf*i1
            ptx.AppendLine("    mul.rn.f32 %f12, %f5, %f9;");
            ptx.AppendLine("    mul.rn.f32 %f13, %f4, %f7;");
            ptx.AppendLine("    sub.rn.f32 %f13, %f13, %f12;");
            // nor1 = sinHalf*r0 + cosHalf*r1
            ptx.AppendLine("    mul.rn.f32 %f14, %f5, %f6;");
            ptx.AppendLine("    fma.rn.f32 %f14, %f4, %f8, %f14;");
            // noi1 = sinHalf*i0 + cosHalf*i1
            ptx.AppendLine("    mul.rn.f32 %f15, %f5, %f7;");
            ptx.AppendLine("    fma.rn.f32 %f15, %f4, %f9, %f15;");
            string workingStore = stageState ? "st.shared.f32" : "st.global.f32";
            ptx.AppendLine($"    {workingStore} [%rd17], %f11;");
            ptx.AppendLine($"    {workingStore} [%rd18], %f13;");
            ptx.AppendLine($"    {workingStore} [%rd19], %f14;");
            ptx.AppendLine($"    {workingStore} [%rd20], %f15;");
            ptx.AppendLine($"    add.u32 %r2, %r2, {BlockThreads};");
            ptx.AppendLine($"    bra.uni $QR_STEP{q};");
            ptx.AppendLine($"$QR_STEP{q}_END:");
            if (q < numQubits - 1)
                ptx.AppendLine(q < 4 ? "    bar.warp.sync 0xffffffff;" : "    bar.sync 0;");
            else if (stageState)
                ptx.AppendLine("    bar.sync 0;");
        }

        if (stageState)
        {
            // Publish the final shared row to global output exactly once.
            ptx.AppendLine("    mov.u32 %r2, %r0;");
            ptx.AppendLine("$QR_STORE:");
            ptx.AppendLine($"    setp.ge.u32 %p2, %r2, {dim};");
            ptx.AppendLine("    @%p2 bra $QR_STORE_END;");
            ptx.AppendLine("    mul.wide.u32 %rd10, %r2, 4;");
            ptx.AppendLine("    add.u64 %rd11, %rd21, %rd10;");
            ptx.AppendLine("    add.u64 %rd12, %rd22, %rd10;");
            ptx.AppendLine("    add.u64 %rd13, %rd8, %rd10;");
            ptx.AppendLine("    add.u64 %rd14, %rd9, %rd10;");
            ptx.AppendLine("    ld.shared.f32 %f0, [%rd11];");
            ptx.AppendLine("    ld.shared.f32 %f1, [%rd12];");
            ptx.AppendLine("    st.global.f32 [%rd13], %f0;");
            ptx.AppendLine("    st.global.f32 [%rd14], %f1;");
            ptx.AppendLine($"    add.u32 %r2, %r2, {BlockThreads};");
            ptx.AppendLine("    bra.uni $QR_STORE;");
            ptx.AppendLine("$QR_STORE_END:");
        }
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static DirectPtxKernelBlueprint CreateBlueprint(DirectPtxArchitectureFamily architecture, int numQubits, int batchSize)
    {
        int dim = 1 << numQubits;
        bool stageState = dim <= MaxStagedDim;
        var stateExtent = new DirectPtxExtent(batchSize * dim);
        var anglesExtent = new DirectPtxExtent(numQubits);
        return new DirectPtxKernelBlueprint(
            Operation: "quantum-rotation",
            Version: 3,
            Architecture: architecture,
            Variant: $"fp32-{(stageState ? "shared-state" : "global-state")}-q{numQubits}-b{batchSize}",
            Tensors:
            [
                new("stateReal", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    stateExtent, stateExtent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("stateImag", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    stateExtent, stateExtent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("outReal", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    stateExtent, stateExtent, 16, DirectPtxTensorAccess.ReadWrite, DirectPtxExtentMode.Exact),
                new("outImag", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    stateExtent, stateExtent, 16, DirectPtxTensorAccess.ReadWrite, DirectPtxExtentMode.Exact),
                new("angles", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    anglesExtent, anglesExtent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: DirectPtxResourceBudget.FromDriverMeasurement(
                measuredRegistersPerThread: 36,
                maxStaticSharedBytes: (numQubits * 2 + (stageState ? dim * 2 : 0)) * sizeof(float),
                maxLocalBytesPerThread: 0,
                minBlocksPerMultiprocessor: 1),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["formula"] = "apply Ry(angles[q]) to each qubit q in turn; Ry = [[c,-s],[s,c]], c=cos(t/2), s=sin(t/2)",
                ["approximation"] = "cos/sin via cos.approx.f32 / sin.approx.f32",
                ["gate-cache"] = "one thread per qubit computes cos/sin into block-shared storage",
                ["state-staging"] = stageState
                    ? "split-complex row staged in shared memory; one global input read and final output write"
                    : "global in-place working row for dimensions above the shared-memory ceiling",
                ["barriers"] = "warp barriers after q0-q3; block barriers before cross-warp stages; staged rows synchronize before final publish",
                ["temporary-device-allocation"] = "none",
                ["stride-parameters"] = "none"
            });
    }

    internal static bool IsSupportedShape(int numQubits, int batchSize)
    {
        if (numQubits < 1 || numQubits > MaxQubits || batchSize <= 0) return false;
        long dim = 1L << numQubits;
        return checked((long)batchSize * dim) <= MaxBatch;
    }

    internal static bool IsPromotedShape(int numQubits, int batchSize) => false;

    private static void ValidateShape(int numQubits, int batchSize)
    {
        if (!IsSupportedShape(numQubits, batchSize))
            throw new ArgumentOutOfRangeException(
                nameof(numQubits),
                $"Quantum rotation requires numQubits in [1,{MaxQubits}] and batchSize*2^numQubits <= {MaxBatch}.");
    }

}
