using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Exact-shape FP32 lower Cholesky for batches of contiguous 4x4 matrices.
/// One thread owns a matrix and retains its complete factorization in registers.
/// </summary>
internal sealed class PtxRegisterCholesky4x4F32Kernel : IDisposable
{
    internal const int MatrixOrder = 4;
    internal const int MatrixElements = MatrixOrder * MatrixOrder;
    internal const int DefaultBlockThreads = DirectPtxSolver4x4Autotuner.DefaultBlockThreads;
    internal const string EntryPoint = "aidotnet_register_cholesky_4x4_f32";

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int BatchCount { get; }
    internal int BlockThreads { get; }
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxRegisterCholesky4x4F32Kernel(
        DirectPtxRuntime runtime,
        int batchCount,
        int blockThreads = DefaultBlockThreads)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.IsCholesky4x4ExperimentArchitecture(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The checked-in register Cholesky specialization is admitted only on SM86.");
        ValidateBatchCount(batchCount);
        DirectPtxSolver4x4Autotuner.ValidateBlockThreads(blockThreads);

        BatchCount = batchCount;
        BlockThreads = blockThreads;
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, batchCount, blockThreads);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor, batchCount, blockThreads);
        _module = runtime.LoadModule(Ptx);
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo info);
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, BlockThreads);
        Blueprint.ResourceBudget.Validate(EntryPoint, info, BlockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Ptx, info,
            BlockThreads, activeBlocks, _module.JitInfoLog);
    }

    internal unsafe void Launch(
        DirectPtxTensorView input,
        DirectPtxTensorView output,
        DirectPtxTensorView info)
    {
        Require(input, Blueprint.Tensors[0], nameof(input));
        Require(output, Blueprint.Tensors[1], nameof(output));
        Require(info, Blueprint.Tensors[2], nameof(info));
        if (Overlaps(input, output) || Overlaps(input, info) || Overlaps(output, info))
            throw new ArgumentException("Register Cholesky input, output, and info allocations must be disjoint.");

        IntPtr inputPointer = input.Pointer;
        IntPtr outputPointer = output.Pointer;
        IntPtr infoPointer = info.Pointer;
        void** arguments = stackalloc void*[3];
        arguments[0] = &inputPointer;
        arguments[1] = &outputPointer;
        arguments[2] = &infoPointer;
        _module.Launch(
            _function,
            checked((uint)(BatchCount / BlockThreads)), 1, 1,
            checked((uint)BlockThreads), 1, 1, 0, arguments);
    }

    public void Dispose() => _module.Dispose();

    internal static bool IsSupportedBatchCount(int batchCount) =>
        batchCount is 1024 or 4096 or 16384 or 65536;

    internal static bool IsPromotedShape(int batchCount) => false;

    internal static string EmitPtx(
        int ccMajor,
        int ccMinor,
        int batchCount,
        int blockThreads = DefaultBlockThreads)
    {
        ValidateBatchCount(batchCount);
        DirectPtxSolver4x4Autotuner.ValidateBlockThreads(blockThreads);
        var ptx = new StringBuilder(8_192);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 input_ptr,");
        ptx.AppendLine("    .param .u64 output_ptr,");
        ptx.AppendLine("    .param .u64 info_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<2>;");
        ptx.AppendLine("    .reg .b32 %r<8>;");
        ptx.AppendLine("    .reg .b64 %rd<10>;");
        ptx.AppendLine("    .reg .f32 %f<24>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [input_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [output_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [info_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {blockThreads}, %r0;");
        ptx.AppendLine("    mul.wide.u32 %rd3, %r2, 64;");
        ptx.AppendLine("    add.u64 %rd4, %rd0, %rd3;");
        ptx.AppendLine("    add.u64 %rd5, %rd1, %rd3;");
        ptx.AppendLine("    mul.wide.u32 %rd6, %r2, 4;");
        ptx.AppendLine("    add.u64 %rd7, %rd2, %rd6;");
        for (int row = 0; row < 4; row++)
            ptx.AppendLine($"    ld.global.v4.f32 {{%f{row * 4},%f{row * 4 + 1},%f{row * 4 + 2},%f{row * 4 + 3}}}, [%rd4+{row * 16}];");
        ptx.AppendLine("    mov.u32 %r3, 0;");
        ptx.AppendLine("    setp.le.f32 %p0, %f0, 0f00000000;");
        ptx.AppendLine("    @%p0 bra FAIL_1;");
        ptx.AppendLine("    sqrt.rn.f32 %f0, %f0;");
        ptx.AppendLine("    div.rn.f32 %f4, %f4, %f0;");
        ptx.AppendLine("    div.rn.f32 %f8, %f8, %f0;");
        ptx.AppendLine("    div.rn.f32 %f12, %f12, %f0;");
        ptx.AppendLine("    fma.rn.f32 %f5, -%f4, %f4, %f5;");
        ptx.AppendLine("    setp.le.f32 %p0, %f5, 0f00000000;");
        ptx.AppendLine("    @%p0 bra FAIL_2;");
        ptx.AppendLine("    sqrt.rn.f32 %f5, %f5;");
        ptx.AppendLine("    fma.rn.f32 %f9, -%f8, %f4, %f9;");
        ptx.AppendLine("    div.rn.f32 %f9, %f9, %f5;");
        ptx.AppendLine("    fma.rn.f32 %f13, -%f12, %f4, %f13;");
        ptx.AppendLine("    div.rn.f32 %f13, %f13, %f5;");
        ptx.AppendLine("    fma.rn.f32 %f10, -%f8, %f8, %f10;");
        ptx.AppendLine("    fma.rn.f32 %f10, -%f9, %f9, %f10;");
        ptx.AppendLine("    setp.le.f32 %p0, %f10, 0f00000000;");
        ptx.AppendLine("    @%p0 bra FAIL_3;");
        ptx.AppendLine("    sqrt.rn.f32 %f10, %f10;");
        ptx.AppendLine("    fma.rn.f32 %f14, -%f12, %f8, %f14;");
        ptx.AppendLine("    fma.rn.f32 %f14, -%f13, %f9, %f14;");
        ptx.AppendLine("    div.rn.f32 %f14, %f14, %f10;");
        ptx.AppendLine("    fma.rn.f32 %f15, -%f12, %f12, %f15;");
        ptx.AppendLine("    fma.rn.f32 %f15, -%f13, %f13, %f15;");
        ptx.AppendLine("    fma.rn.f32 %f15, -%f14, %f14, %f15;");
        ptx.AppendLine("    setp.le.f32 %p0, %f15, 0f00000000;");
        ptx.AppendLine("    @%p0 bra FAIL_4;");
        ptx.AppendLine("    sqrt.rn.f32 %f15, %f15;");
        ptx.AppendLine("    bra.uni STORE;");
        for (int failure = 1; failure <= 4; failure++)
        {
            ptx.AppendLine($"FAIL_{failure}:");
            int diagonal = (failure - 1) * 5;
            ptx.AppendLine($"    mov.f32 %f{diagonal}, 0f00000000;");
            ptx.AppendLine($"    mov.u32 %r3, {failure};");
            ptx.AppendLine("    bra.uni STORE;");
        }
        ptx.AppendLine("STORE:");
        ptx.AppendLine("    mov.f32 %f16, 0f00000000;");
        ptx.AppendLine("    st.global.v4.f32 [%rd5+0], {%f0,%f16,%f16,%f16};");
        ptx.AppendLine("    st.global.v4.f32 [%rd5+16], {%f4,%f5,%f16,%f16};");
        ptx.AppendLine("    st.global.v4.f32 [%rd5+32], {%f8,%f9,%f10,%f16};");
        ptx.AppendLine("    st.global.v4.f32 [%rd5+48], {%f12,%f13,%f14,%f15};");
        ptx.AppendLine("    st.global.u32 [%rd7], %r3;");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture,
        int batchCount,
        int blockThreads)
    {
        var matrices = new DirectPtxExtent(batchCount, MatrixOrder, MatrixOrder);
        var info = new DirectPtxExtent(batchCount);
        return new DirectPtxKernelBlueprint(
            Operation: "cholesky-lower-4x4-f32",
            Version: 1,
            Architecture: architecture,
            Variant: $"register-batch{batchCount}-block{blockThreads}",
            Tensors:
            [
                new("input", DirectPtxPhysicalType.Float32,
                    DirectPtxPhysicalLayout.BatchedRowMajorMatrix,
                    matrices, matrices, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("output", DirectPtxPhysicalType.Float32,
                    DirectPtxPhysicalLayout.BatchedRowMajorMatrix,
                    matrices, matrices, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact),
                new("info", DirectPtxPhysicalType.Int32, DirectPtxPhysicalLayout.Vector,
                    info, info, 4, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 40,
                MaxStaticSharedBytes: 0,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 4),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["factorization"] = "A=L*transpose(L)",
                ["triangle"] = "lower-output-upper-zero",
                ["failure"] = "info=first-non-positive-leading-minor; later-lower-elements-preserve-input",
                ["ownership"] = "one-thread-per-matrix-register-resident",
                ["global-reads"] = "one-16-float-input-read-per-matrix",
                ["global-writes"] = "one-16-float-output-write-plus-one-info-write-per-matrix",
                ["global-intermediates"] = "none",
                ["temporary-device-allocation"] = "none",
                ["stride-parameters"] = "none"
            });
    }

    private static void ValidateBatchCount(int batchCount)
    {
        if (!IsSupportedBatchCount(batchCount))
            throw new ArgumentOutOfRangeException(nameof(batchCount),
                "Supported exact batch buckets are 1024, 4096, 16384, and 65536.");
    }

    private static void Require(
        DirectPtxTensorView view,
        DirectPtxTensorContract contract,
        string parameter)
    {
        if (view.Pointer == IntPtr.Zero || view.PhysicalType != contract.PhysicalType ||
            view.Layout != contract.Layout || view.LogicalExtent != contract.LogicalExtent ||
            view.PhysicalExtent != contract.PhysicalExtent || view.ByteLength != contract.RequiredBytes)
            throw new ArgumentException(
                $"{parameter} does not satisfy physical ABI '{contract.Name}'.", parameter);
    }

    private static bool Overlaps(DirectPtxTensorView left, DirectPtxTensorView right)
    {
        nuint leftStart = PtxCompat.ToNuint(left.Pointer);
        nuint rightStart = PtxCompat.ToNuint(right.Pointer);
        nuint leftEnd = checked(leftStart + left.ByteLength);
        nuint rightEnd = checked(rightStart + right.ByteLength);
        return leftStart < rightEnd && rightStart < leftEnd;
    }
}
