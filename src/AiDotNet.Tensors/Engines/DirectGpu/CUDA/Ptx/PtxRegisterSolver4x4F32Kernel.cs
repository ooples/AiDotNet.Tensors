#if NET5_0_OR_GREATER
using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

internal enum DirectPtxSolver4x4Operation
{
    LuFactor,
    QrReduced,
    EighUpper,
    EighLower,
    SvdReduced,
    LuSolveVector,
    LdlFactorLower,
    LdlSolveVectorLower,
    GeneralSolveVector,
    TriangularSolveVectorLower,
    TriangularSolveVectorUpper,
    CholeskyBackwardLower,
    SolveBackwardVector
}

/// <summary>
/// Pointer-only, exact-shape register kernels for the remaining 4x4 dense
/// solver family. Each thread owns one matrix; no workspace or global
/// intermediate is materialized.
/// </summary>
internal sealed class PtxRegisterSolver4x4F32Kernel : IDisposable
{
    internal const int Order = 4;
    internal const int MatrixElements = 16;
    internal const int DefaultBlockThreads = DirectPtxSolver4x4Autotuner.DefaultBlockThreads;

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal DirectPtxSolver4x4Operation Operation { get; }
    internal int BatchCount { get; }
    internal int BlockThreads { get; }
    internal string EntryPoint { get; }
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxRegisterSolver4x4F32Kernel(
        DirectPtxRuntime runtime,
        DirectPtxSolver4x4Operation operation,
        int batchCount,
        int blockThreads = DefaultBlockThreads)
    {
        ArgumentNullException.ThrowIfNull(runtime);
        if (!DirectPtxArchitecture.IsSolver4x4ExperimentArchitecture(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The checked-in register solver specializations are admitted only on SM86.");
        ValidateBatchCount(batchCount);
        DirectPtxSolver4x4Autotuner.ValidateBlockThreads(blockThreads);
        Operation = operation;
        BatchCount = batchCount;
        BlockThreads = blockThreads;
        EntryPoint = EntryPointFor(operation);
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, operation, batchCount, blockThreads);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor, operation, batchCount, blockThreads);
        _module = runtime.LoadModule(Ptx);
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo info);
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, BlockThreads);
        Blueprint.ResourceBudget.Validate(EntryPoint, info, BlockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Ptx, info,
            BlockThreads, activeBlocks, _module.JitInfoLog);
    }

    internal unsafe void Launch3(
        DirectPtxTensorView first,
        DirectPtxTensorView second,
        DirectPtxTensorView third)
    {
        if (Blueprint.Tensors.Count != 3) throw new InvalidOperationException("Kernel ABI is not three pointers.");
        Require(first, Blueprint.Tensors[0], nameof(first));
        Require(second, Blueprint.Tensors[1], nameof(second));
        Require(third, Blueprint.Tensors[2], nameof(third));
        if (Overlaps(first, second) || Overlaps(first, third) || Overlaps(second, third))
            throw new ArgumentException("Register solver inputs and outputs must use disjoint allocations.");
        IntPtr p0 = first.Pointer, p1 = second.Pointer, p2 = third.Pointer;
        void** args = stackalloc void*[3];
        args[0] = &p0; args[1] = &p1; args[2] = &p2;
        Launch(args);
    }

    internal unsafe void Launch4(
        DirectPtxTensorView first,
        DirectPtxTensorView second,
        DirectPtxTensorView third,
        DirectPtxTensorView fourth)
    {
        if (Blueprint.Tensors.Count != 4) throw new InvalidOperationException("Kernel ABI is not four pointers.");
        Require(first, Blueprint.Tensors[0], nameof(first));
        Require(second, Blueprint.Tensors[1], nameof(second));
        Require(third, Blueprint.Tensors[2], nameof(third));
        Require(fourth, Blueprint.Tensors[3], nameof(fourth));
        if (Overlaps(first, second) || Overlaps(first, third) || Overlaps(first, fourth) ||
            Overlaps(second, third) || Overlaps(second, fourth) || Overlaps(third, fourth))
            throw new ArgumentException("Register solver inputs and outputs must use disjoint allocations.");
        IntPtr p0 = first.Pointer, p1 = second.Pointer, p2 = third.Pointer, p3 = fourth.Pointer;
        void** args = stackalloc void*[4];
        args[0] = &p0; args[1] = &p1; args[2] = &p2; args[3] = &p3;
        Launch(args);
    }

    internal unsafe void Launch5(
        DirectPtxTensorView first,
        DirectPtxTensorView second,
        DirectPtxTensorView third,
        DirectPtxTensorView fourth,
        DirectPtxTensorView fifth)
    {
        if (Blueprint.Tensors.Count != 5) throw new InvalidOperationException("Kernel ABI is not five pointers.");
        DirectPtxTensorView[] views = [first, second, third, fourth, fifth];
        for (int i = 0; i < views.Length; i++) Require(views[i], Blueprint.Tensors[i], $"argument{i}");
        for (int i = 0; i < views.Length; i++)
        for (int j = i + 1; j < views.Length; j++)
            if (Overlaps(views[i], views[j]))
                throw new ArgumentException("Register solver inputs and outputs must use disjoint allocations.");
        IntPtr p0 = first.Pointer, p1 = second.Pointer, p2 = third.Pointer,
            p3 = fourth.Pointer, p4 = fifth.Pointer;
        void** args = stackalloc void*[5];
        args[0] = &p0; args[1] = &p1; args[2] = &p2; args[3] = &p3; args[4] = &p4;
        Launch(args);
    }

    private unsafe void Launch(void** arguments) => _module.Launch(
        _function, checked((uint)(BatchCount / BlockThreads)), 1, 1,
        checked((uint)BlockThreads), 1, 1, 0, arguments);

    public void Dispose() => _module.Dispose();

    internal static bool IsSupportedBatchCount(int batchCount) =>
        batchCount is 1024 or 4096 or 16384 or 65536;

    internal static bool IsPromotedShape(DirectPtxSolver4x4Operation operation, int batchCount) => false;

    internal static string EmitPtx(
        int ccMajor,
        int ccMinor,
        DirectPtxSolver4x4Operation operation,
        int batchCount,
        int blockThreads = DefaultBlockThreads)
    {
        ValidateBatchCount(batchCount);
        DirectPtxSolver4x4Autotuner.ValidateBlockThreads(blockThreads);
        var ptx = new StringBuilder(operation is DirectPtxSolver4x4Operation.SvdReduced or
            DirectPtxSolver4x4Operation.CholeskyBackwardLower ? 48_000 : 32_000);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine();
        string[] parameters = ParameterNames(operation);
        ptx.AppendLine($".visible .entry {EntryPointFor(operation)}(");
        for (int i = 0; i < parameters.Length; i++)
            ptx.AppendLine($"    .param .u64 {parameters[i]}{(i + 1 == parameters.Length ? string.Empty : ",")}");
        ptx.AppendLine(")");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<4>;");
        ptx.AppendLine("    .reg .b32 %r<16>;");
        ptx.AppendLine("    .reg .b64 %rd<24>;");
        ptx.AppendLine(operation switch
        {
            DirectPtxSolver4x4Operation.SvdReduced => "    .reg .f32 %f<80>;",
            DirectPtxSolver4x4Operation.CholeskyBackwardLower => "    .reg .f32 %f<104>;",
            _ => "    .reg .f32 %f<64>;"
        });
        for (int i = 0; i < parameters.Length; i++)
            ptx.AppendLine($"    ld.param.u64 %rd{i}, [{parameters[i]}];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {blockThreads}, %r0;");
        switch (operation)
        {
            case DirectPtxSolver4x4Operation.LuFactor: EmitLu(ptx); break;
            case DirectPtxSolver4x4Operation.QrReduced: EmitQr(ptx); break;
            case DirectPtxSolver4x4Operation.EighUpper: EmitEigh(ptx, upper: true); break;
            case DirectPtxSolver4x4Operation.EighLower: EmitEigh(ptx, upper: false); break;
            case DirectPtxSolver4x4Operation.SvdReduced: EmitSvd(ptx); break;
            case DirectPtxSolver4x4Operation.LuSolveVector: EmitLuSolve(ptx); break;
            case DirectPtxSolver4x4Operation.LdlFactorLower: EmitLdlFactor(ptx); break;
            case DirectPtxSolver4x4Operation.LdlSolveVectorLower: EmitLdlSolve(ptx); break;
            case DirectPtxSolver4x4Operation.GeneralSolveVector: EmitGeneralSolve(ptx); break;
            case DirectPtxSolver4x4Operation.TriangularSolveVectorLower: EmitTriangularSolve(ptx, upper: false); break;
            case DirectPtxSolver4x4Operation.TriangularSolveVectorUpper: EmitTriangularSolve(ptx, upper: true); break;
            case DirectPtxSolver4x4Operation.CholeskyBackwardLower: EmitCholeskyBackward(ptx); break;
            case DirectPtxSolver4x4Operation.SolveBackwardVector: EmitSolveBackward(ptx); break;
            default: throw new ArgumentOutOfRangeException(nameof(operation));
        }
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static void EmitLu(StringBuilder ptx)
    {
        EmitMatrixPointer(ptx, 0, 3, 4);
        EmitMatrixPointer(ptx, 1, 3, 5);
        EmitVectorPointer(ptx, 2, 2, 6, 7, 16);
        EmitLoadMatrix(ptx, 4, 0);
        for (int j = 0; j < 4; j++)
        {
            int pivot = j * 4 + j;
            ptx.AppendLine($"    abs.f32 %f48, %f{pivot};");
            ptx.AppendLine($"    mov.u32 %r4, {j};");
            for (int row = j + 1; row < 4; row++)
            {
                ptx.AppendLine($"    abs.f32 %f49, %f{row * 4 + j};");
                ptx.AppendLine("    setp.gt.f32 %p0, %f49, %f48;");
                ptx.AppendLine($"    selp.u32 %r4, {row}, %r4, %p0;");
                ptx.AppendLine("    selp.f32 %f48, %f49, %f48, %p0;");
            }
            ptx.AppendLine($"    st.global.u32 [%rd7+{j * 4}], %r4;");
            for (int row = j + 1; row < 4; row++)
            {
                ptx.AppendLine($"    setp.eq.u32 %p0, %r4, {row};");
                for (int col = 0; col < 4; col++)
                {
                    int a = j * 4 + col, b = row * 4 + col;
                    ptx.AppendLine($"    @%p0 mov.f32 %f50, %f{a};");
                    ptx.AppendLine($"    @%p0 mov.f32 %f{a}, %f{b};");
                    ptx.AppendLine($"    @%p0 mov.f32 %f{b}, %f50;");
                }
            }
            ptx.AppendLine($"    setp.eq.f32 %p0, %f{pivot}, 0f00000000;");
            ptx.AppendLine($"    @%p0 bra LU_SKIP_{j};");
            for (int row = j + 1; row < 4; row++)
            {
                int factor = row * 4 + j;
                ptx.AppendLine($"    div.rn.f32 %f{factor}, %f{factor}, %f{pivot};");
                for (int col = j + 1; col < 4; col++)
                    ptx.AppendLine($"    fma.rn.f32 %f{row * 4 + col}, -%f{factor}, %f{j * 4 + col}, %f{row * 4 + col};");
            }
            ptx.AppendLine($"LU_SKIP_{j}:");
        }
        EmitStoreMatrix(ptx, 5, 0);
    }

    private static void EmitQr(StringBuilder ptx)
    {
        EmitMatrixPointer(ptx, 0, 3, 4);
        EmitMatrixPointer(ptx, 1, 3, 5);
        EmitMatrixPointer(ptx, 2, 3, 6);
        EmitLoadMatrix(ptx, 4, 0);
        for (int i = 16; i < 32; i++) ptx.AppendLine($"    mov.f32 %f{i}, 0f00000000;");
        for (int col = 0; col < 4; col++)
        {
            for (int previous = 0; previous < col; previous++)
            {
                int r = 16 + previous * 4 + col;
                ptx.AppendLine($"    mul.rn.f32 %f{r}, %f{previous}, %f{col};");
                for (int row = 1; row < 4; row++)
                    ptx.AppendLine($"    fma.rn.f32 %f{r}, %f{row * 4 + previous}, %f{row * 4 + col}, %f{r};");
                for (int row = 0; row < 4; row++)
                    ptx.AppendLine($"    fma.rn.f32 %f{row * 4 + col}, -%f{r}, %f{row * 4 + previous}, %f{row * 4 + col};");
            }
            int diag = 16 + col * 4 + col;
            ptx.AppendLine($"    mul.rn.f32 %f{diag}, %f{col}, %f{col};");
            for (int row = 1; row < 4; row++)
                ptx.AppendLine($"    fma.rn.f32 %f{diag}, %f{row * 4 + col}, %f{row * 4 + col}, %f{diag};");
            ptx.AppendLine($"    sqrt.rn.f32 %f{diag}, %f{diag};");
            ptx.AppendLine($"    setp.gt.f32 %p0, %f{diag}, 0f0da24260;");
            ptx.AppendLine($"    @!%p0 bra QR_ZERO_{col};");
            for (int row = 0; row < 4; row++)
                ptx.AppendLine($"    div.rn.f32 %f{row * 4 + col}, %f{row * 4 + col}, %f{diag};");
            ptx.AppendLine($"    bra.uni QR_DONE_{col};");
            ptx.AppendLine($"QR_ZERO_{col}:");
            for (int row = 0; row < 4; row++)
                ptx.AppendLine($"    mov.f32 %f{row * 4 + col}, 0f00000000;");
            ptx.AppendLine($"QR_DONE_{col}:");
        }
        EmitStoreMatrix(ptx, 5, 0);
        EmitStoreMatrix(ptx, 6, 16);
    }

    private static void EmitEigh(StringBuilder ptx, bool upper)
    {
        EmitMatrixPointer(ptx, 0, 3, 4);
        EmitVectorPointer(ptx, 1, 2, 5, 6, 16);
        EmitMatrixPointer(ptx, 2, 3, 7);
        EmitLoadMatrix(ptx, 4, 0);
        if (upper) SymmetrizeUpper(ptx, 0);
        else SymmetrizeLower(ptx, 0);
        EmitIdentity(ptx, 16);
        EmitJacobiSweeps(ptx, 0, 16, "EIGH", 8);
        EmitSortEigenpairs(ptx, 0, 16, ascending: true, "EIGH");
        for (int i = 0; i < 4; i++)
            ptx.AppendLine($"    st.global.f32 [%rd6+{i * 4}], %f{i * 5};");
        EmitStoreMatrix(ptx, 7, 16);
    }

    private static void EmitSvd(StringBuilder ptx)
    {
        EmitMatrixPointer(ptx, 0, 3, 4);
        EmitMatrixPointer(ptx, 1, 3, 5);
        EmitVectorPointer(ptx, 2, 2, 6, 7, 16);
        EmitMatrixPointer(ptx, 3, 3, 8);
        EmitLoadMatrix(ptx, 4, 0);
        for (int row = 0; row < 4; row++)
        for (int col = 0; col < 4; col++)
        {
            int dst = 16 + row * 4 + col;
            ptx.AppendLine($"    mul.rn.f32 %f{dst}, %f{row}, %f{col};");
            for (int k = 1; k < 4; k++)
                ptx.AppendLine($"    fma.rn.f32 %f{dst}, %f{k * 4 + row}, %f{k * 4 + col}, %f{dst};");
        }
        EmitIdentity(ptx, 32);
        EmitJacobiSweeps(ptx, 16, 32, "SVD", 10);
        EmitSortEigenpairs(ptx, 16, 32, ascending: false, "SVD");
        for (int i = 0; i < 4; i++)
        {
            int eig = 16 + i * 5;
            ptx.AppendLine($"    max.f32 %f{64 + i}, %f{eig}, 0f00000000;");
            ptx.AppendLine($"    sqrt.rn.f32 %f{64 + i}, %f{64 + i};");
            ptx.AppendLine($"    st.global.f32 [%rd7+{i * 4}], %f{64 + i};");
        }
        for (int row = 0; row < 4; row++)
        for (int col = 0; col < 4; col++)
        {
            int dst = 16 + row * 4 + col;
            ptx.AppendLine($"    mul.rn.f32 %f{dst}, %f{row * 4}, %f{32 + col};");
            for (int k = 1; k < 4; k++)
                ptx.AppendLine($"    fma.rn.f32 %f{dst}, %f{row * 4 + k}, %f{32 + k * 4 + col}, %f{dst};");
            ptx.AppendLine($"    setp.gt.f32 %p0, %f{64 + col}, 0f0da24260;");
            ptx.AppendLine($"    @%p0 div.rn.f32 %f{dst}, %f{dst}, %f{64 + col};");
            ptx.AppendLine($"    @!%p0 mov.f32 %f{dst}, 0f00000000;");
        }
        EmitStoreMatrix(ptx, 5, 16);
        for (int row = 0; row < 4; row++)
            ptx.AppendLine($"    st.global.v4.f32 [%rd8+{row * 16}], {{%f{32 + row},%f{36 + row},%f{40 + row},%f{44 + row}}};");
    }

    private static void EmitLuSolve(StringBuilder ptx)
    {
        EmitMatrixPointer(ptx, 0, 3, 4);
        EmitVectorPointer(ptx, 1, 2, 5, 6, 16);
        EmitVectorPointer(ptx, 2, 2, 7, 8, 16);
        EmitVectorPointer(ptx, 3, 2, 9, 10, 16);
        EmitLoadMatrix(ptx, 4, 0);
        ptx.AppendLine("    ld.global.v4.u32 {%r4,%r5,%r6,%r7}, [%rd6];");
        ptx.AppendLine("    ld.global.v4.f32 {%f16,%f17,%f18,%f19}, [%rd8];");
        for (int i = 0; i < 4; i++)
        for (int p = i + 1; p < 4; p++)
        {
            ptx.AppendLine($"    setp.eq.u32 %p0, %r{4 + i}, {p};");
            ptx.AppendLine($"    @%p0 mov.f32 %f20, %f{16 + i};");
            ptx.AppendLine($"    @%p0 mov.f32 %f{16 + i}, %f{16 + p};");
            ptx.AppendLine($"    @%p0 mov.f32 %f{16 + p}, %f20;");
        }
        for (int i = 1; i < 4; i++)
            for (int j = 0; j < i; j++)
                ptx.AppendLine($"    fma.rn.f32 %f{16 + i}, -%f{i * 4 + j}, %f{16 + j}, %f{16 + i};");
        for (int i = 3; i >= 0; i--)
        {
            for (int j = i + 1; j < 4; j++)
                ptx.AppendLine($"    fma.rn.f32 %f{16 + i}, -%f{i * 4 + j}, %f{16 + j}, %f{16 + i};");
            ptx.AppendLine($"    setp.eq.f32 %p0, %f{i * 5}, 0f00000000;");
            ptx.AppendLine($"    @%p0 mov.f32 %f{16 + i}, 0f7fffffff;");
            ptx.AppendLine($"    @!%p0 div.rn.f32 %f{16 + i}, %f{16 + i}, %f{i * 5};");
        }
        ptx.AppendLine("    st.global.v4.f32 [%rd10], {%f16,%f17,%f18,%f19};");
    }

    private static void EmitLdlFactor(StringBuilder ptx)
    {
        EmitMatrixPointer(ptx, 0, 3, 4);
        EmitMatrixPointer(ptx, 1, 3, 5);
        EmitVectorPointer(ptx, 2, 2, 6, 7, 16);
        EmitLoadMatrix(ptx, 4, 0);
        for (int j = 0; j < 4; j++)
        {
            ptx.AppendLine($"    abs.f32 %f48, %f{j * 5};");
            ptx.AppendLine($"    mov.u32 %r4, {j};");
            for (int row = j + 1; row < 4; row++)
            {
                ptx.AppendLine($"    abs.f32 %f49, %f{row * 5};");
                ptx.AppendLine("    setp.gt.f32 %p0, %f49, %f48;");
                ptx.AppendLine($"    selp.u32 %r4, {row}, %r4, %p0;");
                ptx.AppendLine("    selp.f32 %f48, %f49, %f48, %p0;");
            }
            ptx.AppendLine($"    st.global.u32 [%rd7+{j * 4}], %r4;");
            for (int pivot = j + 1; pivot < 4; pivot++)
            {
                ptx.AppendLine($"    setp.eq.u32 %p0, %r4, {pivot};");
                for (int k = 0; k < 4; k++)
                {
                    int left = j * 4 + k, right = pivot * 4 + k;
                    ptx.AppendLine($"    @%p0 mov.f32 %f50, %f{left};");
                    ptx.AppendLine($"    @%p0 mov.f32 %f{left}, %f{right};");
                    ptx.AppendLine($"    @%p0 mov.f32 %f{right}, %f50;");
                }
                for (int k = 0; k < 4; k++)
                {
                    int top = k * 4 + j, bottom = k * 4 + pivot;
                    ptx.AppendLine($"    @%p0 mov.f32 %f50, %f{top};");
                    ptx.AppendLine($"    @%p0 mov.f32 %f{top}, %f{bottom};");
                    ptx.AppendLine($"    @%p0 mov.f32 %f{bottom}, %f50;");
                }
            }
            ptx.AppendLine($"    setp.eq.f32 %p0, %f{j * 5}, 0f00000000;");
            ptx.AppendLine($"    @%p0 bra LDL_SKIP_{j};");
            for (int row = j + 1; row < 4; row++)
            {
                int factor = row * 4 + j;
                ptx.AppendLine($"    div.rn.f32 %f{factor}, %f{factor}, %f{j * 5};");
                for (int col = j + 1; col < 4; col++)
                    ptx.AppendLine($"    fma.rn.f32 %f{row * 4 + col}, -%f{factor}, %f{j * 4 + col}, %f{row * 4 + col};");
            }
            ptx.AppendLine($"LDL_SKIP_{j}:");
        }
        for (int row = 0; row < 4; row++)
        for (int col = row + 1; col < 4; col++)
            ptx.AppendLine($"    mov.f32 %f{row * 4 + col}, 0f00000000;");
        EmitStoreMatrix(ptx, 5, 0);
    }

    private static void EmitLdlSolve(StringBuilder ptx)
    {
        EmitMatrixPointer(ptx, 0, 4, 5);
        EmitVectorPointer(ptx, 1, 2, 6, 7, 16);
        EmitVectorPointer(ptx, 2, 2, 8, 9, 16);
        EmitVectorPointer(ptx, 3, 2, 10, 11, 16);
        EmitLoadMatrix(ptx, 5, 0);
        ptx.AppendLine("    ld.global.v4.u32 {%r4,%r5,%r6,%r7}, [%rd7];");
        ptx.AppendLine("    ld.global.v4.f32 {%f16,%f17,%f18,%f19}, [%rd9];");
        for (int step = 0; step < 4; step++)
        for (int pivot = step + 1; pivot < 4; pivot++)
        {
            ptx.AppendLine($"    setp.eq.u32 %p0, %r{4 + step}, {pivot};");
            ptx.AppendLine($"    @%p0 mov.f32 %f20, %f{16 + step};");
            ptx.AppendLine($"    @%p0 mov.f32 %f{16 + step}, %f{16 + pivot};");
            ptx.AppendLine($"    @%p0 mov.f32 %f{16 + pivot}, %f20;");
        }
        for (int row = 1; row < 4; row++)
        for (int col = 0; col < row; col++)
            ptx.AppendLine($"    fma.rn.f32 %f{16 + row}, -%f{row * 4 + col}, %f{16 + col}, %f{16 + row};");
        for (int row = 0; row < 4; row++)
        {
            ptx.AppendLine($"    setp.eq.f32 %p0, %f{row * 5}, 0f00000000;");
            ptx.AppendLine($"    @%p0 mov.f32 %f{16 + row}, 0f7fffffff;");
            ptx.AppendLine($"    @!%p0 div.rn.f32 %f{16 + row}, %f{16 + row}, %f{row * 5};");
        }
        for (int row = 3; row >= 0; row--)
        for (int col = row + 1; col < 4; col++)
            ptx.AppendLine($"    fma.rn.f32 %f{16 + row}, -%f{col * 4 + row}, %f{16 + col}, %f{16 + row};");
        for (int step = 3; step >= 0; step--)
        for (int pivot = step + 1; pivot < 4; pivot++)
        {
            ptx.AppendLine($"    setp.eq.u32 %p0, %r{4 + step}, {pivot};");
            ptx.AppendLine($"    @%p0 mov.f32 %f20, %f{16 + step};");
            ptx.AppendLine($"    @%p0 mov.f32 %f{16 + step}, %f{16 + pivot};");
            ptx.AppendLine($"    @%p0 mov.f32 %f{16 + pivot}, %f20;");
        }
        ptx.AppendLine("    st.global.v4.f32 [%rd11], {%f16,%f17,%f18,%f19};");
    }

    private static void EmitGeneralSolve(StringBuilder ptx)
    {
        EmitMatrixPointer(ptx, 0, 4, 5);
        EmitVectorPointer(ptx, 1, 2, 6, 7, 16);
        EmitVectorPointer(ptx, 2, 2, 8, 9, 16);
        EmitVectorPointer(ptx, 3, 2, 10, 11, 4);
        EmitLoadMatrix(ptx, 5, 0);
        ptx.AppendLine("    ld.global.v4.f32 {%f16,%f17,%f18,%f19}, [%rd7];");
        ptx.AppendLine("    mov.u32 %r8, 0;");
        EmitFusedLuSolve(ptx, 16, 8, "SOLVE");
        ptx.AppendLine("    st.global.v4.f32 [%rd9], {%f16,%f17,%f18,%f19};");
        ptx.AppendLine("    st.global.u32 [%rd11], %r8;");
    }

    private static void EmitTriangularSolve(StringBuilder ptx, bool upper)
    {
        EmitMatrixPointer(ptx, 0, 3, 4);
        EmitVectorPointer(ptx, 1, 2, 5, 6, 16);
        EmitVectorPointer(ptx, 2, 2, 7, 8, 16);
        EmitLoadMatrix(ptx, 4, 0);
        ptx.AppendLine("    ld.global.v4.f32 {%f16,%f17,%f18,%f19}, [%rd6];");
        if (upper)
        {
            for (int row = 3; row >= 0; row--)
            {
                for (int col = row + 1; col < 4; col++)
                    ptx.AppendLine($"    fma.rn.f32 %f{16 + row}, -%f{row * 4 + col}, %f{16 + col}, %f{16 + row};");
                EmitSafeDivision(ptx, 16 + row, row * 5);
            }
        }
        else
        {
            for (int row = 0; row < 4; row++)
            {
                for (int col = 0; col < row; col++)
                    ptx.AppendLine($"    fma.rn.f32 %f{16 + row}, -%f{row * 4 + col}, %f{16 + col}, %f{16 + row};");
                EmitSafeDivision(ptx, 16 + row, row * 5);
            }
        }
        ptx.AppendLine("    st.global.v4.f32 [%rd8], {%f16,%f17,%f18,%f19};");
    }

    private static void EmitCholeskyBackward(StringBuilder ptx)
    {
        EmitMatrixPointer(ptx, 0, 3, 4);
        EmitMatrixPointer(ptx, 1, 3, 5);
        EmitMatrixPointer(ptx, 2, 3, 6);
        EmitLoadMatrix(ptx, 4, 0);
        EmitLoadMatrix(ptx, 5, 16);
        // tmp = L^T * gradL, then Phi(tmp).
        for (int row = 0; row < 4; row++)
        for (int col = 0; col < 4; col++)
        {
            int dst = 32 + row * 4 + col;
            ptx.AppendLine($"    mul.rn.f32 %f{dst}, %f{row}, %f{16 + col};");
            for (int k = 1; k < 4; k++)
                ptx.AppendLine($"    fma.rn.f32 %f{dst}, %f{k * 4 + row}, %f{16 + k * 4 + col}, %f{dst};");
            if (col > row) ptx.AppendLine($"    mov.f32 %f{dst}, 0f00000000;");
            else if (col == row) ptx.AppendLine($"    mul.rn.f32 %f{dst}, %f{dst}, 0f3f000000;");
        }
        for (int row = 0; row < 4; row++)
        for (int col = 0; col < 4; col++)
        {
            int dst = 16 + row * 4 + col;
            int phi = 32 + Math.Max(row, col) * 4 + Math.Min(row, col);
            if (row == col) ptx.AppendLine($"    add.rn.f32 %f{dst}, %f{phi}, %f{phi};");
            else ptx.AppendLine($"    mov.f32 %f{dst}, %f{phi};");
        }
        // Explicit inverse of unit-free lower triangular L by forward solves.
        for (int row = 0; row < 4; row++)
        for (int col = 0; col < 4; col++)
        {
            int dst = 48 + row * 4 + col;
            if (row < col) ptx.AppendLine($"    mov.f32 %f{dst}, 0f00000000;");
            else if (row == col) ptx.AppendLine($"    div.rn.f32 %f{dst}, 0f3f800000, %f{row * 5};");
            else
            {
                ptx.AppendLine($"    mov.f32 %f{dst}, 0f00000000;");
                for (int k = col; k < row; k++)
                    ptx.AppendLine($"    fma.rn.f32 %f{dst}, -%f{row * 4 + k}, %f{48 + k * 4 + col}, %f{dst};");
                ptx.AppendLine($"    div.rn.f32 %f{dst}, %f{dst}, %f{row * 5};");
            }
        }
        // step = L^-T * (Phi + Phi^T).
        for (int row = 0; row < 4; row++)
        for (int col = 0; col < 4; col++)
        {
            int dst = 64 + row * 4 + col;
            ptx.AppendLine($"    mul.rn.f32 %f{dst}, %f{48 + row}, %f{16 + col};");
            for (int k = 1; k < 4; k++)
                ptx.AppendLine($"    fma.rn.f32 %f{dst}, %f{48 + k * 4 + row}, %f{16 + k * 4 + col}, %f{dst};");
        }
        for (int row = 0; row < 4; row++)
        for (int col = 0; col < 4; col++)
        {
            int dst = 80 + row * 4 + col;
            ptx.AppendLine($"    mul.rn.f32 %f{dst}, %f{64 + row * 4}, %f{48 + col};");
            for (int k = 1; k < 4; k++)
                ptx.AppendLine($"    fma.rn.f32 %f{dst}, %f{64 + row * 4 + k}, %f{48 + k * 4 + col}, %f{dst};");
            ptx.AppendLine($"    mul.rn.f32 %f{dst}, %f{dst}, 0f3f000000;");
        }
        EmitStoreMatrix(ptx, 6, 80);
    }

    private static void EmitSolveBackward(StringBuilder ptx)
    {
        EmitMatrixPointer(ptx, 0, 5, 6);
        EmitVectorPointer(ptx, 1, 2, 7, 8, 16);
        EmitVectorPointer(ptx, 2, 2, 9, 10, 16);
        EmitMatrixPointer(ptx, 3, 5, 11);
        EmitVectorPointer(ptx, 4, 2, 12, 13, 16);
        EmitLoadMatrix(ptx, 6, 0);
        // Factor A^T in registers.
        for (int row = 0; row < 4; row++)
        for (int col = row + 1; col < 4; col++)
        {
            ptx.AppendLine($"    mov.f32 %f48, %f{row * 4 + col};");
            ptx.AppendLine($"    mov.f32 %f{row * 4 + col}, %f{col * 4 + row};");
            ptx.AppendLine($"    mov.f32 %f{col * 4 + row}, %f48;");
        }
        ptx.AppendLine("    ld.global.v4.f32 {%f16,%f17,%f18,%f19}, [%rd8];");
        ptx.AppendLine("    ld.global.v4.f32 {%f20,%f21,%f22,%f23}, [%rd10];");
        ptx.AppendLine("    mov.u32 %r8, 0;");
        EmitFusedLuSolve(ptx, 20, 8, "SOLVE_BACKWARD");
        for (int row = 0; row < 4; row++)
        for (int col = 0; col < 4; col++)
        {
            int dst = 24 + row * 4 + col;
            ptx.AppendLine($"    mul.rn.f32 %f{dst}, %f{20 + row}, %f{16 + col};");
            ptx.AppendLine($"    neg.f32 %f{dst}, %f{dst};");
        }
        EmitStoreMatrix(ptx, 11, 24);
        ptx.AppendLine("    st.global.v4.f32 [%rd13], {%f20,%f21,%f22,%f23};");
    }

    private static void EmitFusedLuSolve(StringBuilder ptx, int rhsBase, int infoRegister, string label)
    {
        for (int j = 0; j < 4; j++)
        {
            int diagonal = j * 5;
            ptx.AppendLine($"    abs.f32 %f48, %f{diagonal};");
            ptx.AppendLine($"    mov.u32 %r4, {j};");
            for (int row = j + 1; row < 4; row++)
            {
                ptx.AppendLine($"    abs.f32 %f49, %f{row * 4 + j};");
                ptx.AppendLine("    setp.gt.f32 %p0, %f49, %f48;");
                ptx.AppendLine($"    selp.u32 %r4, {row}, %r4, %p0;");
                ptx.AppendLine("    selp.f32 %f48, %f49, %f48, %p0;");
            }
            for (int pivot = j + 1; pivot < 4; pivot++)
            {
                ptx.AppendLine($"    setp.eq.u32 %p0, %r4, {pivot};");
                for (int col = 0; col < 4; col++)
                {
                    int left = j * 4 + col, right = pivot * 4 + col;
                    ptx.AppendLine($"    @%p0 mov.f32 %f50, %f{left};");
                    ptx.AppendLine($"    @%p0 mov.f32 %f{left}, %f{right};");
                    ptx.AppendLine($"    @%p0 mov.f32 %f{right}, %f50;");
                }
                ptx.AppendLine($"    @%p0 mov.f32 %f50, %f{rhsBase + j};");
                ptx.AppendLine($"    @%p0 mov.f32 %f{rhsBase + j}, %f{rhsBase + pivot};");
                ptx.AppendLine($"    @%p0 mov.f32 %f{rhsBase + pivot}, %f50;");
            }
            ptx.AppendLine($"    setp.eq.f32 %p0, %f{diagonal}, 0f00000000;");
            ptx.AppendLine($"    setp.eq.u32 %p1, %r{infoRegister}, 0;");
            ptx.AppendLine("    and.pred %p2, %p0, %p1;");
            ptx.AppendLine($"    @%p2 mov.u32 %r{infoRegister}, {j + 1};");
            ptx.AppendLine($"    @%p0 bra {label}_SKIP_{j};");
            for (int row = j + 1; row < 4; row++)
            {
                int factor = row * 4 + j;
                ptx.AppendLine($"    div.rn.f32 %f{factor}, %f{factor}, %f{diagonal};");
                for (int col = j + 1; col < 4; col++)
                    ptx.AppendLine($"    fma.rn.f32 %f{row * 4 + col}, -%f{factor}, %f{j * 4 + col}, %f{row * 4 + col};");
                ptx.AppendLine($"    fma.rn.f32 %f{rhsBase + row}, -%f{factor}, %f{rhsBase + j}, %f{rhsBase + row};");
            }
            ptx.AppendLine($"{label}_SKIP_{j}:");
        }
        for (int row = 3; row >= 0; row--)
        {
            for (int col = row + 1; col < 4; col++)
                ptx.AppendLine($"    fma.rn.f32 %f{rhsBase + row}, -%f{row * 4 + col}, %f{rhsBase + col}, %f{rhsBase + row};");
            EmitSafeDivision(ptx, rhsBase + row, row * 5);
        }
    }

    private static void EmitSafeDivision(StringBuilder ptx, int numerator, int denominator)
    {
        ptx.AppendLine($"    setp.eq.f32 %p0, %f{denominator}, 0f00000000;");
        ptx.AppendLine($"    @%p0 mov.f32 %f{numerator}, 0f7fffffff;");
        ptx.AppendLine($"    @!%p0 div.rn.f32 %f{numerator}, %f{numerator}, %f{denominator};");
    }

    private static void EmitJacobiSweeps(
        StringBuilder ptx,
        int aBase,
        int vBase,
        string labelPrefix,
        int sweeps)
    {
        int label = 0;
        for (int sweep = 0; sweep < sweeps; sweep++)
        for (int p = 0; p < 3; p++)
        for (int q = p + 1; q < 4; q++)
        {
            int app = aBase + p * 4 + p, aqq = aBase + q * 4 + q;
            int apq = aBase + p * 4 + q;
            string skip = $"{labelPrefix}_J_SKIP_{label++}";
            ptx.AppendLine($"    abs.f32 %f48, %f{apq};");
            ptx.AppendLine("    setp.le.f32 %p0, %f48, 0f2edbe6ff;");
            ptx.AppendLine($"    @%p0 bra {skip};");
            ptx.AppendLine($"    sub.rn.f32 %f49, %f{aqq}, %f{app};");
            ptx.AppendLine($"    add.rn.f32 %f50, %f{apq}, %f{apq};");
            ptx.AppendLine("    div.rn.f32 %f49, %f49, %f50;");
            ptx.AppendLine("    abs.f32 %f50, %f49;");
            ptx.AppendLine("    mul.rn.f32 %f51, %f49, %f49;");
            ptx.AppendLine("    add.rn.f32 %f51, %f51, 0f3f800000;");
            ptx.AppendLine("    sqrt.rn.f32 %f51, %f51;");
            ptx.AppendLine("    add.rn.f32 %f50, %f50, %f51;");
            ptx.AppendLine("    div.rn.f32 %f52, 0f3f800000, %f50;");
            ptx.AppendLine("    setp.lt.f32 %p1, %f49, 0f00000000;");
            ptx.AppendLine("    @%p1 neg.f32 %f52, %f52;");
            ptx.AppendLine("    mul.rn.f32 %f53, %f52, %f52;");
            ptx.AppendLine("    add.rn.f32 %f53, %f53, 0f3f800000;");
            ptx.AppendLine("    rsqrt.approx.f32 %f53, %f53;");
            ptx.AppendLine("    mul.rn.f32 %f54, %f52, %f53;");
            ptx.AppendLine($"    mul.rn.f32 %f55, %f53, %f53;");
            ptx.AppendLine($"    mul.rn.f32 %f56, %f54, %f54;");
            ptx.AppendLine($"    mul.rn.f32 %f57, %f54, %f53;");
            ptx.AppendLine($"    mul.rn.f32 %f58, %f55, %f{app};");
            ptx.AppendLine($"    fma.rn.f32 %f58, -%f57, %f{apq}, %f58;");
            ptx.AppendLine($"    fma.rn.f32 %f58, -%f57, %f{apq}, %f58;");
            ptx.AppendLine($"    fma.rn.f32 %f58, %f56, %f{aqq}, %f58;");
            ptx.AppendLine($"    mul.rn.f32 %f59, %f56, %f{app};");
            ptx.AppendLine($"    fma.rn.f32 %f59, %f57, %f{apq}, %f59;");
            ptx.AppendLine($"    fma.rn.f32 %f59, %f57, %f{apq}, %f59;");
            ptx.AppendLine($"    fma.rn.f32 %f59, %f55, %f{aqq}, %f59;");
            ptx.AppendLine($"    mov.f32 %f{app}, %f58;");
            ptx.AppendLine($"    mov.f32 %f{aqq}, %f59;");
            ptx.AppendLine($"    mov.f32 %f{apq}, 0f00000000;");
            ptx.AppendLine($"    mov.f32 %f{aBase + q * 4 + p}, 0f00000000;");
            for (int k = 0; k < 4; k++)
            {
                if (k == p || k == q) continue;
                int akp = aBase + k * 4 + p, akq = aBase + k * 4 + q;
                ptx.AppendLine($"    mul.rn.f32 %f60, %f53, %f{akp};");
                ptx.AppendLine($"    fma.rn.f32 %f60, -%f54, %f{akq}, %f60;");
                ptx.AppendLine($"    mul.rn.f32 %f61, %f54, %f{akp};");
                ptx.AppendLine($"    fma.rn.f32 %f61, %f53, %f{akq}, %f61;");
                ptx.AppendLine($"    mov.f32 %f{akp}, %f60;");
                ptx.AppendLine($"    mov.f32 %f{aBase + p * 4 + k}, %f60;");
                ptx.AppendLine($"    mov.f32 %f{akq}, %f61;");
                ptx.AppendLine($"    mov.f32 %f{aBase + q * 4 + k}, %f61;");
            }
            for (int row = 0; row < 4; row++)
            {
                int vp = vBase + row * 4 + p, vq = vBase + row * 4 + q;
                ptx.AppendLine($"    mul.rn.f32 %f60, %f53, %f{vp};");
                ptx.AppendLine($"    fma.rn.f32 %f60, -%f54, %f{vq}, %f60;");
                ptx.AppendLine($"    mul.rn.f32 %f61, %f54, %f{vp};");
                ptx.AppendLine($"    fma.rn.f32 %f61, %f53, %f{vq}, %f61;");
                ptx.AppendLine($"    mov.f32 %f{vp}, %f60;");
                ptx.AppendLine($"    mov.f32 %f{vq}, %f61;");
            }
            ptx.AppendLine($"{skip}:");
        }
    }

    private static void EmitSortEigenpairs(
        StringBuilder ptx,
        int aBase,
        int vBase,
        bool ascending,
        string labelPrefix)
    {
        int step = 0;
        for (int pass = 0; pass < 3; pass++)
        for (int left = 0; left < 3 - pass; left++)
        {
            int right = left + 1;
            int l = aBase + left * 5, r = aBase + right * 5;
            string skip = $"{labelPrefix}_SORT_SKIP_{step++}";
            ptx.AppendLine($"    setp.{(ascending ? "le" : "ge")}.f32 %p0, %f{l}, %f{r};");
            ptx.AppendLine($"    @%p0 bra {skip};");
            ptx.AppendLine($"    mov.f32 %f48, %f{l};");
            ptx.AppendLine($"    mov.f32 %f{l}, %f{r};");
            ptx.AppendLine($"    mov.f32 %f{r}, %f48;");
            for (int row = 0; row < 4; row++)
            {
                int vl = vBase + row * 4 + left, vr = vBase + row * 4 + right;
                ptx.AppendLine($"    mov.f32 %f48, %f{vl};");
                ptx.AppendLine($"    mov.f32 %f{vl}, %f{vr};");
                ptx.AppendLine($"    mov.f32 %f{vr}, %f48;");
            }
            ptx.AppendLine($"{skip}:");
        }
    }

    private static void EmitMatrixPointer(StringBuilder ptx, int parameterRd, int offsetRd, int resultRd)
    {
        ptx.AppendLine($"    mul.wide.u32 %rd{offsetRd}, %r2, 64;");
        ptx.AppendLine($"    add.u64 %rd{resultRd}, %rd{parameterRd}, %rd{offsetRd};");
    }

    private static void EmitVectorPointer(
        StringBuilder ptx, int parameterRd, int indexRegister, int offsetRd, int resultRd, int bytes)
    {
        ptx.AppendLine($"    mul.wide.u32 %rd{offsetRd}, %r{indexRegister}, {bytes};");
        ptx.AppendLine($"    add.u64 %rd{resultRd}, %rd{parameterRd}, %rd{offsetRd};");
    }

    private static void EmitLoadMatrix(StringBuilder ptx, int pointerRd, int floatBase)
    {
        for (int row = 0; row < 4; row++)
            ptx.AppendLine($"    ld.global.v4.f32 {{%f{floatBase + row * 4},%f{floatBase + row * 4 + 1},%f{floatBase + row * 4 + 2},%f{floatBase + row * 4 + 3}}}, [%rd{pointerRd}+{row * 16}];");
    }

    private static void EmitStoreMatrix(StringBuilder ptx, int pointerRd, int floatBase)
    {
        for (int row = 0; row < 4; row++)
            ptx.AppendLine($"    st.global.v4.f32 [%rd{pointerRd}+{row * 16}], {{%f{floatBase + row * 4},%f{floatBase + row * 4 + 1},%f{floatBase + row * 4 + 2},%f{floatBase + row * 4 + 3}}};");
    }

    private static void EmitIdentity(StringBuilder ptx, int floatBase)
    {
        for (int row = 0; row < 4; row++)
        for (int col = 0; col < 4; col++)
            ptx.AppendLine($"    mov.f32 %f{floatBase + row * 4 + col}, {(row == col ? "0f3f800000" : "0f00000000")};");
    }

    private static void SymmetrizeUpper(StringBuilder ptx, int floatBase)
    {
        for (int row = 1; row < 4; row++)
        for (int col = 0; col < row; col++)
            ptx.AppendLine($"    mov.f32 %f{floatBase + row * 4 + col}, %f{floatBase + col * 4 + row};");
    }

    private static void SymmetrizeLower(StringBuilder ptx, int floatBase)
    {
        for (int row = 0; row < 3; row++)
        for (int col = row + 1; col < 4; col++)
            ptx.AppendLine($"    mov.f32 %f{floatBase + row * 4 + col}, %f{floatBase + col * 4 + row};");
    }

    private static string EntryPointFor(DirectPtxSolver4x4Operation operation) => operation switch
    {
        DirectPtxSolver4x4Operation.LuFactor => "aidotnet_register_lu_factor_4x4_f32",
        DirectPtxSolver4x4Operation.QrReduced => "aidotnet_register_qr_reduced_4x4_f32",
        DirectPtxSolver4x4Operation.EighUpper => "aidotnet_register_eigh_upper_4x4_f32",
        DirectPtxSolver4x4Operation.EighLower => "aidotnet_register_eigh_lower_4x4_f32",
        DirectPtxSolver4x4Operation.SvdReduced => "aidotnet_register_svd_reduced_4x4_f32",
        DirectPtxSolver4x4Operation.LuSolveVector => "aidotnet_register_lu_solve_vector_4x4_f32",
        DirectPtxSolver4x4Operation.LdlFactorLower => "aidotnet_register_ldl_factor_lower_4x4_f32",
        DirectPtxSolver4x4Operation.LdlSolveVectorLower => "aidotnet_register_ldl_solve_lower_vector_4x4_f32",
        DirectPtxSolver4x4Operation.GeneralSolveVector => "aidotnet_register_solve_vector_4x4_f32",
        DirectPtxSolver4x4Operation.TriangularSolveVectorLower => "aidotnet_register_triangular_solve_lower_vector_4x4_f32",
        DirectPtxSolver4x4Operation.TriangularSolveVectorUpper => "aidotnet_register_triangular_solve_upper_vector_4x4_f32",
        DirectPtxSolver4x4Operation.CholeskyBackwardLower => "aidotnet_register_cholesky_backward_lower_4x4_f32",
        DirectPtxSolver4x4Operation.SolveBackwardVector => "aidotnet_register_solve_backward_vector_4x4_f32",
        _ => throw new ArgumentOutOfRangeException(nameof(operation))
    };

    private static string[] ParameterNames(DirectPtxSolver4x4Operation operation) => operation switch
    {
        DirectPtxSolver4x4Operation.LuFactor => ["input_ptr", "lu_ptr", "pivots_ptr"],
        DirectPtxSolver4x4Operation.QrReduced => ["input_ptr", "q_ptr", "r_ptr"],
        DirectPtxSolver4x4Operation.EighUpper => ["input_ptr", "eigenvalues_ptr", "eigenvectors_ptr"],
        DirectPtxSolver4x4Operation.EighLower => ["input_ptr", "eigenvalues_ptr", "eigenvectors_ptr"],
        DirectPtxSolver4x4Operation.SvdReduced => ["input_ptr", "u_ptr", "singular_values_ptr", "vh_ptr"],
        DirectPtxSolver4x4Operation.LuSolveVector => ["lu_ptr", "pivots_ptr", "rhs_ptr", "solution_ptr"],
        DirectPtxSolver4x4Operation.LdlFactorLower => ["input_ptr", "ld_ptr", "pivots_ptr"],
        DirectPtxSolver4x4Operation.LdlSolveVectorLower => ["ld_ptr", "pivots_ptr", "rhs_ptr", "solution_ptr"],
        DirectPtxSolver4x4Operation.GeneralSolveVector => ["input_ptr", "rhs_ptr", "solution_ptr", "info_ptr"],
        DirectPtxSolver4x4Operation.TriangularSolveVectorLower => ["input_ptr", "rhs_ptr", "solution_ptr"],
        DirectPtxSolver4x4Operation.TriangularSolveVectorUpper => ["input_ptr", "rhs_ptr", "solution_ptr"],
        DirectPtxSolver4x4Operation.CholeskyBackwardLower => ["factor_ptr", "grad_output_ptr", "grad_input_ptr"],
        DirectPtxSolver4x4Operation.SolveBackwardVector => ["input_ptr", "solution_ptr", "grad_output_ptr", "grad_input_ptr", "grad_rhs_ptr"],
        _ => throw new ArgumentOutOfRangeException(nameof(operation))
    };

    private static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture,
        DirectPtxSolver4x4Operation operation,
        int batchCount,
        int blockThreads)
    {
        var matrix = new DirectPtxExtent(batchCount, 4, 4);
        var vector = new DirectPtxExtent(batchCount, 4);
        var scalarBatch = new DirectPtxExtent(batchCount);
        DirectPtxTensorContract Matrix(string name, DirectPtxTensorAccess access) =>
            new(name, DirectPtxPhysicalType.Float32,
                DirectPtxPhysicalLayout.BatchedRowMajorMatrix,
                matrix, matrix, 16, access, DirectPtxExtentMode.Exact);
        DirectPtxTensorContract FloatVector(string name, DirectPtxTensorAccess access) =>
            new(name, DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                vector, vector, 16, access, DirectPtxExtentMode.Exact);
        DirectPtxTensorContract IntVector(string name, DirectPtxTensorAccess access) =>
            new(name, DirectPtxPhysicalType.Int32, DirectPtxPhysicalLayout.RowMajor2D,
                vector, vector, 16, access, DirectPtxExtentMode.Exact);
        DirectPtxTensorContract IntScalarBatch(string name, DirectPtxTensorAccess access) =>
            new(name, DirectPtxPhysicalType.Int32, DirectPtxPhysicalLayout.Vector,
                scalarBatch, scalarBatch, 16, access, DirectPtxExtentMode.Exact);
        IReadOnlyList<DirectPtxTensorContract> tensors = operation switch
        {
            DirectPtxSolver4x4Operation.LuFactor =>
                [Matrix("input", DirectPtxTensorAccess.Read), Matrix("lu", DirectPtxTensorAccess.Write), IntVector("pivots", DirectPtxTensorAccess.Write)],
            DirectPtxSolver4x4Operation.QrReduced =>
                [Matrix("input", DirectPtxTensorAccess.Read), Matrix("q", DirectPtxTensorAccess.Write), Matrix("r", DirectPtxTensorAccess.Write)],
            DirectPtxSolver4x4Operation.EighUpper =>
                [Matrix("input", DirectPtxTensorAccess.Read), FloatVector("eigenvalues", DirectPtxTensorAccess.Write), Matrix("eigenvectors", DirectPtxTensorAccess.Write)],
            DirectPtxSolver4x4Operation.EighLower =>
                [Matrix("input", DirectPtxTensorAccess.Read), FloatVector("eigenvalues", DirectPtxTensorAccess.Write), Matrix("eigenvectors", DirectPtxTensorAccess.Write)],
            DirectPtxSolver4x4Operation.SvdReduced =>
                [Matrix("input", DirectPtxTensorAccess.Read), Matrix("u", DirectPtxTensorAccess.Write), FloatVector("singular-values", DirectPtxTensorAccess.Write), Matrix("vh", DirectPtxTensorAccess.Write)],
            DirectPtxSolver4x4Operation.LuSolveVector =>
                [Matrix("lu", DirectPtxTensorAccess.Read), IntVector("pivots", DirectPtxTensorAccess.Read), FloatVector("rhs", DirectPtxTensorAccess.Read), FloatVector("solution", DirectPtxTensorAccess.Write)],
            DirectPtxSolver4x4Operation.LdlFactorLower =>
                [Matrix("input", DirectPtxTensorAccess.Read), Matrix("ld", DirectPtxTensorAccess.Write), IntVector("pivots", DirectPtxTensorAccess.Write)],
            DirectPtxSolver4x4Operation.LdlSolveVectorLower =>
                [Matrix("ld", DirectPtxTensorAccess.Read), IntVector("pivots", DirectPtxTensorAccess.Read), FloatVector("rhs", DirectPtxTensorAccess.Read), FloatVector("solution", DirectPtxTensorAccess.Write)],
            DirectPtxSolver4x4Operation.GeneralSolveVector =>
                [Matrix("input", DirectPtxTensorAccess.Read), FloatVector("rhs", DirectPtxTensorAccess.Read), FloatVector("solution", DirectPtxTensorAccess.Write), IntScalarBatch("info", DirectPtxTensorAccess.Write)],
            DirectPtxSolver4x4Operation.TriangularSolveVectorLower or
            DirectPtxSolver4x4Operation.TriangularSolveVectorUpper =>
                [Matrix("input", DirectPtxTensorAccess.Read), FloatVector("rhs", DirectPtxTensorAccess.Read), FloatVector("solution", DirectPtxTensorAccess.Write)],
            DirectPtxSolver4x4Operation.CholeskyBackwardLower =>
                [Matrix("factor", DirectPtxTensorAccess.Read), Matrix("grad-output", DirectPtxTensorAccess.Read), Matrix("grad-input", DirectPtxTensorAccess.Write)],
            DirectPtxSolver4x4Operation.SolveBackwardVector =>
                [Matrix("input", DirectPtxTensorAccess.Read), FloatVector("solution", DirectPtxTensorAccess.Read), FloatVector("grad-output", DirectPtxTensorAccess.Read), Matrix("grad-input", DirectPtxTensorAccess.Write), FloatVector("grad-rhs", DirectPtxTensorAccess.Write)],
            _ => throw new ArgumentOutOfRangeException(nameof(operation))
        };
        return new DirectPtxKernelBlueprint(
            Operation: operation.ToString(), Version: 1, Architecture: architecture,
            Variant: $"register-4x4-batch{batchCount}-block{blockThreads}", Tensors: tensors,
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: operation switch
                {
                    DirectPtxSolver4x4Operation.SvdReduced => 80,
                    DirectPtxSolver4x4Operation.CholeskyBackwardLower => 104,
                    _ => 64
                },
                MaxStaticSharedBytes: 0, MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 2),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["matrix-order"] = "4",
                ["ownership"] = "one-thread-per-matrix-register-resident",
                ["global-intermediates"] = "none",
                ["temporary-device-allocation"] = "none",
                ["stride-parameters"] = "none",
                ["promotion"] = "experimental-unpromoted"
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
            throw new ArgumentException($"{parameter} does not satisfy physical ABI '{contract.Name}'.", parameter);
    }

    private static bool Overlaps(DirectPtxTensorView left, DirectPtxTensorView right)
    {
        nuint leftStart = (nuint)left.Pointer, rightStart = (nuint)right.Pointer;
        nuint leftEnd = checked(leftStart + left.ByteLength);
        nuint rightEnd = checked(rightStart + right.ByteLength);
        return leftStart < rightEnd && rightStart < leftEnd;
    }
}
#endif
