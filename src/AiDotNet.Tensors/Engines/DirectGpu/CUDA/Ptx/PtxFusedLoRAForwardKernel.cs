using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Fused LoRA forward delta+residual:
/// <c>output[M,N] = base[M,N] + scale * (X[M,K] @ transpose(A[r,K])) @ transpose(B[N,r])</c>,
/// where A (down-projection) and B (up-projection) are canonical output-major
/// low-rank factors (row contiguous over the contracted dimension, identical to the
/// fused-linear weight contract) and <c>base</c> is the already-computed dense linear
/// output. The two rank GEMMs are fused into a single launch: each row-block reduces
/// the full intermediate <c>Z[BM,r] = X @ transpose(A)</c> into shared memory exactly
/// once, then streams every N-column tile through <c>Z @ transpose(B)</c>, so there is
/// no global intermediate and no redundant recomputation of Z across column tiles.
///
/// The checked-in specialization fixes rank r = <see cref="Rank"/> (64); other ranks
/// are tracked follow-ups. Tile BM=BN=64, BK=8, TM=TN=4 → 256 threads, 16 FP32
/// accumulators/thread. Shared: 2 KiB X + 2 KiB A + 16 KiB Z + 2 KiB B. Grid = (M/BM).
/// </summary>
internal sealed class PtxFusedLoRAForwardKernel : IDisposable
{
    internal const int BlockM = 64;
    internal const int BlockN = 64;
    internal const int BlockK = 8;
    internal const int ThreadM = 4;
    internal const int ThreadN = 4;
    internal const int Rank = 64;
    internal const int BlockThreads = (BlockM / ThreadM) * (BlockN / ThreadN); // 256
    internal const string EntryPoint = "aidotnet_fused_lora_forward";

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int M { get; }
    internal int K { get; }
    internal int N { get; }
    internal float Scale { get; }
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxFusedLoRAForwardKernel(DirectPtxRuntime runtime, int m, int k, int n, float scale)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasValidatedFusedLinear(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The checked-in fused-LoRA specialization is measured only on GA10x/SM86.");
        ValidateShape(m, k, n);
        M = m;
        K = k;
        N = n;
        Scale = scale;
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, m, k, n);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor, m, k, n);
        _module = runtime.LoadModule(Ptx);
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo info);
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, BlockThreads);
        Blueprint.ResourceBudget.Validate(EntryPoint, info, BlockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Ptx, info, BlockThreads, activeBlocks, _module.JitInfoLog);
    }

    internal unsafe void Launch(
        DirectPtxTensorView x,
        DirectPtxTensorView loraA,
        DirectPtxTensorView loraB,
        DirectPtxTensorView baseOutput,
        DirectPtxTensorView output)
    {
        Require(x, Blueprint.Tensors[0], nameof(x));
        Require(loraA, Blueprint.Tensors[1], nameof(loraA));
        Require(loraB, Blueprint.Tensors[2], nameof(loraB));
        Require(baseOutput, Blueprint.Tensors[3], nameof(baseOutput));
        Require(output, Blueprint.Tensors[4], nameof(output));
        if (Overlaps(output, x) || Overlaps(output, loraA) || Overlaps(output, loraB))
            throw new ArgumentException("Fused-LoRA output may not alias X, A, or B.");

        IntPtr xPointer = x.Pointer;
        IntPtr aPointer = loraA.Pointer;
        IntPtr bPointer = loraB.Pointer;
        IntPtr basePointer = baseOutput.Pointer;
        IntPtr outputPointer = output.Pointer;
        float scale = Scale;
        void** arguments = stackalloc void*[6];
        arguments[0] = &xPointer;
        arguments[1] = &aPointer;
        arguments[2] = &bPointer;
        arguments[3] = &basePointer;
        arguments[4] = &outputPointer;
        arguments[5] = &scale;
        _module.Launch(_function, (uint)(M / BlockM), 1, 1, BlockThreads, 1, 1, 0, arguments);
    }

    public void Dispose() => _module.Dispose();

    internal static string EmitPtx(int ccMajor, int ccMinor, int m, int k, int n)
    {
        ValidateShape(m, k, n);
        int kBytes = checked(k * sizeof(float));   // X row stride
        int rBytes = checked(Rank * sizeof(float)); // A row stride and B row stride
        int nBytes = checked(n * sizeof(float));   // base/output row stride

        var ptx = new StringBuilder(32_000);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine($"// fused-lora M={m} K={k} N={n} r={Rank} tile={BlockM}x{BlockN}x{BlockK}");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 x_ptr,");
        ptx.AppendLine("    .param .u64 a_ptr,");
        ptx.AppendLine("    .param .u64 b_ptr,");
        ptx.AppendLine("    .param .u64 base_ptr,");
        ptx.AppendLine("    .param .u64 output_ptr,");
        ptx.AppendLine("    .param .f32 scale");
        ptx.AppendLine(")");
        ptx.AppendLine($".maxntid {BlockThreads}, 1, 1");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<4>;");
        ptx.AppendLine("    .reg .b32 %r<48>;");
        ptx.AppendLine("    .reg .b64 %rd<56>;");
        ptx.AppendLine("    .reg .f32 %f<48>;");
        ptx.AppendLine($"    .shared .align 16 .b8 x_tile[{BlockM * BlockK * sizeof(float)}];");
        ptx.AppendLine($"    .shared .align 16 .b8 a_tile[{BlockM * BlockK * sizeof(float)}];");
        ptx.AppendLine($"    .shared .align 16 .b8 z_tile[{BlockM * Rank * sizeof(float)}];");
        ptx.AppendLine($"    .shared .align 16 .b8 b_tile[{BlockN * BlockK * sizeof(float)}];");
        ptx.AppendLine("    ld.param.u64 %rd0, [x_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [a_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [b_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd6, [base_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd3, [output_ptr];");
        ptx.AppendLine("    ld.param.f32 %f26, [scale];");
        ptx.AppendLine("    mov.u64 %rd4, x_tile;");
        ptx.AppendLine("    mov.u64 %rd5, a_tile;");
        ptx.AppendLine("    mov.u64 %rd7, z_tile;");
        ptx.AppendLine("    mov.u64 %rd20, b_tile;");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");                 // row block only
        ptx.AppendLine("    shr.u32 %r3, %r0, 4;");                   // threadRow
        ptx.AppendLine("    and.b32 %r4, %r0, 15;");                  // threadCol
        ptx.AppendLine($"    mul.lo.u32 %r5, %r1, {BlockM};");        // rowBase = blockRow*BM
        ptx.AppendLine($"    mul.lo.u32 %r7, %r3, {ThreadM};");       // threadRow*TM
        ptx.AppendLine($"    mul.lo.u32 %r8, %r4, {ThreadN};");       // threadCol*TN

        // ---------------- Stage 1: Z[BM,r] = X[BM,K] @ transpose(A[r,K]) ----------------
        for (int i = 0; i < ThreadM * ThreadN; i++)
            ptx.AppendLine($"    mov.f32 %f{i}, 0f00000000;");
        ptx.AppendLine("    mov.u32 %r9, 0;");
        ptx.AppendLine("STAGE1_K_LOOP:");
        EmitStage1Load(ptx, kBytes);
        ptx.AppendLine("    bar.sync 0;");
        EmitTiledInner(ptx, "%rd13", "%rd14", BlockK);
        ptx.AppendLine("    bar.sync 0;");
        ptx.AppendLine($"    add.u32 %r9, %r9, {BlockK};");
        ptx.AppendLine($"    setp.lt.u32 %p0, %r9, {k};");
        ptx.AppendLine("    @%p0 bra.uni STAGE1_K_LOOP;");
        // Store the 16 accumulators into z_tile[(threadRow*TM+i)*r + (threadCol*TN+j)].
        for (int i = 0; i < ThreadM; i++)
            for (int j = 0; j < ThreadN; j++)
            {
                ptx.AppendLine($"    add.u32 %r22, %r7, {i};");                 // z row
                ptx.AppendLine($"    add.u32 %r23, %r8, {j};");                 // z col
                ptx.AppendLine($"    mul.lo.u32 %r24, %r22, {Rank};");
                ptx.AppendLine("    add.u32 %r24, %r24, %r23;");
                ptx.AppendLine("    mul.wide.u32 %rd15, %r24, 4;");
                ptx.AppendLine("    add.u64 %rd16, %rd7, %rd15;");
                ptx.AppendLine($"    st.shared.f32 [%rd16], %f{i * ThreadN + j};");
            }
        ptx.AppendLine("    bar.sync 0;");

        // ---------------- Stage 2: for each N tile, dY = Z @ transpose(B), fused residual ----------------
        ptx.AppendLine("    mov.u32 %r30, 0;");                      // nb = column-block base
        ptx.AppendLine("COL_LOOP:");
        for (int i = 0; i < ThreadM * ThreadN; i++)
            ptx.AppendLine($"    mov.f32 %f{i}, 0f00000000;");
        ptx.AppendLine("    mov.u32 %r9, 0;");                       // r0 within rank
        ptx.AppendLine("STAGE2_R_LOOP:");
        EmitStage2LoadB(ptx, rBytes);
        ptx.AppendLine("    bar.sync 0;");
        EmitStage2Inner(ptx);
        ptx.AppendLine("    bar.sync 0;");
        ptx.AppendLine($"    add.u32 %r9, %r9, {BlockK};");
        ptx.AppendLine($"    setp.lt.u32 %p1, %r9, {Rank};");
        ptx.AppendLine("    @%p1 bra.uni STAGE2_R_LOOP;");
        EmitStage2Epilogue(ptx, nBytes);
        ptx.AppendLine($"    add.u32 %r30, %r30, {BlockN};");
        ptx.AppendLine($"    setp.lt.u32 %p2, %r30, {n};");
        ptx.AppendLine("    @%p2 bra.uni COL_LOOP;");

        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    // Stage 1 cooperative load: 64x8 X (rows = rowBase+..) and 64x8 A (rows = 0+..),
    // both row-major over BK. Leaves fragment base pointers in %rd13 (X) / %rd14 (A).
    private static void EmitStage1Load(StringBuilder ptx, int kBytes)
    {
        for (int slot = 0; slot < 2; slot++)
        {
            string e = slot == 0 ? "%r10" : "%r11";
            ptx.AppendLine($"    add.u32 {e}, %r0, {slot * BlockThreads};");
            ptx.AppendLine($"    shr.u32 %r12, {e}, 3;");                   // row
            ptx.AppendLine($"    and.b32 %r13, {e}, 7;");                   // kk
            ptx.AppendLine("    add.u32 %r14, %r5, %r12;");                 // X global row
            ptx.AppendLine("    add.u32 %r15, %r9, %r13;");                 // K col
            ptx.AppendLine($"    mul.wide.u32 %rd8, %r14, {kBytes};");
            ptx.AppendLine("    mul.wide.u32 %rd9, %r15, 4;");
            ptx.AppendLine("    add.u64 %rd10, %rd0, %rd8;");
            ptx.AppendLine("    add.u64 %rd10, %rd10, %rd9;");
            ptx.AppendLine("    ld.global.nc.f32 %f16, [%rd10];");
            ptx.AppendLine($"    mul.wide.u32 %rd11, {e}, 4;");
            ptx.AppendLine("    add.u64 %rd12, %rd4, %rd11;");
            ptx.AppendLine("    st.shared.f32 [%rd12], %f16;");
            // A: output-major row over r; global row = row (block covers all r=BN).
            ptx.AppendLine($"    mul.wide.u32 %rd8, %r12, {kBytes};");
            ptx.AppendLine("    add.u64 %rd10, %rd1, %rd8;");
            ptx.AppendLine("    add.u64 %rd10, %rd10, %rd9;");
            ptx.AppendLine("    ld.global.nc.f32 %f16, [%rd10];");
            ptx.AppendLine("    add.u64 %rd12, %rd5, %rd11;");
            ptx.AppendLine("    st.shared.f32 [%rd12], %f16;");
        }
        // Fragment base pointers for the inner product.
        ptx.AppendLine($"    mul.lo.u32 %r16, %r7, {BlockK};");
        ptx.AppendLine($"    mul.lo.u32 %r17, %r8, {BlockK};");
        ptx.AppendLine("    mul.wide.u32 %rd13, %r16, 4;");
        ptx.AppendLine("    add.u64 %rd13, %rd4, %rd13;");
        ptx.AppendLine("    mul.wide.u32 %rd14, %r17, 4;");
        ptx.AppendLine("    add.u64 %rd14, %rd5, %rd14;");
    }

    // Register-blocked TM×TN accumulate over one BK slice from two row-major shared
    // fragments whose base byte pointers are supplied.
    private static void EmitTiledInner(StringBuilder ptx, string aBase, string bBase, int bk)
    {
        for (int k = 0; k < bk; k++)
        {
            for (int i = 0; i < ThreadM; i++)
                ptx.AppendLine($"    ld.shared.f32 %f{16 + i}, [{aBase}+{(i * bk + k) * sizeof(float)}];");
            for (int j = 0; j < ThreadN; j++)
                ptx.AppendLine($"    ld.shared.f32 %f{20 + j}, [{bBase}+{(j * bk + k) * sizeof(float)}];");
            for (int i = 0; i < ThreadM; i++)
                for (int j = 0; j < ThreadN; j++)
                    ptx.AppendLine(
                        $"    fma.rn.f32 %f{i * ThreadN + j}, %f{16 + i}, %f{20 + j}, %f{i * ThreadN + j};");
        }
    }

    // Stage 2 cooperative load of the BN×BK B slice: B row = nb+row (output column),
    // B col = r0 + kk. B is output-major [N,r].
    private static void EmitStage2LoadB(StringBuilder ptx, int rBytes)
    {
        for (int slot = 0; slot < 2; slot++)
        {
            string e = slot == 0 ? "%r10" : "%r11";
            ptx.AppendLine($"    add.u32 {e}, %r0, {slot * BlockThreads};");
            ptx.AppendLine($"    shr.u32 %r12, {e}, 3;");                   // row (col-within-block)
            ptx.AppendLine($"    and.b32 %r13, {e}, 7;");                   // kk within rank slice
            ptx.AppendLine("    add.u32 %r14, %r30, %r12;");                // global output column n
            ptx.AppendLine("    add.u32 %r15, %r9, %r13;");                 // rank index r0+kk
            ptx.AppendLine($"    mul.wide.u32 %rd8, %r14, {rBytes};");
            ptx.AppendLine("    mul.wide.u32 %rd9, %r15, 4;");
            ptx.AppendLine("    add.u64 %rd10, %rd2, %rd8;");
            ptx.AppendLine("    add.u64 %rd10, %rd10, %rd9;");
            ptx.AppendLine("    ld.global.nc.f32 %f16, [%rd10];");
            ptx.AppendLine($"    mul.wide.u32 %rd11, {e}, 4;");
            ptx.AppendLine("    add.u64 %rd12, %rd20, %rd11;");
            ptx.AppendLine("    st.shared.f32 [%rd12], %f16;");
        }
    }

    // dY[myRow,myCol] += sum_kk Z[myRow, r0+kk] * B_tile[myCol, kk].
    private static void EmitStage2Inner(StringBuilder ptx)
    {
        // B fragment base: &b_tile[(threadCol*TN)*BK].
        ptx.AppendLine($"    mul.lo.u32 %r17, %r8, {BlockK};");
        ptx.AppendLine("    mul.wide.u32 %rd14, %r17, 4;");
        ptx.AppendLine("    add.u64 %rd14, %rd20, %rd14;");
        for (int k = 0; k < BlockK; k++)
        {
            // Z fragment: z_tile[(threadRow*TM+i)*r + (r0+k)].
            for (int i = 0; i < ThreadM; i++)
            {
                ptx.AppendLine($"    add.u32 %r22, %r7, {i};");
                ptx.AppendLine($"    mul.lo.u32 %r22, %r22, {Rank};");
                ptx.AppendLine("    add.u32 %r22, %r22, %r9;");            // + r0
                ptx.AppendLine($"    add.u32 %r22, %r22, {k};");          // + kk
                ptx.AppendLine("    mul.wide.u32 %rd15, %r22, 4;");
                ptx.AppendLine("    add.u64 %rd16, %rd7, %rd15;");
                ptx.AppendLine($"    ld.shared.f32 %f{16 + i}, [%rd16];");
            }
            for (int j = 0; j < ThreadN; j++)
                ptx.AppendLine($"    ld.shared.f32 %f{20 + j}, [%rd14+{(j * BlockK + k) * sizeof(float)}];");
            for (int i = 0; i < ThreadM; i++)
                for (int j = 0; j < ThreadN; j++)
                    ptx.AppendLine(
                        $"    fma.rn.f32 %f{i * ThreadN + j}, %f{16 + i}, %f{20 + j}, %f{i * ThreadN + j};");
        }
    }

    // output[m,n] = base[m,n] + scale * dY[m,n].
    private static void EmitStage2Epilogue(StringBuilder ptx, int nBytes)
    {
        ptx.AppendLine("    add.u32 %r18, %r5, %r7;");                      // m0
        ptx.AppendLine("    add.u32 %r19, %r30, %r8;");                     // n0
        for (int i = 0; i < ThreadM; i++)
        {
            ptx.AppendLine($"    add.u32 %r20, %r18, {i};");
            ptx.AppendLine($"    mul.wide.u32 %rd17, %r20, {nBytes};");
            ptx.AppendLine("    add.u64 %rd18, %rd6, %rd17;");             // &base[m,0]
            ptx.AppendLine("    add.u64 %rd19, %rd3, %rd17;");             // &output[m,0]
            for (int j = 0; j < ThreadN; j++)
            {
                int acc = i * ThreadN + j;
                ptx.AppendLine($"    add.u32 %r21, %r19, {j};");
                ptx.AppendLine("    mul.wide.u32 %rd21, %r21, 4;");
                ptx.AppendLine("    add.u64 %rd22, %rd18, %rd21;");
                ptx.AppendLine("    ld.global.nc.f32 %f24, [%rd22];");     // base
                ptx.AppendLine($"    fma.rn.f32 %f{acc}, %f26, %f{acc}, %f24;"); // base + scale*dY
                ptx.AppendLine("    add.u64 %rd23, %rd19, %rd21;");
                ptx.AppendLine($"    st.global.f32 [%rd23], %f{acc};");
            }
        }
    }

    private static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture, int m, int k, int n)
    {
        var x = new DirectPtxExtent(m, k);
        var a = new DirectPtxExtent(Rank, k);
        var b = new DirectPtxExtent(n, Rank);
        var baseOut = new DirectPtxExtent(m, n);
        var output = new DirectPtxExtent(m, n);
        return new DirectPtxKernelBlueprint(
            Operation: "fused-lora-forward",
            Version: 1,
            Architecture: architecture,
            Variant: $"fp32-m{m}-k{k}-n{n}-r{Rank}",
            Tensors:
            [
                new("x", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    x, x, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("loraA", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.LinearWeightOutputMajor,
                    a, a, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("loraB", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.LinearWeightOutputMajor,
                    b, b, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("base", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    baseOut, baseOut, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    output, output, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 120,
                MaxStaticSharedBytes: (BlockM * BlockK + BlockM * BlockK + BlockM * Rank + BlockN * BlockK) * sizeof(float),
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 1),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["formula"] = "output = base + scale * (X @ transpose(A[r,K])) @ transpose(B[N,r])",
                ["rank"] = Rank.ToString(),
                ["factors"] = "output-major-row-major-fp32",
                ["fusion"] = "single-launch-shared-resident-intermediate-Z",
                ["intermediate-Z"] = "computed-once-per-row-block-in-shared",
                ["tile"] = $"{BlockM}x{BlockN}x{BlockK}-register-{ThreadM}x{ThreadN}",
                ["global-intermediates"] = "none",
                ["temporary-device-allocation"] = "none",
                ["stride-parameters"] = "none"
            });
    }

    internal static bool IsSupportedShape(int m, int k, int n) =>
        m > 0 && m % BlockM == 0 &&
        n > 0 && n % BlockN == 0 &&
        k > 0 && k % BlockK == 0 &&
        m is 64 or 128 or 256 or 512 or 1024 or 2048 &&
        k is 256 or 512 or 1024 or 2048 or 4096 &&
        n is 256 or 512 or 1024 or 2048 or 4096;

    internal static bool IsPromotedShape(int m, int k, int n) => false;

    private static void ValidateShape(int m, int k, int n)
    {
        if (!IsSupportedShape(m, k, n))
            throw new ArgumentOutOfRangeException(
                nameof(m),
                "Fused-LoRA supports M in {64,128,256,512,1024,2048}, K/N in {256,512,1024,2048,4096}, r=64.");
    }

    private static void Require(DirectPtxTensorView view, DirectPtxTensorContract contract, string parameter)
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
