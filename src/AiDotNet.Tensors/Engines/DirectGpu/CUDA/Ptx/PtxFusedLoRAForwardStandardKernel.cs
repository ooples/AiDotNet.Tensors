using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Fused LoRA forward in the engine's canonical standard-factor layout:
/// <c>output[M,N] = base[M,N] + scale * (X[M,K] @ loraA[K,R]) @ loraB[R,N]</c>, where
/// loraA and loraB are ordinary row-major factors (down-projection [K,R], up-projection
/// [R,N]) — the exact contract of <c>CudaBackend.FusedLoRAForward</c>. Both rank GEMMs are
/// standard (non-transposed) B products, fused into a single launch: each row-block reduces
/// the intermediate <c>Z[BM,R] = X @ loraA</c> into shared memory once, then streams every
/// N-column tile through <c>Z @ loraB</c> with a fused base+scale residual. No global
/// intermediate, no per-column Z recompute. This is the standard-layout sibling of the
/// output-major <see cref="PtxFusedLoRAForwardKernel"/>.
///
/// The rank R is a runtime specialization in <see cref="SupportedRanks"/> (16/32/64). The
/// resident Z tile is always sized for <see cref="MaxRank"/> (stride 64) so unused columns
/// never collide with neighbouring rows; the loraA load is bounds-guarded on <c>col &lt; R</c>
/// so no out-of-range factor element is read. Tile BM=BN=64, BK=8, TM=TN=4 → 256 threads,
/// 16 FP32 accumulators/thread. Grid = (M/BM).
/// </summary>
internal sealed class PtxFusedLoRAForwardStandardKernel : IDisposable
{
    internal const int BlockM = 64;
    internal const int BlockN = 64;
    internal const int BlockK = 8;
    internal const int ThreadM = 4;
    internal const int ThreadN = 4;
    internal const int MaxRank = 64; // Z-tile column stride; the largest supported rank.
    internal const int BlockThreads = (BlockM / ThreadM) * (BlockN / ThreadN); // 256

    internal static readonly int[] SupportedRanks = { 16, 32, 64 };
    internal const string EntryPoint = "aidotnet_fused_lora_forward_standard";

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int M { get; }
    internal int K { get; }
    internal int N { get; }
    internal int Rank { get; }
    internal float Scale { get; }
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxFusedLoRAForwardStandardKernel(DirectPtxRuntime runtime, int m, int k, int n, int rank, float scale)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasValidatedFusedLinear(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The checked-in standard-layout fused-LoRA specialization is measured only on GA10x/SM86.");
        ValidateShape(m, k, n, rank);
        M = m;
        K = k;
        N = n;
        Rank = rank;
        Scale = scale;
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, m, k, n, rank);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor, m, k, n, rank);
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
            throw new ArgumentException("Fused-LoRA output may not alias X, loraA, or loraB.");

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

    internal static string EmitPtx(int ccMajor, int ccMinor, int m, int k, int n, int rank)
    {
        ValidateShape(m, k, n, rank);
        int kBytes = checked(k * sizeof(float));    // X row stride
        int rBytes = checked(rank * sizeof(float)); // loraA row stride (over R)
        int nBytes = checked(n * sizeof(float));    // loraB row stride, base/output row stride

        var ptx = new StringBuilder(32_000);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine($"// fused-lora-standard M={m} K={k} N={n} r={rank} tile={BlockM}x{BlockN}x{BlockK}");
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
        ptx.AppendLine($"    .shared .align 16 .b8 a_tile[{BlockK * BlockN * sizeof(float)}];");
        ptx.AppendLine($"    .shared .align 16 .b8 z_tile[{BlockM * MaxRank * sizeof(float)}];");
        ptx.AppendLine($"    .shared .align 16 .b8 b_tile[{BlockK * BlockN * sizeof(float)}];");
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
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine("    shr.u32 %r3, %r0, 4;");
        ptx.AppendLine("    and.b32 %r4, %r0, 15;");
        ptx.AppendLine($"    mul.lo.u32 %r5, %r1, {BlockM};");
        ptx.AppendLine($"    mul.lo.u32 %r7, %r3, {ThreadM};");
        ptx.AppendLine($"    mul.lo.u32 %r8, %r4, {ThreadN};");

        // ---- Stage 1: Z[BM,R] = X[BM,K] @ loraA[K,R] (standard-B, loraA col-guarded) ----
        for (int i = 0; i < ThreadM * ThreadN; i++)
            ptx.AppendLine($"    mov.f32 %f{i}, 0f00000000;");
        ptx.AppendLine("    mov.u32 %r9, 0;");
        ptx.AppendLine("STAGE1_K_LOOP:");
        EmitStage1Load(ptx, kBytes, rBytes, rank);
        ptx.AppendLine("    bar.sync 0;");
        EmitStandardInner(ptx, "%rd13", "%rd14");
        ptx.AppendLine("    bar.sync 0;");
        ptx.AppendLine($"    add.u32 %r9, %r9, {BlockK};");
        ptx.AppendLine($"    setp.lt.u32 %p0, %r9, {k};");
        ptx.AppendLine("    @%p0 bra.uni STAGE1_K_LOOP;");
        // Store Z accumulators into z_tile[(threadRow*TM+i)*MaxRank + (threadCol*TN+j)].
        for (int i = 0; i < ThreadM; i++)
            for (int j = 0; j < ThreadN; j++)
            {
                ptx.AppendLine($"    add.u32 %r22, %r7, {i};");
                ptx.AppendLine($"    add.u32 %r23, %r8, {j};");
                ptx.AppendLine($"    mul.lo.u32 %r24, %r22, {MaxRank};");
                ptx.AppendLine("    add.u32 %r24, %r24, %r23;");
                ptx.AppendLine("    mul.wide.u32 %rd15, %r24, 4;");
                ptx.AppendLine("    add.u64 %rd16, %rd7, %rd15;");
                ptx.AppendLine($"    st.shared.f32 [%rd16], %f{i * ThreadN + j};");
            }
        ptx.AppendLine("    bar.sync 0;");

        // ---- Stage 2: for each N tile, out = base + scale * Z @ loraB (standard-B) ----
        ptx.AppendLine("    mov.u32 %r30, 0;");                      // nb = column-block base
        ptx.AppendLine("COL_LOOP:");
        for (int i = 0; i < ThreadM * ThreadN; i++)
            ptx.AppendLine($"    mov.f32 %f{i}, 0f00000000;");
        ptx.AppendLine("    mov.u32 %r9, 0;");                       // r0 within rank
        ptx.AppendLine("STAGE2_R_LOOP:");
        EmitStage2LoadB(ptx, nBytes);
        ptx.AppendLine("    bar.sync 0;");
        EmitStage2Inner(ptx);
        ptx.AppendLine("    bar.sync 0;");
        ptx.AppendLine($"    add.u32 %r9, %r9, {BlockK};");
        ptx.AppendLine($"    setp.lt.u32 %p1, %r9, {rank};");
        ptx.AppendLine("    @%p1 bra.uni STAGE2_R_LOOP;");
        EmitStage2Epilogue(ptx, nBytes);
        ptx.AppendLine($"    add.u32 %r30, %r30, {BlockN};");
        ptx.AppendLine($"    setp.lt.u32 %p2, %r30, {n};");
        ptx.AppendLine("    @%p2 bra.uni COL_LOOP;");

        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    // Stage 1: X tile [BM][BK] (rows over M, cols over K) and loraA tile [BK][BN] (rows over
    // K, cols over R). loraA columns >= rank load 0 (bounds guard), so their Z columns are 0
    // and are never read in stage 2. Leaves fragment bases in %rd13/%rd14.
    private static void EmitStage1Load(StringBuilder ptx, int kBytes, int rBytes, int rank)
    {
        for (int slot = 0; slot < 2; slot++)
        {
            string e = slot == 0 ? "%r10" : "%r11";
            ptx.AppendLine($"    add.u32 {e}, %r0, {slot * BlockThreads};");
            // X: row = e>>3 (over M), kk = e&7 (over K).
            ptx.AppendLine($"    shr.u32 %r12, {e}, 3;");
            ptx.AppendLine($"    and.b32 %r13, {e}, 7;");
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
            // loraA: row = e>>6 (over K within tile), col = e&63 (over R). global K row = k0 + row.
            ptx.AppendLine($"    shr.u32 %r16, {e}, 6;");
            ptx.AppendLine($"    and.b32 %r17, {e}, 63;");
            ptx.AppendLine("    add.u32 %r18, %r9, %r16;");                 // global K row
            ptx.AppendLine($"    mul.wide.u32 %rd8, %r18, {rBytes};");
            ptx.AppendLine("    mul.wide.u32 %rd9, %r17, 4;");
            ptx.AppendLine("    add.u64 %rd10, %rd1, %rd8;");
            ptx.AppendLine("    add.u64 %rd10, %rd10, %rd9;");
            ptx.AppendLine("    mov.f32 %f16, 0f00000000;");                // default 0 for col >= rank
            ptx.AppendLine($"    setp.lt.u32 %p3, %r17, {rank};");
            ptx.AppendLine("    @%p3 ld.global.nc.f32 %f16, [%rd10];");
            ptx.AppendLine("    add.u64 %rd12, %rd5, %rd11;");
            ptx.AppendLine("    st.shared.f32 [%rd12], %f16;");
        }
        // Fragment bases: X row-major [BM][BK] -> a_frag base &x_sh[(threadRow*TM)*BK];
        // loraA [BK][BN] -> b_frag base &a_sh[threadCol*TN].
        ptx.AppendLine($"    mul.lo.u32 %r16, %r7, {BlockK};");
        ptx.AppendLine("    mul.wide.u32 %rd13, %r16, 4;");
        ptx.AppendLine("    add.u64 %rd13, %rd4, %rd13;");
        ptx.AppendLine("    mul.wide.u32 %rd14, %r8, 4;");
        ptx.AppendLine("    add.u64 %rd14, %rd5, %rd14;");
    }

    // Standard-B register-blocked accumulate over one BK slice: A fragment row-major over
    // BK (stride BK), B fragment row-major over BN (stride BN). Bases supplied.
    private static void EmitStandardInner(StringBuilder ptx, string aBase, string bBase)
    {
        for (int k = 0; k < BlockK; k++)
        {
            for (int i = 0; i < ThreadM; i++)
                ptx.AppendLine($"    ld.shared.f32 %f{16 + i}, [{aBase}+{(i * BlockK + k) * sizeof(float)}];");
            for (int j = 0; j < ThreadN; j++)
                ptx.AppendLine($"    ld.shared.f32 %f{20 + j}, [{bBase}+{(k * BlockN + j) * sizeof(float)}];");
            for (int i = 0; i < ThreadM; i++)
                for (int j = 0; j < ThreadN; j++)
                    ptx.AppendLine(
                        $"    fma.rn.f32 %f{i * ThreadN + j}, %f{16 + i}, %f{20 + j}, %f{i * ThreadN + j};");
        }
    }

    // Stage 2 load of the BK×BN loraB slice: loraB row = r0 + row (over R), col = nb + col (over N).
    private static void EmitStage2LoadB(StringBuilder ptx, int nBytes)
    {
        for (int slot = 0; slot < 2; slot++)
        {
            string e = slot == 0 ? "%r10" : "%r11";
            ptx.AppendLine($"    add.u32 {e}, %r0, {slot * BlockThreads};");
            ptx.AppendLine($"    shr.u32 %r12, {e}, 6;");                   // row over R within tile
            ptx.AppendLine($"    and.b32 %r13, {e}, 63;");                  // col over N within tile
            ptx.AppendLine("    add.u32 %r14, %r9, %r12;");                 // global R row = r0 + row
            ptx.AppendLine("    add.u32 %r15, %r30, %r13;");                // global N col = nb + col
            ptx.AppendLine($"    mul.wide.u32 %rd8, %r14, {nBytes};");
            ptx.AppendLine("    mul.wide.u32 %rd9, %r15, 4;");
            ptx.AppendLine("    add.u64 %rd10, %rd2, %rd8;");
            ptx.AppendLine("    add.u64 %rd10, %rd10, %rd9;");
            ptx.AppendLine("    ld.global.nc.f32 %f16, [%rd10];");
            ptx.AppendLine($"    mul.wide.u32 %rd11, {e}, 4;");
            ptx.AppendLine("    add.u64 %rd12, %rd20, %rd11;");
            ptx.AppendLine("    st.shared.f32 [%rd12], %f16;");
        }
    }

    // out[myRow,myCol] += sum_kk Z[myRow, r0+kk] * loraB_tile[kk, myCol]. Z stride = MaxRank.
    private static void EmitStage2Inner(StringBuilder ptx)
    {
        // loraB fragment base: &b_tile[threadCol*TN] (row-major [BK][BN], stride BN).
        ptx.AppendLine("    mul.wide.u32 %rd14, %r8, 4;");
        ptx.AppendLine("    add.u64 %rd14, %rd20, %rd14;");
        for (int k = 0; k < BlockK; k++)
        {
            // Z fragment: z_tile[(threadRow*TM+i)*MaxRank + (r0+k)].
            for (int i = 0; i < ThreadM; i++)
            {
                ptx.AppendLine($"    add.u32 %r22, %r7, {i};");
                ptx.AppendLine($"    mul.lo.u32 %r22, %r22, {MaxRank};");
                ptx.AppendLine("    add.u32 %r22, %r22, %r9;");            // + r0
                ptx.AppendLine($"    add.u32 %r22, %r22, {k};");          // + kk
                ptx.AppendLine("    mul.wide.u32 %rd15, %r22, 4;");
                ptx.AppendLine("    add.u64 %rd16, %rd7, %rd15;");
                ptx.AppendLine($"    ld.shared.f32 %f{16 + i}, [%rd16];");
            }
            for (int j = 0; j < ThreadN; j++)
                ptx.AppendLine($"    ld.shared.f32 %f{20 + j}, [%rd14+{(k * BlockN + j) * sizeof(float)}];");
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
        DirectPtxArchitectureFamily architecture, int m, int k, int n, int rank)
    {
        var x = new DirectPtxExtent(m, k);
        var a = new DirectPtxExtent(k, rank);
        var b = new DirectPtxExtent(rank, n);
        var baseOut = new DirectPtxExtent(m, n);
        var output = new DirectPtxExtent(m, n);
        return new DirectPtxKernelBlueprint(
            Operation: "fused-lora-forward-standard",
            Version: 1,
            Architecture: architecture,
            Variant: $"fp32-m{m}-k{k}-n{n}-r{rank}",
            Tensors:
            [
                new("x", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    x, x, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("loraA", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    a, a, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("loraB", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    b, b, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("base", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    baseOut, baseOut, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    output, output, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 120,
                MaxStaticSharedBytes: (BlockM * BlockK + BlockK * BlockN + BlockM * MaxRank + BlockK * BlockN) * sizeof(float),
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 1),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["formula"] = "output = base + scale * (X @ loraA[K,R]) @ loraB[R,N]",
                ["rank"] = rank.ToString(),
                ["factors"] = "standard-row-major-fp32-down[K,R]-up[R,N]",
                ["fusion"] = "single-launch-shared-resident-intermediate-Z",
                ["z-stride"] = MaxRank.ToString(),
                ["tile"] = $"{BlockM}x{BlockN}x{BlockK}-register-{ThreadM}x{ThreadN}",
                ["global-intermediates"] = "none",
                ["temporary-device-allocation"] = "none",
                ["stride-parameters"] = "none"
            });
    }

    internal static bool IsSupportedRank(int rank) => Array.IndexOf(SupportedRanks, rank) >= 0;

    internal static bool IsSupportedShape(int m, int k, int n, int rank) =>
        IsSupportedRank(rank) &&
        m > 0 && m % BlockM == 0 &&
        n > 0 && n % BlockN == 0 &&
        k > 0 && k % BlockK == 0 &&
        m is 64 or 128 or 256 or 512 or 1024 or 2048 &&
        k is 256 or 512 or 1024 or 2048 or 4096 &&
        n is 256 or 512 or 1024 or 2048 or 4096;

    internal static bool IsPromotedShape(int m, int k, int n, int rank) => false;

    private static void ValidateShape(int m, int k, int n, int rank)
    {
        if (!IsSupportedShape(m, k, n, rank))
            throw new ArgumentOutOfRangeException(
                nameof(m),
                "Standard fused-LoRA supports M in {64,128,256,512,1024,2048}, K/N in {256,512,1024,2048,4096}, r in {16,32,64}.");
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
