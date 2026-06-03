#if !NET471
using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;

namespace AiDotNet.Tensors.Engines.Simd;

/// <summary>
/// Runtime-JIT'd AVX2 SGEMM for float, row-major C[M,N] = A[M,K]·B[K,N].
///
/// <para>A C# "Xbyak": emits a hand-scheduled, register-blocked 6×16 AVX2/FMA
/// microkernel as raw machine code into W^X executable memory and calls it via a
/// <c>delegate* unmanaged</c> pointer. The microkernel is UNPACKED (reads A via
/// <c>lda</c>, B via <c>ldb</c>, writes C via <c>ldc</c>) so it needs no per-call
/// packing or scatter, and K + strides are runtime arguments — so a SINGLE emitted
/// kernel serves every shape (no per-shape JIT cache needed). The aligned interior
/// (M−M%6 rows × N−N%16 cols) runs on the JIT kernel parallelized over
/// <see cref="Helpers.PersistentParallelExecutor"/> (one pool — no oversubscription,
/// unlike a bolted-on native BLAS); the M/N tails run on a managed scalar edge path.</para>
///
/// <para>Measured (AVX2, 16 threads, our own pool): 600–700 GFLOP/s on the small-K/N
/// transformer shapes (QKV/FFN) vs ~70–113 for the generic managed kernel and ~540
/// for oneDNN. Windows-x64 + AVX2 + FMA only; <see cref="Available"/> is false
/// otherwise and callers fall back to the managed kernel.</para>
/// </summary>
internal static unsafe class JitGemmAvx2
{
    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern IntPtr VirtualAlloc(IntPtr addr, UIntPtr size, uint type, uint protect);
    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool VirtualProtect(IntPtr addr, UIntPtr size, uint newProtect, out uint old);
    private const uint MEM_COMMIT = 0x1000, MEM_RESERVE = 0x2000, PAGE_RW = 0x04, PAGE_EXEC_R = 0x20;

    // void kernel(float* A, float* B, float* C, long K, long lda, long ldb, long ldc)
    private static readonly delegate* unmanaged[Cdecl]<float*, float*, float*, long, long, long, long, void> _kernel;

    /// <summary>True when the JIT kernel was emitted successfully (Windows x64 + AVX2 + FMA).</summary>
    internal static readonly bool Available;

    static JitGemmAvx2()
    {
        try
        {
            if (!OperatingSystem.IsWindows()
                || RuntimeInformation.ProcessArchitecture != Architecture.X64
                || !Avx2.IsSupported || !Fma.IsSupported)
            {
                Available = false;
                return;
            }
            byte[] mc = Emit6x16Unpacked();
            IntPtr mem = VirtualAlloc(IntPtr.Zero, (UIntPtr)mc.Length, MEM_COMMIT | MEM_RESERVE, PAGE_RW);
            if (mem == IntPtr.Zero) { Available = false; return; }
            Marshal.Copy(mc, 0, mem, mc.Length);
            if (!VirtualProtect(mem, (UIntPtr)mc.Length, PAGE_EXEC_R, out _)) { Available = false; return; }
            _kernel = (delegate* unmanaged[Cdecl]<float*, float*, float*, long, long, long, long, void>)mem;
            Available = true;
        }
        catch { Available = false; }
    }

    private const int JIT_MIN_WORK = 1 << 18; // skip tiny GEMMs (dispatch not amortized)

    // Opt-in master gate (AIDOTNET_JIT_GEMM=1). Centralized here so every GEMM
    // entry point (Sgemm, SgemmWithCachedB, TensorMatMul2D, FusedLinear, MlpForward,
    // CompiledMlp) can route through TryMultiply without each re-reading the env.
    private static readonly bool _enabled =
        System.Environment.GetEnvironmentVariable("AIDOTNET_JIT_GEMM") == "1";


    /// <summary>
    /// C[M,N] = A[M,K]·B[K,N], row-major, contiguous (lda=K, ldb=N, ldc=N).
    /// Returns false (no mutation) when unavailable / too small, so the caller
    /// falls back to the managed kernel.
    /// </summary>
    // Per-shape auto-tune cache: true=JIT wins, false=managed wins. On first
    // encounter of a (M,N,K) we time both and lock in the winner — so the JIT
    // engages ONLY where it's actually faster than the managed kernel end-to-end
    // (it wins large-M GEMM, loses small-M like MLP/LSTM gates), with no fixed
    // threshold to mis-tune. Weights/shapes are stable across inferences, so the
    // one-time tuning cost amortizes immediately.
    private static readonly System.Collections.Concurrent.ConcurrentDictionary<(int, int, int), bool> _decision = new();

    internal static bool TryMultiply(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> c, int M, int N, int K)
    {
        if (!_enabled || !Available || M < 6 || N < 16 || (long)M * N * K < JIT_MIN_WORK) return false;

        var key = (M, N, K);
        if (_decision.TryGetValue(key, out bool useJit))
        {
            if (!useJit) return false;
            RunJit(a, b, c, M, N, K);
            return true;
        }
        return TuneAndRun(a, b, c, M, N, K, key);
    }

    // First-encounter tuning: time JIT (into c) vs the managed kernel (into a
    // scratch), cache the winner. If managed wins, returns false so the caller
    // runs its own (managed) path into c.
    private static bool TuneAndRun(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> c, int M, int N, int K, (int, int, int) key)
    {
        float[] scratch = System.Buffers.ArrayPool<float>.Shared.Rent(M * N);
        try
        {
            long jit = long.MaxValue, mgd = long.MaxValue;
            for (int r = 0; r < 4; r++)
            {
                long t0 = System.Diagnostics.Stopwatch.GetTimestamp();
                RunJit(a, b, c, M, N, K);
                long t1 = System.Diagnostics.Stopwatch.GetTimestamp();
                // Managed reference: SgemmAddInternal is the managed core (no JIT re-entry).
                SimdGemm.SgemmAddInternal(a, K, false, b, N, false, scratch.AsSpan(0, M * N), M, K, N, allowParallel: true, clearedOutput: true);
                long t2 = System.Diagnostics.Stopwatch.GetTimestamp();
                if (t1 - t0 < jit) jit = t1 - t0;
                if (t2 - t1 < mgd) mgd = t2 - t1;
            }
            bool jitWins = jit <= mgd;
            _decision[key] = jitWins;
            if (jitWins) { RunJit(a, b, c, M, N, K); return true; } // c re-run clean (scratch had managed)
            return false;
        }
        finally { System.Buffers.ArrayPool<float>.Shared.Return(scratch); }
    }

    private static void RunJit(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> c, int M, int N, int K)
    {
        int Mfull = M - M % 6, Nfull = N - N % 16;
        int numRB = Mfull / 6, numCB = Nfull / 16;

        fixed (float* pa = a, pb = b, pc = c)
        {
            nint A = (nint)pa, B = (nint)pb, C = (nint)pc;
            int kk = K, nn = N, ncb = numCB;

            if (numRB > 0 && numCB > 0)
            {
                // Parallelize over PersistentParallelExecutor only when the GEMM is
                // big enough to amortize the dispatch. Small GEMMs (e.g. a low-batch
                // transformer's QKV/FFN with few row-blocks) run SERIALLY on the
                // calling thread — the per-GEMM PPE wake-up otherwise dwarfs the
                // compute and regresses the forward pass (the kernel is fast; the
                // dispatch is the cost). Measured crossover ~ a few M·N·K.
                bool parallel = (long)M * N * K >= 4_000_000 && numRB >= 2;
                if (parallel)
                {
                    int maxT = Helpers.CpuParallelSettings.MaxDegreeOfParallelism;
                    int chunks = Math.Max(1, Math.Min(numRB, maxT));
                    int perChunk = (numRB + chunks - 1) / chunks;
                    int totalRB = numRB;
                    Helpers.PersistentParallelExecutor.Instance.Execute(chunks, chunk =>
                    {
                        int lo = chunk * perChunk, hi = Math.Min(lo + perChunk, totalRB);
                        var kf = _kernel;
                        float* aP = (float*)A, bP = (float*)B, cP = (float*)C;
                        for (int rb = lo; rb < hi; rb++)
                        {
                            float* aRow = aP + (long)(rb * 6) * kk;
                            float* cRow = cP + (long)(rb * 6) * nn;
                            for (int cb = 0; cb < ncb; cb++)
                                kf(aRow, bP + cb * 16, cRow + cb * 16, kk, kk, nn, nn);
                        }
                    });
                }
                else
                {
                    var kf = _kernel;
                    float* aP = (float*)A, bP = (float*)B, cP = (float*)C;
                    for (int rb = 0; rb < numRB; rb++)
                    {
                        float* aRow = aP + (long)(rb * 6) * kk;
                        float* cRow = cP + (long)(rb * 6) * nn;
                        for (int cb = 0; cb < ncb; cb++)
                            kf(aRow, bP + cb * 16, cRow + cb * 16, kk, kk, nn, nn);
                    }
                }
            }
        }

        // ---- managed edges (tiny): right strip cols [Nfull,N) × all rows;
        //      bottom strip rows [Mfull,M) × cols [0,Nfull). No overlap. ----
        if (Nfull < N) EdgeBlock(a, b, c, M, N, K, 0, M, Nfull, N);
        if (Mfull < M && Nfull > 0) EdgeBlock(a, b, c, M, N, K, Mfull, M, 0, Nfull);
    }

    private static void EdgeBlock(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> c,
        int M, int N, int K, int i0, int i1, int j0, int j1)
    {
        for (int i = i0; i < i1; i++)
        {
            int aRow = i * K, cRow = i * N;
            for (int j = j0; j < j1; j++)
            {
                float acc = 0f;
                for (int k = 0; k < K; k++) acc += a[aRow + k] * b[k * N + j];
                c[cRow + j] = acc;
            }
        }
    }

    // ===================== machine-code emitter =====================
    // Registers: ymm0..11 = 12 accumulators (6 rows × 2 cols), ymm12/13 = B halves,
    // ymm14 = A broadcast. 3-base-pointer trick for the 6 strided A rows: rows 0/2/4
    // bases in RCX/RBX/RSI, index RAX=lda*4, so [base] and [base+RAX] cover all rows.
    // Win64 args: RCX=A RDX=B R8=C R9=K ; lda=[rsp+0x28] ldb=[rsp+0x30] ldc=[rsp+0x38].
    private static byte[] Emit6x16Unpacked()
    {
        var code = new List<byte>(512);
        void Bytes(params byte[] bs) => code.AddRange(bs);
        void Vrr(int mm, int pp, byte op, int reg, int vvvv, int rm) => Bytes(0xC4,
            (byte)(((reg < 8 ? 1 : 0) << 7) | (1 << 6) | ((rm < 8 ? 1 : 0) << 5) | mm),
            (byte)((((~vvvv) & 0xF) << 3) | (1 << 2) | pp),
            op, (byte)(0xC0 | ((reg & 7) << 3) | (rm & 7)));
        void Vrm(int mm, int pp, byte op, int reg, int b, int disp)
        {
            Bytes(0xC4, (byte)(((reg < 8 ? 1 : 0) << 7) | (1 << 6) | ((b < 8 ? 1 : 0) << 5) | mm),
                  (byte)((0xF << 3) | (1 << 2) | pp), op);
            int rm = b & 7; bool sib = rm == 4; int mod = disp == 0 ? 0 : (disp is >= -128 and <= 127 ? 1 : 2);
            Bytes((byte)((mod << 6) | ((reg & 7) << 3) | rm));
            if (sib) Bytes(0x24);
            if (mod == 1) Bytes((byte)(sbyte)disp);
            else if (mod == 2) Bytes((byte)disp, (byte)(disp >> 8), (byte)(disp >> 16), (byte)(disp >> 24));
        }
        void VrmBase(int mm, int pp, byte op, int reg, int b) => Bytes(0xC4,
            (byte)(((reg < 8 ? 1 : 0) << 7) | (1 << 6) | ((b < 8 ? 1 : 0) << 5) | mm),
            (byte)((0xF << 3) | (1 << 2) | pp), op, (byte)(((reg & 7) << 3) | (b & 7)));
        void VrmIdx(int mm, int pp, byte op, int reg, int b, int idx) => Bytes(0xC4,
            (byte)(((reg < 8 ? 1 : 0) << 7) | ((idx < 8 ? 1 : 0) << 6) | ((b < 8 ? 1 : 0) << 5) | mm),
            (byte)((0xF << 3) | (1 << 2) | pp), op, (byte)(((reg & 7) << 3) | 4), (byte)(((idx & 7) << 3) | (b & 7)));
        void Mov128(byte op, int reg, int b, int disp)
        {
            Bytes(0xC4, (byte)(((reg < 8 ? 1 : 0) << 7) | (1 << 6) | ((b < 8 ? 1 : 0) << 5) | 1),
                  (byte)((0xF << 3) | 0), op);
            int rm = b & 7; bool sib = rm == 4; int mod = disp == 0 ? 0 : 1;
            Bytes((byte)((mod << 6) | ((reg & 7) << 3) | rm)); if (sib) Bytes(0x24); if (mod == 1) Bytes((byte)disp);
        }
        void MovFromStack(int dst, byte disp) => Bytes((byte)(0x48 | (dst >= 8 ? 4 : 0)), 0x8B, (byte)(0x44 | ((dst & 7) << 3)), 0x24, disp);
        void Shl4(int r) => Bytes((byte)(0x48 | (r >= 8 ? 1 : 0)), 0xC1, (byte)(0xE0 | (r & 7)), 2);
        void Lea(int dst, int b, int idx, int scale)
        {
            int sc = scale == 2 ? 1 : scale == 4 ? 2 : scale == 8 ? 3 : 0;
            Bytes((byte)(0x48 | (dst >= 8 ? 4 : 0) | (idx >= 8 ? 2 : 0) | (b >= 8 ? 1 : 0)), 0x8D,
                  (byte)(((dst & 7) << 3) | 4), (byte)((sc << 6) | ((idx & 7) << 3) | (b & 7)));
        }
        void AddImm(int r, byte imm) => Bytes((byte)(0x48 | (r >= 8 ? 1 : 0)), 0x83, (byte)(0xC0 | (r & 7)), imm);
        void AddRegReg(int dst, int src) => Bytes((byte)(0x48 | (src >= 8 ? 4 : 0) | (dst >= 8 ? 1 : 0)), 0x01, (byte)(0xC0 | ((src & 7) << 3) | (dst & 7)));
        void Push(int r) { if (r >= 8) Bytes(0x41); Bytes((byte)(0x50 | (r & 7))); }
        void Pop(int r) { if (r >= 8) Bytes(0x41); Bytes((byte)(0x58 | (r & 7))); }
        const int RAX = 0, RCX = 1, RDX = 2, RBX = 3, RSP = 4, RSI = 6, R8 = 8, R10 = 10, R11 = 11;

        MovFromStack(RAX, 0x28); Shl4(RAX);  // rax = lda*4
        MovFromStack(R10, 0x30); Shl4(R10);  // r10 = ldb*4
        MovFromStack(R11, 0x38); Shl4(R11);  // r11 = ldc*4
        Push(RBX); Push(RSI);
        Bytes(0x48, 0x81, 0xEC, 160, 0, 0, 0);                 // sub rsp,160
        for (int x = 6; x <= 15; x++) Mov128(0x11, x, RSP, (x - 6) * 16); // save xmm6..15
        Lea(RBX, RCX, RAX, 2);  // row2 base = A + 2*lda
        Lea(RSI, RCX, RAX, 4);  // row4 base = A + 4*lda
        for (int d = 0; d < 12; d++) Vrr(1, 0, 0x57, d, d, d);  // zero ymm0..11

        int loop = code.Count;
        Vrm(1, 0, 0x10, 12, RDX, 0); Vrm(1, 0, 0x10, 13, RDX, 32);   // B halves
        VrmBase(2, 1, 0x18, 14, RCX); Vrr(2, 1, 0xB8, 0, 14, 12); Vrr(2, 1, 0xB8, 1, 14, 13);
        VrmIdx(2, 1, 0x18, 14, RCX, RAX); Vrr(2, 1, 0xB8, 2, 14, 12); Vrr(2, 1, 0xB8, 3, 14, 13);
        VrmBase(2, 1, 0x18, 14, RBX); Vrr(2, 1, 0xB8, 4, 14, 12); Vrr(2, 1, 0xB8, 5, 14, 13);
        VrmIdx(2, 1, 0x18, 14, RBX, RAX); Vrr(2, 1, 0xB8, 6, 14, 12); Vrr(2, 1, 0xB8, 7, 14, 13);
        VrmBase(2, 1, 0x18, 14, RSI); Vrr(2, 1, 0xB8, 8, 14, 12); Vrr(2, 1, 0xB8, 9, 14, 13);
        VrmIdx(2, 1, 0x18, 14, RSI, RAX); Vrr(2, 1, 0xB8, 10, 14, 12); Vrr(2, 1, 0xB8, 11, 14, 13);
        AddImm(RCX, 4); AddImm(RBX, 4); AddImm(RSI, 4); AddRegReg(RDX, R10);
        Bytes(0x49, 0xFF, 0xC9);                               // dec r9
        int after = code.Count;
        Bytes(0x75, (byte)(sbyte)(loop - (after + 2)));        // jnz loop

        for (int r = 0; r < 6; r++)
        {
            Vrm(1, 0, 0x11, r * 2, R8, 0); Vrm(1, 0, 0x11, r * 2 + 1, R8, 32);
            if (r < 5) AddRegReg(R8, R11);
        }
        for (int x = 6; x <= 15; x++) Mov128(0x10, x, RSP, (x - 6) * 16); // restore xmm
        Bytes(0x48, 0x81, 0xC4, 160, 0, 0, 0);                 // add rsp,160
        Pop(RSI); Pop(RBX);
        Bytes(0xC5, 0xF8, 0x77); Bytes(0xC3);                  // vzeroupper; ret
        return code.ToArray();
    }
}
#endif
