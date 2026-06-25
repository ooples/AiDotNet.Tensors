using System;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Threading;
using AiDotNet.Tensors.Engines.BlasManaged;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using TorchSharp;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>
/// CCX-aware hierarchical GEMM PROTOTYPE (the path to MKL on the 3990X). The flat per-tile scheme caps
/// at ~16-41% of MKL because it re-reads B from DRAM per tile and, when B is shared, the reads cross
/// CCXs over Infinity Fabric. This prototype partitions the N-columns across the 16 CCX domains and pins
/// a thread-group per CCX, so each CCX's B-panel is pulled into ITS OWN 16 MB L3 ONCE and re-used by that
/// CCX's threads with no cross-CCX traffic. Each (ic,jc) tile uses the proven GotoGemmFp32.RunTile.
/// Run: --ab-ccx. Once it beats the per-tile RunParallel toward MKL, it graduates into src as the
/// production parallel path (a persistent pinned pool + the same partition).
/// </summary>
internal static unsafe class CcxGemmBench
{
    [StructLayout(LayoutKind.Sequential)]
    private struct GROUP_AFFINITY { public ulong Mask; public ushort Group; public ushort R0, R1, R2; }

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool SetThreadGroupAffinity(IntPtr hThread, ref GROUP_AFFINITY ga, IntPtr prev);
    [DllImport("kernel32.dll")]
    private static extern IntPtr GetCurrentThread();
    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool GetLogicalProcessorInformationEx(int rel, IntPtr buffer, ref uint len);

    private static (ulong mask, ushort group)[] _ccx = Array.Empty<(ulong, ushort)>();
    private static int _numCcx, _threadsPerCcx;
    private static ManualResetEventSlim[] _go = Array.Empty<ManualResetEventSlim>();
    private static CountdownEvent? _done;
    private static volatile bool _shutdown;

    // GEMM params set per Run.
    private static float* _a, _b, _c;
    private static int _m, _n, _k, _mc, _nc, _kc, _lda, _ldb, _ldc;

    private static (ulong mask, ushort group)[] DetectCcx()
    {
        const int RelationCache = 2;
        uint len = 0;
        GetLogicalProcessorInformationEx(RelationCache, IntPtr.Zero, ref len);
        if (len == 0) return Array.Empty<(ulong, ushort)>();
        IntPtr buf = Marshal.AllocHGlobal((int)len);
        try
        {
            if (!GetLogicalProcessorInformationEx(RelationCache, buf, ref len)) return Array.Empty<(ulong, ushort)>();
            var list = new System.Collections.Generic.List<(ulong, ushort)>();
            long ptr = (long)buf, end = ptr + len;
            while (ptr < end)
            {
                int rel = Marshal.ReadInt32((IntPtr)ptr);
                int size = Marshal.ReadInt32((IntPtr)(ptr + 4));
                if (size <= 0) break;
                if (rel == RelationCache && Marshal.ReadByte((IntPtr)(ptr + 8)) == 3) // Level 3
                {
                    ulong mask = (ulong)Marshal.ReadInt64((IntPtr)(ptr + 40));
                    ushort group = (ushort)Marshal.ReadInt16((IntPtr)(ptr + 48));
                    list.Add((mask, group));
                }
                ptr += size;
            }
            return list.ToArray();
        }
        finally { Marshal.FreeHGlobal(buf); }
    }

    public static void Init(int threadsPerCcx)
    {
        _ccx = DetectCcx();
        _numCcx = _ccx.Length;
        _threadsPerCcx = threadsPerCcx;
        int total = _numCcx * threadsPerCcx;
        _go = new ManualResetEventSlim[total];
        _done = new CountdownEvent(total);
        for (int i = 0; i < total; i++)
        {
            _go[i] = new ManualResetEventSlim(false);
            int id = i;
            var t = new Thread(() => WorkerLoop(id)) { IsBackground = true, Name = $"ccx-{id}" };
            t.Start();
        }
        Thread.Sleep(50); // let workers pin before first dispatch
    }

    private static void WorkerLoop(int id)
    {
        int ccx = id / _threadsPerCcx;
        var ga = new GROUP_AFFINITY { Mask = _ccx[ccx].mask, Group = _ccx[ccx].group };
        SetThreadGroupAffinity(GetCurrentThread(), ref ga, IntPtr.Zero);
        int lane = id % _threadsPerCcx;
        while (true)
        {
            _go[id].Wait();
            _go[id].Reset();
            if (_shutdown) return;
            try { DoWork(ccx, lane); }
            finally { _done!.Signal(); }
        }
    }

    // CCX owns a contiguous jc-block range → its B-columns live in its own L3. The CCX's lanes split the
    // (jc-range × all ic-blocks) tiles round-robin; each tile = GotoGemmFp32.RunTile (per-thread A/B pack).
    private static void DoWork(int ccx, int lane)
    {
        int numJc = (_n + _nc - 1) / _nc;
        int numIc = (_m + _mc - 1) / _mc;
        int jcStart = (int)((long)ccx * numJc / _numCcx);
        int jcEnd = (int)((long)(ccx + 1) * numJc / _numCcx);
        int tileIdx = 0;
        for (int jb = jcStart; jb < jcEnd; jb++)
        {
            int jc = jb * _nc; int effNc = Math.Min(_nc, _n - jc); if (effNc <= 0) continue;
            for (int ib = 0; ib < numIc; ib++)
            {
                if ((tileIdx++ % _threadsPerCcx) != lane) continue;
                int ic = ib * _mc; int effMc = Math.Min(_mc, _m - ic); if (effMc <= 0) continue;
                GotoGemmFp32.RunTile(_a, _lda, _b, _ldb, _c, _ldc, ic, jc, effMc, effNc, _k, _mc, _nc, _kc);
            }
        }
    }

    public static void Run(float* a, int lda, float* b, int ldb, float* c, int ldc,
        int m, int n, int k, int mc, int nc, int kc)
    {
        _a = a; _b = b; _c = c; _lda = lda; _ldb = ldb; _ldc = ldc;
        _m = m; _n = n; _k = k; _mc = mc; _nc = nc; _kc = kc;
        _done!.Reset();
        for (int i = 0; i < _go.Length; i++) _go[i].Set();
        _done.Wait();
    }

    public static void Bench()
    {
        torch.set_grad_enabled(false);
        int P = Environment.ProcessorCount;
        if (!GotoGemmFp32.IsAvailable) { Console.WriteLine("GotoGemmFp32 not available"); return; }
        int tpc = int.TryParse(Environment.GetEnvironmentVariable("CCX_TPC"), out var v) ? v : 4;
        Init(tpc);
        Console.WriteLine($"=== CCX-aware GEMM vs MKL vs per-tile (cores={P}, CCX={_numCcx}, threads/CCX={_threadsPerCcx}) ===");
        int kc = GotoGemmFp32.DefaultKc;
        int saved = CpuParallelSettings.MaxDegreeOfParallelism;
        CpuParallelSettings.MaxDegreeOfParallelism = P;
        var rng = new Random(13);
        var shapes = new (int M, int N, int K, string tag)[]
        {
            (256, 1152, 1152, "attn-proj"), (256, 1152, 4608, "mlp-fc"),
            (1024, 1024, 1024, "square1024"), (2048, 2048, 2048, "square2048"),
        };
        try
        {
            // Warm to steady thermal state.
            { var wa = torch.rand(2048, 2048); var wb = torch.rand(2048, 2048); torch.set_num_threads(P);
              var ww = Stopwatch.StartNew(); while (ww.ElapsedMilliseconds < 1500) { using var _ = torch.matmul(wa, wb); } }

            foreach (var (M, N, K, tag) in shapes)
            {
                var A = new float[(long)M * K]; var B = new float[(long)K * N]; var C = new float[(long)M * N];
                for (int i = 0; i < A.Length; i++) A[i] = (float)(rng.NextDouble() - 0.5);
                for (int i = 0; i < B.Length; i++) B[i] = (float)(rng.NextDouble() - 0.5);
                double gf = 2.0 * M * N * K / 1e9;
                // nc ~ N/numCcx (each CCX ~1 jc-block → all CCXs busy + B fits its L3); mc ~ M/tpc (>=tpc ic-blocks).
                var cands = new (int mc, int nc)[]
                {
                    (RoundMr(Math.Max(48, M / _threadsPerCcx)), RoundNr(Math.Max(64, N / _numCcx))),
                    (RoundMr(Math.Max(48, M / (_threadsPerCcx * 2))), RoundNr(Math.Max(64, N / _numCcx))),
                    (120, RoundNr(Math.Max(64, N / _numCcx))),
                    (RoundMr(Math.Max(48, M / _threadsPerCcx)), RoundNr(Math.Max(64, N / (_numCcx * 2)))),
                };
                double bestGf = 0; int bMc = 0, bNc = 0; string corr = "?";
                fixed (float* pa = A, pb = B, pc = C)
                {
                    foreach (var (mc, nc) in cands)
                    {
                        Run(pa, K, pb, N, pc, N, M, N, K, mc, nc, kc);
                        double maxRel = 0; var rv = new Random(5);
                        for (int t = 0; t < 60; t++)
                        {
                            int r = rv.Next(M), col = rv.Next(N);
                            double s = 0; for (int kk = 0; kk < K; kk++) s += (double)A[(long)r * K + kk] * B[(long)kk * N + col];
                            double err = Math.Abs(C[(long)r * N + col] - s);
                            maxRel = Math.Max(maxRel, Math.Abs(s) > 1e-3 ? err / Math.Abs(s) : err);
                        }
                        double gbest = double.PositiveInfinity;
                        var w = Stopwatch.StartNew(); while (w.ElapsedMilliseconds < 40) Run(pa, K, pb, N, pc, N, M, N, K, mc, nc, kc);
                        for (int r = 0; r < 5; r++)
                        {
                            var sw = Stopwatch.StartNew(); int reps = 0;
                            do { Run(pa, K, pb, N, pc, N, M, N, K, mc, nc, kc); reps++; } while (sw.Elapsed.TotalMilliseconds < 40);
                            sw.Stop(); gbest = Math.Min(gbest, sw.Elapsed.TotalMilliseconds / reps);
                        }
                        double g = gf / (gbest / 1000);
                        Console.WriteLine($"    ccx cand mc={mc,4} nc={nc,4}: {g,7:F0} GF  {(maxRel < 1e-3 ? "OK" : "WRONG " + maxRel.ToString("E1"))}");
                        if (g > bestGf) { bestGf = g; bMc = mc; bNc = nc; corr = maxRel < 1e-3 ? "OK" : "WRONG"; }
                    }
                }
                // Thermally-fair three-way: CCX(best) / per-tile RunParallel / MKL, min-of-10 interleaved.
                var ta = torch.rand(M, K); var tb = torch.rand(K, N);
                var (pmc, pnc, pkc) = GotoGemmFp32.ChooseParallelBlocks(M, N);
                double ccxMs = double.PositiveInfinity, ptMs = double.PositiveInfinity, mklMs = double.PositiveInfinity;
                fixed (float* pa = A, pb = B, pc = C)
                {
                    for (int r = 0; r < 10; r++)
                    {
                        { var sw = Stopwatch.StartNew(); int reps = 0; do { Run(pa, K, pb, N, pc, N, M, N, K, bMc, bNc, kc); reps++; } while (sw.Elapsed.TotalMilliseconds < 30); sw.Stop(); ccxMs = Math.Min(ccxMs, sw.Elapsed.TotalMilliseconds / reps); }
                        { var sw = Stopwatch.StartNew(); int reps = 0; do { GotoGemmFp32.RunParallel(pa, K, pb, N, pc, N, M, N, K, pmc, pnc, pkc); reps++; } while (sw.Elapsed.TotalMilliseconds < 30); sw.Stop(); ptMs = Math.Min(ptMs, sw.Elapsed.TotalMilliseconds / reps); }
                        { var sw = Stopwatch.StartNew(); int reps = 0; do { using var _ = torch.matmul(ta, tb); reps++; } while (sw.Elapsed.TotalMilliseconds < 30); sw.Stop(); mklMs = Math.Min(mklMs, sw.Elapsed.TotalMilliseconds / reps); }
                    }
                }
                double ccxGf = gf / (ccxMs / 1000), ptGf = gf / (ptMs / 1000), mklGf = gf / (mklMs / 1000);
                Console.WriteLine($"{tag,-11} CCX={ccxGf,7:F0} (mc={bMc} nc={bNc})  per-tile={ptGf,7:F0}  MKL={mklGf,7:F0}   CCX/per-tile={(ptGf>0?ccxGf/ptGf:0),4:F2}x  CCX/MKL={(mklGf>0?ccxGf/mklGf*100:0),4:F0}%  {corr}");
            }
        }
        finally { CpuParallelSettings.MaxDegreeOfParallelism = saved; }
    }

    private static int RoundMr(int x) { int r = (x / 6) * 6; return r < 6 ? 6 : r; }
    private static int RoundNr(int x) { int r = (x / 16) * 16; return r < 16 ? 16 : r; }
}
