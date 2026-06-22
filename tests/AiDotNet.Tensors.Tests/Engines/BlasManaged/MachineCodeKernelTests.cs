using System;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using AiDotNet.Tensors.Engines.BlasManaged;
using AiDotNet.Tensors.Helpers;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

/// <summary>
/// Sub-S (#409): validates the hand-emitted machine-code path — first the
/// executable-memory + call infrastructure (a trivial x+1 function), then the
/// 12-accumulator FMA-throughput kernel that breaks RyuJIT's 8-register cap.
/// </summary>
public class MachineCodeKernelTests
{
    private readonly ITestOutputHelper _output;
    public MachineCodeKernelTests(ITestOutputHelper output) { _output = output; }

    private static bool IsX64Windows =>
        RuntimeInformation.OSArchitecture == Architecture.X64
        && RuntimeInformation.IsOSPlatform(OSPlatform.Windows);

    private static bool IsX64Unix =>
        RuntimeInformation.OSArchitecture == Architecture.X64
        && (RuntimeInformation.IsOSPlatform(OSPlatform.Linux) || RuntimeInformation.IsOSPlatform(OSPlatform.OSX));

    [Fact]
    public unsafe void Infra_TrivialFunction_ExecutesAndReturns()
    {
        if (!IsX64Windows && !IsX64Unix) return; // x64-only proof

        // long f(long x) => x + 1.
        // Windows x64: arg0 in RCX. SysV (Linux/macOS): arg0 in RDI.
        byte[] code = IsX64Windows
            ? new byte[] { 0x48, 0x89, 0xC8,       // mov rax, rcx
                           0x48, 0xFF, 0xC0,       // inc rax
                           0xC3 }                  // ret
            : new byte[] { 0x48, 0x89, 0xF8,       // mov rax, rdi
                           0x48, 0xFF, 0xC0,       // inc rax
                           0xC3 };                 // ret

        using var mem = ExecutableMemory.TryAllocate(code);
        if (mem is null) { _output.WriteLine("executable memory not supported — skipping"); return; }

        var fn = (delegate* unmanaged<long, long>)mem.Pointer;
        Assert.Equal(42L, fn(41L));
        Assert.Equal(1L, fn(0L));
        Assert.Equal(-4L, fn(-5L));
    }

    // Wall-clock GFLOPS gate (>44) — flakes under parallel CPU contention in the CI
    // correctness run; tag Performance so the Category!=Performance filter excludes it,
    // matching the repo's other perf gates (#375 de-flake). Correctness of the kernel is
    // covered by the bit-match tests in this class.
    [Trait("Category", "Performance")]
    [Fact]
    public unsafe void Fp64x12_MachineCode_BreaksRyuJitCeiling()
    {
        if (!IsX64Windows) return;                 // proof encoded for Windows x64
        if (!Avx2.IsSupported || !Fma.IsSupported) return;

        byte[] code = MachineCodeFmaKernel.EmitFp64x12Windows();
        using var mem = ExecutableMemory.TryAllocate(code);
        if (mem is null) { _output.WriteLine("executable memory not supported — skipping"); return; }

        var fn = (delegate* unmanaged<long, double*, double*, void>)mem.Pointer;
        const double a = 1.0000001, b = 0.9999999;
        var ab = new double[] { a, b };
        var res = new double[4];

        fixed (double* pab = ab, pres = res)
        {
            // Correctness: each of 12 accumulators sums `iters` copies of a*b, then
            // they're reduced lane-wise → result = 12 * iters * a*b.
            fn(1000, pres, pab);
            double expected = 12.0 * 1000 * (a * b);
            Assert.True(Math.Abs(pres[0] - expected) < expected * 1e-9,
                $"machine-code FMA wrong: got {pres[0]:G17}, expected {expected:G17}");
            // All 4 lanes identical (broadcast operands).
            Assert.Equal(pres[0], pres[1]);
            Assert.Equal(pres[0], pres[3]);

            // Perf: does our own register allocation beat RyuJIT's 8-accumulator
            // ceiling (~44 GFLOPS) and approach the hardware peak (~64)?
            // This is a SINGLE-THREADED tight FMA loop, so its true per-core throughput
            // is the BEST (least time-sliced) window — a single wall-clock sample is
            // depressed when the full suite saturates every core. Take the MAX GFLOPS
            // over several short runs (only one core needs a brief clean slice) with the
            // thread at Highest priority so it wins scheduling. The 44 GFLOPS bar is
            // unchanged — this just measures the kernel's real ceiling under load.
            const long iters = 30_000_000;
            fn(1_000_000, pres, pab); // warm
            double bestGflops = 0.0;
            var prevPriority = System.Threading.Thread.CurrentThread.Priority;
            try
            {
                System.Threading.Thread.CurrentThread.Priority = System.Threading.ThreadPriority.Highest;
                for (int run = 0; run < 30; run++)
                {
                    var sw = Stopwatch.StartNew();
                    fn(iters, pres, pab);
                    sw.Stop();
                    double g = (12.0 * iters * 4 * 2) / sw.Elapsed.TotalSeconds / 1e9; // 12 acc × 4 lanes × 2 flops/FMA
                    if (g > bestGflops) bestGflops = g;
                }
            }
            finally { System.Threading.Thread.CurrentThread.Priority = prevPriority; }
            _output.WriteLine($"machine-code 12-acc FP64 FMA: {bestGflops:F1} GFLOPS " +
                "(best of 30; RyuJIT 8-acc ceiling ~44; hardware peak ~64)");
            Assert.True(bestGflops > 44.0,
                $"machine-code kernel {bestGflops:F1} GFLOPS did not beat the 44 GFLOPS RyuJIT ceiling");
        }
    }

    [Theory]
    [InlineData(1, 8)]
    [InlineData(7, 8)]
    [InlineData(64, 8)]
    [InlineData(256, 8)]
    [InlineData(37, 11)]   // ldc > Nr (padded C), odd kc
    public unsafe void Fp64_6x8_MachineCode_MatchesFmaReference(int kc, int ldc)
    {
        if (!IsX64Windows || !Avx2.IsSupported || !Fma.IsSupported) return;

        const int Mr = 6, Nr = 8;
        byte[] code = MachineCodeFmaKernel.EmitFp64_6x8_PackedWindows();
        using var mem = ExecutableMemory.TryAllocate(code);
        if (mem is null) return;
        var fn = (delegate* unmanaged<double*, double*, double*, long, long, void>)mem.Pointer;

        var rng = new Random(99 + kc * 7 + ldc);
        var packedA = new double[kc * Mr];   // [k*Mr + r]
        var packedB = new double[kc * Nr];   // [k*Nr + col]
        for (int i = 0; i < packedA.Length; i++) packedA[i] = rng.NextDouble() * 2 - 1;
        for (int i = 0; i < packedB.Length; i++) packedB[i] = rng.NextDouble() * 2 - 1;

        int cLen = (Mr - 1) * ldc + Nr;
        var c0 = new double[cLen];
        for (int i = 0; i < cLen; i++) c0[i] = rng.NextDouble() * 2 - 1;
        var cKernel = (double[])c0.Clone();
        var cRef = (double[])c0.Clone();

        fixed (double* pa = packedA, pb = packedB, pc = cKernel)
            fn(pa, pb, pc, ldc, kc);

        // Reference: lane-wise FMA accumulation in k order (matches vfmadd231pd rounding).
        for (int r = 0; r < Mr; r++)
            for (int col = 0; col < Nr; col++)
            {
                double acc = cRef[r * ldc + col];
                for (int k = 0; k < kc; k++)
                    acc = Math.FusedMultiplyAdd(packedA[k * Mr + r], packedB[k * Nr + col], acc);
                cRef[r * ldc + col] = acc;
            }

        for (int i = 0; i < cLen; i++)
            Assert.True(BitConverter.DoubleToInt64Bits(cKernel[i]) == BitConverter.DoubleToInt64Bits(cRef[i]),
                $"6x8 machine kernel mismatch at {i}: kernel={cKernel[i]:G17}, ref={cRef[i]:G17} (kc={kc}, ldc={ldc})");
    }

    [Theory]
    [InlineData(1, 8)]
    [InlineData(7, 8)]
    [InlineData(64, 8)]
    [InlineData(256, 8)]
    [InlineData(37, 11)]
    [InlineData(6, 9)]     // kc divisible by U=4? no (6%4=2) — exercises remainder
    public unsafe void Fp64_6x8_U4_MatchesFmaReference(int kc, int ldc)
    {
        if (!IsX64Windows || !Avx2.IsSupported || !Fma.IsSupported) return;

        const int Mr = 6, Nr = 8;
        byte[] code = MachineCodeFmaKernel.EmitFp64_6x8_PackedWindowsU4();
        using var mem = ExecutableMemory.TryAllocate(code);
        if (mem is null) return;
        var fn = (delegate* unmanaged<double*, double*, double*, long, long, void>)mem.Pointer;

        var rng = new Random(7 + kc * 13 + ldc);
        var packedA = new double[kc * Mr];
        var packedB = new double[kc * Nr];
        for (int i = 0; i < packedA.Length; i++) packedA[i] = rng.NextDouble() * 2 - 1;
        for (int i = 0; i < packedB.Length; i++) packedB[i] = rng.NextDouble() * 2 - 1;
        int cLen = (Mr - 1) * ldc + Nr;
        var c0 = new double[cLen];
        for (int i = 0; i < cLen; i++) c0[i] = rng.NextDouble() * 2 - 1;
        var cKernel = (double[])c0.Clone();
        var cRef = (double[])c0.Clone();

        fixed (double* pa = packedA, pb = packedB, pc = cKernel) fn(pa, pb, pc, ldc, kc);

        for (int r = 0; r < Mr; r++)
            for (int col = 0; col < Nr; col++)
            {
                double acc = cRef[r * ldc + col];
                for (int k = 0; k < kc; k++)
                    acc = Math.FusedMultiplyAdd(packedA[k * Mr + r], packedB[k * Nr + col], acc);
                cRef[r * ldc + col] = acc;
            }
        for (int i = 0; i < cLen; i++)
            Assert.True(BitConverter.DoubleToInt64Bits(cKernel[i]) == BitConverter.DoubleToInt64Bits(cRef[i]),
                $"6x8-U4 mismatch at {i}: kernel={cKernel[i]:G17}, ref={cRef[i]:G17} (kc={kc}, ldc={ldc})");
    }

    [Theory]
    [InlineData(1, 16)]
    [InlineData(7, 16)]
    [InlineData(64, 16)]
    [InlineData(256, 16)]
    [InlineData(37, 19)]   // ldc > Nr (padded C), odd kc
    public unsafe void Fp32_6x16_MachineCode_MatchesFmaReference(int kc, int ldc)
    {
        if (!IsX64Windows || !Avx2.IsSupported || !Fma.IsSupported) return;

        const int Mr = 6, Nr = 16;
        byte[] code = MachineCodeFmaKernel.EmitFp32_6x16_PackedWindows();
        using var mem = ExecutableMemory.TryAllocate(code);
        if (mem is null) return;
        var fn = (delegate* unmanaged<float*, float*, float*, long, long, void>)mem.Pointer;

        var rng = new Random(13 + kc * 17 + ldc);
        var packedA = new float[kc * Mr];   // [k*Mr + r]
        var packedB = new float[kc * Nr];   // [k*Nr + col]
        for (int i = 0; i < packedA.Length; i++) packedA[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < packedB.Length; i++) packedB[i] = (float)(rng.NextDouble() * 2 - 1);

        int cLen = (Mr - 1) * ldc + Nr;
        var c0 = new float[cLen];
        for (int i = 0; i < cLen; i++) c0[i] = (float)(rng.NextDouble() * 2 - 1);
        var cKernel = (float[])c0.Clone();
        var cRef = (float[])c0.Clone();

        fixed (float* pa = packedA, pb = packedB, pc = cKernel)
            fn(pa, pb, pc, ldc, kc);

        // Reference: lane-wise FMA accumulation in k order (matches vfmadd231ps rounding).
        for (int r = 0; r < Mr; r++)
            for (int col = 0; col < Nr; col++)
            {
                float acc = cRef[r * ldc + col];
                for (int k = 0; k < kc; k++)
                    acc = MathF.FusedMultiplyAdd(packedA[k * Mr + r], packedB[k * Nr + col], acc);
                cRef[r * ldc + col] = acc;
            }

        for (int i = 0; i < cLen; i++)
            Assert.True(BitConverter.SingleToInt32Bits(cKernel[i]) == BitConverter.SingleToInt32Bits(cRef[i]),
                $"6x16 FP32 machine kernel mismatch at {i}: kernel={cKernel[i]:G9}, ref={cRef[i]:G9} (kc={kc}, ldc={ldc})");
    }

    [Theory]
    [InlineData(256, 4, 64)]    // 4 tiles, ldc == nTiles*Nr
    [InlineData(64, 3, 48)]
    [InlineData(7, 8, 128)]
    [InlineData(1, 2, 32)]      // kc=1, smallest panel
    [InlineData(128, 5, 83)]    // ldc > nTiles*Nr (padded C)
    public unsafe void Fp32_6x16_Panel_MatchesPerTileReference(int kc, int nTiles, int ldc)
    {
        if (!IsX64Windows || !Avx2.IsSupported || !Fma.IsSupported) return;

        const int Mr = 6, Nr = 16;
        Assert.True(ldc >= nTiles * Nr, "test setup: ldc must hold nTiles N-tile blocks");

        byte[] code = MachineCodeFmaKernel.EmitFp32_6x16_PanelWindows();
        using var mem = ExecutableMemory.TryAllocate(code);
        if (mem is null) return;
        // void kern(float* packedA, float* packedB, float* c, long ldc, long kc, long nTiles)
        var fn = (delegate* unmanaged<float*, float*, float*, long, long, long, void>)mem.Pointer;

        var rng = new Random(409 + kc * 31 + nTiles * 7 + ldc);
        var packedA = new float[kc * Mr];                       // ONE stripe, reused for every tile
        var packedB = new float[(long)nTiles * kc * Nr];        // contiguous [tile][k][Nr] panel
        for (int i = 0; i < packedA.Length; i++) packedA[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < packedB.Length; i++) packedB[i] = (float)(rng.NextDouble() * 2 - 1);

        int cLen = (Mr - 1) * ldc + nTiles * Nr;
        var c0 = new float[cLen];
        for (int i = 0; i < cLen; i++) c0[i] = (float)(rng.NextDouble() * 2 - 1);
        var cKernel = (float[])c0.Clone();
        var cRef = (float[])c0.Clone();

        fixed (float* pa = packedA, pb = packedB, pc = cKernel)
            fn(pa, pb, pc, ldc, kc, nTiles);

        // Reference: nTiles independent 6×16 tiles, lane-wise FMA in k order (matches the panel,
        // which must equal calling the single-tile kernel nTiles times). Tile js reads B at
        // js*kc*Nr and writes C columns [js*Nr, js*Nr+Nr).
        for (int js = 0; js < nTiles; js++)
            for (int r = 0; r < Mr; r++)
                for (int col = 0; col < Nr; col++)
                {
                    float acc = cRef[r * ldc + js * Nr + col];
                    for (int k = 0; k < kc; k++)
                        acc = MathF.FusedMultiplyAdd(packedA[k * Mr + r], packedB[(long)js * kc * Nr + k * Nr + col], acc);
                    cRef[r * ldc + js * Nr + col] = acc;
                }

        for (int i = 0; i < cLen; i++)
            Assert.True(BitConverter.SingleToInt32Bits(cKernel[i]) == BitConverter.SingleToInt32Bits(cRef[i]),
                $"6x16 FP32 panel kernel mismatch at {i}: kernel={cKernel[i]:G9}, ref={cRef[i]:G9} (kc={kc}, nTiles={nTiles}, ldc={ldc})");
    }

    [Theory]
    [InlineData(1, 8)]
    [InlineData(7, 8)]
    [InlineData(64, 8)]
    [InlineData(256, 8)]
    [InlineData(37, 11)]
    [InlineData(6, 9)]
    public unsafe void Fp64_6x8_SysV_MatchesFmaReference(int kc, int ldc)
    {
        // Runs on Linux/macOS x64 (incl. CI) — validates the System V ABI encoding,
        // which can't be executed on Windows. delegate* unmanaged uses the platform
        // default convention (SysV on Linux/macOS x64), matching the emitted kernel.
        if (!IsX64Unix || !Avx2.IsSupported || !Fma.IsSupported) return;

        const int Mr = 6, Nr = 8;
        byte[] code = MachineCodeFmaKernel.EmitFp64_6x8_PackedSysVU4();
        using var mem = ExecutableMemory.TryAllocate(code);
        if (mem is null) return;
        var fn = (delegate* unmanaged<double*, double*, double*, long, long, void>)mem.Pointer;

        var rng = new Random(7 + kc * 13 + ldc);
        var packedA = new double[kc * Mr];
        var packedB = new double[kc * Nr];
        for (int i = 0; i < packedA.Length; i++) packedA[i] = rng.NextDouble() * 2 - 1;
        for (int i = 0; i < packedB.Length; i++) packedB[i] = rng.NextDouble() * 2 - 1;
        int cLen = (Mr - 1) * ldc + Nr;
        var c0 = new double[cLen];
        for (int i = 0; i < cLen; i++) c0[i] = rng.NextDouble() * 2 - 1;
        var cKernel = (double[])c0.Clone();
        var cRef = (double[])c0.Clone();

        fixed (double* pa = packedA, pb = packedB, pc = cKernel) fn(pa, pb, pc, ldc, kc);

        for (int r = 0; r < Mr; r++)
            for (int col = 0; col < Nr; col++)
            {
                double acc = cRef[r * ldc + col];
                for (int kk = 0; kk < kc; kk++)
                    acc = Math.FusedMultiplyAdd(packedA[kk * Mr + r], packedB[kk * Nr + col], acc);
                cRef[r * ldc + col] = acc;
            }
        for (int i = 0; i < cLen; i++)
            Assert.True(BitConverter.DoubleToInt64Bits(cKernel[i]) == BitConverter.DoubleToInt64Bits(cRef[i]),
                $"6x8 SysV mismatch at {i}: kernel={cKernel[i]:G17}, ref={cRef[i]:G17} (kc={kc}, ldc={ldc})");
    }

    [Theory]
    [InlineData(1, 16)]
    [InlineData(7, 16)]
    [InlineData(256, 16)]
    [InlineData(37, 19)]
    public unsafe void Fp32_6x16_SysV_MatchesFmaReference(int kc, int ldc)
    {
        if (!IsX64Unix || !Avx2.IsSupported || !Fma.IsSupported) return;

        const int Mr = 6, Nr = 16;
        byte[] code = MachineCodeFmaKernel.EmitFp32_6x16_PackedSysV();
        using var mem = ExecutableMemory.TryAllocate(code);
        if (mem is null) return;
        var fn = (delegate* unmanaged<float*, float*, float*, long, long, void>)mem.Pointer;

        var rng = new Random(13 + kc * 17 + ldc);
        var packedA = new float[kc * Mr];
        var packedB = new float[kc * Nr];
        for (int i = 0; i < packedA.Length; i++) packedA[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < packedB.Length; i++) packedB[i] = (float)(rng.NextDouble() * 2 - 1);
        int cLen = (Mr - 1) * ldc + Nr;
        var c0 = new float[cLen];
        for (int i = 0; i < cLen; i++) c0[i] = (float)(rng.NextDouble() * 2 - 1);
        var cKernel = (float[])c0.Clone();
        var cRef = (float[])c0.Clone();

        fixed (float* pa = packedA, pb = packedB, pc = cKernel) fn(pa, pb, pc, ldc, kc);

        for (int r = 0; r < Mr; r++)
            for (int col = 0; col < Nr; col++)
            {
                float acc = cRef[r * ldc + col];
                for (int kk = 0; kk < kc; kk++)
                    acc = MathF.FusedMultiplyAdd(packedA[kk * Mr + r], packedB[kk * Nr + col], acc);
                cRef[r * ldc + col] = acc;
            }
        for (int i = 0; i < cLen; i++)
            Assert.True(BitConverter.SingleToInt32Bits(cKernel[i]) == BitConverter.SingleToInt32Bits(cRef[i]),
                $"6x16 FP32 SysV mismatch at {i}: kernel={cKernel[i]:G9}, ref={cRef[i]:G9} (kc={kc}, ldc={ldc})");
    }

    // The exact emitted lengths and prologue bytes were verified instruction-by-
    // instruction with the capstone disassembler (0 bad, full byte coverage, correct
    // EVEX + ABI). These platform-independent invariants catch any later encoder
    // regression even on machines (like the CI runner) without AVX-512.
    [Theory]
    [InlineData("fp64-win", 688, 0x4C)] // Windows: starts `mov r10,[rsp+0x28]` (4C 8B ...)
    [InlineData("fp64-sysv", 484, 0x4D)] // SysV:   starts `mov r10,r8`        (4D 89 C2)
    [InlineData("fp32-win", 688, 0x4C)]
    [InlineData("fp32-sysv", 484, 0x4D)]
    public void Avx512_Encoding_StructuralInvariants(string which, int expectedLen, byte firstByte)
    {
        byte[] code = which switch
        {
            "fp64-win" => MachineCodeFmaKernel.EmitFp64_6x16_Avx512Windows(),
            "fp64-sysv" => MachineCodeFmaKernel.EmitFp64_6x16_Avx512SysV(),
            "fp32-win" => MachineCodeFmaKernel.EmitFp32_6x32_Avx512Windows(),
            _ => MachineCodeFmaKernel.EmitFp32_6x32_Avx512SysV(),
        };
        Assert.Equal(expectedLen, code.Length);
        Assert.Equal(firstByte, code[0]);
        Assert.Equal(0xC3, code[code.Length - 1]);                 // ends with `ret`
        Assert.Equal(0x77, code[code.Length - 2]);                 // `vzeroupper` (C5 F8 77)
        int evex = 0;                                              // EVEX prefixes (0x62) ≈ ZMM ops
        foreach (byte b in code) if (b == 0x62) evex++;
        Assert.True(evex >= 12, $"{which}: expected ≥12 EVEX ops, found {evex}");
    }

    [Theory]
    [InlineData(1)]
    [InlineData(7)]
    [InlineData(64)]
    [InlineData(256)]
    public unsafe void Fp64_6x16_Avx512_MatchesFmaReference(int kc)
    {
        // Runs only on AVX-512 hardware (none in our CI today — verifies on any such
        // box/runner). The encoding is already capstone-verified; this confirms the
        // numerical result bit-exactly.
        if (!Avx512F.IsSupported || !(IsX64Windows || IsX64Unix)) return;

        const int Mr = 6, Nr = 16, ldc = 16;
        byte[] code = IsX64Windows
            ? MachineCodeFmaKernel.EmitFp64_6x16_Avx512Windows()
            : MachineCodeFmaKernel.EmitFp64_6x16_Avx512SysV();
        using var mem = ExecutableMemory.TryAllocate(code);
        if (mem is null) return;
        var fn = (delegate* unmanaged<double*, double*, double*, long, long, void>)mem.Pointer;

        var rng = new Random(101 + kc);
        var packedA = new double[kc * Mr];
        var packedB = new double[kc * Nr];
        for (int i = 0; i < packedA.Length; i++) packedA[i] = rng.NextDouble() * 2 - 1;
        for (int i = 0; i < packedB.Length; i++) packedB[i] = rng.NextDouble() * 2 - 1;
        int cLen = (Mr - 1) * ldc + Nr;
        var cKernel = new double[cLen];
        var cRef = new double[cLen];
        for (int i = 0; i < cLen; i++) { double v = rng.NextDouble() * 2 - 1; cKernel[i] = v; cRef[i] = v; }

        fixed (double* pa = packedA, pb = packedB, pc = cKernel) fn(pa, pb, pc, ldc, kc);
        for (int r = 0; r < Mr; r++)
            for (int col = 0; col < Nr; col++)
            {
                double acc = cRef[r * ldc + col];
                for (int kk = 0; kk < kc; kk++)
                    acc = Math.FusedMultiplyAdd(packedA[kk * Mr + r], packedB[kk * Nr + col], acc);
                cRef[r * ldc + col] = acc;
            }
        for (int i = 0; i < cLen; i++)
            Assert.True(BitConverter.DoubleToInt64Bits(cKernel[i]) == BitConverter.DoubleToInt64Bits(cRef[i]),
                $"AVX-512 6x16 mismatch at {i} (kc={kc})");
    }

    [Theory]
    [InlineData(1)]
    [InlineData(7)]
    [InlineData(256)]
    public unsafe void Fp32_6x32_Avx512_MatchesFmaReference(int kc)
    {
        if (!Avx512F.IsSupported || !(IsX64Windows || IsX64Unix)) return;

        const int Mr = 6, Nr = 32, ldc = 32;
        byte[] code = IsX64Windows
            ? MachineCodeFmaKernel.EmitFp32_6x32_Avx512Windows()
            : MachineCodeFmaKernel.EmitFp32_6x32_Avx512SysV();
        using var mem = ExecutableMemory.TryAllocate(code);
        if (mem is null) return;
        var fn = (delegate* unmanaged<float*, float*, float*, long, long, void>)mem.Pointer;

        var rng = new Random(202 + kc);
        var packedA = new float[kc * Mr];
        var packedB = new float[kc * Nr];
        for (int i = 0; i < packedA.Length; i++) packedA[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < packedB.Length; i++) packedB[i] = (float)(rng.NextDouble() * 2 - 1);
        int cLen = (Mr - 1) * ldc + Nr;
        var cKernel = new float[cLen];
        var cRef = new float[cLen];
        for (int i = 0; i < cLen; i++) { float v = (float)(rng.NextDouble() * 2 - 1); cKernel[i] = v; cRef[i] = v; }

        fixed (float* pa = packedA, pb = packedB, pc = cKernel) fn(pa, pb, pc, ldc, kc);
        for (int r = 0; r < Mr; r++)
            for (int col = 0; col < Nr; col++)
            {
                float acc = cRef[r * ldc + col];
                for (int kk = 0; kk < kc; kk++)
                    acc = MathF.FusedMultiplyAdd(packedA[kk * Mr + r], packedB[kk * Nr + col], acc);
                cRef[r * ldc + col] = acc;
            }
        for (int i = 0; i < cLen; i++)
            Assert.True(BitConverter.SingleToInt32Bits(cKernel[i]) == BitConverter.SingleToInt32Bits(cRef[i]),
                $"AVX-512 6x32 FP32 mismatch at {i} (kc={kc})");
    }

    [Fact]
    public unsafe void Fp64_6x8_MachineCode_Perf()
    {
        if (Environment.GetEnvironmentVariable("AIDOTNET_RUN_JIT_PERF") != "1") return;
        if (!IsX64Windows || !Avx2.IsSupported || !Fma.IsSupported) return;

        const int Mr = 6, Nr = 8, kc = 256, ldc = 8;
        var rng = new Random(42);
        var packedA = new double[kc * Mr];
        var packedB = new double[kc * Nr];
        for (int i = 0; i < packedA.Length; i++) packedA[i] = rng.NextDouble() * 2 - 1;
        for (int i = 0; i < packedB.Length; i++) packedB[i] = rng.NextDouble() * 2 - 1;
        var c = new double[Mr * ldc];

        Measure("6x8 base ", MachineCodeFmaKernel.EmitFp64_6x8_PackedWindows(), packedA, packedB, c, ldc, kc);
        Measure("6x8 unroll4", MachineCodeFmaKernel.EmitFp64_6x8_PackedWindowsU4(), packedA, packedB, c, ldc, kc);
    }

    [Fact]
    public unsafe void Fp32_6x16_MachineCode_Perf()
    {
        if (Environment.GetEnvironmentVariable("AIDOTNET_RUN_JIT_PERF") != "1") return;
        if (!IsX64Windows || !Avx2.IsSupported || !Fma.IsSupported) return;

        const int Mr = 6, Nr = 16, kc = 256, ldc = 16;
        var rng = new Random(42);
        var packedA = new float[kc * Mr];
        var packedB = new float[kc * Nr];
        for (int i = 0; i < packedA.Length; i++) packedA[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < packedB.Length; i++) packedB[i] = (float)(rng.NextDouble() * 2 - 1);
        var c = new float[Mr * ldc];

        using var mem = ExecutableMemory.TryAllocate(MachineCodeFmaKernel.EmitFp32_6x16_PackedWindows());
        if (mem is null) { _output.WriteLine("6x16: no exec mem"); return; }
        var fn = (delegate* unmanaged<float*, float*, float*, long, long, void>)mem.Pointer;
        fixed (float* pa = packedA, pb = packedB, pc = c)
        {
            for (int i = 0; i < 20000; i++) fn(pa, pb, pc, ldc, kc);
            const int iters = 2_000_000;
            double best = double.MaxValue;
            for (int w = 0; w < 7; w++)
            {
                var sw = Stopwatch.StartNew();
                for (int i = 0; i < iters; i++) fn(pa, pb, pc, ldc, kc);
                sw.Stop();
                best = Math.Min(best, sw.Elapsed.TotalSeconds);
            }
            double gflops = 2.0 * Mr * Nr * kc * iters / best / 1e9;
            _output.WriteLine($"machine-code 6x16 FP32: {gflops:F1} GFLOPS (best-of-7) " +
                "(FP64 6x8 ~57; FP32 should be ~2× since 8 lanes/FMA; HW FP32 peak ~128)");
        }
    }

    [Fact]
    public unsafe void Fp32_GebpPanel_vs_PerTile_GemmPath_Perf()
    {
        if (Environment.GetEnvironmentVariable("AIDOTNET_RUN_JIT_PERF") != "1") return;
        if (!IsX64Windows || !Avx2.IsSupported || !Fma.IsSupported) return;
        if (!MachineKernelGemm.IsFp32Available) { _output.WriteLine("FP32 machine kernel unavailable"); return; }

        // Real GEMM path (parallel, BLIS-blocked). The per-N-tile call paid the Windows xmm6–15
        // save/restore (20 movups) on EVERY tile; the GEBP panel pays it once per M-stripe. The
        // benefit is only visible when kernel work dominates B-packing — i.e. large m (kernel/pack
        // ratio ≈ m) — so we sweep square-ish sizes through the actual TryGemmFp32 path.
        // m must be ÷ Mr(6), n must be ÷ Nr(16) (TryGemmFp32 requires a tile-aligned whole shape).
        foreach (var (m, n, k, iters) in new[] { (504, 512, 512, 300), (1020, 1024, 1024, 60), (1536, 1536, 1536, 25) })
        {
            var rng = new Random(42);
            var a = new float[m * k];
            var b = new float[k * n];
            var c = new float[m * n];
            for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
            for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);

            double MeasureGemm(bool panel)
            {
                bool prev = MachineKernelGemm.EnablePanelKernel;
                MachineKernelGemm.EnablePanelKernel = panel;
                try
                {
                    for (int i = 0; i < 5; i++)
                    {
                        Array.Clear(c);
                        MachineKernelGemm.TryGemmFp32(a, k, false, b, n, false, c, n, m, n, k, false, false); // warm
                    }
                    double best = double.MaxValue;
                    for (int w = 0; w < 7; w++)
                    {
                        var sw = Stopwatch.StartNew();
                        for (int i = 0; i < iters; i++)
                        {
                            Array.Clear(c);
                            MachineKernelGemm.TryGemmFp32(a, k, false, b, n, false, c, n, m, n, k, false, false);
                        }
                        sw.Stop();
                        best = Math.Min(best, sw.Elapsed.TotalSeconds);
                    }
                    return 2.0 * m * n * k * iters / best / 1e9;
                }
                finally { MachineKernelGemm.EnablePanelKernel = prev; }
            }

            double perTile = MeasureGemm(panel: false);
            double panelG = MeasureGemm(panel: true);

            // Single-thread per-core efficiency vs the ~115 GF/s raw kernel ceiling: isolates how much
            // of the per-core compute survives packing + memory traffic (vs parallel-scaling loss).
            int prevMax = CpuParallelSettings.MaxDegreeOfParallelism;
            CpuParallelSettings.MaxDegreeOfParallelism = 1;
            double st1;
            try { st1 = MeasureGemm(panel: true); }
            finally { CpuParallelSettings.MaxDegreeOfParallelism = prevMax; }

            // Current production default for large FP32: BlasManaged.Gemm routes ≥250M-work shapes
            // to PackBoth (the machine kernel is gated off there). Measure it to decide the gate flip.
            double packBoth;
            {
                for (int i = 0; i < 5; i++) { Array.Clear(c); AiDotNet.Tensors.Engines.BlasManaged.BlasManaged.Gemm<float>(a, k, false, b, n, false, c, n, m, n, k); }
                double best = double.MaxValue;
                for (int w = 0; w < 7; w++)
                {
                    var sw = Stopwatch.StartNew();
                    for (int i = 0; i < iters; i++) { Array.Clear(c); AiDotNet.Tensors.Engines.BlasManaged.BlasManaged.Gemm<float>(a, k, false, b, n, false, c, n, m, n, k); }
                    sw.Stop();
                    best = Math.Min(best, sw.Elapsed.TotalSeconds);
                }
                packBoth = 2.0 * m * n * k * iters / best / 1e9;
            }

            // Native OpenBLAS reference (shipped libopenblas.dll) — the bar to beat.
            double openblas = 0;
            if (BlasProvider.HasRawSgemm)
            {
                fixed (float* pa = a, pb = b, pc = c)
                {
                    for (int i = 0; i < 5; i++) BlasProvider.SgemmRaw(m, n, k, pa, k, pb, n, pc, n);
                    double best = double.MaxValue;
                    for (int w = 0; w < 7; w++)
                    {
                        var sw = Stopwatch.StartNew();
                        for (int i = 0; i < iters; i++) BlasProvider.SgemmRaw(m, n, k, pa, k, pb, n, pc, n);
                        sw.Stop();
                        best = Math.Min(best, sw.Elapsed.TotalSeconds);
                    }
                    openblas = 2.0 * m * n * k * iters / best / 1e9;
                }
            }

            _output.WriteLine($"GEMM {m}x{n}x{k}: OpenBLAS {openblas:F0}, PackBoth(ours) {packBoth:F0} " +
                $"({(packBoth / openblas - 1) * 100:+0.0;-0.0}% vs OpenBLAS), machine-kernel {panelG:F0} GF/s | " +
                $"per-tile {perTile:F0}, single-thread/core {st1:F0} (raw ceiling ~115)");
        }
    }

    private unsafe void Measure(string name, byte[] code, double[] packedA, double[] packedB, double[] c, int ldc, int kc)
    {
        const int Mr = 6, Nr = 8;
        using var mem = ExecutableMemory.TryAllocate(code);
        if (mem is null) { _output.WriteLine($"{name}: no exec mem"); return; }
        var fn = (delegate* unmanaged<double*, double*, double*, long, long, void>)mem.Pointer;
        fixed (double* pa = packedA, pb = packedB, pc = c)
        {
            for (int i = 0; i < 20000; i++) fn(pa, pb, pc, ldc, kc); // warm
            const int iters = 2_000_000;
            // Min over windows: noise/contention only ADDS time, so the minimum
            // window is the most representative peak throughput (this box is noisy).
            double best = double.MaxValue;
            for (int w = 0; w < 7; w++)
            {
                var sw = Stopwatch.StartNew();
                for (int i = 0; i < iters; i++) fn(pa, pb, pc, ldc, kc);
                sw.Stop();
                best = Math.Min(best, sw.Elapsed.TotalSeconds);
            }
            double gflops = 2.0 * Mr * Nr * kc * iters / best / 1e9;
            _output.WriteLine($"machine-code {name}: {gflops:F1} GFLOPS (best-of-7) " +
                "(hand-written ~38; managed FMA ceiling 44; OpenBLAS ~60; HW ~64)");
        }
    }
}
