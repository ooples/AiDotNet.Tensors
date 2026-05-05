using System;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Simd;
using AiDotNet.Tensors.Interfaces;

namespace AiDotNet.Tensors.Helpers;

internal static class MatrixMultiplyHelper
{
    // BLAS threshold: work = m * k * n must exceed this to use MKL/BLAS.
    // MKL SGEMM overhead is ~2-3us on modern CPUs. Any matmul with work > 4096
    // (e.g., 16x16x16) takes >3us naively, so BLAS is worthwhile.
    // PyTorch uses BLAS for ALL matmuls regardless of size.
    private const long DefaultBlasWorkThreshold = 4096;
    private const long DefaultBlockedWorkThreshold = 64L * 64L * 64L;
    private const long DefaultParallelThreshold = 16384;
    private static readonly int? BlockSizeOverride = ReadEnvInt("AIDOTNET_MATMUL_BLOCK_SIZE");
    private static readonly long? BlasWorkThresholdOverride = ReadEnvLong("AIDOTNET_MATMUL_BLAS_THRESHOLD");
    private static readonly long? BlockedWorkThresholdOverride = ReadEnvLong("AIDOTNET_MATMUL_BLOCKED_THRESHOLD");
    private static readonly long? ParallelThresholdOverride = ReadEnvLong("AIDOTNET_MATMUL_PARALLEL_THRESHOLD");
    private static readonly bool TraceEnabled = ReadEnvBool("AIDOTNET_MATMUL_TRACE");

    internal static bool TryGemm<T>(ReadOnlyMemory<T> a, int aOffset, ReadOnlyMemory<T> b, int bOffset, Memory<T> c, int cOffset, int m, int k, int n)
    {
        if (TraceEnabled)
        {
            Console.WriteLine($"[MATMUL-TRACE] TryGemm<{typeof(T).Name}>: m={m}, k={k}, n={n}");
        }

        if (!(typeof(T) == typeof(float) || typeof(T) == typeof(double)))
        {
            if (TraceEnabled) Console.WriteLine("[MATMUL-TRACE] Type check failed - not float/double");
            return false;
        }

        long work = (long)m * k * n;
        long threshold = GetBlasWorkThreshold();
        if (TraceEnabled)
        {
            Console.WriteLine($"[MATMUL-TRACE] work={work:N0}, threshold={threshold:N0}");
        }
        if (work < threshold)
        {
            if (TraceEnabled) Console.WriteLine("[MATMUL-TRACE] Below threshold - using non-BLAS path");
            return false;
        }

        long aLength = (long)m * k;
        long bLength = (long)k * n;
        long cLength = (long)m * n;
        if (aLength <= 0 || bLength <= 0 || cLength <= 0 ||
            aLength > int.MaxValue || bLength > int.MaxValue || cLength > int.MaxValue)
        {
            return false;
        }

        if (aOffset < 0 || bOffset < 0 || cOffset < 0 ||
            a.Length < aOffset + aLength ||
            b.Length < bOffset + bLength ||
            c.Length < cOffset + cLength)
        {
            return false;
        }

        bool copyBack = false;

        T[] aArray;
        int aStart;
        if (TryGetArraySegment(a, out var aSegment, out var aBaseOffset))
        {
            aArray = aSegment;
            aStart = aBaseOffset + aOffset;
        }
        else
        {
            aArray = a.Span.Slice(aOffset, (int)aLength).ToArray();
            aStart = 0;
        }

        T[] bArray;
        int bStart;
        if (TryGetArraySegment(b, out var bSegment, out var bBaseOffset))
        {
            bArray = bSegment;
            bStart = bBaseOffset + bOffset;
        }
        else
        {
            bArray = b.Span.Slice(bOffset, (int)bLength).ToArray();
            bStart = 0;
        }

        T[] cArray;
        int cStart;
        if (TryGetArraySegment(c, out var cSegment, out var cBaseOffset))
        {
            cArray = cSegment;
            cStart = cBaseOffset + cOffset;
        }
        else
        {
            cArray = new T[(int)cLength];
            cStart = 0;
            copyBack = true;
        }

        bool used = TryGemmFromArray(aArray, aStart, bArray, bStart, cArray, cStart, m, k, n);
        if (TraceEnabled)
        {
            Console.WriteLine($"[MATMUL-TRACE] TryGemmFromArray returned: {used}");
        }
        if (used && copyBack)
        {
            cArray.AsSpan(0, (int)cLength).CopyTo(c.Span.Slice(cOffset, (int)cLength));
        }

        return used;
    }

    internal static bool ShouldUseBlocked<T>(int m, int k, int n)
    {
        if (!(typeof(T) == typeof(float) || typeof(T) == typeof(double)))
        {
            return false;
        }

        long work = (long)m * k * n;
        return work >= GetBlockedWorkThreshold();
    }

    internal static void TraceMatmul(string path, int m, int n, int k)
    {
        if (!TraceEnabled)
        {
            return;
        }

        Console.WriteLine($"[MATMUL-TRACE CPU] {m}x{n}x{k} {path}");
    }

    internal static void MultiplyBlocked<T>(
        INumericOperations<T> numOps,
        ReadOnlyMemory<T> a,
        ReadOnlyMemory<T> b,
        Memory<T> c,
        int m,
        int k,
        int n,
        int aStride,
        int bStride,
        int cStride,
        int aOffset = 0,
        int bOffset = 0,
        int cOffset = 0,
        bool allowParallel = true)
    {
        int block = GetBlockSize<T>();
        int numRowBlocks = (m + block - 1) / block;
        int numColBlocks = (n + block - 1) / block;
        bool parallel = allowParallel &&
            (long)m * n >= GetParallelThreshold() &&
            Environment.ProcessorCount > 1;

        // Single-block worker — processes one (iiBlock, jjBlock) tile over
        // the full K axis. Writes to the disjoint C[iStart..iEnd, jStart..jEnd]
        // slice so parallel invocations across the (M, N) grid are race-free.
        Action<int, int> multiplyTile = (iiBlock, jjBlock) =>
        {
            int iStart = iiBlock * block;
            int iEnd = Math.Min(iStart + block, m);
            int jStart = jjBlock * block;
            int jEnd = Math.Min(jStart + block, n);
            int nLenTotal = jEnd - jStart;
            var aSpan = a.Span;
            var bSpan = b.Span;
            var cSpan = c.Span;

            for (int kk = 0; kk < k; kk += block)
            {
                int kLen = Math.Min(block, k - kk);
                for (int i = iStart; i < iEnd; i++)
                {
                    int aRowOffset = aOffset + (i * aStride) + kk;
                    int cRowOffset = cOffset + (i * cStride) + jStart;

                    for (int kIndex = 0; kIndex < kLen; kIndex++)
                    {
                        T aik = aSpan[aRowOffset + kIndex];
                        int bRowOffset = bOffset + ((kk + kIndex) * bStride) + jStart;
                        var cBlock = cSpan.Slice(cRowOffset, nLenTotal);
                        var bBlock = bSpan.Slice(bRowOffset, nLenTotal);
                        numOps.MultiplyAdd(cBlock, bBlock, aik, cBlock);
                    }
                }
            }
        };

        if (parallel)
        {
            // 2D parallelism: parallelize over the (iiBlock, jjBlock) grid so
            // transformer shapes with small M (M≤128 → numRowBlocks ≤ 2) still
            // saturate multi-core machines. At M=64 with block=45 on a 16-core
            // box, the old M-axis-only partition spawned 2 tasks; the 2D grid
            // with N=512, block=45 gives 2×12=24 tasks — full utilization.
            int total = numRowBlocks * numColBlocks;
            int procs = Environment.ProcessorCount;
            // Only go 2D when the row partition alone under-subscribes the
            // available cores. DiT-XL-class square shapes with large M are
            // already saturated by M-axis; keep them on the simpler path.
            if (numRowBlocks * 2 < procs && numColBlocks > 1)
            {
                Parallel.For(0, total, blockIdx =>
                {
                    int ii = blockIdx / numColBlocks;
                    int jj = blockIdx % numColBlocks;
                    multiplyTile(ii, jj);
                });
            }
            else
            {
                Parallel.For(0, numRowBlocks, iiBlock =>
                {
                    for (int jjBlock = 0; jjBlock < numColBlocks; jjBlock++)
                        multiplyTile(iiBlock, jjBlock);
                });
            }
            return;
        }

        for (int iiBlock = 0; iiBlock < numRowBlocks; iiBlock++)
        for (int jjBlock = 0; jjBlock < numColBlocks; jjBlock++)
            multiplyTile(iiBlock, jjBlock);
    }

    private static bool TryGetArraySegment<T>(ReadOnlyMemory<T> memory, out T[] array, out int offset)
    {
        if (MemoryMarshal.TryGetArray(memory, out var segment) && segment.Array != null)
        {
            array = segment.Array;
            offset = segment.Offset;
            return true;
        }

        array = Array.Empty<T>();
        offset = 0;
        return false;
    }

    private static bool TryGetArraySegment<T>(Memory<T> memory, out T[] array, out int offset)
        => TryGetArraySegment((ReadOnlyMemory<T>)memory, out array, out offset);

    private static bool TryGemmFromArray<T>(T[] a, int aOffset, T[] b, int bOffset, T[] c, int cOffset, int m, int k, int n)
    {
        if (typeof(T) == typeof(float) && a is float[] af && b is float[] bf && c is float[] cf)
        {
            if (BlasProvider.TryGemm(m, n, k, af, aOffset, k, bf, bOffset, n, cf, cOffset, n))
            {
                return true;
            }

            // BLAS unavailable - fall back to our SIMD GEMM (BLIS-style tiled FMA kernel)
            if (TraceEnabled) Console.WriteLine("[MATMUL-TRACE] BLAS unavailable, using SimdGemm.Sgemm fallback");
            SimdGemm.Sgemm(
                af.AsSpan(aOffset, m * k),
                bf.AsSpan(bOffset, k * n),
                cf.AsSpan(cOffset, m * n),
                m, k, n);
            return true;
        }
        else if (typeof(T) == typeof(double) && a is double[] ad && b is double[] bd && c is double[] cd)
        {
            if (BlasProvider.TryGemm(m, n, k, ad, aOffset, k, bd, bOffset, n, cd, cOffset, n))
            {
                return true;
            }

            // BLAS unavailable — route double through SimdGemm.Dgemm (issue
            // #243). The scalar fallback in MultiplyBlocked handles the
            // below-threshold path; this gives the mid-to-large shapes an
            // AVX2 FMA kernel without needing OpenBLAS.
            if (TraceEnabled) Console.WriteLine("[MATMUL-TRACE] BLAS unavailable, using SimdGemm.Dgemm fallback");
            SimdGemm.Dgemm(
                ad.AsSpan(aOffset, m * k),
                bd.AsSpan(bOffset, k * n),
                cd.AsSpan(cOffset, m * n),
                m, k, n);
            return true;
        }

        return false;
    }

    private static int GetBlockSize<T>()
    {
        if (BlockSizeOverride.HasValue)
        {
            return Clamp(BlockSizeOverride.Value, 16, 128);
        }

        // #294 Phase 4 wiring: delegate to CacheOptimizer's L1-aware
        // tile picker for the float / double cases — single source of
        // truth for tile sizing across the codebase. The picker uses
        // Unsafe.SizeOf<T> internally so it stays in step with what
        // the caller's actual element size is. Other T (BFloat16 /
        // Half / custom numerics) fall through to the local heuristic
        // because CacheOptimizer.ComputeOptimalTiling has an unmanaged
        // constraint that this method intentionally does not — matmul
        // is generic over a wider T set than the optimization helper.
        if (typeof(T) == typeof(float))
        {
            var (m, _, _) = AiDotNet.Tensors.Engines.Optimization.CacheOptimizer
                .ComputeOptimalTiling<float>(128, 128, 128);
            return Clamp(m, 16, 128);
        }
        if (typeof(T) == typeof(double))
        {
            var (m, _, _) = AiDotNet.Tensors.Engines.Optimization.CacheOptimizer
                .ComputeOptimalTiling<double>(128, 128, 128);
            return Clamp(m, 16, 128);
        }

        // Fallback for non-primitive T using the same L1-aware formula
        // CacheOptimizer uses (3-block working set: A_tile + B_tile +
        // C_tile). Element size estimated by typeof check since T may
        // not be unmanaged here.
        int elementSize = typeof(T) == typeof(double) ? 8 : 4;
        int l1 = PlatformDetector.Capabilities?.L1CacheSize ?? 0;
        if (l1 <= 0) l1 = 32 * 1024;
        int block = (int)Math.Sqrt(l1 / (2.0 * elementSize));
        return Clamp(block, 16, 128);
    }

    private static long GetBlasWorkThreshold()
    {
        if (BlasWorkThresholdOverride.HasValue && BlasWorkThresholdOverride.Value > 0)
        {
            return BlasWorkThresholdOverride.Value;
        }

        return DefaultBlasWorkThreshold;
    }

    private static long GetBlockedWorkThreshold()
    {
        if (BlockedWorkThresholdOverride.HasValue && BlockedWorkThresholdOverride.Value > 0)
        {
            return BlockedWorkThresholdOverride.Value;
        }

        return DefaultBlockedWorkThreshold;
    }

    private static long GetParallelThreshold()
    {
        if (ParallelThresholdOverride.HasValue && ParallelThresholdOverride.Value > 0)
        {
            return ParallelThresholdOverride.Value;
        }

        return DefaultParallelThreshold;
    }

    private static int Clamp(int value, int min, int max)
    {
        if (value < min)
        {
            return min;
        }

        return value > max ? max : value;
    }

    private static bool ReadEnvBool(string name)
    {
        var raw = Environment.GetEnvironmentVariable(name);
        if (string.IsNullOrWhiteSpace(raw))
        {
            return false;
        }

        return string.Equals(raw, "1", StringComparison.OrdinalIgnoreCase) ||
               string.Equals(raw, "true", StringComparison.OrdinalIgnoreCase) ||
               string.Equals(raw, "yes", StringComparison.OrdinalIgnoreCase);
    }

    private static int? ReadEnvInt(string name)
    {
        var raw = Environment.GetEnvironmentVariable(name);
        if (string.IsNullOrWhiteSpace(raw))
        {
            return null;
        }

        return int.TryParse(raw, out var value) && value > 0 ? value : null;
    }

    private static long? ReadEnvLong(string name)
    {
        var raw = Environment.GetEnvironmentVariable(name);
        if (string.IsNullOrWhiteSpace(raw))
        {
            return null;
        }

        return long.TryParse(raw, out var value) && value > 0 ? value : null;
    }
}
