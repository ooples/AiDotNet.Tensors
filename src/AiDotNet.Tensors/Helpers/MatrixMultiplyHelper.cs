#pragma warning disable CS0618 // SimdGemm.Sgemm/Dgemm (no-trans shims) are [Obsolete] — pending migration to BlasManaged.Gemm<T> in a later K task.
using System;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Optimization;
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

        // #378: FP16 (System.Half) and BF16 route to their SIMD GEMM microkernels
        // (HalfKernels / BFloat16Kernels — float-accumulated upcast emulation) on net5.0+;
        // net471 has no Vector256 intrinsics so they stay on the generic blocked path.
        // Half and BFloat16 are always taken here, on every target and at every size: the generic blocked
        // fallback would accumulate in the element type, where a Half sum stops growing at 2048.
        bool lowPrecision = IsLowPrecision<T>();
        bool isSupportedType = typeof(T) == typeof(float) || typeof(T) == typeof(double) || lowPrecision;
        if (!isSupportedType)
        {
            if (TraceEnabled) Console.WriteLine("[MATMUL-TRACE] Type check failed - not a SIMD-GEMM type");
            return false;
        }

        long work = (long)m * k * n;
        long threshold = GetBlasWorkThreshold();
        if (TraceEnabled)
        {
            Console.WriteLine($"[MATMUL-TRACE] work={work:N0}, threshold={threshold:N0}");
        }
        if (work < threshold && !lowPrecision)
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
        // Half/BFloat16: accumulate in float32 (MultiplyAdd in T would round every partial sum).
        if (IsLowPrecision<T>())
        {
            MultiplyBlockedLowPrecision(a, b, c, m, k, n, aStride, bStride, cStride, aOffset, bOffset, cOffset);
            return;
        }

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
                CpuParallelSettings.ParallelForOrSerial(0, total, (long)m * n * k, blockIdx =>
                {
                    int ii = blockIdx / numColBlocks;
                    int jj = blockIdx % numColBlocks;
                    multiplyTile(ii, jj);
                });
            }
            else
            {
                CpuParallelSettings.ParallelForOrSerial(0, numRowBlocks, (long)m * n * k, iiBlock =>
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
            BlasProvider.Gemm(m, n, k, af, aOffset, k, bf, bOffset, n, cf, cOffset, n);
            return true;
        }
        else if (typeof(T) == typeof(double) && a is double[] ad && b is double[] bd && c is double[] cd)
        {
            BlasProvider.Gemm(m, n, k, ad, aOffset, k, bd, bOffset, n, cd, cOffset, n);
            return true;
        }
#if NET5_0_OR_GREATER
        // #378: FP16 — SIMD upcast-emulation microkernel, float accumulator.
        else if (typeof(T) == typeof(Half) && a is Half[] ah && b is Half[] bh && c is Half[] ch)
        {
            Engines.Simd.HalfKernels.Matmul(
                ah.AsSpan(aOffset, m * k), k,
                bh.AsSpan(bOffset, k * n), n,
                ch.AsSpan(cOffset, m * n), n,
                m, k, n);
            return true;
        }
        // #378: BF16 — wire the (previously orphan) BF16 microkernel into the GEMM dispatch.
        else if (typeof(T) == typeof(NumericOperations.BFloat16)
                 && a is NumericOperations.BFloat16[] abf
                 && b is NumericOperations.BFloat16[] bbf
                 && c is NumericOperations.BFloat16[] cbf)
        {
            Engines.Simd.BFloat16Kernels.Matmul(
                abf.AsSpan(aOffset, m * k), k,
                bbf.AsSpan(bOffset, k * n), n,
                cbf.AsSpan(cOffset, m * n), n,
                m, k, n);
            return true;
        }
#else
        else if (IsLowPrecision<T>())
        {
            // No Vector256 on this target: widen, multiply in float32, round each output once.
            LowPrecisionGemm(a, aOffset, k, b, bOffset, n, c, cOffset, n, m, k, n, accumulate: false);
            return true;
        }
#endif

        return false;
    }

    /// <summary>True for the 16-bit floating-point element types whose matmul accumulates in float32.</summary>
    internal static bool IsLowPrecision<T>()
        => typeof(T) == typeof(Half) || typeof(T) == typeof(AiDotNet.Tensors.NumericOperations.BFloat16);

    /// <summary>
    /// <c>C = A·B</c> (or <c>C += A·B</c> when <paramref name="accumulate"/>) for Half or BFloat16 row-major operands
    /// with the given row strides. A and B are widened to float32 once, multiplied by the float32 GEMM, and each output
    /// (plus its previous value when accumulating) is rounded to the element type once, as PyTorch's half matmul does
    /// with float32 accumulation (cuBLAS <c>CUBLAS_COMPUTE_32F</c>, the CPU <c>opmath_type</c> GEMM).
    /// </summary>
    internal static void LowPrecisionGemm<T>(
        T[] a, int aOffset, int aStride,
        T[] b, int bOffset, int bStride,
        T[] c, int cOffset, int cStride,
        int m, int k, int n, bool accumulate)
    {
        if (m <= 0 || n <= 0) return;
        var cf = new float[m * n];
        if (k > 0)
        {
            var af = new float[m * k];
            for (int i = 0; i < m; i++) CpuEngine.WidenInto(a, aOffset + i * aStride, af, i * k, k);
            var bf = new float[k * n];
            for (int p = 0; p < k; p++) CpuEngine.WidenInto(b, bOffset + p * bStride, bf, p * n, n);
            Engines.BlasManaged.BlasManaged.Gemm<float>(af, k, false, bf, n, false, cf, n, m, n, k);
        }
        var previous = accumulate ? new float[n] : null;
        for (int i = 0; i < m; i++)
        {
            if (previous is not null)
            {
                CpuEngine.WidenInto(c, cOffset + i * cStride, previous, 0, n);
                for (int j = 0; j < n; j++) cf[i * n + j] += previous[j];
            }
            CpuEngine.NarrowInto(cf, i * n, c, cOffset + i * cStride, n);
        }
    }

    // MultiplyBlocked's low-precision route: unwrap the memories to arrays (copying only when one is not array-backed).
    private static void MultiplyBlockedLowPrecision<T>(
        ReadOnlyMemory<T> a, ReadOnlyMemory<T> b, Memory<T> c,
        int m, int k, int n, int aStride, int bStride, int cStride, int aOffset, int bOffset, int cOffset)
    {
        T[] aArray = TryGetArraySegment(a, out var aSeg, out var aBase) ? aSeg : a.ToArray();
        int aStart = ReferenceEquals(aArray, aSeg) ? aBase + aOffset : aOffset;
        T[] bArray = TryGetArraySegment(b, out var bSeg, out var bBase) ? bSeg : b.ToArray();
        int bStart = ReferenceEquals(bArray, bSeg) ? bBase + bOffset : bOffset;
        if (TryGetArraySegment(c, out var cSeg, out var cBase))
        {
            LowPrecisionGemm(aArray, aStart, aStride, bArray, bStart, bStride, cSeg, cBase + cOffset, cStride, m, k, n, accumulate: true);
            return;
        }
        var cCopy = c.ToArray();
        LowPrecisionGemm(aArray, aStart, aStride, bArray, bStart, bStride, cCopy, cOffset, cStride, m, k, n, accumulate: true);
        cCopy.AsSpan().CopyTo(c.Span);
    }

    private static int GetBlockSize<T>()
    {
        if (BlockSizeOverride.HasValue)
        {
            return Clamp(BlockSizeOverride.Value, 16, 128);
        }

        // #294 Phase 4 wiring: delegate to CacheOptimizer's L1-aware
        // tile picker — single source of truth for tile sizing across
        // the codebase. CacheOptimizer.ComputeOptimalTiling<T> uses
        // Unsafe.SizeOf<T> internally so it returns the right tile
        // for any element size: float (4B), double (8B), Half (2B),
        // BFloat16 (2B), int (4B), long (8B). Routing each typeof(T)
        // case explicitly ensures the per-T fast path is the SAME
        // formula the rest of the codebase uses; without this the
        // Half/BFloat16 path would fall back to the old 4-byte
        // heuristic and pick tiles half as large as L1 actually
        // permits, undermining the single-source-of-truth refactor.
        if (typeof(T) == typeof(float))
            return Clamp(CacheOptimizer.ComputeOptimalTiling<float>(128, 128, 128).tileM, 16, 128);
        if (typeof(T) == typeof(double))
            return Clamp(CacheOptimizer.ComputeOptimalTiling<double>(128, 128, 128).tileM, 16, 128);
        if (typeof(T) == typeof(Half))
            return Clamp(CacheOptimizer.ComputeOptimalTiling<Half>(128, 128, 128).tileM, 16, 128);
        if (typeof(T) == typeof(AiDotNet.Tensors.NumericOperations.BFloat16))
            return Clamp(CacheOptimizer.ComputeOptimalTiling<AiDotNet.Tensors.NumericOperations.BFloat16>(128, 128, 128).tileM, 16, 128);
        if (typeof(T) == typeof(int))
            return Clamp(CacheOptimizer.ComputeOptimalTiling<int>(128, 128, 128).tileM, 16, 128);
        if (typeof(T) == typeof(long))
            return Clamp(CacheOptimizer.ComputeOptimalTiling<long>(128, 128, 128).tileM, 16, 128);

        // Last-resort fallback for non-primitive T (Complex,
        // Multivector, custom numerics) where Unsafe.SizeOf would
        // need the unmanaged constraint that this method's signature
        // doesn't provide. Use Marshal.SizeOf as a runtime-resolvable
        // size — at least gives the right tile for blittable structs;
        // for non-blittable types the runtime throws and we'd need
        // an explicit case above. Same L1-aware formula as
        // CacheOptimizer (3-block working set sqrt).
        int elementSize;
        try { elementSize = System.Runtime.InteropServices.Marshal.SizeOf<T>(); }
        catch { elementSize = 4; } // non-blittable: safe default for matmul tile heuristic
        int l1 = PlatformDetector.Capabilities?.L1CacheSize ?? 0;
        if (l1 <= 0) l1 = 32 * 1024;
        int block = (int)Math.Sqrt(l1 / (3.0 * Math.Max(1, elementSize)));
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
