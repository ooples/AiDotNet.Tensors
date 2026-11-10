using System;
using System.Numerics;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tensors.Engines.Simd;

internal static partial class SimdGemm
{
    /// <summary>The most rows of A the skinny A·Bᵀ kernel takes. Measured against a [51865, 384] B: 1 row 3.7 ms (BLAS 51), 16 rows 13 ms (BLAS 42), 64 rows 65 ms (BLAS 39).</summary>
    internal const int SkinnyTransBMaxM = 16;

    /// <summary>The most B rows per parallel work item in the skinny A·Bᵀ kernel.</summary>
    private const int SkinnyTransBChunk = 256;

    /// <summary>The fewest B rows per parallel work item: below this the per-item dispatch outweighs the dots.</summary>
    private const int SkinnyTransBMinChunk = 32;

    /// <summary>
    /// Whether C += A·Bᵀ should take <see cref="SgemmSkinnyTransB"/>: a few rows of A against a B stored [n, k].
    /// </summary>
    /// <remarks>
    /// Both operands are then contiguous along k, so every output is one dot product and B is streamed exactly once.
    /// The general dispatcher instead materialized a transposed copy of all of B on every call - for fewer than
    /// <see cref="Mr"/> rows with a scalar double loop - which made a language model's tied output head
    /// (one hidden row against a [51865, 384] embedding table) 13x slower than the same product with B pre-transposed.
    /// </remarks>
    internal static bool QualifiesForSkinnyTransB(bool transA, bool transB, int m, int k, int n)
        => !transA && transB && m > 0 && m <= SkinnyTransBMaxM && k >= Vector<float>.Count && n > 0;

    /// <summary>
    /// C[i, j] += dot(A[i, :], B[j, :]) for A [m, k] (row stride <paramref name="lda"/>) and B stored [n, k] (row stride
    /// <paramref name="ldb"/>); C is [m, n] with row stride n.
    /// </summary>
    internal static unsafe void SgemmSkinnyTransB(
        ReadOnlySpan<float> a, int lda, ReadOnlySpan<float> b, int ldb, Span<float> c, int m, int k, int n,
        bool allowParallel)
    {
        fixed (float* pa = a, pb = b, pc = c)
        {
            var aPointer = (IntPtr)pa;
            var bPointer = (IntPtr)pb;
            var cPointer = (IntPtr)pc;
            // Enough work items for every core: a fixed 256-row chunk left [16, 1024]·[1024, 1024]ᵀ four items and ran
            // it 4x slower than the packed BLAS. Each output is still one dot product, so the result does not depend
            // on the split.
            int perCore = (n + (2 * Environment.ProcessorCount) - 1) / (2 * Environment.ProcessorCount);
            int rowsPerChunk = Math.Max(SkinnyTransBMinChunk, Math.Min(SkinnyTransBChunk, perCore));
            int chunks = (n + rowsPerChunk - 1) / rowsPerChunk;

            void Chunk(int chunk)
            {
                var aBase = (float*)aPointer;
                var bBase = (float*)bPointer;
                var cBase = (float*)cPointer;
                int end = Math.Min(n, (chunk + 1) * rowsPerChunk);
                for (int j = chunk * rowsPerChunk; j < end; j++)
                {
                    float* bRow = bBase + ((long)j * ldb);
                    for (int i = 0; i < m; i++)
                        cBase[((long)i * n) + j] += Dot(aBase + ((long)i * lda), bRow, k);
                }
            }

            long work = (long)m * k * n;
            if (allowParallel && work >= ParallelWorkThreshold && chunks > 1)
                CpuParallelSettings.ParallelForOrSerial(0, chunks, work, Chunk);
            else
                for (int chunk = 0; chunk < chunks; chunk++) Chunk(chunk);
        }
    }

    private static unsafe float Dot(float* x, float* y, int length)
    {
        int width = Vector<float>.Count;
        var sum0 = Vector<float>.Zero;
        var sum1 = Vector<float>.Zero;
        int p = 0;
        for (; p + (2 * width) <= length; p += 2 * width)
        {
            sum0 += *(Vector<float>*)(x + p) * *(Vector<float>*)(y + p);
            sum1 += *(Vector<float>*)(x + p + width) * *(Vector<float>*)(y + p + width);
        }

        for (; p + width <= length; p += width)
            sum0 += *(Vector<float>*)(x + p) * *(Vector<float>*)(y + p);

        float total = Vector.Dot(sum0 + sum1, Vector<float>.One);
        for (; p < length; p++) total += x[p] * y[p];
        return total;
    }
}
