using System;
using System.Buffers;
using System.Runtime.InteropServices;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tensors.Engines.BlasManaged;

internal static partial class PackBothStrategy
{
    // Convolution workers already own disjoint output panels. Re-entering the joined
    // dispatcher for each inner GEMM created participant objects and closures even
    // though both dispatches ran serially. Keep their exact packing and K-panel order.
    private static unsafe void RunParallelSingleRegionUnsafe<T>(
        T* a, int aLen, int lda, T* b, int bLen, int ldb, T* c, int cLen, int ldc,
        int m, int n, int k, int mc, int kc, int mr, int nr,
        in BlasOptions<T> options, int elemSize) where T : unmanaged
    {
        int dop = options.NumThreads > 0 ? options.NumThreads : Environment.ProcessorCount;
        if (options.NumThreads < 0) dop = 1;
        if (dop == 1 || CpuParallelSettings.MaxDegreeOfParallelism == 1 || CpuParallelSettings.IsInParallelRegion)
        {
            RunSingleRegionSerialUnsafe(a, aLen, lda, b, bLen, ldb, c, cLen, ldc,
                m, n, k, mc, kc, mr, nr, elemSize, options.ConvPacking);
            return;
        }
        RunParallelSingleRegionJoinedUnsafe(a, aLen, lda, b, bLen, ldb, c, cLen, ldc,
            m, n, k, mc, kc, mr, nr, in options, elemSize);
    }

    internal static unsafe void RunSingleRegionSerialUnsafe<T>(
        T* a, int aLen, int lda, T* b, int bLen, int ldb, T* c, int cLen, int ldc,
        int m, int n, int k, int mc, int kc, int mr, int nr, int elemSize,
        ConvGemmPacking<T>.Worker? packing = null) where T : unmanaged
    {
        int panels = (k + kc - 1) / kc;
        int stripes = (n + nr - 1) / nr;
        int packedNc = stripes * nr;
        // The last slab stores its actual K rows, immediately after the preceding full slabs.
        int bBytes = checked(k * packedNc * elemSize + 31);
        var workerB = packing?.TryUseB(bBytes);
        byte[] packBArray = workerB ?? CpuStorageAllocationDiagnostics.RentWorkspace<byte>(bBytes, "workspace.blas_single_region_packB");
        try
        {
            int alignedOffset;
            fixed (byte* address = packBArray)
                alignedOffset = (int)((32 - ((nuint)address & 31)) & 31);
            for (int panel = 0; panel < panels; panel++)
            {
                int pc = panel * kc, effKc = Math.Min(kc, k - pc);
                int panelOffset = panel * kc * packedNc;
                var source = new ReadOnlySpan<T>(b + pc * ldb, bLen - pc * ldb);
                var destination = MemoryMarshal.Cast<byte, T>(packBArray.AsSpan(
                    alignedOffset + panelOffset * elemSize, effKc * packedNc * elemSize));
                Avx2Pack.PackB(source, ldb, false, destination, n, effKc, nr);
            }
            int aBytes = checked(Math.Min(mc, m) * Math.Min(kc, k) * elemSize);
            int firstRow = 0;
            var filter = packing?.Filter.ForRows(a, aLen, lda, m, k, mc, mr, out firstRow);
            using var aLease = filter is null
                ? new PackBothParticipantWorkspace.SerialLease(aBytes, "workspace.blas_private_packA") : default;
            for (int ic = 0; ic < m; ic += mc)
            {
                int effMc = Math.Min(mc, m - ic);
                if (effMc < mr) continue;
                for (int panel = 0; panel < panels; panel++)
                {
                    int pc = panel * kc, effKc = Math.Min(kc, k - pc);
                    int panelOffset = panel * kc * packedNc;
                    Span<T> packA;
                    int stripeStride;
                    if (filter is null)
                    {
                        packA = MemoryMarshal.Cast<byte, T>(aLease.Span.Slice(0, effMc * effKc * elemSize));
                        int sourceOffset = ic * lda + pc;
                        Avx2Pack.PackA(new ReadOnlySpan<T>(a + sourceOffset, aLen - sourceOffset), lda, false, packA, effMc, effKc, mr);
                        stripeStride = effKc;
                    }
                    else
                    {
                        packA = filter.AsSpan((firstRow + ic) * k + pc * mr);
                        stripeStride = k;
                    }
                    var packB = MemoryMarshal.Cast<byte, T>(packBArray.AsSpan(
                        alignedOffset + panelOffset * elemSize, effKc * packedNc * elemSize));
                    for (int jr = 0; jr < n; jr += nr)
                    {
                        int effNr = Math.Min(nr, n - jr);
                        int bOffset = (jr / nr) * effKc * nr;
                        for (int ir = 0; ir + mr <= effMc; ir += mr)
                        {
                            int cOffset = (ic + ir) * ldc + jr;
                            DispatchMicrokernelWithTail(packA.Slice(ir * stripeStride, effKc * mr), packB.Slice(bOffset, effKc * nr),
                                new Span<T>(c + cOffset, cLen - cOffset), ldc, effKc, mr, nr, effNr);
                        }
                    }
                }
            }
        }
        finally
        {
            if (workerB is null) ArrayPool<byte>.Shared.Return(packBArray);
            else packing?.ReleaseB();
        }
    }
}
