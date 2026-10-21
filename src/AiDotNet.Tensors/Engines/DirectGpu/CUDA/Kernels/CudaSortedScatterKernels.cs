// Copyright (c) AiDotNet. All rights reserved.
// Deterministic scatter-add without floating-point atomics (#890 phase 4e): a stable sort of the indices, then each
// destination sums its own contributions in increasing source order.

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels
{
    /// <summary>
    /// The sorted form of a deterministic scatter-add (embedding backward, scatter_add under GpuDeterminism). The
    /// kernels it replaces gave each destination one thread that scanned EVERY index for its own: O(destinations x
    /// sources) -- 429 ms for a 4096-token batch into a 32000-row embedding on a GTX 1660 Ti, on the default
    /// (deterministic) path. Here the indices are stably sorted (LSD radix, 4 bits a pass; counts are integer, so the
    /// order is the same on every run and every device), each destination's run of sources is located by a scan of the
    /// counts, and one thread per output element sums its run in increasing source order -- the same additions in the
    /// same order as the scanning kernel, so the result is bit-identical to it.
    /// </summary>
    internal static class CudaSortedScatterKernels
    {
        /// <summary>Items per sort tile: one per thread of a 256-thread block.</summary>
        internal const int Tile = 256;

        /// <summary>Buckets per radix pass (4 bits).</summary>
        internal const int Buckets = 16;

        public static string GetSource()
        {
            return @"
// keys[i] = indices[i] when it addresses [0, destinations), else destinations (sorted last, summed nowhere);
// values[i] = i.
extern ""C"" __global__ __launch_bounds__(256) void sorted_scatter_init(
    const int* __restrict__ indices, int* __restrict__ keys, int* __restrict__ values, int count, int destinations)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    int k = indices[i];
    keys[i] = (k >= 0 && k < destinations) ? k : destinations;
    values[i] = i;
}

// Per tile, how many of its keys have each 4-bit digit at `shift`: tileCounts[digit * tiles + tile].
extern ""C"" __global__ __launch_bounds__(256) void sorted_scatter_histogram(
    const int* __restrict__ keys, int* __restrict__ tileCounts, int count, int shift, int tiles)
{
    __shared__ int counts[16];
    if (threadIdx.x < 16) counts[threadIdx.x] = 0;
    __syncthreads();
    int i = blockIdx.x * 256 + threadIdx.x;
    if (i < count) atomicAdd(&counts[(keys[i] >> shift) & 15], 1);
    __syncthreads();
    if (threadIdx.x < 16) tileCounts[threadIdx.x * tiles + blockIdx.x] = counts[threadIdx.x];
}

// Exclusive scan of n ints in place, by one block of 1024 threads: each thread owns a contiguous chunk.
extern ""C"" __global__ __launch_bounds__(1024) void sorted_scatter_scan(int* __restrict__ data, int n)
{
    __shared__ int partial[1024];
    int t = threadIdx.x;
    int chunk = (n + 1023) / 1024;
    int begin = min(t * chunk, n), end = min(begin + chunk, n);
    int sum = 0;
    for (int i = begin; i < end; i++) sum += data[i];
    partial[t] = sum;
    __syncthreads();
    for (int offset = 1; offset < 1024; offset <<= 1)
    {
        int add = t >= offset ? partial[t - offset] : 0;
        __syncthreads();
        partial[t] += add;
        __syncthreads();
    }
    int running = partial[t] - sum;
    for (int i = begin; i < end; i++)
    {
        int v = data[i];
        data[i] = running;
        running += v;
    }
}

// Stable scatter of one radix pass: an item's place is its digit's offset for its tile plus the items before it in
// the tile with the same digit (warp ballots, then the earlier warps' counts).
extern ""C"" __global__ __launch_bounds__(256) void sorted_scatter_place(
    const int* __restrict__ keysIn, const int* __restrict__ valuesIn, int* __restrict__ keysOut,
    int* __restrict__ valuesOut, const int* __restrict__ offsets, int count, int shift, int tiles)
{
    __shared__ int warpCounts[8][16];
    int i = blockIdx.x * 256 + threadIdx.x;
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    bool live = i < count;
    int key = live ? keysIn[i] : 0;
    int digit = live ? (key >> shift) & 15 : -1;
    unsigned lower = (1u << lane) - 1u;
    int rank = 0;
    for (int b = 0; b < 16; b++)
    {
        unsigned mask = __ballot_sync(0xffffffffu, digit == b);
        if (digit == b) rank = __popc(mask & lower);
        if (lane == 0) warpCounts[warp][b] = __popc(mask);
    }
    __syncthreads();
    if (!live) return;
    for (int w = 0; w < warp; w++) rank += warpCounts[w][digit];
    int place = offsets[digit * tiles + blockIdx.x] + rank;
    keysOut[place] = key;
    valuesOut[place] = valuesIn[i];
}

// starts[d] = the first sorted position whose key is >= d, for d in [0, destinations]: run d is
// [starts[d], starts[d + 1]).
extern ""C"" __global__ __launch_bounds__(256) void sorted_scatter_starts(
    const int* __restrict__ keys, int* __restrict__ starts, int count, int destinations)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p > count) return;
    int previous = p == 0 ? -1 : keys[p - 1];
    int current = p == count ? destinations + 1 : keys[p];
    for (int d = previous + 1; d <= current && d <= destinations; d++) starts[d] = p;
}

// out[d, f] = (accumulate ? out[d, f] : 0) + the sum of source rows sorted into run d, in increasing source order.
extern ""C"" __global__ __launch_bounds__(256) void sorted_scatter_sum(
    const float* __restrict__ source, const int* __restrict__ order, const int* __restrict__ starts,
    float* __restrict__ output, int destinations, int features, int accumulate)
{
    // The output is one buffer, so it holds fewer than 2^31 elements: 32-bit index arithmetic throughout.
    int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= destinations * features) return;
    int d = cell / features, f = cell - d * features;
    float sum = 0.0f;
    for (int p = starts[d], end = starts[d + 1]; p < end; p++) sum += source[(long long)order[p] * features + f];
    if (accumulate) output[cell] += sum;
    else output[cell] = sum;
}
";
        }

        public static string[] GetKernelNames()
        {
            return new[]
            {
                "sorted_scatter_init",
                "sorted_scatter_histogram",
                "sorted_scatter_scan",
                "sorted_scatter_place",
                "sorted_scatter_starts",
                "sorted_scatter_sum",
            };
        }
    }
}
