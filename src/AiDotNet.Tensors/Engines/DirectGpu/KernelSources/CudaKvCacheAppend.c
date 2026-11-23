// kv_cache_append: IKvCacheAppendBackend. ONE block: every thread reads the position, writes its share of the row and
// of the mask, and after the block barrier thread 0 increments the position (no thread reads it after the barrier).
// A position that is not a non-negative integer below capacity (NaN, an infinity, a negative or fractional value, or a
// full cache) writes nothing to the cache or the position, poisons the mask with NaN and records the first such fault
// in `fault` (page-locked host memory the host reads after the stream synchronizes): fault[0] = 1 for a full cache,
// 2 for an invalid position; fault[1] = the position's bits; fault[2] = capacity.
extern "C" __global__ __launch_bounds__(1024) void kv_cache_append(
    float* __restrict__ cache, float* __restrict__ position, const float* __restrict__ value, float* __restrict__ mask,
    int* __restrict__ fault, int batch, int capacity, int width)
{
    float pf = position[0];
    bool valid = isfinite(pf) && pf >= 0.0f && pf == floorf(pf);
    bool ok = valid && pf < (float)capacity;   // the host bounds capacity by 2^24, so the comparison is exact
    if (!ok)
    {
        for (int s = threadIdx.x; s < capacity; s += blockDim.x) mask[s] = __int_as_float(0x7fc00000);
        if (threadIdx.x == 0 && fault[0] == 0)
        {
            fault[1] = __float_as_int(pf);
            fault[2] = capacity;
            __threadfence_system();
            fault[0] = valid ? 1 : 2;
        }
        return;
    }
    int p = (int)pf;
    for (int s = threadIdx.x; s < capacity; s += blockDim.x)
        mask[s] = s <= p ? 0.0f : __int_as_float(0xff800000);
    long long n = (long long)batch * width;
    for (long long i = threadIdx.x; i < n; i += blockDim.x)
    {
        long long b = i / width, j = i - b * width;
        cache[(b * capacity + p) * width + j] = value[i];
    }
    __syncthreads();
    if (threadIdx.x == 0) position[0] = (float)(p + 1);
}
