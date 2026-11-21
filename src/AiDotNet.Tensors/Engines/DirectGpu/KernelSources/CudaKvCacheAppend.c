// kv_cache_append: IKvCacheAppendBackend. ONE block: every thread reads the position, writes its share of the row and
// of the mask, and after the block barrier thread 0 increments the position (no thread reads it after the barrier).
extern "C" __global__ __launch_bounds__(1024) void kv_cache_append(
    float* __restrict__ cache, float* __restrict__ position, const float* __restrict__ value, float* __restrict__ mask,
    int batch, int capacity, int width)
{
    int p = (int)position[0];
    bool full = p >= capacity;
    for (int s = threadIdx.x; s < capacity; s += blockDim.x)
        mask[s] = (full || s <= p) ? 0.0f : __int_as_float(0xff800000);
    if (!full)
        for (int i = threadIdx.x; i < batch * width; i += blockDim.x)
        {
            int b = i / width, j = i - b * width;
            cache[((long long)b * capacity + p) * width + j] = value[i];
        }
    __syncthreads();
    if (threadIdx.x == 0 && !full) position[0] = (float)(p + 1);
}
