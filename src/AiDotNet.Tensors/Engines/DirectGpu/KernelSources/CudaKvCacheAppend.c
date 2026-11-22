// kv_cache_append: IKvCacheAppendBackend. ONE block: every thread reads the position, writes its share of the row and
// of the mask, and after the block barrier thread 0 increments the position (no thread reads it after the barrier).
// Negative, NaN, infinite or fractional positions on the device are refused without writing out of bounds.
extern "C" __global__ __launch_bounds__(1024) void kv_cache_append(
    float* __restrict__ cache, float* __restrict__ position, const float* __restrict__ value, float* __restrict__ mask,
    int batch, int capacity, int width)
{
    float raw = position[0];
    bool invalid = !(raw >= 0.0f && raw <= 2147483520.0f);
    int p = invalid ? 0 : (int)raw;
    if (!invalid && (float)p != raw) invalid = true;
    bool full = !invalid && p >= capacity;
    bool writable = !invalid && !full;
    for (int s = threadIdx.x; s < capacity; s += blockDim.x)
        mask[s] = (!invalid && (full || s <= p)) ? 0.0f : __int_as_float(0xff800000);
    if (writable)
        for (int i = threadIdx.x; i < batch * width; i += blockDim.x)
        {
            int b = i / width, j = i - b * width;
            cache[((long long)b * capacity + p) * width + j] = value[i];
        }
    __syncthreads();
    if (threadIdx.x == 0 && writable) position[0] = (float)(p + 1);
}
