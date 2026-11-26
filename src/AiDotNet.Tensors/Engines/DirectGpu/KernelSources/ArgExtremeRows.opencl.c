// argmax_axis / argmin_axis on wide rows (CUDA's arg_extreme_axis_rows): one 256-item work group per row instead of one
// work item per row, which scanned a 49,152-wide vocabulary row serially. Bit-exact with the scan: a leading NaN wins;
// otherwise the first index of the extreme over the non-NaN values by exact bit key, so ties go to the smaller index.
// Each item keeps the first extreme of its strided slice; a merge takes the strictly better key, else the smaller
// index, which is associative, so the tree gives the scan's answer. The early returns are uniform across the group.
__kernel __attribute__((reqd_work_group_size(256, 1, 1))) void arg_extreme_axis_rows(
    __global const float* input,
    __global float* indices,
    const int outerSize,
    const int reduceSize,
    const int findMin)
{
    __local int sKey[256];
    __local int sIdx[256];
    const int outer = get_group_id(0);
    const int tid = get_local_id(0);
    if (outer >= outerSize) return;
    if (reduceSize <= 0) { if (tid == 0) indices[outer] = 0.0f; return; }
    __global const uint* row = (__global const uint*)input + (size_t)outer * reduceSize;
    if (xf_isnan(row[0])) { if (tid == 0) indices[outer] = 0.0f; return; }

    // The max of key' = findMin ? -key : key. |xf_key| <= 0x7F800000, so the negation cannot overflow.
    int have = 0, bestKey = 0, bestIdx = 0;
    for (int i = tid; i < reduceSize; i += 256) {
        uint v = row[i];
        if (xf_isnan(v)) continue;
        int key = findMin ? -xf_key(v) : xf_key(v);
        if (!have || key > bestKey) { have = 1; bestKey = key; bestIdx = i; }
    }
    sKey[tid] = have ? bestKey : (int)0x80000000;   // INT_MIN: below every finite or infinite key
    sIdx[tid] = have ? bestIdx : 0x7FFFFFFF;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            int k2 = sKey[tid + stride], i2 = sIdx[tid + stride];
            if (k2 > sKey[tid] || (k2 == sKey[tid] && i2 < sIdx[tid])) { sKey[tid] = k2; sIdx[tid] = i2; }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (tid == 0) indices[outer] = (float)sIdx[0];
}
