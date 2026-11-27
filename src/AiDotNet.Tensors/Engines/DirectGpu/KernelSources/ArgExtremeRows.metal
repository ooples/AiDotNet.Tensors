// arg_extrema_axis on a row at least ArgExtremeRows.MinWidth wide: one 256-thread threadgroup per row
// (CUDA's arg_extreme_axis_rows), bit-exact with the scan above. Each thread keeps the first extreme of its strided
// slice by exact bit key, and a merge takes the strictly better key, else the smaller index, which is associative, so
// the tree gives the scan's answer; a leading NaN wins. No thread returns before a barrier.
kernel void arg_extreme_axis_rows(
    device const float* input [[buffer(0)]], device float* indices [[buffer(1)]],
    constant uint& outerSize [[buffer(2)]], constant uint& reduceSize [[buffer(3)]], constant uint& maximum [[buffer(4)]],
    uint outer [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]])
{
    threadgroup int sKey[256];
    threadgroup uint sIdx[256];
    bool live = outer < outerSize;
    uint offset = outer * reduceSize;
    bool have = false; int bestKey = 0; uint bestIdx = 0u;
    for (uint i = tid; live && i < reduceSize; i += 256u) {
        uint v = as_type<uint>(input[offset + i]);
        if (xf_isnan(v)) continue;
        int key = maximum != 0u ? xf_key(v) : -xf_key(v);
        if (!have || key > bestKey) { have = true; bestKey = key; bestIdx = i; }
    }
    sKey[tid] = have ? bestKey : int(0x80000000u);
    sIdx[tid] = have ? bestIdx : 0xFFFFFFFFu;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            int k2 = sKey[tid + stride]; uint i2 = sIdx[tid + stride];
            if (k2 > sKey[tid] || (k2 == sKey[tid] && i2 < sIdx[tid])) { sKey[tid] = k2; sIdx[tid] = i2; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u && live) indices[outer] = xf_isnan(as_type<uint>(input[offset])) ? 0.0f : float(sIdx[0]);
}
