layout(set = 0, binding = 0) readonly buffer A { float a[]; };
layout(set = 0, binding = 1) writeonly buffer B { float b[]; };
layout(push_constant) uniform Params { uint outerSize; uint reduceSize; uint findMax; };
shared int sKey[256];
shared uint sIdx[256];
void main() {
    uint outer = gl_WorkGroupID.x;
    uint tid = gl_LocalInvocationID.x;
    bool live = outer < outerSize;
    uint offset = outer * reduceSize;
    // The max of key' = findMax ? key : -key. |xf_key| <= 0x7F800000, so the negation cannot overflow.
    bool have = false;
    int bestKey = 0;
    uint bestIdx = 0u;
    for (uint i = tid; live && i < reduceSize; i += 256u) {
        uint v = floatBitsToUint(a[offset + i]);
        if (xf_isnan(v)) continue;
        int key = findMax != 0u ? xf_key(v) : -xf_key(v);
        if (!have || key > bestKey) { have = true; bestKey = key; bestIdx = i; }
    }
    sKey[tid] = have ? bestKey : int(0x80000000u);   // INT_MIN: below every finite or infinite key
    sIdx[tid] = have ? bestIdx : 0xFFFFFFFFu;
    memoryBarrierShared();
    barrier();
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            int k2 = sKey[tid + stride];
            uint i2 = sIdx[tid + stride];
            if (k2 > sKey[tid] || (k2 == sKey[tid] && i2 < sIdx[tid])) { sKey[tid] = k2; sIdx[tid] = i2; }
        }
        memoryBarrierShared();
        barrier();
    }
    if (tid == 0u && live)
        b[outer] = xf_isnan(floatBitsToUint(a[offset])) ? 0.0 : float(sIdx[0]);
}
