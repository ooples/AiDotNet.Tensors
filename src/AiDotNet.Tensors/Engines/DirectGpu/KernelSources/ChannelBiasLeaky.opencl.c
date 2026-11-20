inline float bx_scale_tiny_sum_operand(uint bits) {
    uint magnitude = bits & 0x7fffffffu;
    if (magnitude >= 0x00800000u) return as_float(bits + (24u << 23));
    float scaled = convert_float(magnitude) * 0x1p-125f;
    return as_float(as_uint(scaled) | (bits & 0x80000000u));
}
inline float bx_bias_add_preserve_denorm(float x, float bias) {
#ifdef __FAST_RELAXED_MATH__
    uint a = as_uint(x), b = as_uint(bias);
    uint am = a & 0x7fffffffu, bm = b & 0x7fffffffu;
    if ((am > 0 && am < 0x00800000u) || (bm > 0 && bm < 0x00800000u)) {
        // Above exponent 25 even the predecessor of a power of two is too far
        // away for a subnormal addend to change the correctly rounded result.
        if ((am >> 23) > 25u) return x;
        if ((bm >> 23) > 25u) return bias;
        // Scaling by 2^24 is exact and makes every nonzero operand and sum normal.
        float sum = bx_scale_tiny_sum_operand(a) + bx_scale_tiny_sum_operand(b);
        uint bits = as_uint(sum);
        if (((bits >> 23) & 255u) > 24u) return as_float(bits - (24u << 23));
        uint significand = convert_uint_rte(fabs(sum) * 0x1p125f);
        return as_float((bits & 0x80000000u) | significand);
    }
#endif
    return x + bias;
}
__kernel void bx_channel_bias_leaky(__global const float* input, __global const float* bias,
    __global float* output, int batch, int channels, int spatial, float alpha) {
    int i = (int)get_global_id(0);
    if (i >= batch * channels * spatial) return;
    float rounded = bx_bias_add_preserve_denorm(input[i], bias[(i / spatial) % channels]);
    output[i] = rounded > 0.0f ? rounded : alpha * rounded;
}
