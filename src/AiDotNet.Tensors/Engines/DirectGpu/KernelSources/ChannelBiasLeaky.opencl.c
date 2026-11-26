// The bias add and the leaky slope keep subnormals (PackedFloatLibrary.OpenCl, prepended by the program): an OpenCL
// device may flush them with no build option to keep them, and the CPU does not.
__kernel void bx_channel_bias_leaky(__global const float* input, __global const float* bias,
    __global float* output, int batch, int channels, int spatial, float alpha) {
    int i = (int)get_global_id(0);
    if (i >= batch * channels * spatial) return;
    float rounded = pk_add_keep(input[i], bias[(i / spatial) % channels]);
    output[i] = pk_positive(rounded) ? rounded : pk_mul_keep(alpha, rounded);
}
