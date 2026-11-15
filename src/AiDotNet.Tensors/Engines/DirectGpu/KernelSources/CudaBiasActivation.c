// data[r, c] = act(data[r, c] + bias[c]) in place, over the flat rows * cols range (grid-stride). act 0: none (bias_add);
// act 1: the `gelu` kernel's GELU (tanh form as x * sigmoid(2 * inner)), so this equals bias_add followed by gelu.
// vec4 = 1 (cols % 4 == 0, 16-byte aligned data): each thread moves a float4.
__device__ __forceinline__ float ba_gelu(float x)
{
    const float sqrt2OverPi = 0.7978845608f;
    const float coeff = 0.044715f;
    float inner = sqrt2OverPi * (x + coeff * x * x * x);
    float e = expf(-fabsf(2.0f * inner));
    return x * (inner >= 0.0f ? 1.0f / (1.0f + e) : e / (1.0f + e));
}

extern "C" __global__ __launch_bounds__(256) void bias_act_inplace(float* __restrict__ data,
    const float* __restrict__ bias, int rows, int cols, int vec4, int act)
{
    long long total = (long long)rows * cols;
    long long stride = (long long)gridDim.x * blockDim.x;
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (vec4 && total < 0x7fffffffLL)
    {
        // 32-bit indices: a 64-bit modulo per float4 cost more than the bias add itself.
        int total4 = (int)(total >> 2), cols4 = cols >> 2, step = gridDim.x * blockDim.x;
        float4* d4 = reinterpret_cast<float4*>(data);
        const float4* b4 = reinterpret_cast<const float4*>(bias);
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total4; i += step)
        {
            float4 v = d4[i];
            float4 b = b4[i % cols4];
            v.x += b.x; v.y += b.y; v.z += b.z; v.w += b.w;
            if (act == 1) { v.x = ba_gelu(v.x); v.y = ba_gelu(v.y); v.z = ba_gelu(v.z); v.w = ba_gelu(v.w); }
            d4[i] = v;
        }
        return;
    }
    for (long long i = t; i < total; i += stride)
        { float v = data[i] + bias[i % cols]; data[i] = act == 1 ? ba_gelu(v) : v; }
}
