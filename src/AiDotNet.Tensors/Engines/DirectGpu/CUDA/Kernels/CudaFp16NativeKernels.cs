namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels;

/// <summary>
/// Self-contained FP16 kernels (#558 layer 7) that do NOT depend on &lt;cuda_fp16.h&gt;. The toolkit
/// header is absent on driver-only machines (the bundled nvrtc 12.0 doesn't ship it), which silently
/// disabled the entire existing FP16 path (CudaFp16Kernels: "could not open source file cuda_fp16.h").
/// These kernels convert half↔float with the standard IEEE-754 bit algorithm in-kernel, so they compile
/// with the driver alone — confirming the FP16-native approach works on this hardware. (The __half2
/// packed path in CudaFp16Kernels remains the faster variant for toolkit machines.)
/// </summary>
public static class CudaFp16NativeKernels
{
    public static string GetSource()
    {
        return @"
// ── IEEE-754 half <-> float, no cuda_fp16.h ──────────────────────────────
__device__ __forceinline__ float h2f(unsigned short h)
{
    unsigned int sign = ((unsigned int)(h & 0x8000u)) << 16;
    unsigned int exp  = (h >> 10) & 0x1Fu;
    unsigned int mant = h & 0x3FFu;
    unsigned int f;
    if (exp == 0u) {
        if (mant == 0u) { f = sign; }
        else {
            int e = 1;
            while ((mant & 0x400u) == 0u) { mant <<= 1; e--; }
            mant &= 0x3FFu;
            f = sign | ((unsigned int)(e + 112) << 23) | (mant << 13);
        }
    } else if (exp == 0x1Fu) {
        f = sign | 0x7F800000u | (mant << 13);
    } else {
        f = sign | ((exp + 112u) << 23) | (mant << 13);
    }
    return __int_as_float((int)f);
}

__device__ __forceinline__ unsigned short f2h(float x)
{
    unsigned int f = (unsigned int)__float_as_int(x);
    unsigned int sign = (f >> 16) & 0x8000u;
    int exp = (int)((f >> 23) & 0xFFu) - 127 + 15;
    unsigned int mant = f & 0x7FFFFFu;
    if (((f >> 23) & 0xFFu) == 0xFFu) {                 // inf / nan
        return (unsigned short)(sign | 0x7C00u | (mant ? 0x200u : 0u));
    }
    if (exp >= 0x1F) { return (unsigned short)(sign | 0x7C00u); }   // overflow -> inf
    if (exp <= 0) {                                     // subnormal / underflow
        if (exp < -10) return (unsigned short)sign;
        mant |= 0x800000u;
        int shift = 14 - exp;
        unsigned int h = mant >> shift;
        if ((mant >> (shift - 1)) & 1u) h++;           // round to nearest
        return (unsigned short)(sign | h);
    }
    unsigned int h = ((unsigned int)exp << 10) | (mant >> 13);
    if ((mant >> 12) & 1u) h++;                         // round to nearest
    return (unsigned short)(sign | h);
}

extern ""C"" __global__ __launch_bounds__(256) void convert_fp32_to_fp16_native(
    const float* __restrict__ input, unsigned short* __restrict__ output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    output[idx] = f2h(input[idx]);
}

extern ""C"" __global__ __launch_bounds__(256) void convert_fp16_to_fp32_native(
    const unsigned short* __restrict__ input, float* __restrict__ output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    output[idx] = h2f(input[idx]);
}

// FP16-NATIVE GELU: reads FP16 directly, computes in FP32 in-register, writes FP16. No FP32 buffer is
// materialized — half the bandwidth + half-resident activation, the property the GPU memory/util win needs.
extern ""C"" __global__ __launch_bounds__(256) void fp16_gelu_native(
    const unsigned short* __restrict__ input, unsigned short* __restrict__ output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    float x = h2f(input[idx]);
    float x3 = x * x * x;
    float inner = 0.7978845608f * (x + 0.044715f * x3);
    float r = 0.5f * x * (1.0f + tanhf(inner));
    output[idx] = f2h(r);
}

// FP16-NATIVE ReLU: reads FP16 directly, max(x,0) in FP32, writes FP16.
extern ""C"" __global__ __launch_bounds__(256) void fp16_relu_native(
    const unsigned short* __restrict__ input, unsigned short* __restrict__ output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    float x = h2f(input[idx]);
    output[idx] = f2h(x > 0.0f ? x : 0.0f);
}

// FP16-NATIVE residual add: out = a + b (same shape), reads FP16 directly, FP32 accumulate, writes FP16.
extern ""C"" __global__ __launch_bounds__(256) void fp16_add_native(
    const unsigned short* __restrict__ a, const unsigned short* __restrict__ b,
    unsigned short* __restrict__ output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    output[idx] = f2h(h2f(a[idx]) + h2f(b[idx]));
}

// FP16-NATIVE row softmax over the last axis: ONE BLOCK PER ROW. Reads FP16 directly, the max/sum reductions
// + exp run in FP32 (numerically stable: subtract row max), writes FP16. Shared mem = blockDim.x floats.
extern ""C"" __global__ void fp16_softmax_native(
    const unsigned short* __restrict__ input, unsigned short* __restrict__ output, int rows, int cols)
{
    int row = blockIdx.x;
    if (row >= rows) return;
    const unsigned short* in = input + (long long)row * cols;
    unsigned short* out = output + (long long)row * cols;
    extern __shared__ float sdata[];
    int tid = threadIdx.x; int bs = blockDim.x;
    // row max
    float m = -3.4e38f;
    for (int i = tid; i < cols; i += bs) { float v = h2f(in[i]); if (v > m) m = v; }
    sdata[tid] = m; __syncthreads();
    for (int s = bs >> 1; s > 0; s >>= 1) { if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]); __syncthreads(); }
    float rowmax = sdata[0]; __syncthreads();
    // sum exp
    float sum = 0.0f;
    for (int i = tid; i < cols; i += bs) sum += expf(h2f(in[i]) - rowmax);
    sdata[tid] = sum; __syncthreads();
    for (int s = bs >> 1; s > 0; s >>= 1) { if (tid < s) sdata[tid] += sdata[tid + s]; __syncthreads(); }
    float inv = 1.0f / sdata[0]; __syncthreads();
    for (int i = tid; i < cols; i += bs) out[i] = f2h(expf(h2f(in[i]) - rowmax) * inv);
}

// FP16-NATIVE row layernorm over the last axis with FP16 gamma/beta: ONE BLOCK PER ROW. Reads FP16 directly,
// mean/var reductions in FP32, writes FP16 output; also writes the per-row FP32 mean + variance (for the
// backward). Population variance (÷cols), eps inside the rsqrt — matches the engine's LayerNorm convention.
extern ""C"" __global__ void fp16_layernorm_native(
    const unsigned short* __restrict__ input, const unsigned short* __restrict__ gamma,
    const unsigned short* __restrict__ beta, unsigned short* __restrict__ output,
    float* __restrict__ meanOut, float* __restrict__ varOut, int rows, int cols, float eps)
{
    int row = blockIdx.x;
    if (row >= rows) return;
    const unsigned short* in = input + (long long)row * cols;
    unsigned short* out = output + (long long)row * cols;
    extern __shared__ float sdata[];
    int tid = threadIdx.x; int bs = blockDim.x;
    // mean
    float s = 0.0f;
    for (int i = tid; i < cols; i += bs) s += h2f(in[i]);
    sdata[tid] = s; __syncthreads();
    for (int st = bs >> 1; st > 0; st >>= 1) { if (tid < st) sdata[tid] += sdata[tid + st]; __syncthreads(); }
    float mean = sdata[0] / (float)cols; __syncthreads();
    // variance
    float v = 0.0f;
    for (int i = tid; i < cols; i += bs) { float d = h2f(in[i]) - mean; v += d * d; }
    sdata[tid] = v; __syncthreads();
    for (int st = bs >> 1; st > 0; st >>= 1) { if (tid < st) sdata[tid] += sdata[tid + st]; __syncthreads(); }
    float var = sdata[0] / (float)cols; __syncthreads();
    float invstd = rsqrtf(var + eps);
    if (tid == 0) {                                     // meanOut / varOut are optional + independent
        if (meanOut) meanOut[row] = mean;
        if (varOut) varOut[row] = var;
    }
    for (int i = tid; i < cols; i += bs) {
        float norm = (h2f(in[i]) - mean) * invstd;
        out[i] = f2h(norm * h2f(gamma[i]) + h2f(beta[i]));
    }
}

// ── FP16-NATIVE BACKWARD kernels (read FP16 grad + FP16 saved activation, FP32 math, write FP16 grad) ──
// Each mirrors the EXACT formula of its FP32 counterpart so the Half backward is parity-equivalent to the
// FP32 backward — only the storage dtype differs (the converts that scaled with data size are eliminated).

// GELU backward: gradIn = gradOut * gelu'(x). Mirrors gelu_backward (tanh approximation).
extern ""C"" __global__ __launch_bounds__(256) void fp16_gelu_backward_native(
    const unsigned short* __restrict__ gradOutput, const unsigned short* __restrict__ input,
    unsigned short* __restrict__ gradInput, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    const float sqrt2OverPi = 0.7978845608f;
    const float coeff = 0.044715f;
    float x = h2f(input[idx]);
    float x2 = x * x;
    float x3 = x2 * x;
    float inner = sqrt2OverPi * (x + coeff * x3);
    float t = tanhf(inner);
    float sech2 = 1.0f - t * t;
    float dInner = sqrt2OverPi * (1.0f + 3.0f * coeff * x2);
    float dgelu = 0.5f * (1.0f + t) + 0.5f * x * sech2 * dInner;
    gradInput[idx] = f2h(h2f(gradOutput[idx]) * dgelu);
}

// ReLU backward: gradIn = (x>0) ? gradOut : 0. Mirrors relu_backward (copies the Half grad bit-exactly when x>0).
extern ""C"" __global__ __launch_bounds__(256) void fp16_relu_backward_native(
    const unsigned short* __restrict__ gradOutput, const unsigned short* __restrict__ input,
    unsigned short* __restrict__ gradInput, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    gradInput[idx] = (h2f(input[idx]) > 0.0f) ? gradOutput[idx] : (unsigned short)0;
}

// Softmax backward: ONE THREAD PER ROW (mirrors softmax_backward). gradIn = y * (gradOut - sum_f gradOut*y).
extern ""C"" __global__ __launch_bounds__(256) void fp16_softmax_backward_native(
    const unsigned short* __restrict__ gradOutput, const unsigned short* __restrict__ output,
    unsigned short* __restrict__ gradInput, int batchSize, int features)
{
    int batch = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch >= batchSize) return;
    long long baseIdx = (long long)batch * features;
    float dot = 0.0f;
    for (int f = 0; f < features; f++)
        dot += h2f(gradOutput[baseIdx + f]) * h2f(output[baseIdx + f]);
    for (int f = 0; f < features; f++) {
        long long idx = baseIdx + f;
        gradInput[idx] = f2h(h2f(output[idx]) * (h2f(gradOutput[idx]) - dot));
    }
}

// LayerNorm backward — gradInput. ONE BLOCK PER ROW, reductions in FP32 (mirrors layernorm_backward). Takes
// the FP32 saveMean + saveInvVar (1/sqrt(var+eps)) exactly like the FP32 engine path; grad/input/gamma/gradInput
// are FP16. Shared mem = 2*blockDim.x floats (sumDy, sumDyXmu).
extern ""C"" __global__ void fp16_layernorm_backward_native(
    const unsigned short* __restrict__ gradOutput, const unsigned short* __restrict__ input,
    const unsigned short* __restrict__ gamma, const float* __restrict__ saveMean, const float* __restrict__ saveInvVar,
    unsigned short* __restrict__ gradInput, int rows, int cols)
{
    int b = blockIdx.x;
    if (b >= rows) return;
    extern __shared__ float smem[];
    float* smem2 = smem + blockDim.x;
    int tid = threadIdx.x; int bs = blockDim.x;
    long long base = (long long)b * cols;
    float mean = saveMean[b];
    float invVar = saveInvVar[b];
    float locSumDy = 0.0f, locSumDyXmu = 0.0f;
    for (int i = tid; i < cols; i += bs) {
        float dy = h2f(gradOutput[base + i]) * h2f(gamma[i]);
        locSumDy += dy;
        locSumDyXmu += dy * (h2f(input[base + i]) - mean);
    }
    smem[tid] = locSumDy; smem2[tid] = locSumDyXmu; __syncthreads();
    for (int s = bs >> 1; s > 0; s >>= 1) { if (tid < s) { smem[tid] += smem[tid + s]; smem2[tid] += smem2[tid + s]; } __syncthreads(); }
    float sumDy = smem[0], sumDyXmu = smem2[0]; __syncthreads();
    for (int i = tid; i < cols; i += bs) {
        float xmu = h2f(input[base + i]) - mean;
        float dxhat = h2f(gradOutput[base + i]) * h2f(gamma[i]);
        float gi = invVar * (dxhat - (sumDy + xmu * invVar * invVar * sumDyXmu) / (float)cols);
        gradInput[base + i] = f2h(gi);
    }
}

// LayerNorm backward — gamma/beta param grads. ONE THREAD PER FEATURE, serial over rows, FP32 accumulate, Half
// out (mirrors layernorm_grad_params). gradGamma += gradOut*normalized; gradBeta += gradOut.
extern ""C"" __global__ __launch_bounds__(256) void fp16_layernorm_grad_params_native(
    const unsigned short* __restrict__ gradOutput, const unsigned short* __restrict__ input,
    const float* __restrict__ saveMean, const float* __restrict__ saveInvVar,
    unsigned short* __restrict__ gradGamma, unsigned short* __restrict__ gradBeta, int rows, int cols)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= cols) return;
    float dGamma = 0.0f, dBeta = 0.0f;
    for (int b = 0; b < rows; b++) {
        long long idx = (long long)b * cols + i;
        float normalized = (h2f(input[idx]) - saveMean[b]) * saveInvVar[b];
        float go = h2f(gradOutput[idx]);
        dGamma += go * normalized;
        dBeta += go;
    }
    gradGamma[i] = f2h(dGamma);
    gradBeta[i] = f2h(dBeta);
}
";
    }

    public static string[] GetKernelNames() => new[]
    {
        "convert_fp32_to_fp16_native",
        "convert_fp16_to_fp32_native",
        "fp16_gelu_native",
        "fp16_relu_native",
        "fp16_add_native",
        "fp16_softmax_native",
        "fp16_layernorm_native",
        "fp16_gelu_backward_native",
        "fp16_relu_backward_native",
        "fp16_softmax_backward_native",
        "fp16_layernorm_backward_native",
        "fp16_layernorm_grad_params_native"
    };
}
