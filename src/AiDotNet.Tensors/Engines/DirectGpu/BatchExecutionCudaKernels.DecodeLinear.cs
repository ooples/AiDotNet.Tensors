namespace AiDotNet.Tensors.Engines.DirectGpu;

internal static partial class BatchExecutionCudaKernels
{
    public const string DlLinearGelu = "dl_p_linear_gelu", DlLinearGeluFp16 = "dl_p_linear_gelu_fp16",
        DlW8A8LinearGelu = "dl_p_w8a8_linear_gelu", DlW8A8Epilogue = "dl_p_w8a8_epilogue";

    // #974: the decode-linear routes (IFusedDecodeLinearBackend) for every backend without CUDA's direct-PTX and
    // cuBLAS paths. One thread per output element and no work-group coupling, so the source translates mechanically
    // to OpenCL, Metal and GLSL like the other shared families.
    //
    // Layouts are the routes' documented ones: weights are canonical output-major [N, K] (row n holds output n's K
    // weights), activations row-major [M, K], output [M, N]. GELU is the tanh form with CUDA's constants.
    //
    // The FP16 and quantized kernels read their operands through int words (little-endian: half i in word i / 2,
    // int8 i in word i / 4) because that is the one pointer type every translation carries; the caller guarantees
    // the buffers hold whole words. DL_HALF widens one half to FP32 on its bit pattern, exactly (zero, subnormals,
    // infinity and NaN included), so the FP16 route has FP16 operands and FP32 accumulation like CUDA's. The int8
    // products accumulate in int32 exactly: |x * w| <= 128 * 128, so the sum cannot overflow below K = 131,071, and
    // the launch code rejects anything larger.
    internal const string DecodeLinearSource = @"
#define DL_HALF(V, W, I) { int dl_b = (W[(I) >> 1] >> (((I) & 1) * 16)) & 65535; int dl_e = (dl_b >> 10) & 31; int dl_m = dl_b & 1023; if (dl_e == 0) { float dl_f = dl_m; V = dl_f * 5.9604644775390625e-8f; } else { V = __int_as_float(((dl_e == 31 ? 255 : dl_e + 112) << 23) | (dl_m << 13)); } if ((dl_b & 32768) != 0) { V = -V; } }
extern ""C"" __global__ __launch_bounds__(256) void dl_p_linear_gelu(const float* x, const float* w, const float* bias, float* y, int M, int K, int N) {
    int t = bx_gid(); if (t >= M * N) return;
    int m = t / N; int n = t - m * N;
    float s = 0.0f;
    for (int k = 0; k < K; k++) s += x[m * K + k] * w[n * K + k];
    s += bias[n];
    y[t] = 0.5f * s * (1.0f + tanhf(0.7978845608028654f * (s + 0.044715f * s * s * s)));
}

extern ""C"" __global__ __launch_bounds__(256) void dl_p_linear_gelu_fp16(const int* xh, const int* wh, const float* bias, float* y, int M, int K, int N) {
    int t = bx_gid(); if (t >= M * N) return;
    int m = t / N; int n = t - m * N;
    float s = 0.0f;
    for (int k = 0; k < K; k++) {
        float xv; float wv;
        DL_HALF(xv, xh, m * K + k)
        DL_HALF(wv, wh, n * K + k)
        s += xv * wv;
    }
    s += bias[n];
    y[t] = 0.5f * s * (1.0f + tanhf(0.7978845608028654f * (s + 0.044715f * s * s * s)));
}

extern ""C"" __global__ __launch_bounds__(256) void dl_p_w8a8_linear_gelu(const int* x8, const int* w8, const float* actScale, const float* wScales, const float* bias, float* y, int K, int N) {
    int n = bx_gid(); if (n >= N) return;
    int acc = 0;
    for (int k = 0; k < K; k++) {
        int xi = (x8[k >> 2] >> ((k & 3) * 8)) & 255; if (xi > 127) xi -= 256;
        int j = n * K + k;
        int wi = (w8[j >> 2] >> ((j & 3) * 8)) & 255; if (wi > 127) wi -= 256;
        acc += xi * wi;
    }
    float af = acc;
    float s = af * actScale[0] * wScales[n] + bias[n];
    y[n] = 0.5f * s * (1.0f + tanhf(0.7978845608028654f * (s + 0.044715f * s * s * s)));
}

extern ""C"" __global__ __launch_bounds__(256) void dl_p_w8a8_epilogue(const int* acc, const float* actScale, const float* wScales, const float* bias, float* y, int N) {
    int n = bx_gid(); if (n >= N) return;
    float af = acc[n];
    float s = af * actScale[0] * wScales[n] + bias[n];
    y[n] = 0.5f * s * (1.0f + tanhf(0.7978845608028654f * (s + 0.044715f * s * s * s)));
}
";
}
