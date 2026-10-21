// Copyright (c) AiDotNet. All rights reserved.
// CUDA GPU kernels for gradient-based optimizers.
// Provides element-wise parameter updates for various optimization algorithms.

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels;

/// <summary>
/// CUDA GPU kernels for gradient-based optimizers.
/// These kernels enable GPU-resident training by keeping parameters
/// and optimizer state entirely on the GPU during training.
/// </summary>
internal static class CudaOptimizerKernels
{
    /// <summary>
    /// Gets all optimizer kernel sources.
    /// </summary>
    public static string GetSource()
    {
        return Compilation.DynamicQuantizationCodebook.KernelPrelude(Compilation.DynamicQuantizationCodebook.KernelLanguage.CudaHip) + @"
// ===========================================================================
// CUDA OPTIMIZER KERNELS
// ===========================================================================

// 1 - beta^t for the optimizer bias corrections, without cancellation. These modules compile with --use_fast_math,
// which maps powf to __powf: its ~1e-7 relative error, subtracted from 1, grows by 1/(1 - beta) (1000x for
// beta2 = 0.999 at step 1) and moved an AMSGrad step by 1.7e-4 relative. log1pf and expm1f are not remapped.
__device__ __forceinline__ float bias_correction(float beta, float t)
{
    return -expm1f(t * log1pf(-(1.0f - beta)));
}

__device__ __forceinline__ float bf16_to_float(unsigned short x)
{
    unsigned int bits = ((unsigned int)x) << 16;
    return __uint_as_float(bits);
}

__device__ __forceinline__ unsigned short float_to_bf16_rne(float x)
{
    unsigned int bits = __float_as_uint(x);
    if ((bits & 0x7FFFFFFFu) > 0x7F800000u) {
        return (unsigned short)((bits >> 16) | 0x0040u);
    }
    unsigned int rounding = 0x7FFFu + ((bits >> 16) & 1u);
    return (unsigned short)((bits + rounding) >> 16);
}

// ---------------------------------------------------------------------------
// SGD with momentum update
// ---------------------------------------------------------------------------
extern ""C"" __global__ __launch_bounds__(256) void sgd_momentum_update(
    float* __restrict__ param, const float* __restrict__ gradient, float* __restrict__ velocity,
    float learningRate, float momentum, float weightDecay, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float grad = gradient[idx];
    if (weightDecay > 0.0f) {
        grad += weightDecay * param[idx];
    }

    float v = momentum * velocity[idx] + grad;
    velocity[idx] = v;
    param[idx] -= learningRate * v;
}

// ---------------------------------------------------------------------------
// Vanilla SGD update (no momentum)
// ---------------------------------------------------------------------------
extern ""C"" __global__ __launch_bounds__(256) void sgd_update(
    float* __restrict__ param, const float* __restrict__ gradient,
    float learningRate, float weightDecay, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float grad = gradient[idx];
    if (weightDecay > 0.0f) {
        grad += weightDecay * param[idx];
    }

    param[idx] -= learningRate * grad;
}

// ---------------------------------------------------------------------------
// Adam optimizer update
// ---------------------------------------------------------------------------
extern ""C"" __global__ __launch_bounds__(256) void adam_update(
    float* __restrict__ param, const float* __restrict__ gradient, float* __restrict__ m, float* __restrict__ v,
    float learningRate, float beta1, float beta2, float epsilon,
    float weightDecay, int step, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float grad = gradient[idx];
    if (weightDecay > 0.0f) {
        grad += weightDecay * param[idx];
    }

    float mVal = beta1 * m[idx] + (1.0f - beta1) * grad;
    float vVal = beta2 * v[idx] + (1.0f - beta2) * grad * grad;
    m[idx] = mVal;
    v[idx] = vVal;

    float mHat = mVal / bias_correction(beta1, (float)step);
    float vHat = vVal / bias_correction(beta2, (float)step);

    param[idx] -= learningRate * mHat / (sqrtf(vHat) + epsilon);
}

// ---------------------------------------------------------------------------
// AdamW optimizer update (decoupled weight decay)
// ---------------------------------------------------------------------------
extern ""C"" __global__ __launch_bounds__(256) void adamw_update(
    float* __restrict__ param, const float* __restrict__ gradient, float* __restrict__ m, float* __restrict__ v,
    float learningRate, float beta1, float beta2, float epsilon,
    float weightDecay, int step, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float grad = gradient[idx];

    // Decoupled weight decay
    if (weightDecay > 0.0f) {
        param[idx] *= (1.0f - learningRate * weightDecay);
    }

    float mVal = beta1 * m[idx] + (1.0f - beta1) * grad;
    float vVal = beta2 * v[idx] + (1.0f - beta2) * grad * grad;
    m[idx] = mVal;
    v[idx] = vVal;

    float mHat = mVal / bias_correction(beta1, (float)step);
    float vHat = vVal / bias_correction(beta2, (float)step);

    param[idx] -= learningRate * mHat / (sqrtf(vHat) + epsilon);
}

// ---------------------------------------------------------------------------
// Multi-tensor (apex multi_tensor_apply-style) Adam / AdamW.
// One kernel LAUNCH updates EVERY fp32 Adam/AdamW parameter tensor, instead of
// one launch per tensor. A deep model with hundreds of parameter tensors pays
// hundreds of kernel-launch + arg-marshal + scheduling round-trips per step on
// the per-tensor path; those dominate when the tensors are small. Here the host
// splits the tensor list into fixed CHUNK-element chunks and builds a chunk
// table: each CUDA block handles ONE chunk of ONE tensor. Block b reads its
// tensor index t = chunkTensor[b] and element base s = chunkStart[b], then the
// param/grad/m/v base addresses for tensor t out of the device pointer arrays.
// gridDim.x = total chunks; blockDim.x = CHUNK (256). Math is identical to the
// single-tensor adam_update / adamw_update kernels — this only changes launch
// batching, so results match the per-tensor kernels bit-for-bit.
// Multi-tensor plain SGD (torch.optim.SGD foreach): the tensor list travels in the kernel's parameter block, so a
// launch needs no device-side table, allocation or upload. Up to SGD_MT_MAX tensors per launch; block b works on
// tensor t where blockStart[t] <= b < blockStart[t + 1], SGD_MT_CHUNK elements per block. The arithmetic is
// sgd_update's, so the result is bit-identical to one sgd_update launch per tensor.
#define SGD_MT_MAX 48
#define SGD_MT_CHUNK 1024
struct SgdMultiTensorArgs
{
    unsigned long long param[SGD_MT_MAX];
    unsigned long long grad[SGD_MT_MAX];
    int size[SGD_MT_MAX];
    int blockStart[SGD_MT_MAX + 1];
    int count;
};

// dyn (optional): device float[2] whose [0] replaces learningRate -- a captured step reads the step's rate there.
extern ""C"" __global__ __launch_bounds__(256) void sgd_multi_tensor_update(
    SgdMultiTensorArgs a, float learningRate, float weightDecay, const double* __restrict__ finiteGuard,
    const float* __restrict__ dyn)
{
    if (dyn != 0) learningRate = dyn[0];
    // Guarded launch: the gradients' double sum of squares, written earlier on this stream; not finite = skip the step.
    if (finiteGuard != 0 && !isfinite(*finiteGuard)) return;
    int b = blockIdx.x;
    int t = 0;
    while (t + 1 < a.count && a.blockStart[t + 1] <= b) t++;
    float* __restrict__ param = (float*)a.param[t];
    const float* __restrict__ gradient = (const float*)a.grad[t];
    int size = a.size[t];
    int base = (b - a.blockStart[t]) * SGD_MT_CHUNK + threadIdx.x;
    #pragma unroll
    for (int k = 0; k < SGD_MT_CHUNK / 256; k++)
    {
        int idx = base + k * 256;
        if (idx >= size) return;
        float grad = gradient[idx];
        if (weightDecay > 0.0f) {
            grad += weightDecay * param[idx];
        }
        param[idx] -= learningRate * grad;
    }
}

extern ""C"" __global__ __launch_bounds__(256) void adam_multi_tensor_update(
    const unsigned long long* __restrict__ paramPtrs,
    const unsigned long long* __restrict__ gradPtrs,
    const unsigned long long* __restrict__ mPtrs,
    const unsigned long long* __restrict__ vPtrs,
    const int* __restrict__ sizes,
    const int* __restrict__ chunkTensor,
    const int* __restrict__ chunkStart,
    float learningRate, float beta1, float beta2, float epsilon,
    float weightDecay, int step, int chunkSize,
    const double* __restrict__ finiteGuard,
    const float* __restrict__ dyn)
{
    // A non-finite gradient sum of squares (written on the device before this launch) discards the whole
    // step: no moment or parameter is written, exactly as the host-side check skipped the launch.
    if (finiteGuard != 0 && !isfinite(*finiteGuard)) return;
    // dyn (optional): device float[2] = this step's learning rate and step number, for a captured step.
    if (dyn != 0) { learningRate = dyn[0]; step = (int)dyn[1]; }
    int t = chunkTensor[blockIdx.x];
    int base = chunkStart[blockIdx.x];
    int i = base + threadIdx.x;
    int size = sizes[t];
    if (i >= size) return;

    float* __restrict__ param = (float*)paramPtrs[t];
    const float* __restrict__ gradient = (const float*)gradPtrs[t];
    float* __restrict__ m = (float*)mPtrs[t];
    float* __restrict__ v = (float*)vPtrs[t];

    float grad = gradient[i];
    if (weightDecay > 0.0f) {
        grad += weightDecay * param[i];
    }
    float mVal = beta1 * m[i] + (1.0f - beta1) * grad;
    float vVal = beta2 * v[i] + (1.0f - beta2) * grad * grad;
    m[i] = mVal;
    v[i] = vVal;
    float mHat = mVal / bias_correction(beta1, (float)step);
    float vHat = vVal / bias_correction(beta2, (float)step);
    param[i] -= learningRate * mHat / (sqrtf(vHat) + epsilon);
}

extern ""C"" __global__ __launch_bounds__(256) void adamw_multi_tensor_update(
    const unsigned long long* __restrict__ paramPtrs,
    const unsigned long long* __restrict__ gradPtrs,
    const unsigned long long* __restrict__ mPtrs,
    const unsigned long long* __restrict__ vPtrs,
    const int* __restrict__ sizes,
    const int* __restrict__ chunkTensor,
    const int* __restrict__ chunkStart,
    float learningRate, float beta1, float beta2, float epsilon,
    float weightDecay, int step, int chunkSize,
    const double* __restrict__ finiteGuard,
    const float* __restrict__ dyn)
{
    // A non-finite gradient sum of squares (written on the device before this launch) discards the whole
    // step: no moment or parameter is written, exactly as the host-side check skipped the launch.
    if (finiteGuard != 0 && !isfinite(*finiteGuard)) return;
    // dyn (optional): device float[2] = this step's learning rate and step number, for a captured step.
    if (dyn != 0) { learningRate = dyn[0]; step = (int)dyn[1]; }
    int t = chunkTensor[blockIdx.x];
    int base = chunkStart[blockIdx.x];
    int i = base + threadIdx.x;
    int size = sizes[t];
    if (i >= size) return;

    float* __restrict__ param = (float*)paramPtrs[t];
    const float* __restrict__ gradient = (const float*)gradPtrs[t];
    float* __restrict__ m = (float*)mPtrs[t];
    float* __restrict__ v = (float*)vPtrs[t];

    float grad = gradient[i];
    if (weightDecay > 0.0f) {
        param[i] *= (1.0f - learningRate * weightDecay);
    }
    float mVal = beta1 * m[i] + (1.0f - beta1) * grad;
    float vVal = beta2 * v[i] + (1.0f - beta2) * grad * grad;
    m[i] = mVal;
    v[i] = vVal;
    float mHat = mVal / bias_correction(beta1, (float)step);
    float vHat = vVal / bias_correction(beta2, (float)step);
    param[i] -= learningRate * mHat / (sqrtf(vHat) + epsilon);
}

// ---------------------------------------------------------------------------
// Adam optimizer update with bfloat16 moment storage
// ---------------------------------------------------------------------------
extern ""C"" __global__ __launch_bounds__(256) void adam_bf16_update(
    float* __restrict__ param, const float* __restrict__ gradient, unsigned short* __restrict__ m, unsigned short* __restrict__ v,
    float learningRate, float beta1, float beta2, float epsilon,
    float weightDecay, int step, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float grad = gradient[idx];
    if (weightDecay > 0.0f) {
        grad += weightDecay * param[idx];
    }
    float oldM = step <= 1 ? 0.0f : bf16_to_float(m[idx]);
    float oldV = step <= 1 ? 0.0f : bf16_to_float(v[idx]);
    float mVal = beta1 * oldM + (1.0f - beta1) * grad;
    float vVal = beta2 * oldV + (1.0f - beta2) * grad * grad;
    m[idx] = float_to_bf16_rne(mVal);
    v[idx] = float_to_bf16_rne(vVal);

    float mHat = mVal / bias_correction(beta1, (float)step);
    float vHat = vVal / bias_correction(beta2, (float)step);

    param[idx] -= learningRate * mHat / (sqrtf(vHat) + epsilon);
}

// ---------------------------------------------------------------------------
// AdamW optimizer update with bfloat16 moment storage
// ---------------------------------------------------------------------------
extern ""C"" __global__ __launch_bounds__(256) void adamw_bf16_update(
    float* __restrict__ param, const float* __restrict__ gradient, unsigned short* __restrict__ m, unsigned short* __restrict__ v,
    float learningRate, float beta1, float beta2, float epsilon,
    float weightDecay, int step, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float grad = gradient[idx];
    if (weightDecay > 0.0f) {
        param[idx] *= (1.0f - learningRate * weightDecay);
    }

    float oldM = step <= 1 ? 0.0f : bf16_to_float(m[idx]);
    float oldV = step <= 1 ? 0.0f : bf16_to_float(v[idx]);
    float mVal = beta1 * oldM + (1.0f - beta1) * grad;
    float vVal = beta2 * oldV + (1.0f - beta2) * grad * grad;
    m[idx] = float_to_bf16_rne(mVal);
    v[idx] = float_to_bf16_rne(vVal);

    float mHat = mVal / bias_correction(beta1, (float)step);
    float vHat = vVal / bias_correction(beta2, (float)step);

    param[idx] -= learningRate * mHat / (sqrtf(vHat) + epsilon);
}

// ---------------------------------------------------------------------------
// RMSprop optimizer update
// ---------------------------------------------------------------------------
extern ""C"" __global__ __launch_bounds__(256) void rmsprop_update(
    float* __restrict__ param, const float* __restrict__ gradient, float* __restrict__ squaredAvg,
    float learningRate, float rho, float epsilon, float weightDecay, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float grad = gradient[idx];
    if (weightDecay > 0.0f) {
        grad += weightDecay * param[idx];
    }

    float sqAvg = rho * squaredAvg[idx] + (1.0f - rho) * grad * grad;
    squaredAvg[idx] = sqAvg;

    param[idx] -= learningRate * grad / (sqrtf(sqAvg) + epsilon);
}

// ---------------------------------------------------------------------------
// Adagrad optimizer update
// ---------------------------------------------------------------------------
extern ""C"" __global__ __launch_bounds__(256) void adagrad_update(
    float* __restrict__ param, const float* __restrict__ gradient, float* __restrict__ accumulatedGrad,
    float learningRate, float epsilon, float weightDecay, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float grad = gradient[idx];
    if (weightDecay > 0.0f) {
        grad += weightDecay * param[idx];
    }

    float accum = accumulatedGrad[idx] + grad * grad;
    accumulatedGrad[idx] = accum;

    param[idx] -= learningRate * grad / (sqrtf(accum) + epsilon);
}

// ---------------------------------------------------------------------------
// Nesterov Accelerated Gradient (NAG) optimizer update
// ---------------------------------------------------------------------------
extern ""C"" __global__ __launch_bounds__(256) void nag_update(
    float* __restrict__ param, const float* __restrict__ gradient, float* __restrict__ velocity,
    float learningRate, float momentum, float weightDecay, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float grad = gradient[idx];
    if (weightDecay > 0.0f) {
        grad += weightDecay * param[idx];
    }

    float v = velocity[idx];
    float vNew = momentum * v - learningRate * grad;
    velocity[idx] = vNew;

    param[idx] += momentum * vNew - learningRate * grad;
}

// ---------------------------------------------------------------------------
// LARS optimizer update
// ---------------------------------------------------------------------------
extern ""C"" __global__ __launch_bounds__(256) void lars_update(
    float* __restrict__ param, const float* __restrict__ gradient, float* __restrict__ velocity,
    float learningRate, float momentum, float weightDecay, float trustCoeff, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float grad = gradient[idx];
    float p = param[idx];

    if (weightDecay > 0.0f) {
        grad += weightDecay * p;
    }

    float v = momentum * velocity[idx] + grad;
    velocity[idx] = v;

    param[idx] = p - learningRate * v;
}

// ---------------------------------------------------------------------------
// LAMB optimizer update
// Note: LAMB requires layer-wise trust ratio computation (||w|| / ||update||).
// The trust ratio must be pre-computed externally and passed to this kernel.
// Set trustRatio=1.0f to disable trust ratio scaling (degenerates to AdamW).
// ---------------------------------------------------------------------------
extern ""C"" __global__ __launch_bounds__(256) void lamb_update(
    float* __restrict__ param, const float* __restrict__ gradient, float* __restrict__ m, float* __restrict__ v,
    float learningRate, float beta1, float beta2, float epsilon,
    float weightDecay, float trustRatio, int step, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float grad = gradient[idx];
    float p = param[idx];

    float mVal = beta1 * m[idx] + (1.0f - beta1) * grad;
    float vVal = beta2 * v[idx] + (1.0f - beta2) * grad * grad;
    m[idx] = mVal;
    v[idx] = vVal;

    float mHat = mVal / bias_correction(beta1, (float)step);
    float vHat = vVal / bias_correction(beta2, (float)step);

    float adamUpdate = mHat / (sqrtf(vHat) + epsilon);
    float update = adamUpdate + weightDecay * p;

    // Apply trust ratio scaling (LAMB's layer-wise adaptive learning rate)
    param[idx] = p - learningRate * trustRatio * update;
}

// ---------------------------------------------------------------------------
// AdaDelta optimizer update
// ---------------------------------------------------------------------------
extern ""C"" __global__ __launch_bounds__(256) void adadelta_update(
    float* __restrict__ param, const float* __restrict__ gradient, float* __restrict__ accumGrad, float* __restrict__ accumUpdate,
    float rho, float epsilon, float weightDecay, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float grad = gradient[idx];
    if (weightDecay > 0.0f) {
        grad += weightDecay * param[idx];
    }

    float ag = rho * accumGrad[idx] + (1.0f - rho) * grad * grad;
    accumGrad[idx] = ag;

    float rmsUpdate = sqrtf(accumUpdate[idx] + epsilon);
    float rmsGrad = sqrtf(ag + epsilon);
    float update = (rmsUpdate / rmsGrad) * grad;

    accumUpdate[idx] = rho * accumUpdate[idx] + (1.0f - rho) * update * update;

    param[idx] -= update;
}

// ---------------------------------------------------------------------------
// AMSGrad optimizer update
// ---------------------------------------------------------------------------
extern ""C"" __global__ __launch_bounds__(256) void amsgrad_update(
    float* __restrict__ param, const float* __restrict__ gradient, float* __restrict__ m, float* __restrict__ v, float* __restrict__ vMax,
    float learningRate, float beta1, float beta2, float epsilon,
    float weightDecay, int step, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float grad = gradient[idx];
    if (weightDecay > 0.0f) {
        grad += weightDecay * param[idx];
    }

    float mVal = beta1 * m[idx] + (1.0f - beta1) * grad;
    m[idx] = mVal;

    float vVal = beta2 * v[idx] + (1.0f - beta2) * grad * grad;
    v[idx] = vVal;

    float vMaxVal = fmaxf(vMax[idx], vVal);
    vMax[idx] = vMaxVal;

    float beta2Corr = bias_correction(beta2, (float)step);
    float mHat = mVal / bias_correction(beta1, (float)step);
    float vMaxHat = vMaxVal / beta2Corr;

    param[idx] -= learningRate * mHat / (sqrtf(vMaxHat) + epsilon);
}

// ---------------------------------------------------------------------------
// AdaMax optimizer update
// ---------------------------------------------------------------------------
extern ""C"" __global__ __launch_bounds__(256) void adamax_update(
    float* __restrict__ param, const float* __restrict__ gradient, float* __restrict__ m, float* __restrict__ u,
    float learningRate, float beta1, float beta2, float epsilon,
    float weightDecay, int step, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float grad = gradient[idx];
    if (weightDecay > 0.0f) {
        grad += weightDecay * param[idx];
    }

    float mVal = beta1 * m[idx] + (1.0f - beta1) * grad;
    m[idx] = mVal;

    float uVal = fmaxf(beta2 * u[idx], fabsf(grad));
    u[idx] = uVal;

    float biasCorrection = bias_correction(beta1, (float)step);

    param[idx] -= (learningRate / biasCorrection) * mVal / (uVal + epsilon);
}

// ---------------------------------------------------------------------------
// Lion optimizer update (Evolved Sign Momentum)
// ---------------------------------------------------------------------------
extern ""C"" __global__ __launch_bounds__(256) void lion_update(
    float* __restrict__ param, const float* __restrict__ gradient, float* __restrict__ m,
    float learningRate, float beta1, float beta2, float weightDecay, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float grad = gradient[idx];
    float mVal = m[idx];

    float interp = beta1 * mVal + (1.0f - beta1) * grad;
    float update = (interp > 0.0f) ? 1.0f : ((interp < 0.0f) ? -1.0f : 0.0f);

    m[idx] = beta2 * mVal + (1.0f - beta2) * grad;

    if (weightDecay > 0.0f) {
        update += weightDecay * param[idx];
    }

    param[idx] -= learningRate * update;
}

// ---------------------------------------------------------------------------
// Nadam optimizer update (Nesterov-accelerated Adam)
// ---------------------------------------------------------------------------
extern ""C"" __global__ __launch_bounds__(256) void nadam_update(
    float* __restrict__ param, const float* __restrict__ gradient, float* __restrict__ m, float* __restrict__ v,
    float learningRate, float beta1, float beta2, float epsilon,
    float weightDecay, int step, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float grad = gradient[idx];
    if (weightDecay > 0.0f) {
        grad += weightDecay * param[idx];
    }

    float mVal = beta1 * m[idx] + (1.0f - beta1) * grad;
    m[idx] = mVal;

    float vVal = beta2 * v[idx] + (1.0f - beta2) * grad * grad;
    v[idx] = vVal;

    float beta1Corr = bias_correction(beta1, (float)step);
    float beta1CorrNext = bias_correction(beta1, (float)(step + 1));
    float beta2Corr = bias_correction(beta2, (float)step);
    float vHat = vVal / beta2Corr;

    // Nadam (Dozat 2016, Alg. 2, constant mu = beta1): the momentum look-ahead uses the NEXT step's bias
    // correction (1 - beta1^(t+1)), the current gradient this step's (1 - beta1^t). Matches PyTorch NAdam at
    // constant momentum and FusedOptimizer.NadamUpdateSimd.
    float mNesterov = beta1 * mVal / beta1CorrNext + (1.0f - beta1) * grad / beta1Corr;

    param[idx] -= learningRate * mNesterov / (sqrtf(vHat) + epsilon);
}

// ---------------------------------------------------------------------------
// FTRL optimizer update
// ---------------------------------------------------------------------------
extern ""C"" __global__ __launch_bounds__(256) void ftrl_update(
    float* __restrict__ param, const float* __restrict__ gradient, float* __restrict__ z, float* __restrict__ n,
    float learningRate, float l1Reg, float l2Reg, float beta, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float grad = gradient[idx];
    float nVal = n[idx];
    float zVal = z[idx];
    float pVal = param[idx];

    float nNew = nVal + grad * grad;
    n[idx] = nNew;

    float sigma = (sqrtf(nNew) - sqrtf(nVal)) / learningRate;

    zVal = zVal + grad - sigma * pVal;
    z[idx] = zVal;

    float zSign = (zVal > 0.0f) ? 1.0f : ((zVal < 0.0f) ? -1.0f : 0.0f);
    float zAbs = fabsf(zVal);

    if (zAbs <= l1Reg) {
        param[idx] = 0.0f;
    } else {
        float denom = (beta + sqrtf(nNew)) / learningRate + l2Reg;
        param[idx] = -zSign * (zAbs - l1Reg) / denom;
    }
}

// ---------------------------------------------------------------------------
// Proximal gradient (ISTA) with L1 soft-threshold prox:
//   tmp = param - lr*grad;  param = sign(tmp) * max(|tmp| - lr*l1Strength, 0)
// ---------------------------------------------------------------------------
extern ""C"" __global__ __launch_bounds__(256) void proximal_l1_update(
    float* __restrict__ param, const float* __restrict__ gradient,
    float learningRate, float l1Strength, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float tmp = param[idx] - learningRate * gradient[idx];
    float a = fabsf(tmp) - (learningRate * l1Strength);
    float sgn = (tmp > 0.0f) ? 1.0f : ((tmp < 0.0f) ? -1.0f : 0.0f);
    param[idx] = sgn * (a > 0.0f ? a : 0.0f);
}

// ---------------------------------------------------------------------------
// 8-bit Adam (bitsandbytes-style blockwise dynamic quantization). Matches the
// CPU Adam8BitOptimizer (CompressBothMoments, QuantizationPercentile>=100,
// no stochastic rounding) per block: m and v are codebook INDICES into the signed
// and unsigned dynamic maps (ADAM8_S / ADAM8_U), scaled by the block absmax. One CUDA block per quant
// block; blockDim=256; shared-memory maxAbs reduction recomputes the per-block
// scale before requantizing. Adam math is float (matching the CPU Tensor ops);
// quant/dequant scaling is float on GPU; rint() = round-half-to-even.
// Launch with gridDim.x = numBlocks.
// ---------------------------------------------------------------------------
extern ""C"" __global__ __launch_bounds__(256) void adam8bit_update(
    float* __restrict__ param, const float* __restrict__ gradient,
    unsigned char* __restrict__ mQ, unsigned char* __restrict__ vQ,
    float* __restrict__ mScales, float* __restrict__ vScales,
    float learningRate, float beta1, float beta2, float epsilon,
    float oneMinusBeta1, float oneMinusBeta2, float biasCorrection1, float biasCorrection2,
    int blockSize, int paramLength, int numBlocks)
{
    int blk = blockIdx.x;
    if (blk >= numBlocks) return;
    int start = blk * blockSize;
    int endIdx = start + blockSize; if (endIdx > paramLength) endIdx = paramLength;
    int firstStep = (biasCorrection1 <= oneMinusBeta1 + 1e-7f) && (biasCorrection2 <= oneMinusBeta2 + 1e-7f); /* both corrections must indicate step 1 — bc1 alone is always-true when beta1==0, wrongly discarding v history every step */
    float mScale = firstStep ? 0.0f : mScales[blk];
    float vScale = firstStep ? 0.0f : vScales[blk];

    __shared__ float sMaxM[256];
    __shared__ float sMaxV[256];

    // Phase 1: compute newM/newV, reduce per-block maxAbs for the new scales.
    float locM = 0.0f, locV = 0.0f;
    for (int i = start + threadIdx.x; i < endIdx; i += blockDim.x) {
        float m_i = firstStep ? 0.0f : adam8_dec_s(mQ[i]) * mScale;
        float v_i = firstStep ? 0.0f : adam8_dec_u(vQ[i]) * vScale;
        float g = gradient[i];
        float newM = beta1 * m_i + oneMinusBeta1 * g;
        float newV = beta2 * v_i + oneMinusBeta2 * (g * g);
        locM = fmaxf(locM, fabsf(newM));
        locV = fmaxf(locV, fabsf(newV));
    }
    sMaxM[threadIdx.x] = locM; sMaxV[threadIdx.x] = locV;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sMaxM[threadIdx.x] = fmaxf(sMaxM[threadIdx.x], sMaxM[threadIdx.x + s]);
            sMaxV[threadIdx.x] = fmaxf(sMaxV[threadIdx.x], sMaxV[threadIdx.x + s]);
        }
        __syncthreads();
    }
    // Block-wise dynamic quantization: scale = block absmax, value = nearest codebook entry (linear /127 and
    // /255 rounded small second moments to zero and exploded the update).
    float newMScale = fmax(sMaxM[0], 1e-30f);
    float newVScale = fmax(sMaxV[0], 1e-30f);
    if (threadIdx.x == 0) { mScales[blk] = newMScale; vScales[blk] = newVScale; }
    __syncthreads();

    // Phase 2: recompute, apply bias-corrected param update, requantize.
    for (int i = start + threadIdx.x; i < endIdx; i += blockDim.x) {
        float m_i = firstStep ? 0.0f : adam8_dec_s(mQ[i]) * mScale;
        float v_i = firstStep ? 0.0f : adam8_dec_u(vQ[i]) * vScale;
        float g = gradient[i];
        float newM = beta1 * m_i + oneMinusBeta1 * g;
        float newV = beta2 * v_i + oneMinusBeta2 * (g * g);

        float mHat = newM / biasCorrection1;
        float vHat = newV / biasCorrection2;
        param[i] = param[i] - learningRate * mHat / (sqrtf(vHat) + epsilon);

        mQ[i] = (unsigned char)adam8_enc_s(newM / newMScale);
        vQ[i] = (unsigned char)adam8_enc_u(newV / newVScale);
    }
}

// ===========================================================================
// SPARSE OPTIMIZER KERNELS (PR #567)
// ===========================================================================
// Each kernel below is the sparse counterpart of the dense optimizer kernel
// above. Instead of one thread per parameter element, we launch one thread
// per non-zero gradient (nnz). Each thread reads (idx[k], val[k]), scatter-
// updates ONLY (param[idx], state[idx]); the other (N - nnz) entries are
// never touched — neither read nor written. This is the actual GPU sparse
// fast path: O(nnz) memory traffic instead of O(N).
//
// Grid dim = ceil(nnz / 256); the consumer launches with this dimension and
// passes nnz as the size argument.
// ===========================================================================

__device__ __forceinline__ int decode_sparse_index(float raw, int param_size)
{
    int bitcast = __float_as_int(raw);
    if (bitcast >= 0 && bitcast < param_size) return bitcast;
    // Validate in the FLOAT domain before the (int) cast — (int)raw is undefined for NaN/Inf or
    // out-of-int-range values on the device. Only convert once raw is finite, non-negative, in range
    // and integral.
    if (isfinite(raw) && raw >= 0.0f && raw < (float)param_size && raw == truncf(raw)) return (int)raw;
    return -1;
}

// ---------------------------------------------------------------------------
// Sparse SGD (no momentum)
// ---------------------------------------------------------------------------
extern ""C"" __global__ __launch_bounds__(256) void sparse_sgd_update(
    float* __restrict__ param,
    const float* __restrict__ indices,
    const float* __restrict__ values,
    float learningRate, float weightDecay, int nnz, int param_size)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= nnz) return;
    int i = decode_sparse_index(indices[k], param_size); if (i < 0) return;
    float grad = values[k];
    if (weightDecay > 0.0f) grad += weightDecay * param[i];
    param[i] -= learningRate * grad;
}

// ---------------------------------------------------------------------------
// Sparse SGD with momentum
// ---------------------------------------------------------------------------
extern ""C"" __global__ __launch_bounds__(256) void sparse_sgd_momentum_update(
    float* __restrict__ param,
    const float* __restrict__ indices,
    const float* __restrict__ values,
    float* __restrict__ velocity,
    float learningRate, float momentum, float weightDecay, int nnz, int param_size)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= nnz) return;
    int i = decode_sparse_index(indices[k], param_size); if (i < 0) return;
    float grad = values[k];
    if (weightDecay > 0.0f) grad += weightDecay * param[i];
    float v = momentum * velocity[i] + grad;
    velocity[i] = v;
    param[i] -= learningRate * v;
}

// ---------------------------------------------------------------------------
// Sparse Adam
// ---------------------------------------------------------------------------
extern ""C"" __global__ __launch_bounds__(256) void sparse_adam_update(
    float* __restrict__ param,
    const float* __restrict__ indices,
    const float* __restrict__ values,
    float* __restrict__ m, float* __restrict__ v,
    float learningRate, float beta1, float beta2, float epsilon,
    float weightDecay, int step, int nnz, int param_size)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= nnz) return;
    int i = decode_sparse_index(indices[k], param_size); if (i < 0) return;
    float grad = values[k];
    if (weightDecay > 0.0f) grad += weightDecay * param[i];
    float mVal = beta1 * m[i] + (1.0f - beta1) * grad;
    float vVal = beta2 * v[i] + (1.0f - beta2) * grad * grad;
    m[i] = mVal;
    v[i] = vVal;
    float mHat = mVal / bias_correction(beta1, (float)step);
    float vHat = vVal / bias_correction(beta2, (float)step);
    float update = learningRate * mHat / (sqrtf(vHat) + epsilon);
    param[i] -= update;
}

// ---------------------------------------------------------------------------
// Sparse AdamW (decoupled weight decay)
// ---------------------------------------------------------------------------
extern ""C"" __global__ __launch_bounds__(256) void sparse_adamw_update(
    float* __restrict__ param,
    const float* __restrict__ indices,
    const float* __restrict__ values,
    float* __restrict__ m, float* __restrict__ v,
    float learningRate, float beta1, float beta2, float epsilon,
    float weightDecay, int step, int nnz, int param_size)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= nnz) return;
    int i = decode_sparse_index(indices[k], param_size); if (i < 0) return;
    float grad = values[k];
    float p = param[i];
    if (weightDecay > 0.0f) p -= learningRate * weightDecay * p;
    float mVal = beta1 * m[i] + (1.0f - beta1) * grad;
    float vVal = beta2 * v[i] + (1.0f - beta2) * grad * grad;
    m[i] = mVal;
    v[i] = vVal;
    float mHat = mVal / bias_correction(beta1, (float)step);
    float vHat = vVal / bias_correction(beta2, (float)step);
    param[i] = p - learningRate * mHat / (sqrtf(vHat) + epsilon);
}

// ---------------------------------------------------------------------------
// Sparse RMSProp
// ---------------------------------------------------------------------------
extern ""C"" __global__ __launch_bounds__(256) void sparse_rmsprop_update(
    float* __restrict__ param,
    const float* __restrict__ indices,
    const float* __restrict__ values,
    float* __restrict__ squaredAvg,
    float learningRate, float rho, float epsilon, float weightDecay, int nnz, int param_size)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= nnz) return;
    int i = decode_sparse_index(indices[k], param_size); if (i < 0) return;
    float grad = values[k];
    if (weightDecay > 0.0f) grad += weightDecay * param[i];
    float sq = rho * squaredAvg[i] + (1.0f - rho) * grad * grad;
    squaredAvg[i] = sq;
    param[i] -= learningRate * grad / (sqrtf(sq) + epsilon);
}

// ---------------------------------------------------------------------------
// Sparse Adagrad
// ---------------------------------------------------------------------------
extern ""C"" __global__ __launch_bounds__(256) void sparse_adagrad_update(
    float* __restrict__ param,
    const float* __restrict__ indices,
    const float* __restrict__ values,
    float* __restrict__ accum,
    float learningRate, float epsilon, float weightDecay, int nnz, int param_size)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= nnz) return;
    int i = decode_sparse_index(indices[k], param_size); if (i < 0) return;
    float grad = values[k];
    if (weightDecay > 0.0f) grad += weightDecay * param[i];
    float a = accum[i] + grad * grad;
    accum[i] = a;
    param[i] -= learningRate * grad / (sqrtf(a) + epsilon);
}

// ---------------------------------------------------------------------------
// Sparse NAG (Nesterov-accelerated SGD with momentum)
// ---------------------------------------------------------------------------
extern ""C"" __global__ __launch_bounds__(256) void sparse_nag_update(
    float* __restrict__ param,
    const float* __restrict__ indices,
    const float* __restrict__ values,
    float* __restrict__ velocity,
    float learningRate, float momentum, float weightDecay, int nnz, int param_size)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= nnz) return;
    int i = decode_sparse_index(indices[k], param_size); if (i < 0) return;
    float grad = values[k];
    if (weightDecay > 0.0f) grad += weightDecay * param[i];
    float vOld = velocity[i];
    float vNew = momentum * vOld + grad;
    velocity[i] = vNew;
    // Nesterov look-ahead: use (1 + momentum) * vNew - momentum * vOld
    param[i] -= learningRate * ((1.0f + momentum) * vNew - momentum * vOld);
}

// ---------------------------------------------------------------------------
// Sparse AdaDelta
// ---------------------------------------------------------------------------
extern ""C"" __global__ __launch_bounds__(256) void sparse_adadelta_update(
    float* __restrict__ param,
    const float* __restrict__ indices,
    const float* __restrict__ values,
    float* __restrict__ accumGrad, float* __restrict__ accumUpdate,
    float rho, float epsilon, float weightDecay, int nnz, int param_size)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= nnz) return;
    int i = decode_sparse_index(indices[k], param_size); if (i < 0) return;
    float grad = values[k];
    if (weightDecay > 0.0f) grad += weightDecay * param[i];
    float oneMinusRho = 1.0f - rho;
    float ag = rho * accumGrad[i] + oneMinusRho * grad * grad;
    accumGrad[i] = ag;
    float dx = sqrtf(accumUpdate[i] + epsilon) / sqrtf(ag + epsilon) * grad;
    accumUpdate[i] = rho * accumUpdate[i] + oneMinusRho * dx * dx;
    param[i] -= dx;
}

// ---------------------------------------------------------------------------
// Sparse AMSGrad
// ---------------------------------------------------------------------------
extern ""C"" __global__ __launch_bounds__(256) void sparse_amsgrad_update(
    float* __restrict__ param,
    const float* __restrict__ indices,
    const float* __restrict__ values,
    float* __restrict__ m, float* __restrict__ v, float* __restrict__ vMax,
    float learningRate, float beta1, float beta2, float epsilon,
    float weightDecay, int step, int nnz, int param_size)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= nnz) return;
    int i = decode_sparse_index(indices[k], param_size); if (i < 0) return;
    float grad = values[k];
    if (weightDecay > 0.0f) grad += weightDecay * param[i];
    float mVal = beta1 * m[i] + (1.0f - beta1) * grad;
    float vVal = beta2 * v[i] + (1.0f - beta2) * grad * grad;
    m[i] = mVal;
    v[i] = vVal;
    float vMaxNew = fmaxf(vMax[i], vVal);
    vMax[i] = vMaxNew;
    float mHat = mVal / bias_correction(beta1, (float)step);
    float vHat = vMaxNew / bias_correction(beta2, (float)step);
    param[i] -= learningRate * mHat / (sqrtf(vHat) + epsilon);
}

// ---------------------------------------------------------------------------
// Sparse Adamax (Adam with L-infinity norm)
// ---------------------------------------------------------------------------
extern ""C"" __global__ __launch_bounds__(256) void sparse_adamax_update(
    float* __restrict__ param,
    const float* __restrict__ indices,
    const float* __restrict__ values,
    float* __restrict__ m, float* __restrict__ u,
    float learningRate, float beta1, float beta2, float epsilon,
    float weightDecay, int step, int nnz, int param_size)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= nnz) return;
    int i = decode_sparse_index(indices[k], param_size); if (i < 0) return;
    float grad = values[k];
    if (weightDecay > 0.0f) grad += weightDecay * param[i];
    float mVal = beta1 * m[i] + (1.0f - beta1) * grad;
    float uVal = fmaxf(beta2 * u[i], fabsf(grad));
    m[i] = mVal;
    u[i] = uVal;
    float lrAdj = learningRate / bias_correction(beta1, (float)step);
    param[i] -= lrAdj * mVal / (uVal + epsilon);
}

// ---------------------------------------------------------------------------
// Sparse Lion
// ---------------------------------------------------------------------------
extern ""C"" __global__ __launch_bounds__(256) void sparse_lion_update(
    float* __restrict__ param,
    const float* __restrict__ indices,
    const float* __restrict__ values,
    float* __restrict__ m,
    float learningRate, float beta1, float beta2, float weightDecay, int nnz, int param_size)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= nnz) return;
    int i = decode_sparse_index(indices[k], param_size); if (i < 0) return;
    float grad = values[k];
    float c = beta1 * m[i] + (1.0f - beta1) * grad;
    float sign = (c > 0.0f) ? 1.0f : ((c < 0.0f) ? -1.0f : 0.0f);
    float update = sign;
    if (weightDecay > 0.0f) update += weightDecay * param[i];
    param[i] -= learningRate * update;
    m[i] = beta2 * m[i] + (1.0f - beta2) * grad;
}

// ---------------------------------------------------------------------------
// Sparse Nadam
// ---------------------------------------------------------------------------
extern ""C"" __global__ __launch_bounds__(256) void sparse_nadam_update(
    float* __restrict__ param,
    const float* __restrict__ indices,
    const float* __restrict__ values,
    float* __restrict__ m, float* __restrict__ v,
    float learningRate, float beta1, float beta2, float epsilon,
    float weightDecay, int step, int nnz, int param_size)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= nnz) return;
    int i = decode_sparse_index(indices[k], param_size); if (i < 0) return;
    float grad = values[k];
    if (weightDecay > 0.0f) grad += weightDecay * param[i];
    float mVal = beta1 * m[i] + (1.0f - beta1) * grad;
    float vVal = beta2 * v[i] + (1.0f - beta2) * grad * grad;
    m[i] = mVal;
    v[i] = vVal;
    float bc1 = bias_correction(beta1, (float)step);
    float bc1Next = bias_correction(beta1, (float)(step + 1));
    float bc2 = bias_correction(beta2, (float)step);
    float mHat = beta1 * (mVal / bc1Next) + (1.0f - beta1) * grad / bc1;   // Dozat 2016 Alg. 2; see nadam_update
    float vHat = vVal / bc2;
    param[i] -= learningRate * mHat / (sqrtf(vHat) + epsilon);
}

// ---------------------------------------------------------------------------
// Sparse FTRL (Follow-The-Regularized-Leader)
// ---------------------------------------------------------------------------
extern ""C"" __global__ __launch_bounds__(256) void sparse_ftrl_update(
    float* __restrict__ param,
    const float* __restrict__ indices,
    const float* __restrict__ values,
    float* __restrict__ z, float* __restrict__ n,
    float learningRate, float l1Reg, float l2Reg, float beta, int nnz, int param_size)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= nnz) return;
    int i = decode_sparse_index(indices[k], param_size); if (i < 0) return;
    float grad = values[k];
    float nOld = n[i];
    float nNew = nOld + grad * grad;
    n[i] = nNew;
    float sigma = (sqrtf(nNew) - sqrtf(nOld)) / learningRate;
    z[i] += grad - sigma * param[i];
    float zVal = z[i];
    if (fabsf(zVal) <= l1Reg) {
        param[i] = 0.0f;
    } else {
        float sign = (zVal > 0.0f) ? 1.0f : -1.0f;
        param[i] = (sign * l1Reg - zVal) / ((beta + sqrtf(nNew)) / learningRate + l2Reg);
    }
}

// ---------------------------------------------------------------------------
// Sparse Proximal-L1
// ---------------------------------------------------------------------------
extern ""C"" __global__ __launch_bounds__(256) void sparse_proximal_l1_update(
    float* __restrict__ param,
    const float* __restrict__ indices,
    const float* __restrict__ values,
    float learningRate, float l1Strength, int nnz, int param_size)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= nnz) return;
    int i = decode_sparse_index(indices[k], param_size); if (i < 0) return;
    float grad = values[k];
    float p = param[i] - learningRate * grad;
    float threshold = learningRate * l1Strength;
    if (p > threshold)      param[i] = p - threshold;
    else if (p < -threshold) param[i] = p + threshold;
    else                     param[i] = 0.0f;
}

// ---- Multi-tensor gradient reductions for the compiled training step -----------------------------------------
// One launch over every gradient tensor instead of a fill + square + reduce + add (and a scale) per tensor: the
// per-tensor form issued ~630 launches per N-BEATS step (AiDotNet #1804). Same chunk table as the multi-tensor
// optimizers: block b covers elements [chunkStart[b], chunkStart[b] + blockDim.x) of tensor chunkTensor[b].

__device__ __forceinline__ void mt_atomic_add_double(double* address, double value)
{
#if __CUDA_ARCH__ >= 600
    atomicAdd(address, value);
#else
    unsigned long long* p = (unsigned long long*)address;
    unsigned long long old = *p, assumed;
    do
    {
        assumed = old;
        old = atomicCAS(p, assumed, __double_as_longlong(value + __longlong_as_double(assumed)));
    } while (assumed != old);
#endif
}

// Elements per multi-tensor gradient-table chunk (CudaBackend.GradientTableChunk must match).
#define MT_CHUNK 4096

// Sum of squares of every element of every tensor, accumulated in double. A float squared cannot overflow a double,
// so the result is finite exactly when every element is finite: one reduction serves both the global-norm clip and
// the non-finite-gradient check. *out must be zeroed before the launch.
extern ""C"" __global__ __launch_bounds__(256) void multi_tensor_sum_squares(
    const unsigned long long* __restrict__ ptrs,
    const int* __restrict__ sizes,
    const int* __restrict__ chunkTensor,
    const int* __restrict__ chunkStart,
    int totalChunks,
    double* __restrict__ out)
{
    __shared__ double warpSums[8];
    // A bounded grid strides over the chunks and accumulates locally, so the result takes one atomic per block
    // rather than one per chunk: ~10k chunk-atomics on one address serialized to ~300 us per call.
    // Each chunk is up to MT_CHUNK (4096) elements; the block strides over it in float4s when the tensor's base
    // pointer is 16-byte aligned (chunk starts are multiples of 4096, so every chunk then is), scalar otherwise and
    // for the tail (#1106: one 256-element chunk per block-iteration and scalar loads left the kernel launch-bound).
    double v = 0.0;
    for (int c = blockIdx.x; c < totalChunks; c += gridDim.x)
    {
        int t = chunkTensor[c];
        int start = chunkStart[c];
        int end = min(start + MT_CHUNK, sizes[t]);
        const float* p = (const float*)ptrs[t];
        int i = start;
        if ((ptrs[t] & 15ULL) == 0)
        {
            int nvec = (end - start) >> 2;
            const float4* p4 = (const float4*)(p + start);
            for (int k = threadIdx.x; k < nvec; k += blockDim.x)
            {
                float4 x = p4[k];
                // Four squares summed in float, one double add per float4: the double multiply-adds ran at consumer
                // GPUs' 1/64-rate FP64 and made the kernel compute-bound (128 us for ResNet-18's 11.2M gradients on an
                // RTX 3080, ~2x its memory time). A float partial that is not finite -- a non-finite element, or
                // squares beyond float range (|x| > ~1.8e19) -- is redone in double, so the result stays finite
                // exactly when every element is.
                float s = x.x * x.x + x.y * x.y + x.z * x.z + x.w * x.w;
                v += isfinite(s) ? (double)s
                    : (double)x.x * (double)x.x + (double)x.y * (double)x.y + (double)x.z * (double)x.z + (double)x.w * (double)x.w;
            }
            i = start + (nvec << 2);
        }
        for (i += threadIdx.x; i < end; i += blockDim.x)
        {
            double x = (double)p[i];
            v += x * x;
        }
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        v += __shfl_down_sync(0xFFFFFFFF, v, o);
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    if (lane == 0)
        warpSums[warp] = v;
    __syncthreads();
    if (warp == 0)
    {
        v = lane < (int)((blockDim.x + 31) >> 5) ? warpSums[lane] : 0.0;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1)
            v += __shfl_down_sync(0xFFFFFFFF, v, o);
        if (lane == 0)
            mt_atomic_add_double(out, v);
    }
}

// PyTorch clip_grad_norm_ coefficient on the device: min(1, maxNorm / (sqrt(sumSq) + 1e-6)). A non-finite norm
// leaves the gradients unscaled (1); the non-finite check then discards the step.
extern ""C"" __global__ void clip_scale_from_sum_squares(
    const double* __restrict__ sumSquares, float maxNorm, float* __restrict__ scale)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    double norm = sqrt(*sumSquares);
    // Exponent bits, not isfinite(): under -ffast-math (finite-math-only) isfinite may fold to true.
    double c = (__double_as_longlong(norm) & 0x7ff0000000000000LL) != 0x7ff0000000000000LL ? (double)maxNorm / (norm + 1e-6) : 1.0;
    *scale = (float)(c < 1.0 ? c : 1.0);
}

// Every element of every tensor multiplied by the device scalar *scale: a bounded grid strides over the MT_CHUNK
// chunks, in float4s when the tensor's base pointer is 16-byte aligned (scalar otherwise and for the tail).
extern ""C"" __global__ __launch_bounds__(256) void multi_tensor_scale_by_device_scalar(
    const unsigned long long* __restrict__ ptrs,
    const int* __restrict__ sizes,
    const int* __restrict__ chunkTensor,
    const int* __restrict__ chunkStart,
    const float* __restrict__ scale,
    int totalChunks)
{
    float s = *scale;
    for (int c = blockIdx.x; c < totalChunks; c += gridDim.x)
    {
        int t = chunkTensor[c];
        int start = chunkStart[c];
        int end = min(start + MT_CHUNK, sizes[t]);
        float* p = (float*)ptrs[t];
        int i = start;
        if ((ptrs[t] & 15ULL) == 0)
        {
            int nvec = (end - start) >> 2;
            float4* p4 = (float4*)(p + start);
            for (int k = threadIdx.x; k < nvec; k += blockDim.x)
            {
                float4 x = p4[k];
                x.x *= s; x.y *= s; x.z *= s; x.w *= s;
                p4[k] = x;
            }
            i = start + (nvec << 2);
        }
        for (i += threadIdx.x; i < end; i += blockDim.x)
            p[i] *= s;
    }
}

// ---- Ping-pong SGD for captured step graphs ------------------------------------------------------------------
// Two step graphs alternate: one reads the weights from buffer A and writes the updated weights to buffer B, the
// other the reverse. Writing the update elsewhere lets one pass over the gradients both update and take their sum of
// squares: an in-place update has to wait for the whole reduction (a non-finite gradient anywhere skips the step), so
// it read the gradients twice. A non-finite step is undone inside sgd_pingpong_update, by the last block to finish.
// Table: ptrs[t] = tensor t's source weights, ptrs[n + t] its destination, ptrs[2n + t] its gradient, cut into
// PP_CHUNK-element chunks (CudaBackend.PingPongChunk, 1024) that a bounded grid strides over.

// The block's sum, valid in thread 0. warpSums: 8 doubles of shared memory, free on entry and on return.
__device__ __forceinline__ double pp_block_sum(double v, double* warpSums)
{
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        v += __shfl_down_sync(0xFFFFFFFF, v, o);
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) warpSums[warp] = v;
    __syncthreads();
    v = 0.0;
    if (warp == 0)
    {
        v = lane < (int)((blockDim.x + 31) >> 5) ? warpSums[lane] : 0.0;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1)
            v += __shfl_down_sync(0xFFFFFFFF, v, o);
    }
    __syncthreads();
    return v;
}

// Elements per ping-pong chunk (CudaBackend.PingPongChunk must match): one 16-byte load of each operand per thread.
#define PP_CHUNK 1024

// dst = src - lr * (g + wd * src) (sgd_multi_tensor_update's arithmetic, so the weights match it bit for bit), and the
// gradients' sum of squares (multi_tensor_sum_squares' arithmetic) onto *guard, for a grid that strides over the
// PP_CHUNK-element chunks (the host bounds it at a few blocks per SM). Each thread issues each of its 16-byte quads' two
// loads before that quad's store (one quad per thread at 256 threads; the quad loop strides by blockDim.x, so any block
// size covers the chunk). Each block leaves one partial sum; the last block to finish has one warp add them in block
// order, so on one device the sum does not depend on scheduling (its last bits do depend on the SM count, which sets
// the grid: the guard only gates finiteness, which no grouping changes), adds that onto the guard and resets *done.
// If the guard is then not finite (a non-finite gradient, or a learning-rate fault that poisoned it), that block also
// undoes the step, copying every source over its destination: slow, but only on a step that is being discarded.
// dyn: device float[2], [0] = this step's learning rate.
extern ""C"" __global__ __launch_bounds__(256) void sgd_pingpong_update(
    const unsigned long long* __restrict__ ptrs,
    const int* __restrict__ sizes,
    const int* __restrict__ chunkTensor,
    const int* __restrict__ chunkStart,
    int n, int totalChunks, float weightDecay,
    const float* __restrict__ dyn,
    double* __restrict__ partials,
    unsigned int* __restrict__ done,
    double* __restrict__ guard)
{
    __shared__ double warpSums[8];
    __shared__ int isLast, undo;
    const float learningRate = dyn[0];
    double v = 0.0;
    for (int c = blockIdx.x; c < totalChunks; c += gridDim.x)
    {
        const int t = chunkTensor[c];
        const int start = chunkStart[c];
        const int end = min(start + PP_CHUNK, sizes[t]);
        const float* src = (const float*)ptrs[t];
        float* dst = (float*)ptrs[n + t];
        const float* gradient = (const float*)ptrs[2 * n + t];
        if (((ptrs[t] | ptrs[n + t] | ptrs[2 * n + t]) & 15ULL) == 0)
        {
            for (int e = start + threadIdx.x * 4; e < end; e += blockDim.x * 4)
            {
                if (e + 4 <= end)
                {
                    float4 g = *reinterpret_cast<const float4*>(gradient + e);
                    float4 w = *reinterpret_cast<const float4*>(src + e);
                    float s = g.x * g.x + g.y * g.y + g.z * g.z + g.w * g.w;
                    v += isfinite(s) ? (double)s
                        : (double)g.x * (double)g.x + (double)g.y * (double)g.y + (double)g.z * (double)g.z + (double)g.w * (double)g.w;
                    if (weightDecay > 0.0f) {
                        g.x += weightDecay * w.x; g.y += weightDecay * w.y; g.z += weightDecay * w.z; g.w += weightDecay * w.w;
                    }
                    w.x -= learningRate * g.x; w.y -= learningRate * g.y; w.z -= learningRate * g.z; w.w -= learningRate * g.w;
                    *reinterpret_cast<float4*>(dst + e) = w;
                }
                else
                {
                    for (int i = e; i < end; i++)
                    {
                        float g = gradient[i];
                        float w = src[i];
                        v += (double)g * (double)g;
                        if (weightDecay > 0.0f) {
                            g += weightDecay * w;
                        }
                        w -= learningRate * g;
                        dst[i] = w;
                    }
                }
            }
        }
        else
        {
            for (int i = start + threadIdx.x; i < end; i += blockDim.x)
            {
                float g = gradient[i];
                float w = src[i];
                v += (double)g * (double)g;
                if (weightDecay > 0.0f) {
                    g += weightDecay * w;
                }
                w -= learningRate * g;
                dst[i] = w;
            }
        }
    }
    v = pp_block_sum(v, warpSums);
    if (threadIdx.x == 0)
    {
        partials[blockIdx.x] = v;
        __threadfence();   // the partial is visible device-wide before the ticket that says it is there
        isLast = atomicAdd(done, 1u) == gridDim.x - 1u;
    }
    __syncthreads();
    if (!isLast) return;
    if (threadIdx.x < 32)
    {
        double total = 0.0;
        for (int k = threadIdx.x; k < (int)gridDim.x; k += 32) total += __ldcg(partials + k);
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1)
            total += __shfl_down_sync(0xFFFFFFFF, total, o);
        if (threadIdx.x == 0)
        {
            double g = *guard + total;   // onto the guard, which a learning-rate fault may already have poisoned
            *guard = g;
            *done = 0u;
            // Exponent bits, not isfinite(): see clip_scale_from_sum_squares.
            undo = (__double_as_longlong(g) & 0x7ff0000000000000LL) == 0x7ff0000000000000LL;
        }
    }
    __syncthreads();
    if (!undo) return;
    __threadfence();   // every block's stores are visible before they are overwritten
    for (int c = 0; c < totalChunks; c++)
    {
        const int t = chunkTensor[c];
        const int start = chunkStart[c];
        const int end = min(start + PP_CHUNK, sizes[t]);
        const float* src = (const float*)ptrs[t];
        float* dst = (float*)ptrs[n + t];
        for (int i = start + threadIdx.x; i < end; i += blockDim.x) dst[i] = __ldcg(src + i);
    }
}
";
    }

    /// <summary>
    /// Gets the list of kernel names for compilation.
    /// </summary>
    public static string[] GetKernelNames()
    {
        return new[]
        {
            "sgd_momentum_update",
            "sgd_update",
            "adam_update",
            "adamw_update",
            "sgd_multi_tensor_update",
            "adam_multi_tensor_update",
            "adamw_multi_tensor_update",
            "multi_tensor_sum_squares",
            "clip_scale_from_sum_squares",
            "multi_tensor_scale_by_device_scalar",
            "sgd_pingpong_update",
            "adam_bf16_update",
            "adamw_bf16_update",
            "rmsprop_update",
            "adagrad_update",
            "nag_update",
            "lars_update",
            "lamb_update",
            "adadelta_update",
            "amsgrad_update",
            "adamax_update",
            "lion_update",
            "nadam_update",
            "ftrl_update",
            "proximal_l1_update",
            "adam8bit_update",
            // PR #567 — sparse counterparts (one thread per nnz, scatter-update).
            "sparse_sgd_update",
            "sparse_sgd_momentum_update",
            "sparse_adam_update",
            "sparse_adamw_update",
            "sparse_rmsprop_update",
            "sparse_adagrad_update",
            "sparse_nag_update",
            "sparse_adadelta_update",
            "sparse_amsgrad_update",
            "sparse_adamax_update",
            "sparse_lion_update",
            "sparse_nadam_update",
            "sparse_ftrl_update",
            "sparse_proximal_l1_update",
        };
    }
}
