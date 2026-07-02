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
        return @"
// ===========================================================================
// CUDA OPTIMIZER KERNELS
// ===========================================================================

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

    float mVal = beta1 * m[idx] + (1.0f - beta1) * grad;
    float vVal = beta2 * v[idx] + (1.0f - beta2) * grad * grad;
    m[idx] = mVal;
    v[idx] = vVal;

    float mHat = mVal / (1.0f - powf(beta1, (float)step));
    float vHat = vVal / (1.0f - powf(beta2, (float)step));

    float update = learningRate * mHat / (sqrtf(vHat) + epsilon);
    if (weightDecay > 0.0f) {
        update += learningRate * weightDecay * param[idx];
    }
    param[idx] -= update;
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

    float mHat = mVal / (1.0f - powf(beta1, (float)step));
    float vHat = vVal / (1.0f - powf(beta2, (float)step));

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

    float mHat = mVal / (1.0f - powf(beta1, (float)step));
    float vHat = vVal / (1.0f - powf(beta2, (float)step));

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

    float mHat = mVal / (1.0f - powf(beta1, (float)step));

    param[idx] -= learningRate * mHat / (sqrtf(vMaxVal) + epsilon);
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

    float biasCorrection = 1.0f - powf(beta1, (float)step);

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

    float beta1Pow = powf(beta1, (float)step);
    float beta2Pow = powf(beta2, (float)step);
    float mHat = mVal / (1.0f - beta1Pow);
    float vHat = vVal / (1.0f - beta2Pow);

    float mNesterov = beta1 * mHat + (1.0f - beta1) * grad / (1.0f - beta1Pow);

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
// no stochastic rounding) per block: m is SIGNED int8 (scale=maxAbs/127, stored
// as q+128), v is UNSIGNED int8 (scale=maxAbs/255). One CUDA block per quant
// block; blockDim=256; shared-memory maxAbs reduction recomputes the per-block
// scale before requantizing. Adam math is float (matching the CPU Tensor ops);
// quant/dequant scaling is double (matching NumOps.ToDouble); rint() = round-
// half-to-even = C# Math.Round. Launch with gridDim.x = numBlocks.
// ---------------------------------------------------------------------------
extern ""C"" __global__ __launch_bounds__(256) void adam8bit_update(
    float* __restrict__ param, const float* __restrict__ gradient,
    unsigned char* __restrict__ mQ, unsigned char* __restrict__ vQ,
    double* __restrict__ mScales, double* __restrict__ vScales,
    float learningRate, float beta1, float beta2, float epsilon,
    float oneMinusBeta1, float oneMinusBeta2, float biasCorrection1, float biasCorrection2,
    int blockSize, int paramLength, int numBlocks)
{
    int blk = blockIdx.x;
    if (blk >= numBlocks) return;
    int start = blk * blockSize;
    int endIdx = start + blockSize; if (endIdx > paramLength) endIdx = paramLength;
    double mScale = mScales[blk];
    double vScale = vScales[blk];

    __shared__ float sMaxM[256];
    __shared__ float sMaxV[256];

    // Phase 1: compute newM/newV, reduce per-block maxAbs for the new scales.
    float locM = 0.0f, locV = 0.0f;
    for (int i = start + threadIdx.x; i < endIdx; i += blockDim.x) {
        float m_i = (float)((double)((int)mQ[i] - 128) * mScale);
        float v_i = (float)((double)((int)vQ[i]) * vScale);
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
    double newMScale = (double)sMaxM[0] / 127.0; if (newMScale < 1e-10) newMScale = 1e-10;
    double newVScale = (double)sMaxV[0] / 255.0; if (newVScale < 1e-10) newVScale = 1e-10;
    if (threadIdx.x == 0) { mScales[blk] = newMScale; vScales[blk] = newVScale; }
    __syncthreads();

    // Phase 2: recompute, apply bias-corrected param update, requantize.
    for (int i = start + threadIdx.x; i < endIdx; i += blockDim.x) {
        float m_i = (float)((double)((int)mQ[i] - 128) * mScale);
        float v_i = (float)((double)((int)vQ[i]) * vScale);
        float g = gradient[i];
        float newM = beta1 * m_i + oneMinusBeta1 * g;
        float newV = beta2 * v_i + oneMinusBeta2 * (g * g);

        float mHat = newM / biasCorrection1;
        float vHat = newV / biasCorrection2;
        param[i] = param[i] - learningRate * mHat / (sqrtf(vHat) + epsilon);

        int qm = (int)rint((double)newM / newMScale);
        if (qm < -127) qm = -127; if (qm > 127) qm = 127;
        mQ[i] = (unsigned char)(qm + 128);

        int qv = (int)rint((double)newV / newVScale);
        if (qv < 0) qv = 0; if (qv > 255) qv = 255;
        vQ[i] = (unsigned char)qv;
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
    float mVal = beta1 * m[i] + (1.0f - beta1) * grad;
    float vVal = beta2 * v[i] + (1.0f - beta2) * grad * grad;
    m[i] = mVal;
    v[i] = vVal;
    float mHat = mVal / (1.0f - powf(beta1, (float)step));
    float vHat = vVal / (1.0f - powf(beta2, (float)step));
    float update = learningRate * mHat / (sqrtf(vHat) + epsilon);
    if (weightDecay > 0.0f) update += learningRate * weightDecay * param[i];
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
    float mHat = mVal / (1.0f - powf(beta1, (float)step));
    float vHat = vVal / (1.0f - powf(beta2, (float)step));
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
    float mHat = mVal / (1.0f - powf(beta1, (float)step));
    float vHat = vMaxNew / (1.0f - powf(beta2, (float)step));
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
    float lrAdj = learningRate / (1.0f - powf(beta1, (float)step));
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
    float bc1 = 1.0f - powf(beta1, (float)step);
    float bc2 = 1.0f - powf(beta2, (float)step);
    float mHat = (beta1 * mVal + (1.0f - beta1) * grad) / bc1;
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
