// Copyright (c) AiDotNet. All rights reserved.
// GPU kernels for gradient-based optimizers.
// Provides element-wise parameter updates for various optimization algorithms.

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL.Kernels;

/// <summary>
/// GPU kernels for gradient-based optimizers.
/// These kernels enable GPU-resident training by keeping parameters
/// and optimizer state entirely on the GPU during training.
/// </summary>
internal static class OptimizerKernels
{
    /// <summary>
    /// Gets all optimizer kernel sources.
    /// </summary>
    public static string GetSource()
    {
        return @"
// ===========================================================================
// OPTIMIZER KERNELS
// ===========================================================================
// These kernels implement element-wise parameter updates for various
// gradient-based optimization algorithms. Each kernel operates on individual
// parameter elements in parallel.

static inline float bf16_to_float(ushort x)
{
    return as_float(((uint)x) << 16);
}

static inline ushort float_to_bf16_rne(float x)
{
    uint bits = as_uint(x);
    if ((bits & 0x7FFFFFFFU) > 0x7F800000U) {
        return (ushort)((bits >> 16) | 0x0040U);
    }
    uint rounding = 0x7FFFU + ((bits >> 16) & 1U);
    return (ushort)((bits + rounding) >> 16);
}

// ---------------------------------------------------------------------------
// SGD with momentum update
// Formula: v = momentum * v + grad
//          param = param - lr * v
// ---------------------------------------------------------------------------
__kernel void sgd_momentum_update(
    __global float* param,
    __global const float* gradient,
    __global float* velocity,
    const float learningRate,
    const float momentum,
    const float weightDecay,
    const int size)
{
    const int idx = get_global_id(0);
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
// Vanilla SGD update
// Formula: param = param - lr * grad
// ---------------------------------------------------------------------------
__kernel void sgd_update(
    __global float* param,
    __global const float* gradient,
    const float learningRate,
    const float weightDecay,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    float grad = gradient[idx];
    if (weightDecay > 0.0f) {
        grad += weightDecay * param[idx];
    }

    param[idx] -= learningRate * grad;
}

// ---------------------------------------------------------------------------
// Adam optimizer update
// Formula: m = beta1 * m + (1 - beta1) * grad
//          v = beta2 * v + (1 - beta2) * grad^2
//          m_hat = m / (1 - beta1^t)
//          v_hat = v / (1 - beta2^t)
//          param = param - lr * m_hat / (sqrt(v_hat) + epsilon)
// ---------------------------------------------------------------------------
__kernel void adam_update(
    __global float* param,
    __global const float* gradient,
    __global float* m,
    __global float* v,
    const float learningRate,
    const float beta1,
    const float beta2,
    const float epsilon,
    const float weightDecay,
    const int step,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    float grad = gradient[idx];
    if (weightDecay > 0.0f) {
        grad += weightDecay * param[idx];
    }

    // Update biased first moment estimate
    float m_new = beta1 * m[idx] + (1.0f - beta1) * grad;
    m[idx] = m_new;

    // Update biased second moment estimate
    float v_new = beta2 * v[idx] + (1.0f - beta2) * grad * grad;
    v[idx] = v_new;

    // Bias correction - guard against step==0 which causes division by zero
    // Step should always be >= 1; if step==0, use step==1 to avoid NaN/Inf
    int safe_step = step < 1 ? 1 : step;
    float m_hat = m_new / (1.0f - pow(beta1, (float)safe_step));
    float v_hat = v_new / (1.0f - pow(beta2, (float)safe_step));

    // Update parameters
    param[idx] -= learningRate * m_hat / (sqrt(v_hat) + epsilon);
}

// ---------------------------------------------------------------------------
// AdamW optimizer update (decoupled weight decay)
// Like Adam but weight decay is applied directly to params, not gradients
// ---------------------------------------------------------------------------
__kernel void adamw_update(
    __global float* param,
    __global const float* gradient,
    __global float* m,
    __global float* v,
    const float learningRate,
    const float beta1,
    const float beta2,
    const float epsilon,
    const float weightDecay,
    const int step,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    float grad = gradient[idx];

    // Decoupled weight decay (applied directly to params, not gradients)
    if (weightDecay > 0.0f) {
        param[idx] *= (1.0f - learningRate * weightDecay);
    }

    // Update biased first moment estimate
    float m_new = beta1 * m[idx] + (1.0f - beta1) * grad;
    m[idx] = m_new;

    // Update biased second moment estimate
    float v_new = beta2 * v[idx] + (1.0f - beta2) * grad * grad;
    v[idx] = v_new;

    // Bias correction - guard against step==0 which causes division by zero
    int safe_step = step < 1 ? 1 : step;
    float m_hat = m_new / (1.0f - pow(beta1, (float)safe_step));
    float v_hat = v_new / (1.0f - pow(beta2, (float)safe_step));

    // Update parameters
    param[idx] -= learningRate * m_hat / (sqrt(v_hat) + epsilon);
}

// ---------------------------------------------------------------------------
// Adam optimizer update with bfloat16 moment storage
// ---------------------------------------------------------------------------
__kernel void adam_bf16_update(
    __global float* param,
    __global const float* gradient,
    __global ushort* m,
    __global ushort* v,
    const float learningRate,
    const float beta1,
    const float beta2,
    const float epsilon,
    const float weightDecay,
    const int step,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    float grad = gradient[idx];
    if (weightDecay > 0.0f) {
        grad += weightDecay * param[idx];
    }
    float oldM = step <= 1 ? 0.0f : bf16_to_float(m[idx]);
    float oldV = step <= 1 ? 0.0f : bf16_to_float(v[idx]);
    float m_new = beta1 * oldM + (1.0f - beta1) * grad;
    float v_new = beta2 * oldV + (1.0f - beta2) * grad * grad;
    m[idx] = float_to_bf16_rne(m_new);
    v[idx] = float_to_bf16_rne(v_new);

    int safe_step = step < 1 ? 1 : step;
    float m_hat = m_new / (1.0f - pow(beta1, (float)safe_step));
    float v_hat = v_new / (1.0f - pow(beta2, (float)safe_step));

    param[idx] -= learningRate * m_hat / (sqrt(v_hat) + epsilon);
}

// ---------------------------------------------------------------------------
// AdamW optimizer update with bfloat16 moment storage
// ---------------------------------------------------------------------------
__kernel void adamw_bf16_update(
    __global float* param,
    __global const float* gradient,
    __global ushort* m,
    __global ushort* v,
    const float learningRate,
    const float beta1,
    const float beta2,
    const float epsilon,
    const float weightDecay,
    const int step,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    float grad = gradient[idx];
    if (weightDecay > 0.0f) {
        param[idx] *= (1.0f - learningRate * weightDecay);
    }

    float oldM = step <= 1 ? 0.0f : bf16_to_float(m[idx]);
    float oldV = step <= 1 ? 0.0f : bf16_to_float(v[idx]);
    float m_new = beta1 * oldM + (1.0f - beta1) * grad;
    float v_new = beta2 * oldV + (1.0f - beta2) * grad * grad;
    m[idx] = float_to_bf16_rne(m_new);
    v[idx] = float_to_bf16_rne(v_new);

    int safe_step = step < 1 ? 1 : step;
    float m_hat = m_new / (1.0f - pow(beta1, (float)safe_step));
    float v_hat = v_new / (1.0f - pow(beta2, (float)safe_step));

    param[idx] -= learningRate * m_hat / (sqrt(v_hat) + epsilon);
}

// ---------------------------------------------------------------------------
// 8-bit Adam blockwise dynamic quantization. One OpenCL workgroup per quant block.
// ---------------------------------------------------------------------------
__kernel void adam8bit_update(
    __global float* param,
    __global const float* gradient,
    __global uchar* mQ,
    __global uchar* vQ,
    __global float* mScales,
    __global float* vScales,
    const float learningRate,
    const float beta1,
    const float beta2,
    const float epsilon,
    const float oneMinusBeta1,
    const float oneMinusBeta2,
    const float biasCorrection1,
    const float biasCorrection2,
    const int blockSize,
    const int paramLength,
    const int numBlocks)
{
    const int blk = get_group_id(0);
    const int tid = get_local_id(0);
    if (blk >= numBlocks) return;

    int start = blk * blockSize;
    int endIdx = start + blockSize;
    if (endIdx > paramLength) endIdx = paramLength;
    int firstStep = biasCorrection1 <= oneMinusBeta1 + 1e-7f;
    float mScale = firstStep ? 0.0f : mScales[blk];
    float vScale = firstStep ? 0.0f : vScales[blk];

    __local float sMaxM[256];
    __local float sMaxV[256];

    float locM = 0.0f;
    float locV = 0.0f;
    for (int i = start + tid; i < endIdx; i += 256) {
        float m_i = firstStep ? 0.0f : (float)((int)mQ[i] - 128) * mScale;
        float v_i = firstStep ? 0.0f : (float)((int)vQ[i]) * vScale;
        float g = gradient[i];
        float newM = beta1 * m_i + oneMinusBeta1 * g;
        float newV = beta2 * v_i + oneMinusBeta2 * (g * g);
        locM = fmax(locM, fabs(newM));
        locV = fmax(locV, fabs(newV));
    }

    sMaxM[tid] = locM;
    sMaxV[tid] = locV;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) {
            sMaxM[tid] = fmax(sMaxM[tid], sMaxM[tid + s]);
            sMaxV[tid] = fmax(sMaxV[tid], sMaxV[tid + s]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    float newMScale = sMaxM[0] / 127.0f;
    if (newMScale < 1e-10f) newMScale = 1e-10f;
    float newVScale = sMaxV[0] / 255.0f;
    if (newVScale < 1e-10f) newVScale = 1e-10f;
    if (tid == 0) {
        mScales[blk] = newMScale;
        vScales[blk] = newVScale;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int i = start + tid; i < endIdx; i += 256) {
        float m_i = firstStep ? 0.0f : (float)((int)mQ[i] - 128) * mScale;
        float v_i = firstStep ? 0.0f : (float)((int)vQ[i]) * vScale;
        float g = gradient[i];
        float newM = beta1 * m_i + oneMinusBeta1 * g;
        float newV = beta2 * v_i + oneMinusBeta2 * (g * g);

        float mHat = newM / biasCorrection1;
        float vHat = newV / biasCorrection2;
        param[i] = param[i] - learningRate * mHat / (sqrt(vHat) + epsilon);

        int qm = (int)rint(newM / newMScale);
        if (qm < -127) qm = -127;
        if (qm > 127) qm = 127;
        mQ[i] = (uchar)(qm + 128);

        int qv = (int)rint(newV / newVScale);
        if (qv < 0) qv = 0;
        if (qv > 255) qv = 255;
        vQ[i] = (uchar)qv;
    }
}

// ---------------------------------------------------------------------------
// RMSprop optimizer update
// Formula: sq_avg = rho * sq_avg + (1 - rho) * grad^2
//          param = param - lr * grad / (sqrt(sq_avg) + epsilon)
// ---------------------------------------------------------------------------
__kernel void rmsprop_update(
    __global float* param,
    __global const float* gradient,
    __global float* squaredAvg,
    const float learningRate,
    const float rho,
    const float epsilon,
    const float weightDecay,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    float grad = gradient[idx];
    if (weightDecay > 0.0f) {
        grad += weightDecay * param[idx];
    }

    // Update moving average of squared gradients
    float sqAvg = rho * squaredAvg[idx] + (1.0f - rho) * grad * grad;
    squaredAvg[idx] = sqAvg;

    // Update parameters
    param[idx] -= learningRate * grad / (sqrt(sqAvg) + epsilon);
}

// ---------------------------------------------------------------------------
// Adagrad optimizer update
// Formula: accum = accum + grad^2
//          param = param - lr * grad / (sqrt(accum) + epsilon)
// ---------------------------------------------------------------------------
__kernel void adagrad_update(
    __global float* param,
    __global const float* gradient,
    __global float* accumulatedGrad,
    const float learningRate,
    const float epsilon,
    const float weightDecay,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    float grad = gradient[idx];
    if (weightDecay > 0.0f) {
        grad += weightDecay * param[idx];
    }

    // Accumulate squared gradients
    float accum = accumulatedGrad[idx] + grad * grad;
    accumulatedGrad[idx] = accum;

    // Update parameters
    param[idx] -= learningRate * grad / (sqrt(accum) + epsilon);
}

// ---------------------------------------------------------------------------
// Nesterov Accelerated Gradient (NAG) optimizer update
// Formula: v_new = momentum * v - lr * grad
//          param = param + momentum * v_new - lr * grad (lookahead)
// ---------------------------------------------------------------------------
__kernel void nag_update(
    __global float* param,
    __global const float* gradient,
    __global float* velocity,
    const float learningRate,
    const float momentum,
    const float weightDecay,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    float grad = gradient[idx];
    if (weightDecay > 0.0f) {
        grad += weightDecay * param[idx];
    }

    // Nesterov momentum (Sutskever-style NAG)
    // Note: Caller should compute gradient at lookahead position (theta + mu * v) for true NAG
    float v = velocity[idx];
    float vNew = momentum * v - learningRate * grad;
    velocity[idx] = vNew;

    // NAG update - apply velocity directly
    param[idx] += vNew;
}

// ---------------------------------------------------------------------------
// LARS (Layer-wise Adaptive Rate Scaling) optimizer update
// Note: LARS applies trustCoeff to scale the learning rate adaptively per layer.
// The trustCoeff should be pre-computed as: trustCoeff * ||w|| / (||grad|| + ||w|| * weightDecay)
// Set trustCoeff=1.0f to disable trust coefficient scaling.
// ---------------------------------------------------------------------------
__kernel void lars_update(
    __global float* param,
    __global const float* gradient,
    __global float* velocity,
    const float learningRate,
    const float momentum,
    const float weightDecay,
    const float trustCoeff,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    float grad = gradient[idx];
    float p = param[idx];

    // Apply weight decay
    if (weightDecay > 0.0f) {
        grad += weightDecay * p;
    }

    // Update velocity with momentum
    float v = momentum * velocity[idx] + grad;
    velocity[idx] = v;

    // Update parameters with trust coefficient scaling
    param[idx] = p - learningRate * trustCoeff * v;
}

// ---------------------------------------------------------------------------
// LAMB (Layer-wise Adaptive Moments) optimizer update
// Combines Adam moments with layer-wise trust ratio.
// The trust ratio (||w|| / ||update||) must be pre-computed externally.
// Set trustRatio=1.0f to disable trust ratio scaling (degenerates to AdamW).
// ---------------------------------------------------------------------------
__kernel void lamb_update(
    __global float* param,
    __global const float* gradient,
    __global float* m,
    __global float* v,
    const float learningRate,
    const float beta1,
    const float beta2,
    const float epsilon,
    const float weightDecay,
    const float trustRatio,    // Pre-computed: ||param|| / ||update||, or 1.0 to disable
    const int step,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    float grad = gradient[idx];
    float p = param[idx];

    // Adam-like moment updates
    float mVal = beta1 * m[idx] + (1.0f - beta1) * grad;
    float vVal = beta2 * v[idx] + (1.0f - beta2) * grad * grad;
    m[idx] = mVal;
    v[idx] = vVal;

    // Bias correction - guard against step==0 which causes division by zero
    int safe_step = step < 1 ? 1 : step;
    float mHat = mVal / (1.0f - pow(beta1, (float)safe_step));
    float vHat = vVal / (1.0f - pow(beta2, (float)safe_step));

    // LAMB: Adam update direction with weight decay
    float adamUpdate = mHat / (sqrt(vHat) + epsilon);
    float update = adamUpdate + weightDecay * p;

    // Apply trust ratio scaling (LAMB's layer-wise adaptive learning rate)
    param[idx] = p - learningRate * trustRatio * update;
}

// ---------------------------------------------------------------------------
// AdaDelta optimizer update
// Uses running averages of gradients and updates (no learning rate needed)
// Formula: accum_grad = rho * accum_grad + (1 - rho) * grad^2
//          update = sqrt(accum_update + eps) / sqrt(accum_grad + eps) * grad
//          accum_update = rho * accum_update + (1 - rho) * update^2
//          param = param - update
// ---------------------------------------------------------------------------
__kernel void adadelta_update(
    __global float* param,
    __global const float* gradient,
    __global float* accumGrad,
    __global float* accumUpdate,
    const float rho,
    const float epsilon,
    const float weightDecay,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    float grad = gradient[idx];
    if (weightDecay > 0.0f) {
        grad += weightDecay * param[idx];
    }

    // Update accumulated gradient
    float ag = rho * accumGrad[idx] + (1.0f - rho) * grad * grad;
    accumGrad[idx] = ag;

    // Compute update using accumulated update (from previous step)
    float rmsUpdate = sqrt(accumUpdate[idx] + epsilon);
    float rmsGrad = sqrt(ag + epsilon);
    float update = (rmsUpdate / rmsGrad) * grad;

    // Update accumulated update
    accumUpdate[idx] = rho * accumUpdate[idx] + (1.0f - rho) * update * update;

    // Update parameters
    param[idx] -= update;
}

// ---------------------------------------------------------------------------
// AMSGrad optimizer update
// Like Adam but uses max of v instead of v for stability
// Formula: m = beta1 * m + (1 - beta1) * grad
//          v = beta2 * v + (1 - beta2) * grad^2
//          v_hat = max(v_hat, v)  (element-wise max)
//          param = param - lr * m / (sqrt(v_hat) + epsilon)
// ---------------------------------------------------------------------------
__kernel void amsgrad_update(
    __global float* param,
    __global const float* gradient,
    __global float* m,
    __global float* v,
    __global float* vMax,
    const float learningRate,
    const float beta1,
    const float beta2,
    const float epsilon,
    const float weightDecay,
    const int step,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    float grad = gradient[idx];
    if (weightDecay > 0.0f) {
        grad += weightDecay * param[idx];
    }

    // Update biased first moment estimate
    float mVal = beta1 * m[idx] + (1.0f - beta1) * grad;
    m[idx] = mVal;

    // Update biased second moment estimate
    float vVal = beta2 * v[idx] + (1.0f - beta2) * grad * grad;
    v[idx] = vVal;

    // Compute max of v_hat (AMSGrad modification)
    float vMaxVal = fmax(vMax[idx], vVal);
    vMax[idx] = vMaxVal;

    // Bias correction.
    // Guard against step==0 which causes division by zero
    int safe_step = step < 1 ? 1 : step;
    float mHat = mVal / (1.0f - pow(beta1, (float)safe_step));
    float vMaxHat = vMaxVal / (1.0f - pow(beta2, (float)safe_step));

    // Update parameters using v_max instead of v
    param[idx] -= learningRate * mHat / (sqrt(vMaxHat) + epsilon);
}

// ---------------------------------------------------------------------------
// AdaMax optimizer update
// Variant of Adam using infinity norm instead of L2 norm
// Formula: m = beta1 * m + (1 - beta1) * grad
//          u = max(beta2 * u, |grad|)
//          param = param - lr / (1 - beta1^t) * m / u
// ---------------------------------------------------------------------------
__kernel void adamax_update(
    __global float* param,
    __global const float* gradient,
    __global float* m,
    __global float* u,
    const float learningRate,
    const float beta1,
    const float beta2,
    const float epsilon,
    const float weightDecay,
    const int step,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    float grad = gradient[idx];
    if (weightDecay > 0.0f) {
        grad += weightDecay * param[idx];
    }

    // Update biased first moment estimate
    float mVal = beta1 * m[idx] + (1.0f - beta1) * grad;
    m[idx] = mVal;

    // Update infinity norm (element-wise max)
    float uVal = fmax(beta2 * u[idx], fabs(grad));
    u[idx] = uVal;

    // Bias correction for learning rate - guard against step==0 which causes division by zero
    int safe_step = step < 1 ? 1 : step;
    float biasCorrection = 1.0f - pow(beta1, (float)safe_step);

    // Update parameters
    param[idx] -= (learningRate / biasCorrection) * mVal / (uVal + epsilon);
}

// ---------------------------------------------------------------------------
// Lion optimizer update (Evolved Sign Momentum)
// Formula: update = sign(beta1 * m + (1 - beta1) * grad)
//          m = beta2 * m + (1 - beta2) * grad
//          param = param - lr * (update + weight_decay * param)
// ---------------------------------------------------------------------------
__kernel void lion_update(
    __global float* param,
    __global const float* gradient,
    __global float* m,
    const float learningRate,
    const float beta1,
    const float beta2,
    const float weightDecay,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    float grad = gradient[idx];
    float mVal = m[idx];

    // Compute update direction (interpolation + sign)
    float interp = beta1 * mVal + (1.0f - beta1) * grad;
    float update = (interp > 0.0f) ? 1.0f : ((interp < 0.0f) ? -1.0f : 0.0f);

    // Update momentum (uses different beta than update computation)
    m[idx] = beta2 * mVal + (1.0f - beta2) * grad;

    // Apply weight decay (decoupled, like AdamW)
    if (weightDecay > 0.0f) {
        update += weightDecay * param[idx];
    }

    // Update parameters
    param[idx] -= learningRate * update;
}

// ---------------------------------------------------------------------------
// Nadam optimizer update (Nesterov-accelerated Adam)
// Combines Adam with Nesterov momentum for lookahead
// ---------------------------------------------------------------------------
__kernel void nadam_update(
    __global float* param,
    __global const float* gradient,
    __global float* m,
    __global float* v,
    const float learningRate,
    const float beta1,
    const float beta2,
    const float epsilon,
    const float weightDecay,
    const int step,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    float grad = gradient[idx];
    if (weightDecay > 0.0f) {
        grad += weightDecay * param[idx];
    }

    // Update biased first moment estimate
    float mVal = beta1 * m[idx] + (1.0f - beta1) * grad;
    m[idx] = mVal;

    // Update biased second moment estimate
    float vVal = beta2 * v[idx] + (1.0f - beta2) * grad * grad;
    v[idx] = vVal;

    // Bias corrections - guard against step==0 which causes division by zero
    int safe_step = step < 1 ? 1 : step;
    float beta1Pow = pow(beta1, (float)safe_step);
    float beta1PowNext = pow(beta1, (float)(safe_step + 1));
    float beta2Pow = pow(beta2, (float)safe_step);
    float mHat = mVal / (1.0f - beta1Pow);
    float vHat = vVal / (1.0f - beta2Pow);

    // Nesterov lookahead: use next step's momentum estimate
    float mNesterov = beta1 * mHat + (1.0f - beta1) * grad / (1.0f - beta1PowNext);

    // Update parameters
    param[idx] -= learningRate * mNesterov / (sqrt(vHat) + epsilon);
}

// ---------------------------------------------------------------------------
// FTRL (Follow The Regularized Leader) optimizer update
// Used for sparse data with L1/L2 regularization
// ---------------------------------------------------------------------------
__kernel void ftrl_update(
    __global float* param,
    __global const float* gradient,
    __global float* z,
    __global float* n,
    const float learningRate,
    const float l1Reg,
    const float l2Reg,
    const float beta,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    float grad = gradient[idx];
    float nVal = n[idx];
    float zVal = z[idx];
    float pVal = param[idx];

    // Update n (accumulated squared gradients)
    float nNew = nVal + grad * grad;
    n[idx] = nNew;

    // Compute sigma = (sqrt(n_new) - sqrt(n)) / lr
    float sigma = (sqrt(nNew) - sqrt(nVal)) / learningRate;

    // Update z
    zVal = zVal + grad - sigma * pVal;
    z[idx] = zVal;

    // Compute new param with L1 regularization (soft threshold)
    float zSign = (zVal > 0.0f) ? 1.0f : ((zVal < 0.0f) ? -1.0f : 0.0f);
    float zAbs = fabs(zVal);

    if (zAbs <= l1Reg) {
        param[idx] = 0.0f;
    } else {
        float denom = (beta + sqrt(nNew)) / learningRate + l2Reg;
        param[idx] = -zSign * (zAbs - l1Reg) / denom;
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
            "adam_bf16_update",
            "adamw_bf16_update",
            "adam8bit_update",
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
            "ftrl_update"
        };
    }
}
