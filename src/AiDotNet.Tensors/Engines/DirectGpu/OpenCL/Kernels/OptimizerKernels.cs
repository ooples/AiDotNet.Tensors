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
    float update = learningRate * m_hat / (sqrt(v_hat) + epsilon);
    if (weightDecay > 0.0f) {
        update += learningRate * weightDecay * param[idx];
    }
    param[idx] -= update;
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

    // Bias correction for m only (AMSGrad uses raw v_max)
    // Guard against step==0 which causes division by zero
    int safe_step = step < 1 ? 1 : step;
    float mHat = mVal / (1.0f - pow(beta1, (float)safe_step));

    // Update parameters using v_max instead of v
    param[idx] -= learningRate * mHat / (sqrt(vMaxVal) + epsilon);
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
    float beta2Pow = pow(beta2, (float)safe_step);
    float mHat = mVal / (1.0f - beta1Pow);
    float vHat = vVal / (1.0f - beta2Pow);

    // Nesterov lookahead: use next step's momentum estimate
    float mNesterov = beta1 * mHat + (1.0f - beta1) * grad / (1.0f - beta1Pow);

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

// ---------------------------------------------------------------------------
// Vanilla SGD update (no momentum)
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
// Proximal gradient (ISTA) update with L1 soft-thresholding
// Formula: tmp = param - lr * grad
//          param = sign(tmp) * max(abs(tmp) - lr * l1Strength, 0)
// ---------------------------------------------------------------------------
__kernel void proximal_l1_update(
    __global float* param,
    __global const float* gradient,
    const float learningRate,
    const float l1Strength,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    float tmp = param[idx] - learningRate * gradient[idx];
    float mag = fabs(tmp) - (learningRate * l1Strength);
    if (mag <= 0.0f) {
        param[idx] = 0.0f;
    } else {
        float signTmp = (tmp > 0.0f) ? 1.0f : -1.0f;
        param[idx] = signTmp * mag;
    }
}

inline int decode_sparse_index(float raw, int param_size)
{
    int bitcast = as_int(raw);
    if (bitcast >= 0 && bitcast < param_size) return bitcast;
    // Validate in the FLOAT domain before the (int) cast — (int)raw is undefined for NaN/Inf or
    // out-of-int-range values on the device. Only convert once raw is finite, non-negative, in range
    // and integral.
    if (isfinite(raw) && raw >= 0.0f && raw < (float)param_size && raw == trunc(raw)) return (int)raw;
    return -1;
}

__kernel void sparse_sgd_update(
    __global float* param, __global const float* indices, __global const float* values,
    const float learningRate, const float weightDecay, const int nnz, const int param_size)
{
    const int k = get_global_id(0);
    if (k >= nnz) return;
    int i = decode_sparse_index(indices[k], param_size); if (i < 0) return;
    float grad = values[k];
    if (weightDecay > 0.0f) grad += weightDecay * param[i];
    param[i] -= learningRate * grad;
}

__kernel void sparse_sgd_momentum_update(
    __global float* param, __global const float* indices, __global const float* values,
    __global float* velocity,
    const float learningRate, const float momentum, const float weightDecay, const int nnz, const int param_size)
{
    const int k = get_global_id(0);
    if (k >= nnz) return;
    int i = decode_sparse_index(indices[k], param_size); if (i < 0) return;
    float grad = values[k];
    if (weightDecay > 0.0f) grad += weightDecay * param[i];
    float v = momentum * velocity[i] + grad;
    velocity[i] = v;
    param[i] -= learningRate * v;
}

__kernel void sparse_adam_update(
    __global float* param, __global const float* indices, __global const float* values,
    __global float* m, __global float* v,
    const float learningRate, const float beta1, const float beta2, const float epsilon,
    const float weightDecay, const int step, const int nnz, const int param_size)
{
    const int k = get_global_id(0);
    if (k >= nnz) return;
    int i = decode_sparse_index(indices[k], param_size); if (i < 0) return;
    float grad = values[k];
    float mVal = beta1 * m[i] + (1.0f - beta1) * grad;
    float vVal = beta2 * v[i] + (1.0f - beta2) * grad * grad;
    m[i] = mVal;
    v[i] = vVal;
    float mHat = mVal / (1.0f - pow(beta1, (float)step));
    float vHat = vVal / (1.0f - pow(beta2, (float)step));
    float update = learningRate * mHat / (sqrt(vHat) + epsilon);
    if (weightDecay > 0.0f) update += learningRate * weightDecay * param[i];
    param[i] -= update;
}

__kernel void sparse_adamw_update(
    __global float* param, __global const float* indices, __global const float* values,
    __global float* m, __global float* v,
    const float learningRate, const float beta1, const float beta2, const float epsilon,
    const float weightDecay, const int step, const int nnz, const int param_size)
{
    const int k = get_global_id(0);
    if (k >= nnz) return;
    int i = decode_sparse_index(indices[k], param_size); if (i < 0) return;
    float grad = values[k];
    float p = param[i];
    if (weightDecay > 0.0f) p -= learningRate * weightDecay * p;
    float mVal = beta1 * m[i] + (1.0f - beta1) * grad;
    float vVal = beta2 * v[i] + (1.0f - beta2) * grad * grad;
    m[i] = mVal;
    v[i] = vVal;
    float mHat = mVal / (1.0f - pow(beta1, (float)step));
    float vHat = vVal / (1.0f - pow(beta2, (float)step));
    param[i] = p - learningRate * mHat / (sqrt(vHat) + epsilon);
}

__kernel void sparse_rmsprop_update(
    __global float* param, __global const float* indices, __global const float* values,
    __global float* squaredAvg,
    const float learningRate, const float rho, const float epsilon, const float weightDecay, const int nnz, const int param_size)
{
    const int k = get_global_id(0);
    if (k >= nnz) return;
    int i = decode_sparse_index(indices[k], param_size); if (i < 0) return;
    float grad = values[k];
    if (weightDecay > 0.0f) grad += weightDecay * param[i];
    float sq = rho * squaredAvg[i] + (1.0f - rho) * grad * grad;
    squaredAvg[i] = sq;
    param[i] -= learningRate * grad / (sqrt(sq) + epsilon);
}

__kernel void sparse_adagrad_update(
    __global float* param, __global const float* indices, __global const float* values,
    __global float* accum,
    const float learningRate, const float epsilon, const float weightDecay, const int nnz, const int param_size)
{
    const int k = get_global_id(0);
    if (k >= nnz) return;
    int i = decode_sparse_index(indices[k], param_size); if (i < 0) return;
    float grad = values[k];
    if (weightDecay > 0.0f) grad += weightDecay * param[i];
    float a = accum[i] + grad * grad;
    accum[i] = a;
    param[i] -= learningRate * grad / (sqrt(a) + epsilon);
}

__kernel void sparse_nag_update(
    __global float* param, __global const float* indices, __global const float* values,
    __global float* velocity,
    const float learningRate, const float momentum, const float weightDecay, const int nnz, const int param_size)
{
    const int k = get_global_id(0);
    if (k >= nnz) return;
    int i = decode_sparse_index(indices[k], param_size); if (i < 0) return;
    float grad = values[k];
    if (weightDecay > 0.0f) grad += weightDecay * param[i];
    float vOld = velocity[i];
    float vNew = momentum * vOld + grad;
    velocity[i] = vNew;
    param[i] -= learningRate * ((1.0f + momentum) * vNew - momentum * vOld);
}

__kernel void sparse_adadelta_update(
    __global float* param, __global const float* indices, __global const float* values,
    __global float* accumGrad, __global float* accumUpdate,
    const float rho, const float epsilon, const float weightDecay, const int nnz, const int param_size)
{
    const int k = get_global_id(0);
    if (k >= nnz) return;
    int i = decode_sparse_index(indices[k], param_size); if (i < 0) return;
    float grad = values[k];
    if (weightDecay > 0.0f) grad += weightDecay * param[i];
    float oneMinusRho = 1.0f - rho;
    float ag = rho * accumGrad[i] + oneMinusRho * grad * grad;
    accumGrad[i] = ag;
    float dx = sqrt(accumUpdate[i] + epsilon) / sqrt(ag + epsilon) * grad;
    accumUpdate[i] = rho * accumUpdate[i] + oneMinusRho * dx * dx;
    param[i] -= dx;
}

__kernel void sparse_amsgrad_update(
    __global float* param, __global const float* indices, __global const float* values,
    __global float* m, __global float* v, __global float* vMax,
    const float learningRate, const float beta1, const float beta2, const float epsilon,
    const float weightDecay, const int step, const int nnz, const int param_size)
{
    const int k = get_global_id(0);
    if (k >= nnz) return;
    int i = decode_sparse_index(indices[k], param_size); if (i < 0) return;
    float grad = values[k];
    if (weightDecay > 0.0f) grad += weightDecay * param[i];
    float mVal = beta1 * m[i] + (1.0f - beta1) * grad;
    float vVal = beta2 * v[i] + (1.0f - beta2) * grad * grad;
    m[i] = mVal;
    v[i] = vVal;
    float vMaxNew = fmax(vMax[i], vVal);
    vMax[i] = vMaxNew;
    float mHat = mVal / (1.0f - pow(beta1, (float)step));
    float vHat = vMaxNew / (1.0f - pow(beta2, (float)step));
    param[i] -= learningRate * mHat / (sqrt(vHat) + epsilon);
}

__kernel void sparse_adamax_update(
    __global float* param, __global const float* indices, __global const float* values,
    __global float* m, __global float* u,
    const float learningRate, const float beta1, const float beta2, const float epsilon,
    const float weightDecay, const int step, const int nnz, const int param_size)
{
    const int k = get_global_id(0);
    if (k >= nnz) return;
    int i = decode_sparse_index(indices[k], param_size); if (i < 0) return;
    float grad = values[k];
    if (weightDecay > 0.0f) grad += weightDecay * param[i];
    float mVal = beta1 * m[i] + (1.0f - beta1) * grad;
    float uVal = fmax(beta2 * u[i], fabs(grad));
    m[i] = mVal;
    u[i] = uVal;
    float lrAdj = learningRate / (1.0f - pow(beta1, (float)step));
    param[i] -= lrAdj * mVal / (uVal + epsilon);
}

__kernel void sparse_lion_update(
    __global float* param, __global const float* indices, __global const float* values,
    __global float* m,
    const float learningRate, const float beta1, const float beta2, const float weightDecay, const int nnz, const int param_size)
{
    const int k = get_global_id(0);
    if (k >= nnz) return;
    int i = decode_sparse_index(indices[k], param_size); if (i < 0) return;
    float grad = values[k];
    float c = beta1 * m[i] + (1.0f - beta1) * grad;
    float update = (c > 0.0f) ? 1.0f : ((c < 0.0f) ? -1.0f : 0.0f);
    if (weightDecay > 0.0f) update += weightDecay * param[i];
    param[i] -= learningRate * update;
    m[i] = beta2 * m[i] + (1.0f - beta2) * grad;
}

__kernel void sparse_nadam_update(
    __global float* param, __global const float* indices, __global const float* values,
    __global float* m, __global float* v,
    const float learningRate, const float beta1, const float beta2, const float epsilon,
    const float weightDecay, const int step, const int nnz, const int param_size)
{
    const int k = get_global_id(0);
    if (k >= nnz) return;
    int i = decode_sparse_index(indices[k], param_size); if (i < 0) return;
    float grad = values[k];
    if (weightDecay > 0.0f) grad += weightDecay * param[i];
    float mVal = beta1 * m[i] + (1.0f - beta1) * grad;
    float vVal = beta2 * v[i] + (1.0f - beta2) * grad * grad;
    m[i] = mVal;
    v[i] = vVal;
    float bc1 = 1.0f - pow(beta1, (float)step);
    float bc2 = 1.0f - pow(beta2, (float)step);
    float mHat = (beta1 * mVal + (1.0f - beta1) * grad) / bc1;
    float vHat = vVal / bc2;
    param[i] -= learningRate * mHat / (sqrt(vHat) + epsilon);
}

__kernel void sparse_ftrl_update(
    __global float* param, __global const float* indices, __global const float* values,
    __global float* z, __global float* n,
    const float learningRate, const float l1Reg, const float l2Reg, const float beta, const int nnz, const int param_size)
{
    const int k = get_global_id(0);
    if (k >= nnz) return;
    int i = decode_sparse_index(indices[k], param_size); if (i < 0) return;
    float grad = values[k];
    float nOld = n[i];
    float nNew = nOld + grad * grad;
    n[i] = nNew;
    float sigma = (sqrt(nNew) - sqrt(nOld)) / learningRate;
    z[i] += grad - sigma * param[i];
    float zVal = z[i];
    if (fabs(zVal) <= l1Reg) {
        param[i] = 0.0f;
    } else {
        float sign = (zVal > 0.0f) ? 1.0f : -1.0f;
        param[i] = (sign * l1Reg - zVal) / ((beta + sqrt(nNew)) / learningRate + l2Reg);
    }
}

__kernel void sparse_proximal_l1_update(
    __global float* param, __global const float* indices, __global const float* values,
    const float learningRate, const float l1Strength, const int nnz, const int param_size)
{
    const int k = get_global_id(0);
    if (k >= nnz) return;
    int i = decode_sparse_index(indices[k], param_size); if (i < 0) return;
    float p = param[i] - learningRate * values[k];
    float threshold = learningRate * l1Strength;
    if (p > threshold)       param[i] = p - threshold;
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
            "sparse_proximal_l1_update"
        };
    }
}
