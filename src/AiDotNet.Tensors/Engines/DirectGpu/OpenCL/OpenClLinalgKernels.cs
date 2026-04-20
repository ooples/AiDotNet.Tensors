// Copyright (c) AiDotNet. All rights reserved.
// OpenCL C kernels for the torch.linalg decomposition surface.
// One work-group per batch slice; work-items cooperate via barriers + local memory.

#if !NET462
namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL;

public static class OpenClLinalgKernels
{
    public static string[] GetKernelNames() => new[]
    {
        "parity211_cholesky",
        "parity211_lu_factor",
        "parity211_qr_reduced",
        "parity211_eigh",
    };

    public static string GetSource() => @"
// ═════════════════════════════════════════════════════════════════════════
// CHOLESKY
// ═════════════════════════════════════════════════════════════════════════
__kernel void parity211_cholesky(
    __global const float* A,
    __global float* L,
    __global int* info,
    const int batchCount, const int n, const int upper)
{
    const int b = get_group_id(0);
    if (b >= batchCount) return;
    const int tid = get_local_id(0);
    const int blockSize = get_local_size(0);

    __global const float* Ab = A + (long)b * n * n;
    __global float* Lb = L + (long)b * n * n;

    __local int localInfo;
    if (tid == 0) localInfo = 0;
    for (int idx = tid; idx < n * n; idx += blockSize) Lb[idx] = Ab[idx];
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    for (int j = 0; j < n; ++j) {
        if (tid == 0) {
            float diag = Lb[j * n + j];
            for (int k = 0; k < j; ++k) {
                float l_jk = upper ? Lb[k * n + j] : Lb[j * n + k];
                diag -= l_jk * l_jk;
            }
            if (diag <= 0.0f) { localInfo = j + 1; Lb[j * n + j] = 0.0f; }
            else Lb[j * n + j] = sqrt(diag);
        }
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
        if (localInfo != 0) break;

        float djj = Lb[j * n + j];
        if (djj == 0.0f) break;

        for (int i = j + 1 + tid; i < n; i += blockSize) {
            float sum = upper ? Lb[j * n + i] : Lb[i * n + j];
            for (int k = 0; k < j; ++k) {
                float l_ik = upper ? Lb[k * n + i] : Lb[i * n + k];
                float l_jk = upper ? Lb[k * n + j] : Lb[j * n + k];
                sum -= l_ik * l_jk;
            }
            if (upper) Lb[j * n + i] = sum / djj;
            else       Lb[i * n + j] = sum / djj;
        }
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    }
    for (int idx = tid; idx < n * n; idx += blockSize) {
        int i = idx / n, j = idx % n;
        int shouldZero = upper ? (i > j) : (i < j);
        if (shouldZero) Lb[idx] = 0.0f;
    }
    if (tid == 0) info[b] = localInfo;
}

// ═════════════════════════════════════════════════════════════════════════
// LU FACTOR
// ═════════════════════════════════════════════════════════════════════════
__kernel void parity211_lu_factor(
    __global const float* A,
    __global float* LU,
    __global int* pivots,
    const int batchCount, const int m, const int n)
{
    const int b = get_group_id(0);
    if (b >= batchCount) return;
    const int tid = get_local_id(0);
    const int blockSize = get_local_size(0);
    const int k = min(m, n);

    __global const float* Ab = A + (long)b * m * n;
    __global float* LUb = LU + (long)b * m * n;
    __global int* pivB = pivots + (long)b * k;

    __local int pivRow;

    for (int idx = tid; idx < m * n; idx += blockSize) LUb[idx] = Ab[idx];
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    for (int j = 0; j < k; ++j) {
        if (tid == 0) {
            int best = j;
            float bestVal = fabs(LUb[j * n + j]);
            for (int i = j + 1; i < m; ++i) {
                float v = fabs(LUb[i * n + j]);
                if (v > bestVal) { bestVal = v; best = i; }
            }
            pivRow = best;
            pivB[j] = best;
        }
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

        int p = pivRow;
        if (p != j) {
            for (int c = tid; c < n; c += blockSize) {
                float t = LUb[p * n + c];
                LUb[p * n + c] = LUb[j * n + c];
                LUb[j * n + c] = t;
            }
            barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
        }
        float pivVal = LUb[j * n + j];
        if (pivVal == 0.0f) continue;
        for (int i = j + 1 + tid; i < m; i += blockSize) LUb[i * n + j] /= pivVal;
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
        for (int idx = tid; idx < (m - j - 1) * (n - j - 1); idx += blockSize) {
            int di = idx / (n - j - 1);
            int dc = idx % (n - j - 1);
            int i = j + 1 + di;
            int c = j + 1 + dc;
            LUb[i * n + c] -= LUb[i * n + j] * LUb[j * n + c];
        }
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    }
}

// ═════════════════════════════════════════════════════════════════════════
// QR REDUCED — Modified Gram–Schmidt. See CudaLinalgKernels for the
// algorithm derivation; this mirrors it in OpenCL C.
// ═════════════════════════════════════════════════════════════════════════
__kernel void parity211_qr_reduced(
    __global const float* A,
    __global float* Q,
    __global float* R,
    const int batchCount, const int m, const int n)
{
    const int b = get_group_id(0);
    if (b >= batchCount) return;
    const int tid = get_local_id(0);
    const int blockSize = get_local_size(0);
    const int k = min(m, n);

    __global const float* Ab = A + (long)b * m * n;
    __global float* Qb = Q + (long)b * m * k;
    __global float* Rb = R + (long)b * k * n;

    __local float partials[1024];
    __local float sScalar;

    // Zero R.
    for (int idx = tid; idx < k * n; idx += blockSize) Rb[idx] = 0.0f;
    barrier(CLK_GLOBAL_MEM_FENCE);

    // Stage 1: MGS on the first k columns.
    for (int j = 0; j < k; ++j) {
        for (int i = tid; i < m; i += blockSize) Qb[i * k + j] = Ab[i * n + j];
        barrier(CLK_GLOBAL_MEM_FENCE);

        for (int p = 0; p < j; ++p) {
            float partial = 0.0f;
            for (int i = tid; i < m; i += blockSize)
                partial += Qb[i * k + p] * Qb[i * k + j];
            partials[tid] = partial;
            barrier(CLK_LOCAL_MEM_FENCE);
            for (int s = blockSize / 2; s > 0; s >>= 1) {
                if (tid < s) partials[tid] += partials[tid + s];
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            if (tid == 0) { sScalar = partials[0]; Rb[p * n + j] = sScalar; }
            barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
            float rp = sScalar;
            for (int i = tid; i < m; i += blockSize)
                Qb[i * k + j] -= rp * Qb[i * k + p];
            barrier(CLK_GLOBAL_MEM_FENCE);
        }

        float partialN = 0.0f;
        for (int i = tid; i < m; i += blockSize) {
            float vi = Qb[i * k + j];
            partialN += vi * vi;
        }
        partials[tid] = partialN;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int s = blockSize / 2; s > 0; s >>= 1) {
            if (tid < s) partials[tid] += partials[tid + s];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (tid == 0) { sScalar = sqrt(partials[0]); Rb[j * n + j] = sScalar; }
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

        float norm = sScalar;
        if (norm > 1e-30f) {
            float invNorm = 1.0f / norm;
            for (int i = tid; i < m; i += blockSize) Qb[i * k + j] *= invNorm;
        } else {
            for (int i = tid; i < m; i += blockSize) Qb[i * k + j] = 0.0f;
        }
        barrier(CLK_GLOBAL_MEM_FENCE);
    }

    // Stage 2: fill R[:, c] for c >= k via Qᵀ · A[:, c].
    for (int c = k; c < n; ++c) {
        for (int p = 0; p < k; ++p) {
            float partial = 0.0f;
            for (int i = tid; i < m; i += blockSize)
                partial += Qb[i * k + p] * Ab[i * n + c];
            partials[tid] = partial;
            barrier(CLK_LOCAL_MEM_FENCE);
            for (int s = blockSize / 2; s > 0; s >>= 1) {
                if (tid < s) partials[tid] += partials[tid + s];
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            if (tid == 0) Rb[p * n + c] = partials[0];
            barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
        }
    }
}

// ═════════════════════════════════════════════════════════════════════════
// EIGH (symmetric cyclic Jacobi)
// ═════════════════════════════════════════════════════════════════════════
__kernel void parity211_eigh(
    __global const float* A,
    __global float* W,
    __global float* V,
    const int batchCount, const int n,
    __local float* shared)
{
    const int b = get_group_id(0);
    if (b >= batchCount) return;
    const int tid = get_local_id(0);
    const int blockSize = get_local_size(0);
    const float eps = 1e-7f;
    const int maxSweeps = 64;

    __global const float* Ab = A + (long)b * n * n;
    __global float* Wb = W + (long)b * n;
    __global float* Vb = V + (long)b * n * n;
    __local float* Ash = shared;
    __local float* Vsh = shared + n * n;

    __local float c_shared, s_shared;

    for (int idx = tid; idx < n * n; idx += blockSize) {
        int i = idx / n, j = idx % n;
        Ash[idx] = (i <= j) ? Ab[i * n + j] : Ab[j * n + i];
        Vsh[idx] = (i == j) ? 1.0f : 0.0f;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int sweep = 0; sweep < maxSweeps; ++sweep) {
        float offSum = 0.0f;
        for (int p = 0; p < n; ++p)
            for (int q = p + 1; q < n; ++q)
                offSum += Ash[p * n + q] * Ash[p * n + q];
        if (sqrt(offSum) < eps) break;

        for (int p = 0; p < n - 1; ++p) {
            for (int q = p + 1; q < n; ++q) {
                if (tid == 0) {
                    float apq = Ash[p * n + q];
                    if (fabs(apq) < eps) { c_shared = 1.0f; s_shared = 0.0f; }
                    else {
                        float app = Ash[p * n + p];
                        float aqq = Ash[q * n + q];
                        float theta = (aqq - app) / (2.0f * apq);
                        float t = (theta == 0.0f) ? 1.0f : copysign(1.0f, theta) / (fabs(theta) + sqrt(1.0f + theta * theta));
                        c_shared = 1.0f / sqrt(1.0f + t * t);
                        s_shared = t * c_shared;
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                if (s_shared == 0.0f) continue;
                float cc = c_shared, ss = s_shared;

                for (int i = tid; i < n; i += blockSize) {
                    float aip = Ash[i * n + p];
                    float aiq = Ash[i * n + q];
                    Ash[i * n + p] = cc * aip - ss * aiq;
                    Ash[i * n + q] = ss * aip + cc * aiq;
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                for (int j = tid; j < n; j += blockSize) {
                    float apj = Ash[p * n + j];
                    float aqj = Ash[q * n + j];
                    Ash[p * n + j] = cc * apj - ss * aqj;
                    Ash[q * n + j] = ss * apj + cc * aqj;
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                for (int i = tid; i < n; i += blockSize) {
                    float vip = Vsh[i * n + p];
                    float viq = Vsh[i * n + q];
                    Vsh[i * n + p] = cc * vip - ss * viq;
                    Vsh[i * n + q] = ss * vip + cc * viq;
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
        }
    }

    if (tid == 0) {
        for (int i = 0; i < n; ++i) Wb[i] = Ash[i * n + i];
        for (int i = 0; i < n - 1; ++i) {
            int minIdx = i;
            for (int j = i + 1; j < n; ++j) if (Wb[j] < Wb[minIdx]) minIdx = j;
            if (minIdx != i) {
                float tmp = Wb[i]; Wb[i] = Wb[minIdx]; Wb[minIdx] = tmp;
                for (int r = 0; r < n; ++r) {
                    float v = Vsh[r * n + i]; Vsh[r * n + i] = Vsh[r * n + minIdx]; Vsh[r * n + minIdx] = v;
                }
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    for (int idx = tid; idx < n * n; idx += blockSize) Vb[idx] = Vsh[idx];
}
";
}
#endif
