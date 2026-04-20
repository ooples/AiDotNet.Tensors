// Copyright (c) AiDotNet. All rights reserved.
// Metal Shading Language kernels for the torch.linalg decomposition surface.
// One threadgroup per batch slice; threads cooperate via threadgroup memory
// and threadgroup_barrier.

namespace AiDotNet.Tensors.Engines.DirectGpu.Metal
{
    internal static class MetalLinalgKernels
    {
        public static string[] GetKernelNames() => new[]
        {
            "parity211_cholesky",
            "parity211_lu_factor",
            "parity211_qr_reduced",
            "parity211_eigh",
        };

        public const string Source = @"
#include <metal_stdlib>
#include <metal_math>
using namespace metal;

// ═════════════════════════════════════════════════════════════════════════
// CHOLESKY
// ═════════════════════════════════════════════════════════════════════════
kernel void parity211_cholesky(
    device const float* A [[buffer(0)]],
    device float* L [[buffer(1)]],
    device int* info [[buffer(2)]],
    constant int& batchCount [[buffer(3)]],
    constant int& n [[buffer(4)]],
    constant int& upper [[buffer(5)]],
    uint b [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint blockSize [[threads_per_threadgroup]],
    threadgroup int& localInfo [[threadgroup(0)]])
{
    if ((int)b >= batchCount) return;
    device const float* Ab = A + b * n * n;
    device float* Lb = L + b * n * n;

    for (int idx = tid; idx < n * n; idx += blockSize) Lb[idx] = Ab[idx];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) localInfo = 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);

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
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (localInfo != 0) break;

        float djj = Lb[j * n + j];
        if (djj == 0.0f) break;

        for (int i = j + 1 + (int)tid; i < n; i += blockSize) {
            float sum = upper ? Lb[j * n + i] : Lb[i * n + j];
            for (int k = 0; k < j; ++k) {
                float l_ik = upper ? Lb[k * n + i] : Lb[i * n + k];
                float l_jk = upper ? Lb[k * n + j] : Lb[j * n + k];
                sum -= l_ik * l_jk;
            }
            if (upper) Lb[j * n + i] = sum / djj;
            else       Lb[i * n + j] = sum / djj;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (int idx = tid; idx < n * n; idx += blockSize) {
        int i = idx / n, j = idx % n;
        bool shouldZero = upper ? (i > j) : (i < j);
        if (shouldZero) Lb[idx] = 0.0f;
    }
    if (tid == 0) info[b] = localInfo;
}

// ═════════════════════════════════════════════════════════════════════════
// LU FACTORIZATION
// ═════════════════════════════════════════════════════════════════════════
kernel void parity211_lu_factor(
    device const float* A [[buffer(0)]],
    device float* LU [[buffer(1)]],
    device int* pivots [[buffer(2)]],
    constant int& batchCount [[buffer(3)]],
    constant int& m [[buffer(4)]],
    constant int& n [[buffer(5)]],
    uint b [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint blockSize [[threads_per_threadgroup]],
    threadgroup int& pivRow [[threadgroup(0)]])
{
    if ((int)b >= batchCount) return;
    int k = min(m, n);
    device const float* Ab = A + b * m * n;
    device float* LUb = LU + b * m * n;
    device int* pivB = pivots + b * k;

    for (int idx = tid; idx < m * n; idx += blockSize) LUb[idx] = Ab[idx];
    threadgroup_barrier(mem_flags::mem_threadgroup);

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
        threadgroup_barrier(mem_flags::mem_threadgroup);

        int p = pivRow;
        if (p != j) {
            for (int c = tid; c < n; c += blockSize) {
                float t = LUb[p * n + c];
                LUb[p * n + c] = LUb[j * n + c];
                LUb[j * n + c] = t;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float pivVal = LUb[j * n + j];
        if (pivVal == 0.0f) continue;
        for (int i = j + 1 + (int)tid; i < m; i += blockSize) LUb[i * n + j] /= pivVal;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int idx = tid; idx < (m - j - 1) * (n - j - 1); idx += blockSize) {
            int di = idx / (n - j - 1);
            int dc = idx % (n - j - 1);
            int i = j + 1 + di;
            int c = j + 1 + dc;
            LUb[i * n + c] -= LUb[i * n + j] * LUb[j * n + c];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// ═════════════════════════════════════════════════════════════════════════
// QR FACTORIZATION (Modified Gram–Schmidt, reduced). See CudaLinalgKernels
// for the algorithm derivation; this mirrors it in Metal Shading Language.
// ═════════════════════════════════════════════════════════════════════════
kernel void parity211_qr_reduced(
    device const float* A [[buffer(0)]],
    device float* Q [[buffer(1)]],
    device float* R [[buffer(2)]],
    constant int& batchCount [[buffer(3)]],
    constant int& m [[buffer(4)]],
    constant int& n [[buffer(5)]],
    uint b [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint blockSize [[threads_per_threadgroup]],
    threadgroup float& sScalar [[threadgroup(0)]])
{
    if ((int)b >= batchCount) return;
    int k = min(m, n);
    device const float* Ab = A + b * m * n;
    device float* Qb = Q + b * m * k;
    device float* Rb = R + b * k * n;

    threadgroup float partials[1024];

    // Zero R.
    for (int idx = tid; idx < k * n; idx += blockSize) Rb[idx] = 0.0f;
    threadgroup_barrier(mem_flags::mem_device);

    // Stage 1: MGS on columns 0..k-1.
    for (int j = 0; j < k; ++j) {
        for (int i = tid; i < m; i += blockSize) Qb[i * k + j] = Ab[i * n + j];
        threadgroup_barrier(mem_flags::mem_device);

        for (int p = 0; p < j; ++p) {
            float partial = 0.0f;
            for (int i = tid; i < m; i += blockSize)
                partial += Qb[i * k + p] * Qb[i * k + j];
            partials[tid] = partial;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (int s = (int)blockSize / 2; s > 0; s >>= 1) {
                if ((int)tid < s) partials[tid] += partials[tid + s];
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (tid == 0) { sScalar = partials[0]; Rb[p * n + j] = sScalar; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float rp = sScalar;
            for (int i = tid; i < m; i += blockSize)
                Qb[i * k + j] -= rp * Qb[i * k + p];
            threadgroup_barrier(mem_flags::mem_device);
        }

        float partialN = 0.0f;
        for (int i = tid; i < m; i += blockSize) {
            float vi = Qb[i * k + j];
            partialN += vi * vi;
        }
        partials[tid] = partialN;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int s = (int)blockSize / 2; s > 0; s >>= 1) {
            if ((int)tid < s) partials[tid] += partials[tid + s];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid == 0) { sScalar = sqrt(partials[0]); Rb[j * n + j] = sScalar; }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float norm = sScalar;
        if (norm > 1e-30f) {
            float invNorm = 1.0f / norm;
            for (int i = tid; i < m; i += blockSize) Qb[i * k + j] *= invNorm;
        } else {
            for (int i = tid; i < m; i += blockSize) Qb[i * k + j] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_device);
    }

    // Stage 2: fill R[:, c] for c >= k via Qᵀ · A[:, c].
    for (int c = k; c < n; ++c) {
        for (int p = 0; p < k; ++p) {
            float partial = 0.0f;
            for (int i = tid; i < m; i += blockSize)
                partial += Qb[i * k + p] * Ab[i * n + c];
            partials[tid] = partial;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (int s = (int)blockSize / 2; s > 0; s >>= 1) {
                if ((int)tid < s) partials[tid] += partials[tid + s];
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (tid == 0) Rb[p * n + c] = partials[0];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

// ═════════════════════════════════════════════════════════════════════════
// EIGH (symmetric cyclic Jacobi)
// ═════════════════════════════════════════════════════════════════════════
kernel void parity211_eigh(
    device const float* A [[buffer(0)]],
    device float* W [[buffer(1)]],
    device float* V [[buffer(2)]],
    constant int& batchCount [[buffer(3)]],
    constant int& n [[buffer(4)]],
    uint b [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint blockSize [[threads_per_threadgroup]],
    threadgroup float* shared [[threadgroup(0)]])
{
    if ((int)b >= batchCount) return;
    const float eps = 1e-7f;
    const int maxSweeps = 64;
    device const float* Ab = A + b * n * n;
    device float* Wb = W + b * n;
    device float* Vb = V + b * n * n;
    threadgroup float* Ash = shared;
    threadgroup float* Vsh = shared + n * n;

    for (int idx = tid; idx < n * n; idx += blockSize) {
        int i = idx / n, j = idx % n;
        Ash[idx] = (i <= j) ? Ab[i * n + j] : Ab[j * n + i];
        Vsh[idx] = (i == j) ? 1.0f : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float c, s;
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
                    if (fabs(apq) < eps) { c = 1.0f; s = 0.0f; }
                    else {
                        float app = Ash[p * n + p];
                        float aqq = Ash[q * n + q];
                        float theta = (aqq - app) / (2.0f * apq);
                        float t = (theta == 0.0f) ? 1.0f : copysign(1.0f, theta) / (fabs(theta) + sqrt(1.0f + theta * theta));
                        c = 1.0f / sqrt(1.0f + t * t);
                        s = t * c;
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (s == 0.0f) continue;

                for (int i = tid; i < n; i += blockSize) {
                    float aip = Ash[i * n + p];
                    float aiq = Ash[i * n + q];
                    Ash[i * n + p] = c * aip - s * aiq;
                    Ash[i * n + q] = s * aip + c * aiq;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (int j = tid; j < n; j += blockSize) {
                    float apj = Ash[p * n + j];
                    float aqj = Ash[q * n + j];
                    Ash[p * n + j] = c * apj - s * aqj;
                    Ash[q * n + j] = s * apj + c * aqj;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (int i = tid; i < n; i += blockSize) {
                    float vip = Vsh[i * n + p];
                    float viq = Vsh[i * n + q];
                    Vsh[i * n + p] = c * vip - s * viq;
                    Vsh[i * n + q] = s * vip + c * viq;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
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
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int idx = tid; idx < n * n; idx += blockSize) Vb[idx] = Vsh[idx];
}
";
    }
}
