// Copyright (c) AiDotNet. All rights reserved.
// CUDA NVRTC source for the core torch.linalg decomposition kernels:
//   - parity211_cholesky  (SPD factorization)
//   - parity211_lu_factor (LU with partial pivoting)
//   - parity211_qr_reduced (Householder QR, reduced mode)
//   - parity211_eigh      (symmetric cyclic Jacobi)
//
// Each kernel processes one batch slice per CUDA thread block. Threads
// cooperate via __syncthreads() and shared memory to avoid cross-block
// coordination — this keeps the dispatch path simple at the cost of
// limiting n ≤ blockDim.x (typically n ≤ 1024). Larger problems fall
// back to the managed CPU path through DirectGpuTensorEngine.

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels
{
    internal static class CudaLinalgKernels
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
// CHOLESKY — A = L·Lᵀ (lower) or A = Uᵀ·U (upper).
// One thread block per batch; threads cooperate on per-column scaling and
// trailing-submatrix update. info[b] = 0 on success, k+1 on failure at
// leading minor of order k+1.
// ═════════════════════════════════════════════════════════════════════════
extern ""C"" __global__ void parity211_cholesky(
    const float* __restrict__ A,
    float* __restrict__ L,
    int* __restrict__ info,
    int batchCount, int n, int upper)
{
    int b = blockIdx.x;
    if (b >= batchCount) return;
    int tid = threadIdx.x;

    const float* Ab = A + b * n * n;
    float* Lb = L + b * n * n;

    // Copy A into L as working scratch.
    for (int idx = tid; idx < n * n; idx += blockDim.x)
        Lb[idx] = Ab[idx];
    __syncthreads();

    __shared__ int localInfo;
    if (tid == 0) localInfo = 0;
    __syncthreads();

    for (int j = 0; j < n; ++j) {
        // Diagonal: L[j,j] = sqrt(A[j,j] − Σ_{k<j} L[j,k]²). The diagonal
        // element is on the diagonal regardless of triangle, so no ternary.
        if (tid == 0) {
            float diag = Lb[j * n + j];
            for (int k = 0; k < j; ++k) {
                float l_jk = upper ? Lb[k * n + j] : Lb[j * n + k];
                diag -= l_jk * l_jk;
            }
            if (diag <= 0.0f) {
                localInfo = j + 1;
                Lb[j * n + j] = 0.0f;
            } else {
                Lb[j * n + j] = sqrtf(diag);
            }
        }
        __syncthreads();
        if (localInfo != 0) break;

        float djj = Lb[j * n + j];
        if (djj == 0.0f) break;

        // Below-diagonal column: L[i,j] = (A[i,j] − Σ_{k<j} L[i,k]·L[j,k]) / L[j,j].
        // Threads stride across row indices i ∈ (j, n).
        for (int i = j + 1 + tid; i < n; i += blockDim.x) {
            float sum = upper ? Lb[j * n + i] : Lb[i * n + j];
            for (int k = 0; k < j; ++k) {
                float l_ik = upper ? Lb[k * n + i] : Lb[i * n + k];
                float l_jk = upper ? Lb[k * n + j] : Lb[j * n + k];
                sum -= l_ik * l_jk;
            }
            if (upper) Lb[j * n + i] = sum / djj;
            else       Lb[i * n + j] = sum / djj;
        }
        __syncthreads();
    }

    // Zero out opposite triangle for a clean factor view.
    for (int idx = tid; idx < n * n; idx += blockDim.x) {
        int i = idx / n, j = idx % n;
        bool shouldZero = upper ? (i > j) : (i < j);
        if (shouldZero) Lb[idx] = 0.0f;
    }

    if (tid == 0) info[b] = localInfo;
}

// ═════════════════════════════════════════════════════════════════════════
// LU FACTORIZATION with partial pivoting. One block per batch. Packs L\U
// in the output buffer (unit-diagonal L implicit, U on + above diagonal).
// ═════════════════════════════════════════════════════════════════════════
extern ""C"" __global__ void parity211_lu_factor(
    const float* __restrict__ A,
    float* __restrict__ LU,
    int* __restrict__ pivots,
    int batchCount, int m, int n)
{
    int b = blockIdx.x;
    if (b >= batchCount) return;
    int tid = threadIdx.x;
    int k = min(m, n);

    const float* Ab = A + b * m * n;
    float* LUb = LU + b * m * n;
    int* pivB = pivots + b * k;

    // Copy A → LUb.
    for (int idx = tid; idx < m * n; idx += blockDim.x) LUb[idx] = Ab[idx];
    __syncthreads();

    __shared__ int pivRow;
    __shared__ float pivAbs;

    for (int j = 0; j < k; ++j) {
        // Pivot search: find row with max |A[i, j]| for i ≥ j.
        if (tid == 0) { pivRow = j; pivAbs = fabsf(LUb[j * n + j]); }
        __syncthreads();
        for (int i = j + 1 + tid; i < m; i += blockDim.x) {
            float v = fabsf(LUb[i * n + j]);
            if (v > pivAbs) {
                // Atomically update (simple serial update since pivot search is cheap).
                atomicExch(&pivRow, i);
            }
        }
        __syncthreads();
        // Serialized max refinement (rare contention in practice).
        if (tid == 0) {
            int best = j;
            float bestVal = fabsf(LUb[j * n + j]);
            for (int i = j + 1; i < m; ++i) {
                float v = fabsf(LUb[i * n + j]);
                if (v > bestVal) { bestVal = v; best = i; }
            }
            pivRow = best;
            pivB[j] = best;
        }
        __syncthreads();

        int p = pivRow;
        // Row swap.
        if (p != j) {
            for (int c = tid; c < n; c += blockDim.x) {
                float t = LUb[p * n + c];
                LUb[p * n + c] = LUb[j * n + c];
                LUb[j * n + c] = t;
            }
            __syncthreads();
        }

        // Skip update if pivot is zero (singular).
        float pivVal = LUb[j * n + j];
        if (pivVal == 0.0f) continue;

        // Scale column below pivot.
        for (int i = j + 1 + tid; i < m; i += blockDim.x)
            LUb[i * n + j] /= pivVal;
        __syncthreads();

        // Trailing submatrix update: A[i, c] -= L[i, j] * U[j, c] for i > j, c > j.
        for (int idx = tid; idx < (m - j - 1) * (n - j - 1); idx += blockDim.x) {
            int di = idx / (n - j - 1);
            int dc = idx % (n - j - 1);
            int i = j + 1 + di;
            int c = j + 1 + dc;
            LUb[i * n + c] -= LUb[i * n + j] * LUb[j * n + c];
        }
        __syncthreads();
    }
}

// ═════════════════════════════════════════════════════════════════════════
// QR FACTORIZATION (Modified Gram–Schmidt, reduced). One block per batch.
// Q is (m × k), R is (k × n), where k = min(m, n).
//
// Why MGS (not Householder)? The reflector variant needs an m×n working
// buffer to keep the trailing rows in sync after each reflector; MGS
// writes Q and R directly from the columns of A, with no hidden state.
// Stability is sufficient for the well-conditioned cases ML cares about
// (the managed CPU path still runs Householder for ill-conditioned
// inputs — this is the GPU fast-path only).
//
// Algorithm:
//   Stage 1 (columns 0..k-1):
//     v ← A[:, j]
//     for p = 0..j-1:
//         R[p,j] ← ⟨Q[:, p], v⟩
//         v      ← v − R[p,j] · Q[:, p]
//     R[j,j] ← ‖v‖
//     Q[:, j] ← v / R[j,j]
//     R[i,j] ← 0 for i>j  (upper triangular)
//   Stage 2 (columns k..n-1, only when n > k):
//     R[p,c] ← ⟨Q[:, p], A[:, c]⟩  for p = 0..k-1
// ═════════════════════════════════════════════════════════════════════════
extern ""C"" __global__ void parity211_qr_reduced(
    const float* __restrict__ A,
    float* __restrict__ Q,
    float* __restrict__ R,
    int batchCount, int m, int n)
{
    int b = blockIdx.x;
    if (b >= batchCount) return;
    int tid = threadIdx.x;
    int k = min(m, n);

    const float* Ab = A + b * m * n;
    float* Qb = Q + b * m * k;
    float* Rb = R + b * k * n;

    // Zero R up front — the strict lower triangle stays zero, and stage-2
    // off-diagonal cells above row k are written below.
    for (int idx = tid; idx < k * n; idx += blockDim.x) Rb[idx] = 0.0f;
    __syncthreads();

    // Block-wide reduction scratch.
    __shared__ float partials[1024];
    __shared__ float sScalar;

    // Stage 1: MGS on the first k columns.
    for (int j = 0; j < k; ++j) {
        // Load column j of A into Q[:, j] as the starting v vector.
        for (int i = tid; i < m; i += blockDim.x) Qb[i * k + j] = Ab[i * n + j];
        __syncthreads();

        // Orthogonalize v against each earlier Q column.
        for (int p = 0; p < j; ++p) {
            // dot = ⟨Q[:, p], v⟩ via block-wide reduction.
            float partial = 0.0f;
            for (int i = tid; i < m; i += blockDim.x)
                partial += Qb[i * k + p] * Qb[i * k + j];
            partials[tid] = partial;
            __syncthreads();
            for (int s = blockDim.x / 2; s > 0; s >>= 1) {
                if (tid < s) partials[tid] += partials[tid + s];
                __syncthreads();
            }
            if (tid == 0) { sScalar = partials[0]; Rb[p * n + j] = sScalar; }
            __syncthreads();
            // v ← v − dot · Q[:, p].
            float rp = sScalar;
            for (int i = tid; i < m; i += blockDim.x)
                Qb[i * k + j] -= rp * Qb[i * k + p];
            __syncthreads();
        }

        // R[j, j] = ‖v‖.
        float partialN = 0.0f;
        for (int i = tid; i < m; i += blockDim.x) {
            float vi = Qb[i * k + j];
            partialN += vi * vi;
        }
        partials[tid] = partialN;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) partials[tid] += partials[tid + s];
            __syncthreads();
        }
        if (tid == 0) { sScalar = sqrtf(partials[0]); Rb[j * n + j] = sScalar; }
        __syncthreads();

        // Q[:, j] = v / R[j, j]. If the column is rank-deficient (norm ≈ 0),
        // leave Q[:, j] zero — the consumer can detect via R[j, j].
        float norm = sScalar;
        if (norm > 1e-30f) {
            float invNorm = 1.0f / norm;
            for (int i = tid; i < m; i += blockDim.x) Qb[i * k + j] *= invNorm;
        } else {
            for (int i = tid; i < m; i += blockDim.x) Qb[i * k + j] = 0.0f;
        }
        __syncthreads();
    }

    // Stage 2: fill R[:, c] for c in [k, n) via Qᵀ · A[:, c].
    for (int c = k; c < n; ++c) {
        for (int p = 0; p < k; ++p) {
            float partial = 0.0f;
            for (int i = tid; i < m; i += blockDim.x)
                partial += Qb[i * k + p] * Ab[i * n + c];
            partials[tid] = partial;
            __syncthreads();
            for (int s = blockDim.x / 2; s > 0; s >>= 1) {
                if (tid < s) partials[tid] += partials[tid + s];
                __syncthreads();
            }
            if (tid == 0) Rb[p * n + c] = partials[0];
            __syncthreads();
        }
    }
}

// ═════════════════════════════════════════════════════════════════════════
// SYMMETRIC EIGENDECOMPOSITION (cyclic Jacobi). One block per batch.
// Writes eigenvalues (ascending) and corresponding eigenvectors as columns.
// ═════════════════════════════════════════════════════════════════════════
extern ""C"" __global__ void parity211_eigh(
    const float* __restrict__ A,
    float* __restrict__ W,
    float* __restrict__ V,
    int batchCount, int n)
{
    int b = blockIdx.x;
    if (b >= batchCount) return;
    int tid = threadIdx.x;
    const float eps = 1e-7f;
    const int maxSweeps = 64;

    const float* Ab = A + b * n * n;
    float* Wb = W + b * n;
    float* Vb = V + b * n * n;

    // Scratch for the working matrix — reuse V's slot so we don't need a
    // separate device allocation. Initialize V = I, then copy A to a shared
    // scratch region (per-block). For simplicity, use V's cells pair: the
    // first n*n is the working A, the second is V. Only viable if 2n*n fits
    // in V's buffer — we instead do it via a two-pass layout: copy A to V,
    // accumulate rotations into a local-shared V matrix, then write back.
    // For this v1 path, we apply Jacobi in-place to a separate working matrix
    // held in global memory by interpreting W as a stride into a 2n*n region.
    // To avoid that complexity, copy A into V directly, and use a temporary
    // eigenvector matrix accumulated per-thread then written to V.

    // For medium n (≤ 64), we can use shared memory for both the working A
    // and V. We'll target n ≤ 64 for the fast GPU path; larger falls back
    // to CPU. This is a design trade — GPU Jacobi scales poorly past ~128
    // anyway because the O(n³) inner sweep becomes compute-bound on one SM.
    //
    // Here we use the global V buffer as both the working matrix and the
    // output eigenvectors in an interleaved schedule: compute via V-as-A
    // first to convergence, then do an independent pass to orthogonalize V.
    // That's numerically inexact; instead do the conservative thing and
    // restrict to n ≤ 64 so shared memory holds both. This keeps the kernel
    // correct across its supported range.

    extern __shared__ float shared[];
    float* Ash = shared;              // size n*n
    float* Vsh = shared + n * n;      // size n*n

    for (int idx = tid; idx < n * n; idx += blockDim.x) {
        int i = idx / n, j = idx % n;
        // Symmetrize by reading lower triangle (robust to numerical noise).
        Ash[i * n + j] = (i <= j) ? Ab[i * n + j] : Ab[j * n + i];
        Vsh[i * n + j] = (i == j) ? 1.0f : 0.0f;
    }
    __syncthreads();

    // Cyclic Jacobi sweeps.
    for (int sweep = 0; sweep < maxSweeps; ++sweep) {
        float offSum = 0.0f;
        for (int p = 0; p < n; ++p)
            for (int q = p + 1; q < n; ++q)
                offSum += Ash[p * n + q] * Ash[p * n + q];
        if (sqrtf(offSum) < eps) break;

        for (int p = 0; p < n - 1; ++p) {
            for (int q = p + 1; q < n; ++q) {
                __shared__ float c, s;
                if (tid == 0) {
                    float apq = Ash[p * n + q];
                    if (fabsf(apq) < eps) {
                        c = 1.0f; s = 0.0f;
                    } else {
                        float app = Ash[p * n + p];
                        float aqq = Ash[q * n + q];
                        float theta = (aqq - app) / (2.0f * apq);
                        float t = (theta == 0.0f) ? 1.0f : copysignf(1.0f, theta) / (fabsf(theta) + sqrtf(1.0f + theta * theta));
                        c = 1.0f / sqrtf(1.0f + t * t);
                        s = t * c;
                    }
                }
                __syncthreads();

                if (s == 0.0f) continue;

                // Rotate rows p,q and cols p,q of Ash.
                for (int i = tid; i < n; i += blockDim.x) {
                    float aip = Ash[i * n + p];
                    float aiq = Ash[i * n + q];
                    Ash[i * n + p] = c * aip - s * aiq;
                    Ash[i * n + q] = s * aip + c * aiq;
                }
                __syncthreads();
                for (int j = tid; j < n; j += blockDim.x) {
                    float apj = Ash[p * n + j];
                    float aqj = Ash[q * n + j];
                    Ash[p * n + j] = c * apj - s * aqj;
                    Ash[q * n + j] = s * apj + c * aqj;
                }
                __syncthreads();
                // Accumulate into V.
                for (int i = tid; i < n; i += blockDim.x) {
                    float vip = Vsh[i * n + p];
                    float viq = Vsh[i * n + q];
                    Vsh[i * n + p] = c * vip - s * viq;
                    Vsh[i * n + q] = s * vip + c * viq;
                }
                __syncthreads();
            }
        }
    }

    // Read eigenvalues off the diagonal; sort ascending via selection
    // (serial on thread 0; n is small in the GPU path).
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
    __syncthreads();

    for (int idx = tid; idx < n * n; idx += blockDim.x)
        Vb[idx] = Vsh[idx];
}
";
    }
}
