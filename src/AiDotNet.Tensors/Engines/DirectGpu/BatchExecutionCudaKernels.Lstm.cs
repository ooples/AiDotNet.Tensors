namespace AiDotNet.Tensors.Engines.DirectGpu;

internal static partial class BatchExecutionCudaKernels
{
    public const string LstmGates = "lstm_p_gates", LstmCell = "lstm_p_cell", LstmCellBackward = "lstm_p_cell_bwd",
        LstmHiddenBackward = "lstm_p_dh", LstmInputGrad = "lstm_p_gin", LstmWeightIhGrad = "lstm_p_gwih",
        LstmWeightHhGrad = "lstm_p_gwhh", LstmBiasGrad = "lstm_p_gbias";

    public const string LstmGateCell = "lstm_p_gate_cell", LstmGateCellBackward = "lstm_p_gate_cell_bwd";
    public const string StridedViewCopy = "bx_strided_view_copy";

    // #1107 / #1108: the fused-LSTM training contract (IFusedLstmSequenceTraining) for every backend, as the CPU
    // reference defines it: batch-major x [B, T, In] and y [B, T, H], PyTorch gate order (i, f, g, o) with rows
    // W[g * H + h, :], the two biases summed. One thread per output element and no work-group coupling, so any hidden
    // size runs and the source translates mechanically to OpenCL, Metal and GLSL. Caches: allH / allC [(T + 1), B, H]
    // (index (t * B + b) * H + h), activated gates [T, B, 4H], and the backward's pre-activation gradients dpre [T, B, 4H].
    // The weight and bias gradients sum over (t, b) in a fixed order, so they are deterministic.
    public const string XLstmTrainingForward = "xlstm_train_forward", XLstmTrainingBackward = "xlstm_train_backward";

    internal const string LstmSource = @"
extern ""C"" __global__ __launch_bounds__(256) void bx_strided_view_copy(const float* input, float* output, const int* layout, int rank, int offset, int count) {
    int k = bx_gid(); if (k >= count) return;
    int remaining = k; int source = offset;
    for (int d = rank - 1; d >= 0; d--) {
        source += (remaining % layout[d]) * layout[rank + d];
        remaining /= layout[d];
    }
    output[k] = input[source];
}
extern ""C"" __global__ __launch_bounds__(256) void lstm_p_gates(const float* x, const float* wih, const float* whh, const float* bih, const float* bhh, const float* allH, float* gates, int t, int B, int T, int In, int H) {
    int k = bx_gid(); if (k >= B * 4 * H) return;
    int b = k / (4 * H); int row = k - b * 4 * H;
    float s = bih[row] + bhh[row];
    for (int i = 0; i < In; i++) s += wih[row * In + i] * x[(b * T + t) * In + i];
    for (int j = 0; j < H; j++) s += whh[row * H + j] * allH[(t * B + b) * H + j];
    int g = row / H;
    gates[(t * B + b) * 4 * H + row] = g == 2 ? tanhf(s) : 1.0f / (1.0f + expf(-s));
}

extern ""C"" __global__ __launch_bounds__(256) void lstm_p_cell(const float* gates, float* allC, float* allH, float* y, int t, int B, int T, int H) {
    int k = bx_gid(); if (k >= B * H) return;
    int b = k / H; int h = k - b * H;
    int gb = (t * B + b) * 4 * H;
    float ig = gates[gb + h]; float fg = gates[gb + H + h]; float gg = gates[gb + 2 * H + h]; float og = gates[gb + 3 * H + h];
    float c = fg * allC[(t * B + b) * H + h] + ig * gg;
    float hv = og * tanhf(c);
    allC[((t + 1) * B + b) * H + h] = c;
    allH[((t + 1) * B + b) * H + h] = hv;
    y[(b * T + t) * H + h] = hv;
}

extern ""C"" __global__ __launch_bounds__(256) void lstm_p_cell_bwd(const float* gy, const float* gates, const float* allC, const float* dh, float* dc, float* dpre, int t, int B, int T, int H) {
    int k = bx_gid(); if (k >= B * H) return;
    int b = k / H; int h = k - b * H;
    int gb = (t * B + b) * 4 * H;
    float ig = gates[gb + h]; float fg = gates[gb + H + h]; float gg = gates[gb + 2 * H + h]; float og = gates[gb + 3 * H + h];
    float c = allC[((t + 1) * B + b) * H + h]; float tc = tanhf(c);
    float dhv = gy[(b * T + t) * H + h] + dh[k];
    float dcv = dhv * og * (1.0f - tc * tc) + dc[k];
    dpre[gb + h] = dcv * gg * ig * (1.0f - ig);
    dpre[gb + H + h] = dcv * allC[(t * B + b) * H + h] * fg * (1.0f - fg);
    dpre[gb + 2 * H + h] = dcv * ig * (1.0f - gg * gg);
    dpre[gb + 3 * H + h] = dhv * tc * og * (1.0f - og);
    dc[k] = dcv * fg;
}

extern ""C"" __global__ __launch_bounds__(256) void lstm_p_dh(const float* dpre, const float* whh, float* dh, int t, int B, int H) {
    int k = bx_gid(); if (k >= B * H) return;
    int b = k / H; int j = k - b * H;
    int gb = (t * B + b) * 4 * H;
    float s = 0.0f;
    for (int row = 0; row < 4 * H; row++) s += dpre[gb + row] * whh[row * H + j];
    dh[k] = s;
}

extern ""C"" __global__ __launch_bounds__(256) void lstm_p_gin(const float* dpre, const float* wih, float* gin, int B, int T, int In, int H) {
    int k = bx_gid(); if (k >= B * T * In) return;
    int i = k % In; int bt = k / In; int t = bt % T; int b = bt / T;
    int gb = (t * B + b) * 4 * H;
    float s = 0.0f;
    for (int row = 0; row < 4 * H; row++) s += dpre[gb + row] * wih[row * In + i];
    gin[k] = s;
}

extern ""C"" __global__ __launch_bounds__(256) void lstm_p_gwih(const float* dpre, const float* x, float* gwih, int B, int T, int In, int H) {
    int k = bx_gid(); if (k >= 4 * H * In) return;
    int row = k / In; int i = k - row * In;
    float s = 0.0f;
    for (int t = 0; t < T; t++)
        for (int b = 0; b < B; b++) s += dpre[(t * B + b) * 4 * H + row] * x[(b * T + t) * In + i];
    gwih[k] = s;
}

extern ""C"" __global__ __launch_bounds__(256) void lstm_p_gwhh(const float* dpre, const float* allH, float* gwhh, int B, int T, int H) {
    int k = bx_gid(); if (k >= 4 * H * H) return;
    int row = k / H; int j = k - row * H;
    float s = 0.0f;
    for (int t = 0; t < T; t++)
        for (int b = 0; b < B; b++) s += dpre[(t * B + b) * 4 * H + row] * allH[(t * B + b) * H + j];
    gwhh[k] = s;
}

extern ""C"" __global__ __launch_bounds__(256) void lstm_p_gbias(const float* dpre, float* gbih, float* gbhh, int B, int T, int H) {
    int row = bx_gid(); if (row >= 4 * H) return;
    float s = 0.0f;
    for (int t = 0; t < T; t++)
        for (int b = 0; b < B; b++) s += dpre[(t * B + b) * 4 * H + row];
    gbih[row] = s;
    gbhh[row] = s;
}

extern ""C"" __global__ __launch_bounds__(256) void lstm_p_gate_cell(const float* pre, const float* old, float* y, int N, int H) {
    int k = bx_gid(); if (k >= N) return;
    int g = (k / H) * 4 * H + k % H;
    float ig = 1.0f / (1.0f + expf(-pre[g]));
    float fg = 1.0f / (1.0f + expf(-pre[g + H]));
    float z = tanhf(pre[g + 2 * H]);
    float og = 1.0f / (1.0f + expf(-pre[g + 3 * H]));
    float c = fg * old[k] + ig * z;
    y[k] = og * tanhf(c); y[N + k] = c;
}
extern ""C"" __global__ __launch_bounds__(256) void lstm_p_gate_cell_bwd(const float* pre, const float* old, const float* y, const float* dy, float* dg, float* dc, int N, int H) {
    int k = bx_gid(); if (k >= N) return;
    int g = (k / H) * 4 * H + k % H;
    float ig = 1.0f / (1.0f + expf(-pre[g]));
    float fg = 1.0f / (1.0f + expf(-pre[g + H]));
    float z = tanhf(pre[g + 2 * H]);
    float og = 1.0f / (1.0f + expf(-pre[g + 3 * H]));
    float tc = tanhf(y[N + k]); float dh = dy[k];
    float d = dy[N + k] + dh * og * (1.0f - tc * tc);
    dg[g] = d * z * ig * (1.0f - ig);
    dg[g + H] = d * old[k] * fg * (1.0f - fg);
    dg[g + 2 * H] = d * ig * (1.0f - z * z);
    dg[g + 3 * H] = dh * tc * og * (1.0f - og);
    dc[k] = d * fg;
}

extern ""C"" __global__ __launch_bounds__(256) void xlstm_train_forward(const float* Q, const float* K, const float* V, const float* I, const float* F, const float* O, float* Y, float* S, int B, int T, int M, int H) {
    int bh = bx_gid(); if (bh >= B * H) return;
    int D = M / H; int W = D * D + D; int z = bh * (T + 1) * W;
    float scale = 1.0f / sqrtf((float)D);
    for (int j = 0; j < W; j++) S[z + j] = 0.0f;
    for (int t = 0; t < T; t++) {
        int x = (bh / H * T + t) * M + (bh % H) * D;
        int g = (bh / H * T + t) * H + bh % H;
        int a = z + t * W; int c = a + W;
        float iv = (I[g] > 4.85e8f ? 4.85e8f : I[g]); float f = F[g];
        for (int j = 0; j < D; j++) {
            S[c + D * D + j] = f * S[a + D * D + j] + iv * K[x + j] * scale;
            for (int r = 0; r < D; r++) S[c + r * D + j] = f * S[a + r * D + j] + iv * V[x + r] * K[x + j] * scale;
        }
        float nq = 0.0f;
        for (int j = 0; j < D; j++) nq += S[c + D * D + j] * Q[x + j];
        float nf = fmaxf(fabsf(nq), 1.0f);
        for (int r = 0; r < D; r++) {
            float v = 0.0f;
            for (int j = 0; j < D; j++) v += S[c + r * D + j] * Q[x + j];
            Y[x + r] = O[g] * v / nf;
        }
    }
}
extern ""C"" __global__ __launch_bounds__(256) void xlstm_train_backward(const float* Q, const float* K, const float* V, const float* I, const float* F, const float* O, const float* DY, const float* S, float* A, float* G, int B, int T, int M, int H) {
    int bh = bx_gid(); if (bh >= B * H) return;
    int D = M / H; int W = D * D + D; int z = bh * (T + 1) * W; int az = bh * W;
    int N = B * T * M; int NG = B * T * H;
    float scale = 1.0f / sqrtf((float)D);
    for (int j = 0; j < W; j++) A[az + j] = 0.0f;
    for (int t = T - 1; t >= 0; t--) {
        int x = (bh / H * T + t) * M + (bh % H) * D;
        int g = (bh / H * T + t) * H + bh % H;
        int a = z + t * W; int c = a + W;
        float iv = (I[g] > 4.85e8f ? 4.85e8f : I[g]); float f = F[g]; float o = O[g];
        float nq = 0.0f;
        for (int j = 0; j < D; j++) { nq += S[c + D * D + j] * Q[x + j]; G[x + j] = 0.0f; }
        float nf = fmaxf(fabsf(nq), 1.0f); float dnf = 0.0f; float doGate = 0.0f;
        for (int r = 0; r < D; r++) {
            float v = 0.0f;
            for (int j = 0; j < D; j++) v += S[c + r * D + j] * Q[x + j];
            doGate += DY[x + r] * v / nf;
            dnf -= DY[x + r] * o * v / (nf * nf);
            float dn = DY[x + r] * o / nf;
            for (int j = 0; j < D; j++) {
                A[az + r * D + j] += dn * Q[x + j];
                G[x + j] += S[c + r * D + j] * dn;
            }
        }
        float dnq = fabsf(nq) > 1.0f ? dnf * (nq > 0.0f ? 1.0f : -1.0f) : 0.0f;
        float di = 0.0f; float df = 0.0f;
        for (int j = 0; j < D; j++) {
            A[az + D * D + j] += dnq * Q[x + j];
            G[x + j] += dnq * S[c + D * D + j];
            di += A[az + D * D + j] * K[x + j] * scale;
            df += A[az + D * D + j] * S[a + D * D + j];
            float dk = A[az + D * D + j] * iv;
            for (int r = 0; r < D; r++) dk += A[az + r * D + j] * iv * V[x + r];
            G[N + x + j] = dk * scale;
        }
        for (int r = 0; r < D; r++) {
            float dv = 0.0f;
            for (int j = 0; j < D; j++) {
                float dc = A[az + r * D + j];
                dv += dc * iv * K[x + j] * scale;
                di += dc * V[x + r] * K[x + j] * scale;
                df += dc * S[a + r * D + j];
            }
            G[2 * N + x + r] = dv;
        }
        G[3 * N + g] = I[g] > 4.85e8f ? 0.0f : di;
        G[3 * N + NG + g] = df; G[3 * N + 2 * NG + g] = doGate;
        for (int j = 0; j < W; j++) A[az + j] *= f;
    }
}
";
}
