namespace AiDotNet.Tensors.Engines.DirectGpu;

internal static partial class BatchExecutionCudaKernels
{
    public const string LstmGates = "lstm_p_gates", LstmCell = "lstm_p_cell", LstmCellBackward = "lstm_p_cell_bwd",
        LstmHiddenBackward = "lstm_p_dh", LstmInputGrad = "lstm_p_gin", LstmWeightIhGrad = "lstm_p_gwih",
        LstmWeightHhGrad = "lstm_p_gwhh", LstmBiasGrad = "lstm_p_gbias";

    // #1107 / #1108: the fused-LSTM training contract (IFusedLstmSequenceTraining) for every backend, as the CPU
    // reference defines it: batch-major x [B, T, In] and y [B, T, H], PyTorch gate order (i, f, g, o) with rows
    // W[g * H + h, :], the two biases summed. One thread per output element and no work-group coupling, so any hidden
    // size runs and the source translates mechanically to OpenCL, Metal and GLSL. Caches: allH / allC [(T + 1), B, H]
    // (index (t * B + b) * H + h), activated gates [T, B, 4H], and the backward's pre-activation gradients dpre [T, B, 4H].
    // The weight and bias gradients sum over (t, b) in a fixed order, so they are deterministic.
    internal const string LstmSource = @"
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
";
}
