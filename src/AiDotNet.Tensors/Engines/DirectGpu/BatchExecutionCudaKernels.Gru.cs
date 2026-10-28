namespace AiDotNet.Tensors.Engines.DirectGpu;

internal static partial class BatchExecutionCudaKernels
{
    public const string GruProjection = "gru_p_proj", GruCell = "gru_p_cell", GruCellBackward = "gru_p_cell_bwd",
        GruHiddenBackward = "gru_p_dh", GruInputGrad = "gru_p_gin", GruWeightIhGrad = "gru_p_gwih",
        GruWeightHhGrad = "gru_p_gwhh", GruBiasGrad = "gru_p_gbias";

    // The reset gate before the recurrent product (Cho et al. 2014, Keras reset_after=False):
    // n = tanh(Win x + bin + Whn (r * h) + bhn). gru_p_proj, gru_p_gin, gru_p_gwih and gru_p_gbias serve both forms.
    public const string GruRbResetHidden = "gru_rb_rh", GruRbCandidateProjection = "gru_rb_projn", GruRbCell = "gru_rb_cell",
        GruRbCellBackward = "gru_rb_cell_bwd", GruRbResetBackward = "gru_rb_drh", GruRbHiddenBackward = "gru_rb_dh",
        GruRbWeightHhGrad = "gru_rb_gwhh";

    // #1568: the fused GRU sequence, PyTorch nn.GRU semantics, as the CPU reference (CpuEngine.GruSequence) defines it:
    // batch-major x [B, T, In] and y [B, T, H], gate rows W[g * H + h, :] in the order (r, z, n), and
    //   r = sigmoid(Wir x + bir + Whr h + bhr), z = sigmoid(Wiz x + biz + Whz h + bhz),
    //   n = tanh(Win x + bin + r * (Whn h + bhn)), h' = (1 - z) * n + z * h.
    // r multiplies the recurrent term after its bias, so the two biases stay separate. One thread per output element
    // and no work-group coupling, as LstmSource: any hidden size runs and the source translates mechanically to
    // OpenCL, Metal and GLSL. Scratch: pre [B, 6H] per step (input sums, then recurrent sums). Caches: allH [(T + 1), B, H],
    // gates [T, B, 3H] (r, z, n), ghn [T, B, H] (Whn h + bhn), and the backward's dgi / dgh [T, B, 3H], the
    // pre-activation gradients of the input-side and recurrent-side sums. Weight and bias gradients sum over (t, b) in a
    // fixed order, so they are deterministic.
    internal const string GruSource = @"
extern ""C"" __global__ __launch_bounds__(256) void gru_p_proj(const float* x, const float* wih, const float* whh, const float* bih, const float* bhh, const float* allH, float* pre, int t, int B, int T, int In, int H) {
    int k = bx_gid(); if (k >= B * 3 * H) return;
    int b = k / (3 * H); int row = k - b * 3 * H;
    float si = bih[row];
    for (int i = 0; i < In; i++) si += wih[row * In + i] * x[(b * T + t) * In + i];
    float sh = bhh[row];
    for (int j = 0; j < H; j++) sh += whh[row * H + j] * allH[(t * B + b) * H + j];
    pre[b * 6 * H + row] = si;
    pre[b * 6 * H + 3 * H + row] = sh;
}

extern ""C"" __global__ __launch_bounds__(256) void gru_p_cell(const float* pre, float* gates, float* ghn, float* allH, float* y, int t, int B, int T, int H) {
    int k = bx_gid(); if (k >= B * H) return;
    int b = k / H; int h = k - b * H;
    int p = b * 6 * H;
    float r = 1.0f / (1.0f + expf(-(pre[p + h] + pre[p + 3 * H + h])));
    float z = 1.0f / (1.0f + expf(-(pre[p + H + h] + pre[p + 4 * H + h])));
    float hn = pre[p + 5 * H + h];
    float n = tanhf(pre[p + 2 * H + h] + r * hn);
    float hp = allH[(t * B + b) * H + h];
    float hv = (1.0f - z) * n + z * hp;
    int gb = (t * B + b) * 3 * H;
    gates[gb + h] = r; gates[gb + H + h] = z; gates[gb + 2 * H + h] = n;
    ghn[(t * B + b) * H + h] = hn;
    allH[((t + 1) * B + b) * H + h] = hv;
    y[(b * T + t) * H + h] = hv;
}

extern ""C"" __global__ __launch_bounds__(256) void gru_p_cell_bwd(const float* gy, const float* gates, const float* ghn, const float* allH, const float* dh, float* dgi, float* dgh, int t, int B, int T, int H) {
    int k = bx_gid(); if (k >= B * H) return;
    int b = k / H; int h = k - b * H;
    int gb = (t * B + b) * 3 * H;
    float r = gates[gb + h]; float z = gates[gb + H + h]; float n = gates[gb + 2 * H + h];
    float hn = ghn[(t * B + b) * H + h];
    float hp = allH[(t * B + b) * H + h];
    float dhv = gy[(b * T + t) * H + h] + dh[k];
    float dnp = dhv * (1.0f - z) * (1.0f - n * n);
    float drp = dnp * hn * r * (1.0f - r);
    float dzp = dhv * (hp - n) * z * (1.0f - z);
    dgi[gb + h] = drp; dgi[gb + H + h] = dzp; dgi[gb + 2 * H + h] = dnp;
    dgh[gb + h] = drp; dgh[gb + H + h] = dzp; dgh[gb + 2 * H + h] = dnp * r;
}

extern ""C"" __global__ __launch_bounds__(256) void gru_p_dh(const float* gy, const float* gates, const float* dgh, const float* whh, float* dh, int t, int B, int T, int H) {
    int k = bx_gid(); if (k >= B * H) return;
    int b = k / H; int j = k - b * H;
    int gb = (t * B + b) * 3 * H;
    float s = (gy[(b * T + t) * H + j] + dh[k]) * gates[gb + H + j];
    for (int row = 0; row < 3 * H; row++) s += dgh[gb + row] * whh[row * H + j];
    dh[k] = s;
}

extern ""C"" __global__ __launch_bounds__(256) void gru_p_gin(const float* dgi, const float* wih, float* gin, int B, int T, int In, int H) {
    int k = bx_gid(); if (k >= B * T * In) return;
    int i = k % In; int bt = k / In; int t = bt % T; int b = bt / T;
    int gb = (t * B + b) * 3 * H;
    float s = 0.0f;
    for (int row = 0; row < 3 * H; row++) s += dgi[gb + row] * wih[row * In + i];
    gin[k] = s;
}

extern ""C"" __global__ __launch_bounds__(256) void gru_p_gwih(const float* dgi, const float* x, float* gwih, int B, int T, int In, int H) {
    int k = bx_gid(); if (k >= 3 * H * In) return;
    int row = k / In; int i = k - row * In;
    float s = 0.0f;
    for (int t = 0; t < T; t++)
        for (int b = 0; b < B; b++) s += dgi[(t * B + b) * 3 * H + row] * x[(b * T + t) * In + i];
    gwih[k] = s;
}

extern ""C"" __global__ __launch_bounds__(256) void gru_p_gwhh(const float* dgh, const float* allH, float* gwhh, int B, int T, int H) {
    int k = bx_gid(); if (k >= 3 * H * H) return;
    int row = k / H; int j = k - row * H;
    float s = 0.0f;
    for (int t = 0; t < T; t++)
        for (int b = 0; b < B; b++) s += dgh[(t * B + b) * 3 * H + row] * allH[(t * B + b) * H + j];
    gwhh[k] = s;
}

extern ""C"" __global__ __launch_bounds__(256) void gru_p_gbias(const float* dgi, const float* dgh, float* gbih, float* gbhh, int B, int T, int H) {
    int row = bx_gid(); if (row >= 3 * H) return;
    float si = 0.0f; float sh = 0.0f;
    for (int t = 0; t < T; t++)
        for (int b = 0; b < B; b++) { si += dgi[(t * B + b) * 3 * H + row]; sh += dgh[(t * B + b) * 3 * H + row]; }
    gbih[row] = si;
    gbhh[row] = sh;
}
extern ""C"" __global__ __launch_bounds__(256) void gru_rb_rh(const float* pre, const float* allH, float* gates, float* rh, int t, int B, int H) {
    int k = bx_gid(); if (k >= B * H) return;
    int b = k / H; int h = k - b * H;
    int p = b * 6 * H;
    float r = 1.0f / (1.0f + expf(-(pre[p + h] + pre[p + 3 * H + h])));
    gates[(t * B + b) * 3 * H + h] = r;
    rh[(t * B + b) * H + h] = r * allH[(t * B + b) * H + h];
}

extern ""C"" __global__ __launch_bounds__(256) void gru_rb_projn(const float* whh, const float* bhh, const float* rh, float* pre, int t, int B, int H) {
    int k = bx_gid(); if (k >= B * H) return;
    int b = k / H; int j = k - b * H;
    float s = bhh[2 * H + j];
    for (int q = 0; q < H; q++) s += whh[(2 * H + j) * H + q] * rh[(t * B + b) * H + q];
    pre[b * 6 * H + 5 * H + j] = s;
}

extern ""C"" __global__ __launch_bounds__(256) void gru_rb_cell(const float* pre, float* gates, float* ghn, float* allH, float* y, int t, int B, int T, int H) {
    int k = bx_gid(); if (k >= B * H) return;
    int b = k / H; int h = k - b * H;
    int p = b * 6 * H;
    float z = 1.0f / (1.0f + expf(-(pre[p + H + h] + pre[p + 4 * H + h])));
    float hn = pre[p + 5 * H + h];
    float n = tanhf(pre[p + 2 * H + h] + hn);
    float hp = allH[(t * B + b) * H + h];
    float hv = (1.0f - z) * n + z * hp;
    int gb = (t * B + b) * 3 * H;
    gates[gb + H + h] = z; gates[gb + 2 * H + h] = n;
    ghn[(t * B + b) * H + h] = hn;
    allH[((t + 1) * B + b) * H + h] = hv;
    y[(b * T + t) * H + h] = hv;
}

extern ""C"" __global__ __launch_bounds__(256) void gru_rb_cell_bwd(const float* gy, const float* gates, const float* allH, const float* dh, float* dgi, float* dgh, int t, int B, int T, int H) {
    int k = bx_gid(); if (k >= B * H) return;
    int b = k / H; int h = k - b * H;
    int gb = (t * B + b) * 3 * H;
    float z = gates[gb + H + h]; float n = gates[gb + 2 * H + h];
    float hp = allH[(t * B + b) * H + h];
    float dhv = gy[(b * T + t) * H + h] + dh[k];
    float dnp = dhv * (1.0f - z) * (1.0f - n * n);
    float dzp = dhv * (hp - n) * z * (1.0f - z);
    dgi[gb + H + h] = dzp; dgh[gb + H + h] = dzp;
    dgi[gb + 2 * H + h] = dnp; dgh[gb + 2 * H + h] = dnp;
}

extern ""C"" __global__ __launch_bounds__(256) void gru_rb_drh(const float* gates, const float* allH, const float* whh, float* dgi, float* dgh, float* drr, int t, int B, int H) {
    int k = bx_gid(); if (k >= B * H) return;
    int b = k / H; int q = k - b * H;
    int gb = (t * B + b) * 3 * H;
    float drh = 0.0f;
    for (int j = 0; j < H; j++) drh += dgh[gb + 2 * H + j] * whh[(2 * H + j) * H + q];
    float r = gates[gb + q];
    float drp = drh * allH[(t * B + b) * H + q] * r * (1.0f - r);
    dgi[gb + q] = drp; dgh[gb + q] = drp;
    drr[k] = drh * r;
}

extern ""C"" __global__ __launch_bounds__(256) void gru_rb_dh(const float* gy, const float* gates, const float* dgh, const float* whh, const float* drr, float* dh, int t, int B, int T, int H) {
    int k = bx_gid(); if (k >= B * H) return;
    int b = k / H; int j = k - b * H;
    int gb = (t * B + b) * 3 * H;
    float s = (gy[(b * T + t) * H + j] + dh[k]) * gates[gb + H + j] + drr[k];
    for (int row = 0; row < 2 * H; row++) s += dgh[gb + row] * whh[row * H + j];
    dh[k] = s;
}

extern ""C"" __global__ __launch_bounds__(256) void gru_rb_gwhh(const float* dgh, const float* allH, const float* rh, float* gwhh, int B, int T, int H) {
    int k = bx_gid(); if (k >= 3 * H * H) return;
    int row = k / H; int j = k - row * H;
    float s = 0.0f;
    for (int t = 0; t < T; t++)
        for (int b = 0; b < B; b++)
            s += dgh[(t * B + b) * 3 * H + row] * (row < 2 * H ? allH[(t * B + b) * H + j] : rh[(t * B + b) * H + j]);
    gwhh[k] = s;
}
";
}
