namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

/// <summary>
/// GLSL compute shader source strings for Vulkan fused kernels.
/// Compiled to SPIR-V at runtime via VulkanGlslCompiler (libshaderc).
/// Each kernel is a standalone compute shader with push constants for parameters.
/// </summary>
public static class VulkanGlslKernels
{
    private const string Header = @"#version 450
layout(local_size_x = 256) in;
";

    private const string TwoBufferLayout = @"
layout(set = 0, binding = 0) readonly buffer A { float a[]; };
layout(set = 0, binding = 1) writeonly buffer B { float b[]; };
";

    private const string ThreeBufferLayout = @"
layout(set = 0, binding = 0) readonly buffer A { float a[]; };
layout(set = 0, binding = 1) readonly buffer B { float bdata[]; };
layout(set = 0, binding = 2) writeonly buffer C { float c[]; };
";

    // =====================================================================
    // Reductions
    // =====================================================================

    public static string MeanAxis => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { uint outerSize; uint reduceSize; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= outerSize) return;
    float sum = 0.0;
    uint base_idx = idx * reduceSize;
    for (uint j = 0; j < reduceSize; j++) sum += a[base_idx + j];
    b[idx] = sum / float(reduceSize);
}";

    public static string VarianceAxis => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { uint outerSize; uint reduceSize; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= outerSize) return;
    uint base_idx = idx * reduceSize;
    float sum = 0.0;
    for (uint j = 0; j < reduceSize; j++) sum += a[base_idx + j];
    float mean = sum / float(reduceSize);
    float var_sum = 0.0;
    for (uint j = 0; j < reduceSize; j++) { float d = a[base_idx + j] - mean; var_sum += d * d; }
    b[idx] = var_sum / float(reduceSize);
}";

    public static string StdAxis => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { uint outerSize; uint reduceSize; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= outerSize) return;
    uint base_idx = idx * reduceSize;
    float sum = 0.0;
    for (uint j = 0; j < reduceSize; j++) sum += a[base_idx + j];
    float mean = sum / float(reduceSize);
    float var_sum = 0.0;
    for (uint j = 0; j < reduceSize; j++) { float d = a[base_idx + j] - mean; var_sum += d * d; }
    b[idx] = sqrt(var_sum / float(reduceSize));
}";

    public static string NormAxis => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { uint outerSize; uint reduceSize; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= outerSize) return;
    uint base_idx = idx * reduceSize;
    float sum_sq = 0.0;
    for (uint j = 0; j < reduceSize; j++) { float v = a[base_idx + j]; sum_sq += v * v; }
    b[idx] = sqrt(sum_sq);
}";

    public static string LogSumExpAxis => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { uint outerSize; uint reduceSize; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= outerSize) return;
    uint base_idx = idx * reduceSize;
    float max_val = a[base_idx];
    for (uint j = 1; j < reduceSize; j++) max_val = max(max_val, a[base_idx + j]);
    float sum_exp = 0.0;
    for (uint j = 0; j < reduceSize; j++) sum_exp += exp(a[base_idx + j] - max_val);
    b[idx] = max_val + log(sum_exp);
}";

    public static string CumSumAxis => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { uint outerSize; uint innerSize; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= outerSize) return;
    uint base_idx = idx * innerSize;
    float running = 0.0;
    for (uint j = 0; j < innerSize; j++) { running += a[base_idx + j]; b[base_idx + j] = running; }
}";

    public static string ProductAxis => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { uint outerSize; uint reduceSize; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= outerSize) return;
    uint base_idx = idx * reduceSize;
    float prod = 1.0;
    for (uint j = 0; j < reduceSize; j++) prod *= a[base_idx + j];
    b[idx] = prod;
}";

    public static string NormalizeL2 => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { uint outerSize; uint innerSize; };
void main() {
    uint row = gl_GlobalInvocationID.x;
    if (row >= outerSize) return;
    uint base_idx = row * innerSize;
    float sum_sq = 0.0;
    for (uint j = 0; j < innerSize; j++) { float v = a[base_idx + j]; sum_sq += v * v; }
    float inv_norm = 1.0 / (sqrt(sum_sq) + 1e-12);
    for (uint j = 0; j < innerSize; j++) b[base_idx + j] = a[base_idx + j] * inv_norm;
}";

    // =====================================================================
    // Broadcast / Scalar
    // =====================================================================

    public static string ScalarMinusTensor => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { float scalar; uint size; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= size) return;
    b[idx] = scalar - a[idx];
}";

    public static string BroadcastAddLast => Header + ThreeBufferLayout + @"
layout(push_constant) uniform Params { uint outerSize; uint innerSize; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= outerSize * innerSize) return;
    c[idx] = a[idx] + bdata[idx % innerSize];
}";

    public static string BroadcastMulLast => Header + ThreeBufferLayout + @"
layout(push_constant) uniform Params { uint outerSize; uint innerSize; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= outerSize * innerSize) return;
    c[idx] = a[idx] * bdata[idx % innerSize];
}";

    public static string AddScalar => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { float scalar; uint size; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= size) return;
    b[idx] = a[idx] + scalar;
}";

    public static string ClipKernel => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { float minVal; float maxVal; uint size; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= size) return;
    b[idx] = clamp(a[idx], minVal, maxVal);
}";

    public static string RsqrtKernel => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { uint size; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= size) return;
    b[idx] = inversesqrt(a[idx] + 1e-12);
}";

    // =====================================================================
    // Gated Activations
    // =====================================================================

    public static string GluForward => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { uint outerSize; uint halfDim; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    uint total = outerSize * halfDim;
    if (idx >= total) return;
    uint outer = idx / halfDim; uint d = idx % halfDim; uint fullDim = halfDim * 2;
    float value = a[outer * fullDim + d];
    float gate = a[outer * fullDim + halfDim + d];
    b[idx] = value * (1.0 / (1.0 + exp(-gate)));
}";

    public static string GeGluForward => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { uint outerSize; uint halfDim; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    uint total = outerSize * halfDim;
    if (idx >= total) return;
    uint outer = idx / halfDim; uint d = idx % halfDim; uint fullDim = halfDim * 2;
    float value = a[outer * fullDim + d];
    float gate = a[outer * fullDim + halfDim + d];
    float x3 = value * value * value;
    float gelu = 0.5 * value * (1.0 + tanh(0.7978845608 * (value + 0.044715 * x3)));
    b[idx] = gelu * gate;
}";

    public static string ReGluForward => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { uint outerSize; uint halfDim; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    uint total = outerSize * halfDim;
    if (idx >= total) return;
    uint outer = idx / halfDim; uint d = idx % halfDim; uint fullDim = halfDim * 2;
    float value = a[outer * fullDim + d];
    float gate = a[outer * fullDim + halfDim + d];
    b[idx] = max(value, 0.0) * gate;
}";

    public static string SwiGluForward => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { uint outerSize; uint halfDim; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    uint total = outerSize * halfDim;
    if (idx >= total) return;
    uint outer = idx / halfDim; uint d = idx % halfDim; uint fullDim = halfDim * 2;
    float value = a[outer * fullDim + d];
    float gate = a[outer * fullDim + halfDim + d];
    float sig = 1.0 / (1.0 + exp(-value));
    b[idx] = value * sig * gate;
}";

    public static string ReluDerivative => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { uint size; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= size) return;
    b[idx] = a[idx] > 0.0 ? 1.0 : 0.0;
}";

    public static string SigmoidDerivative => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { uint size; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= size) return;
    float s = a[idx]; b[idx] = s * (1.0 - s);
}";

    public static string TanhDerivative => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { uint size; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= size) return;
    float t = a[idx]; b[idx] = 1.0 - t * t;
}";

    // =====================================================================
    // Softmax Variants + Distance
    // =====================================================================

    public static string LogSoftmax => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { uint outerSize; uint innerSize; };
void main() {
    uint row = gl_GlobalInvocationID.x;
    if (row >= outerSize) return;
    uint base_idx = row * innerSize;
    float max_val = a[base_idx];
    for (uint j = 1; j < innerSize; j++) max_val = max(max_val, a[base_idx + j]);
    float sum_exp = 0.0;
    for (uint j = 0; j < innerSize; j++) sum_exp += exp(a[base_idx + j] - max_val);
    float log_sum = log(sum_exp);
    for (uint j = 0; j < innerSize; j++) b[base_idx + j] = a[base_idx + j] - max_val - log_sum;
}";

    public static string OuterProduct => Header + ThreeBufferLayout + @"
layout(push_constant) uniform Params { uint M; uint N; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= M * N) return;
    c[idx] = a[idx / N] * bdata[idx % N];
}";

    public static string CosineSimilarity => Header + ThreeBufferLayout + @"
layout(push_constant) uniform Params { uint batchSize; uint dim; };
void main() {
    uint batch = gl_GlobalInvocationID.x;
    if (batch >= batchSize) return;
    float dot_val = 0.0, na = 0.0, nb = 0.0;
    for (uint i = 0; i < dim; i++) {
        float ai = a[batch * dim + i]; float bi = bdata[batch * dim + i];
        dot_val += ai * bi; na += ai * ai; nb += bi * bi;
    }
    c[batch] = dot_val / (sqrt(na) * sqrt(nb) + 1e-8);
}";

    // =====================================================================
    // Shape / Layout
    // =====================================================================

    public static string EyeKernel => Header + @"
layout(set = 0, binding = 0) writeonly buffer B { float b[]; };
layout(push_constant) uniform Params { uint n; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= n * n) return;
    b[idx] = (idx / n == idx % n) ? 1.0 : 0.0;
}";

    public static string OneHotKernel => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { uint batchSize; uint numClasses; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= batchSize * numClasses) return;
    uint batch = idx / numClasses; uint cls = idx % numClasses;
    b[idx] = (uint(a[batch]) == cls) ? 1.0 : 0.0;
}";

    public static string DiagKernel => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { uint n; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= n * n) return;
    b[idx] = (idx / n == idx % n) ? a[idx / n] : 0.0;
}";

    public static string TriangularMask => Header + @"
layout(set = 0, binding = 0) writeonly buffer B { float b[]; };
layout(push_constant) uniform Params { uint rows; uint cols; int diagonal; float maskValue; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= rows * cols) return;
    uint row = idx / cols; uint col = idx % cols;
    b[idx] = (col > row + uint(diagonal)) ? maskValue : 0.0;
}";

    // =====================================================================
    // Loss
    // =====================================================================

    public static string CrossEntropyLoss => Header + ThreeBufferLayout + @"
layout(push_constant) uniform Params { uint batchSize; uint numClasses; };
void main() {
    uint batch = gl_GlobalInvocationID.x;
    if (batch >= batchSize) return;
    float loss = 0.0;
    for (uint cls = 0; cls < numClasses; cls++) {
        float target = bdata[batch * numClasses + cls];
        if (target > 0.0) loss -= target * log(max(a[batch * numClasses + cls], 1e-7));
    }
    c[batch] = loss;
}";

    public static string MseLoss => Header + ThreeBufferLayout + @"
layout(push_constant) uniform Params { uint batchSize; uint numFeatures; };
void main() {
    uint batch = gl_GlobalInvocationID.x;
    if (batch >= batchSize) return;
    float sum_sq = 0.0;
    for (uint f = 0; f < numFeatures; f++) {
        float d = a[batch * numFeatures + f] - bdata[batch * numFeatures + f];
        sum_sq += d * d;
    }
    c[batch] = sum_sq / float(numFeatures);
}";

    // Shape ops
    public static string ConcatAxisGlsl => Header + ThreeBufferLayout + @"
layout(push_constant) uniform Params { uint outerSize; uint aInnerSize; uint bInnerSize; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    uint totalInner = aInnerSize + bInnerSize;
    if (idx >= outerSize * totalInner) return;
    uint outer = idx / totalInner; uint inner = idx % totalInner;
    c[idx] = (inner < aInnerSize) ? a[outer * aInnerSize + inner] : bdata[outer * bInnerSize + (inner - aInnerSize)];
}";

    public static string SliceLastAxisGlsl => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { uint outerSize; uint inputInnerSize; uint start; uint sliceSize; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= outerSize * sliceSize) return;
    b[idx] = a[(idx / sliceSize) * inputInnerSize + start + (idx % sliceSize)];
}";

    public static string Pad2DGlsl => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { uint batch; uint channels; uint inH; uint inW; uint outH; uint outW; uint padTop; uint padLeft; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= batch * channels * outH * outW) return;
    uint w = idx % outW; uint temp = idx / outW; uint h = temp % outH; temp /= outH; uint ch = temp % channels; uint ba = temp / channels;
    int srcH = int(h) - int(padTop); int srcW = int(w) - int(padLeft);
    b[idx] = (srcH >= 0 && srcH < int(inH) && srcW >= 0 && srcW < int(inW)) ? a[((ba * channels + ch) * inH + uint(srcH)) * inW + uint(srcW)] : 0.0;
}";

    public static string TileLastAxisGlsl => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { uint outerSize; uint innerSize; uint repeats; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    uint tiledInner = innerSize * repeats;
    if (idx >= outerSize * tiledInner) return;
    b[idx] = a[(idx / tiledInner) * innerSize + ((idx % tiledInner) % innerSize)];
}";

    public static string PixelShuffleGlsl => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { uint batch; uint channels; uint inH; uint inW; uint scale; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    uint outH = inH * scale; uint outW = inW * scale;
    if (idx >= batch * channels * outH * outW) return;
    uint ow = idx % outW; uint temp = idx / outW; uint oh = temp % outH; temp /= outH; uint oc = temp % channels; uint ba = temp / channels;
    uint srcC = oc * scale * scale + (oh % scale) * scale + (ow % scale);
    b[idx] = a[((ba * channels * scale * scale + srcC) * inH + oh / scale) * inW + ow / scale];
}";

    public static string EyeKernelGlsl => Header + @"
layout(set = 0, binding = 0) writeonly buffer B { float b[]; };
layout(push_constant) uniform Params { uint n; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= n * n) return;
    b[idx] = (idx / n == idx % n) ? 1.0 : 0.0;
}";

    public static string LinspaceKernelGlsl => Header + @"
layout(set = 0, binding = 0) writeonly buffer B { float b[]; };
layout(push_constant) uniform Params { float start; float step; uint size; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= size) return;
    b[idx] = start + step * float(idx);
}";

    public static string DiagKernelGlsl => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { uint n; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= n * n) return;
    b[idx] = (idx / n == idx % n) ? a[idx / n] : 0.0;
}";

    public static string TriangularMaskGlsl => Header + @"
layout(set = 0, binding = 0) writeonly buffer B { float b[]; };
layout(push_constant) uniform Params { uint rows; uint cols; int diagonal; float maskValue; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= rows * cols) return;
    b[idx] = (int(idx % cols) > int(idx / cols) + diagonal) ? maskValue : 0.0;
}";

    public static string IndexSelectGlsl => Header + ThreeBufferLayout + @"
layout(push_constant) uniform Params { uint numIndices; uint innerSize; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= numIndices * innerSize) return;
    c[idx] = a[uint(bdata[idx / innerSize]) * innerSize + (idx % innerSize)];
}";

    public static string BatchDotProductGlsl => Header + ThreeBufferLayout + @"
layout(push_constant) uniform Params { uint batchSize; uint dim; };
void main() {
    uint batch = gl_GlobalInvocationID.x;
    if (batch >= batchSize) return;
    float sum = 0.0;
    for (uint i = 0; i < dim; i++) sum += a[batch * dim + i] * bdata[batch * dim + i];
    c[batch] = sum;
}";

    public static string PairwiseDistanceGlsl => Header + ThreeBufferLayout + @"
layout(push_constant) uniform Params { uint M; uint N; uint dim; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= M * N) return;
    uint i = idx / N; uint j = idx % N;
    float d = 0.0;
    for (uint k = 0; k < dim; k++) { float diff = a[i*dim+k] - bdata[j*dim+k]; d += diff*diff; }
    c[idx] = sqrt(d);
}";

    public static string BceLossGlsl => Header + ThreeBufferLayout + @"
layout(push_constant) uniform Params { uint size; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= size) return;
    float p = clamp(a[idx], 1e-7, 1.0 - 1e-7);
    float t = bdata[idx];
    c[idx] = -(t * log(p) + (1.0 - t) * log(1.0 - p));
}";

    public static string CrossEntropyLossGlsl => Header + ThreeBufferLayout + @"
layout(push_constant) uniform Params { uint batchSize; uint numClasses; };
void main() {
    uint batch = gl_GlobalInvocationID.x;
    if (batch >= batchSize) return;
    float loss = 0.0;
    for (uint cls = 0; cls < numClasses; cls++) {
        float target = bdata[batch * numClasses + cls];
        if (target > 0.0) loss -= target * log(max(a[batch * numClasses + cls], 1e-7));
    }
    c[batch] = loss;
}";
}
