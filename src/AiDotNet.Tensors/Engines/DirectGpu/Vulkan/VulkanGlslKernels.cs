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

    public static string SumOfSquaresAxis => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { uint outerSize; uint reduceSize; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= outerSize) return;
    uint base_idx = idx * reduceSize;
    float sum_sq = 0.0;
    for (uint j = 0; j < reduceSize; j++) { float v = a[base_idx + j]; sum_sq += v * v; }
    b[idx] = sum_sq;
}";

    public static string MaxMagnitudeAxis => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { uint outerSize; uint reduceSize; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= outerSize) return;
    uint base_idx = idx * reduceSize;
    float mx = 0.0;
    for (uint j = 0; j < reduceSize; j++) { float v = abs(a[base_idx + j]); mx = max(mx, v); }
    b[idx] = mx;
}";

    public static string MinMagnitudeAxis => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { uint outerSize; uint reduceSize; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= outerSize) return;
    uint base_idx = idx * reduceSize;
    float mn = abs(a[base_idx]);
    for (uint j = 1; j < reduceSize; j++) { float v = abs(a[base_idx + j]); mn = min(mn, v); }
    b[idx] = mn;
}";

    public static string LogVarianceAxis => Header + TwoBufferLayout + @"
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
    b[idx] = log(var_sum / float(reduceSize) + 1e-8);
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

    // Backward gated activation GLSL kernels.
    // a[] = gradOutput [outerSize * halfDim]
    // bdata[] = original input [outerSize * halfDim * 2]  (first half = value, second half = gate)
    // c[] = gradInput [outerSize * halfDim * 2]  (first half = dValue, second half = dGate)
    // Each thread computes ONE output pair (dValue[idx], dGate[idx]) from gradOutput[idx] and input.

    public static string GluBackward => Header + ThreeBufferLayout + @"
layout(push_constant) uniform Params { uint outerSize; uint halfDim; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    uint total = outerSize * halfDim;
    if (idx >= total) return;
    uint outer = idx / halfDim; uint d = idx % halfDim; uint fullDim = halfDim * 2;
    float grad = a[idx];
    float value = bdata[outer * fullDim + d];
    float gate = bdata[outer * fullDim + halfDim + d];
    float sig = 1.0 / (1.0 + exp(-gate));
    c[outer * fullDim + d] = grad * sig;
    c[outer * fullDim + halfDim + d] = grad * value * sig * (1.0 - sig);
}";

    public static string GeGluBackward => Header + ThreeBufferLayout + @"
layout(push_constant) uniform Params { uint outerSize; uint halfDim; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    uint total = outerSize * halfDim;
    if (idx >= total) return;
    uint outer = idx / halfDim; uint d = idx % halfDim; uint fullDim = halfDim * 2;
    float grad = a[idx];
    float value = bdata[outer * fullDim + d];
    float gate = bdata[outer * fullDim + halfDim + d];
    float x3 = value * value * value;
    float z = 0.7978845608 * (value + 0.044715 * x3);
    float tanh_z = tanh(z);
    float gelu = 0.5 * value * (1.0 + tanh_z);
    // GELU derivative: 0.5*(1+tanh(z)) + 0.5*x*sech^2(z)*z'
    // where z' = 0.7978845608 * (1 + 3*0.044715*x^2)
    float sech2 = 1.0 - tanh_z * tanh_z;
    float z_deriv = 0.7978845608 * (1.0 + 3.0 * 0.044715 * value * value);
    float gelu_deriv = 0.5 * (1.0 + tanh_z) + 0.5 * value * sech2 * z_deriv;
    c[outer * fullDim + d] = grad * gate * gelu_deriv;
    c[outer * fullDim + halfDim + d] = grad * gelu;
}";

    public static string ReGluBackward => Header + ThreeBufferLayout + @"
layout(push_constant) uniform Params { uint outerSize; uint halfDim; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    uint total = outerSize * halfDim;
    if (idx >= total) return;
    uint outer = idx / halfDim; uint d = idx % halfDim; uint fullDim = halfDim * 2;
    float grad = a[idx];
    float value = bdata[outer * fullDim + d];
    float gate = bdata[outer * fullDim + halfDim + d];
    c[outer * fullDim + d] = grad * gate * (value > 0.0 ? 1.0 : 0.0);
    c[outer * fullDim + halfDim + d] = grad * max(value, 0.0);
}";

    public static string SwiGluBackward => Header + ThreeBufferLayout + @"
layout(push_constant) uniform Params { uint outerSize; uint halfDim; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    uint total = outerSize * halfDim;
    if (idx >= total) return;
    uint outer = idx / halfDim; uint d = idx % halfDim; uint fullDim = halfDim * 2;
    float grad = a[idx];
    float value = bdata[outer * fullDim + d];
    float gate = bdata[outer * fullDim + halfDim + d];
    float sig = 1.0 / (1.0 + exp(-value));
    float swish = value * sig;
    float swish_deriv = sig + value * sig * (1.0 - sig);
    c[outer * fullDim + d] = grad * gate * swish_deriv;
    c[outer * fullDim + halfDim + d] = grad * swish;
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

    // =====================================================================
    // Additional Scalar / Broadcast kernels
    // =====================================================================

    public static string SubScalar => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { float scalar; uint size; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= size) return;
    b[idx] = a[idx] - scalar;
}";

    public static string DivScalar => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { float scalar; uint size; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= size) return;
    b[idx] = a[idx] / scalar;
}";

    public static string PowScalar => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { float exponent; uint size; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= size) return;
    b[idx] = pow(a[idx], exponent);
}";

    public static string FracKernel => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { uint size; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= size) return;
    b[idx] = fract(a[idx]);
}";

    public static string SinKernel => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { uint size; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= size) return;
    b[idx] = sin(a[idx]);
}";

    public static string CosKernel => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { uint size; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= size) return;
    b[idx] = cos(a[idx]);
}";

    public static string BroadcastSubLast => Header + ThreeBufferLayout + @"
layout(push_constant) uniform Params { uint outerSize; uint innerSize; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= outerSize * innerSize) return;
    c[idx] = a[idx] - bdata[idx % innerSize];
}";

    public static string BroadcastDivLast => Header + ThreeBufferLayout + @"
layout(push_constant) uniform Params { uint outerSize; uint innerSize; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= outerSize * innerSize) return;
    c[idx] = a[idx] / (bdata[idx % innerSize] + 1e-12);
}";

    public static string BroadcastAddFirst => Header + ThreeBufferLayout + @"
layout(push_constant) uniform Params { uint outerSize; uint innerSize; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= outerSize * innerSize) return;
    c[idx] = a[idx / innerSize] + bdata[idx];
}";

    public static string BroadcastMulFirst => Header + ThreeBufferLayout + @"
layout(push_constant) uniform Params { uint outerSize; uint innerSize; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= outerSize * innerSize) return;
    c[idx] = a[idx / innerSize] * bdata[idx];
}";

    public static string EqualsKernel => Header + ThreeBufferLayout + @"
layout(push_constant) uniform Params { uint size; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= size) return;
    c[idx] = (a[idx] == bdata[idx]) ? 1.0 : 0.0;
}";

    public static string NotEqualsKernel => Header + ThreeBufferLayout + @"
layout(push_constant) uniform Params { uint size; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= size) return;
    c[idx] = (a[idx] != bdata[idx]) ? 1.0 : 0.0;
}";

    public static string MaskedFillKernel => Header + ThreeBufferLayout + @"
layout(push_constant) uniform Params { float fillValue; uint size; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= size) return;
    c[idx] = (bdata[idx] != 0.0) ? fillValue : a[idx];
}";

    // =====================================================================
    // Additional Shape / Layout kernels
    // =====================================================================

    public static string SetSliceLastAxisGlsl => Header + @"
layout(set = 0, binding = 0) readonly buffer A { float a[]; };
layout(set = 0, binding = 1) buffer B { float b[]; };
layout(push_constant) uniform Params { uint outerSize; uint outputInnerSize; uint start; uint sliceSize; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= outerSize * sliceSize) return;
    uint outer = idx / sliceSize; uint inner = idx % sliceSize;
    b[outer * outputInnerSize + start + inner] = a[outer * sliceSize + inner];
}";

    public static string Stack2Glsl => Header + ThreeBufferLayout + @"
layout(push_constant) uniform Params { uint size; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= size * 2) return;
    c[idx] = ((idx % 2) == 0) ? a[idx / 2] : bdata[idx / 2];
}";

    public static string Pad2DBackwardGlsl => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { uint batch; uint channels; uint inH; uint inW; uint outH; uint outW; uint padTop; uint padLeft; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= batch * channels * inH * inW) return;
    uint w = idx % inW; uint temp = idx / inW; uint h = temp % inH; temp /= inH; uint ch = temp % channels; uint ba = temp / channels;
    b[idx] = a[((ba * channels + ch) * outH + h + padTop) * outW + w + padLeft];
}";

    public static string RepeatElementsGlsl => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { uint totalElements; uint innerSize; uint repeats; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= totalElements) return;
    b[idx] = a[idx / repeats];
}";

    public static string PixelShuffleBackwardGlsl => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { uint batch; uint channels; uint inH; uint inW; uint scale; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    uint totalInputChannels = channels * scale * scale;
    if (idx >= batch * totalInputChannels * inH * inW) return;
    uint iw = idx % inW; uint temp = idx / inW; uint ih = temp % inH; temp /= inH; uint ic = temp % totalInputChannels; uint ba = temp / totalInputChannels;
    uint oc = ic / (scale * scale); uint sub = ic % (scale * scale); uint subH = sub / scale; uint subW = sub % scale;
    uint outH = inH * scale; uint outW = inW * scale;
    b[idx] = a[((ba * channels + oc) * outH + ih * scale + subH) * outW + iw * scale + subW];
}";

    public static string Crop2DGlsl => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { uint batch; uint channels; uint inH; uint inW; uint outH; uint outW; uint offsetH; uint offsetW; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= batch * channels * outH * outW) return;
    uint w = idx % outW; uint temp = idx / outW; uint h = temp % outH; temp /= outH; uint ch = temp % channels; uint ba = temp / channels;
    b[idx] = a[((ba * channels + ch) * inH + h + offsetH) * inW + w + offsetW];
}";

    public static string Crop2DBackwardGlsl => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { uint batch; uint channels; uint inH; uint inW; uint outH; uint outW; uint offsetH; uint offsetW; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= batch * channels * outH * outW) return;
    uint w = idx % outW; uint temp = idx / outW; uint h = temp % outH; temp /= outH; uint ch = temp % channels; uint ba = temp / channels;
    // b = gradInput (full size), a = gradOutput (cropped size)
    // Map output position back to input position
    b[((ba * channels + ch) * inH + h + offsetH) * inW + w + offsetW] = a[idx];
}";

    public static string ExtractDiagKernelGlsl => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { uint n; uint cols; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= n) return;
    b[idx] = a[idx * cols + idx];
}";

    // =====================================================================
    // Backward kernels
    // =====================================================================

    public static string ReduceSumBackwardGlsl => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { uint outerSize; uint reduceSize; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= outerSize * reduceSize) return;
    b[idx] = a[idx / reduceSize];
}";

    public static string ReduceMeanBackwardGlsl => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { uint outerSize; uint reduceSize; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= outerSize * reduceSize) return;
    b[idx] = a[idx / reduceSize] / float(reduceSize);
}";

    private const string FourBufferLayout = @"
layout(set = 0, binding = 0) readonly buffer A { float a[]; };
layout(set = 0, binding = 1) readonly buffer B { float bdata[]; };
layout(set = 0, binding = 2) readonly buffer C { float c[]; };
layout(set = 0, binding = 3) writeonly buffer D { float d[]; };
";

    private const string FiveBufferLayout = @"
layout(set = 0, binding = 0) readonly buffer A { float a[]; };
layout(set = 0, binding = 1) readonly buffer B { float bdata[]; };
layout(set = 0, binding = 2) readonly buffer C { float c[]; };
layout(set = 0, binding = 3) readonly buffer D { float d[]; };
layout(set = 0, binding = 4) writeonly buffer E { float e[]; };
";

    public static string ReduceMaxBackwardGlsl => Header + FourBufferLayout + @"
layout(push_constant) uniform Params { uint outerSize; uint reduceSize; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= outerSize * reduceSize) return;
    // a=gradOutput, bdata=input, c=maxValues, d=gradInput
    d[idx] = (bdata[idx] == c[idx / reduceSize]) ? a[idx / reduceSize] : 0.0;
}";

    public static string ReduceVarianceBackwardGlsl => Header + FourBufferLayout + @"
layout(push_constant) uniform Params { uint outerSize; uint reduceSize; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= outerSize * reduceSize) return;
    // a=gradOutput, bdata=input, c=means, d=gradInput
    uint outer = idx / reduceSize;
    d[idx] = a[outer] * 2.0 * (bdata[idx] - c[outer]) / float(reduceSize);
}";

    public static string ReduceLogVarianceBackwardGlsl => Header + FiveBufferLayout + @"
layout(push_constant) uniform Params { uint outerSize; uint reduceSize; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= outerSize * reduceSize) return;
    // a=gradOutput, bdata=input, c=means, d=variances, e=gradInput
    uint outer = idx / reduceSize;
    e[idx] = a[outer] * (1.0 / (d[outer] + 1e-8)) * 2.0 * (bdata[idx] - c[outer]) / float(reduceSize);
}";

    // =====================================================================
    // Softmax Variants
    // =====================================================================

    public static string TaylorSoftmaxGlsl => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { uint outerSize; uint innerSize; };
void main() {
    uint row = gl_GlobalInvocationID.x;
    if (row >= outerSize) return;
    uint base_idx = row * innerSize;
    float sum = 0.0;
    for (uint j = 0; j < innerSize; j++) {
        float x = a[base_idx + j];
        float t = max(1.0 + x + 0.5 * x * x, 1e-10);
        b[base_idx + j] = t;
        sum += t;
    }
    for (uint j = 0; j < innerSize; j++) b[base_idx + j] /= sum;
}";

    // Spherical softmax: normalize by exp-weighted L2 norm (not regular softmax)
    // output[i] = exp(x[i]) / sqrt(sum(exp(2*x[j])))
    public static string SphericalSoftmaxGlsl => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { uint outerSize; uint innerSize; };
void main() {
    uint row = gl_GlobalInvocationID.x;
    if (row >= outerSize) return;
    uint base_idx = row * innerSize;
    float max_val = a[base_idx];
    for (uint j = 1; j < innerSize; j++) max_val = max(max_val, a[base_idx + j]);
    float sum_sq = 0.0;
    for (uint j = 0; j < innerSize; j++) {
        float e = exp(a[base_idx + j] - max_val);
        b[base_idx + j] = e;
        sum_sq += e * e;
    }
    float inv_norm = 1.0 / (sqrt(sum_sq) + 1e-10);
    for (uint j = 0; j < innerSize; j++) b[base_idx + j] *= inv_norm;
}";

    // =====================================================================
    // Distance
    // =====================================================================

    public static string PairwiseDistanceSquaredGlsl => Header + ThreeBufferLayout + @"
layout(push_constant) uniform Params { uint M; uint N; uint dim; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= M * N) return;
    uint i = idx / N; uint j = idx % N;
    float d = 0.0;
    for (uint k = 0; k < dim; k++) { float diff = a[i*dim+k] - bdata[j*dim+k]; d += diff*diff; }
    c[idx] = d;
}";

    public static string BatchOuterProductGlsl => Header + ThreeBufferLayout + @"
layout(push_constant) uniform Params { uint batchSize; uint M; uint N; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= batchSize * M * N) return;
    uint ba = idx / (M * N); uint rem = idx % (M * N);
    uint i = rem / N; uint j = rem % N;
    c[idx] = a[ba * M + i] * bdata[ba * N + j];
}";

    public static string MseLossGlsl => Header + ThreeBufferLayout + @"
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

    // =====================================================================
    // Philox counter-based PRNG kernels (GPU-native RNG)
    // =====================================================================

    // Philox 2x32-10 PRNG: deterministic, parallel-safe counter-based RNG
    private const string PhiloxPrng = @"
// Philox 2x32 round function
uvec2 philox2x32_round(uvec2 ctr, uint key) {
    uint hi = uint((uint64_t(ctr.x) * uint64_t(0xD256D193u)) >> 32);
    uint lo = ctr.x * 0xD256D193u;
    return uvec2(hi ^ key ^ ctr.y, lo);
}
// Philox 2x32-10: 10 rounds for full period
uvec2 philox2x32(uvec2 ctr, uint key) {
    for (int i = 0; i < 10; i++) {
        ctr = philox2x32_round(ctr, key);
        key += 0x9E3779B9u; // golden ratio
    }
    return ctr;
}
// Convert uint to float in [0,1)
float uint_to_float01(uint x) { return float(x >> 8) * (1.0 / 16777216.0); }
";

    public static string DropoutMaskGlsl => Header + @"
layout(set = 0, binding = 0) writeonly buffer B { float b[]; };
layout(push_constant) uniform Params { uint size; float keepProb; uint seedLo; uint seedHi; };
" + PhiloxPrng + @"
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= size) return;
    uvec2 rng = philox2x32(uvec2(idx, seedHi), seedLo);
    float u = uint_to_float01(rng.x);
    float inv = 1.0 / max(keepProb, 1e-7);
    b[idx] = (u < keepProb) ? inv : 0.0;
}";

    public static string GaussianNoiseGlsl => Header + @"
layout(set = 0, binding = 0) writeonly buffer B { float b[]; };
layout(push_constant) uniform Params { uint size; float mean; float stddev; uint seedLo; uint seedHi; };
" + PhiloxPrng + @"
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= size) return;
    uvec2 rng = philox2x32(uvec2(idx, seedHi), seedLo);
    float u1 = max(uint_to_float01(rng.x), 1e-10);
    float u2 = uint_to_float01(rng.y);
    // Box-Muller transform
    float z = sqrt(-2.0 * log(u1)) * cos(6.283185307 * u2);
    b[idx] = mean + stddev * z;
}";

    public static string GumbelSoftmaxGlsl => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { uint outerSize; uint innerSize; float temperature; uint seedLo; uint seedHi; };
" + PhiloxPrng + @"
void main() {
    uint row = gl_GlobalInvocationID.x;
    if (row >= outerSize) return;
    uint base_idx = row * innerSize;

    // Add Gumbel noise and divide by temperature
    float max_val = -1e30;
    for (uint j = 0; j < innerSize; j++) {
        uvec2 rng = philox2x32(uvec2(base_idx + j, seedHi), seedLo);
        float u = max(uint_to_float01(rng.x), 1e-10);
        float gumbel = -log(-log(u));
        float val = (a[base_idx + j] + gumbel) / max(temperature, 1e-7);
        b[base_idx + j] = val;
        max_val = max(max_val, val);
    }
    // Softmax
    float sum_exp = 0.0;
    for (uint j = 0; j < innerSize; j++) {
        b[base_idx + j] = exp(b[base_idx + j] - max_val);
        sum_exp += b[base_idx + j];
    }
    for (uint j = 0; j < innerSize; j++) b[base_idx + j] /= (sum_exp + 1e-10);
}";

    public static string SparsemaxGlsl => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { uint outerSize; uint innerSize; };
void main() {
    uint row = gl_GlobalInvocationID.x;
    if (row >= outerSize) return;
    if (innerSize > 256) return; // Safety: skip rows with too many classes
    uint base_idx = row * innerSize;

    // Copy to output and sort in-place (each thread owns its row — no shared memory race)
    for (uint j = 0; j < innerSize; j++) b[base_idx + j] = a[base_idx + j];
    // Sort descending (bubble sort — OK for small innerSize typical in sparsemax)
    for (uint i = 0; i < innerSize; i++) {
        for (uint j = i + 1; j < innerSize; j++) {
            if (b[base_idx + j] > b[base_idx + i]) { float tmp = b[base_idx + i]; b[base_idx + i] = b[base_idx + j]; b[base_idx + j] = tmp; }
        }
    }
    // Find tau using sorted values (now in b[base_idx..])
    float cs = 0.0, tau = 0.0;
    for (uint k = 0; k < innerSize; k++) {
        cs += b[base_idx + k];
        float cand = (cs - 1.0) / float(k + 1);
        if (b[base_idx + k] > cand) tau = cand;
    }
    // Apply sparsemax using original unsorted values
    for (uint j = 0; j < innerSize; j++) b[base_idx + j] = max(a[base_idx + j] - tau, 0.0);
}";

    // =====================================================================
    // Activation kernels (forward + backward)
    // =====================================================================

    public static string Relu6 => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { uint size; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= size) return;
    float x = a[idx];
    b[idx] = min(max(x, 0.0), 6.0);
}";

    public static string Relu6Backward => Header + ThreeBufferLayout + @"
layout(push_constant) uniform Params { uint size; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= size) return;
    float x = bdata[idx];
    c[idx] = (x > 0.0 && x < 6.0) ? a[idx] : 0.0;
}";

    public static string PRelu => Header + ThreeBufferLayout + @"
layout(push_constant) uniform Params { uint size; uint alphaSize; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= size) return;
    float x = a[idx];
    float alpha = bdata[idx % alphaSize];
    c[idx] = x >= 0.0 ? x : alpha * x;
}";

    public static string PReluBackwardInput => Header + @"
layout(set = 0, binding = 0) readonly buffer GO { float gradOut[]; };
layout(set = 0, binding = 1) readonly buffer INP { float inp[]; };
layout(set = 0, binding = 2) readonly buffer ALPHA { float alpha[]; };
layout(set = 0, binding = 3) writeonly buffer GI { float gradIn[]; };
layout(push_constant) uniform Params { uint size; uint alphaSize; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= size) return;
    float x = inp[idx];
    float a = alpha[idx % alphaSize];
    gradIn[idx] = x >= 0.0 ? gradOut[idx] : a * gradOut[idx];
}";

    public static string PReluBackwardAlpha => Header + @"
layout(set = 0, binding = 0) readonly buffer GO { float gradOut[]; };
layout(set = 0, binding = 1) readonly buffer INP { float inp[]; };
layout(set = 0, binding = 2) writeonly buffer GA { float gradAlpha[]; };
layout(push_constant) uniform Params { uint size; uint alphaSize; };
void main() {
    uint c = gl_GlobalInvocationID.x;
    if (c >= alphaSize) return;
    float sum = 0.0;
    for (uint i = c; i < size; i += alphaSize) {
        if (inp[i] < 0.0) sum += inp[i] * gradOut[i];
    }
    gradAlpha[c] = sum;
}";

    public static string RRelu => Header + ThreeBufferLayout + @"
layout(push_constant) uniform Params { uint size; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= size) return;
    float x = a[idx];
    c[idx] = x >= 0.0 ? x : bdata[idx] * x;
}";

    public static string RReluBackward => Header + @"
layout(set = 0, binding = 0) readonly buffer GO { float gradOut[]; };
layout(set = 0, binding = 1) readonly buffer INP { float inp[]; };
layout(set = 0, binding = 2) readonly buffer NOISE { float noise[]; };
layout(set = 0, binding = 3) writeonly buffer GI { float gradIn[]; };
layout(push_constant) uniform Params { uint size; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= size) return;
    float x = inp[idx];
    gradIn[idx] = gradOut[idx] * (x >= 0.0 ? 1.0 : noise[idx]);
}";

    public static string Threshold => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { uint size; float thresh; float val; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= size) return;
    b[idx] = a[idx] > thresh ? a[idx] : val;
}";

    public static string ThresholdBackward => Header + ThreeBufferLayout + @"
layout(push_constant) uniform Params { uint size; float thresh; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= size) return;
    c[idx] = bdata[idx] > thresh ? a[idx] : 0.0;
}";

    public static string ReciprocalBackward => Header + ThreeBufferLayout + @"
layout(push_constant) uniform Params { uint size; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= size) return;
    float x = bdata[idx];
    c[idx] = -a[idx] / (x * x);
}";

    public static string VarBackwardGlsl => Header + FourBufferLayout + @"
layout(push_constant) uniform Params { uint outerSize; uint reduceSize; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    uint total = outerSize * reduceSize;
    if (idx >= total) return;
    uint outer = idx / reduceSize;
    d[idx] = a[outer] * 2.0 * (bdata[idx] - c[outer]) / float(reduceSize);
}";

    public static string StdBackwardGlsl => Header + FiveBufferLayout + @"
layout(push_constant) uniform Params { uint outerSize; uint reduceSize; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    uint total = outerSize * reduceSize;
    if (idx >= total) return;
    uint outer = idx / reduceSize;
    float s = max(d[outer], 1e-8);
    e[idx] = a[outer] * (bdata[idx] - c[outer]) / (float(reduceSize) * s);
}";

    public static string MaskedFillBackwardGlsl => Header + ThreeBufferLayout + @"
layout(push_constant) uniform Params { uint size; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= size) return;
    c[idx] = (bdata[idx] != 0.0) ? 0.0 : a[idx];
}";

    public static string WhereBackwardGlsl => Header + FourBufferLayout + @"
layout(push_constant) uniform Params { uint size; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= size) return;
    float cond = bdata[idx];
    c[idx] = (cond != 0.0) ? a[idx] : 0.0;
    d[idx] = (cond != 0.0) ? 0.0 : a[idx];
}";

    public static string NormBackwardGlsl => Header + FourBufferLayout + @"
layout(push_constant) uniform Params { uint outerSize; uint reduceSize; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    uint total = outerSize * reduceSize;
    if (idx >= total) return;
    uint outer = idx / reduceSize;
    float n = max(c[outer], 1e-8);
    d[idx] = a[outer] * bdata[idx] / n;
}";

    public static string LogSumExpBackwardGlsl => Header + FourBufferLayout + @"
layout(push_constant) uniform Params { uint outerSize; uint reduceSize; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    uint total = outerSize * reduceSize;
    if (idx >= total) return;
    uint outer = idx / reduceSize;
    float softmax_val = exp(bdata[idx] - c[outer]);
    d[idx] = a[outer] * softmax_val;
}";

    // =====================================================================
    // Loss function kernels (forward + backward)
    // =====================================================================

    public static string MseLossElementwise => Header + ThreeBufferLayout + @"
layout(push_constant) uniform Params { uint size; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= size) return;
    float d = a[idx] - bdata[idx];
    c[idx] = d * d;
}";

    public static string L1LossElementwise => Header + ThreeBufferLayout + @"
layout(push_constant) uniform Params { uint size; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= size) return;
    c[idx] = abs(a[idx] - bdata[idx]);
}";

    public static string HuberLossElementwise => Header + ThreeBufferLayout + @"
layout(push_constant) uniform Params { uint size; float delta; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= size) return;
    float d = a[idx] - bdata[idx];
    float ad = abs(d);
    c[idx] = (ad <= delta) ? (0.5 * d * d) : (delta * (ad - 0.5 * delta));
}";

    public static string BceWithLogitsElementwise => Header + ThreeBufferLayout + @"
layout(push_constant) uniform Params { uint size; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= size) return;
    float x = a[idx]; float t = bdata[idx];
    float ax = abs(x);
    c[idx] = max(x, 0.0) - x * t + log(1.0 + exp(-ax));
}";

    public static string L1LossBatch => Header + ThreeBufferLayout + @"
layout(push_constant) uniform Params { uint batchSize; uint numFeatures; };
void main() {
    uint b = gl_GlobalInvocationID.x;
    if (b >= batchSize) return;
    float sum_abs = 0.0;
    for (uint f = 0; f < numFeatures; f++) sum_abs += abs(a[b * numFeatures + f] - bdata[b * numFeatures + f]);
    c[b] = sum_abs / float(numFeatures);
}";

    public static string HuberLossBatch => Header + ThreeBufferLayout + @"
layout(push_constant) uniform Params { uint batchSize; uint numFeatures; float delta; };
void main() {
    uint b = gl_GlobalInvocationID.x;
    if (b >= batchSize) return;
    float sum = 0.0;
    for (uint f = 0; f < numFeatures; f++) {
        float d = a[b * numFeatures + f] - bdata[b * numFeatures + f];
        float ad = abs(d);
        sum += (ad <= delta) ? (0.5 * d * d) : (delta * (ad - 0.5 * delta));
    }
    c[b] = sum / float(numFeatures);
}";

    public static string NllLossBatch => Header + ThreeBufferLayout + @"
layout(push_constant) uniform Params { uint batchSize; uint numClasses; };
void main() {
    uint b = gl_GlobalInvocationID.x;
    if (b >= batchSize) return;
    int tc = int(bdata[b]);
    c[b] = (tc >= 0 && tc < int(numClasses)) ? -a[b * numClasses + uint(tc)] : 0.0;
}";

    public static string KlDivElementwise => Header + ThreeBufferLayout + @"
layout(push_constant) uniform Params { uint size; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= size) return;
    float t = bdata[idx];
    c[idx] = (t > 0.0) ? t * (log(t) - a[idx]) : 0.0;
}";

    public static string HuberLossBackward => Header + @"
layout(set = 0, binding = 0) readonly buffer GO { float gradOut[]; };
layout(set = 0, binding = 1) readonly buffer P { float pred[]; };
layout(set = 0, binding = 2) readonly buffer T { float tgt[]; };
layout(set = 0, binding = 3) writeonly buffer GI { float gradIn[]; };
layout(push_constant) uniform Params { uint size; float invN; float delta; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= size) return;
    float d = pred[idx] - tgt[idx];
    float ad = abs(d);
    float grad = (ad <= delta) ? d : (delta * ((d > 0.0) ? 1.0 : -1.0));
    gradIn[idx] = gradOut[0] * grad * invN;
}";

    public static string MseLossBackward => Header + @"
layout(set = 0, binding = 0) readonly buffer GO { float gradOut[]; };
layout(set = 0, binding = 1) readonly buffer P { float pred[]; };
layout(set = 0, binding = 2) readonly buffer T { float tgt[]; };
layout(set = 0, binding = 3) writeonly buffer GI { float gradIn[]; };
layout(push_constant) uniform Params { uint size; float invN; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= size) return;
    gradIn[idx] = gradOut[0] * 2.0 * (pred[idx] - tgt[idx]) * invN;
}";

    public static string L1LossBackward => Header + @"
layout(set = 0, binding = 0) readonly buffer GO { float gradOut[]; };
layout(set = 0, binding = 1) readonly buffer P { float pred[]; };
layout(set = 0, binding = 2) readonly buffer T { float tgt[]; };
layout(set = 0, binding = 3) writeonly buffer GI { float gradIn[]; };
layout(push_constant) uniform Params { uint size; float invN; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= size) return;
    float d = pred[idx] - tgt[idx];
    float s = (d > 0.0) ? 1.0 : ((d < 0.0) ? -1.0 : 0.0);
    gradIn[idx] = gradOut[0] * s * invN;
}";

    public static string BceWithLogitsBackward => Header + @"
layout(set = 0, binding = 0) readonly buffer GO { float gradOut[]; };
layout(set = 0, binding = 1) readonly buffer L { float logits[]; };
layout(set = 0, binding = 2) readonly buffer T { float tgt[]; };
layout(set = 0, binding = 3) writeonly buffer GI { float gradIn[]; };
layout(push_constant) uniform Params { uint size; float invN; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= size) return;
    float sig = 1.0 / (1.0 + exp(-logits[idx]));
    gradIn[idx] = gradOut[0] * (sig - tgt[idx]) * invN;
}";
}
