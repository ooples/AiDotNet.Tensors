namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels;

/// <summary>
/// Fused CUDA kernels for gated linear units (GLU, GeGLU, ReGLU, SwiGLU)
/// and activation derivatives (ReLU', Sigmoid', Tanh').
/// All gated activations split the input in half and apply gate * activation(value).
/// Single-pass fused kernels: one read of input, one write of output.
/// </summary>
public static class CudaGatedActivationKernels
{
    public static string GetSource()
    {
        return @"
// ============================================================================
// Gated Linear Units: input[..., 2*D] → output[..., D]
// Split at last dim: value = input[..., :D], gate = input[..., D:]
// ============================================================================

// GLU: value * sigmoid(gate)
extern ""C"" __global__ __launch_bounds__(256) void glu_forward(
    const float* __restrict__ input, float* __restrict__ output,
    int outerSize, int halfDim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = outerSize * halfDim;
    if (idx >= total) return;

    int outer = idx / halfDim;
    int d = idx % halfDim;
    int fullDim = halfDim * 2;

    float value = input[outer * fullDim + d];
    float gate = input[outer * fullDim + halfDim + d];
    float sig_gate = 1.0f / (1.0f + expf(-gate));

    output[idx] = value * sig_gate;
}

extern ""C"" __global__ __launch_bounds__(256) void glu_backward(
    const float* __restrict__ grad_output,
    const float* __restrict__ input,
    float* __restrict__ grad_input,
    int outerSize, int halfDim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = outerSize * halfDim;
    if (idx >= total) return;

    int outer = idx / halfDim;
    int d = idx % halfDim;
    int fullDim = halfDim * 2;

    float value = input[outer * fullDim + d];
    float gate = input[outer * fullDim + halfDim + d];
    float sig_gate = 1.0f / (1.0f + expf(-gate));
    float grad = grad_output[idx];

    // dL/d_value = grad * sigmoid(gate)
    grad_input[outer * fullDim + d] = grad * sig_gate;
    // dL/d_gate = grad * value * sigmoid(gate) * (1 - sigmoid(gate))
    grad_input[outer * fullDim + halfDim + d] = grad * value * sig_gate * (1.0f - sig_gate);
}

// GeGLU: value * GELU(gate)  (#775: activation applies to the GATE / 2nd half)
extern ""C"" __global__ __launch_bounds__(256) void geglu_forward(
    const float* __restrict__ input, float* __restrict__ output,
    int outerSize, int halfDim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = outerSize * halfDim;
    if (idx >= total) return;

    int outer = idx / halfDim;
    int d = idx % halfDim;
    int fullDim = halfDim * 2;

    float value = input[outer * fullDim + d];
    float gate = input[outer * fullDim + halfDim + d];

    // #775: activation applies to the GATE (second half), output = value * GELU(gate) — matches
    // glu_forward, the CpuEngine reference, and the validated OpenCL fix (was GELU(value)*gate).
    float g3 = gate * gate * gate;
    float inner = 0.7978845608f * (gate + 0.044715f * g3);
    float gelu_gate = 0.5f * gate * (1.0f + tanhf(inner));

    output[idx] = value * gelu_gate;
}

extern ""C"" __global__ __launch_bounds__(256) void geglu_backward(
    const float* __restrict__ grad_output,
    const float* __restrict__ input,
    float* __restrict__ grad_input,
    int outerSize, int halfDim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = outerSize * halfDim;
    if (idx >= total) return;

    int outer = idx / halfDim;
    int d = idx % halfDim;
    int fullDim = halfDim * 2;

    float value = input[outer * fullDim + d];
    float gate = input[outer * fullDim + halfDim + d];
    float grad = grad_output[idx];

    // #775: out = value * GELU(gate) → d/value = grad*GELU(gate); d/gate = grad*value*GELU'(gate).
    float g3 = gate * gate * gate;
    float inner = 0.7978845608f * (gate + 0.044715f * g3);
    float tanh_inner = tanhf(inner);
    float gelu_gate = 0.5f * gate * (1.0f + tanh_inner);
    float gelu_deriv = 0.5f * (1.0f + tanh_inner) +
        0.5f * gate * (1.0f - tanh_inner * tanh_inner) * 0.7978845608f * (1.0f + 0.134145f * gate * gate);

    grad_input[outer * fullDim + d] = grad * gelu_gate;
    grad_input[outer * fullDim + halfDim + d] = grad * value * gelu_deriv;
}

// ReGLU: value * ReLU(gate)  (#775: activation applies to the GATE / 2nd half)
extern ""C"" __global__ __launch_bounds__(256) void reglu_forward(
    const float* __restrict__ input, float* __restrict__ output,
    int outerSize, int halfDim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = outerSize * halfDim;
    if (idx >= total) return;

    int outer = idx / halfDim;
    int d = idx % halfDim;
    int fullDim = halfDim * 2;

    float value = input[outer * fullDim + d];
    float gate = input[outer * fullDim + halfDim + d];

    output[idx] = value * fmaxf(gate, 0.0f); // #775: act on gate (see geglu_forward)
}

extern ""C"" __global__ __launch_bounds__(256) void reglu_backward(
    const float* __restrict__ grad_output,
    const float* __restrict__ input,
    float* __restrict__ grad_input,
    int outerSize, int halfDim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = outerSize * halfDim;
    if (idx >= total) return;

    int outer = idx / halfDim;
    int d = idx % halfDim;
    int fullDim = halfDim * 2;

    float value = input[outer * fullDim + d];
    float gate = input[outer * fullDim + halfDim + d];
    float grad = grad_output[idx];

    // #775: out = value * ReLU(gate).
    float relu_gate = fmaxf(gate, 0.0f);
    float relu_deriv = (gate > 0.0f) ? 1.0f : 0.0f;

    grad_input[outer * fullDim + d] = grad * relu_gate;
    grad_input[outer * fullDim + halfDim + d] = grad * value * relu_deriv;
}

// SwiGLU: value * Swish(gate) = value * (gate * sigmoid(gate))  (#775: activation applies to the GATE)
extern ""C"" __global__ __launch_bounds__(256) void swiglu_forward(
    const float* __restrict__ input, float* __restrict__ output,
    int outerSize, int halfDim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = outerSize * halfDim;
    if (idx >= total) return;

    int outer = idx / halfDim;
    int d = idx % halfDim;
    int fullDim = halfDim * 2;

    float value = input[outer * fullDim + d];
    float gate = input[outer * fullDim + halfDim + d];

    // #775: out = value * Swish(gate).
    float sig_gate = 1.0f / (1.0f + expf(-gate));
    float swish_gate = gate * sig_gate;

    output[idx] = value * swish_gate;
}

extern ""C"" __global__ __launch_bounds__(256) void swiglu_backward(
    const float* __restrict__ grad_output,
    const float* __restrict__ input,
    float* __restrict__ grad_input,
    int outerSize, int halfDim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = outerSize * halfDim;
    if (idx >= total) return;

    int outer = idx / halfDim;
    int d = idx % halfDim;
    int fullDim = halfDim * 2;

    float value = input[outer * fullDim + d];
    float gate = input[outer * fullDim + halfDim + d];
    float grad = grad_output[idx];

    // #775: out = value * Swish(gate) → d/value = grad*Swish(gate); d/gate = grad*value*Swish'(gate).
    float sig_gate = 1.0f / (1.0f + expf(-gate));
    float swish_gate = gate * sig_gate;
    float swish_deriv = sig_gate + gate * sig_gate * (1.0f - sig_gate);

    grad_input[outer * fullDim + d] = grad * swish_gate;
    grad_input[outer * fullDim + halfDim + d] = grad * value * swish_deriv;
}

// ============================================================================
// Activation derivatives (for custom backward passes)
// ============================================================================

extern ""C"" __global__ __launch_bounds__(256) void relu_derivative(
    const float* __restrict__ input, float* __restrict__ output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    output[idx] = (input[idx] > 0.0f) ? 1.0f : 0.0f;
}

extern ""C"" __global__ __launch_bounds__(256) void sigmoid_derivative(
    const float* __restrict__ sigmoid_output, float* __restrict__ output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    float s = sigmoid_output[idx];
    output[idx] = s * (1.0f - s);
}

extern ""C"" __global__ __launch_bounds__(256) void tanh_derivative(
    const float* __restrict__ tanh_output, float* __restrict__ output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    float t = tanh_output[idx];
    output[idx] = 1.0f - t * t;
}
";
    }

    public static string[] GetKernelNames()
    {
        return
        [
            "glu_forward",
            "glu_backward",
            "geglu_forward",
            "geglu_backward",
            "reglu_forward",
            "reglu_backward",
            "swiglu_forward",
            "swiglu_backward",
            "relu_derivative",
            "sigmoid_derivative",
            "tanh_derivative"
        ];
    }
}
