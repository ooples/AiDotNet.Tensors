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

// GeGLU: GELU(value) * gate
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

    // GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    float x3 = value * value * value;
    float inner = 0.7978845608f * (value + 0.044715f * x3);
    float gelu_val = 0.5f * value * (1.0f + tanhf(inner));

    output[idx] = gelu_val * gate;
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

    float x3 = value * value * value;
    float inner = 0.7978845608f * (value + 0.044715f * x3);
    float tanh_inner = tanhf(inner);
    float gelu_val = 0.5f * value * (1.0f + tanh_inner);
    float gelu_deriv = 0.5f * (1.0f + tanh_inner) +
        0.5f * value * (1.0f - tanh_inner * tanh_inner) * 0.7978845608f * (1.0f + 0.134145f * value * value);

    grad_input[outer * fullDim + d] = grad * gate * gelu_deriv;
    grad_input[outer * fullDim + halfDim + d] = grad * gelu_val;
}

// ReGLU: ReLU(value) * gate
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

    output[idx] = fmaxf(value, 0.0f) * gate;
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

    float relu_val = fmaxf(value, 0.0f);
    float relu_deriv = (value > 0.0f) ? 1.0f : 0.0f;

    grad_input[outer * fullDim + d] = grad * gate * relu_deriv;
    grad_input[outer * fullDim + halfDim + d] = grad * relu_val;
}

// SwiGLU: Swish(value) * gate = (value * sigmoid(value)) * gate
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

    float sig_val = 1.0f / (1.0f + expf(-value));
    float swish_val = value * sig_val;

    output[idx] = swish_val * gate;
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

    float sig_val = 1.0f / (1.0f + expf(-value));
    float swish_val = value * sig_val;
    float swish_deriv = sig_val + value * sig_val * (1.0f - sig_val);

    grad_input[outer * fullDim + d] = grad * gate * swish_deriv;
    grad_input[outer * fullDim + halfDim + d] = grad * swish_val;
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
