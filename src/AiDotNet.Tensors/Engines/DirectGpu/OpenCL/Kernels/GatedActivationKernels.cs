namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL.Kernels;

/// <summary>
/// OpenCL gated activation kernels: GLU, GeGLU, ReGLU, SwiGLU + derivatives.
/// </summary>
public static class GatedActivationKernels
{
    public static string GetSource()
    {
        return @"
__kernel void glu_forward(__global const float* input, __global float* output, int outerSize, int halfDim) {
    int idx = get_global_id(0); int total = outerSize * halfDim; if (idx >= total) return;
    int outer = idx / halfDim; int d = idx % halfDim; int fullDim = halfDim * 2;
    float value = input[outer * fullDim + d]; float gate = input[outer * fullDim + halfDim + d];
    output[idx] = value * (1.0f / (1.0f + exp(-gate)));
}
__kernel void glu_backward(__global const float* grad_output, __global const float* input, __global float* grad_input, int outerSize, int halfDim) {
    int idx = get_global_id(0); int total = outerSize * halfDim; if (idx >= total) return;
    int outer = idx / halfDim; int d = idx % halfDim; int fullDim = halfDim * 2;
    float value = input[outer * fullDim + d]; float gate = input[outer * fullDim + halfDim + d];
    float sig = 1.0f / (1.0f + exp(-gate)); float grad = grad_output[idx];
    grad_input[outer * fullDim + d] = grad * sig;
    grad_input[outer * fullDim + halfDim + d] = grad * value * sig * (1.0f - sig);
}
__kernel void geglu_forward(__global const float* input, __global float* output, int outerSize, int halfDim) {
    int idx = get_global_id(0); int total = outerSize * halfDim; if (idx >= total) return;
    int outer = idx / halfDim; int d = idx % halfDim; int fullDim = halfDim * 2;
    float value = input[outer * fullDim + d]; float gate = input[outer * fullDim + halfDim + d];
    float x3 = value * value * value;
    float gelu = 0.5f * value * (1.0f + tanh(0.7978845608f * (value + 0.044715f * x3)));
    output[idx] = gelu * gate;
}
__kernel void geglu_backward(__global const float* grad_output, __global const float* input, __global float* grad_input, int outerSize, int halfDim) {
    int idx = get_global_id(0); int total = outerSize * halfDim; if (idx >= total) return;
    int outer = idx / halfDim; int d = idx % halfDim; int fullDim = halfDim * 2;
    float value = input[outer * fullDim + d]; float gate = input[outer * fullDim + halfDim + d]; float grad = grad_output[idx];
    float x3 = value * value * value; float inner = 0.7978845608f * (value + 0.044715f * x3);
    float th = tanh(inner); float gelu = 0.5f * value * (1.0f + th);
    float gelu_d = 0.5f * (1.0f + th) + 0.5f * value * (1.0f - th * th) * 0.7978845608f * (1.0f + 0.134145f * value * value);
    grad_input[outer * fullDim + d] = grad * gate * gelu_d;
    grad_input[outer * fullDim + halfDim + d] = grad * gelu;
}
__kernel void reglu_forward(__global const float* input, __global float* output, int outerSize, int halfDim) {
    int idx = get_global_id(0); int total = outerSize * halfDim; if (idx >= total) return;
    int outer = idx / halfDim; int d = idx % halfDim; int fullDim = halfDim * 2;
    float value = input[outer * fullDim + d]; float gate = input[outer * fullDim + halfDim + d];
    // ReGLU: output = value * ReLU(gate)
    output[idx] = value * fmax(gate, 0.0f);
}
__kernel void reglu_backward(__global const float* grad_output, __global const float* input, __global float* grad_input, int outerSize, int halfDim) {
    int idx = get_global_id(0); int total = outerSize * halfDim; if (idx >= total) return;
    int outer = idx / halfDim; int d = idx % halfDim; int fullDim = halfDim * 2;
    float value = input[outer * fullDim + d]; float gate = input[outer * fullDim + halfDim + d]; float grad = grad_output[idx];
    // d/d_value = ReLU(gate), d/d_gate = value * (gate > 0 ? 1 : 0)
    grad_input[outer * fullDim + d] = grad * fmax(gate, 0.0f);
    grad_input[outer * fullDim + halfDim + d] = grad * value * ((gate > 0.0f) ? 1.0f : 0.0f);
}
__kernel void swiglu_forward(__global const float* input, __global float* output, int outerSize, int halfDim) {
    int idx = get_global_id(0); int total = outerSize * halfDim; if (idx >= total) return;
    int outer = idx / halfDim; int d = idx % halfDim; int fullDim = halfDim * 2;
    float value = input[outer * fullDim + d]; float gate = input[outer * fullDim + halfDim + d];
    // SwiGLU: output = value * SiLU(gate) = value * gate * sigmoid(gate)
    float sig = 1.0f / (1.0f + exp(-gate));
    output[idx] = value * gate * sig;
}
__kernel void swiglu_backward(__global const float* grad_output, __global const float* input, __global float* grad_input, int outerSize, int halfDim) {
    int idx = get_global_id(0); int total = outerSize * halfDim; if (idx >= total) return;
    int outer = idx / halfDim; int d = idx % halfDim; int fullDim = halfDim * 2;
    float value = input[outer * fullDim + d]; float gate = input[outer * fullDim + halfDim + d]; float grad = grad_output[idx];
    float sig = 1.0f / (1.0f + exp(-value)); float swish = value * sig;
    float swish_d = sig + value * sig * (1.0f - sig);
    grad_input[outer * fullDim + d] = grad * gate * swish_d;
    grad_input[outer * fullDim + halfDim + d] = grad * swish;
}
__kernel void relu_derivative(__global const float* input, __global float* output, int size) {
    int idx = get_global_id(0); if (idx >= size) return;
    output[idx] = (input[idx] > 0.0f) ? 1.0f : 0.0f;
}
__kernel void sigmoid_derivative(__global const float* sig_out, __global float* output, int size) {
    int idx = get_global_id(0); if (idx >= size) return;
    float s = sig_out[idx]; output[idx] = s * (1.0f - s);
}
__kernel void tanh_derivative(__global const float* tanh_out, __global float* output, int size) {
    int idx = get_global_id(0); if (idx >= size) return;
    float t = tanh_out[idx]; output[idx] = 1.0f - t * t;
}
";
    }

    public static string[] GetKernelNames()
    {
        return new[]
        {
            "glu_forward", "glu_backward", "geglu_forward", "geglu_backward",
            "reglu_forward", "reglu_backward", "swiglu_forward", "swiglu_backward",
            "relu_derivative", "sigmoid_derivative", "tanh_derivative"
        };
    }
}
