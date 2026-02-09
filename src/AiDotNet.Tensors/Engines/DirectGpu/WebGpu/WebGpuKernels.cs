// Copyright (c) AiDotNet. All rights reserved.
// WebGPU compute kernels in WGSL (WebGPU Shading Language).
// Only available in .NET 7+ with Blazor WebAssembly.

#if NET7_0_OR_GREATER
namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

/// <summary>
/// Contains WGSL compute shader source code for tensor operations.
/// </summary>
/// <remarks>
/// <para><b>WGSL Compute Shader Model:</b></para>
/// <para>
/// WGSL compute shaders use a workgroup-based execution model:
/// </para>
/// <list type="bullet">
/// <item><b>@workgroup_size(x, y, z)</b>: Threads per workgroup</item>
/// <item><b>global_invocation_id</b>: Thread's global position</item>
/// <item><b>local_invocation_id</b>: Thread's position within workgroup</item>
/// <item><b>workgroup_id</b>: Workgroup's position in dispatch grid</item>
/// </list>
/// <para><b>Storage Buffers:</b></para>
/// <para>
/// WGSL uses storage buffers for read/write GPU memory:
/// @group(0) @binding(n) var&lt;storage, read&gt; for input
/// @group(0) @binding(n) var&lt;storage, read_write&gt; for output
/// </para>
/// </remarks>
public static class WebGpuKernels
{
    /// <summary>
    /// Common WGSL constants and utility functions.
    /// </summary>
    public const string CommonSource = @"
// WGSL Common Utilities
const PI: f32 = 3.14159265358979323846;
const E: f32 = 2.71828182845904523536;
const EPSILON: f32 = 1e-7;
const NEG_INF: f32 = -3.402823e+38;
const POS_INF: f32 = 3.402823e+38;

// Fast approximation of tanh
fn fast_tanh(x: f32) -> f32 {
    let x2 = x * x;
    let a = x * (135135.0 + x2 * (17325.0 + x2 * (378.0 + x2)));
    let b = 135135.0 + x2 * (62370.0 + x2 * (3150.0 + x2 * 28.0));
    return a / b;
}

// Fast approximation of sigmoid
fn fast_sigmoid(x: f32) -> f32 {
    return 1.0 / (1.0 + exp(-x));
}

// GELU activation
fn gelu(x: f32) -> f32 {
    return 0.5 * x * (1.0 + tanh(sqrt(2.0 / PI) * (x + 0.044715 * x * x * x)));
}

// SiLU/Swish activation
fn silu(x: f32) -> f32 {
    return x / (1.0 + exp(-x));
}
";

    /// <summary>
    /// Element-wise operation kernels.
    /// </summary>
    public const string ElementWiseSource = @"
// Element-wise Addition
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;

struct Params {
    size: u32,
}
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn add(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.size) {
        C[idx] = A[idx] + B[idx];
    }
}

@compute @workgroup_size(256)
fn sub(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.size) {
        C[idx] = A[idx] - B[idx];
    }
}

@compute @workgroup_size(256)
fn mul(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.size) {
        C[idx] = A[idx] * B[idx];
    }
}

@compute @workgroup_size(256)
fn div(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.size) {
        C[idx] = A[idx] / B[idx];
    }
}

@compute @workgroup_size(256)
fn maximum(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.size) {
        C[idx] = max(A[idx], B[idx]);
    }
}

@compute @workgroup_size(256)
fn minimum(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.size) {
        C[idx] = min(A[idx], B[idx]);
    }
}
";

    /// <summary>
    /// Scalar operation kernels.
    /// </summary>
    public const string ScalarOpsSource = @"
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> B: array<f32>;

struct ScalarParams {
    size: u32,
    scalar: f32,
}
@group(0) @binding(2) var<uniform> params: ScalarParams;

@compute @workgroup_size(256)
fn add_scalar(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.size) {
        B[idx] = A[idx] + params.scalar;
    }
}

@compute @workgroup_size(256)
fn mul_scalar(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.size) {
        B[idx] = A[idx] * params.scalar;
    }
}

@compute @workgroup_size(256)
fn pow_scalar(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.size) {
        B[idx] = pow(A[idx], params.scalar);
    }
}

@compute @workgroup_size(256)
fn clamp_scalar(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.size) {
        B[idx] = clamp(A[idx], -params.scalar, params.scalar);
    }
}
";

    /// <summary>
    /// Unary math operations.
    /// </summary>
    public const string UnaryMathSource = @"
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> B: array<f32>;

struct Params {
    size: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn sqrt_op(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.size) {
        B[idx] = sqrt(A[idx]);
    }
}

@compute @workgroup_size(256)
fn exp_op(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.size) {
        B[idx] = exp(A[idx]);
    }
}

@compute @workgroup_size(256)
fn log_op(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.size) {
        B[idx] = log(A[idx]);
    }
}

@compute @workgroup_size(256)
fn abs_op(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.size) {
        B[idx] = abs(A[idx]);
    }
}

@compute @workgroup_size(256)
fn neg_op(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.size) {
        B[idx] = -A[idx];
    }
}

@compute @workgroup_size(256)
fn sign_op(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.size) {
        B[idx] = sign(A[idx]);
    }
}

@compute @workgroup_size(256)
fn floor_op(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.size) {
        B[idx] = floor(A[idx]);
    }
}

@compute @workgroup_size(256)
fn ceil_op(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.size) {
        B[idx] = ceil(A[idx]);
    }
}

@compute @workgroup_size(256)
fn round_op(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.size) {
        B[idx] = round(A[idx]);
    }
}

@compute @workgroup_size(256)
fn reciprocal_op(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.size) {
        B[idx] = 1.0 / A[idx];
    }
}

@compute @workgroup_size(256)
fn square_op(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.size) {
        let v = A[idx];
        B[idx] = v * v;
    }
}
";

    /// <summary>
    /// Trigonometric operations.
    /// </summary>
    public const string TrigSource = @"
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> B: array<f32>;

struct Params {
    size: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn sin_op(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.size) {
        B[idx] = sin(A[idx]);
    }
}

@compute @workgroup_size(256)
fn cos_op(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.size) {
        B[idx] = cos(A[idx]);
    }
}

@compute @workgroup_size(256)
fn tan_op(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.size) {
        B[idx] = tan(A[idx]);
    }
}

@compute @workgroup_size(256)
fn asin_op(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.size) {
        B[idx] = asin(A[idx]);
    }
}

@compute @workgroup_size(256)
fn acos_op(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.size) {
        B[idx] = acos(A[idx]);
    }
}

@compute @workgroup_size(256)
fn atan_op(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.size) {
        B[idx] = atan(A[idx]);
    }
}

@compute @workgroup_size(256)
fn sinh_op(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.size) {
        B[idx] = sinh(A[idx]);
    }
}

@compute @workgroup_size(256)
fn cosh_op(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.size) {
        B[idx] = cosh(A[idx]);
    }
}

@compute @workgroup_size(256)
fn tanh_op(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.size) {
        B[idx] = tanh(A[idx]);
    }
}
";

    /// <summary>
    /// Activation function kernels.
    /// </summary>
    public const string ActivationSource = @"
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> B: array<f32>;

struct Params {
    size: u32,
    alpha: f32,
}
@group(0) @binding(2) var<uniform> params: Params;

const PI: f32 = 3.14159265358979323846;

@compute @workgroup_size(256)
fn relu(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.size) {
        B[idx] = max(0.0, A[idx]);
    }
}

@compute @workgroup_size(256)
fn leaky_relu(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.size) {
        let x = A[idx];
        B[idx] = select(params.alpha * x, x, x > 0.0);
    }
}

@compute @workgroup_size(256)
fn elu(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.size) {
        let x = A[idx];
        B[idx] = select(params.alpha * (exp(x) - 1.0), x, x > 0.0);
    }
}

@compute @workgroup_size(256)
fn selu(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.size) {
        let x = A[idx];
        let alpha = 1.6732632423543772;
        let scale = 1.0507009873554805;
        B[idx] = scale * select(alpha * (exp(x) - 1.0), x, x > 0.0);
    }
}

@compute @workgroup_size(256)
fn sigmoid(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.size) {
        B[idx] = 1.0 / (1.0 + exp(-A[idx]));
    }
}

@compute @workgroup_size(256)
fn swish(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.size) {
        let x = A[idx];
        B[idx] = x / (1.0 + exp(-x));
    }
}

@compute @workgroup_size(256)
fn gelu_op(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.size) {
        let x = A[idx];
        B[idx] = 0.5 * x * (1.0 + tanh(sqrt(2.0 / PI) * (x + 0.044715 * x * x * x)));
    }
}

@compute @workgroup_size(256)
fn softplus(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.size) {
        B[idx] = log(1.0 + exp(A[idx]));
    }
}

@compute @workgroup_size(256)
fn mish(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.size) {
        let x = A[idx];
        B[idx] = x * tanh(log(1.0 + exp(x)));
    }
}

@compute @workgroup_size(256)
fn hardswish(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.size) {
        let x = A[idx];
        B[idx] = x * clamp(x / 6.0 + 0.5, 0.0, 1.0);
    }
}

@compute @workgroup_size(256)
fn hardsigmoid(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.size) {
        B[idx] = clamp(A[idx] / 6.0 + 0.5, 0.0, 1.0);
    }
}
";

    /// <summary>
    /// Reduction operations with workgroup shared memory.
    /// </summary>
    public const string ReductionSource = @"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

struct Params {
    size: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

var<workgroup> shared_data: array<f32, 256>;

@compute @workgroup_size(256)
fn sum_reduce(@builtin(global_invocation_id) gid: vec3<u32>,
              @builtin(local_invocation_id) lid: vec3<u32>,
              @builtin(workgroup_id) wid: vec3<u32>) {
    let idx = gid.x;
    let local_idx = lid.x;

    // Load data to shared memory
    if (idx < params.size) {
        shared_data[local_idx] = input[idx];
    } else {
        shared_data[local_idx] = 0.0;
    }
    workgroupBarrier();

    // Parallel reduction
    for (var stride: u32 = 128u; stride > 0u; stride = stride >> 1u) {
        if (local_idx < stride) {
            shared_data[local_idx] = shared_data[local_idx] + shared_data[local_idx + stride];
        }
        workgroupBarrier();
    }

    // Write result
    if (local_idx == 0u) {
        output[wid.x] = shared_data[0];
    }
}

@compute @workgroup_size(256)
fn max_reduce(@builtin(global_invocation_id) gid: vec3<u32>,
              @builtin(local_invocation_id) lid: vec3<u32>,
              @builtin(workgroup_id) wid: vec3<u32>) {
    let idx = gid.x;
    let local_idx = lid.x;

    if (idx < params.size) {
        shared_data[local_idx] = input[idx];
    } else {
        shared_data[local_idx] = -3.402823e+38; // NEG_INF
    }
    workgroupBarrier();

    for (var stride: u32 = 128u; stride > 0u; stride = stride >> 1u) {
        if (local_idx < stride) {
            shared_data[local_idx] = max(shared_data[local_idx], shared_data[local_idx + stride]);
        }
        workgroupBarrier();
    }

    if (local_idx == 0u) {
        output[wid.x] = shared_data[0];
    }
}

@compute @workgroup_size(256)
fn min_reduce(@builtin(global_invocation_id) gid: vec3<u32>,
              @builtin(local_invocation_id) lid: vec3<u32>,
              @builtin(workgroup_id) wid: vec3<u32>) {
    let idx = gid.x;
    let local_idx = lid.x;

    if (idx < params.size) {
        shared_data[local_idx] = input[idx];
    } else {
        shared_data[local_idx] = 3.402823e+38; // POS_INF
    }
    workgroupBarrier();

    for (var stride: u32 = 128u; stride > 0u; stride = stride >> 1u) {
        if (local_idx < stride) {
            shared_data[local_idx] = min(shared_data[local_idx], shared_data[local_idx + stride]);
        }
        workgroupBarrier();
    }

    if (local_idx == 0u) {
        output[wid.x] = shared_data[0];
    }
}
";

    /// <summary>
    /// Matrix multiplication with tiling.
    /// </summary>
    public const string MatMulSource = @"
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;

struct MatMulParams {
    M: u32,
    N: u32,
    K: u32,
    alpha: f32,
    beta: f32,
}
@group(0) @binding(3) var<uniform> params: MatMulParams;

const TILE_SIZE: u32 = 16u;

var<workgroup> tileA: array<array<f32, 16>, 16>;
var<workgroup> tileB: array<array<f32, 16>, 16>;

@compute @workgroup_size(16, 16)
fn gemm(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let row = gid.y;
    let col = gid.x;
    let local_row = lid.y;
    let local_col = lid.x;

    var acc: f32 = 0.0;

    let num_tiles = (params.K + TILE_SIZE - 1u) / TILE_SIZE;

    for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
        // Load tiles into shared memory
        let a_row = row;
        let a_col = t * TILE_SIZE + local_col;
        if (a_row < params.M && a_col < params.K) {
            tileA[local_row][local_col] = A[a_row * params.K + a_col];
        } else {
            tileA[local_row][local_col] = 0.0;
        }

        let b_row = t * TILE_SIZE + local_row;
        let b_col = col;
        if (b_row < params.K && b_col < params.N) {
            tileB[local_row][local_col] = B[b_row * params.N + b_col];
        } else {
            tileB[local_row][local_col] = 0.0;
        }

        workgroupBarrier();

        // Compute partial dot product
        for (var k: u32 = 0u; k < TILE_SIZE; k = k + 1u) {
            acc = acc + tileA[local_row][k] * tileB[k][local_col];
        }

        workgroupBarrier();
    }

    // Write result
    if (row < params.M && col < params.N) {
        let idx = row * params.N + col;
        C[idx] = params.alpha * acc + params.beta * C[idx];
    }
}

// Simple GEMM without tiling for small matrices
@compute @workgroup_size(256)
fn gemm_simple(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = params.M * params.N;

    if (idx >= total) {
        return;
    }

    let row = idx / params.N;
    let col = idx % params.N;

    var acc: f32 = 0.0;
    for (var k: u32 = 0u; k < params.K; k = k + 1u) {
        acc = acc + A[row * params.K + k] * B[k * params.N + col];
    }

    C[idx] = params.alpha * acc + params.beta * C[idx];
}
";

    /// <summary>
    /// Softmax kernel.
    /// </summary>
    public const string SoftmaxSource = @"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

struct Params {
    batch_size: u32,
    class_count: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

var<workgroup> shared_max: f32;
var<workgroup> shared_sum: f32;
var<workgroup> local_data: array<f32, 256>;

@compute @workgroup_size(256)
fn softmax(@builtin(global_invocation_id) gid: vec3<u32>,
           @builtin(local_invocation_id) lid: vec3<u32>,
           @builtin(workgroup_id) wid: vec3<u32>) {
    let batch_idx = wid.x;
    let local_idx = lid.x;
    let base = batch_idx * params.class_count;

    // Load and find max
    var local_max: f32 = -3.402823e+38;
    for (var i: u32 = local_idx; i < params.class_count; i = i + 256u) {
        local_max = max(local_max, input[base + i]);
    }
    local_data[local_idx] = local_max;
    workgroupBarrier();

    // Reduce to find global max
    for (var stride: u32 = 128u; stride > 0u; stride = stride >> 1u) {
        if (local_idx < stride && local_idx + stride < 256u) {
            local_data[local_idx] = max(local_data[local_idx], local_data[local_idx + stride]);
        }
        workgroupBarrier();
    }

    if (local_idx == 0u) {
        shared_max = local_data[0];
    }
    workgroupBarrier();

    // Compute exp and sum
    var local_sum: f32 = 0.0;
    for (var i: u32 = local_idx; i < params.class_count; i = i + 256u) {
        local_sum = local_sum + exp(input[base + i] - shared_max);
    }
    local_data[local_idx] = local_sum;
    workgroupBarrier();

    for (var stride: u32 = 128u; stride > 0u; stride = stride >> 1u) {
        if (local_idx < stride && local_idx + stride < 256u) {
            local_data[local_idx] = local_data[local_idx] + local_data[local_idx + stride];
        }
        workgroupBarrier();
    }

    if (local_idx == 0u) {
        shared_sum = local_data[0];
    }
    workgroupBarrier();

    // Write normalized output
    for (var i: u32 = local_idx; i < params.class_count; i = i + 256u) {
        output[base + i] = exp(input[base + i] - shared_max) / shared_sum;
    }
}
";

    /// <summary>
    /// Layer normalization kernel.
    /// </summary>
    public const string LayerNormSource = @"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> gamma: array<f32>;
@group(0) @binding(2) var<storage, read> beta: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

struct Params {
    batch_size: u32,
    feature_size: u32,
    epsilon: f32,
}
@group(0) @binding(4) var<uniform> params: Params;

var<workgroup> shared_mean: f32;
var<workgroup> shared_var: f32;
var<workgroup> local_data: array<f32, 256>;

@compute @workgroup_size(256)
fn layer_norm(@builtin(global_invocation_id) gid: vec3<u32>,
              @builtin(local_invocation_id) lid: vec3<u32>,
              @builtin(workgroup_id) wid: vec3<u32>) {
    let batch_idx = wid.x;
    let local_idx = lid.x;
    let base = batch_idx * params.feature_size;

    // Compute mean
    var local_sum: f32 = 0.0;
    for (var i: u32 = local_idx; i < params.feature_size; i = i + 256u) {
        local_sum = local_sum + input[base + i];
    }
    local_data[local_idx] = local_sum;
    workgroupBarrier();

    for (var stride: u32 = 128u; stride > 0u; stride = stride >> 1u) {
        if (local_idx < stride) {
            local_data[local_idx] = local_data[local_idx] + local_data[local_idx + stride];
        }
        workgroupBarrier();
    }

    if (local_idx == 0u) {
        shared_mean = local_data[0] / f32(params.feature_size);
    }
    workgroupBarrier();

    // Compute variance
    var local_var: f32 = 0.0;
    for (var i: u32 = local_idx; i < params.feature_size; i = i + 256u) {
        let diff = input[base + i] - shared_mean;
        local_var = local_var + diff * diff;
    }
    local_data[local_idx] = local_var;
    workgroupBarrier();

    for (var stride: u32 = 128u; stride > 0u; stride = stride >> 1u) {
        if (local_idx < stride) {
            local_data[local_idx] = local_data[local_idx] + local_data[local_idx + stride];
        }
        workgroupBarrier();
    }

    if (local_idx == 0u) {
        shared_var = local_data[0] / f32(params.feature_size);
    }
    workgroupBarrier();

    // Normalize
    let inv_std = 1.0 / sqrt(shared_var + params.epsilon);
    for (var i: u32 = local_idx; i < params.feature_size; i = i + 256u) {
        let normalized = (input[base + i] - shared_mean) * inv_std;
        output[base + i] = gamma[i] * normalized + beta[i];
    }
}
";

    /// <summary>
    /// Batch normalization kernel.
    /// </summary>
    public const string BatchNormSource = @"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> gamma: array<f32>;
@group(0) @binding(2) var<storage, read> beta: array<f32>;
@group(0) @binding(3) var<storage, read> running_mean: array<f32>;
@group(0) @binding(4) var<storage, read> running_var: array<f32>;
@group(0) @binding(5) var<storage, read_write> output: array<f32>;

struct Params {
    batch_size: u32,
    feature_size: u32,
    epsilon: f32,
}
@group(0) @binding(6) var<uniform> params: Params;

@compute @workgroup_size(256)
fn batch_norm_inference(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = params.batch_size * params.feature_size;

    if (idx >= total) {
        return;
    }

    let feature_idx = idx % params.feature_size;

    let mean = running_mean[feature_idx];
    let var_val = running_var[feature_idx];
    let inv_std = 1.0 / sqrt(var_val + params.epsilon);

    let normalized = (input[idx] - mean) * inv_std;
    output[idx] = gamma[feature_idx] * normalized + beta[feature_idx];
}
";

    /// <summary>
    /// Comparison operations.
    /// </summary>
    public const string ComparisonSource = @"
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;

struct Params {
    size: u32,
}
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn greater_than(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.size) {
        C[idx] = select(0.0, 1.0, A[idx] > B[idx]);
    }
}

@compute @workgroup_size(256)
fn less_than(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.size) {
        C[idx] = select(0.0, 1.0, A[idx] < B[idx]);
    }
}

@compute @workgroup_size(256)
fn equal_to(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.size) {
        C[idx] = select(0.0, 1.0, abs(A[idx] - B[idx]) < 1e-7);
    }
}

@compute @workgroup_size(256)
fn where_op(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.size) {
        // A is condition, B is true_val, C is false_val (overwritten with result)
        let condition = A[idx];
        let true_val = B[idx];
        let false_val = C[idx];
        C[idx] = select(false_val, true_val, condition != 0.0);
    }
}
";

    /// <summary>
    /// Loss function kernels.
    /// </summary>
    public const string LossSource = @"
@group(0) @binding(0) var<storage, read> predictions: array<f32>;
@group(0) @binding(1) var<storage, read> targets: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

struct Params {
    size: u32,
    epsilon: f32,
}
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn mse_loss(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.size) {
        let diff = predictions[idx] - targets[idx];
        output[idx] = diff * diff;
    }
}

@compute @workgroup_size(256)
fn mae_loss(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.size) {
        output[idx] = abs(predictions[idx] - targets[idx]);
    }
}

@compute @workgroup_size(256)
fn huber_loss(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.size) {
        let diff = abs(predictions[idx] - targets[idx]);
        let delta = 1.0;
        if (diff < delta) {
            output[idx] = 0.5 * diff * diff;
        } else {
            output[idx] = delta * (diff - 0.5 * delta);
        }
    }
}

@compute @workgroup_size(256)
fn binary_cross_entropy(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.size) {
        let p = clamp(predictions[idx], params.epsilon, 1.0 - params.epsilon);
        let t = targets[idx];
        output[idx] = -(t * log(p) + (1.0 - t) * log(1.0 - p));
    }
}
";

    /// <summary>
    /// Optimizer kernels (SGD, Adam, etc.).
    /// </summary>
    public const string OptimizerSource = @"
@group(0) @binding(0) var<storage, read_write> params_arr: array<f32>;
@group(0) @binding(1) var<storage, read> gradients: array<f32>;
@group(0) @binding(2) var<storage, read_write> momentum: array<f32>;
@group(0) @binding(3) var<storage, read_write> velocity: array<f32>;

struct OptimizerParams {
    size: u32,
    lr: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    weight_decay: f32,
    t: f32,
}
@group(0) @binding(4) var<uniform> opt_params: OptimizerParams;

@compute @workgroup_size(256)
fn sgd(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < opt_params.size) {
        let grad = gradients[idx] + opt_params.weight_decay * params_arr[idx];
        params_arr[idx] = params_arr[idx] - opt_params.lr * grad;
    }
}

@compute @workgroup_size(256)
fn sgd_momentum(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < opt_params.size) {
        let grad = gradients[idx] + opt_params.weight_decay * params_arr[idx];
        momentum[idx] = opt_params.beta1 * momentum[idx] + grad;
        params_arr[idx] = params_arr[idx] - opt_params.lr * momentum[idx];
    }
}

@compute @workgroup_size(256)
fn adam(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < opt_params.size) {
        let grad = gradients[idx] + opt_params.weight_decay * params_arr[idx];

        // Update biased first moment
        momentum[idx] = opt_params.beta1 * momentum[idx] + (1.0 - opt_params.beta1) * grad;

        // Update biased second moment
        velocity[idx] = opt_params.beta2 * velocity[idx] + (1.0 - opt_params.beta2) * grad * grad;

        // Bias correction
        let m_hat = momentum[idx] / (1.0 - pow(opt_params.beta1, opt_params.t));
        let v_hat = velocity[idx] / (1.0 - pow(opt_params.beta2, opt_params.t));

        // Update parameters
        params_arr[idx] = params_arr[idx] - opt_params.lr * m_hat / (sqrt(v_hat) + opt_params.epsilon);
    }
}

@compute @workgroup_size(256)
fn adamw(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < opt_params.size) {
        let grad = gradients[idx];

        // Update moments (no weight decay in gradient)
        momentum[idx] = opt_params.beta1 * momentum[idx] + (1.0 - opt_params.beta1) * grad;
        velocity[idx] = opt_params.beta2 * velocity[idx] + (1.0 - opt_params.beta2) * grad * grad;

        // Bias correction
        let m_hat = momentum[idx] / (1.0 - pow(opt_params.beta1, opt_params.t));
        let v_hat = velocity[idx] / (1.0 - pow(opt_params.beta2, opt_params.t));

        // AdamW: decoupled weight decay
        params_arr[idx] = params_arr[idx] * (1.0 - opt_params.lr * opt_params.weight_decay);
        params_arr[idx] = params_arr[idx] - opt_params.lr * m_hat / (sqrt(v_hat) + opt_params.epsilon);
    }
}
";

    /// <summary>
    /// Transpose operations.
    /// </summary>
    public const string TransposeSource = @"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

struct TransposeParams {
    rows: u32,
    cols: u32,
}
@group(0) @binding(2) var<uniform> params: TransposeParams;

const TILE_SIZE: u32 = 16u;
var<workgroup> tile: array<array<f32, 17>, 16>;  // +1 to avoid bank conflicts

@compute @workgroup_size(16, 16)
fn transpose(@builtin(global_invocation_id) gid: vec3<u32>,
             @builtin(local_invocation_id) lid: vec3<u32>,
             @builtin(workgroup_id) wid: vec3<u32>) {
    let x = wid.x * TILE_SIZE + lid.x;
    let y = wid.y * TILE_SIZE + lid.y;

    // Load tile
    if (x < params.cols && y < params.rows) {
        tile[lid.y][lid.x] = input[y * params.cols + x];
    }
    workgroupBarrier();

    // Write transposed
    let tx = wid.y * TILE_SIZE + lid.x;
    let ty = wid.x * TILE_SIZE + lid.y;

    if (tx < params.rows && ty < params.cols) {
        output[ty * params.rows + tx] = tile[lid.x][lid.y];
    }
}

@compute @workgroup_size(256)
fn transpose_simple(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = params.rows * params.cols;

    if (idx >= total) {
        return;
    }

    let row = idx / params.cols;
    let col = idx % params.cols;

    output[col * params.rows + row] = input[row * params.cols + col];
}
";

    /// <summary>
    /// Convolution operations.
    /// </summary>
    public const string ConvolutionSource = @"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> kernel: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

struct ConvParams {
    batch_size: u32,
    in_channels: u32,
    out_channels: u32,
    in_height: u32,
    in_width: u32,
    out_height: u32,
    out_width: u32,
    kernel_height: u32,
    kernel_width: u32,
    stride_h: u32,
    stride_w: u32,
    pad_h: u32,
    pad_w: u32,
}
@group(0) @binding(3) var<uniform> params: ConvParams;

@compute @workgroup_size(8, 8, 1)
fn conv2d(@builtin(global_invocation_id) gid: vec3<u32>) {
    let out_x = gid.x;
    let out_y = gid.y;
    let out_c = gid.z % params.out_channels;
    let batch = gid.z / params.out_channels;

    if (out_x >= params.out_width || out_y >= params.out_height || batch >= params.batch_size) {
        return;
    }

    var acc: f32 = 0.0;

    for (var ic: u32 = 0u; ic < params.in_channels; ic = ic + 1u) {
        for (var ky: u32 = 0u; ky < params.kernel_height; ky = ky + 1u) {
            for (var kx: u32 = 0u; kx < params.kernel_width; kx = kx + 1u) {
                let in_y = i32(out_y * params.stride_h + ky) - i32(params.pad_h);
                let in_x = i32(out_x * params.stride_w + kx) - i32(params.pad_w);

                if (in_y >= 0 && in_y < i32(params.in_height) && in_x >= 0 && in_x < i32(params.in_width)) {
                    let in_idx = batch * params.in_channels * params.in_height * params.in_width
                               + ic * params.in_height * params.in_width
                               + u32(in_y) * params.in_width + u32(in_x);
                    let k_idx = out_c * params.in_channels * params.kernel_height * params.kernel_width
                              + ic * params.kernel_height * params.kernel_width
                              + ky * params.kernel_width + kx;
                    acc = acc + input[in_idx] * kernel[k_idx];
                }
            }
        }
    }

    let out_idx = batch * params.out_channels * params.out_height * params.out_width
                + out_c * params.out_height * params.out_width
                + out_y * params.out_width + out_x;
    output[out_idx] = acc;
}
";

    /// <summary>
    /// Pooling operations.
    /// </summary>
    public const string PoolingSource = @"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

struct PoolParams {
    batch_size: u32,
    channels: u32,
    in_height: u32,
    in_width: u32,
    out_height: u32,
    out_width: u32,
    kernel_height: u32,
    kernel_width: u32,
    stride_h: u32,
    stride_w: u32,
    pad_h: u32,
    pad_w: u32,
}
@group(0) @binding(2) var<uniform> params: PoolParams;

@compute @workgroup_size(8, 8, 1)
fn max_pool2d(@builtin(global_invocation_id) gid: vec3<u32>) {
    let out_x = gid.x;
    let out_y = gid.y;
    let c = gid.z % params.channels;
    let batch = gid.z / params.channels;

    if (out_x >= params.out_width || out_y >= params.out_height || batch >= params.batch_size) {
        return;
    }

    var max_val: f32 = -3.402823e+38;

    for (var ky: u32 = 0u; ky < params.kernel_height; ky = ky + 1u) {
        for (var kx: u32 = 0u; kx < params.kernel_width; kx = kx + 1u) {
            let in_y = i32(out_y * params.stride_h + ky) - i32(params.pad_h);
            let in_x = i32(out_x * params.stride_w + kx) - i32(params.pad_w);

            if (in_y >= 0 && in_y < i32(params.in_height) && in_x >= 0 && in_x < i32(params.in_width)) {
                let in_idx = batch * params.channels * params.in_height * params.in_width
                           + c * params.in_height * params.in_width
                           + u32(in_y) * params.in_width + u32(in_x);
                max_val = max(max_val, input[in_idx]);
            }
        }
    }

    let out_idx = batch * params.channels * params.out_height * params.out_width
                + c * params.out_height * params.out_width
                + out_y * params.out_width + out_x;
    output[out_idx] = max_val;
}

@compute @workgroup_size(8, 8, 1)
fn avg_pool2d(@builtin(global_invocation_id) gid: vec3<u32>) {
    let out_x = gid.x;
    let out_y = gid.y;
    let c = gid.z % params.channels;
    let batch = gid.z / params.channels;

    if (out_x >= params.out_width || out_y >= params.out_height || batch >= params.batch_size) {
        return;
    }

    var sum: f32 = 0.0;
    var count: f32 = 0.0;

    for (var ky: u32 = 0u; ky < params.kernel_height; ky = ky + 1u) {
        for (var kx: u32 = 0u; kx < params.kernel_width; kx = kx + 1u) {
            let in_y = i32(out_y * params.stride_h + ky) - i32(params.pad_h);
            let in_x = i32(out_x * params.stride_w + kx) - i32(params.pad_w);

            if (in_y >= 0 && in_y < i32(params.in_height) && in_x >= 0 && in_x < i32(params.in_width)) {
                let in_idx = batch * params.channels * params.in_height * params.in_width
                           + c * params.in_height * params.in_width
                           + u32(in_y) * params.in_width + u32(in_x);
                sum = sum + input[in_idx];
                count = count + 1.0;
            }
        }
    }

    let out_idx = batch * params.channels * params.out_height * params.out_width
                + c * params.out_height * params.out_width
                + out_y * params.out_width + out_x;
    output[out_idx] = sum / max(count, 1.0);
}
";

    /// <summary>
    /// Gets all kernel sources combined.
    /// </summary>
    public static string GetCombinedSource()
    {
        return CommonSource + ElementWiseSource + ScalarOpsSource + UnaryMathSource +
               TrigSource + ActivationSource + ReductionSource + MatMulSource +
               SoftmaxSource + LayerNormSource + BatchNormSource + ComparisonSource +
               LossSource + OptimizerSource + TransposeSource + ConvolutionSource + PoolingSource;
    }
}
#endif
