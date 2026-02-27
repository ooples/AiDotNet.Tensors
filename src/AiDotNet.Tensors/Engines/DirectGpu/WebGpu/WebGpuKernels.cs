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
    /// Additional unary math operations not covered by UnaryMathSource.
    /// </summary>
    public const string AdditionalUnarySource = @"
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> B: array<f32>;

struct Params {
    size: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn exp2_op(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.size) {
        B[idx] = exp2(A[idx]);
    }
}

@compute @workgroup_size(256)
fn exp10_op(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.size) {
        B[idx] = exp(A[idx] * 2.302585093);
    }
}

@compute @workgroup_size(256)
fn expm1_op(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.size) {
        B[idx] = exp(A[idx]) - 1.0;
    }
}

@compute @workgroup_size(256)
fn log2_op(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.size) {
        B[idx] = log2(A[idx]);
    }
}

@compute @workgroup_size(256)
fn log1p_op(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.size) {
        B[idx] = log(1.0 + A[idx]);
    }
}

@compute @workgroup_size(256)
fn log10_op(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.size) {
        B[idx] = log(A[idx]) * 0.4342944819;
    }
}

@compute @workgroup_size(256)
fn cbrt_op(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.size) {
        let x = A[idx];
        B[idx] = sign(x) * pow(abs(x), 0.3333333333);
    }
}

@compute @workgroup_size(256)
fn trunc_op(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.size) {
        B[idx] = trunc(A[idx]);
    }
}
";

    /// <summary>
    /// Inverse hyperbolic operations (asinh, acosh, atanh).
    /// </summary>
    public const string InverseHyperbolicSource = @"
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> B: array<f32>;

struct Params {
    size: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn asinh_op(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.size) {
        B[idx] = asinh(A[idx]);
    }
}

@compute @workgroup_size(256)
fn acosh_op(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.size) {
        B[idx] = acosh(A[idx]);
    }
}

@compute @workgroup_size(256)
fn atanh_op(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.size) {
        B[idx] = atanh(A[idx]);
    }
}
";

    /// <summary>
    /// Fill buffer with constant value.
    /// </summary>
    public const string FillSource = @"
@group(0) @binding(0) var<storage, read_write> A: array<f32>;

struct Params {
    size: u32,
    value: f32,
}
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(256)
fn fill_value(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.size) {
        A[idx] = params.value;
    }
}
";

    /// <summary>
    /// Clamp operation with min and max bounds.
    /// </summary>
    public const string ClampSource = @"
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> B: array<f32>;

struct ClampParams {
    size: u32,
    min_val: f32,
    max_val: f32,
}
@group(0) @binding(2) var<uniform> params: ClampParams;

@compute @workgroup_size(256)
fn clamp_op(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.size) {
        B[idx] = clamp(A[idx], params.min_val, params.max_val);
    }
}
";

    /// <summary>
    /// Fused multiply-add: D = A * B + C.
    /// </summary>
    public const string FmaSource = @"
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read> C: array<f32>;
@group(0) @binding(3) var<storage, read_write> D: array<f32>;

struct Params {
    size: u32,
}
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(256)
fn fma_op(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.size) {
        D[idx] = fma(A[idx], B[idx], C[idx]);
    }
}
";

    /// <summary>
    /// Bias addition operations (row-wise and channel-wise).
    /// </summary>
    public const string BiasAddSource = @"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> bias: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

struct BiasParams {
    rows: u32,
    cols: u32,
}
@group(0) @binding(3) var<uniform> params: BiasParams;

@compute @workgroup_size(256)
fn bias_add(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = params.rows * params.cols;
    if (idx < total) {
        output[idx] = input[idx] + bias[idx % params.cols];
    }
}
";

    /// <summary>
    /// Channel-wise bias addition for conv2d output (NCHW format).
    /// </summary>
    public const string Conv2DBiasAddSource = @"
@group(0) @binding(0) var<storage, read_write> output: array<f32>;
@group(0) @binding(1) var<storage, read> bias: array<f32>;

struct Params {
    batch: u32,
    channels: u32,
    spatial_size: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn conv2d_bias_add(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = params.batch * params.channels * params.spatial_size;
    if (idx < total) {
        let channel = (idx / params.spatial_size) % params.channels;
        output[idx] = output[idx] + bias[channel];
    }
}
";

    /// <summary>
    /// Activation function backward pass kernels.
    /// Uses 3 storage buffers: grad_output (read), forward_data (read), grad_input (write).
    /// </summary>
    public const string ActivationBackwardSource = @"
@group(0) @binding(0) var<storage, read> grad_output: array<f32>;
@group(0) @binding(1) var<storage, read> forward_data: array<f32>;
@group(0) @binding(2) var<storage, read_write> grad_input: array<f32>;

struct Params {
    size: u32,
    alpha: f32,
    beta: f32,
}
@group(0) @binding(3) var<uniform> params: Params;

const PI: f32 = 3.14159265358979323846;

@compute @workgroup_size(256)
fn relu_backward(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.size) {
        grad_input[idx] = grad_output[idx] * select(0.0, 1.0, forward_data[idx] > 0.0);
    }
}

@compute @workgroup_size(256)
fn leaky_relu_backward(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.size) {
        grad_input[idx] = grad_output[idx] * select(params.alpha, 1.0, forward_data[idx] >= 0.0);
    }
}

@compute @workgroup_size(256)
fn elu_backward(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.size) {
        let x = forward_data[idx];
        grad_input[idx] = grad_output[idx] * select(params.alpha * exp(x), 1.0, x >= 0.0);
    }
}

@compute @workgroup_size(256)
fn selu_backward(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.size) {
        let x = forward_data[idx];
        // alpha in params.alpha, scale in params.beta
        grad_input[idx] = grad_output[idx] * params.beta * select(params.alpha * exp(x), 1.0, x >= 0.0);
    }
}

@compute @workgroup_size(256)
fn sigmoid_backward(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.size) {
        // forward_data is the sigmoid output
        let s = forward_data[idx];
        grad_input[idx] = grad_output[idx] * s * (1.0 - s);
    }
}

@compute @workgroup_size(256)
fn tanh_backward(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.size) {
        // forward_data is the tanh output
        let t = forward_data[idx];
        grad_input[idx] = grad_output[idx] * (1.0 - t * t);
    }
}

@compute @workgroup_size(256)
fn swish_backward(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.size) {
        let x = forward_data[idx];
        let s = 1.0 / (1.0 + exp(-x));
        grad_input[idx] = grad_output[idx] * (s + x * s * (1.0 - s));
    }
}

@compute @workgroup_size(256)
fn gelu_backward(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.size) {
        let x = forward_data[idx];
        let cdf = 0.5 * (1.0 + tanh(sqrt(2.0 / PI) * (x + 0.044715 * x * x * x)));
        let pdf = 0.3989422804 * exp(-0.5 * x * x);
        grad_input[idx] = grad_output[idx] * (cdf + x * pdf);
    }
}

@compute @workgroup_size(256)
fn mish_backward(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.size) {
        let x = forward_data[idx];
        let sp = log(1.0 + exp(x));
        let th = tanh(sp);
        let sig = 1.0 / (1.0 + exp(-x));
        grad_input[idx] = grad_output[idx] * (th + x * sig * (1.0 - th * th));
    }
}

@compute @workgroup_size(256)
fn softplus_backward(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.size) {
        grad_input[idx] = grad_output[idx] / (1.0 + exp(-forward_data[idx]));
    }
}

@compute @workgroup_size(256)
fn hardswish_backward(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.size) {
        let x = forward_data[idx];
        var grad: f32 = 0.0;
        if (x <= -3.0) { grad = 0.0; }
        else if (x >= 3.0) { grad = 1.0; }
        else { grad = (2.0 * x + 3.0) / 6.0; }
        grad_input[idx] = grad_output[idx] * grad;
    }
}

@compute @workgroup_size(256)
fn hardsigmoid_backward(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.size) {
        let x = forward_data[idx];
        grad_input[idx] = grad_output[idx] * select(0.0, 1.0 / 6.0, x > -3.0 && x < 3.0);
    }
}

@compute @workgroup_size(256)
fn hardtanh_backward(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.size) {
        let x = forward_data[idx];
        // params.alpha = minVal, params.beta = maxVal
        grad_input[idx] = grad_output[idx] * select(0.0, 1.0, x >= params.alpha && x <= params.beta);
    }
}
";

    /// <summary>
    /// Loss function backward pass kernels.
    /// </summary>
    public const string LossBackwardSource = @"
@group(0) @binding(0) var<storage, read> predictions: array<f32>;
@group(0) @binding(1) var<storage, read> targets: array<f32>;
@group(0) @binding(2) var<storage, read_write> grad_input: array<f32>;

struct LossParams {
    size: u32,
    inv_size: f32,
    param1: f32,
    param2: f32,
}
@group(0) @binding(3) var<uniform> params: LossParams;

@compute @workgroup_size(256)
fn mse_backward(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.size) {
        grad_input[idx] = 2.0 * (predictions[idx] - targets[idx]) * params.inv_size;
    }
}

@compute @workgroup_size(256)
fn mae_backward(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.size) {
        grad_input[idx] = sign(predictions[idx] - targets[idx]) * params.inv_size;
    }
}

@compute @workgroup_size(256)
fn huber_backward(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.size) {
        let d = predictions[idx] - targets[idx];
        let delta = params.param1;
        if (abs(d) < delta) {
            grad_input[idx] = d / delta * params.inv_size;
        } else {
            grad_input[idx] = sign(d) * params.inv_size;
        }
    }
}

@compute @workgroup_size(256)
fn bce_backward(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.size) {
        let p = clamp(predictions[idx], 1e-7, 1.0 - 1e-7);
        let t = targets[idx];
        grad_input[idx] = ((p - t) / (p * (1.0 - p))) * params.inv_size;
    }
}

@compute @workgroup_size(256)
fn focal_backward(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.size) {
        let p = clamp(predictions[idx], 1e-7, 1.0 - 1e-7);
        let t = targets[idx];
        let alpha_param = params.param1;
        let gamma_param = params.param2;
        let pt = select(1.0 - p, p, t > 0.5);
        let sign_val = select(-1.0, 1.0, t > 0.5);
        grad_input[idx] = sign_val * alpha_param * pow(1.0 - pt, gamma_param) * (gamma_param * log(pt) / (1.0 - pt + 1e-7) - 1.0 / (pt + 1e-7)) * params.inv_size;
    }
}

@compute @workgroup_size(256)
fn logcosh_backward(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.size) {
        grad_input[idx] = tanh(predictions[idx] - targets[idx]) * params.inv_size;
    }
}

@compute @workgroup_size(256)
fn quantile_backward(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.size) {
        let d = targets[idx] - predictions[idx];
        let q = params.param1;
        grad_input[idx] = select(1.0 - q, -q, d >= 0.0) * params.inv_size;
    }
}

@compute @workgroup_size(256)
fn hinge_backward(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.size) {
        let p = predictions[idx];
        let t = targets[idx];
        grad_input[idx] = select(0.0, -t, t * p < 1.0) * params.inv_size;
    }
}

@compute @workgroup_size(256)
fn squared_hinge_backward(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.size) {
        let p = predictions[idx];
        let t = targets[idx];
        let h = max(0.0, 1.0 - t * p);
        grad_input[idx] = select(0.0, -2.0 * h * t, h > 0.0) * params.inv_size;
    }
}

@compute @workgroup_size(256)
fn poisson_backward(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.size) {
        grad_input[idx] = (exp(predictions[idx]) - targets[idx]) * params.inv_size;
    }
}

@compute @workgroup_size(256)
fn exponential_backward(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.size) {
        grad_input[idx] = -targets[idx] * exp(-targets[idx] * predictions[idx]) * params.inv_size;
    }
}

@compute @workgroup_size(256)
fn modified_huber_backward(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.size) {
        let p = predictions[idx];
        let t = targets[idx];
        let y = t * p;
        var grad: f32 = 0.0;
        if (y >= 1.0) { grad = 0.0; }
        else if (y >= -1.0) { grad = -2.0 * (1.0 - y) * t; }
        else { grad = -4.0 * t; }
        grad_input[idx] = grad * params.inv_size;
    }
}

@compute @workgroup_size(256)
fn categorical_ce_backward(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.size) {
        let p = clamp(predictions[idx], 1e-7, 1.0);
        grad_input[idx] = (-targets[idx] / p) * params.inv_size;
    }
}

@compute @workgroup_size(256)
fn charbonnier_backward(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.size) {
        let d = predictions[idx] - targets[idx];
        let eps = params.param1;
        grad_input[idx] = (d / sqrt(d * d + eps * eps)) * params.inv_size;
    }
}

@compute @workgroup_size(256)
fn elastic_net_backward(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.size) {
        let d = predictions[idx] - targets[idx];
        let l1w = params.param1;
        let l2w = params.param2;
        grad_input[idx] = (l1w * sign(d) + 2.0 * l2w * d) * params.inv_size;
    }
}
";

    /// <summary>
    /// Softmax backward pass kernel.
    /// </summary>
    public const string SoftmaxBackwardSource = @"
@group(0) @binding(0) var<storage, read> grad_output: array<f32>;
@group(0) @binding(1) var<storage, read> softmax_output: array<f32>;
@group(0) @binding(2) var<storage, read_write> grad_input: array<f32>;

struct Params {
    batch_size: u32,
    features: u32,
}
@group(0) @binding(3) var<uniform> params: Params;

var<workgroup> shared_dot: f32;
var<workgroup> local_data: array<f32, 256>;

@compute @workgroup_size(256)
fn softmax_backward(@builtin(global_invocation_id) gid: vec3<u32>,
                    @builtin(local_invocation_id) lid: vec3<u32>,
                    @builtin(workgroup_id) wid: vec3<u32>) {
    let batch_idx = wid.x;
    let local_idx = lid.x;
    let base = batch_idx * params.features;

    // Compute dot product of grad_output and softmax_output
    var local_dot: f32 = 0.0;
    for (var i: u32 = local_idx; i < params.features; i = i + 256u) {
        local_dot = local_dot + grad_output[base + i] * softmax_output[base + i];
    }
    local_data[local_idx] = local_dot;
    workgroupBarrier();

    for (var stride: u32 = 128u; stride > 0u; stride = stride >> 1u) {
        if (local_idx < stride) {
            local_data[local_idx] = local_data[local_idx] + local_data[local_idx + stride];
        }
        workgroupBarrier();
    }

    if (local_idx == 0u) {
        shared_dot = local_data[0];
    }
    workgroupBarrier();

    // grad_input = softmax_output * (grad_output - dot)
    for (var i: u32 = local_idx; i < params.features; i = i + 256u) {
        grad_input[base + i] = softmax_output[base + i] * (grad_output[base + i] - shared_dot);
    }
}
";

    /// <summary>
    /// Additional optimizer kernels (RMSProp, Adagrad, NAG, Adadelta, AMSGrad, Adamax, Lion, Nadam).
    /// </summary>
    public const string AdditionalOptimizerSource = @"
@group(0) @binding(0) var<storage, read_write> params_arr: array<f32>;
@group(0) @binding(1) var<storage, read> gradients: array<f32>;
@group(0) @binding(2) var<storage, read_write> state1: array<f32>;
@group(0) @binding(3) var<storage, read_write> state2: array<f32>;

struct OptimizerParams {
    size: u32,
    lr: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    weight_decay: f32,
    t: f32,
    extra: f32,
}
@group(0) @binding(4) var<uniform> opt_params: OptimizerParams;

@compute @workgroup_size(256)
fn rmsprop(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < opt_params.size) {
        let grad = gradients[idx] + opt_params.weight_decay * params_arr[idx];
        // state1 = squared_avg, beta1 = rho
        state1[idx] = opt_params.beta1 * state1[idx] + (1.0 - opt_params.beta1) * grad * grad;
        params_arr[idx] = params_arr[idx] - opt_params.lr * grad / (sqrt(state1[idx]) + opt_params.epsilon);
    }
}

@compute @workgroup_size(256)
fn adagrad(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < opt_params.size) {
        let grad = gradients[idx] + opt_params.weight_decay * params_arr[idx];
        state1[idx] = state1[idx] + grad * grad;
        params_arr[idx] = params_arr[idx] - opt_params.lr * grad / (sqrt(state1[idx]) + opt_params.epsilon);
    }
}

@compute @workgroup_size(256)
fn nag(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < opt_params.size) {
        let grad = gradients[idx] + opt_params.weight_decay * params_arr[idx];
        let v_prev = state1[idx];
        state1[idx] = opt_params.beta1 * state1[idx] - opt_params.lr * grad;
        params_arr[idx] = params_arr[idx] + (-opt_params.beta1 * v_prev + (1.0 + opt_params.beta1) * state1[idx]);
    }
}

@compute @workgroup_size(256)
fn adadelta(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < opt_params.size) {
        let grad = gradients[idx] + opt_params.weight_decay * params_arr[idx];
        // state1 = accum_grad, state2 = accum_update, beta1 = rho
        state1[idx] = opt_params.beta1 * state1[idx] + (1.0 - opt_params.beta1) * grad * grad;
        let update = sqrt(state2[idx] + opt_params.epsilon) / sqrt(state1[idx] + opt_params.epsilon) * grad;
        state2[idx] = opt_params.beta1 * state2[idx] + (1.0 - opt_params.beta1) * update * update;
        params_arr[idx] = params_arr[idx] - update;
    }
}

@compute @workgroup_size(256)
fn adamax(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < opt_params.size) {
        let grad = gradients[idx] + opt_params.weight_decay * params_arr[idx];
        let bc1 = 1.0 - pow(opt_params.beta1, opt_params.t);
        state1[idx] = opt_params.beta1 * state1[idx] + (1.0 - opt_params.beta1) * grad;
        state2[idx] = max(opt_params.beta2 * state2[idx], abs(grad));
        params_arr[idx] = params_arr[idx] - opt_params.lr / bc1 * state1[idx] / (state2[idx] + opt_params.epsilon);
    }
}

@compute @workgroup_size(256)
fn lion(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < opt_params.size) {
        let update = sign(opt_params.beta1 * state1[idx] + (1.0 - opt_params.beta1) * gradients[idx]);
        params_arr[idx] = params_arr[idx] - opt_params.lr * (update + opt_params.weight_decay * params_arr[idx]);
        state1[idx] = opt_params.beta2 * state1[idx] + (1.0 - opt_params.beta2) * gradients[idx];
    }
}

@compute @workgroup_size(256)
fn nadam(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < opt_params.size) {
        let grad = gradients[idx] + opt_params.weight_decay * params_arr[idx];
        let bc1 = 1.0 - pow(opt_params.beta1, opt_params.t);
        let bc2 = 1.0 - pow(opt_params.beta2, opt_params.t);
        state1[idx] = opt_params.beta1 * state1[idx] + (1.0 - opt_params.beta1) * grad;
        state2[idx] = opt_params.beta2 * state2[idx] + (1.0 - opt_params.beta2) * grad * grad;
        let m_hat = state1[idx] / bc1;
        let v_hat = state2[idx] / bc2;
        params_arr[idx] = params_arr[idx] - opt_params.lr * (opt_params.beta1 * m_hat + (1.0 - opt_params.beta1) * grad / bc1) / (sqrt(v_hat) + opt_params.epsilon);
    }
}
";

    /// <summary>
    /// AMSGrad optimizer (needs 5 buffers: params, gradients, m, v, v_max).
    /// </summary>
    public const string AmsgradOptimizerSource = @"
@group(0) @binding(0) var<storage, read_write> params_arr: array<f32>;
@group(0) @binding(1) var<storage, read> gradients: array<f32>;
@group(0) @binding(2) var<storage, read_write> m: array<f32>;
@group(0) @binding(3) var<storage, read_write> v: array<f32>;

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

// Note: v_max is stored as the second half of the v buffer (offset by size)
// This avoids needing a 5th storage buffer binding
@compute @workgroup_size(256)
fn amsgrad(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < opt_params.size) {
        let grad = gradients[idx] + opt_params.weight_decay * params_arr[idx];
        let bc1 = 1.0 - pow(opt_params.beta1, opt_params.t);
        let bc2 = 1.0 - pow(opt_params.beta2, opt_params.t);
        m[idx] = opt_params.beta1 * m[idx] + (1.0 - opt_params.beta1) * grad;
        v[idx] = opt_params.beta2 * v[idx] + (1.0 - opt_params.beta2) * grad * grad;
        // v_max stored at offset size
        let vm_idx = idx + opt_params.size;
        v[vm_idx] = max(v[vm_idx], v[idx]);
        params_arr[idx] = params_arr[idx] - opt_params.lr * (m[idx] / bc1) / (sqrt(v[vm_idx] / bc2) + opt_params.epsilon);
    }
}
";

    /// <summary>
    /// AMSGrad optimizer with 5 separate storage buffers (params, gradients, m, v, v_max).
    /// Uses Dispatch5BufferAsync so uniform is at binding(5).
    /// </summary>
    public const string Amsgrad5BufferOptimizerSource = @"
@group(0) @binding(0) var<storage, read_write> params_arr: array<f32>;
@group(0) @binding(1) var<storage, read> gradients: array<f32>;
@group(0) @binding(2) var<storage, read_write> m: array<f32>;
@group(0) @binding(3) var<storage, read_write> v: array<f32>;
@group(0) @binding(4) var<storage, read_write> v_max: array<f32>;

struct OptimizerParams {
    size: u32,
    lr: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    weight_decay: f32,
    t: f32,
    _pad: f32,
}
@group(0) @binding(5) var<uniform> opt_params: OptimizerParams;

@compute @workgroup_size(256)
fn amsgrad5(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < opt_params.size) {
        let grad = gradients[idx] + opt_params.weight_decay * params_arr[idx];
        let bc1 = 1.0 - pow(opt_params.beta1, opt_params.t);
        let bc2 = 1.0 - pow(opt_params.beta2, opt_params.t);
        m[idx] = opt_params.beta1 * m[idx] + (1.0 - opt_params.beta1) * grad;
        v[idx] = opt_params.beta2 * v[idx] + (1.0 - opt_params.beta2) * grad * grad;
        v_max[idx] = max(v_max[idx], v[idx]);
        params_arr[idx] = params_arr[idx] - opt_params.lr * (m[idx] / bc1) / (sqrt(v_max[idx] / bc2) + opt_params.epsilon);
    }
}
";

    /// <summary>
    /// Dropout forward and backward kernels.
    /// </summary>
    public const string DropoutSource = @"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<storage, read> mask: array<f32>;

struct DropoutParams {
    size: u32,
    scale: f32,
}
@group(0) @binding(3) var<uniform> params: DropoutParams;

@compute @workgroup_size(256)
fn dropout_forward(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.size) {
        output[idx] = input[idx] * mask[idx] * params.scale;
    }
}

@compute @workgroup_size(256)
fn dropout_backward(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.size) {
        output[idx] = input[idx] * mask[idx] * params.scale;
    }
}
";

    /// <summary>
    /// Embedding lookup and backward scatter kernels.
    /// </summary>
    public const string EmbeddingSource = @"
@group(0) @binding(0) var<storage, read> weights: array<f32>;
@group(0) @binding(1) var<storage, read> indices: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

struct EmbedParams {
    num_indices: u32,
    embedding_dim: u32,
}
@group(0) @binding(3) var<uniform> params: EmbedParams;

@compute @workgroup_size(256)
fn embedding_forward(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = params.num_indices * params.embedding_dim;
    if (idx < total) {
        let token_idx = idx / params.embedding_dim;
        let dim_idx = idx % params.embedding_dim;
        let word_idx = bitcast<u32>(bitcast<i32>(indices[token_idx]));
        output[idx] = weights[word_idx * params.embedding_dim + dim_idx];
    }
}
";

    /// <summary>
    /// Instance normalization kernel.
    /// </summary>
    public const string InstanceNormSource = @"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> gamma: array<f32>;
@group(0) @binding(2) var<storage, read> beta: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

struct Params {
    batch_size: u32,
    channels: u32,
    spatial_size: u32,
    epsilon: f32,
}
@group(0) @binding(4) var<uniform> params: Params;

var<workgroup> shared_mean: f32;
var<workgroup> shared_var: f32;
var<workgroup> local_data: array<f32, 256>;

@compute @workgroup_size(256)
fn instance_norm(@builtin(global_invocation_id) gid: vec3<u32>,
                 @builtin(local_invocation_id) lid: vec3<u32>,
                 @builtin(workgroup_id) wid: vec3<u32>) {
    let instance_idx = wid.x;
    let batch = instance_idx / params.channels;
    let channel = instance_idx % params.channels;
    let local_idx = lid.x;
    let base = (batch * params.channels + channel) * params.spatial_size;

    // Compute mean
    var local_sum: f32 = 0.0;
    for (var i: u32 = local_idx; i < params.spatial_size; i = i + 256u) {
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
        shared_mean = local_data[0] / f32(params.spatial_size);
    }
    workgroupBarrier();

    // Compute variance
    var local_var: f32 = 0.0;
    for (var i: u32 = local_idx; i < params.spatial_size; i = i + 256u) {
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
        shared_var = local_data[0] / f32(params.spatial_size);
    }
    workgroupBarrier();

    // Normalize
    let inv_std = 1.0 / sqrt(shared_var + params.epsilon);
    for (var i: u32 = local_idx; i < params.spatial_size; i = i + 256u) {
        let normalized = (input[base + i] - shared_mean) * inv_std;
        output[base + i] = gamma[channel] * normalized + beta[channel];
    }
}
";

    /// <summary>
    /// RMS normalization kernel.
    /// </summary>
    public const string RMSNormSource = @"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> gamma: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

struct Params {
    batch_size: u32,
    feature_size: u32,
    epsilon: f32,
}
@group(0) @binding(3) var<uniform> params: Params;

var<workgroup> shared_rms: f32;
var<workgroup> local_data: array<f32, 256>;

@compute @workgroup_size(256)
fn rms_norm(@builtin(global_invocation_id) gid: vec3<u32>,
            @builtin(local_invocation_id) lid: vec3<u32>,
            @builtin(workgroup_id) wid: vec3<u32>) {
    let batch_idx = wid.x;
    let local_idx = lid.x;
    let base = batch_idx * params.feature_size;

    // Compute sum of squares
    var local_sum: f32 = 0.0;
    for (var i: u32 = local_idx; i < params.feature_size; i = i + 256u) {
        let v = input[base + i];
        local_sum = local_sum + v * v;
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
        shared_rms = 1.0 / sqrt(local_data[0] / f32(params.feature_size) + params.epsilon);
    }
    workgroupBarrier();

    for (var i: u32 = local_idx; i < params.feature_size; i = i + 256u) {
        output[base + i] = gamma[i] * input[base + i] * shared_rms;
    }
}
";

    /// <summary>
    /// Statistics kernels: argmax, argmin along axis.
    /// </summary>
    public const string StatisticsSource = @"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

struct Params {
    outer_size: u32,
    reduce_size: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn mean_axis(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.outer_size) {
        var sum: f32 = 0.0;
        for (var j: u32 = 0u; j < params.reduce_size; j = j + 1u) {
            sum = sum + input[idx * params.reduce_size + j];
        }
        output[idx] = sum / f32(params.reduce_size);
    }
}

@compute @workgroup_size(256)
fn sum_axis(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.outer_size) {
        var sum: f32 = 0.0;
        for (var j: u32 = 0u; j < params.reduce_size; j = j + 1u) {
            sum = sum + input[idx * params.reduce_size + j];
        }
        output[idx] = sum;
    }
}

@compute @workgroup_size(256)
fn max_axis(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.outer_size) {
        var max_val: f32 = -3.402823e+38;
        for (var j: u32 = 0u; j < params.reduce_size; j = j + 1u) {
            max_val = max(max_val, input[idx * params.reduce_size + j]);
        }
        output[idx] = max_val;
    }
}

@compute @workgroup_size(256)
fn argmax_axis(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.outer_size) {
        var max_val: f32 = -3.402823e+38;
        var max_idx: u32 = 0u;
        for (var j: u32 = 0u; j < params.reduce_size; j = j + 1u) {
            let v = input[idx * params.reduce_size + j];
            if (v > max_val) {
                max_val = v;
                max_idx = j;
            }
        }
        output[idx] = bitcast<f32>(i32(max_idx));
    }
}

@compute @workgroup_size(256)
fn argmin_axis(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.outer_size) {
        var min_val: f32 = 3.402823e+38;
        var min_idx: u32 = 0u;
        for (var j: u32 = 0u; j < params.reduce_size; j = j + 1u) {
            let v = input[idx * params.reduce_size + j];
            if (v < min_val) {
                min_val = v;
                min_idx = j;
            }
        }
        output[idx] = bitcast<f32>(i32(min_idx));
    }
}
";

    /// <summary>
    /// Broadcast multiply operations.
    /// </summary>
    public const string BroadcastSource = @"
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;

struct Params {
    outer_size: u32,
    inner_size: u32,
}
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn broadcast_mul_last(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = params.outer_size * params.inner_size;
    if (idx < total) {
        C[idx] = A[idx] * B[idx % params.inner_size];
    }
}

@compute @workgroup_size(256)
fn broadcast_mul_first(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = params.outer_size * params.inner_size;
    if (idx < total) {
        C[idx] = A[idx] * B[idx / params.inner_size];
    }
}
";

    /// <summary>
    /// Gradient clipping kernels.
    /// </summary>
    public const string GradientClipSource = @"
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> B: array<f32>;

struct ClipParams {
    size: u32,
    clip_value: f32,
    norm: f32,
    max_norm: f32,
}
@group(0) @binding(2) var<uniform> params: ClipParams;

@compute @workgroup_size(256)
fn clip_by_value(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.size) {
        B[idx] = clamp(A[idx], -params.clip_value, params.clip_value);
    }
}

@compute @workgroup_size(256)
fn clip_by_norm(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.size) {
        if (params.norm > params.max_norm) {
            B[idx] = A[idx] * (params.max_norm / params.norm);
        } else {
            B[idx] = A[idx];
        }
    }
}
";

    /// <summary>
    /// Scatter add and gather kernels.
    /// </summary>
    public const string ScatterGatherSource = @"
@group(0) @binding(0) var<storage, read> source: array<f32>;
@group(0) @binding(1) var<storage, read> indices: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

struct Params {
    num_indices: u32,
    feature_size: u32,
}
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn gather(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = params.num_indices * params.feature_size;
    if (idx < total) {
        let token_idx = idx / params.feature_size;
        let feat_idx = idx % params.feature_size;
        let src_idx = bitcast<u32>(bitcast<i32>(indices[token_idx]));
        output[idx] = source[src_idx * params.feature_size + feat_idx];
    }
}
";

    /// <summary>
    /// Conv2D backward pass kernels.
    /// </summary>
    public const string Conv2DBackwardSource = @"
@group(0) @binding(0) var<storage, read> grad_output: array<f32>;
@group(0) @binding(1) var<storage, read> kernel: array<f32>;
@group(0) @binding(2) var<storage, read_write> grad_input: array<f32>;

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
fn conv2d_backward_input(@builtin(global_invocation_id) gid: vec3<u32>) {
    let in_x = gid.x;
    let in_y = gid.y;
    let ic = gid.z % params.in_channels;
    let batch = gid.z / params.in_channels;

    if (in_x >= params.in_width || in_y >= params.in_height || batch >= params.batch_size) {
        return;
    }

    var acc: f32 = 0.0;
    for (var oc: u32 = 0u; oc < params.out_channels; oc = oc + 1u) {
        for (var ky: u32 = 0u; ky < params.kernel_height; ky = ky + 1u) {
            for (var kx: u32 = 0u; kx < params.kernel_width; kx = kx + 1u) {
                let out_y_check = i32(in_y) + i32(params.pad_h) - i32(ky);
                let out_x_check = i32(in_x) + i32(params.pad_w) - i32(kx);
                if (out_y_check >= 0 && out_x_check >= 0 &&
                    out_y_check % i32(params.stride_h) == 0 && out_x_check % i32(params.stride_w) == 0) {
                    let out_y = u32(out_y_check) / params.stride_h;
                    let out_x = u32(out_x_check) / params.stride_w;
                    if (out_y < params.out_height && out_x < params.out_width) {
                        let go_idx = batch * params.out_channels * params.out_height * params.out_width
                                   + oc * params.out_height * params.out_width
                                   + out_y * params.out_width + out_x;
                        let k_idx = oc * params.in_channels * params.kernel_height * params.kernel_width
                                  + ic * params.kernel_height * params.kernel_width
                                  + ky * params.kernel_width + kx;
                        acc = acc + grad_output[go_idx] * kernel[k_idx];
                    }
                }
            }
        }
    }

    let gi_idx = batch * params.in_channels * params.in_height * params.in_width
               + ic * params.in_height * params.in_width
               + in_y * params.in_width + in_x;
    grad_input[gi_idx] = acc;
}
";

    /// <summary>
    /// Conv2D backward kernel weight gradient.
    /// </summary>
    public const string Conv2DBackwardKernelSource = @"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> grad_output: array<f32>;
@group(0) @binding(2) var<storage, read_write> grad_kernel: array<f32>;

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

@compute @workgroup_size(256)
fn conv2d_backward_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = params.out_channels * params.in_channels * params.kernel_height * params.kernel_width;
    if (idx >= total) { return; }

    let kw = idx % params.kernel_width;
    let kh = (idx / params.kernel_width) % params.kernel_height;
    let ic = (idx / (params.kernel_width * params.kernel_height)) % params.in_channels;
    let oc = idx / (params.kernel_width * params.kernel_height * params.in_channels);

    var acc: f32 = 0.0;
    for (var n: u32 = 0u; n < params.batch_size; n = n + 1u) {
        for (var oh: u32 = 0u; oh < params.out_height; oh = oh + 1u) {
            for (var ow: u32 = 0u; ow < params.out_width; ow = ow + 1u) {
                let ih = i32(oh * params.stride_h + kh) - i32(params.pad_h);
                let iw = i32(ow * params.stride_w + kw) - i32(params.pad_w);
                if (ih >= 0 && ih < i32(params.in_height) && iw >= 0 && iw < i32(params.in_width)) {
                    let in_idx = n * params.in_channels * params.in_height * params.in_width
                               + ic * params.in_height * params.in_width
                               + u32(ih) * params.in_width + u32(iw);
                    let go_idx = n * params.out_channels * params.out_height * params.out_width
                               + oc * params.out_height * params.out_width
                               + oh * params.out_width + ow;
                    acc = acc + input[in_idx] * grad_output[go_idx];
                }
            }
        }
    }
    grad_kernel[idx] = acc;
}
";

    /// <summary>
    /// Pool backward pass kernels.
    /// </summary>
    public const string PoolBackwardSource = @"
@group(0) @binding(0) var<storage, read> grad_output: array<f32>;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> grad_input: array<f32>;

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
@group(0) @binding(3) var<uniform> params: PoolParams;

@compute @workgroup_size(8, 8, 1)
fn avg_pool2d_backward(@builtin(global_invocation_id) gid: vec3<u32>) {
    let in_x = gid.x;
    let in_y = gid.y;
    let c = gid.z % params.channels;
    let batch = gid.z / params.channels;

    if (in_x >= params.in_width || in_y >= params.in_height || batch >= params.batch_size) {
        return;
    }

    var acc: f32 = 0.0;
    for (var oh: u32 = 0u; oh < params.out_height; oh = oh + 1u) {
        for (var ow: u32 = 0u; ow < params.out_width; ow = ow + 1u) {
            let in_y_start = oh * params.stride_h;
            let in_x_start = ow * params.stride_w;
            if (in_y >= in_y_start && in_y < in_y_start + params.kernel_height &&
                in_x >= in_x_start && in_x < in_x_start + params.kernel_width) {
                let pool_size = f32(params.kernel_height * params.kernel_width);
                let go_idx = batch * params.channels * params.out_height * params.out_width
                           + c * params.out_height * params.out_width
                           + oh * params.out_width + ow;
                acc = acc + grad_output[go_idx] / pool_size;
            }
        }
    }

    let gi_idx = batch * params.channels * params.in_height * params.in_width
               + c * params.in_height * params.in_width
               + in_y * params.in_width + in_x;
    grad_input[gi_idx] = acc;
}

@compute @workgroup_size(8, 8, 1)
fn max_pool2d_backward(@builtin(global_invocation_id) gid: vec3<u32>) {
    let out_x = gid.x;
    let out_y = gid.y;
    let c = gid.z % params.channels;
    let batch = gid.z / params.channels;

    if (out_x >= params.out_width || out_y >= params.out_height || batch >= params.batch_size) {
        return;
    }

    // Find the max input position for this output position
    var max_val: f32 = -3.402823e+38;
    var max_ih: u32 = 0u;
    var max_iw: u32 = 0u;

    for (var ky: u32 = 0u; ky < params.kernel_height; ky = ky + 1u) {
        for (var kx: u32 = 0u; kx < params.kernel_width; kx = kx + 1u) {
            let in_y = i32(out_y * params.stride_h + ky) - i32(params.pad_h);
            let in_x = i32(out_x * params.stride_w + kx) - i32(params.pad_w);
            if (in_y >= 0 && in_y < i32(params.in_height) && in_x >= 0 && in_x < i32(params.in_width)) {
                let in_idx = batch * params.channels * params.in_height * params.in_width
                           + c * params.in_height * params.in_width
                           + u32(in_y) * params.in_width + u32(in_x);
                if (input[in_idx] > max_val) {
                    max_val = input[in_idx];
                    max_ih = u32(in_y);
                    max_iw = u32(in_x);
                }
            }
        }
    }

    // Route gradient to the max position (atomic add not available, use direct write)
    let go_idx = batch * params.channels * params.out_height * params.out_width
               + c * params.out_height * params.out_width
               + out_y * params.out_width + out_x;
    let gi_idx = batch * params.channels * params.in_height * params.in_width
               + c * params.in_height * params.in_width
               + max_ih * params.in_width + max_iw;
    // Note: This has race conditions if multiple output positions map to the same max input.
    // For production, an atomic add or a different approach would be needed.
    grad_input[gi_idx] = grad_input[gi_idx] + grad_output[go_idx];
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
               LossSource + OptimizerSource + TransposeSource + ConvolutionSource + PoolingSource +
               AdditionalUnarySource + InverseHyperbolicSource + FillSource + ClampSource +
               FmaSource + BiasAddSource + Conv2DBiasAddSource + ActivationBackwardSource +
               LossBackwardSource + SoftmaxBackwardSource + AdditionalOptimizerSource +
               AmsgradOptimizerSource + Amsgrad5BufferOptimizerSource +
               DropoutSource + EmbeddingSource + InstanceNormSource +
               RMSNormSource + StatisticsSource + BroadcastSource + GradientClipSource +
               ScatterGatherSource + Conv2DBackwardSource + Conv2DBackwardKernelSource +
               PoolBackwardSource;
    }
}
#endif
