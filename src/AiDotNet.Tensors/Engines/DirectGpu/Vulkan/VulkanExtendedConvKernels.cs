namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

// #775: Vulkan (GLSL compute) mirrors of the OpenCL extended-conv/geometry kernels. Each kernel is a
// standalone GLSL shader string (compiled to SPIR-V on demand by GlslDispatchN). The per-element
// arithmetic is kept as close to the OpenCL reference as GLSL allows (GLSL float literals drop the `f`
// suffix and fmin/fmax/fabs become min/max/abs), so the source-parity tests assert GLSL-form markers for
// the Vulkan rows. Int params travel in the push-constant block; a float param (upperEps) is passed as
// its raw bit pattern in the same uint[] and declared `float` in the block.
internal static class VulkanExtendedConvKernels
{
    public const string TrilinearInterpolate = @"#version 450
layout(local_size_x = 256) in;
layout(set=0,binding=0) readonly buffer Grid { float grid[]; };
layout(set=0,binding=1) readonly buffer Positions { float positions[]; };
layout(set=0,binding=2) writeonly buffer Output { float output_[]; };
layout(push_constant) uniform PC { int D; int H; int W; int C; int P; float upperEps; };
void main() {
    int idx = int(gl_GlobalInvocationID.x);
    if (idx >= P * C) return;
    int c = idx % C;
    int n = idx / C;
    float z = max(0.0, min(float(D - 1) - upperEps, positions[n * 3 + 0]));
    float y = max(0.0, min(float(H - 1) - upperEps, positions[n * 3 + 1]));
    float x = max(0.0, min(float(W - 1) - upperEps, positions[n * 3 + 2]));
    int z0 = int(floor(z)), y0 = int(floor(y)), x0 = int(floor(x));
    int z1 = min(z0 + 1, D - 1), y1 = min(y0 + 1, H - 1), x1 = min(x0 + 1, W - 1);
    float fz = z - float(z0), fy = y - float(y0), fx = x - float(x0);
    float w000 = (1 - fz) * (1 - fy) * (1 - fx), w001 = (1 - fz) * (1 - fy) * fx;
    float w010 = (1 - fz) * fy * (1 - fx),       w011 = (1 - fz) * fy * fx;
    float w100 = fz * (1 - fy) * (1 - fx),       w101 = fz * (1 - fy) * fx;
    float w110 = fz * fy * (1 - fx),             w111 = fz * fy * fx;
    output_[n * C + c] =
        w000 * grid[(((z0 * H + y0) * W + x0) * C) + c] + w001 * grid[(((z0 * H + y0) * W + x1) * C) + c] +
        w010 * grid[(((z0 * H + y1) * W + x0) * C) + c] + w011 * grid[(((z0 * H + y1) * W + x1) * C) + c] +
        w100 * grid[(((z1 * H + y0) * W + x0) * C) + c] + w101 * grid[(((z1 * H + y0) * W + x1) * C) + c] +
        w110 * grid[(((z1 * H + y1) * W + x0) * C) + c] + w111 * grid[(((z1 * H + y1) * W + x1) * C) + c];
}";

    public const string TrilinearInterpolateBackward = @"#version 450
layout(local_size_x = 256) in;
layout(set=0,binding=0) readonly buffer GradOutput { float gradOutput[]; };
layout(set=0,binding=1) readonly buffer Positions { float positions[]; };
layout(set=0,binding=2) writeonly buffer GradGrid { float gradGrid[]; };
layout(push_constant) uniform PC { int D; int H; int W; int C; int P; float upperEps; };
void main() {
    int idx = int(gl_GlobalInvocationID.x);
    if (idx >= D * H * W * C) return;
    int c = idx % C;
    int gx = (idx / C) % W;
    int gy = (idx / (C * W)) % H;
    int gz = idx / (C * W * H);
    float sum = 0.0;
    for (int n = 0; n < P; n++) {
        float z = max(0.0, min(float(D - 1) - upperEps, positions[n * 3 + 0]));
        float y = max(0.0, min(float(H - 1) - upperEps, positions[n * 3 + 1]));
        float x = max(0.0, min(float(W - 1) - upperEps, positions[n * 3 + 2]));
        int z0 = int(floor(z)), y0 = int(floor(y)), x0 = int(floor(x));
        int z1 = min(z0 + 1, D - 1), y1 = min(y0 + 1, H - 1), x1 = min(x0 + 1, W - 1);
        float fz = z - float(z0), fy = y - float(y0), fx = x - float(x0);
        float wz = (gz == z0 ? (1.0 - fz) : 0.0) + (gz == z1 ? fz : 0.0);
        if (wz == 0.0) continue;
        float wy = (gy == y0 ? (1.0 - fy) : 0.0) + (gy == y1 ? fy : 0.0);
        if (wy == 0.0) continue;
        float wx = (gx == x0 ? (1.0 - fx) : 0.0) + (gx == x1 ? fx : 0.0);
        sum += wz * wy * wx * gradOutput[n * C + c];
    }
    gradGrid[idx] = sum;
}";
}
