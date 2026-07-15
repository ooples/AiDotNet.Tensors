namespace AiDotNet.Tensors.Engines.DirectGpu.Metal;

// #775: Metal (MSL) mirrors of the OpenCL extended-conv/geometry kernels. Kept in one library compiled
// by MetalBackend; the per-family capability interfaces are implemented in MetalBackend.ExtendedConv.cs.
// The per-element arithmetic is kept byte-identical to the OpenCL reference so the cross-backend
// source-parity tests can assert it; only the kernel signature/dispatch differs per shader language.
internal static class MetalExtendedConvKernels
{
    public const string Source = @"
#include <metal_stdlib>
using namespace metal;

kernel void trilinear_interpolate(
    device const float* grid [[buffer(0)]],
    device const float* positions [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant int& D [[buffer(3)]],
    constant int& H [[buffer(4)]],
    constant int& W [[buffer(5)]],
    constant int& C [[buffer(6)]],
    constant int& P [[buffer(7)]],
    constant float& upperEps [[buffer(8)]],
    uint gid [[thread_position_in_grid]])
{
    int idx = int(gid);
    if (idx >= P * C) return;
    int c = idx % C;
    int n = idx / C;
    float z = fmax(0.0f, fmin((float)(D - 1) - upperEps, positions[n * 3 + 0]));
    float y = fmax(0.0f, fmin((float)(H - 1) - upperEps, positions[n * 3 + 1]));
    float x = fmax(0.0f, fmin((float)(W - 1) - upperEps, positions[n * 3 + 2]));
    int z0 = (int)floor(z), y0 = (int)floor(y), x0 = (int)floor(x);
    int z1 = min(z0 + 1, D - 1), y1 = min(y0 + 1, H - 1), x1 = min(x0 + 1, W - 1);
    float fz = z - z0, fy = y - y0, fx = x - x0;
    float w000 = (1 - fz) * (1 - fy) * (1 - fx), w001 = (1 - fz) * (1 - fy) * fx;
    float w010 = (1 - fz) * fy * (1 - fx),       w011 = (1 - fz) * fy * fx;
    float w100 = fz * (1 - fy) * (1 - fx),       w101 = fz * (1 - fy) * fx;
    float w110 = fz * fy * (1 - fx),             w111 = fz * fy * fx;
    output[n * C + c] =
        w000 * grid[(((z0 * H + y0) * W + x0) * C) + c] + w001 * grid[(((z0 * H + y0) * W + x1) * C) + c] +
        w010 * grid[(((z0 * H + y1) * W + x0) * C) + c] + w011 * grid[(((z0 * H + y1) * W + x1) * C) + c] +
        w100 * grid[(((z1 * H + y0) * W + x0) * C) + c] + w101 * grid[(((z1 * H + y0) * W + x1) * C) + c] +
        w110 * grid[(((z1 * H + y1) * W + x0) * C) + c] + w111 * grid[(((z1 * H + y1) * W + x1) * C) + c];
}

kernel void trilinear_interpolate_backward(
    device const float* gradOutput [[buffer(0)]],
    device const float* positions [[buffer(1)]],
    device float* gradGrid [[buffer(2)]],
    constant int& D [[buffer(3)]],
    constant int& H [[buffer(4)]],
    constant int& W [[buffer(5)]],
    constant int& C [[buffer(6)]],
    constant int& P [[buffer(7)]],
    constant float& upperEps [[buffer(8)]],
    uint gid [[thread_position_in_grid]])
{
    int idx = int(gid);
    if (idx >= D * H * W * C) return;
    int c = idx % C;
    int gx = (idx / C) % W;
    int gy = (idx / (C * W)) % H;
    int gz = idx / (C * W * H);
    float sum = 0.0f;
    for (int n = 0; n < P; n++) {
        float z = fmax(0.0f, fmin((float)(D - 1) - upperEps, positions[n * 3 + 0]));
        float y = fmax(0.0f, fmin((float)(H - 1) - upperEps, positions[n * 3 + 1]));
        float x = fmax(0.0f, fmin((float)(W - 1) - upperEps, positions[n * 3 + 2]));
        int z0 = (int)floor(z), y0 = (int)floor(y), x0 = (int)floor(x);
        int z1 = min(z0 + 1, D - 1), y1 = min(y0 + 1, H - 1), x1 = min(x0 + 1, W - 1);
        float fz = z - z0, fy = y - y0, fx = x - x0;
        float wz = (gz == z0 ? (1.0f - fz) : 0.0f) + (gz == z1 ? fz : 0.0f);
        if (wz == 0.0f) continue;
        float wy = (gy == y0 ? (1.0f - fy) : 0.0f) + (gy == y1 ? fy : 0.0f);
        if (wy == 0.0f) continue;
        float wx = (gx == x0 ? (1.0f - fx) : 0.0f) + (gx == x1 ? fx : 0.0f);
        sum += wz * wy * wx * gradOutput[n * C + c];
    }
    gradGrid[idx] = sum;
}
";
}
