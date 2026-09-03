namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels;

public static class CudaInstantNgpKernels
{
    public static string[] GetKernelNames() =>
        ["instant_ngp_hash_encode_level", "instant_ngp_hash_encode_level_backward",
            "frexp_decompose", "unique_consecutive_compact", "unique_consecutive_with_info", "unique_sorted_with_info", "project_gaussians_3d_to_2d", "sample_rays_with_occupancy", "resident_mode", "resident_indices_to_int32", "resident_index_add", "resident_index_select", "resident_scatter_max_argmax_rows", "resident_uniform_mesh_laplacian", "resident_scatter_add_rows", "resident_scatter_mean_rows_counts", "resident_scatter_softmax_rows", "resident_scatter_add_backward_rows", "resident_scatter_mean_backward_rows", "resident_scatter_max_backward_rows", "resident_scatter_softmax_backward_rows", "nonzero_compact", "ctc_loss_forward", "importance_sampling", "resident_nms",
            "generate_spiral_indices"];

    public static string GetSource() => @"
#include <math.h>

__device__ __forceinline__ unsigned int instant_ngp_hash(int x, int y, int z, int tableSize)
{
    unsigned int hx = (unsigned int)x * 73856093u;
    unsigned int hy = (unsigned int)y * 19349663u;
    unsigned int hz = (unsigned int)z * 83492791u;
    return (hx ^ hy ^ hz) % (unsigned int)tableSize;
}

__device__ __forceinline__ float instant_ngp_clamp_position(float value)
{
    if (value <= 0.0f) return 0.0f;
    if (value >= 0.999999f) return 0.999999f;
    return value;
}

extern ""C"" __global__ __launch_bounds__(256) void instant_ngp_hash_encode_level(
    const float* __restrict__ positions,
    const float* __restrict__ hashTable,
    float* __restrict__ output,
    int numPoints, int resolution, int tableSize, int featuresPerLevel,
    int levelOffset, int outputStride)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = numPoints * featuresPerLevel;
    if (gid >= total) return;
    int n = gid / featuresPerLevel;
    int f = gid - n * featuresPerLevel;
    float gx = instant_ngp_clamp_position(positions[n * 3]) * (float)resolution;
    float gy = instant_ngp_clamp_position(positions[n * 3 + 1]) * (float)resolution;
    float gz = instant_ngp_clamp_position(positions[n * 3 + 2]) * (float)resolution;
    int x0 = (int)floorf(gx), y0 = (int)floorf(gy), z0 = (int)floorf(gz);
    int x1 = x0 + 1, y1 = y0 + 1, z1 = z0 + 1;
    float fx = gx - (float)x0, fy = gy - (float)y0, fz = gz - (float)z0;
    float ix = 1.0f - fx, iy = 1.0f - fy, iz = 1.0f - fz;
    unsigned int h000 = instant_ngp_hash(x0, y0, z0, tableSize);
    unsigned int h001 = instant_ngp_hash(x0, y0, z1, tableSize);
    unsigned int h010 = instant_ngp_hash(x0, y1, z0, tableSize);
    unsigned int h011 = instant_ngp_hash(x0, y1, z1, tableSize);
    unsigned int h100 = instant_ngp_hash(x1, y0, z0, tableSize);
    unsigned int h101 = instant_ngp_hash(x1, y0, z1, tableSize);
    unsigned int h110 = instant_ngp_hash(x1, y1, z0, tableSize);
    unsigned int h111 = instant_ngp_hash(x1, y1, z1, tableSize);
    float value =
        ix * iy * iz * hashTable[h000 * featuresPerLevel + f] +
        ix * iy * fz * hashTable[h001 * featuresPerLevel + f] +
        ix * fy * iz * hashTable[h010 * featuresPerLevel + f] +
        ix * fy * fz * hashTable[h011 * featuresPerLevel + f] +
        fx * iy * iz * hashTable[h100 * featuresPerLevel + f] +
        fx * iy * fz * hashTable[h101 * featuresPerLevel + f] +
        fx * fy * iz * hashTable[h110 * featuresPerLevel + f] +
        fx * fy * fz * hashTable[h111 * featuresPerLevel + f];
    output[n * outputStride + levelOffset + f] = value;
}

extern ""C"" __global__ __launch_bounds__(256) void instant_ngp_hash_encode_level_backward(
    const float* __restrict__ positions,
    const float* __restrict__ outputGradient,
    float* __restrict__ tableGradient,
    int numPoints, int resolution, int tableSize, int featuresPerLevel,
    int levelOffset, int outputStride)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = tableSize * featuresPerLevel;
    if (gid >= total) return;
    unsigned int entry = (unsigned int)(gid / featuresPerLevel);
    int f = gid - (int)entry * featuresPerLevel;
    float acc = 0.0f;
    for (int n = 0; n < numPoints; n++) {
        float gx = instant_ngp_clamp_position(positions[n * 3]) * (float)resolution;
        float gy = instant_ngp_clamp_position(positions[n * 3 + 1]) * (float)resolution;
        float gz = instant_ngp_clamp_position(positions[n * 3 + 2]) * (float)resolution;
        int x0 = (int)floorf(gx), y0 = (int)floorf(gy), z0 = (int)floorf(gz);
        int x1 = x0 + 1, y1 = y0 + 1, z1 = z0 + 1;
        float fx = gx - (float)x0, fy = gy - (float)y0, fz = gz - (float)z0;
        float ix = 1.0f - fx, iy = 1.0f - fy, iz = 1.0f - fz;
        float grad = outputGradient[n * outputStride + levelOffset + f];
        if (fabsf(grad) < 1.0e-10f) continue;
        if (instant_ngp_hash(x0, y0, z0, tableSize) == entry) acc += grad * ix * iy * iz;
        if (instant_ngp_hash(x0, y0, z1, tableSize) == entry) acc += grad * ix * iy * fz;
        if (instant_ngp_hash(x0, y1, z0, tableSize) == entry) acc += grad * ix * fy * iz;
        if (instant_ngp_hash(x0, y1, z1, tableSize) == entry) acc += grad * ix * fy * fz;
        if (instant_ngp_hash(x1, y0, z0, tableSize) == entry) acc += grad * fx * iy * iz;
        if (instant_ngp_hash(x1, y0, z1, tableSize) == entry) acc += grad * fx * iy * fz;
        if (instant_ngp_hash(x1, y1, z0, tableSize) == entry) acc += grad * fx * fy * iz;
        if (instant_ngp_hash(x1, y1, z1, tableSize) == entry) acc += grad * fx * fy * fz;
    }
    tableGradient[gid] = acc;
}

extern ""C"" __global__ void unique_consecutive_compact(
    const float* __restrict__ input,
    float* __restrict__ output,
    float* __restrict__ outputCount,
    int length)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    if (length <= 0) { outputCount[0] = 0.0f; return; }
    int count = 1;
    output[0] = input[0];
    for (int i = 1; i < length; i++) {
        if (input[i] != input[i - 1]) output[count++] = input[i];
    }
    outputCount[0] = (float)count;
}

extern ""C"" __global__ void unique_consecutive_with_info(
    const float* __restrict__ input,
    float* __restrict__ outputValues,
    float* __restrict__ outputInverse,
    float* __restrict__ outputCounts,
    float* __restrict__ outputCount,
    int length, int returnInverse, int returnCounts)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    if (length <= 0) { outputCount[0] = 0.0f; return; }
    int count = 1;
    outputValues[0] = input[0];
    if (returnInverse != 0) outputInverse[0] = 0.0f;
    if (returnCounts != 0) outputCounts[0] = 1.0f;
    for (int i = 1; i < length; i++) {
        if (input[i] != input[i - 1]) {
            outputValues[count] = input[i];
            if (returnCounts != 0) outputCounts[count] = 1.0f;
            count++;
        } else if (returnCounts != 0) {
            outputCounts[count - 1] += 1.0f;
        }
        if (returnInverse != 0) outputInverse[i] = (float)(count - 1);
    }
    outputCount[0] = (float)count;
}

extern ""C"" __global__ void unique_sorted_with_info(
    const float* __restrict__ sortedInput,
    const float* __restrict__ sortedOriginalIndices,
    float* __restrict__ outputValues,
    float* __restrict__ outputInverse,
    float* __restrict__ outputCounts,
    float* __restrict__ outputCount,
    int length, int returnInverse, int returnCounts)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    int count = 0;
    for (int i = 0; i < length; i++) {
        if (i == 0 || sortedInput[i] != sortedInput[i - 1]) {
            outputValues[count] = sortedInput[i];
            if (returnCounts != 0) outputCounts[count] = 0.0f;
            count++;
        }
        int group = count - 1;
        if (returnCounts != 0) outputCounts[group] += 1.0f;
        if (returnInverse != 0)
            outputInverse[(int)sortedOriginalIndices[i]] = (float)group;
    }
    outputCount[0] = (float)count;
}

extern ""C"" __global__ __launch_bounds__(256) void frexp_decompose(
    const float* __restrict__ input,
    float* __restrict__ mantissa,
    float* __restrict__ exponent,
    int length)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= length) return;
    float value = input[gid];
    unsigned int bits = __float_as_uint(value);
    unsigned int magnitude = bits & 0x7fffffffu;
    unsigned int exponentBits = magnitude >> 23;
    if (exponentBits == 0xffu || magnitude == 0u) {
        mantissa[gid] = value;
        exponent[gid] = 0.0f;
        return;
    }
    int scaleAdjustment = 0;
    if (exponentBits == 0u) {
        value *= 16777216.0f;
        bits = __float_as_uint(value);
        exponentBits = (bits & 0x7fffffffu) >> 23;
        scaleAdjustment = -24;
    }
    mantissa[gid] = __uint_as_float((bits & 0x807fffffu) | (126u << 23));
    exponent[gid] = (float)((int)exponentBits - 126 + scaleAdjustment);
}

extern ""C"" __global__ __launch_bounds__(256) void project_gaussians_3d_to_2d(
    const float* __restrict__ means3D,
    const float* __restrict__ covariances3D,
    float* __restrict__ means2D,
    float* __restrict__ covariances2D,
    float* __restrict__ depths,
    float* __restrict__ visible,
    int numGaussians, int covarianceStride, int imageWidth, int imageHeight,
    float v00, float v01, float v02, float v03,
    float v10, float v11, float v12, float v13,
    float v20, float v21, float v22, float v23,
    float p00, float p11)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numGaussians) return;
    means2D[i * 2] = 0.0f; means2D[i * 2 + 1] = 0.0f;
    covariances2D[i * 3] = 0.0f; covariances2D[i * 3 + 1] = 0.0f;
    covariances2D[i * 3 + 2] = 0.0f; depths[i] = 0.0f; visible[i] = 0.0f;

    float mx = means3D[i * 3], my = means3D[i * 3 + 1], mz = means3D[i * 3 + 2];
    float camX = v00 * mx + v01 * my + v02 * mz + v03;
    float camY = v10 * mx + v11 * my + v12 * mz + v13;
    float camZ = v20 * mx + v21 * my + v22 * mz + v23;
    if (camZ <= 0.001f) return;
    float invZ = 1.0f / camZ;
    float cx = (float)imageWidth * 0.5f, cy = (float)imageHeight * 0.5f;
    float screenX = p00 * camX * invZ * cx + cx;
    float screenY = p11 * camY * invZ * cy + cy;
    if (screenX < -(float)imageWidth || screenX > 2.0f * (float)imageWidth ||
        screenY < -(float)imageHeight || screenY > 2.0f * (float)imageHeight) return;

    int c = i * covarianceStride;
    float c00 = covariances3D[c];
    float c01 = covariances3D[c + 1];
    float c02 = covariances3D[c + 2];
    float c11 = covariances3D[c + (covarianceStride == 6 ? 3 : 4)];
    float c12 = covariances3D[c + (covarianceStride == 6 ? 4 : 5)];
    float c22 = covariances3D[c + (covarianceStride == 6 ? 5 : 8)];
    float j00 = p00 * invZ, j02 = -p00 * camX * invZ * invZ;
    float j11 = p11 * invZ, j12 = -p11 * camY * invZ * invZ;
    float cov00 = j00*j00*c00 + 2.0f*j00*j02*c02 + j02*j02*c22 + 0.3f;
    float cov01 = j00*j11*c01 + j00*j12*c02 + j02*j11*c12 + j02*j12*c22;
    float cov11 = j11*j11*c11 + 2.0f*j11*j12*c12 + j12*j12*c22 + 0.3f;
    means2D[i * 2] = screenX; means2D[i * 2 + 1] = screenY;
    covariances2D[i * 3] = cov00; covariances2D[i * 3 + 1] = cov01;
    covariances2D[i * 3 + 2] = cov11; depths[i] = camZ; visible[i] = 1.0f;
}

__device__ __forceinline__ bool sample_ray_axis(
    float origin, float direction, float minimum, float maximum,
    float* tMin, float* tMax)
{
    if (fabsf(direction) < 1.0e-8f)
        return origin >= minimum && origin <= maximum;
    float t1 = (minimum - origin) / direction;
    float t2 = (maximum - origin) / direction;
    if (t1 > t2) { float temporary = t1; t1 = t2; t2 = temporary; }
    *tMin = fmaxf(*tMin, t1); *tMax = fminf(*tMax, t2);
    return *tMax >= *tMin;
}

extern ""C"" __global__ __launch_bounds__(256) void sample_rays_with_occupancy(
    const float* __restrict__ rayOrigins,
    const float* __restrict__ rayDirections,
    const unsigned int* __restrict__ occupancyBitfield,
    float* __restrict__ positions,
    float* __restrict__ directions,
    float* __restrict__ validMask,
    float* __restrict__ tValues,
    int numRays, int occupancyWordCount, int gridSize, int maxSamples,
    float minX, float minY, float minZ, float maxX, float maxY, float maxZ,
    float nearBound, float farBound)
{
    int sampleIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int totalSamples = numRays * maxSamples;
    if (sampleIndex >= totalSamples) return;
    int positionIndex = sampleIndex * 3;
    positions[positionIndex] = 0.0f; positions[positionIndex + 1] = 0.0f;
    positions[positionIndex + 2] = 0.0f; directions[positionIndex] = 0.0f;
    directions[positionIndex + 1] = 0.0f; directions[positionIndex + 2] = 0.0f;
    validMask[sampleIndex] = 0.0f; tValues[sampleIndex] = 0.0f;

    int ray = sampleIndex / maxSamples, sample = sampleIndex - ray * maxSamples;
    float ox = rayOrigins[ray * 3], oy = rayOrigins[ray * 3 + 1], oz = rayOrigins[ray * 3 + 2];
    float dx = rayDirections[ray * 3], dy = rayDirections[ray * 3 + 1], dz = rayDirections[ray * 3 + 2];
    float tMin = nearBound, tMax = farBound;
    if (!sample_ray_axis(ox, dx, minX, maxX, &tMin, &tMax) ||
        !sample_ray_axis(oy, dy, minY, maxY, &tMin, &tMax) ||
        !sample_ray_axis(oz, dz, minZ, maxZ, &tMin, &tMax) || tMax < tMin) return;

    float t = tMin + ((tMax - tMin) / (float)maxSamples) * ((float)sample + 0.5f);
    float px = ox + t * dx, py = oy + t * dy, pz = oz + t * dz;
    float invX = 1.0f / fmaxf(1.0e-10f, maxX - minX);
    float invY = 1.0f / fmaxf(1.0e-10f, maxY - minY);
    float invZ = 1.0f / fmaxf(1.0e-10f, maxZ - minZ);
    float nx = fminf(fmaxf((px - minX) * invX, 0.0f), 0.999999f);
    float ny = fminf(fmaxf((py - minY) * invY, 0.0f), 0.999999f);
    float nz = fminf(fmaxf((pz - minZ) * invZ, 0.0f), 0.999999f);
    int gx = min((int)(nx * (float)gridSize), gridSize - 1);
    int gy = min((int)(ny * (float)gridSize), gridSize - 1);
    int gz = min((int)(nz * (float)gridSize), gridSize - 1);
    int cell = (gx * gridSize + gy) * gridSize + gz;
    int word = cell >> 5, bit = cell & 31;
    bool occupied = word < occupancyWordCount &&
        (occupancyBitfield[word] & (1u << bit)) != 0u;
    positions[positionIndex] = px; positions[positionIndex + 1] = py; positions[positionIndex + 2] = pz;
    directions[positionIndex] = dx; directions[positionIndex + 1] = dy; directions[positionIndex + 2] = dz;
    validMask[sampleIndex] = occupied ? 1.0f : 0.0f; tValues[sampleIndex] = t;
}

extern ""C"" __global__ void resident_mode(
    const float* __restrict__ input,
    float* __restrict__ output,
    int length)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    float bestValue = 0.0f;
    int bestCount = -1;
    for (int i = 0; i < length; i++) {
        float candidate = input[i];
        bool first = true;
        for (int j = 0; j < i; j++) {
            float prior = input[j];
            if (candidate == prior || (isnan(candidate) && isnan(prior))) {
                first = false;
                break;
            }
        }
        if (!first) continue;
        int count = 0;
        for (int j = 0; j < length; j++) {
            float value = input[j];
            if (candidate == value || (isnan(candidate) && isnan(value))) count++;
        }
        if (bestCount < 0 || count > bestCount ||
            (count == bestCount && candidate < bestValue)) {
            bestValue = candidate;
            bestCount = count;
        }
    }
    output[0] = bestValue;
    output[1] = (float)bestCount;
}

extern ""C"" __global__ __launch_bounds__(256) void resident_indices_to_int32(
    const float* __restrict__ input,
    int* __restrict__ output,
    int length)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < length) output[gid] = (int)input[gid];
}

extern ""C"" __global__ __launch_bounds__(256) void resident_index_add(
    const float* __restrict__ destination,
    const int* __restrict__ indices,
    const float* __restrict__ source,
    float* __restrict__ output,
    int outerSize, int sourceAxis, int destinationAxis, int innerSize)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = outerSize * destinationAxis * innerSize;
    if (gid >= total) return;
    int inner = gid % innerSize;
    int dst = (gid / innerSize) % destinationAxis;
    int outer = (gid / innerSize) / destinationAxis;
    float value = destination[gid];
    for (int j = 0; j < sourceAxis; j++) {
        if (indices[j] == dst)
            value += source[(outer * sourceAxis + j) * innerSize + inner];
    }
    output[gid] = value;
}

extern ""C"" __global__ __launch_bounds__(256) void resident_index_select(
    const float* __restrict__ source,
    const int* __restrict__ indices,
    float* __restrict__ output,
    int outerSize, int sourceAxis, int indexAxis, int innerSize)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = outerSize * indexAxis * innerSize;
    if (gid >= total) return;
    int inner = gid % innerSize;
    int j = (gid / innerSize) % indexAxis;
    int outer = (gid / innerSize) / indexAxis;
    int sourceIndex = indices[j];
    output[gid] = source[(outer * sourceAxis + sourceIndex) * innerSize + inner];
}

extern ""C"" __global__ __launch_bounds__(256) void resident_scatter_max_argmax_rows(
    const float* __restrict__ source,
    const int* __restrict__ indices,
    float* __restrict__ output,
    float* __restrict__ argmax,
    int sourceRows, int innerSize, int outputRows)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = outputRows * innerSize;
    if (gid >= total) return;
    int group = gid / innerSize;
    int inner = gid % innerSize;
    float best = -INFINITY;
    int bestRow = -1;
    for (int row = 0; row < sourceRows; row++) {
        float value = source[row * innerSize + inner];
        if (indices[row] == group && value > best) {
            best = value;
            bestRow = row;
        }
    }
    output[gid] = best;
    argmax[gid] = (float)bestRow;
}

extern ""C"" __global__ __launch_bounds__(256) void resident_uniform_mesh_laplacian(
    const int* __restrict__ faces,
    float* __restrict__ output,
    int numFaces, int numVertices)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = numVertices * numVertices;
    if (gid >= total) return;
    int row = gid / numVertices;
    int column = gid % numVertices;
    float value = 0.0f;
    for (int face = 0; face < numFaces; face++) {
        int v0 = faces[face * 3];
        int v1 = faces[face * 3 + 1];
        int v2 = faces[face * 3 + 2];
        if (row == v0 && column == v1) value -= 1.0f;
        if (row == v1 && column == v0) value -= 1.0f;
        if (row == v0 && column == v0) value += 1.0f;
        if (row == v1 && column == v1) value += 1.0f;
        if (row == v1 && column == v2) value -= 1.0f;
        if (row == v2 && column == v1) value -= 1.0f;
        if (row == v1 && column == v1) value += 1.0f;
        if (row == v2 && column == v2) value += 1.0f;
        if (row == v2 && column == v0) value -= 1.0f;
        if (row == v0 && column == v2) value -= 1.0f;
        if (row == v2 && column == v2) value += 1.0f;
        if (row == v0 && column == v0) value += 1.0f;
    }
    output[gid] = value;
}

extern ""C"" __global__ __launch_bounds__(256) void resident_scatter_add_rows(
    const float* source, const int* indices, float* output,
    int sourceRows, int innerSize, int outputRows)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = outputRows * innerSize;
    if (gid >= total) return;
    int inner = gid % innerSize;
    int group = gid / innerSize;
    float sum = 0.0f;
    for (int row = 0; row < sourceRows; row++)
        if (indices[row] == group) sum += source[row * innerSize + inner];
    output[gid] = sum;
}

extern ""C"" __global__ __launch_bounds__(256) void resident_scatter_mean_rows_counts(
    const float* source, const int* indices, float* output, float* counts,
    int sourceRows, int innerSize, int outputRows)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int outputTotal = outputRows * innerSize;
    if (gid < outputRows) {
        int count = 0;
        for (int row = 0; row < sourceRows; row++) if (indices[row] == gid) count++;
        counts[gid] = (float)count;
    }
    if (gid >= outputTotal) return;
    int inner = gid % innerSize;
    int group = gid / innerSize;
    float sum = 0.0f;
    int count = 0;
    for (int row = 0; row < sourceRows; row++) {
        if (indices[row] == group) { sum += source[row * innerSize + inner]; count++; }
    }
    output[gid] = count > 0 ? sum * (1.0f / (float)count) : 0.0f;
}

extern ""C"" __global__ __launch_bounds__(256) void resident_scatter_softmax_rows(
    const float* source, const int* indices, float* output,
    int sourceRows, int innerSize, int numGroups)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = sourceRows * innerSize;
    if (gid >= total) return;
    int inner = gid % innerSize;
    int row = gid / innerSize;
    int group = indices[row];
    if (group < 0 || group >= numGroups) { output[gid] = 0.0f; return; }
    float maximum = -INFINITY;
    for (int other = 0; other < sourceRows; other++)
        if (indices[other] == group) {
            float value = source[other * innerSize + inner];
            if (value > maximum) maximum = value;
        }
    float sum = 0.0f;
    for (int other = 0; other < sourceRows; other++)
        if (indices[other] == group) sum += expf(source[other * innerSize + inner] - maximum);
    float value = expf(source[gid] - maximum);
    output[gid] = sum != 0.0f ? value / sum : value;
}

extern ""C"" __global__ __launch_bounds__(256) void resident_scatter_add_backward_rows(
    const float* gradOutput, const int* indices, float* gradSource,
    int sourceRows, int innerSize, int outputRows)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = sourceRows * innerSize;
    if (gid >= total) return;
    int inner = gid % innerSize;
    int group = indices[gid / innerSize];
    gradSource[gid] = group >= 0 && group < outputRows
        ? gradOutput[group * innerSize + inner] : 0.0f;
}

extern ""C"" __global__ __launch_bounds__(256) void resident_scatter_mean_backward_rows(
    const float* gradOutput, const int* indices, const int* counts, float* gradSource,
    int sourceRows, int innerSize, int outputRows)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = sourceRows * innerSize;
    if (gid >= total) return;
    int inner = gid % innerSize;
    int group = indices[gid / innerSize];
    if (group < 0 || group >= outputRows) { gradSource[gid] = 0.0f; return; }
    int count = counts[group];
    float divisor = count > 0 ? (float)count : 1.0f;
    gradSource[gid] = gradOutput[group * innerSize + inner] / divisor;
}

extern ""C"" __global__ __launch_bounds__(256) void resident_scatter_max_backward_rows(
    const float* gradOutput, const int* argmax, float* gradSource,
    int sourceRows, int innerSize, int outputRows)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = sourceRows * innerSize;
    if (gid >= total) return;
    int inner = gid % innerSize;
    int sourceRow = gid / innerSize;
    float value = 0.0f;
    for (int group = 0; group < outputRows; group++)
        if (argmax[group * innerSize + inner] == sourceRow)
            value = gradOutput[group * innerSize + inner];
    gradSource[gid] = value;
}

extern ""C"" __global__ __launch_bounds__(256) void resident_scatter_softmax_backward_rows(
    const float* gradOutput, const float* output, const int* indices, float* gradSource,
    int sourceRows, int innerSize, int numGroups)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = sourceRows * innerSize;
    if (gid >= total) return;
    int inner = gid % innerSize;
    int group = indices[gid / innerSize];
    if (group < 0 || group >= numGroups) { gradSource[gid] = 0.0f; return; }
    float sum = 0.0f;
    for (int row = 0; row < sourceRows; row++)
        if (indices[row] == group)
            sum += output[row * innerSize + inner] * gradOutput[row * innerSize + inner];
    gradSource[gid] = output[gid] * (gradOutput[gid] - sum);
}

extern ""C"" __global__ void nonzero_compact(
    const float* __restrict__ input,
    const int* __restrict__ strides,
    float* __restrict__ output,
    float* __restrict__ outputCount,
    int length, int rank)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    int count = 0;
    for (int i = 0; i < length; i++) {
        if (input[i] != 0.0f) {
            int rem = i;
            for (int d = 0; d < rank; d++) {
                int stride = strides[d];
                output[count * rank + d] = (float)(rem / stride);
                rem %= stride;
            }
            count++;
        }
    }
    outputCount[0] = (float)count;
}

__device__ __forceinline__ unsigned int importance_hash(unsigned int x)
{
    x ^= x >> 16; x *= 0x7feb352du;
    x ^= x >> 15; x *= 0x846ca68bu;
    return x ^ (x >> 16);
}

extern ""C"" __global__ void importance_sampling(
    const float* __restrict__ tValues,
    const float* __restrict__ weights,
    float* __restrict__ output,
    int numRays, int numCoarse, int numFine, unsigned int seed)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = numRays * numFine;
    if (gid >= total) return;
    int ray = gid / numFine;
    int sample = gid - ray * numFine;
    int base = ray * numCoarse;
    unsigned int bits = importance_hash(seed ^ ((unsigned int)gid * 747796405u + 2891336453u));
    float random = (float)(bits >> 8) * (1.0f / 16777216.0f);
    float u = ((float)sample + random) / (float)numFine;
    float weightSum = 0.0f;
    for (int s = 0; s < numCoarse; s++) {
        float weight = weights[base + s];
        weightSum += weight > 0.0f ? weight : 0.0f;
    }
    if (weightSum <= 1.0e-10f) {
        float tMin = tValues[base];
        float tMax = tValues[base + numCoarse - 1];
        output[gid] = tMin + u * (tMax - tMin);
        return;
    }
    float previous = 0.0f;
    float current = 0.0f;
    int index = 0;
    for (int s = 0; s < numCoarse; s++) {
        float weight = weights[base + s];
        current += (weight > 0.0f ? weight : 0.0f) / weightSum;
        index = s;
        if (u <= current || s == numCoarse - 1) break;
        previous = current;
    }
    if (index == 0) {
        output[gid] = tValues[base];
        return;
    }
    float denominator = current - previous;
    float t0 = tValues[base + index - 1];
    float t1 = tValues[base + index];
    output[gid] = denominator > 1.0e-10f
        ? t0 + ((u - previous) / denominator) * (t1 - t0)
        : t0;
}

extern ""C"" __global__ void resident_nms(
    const float* __restrict__ boxes,
    const float* __restrict__ scores,
    const float* __restrict__ classIds,
    float* __restrict__ suppressed,
    float* __restrict__ output,
    float* __restrict__ outputCount,
    int length, float threshold, int batched)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    int count = 0;
    for (int iteration = 0; iteration < length; iteration++) {
        int best = -1;
        float bestScore = -3.402823466e+38f;
        for (int i = 0; i < length; i++) {
            if (suppressed[i] != 0.0f) continue;
            float score = scores[i];
            if (best < 0 || (!isnan(score) &&
                (isnan(bestScore) || score > bestScore || (score == bestScore && i < best)))) {
                best = i;
                bestScore = score;
            }
        }
        if (best < 0) break;
        suppressed[best] = 1.0f;
        output[count++] = (float)best;
        float ix1 = boxes[best * 4];
        float iy1 = boxes[best * 4 + 1];
        float ix2 = boxes[best * 4 + 2];
        float iy2 = boxes[best * 4 + 3];
        float iw0 = ix2 - ix1;
        float ih0 = iy2 - iy1;
        float areaI = (iw0 > 0.0f && ih0 > 0.0f) ? iw0 * ih0 : 0.0f;
        for (int j = 0; j < length; j++) {
            if (suppressed[j] != 0.0f) continue;
            if (batched != 0 && classIds[j] != classIds[best]) continue;
            float jx1 = boxes[j * 4];
            float jy1 = boxes[j * 4 + 1];
            float jx2 = boxes[j * 4 + 2];
            float jy2 = boxes[j * 4 + 3];
            float jw0 = jx2 - jx1;
            float jh0 = jy2 - jy1;
            float areaJ = (jw0 > 0.0f && jh0 > 0.0f) ? jw0 * jh0 : 0.0f;
            float overlapW = fminf(ix2, jx2) - fmaxf(ix1, jx1);
            float overlapH = fminf(iy2, jy2) - fmaxf(iy1, jy1);
            if (overlapW < 0.0f) overlapW = 0.0f;
            if (overlapH < 0.0f) overlapH = 0.0f;
            float intersection = overlapW * overlapH;
            float unionArea = areaI + areaJ - intersection;
            if (unionArea > 0.0f && intersection / unionArea > threshold)
                suppressed[j] = 1.0f;
        }
    }
    outputCount[0] = (float)count;
}

__device__ __forceinline__ int spiral_append_unique(
    float* list, int count, int candidate, int capacity)
{
    for (int i = 0; i < count; i++) if ((int)list[i] == candidate) return count;
    if (count < capacity) list[count++] = (float)candidate;
    return count;
}

__device__ __forceinline__ int spiral_build_neighbors(
    const float* faces, int numFaces, int vertex, float* list, int capacity)
{
    int count = 0;
    for (int f = 0; f < numFaces; f++) {
        int v0 = (int)faces[f * 3], v1 = (int)faces[f * 3 + 1], v2 = (int)faces[f * 3 + 2];
        if (vertex == v0) { count = spiral_append_unique(list, count, v1, capacity); count = spiral_append_unique(list, count, v2, capacity); }
        else if (vertex == v1) { count = spiral_append_unique(list, count, v0, capacity); count = spiral_append_unique(list, count, v2, capacity); }
        else if (vertex == v2) { count = spiral_append_unique(list, count, v0, capacity); count = spiral_append_unique(list, count, v1, capacity); }
    }
    return count;
}

__device__ __forceinline__ float spiral_angle(
    const float* vertices, int center, int reference, int vertex)
{
    float cx = vertices[center * 3], cy = vertices[center * 3 + 1], cz = vertices[center * 3 + 2];
    float rx = vertices[reference * 3] - cx;
    float ry = vertices[reference * 3 + 1] - cy;
    float rz = vertices[reference * 3 + 2] - cz;
    float ax = vertices[vertex * 3] - cx;
    float ay = vertices[vertex * 3 + 1] - cy;
    float az = vertices[vertex * 3 + 2] - cz;
    return atan2f(ax * ry - ay * rx, ax * rx + ay * ry + az * rz);
}

extern ""C"" __global__ void generate_spiral_indices(
    const float* __restrict__ vertices,
    const float* __restrict__ faces,
    float* __restrict__ visited,
    float* __restrict__ currentRing,
    float* __restrict__ nextRing,
    float* __restrict__ output,
    int numVertices, int numFaces, int spiralLength)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    for (int center = 0; center < numVertices; center++) {
        for (int i = 0; i < numVertices; i++) visited[i] = 0.0f;
        int currentCount = spiral_build_neighbors(faces, numFaces, center, currentRing, numVertices);
        if (currentCount > 1) {
            int reference = (int)currentRing[0];
            for (int i = 1; i < currentCount; i++) {
                float key = currentRing[i];
                float keyAngle = spiral_angle(vertices, center, reference, (int)key);
                int j = i - 1;
                while (j >= 0 && spiral_angle(vertices, center, reference, (int)currentRing[j]) > keyAngle) {
                    currentRing[j + 1] = currentRing[j]; j--;
                }
                currentRing[j + 1] = key;
            }
        }
        visited[center] = 1.0f;
        int outputIndex = 0;
        while (outputIndex < spiralLength && currentCount > 0) {
            int nextCount = 0;
            for (int r = 0; r < currentCount && outputIndex < spiralLength; r++) {
                int neighbor = (int)currentRing[r];
                if (neighbor < 0 || neighbor >= numVertices || visited[neighbor] != 0.0f) continue;
                output[center * spiralLength + outputIndex++] = (float)neighbor;
                visited[neighbor] = 1.0f;
                for (int f = 0; f < numFaces; f++) {
                    int v0 = (int)faces[f * 3], v1 = (int)faces[f * 3 + 1], v2 = (int)faces[f * 3 + 2];
                    int a = -1, b = -1;
                    if (neighbor == v0) { a = v1; b = v2; }
                    else if (neighbor == v1) { a = v0; b = v2; }
                    else if (neighbor == v2) { a = v0; b = v1; }
                    if (a >= 0 && a < numVertices && visited[a] == 0.0f) nextCount = spiral_append_unique(nextRing, nextCount, a, numVertices);
                    if (b >= 0 && b < numVertices && visited[b] == 0.0f) nextCount = spiral_append_unique(nextRing, nextCount, b, numVertices);
                }
            }
            currentCount = nextCount;
            for (int i = 0; i < nextCount; i++) currentRing[i] = nextRing[i];
        }
        while (outputIndex < spiralLength)
            output[center * spiralLength + outputIndex++] = -1.0f;
    }
}

__device__ __forceinline__ float ctc_log_add(float a, float b)
{
    const float negativeSentinel = -3.402823466e+38f;
    if (a <= negativeSentinel) return b;
    if (b <= negativeSentinel) return a;
    float m = fmaxf(a, b);
    return m + logf(expf(a - m) + expf(b - m));
}

extern ""C"" __global__ void ctc_loss_forward(
    const float* __restrict__ logProbs,
    const float* __restrict__ targets,
    const float* __restrict__ inputLengths,
    const float* __restrict__ targetLengths,
    float* __restrict__ workspace,
    float* __restrict__ losses,
    int maxTime, int batchSize, int numClasses, int maxTargetLength, int blank)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= batchSize) return;
    int timeLength = (int)inputLengths[n];
    int targetLength = (int)targetLengths[n];
    int targetOffset = 0;
    for (int i = 0; i < n; i++) targetOffset += (int)targetLengths[i];
    int states = 2 * targetLength + 1;
    int maxStates = 2 * maxTargetLength + 1;
    int previous = n * 2 * maxStates;
    int current = previous + maxStates;
    for (int s = 0; s < states; s++) workspace[previous + s] = -3.402823466e+38f;

    workspace[previous] = logProbs[n * numClasses + blank];
    if (states > 1) {
        int label = (int)targets[targetOffset];
        workspace[previous + 1] = logProbs[n * numClasses + label];
    }
    for (int t = 1; t < timeLength; t++) {
        for (int s = 0; s < states; s++) {
            int label = (s & 1) == 0 ? blank : (int)targets[targetOffset + s / 2];
            float sum = workspace[previous + s];
            if (s >= 1) sum = ctc_log_add(sum, workspace[previous + s - 1]);
            if (s >= 2) {
                int priorLabel = (s & 1) == 0 ? blank : (int)targets[targetOffset + (s - 2) / 2];
                if (label != blank && label != priorLabel)
                    sum = ctc_log_add(sum, workspace[previous + s - 2]);
            }
            workspace[current + s] = sum + logProbs[(t * batchSize + n) * numClasses + label];
        }
        int swap = previous; previous = current; current = swap;
    }
    float logProbability = workspace[previous + states - 1];
    if (states >= 2)
        logProbability = ctc_log_add(logProbability, workspace[previous + states - 2]);
    losses[n] = -logProbability;
}
";
}
