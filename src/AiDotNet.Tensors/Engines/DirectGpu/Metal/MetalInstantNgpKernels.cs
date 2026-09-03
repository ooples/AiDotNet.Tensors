namespace AiDotNet.Tensors.Engines.DirectGpu.Metal;

public static class MetalInstantNgpKernels
{
    public static string Source => @"
inline uint instant_ngp_hash(int x, int y, int z, int tableSize)
{
    uint hx = uint(x) * 73856093u;
    uint hy = uint(y) * 19349663u;
    uint hz = uint(z) * 83492791u;
    return (hx ^ hy ^ hz) % uint(tableSize);
}

inline float instant_ngp_clamp_position(float value)
{
    if (value <= 0.0f) return 0.0f;
    if (value >= 0.999999f) return 0.999999f;
    return value;
}

kernel void instant_ngp_hash_encode_level(
    device const float* positions [[buffer(0)]],
    device const float* hashTable [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant int& numPoints [[buffer(3)]],
    constant int& resolution [[buffer(4)]],
    constant int& tableSize [[buffer(5)]],
    constant int& featuresPerLevel [[buffer(6)]],
    constant int& levelOffset [[buffer(7)]],
    constant int& outputStride [[buffer(8)]],
    uint gid [[thread_position_in_grid]])
{
    int total = numPoints * featuresPerLevel;
    if (int(gid) >= total) return;
    int n = int(gid) / featuresPerLevel;
    int f = int(gid) - n * featuresPerLevel;
    float gx = instant_ngp_clamp_position(positions[n * 3]) * float(resolution);
    float gy = instant_ngp_clamp_position(positions[n * 3 + 1]) * float(resolution);
    float gz = instant_ngp_clamp_position(positions[n * 3 + 2]) * float(resolution);
    int x0 = int(floor(gx)), y0 = int(floor(gy)), z0 = int(floor(gz));
    int x1 = x0 + 1, y1 = y0 + 1, z1 = z0 + 1;
    float fx = gx - float(x0), fy = gy - float(y0), fz = gz - float(z0);
    float ix = 1.0f - fx, iy = 1.0f - fy, iz = 1.0f - fz;
    uint h000 = instant_ngp_hash(x0, y0, z0, tableSize);
    uint h001 = instant_ngp_hash(x0, y0, z1, tableSize);
    uint h010 = instant_ngp_hash(x0, y1, z0, tableSize);
    uint h011 = instant_ngp_hash(x0, y1, z1, tableSize);
    uint h100 = instant_ngp_hash(x1, y0, z0, tableSize);
    uint h101 = instant_ngp_hash(x1, y0, z1, tableSize);
    uint h110 = instant_ngp_hash(x1, y1, z0, tableSize);
    uint h111 = instant_ngp_hash(x1, y1, z1, tableSize);
    float value =
        ix * iy * iz * hashTable[int(h000) * featuresPerLevel + f] +
        ix * iy * fz * hashTable[int(h001) * featuresPerLevel + f] +
        ix * fy * iz * hashTable[int(h010) * featuresPerLevel + f] +
        ix * fy * fz * hashTable[int(h011) * featuresPerLevel + f] +
        fx * iy * iz * hashTable[int(h100) * featuresPerLevel + f] +
        fx * iy * fz * hashTable[int(h101) * featuresPerLevel + f] +
        fx * fy * iz * hashTable[int(h110) * featuresPerLevel + f] +
        fx * fy * fz * hashTable[int(h111) * featuresPerLevel + f];
    output[n * outputStride + levelOffset + f] = value;
}

kernel void instant_ngp_hash_encode_level_backward(
    device const float* positions [[buffer(0)]],
    device const float* outputGradient [[buffer(1)]],
    device float* tableGradient [[buffer(2)]],
    constant int& numPoints [[buffer(3)]],
    constant int& resolution [[buffer(4)]],
    constant int& tableSize [[buffer(5)]],
    constant int& featuresPerLevel [[buffer(6)]],
    constant int& levelOffset [[buffer(7)]],
    constant int& outputStride [[buffer(8)]],
    uint gid [[thread_position_in_grid]])
{
    int total = tableSize * featuresPerLevel;
    if (int(gid) >= total) return;
    uint entry = uint(int(gid) / featuresPerLevel);
    int f = int(gid) - int(entry) * featuresPerLevel;
    float acc = 0.0f;
    for (int n = 0; n < numPoints; n++) {
        float gx = instant_ngp_clamp_position(positions[n * 3]) * float(resolution);
        float gy = instant_ngp_clamp_position(positions[n * 3 + 1]) * float(resolution);
        float gz = instant_ngp_clamp_position(positions[n * 3 + 2]) * float(resolution);
        int x0 = int(floor(gx)), y0 = int(floor(gy)), z0 = int(floor(gz));
        int x1 = x0 + 1, y1 = y0 + 1, z1 = z0 + 1;
        float fx = gx - float(x0), fy = gy - float(y0), fz = gz - float(z0);
        float ix = 1.0f - fx, iy = 1.0f - fy, iz = 1.0f - fz;
        float grad = outputGradient[n * outputStride + levelOffset + f];
        if (fabs(grad) < 1.0e-10f) continue;
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

kernel void unique_consecutive_compact(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device float* outputCount [[buffer(2)]],
    constant int& length [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid != 0) return;
    if (length <= 0) { outputCount[0] = 0.0f; return; }
    int count = 1;
    output[0] = input[0];
    for (int i = 1; i < length; i++) {
        if (input[i] != input[i - 1]) output[count++] = input[i];
    }
    outputCount[0] = float(count);
}

kernel void unique_consecutive_with_info(
    device const float* input [[buffer(0)]],
    device float* outputValues [[buffer(1)]],
    device float* outputInverse [[buffer(2)]],
    device float* outputCounts [[buffer(3)]],
    device float* outputCount [[buffer(4)]],
    constant int& length [[buffer(5)]],
    constant int& returnInverse [[buffer(6)]],
    constant int& returnCounts [[buffer(7)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid != 0) return;
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
        if (returnInverse != 0) outputInverse[i] = float(count - 1);
    }
    outputCount[0] = float(count);
}

kernel void unique_sorted_with_info(
    device const float* sortedInput [[buffer(0)]],
    device const float* sortedOriginalIndices [[buffer(1)]],
    device float* outputValues [[buffer(2)]],
    device float* outputInverse [[buffer(3)]],
    device float* outputCounts [[buffer(4)]],
    device float* outputCount [[buffer(5)]],
    constant int& length [[buffer(6)]],
    constant int& returnInverse [[buffer(7)]],
    constant int& returnCounts [[buffer(8)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid != 0) return;
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
            outputInverse[int(sortedOriginalIndices[i])] = float(group);
    }
    outputCount[0] = float(count);
}

kernel void frexp_decompose(
    device const float* input [[buffer(0)]],
    device float* mantissa [[buffer(1)]],
    device float* exponent [[buffer(2)]],
    constant int& length [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (int(gid) >= length) return;
    float value = input[gid];
    uint bits = as_type<uint>(value);
    uint magnitude = bits & 0x7fffffffu;
    uint exponentBits = magnitude >> 23;
    if (exponentBits == 0xffu || magnitude == 0u) {
        mantissa[gid] = value;
        exponent[gid] = 0.0f;
        return;
    }
    int scaleAdjustment = 0;
    if (exponentBits == 0u) {
        value *= 16777216.0f;
        bits = as_type<uint>(value);
        exponentBits = (bits & 0x7fffffffu) >> 23;
        scaleAdjustment = -24;
    }
    mantissa[gid] = as_type<float>((bits & 0x807fffffu) | (126u << 23));
    exponent[gid] = float(int(exponentBits) - 126 + scaleAdjustment);
}

kernel void project_gaussians_3d_to_2d(
    device const float* means3D [[buffer(0)]],
    device const float* covariances3D [[buffer(1)]],
    device float* means2D [[buffer(2)]],
    device float* covariances2D [[buffer(3)]],
    device float* depths [[buffer(4)]],
    device float* visible [[buffer(5)]],
    constant int& numGaussians [[buffer(6)]],
    constant int& covarianceStride [[buffer(7)]],
    constant int& imageWidth [[buffer(8)]], constant int& imageHeight [[buffer(9)]],
    constant float& v00 [[buffer(10)]], constant float& v01 [[buffer(11)]],
    constant float& v02 [[buffer(12)]], constant float& v03 [[buffer(13)]],
    constant float& v10 [[buffer(14)]], constant float& v11 [[buffer(15)]],
    constant float& v12 [[buffer(16)]], constant float& v13 [[buffer(17)]],
    constant float& v20 [[buffer(18)]], constant float& v21 [[buffer(19)]],
    constant float& v22 [[buffer(20)]], constant float& v23 [[buffer(21)]],
    constant float& p00 [[buffer(22)]], constant float& p11 [[buffer(23)]],
    uint i [[thread_position_in_grid]])
{
    if (int(i) >= numGaussians) return;
    means2D[i * 2] = 0.0f; means2D[i * 2 + 1] = 0.0f;
    covariances2D[i * 3] = 0.0f; covariances2D[i * 3 + 1] = 0.0f;
    covariances2D[i * 3 + 2] = 0.0f; depths[i] = 0.0f; visible[i] = 0.0f;
    float mx = means3D[i * 3], my = means3D[i * 3 + 1], mz = means3D[i * 3 + 2];
    float camX = v00 * mx + v01 * my + v02 * mz + v03;
    float camY = v10 * mx + v11 * my + v12 * mz + v13;
    float camZ = v20 * mx + v21 * my + v22 * mz + v23;
    if (camZ <= 0.001f) return;
    float invZ = 1.0f / camZ;
    float cx = float(imageWidth) * 0.5f, cy = float(imageHeight) * 0.5f;
    float screenX = p00 * camX * invZ * cx + cx;
    float screenY = p11 * camY * invZ * cy + cy;
    if (screenX < -float(imageWidth) || screenX > 2.0f * float(imageWidth) ||
        screenY < -float(imageHeight) || screenY > 2.0f * float(imageHeight)) return;
    int c = int(i) * covarianceStride;
    float c00 = covariances3D[c], c01 = covariances3D[c + 1], c02 = covariances3D[c + 2];
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

inline bool sample_ray_axis(
    float origin, float direction, float minimum, float maximum,
    thread float& tMin, thread float& tMax)
{
    if (fabs(direction) < 1.0e-8f)
        return origin >= minimum && origin <= maximum;
    float t1 = (minimum - origin) / direction;
    float t2 = (maximum - origin) / direction;
    if (t1 > t2) { float temporary = t1; t1 = t2; t2 = temporary; }
    tMin = fmax(tMin, t1); tMax = fmin(tMax, t2);
    return tMax >= tMin;
}

kernel void sample_rays_with_occupancy(
    device const float* rayOrigins [[buffer(0)]],
    device const float* rayDirections [[buffer(1)]],
    device const uint* occupancyBitfield [[buffer(2)]],
    device float* positions [[buffer(3)]],
    device float* directions [[buffer(4)]],
    device float* validMask [[buffer(5)]],
    device float* tValues [[buffer(6)]],
    constant int& numRays [[buffer(7)]],
    constant int& occupancyWordCount [[buffer(8)]],
    constant int& gridSize [[buffer(9)]], constant int& maxSamples [[buffer(10)]],
    constant float& minX [[buffer(11)]], constant float& minY [[buffer(12)]],
    constant float& minZ [[buffer(13)]], constant float& maxX [[buffer(14)]],
    constant float& maxY [[buffer(15)]], constant float& maxZ [[buffer(16)]],
    constant float& nearBound [[buffer(17)]], constant float& farBound [[buffer(18)]],
    uint sampleIndex [[thread_position_in_grid]])
{
    int totalSamples = numRays * maxSamples;
    if (int(sampleIndex) >= totalSamples) return;
    int positionIndex = int(sampleIndex) * 3;
    positions[positionIndex] = 0.0f; positions[positionIndex + 1] = 0.0f;
    positions[positionIndex + 2] = 0.0f; directions[positionIndex] = 0.0f;
    directions[positionIndex + 1] = 0.0f; directions[positionIndex + 2] = 0.0f;
    validMask[sampleIndex] = 0.0f; tValues[sampleIndex] = 0.0f;
    int ray = int(sampleIndex) / maxSamples, sample = int(sampleIndex) - ray * maxSamples;
    float ox = rayOrigins[ray * 3], oy = rayOrigins[ray * 3 + 1], oz = rayOrigins[ray * 3 + 2];
    float dx = rayDirections[ray * 3], dy = rayDirections[ray * 3 + 1], dz = rayDirections[ray * 3 + 2];
    float tMin = nearBound, tMax = farBound;
    if (!sample_ray_axis(ox, dx, minX, maxX, tMin, tMax) ||
        !sample_ray_axis(oy, dy, minY, maxY, tMin, tMax) ||
        !sample_ray_axis(oz, dz, minZ, maxZ, tMin, tMax) || tMax < tMin) return;
    float t = tMin + ((tMax - tMin) / float(maxSamples)) * (float(sample) + 0.5f);
    float px = ox + t * dx, py = oy + t * dy, pz = oz + t * dz;
    float nx = clamp((px - minX) / fmax(1.0e-10f, maxX - minX), 0.0f, 0.999999f);
    float ny = clamp((py - minY) / fmax(1.0e-10f, maxY - minY), 0.0f, 0.999999f);
    float nz = clamp((pz - minZ) / fmax(1.0e-10f, maxZ - minZ), 0.0f, 0.999999f);
    int gx = min(int(nx * float(gridSize)), gridSize - 1);
    int gy = min(int(ny * float(gridSize)), gridSize - 1);
    int gz = min(int(nz * float(gridSize)), gridSize - 1);
    int cell = (gx * gridSize + gy) * gridSize + gz;
    int word = cell >> 5, bit = cell & 31;
    bool occupied = word < occupancyWordCount &&
        (occupancyBitfield[word] & (1u << uint(bit))) != 0u;
    positions[positionIndex] = px; positions[positionIndex + 1] = py; positions[positionIndex + 2] = pz;
    directions[positionIndex] = dx; directions[positionIndex + 1] = dy; directions[positionIndex + 2] = dz;
    validMask[sampleIndex] = occupied ? 1.0f : 0.0f; tValues[sampleIndex] = t;
}

kernel void resident_mode(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant int& length [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid != 0) return;
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
    output[1] = float(bestCount);
}

kernel void resident_indices_to_int32(
    device const float* input [[buffer(0)]],
    device int* output [[buffer(1)]],
    constant int& length [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (int(gid) < length) output[gid] = int(input[gid]);
}

kernel void resident_index_add(
    device const float* destination [[buffer(0)]],
    device const int* indices [[buffer(1)]],
    device const float* source [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant int& outerSize [[buffer(4)]],
    constant int& sourceAxis [[buffer(5)]],
    constant int& destinationAxis [[buffer(6)]],
    constant int& innerSize [[buffer(7)]],
    uint gid [[thread_position_in_grid]])
{
    int total = outerSize * destinationAxis * innerSize;
    if (int(gid) >= total) return;
    int inner = int(gid) % innerSize;
    int dst = (int(gid) / innerSize) % destinationAxis;
    int outer = (int(gid) / innerSize) / destinationAxis;
    float value = destination[gid];
    for (int j = 0; j < sourceAxis; j++) {
        if (indices[j] == dst)
            value += source[(outer * sourceAxis + j) * innerSize + inner];
    }
    output[gid] = value;
}

kernel void resident_index_select(
    device const float* source [[buffer(0)]],
    device const int* indices [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant int& outerSize [[buffer(3)]],
    constant int& sourceAxis [[buffer(4)]],
    constant int& indexAxis [[buffer(5)]],
    constant int& innerSize [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    int total = outerSize * indexAxis * innerSize;
    if (int(gid) >= total) return;
    int inner = int(gid) % innerSize;
    int j = (int(gid) / innerSize) % indexAxis;
    int outer = (int(gid) / innerSize) / indexAxis;
    int sourceIndex = indices[j];
    output[gid] = source[(outer * sourceAxis + sourceIndex) * innerSize + inner];
}

kernel void resident_scatter_max_argmax_rows(
    device const float* source [[buffer(0)]],
    device const int* indices [[buffer(1)]],
    device float* output [[buffer(2)]],
    device float* argmax [[buffer(3)]],
    constant int& sourceRows [[buffer(4)]],
    constant int& innerSize [[buffer(5)]],
    constant int& outputRows [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    int total = outputRows * innerSize;
    if (int(gid) >= total) return;
    int group = int(gid) / innerSize;
    int inner = int(gid) % innerSize;
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
    argmax[gid] = float(bestRow);
}

kernel void resident_uniform_mesh_laplacian(
    device const int* faces [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant int& numFaces [[buffer(2)]],
    constant int& numVertices [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    int total = numVertices * numVertices;
    if (int(gid) >= total) return;
    int row = int(gid) / numVertices;
    int column = int(gid) % numVertices;
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

kernel void resident_scatter_add_rows(
    device const float* source [[buffer(0)]], device const int* indices [[buffer(1)]],
    device float* output [[buffer(2)]], constant int& sourceRows [[buffer(3)]],
    constant int& innerSize [[buffer(4)]], constant int& outputRows [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    int total = outputRows * innerSize;
    if (int(gid) >= total) return;
    int inner = int(gid) % innerSize;
    int group = int(gid) / innerSize;
    float sum = 0.0f;
    for (int row = 0; row < sourceRows; row++)
        if (indices[row] == group) sum += source[row * innerSize + inner];
    output[gid] = sum;
}

kernel void resident_scatter_mean_rows_counts(
    device const float* source [[buffer(0)]], device const int* indices [[buffer(1)]],
    device float* output [[buffer(2)]], device float* counts [[buffer(3)]],
    constant int& sourceRows [[buffer(4)]], constant int& innerSize [[buffer(5)]],
    constant int& outputRows [[buffer(6)]], uint gid [[thread_position_in_grid]])
{
    int outputTotal = outputRows * innerSize;
    if (int(gid) < outputRows) {
        int count = 0;
        for (int row = 0; row < sourceRows; row++) if (indices[row] == int(gid)) count++;
        counts[gid] = float(count);
    }
    if (int(gid) >= outputTotal) return;
    int inner = int(gid) % innerSize;
    int group = int(gid) / innerSize;
    float sum = 0.0f;
    int count = 0;
    for (int row = 0; row < sourceRows; row++) {
        if (indices[row] == group) { sum += source[row * innerSize + inner]; count++; }
    }
    output[gid] = count > 0 ? sum * (1.0f / float(count)) : 0.0f;
}

kernel void resident_scatter_softmax_rows(
    device const float* source [[buffer(0)]], device const int* indices [[buffer(1)]],
    device float* output [[buffer(2)]], constant int& sourceRows [[buffer(3)]],
    constant int& innerSize [[buffer(4)]], constant int& numGroups [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    int total = sourceRows * innerSize;
    if (int(gid) >= total) return;
    int inner = int(gid) % innerSize;
    int row = int(gid) / innerSize;
    int group = indices[row];
    if (group < 0 || group >= numGroups) { output[gid] = 0.0f; return; }
    float maximum = -INFINITY;
    for (int other = 0; other < sourceRows; other++)
        if (indices[other] == group) maximum = max(maximum, source[other * innerSize + inner]);
    float sum = 0.0f;
    for (int other = 0; other < sourceRows; other++)
        if (indices[other] == group) sum += exp(source[other * innerSize + inner] - maximum);
    float value = exp(source[gid] - maximum);
    output[gid] = sum != 0.0f ? value / sum : value;
}

kernel void resident_scatter_add_backward_rows(
    device const float* gradOutput [[buffer(0)]], device const int* indices [[buffer(1)]],
    device float* gradSource [[buffer(2)]], constant int& sourceRows [[buffer(3)]],
    constant int& innerSize [[buffer(4)]], constant int& outputRows [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    int total = sourceRows * innerSize;
    if (int(gid) >= total) return;
    int inner = int(gid) % innerSize;
    int group = indices[int(gid) / innerSize];
    gradSource[gid] = group >= 0 && group < outputRows
        ? gradOutput[group * innerSize + inner] : 0.0f;
}

kernel void resident_scatter_mean_backward_rows(
    device const float* gradOutput [[buffer(0)]], device const int* indices [[buffer(1)]],
    device const int* counts [[buffer(2)]], device float* gradSource [[buffer(3)]],
    constant int& sourceRows [[buffer(4)]], constant int& innerSize [[buffer(5)]],
    constant int& outputRows [[buffer(6)]], uint gid [[thread_position_in_grid]])
{
    int total = sourceRows * innerSize;
    if (int(gid) >= total) return;
    int inner = int(gid) % innerSize;
    int group = indices[int(gid) / innerSize];
    if (group < 0 || group >= outputRows) { gradSource[gid] = 0.0f; return; }
    int count = counts[group];
    float divisor = count > 0 ? float(count) : 1.0f;
    gradSource[gid] = gradOutput[group * innerSize + inner] / divisor;
}

kernel void resident_scatter_max_backward_rows(
    device const float* gradOutput [[buffer(0)]], device const int* argmax [[buffer(1)]],
    device float* gradSource [[buffer(2)]], constant int& sourceRows [[buffer(3)]],
    constant int& innerSize [[buffer(4)]], constant int& outputRows [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    int total = sourceRows * innerSize;
    if (int(gid) >= total) return;
    int inner = int(gid) % innerSize;
    int sourceRow = int(gid) / innerSize;
    float value = 0.0f;
    for (int group = 0; group < outputRows; group++)
        if (argmax[group * innerSize + inner] == sourceRow)
            value = gradOutput[group * innerSize + inner];
    gradSource[gid] = value;
}

kernel void resident_scatter_softmax_backward_rows(
    device const float* gradOutput [[buffer(0)]], device const float* output [[buffer(1)]],
    device const int* indices [[buffer(2)]], device float* gradSource [[buffer(3)]],
    constant int& sourceRows [[buffer(4)]], constant int& innerSize [[buffer(5)]],
    constant int& numGroups [[buffer(6)]], uint gid [[thread_position_in_grid]])
{
    int total = sourceRows * innerSize;
    if (int(gid) >= total) return;
    int inner = int(gid) % innerSize;
    int group = indices[int(gid) / innerSize];
    if (group < 0 || group >= numGroups) { gradSource[gid] = 0.0f; return; }
    float sum = 0.0f;
    for (int row = 0; row < sourceRows; row++)
        if (indices[row] == group)
            sum += output[row * innerSize + inner] * gradOutput[row * innerSize + inner];
    gradSource[gid] = output[gid] * (gradOutput[gid] - sum);
}

kernel void nonzero_compact(
    device const float* input [[buffer(0)]],
    device const int* strides [[buffer(1)]],
    device float* output [[buffer(2)]],
    device float* outputCount [[buffer(3)]],
    constant int& length [[buffer(4)]],
    constant int& rank [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid != 0) return;
    int count = 0;
    for (int i = 0; i < length; i++) {
        if (input[i] != 0.0f) {
            int rem = i;
            for (int d = 0; d < rank; d++) {
                int stride = strides[d];
                output[count * rank + d] = float(rem / stride);
                rem %= stride;
            }
            count++;
        }
    }
    outputCount[0] = float(count);
}

inline uint importance_hash(uint x)
{
    x ^= x >> 16; x *= 0x7feb352du;
    x ^= x >> 15; x *= 0x846ca68bu;
    return x ^ (x >> 16);
}

kernel void importance_sampling(
    device const float* tValues [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant int& numRays [[buffer(3)]],
    constant int& numCoarse [[buffer(4)]],
    constant int& numFine [[buffer(5)]],
    constant uint& seed [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    int total = numRays * numFine;
    if (int(gid) >= total) return;
    int ray = int(gid) / numFine;
    int sample = int(gid) - ray * numFine;
    int base = ray * numCoarse;
    uint bits = importance_hash(seed ^ (gid * 747796405u + 2891336453u));
    float random = float(bits >> 8) * (1.0f / 16777216.0f);
    float u = (float(sample) + random) / float(numFine);
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
    if (index == 0) { output[gid] = tValues[base]; return; }
    float denominator = current - previous;
    float t0 = tValues[base + index - 1];
    float t1 = tValues[base + index];
    output[gid] = denominator > 1.0e-10f
        ? t0 + ((u - previous) / denominator) * (t1 - t0)
        : t0;
}

kernel void resident_nms(
    device const float* boxes [[buffer(0)]],
    device const float* scores [[buffer(1)]],
    device const float* classIds [[buffer(2)]],
    device float* suppressed [[buffer(3)]],
    device float* output [[buffer(4)]],
    device float* outputCount [[buffer(5)]],
    constant int& length [[buffer(6)]],
    constant float& threshold [[buffer(7)]],
    constant int& batched [[buffer(8)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid != 0) return;
    int count = 0;
    for (int iteration = 0; iteration < length; iteration++) {
        int best = -1;
        float bestScore = -3.402823466e+38f;
        for (int i = 0; i < length; i++) {
            if (suppressed[i] != 0.0f) continue;
            float score = scores[i];
            if (best < 0 || (!isnan(score) &&
                (isnan(bestScore) || score > bestScore || (score == bestScore && i < best)))) {
                best = i; bestScore = score;
            }
        }
        if (best < 0) break;
        suppressed[best] = 1.0f;
        output[count++] = float(best);
        float ix1 = boxes[best * 4], iy1 = boxes[best * 4 + 1];
        float ix2 = boxes[best * 4 + 2], iy2 = boxes[best * 4 + 3];
        float iw0 = ix2 - ix1, ih0 = iy2 - iy1;
        float areaI = (iw0 > 0.0f && ih0 > 0.0f) ? iw0 * ih0 : 0.0f;
        for (int j = 0; j < length; j++) {
            if (suppressed[j] != 0.0f) continue;
            if (batched != 0 && classIds[j] != classIds[best]) continue;
            float jx1 = boxes[j * 4], jy1 = boxes[j * 4 + 1];
            float jx2 = boxes[j * 4 + 2], jy2 = boxes[j * 4 + 3];
            float jw0 = jx2 - jx1, jh0 = jy2 - jy1;
            float areaJ = (jw0 > 0.0f && jh0 > 0.0f) ? jw0 * jh0 : 0.0f;
            float overlapW = min(ix2, jx2) - max(ix1, jx1);
            float overlapH = min(iy2, jy2) - max(iy1, jy1);
            overlapW = max(overlapW, 0.0f);
            overlapH = max(overlapH, 0.0f);
            float intersection = overlapW * overlapH;
            float unionArea = areaI + areaJ - intersection;
            if (unionArea > 0.0f && intersection / unionArea > threshold)
                suppressed[j] = 1.0f;
        }
    }
    outputCount[0] = float(count);
}

inline int spiral_append_unique(device float* list, int count, int candidate, int capacity)
{
    for (int i = 0; i < count; i++) if (int(list[i]) == candidate) return count;
    if (count < capacity) list[count++] = float(candidate);
    return count;
}

inline int spiral_build_neighbors(
    device const float* faces, int numFaces, int vertex, device float* list, int capacity)
{
    int count = 0;
    for (int f = 0; f < numFaces; f++) {
        int v0 = int(faces[f * 3]), v1 = int(faces[f * 3 + 1]), v2 = int(faces[f * 3 + 2]);
        if (vertex == v0) { count = spiral_append_unique(list, count, v1, capacity); count = spiral_append_unique(list, count, v2, capacity); }
        else if (vertex == v1) { count = spiral_append_unique(list, count, v0, capacity); count = spiral_append_unique(list, count, v2, capacity); }
        else if (vertex == v2) { count = spiral_append_unique(list, count, v0, capacity); count = spiral_append_unique(list, count, v1, capacity); }
    }
    return count;
}

inline float spiral_angle(device const float* vertices, int center, int reference, int vertex)
{
    float cx = vertices[center * 3], cy = vertices[center * 3 + 1], cz = vertices[center * 3 + 2];
    float rx = vertices[reference * 3] - cx;
    float ry = vertices[reference * 3 + 1] - cy;
    float rz = vertices[reference * 3 + 2] - cz;
    float ax = vertices[vertex * 3] - cx;
    float ay = vertices[vertex * 3 + 1] - cy;
    float az = vertices[vertex * 3 + 2] - cz;
    return atan2(ax * ry - ay * rx, ax * rx + ay * ry + az * rz);
}

kernel void generate_spiral_indices(
    device const float* vertices [[buffer(0)]],
    device const float* faces [[buffer(1)]],
    device float* visited [[buffer(2)]],
    device float* currentRing [[buffer(3)]],
    device float* nextRing [[buffer(4)]],
    device float* output [[buffer(5)]],
    constant int& numVertices [[buffer(6)]],
    constant int& numFaces [[buffer(7)]],
    constant int& spiralLength [[buffer(8)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid != 0) return;
    for (int center = 0; center < numVertices; center++) {
        for (int i = 0; i < numVertices; i++) visited[i] = 0.0f;
        int currentCount = spiral_build_neighbors(faces, numFaces, center, currentRing, numVertices);
        if (currentCount > 1) {
            int reference = int(currentRing[0]);
            for (int i = 1; i < currentCount; i++) {
                float key = currentRing[i];
                float keyAngle = spiral_angle(vertices, center, reference, int(key));
                int j = i - 1;
                while (j >= 0 && spiral_angle(vertices, center, reference, int(currentRing[j])) > keyAngle) {
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
                int neighbor = int(currentRing[r]);
                if (neighbor < 0 || neighbor >= numVertices || visited[neighbor] != 0.0f) continue;
                output[center * spiralLength + outputIndex++] = float(neighbor);
                visited[neighbor] = 1.0f;
                for (int f = 0; f < numFaces; f++) {
                    int v0 = int(faces[f * 3]), v1 = int(faces[f * 3 + 1]), v2 = int(faces[f * 3 + 2]);
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

inline float ctc_log_add(float a, float b)
{
    const float negativeSentinel = -3.402823466e+38f;
    if (a <= negativeSentinel) return b;
    if (b <= negativeSentinel) return a;
    float m = max(a, b);
    return m + log(exp(a - m) + exp(b - m));
}

kernel void ctc_loss_forward(
    device const float* logProbs [[buffer(0)]],
    device const float* targets [[buffer(1)]],
    device const float* inputLengths [[buffer(2)]],
    device const float* targetLengths [[buffer(3)]],
    device float* workspace [[buffer(4)]],
    device float* losses [[buffer(5)]],
    constant int& maxTime [[buffer(6)]],
    constant int& batchSize [[buffer(7)]],
    constant int& numClasses [[buffer(8)]],
    constant int& maxTargetLength [[buffer(9)]],
    constant int& blank [[buffer(10)]],
    uint gid [[thread_position_in_grid]])
{
    int n = int(gid);
    if (n >= batchSize) return;
    int timeLength = int(inputLengths[n]);
    int targetLength = int(targetLengths[n]);
    int targetOffset = 0;
    for (int i = 0; i < n; i++) targetOffset += int(targetLengths[i]);
    int states = 2 * targetLength + 1;
    int maxStates = 2 * maxTargetLength + 1;
    int previous = n * 2 * maxStates;
    int current = previous + maxStates;
    for (int s = 0; s < states; s++) workspace[previous + s] = -3.402823466e+38f;

    workspace[previous] = logProbs[n * numClasses + blank];
    if (states > 1) {
        int label = int(targets[targetOffset]);
        workspace[previous + 1] = logProbs[n * numClasses + label];
    }
    for (int t = 1; t < timeLength; t++) {
        for (int s = 0; s < states; s++) {
            int label = (s & 1) == 0 ? blank : int(targets[targetOffset + s / 2]);
            float sum = workspace[previous + s];
            if (s >= 1) sum = ctc_log_add(sum, workspace[previous + s - 1]);
            if (s >= 2) {
                int priorLabel = (s & 1) == 0 ? blank : int(targets[targetOffset + (s - 2) / 2]);
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
