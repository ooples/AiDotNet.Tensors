namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

public static class VulkanInstantNgpKernels
{
    private const string Header = @"#version 450
layout(local_size_x = 256) in;
uint instant_ngp_hash(int x, int y, int z, int tableSize) {
    uint hx = uint(x) * 73856093u;
    uint hy = uint(y) * 19349663u;
    uint hz = uint(z) * 83492791u;
    return (hx ^ hy ^ hz) % uint(tableSize);
}
float instant_ngp_clamp_position(float value) {
    if (value <= 0.0) return 0.0;
    if (value >= 0.999999) return 0.999999;
    return value;
}
";

    public static string Forward => Header + @"
layout(set = 0, binding = 0) readonly buffer Positions { float positions[]; };
layout(set = 0, binding = 1) readonly buffer HashTable { float hashTable[]; };
layout(set = 0, binding = 2) writeonly buffer Output { float output_[]; };
layout(push_constant) uniform P {
    int numPoints; int resolution; int tableSize; int featuresPerLevel;
    int levelOffset; int outputStride;
};
void main() {
    int gid = int(gl_GlobalInvocationID.x);
    int total = numPoints * featuresPerLevel;
    if (gid >= total) return;
    int n = gid / featuresPerLevel;
    int f = gid - n * featuresPerLevel;
    float gx = instant_ngp_clamp_position(positions[n * 3]) * float(resolution);
    float gy = instant_ngp_clamp_position(positions[n * 3 + 1]) * float(resolution);
    float gz = instant_ngp_clamp_position(positions[n * 3 + 2]) * float(resolution);
    int x0 = int(floor(gx)), y0 = int(floor(gy)), z0 = int(floor(gz));
    int x1 = x0 + 1, y1 = y0 + 1, z1 = z0 + 1;
    float fx = gx - float(x0), fy = gy - float(y0), fz = gz - float(z0);
    float ix = 1.0 - fx, iy = 1.0 - fy, iz = 1.0 - fz;
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
    output_[n * outputStride + levelOffset + f] = value;
}
";

    public static string Backward => Header + @"
layout(set = 0, binding = 0) readonly buffer Positions { float positions[]; };
layout(set = 0, binding = 1) readonly buffer OutputGradient { float outputGradient[]; };
layout(set = 0, binding = 2) writeonly buffer TableGradient { float tableGradient[]; };
layout(push_constant) uniform P {
    int numPoints; int resolution; int tableSize; int featuresPerLevel;
    int levelOffset; int outputStride;
};
void main() {
    int gid = int(gl_GlobalInvocationID.x);
    int total = tableSize * featuresPerLevel;
    if (gid >= total) return;
    uint entry = uint(gid / featuresPerLevel);
    int f = gid - int(entry) * featuresPerLevel;
    float acc = 0.0;
    for (int n = 0; n < numPoints; n++) {
        float gx = instant_ngp_clamp_position(positions[n * 3]) * float(resolution);
        float gy = instant_ngp_clamp_position(positions[n * 3 + 1]) * float(resolution);
        float gz = instant_ngp_clamp_position(positions[n * 3 + 2]) * float(resolution);
        int x0 = int(floor(gx)), y0 = int(floor(gy)), z0 = int(floor(gz));
        int x1 = x0 + 1, y1 = y0 + 1, z1 = z0 + 1;
        float fx = gx - float(x0), fy = gy - float(y0), fz = gz - float(z0);
        float ix = 1.0 - fx, iy = 1.0 - fy, iz = 1.0 - fz;
        float grad = outputGradient[n * outputStride + levelOffset + f];
        if (abs(grad) < 1.0e-10) continue;
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
";

    public static string UniqueConsecutive => @"#version 450
layout(local_size_x = 1) in;
layout(set = 0, binding = 0) readonly buffer Input { float input_[]; };
layout(set = 0, binding = 1) writeonly buffer Output { float output_[]; };
layout(set = 0, binding = 2) writeonly buffer Count { float outputCount[]; };
layout(push_constant) uniform P { int length; };
void main() {
    if (gl_GlobalInvocationID.x != 0) return;
    if (length <= 0) { outputCount[0] = 0.0; return; }
    int count = 1;
    output_[0] = input_[0];
    for (int i = 1; i < length; i++) {
        if (input_[i] != input_[i - 1]) output_[count++] = input_[i];
    }
    outputCount[0] = float(count);
}
";

    public static string Frexp => @"#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) readonly buffer Input { float input_[]; };
layout(set = 0, binding = 1) writeonly buffer Mantissa { float mantissa[]; };
layout(set = 0, binding = 2) writeonly buffer Exponent { float exponent[]; };
layout(push_constant) uniform P { int length; };
void main() {
    int gid = int(gl_GlobalInvocationID.x);
    if (gid >= length) return;
    float value = input_[gid];
    uint bits = floatBitsToUint(value);
    uint magnitude = bits & 0x7fffffffu;
    uint exponentBits = magnitude >> 23;
    if (exponentBits == 0xffu || magnitude == 0u) {
        mantissa[gid] = value;
        exponent[gid] = 0.0;
        return;
    }
    int scaleAdjustment = 0;
    if (exponentBits == 0u) {
        value *= 16777216.0;
        bits = floatBitsToUint(value);
        exponentBits = (bits & 0x7fffffffu) >> 23;
        scaleAdjustment = -24;
    }
    mantissa[gid] = uintBitsToFloat((bits & 0x807fffffu) | (126u << 23));
    exponent[gid] = float(int(exponentBits) - 126 + scaleAdjustment);
}
";

    public static string UniqueConsecutiveWithInfo => @"#version 450
layout(local_size_x = 1) in;
layout(set = 0, binding = 0) readonly buffer Input { float input_[]; };
layout(set = 0, binding = 1) writeonly buffer Values { float outputValues[]; };
layout(set = 0, binding = 2) writeonly buffer Inverse { float outputInverse[]; };
layout(set = 0, binding = 3) buffer Counts { float outputCounts[]; };
layout(set = 0, binding = 4) writeonly buffer Count { float outputCount[]; };
layout(push_constant) uniform P { int length; int returnInverse; int returnCounts; };
void main() {
    if (gl_GlobalInvocationID.x != 0) return;
    if (length <= 0) { outputCount[0] = 0.0; return; }
    int count = 1;
    outputValues[0] = input_[0];
    if (returnInverse != 0) outputInverse[0] = 0.0;
    if (returnCounts != 0) outputCounts[0] = 1.0;
    for (int i = 1; i < length; i++) {
        if (input_[i] != input_[i - 1]) {
            outputValues[count] = input_[i];
            if (returnCounts != 0) outputCounts[count] = 1.0;
            count++;
        } else if (returnCounts != 0) {
            outputCounts[count - 1] += 1.0;
        }
        if (returnInverse != 0) outputInverse[i] = float(count - 1);
    }
    outputCount[0] = float(count);
}
";

    public static string UniqueSortedWithInfo => @"#version 450
layout(local_size_x = 1) in;
layout(set = 0, binding = 0) readonly buffer Input { float sortedInput[]; };
layout(set = 0, binding = 1) readonly buffer Original { float sortedOriginalIndices[]; };
layout(set = 0, binding = 2) writeonly buffer Values { float outputValues[]; };
layout(set = 0, binding = 3) writeonly buffer Inverse { float outputInverse[]; };
layout(set = 0, binding = 4) buffer Counts { float outputCounts[]; };
layout(set = 0, binding = 5) writeonly buffer Count { float outputCount[]; };
layout(push_constant) uniform P { int length; int returnInverse; int returnCounts; };
void main() {
    if (gl_GlobalInvocationID.x != 0) return;
    int count = 0;
    for (int i = 0; i < length; i++) {
        if (i == 0 || sortedInput[i] != sortedInput[i - 1]) {
            outputValues[count] = sortedInput[i];
            if (returnCounts != 0) outputCounts[count] = 0.0;
            count++;
        }
        int group = count - 1;
        if (returnCounts != 0) outputCounts[group] += 1.0;
        if (returnInverse != 0)
            outputInverse[int(sortedOriginalIndices[i])] = float(group);
    }
    outputCount[0] = float(count);
}
";

    public static string ProjectGaussians3DTo2D => @"#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) readonly buffer Means3D { float means3D[]; };
layout(set = 0, binding = 1) readonly buffer Covariances3D { float covariances3D[]; };
layout(set = 0, binding = 2) writeonly buffer Means2D { float means2D[]; };
layout(set = 0, binding = 3) writeonly buffer Covariances2D { float covariances2D[]; };
layout(set = 0, binding = 4) writeonly buffer Depths { float depths[]; };
layout(set = 0, binding = 5) writeonly buffer Visible { float visible[]; };
layout(push_constant) uniform P {
    int numGaussians; int covarianceStride; int imageWidth; int imageHeight;
    float v00; float v01; float v02; float v03;
    float v10; float v11; float v12; float v13;
    float v20; float v21; float v22; float v23; float p00; float p11;
};
void main() {
    int i = int(gl_GlobalInvocationID.x);
    if (i >= numGaussians) return;
    means2D[i * 2] = 0.0; means2D[i * 2 + 1] = 0.0;
    covariances2D[i * 3] = 0.0; covariances2D[i * 3 + 1] = 0.0;
    covariances2D[i * 3 + 2] = 0.0; depths[i] = 0.0; visible[i] = 0.0;
    float mx = means3D[i * 3], my = means3D[i * 3 + 1], mz = means3D[i * 3 + 2];
    float camX = v00 * mx + v01 * my + v02 * mz + v03;
    float camY = v10 * mx + v11 * my + v12 * mz + v13;
    float camZ = v20 * mx + v21 * my + v22 * mz + v23;
    if (camZ <= 0.001) return;
    float invZ = 1.0 / camZ;
    float cx = float(imageWidth) * 0.5, cy = float(imageHeight) * 0.5;
    float screenX = p00 * camX * invZ * cx + cx;
    float screenY = p11 * camY * invZ * cy + cy;
    if (screenX < -float(imageWidth) || screenX > 2.0 * float(imageWidth) ||
        screenY < -float(imageHeight) || screenY > 2.0 * float(imageHeight)) return;
    int c = i * covarianceStride;
    float c00 = covariances3D[c], c01 = covariances3D[c + 1], c02 = covariances3D[c + 2];
    float c11 = covariances3D[c + (covarianceStride == 6 ? 3 : 4)];
    float c12 = covariances3D[c + (covarianceStride == 6 ? 4 : 5)];
    float c22 = covariances3D[c + (covarianceStride == 6 ? 5 : 8)];
    float j00 = p00 * invZ, j02 = -p00 * camX * invZ * invZ;
    float j11 = p11 * invZ, j12 = -p11 * camY * invZ * invZ;
    float cov00 = j00*j00*c00 + 2.0*j00*j02*c02 + j02*j02*c22 + 0.3;
    float cov01 = j00*j11*c01 + j00*j12*c02 + j02*j11*c12 + j02*j12*c22;
    float cov11 = j11*j11*c11 + 2.0*j11*j12*c12 + j12*j12*c22 + 0.3;
    means2D[i * 2] = screenX; means2D[i * 2 + 1] = screenY;
    covariances2D[i * 3] = cov00; covariances2D[i * 3 + 1] = cov01;
    covariances2D[i * 3 + 2] = cov11; depths[i] = camZ; visible[i] = 1.0;
}
";

    public static string SampleRaysWithOccupancy => @"#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) readonly buffer Origins { float rayOrigins[]; };
layout(set = 0, binding = 1) readonly buffer DirectionsIn { float rayDirections[]; };
layout(set = 0, binding = 2) readonly buffer Occupancy { uint occupancyBitfield[]; };
layout(set = 0, binding = 3) writeonly buffer Positions { float positions[]; };
layout(set = 0, binding = 4) writeonly buffer DirectionsOut { float directions[]; };
layout(set = 0, binding = 5) writeonly buffer Valid { float validMask[]; };
layout(set = 0, binding = 6) writeonly buffer TValues { float tValues[]; };
layout(push_constant) uniform P {
    int numRays; int occupancyWordCount; int gridSize; int maxSamples;
    float minX; float minY; float minZ; float maxX; float maxY; float maxZ;
    float nearBound; float farBound;
};
bool sample_ray_axis(float origin, float direction, float minimum, float maximum,
    inout float tMin, inout float tMax) {
    if (abs(direction) < 1.0e-8) return origin >= minimum && origin <= maximum;
    float t1 = (minimum - origin) / direction;
    float t2 = (maximum - origin) / direction;
    if (t1 > t2) { float temporary = t1; t1 = t2; t2 = temporary; }
    tMin = max(tMin, t1); tMax = min(tMax, t2);
    return tMax >= tMin;
}
void main() {
    int sampleIndex = int(gl_GlobalInvocationID.x);
    int totalSamples = numRays * maxSamples;
    if (sampleIndex >= totalSamples) return;
    int positionIndex = sampleIndex * 3;
    positions[positionIndex] = 0.0; positions[positionIndex + 1] = 0.0;
    positions[positionIndex + 2] = 0.0; directions[positionIndex] = 0.0;
    directions[positionIndex + 1] = 0.0; directions[positionIndex + 2] = 0.0;
    validMask[sampleIndex] = 0.0; tValues[sampleIndex] = 0.0;
    int ray = sampleIndex / maxSamples, sample = sampleIndex - ray * maxSamples;
    float ox = rayOrigins[ray * 3], oy = rayOrigins[ray * 3 + 1], oz = rayOrigins[ray * 3 + 2];
    float dx = rayDirections[ray * 3], dy = rayDirections[ray * 3 + 1], dz = rayDirections[ray * 3 + 2];
    float tMin = nearBound, tMax = farBound;
    if (!sample_ray_axis(ox, dx, minX, maxX, tMin, tMax) ||
        !sample_ray_axis(oy, dy, minY, maxY, tMin, tMax) ||
        !sample_ray_axis(oz, dz, minZ, maxZ, tMin, tMax) || tMax < tMin) return;
    float t = tMin + ((tMax - tMin) / float(maxSamples)) * (float(sample) + 0.5);
    float px = ox + t * dx, py = oy + t * dy, pz = oz + t * dz;
    float nx = clamp((px - minX) / max(1.0e-10, maxX - minX), 0.0, 0.999999);
    float ny = clamp((py - minY) / max(1.0e-10, maxY - minY), 0.0, 0.999999);
    float nz = clamp((pz - minZ) / max(1.0e-10, maxZ - minZ), 0.0, 0.999999);
    int gx = min(int(nx * float(gridSize)), gridSize - 1);
    int gy = min(int(ny * float(gridSize)), gridSize - 1);
    int gz = min(int(nz * float(gridSize)), gridSize - 1);
    int cell = (gx * gridSize + gy) * gridSize + gz;
    int word = cell >> 5, bit = cell & 31;
    bool occupied = word < occupancyWordCount &&
        (occupancyBitfield[word] & (1u << uint(bit))) != 0u;
    positions[positionIndex] = px; positions[positionIndex + 1] = py; positions[positionIndex + 2] = pz;
    directions[positionIndex] = dx; directions[positionIndex + 1] = dy; directions[positionIndex + 2] = dz;
    validMask[sampleIndex] = occupied ? 1.0 : 0.0; tValues[sampleIndex] = t;
}
";

    public static string Nonzero => @"#version 450
layout(local_size_x = 1) in;
layout(set = 0, binding = 0) readonly buffer Input { float input_[]; };
layout(set = 0, binding = 1) readonly buffer Strides { int strides[]; };
layout(set = 0, binding = 2) writeonly buffer Output { float output_[]; };
layout(set = 0, binding = 3) writeonly buffer Count { float outputCount[]; };
layout(push_constant) uniform P { int length; int rank; };
void main() {
    if (gl_GlobalInvocationID.x != 0) return;
    int count = 0;
    for (int i = 0; i < length; i++) {
        if (input_[i] != 0.0) {
            int rem = i;
            for (int d = 0; d < rank; d++) {
                int stride = strides[d];
                output_[count * rank + d] = float(rem / stride);
                rem %= stride;
            }
            count++;
        }
    }
    outputCount[0] = float(count);
}
";

    public static string Mode => @"#version 450
layout(local_size_x = 1) in;
layout(set = 0, binding = 0) readonly buffer Input { float input_[]; };
layout(set = 0, binding = 1) writeonly buffer Output { float output_[]; };
layout(push_constant) uniform P { int length; };
bool mode_equal(float left, float right) {
    return left == right || (isnan(left) && isnan(right));
}
void main() {
    if (gl_GlobalInvocationID.x != 0) return;
    float bestValue = 0.0;
    int bestCount = -1;
    for (int i = 0; i < length; i++) {
        float candidate = input_[i];
        bool first = true;
        for (int j = 0; j < i; j++) {
            if (mode_equal(candidate, input_[j])) { first = false; break; }
        }
        if (!first) continue;
        int count = 0;
        for (int j = 0; j < length; j++) {
            if (mode_equal(candidate, input_[j])) count++;
        }
        if (bestCount < 0 || count > bestCount ||
            (count == bestCount && candidate < bestValue)) {
            bestValue = candidate;
            bestCount = count;
        }
    }
    output_[0] = bestValue;
    output_[1] = float(bestCount);
}
";

    public static string ConvertIndicesToInt32 => @"#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) readonly buffer Input { float input_[]; };
layout(set = 0, binding = 1) writeonly buffer Output { int output_[]; };
layout(push_constant) uniform P { int length; };
void main() {
    int gid = int(gl_GlobalInvocationID.x);
    if (gid < length) output_[gid] = int(input_[gid]);
}
";

    public static string IndexAdd => @"#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) readonly buffer Destination { float destination[]; };
layout(set = 0, binding = 1) readonly buffer Indices { int indices[]; };
layout(set = 0, binding = 2) readonly buffer Source { float source[]; };
layout(set = 0, binding = 3) writeonly buffer Output { float output_[]; };
layout(push_constant) uniform P {
    int outerSize; int sourceAxis; int destinationAxis; int innerSize;
};
void main() {
    int gid = int(gl_GlobalInvocationID.x);
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
    output_[gid] = value;
}
";

    public static string IndexSelect => @"#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) readonly buffer Source { float source[]; };
layout(set = 0, binding = 1) readonly buffer Indices { int indices[]; };
layout(set = 0, binding = 2) writeonly buffer Output { float output_[]; };
layout(push_constant) uniform P {
    int outerSize; int sourceAxis; int indexAxis; int innerSize;
};
void main() {
    int gid = int(gl_GlobalInvocationID.x);
    int total = outerSize * indexAxis * innerSize;
    if (gid >= total) return;
    int inner = gid % innerSize;
    int j = (gid / innerSize) % indexAxis;
    int outer = (gid / innerSize) / indexAxis;
    int sourceIndex = indices[j];
    output_[gid] = source[(outer * sourceAxis + sourceIndex) * innerSize + inner];
}
";

    public static string ScatterMaxWithArgmaxRows => @"#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) readonly buffer Source { float source[]; };
layout(set = 0, binding = 1) readonly buffer Indices { int indices[]; };
layout(set = 0, binding = 2) writeonly buffer Output { float output_[]; };
layout(set = 0, binding = 3) writeonly buffer Argmax { float argmax_[]; };
layout(push_constant) uniform P { int sourceRows; int innerSize; int outputRows; };
void main() {
    int gid = int(gl_GlobalInvocationID.x);
    int total = outputRows * innerSize;
    if (gid >= total) return;
    int group = gid / innerSize;
    int inner = gid % innerSize;
    float best = uintBitsToFloat(0xff800000u);
    int bestRow = -1;
    for (int row = 0; row < sourceRows; row++) {
        float value = source[row * innerSize + inner];
        if (indices[row] == group && value > best) {
            best = value;
            bestRow = row;
        }
    }
    output_[gid] = best;
    argmax_[gid] = float(bestRow);
}
";

    public static string UniformMeshLaplacian => @"#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) readonly buffer Faces { int faces[]; };
layout(set = 0, binding = 1) writeonly buffer Output { float output_[]; };
layout(push_constant) uniform P { int numFaces; int numVertices; };
void main() {
    int gid = int(gl_GlobalInvocationID.x);
    int total = numVertices * numVertices;
    if (gid >= total) return;
    int row = gid / numVertices;
    int column = gid % numVertices;
    float value = 0.0;
    for (int face = 0; face < numFaces; face++) {
        int v0 = faces[face * 3];
        int v1 = faces[face * 3 + 1];
        int v2 = faces[face * 3 + 2];
        if (row == v0 && column == v1) value -= 1.0;
        if (row == v1 && column == v0) value -= 1.0;
        if (row == v0 && column == v0) value += 1.0;
        if (row == v1 && column == v1) value += 1.0;
        if (row == v1 && column == v2) value -= 1.0;
        if (row == v2 && column == v1) value -= 1.0;
        if (row == v1 && column == v1) value += 1.0;
        if (row == v2 && column == v2) value += 1.0;
        if (row == v2 && column == v0) value -= 1.0;
        if (row == v0 && column == v2) value -= 1.0;
        if (row == v2 && column == v2) value += 1.0;
        if (row == v0 && column == v0) value += 1.0;
    }
    output_[gid] = value;
}
";

    public static string ScatterAddRows => @"#version 450
layout(local_size_x = 256) in;
layout(set=0,binding=0) readonly buffer Source { float source[]; };
layout(set=0,binding=1) readonly buffer Indices { int indices[]; };
layout(set=0,binding=2) writeonly buffer Output { float output_[]; };
layout(push_constant) uniform P { int sourceRows; int innerSize; int outputRows; };
void main() {
    int gid=int(gl_GlobalInvocationID.x), total=outputRows*innerSize;
    if(gid>=total)return; int inner=gid%innerSize, group=gid/innerSize; float sum=0.0;
    for(int row=0;row<sourceRows;row++)if(indices[row]==group)sum+=source[row*innerSize+inner];
    output_[gid]=sum;
}";

    public static string ScatterMeanRowsWithCounts => @"#version 450
layout(local_size_x = 256) in;
layout(set=0,binding=0) readonly buffer Source { float source[]; };
layout(set=0,binding=1) readonly buffer Indices { int indices[]; };
layout(set=0,binding=2) writeonly buffer Output { float output_[]; };
layout(set=0,binding=3) writeonly buffer Counts { float counts[]; };
layout(push_constant) uniform P { int sourceRows; int innerSize; int outputRows; };
void main() {
    int gid=int(gl_GlobalInvocationID.x), outputTotal=outputRows*innerSize;
    if(gid<outputRows){int count=0;for(int row=0;row<sourceRows;row++)if(indices[row]==gid)count++;counts[gid]=float(count);}
    if(gid>=outputTotal)return; int inner=gid%innerSize, group=gid/innerSize; float sum=0.0; int count=0;
    for(int row=0;row<sourceRows;row++)if(indices[row]==group){sum+=source[row*innerSize+inner];count++;}
    output_[gid]=count>0?sum*(1.0/float(count)):0.0;
}";

    public static string ScatterSoftmaxRows => @"#version 450
layout(local_size_x = 256) in;
layout(set=0,binding=0) readonly buffer Source { float source[]; };
layout(set=0,binding=1) readonly buffer Indices { int indices[]; };
layout(set=0,binding=2) writeonly buffer Output { float output_[]; };
layout(push_constant) uniform P { int sourceRows; int innerSize; int numGroups; };
void main(){
    int gid=int(gl_GlobalInvocationID.x),total=sourceRows*innerSize;if(gid>=total)return;
    int inner=gid%innerSize,row=gid/innerSize,group=indices[row];if(group<0||group>=numGroups){output_[gid]=0.0;return;}
    float maximum=uintBitsToFloat(0xff800000u);for(int other=0;other<sourceRows;other++)if(indices[other]==group)maximum=max(maximum,source[other*innerSize+inner]);
    float sum=0.0;for(int other=0;other<sourceRows;other++)if(indices[other]==group)sum+=exp(source[other*innerSize+inner]-maximum);
    float value=exp(source[gid]-maximum);output_[gid]=sum!=0.0?value/sum:value;
}";

    public static string ScatterAddBackwardRows => @"#version 450
layout(local_size_x = 256) in;
layout(set=0,binding=0) readonly buffer GradOutput { float gradOutput[]; };
layout(set=0,binding=1) readonly buffer Indices { int indices[]; };
layout(set=0,binding=2) writeonly buffer GradSource { float gradSource[]; };
layout(push_constant) uniform P { int sourceRows; int innerSize; int outputRows; };
void main(){int gid=int(gl_GlobalInvocationID.x),total=sourceRows*innerSize;if(gid>=total)return;int inner=gid%innerSize,group=indices[gid/innerSize];gradSource[gid]=group>=0&&group<outputRows?gradOutput[group*innerSize+inner]:0.0;}";

    public static string ScatterMeanBackwardRows => @"#version 450
layout(local_size_x = 256) in;
layout(set=0,binding=0) readonly buffer GradOutput { float gradOutput[]; };
layout(set=0,binding=1) readonly buffer Indices { int indices[]; };
layout(set=0,binding=2) readonly buffer Counts { int counts[]; };
layout(set=0,binding=3) writeonly buffer GradSource { float gradSource[]; };
layout(push_constant) uniform P { int sourceRows; int innerSize; int outputRows; };
void main(){int gid=int(gl_GlobalInvocationID.x),total=sourceRows*innerSize;if(gid>=total)return;int inner=gid%innerSize,group=indices[gid/innerSize];if(group<0||group>=outputRows){gradSource[gid]=0.0;return;}int count=counts[group];float divisor=count>0?float(count):1.0;gradSource[gid]=gradOutput[group*innerSize+inner]/divisor;}";

    public static string ScatterMaxBackwardRows => @"#version 450
layout(local_size_x = 256) in;
layout(set=0,binding=0) readonly buffer GradOutput { float gradOutput[]; };
layout(set=0,binding=1) readonly buffer Argmax { int argmax_[]; };
layout(set=0,binding=2) writeonly buffer GradSource { float gradSource[]; };
layout(push_constant) uniform P { int sourceRows; int innerSize; int outputRows; };
void main(){int gid=int(gl_GlobalInvocationID.x),total=sourceRows*innerSize;if(gid>=total)return;int inner=gid%innerSize,sourceRow=gid/innerSize;float value=0.0;for(int group=0;group<outputRows;group++)if(argmax_[group*innerSize+inner]==sourceRow)value=gradOutput[group*innerSize+inner];gradSource[gid]=value;}";

    public static string ScatterSoftmaxBackwardRows => @"#version 450
layout(local_size_x = 256) in;
layout(set=0,binding=0) readonly buffer GradOutput { float gradOutput[]; };
layout(set=0,binding=1) readonly buffer Output { float output_[]; };
layout(set=0,binding=2) readonly buffer Indices { int indices[]; };
layout(set=0,binding=3) writeonly buffer GradSource { float gradSource[]; };
layout(push_constant) uniform P { int sourceRows; int innerSize; int numGroups; };
void main(){int gid=int(gl_GlobalInvocationID.x),total=sourceRows*innerSize;if(gid>=total)return;int inner=gid%innerSize,group=indices[gid/innerSize];if(group<0||group>=numGroups){gradSource[gid]=0.0;return;}float sum=0.0;for(int row=0;row<sourceRows;row++)if(indices[row]==group)sum+=output_[row*innerSize+inner]*gradOutput[row*innerSize+inner];gradSource[gid]=output_[gid]*(gradOutput[gid]-sum);}";

    public static string ImportanceSampling => @"#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) readonly buffer TValues { float tValues[]; };
layout(set = 0, binding = 1) readonly buffer Weights { float weights[]; };
layout(set = 0, binding = 2) writeonly buffer Output { float output_[]; };
layout(push_constant) uniform P {
    int numRays; int numCoarse; int numFine; uint seed;
};
uint importance_hash(uint x) {
    x ^= x >> 16; x *= 0x7feb352du;
    x ^= x >> 15; x *= 0x846ca68bu;
    return x ^ (x >> 16);
}
void main() {
    int gid = int(gl_GlobalInvocationID.x);
    int total = numRays * numFine;
    if (gid >= total) return;
    int ray = gid / numFine;
    int sample = gid - ray * numFine;
    int base = ray * numCoarse;
    uint bits = importance_hash(seed ^ (uint(gid) * 747796405u + 2891336453u));
    float random = float(bits >> 8) * (1.0 / 16777216.0);
    float u = (float(sample) + random) / float(numFine);
    float weightSum = 0.0;
    for (int s = 0; s < numCoarse; s++) {
        float weight = weights[base + s];
        weightSum += weight > 0.0 ? weight : 0.0;
    }
    if (weightSum <= 1.0e-10) {
        float tMin = tValues[base];
        float tMax = tValues[base + numCoarse - 1];
        output_[gid] = tMin + u * (tMax - tMin);
        return;
    }
    float previous = 0.0;
    float current = 0.0;
    int index = 0;
    for (int s = 0; s < numCoarse; s++) {
        float weight = weights[base + s];
        current += (weight > 0.0 ? weight : 0.0) / weightSum;
        index = s;
        if (u <= current || s == numCoarse - 1) break;
        previous = current;
    }
    if (index == 0) { output_[gid] = tValues[base]; return; }
    float denominator = current - previous;
    float t0 = tValues[base + index - 1];
    float t1 = tValues[base + index];
    output_[gid] = denominator > 1.0e-10
        ? t0 + ((u - previous) / denominator) * (t1 - t0)
        : t0;
}
";

    public static string Nms => @"#version 450
layout(local_size_x = 1) in;
layout(set = 0, binding = 0) readonly buffer Boxes { float boxes[]; };
layout(set = 0, binding = 1) readonly buffer Scores { float scores[]; };
layout(set = 0, binding = 2) readonly buffer ClassIds { float classIds[]; };
layout(set = 0, binding = 3) buffer Suppressed { float suppressed[]; };
layout(set = 0, binding = 4) writeonly buffer Output { float output_[]; };
layout(set = 0, binding = 5) writeonly buffer Count { float outputCount[]; };
layout(push_constant) uniform P { int length; float threshold; int batched; };
void main() {
    if (gl_GlobalInvocationID.x != 0) return;
    int count = 0;
    for (int iteration = 0; iteration < length; iteration++) {
        int best = -1;
        float bestScore = -3.402823466e+38;
        for (int i = 0; i < length; i++) {
            if (suppressed[i] != 0.0) continue;
            float score = scores[i];
            if (best < 0 || (!isnan(score) &&
                (isnan(bestScore) || score > bestScore || (score == bestScore && i < best)))) {
                best = i; bestScore = score;
            }
        }
        if (best < 0) break;
        suppressed[best] = 1.0;
        output_[count++] = float(best);
        float ix1 = boxes[best * 4], iy1 = boxes[best * 4 + 1];
        float ix2 = boxes[best * 4 + 2], iy2 = boxes[best * 4 + 3];
        float iw0 = ix2 - ix1, ih0 = iy2 - iy1;
        float areaI = (iw0 > 0.0 && ih0 > 0.0) ? iw0 * ih0 : 0.0;
        for (int j = 0; j < length; j++) {
            if (suppressed[j] != 0.0) continue;
            if (batched != 0 && classIds[j] != classIds[best]) continue;
            float jx1 = boxes[j * 4], jy1 = boxes[j * 4 + 1];
            float jx2 = boxes[j * 4 + 2], jy2 = boxes[j * 4 + 3];
            float jw0 = jx2 - jx1, jh0 = jy2 - jy1;
            float areaJ = (jw0 > 0.0 && jh0 > 0.0) ? jw0 * jh0 : 0.0;
            float overlapW = max(0.0, min(ix2, jx2) - max(ix1, jx1));
            float overlapH = max(0.0, min(iy2, jy2) - max(iy1, jy1));
            float intersection = overlapW * overlapH;
            float unionArea = areaI + areaJ - intersection;
            if (unionArea > 0.0 && intersection / unionArea > threshold)
                suppressed[j] = 1.0;
        }
    }
    outputCount[0] = float(count);
}
";

    public static string SpiralIndices => @"#version 450
layout(local_size_x = 1) in;
layout(set = 0, binding = 0) readonly buffer Vertices { float vertices[]; };
layout(set = 0, binding = 1) readonly buffer Faces { float faces[]; };
layout(set = 0, binding = 2) buffer Visited { float visited[]; };
layout(set = 0, binding = 3) buffer CurrentRing { float currentRing[]; };
layout(set = 0, binding = 4) buffer NextRing { float nextRing[]; };
layout(set = 0, binding = 5) writeonly buffer Output { float output_[]; };
layout(push_constant) uniform P { int numVertices; int numFaces; int spiralLength; };
int append_unique(int count, int candidate) {
    for (int i = 0; i < count; i++) if (int(nextRing[i]) == candidate) return count;
    if (count < numVertices) nextRing[count++] = float(candidate);
    return count;
}
int append_current_unique(int count, int candidate) {
    for (int i = 0; i < count; i++) if (int(currentRing[i]) == candidate) return count;
    if (count < numVertices) currentRing[count++] = float(candidate);
    return count;
}
int build_neighbors(int vertex) {
    int count = 0;
    for (int f = 0; f < numFaces; f++) {
        int v0 = int(faces[f * 3]), v1 = int(faces[f * 3 + 1]), v2 = int(faces[f * 3 + 2]);
        if (vertex == v0) { count = append_current_unique(count, v1); count = append_current_unique(count, v2); }
        else if (vertex == v1) { count = append_current_unique(count, v0); count = append_current_unique(count, v2); }
        else if (vertex == v2) { count = append_current_unique(count, v0); count = append_current_unique(count, v1); }
    }
    return count;
}
float spiral_angle(int center, int reference, int vertex) {
    float cx = vertices[center * 3], cy = vertices[center * 3 + 1], cz = vertices[center * 3 + 2];
    float rx = vertices[reference * 3] - cx;
    float ry = vertices[reference * 3 + 1] - cy;
    float rz = vertices[reference * 3 + 2] - cz;
    float ax = vertices[vertex * 3] - cx;
    float ay = vertices[vertex * 3 + 1] - cy;
    float az = vertices[vertex * 3 + 2] - cz;
    return atan(ax * ry - ay * rx, ax * rx + ay * ry + az * rz);
}
void main() {
    if (gl_GlobalInvocationID.x != 0) return;
    for (int center = 0; center < numVertices; center++) {
        for (int i = 0; i < numVertices; i++) visited[i] = 0.0;
        int currentCount = build_neighbors(center);
        if (currentCount > 1) {
            int reference = int(currentRing[0]);
            for (int i = 1; i < currentCount; i++) {
                float key = currentRing[i];
                float keyAngle = spiral_angle(center, reference, int(key));
                int j = i - 1;
                while (j >= 0 && spiral_angle(center, reference, int(currentRing[j])) > keyAngle) {
                    currentRing[j + 1] = currentRing[j]; j--;
                }
                currentRing[j + 1] = key;
            }
        }
        visited[center] = 1.0;
        int outputIndex = 0;
        while (outputIndex < spiralLength && currentCount > 0) {
            int nextCount = 0;
            for (int r = 0; r < currentCount && outputIndex < spiralLength; r++) {
                int neighbor = int(currentRing[r]);
                if (neighbor < 0 || neighbor >= numVertices || visited[neighbor] != 0.0) continue;
                output_[center * spiralLength + outputIndex++] = float(neighbor);
                visited[neighbor] = 1.0;
                for (int f = 0; f < numFaces; f++) {
                    int v0 = int(faces[f * 3]), v1 = int(faces[f * 3 + 1]), v2 = int(faces[f * 3 + 2]);
                    int a = -1, b = -1;
                    if (neighbor == v0) { a = v1; b = v2; }
                    else if (neighbor == v1) { a = v0; b = v2; }
                    else if (neighbor == v2) { a = v0; b = v1; }
                    if (a >= 0 && a < numVertices && visited[a] == 0.0) nextCount = append_unique(nextCount, a);
                    if (b >= 0 && b < numVertices && visited[b] == 0.0) nextCount = append_unique(nextCount, b);
                }
            }
            currentCount = nextCount;
            for (int i = 0; i < nextCount; i++) currentRing[i] = nextRing[i];
        }
        while (outputIndex < spiralLength)
            output_[center * spiralLength + outputIndex++] = -1.0;
    }
}
";

    public static string CtcLoss => @"#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) readonly buffer LogProbs { float logProbs[]; };
layout(set = 0, binding = 1) readonly buffer Targets { float targets[]; };
layout(set = 0, binding = 2) readonly buffer InputLengths { float inputLengths[]; };
layout(set = 0, binding = 3) readonly buffer TargetLengths { float targetLengths[]; };
layout(set = 0, binding = 4) buffer Workspace { float workspace[]; };
layout(set = 0, binding = 5) writeonly buffer Losses { float losses[]; };
layout(push_constant) uniform P {
    int maxTime; int batchSize; int numClasses; int maxTargetLength; int blank;
};
float ctc_log_add(float a, float b) {
    const float negativeSentinel = -3.402823466e+38;
    if (a <= negativeSentinel) return b;
    if (b <= negativeSentinel) return a;
    float m = max(a, b);
    return m + log(exp(a - m) + exp(b - m));
}
void main() {
    int n = int(gl_GlobalInvocationID.x);
    if (n >= batchSize) return;
    int timeLength = int(inputLengths[n]);
    int targetLength = int(targetLengths[n]);
    int targetOffset = 0;
    for (int i = 0; i < n; i++) targetOffset += int(targetLengths[i]);
    int states = 2 * targetLength + 1;
    int maxStates = 2 * maxTargetLength + 1;
    int previous = n * 2 * maxStates;
    int current = previous + maxStates;
    const float negativeSentinel = -3.402823466e+38;
    for (int s = 0; s < states; s++) workspace[previous + s] = negativeSentinel;

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
