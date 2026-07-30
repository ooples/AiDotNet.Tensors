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
