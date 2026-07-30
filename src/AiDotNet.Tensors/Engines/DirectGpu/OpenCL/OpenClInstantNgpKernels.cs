#if !NET462
namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL;

public static class OpenClInstantNgpKernels
{
    public static string[] GetKernelNames() =>
        ["instant_ngp_hash_encode_level", "instant_ngp_hash_encode_level_backward",
            "unique_consecutive_compact", "resident_mode", "resident_indices_to_int32", "resident_index_add", "resident_index_select", "resident_scatter_max_argmax_rows", "resident_uniform_mesh_laplacian", "resident_scatter_mean_rows_counts", "nonzero_compact", "ctc_loss_forward", "importance_sampling", "resident_nms",
            "generate_spiral_indices"];

    public static string GetSource() => @"
inline uint instant_ngp_hash(int x, int y, int z, int tableSize)
{
    uint hx = (uint)x * 73856093u;
    uint hy = (uint)y * 19349663u;
    uint hz = (uint)z * 83492791u;
    return (hx ^ hy ^ hz) % (uint)tableSize;
}

inline float instant_ngp_clamp_position(float value)
{
    if (value <= 0.0f) return 0.0f;
    if (value >= 0.999999f) return 0.999999f;
    return value;
}

__kernel void instant_ngp_hash_encode_level(
    __global const float* positions,
    __global const float* hashTable,
    __global float* output,
    int numPoints, int resolution, int tableSize, int featuresPerLevel,
    int levelOffset, int outputStride)
{
    int gid = get_global_id(0);
    int total = numPoints * featuresPerLevel;
    if (gid >= total) return;
    int n = gid / featuresPerLevel;
    int f = gid - n * featuresPerLevel;
    float gx = instant_ngp_clamp_position(positions[n * 3]) * (float)resolution;
    float gy = instant_ngp_clamp_position(positions[n * 3 + 1]) * (float)resolution;
    float gz = instant_ngp_clamp_position(positions[n * 3 + 2]) * (float)resolution;
    int x0 = (int)floor(gx), y0 = (int)floor(gy), z0 = (int)floor(gz);
    int x1 = x0 + 1, y1 = y0 + 1, z1 = z0 + 1;
    float fx = gx - (float)x0, fy = gy - (float)y0, fz = gz - (float)z0;
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

__kernel void instant_ngp_hash_encode_level_backward(
    __global const float* positions,
    __global const float* outputGradient,
    __global float* tableGradient,
    int numPoints, int resolution, int tableSize, int featuresPerLevel,
    int levelOffset, int outputStride)
{
    int gid = get_global_id(0);
    int total = tableSize * featuresPerLevel;
    if (gid >= total) return;
    uint entry = (uint)(gid / featuresPerLevel);
    int f = gid - (int)entry * featuresPerLevel;
    float acc = 0.0f;
    for (int n = 0; n < numPoints; n++) {
        float gx = instant_ngp_clamp_position(positions[n * 3]) * (float)resolution;
        float gy = instant_ngp_clamp_position(positions[n * 3 + 1]) * (float)resolution;
        float gz = instant_ngp_clamp_position(positions[n * 3 + 2]) * (float)resolution;
        int x0 = (int)floor(gx), y0 = (int)floor(gy), z0 = (int)floor(gz);
        int x1 = x0 + 1, y1 = y0 + 1, z1 = z0 + 1;
        float fx = gx - (float)x0, fy = gy - (float)y0, fz = gz - (float)z0;
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

__kernel void unique_consecutive_compact(
    __global const float* input,
    __global float* output,
    __global float* outputCount,
    int length)
{
    if (get_global_id(0) != 0) return;
    if (length <= 0) { outputCount[0] = 0.0f; return; }
    int count = 1;
    output[0] = input[0];
    for (int i = 1; i < length; i++) {
        if (input[i] != input[i - 1]) output[count++] = input[i];
    }
    outputCount[0] = (float)count;
}

__kernel void resident_mode(
    __global const float* input,
    __global float* output,
    int length)
{
    if (get_global_id(0) != 0) return;
    float bestValue = 0.0f;
    int bestCount = -1;
    for (int i = 0; i < length; i++) {
        float candidate = input[i];
        int first = 1;
        for (int j = 0; j < i; j++) {
            float prior = input[j];
            if (candidate == prior || (isnan(candidate) && isnan(prior))) {
                first = 0;
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

__kernel void resident_indices_to_int32(
    __global const float* input,
    __global int* output,
    int length)
{
    int gid = get_global_id(0);
    if (gid < length) output[gid] = (int)input[gid];
}

__kernel void resident_index_add(
    __global const float* destination,
    __global const int* indices,
    __global const float* source,
    __global float* output,
    int outerSize, int sourceAxis, int destinationAxis, int innerSize)
{
    int gid = get_global_id(0);
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

__kernel void resident_index_select(
    __global const float* source,
    __global const int* indices,
    __global float* output,
    int outerSize, int sourceAxis, int indexAxis, int innerSize)
{
    int gid = get_global_id(0);
    int total = outerSize * indexAxis * innerSize;
    if (gid >= total) return;
    int inner = gid % innerSize;
    int j = (gid / innerSize) % indexAxis;
    int outer = (gid / innerSize) / indexAxis;
    int sourceIndex = indices[j];
    output[gid] = source[(outer * sourceAxis + sourceIndex) * innerSize + inner];
}

__kernel void resident_scatter_max_argmax_rows(
    __global const float* source,
    __global const int* indices,
    __global float* output,
    __global float* argmax,
    int sourceRows, int innerSize, int outputRows)
{
    int gid = get_global_id(0);
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

__kernel void resident_uniform_mesh_laplacian(
    __global const int* faces,
    __global float* output,
    int numFaces, int numVertices)
{
    int gid = get_global_id(0);
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

__kernel void resident_scatter_mean_rows_counts(
    __global const float* source,
    __global const int* indices,
    __global float* output,
    __global float* counts,
    int sourceRows, int innerSize, int outputRows)
{
    int gid = get_global_id(0);
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

__kernel void nonzero_compact(
    __global const float* input,
    __global const int* strides,
    __global float* output,
    __global float* outputCount,
    int length, int rank)
{
    if (get_global_id(0) != 0) return;
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

inline uint importance_hash(uint x)
{
    x ^= x >> 16; x *= 0x7feb352du;
    x ^= x >> 15; x *= 0x846ca68bu;
    return x ^ (x >> 16);
}

__kernel void importance_sampling(
    __global const float* tValues,
    __global const float* weights,
    __global float* output,
    int numRays, int numCoarse, int numFine, uint seed)
{
    int gid = get_global_id(0);
    int total = numRays * numFine;
    if (gid >= total) return;
    int ray = gid / numFine;
    int sample = gid - ray * numFine;
    int base = ray * numCoarse;
    uint bits = importance_hash(seed ^ ((uint)gid * 747796405u + 2891336453u));
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
    if (index == 0) { output[gid] = tValues[base]; return; }
    float denominator = current - previous;
    float t0 = tValues[base + index - 1];
    float t1 = tValues[base + index];
    output[gid] = denominator > 1.0e-10f
        ? t0 + ((u - previous) / denominator) * (t1 - t0)
        : t0;
}

__kernel void resident_nms(
    __global const float* boxes,
    __global const float* scores,
    __global const float* classIds,
    __global float* suppressed,
    __global float* output,
    __global float* outputCount,
    int length, float threshold, int batched)
{
    if (get_global_id(0) != 0) return;
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
        output[count++] = (float)best;
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
            float overlapW = fmin(ix2, jx2) - fmax(ix1, jx1);
            float overlapH = fmin(iy2, jy2) - fmax(iy1, jy1);
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

inline int spiral_append_unique(__global float* list, int count, int candidate, int capacity)
{
    for (int i = 0; i < count; i++) if ((int)list[i] == candidate) return count;
    if (count < capacity) list[count++] = (float)candidate;
    return count;
}

inline int spiral_build_neighbors(
    __global const float* faces, int numFaces, int vertex, __global float* list, int capacity)
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

inline float spiral_angle(
    __global const float* vertices, int center, int reference, int vertex)
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

__kernel void generate_spiral_indices(
    __global const float* vertices,
    __global const float* faces,
    __global float* visited,
    __global float* currentRing,
    __global float* nextRing,
    __global float* output,
    int numVertices, int numFaces, int spiralLength)
{
    if (get_global_id(0) != 0) return;
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

inline float ctc_log_add(float a, float b)
{
    const float negativeSentinel = -3.402823466e+38f;
    if (a <= negativeSentinel) return b;
    if (b <= negativeSentinel) return a;
    float m = fmax(a, b);
    return m + log(exp(a - m) + exp(b - m));
}

__kernel void ctc_loss_forward(
    __global const float* logProbs,
    __global const float* targets,
    __global const float* inputLengths,
    __global const float* targetLengths,
    __global float* workspace,
    __global float* losses,
    int maxTime, int batchSize, int numClasses, int maxTargetLength, int blank)
{
    int n = get_global_id(0);
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
#endif
