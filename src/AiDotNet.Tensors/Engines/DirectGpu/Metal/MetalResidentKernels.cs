// Copyright (c) AiDotNet. All rights reserved.

namespace AiDotNet.Tensors.Engines.DirectGpu.Metal;

/// <summary>Correctness-first resident kernels for operations that cannot use a host fallback.</summary>
internal static class MetalResidentKernels
{
    public const string Source = @"
#include <metal_stdlib>
using namespace metal;

kernel void hyperbolic_linear_forward(
    device const float* input [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device const float* biases [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& batch [[buffer(4)]],
    constant uint& inputFeatures [[buffer(5)]],
    constant uint& outputFeatures [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= batch * outputFeatures) return;
    uint b = gid / outputFeatures;
    uint o = gid % outputFeatures;
    float sum = biases[o];
    for (uint i = 0; i < inputFeatures; ++i)
        sum += input[b * inputFeatures + i] * weights[o * inputFeatures + i];
    output[gid] = sum;
}

kernel void hyperbolic_backward_input(
    device const float* gradOutput [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device float* gradInput [[buffer(2)]],
    constant uint& batch [[buffer(3)]],
    constant uint& inputFeatures [[buffer(4)]],
    constant uint& outputFeatures [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= batch * inputFeatures) return;
    uint b = gid / inputFeatures;
    uint i = gid % inputFeatures;
    float sum = 0.0f;
    for (uint o = 0; o < outputFeatures; ++o)
        sum += gradOutput[b * outputFeatures + o] * weights[o * inputFeatures + i];
    gradInput[gid] = sum;
}

kernel void hyperbolic_backward_weights(
    device const float* gradOutput [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* gradWeights [[buffer(2)]],
    constant uint& batch [[buffer(3)]],
    constant uint& inputFeatures [[buffer(4)]],
    constant uint& outputFeatures [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= outputFeatures * inputFeatures) return;
    uint o = gid / inputFeatures;
    uint i = gid % inputFeatures;
    float sum = 0.0f;
    for (uint b = 0; b < batch; ++b)
        sum += gradOutput[b * outputFeatures + o] * input[b * inputFeatures + i];
    gradWeights[gid] = sum;
}

kernel void hyperbolic_backward_biases(
    device const float* gradOutput [[buffer(0)]],
    device float* gradBiases [[buffer(1)]],
    constant uint& batch [[buffer(2)]],
    constant uint& outputFeatures [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= outputFeatures) return;
    float sum = 0.0f;
    for (uint b = 0; b < batch; ++b)
        sum += gradOutput[b * outputFeatures + gid];
    gradBiases[gid] = sum;
}

kernel void complex_multiply_interleaved(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& pairs [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= pairs) return;
    uint i = gid * 2;
    float ar = a[i], ai = a[i + 1], br = b[i], bi = b[i + 1];
    output[i] = ar * br - ai * bi;
    output[i + 1] = ar * bi + ai * br;
}

kernel void complex_conjugate_interleaved(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& pairs [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= pairs) return;
    uint i = gid * 2;
    float re = input[i], im = input[i + 1];
    output[i] = re;
    output[i + 1] = -im;
}

kernel void complex_magnitude_interleaved(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& pairs [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= pairs) return;
    uint i = gid * 2;
    float re = input[i], im = input[i + 1];
    output[gid] = sqrt(re * re + im * im);
}

kernel void quantum_measurement(
    device const float* realPart [[buffer(0)]],
    device const float* imagPart [[buffer(1)]],
    device float* probabilities [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;
    float re = realPart[gid], im = imagPart[gid];
    probabilities[gid] = re * re + im * im;
}

kernel void normalize_probabilities(
    device float* probabilities [[buffer(0)]],
    constant uint& batch [[buffer(1)]],
    constant uint& stateSize [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= batch) return;
    uint offset = gid * stateSize;
    float sum = 0.0f;
    for (uint s = 0; s < stateSize; ++s) sum += probabilities[offset + s];
    if (sum > 0.0f)
        for (uint s = 0; s < stateSize; ++s) probabilities[offset + s] /= sum;
}

kernel void complex_matvec(
    device const float* matReal [[buffer(0)]],
    device const float* matImag [[buffer(1)]],
    device const float* vecReal [[buffer(2)]],
    device const float* vecImag [[buffer(3)]],
    device float* outReal [[buffer(4)]],
    device float* outImag [[buffer(5)]],
    constant uint& batch [[buffer(6)]],
    constant uint& dim [[buffer(7)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= batch * dim) return;
    uint b = gid / dim, row = gid % dim;
    uint vectorOffset = b * dim, matrixOffset = b * dim * dim + row * dim;
    float sumReal = 0.0f, sumImag = 0.0f;
    for (uint j = 0; j < dim; ++j) {
        float mr = matReal[matrixOffset + j], mi = matImag[matrixOffset + j];
        float vr = vecReal[vectorOffset + j], vi = vecImag[vectorOffset + j];
        sumReal += mr * vr - mi * vi;
        sumImag += mr * vi + mi * vr;
    }
    outReal[gid] = sumReal;
    outImag[gid] = sumImag;
}

kernel void measurement_forward(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;
    float re = input[gid * 2], im = input[gid * 2 + 1];
    output[gid] = re * re + im * im;
}

kernel void quantum_rotation(
    device const float* stateReal [[buffer(0)]],
    device const float* stateImag [[buffer(1)]],
    device const float* angles [[buffer(2)]],
    device float* outReal [[buffer(3)]],
    device float* outImag [[buffer(4)]],
    constant uint& stateSize [[buffer(5)]],
    constant uint& numQubits [[buffer(6)]],
    constant uint& batch [[buffer(7)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= batch * stateSize) return;
    uint b = gid / stateSize;
    float halfAngle = angles[b * numQubits] * 0.5f;
    float cosAngle = cos(halfAngle), sinAngle = sin(halfAngle);
    float re = stateReal[gid], im = stateImag[gid];
    outReal[gid] = cosAngle * re - sinAngle * im;
    outImag[gid] = sinAngle * re + cosAngle * im;
}

kernel void permute_tensor(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const int* inputStrides [[buffer(2)]],
    device const int* outputStrides [[buffer(3)]],
    device const int* permutation [[buffer(4)]],
    constant uint& dimensions [[buffer(5)]],
    constant uint& count [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;
    uint remaining = gid;
    uint inputIndex = 0;
    for (uint d = 0; d < dimensions; ++d) {
        uint coordinate = remaining / uint(outputStrides[d]);
        remaining %= uint(outputStrides[d]);
        inputIndex += coordinate * uint(inputStrides[permutation[d]]);
    }
    output[gid] = input[inputIndex];
}

kernel void copy_2d_strided(
    device const float* source [[buffer(0)]],
    device float* destination [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& sourceColumns [[buffer(3)]],
    constant uint& destinationColumns [[buffer(4)]],
    constant uint& destinationOffset [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    uint count = rows * sourceColumns;
    if (gid >= count) return;
    uint row = gid / sourceColumns, column = gid % sourceColumns;
    destination[row * destinationColumns + destinationOffset + column] = source[gid];
}

kernel void nearest_upsample_2d(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& batchChannels [[buffer(2)]],
    constant uint& height [[buffer(3)]],
    constant uint& width [[buffer(4)]],
    constant uint& scale [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    uint outputHeight = height * scale, outputWidth = width * scale;
    if (gid >= batchChannels * outputHeight * outputWidth) return;
    uint column = gid % outputWidth, row = (gid / outputWidth) % outputHeight;
    uint bc = gid / (outputHeight * outputWidth);
    output[gid] = input[(bc * height + row / scale) * width + column / scale];
}

kernel void nearest_upsample_2d_backward(
    device const float* gradOutput [[buffer(0)]],
    device float* gradInput [[buffer(1)]],
    constant uint& batchChannels [[buffer(2)]],
    constant uint& height [[buffer(3)]],
    constant uint& width [[buffer(4)]],
    constant uint& scale [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= batchChannels * height * width) return;
    uint column = gid % width, row = (gid / width) % height, bc = gid / (height * width);
    uint outputHeight = height * scale, outputWidth = width * scale;
    float sum = 0.0f;
    for (uint y = 0; y < scale; ++y)
        for (uint x = 0; x < scale; ++x)
            sum += gradOutput[(bc * outputHeight + row * scale + y) * outputWidth + column * scale + x];
    gradInput[gid] = sum;
}

kernel void nearest_upsample_3d(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& batch [[buffer(2)]], constant uint& channels [[buffer(3)]],
    constant uint& depth [[buffer(4)]], constant uint& height [[buffer(5)]], constant uint& width [[buffer(6)]],
    constant uint& scaleDepth [[buffer(7)]], constant uint& scaleHeight [[buffer(8)]], constant uint& scaleWidth [[buffer(9)]],
    uint gid [[thread_position_in_grid]])
{
    uint outputDepth = depth * scaleDepth, outputHeight = height * scaleHeight, outputWidth = width * scaleWidth;
    if (gid >= batch * channels * outputDepth * outputHeight * outputWidth) return;
    uint column = gid % outputWidth, value = gid / outputWidth;
    uint row = value % outputHeight; value /= outputHeight;
    uint plane = value % outputDepth; uint bc = value / outputDepth;
    output[gid] = input[((bc * depth + plane / scaleDepth) * height + row / scaleHeight) * width + column / scaleWidth];
}

kernel void nearest_upsample_3d_backward(
    device const float* gradOutput [[buffer(0)]],
    device float* gradInput [[buffer(1)]],
    constant uint& batch [[buffer(2)]], constant uint& channels [[buffer(3)]],
    constant uint& depth [[buffer(4)]], constant uint& height [[buffer(5)]], constant uint& width [[buffer(6)]],
    constant uint& scaleDepth [[buffer(7)]], constant uint& scaleHeight [[buffer(8)]], constant uint& scaleWidth [[buffer(9)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= batch * channels * depth * height * width) return;
    uint column = gid % width, value = gid / width;
    uint row = value % height; value /= height;
    uint plane = value % depth; uint bc = value / depth;
    uint outputDepth = depth * scaleDepth, outputHeight = height * scaleHeight, outputWidth = width * scaleWidth;
    float sum = 0.0f;
    for (uint z = 0; z < scaleDepth; ++z)
        for (uint y = 0; y < scaleHeight; ++y)
            for (uint x = 0; x < scaleWidth; ++x)
                sum += gradOutput[((bc * outputDepth + plane * scaleDepth + z) * outputHeight + row * scaleHeight + y) * outputWidth + column * scaleWidth + x];
    gradInput[gid] = sum;
}

kernel void l2_norm_serial(
    device const float* input [[buffer(0)]], device float* output [[buffer(1)]],
    constant uint& count [[buffer(2)]], uint gid [[thread_position_in_grid]])
{
    if (gid != 0) return;
    float sum = 0.0f;
    for (uint i = 0; i < count; ++i) sum += input[i] * input[i];
    output[0] = sqrt(sum);
}

kernel void clip_by_norm_serial(
    device const float* input [[buffer(0)]], device float* output [[buffer(1)]],
    constant uint& count [[buffer(2)]], constant float& maxNorm [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid != 0) return;
    float sum = 0.0f;
    for (uint i = 0; i < count; ++i) sum += input[i] * input[i];
    float norm = sqrt(sum);
    float scale = norm > maxNorm ? maxNorm / norm : 1.0f;
    for (uint i = 0; i < count; ++i) output[i] = input[i] * scale;
}

kernel void scatter_add_deterministic(
    device const float* source [[buffer(0)]], device const int* indices [[buffer(1)]], device float* destination [[buffer(2)]],
    constant uint& sourceSize [[buffer(3)]], constant uint& destinationSize [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= destinationSize) return;
    float value = destination[gid];
    for (uint i = 0; i < sourceSize; ++i)
        if (indices[i] == int(gid)) value += source[i];
    destination[gid] = value;
}

kernel void indexed_gather(
    device const float* source [[buffer(0)]], device const int* indices [[buffer(1)]], device float* output [[buffer(2)]],
    constant uint& entries [[buffer(3)]], constant uint& featureSize [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= entries * featureSize) return;
    uint entry = gid / featureSize, feature = gid % featureSize;
    output[gid] = source[uint(indices[entry]) * featureSize + feature];
}

inline ushort resident_float_to_half(float value)
{
    uint bits = as_type<uint>(value);
    uint sign = (bits >> 16) & 0x8000u;
    int exponent = int((bits >> 23) & 0xffu) - 112;
    uint mantissa = bits & 0x7fffffu;
    if (exponent <= 0) {
        if (exponent < -10) return ushort(sign);
        mantissa |= 0x800000u;
        mantissa >>= uint(14 - exponent);
        return ushort(sign | mantissa);
    }
    if (exponent >= 31) {
        if ((bits & 0x7fffffffu) > 0x7f800000u) return ushort(0x7fffu);
        return ushort(sign | 0x7c00u);
    }
    return ushort(sign | (uint(exponent) << 10) | (mantissa >> 13));
}

inline float resident_half_to_float(ushort value)
{
    uint sign = (uint(value) & 0x8000u) << 16;
    int exponent = int((uint(value) >> 10) & 0x1fu);
    uint mantissa = uint(value) & 0x3ffu;
    if (exponent == 0) {
        if (mantissa == 0u) return as_type<float>(sign);
        while ((mantissa & 0x400u) == 0u) { mantissa <<= 1; --exponent; }
        ++exponent; mantissa &= 0x3ffu;
    } else if (exponent == 31) {
        return as_type<float>(sign | (mantissa == 0u ? 0x7f800000u : 0x7fc00000u));
    }
    exponent += 112;
    return as_type<float>(sign | (uint(exponent) << 23) | (mantissa << 13));
}

kernel void convert_to_fp16_packed(
    device const float* input [[buffer(0)]], device uint* output [[buffer(1)]],
    constant uint& count [[buffer(2)]], uint gid [[thread_position_in_grid]])
{
    uint packedCount = (count + 1u) / 2u;
    if (gid >= packedCount) return;
    uint i = gid * 2u;
    uint low = uint(resident_float_to_half(input[i]));
    uint high = i + 1u < count ? uint(resident_float_to_half(input[i + 1u])) : 0u;
    output[gid] = low | (high << 16);
}

kernel void convert_from_fp16_packed(
    device const uint* input [[buffer(0)]], device float* output [[buffer(1)]],
    constant uint& count [[buffer(2)]], uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;
    uint packed = input[gid / 2u];
    ushort value = ushort((gid & 1u) == 0u ? packed & 0xffffu : packed >> 16);
    output[gid] = resident_half_to_float(value);
}

kernel void capsule_squash(
    device const float* input [[buffer(0)]], device float* output [[buffer(1)]],
    constant uint& capsules [[buffer(2)]], constant uint& dimension [[buffer(3)]], constant float& epsilon [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= capsules * dimension) return;
    uint capsule = gid / dimension, offset = capsule * dimension;
    float normSquared = 0.0f;
    for (uint d = 0; d < dimension; ++d) { float value = input[offset + d]; normSquared += value * value; }
    float norm = sqrt(normSquared + epsilon);
    float scale = normSquared / ((1.0f + normSquared) * norm);
    output[gid] = input[gid] * scale;
}

kernel void capsule_squash_backward(
    device const float* gradOutput [[buffer(0)]], device const float* input [[buffer(1)]], device float* gradInput [[buffer(2)]],
    constant uint& capsules [[buffer(3)]], constant uint& dimension [[buffer(4)]], constant float& epsilon [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= capsules * dimension) return;
    uint capsule = gid / dimension, offset = capsule * dimension;
    float normSquared = 0.0f;
    for (uint d = 0; d < dimension; ++d) { float value = input[offset + d]; normSquared += value * value; }
    float norm = sqrt(normSquared + epsilon);
    float scale = normSquared / ((1.0f + normSquared) * norm);
    gradInput[gid] = gradOutput[gid] * scale;
}

kernel void capsule_predictions(
    device const float* input [[buffer(0)]], device const float* weights [[buffer(1)]], device float* output [[buffer(2)]],
    constant uint& batch [[buffer(3)]], constant uint& inputCapsules [[buffer(4)]], constant uint& inputDimension [[buffer(5)]],
    constant uint& outputCapsules [[buffer(6)]], constant uint& outputDimension [[buffer(7)]],
    uint gid [[thread_position_in_grid]])
{
    uint count = batch * inputCapsules * outputCapsules * outputDimension;
    if (gid >= count) return;
    uint d = gid % outputDimension, value = gid / outputDimension;
    uint outputCapsule = value % outputCapsules; value /= outputCapsules;
    uint inputCapsule = value % inputCapsules; uint b = value / inputCapsules;
    float sum = 0.0f;
    for (uint k = 0; k < inputDimension; ++k) {
        uint inputIndex = (b * inputCapsules + inputCapsule) * inputDimension + k;
        uint weightIndex = ((inputCapsule * outputCapsules + outputCapsule) * inputDimension + k) * outputDimension + d;
        sum += input[inputIndex] * weights[weightIndex];
    }
    output[gid] = sum;
}

kernel void capsule_weighted_sum(
    device const float* coupling [[buffer(0)]], device const float* predictions [[buffer(1)]], device float* output [[buffer(2)]],
    constant uint& batch [[buffer(3)]], constant uint& inputCapsules [[buffer(4)]],
    constant uint& outputCapsules [[buffer(5)]], constant uint& dimension [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= batch * outputCapsules * dimension) return;
    uint d = gid % dimension, value = gid / dimension;
    uint outputCapsule = value % outputCapsules, b = value / outputCapsules;
    float sum = 0.0f;
    for (uint inputCapsule = 0; inputCapsule < inputCapsules; ++inputCapsule) {
        uint couplingIndex = (b * inputCapsules + inputCapsule) * outputCapsules + outputCapsule;
        uint predictionIndex = ((b * inputCapsules + inputCapsule) * outputCapsules + outputCapsule) * dimension + d;
        sum += coupling[couplingIndex] * predictions[predictionIndex];
    }
    output[gid] = sum;
}

kernel void capsule_agreement(
    device const float* predictions [[buffer(0)]], device const float* output [[buffer(1)]], device float* agreement [[buffer(2)]],
    constant uint& batch [[buffer(3)]], constant uint& inputCapsules [[buffer(4)]],
    constant uint& outputCapsules [[buffer(5)]], constant uint& dimension [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= batch * inputCapsules * outputCapsules) return;
    uint outputCapsule = gid % outputCapsules, value = gid / outputCapsules;
    uint inputCapsule = value % inputCapsules, b = value / inputCapsules;
    float sum = 0.0f;
    for (uint d = 0; d < dimension; ++d) {
        uint predictionIndex = ((b * inputCapsules + inputCapsule) * outputCapsules + outputCapsule) * dimension + d;
        uint outputIndex = (b * outputCapsules + outputCapsule) * dimension + d;
        sum += predictions[predictionIndex] * output[outputIndex];
    }
    agreement[gid] = sum;
}

kernel void tile_batch(
    device const float* input [[buffer(0)]], device float* output [[buffer(1)]],
    constant uint& repeats [[buffer(2)]], constant uint& innerSize [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= repeats * innerSize) return;
    output[gid] = input[gid % innerSize];
}

kernel void enforce_2x4(
    device const float* dense [[buffer(0)]], device float* values [[buffer(1)]], device uchar* indices [[buffer(2)]],
    constant uint& rows [[buffer(3)]], constant uint& columns [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    uint groups = rows * (columns / 4u);
    if (gid >= groups) return;
    uint row = gid / (columns / 4u), group = gid % (columns / 4u), base = row * columns + group * 4u;
    uint order[4] = { 0u, 1u, 2u, 3u };
    float magnitudes[4] = { abs(dense[base]), abs(dense[base + 1u]), abs(dense[base + 2u]), abs(dense[base + 3u]) };
    for (uint i = 0; i < 3u; ++i) for (uint j = i + 1u; j < 4u; ++j) {
        if (magnitudes[j] > magnitudes[i]) {
            float magnitude = magnitudes[i]; magnitudes[i] = magnitudes[j]; magnitudes[j] = magnitude;
            uint index = order[i]; order[i] = order[j]; order[j] = index;
        }
    }
    values[gid * 2u] = dense[base + order[0]];
    values[gid * 2u + 1u] = dense[base + order[1]];
    indices[gid] = uchar((order[0] & 3u) | ((order[1] & 3u) << 2u));
}

kernel void decompress_2x4(
    device const float* values [[buffer(0)]], device const uchar* indices [[buffer(1)]], device float* dense [[buffer(2)]],
    constant uint& rows [[buffer(3)]], constant uint& columns [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= rows * columns) return;
    uint group = gid / 4u, local = gid & 3u;
    uchar packed = indices[group]; uint first = uint(packed) & 3u, second = (uint(packed) >> 2u) & 3u;
    dense[gid] = local == first ? values[group * 2u] : (local == second ? values[group * 2u + 1u] : 0.0f);
}

kernel void scatter_add_edges_deterministic(
    device const float* input [[buffer(0)]], device const int* sources [[buffer(1)]], device const int* targets [[buffer(2)]],
    device const float* edgeValues [[buffer(3)]], device float* output [[buffer(4)]],
    constant uint& nodes [[buffer(5)]], constant uint& edges [[buffer(6)]], constant uint& features [[buffer(7)]],
    constant uint& weighted [[buffer(8)]], uint gid [[thread_position_in_grid]])
{
    if (gid >= nodes * features) return;
    uint node = gid / features, feature = gid % features;
    float sum = output[gid];
    for (uint edge = 0; edge < edges; ++edge) {
        int source = sources[edge], target = targets[edge];
        if (target == int(node) && source >= 0 && source < int(nodes))
            sum += (weighted != 0u ? edgeValues[edge] : 1.0f) * input[uint(source) * features + feature];
    }
    output[gid] = sum;
}

kernel void csr_segmented(
    device const int* columns [[buffer(0)]], device const int* rows [[buffer(1)]],
    device const float* input [[buffer(2)]], device float* output [[buffer(3)]],
    constant uint& rowCount [[buffer(4)]], constant uint& inputRows [[buffer(5)]], constant uint& features [[buffer(6)]],
    constant uint& operation [[buffer(7)]], constant uint& epsilonBits [[buffer(8)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= rowCount * features) return;
    uint row = gid / features, feature = gid % features;
    int start = rows[row], end = rows[row + 1u];
    if (start >= end) { output[gid] = 0.0f; return; }
    if (operation < 2u) {
        float result = operation == 0u ? -3.402823466e+38f : 3.402823466e+38f;
        for (int p = start; p < end; ++p) {
            int column = columns[p];
            if (column >= 0 && column < int(inputRows))
                result = operation == 0u ? max(result, input[uint(column) * features + feature])
                                         : min(result, input[uint(column) * features + feature]);
        }
        output[gid] = result; return;
    }
    float mean = 0.0f;
    for (int p = start; p < end; ++p) { int column = columns[p]; if (column >= 0 && column < int(inputRows)) mean += input[uint(column) * features + feature]; }
    mean /= float(end - start);
    float variance = 0.0f;
    for (int p = start; p < end; ++p) { int column = columns[p]; if (column >= 0 && column < int(inputRows)) { float d = input[uint(column) * features + feature] - mean; variance += d * d; } }
    output[gid] = sqrt(variance / float(end - start) + as_type<float>(epsilonBits));
}

kernel void comparison_binary(
    device const float* a [[buffer(0)]], device const float* b [[buffer(1)]], device float* output [[buffer(2)]],
    constant uint& count [[buffer(3)]], constant uint& operation [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;
    if (operation == 0u) { output[gid] = a[gid] > b[gid] ? 1.0f : 0.0f; return; }
    if (operation == 1u) { output[gid] = a[gid] < b[gid] ? 1.0f : 0.0f; return; }
    uint ab = as_type<uint>(a[gid]);
    uint bb = as_type<uint>(b[gid]);
    uint aa = ab & 0x7FFFFFFFu;
    uint ba = bb & 0x7FFFFFFFu;
    bool equal = aa <= 0x7F800000u && ba <= 0x7F800000u
        && (ab == bb || ((aa | ba) == 0u));
    output[gid] = equal ? 1.0f : 0.0f;
}

kernel void comparison_where(
    device const float* condition [[buffer(0)]], device const float* a [[buffer(1)]], device const float* b [[buffer(2)]], device float* output [[buffer(3)]],
    constant uint& count [[buffer(4)]], uint gid [[thread_position_in_grid]])
{
    if (gid < count) output[gid] = condition[gid] != 0.0f ? a[gid] : b[gid];
}

kernel void comparison_not_equal_scalar(
    device const float* input [[buffer(0)]], device float* output [[buffer(1)]],
    constant uint& count [[buffer(2)]], constant uint& scalarBits [[buffer(3)]], uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;
    uint ab = as_type<uint>(input[gid]);
    uint aa = ab & 0x7FFFFFFFu;
    uint ba = scalarBits & 0x7FFFFFFFu;
    bool equal = aa <= 0x7F800000u && ba <= 0x7F800000u
        && (ab == scalarBits || ((aa | ba) == 0u));
    output[gid] = equal ? 0.0f : 1.0f;
}

kernel void variance_axis(
    device const float* input [[buffer(0)]], device float* mean [[buffer(1)]], device float* variance [[buffer(2)]],
    constant uint& outerSize [[buffer(3)]], constant uint& reduceSize [[buffer(4)]], uint gid [[thread_position_in_grid]])
{
    if (gid >= outerSize) return;
    uint offset = gid * reduceSize; float sum = 0.0f;
    for (uint i = 0; i < reduceSize; ++i) sum += input[offset + i];
    float rowMean = sum / float(reduceSize); mean[gid] = rowMean;
    float sumSquared = 0.0f;
    for (uint i = 0; i < reduceSize; ++i) { float d = input[offset + i] - rowMean; sumSquared += d * d; }
    variance[gid] = sumSquared / float(reduceSize);
}

kernel void arg_extrema_axis(
    device const float* input [[buffer(0)]], device float* indices [[buffer(1)]],
    constant uint& outerSize [[buffer(2)]], constant uint& reduceSize [[buffer(3)]], constant uint& maximum [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= outerSize) return;
    uint offset = gid * reduceSize, bestIndex = 0u;
    float best = maximum != 0u ? -3.402823466e+38f : 3.402823466e+38f;
    for (uint i = 0; i < reduceSize; ++i) {
        float value = input[offset + i];
        if ((maximum != 0u && value > best) || (maximum == 0u && value < best)) { best = value; bestIndex = i; }
    }
    indices[gid] = float(bestIndex);
}

kernel void topk_axis_serial(
    device const float* input [[buffer(0)]], device float* values [[buffer(1)]], device float* indices [[buffer(2)]],
    constant uint& outerSize [[buffer(3)]], constant uint& reduceSize [[buffer(4)]], constant uint& k [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= outerSize) return;
    uint inputOffset = gid * reduceSize, outputOffset = gid * k;
    for (uint rank = 0; rank < k; ++rank) {
        float best = -3.402823466e+38f; uint bestIndex = 0u;
        for (uint column = 0; column < reduceSize; ++column) {
            bool used = false;
            for (uint previous = 0; previous < rank; ++previous)
                if (uint(indices[outputOffset + previous]) == column) { used = true; break; }
            float value = input[inputOffset + column];
            if (!used && value > best) { best = value; bestIndex = column; }
        }
        values[outputOffset + rank] = best; indices[outputOffset + rank] = float(bestIndex);
    }
}

kernel void split_fft_strided_serial(
    device const float* inputReal [[buffer(0)]], device const float* inputImag [[buffer(1)]],
    device float* outputReal [[buffer(2)]], device float* outputImag [[buffer(3)]],
    constant uint& sequences [[buffer(4)]], constant uint& length [[buffer(5)]],
    constant uint& baseStride [[buffer(6)]], constant uint& elementStride [[buffer(7)]],
    constant uint& inverse [[buffer(8)]], uint gid [[thread_position_in_grid]])
{
    if (gid >= sequences) return;
    uint base = gid * baseStride;
    for (uint i = 0; i < length; ++i) {
        uint index = base + i * elementStride;
        float re = inputReal[index], im = inputImag[index];
        outputReal[index] = re; outputImag[index] = im;
    }
    uint bits = 0u; for (uint value = length; value > 1u; value >>= 1u) ++bits;
    for (uint i = 0; i < length; ++i) {
        uint reversed = 0u, value = i;
        for (uint bit = 0; bit < bits; ++bit) { reversed = (reversed << 1u) | (value & 1u); value >>= 1u; }
        if (reversed > i) {
            uint a = base + i * elementStride, b = base + reversed * elementStride;
            float re = outputReal[a], im = outputImag[a];
            outputReal[a] = outputReal[b]; outputImag[a] = outputImag[b];
            outputReal[b] = re; outputImag[b] = im;
        }
    }
    float sign = inverse != 0u ? 1.0f : -1.0f;
    for (uint span = 2u; span <= length; span <<= 1u) {
        float angle = sign * 2.0f * 3.14159265358979323846f / float(span);
        float stepReal = cos(angle), stepImag = sin(angle);
        for (uint start = 0u; start < length; start += span) {
            float currentReal = 1.0f, currentImag = 0.0f;
            for (uint j = 0u; j < span / 2u; ++j) {
                uint even = base + (start + j) * elementStride;
                uint odd = base + (start + j + span / 2u) * elementStride;
                float oddReal = outputReal[odd], oddImag = outputImag[odd];
                float productReal = currentReal * oddReal - currentImag * oddImag;
                float productImag = currentReal * oddImag + currentImag * oddReal;
                float evenReal = outputReal[even], evenImag = outputImag[even];
                outputReal[odd] = evenReal - productReal; outputImag[odd] = evenImag - productImag;
                outputReal[even] = evenReal + productReal; outputImag[even] = evenImag + productImag;
                float nextReal = currentReal * stepReal - currentImag * stepImag;
                currentImag = currentReal * stepImag + currentImag * stepReal;
                currentReal = nextReal;
            }
        }
    }
    if (inverse != 0u)
        for (uint i = 0; i < length; ++i) { uint index = base + i * elementStride; outputReal[index] /= float(length); outputImag[index] /= float(length); }
}

kernel void irfft_reconstruct(
    device const float* halfReal [[buffer(0)]], device const float* halfImag [[buffer(1)]],
    device float* fullReal [[buffer(2)]], device float* fullImag [[buffer(3)]],
    constant uint& length [[buffer(4)]], uint gid [[thread_position_in_grid]])
{
    if (gid >= length) return;
    uint halfLength = length / 2u + 1u;
    if (gid < halfLength) { fullReal[gid] = halfReal[gid]; fullImag[gid] = halfImag[gid]; }
    else { uint mirror = length - gid; fullReal[gid] = halfReal[mirror]; fullImag[gid] = -halfImag[mirror]; }
}

kernel void power_to_db(
    device const float* power [[buffer(0)]], device float* output [[buffer(1)]],
    constant uint& count [[buffer(2)]], constant uint& referenceBits [[buffer(3)]], constant uint& minimumBits [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;
    float value = 10.0f * log10(max(power[gid], 1.0e-10f) / as_type<float>(referenceBits));
    output[gid] = max(value, as_type<float>(minimumBits));
}

kernel void db_to_power(
    device const float* db [[buffer(0)]], device float* output [[buffer(1)]],
    constant uint& count [[buffer(2)]], constant uint& referenceBits [[buffer(3)]], uint gid [[thread_position_in_grid]])
{
    if (gid < count) output[gid] = as_type<float>(referenceBits) * pow(10.0f, db[gid] / 10.0f);
}

kernel void rbf_forward(
    device const float* input [[buffer(0)]], device const float* centers [[buffer(1)]],
    device const float* epsilons [[buffer(2)]], device float* output [[buffer(3)]],
    constant uint& batch [[buffer(4)]], constant uint& centerCount [[buffer(5)]], constant uint& dimension [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= batch * centerCount) return;
    uint b = gid / centerCount, center = gid % centerCount; float distance = 0.0f;
    for (uint d = 0; d < dimension; ++d) { float value = input[b * dimension + d] - centers[center * dimension + d]; distance += value * value; }
    output[gid] = exp(-epsilons[center] * distance);
}

kernel void stdp_update(
    device float* weights [[buffer(0)]], device const float* preTrace [[buffer(1)]], device const float* postTrace [[buffer(2)]],
    device const float* preSpike [[buffer(3)]], device const float* postSpike [[buffer(4)]],
    constant uint& preCount [[buffer(5)]], constant uint& postCount [[buffer(6)]],
    constant uint& ltpBits [[buffer(7)]], constant uint& ltdBits [[buffer(8)]], constant uint& homeostasisBits [[buffer(9)]],
    constant uint& minimumBits [[buffer(10)]], constant uint& maximumBits [[buffer(11)]], uint gid [[thread_position_in_grid]])
{
    if (gid >= preCount * postCount) return;
    uint pre = gid / postCount, post = gid % postCount; float value = weights[gid];
    if (preSpike[pre] > 0.0f) value += as_type<float>(ltpBits) * postTrace[post];
    if (postSpike[post] > 0.0f) value -= as_type<float>(ltdBits) * preTrace[pre];
    value -= as_type<float>(homeostasisBits) * value;
    weights[gid] = clamp(value, as_type<float>(minimumBits), as_type<float>(maximumBits));
}

kernel void update_traces(
    device float* traces [[buffer(0)]], device float* spikes [[buffer(1)]], device const float* input [[buffer(2)]],
    constant uint& count [[buffer(3)]], constant uint& decayBits [[buffer(4)]], constant uint& thresholdBits [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;
    float threshold = as_type<float>(thresholdBits);
    float trace = traces[gid] * as_type<float>(decayBits) + input[gid];
    if (trace >= threshold) { spikes[gid] = 1.0f; trace -= threshold; } else spikes[gid] = 0.0f;
    traces[gid] = trace;
}

kernel void optimizer_update(
    device float* parameter [[buffer(0)]], device const float* gradient [[buffer(1)]],
    device float* state1 [[buffer(2)]], device float* state2 [[buffer(3)]], device float* state3 [[buffer(4)]],
    constant uint& count [[buffer(5)]], constant uint& operation [[buffer(6)]], constant uint& step [[buffer(7)]],
    constant uint& learningRateBits [[buffer(8)]], constant uint& aBits [[buffer(9)]], constant uint& bBits [[buffer(10)]],
    constant uint& cBits [[buffer(11)]], constant uint& dBits [[buffer(12)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;
    float learningRate = as_type<float>(learningRateBits), a = as_type<float>(aBits), b = as_type<float>(bBits);
    float c = as_type<float>(cBits), d = as_type<float>(dBits), value = parameter[gid], g = gradient[gid];
    if (operation == 0u) { g += a * value; parameter[gid] = value - learningRate * g; return; }
    if (operation == 1u) { g += b * value; float velocity = a * state1[gid] + g; state1[gid] = velocity; parameter[gid] = value - learningRate * velocity; return; }
    if (operation == 2u || operation == 3u) {
        if (operation == 2u) g += d * value;
        float m = a * state1[gid] + (1.0f - a) * g;
        float v = b * state2[gid] + (1.0f - b) * g * g;
        state1[gid] = m; state2[gid] = v;
        float mHat = m / (1.0f - pow(a, float(step))), vHat = v / (1.0f - pow(b, float(step)));
        float update = learningRate * mHat / (sqrt(vHat) + c);
        parameter[gid] = operation == 3u ? value * (1.0f - learningRate * d) - update : value - update;
        return;
    }
    if (operation == 4u) { g += c * value; float average = a * state1[gid] + (1.0f - a) * g * g; state1[gid] = average; parameter[gid] = value - learningRate * g / (sqrt(average) + b); return; }
    if (operation == 5u) { g += b * value; float accumulated = state1[gid] + g * g; state1[gid] = accumulated; parameter[gid] = value - learningRate * g / (sqrt(accumulated) + a); return; }
    if (operation == 6u) { g += b * value; float previous = state1[gid], velocity = a * previous - learningRate * g; state1[gid] = velocity; parameter[gid] = value - a * previous + (1.0f + a) * velocity; return; }
    if (operation == 7u) {
        g += c * value; float accumulatedGradient = a * state1[gid] + (1.0f - a) * g * g; state1[gid] = accumulatedGradient;
        float update = sqrt(state2[gid] + b) / sqrt(accumulatedGradient + b) * g;
        state2[gid] = a * state2[gid] + (1.0f - a) * update * update; parameter[gid] = value - update; return;
    }
    if (operation == 8u) {
        g += d * value; float m = a * state1[gid] + (1.0f - a) * g, v = b * state2[gid] + (1.0f - b) * g * g;
        state1[gid] = m; state2[gid] = v; float maximum = max(state3[gid], v); state3[gid] = maximum;
        float mHat = m / (1.0f - pow(a, float(step))), maxHat = maximum / (1.0f - pow(b, float(step)));
        parameter[gid] = value - learningRate * mHat / (sqrt(maxHat) + c); return;
    }
    if (operation == 9u) {
        g += d * value; float m = a * state1[gid] + (1.0f - a) * g, u = max(b * state2[gid], abs(g));
        state1[gid] = m; state2[gid] = u; float mHat = m / (1.0f - pow(a, float(step)));
        parameter[gid] = value - learningRate * mHat / (u + c); return;
    }
    if (operation == 10u) {
        float update = a * state1[gid] + (1.0f - a) * g;
        state1[gid] = b * state1[gid] + (1.0f - b) * g;
        parameter[gid] = value * (1.0f - learningRate * c) - learningRate * sign(update); return;
    }
    if (operation == 11u) {
        g += d * value; float m = a * state1[gid] + (1.0f - a) * g, v = b * state2[gid] + (1.0f - b) * g * g;
        state1[gid] = m; state2[gid] = v;
        float correction1 = 1.0f - pow(a, float(step)), correctionNext = 1.0f - pow(a, float(step + 1u));
        float nesterov = a * (m / correction1) + (1.0f - a) * g / correctionNext;
        parameter[gid] = value - learningRate * nesterov / (sqrt(v / (1.0f - pow(b, float(step)))) + c); return;
    }
    if (operation == 12u) {
        float previous = state2[gid], current = previous + g * g; state2[gid] = current;
        float sigma = (sqrt(current) - sqrt(previous)) / learningRate; float z = state1[gid] + g - sigma * value; state1[gid] = z;
        if (abs(z) <= a) parameter[gid] = 0.0f;
        else parameter[gid] = -(z - sign(z) * a) / ((c + sqrt(current)) / learningRate + b);
        return;
    }
    float updated = value - learningRate * g, threshold = learningRate * a;
    parameter[gid] = updated > threshold ? updated - threshold : (updated < -threshold ? updated + threshold : 0.0f);
}

kernel void lars_update_serial(
    device float* parameter [[buffer(0)]], device const float* gradient [[buffer(1)]], device float* velocity [[buffer(2)]],
    constant uint& count [[buffer(3)]], constant uint& learningRateBits [[buffer(4)]], constant uint& momentumBits [[buffer(5)]],
    constant uint& decayBits [[buffer(6)]], constant uint& trustBits [[buffer(7)]], uint gid [[thread_position_in_grid]])
{
    if (gid != 0u) return;
    float parameterNorm = 0.0f, gradientNorm = 0.0f;
    for (uint i = 0; i < count; ++i) { parameterNorm += parameter[i] * parameter[i]; gradientNorm += gradient[i] * gradient[i]; }
    parameterNorm = sqrt(parameterNorm); gradientNorm = sqrt(gradientNorm);
    float learningRate = as_type<float>(learningRateBits), decay = as_type<float>(decayBits);
    float localRate = learningRate;
    if (parameterNorm > 0.0f && gradientNorm > 0.0f)
        localRate = learningRate * as_type<float>(trustBits) * parameterNorm / (gradientNorm + decay * parameterNorm);
    for (uint i = 0; i < count; ++i) {
        float state = as_type<float>(momentumBits) * velocity[i] + localRate * (gradient[i] + decay * parameter[i]);
        velocity[i] = state; parameter[i] -= state;
    }
}

kernel void lamb_update_serial(
    device float* parameter [[buffer(0)]], device const float* gradient [[buffer(1)]], device float* m [[buffer(2)]], device float* v [[buffer(3)]],
    constant uint& count [[buffer(4)]], constant uint& step [[buffer(5)]], constant uint& learningRateBits [[buffer(6)]],
    constant uint& beta1Bits [[buffer(7)]], constant uint& beta2Bits [[buffer(8)]], constant uint& epsilonBits [[buffer(9)]],
    constant uint& decayBits [[buffer(10)]], uint gid [[thread_position_in_grid]])
{
    if (gid != 0u) return;
    float learningRate = as_type<float>(learningRateBits), beta1 = as_type<float>(beta1Bits), beta2 = as_type<float>(beta2Bits);
    float epsilon = as_type<float>(epsilonBits), decay = as_type<float>(decayBits);
    float correction1 = 1.0f - pow(beta1, float(step)), correction2 = 1.0f - pow(beta2, float(step));
    for (uint i = 0; i < count; ++i) { float g = gradient[i]; m[i] = beta1 * m[i] + (1.0f - beta1) * g; v[i] = beta2 * v[i] + (1.0f - beta2) * g * g; }
    float parameterNorm = 0.0f, updateNorm = 0.0f;
    for (uint i = 0; i < count; ++i) { float update = (m[i] / correction1) / (sqrt(v[i] / correction2) + epsilon) + decay * parameter[i]; parameterNorm += parameter[i] * parameter[i]; updateNorm += update * update; }
    parameterNorm = sqrt(parameterNorm); updateNorm = sqrt(updateNorm);
    float trust = parameterNorm > 0.0f && updateNorm > 0.0f ? parameterNorm / updateNorm : 1.0f;
    for (uint i = 0; i < count; ++i) { float update = (m[i] / correction1) / (sqrt(v[i] / correction2) + epsilon) + decay * parameter[i]; parameter[i] -= learningRate * trust * update; }
}

kernel void sparse_optimizer_serial(
    device float* parameter [[buffer(0)]], device float* state1 [[buffer(1)]], device float* state2 [[buffer(2)]], device float* state3 [[buffer(3)]],
    device const float* indices [[buffer(4)]], device const float* values [[buffer(5)]],
    constant uint& size [[buffer(6)]], constant uint& nonzeroCount [[buffer(7)]], constant uint& operation [[buffer(8)]], constant uint& step [[buffer(9)]],
    constant uint& learningRateBits [[buffer(10)]], constant uint& aBits [[buffer(11)]], constant uint& bBits [[buffer(12)]],
    constant uint& cBits [[buffer(13)]], constant uint& dBits [[buffer(14)]], uint gid [[thread_position_in_grid]])
{
    if (gid != 0u) return;
    float learningRate = as_type<float>(learningRateBits), a = as_type<float>(aBits), b = as_type<float>(bBits);
    float c = as_type<float>(cBits), d = as_type<float>(dBits);
    float correction1 = 1.0f - pow(a, float(step)), correction2 = 1.0f - pow(b, float(step));
    for (uint k = 0u; k < nonzeroCount; ++k) {
        int bitIndex = as_type<int>(indices[k]);
        int numericIndex = int(indices[k]);
        int decoded = bitIndex >= 0 && uint(bitIndex) < size ? bitIndex
                    : (numericIndex >= 0 && uint(numericIndex) < size && indices[k] == float(numericIndex) ? numericIndex : -1);
        if (decoded < 0) continue;
        uint i = uint(decoded); float g = values[k], value = parameter[i];
        if (operation == 0u) { if (a > 0.0f) g += a * value; parameter[i] = value - learningRate * g; }
        else if (operation == 1u) { if (b > 0.0f) g += b * value; float velocity = a * state1[i] + g; state1[i] = velocity; parameter[i] = value - learningRate * velocity; }
        else if (operation == 2u || operation == 3u) {
            float original = value; if (operation == 3u && d > 0.0f) original -= learningRate * d * original;
            float m = a * state1[i] + (1.0f - a) * g, v = b * state2[i] + (1.0f - b) * g * g; state1[i] = m; state2[i] = v;
            float update = learningRate * (m / correction1) / (sqrt(v / correction2) + c);
            if (operation == 2u && d > 0.0f) update += learningRate * d * value;
            parameter[i] = original - update;
        }
        else if (operation == 4u) { if (c > 0.0f) g += c * value; float average = a * state1[i] + (1.0f - a) * g * g; state1[i] = average; parameter[i] = value - learningRate * g / (sqrt(average) + b); }
        else if (operation == 5u) { if (b > 0.0f) g += b * value; float accumulated = state1[i] + g * g; state1[i] = accumulated; parameter[i] = value - learningRate * g / (sqrt(accumulated) + a); }
        else if (operation == 6u) { if (b > 0.0f) g += b * value; float old = state1[i], velocity = a * old + g; state1[i] = velocity; parameter[i] = value - learningRate * ((1.0f + a) * velocity - a * old); }
        else if (operation == 7u) { if (c > 0.0f) g += c * value; float accumulated = a * state1[i] + (1.0f - a) * g * g; state1[i] = accumulated; float update = sqrt(state2[i] + b) / sqrt(accumulated + b) * g; state2[i] = a * state2[i] + (1.0f - a) * update * update; parameter[i] = value - update; }
        else if (operation == 8u) { if (d > 0.0f) g += d * value; float m = a * state1[i] + (1.0f - a) * g, v = b * state2[i] + (1.0f - b) * g * g; state1[i] = m; state2[i] = v; float maximum = max(state3[i], v); state3[i] = maximum; parameter[i] = value - learningRate * (m / correction1) / (sqrt(maximum / correction2) + c); }
        else if (operation == 9u) { if (d > 0.0f) g += d * value; float m = a * state1[i] + (1.0f - a) * g, u = max(b * state2[i], abs(g)); state1[i] = m; state2[i] = u; parameter[i] = value - (learningRate / correction1) * m / (u + c); }
        else if (operation == 10u) { float combined = a * state1[i] + (1.0f - a) * g, update = sign(combined); if (c > 0.0f) update += c * value; parameter[i] = value - learningRate * update; state1[i] = b * state1[i] + (1.0f - b) * g; }
        else if (operation == 11u) { if (d > 0.0f) g += d * value; float m = a * state1[i] + (1.0f - a) * g, v = b * state2[i] + (1.0f - b) * g * g; state1[i] = m; state2[i] = v; float mHat = (a * m + (1.0f - a) * g) / correction1; parameter[i] = value - learningRate * mHat / (sqrt(v / correction2) + c); }
        else if (operation == 12u) { float old = state2[i], current = old + g * g; state2[i] = current; float sigma = (sqrt(current) - sqrt(old)) / learningRate; float z = state1[i] + g - sigma * value; state1[i] = z; parameter[i] = abs(z) <= a ? 0.0f : ((z > 0.0f ? 1.0f : -1.0f) * a - z) / ((c + sqrt(current)) / learningRate + b); }
        else { float updated = value - learningRate * g, threshold = learningRate * a; parameter[i] = updated > threshold ? updated - threshold : (updated < -threshold ? updated + threshold : 0.0f); }
    }
}

kernel void batch_norm_forward_serial_channels(
    device const float* input [[buffer(0)]], device float* output [[buffer(1)]],
    device const float* gamma [[buffer(2)]], device const float* beta [[buffer(3)]],
    device float* runningMean [[buffer(4)]], device float* runningVariance [[buffer(5)]],
    device float* savedMean [[buffer(6)]], device float* savedInverse [[buffer(7)]],
    constant uint& batch [[buffer(8)]], constant uint& channels [[buffer(9)]], constant uint& spatial [[buffer(10)]],
    constant uint& epsilonBits [[buffer(11)]], constant uint& momentumBits [[buffer(12)]], constant uint& training [[buffer(13)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= channels) return;
    float mean, inverse;
    if (training != 0u) {
        float sum = 0.0f;
        for (uint b = 0; b < batch; ++b) for (uint s = 0; s < spatial; ++s) sum += input[(b * channels + gid) * spatial + s];
        mean = sum / float(batch * spatial); float varianceSum = 0.0f;
        for (uint b = 0; b < batch; ++b) for (uint s = 0; s < spatial; ++s) { float d = input[(b * channels + gid) * spatial + s] - mean; varianceSum += d * d; }
        float variance = varianceSum / float(batch * spatial); inverse = 1.0f / sqrt(variance + as_type<float>(epsilonBits));
        float momentum = as_type<float>(momentumBits);
        runningMean[gid] = (1.0f - momentum) * runningMean[gid] + momentum * mean;
        runningVariance[gid] = (1.0f - momentum) * runningVariance[gid] + momentum * variance;
        savedMean[gid] = mean; savedInverse[gid] = inverse;
    } else { mean = runningMean[gid]; inverse = 1.0f / sqrt(runningVariance[gid] + as_type<float>(epsilonBits)); }
    for (uint b = 0; b < batch; ++b) for (uint s = 0; s < spatial; ++s) { uint index = (b * channels + gid) * spatial + s; output[index] = gamma[gid] * ((input[index] - mean) * inverse) + beta[gid]; }
}

kernel void batch_norm_backward_serial_channels(
    device const float* gradOutput [[buffer(0)]], device const float* input [[buffer(1)]], device const float* gamma [[buffer(2)]],
    device const float* savedMean [[buffer(3)]], device const float* savedInverse [[buffer(4)]],
    device float* gradInput [[buffer(5)]], device float* gradGamma [[buffer(6)]], device float* gradBeta [[buffer(7)]],
    constant uint& batch [[buffer(8)]], constant uint& channels [[buffer(9)]], constant uint& spatial [[buffer(10)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= channels) return;
    float mean = savedMean[gid], inverse = savedInverse[gid], dGamma = 0.0f, dBeta = 0.0f;
    for (uint b = 0; b < batch; ++b) for (uint s = 0; s < spatial; ++s) { uint index = (b * channels + gid) * spatial + s; float normalized = (input[index] - mean) * inverse; dGamma += gradOutput[index] * normalized; dBeta += gradOutput[index]; }
    gradGamma[gid] = dGamma; gradBeta[gid] = dBeta;
    float sum1 = 0.0f, sum2 = 0.0f;
    for (uint b = 0; b < batch; ++b) for (uint s = 0; s < spatial; ++s) { uint index = (b * channels + gid) * spatial + s; float normalized = (input[index] - mean) * inverse; sum1 += gradOutput[index]; sum2 += gradOutput[index] * normalized; }
    float count = float(batch * spatial);
    for (uint b = 0; b < batch; ++b) for (uint s = 0; s < spatial; ++s) { uint index = (b * channels + gid) * spatial + s; float normalized = (input[index] - mean) * inverse; gradInput[index] = gamma[gid] * inverse * (gradOutput[index] - sum1 / count - normalized * sum2 / count); }
}

kernel void layer_norm_forward_serial_rows(
    device const float* input [[buffer(0)]], device float* output [[buffer(1)]], device const float* gamma [[buffer(2)]], device const float* beta [[buffer(3)]],
    device float* savedMean [[buffer(4)]], device float* savedInverse [[buffer(5)]],
    constant uint& rows [[buffer(6)]], constant uint& width [[buffer(7)]], constant uint& epsilonBits [[buffer(8)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= rows) return; uint offset = gid * width; float sum = 0.0f;
    for (uint i = 0; i < width; ++i) sum += input[offset + i]; float mean = sum / float(width), variance = 0.0f;
    for (uint i = 0; i < width; ++i) { float d = input[offset + i] - mean; variance += d * d; }
    float inverse = 1.0f / sqrt(variance / float(width) + as_type<float>(epsilonBits)); savedMean[gid] = mean; savedInverse[gid] = inverse;
    for (uint i = 0; i < width; ++i) output[offset + i] = gamma[i] * ((input[offset + i] - mean) * inverse) + beta[i];
}

kernel void layer_norm_backward_serial(
    device const float* gradOutput [[buffer(0)]], device const float* input [[buffer(1)]], device const float* gamma [[buffer(2)]],
    device const float* savedMean [[buffer(3)]], device const float* savedInverse [[buffer(4)]],
    device float* gradInput [[buffer(5)]], device float* gradGamma [[buffer(6)]], device float* gradBeta [[buffer(7)]],
    constant uint& rows [[buffer(8)]], constant uint& width [[buffer(9)]], uint gid [[thread_position_in_grid]])
{
    if (gid != 0u) return; for (uint i = 0; i < width; ++i) { gradGamma[i] = 0.0f; gradBeta[i] = 0.0f; }
    for (uint row = 0; row < rows; ++row) {
        uint offset = row * width; float mean = savedMean[row], inverse = savedInverse[row];
        for (uint i = 0; i < width; ++i) { uint index = offset + i; float normalized = (input[index] - mean) * inverse; gradGamma[i] += gradOutput[index] * normalized; gradBeta[i] += gradOutput[index]; }
        float sum1 = 0.0f, sum2 = 0.0f;
        for (uint i = 0; i < width; ++i) { uint index = offset + i; float normalized = (input[index] - mean) * inverse; sum1 += gradOutput[index] * gamma[i]; sum2 += gradOutput[index] * gamma[i] * normalized; }
        for (uint i = 0; i < width; ++i) { uint index = offset + i; float normalized = (input[index] - mean) * inverse; gradInput[index] = inverse * (gradOutput[index] * gamma[i] - sum1 / float(width) - normalized * sum2 / float(width)); }
    }
}

kernel void group_norm_forward_serial_groups(
    device const float* input [[buffer(0)]], device float* output [[buffer(1)]], device const float* gamma [[buffer(2)]], device const float* beta [[buffer(3)]],
    device float* savedMean [[buffer(4)]], device float* savedInverse [[buffer(5)]],
    constant uint& batch [[buffer(6)]], constant uint& groups [[buffer(7)]], constant uint& channels [[buffer(8)]], constant uint& spatial [[buffer(9)]], constant uint& epsilonBits [[buffer(10)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= batch * groups) return; uint b = gid / groups, group = gid % groups, channelsPerGroup = channels / groups, groupSize = channelsPerGroup * spatial;
    float sum = 0.0f; for (uint c = 0; c < channelsPerGroup; ++c) for (uint s = 0; s < spatial; ++s) sum += input[(b * channels + group * channelsPerGroup + c) * spatial + s];
    float mean = sum / float(groupSize), variance = 0.0f;
    for (uint c = 0; c < channelsPerGroup; ++c) for (uint s = 0; s < spatial; ++s) { float d = input[(b * channels + group * channelsPerGroup + c) * spatial + s] - mean; variance += d * d; }
    float inverse = 1.0f / sqrt(variance / float(groupSize) + as_type<float>(epsilonBits)); savedMean[gid] = mean; savedInverse[gid] = inverse;
    for (uint c = 0; c < channelsPerGroup; ++c) { uint channel = group * channelsPerGroup + c; for (uint s = 0; s < spatial; ++s) { uint index = (b * channels + channel) * spatial + s; output[index] = gamma[channel] * ((input[index] - mean) * inverse) + beta[channel]; } }
}

kernel void instance_norm_forward_serial_channels(
    device const float* input [[buffer(0)]], device float* output [[buffer(1)]], device const float* gamma [[buffer(2)]], device const float* beta [[buffer(3)]],
    device float* savedMean [[buffer(4)]], device float* savedInverse [[buffer(5)]],
    constant uint& batch [[buffer(6)]], constant uint& channels [[buffer(7)]], constant uint& spatial [[buffer(8)]], constant uint& epsilonBits [[buffer(9)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= batch * channels) return; uint channel = gid % channels, offset = gid * spatial; float sum = 0.0f;
    for (uint s = 0; s < spatial; ++s) sum += input[offset + s]; float mean = sum / float(spatial), variance = 0.0f;
    for (uint s = 0; s < spatial; ++s) { float d = input[offset + s] - mean; variance += d * d; }
    float inverse = 1.0f / sqrt(variance / float(spatial) + as_type<float>(epsilonBits)); savedMean[gid] = mean; savedInverse[gid] = inverse;
    for (uint s = 0; s < spatial; ++s) output[offset + s] = gamma[channel] * ((input[offset + s] - mean) * inverse) + beta[channel];
}

kernel void instance_norm_backward_serial(
    device const float* gradOutput [[buffer(0)]], device const float* input [[buffer(1)]], device const float* gamma [[buffer(2)]],
    device const float* savedMean [[buffer(3)]], device const float* savedInverse [[buffer(4)]],
    device float* gradInput [[buffer(5)]], device float* gradGamma [[buffer(6)]], device float* gradBeta [[buffer(7)]],
    constant uint& batch [[buffer(8)]], constant uint& channels [[buffer(9)]], constant uint& spatial [[buffer(10)]], uint gid [[thread_position_in_grid]])
{
    if (gid != 0u) return; for (uint c = 0; c < channels; ++c) { gradGamma[c] = 0.0f; gradBeta[c] = 0.0f; }
    for (uint b = 0; b < batch; ++b) for (uint c = 0; c < channels; ++c) {
        uint unit = b * channels + c, offset = unit * spatial; float mean = savedMean[unit], inverse = savedInverse[unit], dGamma = 0.0f, dBeta = 0.0f;
        for (uint s = 0; s < spatial; ++s) { uint index = offset + s; float normalized = (input[index] - mean) * inverse; dGamma += gradOutput[index] * normalized; dBeta += gradOutput[index]; }
        gradGamma[c] += dGamma; gradBeta[c] += dBeta; float sum1 = 0.0f, sum2 = 0.0f;
        for (uint s = 0; s < spatial; ++s) { uint index = offset + s; float normalized = (input[index] - mean) * inverse; sum1 += gradOutput[index]; sum2 += gradOutput[index] * normalized; }
        for (uint s = 0; s < spatial; ++s) { uint index = offset + s; float normalized = (input[index] - mean) * inverse; gradInput[index] = gamma[c] * inverse * (gradOutput[index] - sum1 / float(spatial) - normalized * sum2 / float(spatial)); }
    }
}

kernel void rms_norm_forward_serial_rows(
    device const float* input [[buffer(0)]], device float* output [[buffer(1)]], device const float* gamma [[buffer(2)]], device float* savedRms [[buffer(3)]],
    constant uint& rows [[buffer(4)]], constant uint& width [[buffer(5)]], constant uint& epsilonBits [[buffer(6)]], uint gid [[thread_position_in_grid]])
{
    if (gid >= rows) return; uint offset = gid * width; float sum = 0.0f;
    for (uint i = 0; i < width; ++i) { float value = input[offset + i]; sum += value * value; }
    float rms = sqrt(sum / float(width) + as_type<float>(epsilonBits)); savedRms[gid] = rms;
    for (uint i = 0; i < width; ++i) output[offset + i] = input[offset + i] / rms * gamma[i];
}

kernel void rms_norm_backward_serial(
    device const float* gradOutput [[buffer(0)]], device const float* input [[buffer(1)]], device const float* gamma [[buffer(2)]], device const float* savedRms [[buffer(3)]],
    device float* gradInput [[buffer(4)]], device float* gradGamma [[buffer(5)]],
    constant uint& rows [[buffer(6)]], constant uint& width [[buffer(7)]], uint gid [[thread_position_in_grid]])
{
    if (gid != 0u) return; for (uint i = 0; i < width; ++i) gradGamma[i] = 0.0f;
    for (uint row = 0; row < rows; ++row) { uint offset = row * width; float rms = savedRms[row], inverse = 1.0f / rms;
        for (uint i = 0; i < width; ++i) { uint index = offset + i; gradGamma[i] += gradOutput[index] * input[index] * inverse; }
        float sum = 0.0f; for (uint i = 0; i < width; ++i) { uint index = offset + i; sum += gradOutput[index] * gamma[i] * input[index]; }
        for (uint i = 0; i < width; ++i) { uint index = offset + i; float normalizedGradient = gradOutput[index] * gamma[i]; gradInput[index] = inverse * (normalizedGradient - input[index] * sum / (rms * rms * float(width))); }
    }
}

inline uint dropout_threshold(float rate)
{
    uint bits = as_type<uint>(rate), exponentBits = (bits >> 23) & 255u, fraction = bits & 0x7fffffu;
    if (exponentBits == 0u) return fraction == 0u ? 0u : 1u;
    int exponent = int(exponentBits) - 127; ulong mantissa = ulong(fraction | 0x800000u);
    uint shift = uint(23 - exponent); ulong product = mantissa * 2147483647ul;
    if (shift >= 64u) return 1u;
    ulong divisor = 1ul << shift; return uint((product + divisor - 1ul) / divisor);
}

kernel void dropout_dotnet_random_serial(
    device const float* input [[buffer(0)]], device float* output [[buffer(1)]], device float* mask [[buffer(2)]],
    constant uint& count [[buffer(3)]], constant uint& rateBits [[buffer(4)]], constant uint& seed [[buffer(5)]], constant uint& training [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid != 0u) return; float rate = as_type<float>(rateBits);
    if (training == 0u || rate <= 0.0f) { for (uint i = 0; i < count; ++i) { mask[i] = 1.0f; output[i] = input[i]; } return; }
    if (rate >= 1.0f) { for (uint i = 0; i < count; ++i) { mask[i] = 0.0f; output[i] = 0.0f; } return; }
    int seedArray[56]; for (uint i = 0; i < 56u; ++i) seedArray[i] = 0;
    int subtraction = int(seed), mj = 161803398 - subtraction, mk = 1; seedArray[55] = mj;
    for (int i = 1; i < 55; ++i) { int ii = (21 * i) % 55; seedArray[ii] = mk; mk = mj - mk; if (mk < 0) mk += 2147483647; mj = seedArray[ii]; }
    for (int pass = 1; pass < 5; ++pass) for (int i = 1; i < 56; ++i) { seedArray[i] -= seedArray[1 + (i + 30) % 55]; if (seedArray[i] < 0) seedArray[i] += 2147483647; }
    int inext = 0, inextp = 21; uint threshold = dropout_threshold(rate); float scale = 1.0f / (1.0f - rate);
    for (uint i = 0; i < count; ++i) { if (++inext >= 56) inext = 1; if (++inextp >= 56) inextp = 1; int sample = seedArray[inext] - seedArray[inextp]; if (sample == 2147483647) --sample; if (sample < 0) sample += 2147483647; seedArray[inext] = sample; bool keep = uint(sample) >= threshold; mask[i] = keep ? scale : 0.0f; output[i] = keep ? input[i] * scale : 0.0f; }
}

kernel void embedding_lookup(
    device const int* indices [[buffer(0)]], device const float* table [[buffer(1)]], device float* output [[buffer(2)]],
    constant uint& entries [[buffer(3)]], constant uint& dimension [[buffer(4)]], constant uint& vocabulary [[buffer(5)]], uint gid [[thread_position_in_grid]])
{
    if (gid >= entries * dimension) return; uint entry = gid / dimension, feature = gid % dimension; int index = indices[entry]; output[gid] = index >= 0 && index < int(vocabulary) ? table[uint(index) * dimension + feature] : 0.0f;
}

kernel void embedding_backward_deterministic(
    device const float* gradOutput [[buffer(0)]], device const int* indices [[buffer(1)]], device float* gradEmbedding [[buffer(2)]],
    constant uint& entries [[buffer(3)]], constant uint& dimension [[buffer(4)]], constant uint& vocabulary [[buffer(5)]], uint gid [[thread_position_in_grid]])
{
    if (gid >= vocabulary * dimension) return; uint word = gid / dimension, feature = gid % dimension; float sum = 0.0f;
    for (uint entry = 0; entry < entries; ++entry) if (indices[entry] == int(word)) sum += gradOutput[entry * dimension + feature]; gradEmbedding[gid] = sum;
}

inline float resident_attention_dot(device const float* query, device const float* key,
    uint queryOffset, uint keyOffset, uint dimension)
{
    float sum = 0.0f; for (uint d = 0; d < dimension; ++d) sum += query[queryOffset + d] * key[keyOffset + d]; return sum;
}

kernel void attention_forward_serial(
    device const float* query [[buffer(0)]], device const float* key [[buffer(1)]], device const float* value [[buffer(2)]],
    device float* output [[buffer(3)]], device float* weights [[buffer(4)]], device const float* mask [[buffer(5)]],
    constant uint& batch [[buffer(6)]], constant uint& heads [[buffer(7)]], constant uint& queryLength [[buffer(8)]], constant uint& keyLength [[buffer(9)]],
    constant uint& dimension [[buffer(10)]], constant uint& scaleBits [[buffer(11)]], constant uint& causal [[buffer(12)]],
    constant uint& hasWeights [[buffer(13)]], constant uint& maskMode [[buffer(14)]], constant uint& softcapBits [[buffer(15)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid != 0u) return; float scale = as_type<float>(scaleBits); float softcap = as_type<float>(softcapBits);
    for (uint b = 0; b < batch; ++b) for (uint h = 0; h < heads; ++h) {
        uint queryOffset = (b * heads + h) * queryLength * dimension;
        uint keyOffset = (b * heads + h) * keyLength * dimension;
        uint weightOffset = (b * heads + h) * queryLength * keyLength;
        uint maskOffset = (maskMode == 2u ? (b * heads + h) * queryLength * keyLength : 0u);
        for (uint i = 0; i < queryLength; ++i) {
            float maximum = -INFINITY;
            for (uint j = 0; j < keyLength; ++j) { float score = resident_attention_dot(query, key, queryOffset + i * dimension, keyOffset + j * dimension, dimension) * scale; if (softcap > 0.0f) score = softcap * tanh(score / softcap); if ((causal != 0u && j > i) || (maskMode != 0u && mask[maskOffset + i * keyLength + j] == 0.0f)) score = -INFINITY; maximum = max(maximum, score); }
            float sumExp = 0.0f;
            for (uint j = 0; j < keyLength; ++j) { float score = resident_attention_dot(query, key, queryOffset + i * dimension, keyOffset + j * dimension, dimension) * scale; if (softcap > 0.0f) score = softcap * tanh(score / softcap); if ((causal != 0u && j > i) || (maskMode != 0u && mask[maskOffset + i * keyLength + j] == 0.0f)) score = -INFINITY; sumExp += exp(score - maximum); }
            for (uint j = 0; j < keyLength; ++j) { float score = resident_attention_dot(query, key, queryOffset + i * dimension, keyOffset + j * dimension, dimension) * scale; if (softcap > 0.0f) score = softcap * tanh(score / softcap); if ((causal != 0u && j > i) || (maskMode != 0u && mask[maskOffset + i * keyLength + j] == 0.0f)) score = -INFINITY; float weight = sumExp > 0.0f ? exp(score - maximum) / sumExp : 0.0f; if (hasWeights != 0u) weights[weightOffset + i * keyLength + j] = weight; }
            for (uint d = 0; d < dimension; ++d) { float sum = 0.0f; for (uint j = 0; j < keyLength; ++j) { float score = resident_attention_dot(query, key, queryOffset + i * dimension, keyOffset + j * dimension, dimension) * scale; if (softcap > 0.0f) score = softcap * tanh(score / softcap); if ((causal != 0u && j > i) || (maskMode != 0u && mask[maskOffset + i * keyLength + j] == 0.0f)) score = -INFINITY; sum += (sumExp > 0.0f ? exp(score - maximum) / sumExp : 0.0f) * value[keyOffset + j * dimension + d]; } output[queryOffset + i * dimension + d] = sum; }
        }
    }
}

kernel void attention_backward_serial(
    device const float* gradOutput [[buffer(0)]], device const float* query [[buffer(1)]], device const float* key [[buffer(2)]], device const float* value [[buffer(3)]],
    device const float* weights [[buffer(4)]], device float* gradQuery [[buffer(5)]], device float* gradKey [[buffer(6)]], device float* gradValue [[buffer(7)]],
    constant uint& batch [[buffer(8)]], constant uint& heads [[buffer(9)]], constant uint& sequence [[buffer(10)]], constant uint& dimension [[buffer(11)]],
    constant uint& scaleBits [[buffer(12)]], constant uint& causal [[buffer(13)]], uint gid [[thread_position_in_grid]])
{
    if (gid != 0u) return; float scale = as_type<float>(scaleBits), zero = 0.0f; uint tensorCount = batch * heads * sequence * dimension;
    for (uint i = 0; i < tensorCount; ++i) { gradQuery[i] = zero; gradKey[i] = zero; gradValue[i] = zero; }
    for (uint b = 0; b < batch; ++b) for (uint h = 0; h < heads; ++h) { uint headOffset = (b * heads + h) * sequence * dimension, weightOffset = (b * heads + h) * sequence * sequence;
        for (uint j = 0; j < sequence; ++j) for (uint d = 0; d < dimension; ++d) { float sum = 0.0f; for (uint i = 0; i < sequence; ++i) sum += weights[weightOffset + i * sequence + j] * gradOutput[headOffset + i * dimension + d]; gradValue[headOffset + j * dimension + d] = sum; }
        for (uint i = 0; i < sequence; ++i) {
            float dot = 0.0f; for (uint j = 0; j < sequence; ++j) { float gradWeight = 0.0f; for (uint d = 0; d < dimension; ++d) gradWeight += gradOutput[headOffset + i * dimension + d] * value[headOffset + j * dimension + d]; dot += gradWeight * weights[weightOffset + i * sequence + j]; }
            for (uint d = 0; d < dimension; ++d) { float sum = 0.0f; for (uint j = 0; j < sequence; ++j) { float gradWeight = 0.0f; for (uint x = 0; x < dimension; ++x) gradWeight += gradOutput[headOffset + i * dimension + x] * value[headOffset + j * dimension + x]; float gradScore = weights[weightOffset + i * sequence + j] * (gradWeight - dot); if (causal != 0u && j > i) gradScore = 0.0f; sum += gradScore * key[headOffset + j * dimension + d]; } gradQuery[headOffset + i * dimension + d] = sum * scale; }
        }
        for (uint j = 0; j < sequence; ++j) for (uint d = 0; d < dimension; ++d) { float sum = 0.0f; for (uint i = 0; i < sequence; ++i) { float dot = 0.0f; for (uint x = 0; x < sequence; ++x) { float gradWeight = 0.0f; for (uint y = 0; y < dimension; ++y) gradWeight += gradOutput[headOffset + i * dimension + y] * value[headOffset + x * dimension + y]; dot += gradWeight * weights[weightOffset + i * sequence + x]; } float gradWeight = 0.0f; for (uint y = 0; y < dimension; ++y) gradWeight += gradOutput[headOffset + i * dimension + y] * value[headOffset + j * dimension + y]; float gradScore = weights[weightOffset + i * sequence + j] * (gradWeight - dot); if (causal != 0u && j > i) gradScore = 0.0f; sum += gradScore * query[headOffset + i * dimension + d]; } gradKey[headOffset + j * dimension + d] = sum * scale; }
    }
}

kernel void flash_attention_forward_serial(
    device const float* query [[buffer(0)]], device const float* key [[buffer(1)]], device const float* value [[buffer(2)]],
    device float* output [[buffer(3)]], device float* stats [[buffer(4)]], device const float* bias [[buffer(5)]],
    constant uint& batch [[buffer(6)]], constant uint& heads [[buffer(7)]], constant uint& queryLength [[buffer(8)]], constant uint& keyLength [[buffer(9)]], constant uint& dimension [[buffer(10)]],
    constant uint& scaleBits [[buffer(11)]], constant uint& causal [[buffer(12)]], constant uint& hasBias [[buffer(13)]], constant uint& biasBatchStride [[buffer(14)]], uint gid [[thread_position_in_grid]])
{
    if (gid != 0u) return; float scale = as_type<float>(scaleBits);
    for (uint b = 0; b < batch; ++b) for (uint h = 0; h < heads; ++h) { uint queryOffset = (b * heads + h) * queryLength * dimension, keyOffset = (b * heads + h) * keyLength * dimension, statsOffset = (b * heads + h) * queryLength, biasOffset = b * biasBatchStride + h * queryLength * keyLength;
        for (uint i = 0; i < queryLength; ++i) { float maximum = -INFINITY;
            for (uint j = 0; j < keyLength; ++j) if (causal == 0u || j <= i) { float score = resident_attention_dot(query, key, queryOffset + i * dimension, keyOffset + j * dimension, dimension) * scale; if (hasBias != 0u) score += bias[biasOffset + i * keyLength + j]; maximum = max(maximum, score); }
            float sumExp = 0.0f; for (uint j = 0; j < keyLength; ++j) if (causal == 0u || j <= i) { float score = resident_attention_dot(query, key, queryOffset + i * dimension, keyOffset + j * dimension, dimension) * scale; if (hasBias != 0u) score += bias[biasOffset + i * keyLength + j]; sumExp += exp(score - maximum); }
            stats[statsOffset + i] = maximum + log(sumExp);
            for (uint d = 0; d < dimension; ++d) { float sum = 0.0f; for (uint j = 0; j < keyLength; ++j) if (causal == 0u || j <= i) { float score = resident_attention_dot(query, key, queryOffset + i * dimension, keyOffset + j * dimension, dimension) * scale; if (hasBias != 0u) score += bias[biasOffset + i * keyLength + j]; sum += (exp(score - maximum) / sumExp) * value[keyOffset + j * dimension + d]; } output[queryOffset + i * dimension + d] = sum; }
        }
    }
}

kernel void flash_attention_backward_serial(
    device const float* gradOutput [[buffer(0)]], device const float* query [[buffer(1)]], device const float* key [[buffer(2)]], device const float* value [[buffer(3)]],
    device const float* stats [[buffer(4)]], device const float* bias [[buffer(5)]], device float* gradQuery [[buffer(6)]], device float* gradKey [[buffer(7)]], device float* gradValue [[buffer(8)]],
    constant uint& batch [[buffer(9)]], constant uint& heads [[buffer(10)]], constant uint& queryLength [[buffer(11)]], constant uint& keyLength [[buffer(12)]], constant uint& dimension [[buffer(13)]],
    constant uint& scaleBits [[buffer(14)]], constant uint& causal [[buffer(15)]], constant uint& hasBias [[buffer(16)]], constant uint& biasBatchStride [[buffer(17)]], uint gid [[thread_position_in_grid]])
{
    if (gid != 0u) return; float scale = as_type<float>(scaleBits); uint queryCount = batch * heads * queryLength * dimension, keyCount = batch * heads * keyLength * dimension;
    for (uint i = 0; i < queryCount; ++i) gradQuery[i] = 0.0f; for (uint i = 0; i < keyCount; ++i) { gradKey[i] = 0.0f; gradValue[i] = 0.0f; }
    for (uint b = 0; b < batch; ++b) for (uint h = 0; h < heads; ++h) { uint queryOffset = (b * heads + h) * queryLength * dimension, keyOffset = (b * heads + h) * keyLength * dimension, statsOffset = (b * heads + h) * queryLength, biasOffset = b * biasBatchStride + h * queryLength * keyLength;
        for (uint i = 0; i < queryLength; ++i) { float lse = stats[statsOffset + i];
            for (uint j = 0; j < keyLength; ++j) { float weight = 0.0f; if (causal == 0u || j <= i) { float score = resident_attention_dot(query, key, queryOffset + i * dimension, keyOffset + j * dimension, dimension) * scale; if (hasBias != 0u) score += bias[biasOffset + i * keyLength + j]; weight = exp(score - lse); } for (uint d = 0; d < dimension; ++d) gradValue[keyOffset + j * dimension + d] += weight * gradOutput[queryOffset + i * dimension + d]; }
            float dot = 0.0f; for (uint d = 0; d < dimension; ++d) { float sum = 0.0f; for (uint j = 0; j < keyLength; ++j) { float weight = 0.0f; if (causal == 0u || j <= i) { float score = resident_attention_dot(query, key, queryOffset + i * dimension, keyOffset + j * dimension, dimension) * scale; if (hasBias != 0u) score += bias[biasOffset + i * keyLength + j]; weight = exp(score - lse); } sum += weight * value[keyOffset + j * dimension + d]; } dot += gradOutput[queryOffset + i * dimension + d] * sum; }
            for (uint j = 0; j < keyLength; ++j) { float weight = 0.0f; if (causal == 0u || j <= i) { float score = resident_attention_dot(query, key, queryOffset + i * dimension, keyOffset + j * dimension, dimension) * scale; if (hasBias != 0u) score += bias[biasOffset + i * keyLength + j]; weight = exp(score - lse); } float gradWeight = 0.0f; for (uint d = 0; d < dimension; ++d) gradWeight += gradOutput[queryOffset + i * dimension + d] * value[keyOffset + j * dimension + d]; float gradScore = weight * (gradWeight - dot) * scale; for (uint d = 0; d < dimension; ++d) { gradQuery[queryOffset + i * dimension + d] += gradScore * key[keyOffset + j * dimension + d]; gradKey[keyOffset + j * dimension + d] += gradScore * query[queryOffset + i * dimension + d]; } }
        }
    }
}

kernel void grouped_attention_forward_serial(
    device const float* query [[buffer(0)]], device const float* key [[buffer(1)]], device const float* value [[buffer(2)]],
    device float* output [[buffer(3)]], device float* weights [[buffer(4)]],
    constant uint& batch [[buffer(5)]], constant uint& queryHeads [[buffer(6)]], constant uint& keyHeads [[buffer(7)]], constant uint& queryLength [[buffer(8)]], constant uint& keyLength [[buffer(9)]], constant uint& dimension [[buffer(10)]],
    constant uint& scaleBits [[buffer(11)]], constant uint& causal [[buffer(12)]], constant uint& hasWeights [[buffer(13)]], uint gid [[thread_position_in_grid]])
{
    if (gid != 0u) return; float scale = as_type<float>(scaleBits); uint headsPerGroup = queryHeads / keyHeads;
    for (uint b = 0; b < batch; ++b) for (uint qh = 0; qh < queryHeads; ++qh) { uint kh = qh / headsPerGroup, queryOffset = (b * queryHeads + qh) * queryLength * dimension, keyOffset = (b * keyHeads + kh) * keyLength * dimension, weightOffset = (b * queryHeads + qh) * queryLength * keyLength;
        for (uint i = 0; i < queryLength; ++i) { float maximum = -INFINITY; for (uint j = 0; j < keyLength; ++j) if (causal == 0u || j <= i) maximum = max(maximum, resident_attention_dot(query, key, queryOffset + i * dimension, keyOffset + j * dimension, dimension) * scale);
            float sumExp = 0.0f; for (uint j = 0; j < keyLength; ++j) if (causal == 0u || j <= i) sumExp += exp(resident_attention_dot(query, key, queryOffset + i * dimension, keyOffset + j * dimension, dimension) * scale - maximum);
            if (hasWeights != 0u) for (uint j = 0; j < keyLength; ++j) weights[weightOffset + i * keyLength + j] = (causal != 0u && j > i) ? 0.0f : exp(resident_attention_dot(query, key, queryOffset + i * dimension, keyOffset + j * dimension, dimension) * scale - maximum) / sumExp;
            for (uint d = 0; d < dimension; ++d) { float sum = 0.0f; for (uint j = 0; j < keyLength; ++j) if (causal == 0u || j <= i) sum += exp(resident_attention_dot(query, key, queryOffset + i * dimension, keyOffset + j * dimension, dimension) * scale - maximum) / sumExp * value[keyOffset + j * dimension + d]; output[queryOffset + i * dimension + d] = sum; }
        }
    }
}

kernel void grouped_attention_backward_serial(
    device const float* gradOutput [[buffer(0)]], device const float* query [[buffer(1)]], device const float* key [[buffer(2)]], device const float* value [[buffer(3)]], device const float* weights [[buffer(4)]],
    device float* gradQuery [[buffer(5)]], device float* gradKey [[buffer(6)]], device float* gradValue [[buffer(7)]],
    constant uint& batch [[buffer(8)]], constant uint& queryHeads [[buffer(9)]], constant uint& keyHeads [[buffer(10)]], constant uint& queryLength [[buffer(11)]], constant uint& keyLength [[buffer(12)]], constant uint& dimension [[buffer(13)]],
    constant uint& scaleBits [[buffer(14)]], constant uint& queriesPerKey [[buffer(15)]], uint gid [[thread_position_in_grid]])
{
    if (gid != 0u) return; float scale = as_type<float>(scaleBits); uint queryCount = batch * queryHeads * queryLength * dimension, keyCount = batch * keyHeads * keyLength * dimension;
    for (uint i = 0; i < queryCount; ++i) gradQuery[i] = 0.0f; for (uint i = 0; i < keyCount; ++i) { gradKey[i] = 0.0f; gradValue[i] = 0.0f; }
    for (uint b = 0; b < batch; ++b) for (uint qh = 0; qh < queryHeads; ++qh) { uint kh = qh / queriesPerKey, queryOffset = (b * queryHeads + qh) * queryLength * dimension, keyOffset = (b * keyHeads + kh) * keyLength * dimension, weightOffset = (b * queryHeads + qh) * queryLength * keyLength;
        for (uint i = 0; i < queryLength; ++i) { for (uint j = 0; j < keyLength; ++j) { float weight = weights[weightOffset + i * keyLength + j]; for (uint d = 0; d < dimension; ++d) gradValue[keyOffset + j * dimension + d] += weight * gradOutput[queryOffset + i * dimension + d]; }
            float dot = 0.0f; for (uint j = 0; j < keyLength; ++j) { float gradWeight = 0.0f; for (uint d = 0; d < dimension; ++d) gradWeight += gradOutput[queryOffset + i * dimension + d] * value[keyOffset + j * dimension + d]; dot += gradWeight * weights[weightOffset + i * keyLength + j]; }
            for (uint j = 0; j < keyLength; ++j) { float gradWeight = 0.0f; for (uint d = 0; d < dimension; ++d) gradWeight += gradOutput[queryOffset + i * dimension + d] * value[keyOffset + j * dimension + d]; float gradScore = weights[weightOffset + i * keyLength + j] * (gradWeight - dot) * scale; for (uint d = 0; d < dimension; ++d) { gradQuery[queryOffset + i * dimension + d] += gradScore * key[keyOffset + j * dimension + d]; gradKey[keyOffset + j * dimension + d] += gradScore * query[queryOffset + i * dimension + d]; } }
        }
    }
}

inline int resident_reflect(int coordinate, int size)
{
    if (size == 1) return 0; while (coordinate < 0 || coordinate >= size) { if (coordinate < 0) coordinate = -coordinate - 1; if (coordinate >= size) coordinate = 2 * size - coordinate - 1; } return coordinate;
}
inline bool resident_map_pixel(thread int& y, thread int& x, int height, int width, uint padding)
{
    if (y >= 0 && y < height && x >= 0 && x < width) return true; if (padding == 0u) return false;
    if (padding == 1u) { y = clamp(y, 0, height - 1); x = clamp(x, 0, width - 1); }
    else { y = resident_reflect(y, height); x = resident_reflect(x, width); } return true;
}
inline float resident_pixel(device const float* input, uint b, uint c, int y, int x, uint channels, uint height, uint width, uint padding)
{
    if (!resident_map_pixel(y, x, int(height), int(width), padding)) return 0.0f; return input[(b * channels + c) * height * width + uint(y) * width + uint(x)];
}

kernel void affine_grid(
    device const float* theta [[buffer(0)]], device float* grid [[buffer(1)]],
    constant uint& batch [[buffer(2)]], constant uint& height [[buffer(3)]], constant uint& width [[buffer(4)]], uint gid [[thread_position_in_grid]])
{
    if (gid >= batch * height * width) return; uint xIndex = gid % width, value = gid / width, yIndex = value % height, b = value / height;
    float y = 2.0f * float(yIndex) / float(height - 1u) - 1.0f, x = 2.0f * float(xIndex) / float(width - 1u) - 1.0f; uint offset = b * 6u;
    grid[gid * 2u] = theta[offset] * x + theta[offset + 1u] * y + theta[offset + 2u]; grid[gid * 2u + 1u] = theta[offset + 3u] * x + theta[offset + 4u] * y + theta[offset + 5u];
}

kernel void grid_sample(
    device const float* input [[buffer(0)]], device const float* grid [[buffer(1)]], device float* output [[buffer(2)]],
    constant uint& batch [[buffer(3)]], constant uint& channels [[buffer(4)]], constant uint& inputHeight [[buffer(5)]], constant uint& inputWidth [[buffer(6)]],
    constant uint& outputHeight [[buffer(7)]], constant uint& outputWidth [[buffer(8)]], constant uint& padding [[buffer(9)]], constant uint& alignCorners [[buffer(10)]], uint gid [[thread_position_in_grid]])
{
    if (gid >= batch * channels * outputHeight * outputWidth) return; uint xIndex = gid % outputWidth, value = gid / outputWidth, yIndex = value % outputHeight; value /= outputHeight; uint c = value % channels, b = value / channels;
    uint gridIndex = (b * outputHeight * outputWidth + yIndex * outputWidth + xIndex) * 2u; float x = grid[gridIndex], y = grid[gridIndex + 1u];
    float inputX = alignCorners != 0u ? (x + 1.0f) * float(inputWidth - 1u) / 2.0f : ((x + 1.0f) * float(inputWidth) - 1.0f) / 2.0f;
    float inputY = alignCorners != 0u ? (y + 1.0f) * float(inputHeight - 1u) / 2.0f : ((y + 1.0f) * float(inputHeight) - 1.0f) / 2.0f;
    int y0 = int(floor(inputY)), x0 = int(floor(inputX)); float fy = inputY - float(y0), fx = inputX - float(x0);
    float v00 = resident_pixel(input, b, c, y0, x0, channels, inputHeight, inputWidth, padding), v01 = resident_pixel(input, b, c, y0, x0 + 1, channels, inputHeight, inputWidth, padding);
    float v10 = resident_pixel(input, b, c, y0 + 1, x0, channels, inputHeight, inputWidth, padding), v11 = resident_pixel(input, b, c, y0 + 1, x0 + 1, channels, inputHeight, inputWidth, padding);
    output[gid] = (1.0f - fy) * (1.0f - fx) * v00 + (1.0f - fy) * fx * v01 + fy * (1.0f - fx) * v10 + fy * fx * v11;
}

kernel void grid_sample_backward_serial(
    device const float* gradOutput [[buffer(0)]], device const float* input [[buffer(1)]], device const float* grid [[buffer(2)]], device float* gradInput [[buffer(3)]], device float* gradGrid [[buffer(4)]],
    constant uint& batch [[buffer(5)]], constant uint& channels [[buffer(6)]], constant uint& inputHeight [[buffer(7)]], constant uint& inputWidth [[buffer(8)]],
    constant uint& outputHeight [[buffer(9)]], constant uint& outputWidth [[buffer(10)]], constant uint& padding [[buffer(11)]], constant uint& alignCorners [[buffer(12)]], uint gid [[thread_position_in_grid]])
{
    if (gid != 0u) return; uint inputCount = batch * channels * inputHeight * inputWidth; for (uint i = 0; i < inputCount; ++i) gradInput[i] = 0.0f;
    for (uint b = 0; b < batch; ++b) for (uint oy = 0; oy < outputHeight; ++oy) for (uint ox = 0; ox < outputWidth; ++ox) { uint gridIndex = (b * outputHeight * outputWidth + oy * outputWidth + ox) * 2u; float x = grid[gridIndex], y = grid[gridIndex + 1u];
        float inputX = alignCorners != 0u ? (x + 1.0f) * float(inputWidth - 1u) / 2.0f : ((x + 1.0f) * float(inputWidth) - 1.0f) / 2.0f, inputY = alignCorners != 0u ? (y + 1.0f) * float(inputHeight - 1u) / 2.0f : ((y + 1.0f) * float(inputHeight) - 1.0f) / 2.0f;
        int y0 = int(floor(inputY)), x0 = int(floor(inputX)); float fy = inputY - float(y0), fx = inputX - float(x0), gradX = 0.0f, gradY = 0.0f;
        for (uint c = 0; c < channels; ++c) { uint outputIndex = (b * channels + c) * outputHeight * outputWidth + oy * outputWidth + ox; float gradient = gradOutput[outputIndex];
            int ys[4] = { y0, y0, y0 + 1, y0 + 1 }, xs[4] = { x0, x0 + 1, x0, x0 + 1 }; float factors[4] = { (1.0f - fy) * (1.0f - fx), (1.0f - fy) * fx, fy * (1.0f - fx), fy * fx };
            for (uint q = 0; q < 4u; ++q) { int py = ys[q], px = xs[q]; if (resident_map_pixel(py, px, int(inputHeight), int(inputWidth), padding)) gradInput[(b * channels + c) * inputHeight * inputWidth + uint(py) * inputWidth + uint(px)] += factors[q] * gradient; }
            float v00 = resident_pixel(input, b, c, y0, x0, channels, inputHeight, inputWidth, padding), v01 = resident_pixel(input, b, c, y0, x0 + 1, channels, inputHeight, inputWidth, padding), v10 = resident_pixel(input, b, c, y0 + 1, x0, channels, inputHeight, inputWidth, padding), v11 = resident_pixel(input, b, c, y0 + 1, x0 + 1, channels, inputHeight, inputWidth, padding);
            float dWidth = (1.0f - fy) * (v01 - v00) + fy * (v11 - v10), dHeight = (1.0f - fx) * (v10 - v00) + fx * (v11 - v01);
            gradX += gradient * dWidth * (alignCorners != 0u ? float(inputWidth - 1u) / 2.0f : float(inputWidth) / 2.0f); gradY += gradient * dHeight * (alignCorners != 0u ? float(inputHeight - 1u) / 2.0f : float(inputHeight) / 2.0f);
        }
        gradGrid[gridIndex] = gradX; gradGrid[gridIndex + 1u] = gradY;
    }
}

inline float resident_stable_sigmoid(float x)
{
    if (x >= 0.0f) { float value = exp(-x); return 1.0f / (1.0f + value); }
    float value = exp(x); return value / (1.0f + value);
}

kernel void lstm_forward_sequence(
    device const float* input [[buffer(0)]], device const float* weightsIh [[buffer(1)]], device const float* weightsHh [[buffer(2)]],
    device const float* biasIh [[buffer(3)]], device const float* biasHh [[buffer(4)]], device float* output [[buffer(5)]],
    device float* allH [[buffer(6)]], device float* allC [[buffer(7)]], device float* gates [[buffer(8)]],
    constant uint& sequence [[buffer(9)]], constant uint& batch [[buffer(10)]], constant uint& inputSize [[buffer(11)]], constant uint& hiddenSize [[buffer(12)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= batch) return; uint stateSize = batch * hiddenSize;
    for (uint t = 0; t < sequence; ++t) { uint inputOffset = t * batch * inputSize + gid * inputSize, previousOffset = t * stateSize + gid * hiddenSize, currentOffset = (t + 1u) * stateSize + gid * hiddenSize, outputOffset = t * stateSize + gid * hiddenSize, gateOffset = (t * batch + gid) * hiddenSize * 4u;
        for (uint h = 0; h < hiddenSize; ++h) { float activated[4];
            for (uint gate = 0; gate < 4u; ++gate) { uint gateRow = gate * hiddenSize + h; float sum = biasIh[gateRow] + biasHh[gateRow]; for (uint i = 0; i < inputSize; ++i) sum += weightsIh[gateRow * inputSize + i] * input[inputOffset + i]; for (uint hp = 0; hp < hiddenSize; ++hp) sum += weightsHh[gateRow * hiddenSize + hp] * allH[previousOffset + hp]; activated[gate] = gate == 2u ? tanh(sum) : resident_stable_sigmoid(sum); }
            for (uint gate = 0; gate < 4u; ++gate) gates[gateOffset + h * 4u + gate] = activated[gate];
            float cell = activated[1] * allC[previousOffset + h] + activated[0] * activated[2], hidden = activated[3] * tanh(cell);
            allC[currentOffset + h] = cell; allH[currentOffset + h] = hidden; output[outputOffset + h] = hidden;
        }
    }
}

kernel void lstm_backward_sequence_serial(
    device const float* gradOutput [[buffer(0)]], device const float* allH [[buffer(1)]], device const float* allC [[buffer(2)]], device const float* gates [[buffer(3)]],
    device const float* weightsIh [[buffer(4)]], device const float* weightsHh [[buffer(5)]], device const float* input [[buffer(6)]],
    device float* gradInput [[buffer(7)]], device float* gradHInit [[buffer(8)]], device float* gradCInit [[buffer(9)]],
    device float* gradWeightsIh [[buffer(10)]], device float* gradWeightsHh [[buffer(11)]], device float* gradBias [[buffer(12)]],
    device float* nextH [[buffer(13)]], device float* nextC [[buffer(14)]],
    constant uint& sequence [[buffer(15)]], constant uint& batch [[buffer(16)]], constant uint& inputSize [[buffer(17)]], constant uint& hiddenSize [[buffer(18)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid != 0u) return; uint stateSize = batch * hiddenSize, gateRows = 4u * hiddenSize;
    for (uint i = 0; i < sequence * batch * inputSize; ++i) gradInput[i] = 0.0f;
    for (uint i = 0; i < stateSize; ++i) { gradHInit[i] = 0.0f; gradCInit[i] = 0.0f; nextH[i] = 0.0f; nextC[i] = 0.0f; }
    for (uint i = 0; i < gateRows * inputSize; ++i) gradWeightsIh[i] = 0.0f; for (uint i = 0; i < gateRows * hiddenSize; ++i) gradWeightsHh[i] = 0.0f; for (uint i = 0; i < gateRows; ++i) gradBias[i] = 0.0f;
    for (int time = int(sequence) - 1; time >= 0; --time) { uint t = uint(time), inputOffsetBase = t * batch * inputSize, previousBase = t * stateSize, currentBase = (t + 1u) * stateSize, outputBase = t * stateSize, gateBase = t * batch * hiddenSize * 4u;
        for (uint b = 0; b < batch; ++b) for (uint h = 0; h < hiddenSize; ++h) { uint cache = gateBase + b * hiddenSize * 4u + h * 4u, state = b * hiddenSize + h; float inputGate = gates[cache], forgetGate = gates[cache + 1u], candidate = gates[cache + 2u], outputGate = gates[cache + 3u], previousCell = allC[previousBase + state], currentCell = allC[currentBase + state];
            float dHidden = gradOutput[outputBase + state] + nextH[state], tanhCell = tanh(currentCell), dOutput = dHidden * tanhCell, dCell = dHidden * outputGate * (1.0f - tanhCell * tanhCell) + nextC[state];
            float raw[4] = { dCell * candidate * inputGate * (1.0f - inputGate), dCell * previousCell * forgetGate * (1.0f - forgetGate), dCell * inputGate * (1.0f - candidate * candidate), dOutput * outputGate * (1.0f - outputGate) };
            for (uint gate = 0; gate < 4u; ++gate) { uint row = gate * hiddenSize + h; gradBias[row] += raw[gate]; for (uint i = 0; i < inputSize; ++i) gradWeightsIh[row * inputSize + i] += raw[gate] * input[inputOffsetBase + b * inputSize + i]; for (uint hp = 0; hp < hiddenSize; ++hp) gradWeightsHh[row * hiddenSize + hp] += raw[gate] * allH[previousBase + b * hiddenSize + hp]; }
            for (uint i = 0; i < inputSize; ++i) { float sum = 0.0f; for (uint gate = 0; gate < 4u; ++gate) sum += raw[gate] * weightsIh[(gate * hiddenSize + h) * inputSize + i]; gradInput[inputOffsetBase + b * inputSize + i] += sum; }
            for (uint hp = 0; hp < hiddenSize; ++hp) { float sum = 0.0f; for (uint gate = 0; gate < 4u; ++gate) sum += raw[gate] * weightsHh[(gate * hiddenSize + hp) * hiddenSize + h]; if (t > 0u) nextH[b * hiddenSize + hp] = sum; else gradHInit[b * hiddenSize + hp] += sum; }
            float previousCellGradient = dCell * forgetGate; if (t > 0u) nextC[state] = previousCellGradient; else gradCInit[state] = previousCellGradient;
        }
    }
}

kernel void gru_forward_sequence(
    device const float* input [[buffer(0)]], device const float* weightsIh [[buffer(1)]], device const float* weightsHh [[buffer(2)]],
    device const float* biasIh [[buffer(3)]], device const float* biasHh [[buffer(4)]], device float* output [[buffer(5)]], device float* allH [[buffer(6)]], device float* gates [[buffer(7)]],
    constant uint& sequence [[buffer(8)]], constant uint& batch [[buffer(9)]], constant uint& inputSize [[buffer(10)]], constant uint& hiddenSize [[buffer(11)]], uint gid [[thread_position_in_grid]])
{
    if (gid >= batch) return; uint stateSize = batch * hiddenSize;
    for (uint t = 0; t < sequence; ++t) { uint inputOffset = (t * batch + gid) * inputSize, previousOffset = t * stateSize + gid * hiddenSize, currentOffset = (t + 1u) * stateSize + gid * hiddenSize, gateOffset = (t * batch + gid) * hiddenSize * 3u;
        for (uint gate = 0; gate < 2u; ++gate) for (uint h = 0; h < hiddenSize; ++h) { uint row = gate * hiddenSize + h; float sum = biasIh[row] + biasHh[row]; for (uint i = 0; i < inputSize; ++i) sum += weightsIh[row * inputSize + i] * input[inputOffset + i]; for (uint hp = 0; hp < hiddenSize; ++hp) sum += weightsHh[row * hiddenSize + hp] * allH[previousOffset + hp]; gates[gateOffset + h * 3u + gate] = resident_stable_sigmoid(sum); }
        for (uint h = 0; h < hiddenSize; ++h) { uint row = 2u * hiddenSize + h; float sum = biasIh[row] + biasHh[row]; for (uint i = 0; i < inputSize; ++i) sum += weightsIh[row * inputSize + i] * input[inputOffset + i]; for (uint hp = 0; hp < hiddenSize; ++hp) sum += weightsHh[row * hiddenSize + hp] * (gates[gateOffset + hp * 3u] * allH[previousOffset + hp]); gates[gateOffset + h * 3u + 2u] = tanh(sum); }
        for (uint h = 0; h < hiddenSize; ++h) { float update = gates[gateOffset + h * 3u + 1u], candidate = gates[gateOffset + h * 3u + 2u], previous = allH[previousOffset + h], current = (1.0f - update) * candidate + update * previous; allH[currentOffset + h] = current; output[t * stateSize + gid * hiddenSize + h] = current; }
    }
}

kernel void gru_backward_sequence_serial(
    device const float* gradOutput [[buffer(0)]], device const float* allH [[buffer(1)]], device const float* gates [[buffer(2)]],
    device const float* weightsIh [[buffer(3)]], device const float* weightsHh [[buffer(4)]], device const float* input [[buffer(5)]],
    device float* gradInput [[buffer(6)]], device float* gradHInit [[buffer(7)]], device float* nextH [[buffer(8)]],
    device float* gradWeightsIh [[buffer(9)]], device float* gradWeightsHh [[buffer(10)]], device float* gradBias [[buffer(11)]],
    constant uint& sequence [[buffer(12)]], constant uint& batch [[buffer(13)]], constant uint& inputSize [[buffer(14)]], constant uint& hiddenSize [[buffer(15)]], uint gid [[thread_position_in_grid]])
{
    if (gid != 0u) return; uint stateSize = batch * hiddenSize, gateRows = 3u * hiddenSize;
    for (uint i = 0; i < sequence * batch * inputSize; ++i) gradInput[i] = 0.0f; for (uint i = 0; i < stateSize; ++i) { gradHInit[i] = 0.0f; nextH[i] = 0.0f; }
    for (uint i = 0; i < gateRows * inputSize; ++i) gradWeightsIh[i] = 0.0f; for (uint i = 0; i < gateRows * hiddenSize; ++i) gradWeightsHh[i] = 0.0f; for (uint i = 0; i < gateRows; ++i) gradBias[i] = 0.0f;
    for (int time = int(sequence) - 1; time >= 0; --time) { uint t = uint(time), inputBase = t * batch * inputSize, previousBase = t * stateSize, outputBase = t * stateSize, gateBase = t * batch * hiddenSize * 3u;
        for (uint b = 0; b < batch; ++b) for (uint h = 0; h < hiddenSize; ++h) { uint state = b * hiddenSize + h, cache = gateBase + state * 3u; float reset = gates[cache], update = gates[cache + 1u], candidate = gates[cache + 2u], previous = allH[previousBase + state], dHidden = gradOutput[outputBase + state] + nextH[state];
            float rawCandidate = dHidden * (1.0f - update) * (1.0f - candidate * candidate), rawUpdate = dHidden * (previous - candidate) * update * (1.0f - update), gatedPreviousGradient = 0.0f;
            for (uint hp = 0; hp < hiddenSize; ++hp) gatedPreviousGradient += rawCandidate * weightsHh[(2u * hiddenSize + h) * hiddenSize + hp];
            float rawReset = gatedPreviousGradient * previous * reset * (1.0f - reset), raw[3] = { rawReset, rawUpdate, rawCandidate };
            for (uint gate = 0; gate < 3u; ++gate) { uint row = gate * hiddenSize + h; gradBias[row] += raw[gate]; for (uint i = 0; i < inputSize; ++i) gradWeightsIh[row * inputSize + i] += raw[gate] * input[inputBase + b * inputSize + i]; for (uint hp = 0; hp < hiddenSize; ++hp) { float hiddenValue = gate == 2u ? reset * allH[previousBase + b * hiddenSize + hp] : allH[previousBase + b * hiddenSize + hp]; gradWeightsHh[row * hiddenSize + hp] += raw[gate] * hiddenValue; } }
            for (uint i = 0; i < inputSize; ++i) { float sum = 0.0f; for (uint gate = 0; gate < 3u; ++gate) sum += raw[gate] * weightsIh[(gate * hiddenSize + h) * inputSize + i]; gradInput[inputBase + b * inputSize + i] += sum; }
            float previousGradient = dHidden * update + gatedPreviousGradient * reset; for (uint gate = 0; gate < 2u; ++gate) for (uint hp = 0; hp < hiddenSize; ++hp) previousGradient += raw[gate] * weightsHh[(gate * hiddenSize + hp) * hiddenSize + h];
            if (t > 0u) nextH[state] = previousGradient; else gradHInit[state] = previousGradient;
        }
    }
}

kernel void gru_cell_backward(
    device const float* gradH [[buffer(0)]], device const float* reset [[buffer(1)]], device const float* update [[buffer(2)]], device const float* candidate [[buffer(3)]],
    device const float* previous [[buffer(4)]], device const float* weightsHh [[buffer(5)]], device float* gradPrevious [[buffer(6)]],
    device float* gradReset [[buffer(7)]], device float* gradUpdate [[buffer(8)]], device float* gradCandidate [[buffer(9)]],
    constant uint& batch [[buffer(10)]], constant uint& hiddenSize [[buffer(11)]], uint gid [[thread_position_in_grid]])
{
    if (gid >= batch * hiddenSize) return; uint h = gid % hiddenSize; float localGradient = gradH[gid], r = reset[gid], z = update[gid], n = candidate[gid], prior = previous[gid];
    float rawCandidate = localGradient * (1.0f - z) * (1.0f - n * n), rawUpdate = localGradient * (prior - n) * z * (1.0f - z), gated = 0.0f;
    for (uint hp = 0; hp < hiddenSize; ++hp) gated += rawCandidate * weightsHh[(2u * hiddenSize + h) * hiddenSize + hp];
    float rawReset = gated * prior * r * (1.0f - r), previousGradient = localGradient * z + gated * r;
    for (uint gate = 0; gate < 2u; ++gate) { float gateGradient = gate == 0u ? rawReset : rawUpdate; for (uint hp = 0; hp < hiddenSize; ++hp) previousGradient += gateGradient * weightsHh[(gate * hiddenSize + hp) * hiddenSize + h]; }
    gradPrevious[gid] = previousGradient; gradReset[gid] = rawReset; gradUpdate[gid] = rawUpdate; gradCandidate[gid] = rawCandidate;
}

inline float resident_loss_sign(float value)
{
    return value > 0.0f ? 1.0f : (value < 0.0f ? -1.0f : 0.0f);
}

kernel void loss_scalar_serial(
    device const float* predictions [[buffer(0)]], device const float* targets [[buffer(1)]], device float* result [[buffer(2)]],
    constant uint& count [[buffer(3)]], constant uint& divisor [[buffer(4)]], constant uint& operation [[buffer(5)]],
    constant uint& parameter0Bits [[buffer(6)]], constant uint& parameter1Bits [[buffer(7)]], uint gid [[thread_position_in_grid]])
{
    if (gid != 0u) return;
    float parameter0 = as_type<float>(parameter0Bits), parameter1 = as_type<float>(parameter1Bits), total = 0.0f;
    for (uint i = 0; i < count; ++i) {
        float prediction = predictions[i], target = targets[i], diff = prediction - target, value = 0.0f;
        if (operation == 0u) { float p = max(prediction, 1.0e-7f); value = -target * log(p); }
        else if (operation == 1u) { float p = max(min(prediction, 1.0f - 1.0e-7f), 1.0e-7f); value = -(target * log(p) + (1.0f - target) * log(1.0f - p)); }
        else if (operation == 2u) value = diff * diff;
        else if (operation == 3u) { float absolute = abs(diff); value = absolute < parameter0 ? 0.5f * absolute * absolute / parameter0 : absolute - 0.5f * parameter0; }
        else if (operation == 4u) { float p = max(min(prediction, 1.0f - 1.0e-7f), 1.0e-7f), pt = target * p + (1.0f - target) * (1.0f - p), alphaT = target * parameter0 + (1.0f - target) * (1.0f - parameter0); value = -alphaT * pow(1.0f - pt, parameter1) * log(pt); }
        else if (operation == 5u) value = abs(diff);
        else if (operation == 6u) value = log(cosh(diff));
        else if (operation == 7u) { float targetDiff = target - prediction; value = targetDiff >= 0.0f ? parameter0 * targetDiff : (parameter0 - 1.0f) * targetDiff; }
        else if (operation == 8u) value = max(0.0f, 1.0f - target * prediction);
        else if (operation == 9u) { float margin = max(0.0f, 1.0f - target * prediction); value = margin * margin; }
        else if (operation == 10u) { float p = max(prediction, 1.0e-7f); value = p - target * log(p); }
        else if (operation == 11u) value = exp(-target * prediction);
        else if (operation == 12u) { float yt = target * prediction; if (yt >= -1.0f) { float margin = max(0.0f, 1.0f - yt); value = margin * margin; } else value = -4.0f * yt; }
        else if (operation == 13u) { float p = max(prediction, 1.0e-7f); value = -target * log(p); }
        else if (operation == 14u) value = sqrt(diff * diff + parameter0 * parameter0);
        else if (operation == 15u) value = parameter0 * abs(diff) + parameter1 * diff * diff;
        total += value;
    }
    result[0] = divisor == 0u ? 0.0f : total / float(divisor);
}

kernel void loss_elementwise_backward(
    device const float* predictions [[buffer(0)]], device const float* targets [[buffer(1)]], device float* gradient [[buffer(2)]],
    constant uint& count [[buffer(3)]], constant uint& operation [[buffer(4)]], constant uint& parameter0Bits [[buffer(5)]],
    constant uint& parameter1Bits [[buffer(6)]], uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;
    float parameter0 = as_type<float>(parameter0Bits), parameter1 = as_type<float>(parameter1Bits);
    float prediction = predictions[gid], target = targets[gid], diff = prediction - target, value = 0.0f, invCount = 1.0f / float(count);
    if (operation == 1u) { float p = max(min(prediction, 1.0f - 1.0e-7f), 1.0e-7f); value = (p - target) / (p * (1.0f - p)) * invCount; }
    else if (operation == 3u) value = (abs(diff) < parameter0 ? diff / parameter0 : resident_loss_sign(diff)) * invCount;
    else if (operation == 4u) { float p = max(min(prediction, 1.0f - 1.0e-7f), 1.0e-7f), pt = target * p + (1.0f - target) * (1.0f - p), alphaT = target * parameter0 + (1.0f - target) * (1.0f - parameter0), factor = pow(1.0f - pt, parameter1); value = alphaT * factor * (parameter1 * pt * log(pt) + pt - 1.0f) * (2.0f * target - 1.0f) * invCount; }
    else if (operation == 5u) value = resident_loss_sign(diff) * invCount;
    else if (operation == 6u) value = tanh(diff) * invCount;
    else if (operation == 7u) value = ((target - prediction) >= 0.0f ? -parameter0 : (1.0f - parameter0)) * invCount;
    else if (operation == 8u) value = (1.0f - target * prediction) > 0.0f ? -target * invCount : 0.0f;
    else if (operation == 9u) { float margin = 1.0f - target * prediction; value = margin > 0.0f ? -2.0f * margin * target * invCount : 0.0f; }
    else if (operation == 10u) { float p = max(prediction, 1.0e-7f); value = (1.0f - target / p) * invCount; }
    else if (operation == 11u) value = -target * exp(-target * prediction) * invCount;
    else if (operation == 12u) { float yt = target * prediction; value = yt >= 1.0f ? 0.0f : (yt >= -1.0f ? -2.0f * (1.0f - yt) * target * invCount : -4.0f * target * invCount); }
    else if (operation == 13u) { float p = max(prediction, 1.0e-7f); value = -target / p * invCount; }
    else if (operation == 14u) value = diff / sqrt(diff * diff + parameter0 * parameter0) * invCount;
    else if (operation == 15u) value = (parameter0 * resident_loss_sign(diff) + 2.0f * parameter1 * diff) * invCount;
    gradient[gid] = value;
}

kernel void triplet_loss_serial(
    device const float* anchor [[buffer(0)]], device const float* positive [[buffer(1)]], device const float* negative [[buffer(2)]], device float* result [[buffer(3)]],
    constant uint& batch [[buffer(4)]], constant uint& embedding [[buffer(5)]], constant uint& marginBits [[buffer(6)]], uint gid [[thread_position_in_grid]])
{
    if (gid != 0u) return; float margin = as_type<float>(marginBits), total = 0.0f;
    for (uint b = 0; b < batch; ++b) { float positiveDistance = 0.0f, negativeDistance = 0.0f; for (uint d = 0; d < embedding; ++d) { uint i = b * embedding + d; float ap = anchor[i] - positive[i], an = anchor[i] - negative[i]; positiveDistance += ap * ap; negativeDistance += an * an; } total += max(0.0f, positiveDistance - negativeDistance + margin); }
    result[0] = batch == 0u ? 0.0f : total / float(batch);
}

kernel void triplet_loss_backward(
    device const float* anchor [[buffer(0)]], device const float* positive [[buffer(1)]], device const float* negative [[buffer(2)]],
    device float* gradAnchor [[buffer(3)]], device float* gradPositive [[buffer(4)]], device float* gradNegative [[buffer(5)]],
    constant uint& batch [[buffer(6)]], constant uint& embedding [[buffer(7)]], constant uint& marginBits [[buffer(8)]], uint gid [[thread_position_in_grid]])
{
    if (gid >= batch) return; float positiveDistance = 0.0f, negativeDistance = 0.0f, margin = as_type<float>(marginBits);
    for (uint d = 0; d < embedding; ++d) { uint i = gid * embedding + d; float ap = anchor[i] - positive[i], an = anchor[i] - negative[i]; positiveDistance += ap * ap; negativeDistance += an * an; }
    bool active = positiveDistance - negativeDistance + margin > 0.0f;
    for (uint d = 0; d < embedding; ++d) { uint i = gid * embedding + d; if (active) { float ap = anchor[i] - positive[i], an = anchor[i] - negative[i]; gradAnchor[i] = 2.0f * (ap - an) / float(batch); gradPositive[i] = -2.0f * ap / float(batch); gradNegative[i] = 2.0f * an / float(batch); } else { gradAnchor[i] = 0.0f; gradPositive[i] = 0.0f; gradNegative[i] = 0.0f; } }
}

kernel void contrastive_loss_serial(
    device const float* output1 [[buffer(0)]], device const float* output2 [[buffer(1)]], device const float* labels [[buffer(2)]], device float* result [[buffer(3)]],
    constant uint& batch [[buffer(4)]], constant uint& embedding [[buffer(5)]], constant uint& marginBits [[buffer(6)]], uint gid [[thread_position_in_grid]])
{
    if (gid != 0u) return; float margin = as_type<float>(marginBits), total = 0.0f;
    for (uint b = 0; b < batch; ++b) { float distance = 0.0f; for (uint d = 0; d < embedding; ++d) { float diff = output1[b * embedding + d] - output2[b * embedding + d]; distance += diff * diff; } distance = sqrt(distance); if (labels[b] == 0.0f) total += distance * distance; else { float marginDistance = max(0.0f, margin - distance); total += marginDistance * marginDistance; } }
    result[0] = batch == 0u ? 0.0f : total / float(batch);
}

kernel void contrastive_loss_backward(
    device const float* output1 [[buffer(0)]], device const float* output2 [[buffer(1)]], device const float* labels [[buffer(2)]],
    device float* gradOutput1 [[buffer(3)]], device float* gradOutput2 [[buffer(4)]], constant uint& batch [[buffer(5)]],
    constant uint& embedding [[buffer(6)]], constant uint& marginBits [[buffer(7)]], uint gid [[thread_position_in_grid]])
{
    if (gid >= batch) return; float distance = 0.0f, margin = as_type<float>(marginBits);
    for (uint d = 0; d < embedding; ++d) { float diff = output1[gid * embedding + d] - output2[gid * embedding + d]; distance += diff * diff; } distance = sqrt(distance + 1.0e-7f);
    for (uint d = 0; d < embedding; ++d) { uint i = gid * embedding + d; float diff = output1[i] - output2[i], first = 0.0f; if (labels[gid] == 0.0f) first = 2.0f * diff / float(batch); else if (margin - distance > 0.0f) first = -2.0f * (margin - distance) * diff / distance / float(batch); gradOutput1[i] = first; gradOutput2[i] = -first; }
}

kernel void loss_backward_with_scalar_gradient(
    device const float* gradOutput [[buffer(0)]], device const float* predictions [[buffer(1)]], device const float* targets [[buffer(2)]], device float* gradInput [[buffer(3)]],
    constant uint& count [[buffer(4)]], constant uint& invNBits [[buffer(5)]], constant uint& parameterBits [[buffer(6)]], constant uint& operation [[buffer(7)]], uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return; float diff = predictions[gid] - targets[gid], scale = gradOutput[0] * as_type<float>(invNBits);
    if (operation == 0u) gradInput[gid] = scale * 2.0f * diff;
    else { float delta = as_type<float>(parameterBits), absolute = abs(diff); gradInput[gid] = absolute <= delta ? scale * diff : scale * delta * (diff > 0.0f ? 1.0f : -1.0f); }
}

inline ulong resident_random_mix(ulong value)
{
    value ^= value >> 30; value *= 0xbf58476d1ce4e5b9ul;
    value ^= value >> 27; value *= 0x94d049bb133111ebul;
    return value ^ (value >> 31);
}

inline float resident_random_uniform01(ulong state)
{
    return float(resident_random_mix(state) >> 40) * (1.0f / 16777216.0f);
}

kernel void random_uniform_resident(
    device float* output [[buffer(0)]], constant uint& count [[buffer(1)]], constant uint& minimumBits [[buffer(2)]],
    constant uint& maximumBits [[buffer(3)]], constant uint& seedLow [[buffer(4)]], constant uint& seedHigh [[buffer(5)]], uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;
    uint seed = seedLow ^ seedHigh;
    uint state = gid * 747796405u + seed + 2891336453u;
    uint word = ((state >> ((state >> 28) + 4u)) ^ state) * 277803737u;
    uint sample = (word >> 22) ^ word;
    float minimum = as_type<float>(minimumBits), maximum = as_type<float>(maximumBits);
    float uniform = float(sample >> 8) * (1.0f / 16777216.0f);
    output[gid] = fma(uniform, maximum - minimum, minimum);
}

kernel void stateless_dropout_mask(
    device float* output [[buffer(0)]], constant uint& count [[buffer(1)]], constant uint& threshold [[buffer(2)]],
    constant uint& scaleBits [[buffer(3)]], constant uint& seed [[buffer(4)]], uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;
    uint state = gid * 747796405u + seed + 2891336453u;
    uint word = ((state >> ((state >> 28) + 4u)) ^ state) * 277803737u;
    uint sample = (word >> 22) ^ word;
    output[gid] = sample < threshold ? 0.0f : as_type<float>(scaleBits);
}

kernel void random_normal_resident(
    device float* output [[buffer(0)]], constant uint& count [[buffer(1)]], constant uint& meanBits [[buffer(2)]],
    constant uint& stdDevBits [[buffer(3)]], constant uint& seedLow [[buffer(4)]], constant uint& seedHigh [[buffer(5)]], uint gid [[thread_position_in_grid]])
{
    uint first = gid * 2u; if (first >= count) return; ulong seed = (ulong(seedHigh) << 32) | ulong(seedLow), state = seed + ulong(first) * 0x9e3779b97f4a7c15ul;
    float u1 = max(resident_random_uniform01(state), 1.0e-10f), u2 = resident_random_uniform01(state + 0x9e3779b97f4a7c15ul);
    float radius = sqrt(-2.0f * log(u1)), angle = 6.28318530717958647692f * u2, mean = as_type<float>(meanBits), stdDev = as_type<float>(stdDevBits);
    output[first] = mean + stdDev * radius * cos(angle); if (first + 1u < count) output[first + 1u] = mean + stdDev * radius * sin(angle);
}
";
}
