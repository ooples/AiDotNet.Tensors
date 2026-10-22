// Copyright (c) AiDotNet. All rights reserved.
// WebGPU JavaScript interop for Blazor WebAssembly tensor operations.
// Production-ready implementation with proper resource management.

"use strict";

/**
 * WebGPU Tensor Engine - Browser-side GPU compute implementation.
 *
 * Architecture:
 * - Uses resource pools for buffer/pipeline reuse
 * - Proper error handling with descriptive messages
 * - Automatic cleanup and memory management
 * - Staging buffer pattern for CPU-GPU data transfer
 */

// Global state
let gpu = null;
let adapter = null;
let device = null;
let queue = null;

// Resource pools with auto-incrementing IDs
let nextBufferId = 0;
let nextShaderModuleId = 0;
let nextPipelineId = 0;
let nextBindGroupId = 0;
let nextTimestampQueryId = 0;

const buffers = new Map();
const shaderModules = new Map();
const shaderModuleUniformBindings = new Map();   // module id -> binding indices it declares a uniform buffer at
const pipelineUniformBindings = new WeakMap();   // pipeline -> { declared, resolved: storage count -> uniform binding }
const pipelines = new Map();
const bindGroups = new Map();
const bindGroupLayouts = new Map();
const timestampQueries = new Map();

// Staging buffer pool for read operations
const stagingBufferPool = [];
const MAX_STAGING_POOL_SIZE = 8;

/**
 * Checks if WebGPU is supported in the current browser.
 * @returns {boolean} True if WebGPU is available.
 */
export function isWebGpuSupported() {
    return typeof navigator !== 'undefined' && 'gpu' in navigator;
}

/**
 * Initializes WebGPU by requesting adapter and device.
 * @returns {Promise<boolean>} True if initialization succeeded.
 */
export async function initializeWebGpu() {
    if (!isWebGpuSupported()) {
        console.error('WebGPU is not supported in this browser');
        return false;
    }

    try {
        gpu = navigator.gpu;

        // Request high-performance adapter
        adapter = await gpu.requestAdapter({
            powerPreference: 'high-performance'
        });

        if (!adapter) {
            console.error('Failed to get WebGPU adapter');
            return false;
        }

        // Request device with maximum limits
        const requiredFeatures = [];
        const requiredLimits = {
            maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
            maxBufferSize: adapter.limits.maxBufferSize,
            maxComputeWorkgroupSizeX: adapter.limits.maxComputeWorkgroupSizeX,
            maxComputeWorkgroupSizeY: adapter.limits.maxComputeWorkgroupSizeY,
            maxComputeWorkgroupSizeZ: adapter.limits.maxComputeWorkgroupSizeZ,
            maxComputeInvocationsPerWorkgroup: adapter.limits.maxComputeInvocationsPerWorkgroup,
            maxComputeWorkgroupsPerDimension: adapter.limits.maxComputeWorkgroupsPerDimension,
            maxBindGroups: adapter.limits.maxBindGroups,
            maxStorageBuffersPerShaderStage: adapter.limits.maxStorageBuffersPerShaderStage
        };

        // Check for timestamp query support
        if (adapter.features.has('timestamp-query')) {
            requiredFeatures.push('timestamp-query');
        }

        device = await adapter.requestDevice({
            requiredFeatures,
            requiredLimits
        });

        if (!device) {
            console.error('Failed to get WebGPU device');
            return false;
        }

        queue = device.queue;

        // Set up error handler
        device.addEventListener('uncapturederror', (event) => {
            console.error('WebGPU uncaptured error:', event.error.message);
        });

        // Handle device loss
        device.lost.then((info) => {
            console.error(`WebGPU device lost: ${info.reason} - ${info.message}`);
            device = null;
            queue = null;
        });

        console.log('WebGPU initialized successfully');
        return true;
    } catch (error) {
        console.error('WebGPU initialization failed:', error);
        return false;
    }
}

/**
 * Gets adapter information string.
 * @returns {string} Adapter description.
 */
export function getAdapterInfo() {
    if (!adapter) return '';

    const info = adapter.info || {};
    return `${info.vendor || 'Unknown'} ${info.architecture || ''} (${info.device || 'Unknown Device'})`.trim();
}

/**
 * Gets device limits as JSON.
 * @returns {string} JSON string of device limits.
 */
export function getDeviceLimitsJson() {
    if (!device) return '{}';

    const limits = device.limits;
    return JSON.stringify({
        maxBufferSize: limits.maxBufferSize,
        maxStorageBufferBindingSize: limits.maxStorageBufferBindingSize,
        maxUniformBufferBindingSize: limits.maxUniformBufferBindingSize,
        maxComputeWorkgroupSizeX: limits.maxComputeWorkgroupSizeX,
        maxComputeWorkgroupSizeY: limits.maxComputeWorkgroupSizeY,
        maxComputeWorkgroupSizeZ: limits.maxComputeWorkgroupSizeZ,
        maxComputeInvocationsPerWorkgroup: limits.maxComputeInvocationsPerWorkgroup,
        maxComputeWorkgroupsPerDimension: limits.maxComputeWorkgroupsPerDimension,
        maxBindGroups: limits.maxBindGroups,
        maxBindingsPerBindGroup: limits.maxBindingsPerBindGroup,
        maxStorageBuffersPerShaderStage: limits.maxStorageBuffersPerShaderStage
    });
}

/**
 * Destroys the WebGPU device and releases all resources.
 */
export function destroyDevice() {
    // Clean up all resources
    for (const [id, buffer] of buffers) {
        try { buffer.destroy(); } catch (e) { }
    }
    buffers.clear();

    for (const buffer of stagingBufferPool) {
        try { buffer.destroy(); } catch (e) { }
    }
    stagingBufferPool.length = 0;

    shaderModules.clear();
    pipelines.clear();
    bindGroups.clear();
    bindGroupLayouts.clear();
    timestampQueries.clear();

    if (device) {
        device.destroy();
        device = null;
        queue = null;
    }

    adapter = null;
    gpu = null;

    // Reset IDs
    nextBufferId = 0;
    nextShaderModuleId = 0;
    nextPipelineId = 0;
    nextBindGroupId = 0;
    nextTimestampQueryId = 0;
}

// ============================================================================
// Buffer Operations
// ============================================================================

/**
 * Creates a GPU buffer.
 * @param {number} sizeBytes - Size in bytes.
 * @param {number} usage - Buffer usage flags.
 * @returns {number} Buffer ID, or -1 on failure.
 */
export function createBuffer(sizeBytes, usage) {
    if (!device) return -1;

    try {
        // Ensure 4-byte alignment
        const alignedSize = Math.max(4, (sizeBytes + 3) & ~3);

        const buffer = device.createBuffer({
            size: alignedSize,
            usage: usage,
            mappedAtCreation: false
        });

        const id = nextBufferId++;
        buffers.set(id, buffer);
        return id;
    } catch (error) {
        console.error('Failed to create buffer:', error);
        return -1;
    }
}

/**
 * Destroys a GPU buffer.
 * @param {number} bufferId - Buffer ID.
 */
export function destroyBuffer(bufferId) {
    const buffer = buffers.get(bufferId);
    if (buffer) {
        try { buffer.destroy(); } catch (e) { }
        buffers.delete(bufferId);
    }
}

/**
 * Writes data to a GPU buffer.
 * @param {number} bufferId - Buffer ID.
 * @param {Float32Array|number[]} data - Data to write.
 * @param {number} offsetBytes - Offset in bytes.
 */
export function writeBuffer(bufferId, data, offsetBytes) {
    if (!device || !queue) return;

    const buffer = buffers.get(bufferId);
    if (!buffer) {
        console.error(`Buffer ${bufferId} not found`);
        return;
    }

    const floatData = data instanceof Float32Array ? data : new Float32Array(data);
    queue.writeBuffer(buffer, offsetBytes, floatData);
}

/**
 * Writes raw byte data to a GPU buffer.
 * @param {number} bufferId - Buffer ID.
 * @param {Uint8Array|number[]} data - Byte data to write.
 * @param {number} offsetBytes - Offset in bytes.
 */
export function writeByteBuffer(bufferId, data, offsetBytes) {
    if (!device || !queue) return;

    const buffer = buffers.get(bufferId);
    if (!buffer) {
        console.error(`Buffer ${bufferId} not found`);
        return;
    }

    const source = data instanceof Uint8Array ? data : new Uint8Array(data);
    const alignedLength = (source.byteLength + 3) & ~3;
    const byteData = source.byteLength === alignedLength ? source : new Uint8Array(alignedLength);
    if (byteData !== source) {
        byteData.set(source);
    }
    queue.writeBuffer(buffer, offsetBytes, byteData);
}

/**
 * Reads data from a GPU buffer.
 * @param {number} bufferId - Buffer ID.
 * @param {number} sizeBytes - Size to read.
 * @param {number} offsetBytes - Offset in bytes.
 * @returns {Promise<Float32Array>} Read data.
 */
export async function readBuffer(bufferId, sizeBytes, offsetBytes) {
    if (!device) return new Float32Array(0);

    const buffer = buffers.get(bufferId);
    if (!buffer) {
        console.error(`Buffer ${bufferId} not found`);
        return new Float32Array(0);
    }

    // Get or create staging buffer
    const alignedSize = (sizeBytes + 3) & ~3;
    let stagingBuffer = getStagingBuffer(alignedSize);

    if (!stagingBuffer) {
        stagingBuffer = device.createBuffer({
            size: alignedSize,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
            mappedAtCreation: false
        });
    }

    // Copy to staging buffer
    const commandEncoder = device.createCommandEncoder();
    commandEncoder.copyBufferToBuffer(buffer, offsetBytes, stagingBuffer, 0, alignedSize);
    queue.submit([commandEncoder.finish()]);

    // Map and read
    await stagingBuffer.mapAsync(GPUMapMode.READ);
    // A pooled staging buffer can be larger than this read: map and return only the requested range.
    const arrayBuffer = stagingBuffer.getMappedRange(0, alignedSize);
    const result = new Float32Array(arrayBuffer.slice(0, alignedSize));
    stagingBuffer.unmap();

    // Return staging buffer to pool
    returnStagingBuffer(stagingBuffer, alignedSize);

    return result;
}

/**
 * Reads data from a GPU buffer and returns as JSON string.
 * This is needed because C# JSImport doesn't support Task<T[]> return types.
 * @param {number} bufferId - Buffer ID.
 * @param {number} sizeBytes - Size to read in bytes.
 * @param {number} offsetBytes - Offset in bytes.
 * @returns {Promise<string>} JSON string of float array.
 */
export async function readBufferAsJson(bufferId, sizeBytes, offsetBytes) {
    const result = await readBuffer(bufferId, sizeBytes, offsetBytes);
    if (!result || result.length === 0) {
        return '[]';
    }
    // JSON has no NaN or Infinity (JSON.stringify writes null for them) and writes -0 as 0, so those travel as the
    // strings "NaN", "Infinity", "-Infinity" and "-0", which the C# side reads back as numbers.
    return JSON.stringify(Array.from(result, v => Number.isFinite(v) ? (Object.is(v, -0) ? '-0' : v) : String(v)));
}

/**
 * Reads raw byte data from a GPU buffer and returns as JSON string.
 * @param {number} bufferId - Buffer ID.
 * @param {number} sizeBytes - Size to read in bytes.
 * @param {number} offsetBytes - Offset in bytes.
 * @returns {Promise<string>} JSON string of byte values.
 */
export async function readByteBufferAsJson(bufferId, sizeBytes, offsetBytes) {
    if (!device) return '[]';

    const buffer = buffers.get(bufferId);
    if (!buffer) {
        console.error(`Buffer ${bufferId} not found`);
        return '[]';
    }

    const alignedSize = (sizeBytes + 3) & ~3;
    let stagingBuffer = getStagingBuffer(alignedSize);

    if (!stagingBuffer) {
        stagingBuffer = device.createBuffer({
            size: alignedSize,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
            mappedAtCreation: false
        });
    }

    const commandEncoder = device.createCommandEncoder();
    commandEncoder.copyBufferToBuffer(buffer, offsetBytes, stagingBuffer, 0, alignedSize);
    queue.submit([commandEncoder.finish()]);

    await stagingBuffer.mapAsync(GPUMapMode.READ);
    const arrayBuffer = stagingBuffer.getMappedRange(0, alignedSize);
    const result = new Uint8Array(arrayBuffer.slice(0, sizeBytes));
    stagingBuffer.unmap();

    returnStagingBuffer(stagingBuffer, alignedSize);

    return JSON.stringify(Array.from(result));
}

// ----------------------------------------------------------------------------
// Bit-exact transfers. Buffer contents cross as base64 text: a number array cannot carry every bit pattern (an int
// stored through a float view is a NaN whenever it is negative, and a NaN's payload does not survive a JS number).
// ----------------------------------------------------------------------------

function base64ToBytes(base64, paddedLength) {
    const binary = atob(base64);
    const bytes = new Uint8Array(paddedLength === undefined ? binary.length : paddedLength);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    return bytes;
}

function bytesToBase64(bytes) {
    let binary = '';
    const chunk = 0x8000;
    for (let i = 0; i < bytes.length; i += chunk) {
        binary += String.fromCharCode.apply(null, bytes.subarray(i, i + chunk));
    }
    return btoa(binary);
}

/**
 * Writes bytes, given as base64, to a GPU buffer. A length that is not a multiple of four is padded with zeros.
 * @param {number} bufferId - Buffer handle ID.
 * @param {string} base64 - The bytes to write.
 * @param {number} offsetBytes - Offset in bytes.
 */
export function writeBufferBase64(bufferId, base64, offsetBytes) {
    if (!device || !queue) return;

    const buffer = buffers.get(bufferId);
    if (!buffer) {
        console.error(`Buffer ${bufferId} not found`);
        return;
    }

    const length = atob(base64).length;
    const bytes = base64ToBytes(base64, (length + 3) & ~3);
    queue.writeBuffer(buffer, offsetBytes, bytes);
}

/**
 * Reads bytes from a GPU buffer as base64.
 * @param {number} bufferId - Buffer handle ID.
 * @param {number} sizeBytes - Size to read in bytes.
 * @param {number} offsetBytes - Offset in bytes.
 * @returns {Promise<string>} The bytes, base64 encoded.
 */
export async function readBufferBase64(bufferId, sizeBytes, offsetBytes) {
    if (!device) return '';

    const buffer = buffers.get(bufferId);
    if (!buffer) {
        console.error(`Buffer ${bufferId} not found`);
        return '';
    }

    const alignedSize = (sizeBytes + 3) & ~3;
    let stagingBuffer = getStagingBuffer(alignedSize);
    if (!stagingBuffer) {
        stagingBuffer = device.createBuffer({
            size: alignedSize,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
            mappedAtCreation: false
        });
    }

    const commandEncoder = device.createCommandEncoder();
    commandEncoder.copyBufferToBuffer(buffer, offsetBytes, stagingBuffer, 0, alignedSize);
    queue.submit([commandEncoder.finish()]);

    await stagingBuffer.mapAsync(GPUMapMode.READ, 0, alignedSize);
    const result = new Uint8Array(stagingBuffer.getMappedRange(0, alignedSize).slice(0, sizeBytes));
    stagingBuffer.unmap();
    returnStagingBuffer(stagingBuffer, alignedSize);

    return bytesToBase64(result);
}

// Readbacks started by beginReadback and not yet finished: token -> { staging, alignedSize, sizeBytes, mapped }.
const pendingReadbacks = new Map();
let nextReadbackToken = 1;

/**
 * Starts a device-to-host copy without waiting for it: the copy is submitted behind the work already queued and the
 * staging buffer's mapAsync is started, so finishReadbackBase64 waits only for this copy, not for work queued after it.
 * @param {number} bufferId - Source buffer ID.
 * @param {number} sizeBytes - Bytes to read.
 * @param {number} offsetBytes - Offset in bytes.
 * @returns {number} A token for finishReadbackBase64, or 0 when nothing was started.
 */
export function beginReadback(bufferId, sizeBytes, offsetBytes) {
    if (!device) return 0;
    const buffer = buffers.get(bufferId);
    if (!buffer) {
        console.error(`Buffer ${bufferId} not found`);
        return 0;
    }
    const alignedSize = (sizeBytes + 3) & ~3;
    // Its own staging buffer: a pooled one could be handed to another read while this one is still mapping.
    const staging = device.createBuffer({
        size: alignedSize,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        mappedAtCreation: false
    });
    const commandEncoder = device.createCommandEncoder();
    commandEncoder.copyBufferToBuffer(buffer, offsetBytes, staging, 0, alignedSize);
    queue.submit([commandEncoder.finish()]);
    const token = nextReadbackToken++;
    pendingReadbacks.set(token, { staging, alignedSize, sizeBytes, mapped: staging.mapAsync(GPUMapMode.READ, 0, alignedSize) });
    return token;
}

/**
 * Waits for a readback beginReadback started and returns its bytes, base64 encoded; the token is then spent.
 * @param {number} token - The token beginReadback returned.
 * @returns {Promise<string>} The bytes, base64 encoded ('' for an unknown token).
 */
export async function finishReadbackBase64(token) {
    const pending = pendingReadbacks.get(token);
    if (!pending) return '';
    pendingReadbacks.delete(token);
    try {
        await pending.mapped;
        const result = new Uint8Array(pending.staging.getMappedRange(0, pending.alignedSize).slice(0, pending.sizeBytes));
        pending.staging.unmap();
        return bytesToBase64(result);
    } finally {
        pending.staging.destroy();
    }
}

/**
 * Copies data between GPU buffers.
 * @param {number} srcBufferId - Source buffer ID.
 * @param {number} srcOffset - Source offset in bytes.
 * @param {number} dstBufferId - Destination buffer ID.
 * @param {number} dstOffset - Destination offset in bytes.
 * @param {number} sizeBytes - Size to copy in bytes.
 */
export function copyBufferToBuffer(srcBufferId, srcOffset, dstBufferId, dstOffset, sizeBytes) {
    if (!device) return;

    const srcBuffer = buffers.get(srcBufferId);
    const dstBuffer = buffers.get(dstBufferId);

    if (!srcBuffer || !dstBuffer) {
        console.error('Source or destination buffer not found');
        return;
    }

    const commandEncoder = device.createCommandEncoder();
    commandEncoder.copyBufferToBuffer(srcBuffer, srcOffset, dstBuffer, dstOffset, sizeBytes);
    queue.submit([commandEncoder.finish()]);
}

// Staging buffer pool management
function getStagingBuffer(size) {
    for (let i = 0; i < stagingBufferPool.length; i++) {
        if (stagingBufferPool[i].size >= size) {
            return stagingBufferPool.splice(i, 1)[0];
        }
    }
    return null;
}

function returnStagingBuffer(buffer, size) {
    if (stagingBufferPool.length < MAX_STAGING_POOL_SIZE) {
        stagingBufferPool.push(buffer);
    } else {
        buffer.destroy();
    }
}

// ============================================================================
// Shader Operations
// ============================================================================

/**
 * Creates a shader module from WGSL source.
 * @param {string} wgslSource - WGSL shader source code.
 * @returns {number} Shader module ID, or -1 on failure.
 */
export function createShaderModule(wgslSource) {
    if (!device) return -1;

    try {
        const shaderModule = device.createShaderModule({
            code: wgslSource
        });

        const id = nextShaderModuleId++;
        shaderModules.set(id, shaderModule);
        // Where this module declares uniform buffers: see resolveUniformBindGroup.
        const declared = new Set();
        for (const match of wgslSource.matchAll(/@binding\((\d+)\)\s*var\s*<\s*uniform\s*>/g)) {
            declared.add(Number(match[1]));
        }
        shaderModuleUniformBindings.set(id, Array.from(declared).sort((a, b) => a - b));
        return id;
    } catch (error) {
        console.error('Failed to create shader module:', error);
        return -1;
    }
}

/**
 * Creates a compute pipeline.
 * @param {number} shaderModuleId - Shader module ID.
 * @param {string} entryPoint - Entry point function name.
 * @returns {Promise<number>} Pipeline ID, or -1 on failure.
 */
export async function createComputePipeline(shaderModuleId, entryPoint) {
    if (!device) return -1;

    const shaderModule = shaderModules.get(shaderModuleId);
    if (!shaderModule) {
        console.error(`Shader module ${shaderModuleId} not found`);
        return -1;
    }

    try {
        const pipeline = await device.createComputePipelineAsync({
            layout: 'auto',
            compute: {
                module: shaderModule,
                entryPoint: entryPoint
            }
        });

        const id = nextPipelineId++;
        pipelines.set(id, pipeline);
        pipelineUniformBindings.set(pipeline, {
            declared: shaderModuleUniformBindings.get(shaderModuleId) || [],
            resolved: new Map()
        });
        return id;
    } catch (error) {
        console.error(`Failed to create compute pipeline for ${entryPoint}:`, error);
        return -1;
    }
}

/**
 * Destroys a shader module.
 * @param {number} shaderModuleId - Shader module ID.
 */
export function destroyShaderModule(shaderModuleId) {
    shaderModules.delete(shaderModuleId);
}

/**
 * Destroys a compute pipeline.
 * @param {number} pipelineId - Pipeline ID.
 */
export function destroyPipeline(pipelineId) {
    pipelines.delete(pipelineId);
}

// ============================================================================
// Compute Dispatch
// ============================================================================

/**
 * Creates a bind group for compute dispatch.
 * @param {number} pipelineId - Pipeline ID.
 * @param {number[]} bufferIds - Array of buffer IDs.
 * @returns {number} Bind group ID, or -1 on failure.
 */
export function createBindGroup(pipelineId, bufferIds) {
    if (!device) return -1;

    const pipeline = pipelines.get(pipelineId);
    if (!pipeline) {
        console.error(`Pipeline ${pipelineId} not found`);
        return -1;
    }

    try {
        const entries = bufferIds.map((bufferId, index) => {
            const buffer = buffers.get(bufferId);
            if (!buffer) {
                throw new Error(`Buffer ${bufferId} not found`);
            }
            return {
                binding: index,
                resource: { buffer: buffer }
            };
        });

        const id = nextBindGroupId++;
        bindGroups.set(id, { entries: entries, resolved: null, resolvedUniform: null });
        return id;
    } catch (error) {
        console.error('Failed to create bind group:', error);
        return -1;
    }
}

/**
 * Creates a bind group that binds a byte range of each buffer, so a buffer view addresses only its own
 * elements (#1142). Offsets must meet the device's minStorageBufferOffsetAlignment.
 * @param {number} pipelineId - Pipeline ID.
 * @param {number[]} bufferIds - Buffer IDs, one per binding in order.
 * @param {number[]} offsets - Byte offset of each binding.
 * @param {number[]} sizes - Byte size of each binding.
 * @returns {number} Bind group ID, or -1 on failure.
 */
export function createBindGroupRanges(pipelineId, bufferIds, offsets, sizes) {
    if (!device) return -1;

    const pipeline = pipelines.get(pipelineId);
    if (!pipeline) {
        console.error(`Pipeline ${pipelineId} not found`);
        return -1;
    }

    try {
        const entries = bufferIds.map((bufferId, index) => {
            const buffer = buffers.get(bufferId);
            if (!buffer) {
                throw new Error(`Buffer ${bufferId} not found`);
            }
            return {
                binding: index,
                resource: { buffer: buffer, offset: offsets[index], size: sizes[index] }
            };
        });

        const id = nextBindGroupId++;
        bindGroups.set(id, { entries: entries, resolved: null, resolvedUniform: null });
        return id;
    } catch (error) {
        console.error('Failed to create ranged bind group:', error);
        return -1;
    }
}
/**
 * Destroys a bind group.
 * @param {number} bindGroupId - Bind group ID.
 */
/**
 * The GPU bind group for a recorded set of entries. createBindGroup only records the storage buffers: a kernel's
 * parameters live in a uniform buffer bound after them (binding index = storage buffer count), and that buffer is
 * only known at dispatch. A bind group has to match the pipeline's layout exactly, so it cannot be created without it.
 */
function resolveBindGroup(record, pipeline) {
    // Keyed on the pipeline too: an automatic layout belongs to one pipeline, and a bind group may be dispatched with
    // its kernel's folded copy (dispatchComputeFolded).
    if (!record.resolved || record.resolvedUniform || record.resolvedPipeline !== pipeline) {
        record.resolved = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: record.entries });
        record.resolvedUniform = null;
        record.resolvedPipeline = pipeline;
    }
    return record.resolved;
}

/**
 * The bind group for a dispatch with a uniform buffer.
 *
 * The uniform buffer's binding index is the shader's choice. Usually it is the one after the storage buffers, but a
 * module shared by several kernels declares it once, after the most storage buffers any of them takes, and a kernel
 * that takes fewer still reads it from there (four storage declarations and a uniform at binding 4, used by an entry
 * point that binds three buffers). A pipeline's layout cannot be read back, so the candidates are tried in order
 * inside a validation error scope: the index after the storage buffers, then each index the module declares a uniform
 * at. The one that validates is remembered per pipeline and storage count, so this costs one probe per kernel.
 */
async function resolveUniformBindGroup(record, pipeline, uniformBuffer) {
    if (record.resolved && record.resolvedUniform === uniformBuffer && record.resolvedPipeline === pipeline) return record.resolved;

    const info = pipelineUniformBindings.get(pipeline) || { declared: [], resolved: new Map() };
    const storageCount = record.entries.length;
    const layout = pipeline.getBindGroupLayout(0);
    // A choice is the uniform's binding index and the storage entries to leave out. An automatic layout holds only
    // the bindings the entry point uses, so a buffer the caller binds for a kernel that does not touch it has to be
    // left out (the kernel then cannot have written it, which the caller's results will show).
    const build = (choice) => device.createBindGroup({
        layout: layout,
        entries: record.entries.filter((_, index) => !choice.omit.includes(index))
            .concat([{ binding: choice.uniform, resource: { buffer: uniformBuffer } }])
    });

    let group = null;
    if (info.resolved.has(storageCount)) {
        group = build(info.resolved.get(storageCount));
    } else {
        const uniforms = [storageCount].concat(info.declared.filter(b => b !== storageCount));
        const omissions = [[]];
        for (let i = 0; i < storageCount; i++) omissions.push([i]);
        for (let i = 0; i < storageCount; i++) for (let j = i + 1; j < storageCount; j++) omissions.push([i, j]);
        let firstError = null;
        search:
        for (const omit of omissions) {
            for (const uniform of uniforms) {
                const choice = { uniform: uniform, omit: omit };
                device.pushErrorScope('validation');
                const attempt = build(choice);
                const error = await device.popErrorScope();
                if (!error) {
                    info.resolved.set(storageCount, choice);
                    group = attempt;
                    break search;
                }
                if (!firstError) firstError = error;
            }
        }
        if (!group) {
            console.error(`No bind group layout matches ${storageCount} storage buffers and a uniform buffer ` +
                `(uniform bindings tried: ${uniforms.join(', ')}): ${firstError ? firstError.message : 'unknown error'}`);
            return null;
        }
    }
    record.resolved = group;
    record.resolvedUniform = uniformBuffer;
    record.resolvedPipeline = pipeline;
    return group;
}

export function destroyBindGroup(bindGroupId) {
    bindGroups.delete(bindGroupId);
}

/**
 * Dispatches a compute operation.
 * @param {number} pipelineId - Pipeline ID.
 * @param {number} bindGroupId - Bind group ID.
 * @param {number} workgroupsX - Workgroups in X dimension.
 * @param {number} workgroupsY - Workgroups in Y dimension.
 * @param {number} workgroupsZ - Workgroups in Z dimension.
 */
export async function dispatchCompute(pipelineId, bindGroupId, workgroupsX, workgroupsY, workgroupsZ) {
    if (!device) return;

    const pipeline = pipelines.get(pipelineId);
    const bindGroup = bindGroups.get(bindGroupId);

    if (!pipeline || !bindGroup) {
        console.error('Pipeline or bind group not found');
        return;
    }

    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();

    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, resolveBindGroup(bindGroup, pipeline));
    passEncoder.dispatchWorkgroups(workgroupsX, workgroupsY, workgroupsZ);
    passEncoder.end();

    queue.submit([commandEncoder.finish()]);
}

/**
 * Dispatches a compute operation with uniforms in a separate bind group.
 * @param {number} pipelineId - Pipeline ID.
 * @param {number} bindGroupId - Bind group ID for storage buffers.
 * @param {number} uniformBufferId - Uniform buffer ID.
 * @param {number} workgroupsX - Workgroups in X dimension.
 * @param {number} workgroupsY - Workgroups in Y dimension.
 * @param {number} workgroupsZ - Workgroups in Z dimension.
 */
export async function dispatchComputeWithUniforms(pipelineId, bindGroupId, uniformBufferId, workgroupsX, workgroupsY, workgroupsZ) {
    if (!device) return;

    const pipeline = pipelines.get(pipelineId);
    const bindGroup = bindGroups.get(bindGroupId);
    const uniformBuffer = buffers.get(uniformBufferId);

    if (!pipeline || !bindGroup) {
        console.error('Pipeline or bind group not found');
        return;
    }
    if (!uniformBuffer) {
        console.error(`Uniform buffer ${uniformBufferId} not found`);
        return;
    }

    const resolved = await resolveUniformBindGroup(bindGroup, pipeline, uniformBuffer);
    if (!resolved) return;

    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();

    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, resolved);
    passEncoder.dispatchWorkgroups(workgroupsX, workgroupsY, workgroupsZ);
    passEncoder.end();

    queue.submit([commandEncoder.finish()]);
}

/**
 * Dispatches a kernel's folded copy (WebGpuDispatchFold in the C# backend) on a 2-D grid: a 1-D dispatch of more
 * workgroups than one dimension holds. The folded entry point reads the true workgroup count from a uniform at group 1.
 * @param {number} pipelineId - The folded pipeline's ID.
 * @param {number} bindGroupId - Storage bind group ID (group 0).
 * @param {number} uniformBufferId - The kernel's uniform buffer ID, or -1 for none.
 * @param {number} groupsBufferId - Uniform buffer holding the true workgroup count (group 1, binding 0).
 * @param {number} workgroupsX - Workgroups in X.
 * @param {number} workgroupsY - Workgroups in Y.
 */
export async function dispatchComputeFolded(pipelineId, bindGroupId, uniformBufferId, groupsBufferId, workgroupsX, workgroupsY) {
    if (!device) return;

    const pipeline = pipelines.get(pipelineId);
    const bindGroup = bindGroups.get(bindGroupId);
    const groupsBuffer = buffers.get(groupsBufferId);
    if (!pipeline || !bindGroup || !groupsBuffer) {
        console.error('dispatchComputeFolded: pipeline, bind group or workgroup-count buffer not found');
        return;
    }
    let storage;
    if (uniformBufferId >= 0) {
        const uniformBuffer = buffers.get(uniformBufferId);
        if (!uniformBuffer) {
            console.error(`Uniform buffer ${uniformBufferId} not found`);
            return;
        }
        storage = await resolveUniformBindGroup(bindGroup, pipeline, uniformBuffer);
        if (!storage) return;
    } else {
        storage = resolveBindGroup(bindGroup, pipeline);
    }
    const groups = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(1),
        entries: [{ binding: 0, resource: { buffer: groupsBuffer } }]
    });

    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, storage);
    passEncoder.setBindGroup(1, groups);
    passEncoder.dispatchWorkgroups(workgroupsX, workgroupsY, 1);
    passEncoder.end();

    queue.submit([commandEncoder.finish()]);
}

/**
 * Dispatches a chain of compute operations, each with its uniforms, in ONE compute pass of one command encoder,
 * submitted once (#1452, the resident program). WebGPU orders the dispatches of a pass, with the storage writes of
 * one visible to the next, so the chain needs no barrier of its own. Every bind group exists before the first
 * dispatch runs, so none may hold a snapshot of a buffer an earlier dispatch writes (WebGpuBindGroup snapshots a
 * buffer bound twice).
 * @param {number[]} pipelineIds - Pipeline ID per dispatch.
 * @param {number[]} bindGroupIds - Storage bind group ID per dispatch.
 * @param {number[]} uniformBufferIds - Uniform buffer ID per dispatch.
 * @param {number[]} workgroupsX - Workgroups in X per dispatch.
 */
export async function dispatchComputeChainWithUniforms(pipelineIds, bindGroupIds, uniformBufferIds, workgroupsX) {
    if (!device) return;

    const stages = [];
    for (let i = 0; i < pipelineIds.length; i++) {
        const pipeline = pipelines.get(pipelineIds[i]);
        const bindGroup = bindGroups.get(bindGroupIds[i]);
        const uniformBuffer = buffers.get(uniformBufferIds[i]);
        if (!pipeline || !bindGroup || !uniformBuffer) {
            console.error(`dispatchComputeChainWithUniforms: stage ${i}: pipeline, bind group or uniform buffer not found`);
            return;
        }
        const resolved = await resolveUniformBindGroup(bindGroup, pipeline, uniformBuffer);
        if (!resolved) return;
        stages.push({ pipeline: pipeline, group: resolved, workgroups: workgroupsX[i] });
    }

    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    for (const stage of stages) {
        passEncoder.setPipeline(stage.pipeline);
        passEncoder.setBindGroup(0, stage.group);
        passEncoder.dispatchWorkgroups(stage.workgroups, 1, 1);
    }
    passEncoder.end();

    queue.submit([commandEncoder.finish()]);
}

/**
 * Submits all pending commands and waits for completion.
 */
export async function submitAndWait() {
    if (!device || !queue) return;

    // Create a fence-like mechanism using a mappable buffer
    const signalBuffer = device.createBuffer({
        size: 4,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
    });

    try {
        await queue.onSubmittedWorkDone();
    } finally {
        signalBuffer.destroy();
    }
}

// ============================================================================
// Timestamp Queries (if supported)
// ============================================================================

/**
 * Creates a timestamp query for profiling.
 * @returns {number} Query ID, or -1 if not supported.
 */
export function createTimestampQuery() {
    if (!device || !device.features.has('timestamp-query')) {
        return -1;
    }

    try {
        const querySet = device.createQuerySet({
            type: 'timestamp',
            count: 2
        });

        const resolveBuffer = device.createBuffer({
            size: 16,
            usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC
        });

        const readBuffer = device.createBuffer({
            size: 16,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
        });

        const id = nextTimestampQueryId++;
        timestampQueries.set(id, { querySet, resolveBuffer, readBuffer });
        return id;
    } catch (error) {
        console.error('Failed to create timestamp query:', error);
        return -1;
    }
}

/**
 * Writes timestamp `index` (0 or 1) of a query, from an empty compute pass in a submission of its own, so the
 * interval between the two writes covers exactly the work submitted between them.
 * @param {number} queryId - Query ID from createTimestampQuery.
 * @param {number} index - 0 for the start, 1 for the end.
 */
export function writeTimestamp(queryId, index) {
    const query = timestampQueries.get(queryId);
    if (!query) return;
    const commandEncoder = device.createCommandEncoder();
    const pass = commandEncoder.beginComputePass({
        timestampWrites: { querySet: query.querySet, beginningOfPassWriteIndex: index }
    });
    pass.end();
    queue.submit([commandEncoder.finish()]);
}
/**
 * Reads timestamp query results.
 * @param {number} queryId - Query ID.
 * @returns {Promise<number>} Duration in nanoseconds, or -1 on error.
 */
export async function readTimestampQuery(queryId) {
    const query = timestampQueries.get(queryId);
    if (!query) return -1;

    try {
        const commandEncoder = device.createCommandEncoder();
        commandEncoder.resolveQuerySet(query.querySet, 0, 2, query.resolveBuffer, 0);
        commandEncoder.copyBufferToBuffer(query.resolveBuffer, 0, query.readBuffer, 0, 16);
        queue.submit([commandEncoder.finish()]);

        await query.readBuffer.mapAsync(GPUMapMode.READ);
        const data = new BigUint64Array(query.readBuffer.getMappedRange());
        const duration = Number(data[1] - data[0]);
        query.readBuffer.unmap();

        return duration;
    } catch (error) {
        console.error('Failed to read timestamp query:', error);
        return -1;
    }
}

/**
 * Destroys a timestamp query.
 * @param {number} queryId - Query ID.
 */
export function destroyTimestampQuery(queryId) {
    const query = timestampQueries.get(queryId);
    if (query) {
        try { query.querySet.destroy(); } catch (e) { }
        try { query.resolveBuffer.destroy(); } catch (e) { }
        try { query.readBuffer.destroy(); } catch (e) { }
        timestampQueries.delete(queryId);
    }
}
