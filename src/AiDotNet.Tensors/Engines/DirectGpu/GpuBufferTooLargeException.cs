// Copyright (c) AiDotNet. All rights reserved.

using System;

namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// Thrown when a single GPU buffer allocation would exceed the device's
/// per-allocation cap (OpenCL's <c>CL_DEVICE_MAX_MEM_ALLOC_SIZE</c>, CUDA's
/// per-allocation virtual address window, …). Distinct from a general
/// out-of-memory: the device may have plenty of total memory free but
/// refuse a single buffer larger than the cap.
/// </summary>
/// <remarks>
/// Catch this in dispatch shims (e.g. <see cref="DirectGpuTensorEngine"/>)
/// to fall through to a CPU path or chunked-GPU path instead of crashing
/// the user's training run. The thrown exception carries the requested and
/// device-cap byte counts so logs / telemetry can show the user a concrete
/// "reduce batch size from X to Y" hint.
/// </remarks>
public sealed class GpuBufferTooLargeException : InvalidOperationException
{
    /// <summary>
    /// Bytes the caller asked the backend to allocate as a single buffer.
    /// </summary>
    public long RequestedBytes { get; }

    /// <summary>
    /// Maximum bytes the device allows in a single allocation
    /// (<c>CL_DEVICE_MAX_MEM_ALLOC_SIZE</c> on OpenCL, equivalent on other
    /// backends). Zero when unknown / not queried.
    /// </summary>
    public long DeviceMaxAllocBytes { get; }

    /// <summary>
    /// Human-readable device name (e.g. "AMD Radeon RX 5500 XT") for use
    /// in error messages and telemetry. Empty when unknown.
    /// </summary>
    public string DeviceName { get; }

    /// <summary>
    /// Backend identifier (e.g. "OpenCL", "CUDA"). Useful for logs that
    /// span multiple backends.
    /// </summary>
    public string Backend { get; }

    public GpuBufferTooLargeException(
        string backend,
        long requestedBytes,
        long deviceMaxAllocBytes,
        string deviceName)
        : base(BuildMessage(backend, requestedBytes, deviceMaxAllocBytes, deviceName))
    {
        Backend = backend ?? string.Empty;
        RequestedBytes = requestedBytes;
        DeviceMaxAllocBytes = deviceMaxAllocBytes;
        DeviceName = deviceName ?? string.Empty;
    }

    private static string BuildMessage(string backend, long requested, long cap, string deviceName)
    {
        string b = string.IsNullOrEmpty(backend) ? "GPU" : backend;
        string device = string.IsNullOrEmpty(deviceName) ? "device" : deviceName;
        if (cap > 0)
        {
            return $"{b} buffer allocation of {requested} bytes ({BytesToHuman(requested)}) " +
                   $"exceeds {device}'s per-allocation cap of {cap} bytes ({BytesToHuman(cap)}). " +
                   $"Reduce batch size or split the tensor — typical fixes: lower " +
                   $"`maxSequenceLength`, `batchSize`, or model dimension; or call " +
                   $"`ConfigureGpuAcceleration(new GpuAccelerationConfig {{ UsageLevel = GpuUsageLevel.AlwaysCpu }})` " +
                   $"to disable GPU for this run.";
        }
        return $"{b} buffer allocation of {requested} bytes ({BytesToHuman(requested)}) " +
               $"refused by {device} (per-allocation cap unknown).";
    }

    private static string BytesToHuman(long bytes)
    {
        if (bytes < 1024) return bytes + " B";
        if (bytes < 1024L * 1024) return (bytes / 1024.0).ToString("F1") + " KiB";
        if (bytes < 1024L * 1024 * 1024) return (bytes / (1024.0 * 1024)).ToString("F1") + " MiB";
        return (bytes / (1024.0 * 1024 * 1024)).ToString("F2") + " GiB";
    }
}
