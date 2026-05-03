// Copyright (c) AiDotNet. All rights reserved.
// Issue #285 — DispatchProxy-based mock IDirectGpuBackend for chunker tests.
// Implements a tiny set of methods (the ones the chunker actually calls)
// and throws NotImplementedException for everything else. Sidesteps the
// 470-member interface surface that would require a hand-written stub.
//
// DispatchProxy lives in System.Reflection from .NET Core 5+; it is NOT
// available on .NET Framework, so the whole mock + its tests are gated
// to non-Framework targets. Coverage on net10.0 is the meaningful one
// for codecov.

#if !NETFRAMEWORK
#nullable disable

using System;
using System.Collections.Generic;
using System.Reflection;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Minimal IGpuBuffer wrapping a host float[] — the mock backend's
/// "device buffer" is just the array itself.
/// </summary>
internal sealed class MockGpuBuffer : IGpuBuffer
{
    public float[] Data { get; }
    public int Size => Data.Length;
    public long SizeInBytes => (long)Data.Length * sizeof(float);
    public IntPtr Handle { get; }

    public MockGpuBuffer(float[] data)
    {
        Data = data;
        // Use a synthetic handle (allocation counter) so distinct buffers
        // get distinct handles; tests don't dereference these pointers.
        Handle = (IntPtr)System.Threading.Interlocked.Increment(ref _handleCounter);
    }

    private static int _handleCounter;
    public void Dispose() { }
}

/// <summary>
/// State shared between the mock backend and tests. Captures every
/// allocation request and configures the per-allocation cap that the
/// mock reports via <see cref="IDirectGpuBackend.MaxBufferAllocBytes"/>.
/// </summary>
internal sealed class MockBackendState
{
    public long MaxBufferAllocBytes { get; set; } = long.MaxValue;
    public string DeviceName { get; set; } = "MockDevice";
    public string DeviceVendor { get; set; } = "Mock";
    public List<int> AllocationSizes { get; } = new();
    public int DownloadBufferCalls { get; set; }
    public int UnaryOpCalls { get; set; }
    public int BinaryOpCalls { get; set; }
}

/// <summary>
/// DispatchProxy implementation that backs <see cref="MockDirectGpuBackend"/>.
/// Implements only the methods the chunker actually calls; everything else
/// throws <see cref="NotImplementedException"/>. The 470-method
/// IDirectGpuBackend surface is too large to stub by hand — this lets us
/// pretend to be a backend cheaply.
/// </summary>
public class MockDirectGpuBackend : DispatchProxy
{
    private MockBackendState _state;

    internal static IDirectGpuBackend Create(MockBackendState state)
    {
        var proxy = (MockDirectGpuBackend)Create<IDirectGpuBackend, MockDirectGpuBackend>();
        proxy._state = state;
        return (IDirectGpuBackend)(object)proxy;
    }

    protected override object Invoke(MethodInfo targetMethod, object[] args)
    {
        switch (targetMethod.Name)
        {
            // ── Identity / device info — implemented ──
            case "get_IsAvailable": return true;
            case "get_BackendName": return "Mock";
            case "get_DeviceType": return TensorDevice.OpenCL; // any value works
            case "get_DeviceName": return _state.DeviceName;
            case "get_DeviceVendor": return _state.DeviceVendor;
            case "get_ComputeUnits": return 1;
            case "get_GlobalMemoryBytes": return _state.MaxBufferAllocBytes;
            case "get_LocalMemoryBytes": return 0L;
            case "get_MaxBufferAllocBytes": return _state.MaxBufferAllocBytes;
            case "get_TheoreticalGflops": return 0.0;
            case "Dispose": return null!;

            // ── Buffer allocation + download — what the chunker uses ──
            case "AllocateBuffer" when args.Length == 1 && args[0] is float[] arr:
            {
                long bytes = (long)arr.Length * sizeof(float);
                GpuBufferSizeGuard.EnsureFits("Mock", bytes, _state.MaxBufferAllocBytes, _state.DeviceName);
                _state.AllocationSizes.Add(arr.Length);
                // Copy so subsequent host writes don't affect the buffer.
                var copy = new float[arr.Length];
                Array.Copy(arr, copy, arr.Length);
                return new MockGpuBuffer(copy);
            }
            case "AllocateBuffer" when args.Length == 1 && args[0] is int size:
            {
                long bytes = (long)size * sizeof(float);
                GpuBufferSizeGuard.EnsureFits("Mock", bytes, _state.MaxBufferAllocBytes, _state.DeviceName);
                _state.AllocationSizes.Add(size);
                return new MockGpuBuffer(new float[size]);
            }
            case "DownloadBuffer" when args.Length == 1:
            {
                _state.DownloadBufferCalls++;
                var b = (MockGpuBuffer)args[0]!;
                var copy = new float[b.Size];
                Array.Copy(b.Data, copy, b.Size);
                return copy;
            }
            case "DownloadBuffer" when args.Length == 2:
            {
                _state.DownloadBufferCalls++;
                var b = (MockGpuBuffer)args[0]!;
                var dest = (float[])args[1]!;
                Array.Copy(b.Data, dest, b.Size);
                return null!;
            }
        }
        throw new NotImplementedException(
            $"MockDirectGpuBackend does not implement {targetMethod.Name}. " +
            "Add a case in MockDirectGpuBackend.Invoke if your test exercises this op.");
    }
}
#endif
