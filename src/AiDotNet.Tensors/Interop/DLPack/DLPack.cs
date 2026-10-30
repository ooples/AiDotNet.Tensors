using System;
using System.Buffers;
using System.Collections.Concurrent;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.NumericOperations;

namespace AiDotNet.Tensors.Interop;

/// <summary>
/// DLPack (dmlc/dlpack v1.x) zero-copy tensor exchange with PyTorch, NumPy, JAX, CuPy, ONNX Runtime and any other
/// DLPack-speaking library.
/// </summary>
/// <remarks>
/// <para><b>Exporting</b> (<see cref="Tensor{T}.ToDLPack"/>): the result is a <c>DLManagedTensorVersioned*</c> whose
/// data is this tensor's memory, not a copy. The consumer owns the pointer and calls its <c>deleter</c> exactly once
/// when done; until then the exported memory stays valid even if every AiDotNet.Tensors reference to the tensor is
/// disposed. <see cref="Tensor{T}.ToDLPackLegacy"/> produces the unversioned <c>DLManagedTensor*</c> for consumers that
/// predate DLPack 0.8 (<c>torch.utils.dlpack.from_dlpack</c> on old PyTorch).</para>
/// <para><b>Importing</b> (<see cref="Tensor{T}.FromDLPack"/>): the tensor aliases the producer's memory. Import takes
/// ownership of the pointer in every case: the producer's deleter runs once when the tensor's storage is released
/// (its last tensor or view disposed), or before the exception if the import is refused. Dispose imported tensors:
/// host memory imported by a tensor that is only finalized is never released (a span over it can outlive the last
/// tensor reference, so freeing it then is unsound), and device memory is released at the backend's next operation,
/// after the work queued on it.</para>
/// <para><b>Devices.</b> CPU memory (and CUDA/ROCm pinned host memory) in both directions. CUDA device memory for
/// <c>float</c> tensors on the CUDA backend: an exported GPU-resident tensor hands out its device pointer, and an
/// imported CUDA tensor becomes a GPU-resident tensor over the producer's allocation.</para>
/// <para><b>Streams.</b> On export the producer orders the consumer's stream after the work that produced the data
/// (<see cref="DLPackStream"/>). On import the producer has done that for the stream the consumer named; pass it
/// <see cref="GetCudaConsumerStream"/> so the ordering lands on the stream AiDotNet.Tensors runs on.</para>
/// <para><b>Element types.</b> <c>float</c>, <c>double</c>, <c>Half</c> (.NET 5+, where it is IEEE binary16),
/// <see cref="BFloat16"/>, <c>sbyte</c>, <c>short</c>, <c>int</c>, <c>long</c>, <c>byte</c>, <c>ushort</c>,
/// <c>uint</c>, <c>ulong</c> and <c>bool</c>.</para>
/// <para><b>Writes.</b> Memory written by the other library bypasses this tensor's change tracking: call
/// <see cref="TensorBase{T}.MarkModified"/> after such a write, as for <see cref="Tensor{T}.FromMemory"/>.</para>
/// </remarks>
public static class DLPack
{
    /// <summary>The DLPack ABI major version produced and accepted.</summary>
    public const int MajorVersion = (int)DLPackNative.MajorVersion;

    /// <summary>The DLPack minor version produced. Imports accept any minor version of <see cref="MajorVersion"/>.</summary>
    public const int MinorVersion = (int)DLPackNative.MinorVersion;

    /// <summary>
    /// The CUDA stream AiDotNet.Tensors orders its work on, as the handle a producer's <c>__dlpack__(stream=...)</c>
    /// expects: request the capsule with this stream before passing it to <see cref="Tensor{T}.FromDLPack"/>, so the
    /// producer's pending writes are complete before AiDotNet.Tensors reads the memory.
    /// </summary>
    /// <param name="deviceId">The CUDA device ordinal.</param>
    /// <exception cref="NotSupportedException">No CUDA backend is active for that device.</exception>
    public static IntPtr GetCudaConsumerStream(int deviceId = 0) => GetCudaBackend(deviceId).Device.WorkStreamHandle;

    // ── Exports ────────────────────────────────────────────────────────────────────────────────────────────────────

    // Every live export, by the address of its native block. The deleter looks itself up here rather than reading the
    // block, so a consumer calling the deleter twice (a contract violation) finds nothing the second time instead of
    // freeing the block again.
    private static readonly ConcurrentDictionary<IntPtr, ExportState> s_liveExports = new();

    /// <summary>Exports whose deleter has not run yet. For tests and leak diagnostics.</summary>
    internal static int LiveExportCount => s_liveExports.Count;

    /// <summary>Deleter calls that matched no live export (a double call, or a foreign pointer).</summary>
    internal static long UnmatchedDeleterCalls => Interlocked.Read(ref s_unmatchedDeleterCalls);
    private static long s_unmatchedDeleterCalls;

    /// <summary>
    /// Host-memory imports whose tensor was finalized without being disposed. Their producer memory is never released
    /// (see <see cref="DLPackDeleterHandle"/>), so a nonzero count is a leak to fix by disposing the tensors.
    /// </summary>
    internal static long LeakedHostImports => Interlocked.Read(ref s_leakedHostImports);
    private static long s_leakedHostImports;

    internal static void CountLeakedHostImport() => Interlocked.Increment(ref s_leakedHostImports);

    /// <summary>Releases that threw inside the deleter (a C callback cannot propagate them).</summary>
    internal static long FailedReleases => Interlocked.Read(ref s_failedReleases);
    private static long s_failedReleases;

    private sealed class ExportState
    {
        public ExportState(IntPtr block, MemoryHandle pin, IDisposable keepAlive)
        {
            Block = block;
            Pin = pin;
            KeepAlive = keepAlive;
        }

        public IntPtr Block { get; }
        public MemoryHandle Pin { get; }
        public IDisposable KeepAlive { get; }

        public void Release()
        {
            try
            {
                Pin.Dispose();
                KeepAlive.Dispose();
            }
            finally
            {
                Marshal.FreeHGlobal(Block);
            }
        }
    }

    /// <summary>
    /// Builds the native managed tensor for an export and registers it. <paramref name="keepAlive"/> holds the
    /// exported memory (a storage-sharing view of the source) and <paramref name="pin"/> fixes host memory in place;
    /// both are released when the consumer calls the deleter.
    /// </summary>
    internal static unsafe IntPtr CreateExport(
        bool versioned,
        IntPtr data,
        int deviceType,
        int deviceId,
        DLPackNative.DLDataType dtype,
        int[] shape,
        int[] strides,
        bool readOnly,
        MemoryHandle pin,
        IDisposable keepAlive,
        bool isCopied = false)
    {
        int ndim = shape.Length;
        int headerBytes = versioned
            ? Unsafe.SizeOf<DLPackNative.DLManagedTensorVersioned>()
            : Unsafe.SizeOf<DLPackNative.DLManagedTensor>();
        // The header and its shape/strides arrays share one block (int64-aligned: the header size is a multiple of 8),
        // so the deleter frees exactly one allocation.
        int headerAligned = (headerBytes + 7) & ~7;
        IntPtr block = Marshal.AllocHGlobal(headerAligned + 2 * ndim * sizeof(long));
        try
        {
            long* shapePtr = (long*)((byte*)block + headerAligned);
            long* stridesPtr = shapePtr + ndim;
            for (int i = 0; i < ndim; i++)
            {
                shapePtr[i] = shape[i];
                stridesPtr[i] = strides[i];
            }

            var tensor = new DLPackNative.DLTensor
            {
                // An empty tensor carries a NULL data pointer, as the spec asks.
                Data = ProductIsZero(shape) ? IntPtr.Zero : data,
                Device = new DLPackNative.DLDevice { DeviceType = deviceType, DeviceId = deviceId },
                NDim = ndim,
                DType = dtype,
                Shape = ndim == 0 ? IntPtr.Zero : (IntPtr)shapePtr,
                Strides = ndim == 0 ? IntPtr.Zero : (IntPtr)stridesPtr,
                ByteOffset = 0,
            };

            if (versioned)
            {
                *(DLPackNative.DLManagedTensorVersioned*)block = new DLPackNative.DLManagedTensorVersioned
                {
                    Version = new DLPackNative.DLPackVersion { Major = DLPackNative.MajorVersion, Minor = DLPackNative.MinorVersion },
                    ManagerCtx = IntPtr.Zero,
                    Deleter = ExportDeleterPointer,
                    Flags = (readOnly ? DLPackNative.FlagReadOnly : 0) | (isCopied ? DLPackNative.FlagIsCopied : 0),
                    DLTensor = tensor,
                };
            }
            else
            {
                *(DLPackNative.DLManagedTensor*)block = new DLPackNative.DLManagedTensor
                {
                    DLTensor = tensor,
                    ManagerCtx = IntPtr.Zero,
                    Deleter = ExportDeleterPointer,
                };
            }

            s_liveExports[block] = new ExportState(block, pin, keepAlive);
            return block;
        }
        catch
        {
            Marshal.FreeHGlobal(block);
            throw;
        }
    }

    private static bool ProductIsZero(int[] shape)
    {
        foreach (int extent in shape)
            if (extent == 0) return true;
        return false;
    }

    /// <summary>The deleter every export carries. The same entry serves both layouts: it only uses the address.</summary>
    internal static void ReleaseExport(IntPtr self)
    {
        if (!s_liveExports.TryRemove(self, out var state))
        {
            Interlocked.Increment(ref s_unmatchedDeleterCalls);
            return;
        }
        state.Release();
    }

#if NET5_0_OR_GREATER
    [UnmanagedCallersOnly(CallConvs = new[] { typeof(CallConvCdecl) })]
    private static void ExportDeleterEntry(IntPtr self)
    {
        try { ReleaseExport(self); }
        catch { Interlocked.Increment(ref s_failedReleases); }   // an exception must not unwind into C
    }

    private static unsafe IntPtr ExportDeleterPointer
        => (IntPtr)(delegate* unmanaged[Cdecl]<IntPtr, void>)&ExportDeleterEntry;
#else
    private static void ExportDeleterEntry(IntPtr self)
    {
        try { ReleaseExport(self); }
        catch { Interlocked.Increment(ref s_failedReleases); }   // an exception must not unwind into C
    }

    // Held in a static so the thunk behind the function pointer lives as long as the process.
    private static readonly DLPackNative.DeleterFn s_exportDeleter = ExportDeleterEntry;
    private static readonly IntPtr s_exportDeleterPointer = Marshal.GetFunctionPointerForDelegate(s_exportDeleter);
    private static IntPtr ExportDeleterPointer => s_exportDeleterPointer;
#endif

    // ── Imports ────────────────────────────────────────────────────────────────────────────────────────────────────

    /// <summary>
    /// Wraps a producer's managed tensor as a <see cref="Tensor{T}"/>. Takes ownership of <paramref name="managed"/>:
    /// on failure its deleter has run before the exception propagates.
    /// </summary>
    internal static unsafe Tensor<T> Import<T>(IntPtr managed, bool versioned)
    {
        if (managed == IntPtr.Zero) throw new ArgumentNullException(nameof(managed));

        IntPtr deleter;
        DLPackNative.DLTensor dl;
        bool readOnly = false;
        uint minor;
        if (versioned)
        {
            var header = (DLPackNative.DLManagedTensorVersioned*)managed;
            deleter = header->Deleter;   // ABI-stable across major versions, so safe to read first
            if (header->Version.Major != DLPackNative.MajorVersion)
            {
                uint major = header->Version.Major;
                DLPackNative.InvokeDeleter(deleter, managed);   // the spec: call the deleter, touch nothing else
                throw new NotSupportedException(
                    $"DLPack tensor has ABI major version {major}; this build reads major version {DLPackNative.MajorVersion}.");
            }
            minor = header->Version.Minor;
            readOnly = (header->Flags & DLPackNative.FlagReadOnly) != 0;
            dl = header->DLTensor;
        }
        else
        {
            var header = (DLPackNative.DLManagedTensor*)managed;
            deleter = header->Deleter;
            minor = 0;   // pre-versioning: NULL strides mean compact row-major
            dl = header->DLTensor;
        }

        var owner = new DLPackDeleterHandle(deleter, managed);
        try
        {
            return dl.Device.DeviceType switch
            {
                DLPackNative.DeviceCpu or DLPackNative.DeviceCudaHost or DLPackNative.DeviceRocmHost
                    => ImportHost<T>(in dl, minor, readOnly, owner),
                DLPackNative.DeviceCuda => ImportCuda<T>(in dl, minor, readOnly, owner, managed),
                _ => throw new NotSupportedException(
                    $"DLPack device type {dl.Device.DeviceType} is not supported; AiDotNet.Tensors imports CPU and " +
                    "pinned host memory (kDLCPU, kDLCUDAHost, kDLROCMHost) and CUDA device memory (kDLCUDA)."),
            };
        }
        catch
        {
            owner.Release();
            throw;
        }
    }

    private static unsafe void ReadLayout<T>(in DLPackNative.DLTensor dl, uint minor, out int[] shape, out long[] strides, out long count)
    {
        if (!TryGetDataType<T>(out var expected))
            throw new NotSupportedException($"Tensor<{typeof(T).Name}> has no DLPack element type.");
        if (dl.DType.Code != expected.Code || dl.DType.Bits != expected.Bits || dl.DType.Lanes != 1)
            throw new ArgumentException(
                $"DLPack tensor has dtype (code {dl.DType.Code}, bits {dl.DType.Bits}, lanes {dl.DType.Lanes}); " +
                $"Tensor<{typeof(T).Name}> needs (code {expected.Code}, bits {expected.Bits}, lanes 1).");
        if (dl.NDim < 0) throw new ArgumentException($"DLPack tensor has negative ndim {dl.NDim}.");
        if (dl.NDim > 0 && dl.Shape == IntPtr.Zero) throw new ArgumentException("DLPack tensor has ndim > 0 but no shape.");

        shape = new int[dl.NDim];
        count = 1;
        var shapePtr = (long*)dl.Shape;
        for (int i = 0; i < dl.NDim; i++)
        {
            long extent = shapePtr[i];
            if (extent < 0 || extent > int.MaxValue)
                throw new ArgumentException($"DLPack tensor dimension {i} has extent {extent}, outside [0, int.MaxValue].");
            shape[i] = (int)extent;
            count *= extent;
            if (count > int.MaxValue)
                throw new NotSupportedException("A DLPack tensor imported into AiDotNet.Tensors holds at most int.MaxValue elements.");
        }

        strides = new long[dl.NDim];
        if (dl.NDim > 0 && dl.Strides == IntPtr.Zero)
        {
            // NULL strides meant "compact row-major" before DLPack 1.2; from 1.2 on they are mandatory.
            if (minor >= DLPackNative.MinorVersionStridesRequired)
                throw new ArgumentException($"DLPack 1.{minor} tensor has ndim > 0 but NULL strides.");
            long stride = 1;
            for (int i = dl.NDim - 1; i >= 0; i--)
            {
                strides[i] = stride;
                stride *= Math.Max(shape[i], 1);
            }
        }
        else
        {
            var stridesPtr = (long*)dl.Strides;
            for (int i = 0; i < dl.NDim; i++) strides[i] = stridesPtr[i];
        }
    }

    /// <summary>True when the strides describe compact row-major order (strides of extent-1 dimensions are free).</summary>
    private static bool IsCompactRowMajor(int[] shape, long[] strides)
    {
        long expected = 1;
        for (int i = shape.Length - 1; i >= 0; i--)
        {
            if (shape[i] != 1 && strides[i] != expected) return false;
            expected *= shape[i];
        }
        return true;
    }

    private static unsafe Tensor<T> ImportHost<T>(in DLPackNative.DLTensor dl, uint minor, bool readOnly, DLPackDeleterHandle owner)
    {
        ReadLayout<T>(in dl, minor, out var shape, out var strides, out long count);
        int elementSize = Unsafe.SizeOf<T>();

        if (count == 0)
        {
            // Nothing to alias: the producer's memory is not needed past this point.
            owner.Release();
            return new Tensor<T>(shape);
        }
        if (dl.Data == IntPtr.Zero) throw new ArgumentException("DLPack tensor has elements but a NULL data pointer.");

        byte* first = (byte*)dl.Data + (long)dl.ByteOffset;
        if (IsCompactRowMajor(shape, strides))
        {
            var memory = new DLPackHostMemory<T>((IntPtr)first, (int)count, owner);
            return Tensor<T>.AliasDLPackMemory(memory, shape, writable: !readOnly);
        }

        // Strided (a transpose, a slice, a broadcast with zero strides, or negative strides): alias the span the
        // strides cover and view it. The view's storage offset puts element [0, ..., 0] back where the producer has it.
        long minIndex = 0, maxIndex = 0;
        for (int i = 0; i < shape.Length; i++)
        {
            if (shape[i] <= 1) continue;
            long extent = (shape[i] - 1) * strides[i];
            if (extent >= 0) maxIndex += extent; else minIndex += extent;
        }
        long covered = maxIndex - minIndex + 1;
        if (covered > int.MaxValue || minIndex < int.MinValue)
            throw new NotSupportedException("A strided DLPack tensor imported into AiDotNet.Tensors spans at most int.MaxValue elements.");
        var intStrides = new int[strides.Length];
        for (int i = 0; i < strides.Length; i++)
        {
            if (strides[i] > int.MaxValue || strides[i] < int.MinValue)
                throw new NotSupportedException($"DLPack tensor stride {strides[i]} does not fit an int.");
            intStrides[i] = (int)strides[i];
        }
        var covering = new DLPackHostMemory<T>((IntPtr)(first + minIndex * elementSize), (int)covered, owner);
        return Tensor<T>.AliasDLPackMemoryStrided(covering, (int)covered, shape, intStrides, (int)(-minIndex), writable: !readOnly);
    }

    private static Tensor<T> ImportCuda<T>(in DLPackNative.DLTensor dl, uint minor, bool readOnly, DLPackDeleterHandle owner, IntPtr managed)
    {
        if (typeof(T) != typeof(float))
            throw new NotSupportedException(
                $"CUDA DLPack import supports Tensor<float> only (the CUDA backend stores device data as float32); got Tensor<{typeof(T).Name}>.");
        ReadLayout<T>(in dl, minor, out var shape, out var strides, out long count);
        if (!IsCompactRowMajor(shape, strides))
            throw new NotSupportedException(
                "CUDA DLPack import needs a compact row-major tensor; call .contiguous() on the producer side first.");
        if (readOnly)
            throw new NotSupportedException(
                "CUDA DLPack import of a read-only tensor is not supported: GPU-resident tensors have no read-only mode.");

        var (backend, device) = GetCudaBackend(dl.Device.DeviceId);
        if (count == 0)
        {
            owner.Release();
            return new Tensor<T>(shape);
        }
        if (dl.Data == IntPtr.Zero) throw new ArgumentException("DLPack tensor has elements but a NULL data pointer.");

        var pointer = new IntPtr(dl.Data.ToInt64() + (long)dl.ByteOffset);
        // The deleter runs only after this backend's queued kernels have finished with the memory: disposing the
        // buffer waits for them, and a finalized import queues its release behind them on this backend.
        owner.SetDeviceOwner(device);
        var buffer = device.WrapExternalDeviceMemory(pointer, (int)count, sizeof(float), owner.Release);
        try
        {
            var tensor = Tensor<float>.FromGpuBuffer(backend, buffer, shape, ownsBuffer: true);
            tensor._gpuDeviceIndex = device.DeviceOrdinal;
            return (Tensor<T>)(object)tensor;
        }
        catch
        {
            buffer.Dispose();   // runs owner.Release; the outer catch's Release is then a no-op
            throw;
        }
    }

    /// <summary>The GPU backend tensors are placed on, as <see cref="Tensor{T}.Gpu"/> chooses it, when it exchanges
    /// CUDA device memory for <paramref name="deviceId"/>.</summary>
    internal static (IDirectGpuBackend Backend, IDLPackDeviceBackend Device) GetCudaBackend(int deviceId)
    {
        var backend = AiDotNetEngine.Current is DirectGpuTensorEngine dispatcher
            ? dispatcher.PlacementBackend
            : Engine.DirectGpu is { IsAvailable: true } directGpu ? directGpu.Backend : null;
        if (backend is not IDLPackDeviceBackend { DLPackDeviceType: DLPackNative.DeviceCuda } device)
            throw new NotSupportedException(
                "CUDA DLPack exchange needs the CUDA backend as the active GPU backend" +
                (backend is null ? "; no GPU backend is active." : $"; the active backend is {backend.GetType().Name}."));
        if (device.DeviceOrdinal != deviceId)
            throw new NotSupportedException(
                $"The DLPack tensor is on CUDA device {deviceId}; the active CUDA backend runs on device {device.DeviceOrdinal}.");
        return (backend, device);
    }

    // ── Element types ──────────────────────────────────────────────────────────────────────────────────────────────

    /// <summary>The DLPack dtype of <typeparamref name="T"/>, when it has one.</summary>
    internal static bool TryGetDataType<T>(out DLPackNative.DLDataType dtype)
    {
        byte code;
        byte bits;
        if (typeof(T) == typeof(float)) { code = DLPackNative.CodeFloat; bits = 32; }
        else if (typeof(T) == typeof(double)) { code = DLPackNative.CodeFloat; bits = 64; }
#if NET5_0_OR_GREATER
        // Only .NET 5+'s Half is IEEE binary16; the .NET Framework shim stores a float.
        else if (typeof(T) == typeof(Half)) { code = DLPackNative.CodeFloat; bits = 16; }
#endif
        else if (typeof(T) == typeof(BFloat16)) { code = DLPackNative.CodeBfloat; bits = 16; }
        else if (typeof(T) == typeof(sbyte)) { code = DLPackNative.CodeInt; bits = 8; }
        else if (typeof(T) == typeof(short)) { code = DLPackNative.CodeInt; bits = 16; }
        else if (typeof(T) == typeof(int)) { code = DLPackNative.CodeInt; bits = 32; }
        else if (typeof(T) == typeof(long)) { code = DLPackNative.CodeInt; bits = 64; }
        else if (typeof(T) == typeof(byte)) { code = DLPackNative.CodeUInt; bits = 8; }
        else if (typeof(T) == typeof(ushort)) { code = DLPackNative.CodeUInt; bits = 16; }
        else if (typeof(T) == typeof(uint)) { code = DLPackNative.CodeUInt; bits = 32; }
        else if (typeof(T) == typeof(ulong)) { code = DLPackNative.CodeUInt; bits = 64; }
        else if (typeof(T) == typeof(bool)) { code = DLPackNative.CodeBool; bits = 8; }
        else
        {
            dtype = default;
            return false;
        }
        dtype = new DLPackNative.DLDataType { Code = code, Bits = bits, Lanes = 1 };
        return true;
    }
}
