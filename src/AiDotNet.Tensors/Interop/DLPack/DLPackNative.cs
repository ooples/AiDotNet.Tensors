using System;
using System.Runtime.InteropServices;

namespace AiDotNet.Tensors.Interop;

/// <summary>
/// The C ABI of DLPack (dmlc/dlpack <c>include/dlpack/dlpack.h</c>, v1.3), field for field. Pointers are
/// <see cref="IntPtr"/> so the sequential layout matches the C struct on 32- and 64-bit processes alike.
/// </summary>
internal static class DLPackNative
{
    /// <summary>DLPACK_MAJOR_VERSION: a different major version means a different ABI.</summary>
    internal const uint MajorVersion = 1;

    /// <summary>DLPACK_MINOR_VERSION of the header this layout was written against.</summary>
    internal const uint MinorVersion = 3;

    /// <summary>DLPACK_FLAG_BITMASK_READ_ONLY: the consumer must not write through the tensor.</summary>
    internal const ulong FlagReadOnly = 1UL << 0;

    /// <summary>DLPACK_FLAG_BITMASK_IS_COPIED: the producer made a copy the consumer owns outright.</summary>
    internal const ulong FlagIsCopied = 1UL << 1;

    /// <summary>The first minor version (v1.2) in which <c>strides</c> must be non-NULL when <c>ndim != 0</c>.</summary>
    internal const uint MinorVersionStridesRequired = 2;

    // DLDeviceType values used here.
    internal const int DeviceCpu = 1;
    internal const int DeviceCuda = 2;
    internal const int DeviceCudaHost = 3;
    internal const int DeviceRocmHost = 11;
    internal const int DeviceCudaManaged = 13;

    // DLDataTypeCode values used here.
    internal const byte CodeInt = 0;
    internal const byte CodeUInt = 1;
    internal const byte CodeFloat = 2;
    internal const byte CodeBfloat = 4;
    internal const byte CodeBool = 6;

    // The array-API stream values for CUDA (`__dlpack__(stream=...)`): 1 is the legacy default stream, 2 the
    // per-thread default stream. The CUDA driver accepts both as stream handles (CU_STREAM_LEGACY,
    // CU_STREAM_PER_THREAD).
    internal static readonly IntPtr CudaLegacyDefaultStream = new IntPtr(1);
    internal static readonly IntPtr CudaPerThreadDefaultStream = new IntPtr(2);

    [StructLayout(LayoutKind.Sequential)]
    internal struct DLPackVersion
    {
        public uint Major;
        public uint Minor;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct DLDevice
    {
        public int DeviceType;
        public int DeviceId;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct DLDataType
    {
        public byte Code;
        public byte Bits;
        public ushort Lanes;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct DLTensor
    {
        public IntPtr Data;
        public DLDevice Device;
        public int NDim;
        public DLDataType DType;
        public IntPtr Shape;     // int64_t*
        public IntPtr Strides;   // int64_t*, in elements
        public ulong ByteOffset;
    }

    /// <summary>The legacy (pre-v0.8, unversioned) managed tensor: <c>torch.utils.dlpack.to_dlpack</c>'s capsule.</summary>
    [StructLayout(LayoutKind.Sequential)]
    internal struct DLManagedTensor
    {
        public DLTensor DLTensor;
        public IntPtr ManagerCtx;
        public IntPtr Deleter;   // void (*)(DLManagedTensor*)
    }

    /// <summary>The current exchange structure. Everything up to <see cref="Flags"/> is ABI-stable across versions.</summary>
    [StructLayout(LayoutKind.Sequential)]
    internal struct DLManagedTensorVersioned
    {
        public DLPackVersion Version;
        public IntPtr ManagerCtx;
        public IntPtr Deleter;   // void (*)(DLManagedTensorVersioned*)
        public ulong Flags;
        public DLTensor DLTensor;
    }

#if !NET5_0_OR_GREATER
    /// <summary>The export deleter's delegate type on targets without UnmanagedCallersOnly.</summary>
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    internal delegate void DeleterFn(IntPtr self);
#endif

    /// <summary>
    /// Calls a producer's C deleter on <paramref name="self"/>, as a raw unmanaged call on every target. On .NET
    /// Framework, GetDelegateForFunctionPointer on a pointer that came from a managed delegate hands back that delegate
    /// and throws InvalidCastException when its type is not the one asked for, so a capsule from another managed
    /// producer could never be released there.
    /// </summary>
    internal static unsafe void InvokeDeleter(IntPtr deleter, IntPtr self)
    {
        if (deleter == IntPtr.Zero) return;
        ((delegate* unmanaged[Cdecl]<IntPtr, void>)deleter)(self);
    }
}
