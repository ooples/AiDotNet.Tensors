// Copyright (c) AiDotNet. All rights reserved.
// Metal native bindings using Objective-C runtime interop.
// Provides P/Invoke declarations for Metal framework on macOS/iOS.

using System;
using System.Runtime.InteropServices;

namespace AiDotNet.Tensors.Engines.DirectGpu.Metal;

/// <summary>
/// Native P/Invoke bindings for Apple's Metal framework via Objective-C runtime.
/// </summary>
/// <remarks>
/// <para><b>Architecture:</b></para>
/// <para>
/// Metal is an Objective-C API, so we use the Objective-C runtime to call methods.
/// This approach allows pure P/Invoke without requiring managed Objective-C bindings.
/// </para>
/// <para><b>Key Functions:</b></para>
/// <list type="bullet">
/// <item>objc_msgSend - Call Objective-C methods</item>
/// <item>objc_getClass - Get Objective-C class by name</item>
/// <item>sel_registerName - Register/get selector</item>
/// </list>
/// </remarks>
public static class MetalNativeBindings
{
    #region Platform Detection

    /// <summary>
    /// Checks if we're running on a platform that supports Metal (macOS, iOS, tvOS).
    /// </summary>
    public static bool IsPlatformSupported
    {
        get
        {
            if (!RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
            {
                return false;
            }

            // Check macOS version - Metal requires macOS 10.11+
            // Note: On .NET 5+, Environment.OSVersion returns Darwin kernel version, not macOS marketing version.
            // Use OperatingSystem.IsMacOSVersionAtLeast when available for correct detection.
            try
            {
#if NET5_0_OR_GREATER
                return OperatingSystem.IsMacOSVersionAtLeast(10, 11);
#else
                // Fallback for .NET Framework: Darwin 15.x maps to macOS 10.11 (El Capitan)
                var version = Environment.OSVersion.Version;
                return version.Major >= 15;
#endif
            }
            catch
            {
                return false;
            }
        }
    }

    #endregion

    #region Objective-C Runtime

    private const string LibObjc = "/usr/lib/libobjc.dylib";
    private const string LibMetal = "/System/Library/Frameworks/Metal.framework/Metal";
    private const string LibMetalPerformanceShaders = "/System/Library/Frameworks/MetalPerformanceShaders.framework/MetalPerformanceShaders";
    private const string LibFoundation = "/System/Library/Frameworks/Foundation.framework/Foundation";

    /// <summary>
    /// Gets an Objective-C class by name.
    /// </summary>
    [DllImport(LibObjc, EntryPoint = "objc_getClass")]
    public static extern IntPtr GetClass([MarshalAs(UnmanagedType.LPStr)] string name);

    /// <summary>
    /// Registers or retrieves a selector by name.
    /// </summary>
    [DllImport(LibObjc, EntryPoint = "sel_registerName")]
    public static extern IntPtr RegisterSelector([MarshalAs(UnmanagedType.LPStr)] string name);

    /// <summary>
    /// Sends a message to an Objective-C object (returns IntPtr).
    /// </summary>
    [DllImport(LibObjc, EntryPoint = "objc_msgSend")]
    public static extern IntPtr SendMessage(IntPtr receiver, IntPtr selector);

    /// <summary>
    /// Sends a message with one IntPtr argument.
    /// </summary>
    [DllImport(LibObjc, EntryPoint = "objc_msgSend")]
    public static extern IntPtr SendMessage(IntPtr receiver, IntPtr selector, IntPtr arg1);

    /// <summary>
    /// Sends a message with two IntPtr arguments.
    /// </summary>
    [DllImport(LibObjc, EntryPoint = "objc_msgSend")]
    public static extern IntPtr SendMessage(IntPtr receiver, IntPtr selector, IntPtr arg1, IntPtr arg2);

    /// <summary>
    /// Sends a message with three IntPtr arguments.
    /// </summary>
    [DllImport(LibObjc, EntryPoint = "objc_msgSend")]
    public static extern IntPtr SendMessage(IntPtr receiver, IntPtr selector, IntPtr arg1, IntPtr arg2, IntPtr arg3);

    /// <summary>
    /// Sends a message with an IntPtr, IntPtr, and ref IntPtr argument (for NSError** out-parameters).
    /// </summary>
    [DllImport(LibObjc, EntryPoint = "objc_msgSend")]
    public static extern IntPtr SendMessageWithError(IntPtr receiver, IntPtr selector, IntPtr arg1, IntPtr arg2, ref IntPtr errorOut);

    /// <summary>
    /// Sends a message with an IntPtr and ref IntPtr argument (for NSError** out-parameters).
    /// </summary>
    [DllImport(LibObjc, EntryPoint = "objc_msgSend")]
    public static extern IntPtr SendMessageWithError(IntPtr receiver, IntPtr selector, IntPtr arg1, ref IntPtr errorOut);

    /// <summary>
    /// Sends a message with an IntPtr and ulong argument.
    /// </summary>
    [DllImport(LibObjc, EntryPoint = "objc_msgSend")]
    public static extern IntPtr SendMessage(IntPtr receiver, IntPtr selector, IntPtr arg1, ulong arg2);

    /// <summary>
    /// Sends a message with a ulong argument (returns IntPtr).
    /// </summary>
    [DllImport(LibObjc, EntryPoint = "objc_msgSend")]
    public static extern IntPtr SendMessageULong(IntPtr receiver, IntPtr selector, ulong arg1);

    /// <summary>
    /// Sends a message with two ulong arguments.
    /// </summary>
    [DllImport(LibObjc, EntryPoint = "objc_msgSend")]
    public static extern IntPtr SendMessageULong2(IntPtr receiver, IntPtr selector, ulong arg1, ulong arg2);

    /// <summary>
    /// Sends a message that returns a ulong.
    /// </summary>
    [DllImport(LibObjc, EntryPoint = "objc_msgSend")]
    public static extern ulong SendMessageULongReturn(IntPtr receiver, IntPtr selector);

    /// <summary>
    /// Sends a message that returns a bool.
    /// </summary>
    [DllImport(LibObjc, EntryPoint = "objc_msgSend")]
    [return: MarshalAs(UnmanagedType.I1)]
    public static extern bool SendMessageBool(IntPtr receiver, IntPtr selector);

    /// <summary>
    /// Sends a message with IntPtr arg that returns bool.
    /// </summary>
    [DllImport(LibObjc, EntryPoint = "objc_msgSend")]
    [return: MarshalAs(UnmanagedType.I1)]
    public static extern bool SendMessageBool(IntPtr receiver, IntPtr selector, IntPtr arg1);

    /// <summary>
    /// Sends a message with pointer and length arguments (for buffer contents).
    /// </summary>
    [DllImport(LibObjc, EntryPoint = "objc_msgSend")]
    public static extern IntPtr SendMessagePtr(IntPtr receiver, IntPtr selector, IntPtr pointer, ulong length, ulong options);

    /// <summary>
    /// Sends a message that returns void.
    /// </summary>
    [DllImport(LibObjc, EntryPoint = "objc_msgSend")]
    public static extern void SendMessageVoid(IntPtr receiver, IntPtr selector);

    /// <summary>
    /// Sends a message with IntPtr arg that returns void.
    /// </summary>
    [DllImport(LibObjc, EntryPoint = "objc_msgSend")]
    public static extern void SendMessageVoid(IntPtr receiver, IntPtr selector, IntPtr arg1);

    /// <summary>
    /// Sends a message with two IntPtr args that returns void.
    /// </summary>
    [DllImport(LibObjc, EntryPoint = "objc_msgSend")]
    public static extern void SendMessageVoid(IntPtr receiver, IntPtr selector, IntPtr arg1, IntPtr arg2);

    /// <summary>
    /// Sends a message with ulong arg that returns void.
    /// </summary>
    [DllImport(LibObjc, EntryPoint = "objc_msgSend")]
    public static extern void SendMessageVoidULong(IntPtr receiver, IntPtr selector, ulong arg1);

    /// <summary>
    /// Sends a message for dispatch with grid/threadgroup sizes.
    /// Maps to dispatchThreadgroups:threadsPerThreadgroup: which takes two MTLSize arguments.
    /// </summary>
    [DllImport(LibObjc, EntryPoint = "objc_msgSend")]
    public static extern void SendMessageDispatch(
        IntPtr receiver,
        IntPtr selector,
        MTLSize gridSize,
        MTLSize threadgroupSize);

    /// <summary>
    /// Sends a message to set buffer at index.
    /// </summary>
    [DllImport(LibObjc, EntryPoint = "objc_msgSend")]
    public static extern void SendMessageSetBuffer(
        IntPtr receiver,
        IntPtr selector,
        IntPtr buffer,
        ulong offset,
        ulong index);

    /// <summary>
    /// Sends a message to set bytes at index.
    /// </summary>
    [DllImport(LibObjc, EntryPoint = "objc_msgSend")]
    public static extern void SendMessageSetBytes(
        IntPtr receiver,
        IntPtr selector,
        IntPtr bytes,
        ulong length,
        ulong index);

    /// <summary>
    /// Sends a message to set threadgroup memory length at index.
    /// </summary>
    [DllImport(LibObjc, EntryPoint = "objc_msgSend")]
    public static extern void SendMessageSetThreadgroupMemory(
        IntPtr receiver,
        IntPtr selector,
        ulong length,
        ulong index);

    /// <summary>
    /// Retains an Objective-C object.
    /// </summary>
    [DllImport(LibObjc, EntryPoint = "objc_retain")]
    public static extern IntPtr Retain(IntPtr obj);

    /// <summary>
    /// Releases an Objective-C object.
    /// </summary>
    [DllImport(LibObjc, EntryPoint = "objc_release")]
    public static extern void Release(IntPtr obj);

    #endregion

    #region Metal Types

    /// <summary>
    /// Metal size structure (width, height, depth).
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct MTLSize
    {
        public ulong Width;
        public ulong Height;
        public ulong Depth;

        public MTLSize(ulong width, ulong height, ulong depth)
        {
            Width = width;
            Height = height;
            Depth = depth;
        }

        public static MTLSize Create(int width, int height = 1, int depth = 1)
        {
            return new MTLSize((ulong)width, (ulong)height, (ulong)depth);
        }
    }

    /// <summary>
    /// Metal origin structure (x, y, z).
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct MTLOrigin
    {
        public ulong X;
        public ulong Y;
        public ulong Z;

        public MTLOrigin(ulong x, ulong y, ulong z)
        {
            X = x;
            Y = y;
            Z = z;
        }
    }

    /// <summary>
    /// Metal region structure (origin + size).
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct MTLRegion
    {
        public MTLOrigin Origin;
        public MTLSize Size;
    }

    /// <summary>
    /// Metal resource options for buffer allocation.
    /// </summary>
    [Flags]
    public enum MTLResourceOptions : ulong
    {
        /// <summary>
        /// Default CPU cache mode.
        /// </summary>
        CPUCacheModeDefaultCache = 0,

        /// <summary>
        /// CPU writes are combined, good for streaming data to GPU.
        /// </summary>
        CPUCacheModeWriteCombined = 1,

        /// <summary>
        /// Resource stored in shared memory accessible by both CPU and GPU.
        /// Best for frequently updated buffers on Apple Silicon.
        /// </summary>
        StorageModeShared = 0 << 4,

        /// <summary>
        /// Resource stored in GPU-private memory.
        /// Best for GPU-only resources on discrete GPUs.
        /// </summary>
        StorageModePrivate = 1 << 4,

        /// <summary>
        /// Resource is managed by Metal, synchronized between CPU and GPU.
        /// </summary>
        StorageModeManaged = 2 << 4,

        /// <summary>
        /// Hazard tracking is untracked (manual synchronization).
        /// </summary>
        HazardTrackingModeUntracked = 1 << 8
    }

    /// <summary>
    /// Metal command buffer status.
    /// </summary>
    public enum MTLCommandBufferStatus : ulong
    {
        NotEnqueued = 0,
        Enqueued = 1,
        Committed = 2,
        Scheduled = 3,
        Completed = 4,
        Error = 5
    }

    /// <summary>
    /// Metal GPU family for feature detection.
    /// </summary>
    public enum MTLGPUFamily : long
    {
        Apple1 = 1001,
        Apple2 = 1002,
        Apple3 = 1003,
        Apple4 = 1004,
        Apple5 = 1005,
        Apple6 = 1006,
        Apple7 = 1007,
        Apple8 = 1008,
        Apple9 = 1009,
        Mac1 = 2001,
        Mac2 = 2002,
        Common1 = 3001,
        Common2 = 3002,
        Common3 = 3003,
        Metal3 = 5001
    }

    #endregion

    #region Cached Selectors

    /// <summary>
    /// Cached selectors for frequently used Metal methods.
    /// </summary>
    public static class Selectors
    {
        // Device selectors
        public static readonly IntPtr CreateSystemDefaultDevice = RegisterSelector("MTLCreateSystemDefaultDevice");
        public static readonly IntPtr Name = RegisterSelector("name");
        public static readonly IntPtr MaxThreadsPerThreadgroup = RegisterSelector("maxThreadsPerThreadgroup");
        public static readonly IntPtr MaxThreadgroupMemoryLength = RegisterSelector("maxThreadgroupMemoryLength");
        public static readonly IntPtr RecommendedMaxWorkingSetSize = RegisterSelector("recommendedMaxWorkingSetSize");
        public static readonly IntPtr CurrentAllocatedSize = RegisterSelector("currentAllocatedSize");
        public static readonly IntPtr SupportsFamily = RegisterSelector("supportsFamily:");
        public static readonly IntPtr RegistryID = RegisterSelector("registryID");

        // Buffer selectors
        public static readonly IntPtr NewBufferWithLength = RegisterSelector("newBufferWithLength:options:");
        public static readonly IntPtr NewBufferWithBytes = RegisterSelector("newBufferWithBytes:length:options:");
        public static readonly IntPtr Contents = RegisterSelector("contents");
        public static readonly IntPtr Length = RegisterSelector("length");
        public static readonly IntPtr DidModifyRange = RegisterSelector("didModifyRange:");

        // Command queue selectors
        public static readonly IntPtr NewCommandQueue = RegisterSelector("newCommandQueue");
        public static readonly IntPtr NewCommandQueueWithMaxCommandBufferCount = RegisterSelector("newCommandQueueWithMaxCommandBufferCount:");
        public static readonly IntPtr CommandBuffer = RegisterSelector("commandBuffer");
        public static readonly IntPtr CommandBufferWithUnretainedReferences = RegisterSelector("commandBufferWithUnretainedReferences");

        // Command buffer selectors
        public static readonly IntPtr ComputeCommandEncoder = RegisterSelector("computeCommandEncoder");
        public static readonly IntPtr ComputeCommandEncoderWithDispatchType = RegisterSelector("computeCommandEncoderWithDispatchType:");
        public static readonly IntPtr BlitCommandEncoder = RegisterSelector("blitCommandEncoder");
        public static readonly IntPtr Commit = RegisterSelector("commit");
        public static readonly IntPtr WaitUntilCompleted = RegisterSelector("waitUntilCompleted");
        public static readonly IntPtr WaitUntilScheduled = RegisterSelector("waitUntilScheduled");
        public static readonly IntPtr Status = RegisterSelector("status");
        public static readonly IntPtr Error = RegisterSelector("error");
        public static readonly IntPtr Enqueue = RegisterSelector("enqueue");
        public static readonly IntPtr PresentDrawable = RegisterSelector("presentDrawable:");
        public static readonly IntPtr AddCompletedHandler = RegisterSelector("addCompletedHandler:");

        // Compute encoder selectors
        public static readonly IntPtr SetComputePipelineState = RegisterSelector("setComputePipelineState:");
        public static readonly IntPtr SetBuffer = RegisterSelector("setBuffer:offset:atIndex:");
        public static readonly IntPtr SetBytes = RegisterSelector("setBytes:length:atIndex:");
        public static readonly IntPtr SetTexture = RegisterSelector("setTexture:atIndex:");
        public static readonly IntPtr SetThreadgroupMemoryLength = RegisterSelector("setThreadgroupMemoryLength:atIndex:");
        public static readonly IntPtr DispatchThreadgroups = RegisterSelector("dispatchThreadgroups:threadsPerThreadgroup:");
        public static readonly IntPtr DispatchThreads = RegisterSelector("dispatchThreads:threadsPerThreadgroup:");
        public static readonly IntPtr EndEncoding = RegisterSelector("endEncoding");
        public static readonly IntPtr MemoryBarrierWithScope = RegisterSelector("memoryBarrierWithScope:");
        public static readonly IntPtr MemoryBarrierWithResources = RegisterSelector("memoryBarrierWithResources:count:");

        // Blit encoder selectors
        public static readonly IntPtr CopyFromBuffer = RegisterSelector("copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:");
        public static readonly IntPtr FillBuffer = RegisterSelector("fillBuffer:range:value:");
        public static readonly IntPtr SynchronizeResource = RegisterSelector("synchronizeResource:");

        // Library/function selectors
        public static readonly IntPtr NewLibraryWithSource = RegisterSelector("newLibraryWithSource:options:error:");
        public static readonly IntPtr NewLibraryWithData = RegisterSelector("newLibraryWithData:error:");
        public static readonly IntPtr NewDefaultLibrary = RegisterSelector("newDefaultLibrary");
        public static readonly IntPtr NewFunctionWithName = RegisterSelector("newFunctionWithName:");
        public static readonly IntPtr FunctionNames = RegisterSelector("functionNames");

        // Pipeline selectors
        public static readonly IntPtr NewComputePipelineStateWithFunction = RegisterSelector("newComputePipelineStateWithFunction:error:");
        public static readonly IntPtr NewComputePipelineStateWithDescriptor = RegisterSelector("newComputePipelineStateWithDescriptor:options:reflection:error:");
        public static readonly IntPtr MaxTotalThreadsPerThreadgroup = RegisterSelector("maxTotalThreadsPerThreadgroup");
        public static readonly IntPtr ThreadExecutionWidth = RegisterSelector("threadExecutionWidth");
        public static readonly IntPtr StaticThreadgroupMemoryLength = RegisterSelector("staticThreadgroupMemoryLength");

        // NSString selectors
        public static readonly IntPtr StringWithUTF8String = RegisterSelector("stringWithUTF8String:");
        public static readonly IntPtr UTF8String = RegisterSelector("UTF8String");

        // NSError selectors
        public static readonly IntPtr LocalizedDescription = RegisterSelector("localizedDescription");
        public static readonly IntPtr Code = RegisterSelector("code");
        public static readonly IntPtr Domain = RegisterSelector("domain");

        // NSArray selectors
        public static readonly IntPtr Count = RegisterSelector("count");
        public static readonly IntPtr ObjectAtIndex = RegisterSelector("objectAtIndex:");

        // General selectors
        public static readonly IntPtr Alloc = RegisterSelector("alloc");
        public static readonly IntPtr Init = RegisterSelector("init");
        public static readonly IntPtr Release = RegisterSelector("release");
        public static readonly IntPtr Retain = RegisterSelector("retain");
        public static readonly IntPtr Autorelease = RegisterSelector("autorelease");
        public static readonly IntPtr Description = RegisterSelector("description");
    }

    /// <summary>
    /// Cached class references for Metal types.
    /// </summary>
    public static class Classes
    {
        public static readonly IntPtr NSString = GetClass("NSString");
        public static readonly IntPtr NSError = GetClass("NSError");
        public static readonly IntPtr NSArray = GetClass("NSArray");
        public static readonly IntPtr MTLCompileOptions = GetClass("MTLCompileOptions");
        public static readonly IntPtr MTLComputePipelineDescriptor = GetClass("MTLComputePipelineDescriptor");
    }

    #endregion

    #region Helper Methods

    /// <summary>
    /// Creates an NSString from a C# string.
    /// </summary>
    public static IntPtr CreateNSString(string str)
    {
        if (string.IsNullOrEmpty(str))
        {
            return IntPtr.Zero;
        }

        var utf8Bytes = System.Text.Encoding.UTF8.GetBytes(str + '\0');
        var handle = GCHandle.Alloc(utf8Bytes, GCHandleType.Pinned);
        try
        {
            return SendMessage(Classes.NSString, Selectors.StringWithUTF8String, handle.AddrOfPinnedObject());
        }
        finally
        {
            handle.Free();
        }
    }

    /// <summary>
    /// Gets a C# string from an NSString.
    /// </summary>
    public static string? GetStringFromNSString(IntPtr nsString)
    {
        if (nsString == IntPtr.Zero)
        {
            return null;
        }

        var utf8Ptr = SendMessage(nsString, Selectors.UTF8String);
        if (utf8Ptr == IntPtr.Zero)
        {
            return null;
        }

        return PtrToStringUTF8Compat(utf8Ptr);
    }

    /// <summary>
    /// .NET Framework compatible implementation of Marshal.PtrToStringUTF8.
    /// </summary>
    private static unsafe string? PtrToStringUTF8Compat(IntPtr ptr)
    {
        if (ptr == IntPtr.Zero)
            return null;

        // Find the length of the UTF-8 string by looking for null terminator
        byte* bytePtr = (byte*)ptr;
        int length = 0;
        while (bytePtr[length] != 0)
            length++;

        if (length == 0)
            return string.Empty;

        // Convert UTF-8 bytes to string
        return System.Text.Encoding.UTF8.GetString(bytePtr, length);
    }

    /// <summary>
    /// Gets the error description from an NSError.
    /// </summary>
    public static string? GetErrorDescription(IntPtr nsError)
    {
        if (nsError == IntPtr.Zero)
        {
            return null;
        }

        var description = SendMessage(nsError, Selectors.LocalizedDescription);
        return GetStringFromNSString(description);
    }

    /// <summary>
    /// Creates the default Metal device.
    /// </summary>
    public static IntPtr CreateDefaultDevice()
    {
        // MTLCreateSystemDefaultDevice is a C function, not an Objective-C method
        return MTLCreateSystemDefaultDevice();
    }

    [DllImport(LibMetal, EntryPoint = "MTLCreateSystemDefaultDevice")]
    private static extern IntPtr MTLCreateSystemDefaultDevice();

    /// <summary>
    /// Copies all Metal devices on the system.
    /// </summary>
    [DllImport(LibMetal, EntryPoint = "MTLCopyAllDevices")]
    public static extern IntPtr CopyAllDevices();

    #endregion

    #region Metal Performance Shaders (MPS)

    /// <summary>
    /// MPS matrix descriptor creation.
    /// </summary>
    public static class MPS
    {
        public static readonly IntPtr MPSMatrixDescriptor = GetClass("MPSMatrixDescriptor");
        public static readonly IntPtr MPSMatrix = GetClass("MPSMatrix");
        public static readonly IntPtr MPSMatrixMultiplication = GetClass("MPSMatrixMultiplication");
        public static readonly IntPtr MPSMatrixVectorMultiplication = GetClass("MPSMatrixVectorMultiplication");
        public static readonly IntPtr MPSMatrixSoftMax = GetClass("MPSMatrixSoftMax");
        public static readonly IntPtr MPSMatrixLogSoftMax = GetClass("MPSMatrixLogSoftMax");
        public static readonly IntPtr MPSMatrixNeuron = GetClass("MPSMatrixNeuron");
        public static readonly IntPtr MPSMatrixNeuronGradient = GetClass("MPSMatrixNeuronGradient");
        public static readonly IntPtr MPSCNNConvolution = GetClass("MPSCNNConvolution");
        public static readonly IntPtr MPSCNNConvolutionDescriptor = GetClass("MPSCNNConvolutionDescriptor");
        public static readonly IntPtr MPSCNNPoolingMax = GetClass("MPSCNNPoolingMax");
        public static readonly IntPtr MPSCNNPoolingAverage = GetClass("MPSCNNPoolingAverage");
        public static readonly IntPtr MPSCNNNeuronReLU = GetClass("MPSCNNNeuronReLU");
        public static readonly IntPtr MPSCNNNeuronSigmoid = GetClass("MPSCNNNeuronSigmoid");
        public static readonly IntPtr MPSCNNNeuronTanH = GetClass("MPSCNNNeuronTanH");

        // MPS selectors
        public static readonly IntPtr MatrixDescriptorWithRows = RegisterSelector("matrixDescriptorWithRows:columns:rowBytes:dataType:");
        public static readonly IntPtr InitWithBuffer = RegisterSelector("initWithBuffer:descriptor:");
        public static readonly IntPtr InitWithDevice = RegisterSelector("initWithDevice:");
        public static readonly IntPtr InitWithDeviceTransposeLeft = RegisterSelector("initWithDevice:transposeLeft:transposeRight:resultRows:resultColumns:interiorColumns:alpha:beta:");
        public static readonly IntPtr EncodeToCommandBuffer = RegisterSelector("encodeToCommandBuffer:leftMatrix:rightMatrix:resultMatrix:");
        public static readonly IntPtr SetNeuronType = RegisterSelector("setNeuronType:parameterA:parameterB:");
    }

    /// <summary>
    /// MPS data types.
    /// </summary>
    public enum MPSDataType : uint
    {
        Float32 = 0x10000000 | 32,
        Float16 = 0x10000000 | 16,
        Int32 = 0x20000000 | 32,
        Int16 = 0x20000000 | 16,
        Int8 = 0x20000000 | 8,
        UInt32 = 32,
        UInt16 = 16,
        UInt8 = 8
    }

    /// <summary>
    /// MPS neuron types for activation functions.
    /// </summary>
    public enum MPSCNNNeuronType : int
    {
        None = 0,
        ReLU = 1,
        Linear = 2,
        Sigmoid = 3,
        HardSigmoid = 4,
        TanH = 5,
        Absolute = 6,
        SoftPlus = 7,
        SoftSign = 8,
        ELU = 9,
        PReLU = 10,
        ReLUN = 11,
        Power = 12,
        Exponential = 13,
        Logarithm = 14,
        GeLU = 15
    }

    #endregion
}
