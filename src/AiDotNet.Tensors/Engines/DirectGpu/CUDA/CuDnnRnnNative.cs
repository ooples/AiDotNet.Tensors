// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Runtime.InteropServices;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

/// <summary>
/// Raw P/Invoke bindings into <c>libcudnn</c>'s RNN entry points.
/// We already bind cuDNN Conv + BN elsewhere; this file adds the RNN /
/// LSTM / GRU surface — cuDNN's most-used "general" layer.
///
/// <para>The wrapper class <see cref="CuDnnRnn"/> owns the
/// rnnDescriptor lifecycle + persistent-kernel selection; this file
/// just declares the entry points.</para>
/// </summary>
internal static class CuDnnRnnNative
{
    private const string Lib = "cudnn64_9";

    public enum Status
    {
        Success = 0,
        NotInitialized = 1,
        AllocFailed = 2,
        BadParam = 3,
        InternalError = 4,
        InvalidValue = 5,
        ArchMismatch = 6,
        MappingError = 7,
        ExecutionFailed = 8,
        NotSupported = 9,
        LicenseError = 10,
        RuntimePrerequisiteMissing = 11,
        RuntimeInProgress = 12,
        RuntimeFpOverflow = 13,
    }

    public enum RnnInputMode { Linear = 0, Skip = 1 }
    public enum RnnDirection { Unidirectional = 0, Bidirectional = 1 }
    public enum RnnMode { RnnRelu = 0, RnnTanh = 1, Lstm = 2, Gru = 3 }
    public enum RnnAlgo { Standard = 0, PersistStatic = 1, PersistDynamic = 2 }
    public enum RnnDataLayout { SeqMajorUnpacked = 0, SeqMajorPacked = 1, BatchMajorUnpacked = 2 }
    public enum DataType { Float = 0, Double = 1, Half = 2, BFloat16 = 14 }

    [DllImport(Lib)] public static extern Status cudnnCreate(out IntPtr handle);
    [DllImport(Lib)] public static extern Status cudnnDestroy(IntPtr handle);

    [DllImport(Lib)] public static extern Status cudnnCreateRNNDescriptor(out IntPtr desc);
    [DllImport(Lib)] public static extern Status cudnnDestroyRNNDescriptor(IntPtr desc);

    [DllImport(Lib)]
    public static extern Status cudnnSetRNNDescriptor_v8(IntPtr rnnDesc,
        RnnAlgo algo, RnnMode cellMode, /* biasMode */ int biasMode,
        RnnDirection direction, RnnInputMode inputMode,
        DataType dataType, DataType mathPrec,
        /* mathType */ int mathType,
        int inputSize, int hiddenSize, int projSize, int numLayers,
        IntPtr dropoutDesc, /* auxFlags */ int auxFlags);

    [DllImport(Lib)]
    public static extern Status cudnnRNNForward(IntPtr handle, IntPtr rnnDesc,
        /* fwdMode */ int fwdMode, IntPtr devSeqLengths,
        IntPtr xDesc, IntPtr x, IntPtr yDesc, IntPtr y,
        IntPtr hDesc, IntPtr hx, IntPtr hy,
        IntPtr cDesc, IntPtr cx, IntPtr cy,
        ulong weightSpaceSize, IntPtr weightSpace,
        ulong workSpaceSize, IntPtr workSpace,
        ulong reserveSpaceSize, IntPtr reserveSpace);

    [DllImport(Lib)]
    public static extern Status cudnnRNNBackwardData_v8(IntPtr handle, IntPtr rnnDesc,
        IntPtr devSeqLengths,
        IntPtr yDesc, IntPtr y, IntPtr dy,
        IntPtr xDesc, IntPtr dx,
        IntPtr hDesc, IntPtr hx, IntPtr dhy, IntPtr dhx,
        IntPtr cDesc, IntPtr cx, IntPtr dcy, IntPtr dcx,
        ulong weightSpaceSize, IntPtr weightSpace,
        ulong workSpaceSize, IntPtr workSpace,
        ulong reserveSpaceSize, IntPtr reserveSpace);

    [DllImport(Lib)]
    public static extern Status cudnnRNNBackwardWeights_v8(IntPtr handle, IntPtr rnnDesc,
        /* addGrad */ int addGrad, IntPtr devSeqLengths,
        IntPtr xDesc, IntPtr x, IntPtr hDesc, IntPtr hx, IntPtr yDesc, IntPtr y,
        ulong weightSpaceSize, IntPtr dweightSpace,
        ulong workSpaceSize, IntPtr workSpace,
        ulong reserveSpaceSize, IntPtr reserveSpace);

    public static readonly bool IsAvailable = Probe();

    private static bool Probe()
    {
        try
        {
            var status = cudnnCreate(out var h);
            if (status == Status.Success) { cudnnDestroy(h); return true; }
            return false;
        }
        catch (DllNotFoundException) { return false; }
        catch (EntryPointNotFoundException) { return false; }
        catch (BadImageFormatException) { return false; }
        catch { return false; }
    }
}
