// Copyright (c) AiDotNet. All rights reserved.
using System;
using K = AiDotNet.Tensors.Engines.DirectGpu.BatchExecutionCudaKernels;

namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// <see cref="IFusedDecodeLinearBackend"/> for every backend without CUDA's direct-PTX and cuBLAS routes (#974), over
/// the shared <c>dl_p_</c> kernels in <see cref="K.DecodeLinearSource"/>: one dispatch per route, one thread per
/// output element, every operand resident.
/// </summary>
internal static class PortableFusedDecodeLinear
{
    /// <summary>
    /// The longest int8 dot product whose int32 accumulator cannot overflow: each product is at most 128 * 128 in
    /// magnitude, and 131,071 of them stay below 2^31.
    /// </summary>
    internal const int MaxW8A8InputFeatures = 131_071;

    private const int FusedRows16 = 16;

    private static void Positive(int value, string name)
    {
        if (value <= 0) throw new ArgumentOutOfRangeException(name, value, "Must be positive.");
    }

    private static void Bytes(IGpuBuffer buffer, long bytes, string name)
    {
        if (buffer is null) throw new ArgumentNullException(name);
        if (buffer.SizeInBytes < bytes)
            throw new ArgumentException($"{name} holds {buffer.SizeInBytes} bytes; the route reads or writes {bytes}.", name);
    }

    private static int Threads(long count, string what)
        => count <= int.MaxValue
            ? (int)count
            : throw new ArgumentOutOfRangeException(what, count, "Exceeds the kernels' int32 indexing.");

    /// <summary>
    /// The kernels read FP16 and int8 operands through 32-bit words, so the last element's word must exist in full.
    /// A buffer that ends mid-word is copied into a padded scratch (the extra bytes are never used: every element is
    /// masked out of its word). Returns the buffer to bind and, through <paramref name="scratch"/>, what to dispose.
    /// </summary>
    internal static IGpuBuffer WholeWords(IDirectGpuBackend backend, IGpuBuffer buffer, long bytes, out IGpuBuffer? scratch)
    {
        long padded = (bytes + 3) / 4 * 4;
        if (buffer.SizeInBytes >= padded)
        {
            scratch = null;
            return buffer;
        }
        scratch = backend.AllocateByteBuffer(Threads(padded, nameof(bytes)));
        try
        {
            PortableFp16Activation.CopyBytes(backend, buffer, scratch, bytes);
        }
        catch
        {
            scratch.Dispose();
            scratch = null;
            throw;
        }
        return scratch;
    }

    public static void Linear(PortableGenericFft.Launcher launch, IGpuBuffer input, IGpuBuffer outputMajorWeight, IGpuBuffer bias,
        IGpuBuffer output, int rows, int inFeatures, int outFeatures)
    {
        Positive(inFeatures, nameof(inFeatures));
        Positive(outFeatures, nameof(outFeatures));
        long outputs = (long)rows * outFeatures;
        Bytes(input, (long)rows * inFeatures * sizeof(float), nameof(input));
        Bytes(outputMajorWeight, (long)inFeatures * outFeatures * sizeof(float), nameof(outputMajorWeight));
        Bytes(bias, (long)outFeatures * sizeof(float), nameof(bias));
        Bytes(output, outputs * sizeof(float), nameof(output));
        Threads((long)inFeatures * outFeatures, nameof(outFeatures));
        launch(K.DlLinearGelu, Threads(outputs, nameof(outFeatures)), input, outputMajorWeight, bias, output, rows, inFeatures, outFeatures);
    }

    public static void LinearFp16(IDirectGpuBackend backend, PortableGenericFft.Launcher launch, IGpuBuffer inputHalf,
        IGpuBuffer outputMajorWeightHalf, IGpuBuffer biasFloat, IGpuBuffer outputFloat, int rows, int inFeatures, int outFeatures)
    {
        Positive(inFeatures, nameof(inFeatures));
        Positive(outFeatures, nameof(outFeatures));
        long outputs = (long)rows * outFeatures;
        long inputBytes = (long)rows * inFeatures * 2, weightBytes = (long)inFeatures * outFeatures * 2;
        Bytes(inputHalf, inputBytes, nameof(inputHalf));
        Bytes(outputMajorWeightHalf, weightBytes, nameof(outputMajorWeightHalf));
        Bytes(biasFloat, (long)outFeatures * sizeof(float), nameof(biasFloat));
        Bytes(outputFloat, outputs * sizeof(float), nameof(outputFloat));
        Threads((long)inFeatures * outFeatures, nameof(outFeatures));
        int threads = Threads(outputs, nameof(outFeatures));

        IGpuBuffer? inputScratch = null, weightScratch = null;
        try
        {
            var x = WholeWords(backend, inputHalf, inputBytes, out inputScratch);
            var w = WholeWords(backend, outputMajorWeightHalf, weightBytes, out weightScratch);
            launch(K.DlLinearGeluFp16, threads, x, w, biasFloat, outputFloat, rows, inFeatures, outFeatures);
        }
        finally
        {
            inputScratch?.Dispose();
            weightScratch?.Dispose();
        }
    }

    public static void LinearFp16Rows16(IDirectGpuBackend backend, PortableGenericFft.Launcher launch, IGpuBuffer inputHalf,
        IGpuBuffer outputMajorWeightHalf, IGpuBuffer biasFloat, IGpuBuffer outputFloat, int inFeatures, int outFeatures)
        => LinearFp16(backend, launch, inputHalf, outputMajorWeightHalf, biasFloat, outputFloat, FusedRows16, inFeatures, outFeatures);

    public static void LinearW8A8(IDirectGpuBackend backend, PortableGenericFft.Launcher launch, IGpuBuffer inputInt8,
        IGpuBuffer outputMajorWeightsInt8, IGpuBuffer activationScaleFloat, IGpuBuffer weightScalesFloat, IGpuBuffer biasFloat,
        IGpuBuffer outputFloat, int inputFeatures, int outputFeatures)
    {
        Positive(inputFeatures, nameof(inputFeatures));
        Positive(outputFeatures, nameof(outputFeatures));
        if (inputFeatures > MaxW8A8InputFeatures)
            throw new ArgumentOutOfRangeException(nameof(inputFeatures), inputFeatures,
                $"An int8 dot product longer than {MaxW8A8InputFeatures} can overflow its int32 accumulator.");
        long weightBytes = (long)inputFeatures * outputFeatures;
        long outputBytes = (long)outputFeatures * sizeof(float);
        Bytes(inputInt8, inputFeatures, nameof(inputInt8));
        Bytes(outputMajorWeightsInt8, weightBytes, nameof(outputMajorWeightsInt8));
        Bytes(activationScaleFloat, sizeof(float), nameof(activationScaleFloat));
        Bytes(weightScalesFloat, outputBytes, nameof(weightScalesFloat));
        Bytes(biasFloat, outputBytes, nameof(biasFloat));
        Bytes(outputFloat, outputBytes, nameof(outputFloat));
        Threads(weightBytes, nameof(outputFeatures));

        IGpuBuffer? inputScratch = null, weightScratch = null;
        try
        {
            var x = WholeWords(backend, inputInt8, inputFeatures, out inputScratch);
            var w = WholeWords(backend, outputMajorWeightsInt8, weightBytes, out weightScratch);
            launch(K.DlW8A8LinearGelu, outputFeatures, x, w, activationScaleFloat, weightScalesFloat, biasFloat, outputFloat,
                inputFeatures, outputFeatures);
        }
        finally
        {
            inputScratch?.Dispose();
            weightScratch?.Dispose();
        }
    }

    public static void W8A8Epilogue(PortableGenericFft.Launcher launch, IGpuBuffer accumulatorInt32, IGpuBuffer activationScaleFloat,
        IGpuBuffer weightScalesFloat, IGpuBuffer biasFloat, IGpuBuffer outputFloat, int outputFeatures)
    {
        Positive(outputFeatures, nameof(outputFeatures));
        long outputBytes = (long)outputFeatures * sizeof(float);
        Bytes(accumulatorInt32, outputBytes, nameof(accumulatorInt32));
        Bytes(activationScaleFloat, sizeof(float), nameof(activationScaleFloat));
        Bytes(weightScalesFloat, outputBytes, nameof(weightScalesFloat));
        Bytes(biasFloat, outputBytes, nameof(biasFloat));
        Bytes(outputFloat, outputBytes, nameof(outputFloat));
        launch(K.DlW8A8Epilogue, outputFeatures, accumulatorInt32, activationScaleFloat, weightScalesFloat, biasFloat, outputFloat,
            outputFeatures);
    }
}
