// Copyright (c) AiDotNet. All rights reserved.

namespace AiDotNet.Tensors.Engines.DirectGpu;

internal static class CompressedMomentHostFallback
{
    private static readonly System.Collections.Generic.HashSet<string> _warnedBackends = new();

    /// <summary>
    /// Emits a one-time warning that this backend runs the compressed-moment (bf16/int8) optimizer
    /// through a HOST fallback (per-step download → CPU compute → upload), NOT a native GPU kernel.
    /// Backends whose compressed-moment methods delegate here (Metal, Vulkan) call this so the host
    /// round-trip is transparent instead of silently masquerading as native GPU acceleration — the
    /// concern raised on the #719 Metal/Vulkan review threads. Native kernels are tracked follow-up.
    /// </summary>
    public static void WarnHostFallbackOnce(string backendName)
    {
        lock (_warnedBackends)
        {
            if (!_warnedBackends.Add(backendName)) return;
        }
        System.Diagnostics.Trace.TraceWarning(
            backendName + ": compressed-moment (bf16/int8) optimizer runs via a HOST fallback " +
            "(per-step download → CPU compute → upload), NOT a native GPU kernel — expect reduced " +
            "throughput vs full-precision GPU moments. Native kernels are tracked as follow-up.");
    }

    public static byte[] DownloadPackedBytes(float[] words, int byteCount)
    {
        var allBytes = new byte[words.Length * sizeof(float)];
        Buffer.BlockCopy(words, 0, allBytes, 0, allBytes.Length);
        if (allBytes.Length == byteCount)
            return allBytes;

        var bytes = new byte[byteCount];
        Array.Copy(allBytes, bytes, byteCount);
        return bytes;
    }

    public static float[] PackBytesAsFloats(byte[] bytes)
    {
        var words = new float[(bytes.Length + sizeof(float) - 1) / sizeof(float)];
        Buffer.BlockCopy(bytes, 0, words, 0, bytes.Length);
        return words;
    }

    public static void AdamBf16(
        float[] param,
        float[] grad,
        byte[] mBytes,
        byte[] vBytes,
        float learningRate,
        float beta1,
        float beta2,
        float epsilon,
        float weightDecay,
        int step,
        int size,
        bool decoupledWeightDecay)
    {
        float bc1 = 1f - MathF.Pow(beta1, step);
        float bc2 = 1f - MathF.Pow(beta2, step);
        bool firstStep = step <= 1;

        for (int i = 0; i < size; i++)
        {
            if (decoupledWeightDecay && weightDecay > 0f)
                param[i] *= 1f - learningRate * weightDecay;

            float g = decoupledWeightDecay ? grad[i] : grad[i] + weightDecay * param[i];
            float oldM = firstStep ? 0f : Bf16ToFloat(ReadUInt16(mBytes, i));
            float oldV = firstStep ? 0f : Bf16ToFloat(ReadUInt16(vBytes, i));
            float newM = beta1 * oldM + (1f - beta1) * g;
            float newV = beta2 * oldV + (1f - beta2) * g * g;
            WriteUInt16(mBytes, i, FloatToBf16Rne(newM));
            WriteUInt16(vBytes, i, FloatToBf16Rne(newV));

            float mHat = newM / bc1;
            float vHat = newV / bc2;
            param[i] -= learningRate * mHat / (MathF.Sqrt(vHat) + epsilon);
        }
    }

    public static void Adam8Bit(
        float[] param,
        float[] grad,
        byte[] mQuant,
        byte[] vQuant,
        float[] mScales,
        float[] vScales,
        float learningRate,
        float beta1,
        float beta2,
        float epsilon,
        float oneMinusBeta1,
        float oneMinusBeta2,
        float biasCorrection1,
        float biasCorrection2,
        int blockSize,
        int paramLength,
        int numBlocks)
    {
        // Both corrections must indicate step 1: bc1 <= (1-beta1) alone is ALWAYS true when beta1==0
        // (bc1 stays 1 for every step), which would wrongly discard v history each step for a
        // beta1==0/beta2>0 config. Requiring bc2 to agree makes step detection robust.
        bool firstStep = biasCorrection1 <= oneMinusBeta1 + 1e-7f && biasCorrection2 <= oneMinusBeta2 + 1e-7f;
        for (int block = 0; block < numBlocks; block++)
        {
            int start = block * blockSize;
            int end = Math.Min(start + blockSize, paramLength);
            float oldMScale = firstStep ? 0f : mScales[block];
            float oldVScale = firstStep ? 0f : vScales[block];
            float maxM = 0f;
            float maxV = 0f;

            for (int i = start; i < end; i++)
            {
                float oldM = firstStep ? 0f : (mQuant[i] - 128) * oldMScale;
                float oldV = firstStep ? 0f : vQuant[i] * oldVScale;
                float g = grad[i];
                float newM = beta1 * oldM + oneMinusBeta1 * g;
                float newV = beta2 * oldV + oneMinusBeta2 * g * g;
                maxM = MathF.Max(maxM, MathF.Abs(newM));
                maxV = MathF.Max(maxV, MathF.Abs(newV));
            }

            float newMScale = MathF.Max(maxM / 127f, 1e-10f);
            float newVScale = MathF.Max(maxV / 255f, 1e-10f);
            mScales[block] = newMScale;
            vScales[block] = newVScale;

            for (int i = start; i < end; i++)
            {
                float oldM = firstStep ? 0f : (mQuant[i] - 128) * oldMScale;
                float oldV = firstStep ? 0f : vQuant[i] * oldVScale;
                float g = grad[i];
                float newM = beta1 * oldM + oneMinusBeta1 * g;
                float newV = beta2 * oldV + oneMinusBeta2 * g * g;

                float mHat = newM / biasCorrection1;
                float vHat = newV / biasCorrection2;
                param[i] -= learningRate * mHat / (MathF.Sqrt(vHat) + epsilon);

                int qm = (int)Math.Round(newM / newMScale, MidpointRounding.ToEven);
                if (qm < -127) qm = -127;
                if (qm > 127) qm = 127;
                mQuant[i] = (byte)(qm + 128);

                int qv = (int)Math.Round(newV / newVScale, MidpointRounding.ToEven);
                if (qv < 0) qv = 0;
                if (qv > 255) qv = 255;
                vQuant[i] = (byte)qv;
            }
        }
    }

    private static ushort ReadUInt16(byte[] bytes, int elementIndex)
    {
        int offset = elementIndex * sizeof(ushort);
        return (ushort)(bytes[offset] | (bytes[offset + 1] << 8));
    }

    private static void WriteUInt16(byte[] bytes, int elementIndex, ushort value)
    {
        int offset = elementIndex * sizeof(ushort);
        bytes[offset] = (byte)value;
        bytes[offset + 1] = (byte)(value >> 8);
    }

    private static unsafe float Bf16ToFloat(ushort value)
    {
        uint bits = (uint)value << 16;
        return *(float*)&bits;
    }

    private static unsafe ushort FloatToBf16Rne(float value)
    {
        uint bits = *(uint*)&value;
        if ((bits & 0x7FFFFFFFu) > 0x7F800000u)
            return (ushort)((bits >> 16) | 0x0040u);

        uint rounding = 0x7FFFu + ((bits >> 16) & 1u);
        return (ushort)((bits + rounding) >> 16);
    }
}
