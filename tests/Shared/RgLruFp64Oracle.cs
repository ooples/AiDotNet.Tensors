using System;

namespace AiDotNet.Tensors.Testing;

internal static class RgLruFp64Oracle
{
    private const int SequenceLength = 128;
    private const int RecurrentDimension = 256;
    private const int SequenceElementCount = SequenceLength * RecurrentDimension;

    internal static double[] Evaluate(
        float[] value,
        float[] recurrence,
        float[] inputGate,
        float[] decay)
    {
        if (value == null)
            throw new ArgumentNullException(nameof(value));
        if (recurrence == null)
            throw new ArgumentNullException(nameof(recurrence));
        if (inputGate == null)
            throw new ArgumentNullException(nameof(inputGate));
        if (decay == null)
            throw new ArgumentNullException(nameof(decay));
        if (value.Length != SequenceElementCount)
            throw new ArgumentException($"Expected {SequenceElementCount} values.", nameof(value));
        if (recurrence.Length != SequenceElementCount)
            throw new ArgumentException($"Expected {SequenceElementCount} recurrence values.", nameof(recurrence));
        if (inputGate.Length != SequenceElementCount)
            throw new ArgumentException($"Expected {SequenceElementCount} input-gate values.", nameof(inputGate));
        if (decay.Length != RecurrentDimension)
            throw new ArgumentException($"Expected {RecurrentDimension} decay values.", nameof(decay));

        var output = new double[SequenceElementCount];
        for (int channel = 0; channel < RecurrentDimension; channel++)
        {
            double state = 0;
            double channelDecay = 1.0 / (1.0 + Math.Exp(decay[channel]));
            for (int timestep = 0; timestep < SequenceLength; timestep++)
            {
                int offset = timestep * RecurrentDimension + channel;
                double a = recurrence[offset] * channelDecay;
                double scale = Math.Sqrt(Math.Max(0, 1 - a * a));
                state = a * state + scale * inputGate[offset] * value[offset];
                output[offset] = state;
            }
        }

        return output;
    }
}
