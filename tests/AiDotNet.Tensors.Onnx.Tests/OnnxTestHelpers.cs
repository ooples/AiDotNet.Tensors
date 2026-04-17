using AiDotNet.Tensors.Engines;
using Xunit;

namespace AiDotNet.Tensors.Onnx.Tests;

/// <summary>
/// Shared test helpers for ONNX operator parity tests.
/// </summary>
internal static class OnnxTestHelpers
{
    internal const float Tolerance = 1e-4f;
    internal const int FLOAT = 1;

    /// <summary>
    /// Runs a serialized ONNX model through our importer, feeds the named
    /// input tensors, executes, and returns a flattened copy of the output.
    /// </summary>
    internal static float[] ImportAndExecute(byte[] modelBytes, params (string name, float[] data)[] inputs)
    {
        using var stream = new MemoryStream(modelBytes);
        var engine = new CpuEngine();
        var result = OnnxImporter.Import<float>(stream, engine);
        Assert.Empty(result.UnsupportedOperators);
        Assert.NotNull(result.Plan);
        foreach (var (name, data) in inputs)
        {
            var t = result.Inputs[name];
            data.AsSpan().CopyTo(t.AsWritableSpan());
        }
        var output = result.Plan!.Execute();
        var arr = new float[output.AsSpan().Length];
        output.AsSpan().CopyTo(arr);
        return arr;
    }

    internal static float[] RandomArray(int seed, int n, float lo = -1f, float hi = 1f)
    {
        var rng = new Random(seed);
        var arr = new float[n];
        for (int i = 0; i < n; i++) arr[i] = lo + (float)rng.NextDouble() * (hi - lo);
        return arr;
    }

    /// <summary>
    /// Asserts <paramref name="expected"/> and <paramref name="actual"/>
    /// agree per-element within <c>Tolerance × max(1, |expected|)</c>.
    /// </summary>
    internal static void AssertClose(float[] expected, float[] actual, float? tolerance = null)
    {
        Assert.Equal(expected.Length, actual.Length);
        float tol = tolerance ?? Tolerance;
        for (int i = 0; i < expected.Length; i++)
        {
            float diff = Math.Abs(expected[i] - actual[i]);
            float scale = Math.Max(Math.Abs(expected[i]), 1f);
            Assert.True(diff <= tol * scale,
                $"Element {i}: expected {expected[i]}, got {actual[i]}, diff={diff}, tol={tol * scale}");
        }
    }
}
