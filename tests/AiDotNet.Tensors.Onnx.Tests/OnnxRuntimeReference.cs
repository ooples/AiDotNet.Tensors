using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace AiDotNet.Tensors.Onnx.Tests;

/// <summary>
/// Runs a serialized ONNX model through <c>Microsoft.ML.OnnxRuntime</c> and
/// returns its output(s). Used as the ground-truth reference for the
/// numerical-parity assertions — our importer's output must match ORT
/// within the tolerance spec'd in Issue #169 (1e-4).
/// </summary>
internal static class OnnxRuntimeReference
{
    internal static float[] RunSingleOutput(
        byte[] modelBytes,
        params (string name, int[] shape, float[] data)[] inputs)
    {
        using var session = new InferenceSession(modelBytes);
        var feeds = new List<NamedOnnxValue>(inputs.Length);
        foreach (var (name, shape, data) in inputs)
        {
            var dims = Array.ConvertAll(shape, i => (long)i);
            var t = new DenseTensor<float>(data, shape);
            feeds.Add(NamedOnnxValue.CreateFromTensor(name, t));
        }
        using var results = session.Run(feeds);
        var first = results.First();
        return first.AsTensor<float>().ToArray();
    }

    internal static float[] RunSingleOutput(
        byte[] modelBytes,
        Dictionary<string, (int[] shape, float[] data)> inputs)
    {
        using var session = new InferenceSession(modelBytes);
        var feeds = new List<NamedOnnxValue>(inputs.Count);
        foreach (var kv in inputs)
        {
            var t = new DenseTensor<float>(kv.Value.data, kv.Value.shape);
            feeds.Add(NamedOnnxValue.CreateFromTensor(kv.Key, t));
        }
        using var results = session.Run(feeds);
        return results.First().AsTensor<float>().ToArray();
    }

    internal static bool IsOrtAvailable
    {
        get
        {
            // ORT's native library may fail to load on some test hosts (e.g.
            // ARM64 without the matching wheel). Probe once and cache.
            try { using var _ = new SessionOptions(); return true; }
            catch { return false; }
        }
    }
}
