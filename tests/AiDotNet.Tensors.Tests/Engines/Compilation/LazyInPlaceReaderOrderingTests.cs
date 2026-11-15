using System;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// A lazy node reads each input as it was when the node was recorded, so an in-place write recorded later must run after
/// every earlier reader of its target, whichever is materialized first. Three kinds of reader were left unordered: an
/// in-place write of another tensor that reads the target (y += x before x *= 2), a reader of a view that shares the
/// target's storage, and a cross-type reader (an FFT). Each one read the written value when the write was materialized
/// first; the cross-type node also ran a later write early when it was materialized first.
/// </summary>
[Collection("CompilationGlobalState")]
public class LazyInPlaceReaderOrderingTests
{
    private static Tensor<float> Values(params float[] values) => new(values, new[] { values.Length });

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void An_in_place_write_that_reads_the_target_runs_before_the_targets_later_write(bool writeReadFirst)
    {
        var engine = new CpuEngine();
        var x = Values(1f, 3f);
        var y = Values(0f, 0f);
        float[] xRead, yRead;
        using (GraphMode.Enable())
        {
            engine.TensorAddInPlace(y, x);              // y += x reads x = [1, 3]
            engine.TensorMultiplyScalarInPlace(x, 2f);  // then x *= 2
            if (writeReadFirst) { xRead = x.ToArray(); yRead = y.ToArray(); }
            else { yRead = y.ToArray(); xRead = x.ToArray(); }
        }
        Assert.Equal(new[] { 1f, 3f }, yRead);
        Assert.Equal(new[] { 2f, 6f }, xRead);
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void A_reader_of_a_view_runs_before_a_later_write_of_the_viewed_tensor(bool writeReadFirst)
    {
        var engine = new CpuEngine();
        var x = Values(1f, 2f, 3f, 4f);
        float[] xRead, rRead;
        using (GraphMode.Enable())
        {
            var v = engine.Reshape(x, new[] { 2, 2 });     // a view: shares x's storage
            var r = engine.TensorMultiplyScalar(v, 1f);    // reads the view
            engine.TensorMultiplyScalarInPlace(x, 2f);     // then x *= 2
            if (writeReadFirst) { xRead = x.ToArray(); rRead = r.ToArray(); }
            else { rRead = r.ToArray(); xRead = x.ToArray(); }
        }
        Assert.Equal(new[] { 1f, 2f, 3f, 4f }, rRead);
        Assert.Equal(new[] { 2f, 4f, 6f, 8f }, xRead);
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void A_cross_type_reader_runs_before_a_later_write_and_does_not_run_it_early(bool writeReadFirst)
    {
        var engine = new CpuEngine();
        var host = new[] { 0.5f, -1.25f, 2f, 0.75f, -0.3f, 1.1f, 0.2f, -0.9f };
        var expected = engine.NativeComplexMagnitude(engine.NativeComplexFFT(Values(host))).ToArray();
        var x = Values(host);
        float[] xRead, magRead;
        using (GraphMode.Enable())
        {
            var spectrum = engine.NativeComplexFFT(x);           // a cross-type node reading x
            var magnitude = engine.NativeComplexMagnitude(spectrum);
            engine.TensorMultiplyScalarInPlace(x, 2f);          // then x *= 2
            if (writeReadFirst) { xRead = x.ToArray(); magRead = magnitude.ToArray(); }
            else { magRead = magnitude.ToArray(); xRead = x.ToArray(); }
        }
        Assert.Equal(host.Select(v => 2f * v).ToArray(), xRead);
        Assert.Equal(expected.Length, magRead.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - magRead[i]) <= 1e-5f * Math.Max(1f, Math.Abs(expected[i])),
                $"|FFT(x)|[{i}] = {magRead[i]}, expected {expected[i]} (the FFT read x after the write)");
    }
}
