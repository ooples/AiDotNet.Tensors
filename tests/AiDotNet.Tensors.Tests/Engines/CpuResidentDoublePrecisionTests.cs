using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// A caller that asks for <c>double</c> and hands the engine CPU-resident data must get double
/// answers, whatever the ambient engine happens to be.
/// </summary>
/// <remarks>
/// <para>
/// Narrowing double -&gt; float belongs at a real device boundary, for data that already lives on the
/// device. It was also happening on the upload-compute-download paths: with
/// <c>AiDotNetEngine.Current</c> resolving to <see cref="DirectGpuTensorEngine"/>, a CPU-resident
/// <c>Vector&lt;double&gt;</c> multiplied by a double came back with 3.974E-008 relative error --
/// float32 epsilon (~1.19e-7), not double epsilon (~2.2e-16) -- because the operands were copied
/// into a single-precision kernel. The same operations on an explicitly constructed
/// <see cref="CpuEngine"/> were exact, so the kernels were never the problem; the dispatch was.
/// </para>
/// <para>
/// The cost was not only accuracy: every such op paid a host-&gt;device-&gt;host round trip for data
/// that never needed to leave the CPU, which is why an optimizer update chain measured ~290 ms per
/// step at 2M parameters against ~17 ms for the same arithmetic in process.
/// </para>
/// <para>
/// These assert EXACT equality on purpose. A tolerance is what let this hide: every value was
/// "close enough" while silently carrying seven digits instead of sixteen. The operands are chosen
/// so the float and double results differ -- thirds and sevenths are not representable, so a
/// single-precision round trip cannot reproduce the double answer.
/// </para>
/// </remarks>
public class CpuResidentDoublePrecisionTests
{
    private static Vector<double> Vec(params double[] values)
    {
        var v = new Vector<double>(values.Length);
        for (int i = 0; i < values.Length; i++) v[i] = values[i];
        return v;
    }

    private const double A = 1.0 / 3.0;
    private const double B = 0.9000000000000001;
    private const double C = 7.0 / 11.0;

    public static TheoryData<string> Engines => new()
    {
        "cpu",
        "gpu",
    };

    /// <summary>
    /// Binds the second case to <see cref="DirectGpuTensorEngine"/> EXPLICITLY, and skips when the
    /// host has no device backend.
    /// </summary>
    /// <remarks>
    /// This case used to resolve <c>AiDotNetEngine.Current</c>, which defaults to
    /// <see cref="CpuEngine"/> and only auto-detects a GPU when one is present (never on net471).
    /// On any CPU-only machine — every CI runner here — it therefore ran the same engine as the
    /// "cpu" case and asserted nothing new, so the regression this whole file exists to catch
    /// (double narrowed to float across the upload/compute/download path) had NO coverage while
    /// the suite reported green. Constructing the engine directly makes the GPU case either run
    /// against the GPU or announce itself as skipped, instead of silently passing as a duplicate.
    /// </remarks>
    private static IEngine Resolve(string which)
    {
        if (which == "cpu") return new CpuEngine();
        var gpu = new DirectGpuTensorEngine();
        Skip.If(!gpu.IsGpuAvailable, "needs a DirectGpu backend (CUDA/OpenCL/…).");
        return gpu;
    }

    [SkippableTheory]
    [MemberData(nameof(Engines))]
    public void MultiplyVectorByVector_IsExactDouble(string which)
    {
        var engine = Resolve(which);
        var result = (Vector<double>)engine.Multiply(Vec(A), Vec(B));
        Assert.Equal(A * B, result[0]);
    }

    [SkippableTheory]
    [MemberData(nameof(Engines))]
    public void AddVectorToVector_IsExactDouble(string which)
    {
        var engine = Resolve(which);
        var result = (Vector<double>)engine.Add(Vec(A), Vec(B));
        Assert.Equal(A + B, result[0]);
    }

    [SkippableTheory]
    [MemberData(nameof(Engines))]
    public void SubtractVectorFromVector_IsExactDouble(string which)
    {
        var engine = Resolve(which);
        var result = (Vector<double>)engine.Subtract(Vec(A), Vec(B));
        Assert.Equal(A - B, result[0]);
    }

    [SkippableTheory]
    [MemberData(nameof(Engines))]
    public void DivideVectorByVector_IsExactDouble(string which)
    {
        var engine = Resolve(which);
        var result = (Vector<double>)engine.Divide(Vec(A), Vec(B));
        Assert.Equal(A / B, result[0]);
    }

    [SkippableTheory]
    [MemberData(nameof(Engines))]
    public void SqrtOfVector_IsExactDouble(string which)
    {
        var engine = Resolve(which);
        var result = (Vector<double>)engine.Sqrt(Vec(C));
        Assert.Equal(Math.Sqrt(C), result[0]);
    }

    /// <summary>
    /// The composite that surfaced this: one Adam update step's worth of arithmetic. A single
    /// narrowing op anywhere in the chain moves the result, so this fails even if only one overload
    /// regresses.
    /// </summary>
    [SkippableTheory]
    [MemberData(nameof(Engines))]
    public void AdamStyleUpdateChain_IsExactDouble(string which)
    {
        var engine = Resolve(which);

        const double Beta1 = 0.9, Beta2 = 0.999, Epsilon = 1e-8, Lr = 0.001;
        var m = Vec(A);
        var v = Vec(C);
        var g = Vec(B);
        var p = Vec(1.0 / 7.0);

        var mNext = (Vector<double>)engine.Add(
            (Vector<double>)engine.Multiply(m, Beta1),
            (Vector<double>)engine.Multiply(g, 1 - Beta1));
        var vNext = (Vector<double>)engine.Add(
            (Vector<double>)engine.Multiply(v, Beta2),
            (Vector<double>)engine.Multiply((Vector<double>)engine.Multiply(g, g), 1 - Beta2));
        var denominator = (Vector<double>)engine.Add(
            (Vector<double>)engine.Sqrt(vNext),
            Vector<double>.CreateDefault(1, Epsilon));
        var update = (Vector<double>)engine.Divide(mNext, denominator);
        var updated = (Vector<double>)engine.Subtract(p, (Vector<double>)engine.Multiply(update, Lr));

        double expectedM = (A * Beta1) + (B * (1 - Beta1));
        double expectedV = (C * Beta2) + ((B * B) * (1 - Beta2));
        double expected = (1.0 / 7.0) - ((expectedM / (Math.Sqrt(expectedV) + Epsilon)) * Lr);

        Assert.Equal(expected, updated[0]);
    }
}
