// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.NumericOperations;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>Why an op whose output moves smoothly with its input still records no gradient.</summary>
public enum NoGradientReason
{
    /// <summary>A random draw: torch's samplers (bernoulli, uniform_, ...) are not differentiable either.</summary>
    RandomDraw,

    /// <summary>An iterative reconstruction algorithm, not an op (GriffinLim runs 60 fixed-point iterations).</summary>
    IterativeAlgorithm,

    /// <summary>Sequential branch-selected recurrence (Fritsch-Carlson Hermite slope limiter).</summary>
    PiecewiseRecurrence,

    /// <summary>Intermediate quantities of a loss whose own backward consumes them (the CTC lattice).</summary>
    ConsumedByOwnLossBackward,

    /// <summary>A scaled low-precision kernel, not differentiable in torch either (torch._scaled_mm).</summary>
    LowPrecisionKernel,

    /// <summary>Intentional gradient blocker (<see cref="IEngine.StopGradient{T}"/>).</summary>
    ExplicitStopGradient,

    /// <summary>Writes into a caller-provided destination or mutates a tensor slice outside functional autograd.</summary>
    InPlaceOrDestinationBuffer,

    /// <summary>Internal memory-layout conversion (NCHW/NCHWc); gradients propagate through tensor layout handling.</summary>
    LayoutReorder,

    /// <summary>Inference-only fused fast path; training decomposes into recorded constituent ops.</summary>
    InferenceOnlyFastPath,

    /// <summary>Value-equality deduplication with data-dependent output shape (torch.unique has no gradient either).</summary>
    DataDependentUniqueSet,

    /// <summary>Phase-vocoder audio augmentation used in preprocessing rather than inside differentiable losses.</summary>
    AudioAugmentation,

    /// <summary>3D/NeRF or detection geometry primitive paired with explicit domain backward helpers or pre/postprocessing.</summary>
    DomainSpecificPrimitive,

    /// <summary>Legacy Native* DSP/math wrapper superseded on the autograd path by recorded Tensor* / spectral ops.</summary>
    NativeWrapper,

    /// <summary>Legacy constructor/matrix helper (<see cref="IEngine.TensorDiag{T}"/>); autograd uses TensorDiagflat / TensorDiagEmbed.</summary>
    ConstructorHelper,
}

/// <summary>Why an op in <see cref="OpRegistry.NonDifferentiableOps"/> cannot be driven by the audit.</summary>
public enum NotAuditableReason
{
    /// <summary>Its valid inputs have a structure neither the argument synthesizer nor a case table entry builds.</summary>
    NoSynthesizableInput,

    /// <summary>Defined only on integral or bitwise domains, so <c>double</c> perturbations throw by contract.</summary>
    IntegerOnlyDomain,

    /// <summary>Takes a caller-supplied delegate (<c>Func&lt;T, T&gt;</c>) rather than pure tensor/scalar parameters.</summary>
    ArbitraryUserDelegate,
}

/// <summary>
/// Audits <see cref="OpRegistry.NonDifferentiableOps"/> against the ops themselves.
/// </summary>
/// <remarks>
/// <para>
/// The classification was never checked. STFT and the FFT family sat in the set on the stated ground that their out
/// parameters cannot flow through the Record* helpers, which is false (STFT now records an edge per out result), and
/// because every sweep selected ops by a tensor return type, nothing ever finite-differenced them: STFT trained nothing
/// upstream of it. A loss through a classified op silently gets no gradient, so a wrong entry is a silent defect.
/// </para>
/// <para>
/// This drives every forward op in the set the sweep can invoke and finite-differences its output (the weighted sum of
/// every out result for an op with no return) at a few elements of each tensor input. Backward kernel helpers
/// (<c>*Backward*</c>, <c>*Derivative</c>) and creation ops with no <c>Tensor&lt;T&gt;</c> input are skipped up front.
/// A derivative that is nonzero and stable under halving the step is a smooth dependence: the op is differentiable and
/// must record a backward, unless <see cref="NoGradientBy"/> says why not. Comparisons, indicators, rounding, integer
/// and index ops have a zero derivative at generic points and pass. An op that varies between identical calls fails
/// unless it is a listed random draw, and an op the audit cannot invoke or probe fails unless <see cref="NotAuditable"/>
/// says why.
/// </para>
/// </remarks>
public partial class DifferentiableOpsGradCheckSweep
{
    /// <summary>The ops that may depend smoothly on their input and still record no gradient, each with its reason.</summary>
    internal static readonly Dictionary<string, NoGradientReason> NoGradientBy = new(StringComparer.Ordinal)
    {
        ["GriffinLim"] = NoGradientReason.IterativeAlgorithm,
        ["TensorFritschCarlsonLimit"] = NoGradientReason.PiecewiseRecurrence,
        ["TensorCTCForwardBackward"] = NoGradientReason.ConsumedByOwnLossBackward,
        ["TensorCTCLabelPosteriors"] = NoGradientReason.ConsumedByOwnLossBackward,
        ["TensorScaledMatMul"] = NoGradientReason.LowPrecisionKernel,
        ["TensorScaledGroupedMatMul"] = NoGradientReason.LowPrecisionKernel,
        ["TensorBernoulli"] = NoGradientReason.RandomDraw,
        ["TensorBinomial"] = NoGradientReason.RandomDraw,
        ["TensorPoisson"] = NoGradientReason.RandomDraw,
        ["TensorUniform"] = NoGradientReason.RandomDraw,
        ["TensorCauchy"] = NoGradientReason.RandomDraw,
        ["TensorExponential"] = NoGradientReason.RandomDraw,
        ["TensorGeometric"] = NoGradientReason.RandomDraw,
        ["TensorLogNormal"] = NoGradientReason.RandomDraw,
        ["TensorMultinomial"] = NoGradientReason.RandomDraw,
        ["TensorRandLike"] = NoGradientReason.RandomDraw,
        ["TensorRandnLike"] = NoGradientReason.RandomDraw,
        ["TensorRandintLike"] = NoGradientReason.RandomDraw,
        ["TensorCategoricalSample"] = NoGradientReason.RandomDraw,
        ["GumbelSoftmax"] = NoGradientReason.RandomDraw,
        ["StopGradient"] = NoGradientReason.ExplicitStopGradient,
        ["GroupNormInto"] = NoGradientReason.InPlaceOrDestinationBuffer,
        ["TensorSetSlice"] = NoGradientReason.InPlaceOrDestinationBuffer,
        ["ReorderToNchw"] = NoGradientReason.LayoutReorder,
        ["ReorderToNchwc"] = NoGradientReason.LayoutReorder,
        ["BatchNormInference"] = NoGradientReason.InferenceOnlyFastPath,
        ["FusedHierarchicalSoftmax"] = NoGradientReason.InferenceOnlyFastPath,
        ["FusedLinearMaxout"] = NoGradientReason.InferenceOnlyFastPath,
        ["LstmSequenceForward"] = NoGradientReason.InferenceOnlyFastPath,
        ["MlpForward"] = NoGradientReason.InferenceOnlyFastPath,
        ["MultiHeadAttentionForward"] = NoGradientReason.InferenceOnlyFastPath,
        ["TensorUnique"] = NoGradientReason.DataDependentUniqueSet,
        ["TensorUniqueConsecutive"] = NoGradientReason.DataDependentUniqueSet,
        ["TensorUniqueDim"] = NoGradientReason.DataDependentUniqueSet,
        ["PitchShift"] = NoGradientReason.AudioAugmentation,
        ["TimeStretch"] = NoGradientReason.AudioAugmentation,
        ["BoxArea"] = NoGradientReason.DomainSpecificPrimitive,
        ["BoxConvert"] = NoGradientReason.DomainSpecificPrimitive,
        ["ComputeGaussianCovariance"] = NoGradientReason.DomainSpecificPrimitive,
        ["EvaluateSphericalHarmonics"] = NoGradientReason.DomainSpecificPrimitive,
        ["ImportanceSampling"] = NoGradientReason.DomainSpecificPrimitive,
        ["MultiresolutionHashEncoding"] = NoGradientReason.DomainSpecificPrimitive,
        ["PositionalEncoding"] = NoGradientReason.DomainSpecificPrimitive,
        ["ProjectGaussians3DTo2D"] = NoGradientReason.DomainSpecificPrimitive,
        ["RasterizeGaussians"] = NoGradientReason.DomainSpecificPrimitive,
        ["UpdateOccupancyGrid"] = NoGradientReason.DomainSpecificPrimitive,
        ["NativeAtan2"] = NoGradientReason.NativeWrapper,
        ["NativeBatchedCavityForward"] = NoGradientReason.NativeWrapper,
        ["NativeExp"] = NoGradientReason.NativeWrapper,
        ["NativeMfccFeatures"] = NoGradientReason.NativeWrapper,
        ["NativeNormalizeRows"] = NoGradientReason.NativeWrapper,
        ["NativeTanh"] = NoGradientReason.NativeWrapper,
        ["NativeWidebandFeatures"] = NoGradientReason.NativeWrapper,
        ["TensorDiag"] = NoGradientReason.ConstructorHelper,
    };

    /// <summary>
    /// Ops the audit cannot invoke, each with its reason. Every other op it fails to drive fails the audit: an op that
    /// is never invoked is never verified, so an unexplained skip would let a wrong classification pass.
    /// </summary>
    internal static readonly Dictionary<string, NotAuditableReason> NotAuditable = new(StringComparer.Ordinal)
    {
        ["TensorBitwiseAnd"] = NotAuditableReason.IntegerOnlyDomain,
        ["TensorBitwiseLeftShift"] = NotAuditableReason.IntegerOnlyDomain,
        ["TensorBitwiseNot"] = NotAuditableReason.IntegerOnlyDomain,
        ["TensorBitwiseOr"] = NotAuditableReason.IntegerOnlyDomain,
        ["TensorBitwiseRightShift"] = NotAuditableReason.IntegerOnlyDomain,
        ["TensorBitwiseXor"] = NotAuditableReason.IntegerOnlyDomain,
        ["TensorGcd"] = NotAuditableReason.IntegerOnlyDomain,
        ["TensorLcm"] = NotAuditableReason.IntegerOnlyDomain,
        ["TensorMap"] = NotAuditableReason.ArbitraryUserDelegate,
    };

    static partial void AddCasesNonDifferentiableAudit(Dictionary<string, Func<Random, object[]>> cases)
    {
        cases["BatchNormInference|T,T,T,T,T,double"] = r =>
            [SafeTensor([2, 3, 2, 2], r), SafeTensor([3], r), SafeTensor([3], r), SafeTensor([3], r), SafeTensor([3], r), 1e-5];
        cases["BatchNormInference|T,T,T,T,T,double,FusedActivationType"] = r =>
            [SafeTensor([2, 3, 2, 2], r), SafeTensor([3], r), SafeTensor([3], r), SafeTensor([3], r), SafeTensor([3], r), 1e-5, FusedActivationType.Tanh];
        cases["BoxArea"] = r => [Boxes(3, r)];
        cases["BoxConvert"] = r => [Boxes(3, r), BoxFormat.XYXY, BoxFormat.XYWH];
        cases["ComputeGaussianCovariance"] = r => [SafeTensor([3, 4], r), SafeTensor([3, 3], r)];
        cases["EvaluateSphericalHarmonics"] = r => [Scaled(SafeTensor([3, 4, 3], r), 0.2), SafeTensor([3, 3], r), 1];
        cases["FusedHierarchicalSoftmax"] = r => [SafeTensor([2, 3], r), SafeTensor([2, 3], r), 4];
        cases["FusedLinearMaxout"] = r => [SafeTensor([2, 3], r), SafeTensor([3, 4], r), SafeTensor([4], r), 2];
        cases["GriffinLim"] = r =>
        {
            var args = new object[7];
            args[0] = SafeTensor([9, 4], r);
            args[1] = 16;
            args[2] = 4;
            args[3] = HannWindowFor(16);
            args[4] = 2;
            args[5] = 0.99;
            return args;
        };
        cases["LstmSequenceForward|T,T,T,T,T,T,T,bool"] = r =>
            [SafeTensor([2, 3, 4], r), SafeTensor([2, 2], r), SafeTensor([2, 2], r), SafeTensor([8, 4], r), SafeTensor([8, 2], r), SafeTensor([8], r), SafeTensor([8], r), true];
        cases["LstmSequenceForward|T,T,T,T,T,T,T,T,T,bool"] = r =>
            [SafeTensor([2, 3, 4], r), SafeTensor([2, 2], r), SafeTensor([2, 2], r), SafeTensor([8, 4], r), SafeTensor([8, 2], r), SafeTensor([8], r), SafeTensor([8], r), OutSlot(), OutSlot(), true];
        cases["MlpForward"] = r =>
        {
            var args = new object[7];
            args[0] = SafeTensor([2, 3], r);
            args[1] = new[] { SafeTensor([3, 4], r), SafeTensor([4, 2], r) };
            args[2] = new[] { SafeTensor([4], r), SafeTensor([2], r) };
            args[3] = FusedActivationType.Tanh;
            args[4] = FusedActivationType.None;
            return args;
        };
        cases["MultiHeadAttentionForward"] = r =>
        {
            var args = new object[7];
            args[0] = SafeTensor([1, 3, 4], r);
            args[1] = SafeTensor([4, 4], r);
            args[2] = SafeTensor([4, 4], r);
            args[3] = SafeTensor([4, 4], r);
            args[4] = SafeTensor([4, 4], r);
            args[5] = 2;
            return args;
        };
        cases["MultiresolutionHashEncoding"] = r =>
        [
            new Tensor<double>(new[] { 0.17, 0.31, 0.43, 0.61, 0.79, 0.83 }, new[] { 2, 3 }),
            new[] { SafeTensor([16, 2], r), SafeTensor([16, 2], r) },
            new[] { 4, 8 },
            2
        ];
        cases["NativeBatchedCavityForward"] = r => [SafeTensor([1, 8], r), ComplexFilterBank(2, 8, r), 1];
        cases["NativeMfccFeatures"] = r => [SafeTensor([1, 32], r), 2, 4, 16];
        cases["NativePacFeatures"] = _ => [PacWaveform(64), 64, 32, 4.0, 8.0, new[] { (16.0, 24.0) }];
        cases["ProjectGaussians3DTo2D"] = r =>
        [
            new Tensor<double>(new[] { 0.2, 0.3, 2.5, -0.3, 0.1, 3.0 }, new[] { 2, 3 }),
            SafeTensor([2, 6], r),
            IdentityMatrix4(),
            IdentityMatrix4(),
            64,
            64,
            OutSlot(),
            OutSlot(),
            OutSlot(),
            new Tensor<bool>(new[] { 1 })
        ];
        cases["ReorderToNchwc"] = r => [SafeTensor([1, 8, 2, 2], r), TensorLayout.Nchwc8];
        cases["TensorBinomial"] = r =>
        {
            var counts = new Tensor<double>([16]);
            for (int i = 0; i < counts.Length; i++) counts[i] = 20.0;
            var args = new object[3];
            args[0] = counts;
            args[1] = SafeTensor([16], r);
            return args;
        };
        cases["TensorCTCLabelPosteriors"] = r => [LogProbs(5, 1, 4, r), CtcTargets(2, 4, r), new[] { 5 }, new[] { 2 }, 0];
        cases["TensorFritschCarlsonLimit"] = r => [SafeTensor([5], r), SafeTensor([4], r)];
        cases["TensorRandintLike"] = r =>
        {
            var args = new object[4];
            args[0] = SafeTensor([16], r);
            args[1] = 0L;
            args[2] = 1_000_000L;
            return args;
        };
        cases["TensorScaledMatMul"] = r =>
        [
            SafeTensor([2, 4], r), SafeTensor([4, 3], r),
            SafeTensor([1], r), ScalingRecipe.TensorWise,
            SafeTensor([1], r), ScalingRecipe.TensorWise,
            ScaleSwizzle.None, ScaleSwizzle.None,
            SafeTensor([3], r)
        ];
        cases["TensorScaledGroupedMatMul"] = r =>
        {
            var args = new object[10];
            args[0] = SafeTensor([2, 2, 4], r);
            args[1] = SafeTensor([2, 4, 3], r);
            args[2] = SafeTensor([1], r);
            args[3] = ScalingRecipe.TensorWise;
            args[4] = SafeTensor([1], r);
            args[5] = ScalingRecipe.TensorWise;
            args[6] = ScaleSwizzle.None;
            args[7] = ScaleSwizzle.None;
            args[8] = SafeTensor([3], r);
            return args;
        };
    }

    private static Matrix<double> IdentityMatrix4()
    {
        var m = new Matrix<double>(4, 4);
        for (int i = 0; i < 4; i++) m[i, i] = 1.0;
        return m;
    }

    private static Tensor<Complex<double>> ComplexFilterBank(int cavities, int n, Random rng)
    {
        var data = new Complex<double>[cavities * n];
        for (int i = 0; i < data.Length; i++)
            data[i] = new Complex<double>(0.2 + rng.NextDouble() * 0.5, 0.1 + rng.NextDouble() * 0.4);
        return new Tensor<Complex<double>>(data, new[] { cavities, n });
    }

    private static Tensor<double> PacWaveform(int n)
    {
        var t = new Tensor<double>([1, n]);
        for (int i = 0; i < n; i++)
        {
            double theta = Math.Sin(2.0 * Math.PI * 6.0 * i / n);
            double gamma = (1.0 + 0.5 * theta) * Math.Cos(2.0 * Math.PI * 20.0 * i / n);
            t[i] = 0.5 + 0.6 * theta + 0.4 * gamma;
        }
        return t;
    }

    private static bool IsBackwardHelper(string name) =>
        (name.Contains("Backward", StringComparison.Ordinal) && name != "TensorCTCForwardBackward")
        || name.EndsWith("Derivative", StringComparison.Ordinal);

    private static bool HasDoubleTensorInput(MethodInfo closedDoubleMethod)
    {
        foreach (var p in closedDoubleMethod.GetParameters())
        {
            if (p.IsOut) continue;
            var pt = p.ParameterType;
            if (pt == typeof(Tensor<double>) || pt == typeof(Tensor<double>[])
                || typeof(IEnumerable<Tensor<double>>).IsAssignableFrom(pt))
                return true;
        }
        return false;
    }

    [Fact]
    public void EveryNonDifferentiableOp_HasNoSmoothDerivative()
    {
        var smooth = new List<string>();
        var exemptedAndSmooth = new HashSet<string>(StringComparer.Ordinal);
        var exemptedAndAudited = new HashSet<string>(StringComparer.Ordinal);
        var notDriven = new List<string>();
        var drivenButExcluded = new HashSet<string>(StringComparer.Ordinal);
        var nondeterministic = new List<string>();
        int audited = 0;

        foreach (var def in typeof(IEngine).GetMethods(BindingFlags.Public | BindingFlags.Instance)
                     .Where(m => !m.IsSpecialName && m.IsGenericMethodDefinition && m.GetGenericArguments().Length == 1)
                     .Where(HasTensorOutput))
        {
            string name = def.Name;
            if (!OpRegistry.NonDifferentiableOps.Contains(name)) continue;
            if (IsBackwardHelper(name)) continue;
            MethodInfo m;
            try { m = def.MakeGenericMethod(typeof(double)); }
            catch (ArgumentException) { if (!NotAuditable.ContainsKey(name)) notDriven.Add($"{name}: cannot bind <double>"); continue; }

            if (!HasDoubleTensorInput(m)) continue; // creation or discrete-input op with no floating-point tensor input

            var args = DriveArgs(m, name);
            if (args is null)
            {
                if (!NotAuditable.ContainsKey(name)) notDriven.Add($"{name}|{ParamFingerprint(m)}: no valid invocation");
                continue;
            }
            if (NotAuditable.ContainsKey(name)) drivenButExcluded.Add(name);

            var inputs = InputTensors<double>(m, args);
            if (inputs.Length == 0) continue; // nothing to depend on (a creation op given only sizes)
            if (!IsDeterministic(m, args))
            {
                // Only a random draw may vary between identical calls; anything else varying is a defect of its own.
                if (!(NoGradientBy.TryGetValue(name, out var reason) && reason == NoGradientReason.RandomDraw))
                    nondeterministic.Add($"{name}|{ParamFingerprint(m)}");
                continue;
            }

            string? finding = SmoothDependence(m, args, inputs, out bool anyProbed);
            if (!anyProbed)
            {
                if (!NotAuditable.ContainsKey(name))
                    notDriven.Add($"{name}|{ParamFingerprint(m)}: all perturbed probes failed");
                continue;
            }

            audited++;
            if (NoGradientBy.TryGetValue(name, out var why) && why != NoGradientReason.RandomDraw)
                exemptedAndAudited.Add(name);
            if (finding is null) continue;
            if (NoGradientBy.ContainsKey(name)) exemptedAndSmooth.Add(name);
            else smooth.Add($"{name}|{ParamFingerprint(m)}: {finding}");
        }

        _out.WriteLine($"audited          : {audited}");
        _out.WriteLine($"smooth, no grad  : {smooth.Count}");
        _out.WriteLine($"exempted         : {string.Join(", ", exemptedAndSmooth.OrderBy(x => x))}");
        foreach (var s in smooth.OrderBy(x => x)) _out.WriteLine("SMOOTH  " + s);
        foreach (var s in nondeterministic.OrderBy(x => x)) _out.WriteLine("varies  " + s);
        foreach (var s in notDriven.OrderBy(x => x)) _out.WriteLine("skip    " + s);

        // An exemption for an op the audit drove and found constant is stale: the reason no longer describes the op.
        var stale = exemptedAndAudited.Except(exemptedAndSmooth).OrderBy(x => x).ToList();
        Assert.True(smooth.Count == 0,
            $"{smooth.Count} op(s) in OpRegistry.NonDifferentiableOps depend smoothly on their input, so a loss through "
            + "them silently gets no gradient. Record a backward and move each to DifferentiableOps, or add it to "
            + "NoGradientBy with its reason:\n" + string.Join("\n", smooth.OrderBy(x => x)));
        Assert.True(stale.Count == 0, "NoGradientBy entries with no smooth dependence (remove them): " + string.Join(", ", stale));
        Assert.True(notDriven.Count == 0,
            $"{notDriven.Count} op(s) in OpRegistry.NonDifferentiableOps were never invoked, so their classification is "
            + "unverified. Add a case table entry that drives each, or list it in NotAuditable with its reason:\n"
            + string.Join("\n", notDriven.OrderBy(x => x)));
        Assert.True(nondeterministic.Count == 0,
            "Ops that vary between identical calls but are not listed as a RandomDraw: " + string.Join(", ", nondeterministic.OrderBy(x => x)));
        Assert.True(drivenButExcluded.Count == 0,
            "NotAuditable entries the audit can drive (remove them): " + string.Join(", ", drivenButExcluded.OrderBy(x => x)));
    }

    private bool IsDeterministic(MethodInfo m, object[] args)
    {
        if (Evaluate(m, args) is not { } f0) return false;
        for (int trial = 0; trial < 3; trial++)
        {
            if (Evaluate(m, args) is not { } fk || !f0.Equals(fk))
                return false;
        }
        return true;
    }

    private object[]? DriveArgs(MethodInfo m, string name)
    {
        string fingerprint = $"{name}|{ParamFingerprint(m)}";
        bool hasTable = OpCases.TryGetValue(fingerprint, out var caseFactory)
                     || OpCases.TryGetValue($"{name}/{m.GetParameters().Length}", out caseFactory)
                     || OpCases.TryGetValue(name, out caseFactory);
        foreach (var shape in hasTable ? new[] { Array.Empty<int>() } : new[] { new[] { 6 }, new[] { 2, 3 } })
        {
            var rng = new Random(1234);
            object[]? args;
            try { args = hasTable && caseFactory is not null ? caseFactory(rng) : BuildArgs(m, shape, rng, out _); }
            catch (Exception ex) when (ex is ArgumentException or InvalidOperationException) { continue; }
            if (args is null) continue;
            if (Evaluate(m, args) is null) continue;
            return args;
        }
        return null;
    }

    // The op's output reduced to one number with non-uniform element weights so permutations or balanced zero-sum
    // draws do not cancel, or null when it throws.
    private double? Evaluate(MethodInfo m, object[] args)
    {
        try
        {
            using var _ = new NoGradScope<double>();
            var output = InvokeOutput<double>(_engine, m, args);
            double sum = 0;
            var flat = output.GetFlattenedData();
            for (int i = 0; i < flat.Length; i++)
                sum += flat[i] * (1.0 + 0.371293 * Math.Sin(1.7320508 * (i + 1)));
            return sum;
        }
        catch (Exception ex) when (ex is TargetInvocationException or ArgumentException or InvalidOperationException
                                       or NotSupportedException or IndexOutOfRangeException)
        {
            return null;
        }
    }

    // The first element of an input along which the output's derivative is nonzero and stable (the same at step h
    // and h/2): a smooth dependence. A jump or a kink is not stable under halving, a plateau is zero.
    private string? SmoothDependence(MethodInfo m, object[] args, Tensor<double>[] inputs, out bool anyProbed)
    {
        anyProbed = false;
        const double h = 1e-4;
        for (int t = 0; t < inputs.Length; t++)
        {
            var x = inputs[t];
            int n = x.Length;
            foreach (int i in new[] { 0, n / 3, (2 * n) / 3, n - 1 }.Where(i => i >= 0 && i < n).Distinct())
            {
                double original = x[i];
                double? Derivative(double step)
                {
                    x[i] = original + step;
                    var plus = Evaluate(m, args);
                    x[i] = original - step;
                    var minus = Evaluate(m, args);
                    x[i] = original;
                    return plus is { } p && minus is { } q ? (p - q) / (2 * step) : null;
                }

                if (Derivative(h) is not { } d1 || Derivative(h / 2) is not { } d2) continue;
                if (double.IsNaN(d1) || double.IsInfinity(d1) || double.IsNaN(d2) || double.IsInfinity(d2)) continue;
                anyProbed = true;
                if (Math.Abs(d1) <= 1e-6) continue;
                if (Math.Abs(d1 - d2) <= 1e-4 + 1e-3 * Math.Abs(d1))
                    return $"d/d(input {t})[{i}] = {d1:G6} (step {h}), {d2:G6} (step {h / 2})";
            }
        }
        return null;
    }
}
