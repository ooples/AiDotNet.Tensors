// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Globalization;
using System.Security.Cryptography;
using System.Text;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ptx;

namespace AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;

/// <summary>
/// Everything that makes a measured lowering winner valid: device, target, spec and emitter.
/// </summary>
/// <remarks>
/// A kernel name alone is not an autotune key. The same name may be rebuilt for a different
/// shape, a new emitter may produce different instructions, and a winner on one GPU or driver
/// is not evidence for another. Keeping all four fingerprints in each row makes stale cache
/// entries fail closed instead of silently becoming dispatch policy.
/// </remarks>
public sealed record CodegenAutotuneIdentity(
    string DeviceFingerprint,
    string Target,
    string SpecFingerprint,
    string EmitterFingerprint)
{
    /// <summary>Builds the identity for one concrete spec and CUDA target.</summary>
    public static CodegenAutotuneIdentity Create(
        CodegenKernelSpec spec, string deviceFingerprint, int computeMajor, int computeMinor)
    {
        if (spec is null) throw new ArgumentNullException(nameof(spec));
        if (string.IsNullOrWhiteSpace(deviceFingerprint))
            throw new ArgumentException("An autotune identity needs a device fingerprint.",
                nameof(deviceFingerprint));
        if (computeMajor <= 0 || computeMinor < 0)
            throw new ArgumentOutOfRangeException(nameof(computeMajor));

        // Fingerprint what the tuner can ACTUALLY emit for this spec. An assembly MVID is
        // too broad and too unstable: repository/build metadata changed it after a benchmark-
        // only commit, silently invalidating every winner even though no emitted instruction
        // changed. Conversely, a hand-maintained version is easy to forget. Hashing the PTX
        // search space invalidates exactly when one of the candidate programs changes.
        string emitter = FingerprintEmitterSearchSpace(spec, computeMajor, computeMinor);
        return new CodegenAutotuneIdentity(
            deviceFingerprint,
            "sm" + computeMajor.ToString(CultureInfo.InvariantCulture) +
                computeMinor.ToString(CultureInfo.InvariantCulture),
            Hash(CanonicalSpec(spec)),
            emitter);
    }

    private static string FingerprintEmitterSearchSpace(
        CodegenKernelSpec spec, int computeMajor, int computeMinor)
    {
        var text = new StringBuilder();
        AppendCandidate(text, "modelled", spec, computeMajor, computeMinor, static _ => { });
        AppendCandidate(text, "no-tile", spec, computeMajor, computeMinor,
            static e => e.Coarsening = 1);
        AppendCandidate(text, "tile2", spec, computeMajor, computeMinor,
            static e => e.Coarsening = 2);
        AppendCandidate(text, "lanes4", spec, computeMajor, computeMinor,
            static e => e.MaxTileLanes = 4);
        AppendCandidate(text, "no-staging", spec, computeMajor, computeMinor,
            static e => e.EnableSharedStaging = false);
        AppendCandidate(text, "no-vector", spec, computeMajor, computeMinor,
            static e => e.EnableVectorLoads = false);
        AppendCandidate(text, "input-staging", spec, computeMajor, computeMinor,
            static e => e.EnableInputStaging = true);
        AppendTiledCandidate(text, spec, computeMajor, computeMinor);

        try
        {
            CodegenSplitPlan? split = CodegenSplitReduction.TryPlan(spec);
            if (split is not null)
            {
                AppendCandidate(text, "split-partial", split.Partial,
                    computeMajor, computeMinor, static _ => { });
                AppendCandidate(text, "split-combine", split.Combine,
                    computeMajor, computeMinor, static _ => { });
            }
        }
        catch (NotSupportedException)
        {
            text.Append("split=unsupported;");
        }

        return "ptxset-" + Hash(text.ToString());
    }

    private static void AppendCandidate(
        StringBuilder text,
        string name,
        CodegenKernelSpec spec,
        int computeMajor,
        int computeMinor,
        Action<PtxAffineEmitter> configure)
    {
        text.Append("candidate=").Append(name).Append(';');
        try
        {
            var emitter = new PtxAffineEmitter();
            configure(emitter);
            text.Append(emitter.Emit(spec, computeMajor, computeMinor));
        }
        catch (NotSupportedException ex)
        {
            text.Append("unsupported=").Append(ex.Message);
        }
        text.Append(";end-candidate;");
    }

    private static void AppendTiledCandidate(
        StringBuilder text, CodegenKernelSpec spec, int computeMajor, int computeMinor)
    {
        text.Append("candidate=tiled-contraction;");
        try
        {
            text.Append(new PtxTiledContractionEmitter().Emit(
                spec, computeMajor, computeMinor));
        }
        catch (NotSupportedException ex)
        {
            text.Append("unsupported=").Append(ex.Message);
        }
        text.Append(";end-candidate;");
    }

    private static string CanonicalSpec(CodegenKernelSpec spec)
    {
        var text = new StringBuilder();
        text.Append("name=").Append(spec.Name).Append(';');
        for (int a = 0; a < spec.Space.Axes.Count; a++)
        {
            var axis = spec.Space.Axes[a];
            text.Append("axis=").Append(axis.Name).Append(',')
                .Append(axis.Extent.ToString(CultureInfo.InvariantCulture)).Append(',')
                .Append(axis.IsReduction ? 'r' : 'p').Append(';');
        }

        for (int i = 0; i < spec.Inputs.Count; i++)
            AppendBinding(text, "in", spec.Inputs[i]);
        AppendBinding(text, "out", spec.Output);

        text.Append("product=");
        for (int i = 0; i < spec.ProductInputs.Count; i++)
            text.Append(spec.ProductInputs[i].ToString(CultureInfo.InvariantCulture)).Append(',');
        text.Append(";reduce=").Append(spec.Reduce)
            .Append(";bias=").Append(Optional(spec.BiasInput))
            .Append(";scale=").Append(Optional(spec.ScaleInput))
            .Append(";activation=").Append(spec.Activation)
            .Append(";reduceScale=").Append(spec.ReduceScale.ToString("R", CultureInfo.InvariantCulture))
            .Append(";preReduce=").Append(spec.PreReduce)
            .Append(";preBias=").Append(Optional(spec.PreBiasInput))
            .Append(";preBiasScale=").Append(spec.PreBiasScale.ToString("R", CultureInfo.InvariantCulture))
            .Append(";algebra=").Append(spec.Algebra).Append(';');

        for (int i = 0; i < spec.ExtraOutputs.Count; i++)
        {
            var extra = spec.ExtraOutputs[i];
            AppendBinding(text, "extra", extra.Binding);
            text.Append("extraKind=").Append(extra.Kind)
                .Append(";extraIndex=");
            if (extra.IndexExpr is not null) AppendExpr(text, extra.IndexExpr);
            text.Append(";extraScale=").Append(extra.Scale.ToString("R", CultureInfo.InvariantCulture))
                .Append(";extraBias=").Append(Optional(extra.BiasInput))
                .Append(";extraBiasScale=")
                .Append(extra.BiasScale.ToString("R", CultureInfo.InvariantCulture)).Append(';');
        }

        return text.ToString();
    }

    private static void AppendBinding(
        StringBuilder text, string role, CodegenTensorBinding binding)
    {
        text.Append(role).Append('=')
            .Append(binding.ParameterIndex.ToString(CultureInfo.InvariantCulture)).Append(',')
            .Append(binding.Name).Append(',').Append(binding.ElementType).Append(',')
            .Append(binding.IsOutput ? 'w' : 'r').Append(';');

        for (int d = 0; d < binding.Shape.Count; d++)
        {
            text.Append("dim=").Append(binding.Shape[d].ToString(CultureInfo.InvariantCulture))
                .Append(',');
            AppendExpr(text, binding.Map[d]);
            var indirect = binding.Indirect[d];
            if (indirect is not null)
            {
                text.Append(",indirect=")
                    .Append(indirect.IndexInput.ToString(CultureInfo.InvariantCulture)).Append(',')
                    .Append(indirect.Bound.ToString(CultureInfo.InvariantCulture)).Append(',')
                    .Append(indirect.OutOfRange).Append(',');
                AppendExpr(text, indirect.Position);
            }
            text.Append(';');
        }
    }

    private static void AppendExpr(StringBuilder text, CodegenAffineExpr expression)
    {
        text.Append('(');
        for (int t = 0; t < expression.Terms.Count; t++)
        {
            var term = expression.Terms[t];
            text.Append(term.Axis.ToString(CultureInfo.InvariantCulture)).Append(':')
                .Append(term.Coefficient.ToString(CultureInfo.InvariantCulture)).Append(',');
        }
        text.Append('|').Append(expression.Constant.ToString(CultureInfo.InvariantCulture))
            .Append('|').Append(expression.Divisor.ToString(CultureInfo.InvariantCulture))
            .Append('|').Append(expression.RequiresExactDivision ? '1' : '0').Append(')');
    }

    private static string Optional(int? value) => value.HasValue
        ? value.Value.ToString(CultureInfo.InvariantCulture)
        : "-";

    private static string Hash(string value)
    {
        using var sha = SHA256.Create();
        byte[] digest = sha.ComputeHash(Encoding.UTF8.GetBytes(value));
        var text = new StringBuilder(digest.Length * 2);
        for (int i = 0; i < digest.Length; i++)
            text.Append(digest[i].ToString("x2", CultureInfo.InvariantCulture));
        return "sha256-" + text;
    }
}
