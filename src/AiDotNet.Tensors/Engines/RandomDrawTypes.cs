namespace AiDotNet.Tensors.Engines;

/// <summary>
/// How a Box-Muller draw obtains its first uniform <c>u1</c> (the one under the logarithm) from
/// <see cref="System.Random.NextDouble"/>. The values are stable: the device kernels never see them (the draws are
/// host-side), but serialized options may.
/// </summary>
public enum BoxMullerFirstUniform
{
    /// <summary><c>u1 = NextDouble()</c>.</summary>
    Raw = 0,

    /// <summary><c>u1 = 1 - NextDouble()</c>, in (0, 1], so the logarithm is finite.</summary>
    OneMinus = 1,

    /// <summary><c>u1 = NextDouble()</c>, drawn again while it is zero; <c>u2</c> is drawn after the accepted
    /// <c>u1</c>.</summary>
    RedrawWhileZero = 2,

    /// <summary>The pair <c>(u1, u2)</c> is drawn again, both values, while <c>u1</c> is zero
    /// (<c>do { u1 = NextDouble(); u2 = NextDouble(); } while (u1 &lt;= double.Epsilon)</c>).</summary>
    RedrawPairWhileZero = 3,
}

/// <summary>How a Box-Muller draw obtains its second uniform <c>u2</c> (the one under the trigonometric function).</summary>
public enum BoxMullerSecondUniform
{
    /// <summary><c>u2 = NextDouble()</c>.</summary>
    Raw = 0,

    /// <summary><c>u2 = 1 - NextDouble()</c>.</summary>
    OneMinus = 1,
}

/// <summary>Which of the pair's two normals a Box-Muller draw keeps (one pair per element).</summary>
public enum BoxMullerBranch
{
    /// <summary><c>z = sqrt(-2 ln u1) cos(2 pi u2)</c>.</summary>
    Cosine = 0,

    /// <summary><c>z = sqrt(-2 ln u1) sin(2 pi u2)</c>.</summary>
    Sine = 1,
}

/// <summary>
/// One Box-Muller convention: how <c>u1</c> and <c>u2</c> come from the caller's <see cref="System.Random"/> and which
/// branch is kept, so an engine draw reproduces a scalar loop's stream and values exactly. Each element consumes one
/// pair (two <c>NextDouble()</c> calls, more only under a redraw rule), <c>u1</c> first.
/// </summary>
/// <remarks>
/// The element is <c>mean + stdDev * z</c> with
/// <c>z = Math.Sqrt(-2.0 * Math.Log(Math.Max(u1, FirstUniformFloor))) * Math.Cos(2.0 * Math.PI * u2)</c> (or
/// <c>Math.Sin</c>), evaluated in double in that order. The default value is the textbook form: raw <c>u1</c> and
/// <c>u2</c>, no floor, cosine branch.
/// </remarks>
public readonly struct BoxMullerVariant
{
    /// <summary>Creates a convention.</summary>
    /// <param name="firstUniform">How <c>u1</c> is drawn.</param>
    /// <param name="firstUniformFloor">The lower bound <c>u1</c> is raised to (<c>Math.Max(u1, floor)</c>); 0 leaves it
    /// unchanged. Must be finite and non-negative.</param>
    /// <param name="secondUniform">How <c>u2</c> is drawn.</param>
    /// <param name="branch">The branch kept.</param>
    public BoxMullerVariant(BoxMullerFirstUniform firstUniform, double firstUniformFloor, BoxMullerSecondUniform secondUniform,
        BoxMullerBranch branch)
    {
        if (!(firstUniformFloor >= 0.0) || double.IsInfinity(firstUniformFloor))
            throw new System.ArgumentOutOfRangeException(nameof(firstUniformFloor), firstUniformFloor,
                "The floor must be finite and non-negative.");
        FirstUniform = firstUniform;
        FirstUniformFloor = firstUniformFloor;
        SecondUniform = secondUniform;
        Branch = branch;
    }

    /// <summary>How <c>u1</c> is drawn.</summary>
    public BoxMullerFirstUniform FirstUniform { get; }

    /// <summary>The lower bound applied to <c>u1</c> (0: none).</summary>
    public double FirstUniformFloor { get; }

    /// <summary>How <c>u2</c> is drawn.</summary>
    public BoxMullerSecondUniform SecondUniform { get; }

    /// <summary>The branch kept.</summary>
    public BoxMullerBranch Branch { get; }
}

/// <summary>Which side of the threshold a uniform draw must fall on to count as a success.</summary>
public enum UniformDrawComparison
{
    /// <summary>Success when <c>NextDouble() &lt; threshold</c> (the Bernoulli convention: probability = threshold).</summary>
    DrawBelowThreshold = 0,

    /// <summary>Success when <c>NextDouble() &gt; threshold</c>.</summary>
    DrawAboveThreshold = 1,
}

/// <summary>How a Rademacher (+/-1) draw decides its sign from one <c>NextDouble()</c>; they differ only at exactly
/// 0.5.</summary>
public enum RademacherRule
{
    /// <summary>-1 when <c>NextDouble() &lt; 0.5</c>, otherwise +1.</summary>
    NegativeWhenDrawBelowHalf = 0,

    /// <summary>+1 when <c>NextDouble() &gt; 0.5</c>, otherwise -1.</summary>
    PositiveWhenDrawAboveHalf = 1,
}

/// <summary>How a floating value becomes an integer in <see cref="TensorDataEngineExtensions.TensorCast{TIn, TOut}"/>.
/// The values are stable: the device kernels receive them as integers.</summary>
public enum RoundingMode
{
    /// <summary>Truncation (C# <c>(int)x</c>, <c>Math.Truncate</c>, NumPy <c>astype(int)</c>).</summary>
    TowardZero = 0,

    /// <summary>Nearest, ties to even (<c>Math.Round</c>, <c>torch.round</c>, <c>numpy.rint</c>).</summary>
    ToNearestEven = 1,

    /// <summary>Nearest, ties away from zero (<c>MidpointRounding.AwayFromZero</c>, C <c>round</c>).</summary>
    ToNearestAwayFromZero = 2,

    /// <summary>Toward negative infinity (<c>Math.Floor</c>).</summary>
    TowardNegativeInfinity = 3,

    /// <summary>Toward positive infinity (<c>Math.Ceiling</c>).</summary>
    TowardPositiveInfinity = 4,
}

/// <summary>What <see cref="TensorDataEngineExtensions.TensorCast{TIn, TOut}"/> does with a value the integral target
/// type cannot hold (out of range, or NaN).</summary>
public enum CastOverflowBehavior
{
    /// <summary>Clamp to the target's range; NaN becomes 0.</summary>
    Saturate = 0,

    /// <summary>Throw <see cref="System.OverflowException"/>.</summary>
    Throw = 1,
}
