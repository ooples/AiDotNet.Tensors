namespace AiDotNet.Tensors.Engines;

/// <summary>
/// Controls whether element-wise operations broadcast mismatched shapes or reject them.
/// </summary>
/// <remarks>
/// <para>
/// Broadcasting is on by default, matching NumPy, PyTorch and JAX: a <c>[rows, 1]</c> operand
/// stretches across a <c>[rows, cols]</c> one, which is what reduce-then-recombine code —
/// normalization, softmax, standardization — reads like on the page.
/// </para>
/// <para>
/// The cost of that convenience is real, and this type exists to give it back on demand. Implicit
/// broadcasting turns a whole class of loud failures into silent successes: a transposed operand or
/// an off-by-one axis that used to throw now quietly produces a plausible-looking tensor of the
/// wrong shape. Wrapping a run in <see cref="Strict"/> restores throw-on-mismatch, so a test suite
/// or a debugging session can prove that nothing is broadcasting by accident.
/// </para>
/// <para>
/// The setting is per-thread. Tests run in parallel, and a global switch would leak between them —
/// one fixture demanding strictness would silently impose it on everything running alongside.
/// </para>
/// <example>
/// <code>
/// // Prove a suite performs no accidental broadcasts:
/// using (ShapePolicy.Strict())
/// {
///     RunTraining();   // any shape mismatch now throws instead of stretching
/// }
/// </code>
/// </example>
/// </remarks>
public static class ShapePolicy
{
    [ThreadStatic]
    private static int _strictDepth;

    /// <summary>
    /// Whether element-wise operations currently reject mismatched shapes instead of broadcasting.
    /// </summary>
    public static bool IsStrict => _strictDepth > 0;

    /// <summary>
    /// Rejects mismatched shapes for the lifetime of the returned scope.
    /// </summary>
    /// <remarks>
    /// Nests correctly: an inner scope does not re-enable broadcasting when it ends while an outer
    /// scope is still open.
    /// </remarks>
    public static StrictScope Strict() => new();

    /// <summary>
    /// The scope handle returned by <see cref="Strict"/>.
    /// </summary>
    public sealed class StrictScope : IDisposable
    {
        private readonly int _ownerThreadId;
        private int _disposed;

        /// <summary>Enters a strict-shape scope. Prefer <see cref="ShapePolicy.Strict"/>.</summary>
        internal StrictScope()
        {
            _ownerThreadId = Environment.CurrentManagedThreadId;
            _strictDepth++;
        }

        /// <summary>Restores the previous shape policy.</summary>
        public void Dispose()
        {
            if (Environment.CurrentManagedThreadId != _ownerThreadId)
                throw new InvalidOperationException(
                    "A ShapePolicy.Strict scope must be disposed on the thread that created it.");
            if (System.Threading.Interlocked.Exchange(ref _disposed, 1) == 0)
                _strictDepth--;
        }
    }
}
