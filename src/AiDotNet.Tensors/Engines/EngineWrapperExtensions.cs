using System.Diagnostics.CodeAnalysis;

namespace AiDotNet.Tensors.Engines;

/// <summary>
/// An <see cref="IEngine"/> that decorates another engine: it forwards (and may observe) calls to
/// <see cref="Inner"/> instead of executing them itself.
/// </summary>
/// <remarks>
/// <para>
/// A wrapper such as a tracing engine is installed as <see cref="AiDotNetEngine.Current"/> in place of the engine it
/// wraps. Code that needs the concrete engine (or one of the capability interfaces it implements) must therefore not
/// test the engine's type directly — <c>engine is DirectGpuTensorEngine</c> is false for a wrapper around one — but go
/// through <see cref="EngineWrapperExtensions"/>, which looks through any chain of wrappers.
/// </para>
/// <para>The repository's TENS0020 analyzer rule flags direct type tests of an <see cref="IEngine"/>.</para>
/// </remarks>
public interface IEngineWrapper
{
    /// <summary>The engine this wrapper forwards to. It may itself be a wrapper.</summary>
    IEngine Inner { get; }
}

/// <summary>
/// Wrapper-aware type tests for <see cref="IEngine"/>: the equivalents of <c>is</c>, <c>as</c>, a cast and
/// <c>GetType()</c> that see through any chain of <see cref="IEngineWrapper"/>s.
/// </summary>
/// <remarks>
/// <para>
/// Each helper walks the chain from the outermost engine inward (the engine itself, then each
/// <see cref="IEngineWrapper.Inner"/>) and stops at the first engine of the requested type, so with no wrapper
/// involved it behaves exactly like the corresponding language operator.
/// </para>
/// <para><b>For Beginners:</b> use <c>engine.Is&lt;CpuEngine&gt;()</c> instead of <c>engine is CpuEngine</c>,
/// <c>engine.TryGet&lt;DirectGpuTensorEngine&gt;(out var gpu)</c> instead of <c>engine is DirectGpuTensorEngine gpu</c>,
/// <c>engine.As&lt;CpuEngine&gt;()</c> instead of <c>engine as CpuEngine</c>, <c>engine.Cast&lt;CpuEngine&gt;()</c>
/// instead of <c>(CpuEngine)engine</c>, and <c>engine.Unwrap().GetType()</c> instead of <c>engine.GetType()</c>.</para>
/// </remarks>
public static class EngineWrapperExtensions
{
    // A wrapper chain this deep is a cycle (a wrapper that wraps itself, directly or not), not a real decoration.
    private const int MaxWrapperDepth = 1024;

    /// <summary>
    /// Returns the innermost engine behind any chain of <see cref="IEngineWrapper"/>s: <paramref name="engine"/>
    /// itself when it is not a wrapper, and <see langword="null"/> when it is <see langword="null"/>.
    /// </summary>
    /// <param name="engine">The engine, possibly a wrapper.</param>
    /// <returns>The first engine in the chain that is not an <see cref="IEngineWrapper"/>.</returns>
    /// <exception cref="InvalidOperationException">The wrapper chain is cyclic.</exception>
    [return: NotNullIfNotNull(nameof(engine))]
    public static IEngine? Unwrap(this IEngine? engine)
    {
        int depth = 0;
        while (engine is IEngineWrapper wrapper)
        {
            if (++depth > MaxWrapperDepth) throw CyclicChain();
            engine = wrapper.Inner;
        }

        return engine;
    }

    /// <summary>
    /// Returns <see langword="true"/> when <paramref name="engine"/>, or an engine it wraps, is a
    /// <typeparamref name="TEngine"/> (the wrapper-aware <c>engine is TEngine</c>).
    /// </summary>
    /// <typeparam name="TEngine">The engine class or capability interface to test for.</typeparam>
    /// <param name="engine">The engine, possibly a wrapper or <see langword="null"/>.</param>
    public static bool Is<TEngine>(this IEngine? engine) where TEngine : class
        => engine.As<TEngine>() is not null;

    /// <summary>
    /// Returns the first engine in the wrapper chain of <paramref name="engine"/> that is a
    /// <typeparamref name="TEngine"/>, or <see langword="null"/> (the wrapper-aware <c>engine as TEngine</c>).
    /// </summary>
    /// <typeparam name="TEngine">The engine class or capability interface to look for.</typeparam>
    /// <param name="engine">The engine, possibly a wrapper or <see langword="null"/>.</param>
    public static TEngine? As<TEngine>(this IEngine? engine) where TEngine : class
    {
        int depth = 0;
        for (var current = engine; current is not null;)
        {
            if (current is TEngine match) return match;
            if (current is not IEngineWrapper wrapper) break;
            if (++depth > MaxWrapperDepth) throw CyclicChain();
            current = wrapper.Inner;
        }

        return null;
    }

    /// <summary>
    /// Finds the first engine in the wrapper chain of <paramref name="engine"/> that is a
    /// <typeparamref name="TEngine"/> (the wrapper-aware <c>engine is TEngine result</c>).
    /// </summary>
    /// <typeparam name="TEngine">The engine class or capability interface to look for.</typeparam>
    /// <param name="engine">The engine, possibly a wrapper or <see langword="null"/>.</param>
    /// <param name="result">The engine found, or <see langword="null"/>.</param>
    /// <returns><see langword="true"/> when an engine of the type was found.</returns>
    /// <remarks>Declare the result with <c>out var</c> (<c>engine.TryGet&lt;CpuEngine&gt;(out var cpu)</c>): it is
    /// typed <typeparamref name="TEngine"/>? and known to be non-null when the method returns <see langword="true"/>.</remarks>
    public static bool TryGet<TEngine>(this IEngine? engine, [NotNullWhen(true)] out TEngine? result)
        where TEngine : class
    {
        result = engine.As<TEngine>();
        return result is not null;
    }

    /// <summary>
    /// Returns the first engine in the wrapper chain of <paramref name="engine"/> that is a
    /// <typeparamref name="TEngine"/>, and throws when there is none (the wrapper-aware <c>(TEngine)engine</c>).
    /// Like a cast, a <see langword="null"/> engine yields <see langword="null"/>.
    /// </summary>
    /// <typeparam name="TEngine">The engine class or capability interface to cast to.</typeparam>
    /// <param name="engine">The engine, possibly a wrapper or <see langword="null"/>.</param>
    /// <exception cref="InvalidCastException">No engine in the chain is a <typeparamref name="TEngine"/>.</exception>
    [return: NotNullIfNotNull(nameof(engine))]
    public static TEngine? Cast<TEngine>(this IEngine? engine) where TEngine : class
    {
        if (engine is null) return null;
        return engine.As<TEngine>() ?? throw new InvalidCastException(
            $"Unable to cast engine of type '{engine.Unwrap().GetType().FullName}' to type '{typeof(TEngine).FullName}'.");
    }

    private static InvalidOperationException CyclicChain() =>
        new("The IEngineWrapper chain is cyclic: a wrapper wraps itself, directly or through other wrappers.");
}
