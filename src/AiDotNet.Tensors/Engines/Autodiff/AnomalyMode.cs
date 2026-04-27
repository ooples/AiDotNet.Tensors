// Copyright (c) AiDotNet. All rights reserved.
// Anomaly detection with forward-op-site localization: points at the
// original forward call when a NaN/Inf appears in the backward pass.

using System;
using System.Runtime.CompilerServices;

namespace AiDotNet.Tensors.Engines.Autodiff;

/// <summary>
/// Enters a scope where the active <see cref="GradientTape{T}"/>
/// validates every backward gradient for NaN/Inf. On detection,
/// throws an <see cref="AnomalyDetectedException"/> that carries the
/// forward op name and, where available, the source file / line
/// where the offending forward op was recorded.
/// </summary>
/// <remarks>
/// <para><b>Why a dedicated scope instead of a per-tape flag:</b></para>
/// <para>
/// <see cref="GradientTape{T}.DetectAnomaly"/> already gates the
/// NaN/Inf check on the backward walk, but has no API for toggling
/// the check from outside the tape and no way to associate the op
/// with the C# source site. This scope wraps the flag, picks up the
/// <see cref="CallerFilePathAttribute"/> / <see cref="CallerLineNumberAttribute"/>
/// of the surrounding code, and makes sure the scope-level state
/// unwinds cleanly even if the tape is disposed first.
/// </para>
/// <para><b>Zero cost when disabled:</b></para>
/// <para>
/// When no scope is active <see cref="IsActive"/> returns false; the
/// existing fast-path check in
/// <see cref="GradientTape{T}.ComputeGradients"/> skips the NaN/Inf
/// validation entirely. PyTorch pays roughly a 10% tax even when
/// anomaly detection is off because its validation hooks are always
/// dispatched — we don't.
/// </para>
/// <para><b>Composition with nested tapes:</b></para>
/// <para>
/// The scope counter is thread-static, and each
/// <see cref="GradientTape{T}"/> reads <see cref="IsActive"/> at
/// <see cref="GradientTape{T}.ComputeGradients"/> time. Nested tapes
/// all observe the same flag while the outermost scope is active.
/// </para>
/// </remarks>
public sealed class AnomalyModeScope : IDisposable
{
    [ThreadStatic]
    private static int _activeCount;

    private readonly string? _enteredAt;
    private readonly int _enteredAtLine;
    private bool _disposed;

    /// <summary>
    /// Gets whether an anomaly-mode scope is currently active on the
    /// calling thread. Read by <see cref="GradientTape{T}.ComputeGradients"/>
    /// when deciding whether to run the per-step NaN/Inf check.
    /// </summary>
    /// <remarks>
    /// Thread-local — matches the scoping of
    /// <see cref="NoGradScope{T}"/> and
    /// <see cref="InferenceModeScope{T}"/>, so one thread entering
    /// anomaly mode does not impose the ~cost-per-grad check on other
    /// threads running unrelated training. Users who want finer-grained
    /// control can set <see cref="GradientTape{T}.DetectAnomaly"/>
    /// directly on a specific tape instead.
    /// </remarks>
    public static bool IsActive => _activeCount > 0;

    /// <summary>
    /// Gets the source file (<see cref="CallerFilePathAttribute"/>)
    /// where the scope was entered, if captured.
    /// </summary>
    public string? EnteredAtFile => _enteredAt;

    /// <summary>
    /// Gets the line number where the scope was entered.
    /// </summary>
    public int EnteredAtLine => _enteredAtLine;

    /// <summary>
    /// Creates a new anomaly-mode scope. The caller's file / line is
    /// captured automatically via <see cref="CallerFilePathAttribute"/>
    /// / <see cref="CallerLineNumberAttribute"/> so a rethrow inside
    /// the scope can tell the user where anomaly was turned on.
    /// </summary>
    public AnomalyModeScope(
        [CallerFilePath] string? enteredAt = null,
        [CallerLineNumber] int enteredAtLine = 0)
    {
        _enteredAt = enteredAt;
        _enteredAtLine = enteredAtLine;
        _activeCount++;
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _activeCount--;
    }
}

/// <summary>
/// Thrown when a NaN or Inf appears in a gradient while an
/// <see cref="AnomalyModeScope"/> (or per-tape
/// <see cref="GradientTape{T}.DetectAnomaly"/>) is active. Carries
/// the forward op name and — when the op was recorded with
/// <see cref="CallerFilePathAttribute"/> info — the source site.
/// </summary>
/// <remarks>
/// Derives from <see cref="ArithmeticException"/> so existing catch
/// clauses written for NaN/Inf arithmetic failures (including the
/// pre-anomaly-scope tape path that threw <c>ArithmeticException</c>
/// directly) continue to match.
/// </remarks>
public sealed class AnomalyDetectedException : ArithmeticException
{
    /// <summary>Gets the name of the forward op whose backward
    /// produced a NaN/Inf.</summary>
    public string OperationName { get; }

    /// <summary>Gets the source file where the offending forward op
    /// was called, if captured.</summary>
    public string? ForwardCallerFile { get; }

    /// <summary>Gets the line number in <see cref="ForwardCallerFile"/>
    /// where the op was called, if captured.</summary>
    public int ForwardCallerLine { get; }

    /// <summary>
    /// Creates a new anomaly exception.
    /// </summary>
    /// <param name="operationName">Name of the backward op that
    /// emitted the NaN/Inf gradient.</param>
    /// <param name="forwardCallerFile">File path of the forward call
    /// site, if recorded.</param>
    /// <param name="forwardCallerLine">Line number of the forward
    /// call site, if recorded.</param>
    public AnomalyDetectedException(
        string operationName,
        string? forwardCallerFile = null,
        int forwardCallerLine = 0)
        : base(BuildMessage(operationName, forwardCallerFile, forwardCallerLine))
    {
        OperationName = operationName;
        ForwardCallerFile = forwardCallerFile;
        ForwardCallerLine = forwardCallerLine;
    }

    private static string BuildMessage(string op, string? file, int line)
    {
        if (file is null) return $"Anomaly detected in backward for '{op}' (NaN/Inf in gradient).";
        return $"Anomaly detected in backward for '{op}' (NaN/Inf in gradient). "
             + $"Forward op recorded at {file}:{line}.";
    }
}
