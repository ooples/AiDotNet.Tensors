// Copyright (c) AiDotNet. All rights reserved.

namespace AiDotNet.Tensors.Licensing;

/// <summary>
/// Thrown when a tensor-layer persistence operation (save / load /
/// serialize / deserialize) is attempted without an active license
/// covering the operation, and the free-trial budget has been
/// exhausted.
/// </summary>
/// <remarks>
/// <para>
/// Training and inference are never gated — only persistence I/O is.
/// Use one of the following to resolve:
/// <list type="bullet">
///   <item><description>Set the license key via environment variable
///   <c>AIDOTNET_LICENSE_KEY=your-key</c> (the same variable upstream
///   <c>AiDotNet</c> reads — one key, both packages).</description></item>
///   <item><description>Save the key to <c>~/.aidotnet/license.key</c>.</description></item>
///   <item><description>Pass a key to
///   <see cref="PersistenceGuard.SetActiveLicenseKey"/> at startup.</description></item>
/// </list>
/// </para>
/// </remarks>
public sealed class LicenseRequiredException : Exception
{
    /// <summary>
    /// Days the trial was active before the cap fired, or <c>null</c>
    /// when expiration was driven by operation-count rather than time.
    /// </summary>
    public int? TrialDaysElapsed { get; }

    /// <summary>
    /// Number of save+load operations performed during the trial, or
    /// <c>null</c> when expiration was driven by elapsed time.
    /// </summary>
    public int? OperationsPerformed { get; }

    /// <summary>
    /// Reason a license was required at the call site — surfaces the
    /// server's validation status for diagnostic purposes.
    /// </summary>
    public LicenseKeyStatus? KeyStatus { get; }

    /// <summary>
    /// The capability the operation needed (e.g. <c>tensors:save</c>),
    /// when the failure mode is <see cref="LicenseKeyStatus.CapabilityMissing"/>.
    /// </summary>
    public string? RequiredCapability { get; }

    /// <summary>Creates an exception with a default message.</summary>
    public LicenseRequiredException()
        : base(DefaultMessage)
    { }

    /// <summary>Creates an exception with a custom message.</summary>
    public LicenseRequiredException(string message) : base(message) { }

    /// <summary>Creates an exception with a message and inner exception.</summary>
    public LicenseRequiredException(string message, Exception inner) : base(message, inner) { }

    /// <summary>
    /// Creates a fully-specified exception. Used by
    /// <see cref="PersistenceGuard"/> so callers can switch on the
    /// individual properties to render a tailored message.
    /// </summary>
    public LicenseRequiredException(
        string message,
        int? trialDaysElapsed,
        int? operationsPerformed,
        LicenseKeyStatus? keyStatus,
        string? requiredCapability) : base(message)
    {
        TrialDaysElapsed = trialDaysElapsed;
        OperationsPerformed = operationsPerformed;
        KeyStatus = keyStatus;
        RequiredCapability = requiredCapability;
    }

    private const string DefaultMessage =
        "AiDotNet.Tensors persistence operations require a license. The free trial has expired or no license is configured. " +
        "Set AIDOTNET_LICENSE_KEY, save your key to ~/.aidotnet/license.key, or call " +
        "PersistenceGuard.SetActiveLicenseKey at startup. See https://aidotnet.dev/license for details.";
}
