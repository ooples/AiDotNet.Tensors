// Copyright (c) AiDotNet. All rights reserved.

using System.Collections.Generic;

namespace AiDotNet.Tensors.Licensing;

/// <summary>
/// Result of a single <see cref="LicenseValidator.Validate"/> call.
/// Mirrors the upstream <c>AiDotNet.Models.LicenseValidationResult</c>
/// shape, with one extension — a server-returned set of
/// <see cref="Capabilities"/> strings used by
/// <see cref="PersistenceGuard"/> to decide whether the key is
/// authorised for the specific operation being attempted.
/// </summary>
/// <remarks>
/// <para>
/// The capability convention is namespace-prefixed strings — e.g.
/// <c>tensors:save</c>, <c>tensors:load</c>, <c>model:save</c>,
/// <c>model:load</c>. The validation server attaches the set
/// corresponding to the purchased tier; a missing capability surfaces
/// as <see cref="LicenseKeyStatus.CapabilityMissing"/> rather than
/// as a generic <see cref="LicenseKeyStatus.Invalid"/>, so the user
/// gets a clear "your tier doesn't include this operation" error
/// instead of the misleading "key invalid".
/// </para>
/// </remarks>
public sealed class LicenseValidationResult
{
    /// <summary>The validation status of the license key.</summary>
    public LicenseKeyStatus Status { get; }

    /// <summary>
    /// The subscription tier associated with this key (e.g.
    /// <c>tensors-only</c>, <c>community</c>, <c>pro</c>,
    /// <c>enterprise</c>), or <c>null</c> if unknown.
    /// </summary>
    public string? Tier { get; }

    /// <summary>The expiry of this license, or <c>null</c> if perpetual.</summary>
    public DateTimeOffset? ExpiresAt { get; }

    /// <summary>UTC timestamp this validation completed.</summary>
    public DateTimeOffset ValidatedAt { get; }

    /// <summary>An optional human-readable message from the server.</summary>
    public string? Message { get; }

    /// <summary>
    /// Set of capability strings this key authorises. The
    /// <see cref="PersistenceGuard"/> compares the operation it's about
    /// to authorise (e.g. <c>tensors:save</c>) against this set; if the
    /// capability is missing, <see cref="HasCapability"/> returns
    /// <c>false</c> and the guard throws.
    /// </summary>
    /// <remarks>
    /// Typed as <see cref="IReadOnlyCollection{T}"/> rather than
    /// <c>IReadOnlySet&lt;string&gt;</c> for net471 compatibility — the
    /// latter type is only available on .NET 6+. Membership is checked
    /// through <see cref="HasCapability"/>, which uses the underlying
    /// hash set so lookup stays O(1).
    /// </remarks>
    public IReadOnlyCollection<string> Capabilities => _capabilities;

    private readonly HashSet<string> _capabilities;

    /// <summary>
    /// True iff <paramref name="capability"/> is present in
    /// <see cref="Capabilities"/>. Comparison is ordinal
    /// case-sensitive — capability strings are server-defined and
    /// must round-trip exactly.
    /// </summary>
    public bool HasCapability(string capability) => _capabilities.Contains(capability);

    /// <summary>
    /// Creates a new validation result.
    /// </summary>
    public LicenseValidationResult(
        LicenseKeyStatus status,
        string? tier,
        DateTimeOffset? expiresAt,
        DateTimeOffset validatedAt,
        string? message,
        IEnumerable<string>? capabilities)
    {
        Status = status;
        Tier = tier;
        ExpiresAt = expiresAt;
        ValidatedAt = validatedAt;
        Message = message;
        _capabilities = capabilities is null
            ? new HashSet<string>(StringComparer.Ordinal)
            : new HashSet<string>(capabilities, StringComparer.Ordinal);
    }
}
