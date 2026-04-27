// Copyright (c) AiDotNet. All rights reserved.

namespace AiDotNet.Tensors.Licensing;

/// <summary>
/// Outcome of validating an <see cref="AiDotNetTensorsLicenseKey"/>.
/// Mirrors the upstream <c>AiDotNet.Enums.LicenseKeyStatus</c> enum so
/// a result returned by either layer's validator carries a stable
/// meaning across the package boundary.
/// </summary>
public enum LicenseKeyStatus
{
    /// <summary>The key is valid and active.</summary>
    Active,

    /// <summary>The key has expired.</summary>
    Expired,

    /// <summary>The key has been revoked by an administrator.</summary>
    Revoked,

    /// <summary>The key has reached its activation/seat limit.</summary>
    SeatLimitReached,

    /// <summary>
    /// The key string is malformed or unknown to the validation server.
    /// </summary>
    Invalid,

    /// <summary>
    /// The server was unreachable; the validator is operating against a
    /// cached result inside its offline grace window.
    /// </summary>
    ValidationPending,

    /// <summary>
    /// The key authenticates but does not include the capability the
    /// caller asked for. E.g. a tier that grants <c>model:save</c>
    /// won't satisfy a <c>tensors:save</c> request unless its tier
    /// also includes the tensors capability set.
    /// </summary>
    CapabilityMissing,
}
