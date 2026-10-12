using System;

namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// Marks a backend capability interface as a vendor-specific route (for example CUDA direct PTX)
/// that may be implemented by fewer than all six backends.
/// </summary>
/// <remarks>
/// The operation itself must still run natively on every backend: <paramref name="crossBackendOperations"/>
/// names the backend methods that compute the same operation, and the TENS0005 analyzer rule fails
/// the build unless each one exists on all six backends. A route is a faster path, never the only path.
/// </remarks>
[AttributeUsage(AttributeTargets.Interface, AllowMultiple = false, Inherited = false)]
internal sealed class BackendSpecificRouteAttribute : Attribute
{
    public BackendSpecificRouteAttribute(string reason, params string[] crossBackendOperations)
    {
        Reason = reason;
        CrossBackendOperations = crossBackendOperations;
    }

    /// <summary>Why only some backends carry this route.</summary>
    public string Reason { get; }

    /// <summary>Backend methods, present on every backend, that compute the same operation.</summary>
    public string[] CrossBackendOperations { get; }
}
