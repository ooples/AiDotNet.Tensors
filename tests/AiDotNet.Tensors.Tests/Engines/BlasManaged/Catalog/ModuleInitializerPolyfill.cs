#if NETFRAMEWORK
namespace System.Runtime.CompilerServices
{
    /// <summary>
    /// Polyfill for net471. <c>ModuleInitializerAttribute</c> ships in .NET 5+;
    /// Roslyn recognizes the type by name and namespace and emits the appropriate
    /// .cctor call regardless of which assembly defines the attribute. A stub with
    /// the correct fully-qualified name is sufficient.
    /// </summary>
    [System.AttributeUsage(System.AttributeTargets.Method, Inherited = false)]
    internal sealed class ModuleInitializerAttribute : System.Attribute
    {
    }
}
#endif
