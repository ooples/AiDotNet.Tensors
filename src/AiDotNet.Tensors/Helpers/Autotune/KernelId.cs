namespace AiDotNet.Tensors.Helpers.Autotune;

/// <summary>
/// Identifies a tunable kernel family. The <see cref="Category"/> groups
/// kernels with the same semantics (e.g. <c>"gemm"</c>, <c>"sdpa"</c>,
/// <c>"conv2d"</c>); the <see cref="Name"/> distinguishes implementations
/// within a category (e.g. <c>"opencl-v1"</c>, <c>"cuda-tensorcore-v2"</c>,
/// <c>"cpu-avx2-blocked"</c>).
/// </summary>
/// <remarks>
/// Used by <see cref="AutotuneCache"/> as the primary key prefix. Category
/// is separated from Name so consumers can evolve a kernel's internal
/// identifier (bumping the version suffix) without invalidating the whole
/// category's cache — only entries with the old <c>Name</c> go stale.
/// </remarks>
public readonly record struct KernelId(string Category, string Name)
{
    /// <summary>
    /// Returns a filesystem-safe string suitable for use as a filename prefix,
    /// e.g. <c>"gemm__opencl-v1"</c>. Double-underscore separates the category
    /// from the name so a filename regex can round-trip a KernelId.
    /// </summary>
    public string ToFileStem() => $"{Sanitize(Category)}__{Sanitize(Name)}";

    private static string Sanitize(string input)
    {
        // Replace any char that isn't safe across Windows + Linux filenames.
        // Keep a-z, A-Z, 0-9, dash, underscore, period.
        var chars = input.ToCharArray();
        for (int i = 0; i < chars.Length; i++)
        {
            char c = chars[i];
            bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')
                   || (c >= '0' && c <= '9') || c == '-' || c == '_' || c == '.';
            if (!ok) chars[i] = '_';
        }
        return new string(chars);
    }
}
