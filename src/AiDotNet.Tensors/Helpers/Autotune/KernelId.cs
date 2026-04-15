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
    /// e.g. <c>"gemm__opencl-v1"</c>. Safe characters
    /// (<c>[A-Za-z0-9._-]</c>) are passed through unchanged; any other byte
    /// (including spaces, slashes, multibyte UTF-8) is encoded as <c>%HH</c>
    /// using its UTF-8 byte value. Double-underscore (<c>__</c>) separates the
    /// category from the name so a filename regex can round-trip a KernelId.
    ///
    /// <para><b>Injectivity:</b> distinct (Category, Name) pairs always produce
    /// distinct stems. The previous implementation collapsed any unsafe char to
    /// <c>_</c>, which made <c>"foo bar"</c> and <c>"foo_bar"</c> alias to the
    /// same cache file — a real risk for kernel names that embed shape or
    /// generated-id metadata.</para>
    /// </summary>
    public string ToFileStem() => $"{Encode(Category)}__{Encode(Name)}";

    private static string Encode(string input)
    {
        if (input is null) return string.Empty;

        // Fast path: all bytes already safe → pass through unchanged.
        var utf8 = System.Text.Encoding.UTF8.GetBytes(input);
        bool needsEncoding = false;
        for (int i = 0; i < utf8.Length; i++)
        {
            if (!IsSafe(utf8[i])) { needsEncoding = true; break; }
        }
        if (!needsEncoding) return input;

        // %HH encoding using UTF-8 byte values — injective (any two distinct
        // strings produce distinct byte sequences and therefore distinct stems).
        var sb = new System.Text.StringBuilder(utf8.Length + 8);
        for (int i = 0; i < utf8.Length; i++)
        {
            byte b = utf8[i];
            if (IsSafe(b))
            {
                sb.Append((char)b);
            }
            else
            {
                sb.Append('%');
                sb.Append(HexChar(b >> 4));
                sb.Append(HexChar(b & 0xF));
            }
        }
        return sb.ToString();
    }

    private static bool IsSafe(byte b)
    {
        // Keep a-z, A-Z, 0-9, dash, underscore, period. '%' itself is NOT safe
        // (so it gets encoded as %25, preserving injectivity).
        return (b >= (byte)'a' && b <= (byte)'z')
            || (b >= (byte)'A' && b <= (byte)'Z')
            || (b >= (byte)'0' && b <= (byte)'9')
            ||  b == (byte)'-' || b == (byte)'_' || b == (byte)'.';
    }

    private static char HexChar(int nibble)
        => (char)(nibble < 10 ? ('0' + nibble) : ('A' + nibble - 10));
}
