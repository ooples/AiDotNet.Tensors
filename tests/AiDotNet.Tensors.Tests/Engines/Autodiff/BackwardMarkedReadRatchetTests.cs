using System;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

// A backward that reads a tensor through AsSpan(), AsWritableSpan(), AsMemory() or GetDataArray() marks its storage
// escaped, and an escaped temporary can never be handed back to the pool: it stays on the heap until a collection.
// Backward kernels that keep none of the storage read through AsSpanUnmarked under KeepAliveScope instead. This
// ratchet counts the marked reads left in the backward sources: a new one fails it, and removing one means lowering
// the count here.
public sealed class BackwardMarkedReadRatchetTests
{
    private static readonly Regex MarkedRead =
        new(@"\.AsSpan\(\)|\.AsWritableSpan\(\)|\.AsMemory\(\)|GetDataArray\(\)", RegexOptions.Compiled);

    // Marked reads per backward source file. Files not listed have none.
    private const int BackwardFunctionsMarkedReads = 73;

    [Fact]
    public void BackwardSources_AddNoMarkedReads()
    {
        string dir = Path.Combine(RepoRoot(), "src", "AiDotNet.Tensors", "Engines", "Autodiff");
        foreach (var file in Directory.GetFiles(dir, "BackwardFunctions*.cs"))
        {
            int count = MarkedRead.Matches(File.ReadAllText(file)).Count;
            string name = Path.GetFileName(file);
            int allowed = name == "BackwardFunctions.cs" ? BackwardFunctionsMarkedReads : 0;
            Assert.True(count <= allowed,
                $"{name} has {count} marked storage reads (AsSpan/AsWritableSpan/AsMemory/GetDataArray), the ratchet allows " +
                $"{allowed}. Read through AsSpanUnmarked under KeepAliveScope when the kernel keeps none of the storage.");
            Assert.True(count >= allowed,
                $"{name} is down to {count} marked storage reads; lower the ratchet in this test from {allowed} to {count}.");
        }
    }

    private static string RepoRoot()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null && !File.Exists(Path.Combine(dir.FullName, "src", "AiDotNet.Tensors", "AiDotNet.Tensors.csproj")))
            dir = dir.Parent;
        return dir?.FullName ?? throw new InvalidOperationException("Repository root not found.");
    }
}
