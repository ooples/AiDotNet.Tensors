using System.Runtime.InteropServices;
using Microsoft.Win32.SafeHandles;

namespace AiDotNet.Tensors.Helpers.Autotune;

internal enum QuarantineCommitMode
{
    Unsupported,
    LinuxDirectorySync
}

[Flags]
internal enum LinuxDirectoryOpenFlags
{
    ReadOnly = 0,
    ArmDirectory = 1 << 14,
    GenericDirectory = 1 << 16,
    CloseOnExec = 1 << 19
}

/// <summary>The native boundary, injectable per store for deterministic I/O failure tests.</summary>
internal interface IQuarantineCommitOperations
{
    QuarantineCommitMode Mode { get; }
    bool MoveNewFile(string source, string destination);
    bool FlushDirectory(string directory);
}

/// <summary>Publishes an already flushed file without replacing an existing tombstone.</summary>
internal static class QuarantineReceiptCommit
{
    internal static IQuarantineCommitOperations Native { get; } = new NativeCommitOperations();

    internal static bool TryGetDirectoryFlags(Architecture architecture, out LinuxDirectoryOpenFlags flags)
    {
        // Linux ARM32 overrides O_DIRECTORY; x86/x64/ARM64 use the generic UAPI.
        // Unknown ABIs are unsupported rather than guessing syscall flag values.
        switch (architecture)
        {
            case Architecture.Arm:
                flags = LinuxDirectoryOpenFlags.ArmDirectory | LinuxDirectoryOpenFlags.CloseOnExec;
                return true;
            case Architecture.X86:
            case Architecture.X64:
            case Architecture.Arm64:
                flags = LinuxDirectoryOpenFlags.GenericDirectory | LinuxDirectoryOpenFlags.CloseOnExec;
                return true;
            default:
                flags = LinuxDirectoryOpenFlags.ReadOnly;
                return false;
        }
    }

    internal static bool TryCommit(string pending, string destination, IQuarantineCommitOperations operations)
    {
        QuarantineCommitMode mode = operations.Mode;
        if (mode != QuarantineCommitMode.Unsupported && mode != QuarantineCommitMode.LinuxDirectorySync)
            return false;

        string? directory = Path.GetDirectoryName(destination);
        if (string.IsNullOrEmpty(directory) || !Path.IsPathRooted(directory) ||
            !string.Equals(directory, Path.GetDirectoryName(pending), StringComparison.Ordinal)) return false;

        if (!operations.MoveNewFile(pending, destination)) return false;
        // Retain the best-effort tombstone even without a verified directory barrier;
        // it protects ordinary restarts, but cannot promise power-loss safety.
        if (mode == QuarantineCommitMode.Unsupported) return false;

        // fsync(file) does not persist the directory entry created by rename.
        // Include ancestors: the journal directory itself may have just been created.
        // A failed barrier leaves the visible tombstone intact but cannot claim durability.
        while (!string.IsNullOrEmpty(directory))
        {
            if (!operations.FlushDirectory(directory)) return false;
            directory = Path.GetDirectoryName(directory);
        }
        return true;
    }

    private sealed class NativeCommitOperations : IQuarantineCommitOperations
    {
        public QuarantineCommitMode Mode => RuntimeInformation.IsOSPlatform(OSPlatform.Linux)
            && TryGetDirectoryFlags(RuntimeInformation.ProcessArchitecture, out _)
            ? QuarantineCommitMode.LinuxDirectorySync : QuarantineCommitMode.Unsupported;

        public bool MoveNewFile(string source, string destination)
        {
            File.Move(source, destination);
            return true;
        }

        public bool FlushDirectory(string directory)
        {
            if (Mode != QuarantineCommitMode.LinuxDirectorySync) return false;
            if (!TryGetDirectoryFlags(RuntimeInformation.ProcessArchitecture, out LinuxDirectoryOpenFlags flags)) return false;
            int descriptor = OpenDirectory(directory, flags);
            if (descriptor < 0) return false;
            using var handle = new DirectoryHandle(descriptor);
            return Synchronize(descriptor) == 0;
        }
    }

    [DllImport("libc", EntryPoint = "open", SetLastError = true)]
    private static extern int OpenDirectory([MarshalAs(UnmanagedType.LPUTF8Str)] string path, LinuxDirectoryOpenFlags flags);

    [DllImport("libc", EntryPoint = "fsync", SetLastError = true)]
    private static extern int Synchronize(int descriptor);

    [DllImport("libc", EntryPoint = "close")]
    private static extern int CloseDirectory(int descriptor);

    private sealed class DirectoryHandle : SafeHandleMinusOneIsInvalid
    {
        internal DirectoryHandle(int descriptor) : base(true) => SetHandle(new IntPtr(descriptor));
        protected override bool ReleaseHandle() => CloseDirectory(handle.ToInt32()) == 0;
    }
}
