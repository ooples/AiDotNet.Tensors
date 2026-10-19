using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL
{
    // The native CLBlast library tunes its GEMM kernels from a built-in database. A device missing from it runs generic
    // parameters; these are measured replacements, applied once per backend before the first library GEMM.
    public sealed partial class OpenClBackend
    {
        private static readonly string[] XgemmParameterNames =
            { "GEMMK", "KREG", "KWG", "KWI", "MDIMA", "MDIMC", "MWG", "NDIMB", "NDIMC", "NWG", "SA", "SB", "STRM", "STRN", "VWM", "VWN" };

        private static readonly string[] XgemmDirectParameterNames =
            { "KWID", "MDIMAD", "MDIMCD", "NDIMBD", "NDIMCD", "PADA", "PADB", "VWMD", "VWND", "WGD" };

        // Radeon RX 5500 XT (gfx1012), from a sampled sweep through CLBlastOverrideParameters:
        //   Xgemm on 1024^3: 2.017 ms with the library's parameters, 0.951 ms with these (2048^3: 10.1 -> 8.8 ms).
        //   XgemmDirect on 36 strided-batched 512x49x512 products (the late Winograd GEMMs): 1.592 -> 1.252 ms.
        private static readonly (string DeviceFragment, ulong[] Xgemm, ulong[] XgemmDirect)[] NativeClBlastTunings =
        {
            ("gfx1012", new ulong[] { 0, 1, 32, 2, 16, 16, 64, 8, 8, 128, 1, 1, 0, 1, 4, 4 },
                new ulong[] { 2, 8, 8, 16, 16, 1, 1, 2, 1, 32 }),
        };

        private int _nativeClBlastTuned;

        private void EnsureNativeClBlastTuned()
        {
            if (_context == null || System.Threading.Interlocked.Exchange(ref _nativeClBlastTuned, 1) != 0) return;
            string device = _context.DeviceName;
            foreach (var (fragment, xgemm, direct) in NativeClBlastTunings)
            {
                if (device.IndexOf(fragment, StringComparison.OrdinalIgnoreCase) < 0) continue;
                OverrideNative("Xgemm", XgemmParameterNames, xgemm);
                OverrideNative("XgemmDirect", XgemmDirectParameterNames, direct);
                return;
            }
        }

        private void OverrideNative(string kernel, string[] names, ulong[] values)
        {
            if (_context == null) return;
            var native = new UIntPtr[values.Length];
            for (int i = 0; i < values.Length; i++) native[i] = (UIntPtr)values[i];
            ClBlastNative.StatusCode status;
            try
            {
                status = ClBlastNative.OverrideParameters(_context.Device, kernel, ClBlastNative.PrecisionSingle,
                    (UIntPtr)names.Length, names, native);
            }
            catch (EntryPointNotFoundException)
            {
                // An older CLBlast without CLBlastOverrideParameters keeps its own database parameters.
                return;
            }
            if (status != ClBlastNative.StatusCode.Success)
                System.Diagnostics.Trace.TraceWarning($"CLBlast {kernel} parameter override rejected: {status}");
        }
    }
}
