# NativeAOT enforcement extraction — design & execution plan

Goal: move the most patch-attractive enforcement decision out of trivially
decompilable managed IL into a **NativeAOT-compiled native component**, so
defeating it requires native reverse-engineering instead of a 2-minute dnSpy
edit. This is a real, multi-platform engineering initiative — it cannot be a
single code edit, and it has an **honest benefit ceiling** (below). This doc is
the complete plan so it can be executed when the multi-platform CI is committed.

## Honest benefit & ceiling (read first)

- **What it buys:** the *check itself* (signature verify / entitlement decision)
  becomes native machine code — no symbol names, no IL, far harder to patch.
- **What it does NOT buy:** the **managed call site** that invokes the native
  check and decides whether to throw is *still managed IL*, so an attacker can
  patch the caller to ignore the result, or replace the native `.dll`/`.so`
  with a stub. To make this meaningful, the native component must gate something
  the app **genuinely needs**, not just return a bool — see "Binding" below.
- Net: this raises attacker cost from "minutes" to "hours/days", but does not
  make client-side enforcement absolute. Prioritize signing + entitlement +
  contract first; do this only if the threat model justifies the cost.

## Architecture

```
AiDotNet.Tensors (managed, net471 + net10.0)
        │  P/Invoke  [DllImport("aidotnet_license_native")]
        ▼
aidotnet_license_native  (NativeAOT -p:NativeLib=Shared)
  - RSA-verify the signed entitlement (embedded public key, native)
  - return a license-derived value (see Binding), not just a bool
  shipped as runtimes/{rid}/native/{lib} in the NuGet package
```

### Binding (so it can't be no-op'd)

Returning a bool is patch-trivial. Instead, have the native component derive a
small value the managed persistence path actually consumes — e.g. it returns a
verified capability set / a key the codec uses to finalize a header — so a
stubbed native lib produces wrong output rather than "free access". Keep this
*cheap* (verify once, cache) so legitimate users pay ~nothing. Designing this
binding without hurting legit performance/usability is the hard part and must be
reviewed carefully — a bad binding degrades the product for paying customers.

## Project layout

```text
src/AiDotNet.Native.LicenseGuard/        # NativeAOT shared lib
  AiDotNet.Native.LicenseGuard.csproj     # <PublishAot>true</PublishAot>, <NativeLib>Shared</NativeLib>
  Exports.cs                              # [UnmanagedCallersOnly(EntryPoint="...")]
src/AiDotNet.Tensors/Licensing/
  NativeLicenseGuard.cs                   # [DllImport] wrapper + managed fallback
```

The managed wrapper must degrade safely if the native lib is absent for a RID
(fall back to the managed `SignedEntitlement` path) so unsupported platforms
still work.

## Multi-platform build (the blocker that needs your CI)

NativeAOT produces a binary **per RID**, each built **on its own OS** with that
platform's toolchain. Add a matrix job to the release pipeline:

```yaml
build-native-guard:
  strategy:
    matrix:
      include:
        - { os: windows-latest, rid: win-x64,    lib: aidotnet_license_native.dll }
        - { os: ubuntu-latest,  rid: linux-x64,  lib: libaidotnet_license_native.so }
        - { os: macos-latest,   rid: osx-arm64,  lib: libaidotnet_license_native.dylib }
        - { os: macos-13,       rid: osx-x64,    lib: libaidotnet_license_native.dylib }
  runs-on: ${{ matrix.os }}
  steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-dotnet@v4
      with: { dotnet-version: '10.0.x' }
    # Linux needs clang + zlib; Windows needs the MSVC C++ build tools (preinstalled on windows-latest)
    - run: dotnet publish src/AiDotNet.Native.LicenseGuard -c Release -r ${{ matrix.rid }} -p:PublishAot=true
    - uses: actions/upload-artifact@v4
      with: { name: native-${{ matrix.rid }}, path: '**/publish/${{ matrix.lib }}' }
```

The pack job then downloads all native artifacts and includes them as
`runtimes/{rid}/native/{lib}` pack items (static `<None>` with `PackagePath`),
exactly like the other `AiDotNet.Native.*` packages already do.

## Why this isn't a single edit / can't be done from one machine

- Each native binary must be compiled on its target OS (cross-compiling AOT is
  not generally supported) → needs the 4-runner matrix above.
- Adding the native dependency to the **shipping** build before all RIDs exist
  would break Linux CI and non-Windows consumers — so it must land together
  with the matrix, behind a safe managed fallback.
- NativeAOT requires the platform C toolchain on each runner (GitHub-hosted
  runners provide them; self-hosted must install clang/MSVC).

## Execution checklist

1. Create `src/AiDotNet.Native.LicenseGuard` (NativeLib=Shared, PublishAot).
2. Implement the RSA-verify + **binding** export; design the binding so a stub
   produces wrong output, reviewed for zero legit-user perf impact.
3. Add `NativeLicenseGuard.cs` P/Invoke wrapper with managed fallback.
4. Add the `build-native-guard` matrix job + wire artifacts into pack.
5. Validate on all 4 RIDs end-to-end before enabling in `PersistenceGuard`.

Until step 4 is in place, this stays **out of the shipping build** to protect CI
and consumers.
