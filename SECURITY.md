# Security policy

## Reporting a vulnerability

Please report vulnerabilities **privately** through GitHub's private vulnerability reporting for this repository:
open the repository's **Security** tab, choose **Advisories**, then **Report a vulnerability**
(<https://github.com/ooples/AiDotNet.Tensors/security/advisories/new>).

Do not open a public issue, discussion or pull request that contains an exploit, a secret, or details that would let
someone attack users before a fix is released.

Include, where you can: the affected package and version, the target framework and OS, a minimal reproduction, and the
impact you observed. You will get an acknowledgement on the advisory thread, and the fix is coordinated and disclosed
through the same advisory.

## Supported versions

Security fixes are made on `main` and shipped in the next release of the packages built from this repository
(`AiDotNet.Tensors` and the `AiDotNet.Native.*` runtime packages). Only the latest published version receives fixes.

## Scope

In scope: the managed code in this repository, the build and release workflows, and the packaging of the native
runtime packages. Vulnerabilities in a redistributed third-party native binary (CLBlast, oneDNN, MoltenVK, the NVIDIA
CUDA runtime, AMD ROCm) should also be reported upstream to that vendor; please tell us as well so the redistributed
version can be updated. Each native package records the SHA-256 of every binary it ships, and its upstream version, in its
`NATIVE-MANIFEST.json`. One binary states no version: `hipblas.dll` in `AiDotNet.Native.ROCm` has no version
resource or string, so its manifest entry names the release it was taken from (the AMD HIP SDK for Windows 6.4, whose
hipBLAS is 2.4.0); its SHA-256 identifies the exact file.

## Release integrity

Starting with the first release built after this policy was added, release packages carry a GitHub build-provenance
attestation and a CycloneDX SBOM is attached to each GitHub release.

The attestation covers the package bytes the release workflow built, which are the `.nupkg` assets attached to the
GitHub release. It signs a provenance statement about those bytes, not the package itself. The release code-signs the
packages only when its signing certificate is configured, so check a package's own signature separately with
`dotnet nuget verify --all <package>` (below). Verify a release asset's provenance with:

```
gh release download v<version> --repo ooples/AiDotNet.Tensors --pattern "AiDotNet.Tensors.<version>.nupkg"
gh attestation verify AiDotNet.Tensors.<version>.nupkg --repo ooples/AiDotNet.Tensors \
  --signer-workflow ooples/AiDotNet.Tensors/.github/workflows/attest-release.yml
```

`--signer-workflow` accepts only attestations made by the release's attestation workflow, not by any workflow in the
repository.

A package downloaded from NuGet.org does not verify this way: NuGet.org adds its repository signature in the package's
`.signature.p7s` entry (creating it, or extending the author signature), which changes its bytes. Its contents are unchanged, so check a NuGet download against
the attested release asset by comparing every entry except `.signature.p7s`. For example, in PowerShell:

```
Add-Type -AssemblyName System.IO.Compression.FileSystem
function Get-EntryHashes($path) {
  $zip = [IO.Compression.ZipFile]::OpenRead((Resolve-Path $path))
  try {
    $sha = [Security.Cryptography.SHA256]::Create()
    $zip.Entries | Where-Object { $_.FullName -ne '.signature.p7s' } | Sort-Object FullName | ForEach-Object {
      $s = $_.Open(); try { '{0} {1}' -f $_.FullName, [BitConverter]::ToString($sha.ComputeHash($s)).Replace('-', '') } finally { $s.Dispose() }
    }
  } finally { $zip.Dispose() }
}
Compare-Object (Get-EntryHashes attested/AiDotNet.Tensors.<version>.nupkg) (Get-EntryHashes nuget/aidotnet.tensors.<version>.nupkg)
```

It runs in Windows PowerShell 5.1 and PowerShell 7. No output means the NuGet package holds exactly the attested
contents. `dotnet nuget verify --all <package>` then checks the package's signatures: NuGet.org's repository signature,
and the author signature when the release was code-signed.
