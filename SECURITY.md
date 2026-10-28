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
version can be updated. Each native package records the exact upstream version and SHA-256 of every binary it ships in
its `NATIVE-MANIFEST.json`.

## Release integrity

Starting with the first release built after this policy was added, release packages carry a GitHub build-provenance
attestation and a CycloneDX SBOM is attached to each GitHub release. Verify a downloaded package with:

```
gh attestation verify AiDotNet.Tensors.<version>.nupkg --repo ooples/AiDotNet.Tensors
```
