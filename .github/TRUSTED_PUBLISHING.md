# NuGet Trusted Publishing

AiDotNet.Tensors publishes through GitHub Actions OIDC. The repository does not store a long-lived NuGet API key.

## NuGet.org policy

Create the policy at [nuget.org](https://www.nuget.org/account/trustedpublishing) with these exact values:

| Field | Value |
|-------|-------|
| Owner | `ooples` |
| Repository | `AiDotNet.Tensors` |
| Workflow file | `automated-release.yml` |
| Environment | Leave blank |

Enter only `automated-release.yml`, not `.github/workflows/automated-release.yml`. NuGet matches the owner, repository, workflow file, and environment values case-insensitively. The policy must authorize every package produced by the release workflow.

See the official [NuGet trusted-publishing documentation](https://learn.microsoft.com/en-us/nuget/nuget-org/trusted-publishing).

## Security boundary

The workflow separates package production from publication:

1. `version-and-build` validates the immutable release tag with read-only repository access.
2. `pack` builds, optionally obfuscates and signs, verifies, and uploads the packages without OIDC permission. The immutable package artifact is retained for 30 days so publication can be retried without rebuilding it.
3. `publish-nuget` downloads the exact immutable artifact ID emitted by `pack`. It has no checkout, build, signing, obfuscation, or package-mutation step. This is the only job with `id-token: write`.
4. `github-release` receives only `contents: write` and attaches the same immutable artifact to the existing release.

`NuGet/login` exchanges the publish job's OIDC identity for a temporary API key. Publishing fails if the policy does not match, the artifact is missing or its digest is invalid, or NuGet does not issue the temporary key.

## Release verification

Syntax and permission-boundary checks run before merge. The OIDC exchange itself can only be proven by a real release event or the manual fallback for an existing tag. For the next deliberate release, verify:

1. `pack` completes with no `id-token: write` permission.
2. `publish-nuget` downloads the artifact ID reported by `pack`.
3. `NuGet/login` issues a temporary key without a repository NuGet secret.
4. Every expected package is published to nuget.org.
5. `github-release` attaches the same package set to the existing GitHub release.

If the workflow filename changes, update the NuGet policy before the renamed workflow is used for a release.
