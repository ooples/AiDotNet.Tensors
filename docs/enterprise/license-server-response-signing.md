# License server: validation-response signing contract

This is the **server-side** half of the anti-spoof / anti-replay hardening. The
client (`LicenseResponseVerifier` + `LicenseValidator`) is already implemented
and will enforce it once (a) the server signs responses per this contract and
(b) the customer sets `AiDotNetTensorsLicenseKey.RequireSignedResponse = true`.
Until both hold, the online path behaves exactly as before (unsigned).

> **Why:** the online response is otherwise unsigned JSON, so an attacker who
> redirects the endpoint (hosts file / proxy / DNS) can return
> `{"status":"active"}` and bypass the online check. Signing the response over a
> client nonce closes that.

## 1. Keypair

Generate an RSA-2048 keypair for response signing. This is a **separate** key
from the entitlement-signing key (compromise isolation).

```bash
openssl genrsa -out response-signing.key 2048
# Export the public key as modulus:exponent (matches RSAParameters import client-side):
openssl rsa -in response-signing.key -noout -modulus            # → Modulus=HEX...
openssl rsa -in response-signing.key -pubout                    # (for reference)
```

Convert modulus + exponent (65537 → `AQAB`) to base64 and embed in the client as
`LicenseResponseVerifier.ResponseSigningPublicKey = "{modulusBase64}:{exponentBase64}"`.
Keep the **private** key in the server's secret store / HSM only.

## 2. Request

The client sends, in the existing JSON POST body, a new field:

```json
{ "license_key": "...", "package": "AiDotNet.Tensors", "nonce": "<base64url, 256-bit, per-request>" }
```

The `nonce` is fresh per request and unpredictable. The server MUST echo it into
the signed payload (below) so a captured response can't be replayed against a
later request.

## 3. Canonical string to sign (MUST match byte-for-byte)

The client reconstructs and verifies exactly this (`LicenseResponseVerifier.BuildCanonical`):

```text
{nonce}\n{status}\n{tier}\n{expires_at}\n{capabilities_csv}
```

- `\n` is U+000A (LF), single byte. Five logical fields, four separators.
- `status`: the same string returned in the body (`active` / `expired` / `revoked` / `seat_limit_reached` / `invalid`).
- `tier`: the tier string, or empty if none.
- `expires_at`: the same string returned in the body's `expires_at`, or empty if none.
- `capabilities_csv`: the capability strings **sorted with ordinal (byte) ordering** and joined by `,` (comma, no spaces). Empty string if none.
- Absent fields are the empty string (not `null`, not the literal `"null"`).

Sign `UTF-8(canonical)` with RSA-2048 **PKCS#1 v1.5**, **SHA-256**.

## 4. Response

Return the existing fields plus a base64 `signature`:

```json
{
  "status": "active",
  "tier": "enterprise",
  "expires_at": "2027-05-22",
  "capabilities": ["tensors:load", "tensors:save"],
  "signature": "<base64 RSA-2048 PKCS#1 SHA-256 over the canonical string>"
}
```

The `capabilities` array order in the JSON body does not matter (the client
sorts before verifying), but the **signed** canonical uses the ordinal-sorted
order — make sure the server sorts identically before signing.

## 5. Client behaviour (already implemented — for reference)

- `RequireSignedResponse == false` (default): signature ignored; unchanged.
- `RequireSignedResponse == true`:
  - missing/invalid `signature`, wrong key, or nonce/field mismatch → result
    downgraded to `Invalid` ("failed signature verification … possible spoofed
    server"), regardless of the body's `status`.
  - valid signature → the body's status/tier/caps are trusted.

## 6. Reference signing snippet (server pseudo-code)

```python
caps_csv = ",".join(sorted(capabilities))           # ordinal sort
canonical = f"{nonce}\n{status}\n{tier or ''}\n{expires_at or ''}\n{caps_csv}"
signature = rsa_pkcs1_sha256_sign(private_key, canonical.encode("utf-8"))
response["signature"] = base64.b64encode(signature).decode()
```

## Honest scope

This hardens the **online/standard** tier against server-spoof trial-bypass. It
is bypassable by patching the client (managed code). The **high-assurance**
trust anchor is the fully-offline `SignedEntitlement` (no server) — see
`licensing-security-model.md`.
