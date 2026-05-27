// Reference implementation — Supabase Edge Function (Deno) response signing.
//
// Drop this into your existing `validate-license` Edge Function. It signs the
// validation response over the client's per-request nonce so the .NET client
// (LicenseResponseVerifier) can reject spoofed/MITM'd servers when a customer
// sets RequireSignedResponse = true. See
// docs/enterprise/license-server-response-signing.md for the full contract.
//
// SECRET: store the RSA-2048 PKCS#8 PEM private key as the Edge Function secret
// RESPONSE_SIGNING_PRIVATE_KEY_PEM (its PUBLIC half is embedded in the client as
// LicenseResponseVerifier.ResponseSigningPublicKey). Never log or return it.

const PEM_HEADER = "-----BEGIN PRIVATE KEY-----";
const PEM_FOOTER = "-----END PRIVATE KEY-----";

/** Imports the PKCS#8 PEM private key for RSASSA-PKCS1-v1_5 / SHA-256 signing. */
async function importPrivateKey(pem: string): Promise<CryptoKey> {
  const b64 = pem
    .replace(PEM_HEADER, "")
    .replace(PEM_FOOTER, "")
    .replace(/\s+/g, "");
  const der = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
  return await crypto.subtle.importKey(
    "pkcs8",
    der,
    { name: "RSASSA-PKCS1-v1_5", hash: "SHA-256" },
    false,
    ["sign"],
  );
}

/**
 * Canonical string — MUST match LicenseResponseVerifier.BuildCanonical
 * byte-for-byte: nonce\nstatus\ntier\nexpires\ncaps  where caps are ordinal-
 * sorted and comma-joined, and absent fields are the empty string.
 */
function buildCanonical(
  nonce: string,
  status: string,
  tier: string | null,
  expiresAt: string | null,
  capabilities: string[],
): string {
  const caps = [...capabilities].filter((c) => c).sort(); // JS default sort is by UTF-16 code unit (ordinal for ASCII capability names)
  return [nonce, status, tier ?? "", expiresAt ?? "", caps.join(",")].join("\n");
}

/** Returns the base64 RSA-2048 PKCS#1 SHA-256 signature over the canonical form. */
export async function signValidationResponse(
  nonce: string,
  status: string,
  tier: string | null,
  expiresAt: string | null,
  capabilities: string[],
): Promise<string> {
  const pem = Deno.env.get("RESPONSE_SIGNING_PRIVATE_KEY_PEM");
  if (!pem) throw new Error("RESPONSE_SIGNING_PRIVATE_KEY_PEM secret is not set.");
  const key = await importPrivateKey(pem);
  const canonical = new TextEncoder().encode(
    buildCanonical(nonce, status, tier, expiresAt, capabilities),
  );
  const sig = await crypto.subtle.sign("RSASSA-PKCS1-v1_5", key, canonical);
  return btoa(String.fromCharCode(...new Uint8Array(sig)));
}

// ── Integration sketch (inside your existing handler) ───────────────────────
//
//   const { license_key, package: pkg, nonce } = await req.json();
//   const result = await validateLicense(license_key, pkg);  // your existing logic
//
//   // NEW: sign the response over the client's nonce when one was supplied.
//   let signature: string | undefined;
//   if (nonce) {
//     signature = await signValidationResponse(
//       nonce, result.status, result.tier ?? null,
//       result.expires_at ?? null, result.capabilities ?? [],
//     );
//   }
//
//   return new Response(JSON.stringify({ ...result, signature }), {
//     headers: { "content-type": "application/json" },
//   });
//
// NOTE on capability ordering: the client sorts capabilities with .NET's
// StringComparer.Ordinal before verifying. JS Array.sort() on ASCII capability
// strings (e.g. "tensors:load", "tensors:save") yields the same order. If you
// ever introduce non-ASCII capability names, switch to an explicit ordinal
// (code-unit) comparator on BOTH sides.
