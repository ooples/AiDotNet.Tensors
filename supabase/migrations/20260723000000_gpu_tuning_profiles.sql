-- Phase 2 community GPU-tuning exchange (opt-in).
-- One row = a locally-measured winning kernel config, keyed by the per-MODEL
-- hardware class (not the per-card UUID) so a config tuned on one card reaches
-- every card of that model on the same driver + architecture.
--
-- Trust model: measured_gflops is advisory. Clients treat a downloaded row as a
-- candidate that must win their own on-device sweep before it is used, so a
-- poisoned or hardware-mismatched row cannot degrade a peer — it just loses.
-- Rows are therefore append-only community reports, never authoritative config.

create table if not exists public.gpu_tuning_profiles (
    id               bigint generated always as identity primary key,
    model_key        text        not null,   -- GpuDeviceFingerprint.ModelKey
    vendor           text        not null,
    model            text        not null,
    architecture     text        not null,   -- e.g. sm86
    driver_version   integer     not null,
    category         text        not null,   -- conv2d | sdpa | gemm | ...
    kernel_name      text        not null,   -- shareable family, no per-card suffix
    shape_key        text        not null,   -- ShapeProfile.ToFileStem()
    variant          text        not null,   -- winning variant id (e.g. tile-16)
    parameters       jsonb       not null default '{}'::jsonb,
    measured_gflops  double precision not null,
    client_hash      text,                    -- pseudonymous reporter id (no PII)
    aidotnet_version text,
    recorded_at      timestamptz not null default now(),
    constraint gpu_tuning_profiles_gflops_nonneg check (measured_gflops >= 0)
);

-- The exact lookup clients issue: (model_key, category, kernel_name, shape_key),
-- best-reported first. The trailing measured_gflops desc lets the ORDER BY be
-- index-served so a client can cheaply pull the top-K community candidates.
create index if not exists gpu_tuning_profiles_lookup
    on public.gpu_tuning_profiles (model_key, category, kernel_name, shape_key, measured_gflops desc);

-- Opt-in clients use the anon role. Community reports are append-only: anon may
-- INSERT (publish) and SELECT (download), but never UPDATE or DELETE, so no one
-- can rewrite or purge others' reports. Server-side aggregation/curation, if
-- added later, runs under the service role.
alter table public.gpu_tuning_profiles enable row level security;

drop policy if exists gpu_tuning_profiles_anon_insert on public.gpu_tuning_profiles;
create policy gpu_tuning_profiles_anon_insert
    on public.gpu_tuning_profiles for insert to anon with check (true);

drop policy if exists gpu_tuning_profiles_anon_select on public.gpu_tuning_profiles;
create policy gpu_tuning_profiles_anon_select
    on public.gpu_tuning_profiles for select to anon using (true);
