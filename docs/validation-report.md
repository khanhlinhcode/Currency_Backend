# Candidate model validation (2026-11-07)

Model artifact: `models/efficientnetb0_best.keras` (SHA-256 `688efeec9b612d6777f450e63c75803f6705b2cb9eddf5518465b288f696f45e`), EfficientNetB0 with 25 output classes. The API's original 26-label mapping was replaced by the training class order in `models/model-manifest.json`.

## Data audit

The existing dataset has 7,065 train, 1,504 validation and 1,537 test images. `training/audit_dataset.py` found 182 groups of byte-identical images crossing splits, including 104 affected test files, and two groups with conflicting labels (`INR_20` versus `INR_200`, and `INR_20` versus `INR_2000`). The existing split has no physical banknote or capture-session IDs. The dataset audit fails by design.

A fresh inventory of `dataset/raw` found 10,106 images, 367 exact-duplicate groups (461 extra copies), and the same two conflicting-label groups. `training/curate_raw.py` built an isolated `dataset/curated_raw` copy with 9,643 unique images, removed 459 redundant copies, and quarantined the four conflict images for label review. No label was guessed. `results/group-review.csv` retains every source image and `results/curated-groups.csv` has empty `group_id` cells that must be filled from physical note or capture-session evidence. This snapshot is not yet release data.

## Diagnostic model metrics

`training/evaluate_release.py --audit results/dataset-audit.json` produced:

| Measure | Existing test split | After excluding 104 exact duplicate test files |
|---|---:|---:|
| Images | 1,537 | 1,433 |
| Accuracy | 96.2915% | 96.0223% |
| Macro F1 | 95.9396% | 95.5717% |

For the existing test split, ten-bin expected calibration error is 0.022638. At confidence threshold 0.6, coverage is 95.7059%, accuracy among accepted predictions is 97.8246%, and 32 accepted predictions are wrong. These figures are **diagnostic only**; exact-duplicate exclusion does not remove all near-duplicates or multiple photos of the same physical note. No production accuracy is certified.

## Speed checks

On the local Apple silicon development machine, the warm single-image benchmark (30 runs, five warmups) measured mean 52.53 ms, p50 45.49 ms, p95 93.47 ms and p99 101.83 ms for model inference. A cold HTTP request in the SQLite smoke test took 1.322 s for inference and 2.160 s total processing. These are not production latency claims; repeat the HTTP load test on the chosen Render instance with PostgreSQL and S3.

The new `scripts/benchmark_api.py` was smoke tested against the local PostgreSQL API with five measured requests, one warmup and concurrency two: five successes, no errors, 10.282 requests/second, mean 159.75 ms and p95 220.56 ms. This tiny sample verifies the tool and end-to-end path only; measure at least 100 requests on staging before setting a service latency target.

A fuller **local staging simulation** used PostgreSQL and a mocked S3 HTTP service. The candidate artifact was uploaded to an immutable staging key, fetched by the API at startup and checksum-verified. Guest prediction returned 401. With three distinct JPEG images, ten warmups, 100 measured requests and concurrency four, the final run had 100/100 successes, 13.299 requests/s, client p95 490.65 ms and p99 715.86 ms. Server processing p95 was 480.9 ms, queue wait p95 273.7 ms and actual model execution p95 126.3 ms. All provisional checks in `docs/staging-targets.json` passed. The benchmark records were tagged, excluded from operational dashboard/drift statistics, then purged from staging storage and database. These figures were measured on the local Apple silicon machine with a mock S3 service; they do **not** predict Render performance.

The same three images were also measured in separate 100-request runs at other concurrency levels:

| Concurrent requests | Successes | Throughput | Client p95 | Queue p95 | Inference p95 |
|---:|---:|---:|---:|---:|---:|
| 1 | 100/100 | 14.354 req/s | 91.77 ms | 0 ms | 57.9 ms |
| 4 | 100/100 | 13.299 req/s | 490.65 ms | 273.7 ms | 126.3 ms |
| 8 | 100/100 | 18.357 req/s | 548.29 ms | 431.4 ms | 70.1 ms |

These are independent short local runs, so throughput is not monotonic; repeat with longer representative traffic on the actual staging plan. JSON reports are at `results/staging-local-c1.json`, `results/staging-local-benchmark.json` and `results/staging-local-c8.json` (ignored by Git).

## Integration checks

- Backend API smoke passed: readiness, admin login, prediction, dashboard statistics, review and deletion against both fresh SQLite and PostgreSQL databases.
- Alembic upgrade passed on a copy of the existing SQLite database and on fresh SQLite and PostgreSQL databases.
- Private S3 storage round trip and signed URL passed with a mocked S3 service.
- Frontend ESLint and Vite production build passed. Route-level loading reduced the shared initial JavaScript chunk from approximately 1,250 KB to 281 KB before gzip; heavy dashboard and history libraries load on demand.
- Linux AMD64 Docker image build completed successfully.
- Production container startup correctly refused the candidate model despite otherwise valid deployment settings.
- Render accepted the Blueprint syntax/fields but account validation returned `need_payment_info` for paid web and database services; cloud staging has not been created.

The release manifest remains `candidate`. Promotion and production startup reject it until a grouped, audited dataset and new evaluation pass the release gates.

The candidate pipeline was executed against the old split as a gate check. It stopped at the dataset audit (182 cross-split duplicate groups) before starting training. No new model was trained from the unverified split.

## Free-tier staging preparation (2026-11-08)

The pinned ONNX candidate is 17.46 MB, SHA-256 `737a0fcf1de09f8acf2f226e9ca54afc4a970f0738ae17dc135d146efda7487c`. Its 25-class output and nearest-neighbor input resize match the Keras source. A same-input comparison across all 1,537 historical test images found 100% top-1 agreement and a maximum absolute probability difference of `3.64e-6`. The ONNX-only evaluation measured 96.2915% accuracy, 95.9396% macro F1, ECE 0.022638, 95.7059% accepted coverage and 97.8246% accuracy among accepted predictions at confidence 0.6. Its Apple silicon inference p50/p95/p99 was 29.17/51.25/97.43 ms. These remain **diagnostic** because the repeated audit still found 182 exact cross-split duplicate groups, two conflicting-label groups and no verified physical-note grouping. The release gate still rejects promotion.

For the Render Free resource profile, a Linux AMD64 container was run locally with a 0.1 CPU quota and 512 MB RAM. It reached `/ready` and used 184.4 MiB at idle. Three distinct JPEGs, ten warmups and 100 measured HTTP predictions were used for each concurrency profile:

| Concurrent requests | Successes | Throughput | Client p95 | Client p99 | Queue p95 | Inference p95 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 100/100 | 0.295 req/s | 4.70 s | 5.60 s | 0.0001 s | 3.10 s |
| 4 | 100/100 | 0.237 req/s | 22.00 s | 24.20 s | 12.00 s | 6.19 s |
| 8 | 100/100 | 0.226 req/s | 42.79 s | 46.69 s | 30.80 s | 6.20 s |

The container was not OOM-killed; a post-run memory snapshot showed 484.5 MiB of 512 MiB in use. The measurements ran through AMD64 emulation on Apple silicon with local SQLite and disk storage, so they are resource-constrained diagnostics, **not** cloud latency evidence. They show severe queuing at concurrent load and little memory headroom. Actual Render measurements with Supabase PostgreSQL and S3 are required before setting a service target.

The isolated Supabase project `currency-ai-staging` and private bucket `currency-ai-staging-private` were created in Singapore. The checksum-pinned ONNX file was uploaded under `staging/models/1.1.0/efficientnetb0_candidate.onnx`. This is staging infrastructure, not an approved production release.

The backend commit `5b9a8e0b` and frontend commit `f48e32ca` passed their GitHub Actions CI workflows. The Render Free static dashboard was deployed at `https://currency-ai-free-dashboard.onrender.com`; both `/` and `/history` returned HTTP 200 with the SPA rewrite. Cloud API and latency findings from the subsequent deployment are recorded below.

Tracked reports: [dataset audit](evidence/2026-11-08-dataset-audit-summary.json), [ONNX evaluation](evidence/2026-11-08-onnx-evaluation.json), [same-input parity](evidence/2026-11-08-onnx-parity.json), [warm ONNX benchmark](evidence/2026-11-08-onnx-warm-benchmark.json), [diagnostic run record](evidence/2026-11-08-diagnostic-run.json), local Free-profile HTTP results at [concurrency 1](evidence/2026-11-08-free-local-c1.json), [4](evidence/2026-11-08-free-local-c4.json) and [8](evidence/2026-11-08-free-local-c8.json).

## Cloud staging results (2026-11-08)

The Render Free backend is live at `https://currency-ai-free-staging.onrender.com` and the static dashboard at `https://currency-ai-free-dashboard.onrender.com`. PostgreSQL uses the isolated Supabase project through its IPv4 session pooler; the model and uploaded images are in the private Storage bucket. The ONNX artifact was downloaded from Storage and its SHA-256 matched the pinned manifest. The backend is using Python 3.11, 0.1 CPU and 512 MiB in Singapore. The candidate remains blocked for production.

The cloud smoke check passed readiness, admin login, model metadata, operational statistics, archived MLOps runs, analyst user visibility, an authenticated prediction, private image retrieval through a five-minute signed API link, history, human review and deletion. During the review check, reviewed accuracy became 100% for that one labelled record, then returned to no reviewed records after cleanup; this is an operational workflow check, not a model quality estimate. A valid unauthenticated multipart prediction returned HTTP 401. Browser checks confirmed login, dashboard, model/run metadata and user administration; the deployed login no longer shows demo credentials. A Supabase S3 presigned GET returned HTTP 403, so private history images are served by the signed API route instead.

Three distinct JPEGs (56–596 KB), ten warmups and 100 measured HTTP requests per profile were sent from the local macOS client to the live Render/Supabase staging stack. Benchmark requests were tagged, excluded from operational statistics, and all 330 benchmark records and images were removed afterward.

| Concurrent requests | Successes | Throughput | Client p95 | Client p99 | Queue p95 | Inference p95 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 100/100 | 1.342 req/s | 1.05 s | 1.71 s | 0 ms | 181 ms |
| 4 | 100/100 | 2.526 req/s | 2.11 s | 4.38 s | 289 ms | 410 ms |
| 8 | 100/100 | 2.566 req/s | 5.18 s | 5.87 s | 1,207 ms | 598 ms |

These are warm, short runs on three images, not an SLA or a representative production workload. The local AMD64-emulated Free-profile figures above were much slower than the native Render deployment and must not be used as cloud estimates. An idle cold-start measurement and longer representative traffic remain necessary before promising availability or latency.

The `20261108T063700Z-onnx-diagnostic` training/evaluation record was archived immutably under `staging/runs/` and appears in the admin MLOps dashboard. The daily GitHub Actions staging monitor was configured with encrypted credentials and passed its manually triggered run. After benchmark cleanup, the monitor reported zero operational predictions, one archived run and all probes healthy. Backend CI at `2aeb27bf` and frontend CI at `3f8d1a90` passed.

Cloud evidence: [deployment snapshot](evidence/2026-11-08-deployment-snapshot.json), [benchmark context](evidence/2026-11-08-cloud-benchmark-context.json), [concurrency 1](evidence/2026-11-08-free-cloud-c1.json), [4](evidence/2026-11-08-free-cloud-c4.json), [8](evidence/2026-11-08-free-cloud-c8.json), [smoke checks](evidence/2026-11-08-cloud-smoke.json), [monitor snapshot](evidence/2026-11-08-cloud-monitor.json).

## MLOps release control staging (2026-11-10)

Backend commit `9c914987` and frontend commit `ab235fb1` passed CI and reached Render `live`; later backend commit `5a5b1dbe` also passed CI and reached `live`. The admin release/connected-device page was inspected in a real WebKit browser at `/mlops/releases`; it displayed an empty staging channel, disabled activation/rollback buttons and the legacy candidate warning. Monitoring displayed the device fleet and drift fields as `Chưa đủ dữ liệu`. The separate [cloud smoke report](evidence/2026-11-10-release-cloud-smoke.json) passed 20/20 checks against the earlier release-control build, including admin/analyst authorization, legacy release rejection, device token and revocation, prediction, private image, history cleanup and archived benchmark visibility.

The same three images (56–596 KB) were measured again on Render Free with ten warmups per profile. Tagged traffic was removed after immutable reports were archived in private Storage and PostgreSQL.

| Concurrent requests | Measured successes | Throughput | Client p50 | Client p95 | Client p99 |
|---:|---:|---:|---:|---:|---:|
| 1 | 100/100 | 1.351 req/s | 786 ms | 1.005 s | 1.222 s |
| 4 | 100/100 | 2.535 req/s | 1.572 s | 2.090 s | 2.266 s |
| 8 | 100/100 | 2.344 req/s | 3.070 s | 5.216 s | 7.496 s |
| 4 sustained | 300/300 | 2.334 req/s | 1.652 s | 2.473 s | 3.595 s |

Render's one-minute samples peaked at 0.136 of 0.15 CPU cores and 322 of 537 MB RAM in the benchmark window. Raw reports: [c1](evidence/20261110-free-cloud-c1.json), [c4](evidence/20261110-free-cloud-c4.json), [c8](evidence/20261110-free-cloud-c8.json), [sustained](evidence/20261110-free-cloud-sustained-c4.json), [resource metrics](evidence/20261110-render-resource-metrics.json). These are candidate 1.1.0 service measurements, not proof of model correctness. The data release gate is still blocked and no production service was started.

Render logged a Free-instance shutdown at 12:14:15 UTC. One client request to `/ready` after the shutdown took **52.688 s**; the next request took **0.262 s**, both HTTP 200. [Cold-start evidence](evidence/2026-11-10-free-cold-start.json). This single observation is separate from warm prediction benchmarks and does not establish an SLA.

The upgraded backup script was exercised locally with PostgreSQL 16 and moto S3. One nonempty database row and two private objects were encrypted, backed up, authenticated, restored to an isolated database/bucket and verified byte-for-byte in 0.23 seconds for the restore phase. [Full local restore evidence](evidence/2026-11-10-local-full-restore-drill.json). No Supabase backup or restore was performed.

On 2026-11-11, backend commit `475edd96` passed CI and reached Render `live`. A read-only [post-deploy smoke report](evidence/2026-11-11-final-staging-smoke.json) passed 9/9 checks: readiness, admin and analyst login, release-channel access control, run registry, benchmarks and monitoring. The release channel had no active approved run and monitoring returned `insufficient_data`. The first request after an overnight idle took 52.634 s, consistent with a Free-instance cold start; this is an additional single observation, not a latency distribution.

A separate [manual staging monitor run](evidence/2026-11-11-manual-staging-monitor.json) passed seven probe/storage checks and saved a timestamped database snapshot. It found only two operational predictions, zero reviewed labels, no verified drift baseline and an `insufficient_data` model-quality status. Its `healthy: true` field means the probes ran and stored successfully; it does **not** mean model accuracy or drift is healthy.

Backend commit `062af5c4` passed CI and reached Render `live` with signed device manifests. The [signed cloud probe](evidence/2026-11-11-signed-device-cloud-probe.json) verified an Ed25519 `no_release` response, detected a changed generation, and confirmed the temporary device token was rejected after revocation. This checks the control plane only: the release channel still has no approved model, and no physical device downloaded an artifact.

On 2026-11-11 the local `.venv` was repaired to use an available Python 3.11.16 interpreter and the pinned API/CI dependencies were installed. All 26 local unit tests and `pip check` passed; TensorFlow 2.16.2 imported in 102.36 seconds on the development Mac. A fresh audit of the **legacy** split found 10,106 images, 182 exact cross-split duplicate groups and two conflicting-label groups. The human review queue still has 0/10,106 complete rows and four conflict rows; the release gate stopped as designed. The loopback review UI returned HTTP 200 with matching counts. [Local recheck evidence](evidence/2026-11-11-local-data-recheck.json). These tests verify tooling and detect legacy leakage; they do not certify model quality.

A metadata-presence scan found EXIF on 5,638/10,106 files, but no `DateTimeOriginal` values. Only 2,188 INR images had a generic EXIF timestamp; no VN image had EXIF. These fields may help a human find source records, but do not prove that two files show the same physical banknote or capture session. If capture logs or physical-note IDs do not exist, those groups require a newly collected, traceable dataset rather than inferred IDs.
