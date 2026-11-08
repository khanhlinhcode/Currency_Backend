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

The backend commit `5b9a8e0b` and frontend commit `f48e32ca` passed their GitHub Actions CI workflows. The Render Free static dashboard was deployed at `https://currency-ai-free-dashboard.onrender.com`; both `/` and `/history` returned HTTP 200 with the SPA rewrite. The backend Blueprint still requires the database and administrator passwords to be entered by the project owner before deployment, so no cloud API or cloud performance result is claimed yet.

Tracked reports: [dataset audit](evidence/2026-11-08-dataset-audit-summary.json), [ONNX evaluation](evidence/2026-11-08-onnx-evaluation.json), [same-input parity](evidence/2026-11-08-onnx-parity.json), [warm ONNX benchmark](evidence/2026-11-08-onnx-warm-benchmark.json), [diagnostic run record](evidence/2026-11-08-diagnostic-run.json), local Free-profile HTTP results at [concurrency 1](evidence/2026-11-08-free-local-c1.json), [4](evidence/2026-11-08-free-local-c4.json) and [8](evidence/2026-11-08-free-local-c8.json).
