# Staging release and measurement

For the current Render Free deployment and dataset v2 quality policy, use [free staging](free-staging-runbook.md) and [MLOps v2](mlops-v2-runbook.md). This document retains the earlier paid-plan benchmark targets for comparison.

Staging runs the candidate model with an isolated PostgreSQL database and a separate private S3 bucket. The backend requires an active login for predictions when `APP_ENV=staging`. The candidate's accuracy is not a production claim.

## 1. Prepare and review data

```bash
python training/inventory_raw.py
python training/curate_raw.py
```

The current inventory contains 10,106 raw images. The curated view contains 9,643 unique images; four conflicting-label images are quarantined. Inspect `results/raw-inventory.json` and fill `reviewed_label` for every conflicting row in `results/group-review.csv`. Run `curate_raw.py` again with a new `--output` and `--group-template` path to include reviewed images. Never guess a label from its directory name when an identical image appears under another class.

Fill `group_id` in the resulting group CSV using the physical banknote or capture session. Exact content hashes in the review CSV are suggestions only. When all groups have been reviewed:

```bash
python training/split_grouped.py --raw dataset/curated_raw --groups results/curated-groups.csv --output dataset/processed_v2 --attested-physical-groups
python training/audit_dataset.py --root dataset/processed_v2 --output results/dataset-audit-v2.json
```

The split script requires a group ID for every curated image. The audit checks that every output file is covered by the split manifest, each group stays in one split, class sets match, and no identical images cross splits. This cannot independently prove a human-provided group ID identifies the correct physical note.

## 2. Record and evaluate model runs

After a grouped split passes the audit, use the isolated candidate pipeline. It stops immediately if the audit fails; it trains using train/validation only, then evaluates the held-out test set, benchmarks inference and records all inputs and outputs under a unique run ID:

```bash
python scripts/run_candidate_pipeline.py --dataset-root dataset/processed_v2 --version 2.0.0 --benchmark-image path/to/representative.jpg --gates-file results/quality-gates-v2.json --training-config results/training-config-v2.json
```

For a manually executed candidate workflow, record each run after generating its audit, evaluation and benchmark reports:

```bash
python scripts/record_run.py --audit results/dataset-audit-v2.json --evaluation results/release-evaluation.json --benchmark results/benchmark.json --training-history results/efficientnetb0_training_history.csv --status candidate
```

Each `results/runs/<run-id>/` folder contains the input reports, their SHA-256 hashes, model and dataset versions, Git commit and environment. These folders are local and ignored by Git; archive them with the model artifact in private storage before treating them as durable records. A `diagnostic` run may use the historical unverified split; candidate and approved runs require a clean, verified audit. The pipeline was tested against the old dataset and correctly stopped at the audit before training.

## 3. Provision isolated staging

1. Create a **separate private staging S3 bucket**. Set `MODEL_S3_KEY=staging/models/<version>/<artifact>` and the bucket credentials in your local secret environment. Publish the checksum-verified candidate with `python -m scripts.publish_artifact --target staging`.
2. Connect the backend repository to Render and create a Blueprint with path `render.staging.yaml`. It creates `currency-ai-staging` and `currency-ai-staging-db` in Singapore. Enter the prompted `sync: false` values: exact frontend origin, admin credentials, staging bucket and S3 endpoint/region. Do not copy production credentials or database data into staging.
3. Deploy a frontend preview with `VITE_API_URL=https://<staging-backend>` and `VITE_APP_ENV=staging`. Login before prediction.
4. Smoke test `/ready`, `/login`, `/predict`, `/history`, `/stats`, review, image signed URL and dashboard. A guest `/predict` must return 401.

Both Render services use paid plans because the TensorFlow container needs 2 GB RAM and the migration uses a pre-deploy command. Render Blueprint validation currently returns `need_payment_info` for this account; billing setup and a connected Git repository are required before cloud provisioning. See the [Render Blueprint specification](https://render.com/docs/blueprint-spec) and [Blueprint setup guide](https://render.com/docs/infrastructure-as-code).

## 4. Benchmark against staging

Choose at least three different JPEG images from different classes and sizes. Get an analyst token from `/login` and pass it through a private environment variable or secure command invocation. The benchmark creates prediction records and images in staging.

```bash
python scripts/benchmark_api.py image-a.jpg image-b.jpg image-c.jpg --url https://<staging-backend> --token-env STAGING_TOKEN --runs 100 --warmups 10 --concurrency 4 --output results/staging-api-benchmark.json
python scripts/check_staging_benchmark.py --report results/staging-api-benchmark.json
```

`docs/staging-targets.json` holds **provisional** gates for the specified 1 CPU / 2 GB service: at least 100 requests, three unique images, concurrency four, at least 99% success, client p95 ≤ 2,000 ms, client p99 ≤ 4,000 ms, server processing p95 ≤ 1,500 ms, model queue wait p95 ≤ 1,000 ms, model execution p95 ≤ 750 ms and throughput ≥ 2 requests/s. These are engineering starting targets, not measured service commitments. Run extra profiles at concurrency one and eight, plus cold start after a deploy; retain every JSON report and hardware/plan details. Investigate error count and latency before changing targets. The benchmark marks its records with `benchmark_run_id`, and the operational dashboard and drift report exclude them. After saving the report, run `APP_ENV=staging python -m scripts.purge_benchmark --run-id <benchmark_run_id> --execute` with the staging database and S3 environment to remove its records and images.

## 5. Monitor and roll back

Monitor `/ready`, HTTP 5xx rate, memory/CPU, database connectivity, storage errors and API p95/p99 through Render service metrics/logs. `/stats` provides current-model inference and processing latency from the latest 1,000 predictions. Run `scripts/drift_report.py` periodically with a clean audit and evaluation baseline; its low-confidence and class-mix signals require human investigation. Reviewed accuracy is based on selected staff reviews and is not a test-set estimate.

If a deploy degrades service, select the previous successful deploy on the Render **Deploys** page, then confirm `/ready`, model version, prediction and history. Preserve the old immutable model object key. A Render rollback reuses the previous build and deploy configuration, but it does not roll back PostgreSQL data; database migrations must remain forward compatible or be restored from a backup. See [Render rollbacks](https://render.com/docs/rollbacks).
