# Production runbook

This runbook describes the supported deployment: Vercel (frontend), Render Docker web service, Render PostgreSQL, and a private S3 compatible bucket for prediction images and model artifacts. `render.yaml` defines the backend service and database. Railway can also use the Dockerfile and `railway.json` with the same environment variables.

## Release gates

1. Inventory and curate raw data as described in `docs/staging-runbook.md`. Resolve conflicting labels, remove redundant exact copies and create a `path,group_id` CSV for the curated view. Each group ID must identify a physical banknote or capture session. Run `python training/split_grouped.py --raw dataset/curated_raw --groups results/curated-groups.csv --output dataset/processed_v2 --attested-physical-groups`, then audit the new output. The existing image-level split does not establish independent evaluation.
2. Run `python scripts/verify_release.py --require-artifact` and `python training/evaluate_release.py`. The evaluator writes real test accuracy, macro F1, per-class metrics and a confusion matrix to `results/release-evaluation.json`. Do not use generated charts or the historical hard-coded accuracy as release evidence.
3. Set `dataset_version` and `test_metrics` in `models/model-manifest.json` from the verified audit and evaluation. Preserve the exact class order used by training. Each model change needs a new version and SHA-256 digest. Set `release_status` to `approved` only after all gates pass; production startup rejects candidate manifests.
4. Run `python -m training.benchmark_inference path/to/test-image.jpg --runs 100` on hardware comparable to production. The result records warm p50/p95/p99 inference latency. After staging deploy, run `python scripts/benchmark_api.py path/to/test-image.jpg --url https://<staging-domain> --runs 100 --concurrency 4 --output results/staging-api-benchmark.json` to measure upload, storage and database time as well. The HTTP benchmark creates prediction records and images; use a staging environment and purge those records after review.
5. Review the proposed thresholds for per-class recall, macro F1, low-confidence rate, and p95 processing latency before promoting a model. Keep the test split fixed across candidate comparisons.

The current manifest has `test_metrics: null`, `dataset_version: legacy-unversioned`, and `release_status: candidate`. This means the historical model is a candidate, not a measured production release. The existing split contains 182 exact duplicate groups across splits.

An isolated staging service may use `APP_ENV=staging` with this candidate model for integration tests. Staging still requires PostgreSQL, private S3 storage, strong credentials and exact CORS origins. Do not expose candidate accuracy as production quality.

### Candidate to approved workflow

Install Python 3.11 and `requirements-training.txt`. After a group CSV exists, run the following with paths and quality thresholds chosen for the use case:

`python scripts/run_candidate_pipeline.py --dataset-root dataset/processed_v2 --version 2.0.0 --benchmark-image path/to/test-image.jpg` automates audit, training, evaluation, warm benchmark and run recording. The detailed commands below can also be run individually for review and debugging.

```bash
python training/split_grouped.py --raw dataset/curated_raw --groups results/curated-groups.csv --output dataset/processed_v2 --attested-physical-groups
python training/audit_dataset.py --root dataset/processed_v2 --output results/dataset-audit-v2.json
DATA_DIR=dataset/processed_v2 python training/train_efficientnetb0.py
python scripts/create_candidate.py --model models/efficientnetb0_best.keras --audit results/dataset-audit-v2.json --train-dir dataset/processed_v2/train --version 2.0.0
python training/evaluate_release.py --test-dir dataset/processed_v2/test --audit results/dataset-audit-v2.json
python -m training.benchmark_inference path/to/test-image.jpg --runs 100 --output results/benchmark.json
python scripts/promote_release.py --audit results/dataset-audit-v2.json --evaluation results/release-evaluation.json --benchmark results/benchmark.json --min-accuracy 0.95 --min-macro-f1 0.93 --min-class-recall 0.85 --max-ece 0.10 --min-accepted-accuracy 0.97 --max-p95-ms 250
python scripts/verify_release.py --require-artifact
python scripts/record_run.py --audit results/dataset-audit-v2.json --evaluation results/release-evaluation.json --benchmark results/benchmark.json --training-history results/efficientnetb0_training_history.csv --status approved
```

The numeric thresholds above are examples for the command; agree on acceptance criteria before training. The model is not approved until every gate passes. Training history is written from actual epochs to `results/efficientnetb0_training_history.csv`.

## Provisioning and deploy

1. Create a private S3 compatible bucket. Set `MODEL_S3_KEY` to an immutable key such as `models/2.0.0/efficientnetb0_best.keras`; run `python -m scripts.publish_artifact --model models/efficientnetb0_best.keras` after approval. The script checks status and SHA-256 before uploading. Do not expose the bucket publicly.
2. Create a Render Blueprint from `render.yaml`. Set `CORS_ORIGINS` to the exact Vercel production domain; set a unique `ADMIN_USERNAME` and a strong `ADMIN_PASSWORD`. Set `S3_BUCKET`, `S3_REGION`, `S3_ENDPOINT_URL` (if needed), `S3_ACCESS_KEY_ID`, `S3_SECRET_ACCESS_KEY`, and `MODEL_S3_KEY`. `SECRET_KEY` is generated by Render. The database URL is provided by Render PostgreSQL.
3. Render builds the Dockerfile, runs `alembic upgrade head` before deployment, downloads the pinned model at startup, verifies SHA-256 and output dimensions, and checks `/ready` before traffic is routed to the new instance.
4. Deploy the frontend to Vercel. Set `VITE_API_URL=https://<backend-domain>` for Production and Preview as appropriate, then rebuild. The value is public in the browser bundle, so it must never contain credentials.
5. Smoke test: `/ready`, `/login`, a real `/predict`, `/history`, `/stats`, image viewing via a short-lived signed URL, review submission, and the dashboard. Run the API benchmark above at the expected concurrency and inspect error count, throughput, p95 and p99. Check logs and p95 timings after traffic begins.

For local development, run `python -m alembic upgrade head` before `uvicorn api.main:app --reload`. The old SQLite file can be adopted by the initial migration. Back it up first. `APP_ENV=production` rejects missing production settings.

## Operations

- Back up PostgreSQL and enable recovery on the chosen plan. Record the model key and manifest with every deploy. Roll back application code and the manifest together; use a compatible forward database migration or restore a database backup when necessary.
- Images are private S3 objects; history responses contain a signed URL valid for five minutes. Run `python -m scripts.purge_history --days 90` on a scheduled job if the retention policy is 90 days. Adjust that period to the approved policy.
- Dashboard latency comes from the latest 1,000 predictions of the current model version. Reviewed accuracy comes only from records manually labeled by staff and may be biased; the independent test set remains the release quality measure.
- Monitor startup failures, database connectivity, storage failures, HTTP error rate, inference p95/p99, total processing p95, low-confidence proportion and review outcomes. Investigate drift before retraining. Keep each new dataset snapshot, model artifact, manifest and evaluation report.
- `suspicious` means low confidence or a background prediction, not counterfeit detection. The current 25-class model has no verified counterfeit class.

## Known work before public launch

- The legacy dataset uses an image-level split and has inconsistent `USA_2 Dollar` spelling. Regenerate a clean grouped split and rerun evaluation before making accuracy claims.
- A production rate limiter or gateway policy should protect the public upload endpoint. Review the signed URL behavior and bucket CORS on the selected S3 provider.
- The SQLite file and 30 uploaded images have been removed from Git tracking in this change; local copies remain. Existing Git history may still contain them, so review repository visibility and history handling before publication.
- Run records are currently local files. After private artifact storage is provisioned, automate archiving them, schedule drift/error checks and rehearse a cloud rollback. CI verifies release-gate logic but cannot run a real model evaluation until a verified dataset and model artifact are available to the runner.
