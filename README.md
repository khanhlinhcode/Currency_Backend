# Currency AI backend

FastAPI service for recognizing the denomination of Vietnamese, US and Indian banknotes. The current EfficientNetB0 model has 25 classes. Low confidence means **review required**; this system has not been validated to detect counterfeit notes.

## Current release status

The checked-in [model manifest](models/model-manifest.json) is `candidate`. A raw-data audit found exact duplicates crossing the old train/validation/test split and two conflicting-label groups. The historical accuracy is diagnostic only. `APP_ENV=production` refuses this candidate until a verified grouped dataset, fresh evaluation and release gates pass.

## Local development

Use Python 3.11 and a local copy of the model artifact whose SHA-256 matches the manifest:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements-api.txt
python -m alembic upgrade head
uvicorn api.main:app --reload
```

The development defaults use SQLite and local image files. Copy `.env.example` into a private environment file or export the variables before starting. API documentation is at `http://127.0.0.1:8000/docs`. The frontend is in the sibling `currency-frontend` repository.

## Data and model lifecycle

1. Run `python training/inventory_raw.py` to list raw images, exact duplicates and conflicting labels.
2. Run `python training/curate_raw.py` to build a deduplicated local view. Review the quarantined images and fill physical-banknote or capture-session `group_id` values in the generated CSV.
3. Split with `training/split_grouped.py --attested-physical-groups`, then run `training/audit_dataset.py`. The audit must pass before training a release candidate.
4. Train, evaluate, benchmark and record the run. `scripts/promote_release.py` checks quality, calibration, group integrity and inference latency before approving a manifest.
5. Deploy the approved artifact and manifest together. Track dashboard latency and reviewed outcomes, then investigate signals from `scripts/drift_report.py`.

Exact commands and acceptance criteria are in the [staging runbook](docs/staging-runbook.md) and [production runbook](docs/production-runbook.md). The [validation report](docs/validation-report.md) contains measured results and known limits.

The [free staging runbook](docs/free-staging-runbook.md) uses a checksum-pinned ONNX conversion on Render Free with isolated Supabase PostgreSQL and private object storage. The live [staging dashboard](https://currency-ai-free-dashboard.onrender.com) calls the [staging API](https://currency-ai-free-staging.onrender.com/ready). `render.free.yaml` describes the native Python service; `Dockerfile.free` supports local resource testing. This staging deployment does not relax the model release gate. `scripts/archive_run.py` saves immutable audit/evaluation/benchmark run folders, `/mlops/runs` exposes their summaries to admins, and the scheduled staging monitor retains JSON snapshots in CI.

The [MLOps v2 runbook](docs/mlops-v2-runbook.md) covers human dataset review, pre-registered quality gates, PostgreSQL run metadata, private artifact reports, dashboard monitoring and encrypted backup/restore drills. Its release path stays blocked until physical group IDs and conflicting labels are reviewed.

For nontechnical human reviewers, [open the local review interface](docs/data-review-ui.md) with `scripts/start-data-review.command`. It never guesses `group_id`; the one-click finalizer produces a new audited dataset version only after every review is complete.

If a trusted capture ledger or source manifest exists, [import its attested `group_id` mappings in batches](docs/batch-data-review.md). The importer checks image paths and hashes, writes an atomic local CSV with a backup, and leaves images without provenance pending.

## Deploy and verification

`render.staging.yaml` defines an isolated staging backend and PostgreSQL database; `render.yaml` defines production. Both use the Dockerfile, Alembic migrations, private S3-compatible image/model storage and `/ready` health checks. The frontend uses `VITE_API_URL` and, on staging, `VITE_APP_ENV=staging` to require login before prediction.

```bash
python -m unittest discover -s tests
python scripts/verify_release.py --require-artifact
python scripts/benchmark_api.py image-a.jpg image-b.jpg image-c.jpg --url https://<staging-api> --token-env STAGING_TOKEN --runs 100 --concurrency 4 --output results/staging-api-benchmark.json
python scripts/check_staging_benchmark.py --report results/staging-api-benchmark.json
```

`/predict` returns model version, queue wait, model execution and total processing time. `/stats` shows operational latency and reviewed accuracy; benchmark traffic is excluded. `/history`, `/users`, `/model-info` and `/stats` require authentication. The staging `/predict` endpoint also requires authentication.
