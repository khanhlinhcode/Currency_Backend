# Release control and connected devices

This is the operational path for a model with the **same input, preprocessing, class order and confidence contract** as the deployed API. A model that changes that contract requires a reviewed application/firmware update. The first production bootstrap also requires an approved manifest and artifact; the current 1.1.0 legacy candidate is blocked.

## Release decision

1. Collect prediction metadata and private evidence images. An admin reviews labels on the dashboard. Review outcomes are operational evidence, not held-out test-set metrics.
2. If monitoring reports drift, low confidence, low reviewed accuracy or a latency regression, investigate image quality, traffic mix and labels. A warning opens an investigation; it does **not** approve retraining or an automatic model replacement.
3. Curate a new human-attested dataset snapshot. All images need a physical-note/capture-session `group_id`; label conflicts and duplicate groups across splits must be resolved. Lock quality thresholds **before** evaluation.
4. Train and evaluate on a suitable training machine using `scripts/run_candidate_pipeline.py`. Verify the grouped audit, per-class recall, calibration, confusion matrix, benchmark and release gate. Export a deployable artifact with `scripts/export_onnx.py` if targeting the free ONNX API or constrained devices, then evaluate and archive evidence for that exact artifact hash. A Keras model can only be activated on a serving environment with compatible TensorFlow installed.
5. After independent review and an approved run record is archived in private Storage, an admin opens **Phát hành & thiết bị** in the dashboard. Select an approved run, enter a reason and activate it. The server rechecks the archived gate, audit, model bytes and inference contract. Candidate and legacy runs are rejected. The channel uses a generation number to reject stale concurrent edits.
6. API instances poll the channel every 60 seconds and atomically swap their in-memory model after download, hash verification and loadability checks. In-flight requests keep their original model snapshot. `/ready`, `/model-info`, prediction history and monitoring show the actual serving version. A failed load preserves the previous model and appears in monitoring.
7. Devices poll every 300 seconds by default, download the channel artifact over HTTPS, verify SHA-256 and atomically replace `current.json`. A local self-test command can be required before pointer replacement. The serving application on the device must watch `current.json` and reload when `generation` changes. This integration is installed once per hardware product; there is no per-model firmware edit.

The dashboard activation does not bypass the release gate. It changes only an approved channel pointer. Production and staging should have separate PostgreSQL databases, private buckets, secrets and device credentials; never register a production device against staging.

## Provision a device once

An admin creates a device under **Phát hành & thiết bị**. The token is shown once. Store it in the device secret store, never in Git, an image, a command-line argument or a public log. Example for a Linux device after securely provisioning the token as `CURRENCY_DEVICE_TOKEN`:

```bash
python3 scripts/device_updater.py \
  --url https://YOUR-PRODUCTION-API \
  --device-id YOUR-DEVICE-ID \
  --state-dir /var/lib/currency-ai \
  --self-test /usr/local/bin/currency-model-self-test
```

The updater appends the candidate model path to the self-test command. It accepts HTTPS (or localhost for isolated tests), keeps prior releases under `releases/`, and reports `ready` or `failed` by heartbeat. Run it under a service supervisor with restricted file permissions and restart policy. If the network is down or the artifact fails its hash/self-test, the old `current.json` remains active. Revoking the device in the dashboard stops future polls/uploads; it cannot erase a model already stored on an offline device.

## Device prediction evidence

After local inference, firmware posts up to 100 metadata events at a time to `POST /devices/{id}/events` using the `X-Device-Token` header. Each event has a UUID `event_id`, an approved `run_id`, `prediction`, confidence between 0 and 1, `observed_at` with timezone, and optional `inference_ms`. Retries with the same ID and content are idempotent. Send no personal identifiers in these fields. A JPEG evidence image of at most 5 MB can be attached at `POST /devices/{id}/events/{event_id}/image`; this requires the product's data-capture consent policy. Images stay in private Storage and are fetched through an admin-only API. An admin can review a label only after evidence is attached. Model training must use a separately curated, versioned dataset; operational events do not silently enter the held-out test set.

Monitoring shows active device count, offline/outdated devices, device sample size, low-confidence proportion, reviewed accuracy and distribution drift. Drift is `insufficient_data` until at least 100 events and a clean approved baseline exist; reviewed accuracy is hidden until at least 30 reviewed events. Offline means no heartbeat for 20 minutes. These are staging thresholds in `docs/free-staging-monitor-targets.json`; production thresholds need a measured baseline and an alert receiver.

## Rollback and recovery

An admin enters a reason and clicks **Quay lại bản trước**. The server verifies that the previous run remains approved and intact, increments the generation and records an immutable release event. API instances and online devices converge on that generation during their next polls. Review per-device heartbeat and run ID; offline devices update after reconnecting. Keep a separate PostgreSQL backup and replicated object store: a channel rollback does not reverse data writes or restore deleted objects.

The rollback path is implemented and locally tested for generation safety. A cloud drill with two approved model runs and physical devices is **pending** because no clean approved v2 run or hardware fleet exists yet. Do not quote a recovery time objective until that drill is measured.

## Current staging evidence and limits (2026-11-10)

- Dashboard: `https://currency-ai-free-dashboard.onrender.com/mlops/releases` (admin login). API: `https://currency-ai-free-staging.onrender.com`.
- Four archived warm/sustained benchmark reports are under `docs/evidence/20261110-free-cloud-*.json` and in the private MLOps registry. All measured requests succeeded. These measurements use the **candidate 1.1.0** model and measure service speed, not classification quality.
- Raw dataset review remains pending for 10,106 images; the legacy split has 182 duplicate groups crossing splits and two conflicting-label groups. No release channel has an approved run, so the dashboard correctly shows no deployable model.
- No physical device is connected. Device artifact and telemetry flows have local contract tests; cloud smoke checks verify access control and the empty-channel behavior. A real firmware integration and two-version rollback drill remain necessary.
- The separate cloud backup bucket, actual Supabase restore drill, production isolation, retention policy for evidence images and independent alert receiver are not yet complete. The local PostgreSQL restore drill is evidence of script behavior only.
