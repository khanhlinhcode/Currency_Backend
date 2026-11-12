# Currency AI backend

FastAPI nhận dạng **mệnh giá** tiền giấy VN, Mỹ và Ấn Độ (25 lớp). `suspicious`/confidence thấp có nghĩa là cần người xem lại; hệ thống **chưa được kiểm chứng để phát hiện tiền giả**.

## Trạng thái hiện tại (12/11/2026)

| Hạng mục | Trạng thái đã xác minh |
|---|---|
| Staging | [API Render Free](https://currency-ai-free-staging.onrender.com/ready) và [dashboard](https://currency-ai-free-dashboard.onrender.com/) hoạt động; backend commit `5fe3be34` đã qua CI và `live`. |
| Model | `1.1.0`, `candidate`, dataset `legacy-unversioned`; **chưa có model được phê duyệt cho production**. Accuracy 96,29% cũ chỉ mang tính chẩn đoán. |
| Dữ liệu | 10.106 ảnh cũ: **0 ảnh được duyệt đủ**, 10.106 thiếu `group_id`, 0 nhãn mâu thuẫn chưa xử lý. Tập chia cũ có 182 nhóm ảnh trùng qua train/val/test. |
| Kiểm tra code | 46/46 unit test cục bộ đạt trên working tree; CI của bản staging `5fe3be34` đạt 45/45. Đây là kiểm tra phần mềm/fixture, không phải accuracy model. |
| Production | Chưa triển khai. `render.yaml` bị Render trả `need_payment_info` cho service và database trả phí; release gate cũng chặn model legacy. |

[Báo cáo hậu triển khai](docs/evidence/2026-11-11-integration-preflight.json) phân biệt rõ kết quả cloud với diễn tập cục bộ. Staging dùng PostgreSQL và private Supabase Storage; hệ thống có MLOps Runs, so sánh model, benchmark, monitoring và trang admin phát hành/rollback. Lần deploy mới nhất chỉ smoke `/ready` và dashboard; các luồng đăng nhập/predict đã được kiểm tra ở các bản staging trước, xem [validation report](docs/validation-report.md).

Kiểm tra ngày 12/11: `/ready` trả HTTP 200 sau 42,564 giây từ client, dashboard trả HTTP 200 sau 0,606 giây; đây là hai yêu cầu đơn lẻ, không phải benchmark hay cam kết độ trễ. Render vẫn báo backend commit `5fe3be34` ở trạng thái `live`. Truy vấn log lỗi Render lần này không thực hiện được do dịch vụ log trả 503.

### Số liệu hiệu năng staging đã đo

Ba ảnh khác lớp/kích thước (56–596 KB), 10 warmup và 100 yêu cầu đo cho mỗi mức tải trên **Render Free, model candidate 1.1.0**, ngày 10/11/2026:

| Đồng thời | Thành công | Throughput | Client p95 | Client p99 |
|---:|---:|---:|---:|---:|
| 1 | 100/100 | 1,351 req/s | 1,005 s | 1,222 s |
| 4 | 100/100 | 2,535 req/s | 2,090 s | 2,266 s |
| 8 | 100/100 | 2,344 req/s | 5,216 s | 7,496 s |

Một lần đánh thức Render Free mất 52,688 s; đây **không phải** phân phối cold start hay SLA. Bảng trên đo tốc độ dịch vụ, không chứng minh model đúng. Có báo cáo tải kéo dài 300 yêu cầu và CPU/RAM trong [validation report](docs/validation-report.md).

## MLOps hiện có và giới hạn

- Mỗi run ghi Git commit, dataset/model version, seed, cấu hình train, lịch sử epoch, hash artifact, test-set metrics từng lớp, confusion matrix, calibration, benchmark và release gate. Metadata ở PostgreSQL, artifact ở private Storage. CI kiểm tra schema/gate trên fixture; huấn luyện và đánh giá nặng chạy riêng.
- Monitoring theo dõi lỗi, latency, confidence thấp và review. Drift chỉ được tính khi có baseline dataset sạch và đủ mẫu; hiện trạng thái chất lượng là **chưa đủ dữ liệu**. Cảnh báo mở điều tra, không tự train hay tự thay model.
- Admin có thể kích hoạt/rollback **run đã được duyệt** từ dashboard, không cần sửa code. API poll channel mỗi 60 giây; thiết bị tương thích poll mỗi 300 giây, kiểm chữ ký, hash, hợp đồng model và self-test trước khi đổi pointer. Chưa thử cài model đã duyệt trên thiết bị vật lý.
- Backup PostgreSQL/object đã diễn tập với hạ tầng cô lập và khóa backup tách biệt. **Chưa backup/restore dữ liệu Supabase thật** vì chưa có kho backup/credential riêng ở phiên vận hành này.

Chi tiết: [MLOps v2 runbook](docs/mlops-v2-runbook.md), [release control](docs/release-control-runbook.md), [production runbook](docs/production-runbook.md).

## Train lại và cập nhật model

1. **Chuẩn bị dữ liệu có nguồn gốc.** Nếu có hồ sơ chụp cho ảnh cũ, dùng [giao diện duyệt](docs/data-review-ui.md) hoặc [nhập `group_id` theo lô](docs/batch-data-review.md). Nếu không, [thu thập ảnh mới](docs/new-capture-intake.md), xác minh và gộp các đợt chụp. Không tạo `group_id` từ tên file, hash hoặc EXIF.
2. **Chốt dataset v2:** mọi ảnh được duyệt; không còn nhãn mâu thuẫn hay nhóm/ảnh trùng qua split. Chạy `scripts/finalize-data-review.command` (hoặc `python -m scripts.finalize_data_review` với `--raw`, `--review-csv`, `--version` tương ứng). Giữ snapshot và manifest bất biến.
3. **Khóa ngưỡng trước test:** sao chép `docs/model-quality-gates.example.json` và `docs/training-config.example.json` vào file riêng tư, đặt dataset version, người/thời điểm phê duyệt, `status=locked` và tham số/seed. Ngưỡng mẫu chưa phải chính sách phát hành.
4. **Train trên máy phù hợp** với Python 3.11 và `requirements-training.txt`:

   ```bash
   python scripts/run_candidate_pipeline.py \
     --dataset-root dataset/v2/<version>/processed \
     --version <model-version> \
     --benchmark-image <anh-dai-dien.jpg> \
     --training-config results/training-config-v2.json \
     --gates-file results/quality-gates-v2.json
   ```

   Pipeline dừng nếu audit/review chưa đạt; khi chạy được, nó tạo **Keras candidate** và bằng chứng đánh giá. Có thể thêm `--archive-to-s3` trên máy đã cấu hình private staging Storage/PostgreSQL. Không dùng accuracy cũ để duyệt.
5. **Xem xét và phát hành:** người phụ trách kiểm tra per-class recall, macro F1, ECE, coverage, confusion matrix, latency và gate; model không đạt thì giữ candidate. Nếu đạt, phải phê duyệt và lưu một **approved run** với đúng hash artifact/metrics/gate, rồi admin mới kích hoạt trên trang **Phát hành & thiết bị**. Khi kênh đổi, API/thiết bị tương thích tải bản mới qua mạng và có thể rollback về approved run trước đó.

**Giới hạn của bước 5 hiện tại:** pipeline tự động chỉ train/đánh giá Keras. API Render Free dùng ONNX Runtime và không nhận Keras khi thiếu TensorFlow. `scripts/export_onnx.py` chuyển đổi được; `training/evaluate_onnx.py --release-grade` kiểm tra lại audit dataset v2 và policy đã khóa, rồi đánh giá đúng ONNX hash. Bước chuyển đổi → benchmark ONNX → release gate → ghi approved run → lưu private Storage hiện vẫn là thao tác riêng, chưa có pipeline end-to-end được kiểm chứng với dataset thật. Không dùng metrics Keras để phê duyệt ONNX. Khi model giảm chất lượng, hệ thống **cảnh báo để người phụ trách điều tra và duyệt lần train mới**, không tự thay model.

## Chạy cục bộ

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements-api.txt
python -m alembic upgrade head
uvicorn api.main:app --reload
```

Mặc định development dùng SQLite và file ảnh cục bộ. Đặt biến môi trường từ `.env.example` trong file riêng tư; không commit secret. API docs: `http://127.0.0.1:8000/docs`. Frontend ở repository `currency-frontend` cùng cấp. Kiểm tra code bằng `python -m unittest discover -s tests` và `python scripts/verify_release.py`; lệnh verify chỉ kiểm tra hợp đồng artifact/manifest, **không phê duyệt chất lượng model**.
