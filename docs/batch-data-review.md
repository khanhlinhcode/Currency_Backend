# Nhập group_id theo lô từ hồ sơ nguồn

Dùng cách này khi có sổ chụp, bảng quản lý mẫu tiền hoặc hồ sơ nguồn khác cho biết **ảnh nào thuộc tờ tiền vật lý hoặc phiên chụp nào**. Công cụ không suy ra `group_id` từ tên file, hash, thư mục lớp hay EXIF. `source_ref` là tham chiếu đến hồ sơ mà người duyệt thực sự đã kiểm tra; chỉ điền khi có căn cứ. Ảnh không có căn cứ vẫn để trống và chờ duyệt hoặc được thay bằng dữ liệu mới có thể truy vết.

## 1. Xuất bảng cần điền

Từ thư mục `currency-backend`:

```bash
.venv/bin/python -m training.import_review_batch --export-template results/source-review-template.csv
```

Bảng gồm `path`, `sha256`, `current_label`, `reviewed_label`, `group_id`, `source_ref`, `reviewed_by`, `reviewed_at`, `review_notes`. Không sửa ba cột đầu. Với mỗi ảnh có bằng chứng, điền:

- `group_id`: mã tờ tiền hoặc phiên chụp từ hồ sơ nguồn. Các ảnh trùng byte phải nhận cùng mã, nhưng hash **không phải** căn cứ để tự tạo mã.
- `source_ref`: tên/mã hồ sơ và dòng tham chiếu cụ thể, ví dụ `capture-ledger-2026-11-11 / row 17`. Lưu hồ sơ gốc riêng tư.
- `reviewed_by`: người đã đối chiếu hồ sơ; `reviewed_at`: thời điểm ISO-8601 có múi giờ, ví dụ `2026-11-11T14:30:00+07:00`.
- `reviewed_label`: để trống để giữ nhãn hiện có, hoặc giữ nhãn đã xác nhận trong mẫu. Chỉ thay khi người duyệt đã kiểm tra lại ảnh.

Có thể nhập từng phần: các dòng chưa có căn cứ để trống `group_id`. Không cần điền 10.106 dòng một lần. Nếu bảng mẫu đã tồn tại, dùng tên file mới để không ghi đè việc đang làm.

## 2. Kiểm tra trước, rồi mới ghi

```bash
.venv/bin/python -m training.import_review_batch --input results/source-review-template.csv --report results/review-batch-check.json
.venv/bin/python -m training.import_review_batch --input results/source-review-template.csv --apply --report results/review-batch-apply.json
```

Lệnh đầu **không sửa** dữ liệu duyệt; báo số quyết định hợp lệ, số ảnh còn thiếu và hash CSV. Lệnh `--apply` kiểm tra lại rồi ghi một lần, tạo bản sao `results/group-review.before-batch-*.csv`, cập nhật `results/review-queue.csv` và `results/review-status.json`. Chạy lại cùng file không ghi trùng. Công cụ từ chối đường dẫn/hash cũ, ảnh gốc đã đổi, nhóm/nhãn mâu thuẫn giữa các bản sao, thiếu `source_ref` hoặc người/thời điểm duyệt. Nếu cần sửa một quyết định đã lưu, mở ảnh trong giao diện duyệt và ghi rõ lý do; công cụ nhập theo lô không tự ghi đè quyết định cũ.

**Đóng giao diện duyệt cục bộ trước khi `--apply`; mở lại sau đó.** Ứng dụng và công cụ nhập dùng khóa file để tránh ghi cùng lúc, nhưng ứng dụng đang mở sẽ yêu cầu khởi động lại khi CSV thay đổi bên ngoài.

CSV mẫu, hồ sơ nguồn, bản sao và báo cáo nhập nằm trong `results/` (Git bỏ qua); không đưa ảnh gốc hoặc thông tin nguồn nhạy cảm vào Git. Báo cáo nhập chỉ xác minh cấu trúc, hash và tính nhất quán của lời xác nhận; nó không thể tự xác thực một mã tờ tiền ngoài hồ sơ nguồn.

## 3. Chốt dataset

Chạy `python -m training.review_queue --require-complete` để kiểm tra hàng chờ. Chỉ khi **0 ảnh còn thiếu** mới chạy `scripts/finalize-data-review.command` để tạo snapshot v2 bất biến và audit chia tập. Việc xử lý xong hai cặp nhãn trước đó không thay thế bước xác nhận `group_id`.

## Nếu bộ ảnh cũ không có hồ sơ nguồn

Thu thập một bộ ảnh mới trong thư mục riêng. Ghi mã tờ tiền/phiên chụp, lớp, người chụp và tham chiếu hồ sơ **ngay lúc chụp**; giữ ảnh gốc và bảng nguồn riêng tư. Không tạo mã nhóm từ tên file sau khi chụp. Quy trình hiện có nhận một `--raw` và `--review-csv` riêng, nên có thể kiểm kê bộ mới mà không sửa CSV duyệt 10.106 ảnh cũ:

```bash
.venv/bin/python training/inventory_raw.py --raw dataset/new-raw --review-csv results/new-review.csv --output results/new-inventory.json
.venv/bin/python -m training.import_review_batch --review-csv results/new-review.csv --export-template results/new-source-template.csv
.venv/bin/python -m training.import_review_batch --raw dataset/new-raw --review-csv results/new-review.csv --input results/new-source-template.csv --apply --queue results/new-review-queue.csv --status results/new-review-status.json --report results/new-review-import.json
```

Sau khi đủ duyệt, chạy `python -m scripts.finalize_data_review --raw dataset/new-raw --review-csv results/new-review.csv --output-root dataset/v2 --version <version-mới>`. Mỗi lớp cần đủ nhóm vật lý độc lập để chia train/val/test; số lượng ảnh hoặc nhãn lớp thiếu vẫn khiến bước huấn luyện/release gate dừng. Không trộn bộ mới với bộ cũ chưa có provenance để đạt số lượng.
