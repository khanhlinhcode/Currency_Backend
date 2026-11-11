# Thu thập dataset mới có thể truy vết

Dùng quy trình này khi ảnh cũ không có hồ sơ đủ để xác nhận `group_id`. Nó tạo một **bộ ảnh mới**, không gán nhóm cho 10.106 ảnh cũ. Ảnh và bảng nguồn ở máy riêng tư; không đưa lên Git hoặc staging. Việc kiểm tra bằng code xác minh hash, định dạng, tính đầy đủ và nhất quán; con người vẫn chịu trách nhiệm xác nhận nhãn và nguồn gốc tờ tiền.

## Tổ chức buổi chụp

- Ghi một `group_id` riêng cho **mỗi tờ tiền vật lý** (hoặc một phiên chụp chỉ chứa một lớp tiền). Chụp nhiều góc của cùng tờ tiền thì giữ cùng mã. Không dùng cùng mã cho hai lớp khác nhau.
- Ghi `source_ref` dẫn đến dòng trong sổ chụp/bảng mẫu tiền; lưu hồ sơ này riêng tư. Ghi người và thời điểm chụp, người và thời điểm duyệt nhãn. Người duyệt phải xem ảnh và đối chiếu hồ sơ.
- Thu nhiều nhóm vật lý độc lập cho từng lớp. Công cụ chia tập yêu cầu tối thiểu ba nhóm/lớp; ba nhóm chỉ là mức để chạy kiểm thử, chưa đủ căn cứ cho một đánh giá chất lượng đáng tin cậy.
- Giữ ảnh gốc trong một thư mục nguồn riêng, ví dụ `dataset/new-capture-source/`. Không trộn với `dataset/raw` của bộ cũ. Tên file và hash chỉ dùng để định vị/kiểm tra toàn vẹn, không dùng để tạo `group_id`.

## Tạo bảng chụp

Từ thư mục `currency-backend`:

```bash
.venv/bin/python -m training.intake_capture --source-root dataset/new-capture-source --export-template results/new-capture-template.csv
```

Mỗi ảnh nhận một dòng với `source_path` tương đối và `sha256` được tính tự động. Trong bảng tính, điền các cột còn trống: `label`, `group_id`, `source_ref`, `captured_by`, `captured_at`, `reviewed_by`, `reviewed_at`, `review_notes`. Nhãn phải trùng một trong 25 `class_names` ở `models/model-manifest.json`. Hai thời điểm dùng ISO-8601 có múi giờ, chẳng hạn `2026-11-11T14:30:00+07:00`; thời điểm duyệt không được sớm hơn thời điểm chụp. Không sửa `source_path` hoặc `sha256`.

Một chiến dịch thu thập có thể chứa nhiều phiên chụp trong cùng thư mục nguồn. Chỉ chốt khi bảng bao phủ **mọi ảnh** trong thư mục đó; các ảnh chưa đủ hồ sơ phải được làm rõ hoặc chuyển ra ngoài chiến dịch trước khi chốt. Không thêm ảnh vào một snapshot đã tạo.

## Kiểm tra và lưu bản intake

```bash
.venv/bin/python -m training.intake_capture --source-root dataset/new-capture-source --capture-csv results/new-capture-template.csv --output dataset/intake/campaign-001 --batch-id campaign-001
.venv/bin/python -m training.intake_capture --source-root dataset/new-capture-source --capture-csv results/new-capture-template.csv --output dataset/intake/campaign-001 --batch-id campaign-001 --apply
```

Lệnh đầu chỉ kiểm tra. Lệnh `--apply` tạo **thư mục mới** gồm `raw/`, `review.csv` và `intake-manifest.json`; nó từ chối ghi đè thư mục đã tồn tại. Công cụ kiểm tra hash và giải mã ảnh, toàn bộ ảnh nguồn có mặt trong CSV, nhãn thuộc bộ lớp, `group_id` không vượt qua lớp và các bản sao byte có cùng nhãn/nhóm. Nó ghi người/thời điểm duyệt và tham chiếu nguồn trong manifest. Nếu ảnh hoặc CSV đổi trong lúc nhập, lệnh dừng.

Có thể kiểm tra lại snapshot intake trước khi dùng:

```bash
.venv/bin/python -m training.intake_capture --verify-intake dataset/intake/campaign-001
```

Lệnh này xác minh hash manifest, CSV duyệt, từng ảnh và phạm vi ảnh. Hash phát hiện sửa đổi sau khi ghi; nó không thay thế bằng chứng nguồn do con người cung cấp.

Snapshot intake mới chỉ là bằng chứng thu thập. Cần kiểm toán và chia tập v2 trước khi huấn luyện:

```bash
.venv/bin/python -m training.review_queue --review-csv dataset/intake/campaign-001/review.csv --queue results/new-capture-queue.csv --report results/new-capture-status.json --require-complete
.venv/bin/python -m scripts.finalize_data_review --raw dataset/intake/campaign-001/raw --review-csv dataset/intake/campaign-001/review.csv --output-root dataset/v2 --version 2.0.0
```

`finalize_data_review` từ chối lớp thiếu nhóm độc lập, nhãn/nhóm mâu thuẫn hoặc ảnh trùng qua split. Sau đó khóa ngưỡng chất lượng **trước** khi đánh giá tập test mới. Không dùng accuracy 96,29% của dataset legacy để phát hành.

## Gộp nhiều chiến dịch thu thập

Mỗi intake là một snapshot riêng, không sửa sau khi chốt. Nếu dữ liệu cần thiết nằm trong nhiều snapshot, kiểm tra và gộp chúng vào một thư mục **mới** trước khi chạy `finalize_data_review`:

```bash
.venv/bin/python -m training.merge_capture_intakes --intake dataset/intake/campaign-001 --intake dataset/intake/campaign-002 --output dataset/merged/currency-v2 --merge-id currency-v2
.venv/bin/python -m training.merge_capture_intakes --intake dataset/intake/campaign-001 --intake dataset/intake/campaign-002 --output dataset/merged/currency-v2 --merge-id currency-v2 --apply
.venv/bin/python -m training.merge_capture_intakes --verify-merge dataset/merged/currency-v2
.venv/bin/python -m scripts.finalize_data_review --raw dataset/merged/currency-v2/raw --review-csv dataset/merged/currency-v2/review.csv --output-root dataset/v2 --version 2.0.0
```

Để gộp an toàn, `source_path` phải duy nhất giữa các chiến dịch (đặt đường dẫn nguồn kèm mã chiến dịch từ lúc chụp), và `group_id` phải là mã **toàn cục** không tái sử dụng cho tờ tiền khác. Công cụ từ chối trùng `group_id` hoặc nội dung ảnh byte giữa các chiến dịch; nó không tự đổi tên nhóm, vì hai đợt có thể đã chụp cùng một tờ tiền. Nếu trùng, người quản lý dữ liệu phải đối chiếu hồ sơ vật lý, sửa bảng nguồn và tạo intake version mới; không sửa snapshot cũ hoặc gán nhóm từ hash. Merge manifest lưu hash từng intake, đường dẫn ảnh gốc, nhãn, mã nhóm, người/thời điểm duyệt và tham chiếu nguồn. Bản gộp cũng được xác minh trước và sau khi tạo.
