# Duyệt dữ liệu v2 trên máy cá nhân

Ứng dụng này giúp người duyệt xem từng ảnh và nhập nhãn, `group_id`, người duyệt và ghi chú **mà không cần sửa CSV hay viết code**. Nó chỉ lắng nghe trên `127.0.0.1`; ảnh gốc và kết quả duyệt ở máy này, không đưa 2,9 GB ảnh thô lên Render/Supabase staging.

## Mở giao diện

Trên macOS, nhấp đúp `scripts/start-data-review.command` trong Finder. Trình duyệt sẽ mở `http://127.0.0.1:8765/`. Giữ cửa sổ Terminal mở khi làm việc; đóng bằng Ctrl+C. Nếu Terminal báo thiếu Python, cài môi trường backend theo README trước. Người vận hành kỹ thuật có thể chạy `python3 -m training.review_app`.

Trang đầu hiển thị tổng ảnh, số đã duyệt, số còn thiếu và nhãn mâu thuẫn. Có ô tìm, bộ lọc và phân trang. Mở một ảnh để xem ảnh gốc, nhãn gốc và các bản sao có cùng hash. **Hash và tên file không chứng minh hai ảnh là cùng tờ tiền vật lý hoặc phiên chụp.** Người duyệt phải đối chiếu hồ sơ nguồn gốc rồi mới nhập `group_id`; nếu không có chứng cứ, để trống và ghi chú. Với ảnh có nhãn mâu thuẫn, chọn `reviewed_label` sau khi kiểm tra. Không chọn tùy tiện để vượt cổng phát hành.

Trang chi tiết cũng hiện EXIF thời gian/máy ảnh nếu có. Scan hiện tại chỉ thấy 2.188 ảnh INR có timestamp EXIF chung và không có ảnh nào có `DateTimeOriginal`; ảnh VN không có EXIF. Metadata này chỉ giúp tìm hồ sơ chụp, không đủ để khẳng định `group_id`.

Mỗi lần lưu, ứng dụng ghi `reviewed_by` và thời gian UTC, cập nhật `results/group-review.csv`, `results/review-queue.csv` và `results/review-status.json`. Bản sao ban đầu `results/group-review.initial-backup.csv` được tạo trước lần sửa đầu. Ứng dụng từ chối ảnh đã đổi hash, CSV bị sửa bên ngoài, xung đột giữa hai tab và hai bản sao byte có nhãn hoặc `group_id` đã duyệt khác nhau. Chỉ dữ liệu có **100% ảnh đầy đủ** mới qua bước tiếp theo.

## Chốt một phiên bản bất biến

Sau khi người duyệt hoàn tất, đóng giao diện rồi nhấp đúp `scripts/finalize-data-review.command`; nhập một version mới, ví dụ `2.0.0`. Nếu vẫn còn ảnh thiếu, script dừng trước khi tạo snapshot. Khi hoàn tất, nó tạo `dataset/v2/<version>/` chứa bản sao CSV duyệt, curation report, split manifest, dataset manifest, audit JSON, hash của các file quan trọng và ảnh đã chia train/val/test. Không thể ghi đè một version đã tồn tại. Người vận hành kỹ thuật có thể chạy `python3 -m scripts.finalize_data_review --version 2.0.0`.

Snapshot chỉ được chốt nếu mọi ảnh có thông tin duyệt, nhãn mâu thuẫn đã xử lý, các group tách biệt qua split và audit xác minh đầy đủ. Đây là bằng chứng cấu trúc và hash; nó không thay thế việc kiểm tra tính đúng đắn của lời xác nhận từ người duyệt. Sau đó dùng [MLOps v2 runbook](mlops-v2-runbook.md) để khóa ngưỡng chất lượng, huấn luyện và đánh giá trên tập test mới. Không dùng số accuracy 96,29% cũ để phát hành.

## Trạng thái hiện tại

Ngày 11/11/2026, giao diện đã được xem bằng Playwright WebKit trên tập thật: 10.106 ảnh, 0 ảnh hoàn tất, bốn ảnh thuộc hai nhóm nhãn mâu thuẫn. Không có nhãn hay `group_id` nào được tự điền. [Bằng chứng kiểm tra giao diện](evidence/2026-11-11-local-review-ui-check.json).
