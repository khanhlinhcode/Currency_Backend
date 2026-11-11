"""Loopback-only browser UI for human review of raw dataset labels and group IDs.

Run ``python3 -m training.review_app`` from the backend root. This app never
guesses a physical-note/capture-session group from a filename or image hash.
"""

import argparse
import csv
import hashlib
import html
import json
import os
import re
import secrets
import shutil
import tempfile
import threading
import urllib.parse
import webbrowser
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from training.review_queue import generate_review_queue, review_issues


FIELDS = ["path", "current_label", "exact_hash_group", "label_conflict",
          "reviewed_label", "group_id", "reviewed_by", "reviewed_at", "review_notes"]
IMAGE_TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
               ".webp": "image/webp"}
CSS = """<style>
*{box-sizing:border-box}body{font:16px system-ui,-apple-system,sans-serif;background:#f5f7fb;color:#172237;margin:0}
header{background:#172b4d;color:white;padding:20px max(24px,calc((100vw - 1120px)/2))}
header a{color:white}main{max-width:1120px;margin:28px auto;padding:0 20px}
.card{background:white;border:1px solid #dce3ec;border-radius:14px;padding:22px;margin:18px 0;box-shadow:0 2px 8px #18283d09}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px}.metric{background:#eef3f9;padding:14px;border-radius:10px}
.metric strong{display:block;font-size:24px}label{display:block;font-weight:600;margin:12px 0 5px}
input,select,textarea{width:100%;font:inherit;padding:10px;border:1px solid #b5c2d2;border-radius:8px;background:white}
input[type=checkbox]{width:auto}button,.button{display:inline-block;border:0;border-radius:9px;background:#2357a5;color:white;padding:11px 16px;font:inherit;font-weight:600;text-decoration:none;cursor:pointer;margin:8px 8px 0 0}
.secondary{background:#e4ebf4;color:#172b4d}.warning{background:#fff2d5;border-left:4px solid #d58309;padding:13px;border-radius:7px}
table{border-collapse:collapse;width:100%}td,th{text-align:left;border-bottom:1px solid #e5eaf0;padding:10px;word-break:break-word}tr:hover{background:#f7f9fc}
.muted{color:#63738a}.image{max-width:100%;max-height:65vh;object-fit:contain;background:#e9edf3;border-radius:9px}
.review-layout{display:grid;grid-template-columns:minmax(260px,1fr) minmax(300px,1fr);gap:22px}
@media(max-width:800px){.review-layout{grid-template-columns:1fr}}
</style>"""


def esc(value):
    return html.escape(str(value), quote=True)


def digest_file(path):
    sha = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def visible_exif(path):
    """Show limited source metadata as context; never derive a group ID from it."""
    try:
        from PIL import Image
        with Image.open(path) as image:
            exif = image.getexif()
        names = ((306, "Thời gian EXIF"), (36867, "Thời gian chụp gốc"),
                 (271, "Hãng máy ảnh"), (272, "Mẫu máy ảnh"))
        return [(label, str(exif[tag])[:160]) for tag, label in names if exif.get(tag)]
    except (ImportError, OSError, ValueError):
        return []


class ReviewStore:
    def __init__(self, raw_root, review_csv, queue_csv, report_json):
        self.raw_root = Path(raw_root).resolve(strict=True)
        if not self.raw_root.is_dir():
            raise ValueError("Raw dataset root must be a directory")
        self.review_csv = Path(review_csv).resolve(strict=True)
        self.queue_csv = Path(queue_csv)
        self.report_json = Path(report_json)
        self.lock = threading.RLock()
        self._load()
        generate_review_queue(self.review_csv, self.queue_csv, self.report_json)

    def _load(self):
        with self.review_csv.open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            if reader.fieldnames != FIELDS:
                raise ValueError("Review CSV schema differs from inventory_raw.py; rerun inventory without --reset-review")
            self.rows = list(reader)
        self.by_path = {}
        for row in self.rows:
            relative = Path(row["path"])
            if relative.is_absolute() or len(relative.parts) != 3 or ".." in relative.parts:
                raise ValueError(f"Invalid review path: {row['path']}")
            if row["path"] in self.by_path:
                raise ValueError(f"Duplicate review path: {row['path']}")
            if not re.fullmatch(r"[0-9a-f]{64}", row["exact_hash_group"]):
                raise ValueError(f"Invalid image hash for {row['path']}")
            self.by_path[row["path"]] = row
        self.labels = sorted({row["current_label"] for row in self.rows})
        self.csv_sha256 = digest_file(self.review_csv)

    def image_path(self, row):
        relative = Path(row["path"])
        target = (self.raw_root / relative).resolve(strict=True)
        if not target.is_relative_to(self.raw_root) or not target.is_file():
            raise ValueError("Image path escapes raw dataset")
        if target.suffix.lower() not in IMAGE_TYPES or digest_file(target) != row["exact_hash_group"]:
            raise ValueError("Raw image changed since inventory; review stopped")
        return target

    def status(self):
        pending = [row for row in self.rows if review_issues(row)]
        return {"total": len(self.rows), "complete": len(self.rows) - len(pending),
                "pending": len(pending),
                "conflicts": sum(row["label_conflict"] == "yes" and not row["reviewed_label"] for row in self.rows)}

    def next_pending(self, path):
        start = next((i for i, row in enumerate(self.rows) if row["path"] == path), -1)
        for offset in range(1, len(self.rows) + 1):
            row = self.rows[(start + offset) % len(self.rows)]
            if review_issues(row):
                return row["path"]
        return None

    def save(self, form):
        with self.lock:
            if digest_file(self.review_csv) != self.csv_sha256:
                raise ValueError("Review CSV changed outside this app. Restart before editing.")
            path = form.get("path", "")
            row = self.by_path.get(path)
            if row is None:
                raise ValueError("Unknown image")
            if form.get("expected_reviewed_at", "") != row["reviewed_at"]:
                raise ValueError("This review changed in another tab. Reload before editing.")
            self.image_path(row)
            label = form.get("reviewed_label", "").strip()
            group = form.get("group_id", "").strip()
            reviewer = form.get("reviewed_by", "").strip()
            notes = form.get("review_notes", "").strip()
            if label and label not in self.labels:
                raise ValueError("Reviewed label must be one of the inventory classes")
            if group and (len(group) > 128 or any(ord(ch) < 32 for ch in group)):
                raise ValueError("Group ID must be at most 128 characters without control characters")
            if not 2 <= len(reviewer) <= 80 or any(ord(ch) < 32 for ch in reviewer):
                raise ValueError("Enter a reviewer name or ID of 2–80 characters")
            if len(notes) > 2000:
                raise ValueError("Notes exceed 2,000 characters")
            exact_copies = [item for item in self.rows if item is not row and
                            item["exact_hash_group"] == row["exact_hash_group"]]
            if label and any(item["reviewed_label"] and item["reviewed_label"] != label
                             for item in exact_copies):
                raise ValueError("Identical image copies have a different reviewed label")
            if group and any(item["group_id"] and item["group_id"] != group
                             for item in exact_copies):
                raise ValueError("Identical image copies have a different group ID")
            old = dict(row)
            old_sha256 = self.csv_sha256
            row.update(reviewed_label=label, group_id=group, reviewed_by=reviewer,
                       reviewed_at=datetime.now(timezone.utc).isoformat(), review_notes=notes)
            try:
                self._write_atomic()
            except Exception:
                if digest_file(self.review_csv) == old_sha256:
                    row.clear()
                    row.update(old)
                else:
                    self._load()
                raise
            return row

    def _write_atomic(self):
        backup = self.review_csv.with_suffix(".initial-backup.csv")
        if not backup.exists():
            shutil.copy2(self.review_csv, backup)
            backup.chmod(0o600)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile("w", dir=self.review_csv.parent,
                                             prefix=".review-", suffix=".csv", delete=False,
                                             newline="", encoding="utf-8") as target:
                temporary = Path(target.name)
                writer = csv.DictWriter(target, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(self.rows)
                target.flush()
                os.fsync(target.fileno())
            temporary.chmod(0o600)
            os.replace(temporary, self.review_csv)
            self.csv_sha256 = digest_file(self.review_csv)
            generate_review_queue(self.review_csv, self.queue_csv, self.report_json)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


class ReviewHandler(BaseHTTPRequestHandler):
    def _allowed_host(self):
        host = self.headers.get("Host", "")
        return host in {f"127.0.0.1:{self.server.server_port}",
                        f"localhost:{self.server.server_port}"}

    def _send(self, status, payload, content_type="text/html; charset=utf-8", extra=None):
        if isinstance(payload, str):
            payload = payload.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' data:; style-src 'unsafe-inline'; script-src 'unsafe-inline'; form-action 'self'; base-uri 'none'")
        if extra:
            for key, value in extra.items():
                self.send_header(key, value)
        self.end_headers()
        self.wfile.write(payload)

    def _page(self, title, body):
        return f'<!doctype html><html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{esc(title)} · Currency AI</title>{CSS}<header><a href="/">Currency AI · Duyệt dữ liệu cục bộ</a></header><main>{body}</main></html>'

    def _bad(self, status, message):
        self._send(status, self._page("Có lỗi", f'<div class="card"><h1>{esc(message)}</h1><a class="button" href="/">Về danh sách</a></div>'))

    def do_GET(self):
        if not self._allowed_host():
            return self._bad(403, "Địa chỉ truy cập không hợp lệ")
        parsed = urllib.parse.urlparse(self.path)
        query = urllib.parse.parse_qs(parsed.query)
        store = self.server.store
        if parsed.path == "/":
            q = query.get("q", [""])[0].strip().casefold()[:100]
            mode = query.get("mode", ["pending"])[0]
            try:
                page = max(1, int(query.get("page", ["1"])[0]))
            except ValueError:
                page = 1
            with store.lock:
                status = store.status()
                rows = [row for row in store.rows if
                        (mode == "all" or (mode == "conflicts" and row["label_conflict"] == "yes" and not row["reviewed_label"])
                         or (mode == "pending" and review_issues(row)))
                        and (not q or q in row["path"].casefold() or q in row["current_label"].casefold())]
                total = len(rows)
                shown = rows[(page - 1) * 30:page * 30]
                table = "".join(f'<tr><td><a href="/review?path={urllib.parse.quote(row["path"])}">{esc(row["path"])}</a></td><td>{esc(row["current_label"])}</td><td>{"⚠ Nhãn mâu thuẫn" if row["label_conflict"] == "yes" else ""}</td><td>{esc(", ".join(review_issues(row)) or "Đã duyệt")}</td></tr>' for row in shown)
            filters = "".join(f'<option value="{value}" {"selected" if mode == value else ""}>{label}</option>' for value, label in (("pending", "Cần duyệt"), ("conflicts", "Nhãn mâu thuẫn"), ("all", "Tất cả")))
            base = f'/?q={urllib.parse.quote(q)}&mode={urllib.parse.quote(mode)}&page='
            nav = (f'<a class="button secondary" href="{base}{page-1}">← Trước</a>' if page > 1 else "") + (f'<a class="button secondary" href="{base}{page+1}">Tiếp →</a>' if page * 30 < total else "")
            body = f'''<h1>Duyệt dữ liệu v2</h1><p class="warning">Chỉ người đã xem ảnh và biết nguồn gốc tờ tiền/phiên chụp mới được nhập group_id. Tên file và hash không chứng minh cùng tờ tiền. Dữ liệu ở máy này, không tải lên staging.</p>
            <div class="grid"><div class="metric"><strong>{status['total']}</strong>Tổng ảnh</div><div class="metric"><strong>{status['complete']}</strong>Đã duyệt đủ</div><div class="metric"><strong>{status['pending']}</strong>Chờ duyệt</div><div class="metric"><strong>{status['conflicts']}</strong>Nhãn mâu thuẫn</div></div>
            <div class="card"><form method="get"><label>Tìm theo đường dẫn hoặc lớp</label><input name="q" value="{esc(q)}"><label>Bộ lọc</label><select name="mode">{filters}</select><button>Tìm</button></form></div>
            <div class="card"><h2>Danh sách ({total} kết quả)</h2><table><thead><tr><th>Ảnh</th><th>Nhãn gốc</th><th>Lưu ý</th><th>Thiếu</th></tr></thead><tbody>{table}</tbody></table>{nav}<p class="muted">Trang {page}; 30 ảnh/trang. Mỗi lần lưu cập nhật CSV và báo cáo trạng thái.</p></div>'''
            return self._send(200, self._page("Danh sách ảnh", body))
        if parsed.path in {"/review", "/image"}:
            path = query.get("path", [""])[0]
            with store.lock:
                row = store.by_path.get(path)
                if row is None:
                    return self._bad(404, "Không tìm thấy ảnh")
                try:
                    image = store.image_path(row)
                except (ValueError, FileNotFoundError) as error:
                    return self._bad(409, str(error))
                if parsed.path == "/image":
                    return self._send(200, image.read_bytes(), IMAGE_TYPES[image.suffix.lower()])
                duplicates = [item for item in store.rows if item["exact_hash_group"] == row["exact_hash_group"]]
                duplicate_links = "".join(f'<li><a href="/review?path={urllib.parse.quote(item["path"])}">{esc(item["path"])}</a> · {esc(item["current_label"])}</li>' for item in duplicates)
                exif = visible_exif(image)
                exif_list = "".join(f"<li>{esc(label)}: {esc(value)}</li>" for label, value in exif)
                metadata = f"<ul>{exif_list}</ul>" if exif else '<p class="muted">Không có EXIF hữu ích.</p>'
                options = '<option value="">Giữ nhãn gốc / chưa quyết định</option>' + "".join(f'<option value="{esc(label)}" {"selected" if row["reviewed_label"] == label else ""}>{esc(label)}</option>' for label in store.labels)
                issues = ", ".join(review_issues(row)) or "Đã duyệt đủ"
                body = f'''<a href="/">← Danh sách</a><h1>{esc(row['path'])}</h1><p>Nhãn gốc: <strong>{esc(row['current_label'])}</strong> · Cần xử lý: {esc(issues)}</p>
                <div class="review-layout"><div class="card"><img class="image" src="/image?path={urllib.parse.quote(path)}" alt="Ảnh cần duyệt"><p class="muted">SHA-256: {esc(row['exact_hash_group'])}</p><h3>Metadata từ ảnh</h3>{metadata}<p class="muted">EXIF chỉ là ngữ cảnh để tìm hồ sơ nguồn gốc; không tự chứng minh group_id.</p><h3>Ảnh có cùng nội dung byte ({len(duplicates)})</h3><ul>{duplicate_links}</ul><p class="warning">Các ảnh trùng byte cần nhãn nhất quán, nhưng group_id vẫn phải dựa trên chứng cứ tờ tiền/phiên chụp.</p></div>
                <form class="card" method="post" action="/save"><input type="hidden" name="csrf" value="{esc(self.server.csrf)}"><input type="hidden" name="path" value="{esc(path)}"><input type="hidden" name="expected_reviewed_at" value="{esc(row['reviewed_at'])}">
                <label>Nhãn đã kiểm tra</label><select name="reviewed_label">{options}</select><p class="muted">Ảnh bị gắn nhãn mâu thuẫn phải chọn nhãn sau khi kiểm tra nguồn gốc.</p>
                <label>group_id theo tờ tiền vật lý hoặc phiên chụp</label><input name="group_id" value="{esc(row['group_id'])}" maxlength="128" placeholder="Nhập ID từ hồ sơ nguồn gốc"><p class="muted">Để trống nếu chưa có chứng cứ. Không lấy từ tên file hay hash.</p>
                <label>Người duyệt</label><input id="reviewer" name="reviewed_by" value="{esc(row['reviewed_by'])}" maxlength="80" required minlength="2">
                <label>Ghi chú/chứng cứ</label><textarea name="review_notes" rows="5" maxlength="2000">{esc(row['review_notes'])}</textarea>
                <button name="action" value="next">Lưu và mở ảnh tiếp theo</button><button class="secondary" name="action" value="stay">Lưu và ở lại</button></form></div>
                <script>const x=document.getElementById('reviewer');if(!x.value)x.value=localStorage.getItem('currency-reviewer')||'';x.addEventListener('change',()=>localStorage.setItem('currency-reviewer',x.value));</script>'''
            return self._send(200, self._page("Duyệt ảnh", body))
        if parsed.path == "/status":
            with store.lock:
                return self._send(200, json.dumps(store.status(), ensure_ascii=False), "application/json; charset=utf-8")
        return self._bad(404, "Không tìm thấy trang")

    def do_POST(self):
        if not self._allowed_host() or self.headers.get("Origin") not in (None, f"http://{self.headers.get('Host')}"):
            return self._bad(403, "Yêu cầu không cùng nguồn")
        if urllib.parse.urlparse(self.path).path != "/save":
            return self._bad(404, "Không tìm thấy trang")
        try:
            size = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            return self._bad(400, "Dữ liệu không hợp lệ")
        if size < 1 or size > 16_384:
            return self._bad(413, "Dữ liệu quá lớn")
        form = {key: values[0] for key, values in urllib.parse.parse_qs(
            self.rfile.read(size).decode("utf-8"), keep_blank_values=True).items()}
        if not secrets.compare_digest(form.get("csrf", ""), self.server.csrf):
            return self._bad(403, "Phiên duyệt không hợp lệ")
        try:
            row = self.server.store.save(form)
        except (ValueError, FileNotFoundError) as error:
            return self._bad(409, str(error))
        next_path = row["path"] if form.get("action") == "stay" else self.server.store.next_pending(row["path"])
        location = "/review?path=" + urllib.parse.quote(next_path) if next_path else "/"
        return self._send(303, b"", extra={"Location": location})


def make_server(store, port=8765):
    server = ThreadingHTTPServer(("127.0.0.1", port), ReviewHandler)
    server.store = store
    server.csrf = secrets.token_urlsafe(32)
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("dataset/raw"))
    parser.add_argument("--review-csv", type=Path, default=Path("results/group-review.csv"))
    parser.add_argument("--queue", type=Path, default=Path("results/review-queue.csv"))
    parser.add_argument("--report", type=Path, default=Path("results/review-status.json"))
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    store = ReviewStore(args.raw, args.review_csv, args.queue, args.report)
    server = make_server(store, args.port)
    url = f"http://127.0.0.1:{server.server_port}/"
    print(f"Human review: {url} (local machine only). Press Ctrl+C to stop.", flush=True)
    if not args.no_browser:
        threading.Timer(0.3, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
