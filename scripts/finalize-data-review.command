#!/bin/zsh
set -u
cd "${0:A:h}/.."
umask 077
if [[ -x .venv/bin/python ]]; then
  .venv/bin/python -m scripts.finalize_data_review
else
  python3 -m scripts.finalize_data_review
fi
result=$?
echo "Hoàn tất với mã: $result. Nhấn Enter để đóng cửa sổ."
read -r
exit "$result"
