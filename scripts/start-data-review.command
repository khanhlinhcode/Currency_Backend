#!/bin/zsh
set -euo pipefail
cd "${0:A:h}/.."
umask 077
if [[ -x .venv/bin/python ]]; then
  exec .venv/bin/python -m training.review_app
fi
exec python3 -m training.review_app
