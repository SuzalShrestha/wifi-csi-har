#!/usr/bin/env bash
# Install ESP-IDF v5.x for ESP32-S3 development (macOS/Linux).
# Idempotent: safe to re-run.
set -euo pipefail

IDF_DIR="${IDF_DIR:-$HOME/esp/esp-idf}"
IDF_BRANCH="${IDF_BRANCH:-v5.3.2}"

if ! command -v cmake >/dev/null; then
  echo "cmake missing — install prerequisites first:"
  echo "  brew install cmake ninja dfu-util ccache"
  exit 1
fi

if [ ! -d "$IDF_DIR" ]; then
  mkdir -p "$(dirname "$IDF_DIR")"
  git clone --depth 1 --branch "$IDF_BRANCH" --recursive \
    https://github.com/espressif/esp-idf.git "$IDF_DIR"
else
  echo "ESP-IDF already present at $IDF_DIR"
fi

"$IDF_DIR/install.sh" esp32s3

echo
echo "Done. In every new shell before using idf.py, run:"
echo "  source $IDF_DIR/export.sh"
