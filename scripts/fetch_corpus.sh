#!/usr/bin/env bash
# Unduh korpus teks untuk training. Pengganti `benchmarks/train_toy.py --corpus hf`
# yang sudah dihapus bersama sisa Python.
#
# Pemakaian:
#   scripts/fetch_corpus.sh                      # TinyStories valid (19 MB)
#   scripts/fetch_corpus.sh tinystories-train    # split train (~1.9 GB)
#   scripts/fetch_corpus.sh file /path/teks.txt  # pakai file lokal
set -euo pipefail

out_dir="${ASCA_DATA_DIR:-$HOME/.cache/asca}"
mkdir -p "$out_dir"

case "${1:-tinystories}" in
  tinystories|tinystories-valid)
    url="https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt"
    dest="$out_dir/TinyStories-valid.txt"
    ;;
  tinystories-train)
    url="https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt"
    dest="$out_dir/TinyStories-train.txt"
    ;;
  file)
    if [ -z "${2:-}" ]; then echo "pemakaian: $0 file /path/teks.txt" >&2; exit 1; fi
    echo "$2"
    exit 0
    ;;
  *)
    echo "split tidak dikenal: $1 (pakai tinystories|tinystories-train|file)" >&2
    exit 1
    ;;
esac

if [ -f "$dest" ]; then
  echo "$dest"
  exit 0
fi

echo "mengunduh $url -> $dest" >&2
curl -fL --progress-bar "$url" -o "$dest.part"
mv "$dest.part" "$dest"
echo "$dest"
