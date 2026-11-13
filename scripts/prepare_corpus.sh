#!/usr/bin/env bash
# Siapkan korpus latih dari file besar: ambil N MB pertama, latih tokenizer BPE di atasnya.
#
# Pemakaian:
#   scripts/prepare_corpus.sh [MB] [VOCAB]
#   scripts/prepare_corpus.sh 100 12288
#
# Kenapa hanya sebagian: BPE dilatih pada statistik kata, dan 100 MB sudah memberi jutaan
# kata unik - cukup untuk vocab yang stabil. Melatih BPE pada 1,9 GB hanya menambah waktu
# tanpa mengubah kosakata secara berarti. Data latih model tetap bisa memakai file utuh.
set -euo pipefail

mb="${1:-100}"
vocab="${2:-12288}"
dir="${ASCA_DATA_DIR:-$HOME/.cache/asca}"
src="$dir/TinyStories-train.txt"
out="$dir/corpus-${mb}m.txt"
tok="$dir/tok${vocab}.bin"

if [ ! -f "$src" ]; then
  echo "korpus sumber tidak ada: $src (jalankan scripts/fetch_corpus.sh tinystories-train)" >&2
  exit 1
fi

bytes=$(( mb * 1024 * 1024 ))
if [ ! -f "$out" ] || [ "$(stat -c%s "$out")" -ne "$bytes" ]; then
  head -c "$bytes" "$src" > "$out"
fi
echo "korpus: $out ($(stat -c%s "$out") byte)"

bin="${ASCA_BIN:-./rust/target/release/asca}"
"$bin" bpe train --data "$out" --vocab "$vocab" --out "$tok"
echo "tokenizer: $tok"
