#!/usr/bin/env bash
# Latih model bahasa dari nol sampai bisa diajak bicara. Satu perintah, tanpa Python.
#
#   scripts/train_llm.sh                 # jalur lengkap dengan nilai bawaan
#   STEPS=100000 scripts/train_llm.sh    # latih lebih lama
#
# Tahapan:
#   1. unduh korpus          (dilewati kalau sudah ada)
#   2. latih tokenizer BPE   (dilewati kalau sudah ada)
#   3. encode korpus sekali  -> file .tok (supaya proses berikutnya tidak meng-encode ulang)
#   4. latih model           (bisa dilanjutkan: --resume, checkpoint berkala)
#   5. contohkan keluaran    (bukti model benar-benar menghasilkan bahasa)
set -euo pipefail

dir="${ASCA_DATA_DIR:-$HOME/.cache/asca}"
bin="${ASCA_BIN:-$(cd "$(dirname "$0")/.." && pwd)/rust/target/release/asca}"
vocab="${VOCAB:-12288}"
mb="${CORPUS_MB:-100}"
steps="${STEPS:-48828}"
batch="${BATCH:-8}"
seq="${SEQ:-256}"
lr="${LR:-3e-3}"

corpus="$dir/corpus-${mb}m.txt"
tok="$dir/tok${vocab}.bin"
tokfile="$dir/corpus-${mb}m.tok"
ckpt="${CKPT:-$dir/asca_llm.bin}"

mkdir -p "$dir/logs"
[ -x "$bin" ] || { echo "binary belum ada: $bin (jalankan: make build)" >&2; exit 1; }

if [ ! -f "$dir/TinyStories-train.txt" ]; then
  echo "== 1/5 mengunduh korpus"
  "$(dirname "$0")/fetch_corpus.sh" tinystories-train
fi

if [ ! -f "$tok" ]; then
  echo "== 2/5 melatih tokenizer BPE vocab $vocab"
  head -c $(( mb * 1024 * 1024 )) "$dir/TinyStories-train.txt" > "$corpus"
  "$bin" bpe train --data "$corpus" --vocab "$vocab" --out "$tok"
fi

if [ ! -f "$tokfile" ]; then
  echo "== 3/5 meng-encode korpus"
  "$bin" tokenize --data "$corpus" --tokenizer "$tok" --out "$tokfile"
fi

echo "== 4/5 melatih model ($steps langkah x $(( batch * seq )) token = $(( steps * batch * seq )) token)"
resume=()
[ -f "$ckpt" ] && resume=(--resume "$ckpt")
"$bin" train --tokenized "$tokfile" --steps "$steps" --seq-len "$seq" --batch "$batch" \
  --lr "$lr" --d-model "${DMODEL:-192}" --n-heads "${NHEADS:-6}" --n-layers "${NLAYERS:-6}" \
  --n-experts "${NEXPERTS:-8}" --d-ff "${DFF:-384}" --eval-every 1000 \
  "${resume[@]}" --out "$ckpt"

echo "== 5/5 contoh keluaran"
printf 'Once upon a time, there was a little girl named Lily\n/quit\n' | \
  "$bin" chat --ckpt "$ckpt" --tokenizer "$tok" --tokens 80
