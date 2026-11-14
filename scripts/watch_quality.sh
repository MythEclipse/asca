#!/usr/bin/env bash
# Pantau kualitas generasi sepanjang training, bukan hanya di akhir.
#
#   scripts/watch_quality.sh [CKPT] [INTERVAL_DETIK]
#
# Setiap kali checkpoint berubah, jalankan `asca genstat` dan catat hasilnya bersama nomor
# langkah. Val loss saja tidak cukup: model yang mulai mengulang kalimat tetap punya loss
# rendah, jadi distinct-n dan rep-4 ikut dicatat sebagai penjaga mode collapse.
set -uo pipefail

dir="${ASCA_DATA_DIR:-$HOME/.cache/asca}"
bin="${ASCA_BIN:-$(cd "$(dirname "$0")/.." && pwd)/rust/target/release/asca}"
ckpt="${1:-$dir/asca_full.bin}"
interval="${2:-600}"
out="$dir/logs/quality.log"
tok="$dir/tok12288.bin"

[ -x "$bin" ] || { echo "binary belum ada: $bin" >&2; exit 1; }
mkdir -p "$dir/logs"
[ -f "$out" ] || echo "# step  distinct-1  distinct-2  distinct-3  rep-4  vocab_unik  waktu" > "$out"

last=""
while true; do
  if [ -f "$ckpt" ]; then
    cur="$(stat -c %Y "$ckpt")"
    if [ "$cur" != "$last" ]; then
      step="$(grep -oE 'step +[0-9]+' "$dir/logs/train_full.log" 2>/dev/null | tail -1 | grep -oE '[0-9]+')"
      res="$("$bin" genstat --ckpt "$ckpt" --tokenizer "$tok" --samples 8 --tokens 120 2>/dev/null \
            | awk '/distinct-1/{d1=$3} /distinct-2/{d2=$3} /distinct-3/{d3=$3} /rep-4/{r4=$3} /vocab unik/{v=$4} END{print d1, d2, d3, r4, v}')"
      [ -n "${res// /}" ] && echo "${step:-?}  $res  $(date +%H:%M)" >> "$out"
      last="$cur"
    fi
  fi
  sleep "$interval"
done
