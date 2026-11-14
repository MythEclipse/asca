#!/usr/bin/env bash
# Sambung training ke jumlah token yang lebih besar TANPA melonjakkan learning rate.
#
#   scripts/continue_training.sh [TARGET_TOKEN]        # default 250.000.000
#
# Masalahnya: `lr_at` menghitung progres cosine dari `--steps`. Kalau run 100 juta token
# (48.828 langkah) disambung dengan `--steps 122070`, progres melompat balik ke 0,4 dan LR
# naik dari 3e-4 ke ~2,7e-3. Lonjakan 9x pada model yang sudah terlatih adalah cara klasik
# merusak bobot yang sudah bagus.
#
# Skrip ini menyelesaikannya dengan membalik jadwal: ia tahu LR di titik sambung HARUS sama
# dengan LR akhir run sebelumnya, lalu menghitung LR puncak baru yang membuat hal itu benar.
set -euo pipefail

dir="${ASCA_DATA_DIR:-$HOME/.cache/asca}"
bin="${ASCA_BIN:-$(cd "$(dirname "$0")/.." && pwd)/rust/target/release/asca}"
ckpt="${CKPT:-$dir/asca_full.bin}"
target="${1:-250000000}"
batch="${BATCH:-8}"
seq="${SEQ:-256}"
lr_final="${LR_FINAL:-3e-4}"   # LR di akhir run sebelumnya (lr 3e-3 x 0,1)
warmup="${WARMUP:-50}"

[ -x "$bin" ] || { echo "binary belum ada: $bin" >&2; exit 1; }
[ -f "$ckpt" ] || { echo "checkpoint belum ada: $ckpt" >&2; exit 1; }

info="$("$bin" ckptinfo --ckpt "$ckpt")"
step="$(awk '/^step /{print $2}' <<<"$info")"
total=$(( target / (batch * seq) ))

read -r lr at_resume < <(awk -v s="$step" -v t="$total" -v w="$warmup" -v lf="$lr_final" 'BEGIN{
  if (t <= w) { printf "%.3e %.3e\n", lf, lf; exit }
  p = (s - w) / (t - w); if (p > 1) p = 1;
  f = 0.1 + 0.45 * (1 + cos(3.141592653589793 * p));
  printf "%.3e %.3e\n", lf / f, lf
}')

cat <<EOF
sambung training
  checkpoint   : $ckpt (langkah $step)
  target       : $target token = $total langkah
  LR akhir lama: $lr_final
  LR puncak baru: $lr  ->  LR di titik sambung $at_resume (harus ~= $lr_final)

EOF

cmd=("$bin" train \
  --tokenized "$dir/corpus-full.tok" \
  --steps "$total" --seq-len "$seq" --batch "$batch" --lr "$lr" \
  --d-model 192 --n-heads 6 --n-layers 6 --n-experts 8 --d-ff 384 \
  --eval-every 1000 --resume "$ckpt" --out "$ckpt")

if [ "${DRY_RUN:-0}" = "1" ]; then
  printf 'perintah:\n  %s\n' "${cmd[*]}"
  exit 0
fi
exec "${cmd[@]}"
