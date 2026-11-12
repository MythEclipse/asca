# ASCA — target operasional.
# Wajib memakai venv bersama (~/.venvs/ml); PYTHONPATH Hermes di-unset agar
# site-packages python3.14 milik Hermes tidak bocor ke interpreter ini.
PY      := env -u PYTHONPATH -u PYTHONHOME ~/.venvs/ml/bin/python
PRESET  ?= nano
STEPS   ?= 600
BATCH   ?= 8
SEQ     ?= 256
SEEDS   ?= 0,1,2
OUT     ?= docs/EXPERIMENTS.md

.PHONY: help test bench vram train ablations all clean

help:
	@echo "make test                      # 12 test: kausalitas, referensi naif, MoE, dll"
	@echo "make bench  PRESET=nano        # FLOPs/token & throughput"
	@echo "make vram   PRESET=nano        # peak VRAM 512..4096 & latensi decode"
	@echo "make train  PRESET=nano        # komparasi dense-matched vs ASCA (3 seed)"
	@echo "make ablations PRESET=nano     # A1..A6 vs A0"
	@echo "make all                       # test + bench + vram + train"

test:
	$(PY) tests/run_tests.py

bench:
	$(PY) benchmarks/bench_flops.py --preset $(PRESET) --out $(OUT)

decode:
	$(PY) benchmarks/bench_decode.py --preset $(PRESET) --contexts 512,1024,2048

vram:
	$(PY) benchmarks/bench_vram.py --preset $(PRESET) --out $(OUT)

train:
	$(PY) benchmarks/train_toy.py --preset $(PRESET) --steps $(STEPS) --batch $(BATCH) \
		--seq-len $(SEQ) --models dense-matched,asca --seeds $(SEEDS) --out $(OUT)

ablations:
	$(PY) benchmarks/train_toy.py --preset $(PRESET) --steps $(STEPS) --batch $(BATCH) \
		--seq-len $(SEQ) --models asca,A1,A2,A3,A4,A5,A6 --seeds 0 --out $(OUT)

all: test bench decode vram train

clean:
	rm -rf data/cache __pycache__ */__pycache__ .pytest_cache
