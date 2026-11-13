# ASCA — target operasional. Seluruh pipeline sekarang Rust; tidak ada Python.
CARGO  ?= cargo
BIN    := rust/target/release/asca
DATA   ?= $(shell ./scripts/fetch_corpus.sh 2>/dev/null || echo "")
STEPS  ?= 400
SEEDS  ?= 0,1,2
LR     ?= 6e-3

.PHONY: help build test gpu train bench curve clean corpus

help:
	@echo "make build                    # cargo build --release"
	@echo "make test                     # 3 gradcheck: delta block, model, dense baseline"
	@echo "make gpu                      # paritas kernel CUDA + benchmark GEMM"
	@echo "make corpus                   # unduh TinyStories ke ~/.cache/asca"
	@echo "make train  STEPS=400         # latih ASCA (checkpoint ke /tmp/asca.ckpt)"
	@echo "make bench  STEPS=400         # ASCA vs dense di engine yang sama"
	@echo "make curve  STEPS=2000        # kurva loss vs token"

corpus:
	@./scripts/fetch_corpus.sh

build:
	cd rust && $(CARGO) build --release

test:
	cd rust && $(CARGO) test --release

gpu:
	$(BIN) gpu

train: build
	$(BIN) train --data "$(DATA)" --steps $(STEPS) --seq-len 128 --batch 4 \
		--lr $(LR) --warmup 40 --out /tmp/asca.ckpt

bench: build
	$(BIN) bench --data "$(DATA)" --steps $(STEPS) --seq-len 128 --batch 4 \
		--lr $(LR) --seeds $(SEEDS)

curve: build
	$(BIN) bench --data "$(DATA)" --steps $(STEPS) --seq-len 128 --batch 4 \
		--lr $(LR) --seeds $(SEEDS) --eval-curve 250 --match total

clean:
	cd rust && $(CARGO) clean
