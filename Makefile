ORIGINALS = originals
ORIGINALS_OUT = $(ORIGINALS)/target
CARGO_OUT = target/release
CARGO_SRC = src/bin
TMPDIR = /tmp

.PHONY: mandelbrot

mandelbrot: mandelbrot-diff mandelbrot-bench

# Do not mark .PHONY because we want this to run each time
mandelbrot-diff: $(CARGO_OUT)/mandelbrot $(ORIGINALS_OUT)/mandelbrot.gcc-run
	@echo "Checking mandelbrot outputs for 200..."
	@$(ORIGINALS_OUT)/mandelbrot.gcc-run 200 > $(TMPDIR)/mandelbrot.gcc-run.bmp
	@$(CARGO_OUT)/mandelbrot 200 > $(TMPDIR)/mandelbrot.rs.bmp
	@cmp --silent $(TMPDIR)/mandelbrot.gcc-run.bmp $(TMPDIR)/mandelbrot.rs.bmp \
		|| { echo "Mandelbrot plots differ!"; exit 1; }

# Do not mark .PHONY because we want this to run each time
mandelbrot-bench: $(CARGO_OUT)/mandelbrot $(ORIGINALS_OUT)/mandelbrot.gcc-run
	@echo "Running mandelbrot benchmarks..."
	PARALLELIZE=true hyperfine \
		'$(CARGO_OUT)/mandelbrot 16000' \
		'$(ORIGINALS_OUT)/mandelbrot.gcc-run 16000'

# Do not mark .PHONY because we want this to run each time
mandelbrot-bench-large: $(CARGO_OUT)/mandelbrot $(ORIGINALS_OUT)/mandelbrot.gcc-run
	@echo "Running mandelbrot benchmarks..."
	PARALLELIZE=true hyperfine \
		'$(CARGO_OUT)/mandelbrot 64000' \
		'$(ORIGINALS_OUT)/mandelbrot.gcc-run 64000'

mandelbrot-par: $(CARGO_OUT)/mandelbrot
	PARALLELIZE=true RAYON_NUM_THREADS=2 RUST_BACKTRACE=1 \
		$(CARGO_OUT)/mandelbrot 16000 >/dev/null

# Do not mark .PHONY because we want this to run each time
mandelbrot-seq-bench: $(CARGO_OUT)/mandelbrot \
		$(ORIGINALS_OUT)/mandelbrot.gcc-seq-run
	@echo "Running mandelbrot sequential impl benchmarks..."
	hyperfine \
		'$(CARGO_OUT)/mandelbrot 16000' \
		'$(ORIGINALS_OUT)/mandelbrot.gcc-seq-run 16000'

$(CARGO_OUT)/mandelbrot: $(CARGO_SRC)/mandelbrot.rs
	cargo build --bin mandelbrot --release

.PHONY: build-all build-rust

build-all: $(ORIGINALS_OUT)/mandelbrot.gcc-run \
		$(ORIGINALS_OUT)/mandelbrot.gcc-seq-run \
		build-rust

$(ORIGINALS_OUT)/mandelbrot.gcc-run: ${ORIGINALS}/mandelbrot.c $(ORIGINALS_OUT)
	gcc -pipe -Wall -O3 -fomit-frame-pointer -march=core2 \
		-mno-fma -fno-finite-math-only -fopenmp $< \
		-o $@

$(ORIGINALS_OUT)/mandelbrot.gcc-seq-run: ${ORIGINALS}/mandelbrot.c \
		$(ORIGINALS_OUT)
	gcc -pipe -Wall -O3 -fomit-frame-pointer -march=core2 \
		-mno-fma -fno-finite-math-only $< \
		-o $@

$(ORIGINALS_OUT):
	mkdir -p $(ORIGINALS_OUT)

build-rust:
	cargo build --bins --release
