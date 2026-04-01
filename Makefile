PYTHON = python3

# Common Arguments
NPZ_FILE ?= ./matrices/cifar100_infl_matrix.npz
MEM_THRESHOLD ?= 0.8
NUM_BINS ?= 10

# Script-specific Arguments
FLATTEN_THRESHOLD ?= 0.90
EPOCH ?= 100
EPOCH_START ?= 15
EPOCH_END ?= 100

.PHONY: all flattening multiples quartiles ratios plots help

all: flattening multiples quartiles ratios plots

flattening:
	$(PYTHON) analyze_flattening.py \
		--flatten_threshold $(FLATTEN_THRESHOLD) \
		--mem_threshold $(MEM_THRESHOLD) \
		--num_bins $(NUM_BINS) \
		--npz_file $(NPZ_FILE)

multiples:
	$(PYTHON) analyze_multiples.py \
		--epoch $(EPOCH) \
		--mem_threshold $(MEM_THRESHOLD) \
		--npz_file $(NPZ_FILE)

quartiles:
	$(PYTHON) analyze_quartiles.py \
		--mem_threshold $(MEM_THRESHOLD) \
		--num_bins $(NUM_BINS) \
		--npz_file $(NPZ_FILE)

ratios:
	$(PYTHON) analyze_ratios.py \
		--epoch_start $(EPOCH_START) \
		--epoch_end $(EPOCH_END) \
		--mem_threshold $(MEM_THRESHOLD) \
		--num_bins $(NUM_BINS) \
		--npz_file $(NPZ_FILE)

plots:
	$(PYTHON) plot_metrics.py

help:
	@echo "Available analysis targets:"
	@echo "  make all        - Run all analyses and plot metrics"
	@echo "  make flattening - Run analyze_flattening.py"
	@echo "  make multiples  - Run analyze_multiples.py"
	@echo "  make quartiles  - Run analyze_quartiles.py"
	@echo "  make ratios     - Run analyze_ratios.py"
	@echo "  make plots      - Run plot_metrics.py"
	@echo ""
	@echo "You can override default arguments by passing them to make, for example:"
	@echo "  make ratios EPOCH_START=10 EPOCH_END=50 NUM_BINS=5"
	@echo "  make flattening FLATTEN_THRESHOLD=0.95"
	@echo "  make multiples EPOCH=50"
