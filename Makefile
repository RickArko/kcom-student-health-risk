COMPETITION := playground-series-s6e7
DATA_DIR   := data/raw
TOKEN_FILE := .kaggle/access_token

.PHONY: all install download test lint format clean list submit notebook-eda notebook-baseline kernel-baseline

all: install download test
	@echo ""
	@echo "========================================================"
	@echo "  Setup complete. Next steps:"
	@echo "    make test            # run tests"
	@echo "    make lint            # code linting"
	@echo "    make format          # code formatting (check)"
	@echo "========================================================"

install: .uv_sync
	uv pip install -e .
	@$(MAKE) _ensure_kaggle_auth
	@echo ""
	@echo "All set. Run 'make download' to fetch the competition data."

download: _ensure_kaggle_cli _ensure_kaggle_auth
	@mkdir -p $(DATA_DIR); \
	TOKEN="$$(cat $(TOKEN_FILE) 2>/dev/null)"; \
	[ -z "$$TOKEN" ] && TOKEN="$$KAGGLE_API_TOKEN"; \
	echo "Downloading $(COMPETITION) data..."; \
	KAGGLE_API_TOKEN="$$TOKEN" kaggle competitions download \
		-c $(COMPETITION) -p $(DATA_DIR) 2>&1 || { \
		exit_code=$$?; \
		echo ""; \
		echo "================================================================"; \
		echo " Download failed (403 Forbidden)."; \
		echo ""; \
		echo " Possible causes:"; \
		echo "   1. You haven't joined the competition yet."; \
		echo "      Go to the page and click 'Join' / 'Accept Rules':"; \
		echo "      https://www.kaggle.com/competitions/$(COMPETITION)"; \
		echo ""; \
		echo "   2. Your API token may be stale."; \
		echo "      Regenerate at https://www.kaggle.com/settings"; \
		echo "      then update $(TOKEN_FILE)"; \
		echo "================================================================"; \
		exit $$exit_code; \
	}; \
	echo "Extracting..."; \
	cd $(DATA_DIR) && unzip -o $(COMPETITION).zip && rm -f $(COMPETITION).zip; \
	echo "  Data ready in $(DATA_DIR)/"

test:
	@if [ -d tests ]; then uv run pytest tests/ -v $(ARGS); else echo "WARNING: tests/ directory not found — skipping tests."; fi

lint:
	@uv run ruff check src/ scripts/ $(shell [ -d tests ] && echo tests/)

format:
	@uv run ruff format src/ scripts/ $(shell [ -d tests ] && echo tests/) --check

format-fix:
	@uv run ruff format src/ scripts/ $(shell [ -d tests ] && echo tests/)

list:
	@uv run python scripts/list_stats.py

submit: _ensure_kaggle_cli _ensure_kaggle_auth
	@SUBMISSION="submissions/submission.csv"; \
	if [ ! -f "$$SUBMISSION" ]; then \
		echo "ERROR: $$SUBMISSION not found. Run prediction first."; \
		exit 1; \
	fi; \
	METRICS=""; \
	if [ -f experiments/latest_run/metrics.json ]; then \
		METRICS="$$(uv run python -c "import json; m=json.load(open('experiments/latest_run/metrics.json')); print(f'acc={m[\"accuracy\"]:.4f}_f1={m[\"f1\"]:.4f}')" 2>/dev/null)"; \
	fi; \
	MSG="$${METRICS:+$$METRICS }$$(date +%Y%m%d-%H%M)"; \
	echo "Submitting $$SUBMISSION to $(COMPETITION)..."; \
	KAGGLE_API_TOKEN="$$(cat $(TOKEN_FILE) 2>/dev/null)" kaggle competitions submit \
		-c $(COMPETITION) -f "$$SUBMISSION" -m "$$MSG"

_uv_sync: pyproject.toml uv.lock
	uv sync --extra dev
	@touch .uv_sync

.uv_sync: _uv_sync
	@echo "Dependencies installed."

_ensure_kaggle_cli:
	@which kaggle >/dev/null 2>&1 || { \
		echo "Installing Kaggle CLI via uv tool..."; \
		uv tool install kaggle; \
	}

_ensure_kaggle_token:
	@mkdir -p .kaggle; \
	PLACEHOLDER="KGAT_your-kaggle-api-token-here"; \
	TOKEN=""; \
	\
	if [ -n "$$KAGGLE_API_TOKEN" ]; then \
		TOKEN="$$KAGGLE_API_TOKEN"; \
		if [ ! -f $(TOKEN_FILE) ] || [ "$$(cat $(TOKEN_FILE))" != "$$TOKEN" ]; then \
			printf '%s' "$$TOKEN" > $(TOKEN_FILE); \
			chmod 600 $(TOKEN_FILE); \
		fi; \
	elif [ -f $(TOKEN_FILE) ] && [ -s $(TOKEN_FILE) ] && ! grep -q "$$PLACEHOLDER" $(TOKEN_FILE) 2>/dev/null; then \
		TOKEN=$$(cat $(TOKEN_FILE)); \
	elif [ -f ~/.kaggle/access_token ] && [ -s ~/.kaggle/access_token ] && ! grep -q "$$PLACEHOLDER" ~/.kaggle/access_token 2>/dev/null; then \
		cp ~/.kaggle/access_token $(TOKEN_FILE); \
		chmod 600 $(TOKEN_FILE); \
		echo "  Copied token from ~/.kaggle/access_token"; \
	elif [ -f ~/.kaggle/kaggle.json ] && ! grep -q "your-kaggle-username" ~/.kaggle/kaggle.json 2>/dev/null; then \
		echo "Using legacy credentials from ~/.kaggle/kaggle.json"; \
		exit 0; \
	else \
		if [ -f .kaggle/access_token.example ]; then \
			cp .kaggle/access_token.example $(TOKEN_FILE); \
		else \
			printf 'KGAT_your-kaggle-api-token-here\n' > $(TOKEN_FILE); \
		fi; \
		chmod 600 $(TOKEN_FILE); \
		echo ""; \
		echo " Token template written to $(TOKEN_FILE)."; \
		echo ""; \
		echo " To configure:"; \
		echo "   1. Go to https://www.kaggle.com/settings"; \
		echo "   2. Under API, click 'Create New Token'"; \
		echo "   3. Copy the token (starts with KGAT_)"; \
		echo "   4. Paste it into $(TOKEN_FILE) (and nothing else)"; \
		echo "   5. Run 'make download' again to verify"; \
		echo ""; \
		exit 1; \
	fi

_ensure_kaggle_auth: _ensure_kaggle_token
	@TOKEN="$$(cat $(TOKEN_FILE) 2>/dev/null)"; \
	[ -z "$$TOKEN" ] && TOKEN="$$KAGGLE_API_TOKEN"; \
	echo "Verifying..." && \
	KAGGLE_API_TOKEN="$$TOKEN" kaggle competitions list >/dev/null 2>&1 && \
	echo "  Authenticated successfully." || \
	{ echo "  WARNING: Authentication check failed."; exit 1; }

notebook-eda:
	@uv run python scripts/nbs/build_eda.py

notebook-baseline:
	@uv run python scripts/nbs/build_baseline.py

notebooks: notebook-eda notebook-baseline

kernel-baseline:
	@echo "Copy scripts/kernels/baseline.py into a Kaggle Notebook cell."
	@echo "Or run locally: uv run python scripts/kernels/baseline.py"
	@uv run python scripts/kernels/baseline.py 2>&1 | head -30 || true
	@echo "... (use CONFIG=config/baseline.yaml for config-driven training)"

_clean:
	rm -f .uv_sync
	rm -rf data/raw/*
	rm -rf notebooks/*.ipynb