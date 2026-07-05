COMPETITION := playground-series-s6e7
DATA_DIR   := data/raw
TOKEN_FILE := .kaggle/access_token

.PHONY: all install download sim-download test lint format clean list

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

sim-download:
	@mkdir -p $(DATA_DIR); \
	$(MAKE) _ensure_kaggle_token; \
	TOKEN="$$(cat $(TOKEN_FILE) 2>/dev/null)"; \
	[ -z "$$TOKEN" ] && TOKEN="$$KAGGLE_API_TOKEN"; \
	echo "Downloading $(COMPETITION) sample data..."; \
	KAGGLE_API_TOKEN="$$TOKEN" uv run kaggle competitions download \
		-c $(COMPETITION) -f train.csv -p $(DATA_DIR) && \
	KAGGLE_API_TOKEN="$$TOKEN" uv run kaggle competitions download \
		-c $(COMPETITION) -f test.csv -p $(DATA_DIR) 2>&1 || { \
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
	echo "  Sample data ready in $(DATA_DIR)/"

test:
	@uv run pytest tests/ -v $(ARGS)

lint:
	@uv run ruff check src/ scripts/ tests/

format:
	@uv run ruff format src/ scripts/ tests/ --check

format-fix:
	@uv run ruff format src/ scripts/ tests/

list:
	@uv run python scripts/list_stats.py

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
elif [ -f ~/.kaggle/kaggle.json ] && ! grep -q "your-kaggle-username" ~/.kaggle/kaggle.json 2>/dev/null; then \
		echo "Using legacy credentials from ~/.kaggle/kaggle.json"; \
		exit 0; \
	else \
		if [ -n "$$(ls ~/.kaggle/acc* 2>/dev/null)" ] && ls ~/.kaggle/acc* | xargs grep -L "^$$PLACEHOLDER" 2>/dev/null | head -1 > $(TOKEN_FILE) 2>/dev/null; then \
			if [ -s "$$(ls ~/.kaggle/acc* 2>/dev/null | grep -L "^$$PLACEHOLDER" 2>/dev/null | head -1)" ]; then \
				TOKEN=$$(cat $(TOKEN_FILE)); \
			else \
				echo "Illegal token found"; \
				touch $(TOKEN_FILE); \
				chmod 600 $(TOKEN_FILE); \
			fi; \
		else \
			if [ -f .kaggle/access_token.example ]; then \
				cp .kaggle/access_token.example $(TOKEN_FILE); \
			else \
				printf 'KGAT_your-kaggle-api-token-here\n' > $(TOKEN_FILE); \
			fi; \
			chmod 600 $(TOKEN_FILE); \
			fi; \
	fi

_ensure_kaggle_auth: _ensure_kaggle_token
	@TOKEN="$$(cat $(TOKEN_FILE) 2>/dev/null)"; \
	[ -z "$$TOKEN" ] && TOKEN="$$KAGGLE_API_TOKEN"; \
	echo "Verifying..." && \
	KAGGLE_API_TOKEN="$$TOKEN" kaggle competitions list >/dev/null 2>&1 && \
	echo "  Authenticated successfully." || \
	{ echo "  WARNING: Authentication check failed."; exit 1; }

_clean:
	rm -f .uv_sync
	rm -rf data/raw/*
