PYTHON ?= uv run --frozen python
DATA ?= archived_results/iconip2026
OUT ?= reproduced_artifacts/iconip2026
ARCHIVE ?=
RESULTS_URL ?=
SOURCE_ROOT ?=

.PHONY: help paper-data paper-verify paper-tables paper-plots paper-artifacts paper-reproduce paper-audit paper-check train-plan paper-bundle
help:
	@echo 'paper-reproduce: fetch verified results and generate all paper artifacts'
	@echo 'paper-artifacts: regenerate offline from existing results'
	@echo 'paper-check: verify archive and run numerical/control regressions'
	@echo 'train-plan: print released configurations (no training)'
paper-data:
	$(PYTHON) -m scripts.paper.archive fetch --data "$(DATA)" $(if $(ARCHIVE),--archive "$(ARCHIVE)") $(if $(RESULTS_URL),--url "$(RESULTS_URL)")
paper-verify:
	$(PYTHON) -m scripts.paper.archive verify --data "$(DATA)"
paper-tables:
	$(PYTHON) -m scripts.paper.reproduce --data "$(DATA)" --out "$(OUT)" --kind tables
paper-plots:
	$(PYTHON) -m scripts.paper.reproduce --data "$(DATA)" --out "$(OUT)" --kind figures
paper-artifacts:
	$(PYTHON) -m scripts.paper.reproduce --data "$(DATA)" --out "$(OUT)"
paper-reproduce: paper-data
	$(MAKE) paper-artifacts
paper-audit:
	$(PYTHON) -m scripts.paper.reproduce --data "$(DATA)" --out "$(OUT)" --kind audit
paper-check: paper-verify
	ICONIP_DATA="$(abspath $(DATA))" $(PYTHON) -m pytest -q
train-plan:
	$(PYTHON) -m scripts.paper.training
paper-bundle:
	@test -n "$(SOURCE_ROOT)" || (echo 'Set SOURCE_ROOT to the thesis archived_results directory'; exit 1)
	$(PYTHON) -m scripts.paper.archive pack --source-root "$(SOURCE_ROOT)" --archive dist/iconip2026-results-v1.zip
