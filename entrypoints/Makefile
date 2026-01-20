PYTHON ?= python
BLACK ?= $(PYTHON) -m black
PYTEST ?= $(PYTHON) -m pytest
PY_SRC := src scripts tests *.py

.PHONY: install-deps lint format test check

install-deps:
	$(PYTHON) -m pip install -r requirements.txt

lint:
	$(BLACK) --check $(PY_SRC)

format:
	$(BLACK) $(PY_SRC)

test:
	$(PYTEST)

check: lint test
