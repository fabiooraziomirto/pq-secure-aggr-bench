PYTHON ?= python

.PHONY: test test-real
test:
	$(PYTHON) -m pytest -q

test-real:
	$(PYTHON) -m pytest -q -rs -m requires_oqs --fail-on-oqs-skip
