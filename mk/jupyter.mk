# mk/jupyter.mk — Jupyter kernel venv + notebook gates.

# Jupyter kernel (venv in packages/jupyter/.venv)
# Apple-bundled /usr/bin/python3 is 3.9 (too old for our deps), so we
# prefer a managed 3.12+. Resolution order:
#   1. uv-managed python, if uv is installed (most projects on this
#      codebase already have it via the pyproject/uv.lock pattern)
#   2. system `python3` on $PATH — falls back loudly if it's < 3.12
#
# Deliberately not falling back to `nix build nixpkgs#python3`: that
# materialises python3 in the nix store on every build regardless of
# user config (same pattern as the removed mlx fallback above).
.PHONY: jupyter-install jupyter-lab test-e2e-jupyter \
        test-integration-jupyter-cellparser test-e2e-notebooks

UV_PYTHON := $(shell uv python find 2>/dev/null)
VENV_PYTHON := $(shell [ -x "$(UV_PYTHON)" ] && echo "$(UV_PYTHON)" || echo python3)
JUPYTER_VENV := packages/jupyter/.venv
JUPYTER_PIP := $(JUPYTER_VENV)/bin/pip
JUPYTER_PYTHON := $(JUPYTER_VENV)/bin/python3
JUPYTER_PYTEST := $(JUPYTER_VENV)/bin/pytest

# Linux: pip's pyzmq wheel (the jupyter_client ZMQ backend) dlopens
# libstdc++.so.6 via LD_LIBRARY_PATH, which the nix shell doesn't expose →
# "libstdc++.so.6: cannot open shared object file" on the kernel install /
# notebook execute (run 28318698677). IDRISML_CXX_LIB is exported by the flake's
# Linux shellHook (nix's own libstdc++); prepend it for the venv runtime calls
# only. Empty on macOS (var unset; dyld is used anyway) → recipes unchanged.
JUPYTER_LDPATH := $(if $(IDRISML_CXX_LIB),LD_LIBRARY_PATH=$(IDRISML_CXX_LIB):$$LD_LIBRARY_PATH ,)

$(JUPYTER_VENV)/bin/activate:
	$(VENV_PYTHON) -m venv $(JUPYTER_VENV)
	$(JUPYTER_PIP) install --upgrade pip setuptools >/dev/null

jupyter-install: backend check $(JUPYTER_VENV)/bin/activate
	$(JUPYTER_PIP) install -e packages/jupyter/.[dev]
	$(JUPYTER_LDPATH)$(JUPYTER_PYTHON) -m idris_ml_kernel.install

jupyter-lab: jupyter-install
	$(JUPYTER_PIP) install -q jupyterlab
	$(JUPYTER_LDPATH)$(JUPYTER_VENV)/bin/jupyter lab --notebook-dir=packages/jupyter/notebooks

# Jupyter kernel tests (requires backend + idris2). IDRIS_ML_BUILD_DIR
# pins the REPL wrapper to the per-set tree this make just built
# (repl.py falls back to newest-dylib discovery without it).
test-e2e-jupyter: backend check $(JUPYTER_VENV)/bin/activate
	$(JUPYTER_PIP) install -q -e packages/jupyter/.[dev]
	cd packages/jupyter && $(JUPYTER_LDPATH)IDRIS_ML_BUILD_DIR=$(CURDIR)/$(BUILD) ../../$(JUPYTER_PYTEST) tests/ -v

# Quick: just cell parser (no REPL, no backend needed)
test-integration-jupyter-cellparser: $(JUPYTER_VENV)/bin/activate
	$(JUPYTER_PIP) install -q -e packages/jupyter/.[dev]
	cd packages/jupyter && ../../$(JUPYTER_PYTEST) tests/test_cell_parser.py -v

# Run all notebooks headless to check for API breakage.
# install-notebook (→ install-core) installs the idris-ml + idris-ml-notebook
# packages into IDRIS2_PACKAGE_PATH ($(IDRIS2_LOCAL)/idris2-0.8.0) — exactly the
# pair the kernel's REPL requests (`-p idris-ml -p idris-ml-notebook`). Without
# it the kernel dies on a cold cache with "Can't find package idris-ml (any)"
# (run 28325498685); it only ever passed off a warm install cache.
test-e2e-notebooks: install-notebook jupyter-install
	@fail=0; \
	for nb in packages/jupyter/notebooks/tutorials/*.ipynb packages/jupyter/notebooks/models/*.ipynb; do \
		echo "--- $$nb ---"; \
		$(JUPYTER_LDPATH)$(JUPYTER_VENV)/bin/jupyter nbconvert --execute --to notebook \
			--ExecutePreprocessor.timeout=120 "$$nb" \
			--output /tmp/test_nb_out.ipynb 2>&1 || { echo "FAIL: $$nb"; fail=1; continue; }; \
		echo "ok"; \
	done; \
	rm -f /tmp/test_nb_out.ipynb; \
	[ $$fail -eq 0 ] && echo "All notebooks passed" || { echo "Some notebooks failed"; exit 1; }
