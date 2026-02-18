# Minimal makefile for Sphinx documentation
# Default target "all" builds main docs + v0.2. Use "make html" for the same.
#

SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = build
V02_DIR       = _versions/v0.2
V02_BUILD     = $(V02_DIR)/build/html

# Default: build everything (main docs + v0.2).
all: html

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

apidoc:
	@sphinx-apidoc -T --force ../robocasa -o api

# Build v0.2 docs and copy into main build so v0_2.html redirect works (v0.2/index.html).
v0.2:
	@echo "Building v0.2 docs..."
	@$(MAKE) -C $(V02_DIR) html
	@mkdir -p $(BUILDDIR)/html
	@cp -R $(V02_BUILD)/. $(BUILDDIR)/html/v0.2/
	@echo "v0.2 docs copied to $(BUILDDIR)/html/v0.2/"

# Build main docs then v0.2 (must be explicit so catch-all does not override).
html:
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@$(MAKE) v0.2

.PHONY: all help Makefile apidoc v0.2 html

# Catch-all: other targets (e.g. clean, latex) go to Sphinx. "html" is handled above.
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
