SPHINXOPTS ?=
SPHINXBUILD ?= sphinx-build
PYTHON ?= python3
SERVEPORT ?= 8000
SOURCEDIR = .
BUILDDIR = _build
# Zip of the built site, dropped in the repo root (one level above docs/).
ZIPFILE = $(CURDIR)/../waxmorph-docs.zip

.PHONY: html clean serve

html:
	$(SPHINXBUILD) -b html $(SPHINXOPTS) $(SOURCEDIR) $(BUILDDIR)/html
	rm -f "$(ZIPFILE)"
	cd $(BUILDDIR)/html && zip -r -q "$(ZIPFILE)" .
	@echo "Zipped build -> $(ZIPFILE)"

# Serve the built site over HTTP — required for the interactive gallery, whose
# ES modules and data files are blocked by browsers on file://.
serve:
	@echo "Serving docs at http://localhost:$(SERVEPORT)/  (Ctrl-C to stop)"
	cd $(BUILDDIR)/html && $(PYTHON) -m http.server $(SERVEPORT)

clean:
	rm -rf $(BUILDDIR) generated
	rm -f "$(ZIPFILE)"
