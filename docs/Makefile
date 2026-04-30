SPHINXOPTS ?=
SOURCEDIR = .
BUILDDIR = _build

.PHONY: html clean

html:
	python -m sphinx -b html $(SPHINXOPTS) $(SOURCEDIR) $(BUILDDIR)/html

clean:
	rm -rf $(BUILDDIR)
