# Makefile for Sphinx documentation

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = python -m sphinx
SOURCEDIR     = source
BUILDDIR      = build

.PHONY: help clean html

help:
	@echo "Please use 'make <target>' where <target> is one of"
	@echo "  clean            to clear all built documentation files"
	@echo "  html             to make all standalone HTML files"

clean:
	-rm -rf $(BUILDDIR)/*

html: clean
	( THEME=sphinx_rtd_theme sphinx-multiversion $(SOURCEDIR) $(BUILDDIR)/html )
	( THEME=pydata_sphinx_theme sphinx-multiversion $(SOURCEDIR) $(BUILDDIR)/html )
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."
