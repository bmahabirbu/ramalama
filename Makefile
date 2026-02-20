MAKEFLAGS += -j2
OS := $(shell uname;)
SELINUXOPT ?= $(shell test -x /usr/sbin/selinuxenabled && selinuxenabled && echo -Z)
PREFIX ?= /usr/local
BINDIR ?= ${PREFIX}/bin
SHAREDIR ?= ${PREFIX}/share
PYTHON ?= $(shell command -v python3 python|head -n1)
DESTDIR ?= /
PATH := $(PATH):$(HOME)/.local/bin
MYPIP ?= pip
IMAGE ?= ramalama
PROJECT_DIR ?= $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
EXCLUDE_DIRS ?= .venv venv .tox build
EXCLUDE_OPTS ?= $(addprefix --exclude-dir=,$(EXCLUDE_DIRS))
PYTHON_SCRIPTS ?= $(shell grep -lEr "^\#\!\s*/usr/bin/(env +)?python(3)?(\s|$$)" $(EXCLUDE_OPTS) $(PROJECT_DIR) || true)
RUFF_TARGETS ?= ramalama scripts bin/ramalama

default: help

help:
	@echo "Build Container Image"
	@echo
	@echo "  - make build"
	@echo "  - make build IMAGE=ramalama"
	@echo "  - make multi-arch"
	@echo "  - make multi-arch IMAGE=ramalama"
	@echo "  Build using build cache, for development only"
	@echo "  - make build IMAGE=ramalama CACHE=-C"
	@echo
	@echo "Build docs"
	@echo
	@echo "  - make docs"
	@echo
	@echo "Install ramalama"
	@echo
	@echo "  - make install"
	@echo
	@echo "Clean the repository"
	@echo
	@echo "  - make clean"
	@echo

.PHONY: install-uv
install-uv:
	./install-uv.sh

.PHONY: install-requirements
install-requirements:
	${MYPIP} install ".[dev]"

.PHONY: install-completions
install-completions: completions
	install ${SELINUXOPT} -d -m 755 $(DESTDIR)${SHAREDIR}/bash-completion/completions
	install ${SELINUXOPT} -m 644 completions/bash-completion/completions/ramalama \
		$(DESTDIR)${SHAREDIR}/bash-completion/completions/ramalama
	install ${SELINUXOPT} -d -m 755 $(DESTDIR)${SHAREDIR}/fish/vendor_completions.d
	install ${SELINUXOPT} -m 644 completions/fish/vendor_completions.d/ramalama.fish \
		$(DESTDIR)${SHAREDIR}/fish/vendor_completions.d/ramalama.fish
	install ${SELINUXOPT} -d -m 755 $(DESTDIR)${SHAREDIR}/zsh/site-functions
	install ${SELINUXOPT} -m 644 completions/zsh/site-functions/_ramalama \
		$(DESTDIR)${SHAREDIR}/zsh/site-functions/_ramalama

.PHONY: install-shortnames
install-shortnames:
	install ${SELINUXOPT} -d -m 755 $(DESTDIR)$(SHAREDIR)/ramalama
	install ${SELINUXOPT} -m 644 shortnames/shortnames.conf \
		$(DESTDIR)$(SHAREDIR)/ramalama

.PHONY: completions
completions:
	mkdir -p completions/bash-completion/completions
	register-python-argcomplete --shell bash ramalama > completions/bash-completion/completions/ramalama

	mkdir -p completions/fish/vendor_completions.d
	register-python-argcomplete --shell fish ramalama > completions/fish/vendor_completions.d/ramalama.fish

	mkdir -p completions/zsh/site-functions
	-register-python-argcomplete --shell zsh ramalama > completions/zsh/site-functions/_ramalama

.PHONY: install
install: docs completions
	RAMALAMA_VERSION=$(RAMALAMA_VERSION) \
	${MYPIP} install . --no-deps --root $(DESTDIR) --prefix ${PREFIX}

.PHONY: build
build:
	./container_build.sh ${CACHE} build $(IMAGE) -v "$(VERSION)"

.PHONY: build-rm
build-rm:
	./container_build.sh ${CACHE} -r build $(IMAGE) -v "$(VERSION)"

.PHONY: build_multi_arch
build_multi_arch:
	./container_build.sh ${CACHE} multi-arch $(IMAGE) -v "$(VERSION)"

.PHONY: install-docs
install-docs: docs
	make -C docs install

.PHONY: docs docs-manpages docsite-docs
docs: docs-manpages docsite-docs

docs-manpages:
	$(MAKE) -C docs

docsite-docs:
	$(MAKE) -C docsite convert

.PHONY: lint
lint:
	! git grep -n -- '#!/usr/bin/python3' -- ':!Makefile'
	ruff check $(RUFF_TARGETS)
	shellcheck *.sh */*.sh */*/*.sh

.PHONY: check-format
check-format:
	ruff format --check $(RUFF_TARGETS)

.PHONY: format
format:
	ruff format $(RUFF_TARGETS)
	ruff check --fix $(RUFF_TARGETS)

.PHONY: codespell
codespell:
	codespell $(PROJECT_DIR) $(PYTHON_SCRIPTS)

.PHONY: man-check
man-check:
ifeq ($(OS),Linux)
	hack/man-page-checker
	hack/xref-helpmsgs-manpages
endif

.PHONY: type-check
type-check:
	mypy $(addprefix --exclude=,$(EXCLUDE_DIRS)) --exclude test $(PROJECT_DIR)

.PHONY: validate
validate: codespell lint man-check type-check

.PHONY: pypi-build
pypi-build:   clean
	make docs
	python3 -m build --sdist
	python3 -m build --wheel

.PHONY: pypi
pypi: pypi-build
	python3 -m twine upload dist/*

.PHONY: clean
clean:
	make -C docs clean
	make -C docsite clean clean-generated
	find . -depth -print0 | git check-ignore --stdin -z | xargs -0 rm -rf

