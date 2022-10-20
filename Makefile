.DEFAULT_GOAL := help

.PHONY: help
help: ## Generate list of targets with descriptions
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	| sed -n 's/^\(.*\): \(.*\)##\(.*\)/\1~\3/p' \
	| column -t -s "~"

.PHONY: install
install: build ## Build and install meanshift to current Python environment
	cd wheels && pip install --force-reinstall -U meanshift_rs-*.whl && cd ..

.PHONY: build
build: ## Build s2gpp Python package with RUSTFLAGS
	@pip install -r requirements.txt
	bash ./tasks.sh release-build
