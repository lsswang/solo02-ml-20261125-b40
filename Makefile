.PHONY: help generate test demo all reproduce

help:
	@echo "make generate  Generate examples/*.json"
	@echo "make test      Run standard-library self-tests"
	@echo "make all       Learn/plan/evaluate the canonical tray"
	@echo "make demo      Run the persistent minimal-tray demo"

generate:
	python3 run.py generate --out examples

test:
	python3 run.py selftest
	python3 -m unittest discover -s tests

all: generate
	python3 run.py all --input examples/sample_input.json --out deliverables/sample_input
	python3 run.py all --input examples/minimal_tray.json --out deliverables/minimal

demo:
	./scripts/demo_minimal.sh

reproduce:
	./scripts/reproduce_all.sh
