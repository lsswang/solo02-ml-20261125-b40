.PHONY: demo test handcheck clean-demo

demo:
	bash scripts/run_demo.sh

test:
	python3 -m unittest discover -s tests -v

handcheck:
	python3 -m boulder_judge.cli handcheck --strict

fresh:
	bash scripts/rerun_fresh_seed.sh 777 /tmp/boulder_fresh_777

clean-demo:
	rm -rf cases/APP-2026-0001
