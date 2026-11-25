.PHONY: demo test clean

demo:
	bash run_demo.sh

test:
	python3 -m unittest discover -s tests -v

clean:
	rm -rf out/demo
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
