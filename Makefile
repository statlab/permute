.PHONY: all clean test

all:
	python setup.py build_ext --inplace

clean:
	find . -name "*.so" -o -name "*.pyc" -o -name "*.pyx.md5" | xargs rm -f
	rm -rf build dist permute.egg-info
	rm -rf .ipynb_checkpoints .coverage .cache

test:
	pytest --durations=10 --pyargs permute

doctest:
	pytest --doctest-modules --durations=10 --pyargs permute

test-all:
	pytest --runslow --doctest-modules --durations=10 --pyargs permute

coverage:
	pytest --cov=permute --runslow --doctest-modules --durations=10 --pyargs permute
