.PHONY: all clean test

all:
	python setup.py build_ext --inplace

clean:
	find . -name "*.so" -o -name "*.pyc" -o -name "*.pyx.md5" | xargs rm -f
	rm -rf build dist permute.egg-info
	rm -rf .ipynb_checkpoints .coverage .cache

test:
	nosetests permute -A 'not slow' --ignore-files=^_test -v -s

test-all:
	nosetests permute --ignore-files=^_test -v -s

doctest:
	nosetests permute --ignore-files=^_test -v -s --with-doctest --ignore-files=^\. --ignore-files=^setup\.py$$ --ignore-files=test

coverage:
	nosetests permute --with-coverage --cover-package=permute --ignore-files=^_test  -v -s
