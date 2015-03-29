.PHONY: all clean test

all:
	python setup.py build_ext --inplace

clean:
	find . -name "*.so" -o -name "*.pyc" -o -name "*.pyx.md5" | xargs rm -f

test:
	python -c "import permute, sys, io; sys.exit(permute.test_verbose())"

doctest:
	python -c "import permute, sys, io; sys.exit(permute.doctest_verbose())"

coverage:
	nosetests permute --with-coverage --cover-package=permute
