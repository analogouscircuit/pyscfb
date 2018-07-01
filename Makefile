all:
	python setup.py build_ext -if

clean:
	-rm -r build scfbutils.c scfbutils*.so
