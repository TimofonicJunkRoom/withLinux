CXXFLAGS += -Wall -Wpedantic -std=c++11 -O2 -fPIE -pie

main:
	$(foreach ccfile,$(shell ls *.cc),\
		make $(patsubst %.cc,%,$(ccfile)))

# the *.cc file compilation rule is implicit
# TODO: add the clean target
