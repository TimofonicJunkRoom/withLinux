CXXFLAGS += -Wall -Wpedantic -std=c++11 -O2 -fPIE -pie -fstack-protector-strong -Wformat -Werror=format-security
CPPFLAGS += -Wdate-time -D_FORTIFY_SOURCE=2
LDFLAGS  += -Wl,-z,relro -Wl,-z,now
# Query system flags with e.g. dpkg-buildflags --get CFLAGS

main:
	@parallel 'printf " CXX %s\n" {} ; $(CXX) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS) {} -o {.}.elf' ::: *.cc
	# TODO: compile those cc sources without a main function into object files
	# with an common header.

test:
	@for TESTBIN in $(shell ls *.elf); do ./$$TESTBIN; done

clean:
	$(RM) *.elf

pdf:
	python3 z.genbook.py
	evince *.pdf

.PHONY: main test clean pdf
