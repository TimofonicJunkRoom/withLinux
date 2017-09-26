CXXFLAGS += -Wall -Wpedantic -std=c++11 -O2 -fPIE -pie

main:
	@parallel 'printf " CXX %s\n" {} ; $(CXX) $(CXXFLAGS) {} -o {.}.elf' ::: *.cc

clean:
	$(RM) *.elf
