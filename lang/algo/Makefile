CXXFLAGS += -Wall -Wpedantic -std=c++11 -O2 -fPIE -pie -fstack-protector-strong -Wformat -Werror=format-security
CPPFLAGS += -Wdate-time -D_FORTIFY_SOURCE=2
LDFLAGS  += -Wl,-z,relro -Wl,-z,now
# Query system flags with e.g. dpkg-buildflags --get CFLAGS

main:
	@parallel 'printf " CXX %s\n" {} ; $(CXX) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS) {} -o {.}.elf' ::: *.cc

clean:
	$(RM) *.elf
