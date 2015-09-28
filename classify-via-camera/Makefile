CXX ?= /usr/bin/g++
CXXFLAGS := -g -Wall
LIBS := -lopencv_core -lopencv_highgui -lglog

caffetoy: clean
	$(CXX) $(CXXFLAGS) -c caffetoy.cc -o caffetoy.cc.o
	$(CXX) $(CXXFLAGS) caffetoy.cc.o $(LIBS) -o caffetoy

clean:
	-$(RM) *.o caffetoy Frame.jpg
