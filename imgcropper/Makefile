CXX ?= /usr/bin/g++
CXXFLAGS := -g -Wall
INCLUDE := -I/usr/include/ImageMagick-6/ -I/usr/include/ImageMagick-6/Magic++/ -I/usr/include/ImageMagick-6/magick/ -I/usr/include/x86_64-linux-gnu/ImageMagick-6/
LIBS := -lglog -lMagick++-6.Q16
CONFIG := `pkg-config pkg-config --cflags --libs Magick++`

caffetoy: clean
	$(CXX) $(CXXFLAGS) $(LIBS) $(CONFIG) $(INCLUDE) -o imgcropper main.cc

clean:
	-$(RM) *.o imgcropper
