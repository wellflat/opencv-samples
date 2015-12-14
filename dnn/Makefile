CXX = g++
CXXFLAGS = `pkg-config --cflags opencv` -std=c++11
LDFLAGS = `pkg-config --libs opencv`
TARGET = $@
SOURCE = $(TARGET).cpp

TARGET:
	$(CXX) -o $(TARGET) $(SOURCE) $(CXXFLAGS) $(LDFLAGS)

.PHONY: clean
clean:
	rm -rf *.o *~


