CXX = g++ -Wall -std=c++0x
#CXXFLAGS = -g -Wall `pkg-config --cflags opencv`
CXXFLAGS = `pkg-config --cflags opencv`
#LDFLAGS = -L/usr/local/opencv-2.4.5/lib -lopencv_world -lboost_filesystem -lboost_filesystem-mt
LDFLAGS = -L/usr/local/opencv-2.4.5/lib -lopencv_world -lboost_filesystem
TARGET = main
SOURCE = $(TARGET).cpp feature.cpp

TARGET:
	$(CXX) -o $(TARGET) $(SOURCE) $(CXXFLAGS) $(LDFLAGS)

.PHONY: clean
clean:
	rm -f *.o *~ $(TARGET)
