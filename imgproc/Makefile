CXX = g++
#CXXFLAGS = `pkg-config --cflags opencv` -std=c++11 -O2
CXXFLAGS = `pkg-config --cflags opencv` -g
LDFLAGS = `pkg-config --libs opencv`
#TARGET = $@
TARGET = lsd_sample
SRCS = $(wildcard *.cpp)
OBJS := $(SRCS:.cpp=.o)
DEPS := $(SRCS:.cpp=.d)

.PHONY: all clean depend
all: $(TARGET)
-include $(DEPS)

$(TARGET): $(OBJS)
	$(CXX) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -o $@ -c -MMD $<

clean:
	$(RM) $(TARGET) $(OBJS) $(DEPS) *~ core*

depend:
	$(CXX) -MM $(SRCS)
