CXX = /usr/bin/g++
#CXX = /usr/local/bin/clang++
CXXFLAGS = `pkg-config --cflags opencv` -O2 -march=native
LDFLAGS = `pkg-config --libs opencv`
TARGET = train_sample
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
	$(RM) $(TARGET) $(OBJS) $(DEPS) *~

depend:
	$(CXX) -MM $(SRCS)


