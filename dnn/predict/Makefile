CXX = /usr/bin/g++
#CXX = /usr/local/bin/clang++
CXXFLAGS = `pkg-config --cflags opencv` -O2 -march=native
LDFLAGS = `pkg-config --libs opencv`
TARGET = predict_sample
SRCS = $(TARGET).cpp
OBJS := $(SRCS:.cpp=.o)

.PHONY: all clean
all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -o $@ -c $<

clean:
	$(RM) $(TARGET) $(OBJS) *~


