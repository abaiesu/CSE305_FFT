# Compiler
CXX = g++

# Compiler flags
CXXFLAGS = -Wall -Wextra -std=c++11

# Include directories
INCLUDES = -I eigen

# Source files
SRCS = CS.cpp helpers.cpp fft.cpp perf_test.cpp demo.cpp

# Object files
OBJS = $(SRCS:.cpp=.o)

# Target executable
TARGET = CS

# Rule to link object files into the target executable
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $^ -o $@

# Rule to compile source files into object files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Clean rule
clean:
	rm -f $(OBJS) $(TARGET)

# Phony target to avoid conflicts with files named "clean"
.PHONY: clean