# Compiler
CXX = g++

# Compiler flags
CXXFLAGS = -Wall -Wextra -std=c++11

# Include directories
INCLUDES = -I eigen

# Source files
SRCS_CS_demo = CS.cpp helpers.cpp dft.cpp dct.cpp CS_demo.cpp
SRCS_perf_test = CS.cpp helpers.cpp dft.cpp perf_test.cpp dct.cpp 

# Object files
OBJS_CS_demo = $(SRCS_CS_demo:.cpp=.o)
OBJS_perf_test = $(SRCS_perf_test:.cpp=.o)

# Target executable
CS_demo = CS
perf_test = perf_test

# Rule to link object files into the target executable
$(CS_demo): $(OBJS_CS_demo)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $^ -o $@

$(perf_test): $(OBJS_perf_test)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $^ -o $@


# Rule to compile source files into object files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Clean rule
clean:
	rm -f $(OBJS_CS_demo) $(OBJS_perf_test) $(CS_demo) $(perf_test)

all: $(CS_demo) $(perf_test)