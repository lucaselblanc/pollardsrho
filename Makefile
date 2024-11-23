TARGET = pollardsrho

CXX = g++
CXXFLAGS = -O3 -march=native -Wall -std=c++17 -fopenmp

INCLUDES = -I/usr/local/include/boost
LIBS = ./libsecp256k1.a -latomic -lboost_system -lboost_filesystem

SRC = pollardsrho.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(SRC) -o $(TARGET) $(LIBS)

clean:
	rm -f $(TARGET)
