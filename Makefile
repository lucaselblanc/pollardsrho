TARGET = pollardsrho

CXX = g++ -g
CXXFLAGS = -O3 -march=native -Wall -std=c++17 -fopenmp -Wno-deprecated-declarations

INCLUDES = -I/usr/local/include/boost
LIBS = -latomic -lgmp -lboost_system -lboost_filesystem

SRC = pollardsrho.cpp ec.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(SRC) -o $(TARGET) $(LIBS)

ec.o: ec.c ec.h
	$(CXX) $(CFLAGS) -c $< -o $@

pollardsrho.o: pollardsrho.cpp ec.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: %.c
	$(CXX) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET)
