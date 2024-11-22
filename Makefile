TARGET = pollardsrho

CXX = g++
CXXFLAGS = -O2 -Wall -std=c++17

INCLUDES = -I/usr/local/include -I/usr/local/include/boost
LIBS = /usr/local/lib/libsecp256k1.a -latomic -lboost_system -lboost_filesystem

SRC = pollardsrho.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(SRC) -o $(TARGET) $(LIBS)

clean:
	rm -f $(TARGET)
