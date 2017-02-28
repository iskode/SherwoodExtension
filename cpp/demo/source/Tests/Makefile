

#link a static lib:  g++   test.cpp -I libMvtnorm/  libMvtnorm/libMvtnorm.a -o exe

LDIR=../../../lib/
CDF=$(LDIR)libMvtnorm-master/libMvtnorm/
LIB=$(CDF)libMvtnorm.a
INCLUDE=-I$(LDIR) -I$(CDF)
EXEC=tests
all:
	g++ -std=c++11 $(INCLUDE) GaussianNDAggregatorsTests.cpp -L. $(LIB) -o $(EXEC)

run:
	./$(EXEC)

.PHONY: all run