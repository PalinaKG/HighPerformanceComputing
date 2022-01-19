TARGET= jacobi
OBJS	= main.o jacobi.o alloc3d.o alloc3d_gpu.o transfer3d_gpu.o print.o

OPT	= -g -Ofast -funsafe-loop-optimizations
ISA	= 
PARA	= -fopenmp

XOPTS = -Xptxas=-v -arch=sm_80 -lineinfo

CC	= gcc

CCC	= g++
CXX	= nvcc
CXXFLAGS= --compiler-options "$(OPT) $(PARA)" $(XOPTS)

CFLAGS	= $(OPT) $(ISA) $(PARA) $(XOPT)

CUDA_PATH ?= /appl/cuda/11.5.1
INCLUDES = -I$(CUDA_PATH)/include -I$(CUDA_PATH)/samples/common/inc

all: $(TARGET)

$(TARGET): $(OBJS) 
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LIBS)

.SUFFIXES: .cu
.cu.o:
	$(CXX) -o $*.o -c $*.cu $(CXXFLAGS) $(SOFLAGS) $(INCLUDES)

clean:
	@/bin/rm -f *.o $(TARGET)

# DO NOT DELETE

main.o: main.cu jacobi.h alloc3d.h alloc3d_gpu.h transfer3d_gpu.h print.h
alloc3d.o : alloc3d.h
alloc3d_gpu.o : alloc3d_gpu.h 
print.o: print.h