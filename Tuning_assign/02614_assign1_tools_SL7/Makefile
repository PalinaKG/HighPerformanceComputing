TARGET	= libmatmult.so
LIBSRCS	= matmult_fun.c 
LIBOBJS	= $(LIBSRCS:.c=.o)

OPT	= -g 
PIC	= -fPIC	

CC	= gcc
CFLAGS= $(OPT) $(PIC) $(XOPTS)

SOFLAGS = -shared 
XLIBS	= -L/usr/lib64/atlas -lsatlas

$(TARGET): $(LIBOBJS)
	$(CC) -o $@ $(SOFLAGS) $(LIBOBJS) $(XLIBS)

clean:
	@/bin/rm -f core core.* $(LIBOBJS) 
