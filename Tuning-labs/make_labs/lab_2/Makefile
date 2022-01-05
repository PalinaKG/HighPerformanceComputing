CC = gcc
OPT = -g -O3
WARN = -Wall
CFLAGS = $(OPT) $(WARN)
OBJECTS = data.o main.o io.o

excer2.exe: $(OBJECTS)
	$(CC) $(CFLAGS) -o excer2.exe $(OBJECTS)
clean:
	@rm -f *.o core
realclean : clean
	@rm -f excer2.exe

data.o : data.c data.h
main.o : data.h io.h main.c
io.o   : io.h
