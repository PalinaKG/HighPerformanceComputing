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


# DO NOT DELETE

data.o: data.h
io.o: io.h
main.o: io.h data.h
