CFLAGS = -g -ggdb -gdwarf-3 -gstrict-dwarf -O0 $(shell pkg-config --cflags starpu-1.3)
LDFLAGS = $(shell pkg-config --libs starpu-1.3)
CC = gcc

color_tests: color_tests.c
	$(CC) color_tests.c -o color_test $(CFLAGS) $(LDFLAGS) 

clean:
	rm color_test 
