# CFLAGS = -g -ggdb -gdwarf-3 -gstrict-dwarf -O0 $(shell pkg-config --cflags starpu-1.3)
CFLAGS = -O3 $(shell pkg-config --cflags starpu-1.4)
LDFLAGS =  $(shell pkg-config --libs starpu-1.4)
CC = gcc

color_tests: color_tests.c
	$(CC) color_tests.c -o color_test $(CFLAGS) $(LDFLAGS) 

merge_sort_cu.o: bubble_merge_sort.cu
	nvcc -I$(STARPU_INCLUDE) -I/usr/local/include -lcublas  -DSTARPU -Isrc $< -c -o merge_sort_cu.o

merge_sort: bubble_merge_sort.c merge_sort_cu.o
	$(CC) -lcublas -lstdc++ -lcudart bubble_merge_sort.c merge_sort_cu.o -o merge_sort $(CFLAGS) $(LDFLAGS) -lcublas

vector: vector.c
	$(CC) vector.c -o vector $(CFLAGS) $(LDFLAGS) 

vector_mpi: vector_mpi.c
	mpicc vector_mpi.c -o vector_mpi -O0 -g -ggdb -gdwarf-3 -gstrict-dwarf $(shell pkg-config --cflags starpumpi-1.4) $(shell pkg-config --libs starpumpi-1.4)

vector_mpi_auto_req: vector_mpi_auto_req.c
	mpicc vector_mpi_auto_req.c -o vector_mpi_auto_req -O0 -g -ggdb -gdwarf-3 -gstrict-dwarf $(shell pkg-config --cflags starpumpi-1.4) $(shell pkg-config --libs starpumpi-1.4)



clean:
	rm color_test merge_sort vector 
