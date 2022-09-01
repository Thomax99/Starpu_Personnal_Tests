// This test is a test of StarPU when coloring task
// The idea is to have chain of size 3 of tasks
// for a chain a-b-c a has to be executed before b which has to be executed before c
// Moreover, for each task c of a chain, we give a color to c. Two tasks of the same color can be executed at the same time, but two tasks of different color cannot be executed at the same time
//
// With more details, we have an array of size arg0. We have also arg1 chain of tasks.
// We give randomly at each chain of task an area (an index of the array) for writing.
// Then, we color each chain of task ; and the idea of the coloration is that two tasks have the same color if they could be perform at the same time, ie if they write on different area.
//
// At the initialization, we have an array A randomly initiate of size ntasks*15.
// We give at each chain a-b-c an index i
// a is going to calculate the sum of A[15*i:15*(i+1)] and write it on a buffer
// b is going to read on the buffer and write it on an other buffer
// c is going to read the other buffer and write it on it given area.


#include <starpu.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>





int main(int argc, char * argv[]) {
	if (argc < 3) {
		fprintf(stderr, "We need 2 arguments : ./a.out sizeOfArray numberOfChainOfTasks\n");
	}
	int sizeOfArray = atoi(argv[1]), nbChain = atoi(argv[2]) ;

	int * array = calloc(sizeOfArray, sizeof(int)) ;

	int * givenAreas = calloc(nbChain, sizeof(int)) ;
	
	srand(time(NULL)) ;

	for (int i = 0 ; i < nbChain ; i++) {
		givenAreas[i] = rand() % sizeOfArray ;
	}

	// now we need to make the coloration
	// For this, this is simple ; we start by making an adjacency matrix
	// It means a matrix M of size nbChain * nbChain ; and M[i][j] = (givenAreas[i] == givenAreas[j]) *
	
	int * M = calloc(nbChain * nbChain, sizeof(int)) ;
	for (int i = 0 ; i < nbChain ; i++)
	for (int j = 0 ; j < nbChain ; j++) {
		M[i*nbChain + j] = (givenAreas[i] == givenAreas[j]) ;
	}
	
	// now we can colorate
	int * colors = calloc(nbChain, sizeof(int)) ;
	for (int t = 0 ; t < nbChain ; t++) {
		// we color task t
		int * usedColors = calloc(nbChain, sizeof(int)) ; // we can color in at most nbChain colors all the tasks
		for (int at = 0 ; at < t ; at ++ ){
			if (M[t*nbChain +at]) {
				usedColors[colors[at]] = 1 ;
			}
		}
		int goodC ;
		for (goodC = 0 ; usedColors[goodC] ; goodC++) ;
		colors[t] = goodC ;
		free(usedColors) ;
	}
	
	// here we have given a color for each task
	
	// now we create the array of reading
	
	int * A = calloc(nbChain*16, sizeof(int)) ;
	
	for (int i = 0 ; i < nbChain * 16 ; i++) {
		A[i] = rand() % 25 ;
	}

	
		
}


