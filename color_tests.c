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

#define FACTOR_TASK 16
// create the solution of a problem, by allocating an array and solve the problem
int * createSolutionOfProblem(int nbTasks, int * givenAreas, int nbAreas, int * arrayProblem) {
	int * solution = calloc(nbAreas, sizeof(int)) ;
	
	for (int i = 0 ; i < nbTasks ; i++) {
		// we consider the task i
		int areaToWrite = givenAreas[i] ;
		for (int j = 0 ; j < FACTOR_TASK ; j++) {
			solution[areaToWrite] += arrayProblem[FACTOR_TASK*i + j] ;
		}
	}
	return solution ;
}

void cpu_func_read_data(void * buffers[], void * cl_arg) {
	// first we recover the index of the task
	unsigned int index = (unsigned int) ((uintptr_t) cl_arg) ;
	// then the array of the problem (read-mode)
	int * arrayProblem = (int*) STARPU_VECTOR_GET_PTR(buffers[0]) ;
	// and finally the buffer for writing
	int * buffer = (int*) STARPU_VARIABLE_GET_PTR(buffers[1]) ;

	(*buffer) = 0 ;
	// now we can make the resolution
	for (int i = 0 ; i < FACTOR_TASK ; i++) {
		(*buffer) += arrayProblem[index*FACTOR_TASK+i] ;
	}
}

void cpu_func_transmit_data(void * buffers[], void * cl_arg) {
	// first we recover the buffers for reading / writing
	
	int * bufferReader = (int *) STARPU_VARIABLE_GET_PTR(buffers[0]) ;
	int * bufferWriter = (int *) STARPU_VARIABLE_GET_PTR(buffers[1]) ;

	// now we sleep
	usleep(1000) ;

	// now we transmit the value
	
	bufferWriter[0] = bufferReader[0] ;

	// and now we sleep again
	
	usleep(1000) ;
}

void cpu_func_write_data(void * buffers [], void * cl_arg) {
	// first, we recover all the pieces of data
	
	int * bufferReader = (int*) STARPU_VARIABLE_GET_PTR(buffers[0]) ; // the value to write
	int * areasForWriting = (int*) STARPU_VECTOR_GET_PTR(buffers[1]) ; // the array of the location for writing
	int * solutionArray = (int*) STARPU_VECTOR_GET_PTR(buffers[2]) ; // the array to write : coherency has to be ensured by hand

	int index = (int)((uintptr_t) cl_arg) ;

	solutionArray[areasForWriting[index]] += (*bufferReader) ;
}

struct starpu_codelet cl_task_reader = {
	.cpu_funcs = {cpu_func_read_data},
	.nbuffers = 2,
	.modes = {STARPU_R, STARPU_W}

} ;

struct starpu_codelet cl_task_transmitter = {
	.cpu_funcs = {cpu_func_transmit_data},
	.nbuffers = 2,
	.modes = {STARPU_R, STARPU_W}
} ;

struct starpu_codelet cl_task_writer = {
	.cpu_funcs = {cpu_func_write_data},
	.nbuffers = 3,
	.modes = {STARPU_R, STARPU_R, STARPU_R} // in fact, the last R is a read-write mode, but we manage data coherency by hand
} ;

int submissionNaive(starpu_data_handle_t arrayProblem_handle, starpu_data_handle_t areasProblem_handle, starpu_data_handle_t solution_handle,
        		starpu_data_handle_t * bufferTaskA_handle, starpu_data_handle_t * bufferTaskB_handle, int nbChain) {
	for (int t = 0 ; t < nbChain ; t++) {

		struct starpu_task * task = starpu_task_create() ;
		task->cl = &cl_task_reader ;
		task->cl_arg = (void*) (uintptr_t) t ;
		task->cl_arg_size = sizeof(int) ;
		task->synchronous = 0 ;

		task->handles[0] = arrayProblem_handle ;
		task->handles[1] = bufferTaskA_handle[t] ;

		starpu_task_submit(task) ;

		task = starpu_task_create() ;

		task->cl = &cl_task_transmitter ;
		task->synchronous = 0 ;

		task->synchronous = 0 ;

		task->handles[0] = bufferTaskA_handle[t] ;
		task->handles[1] = bufferTaskB_handle[t] ;

		starpu_task_submit(task) ;

		task = starpu_task_create() ;
		task->cl = &cl_task_writer ;
		task->cl_arg = (void*) (uintptr_t) t ;
		task->cl_arg_size = sizeof(int) ;
		task->synchronous = 0 ;

		task->handles[0] = bufferTaskB_handle[t] ;
		task->handles[1] = areasProblem_handle ;
		task->handles[2] = solution_handle ;
		starpu_task_submit(task) ;
	}



}

#define NAIVE 0
#define BARRIER 1
#define COLORING_LOCK 2
#define COLORING_BUFFER 3

int main(int argc, char * argv[]) {
	if (argc < 4) {
		fprintf(stderr, "We need 3 arguments : ./a.out nbAreas numberOfChainOfTasks submissionMode\nnbAreas is the number of different areas the tasks will write\nnumberOfChainOfTasks is the number of different tasks\nsubmissionMode is %d for naive, %d for barrier, %d for coloring with lock (todo), %d for bufferized coloring (todo)\n", NAIVE, BARRIER, COLORING_LOCK, COLORING_BUFFER);
		exit(1);
	}
	int nbAreas = atoi(argv[1]), nbChain = atoi(argv[2]), mode = atoi(argv[3]) ;

	int * arraySolution = calloc(nbAreas, sizeof(int)) ;

	int * givenAreas = calloc(nbChain, sizeof(int)) ;
	
	srand(time(NULL)) ;

	for (int i = 0 ; i < nbChain ; i++) {
		givenAreas[i] = rand() % nbAreas ;
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
	
	int * A = calloc(nbChain*FACTOR_TASK, sizeof(int)) ;
	
	for (int i = 0 ; i < nbChain * FACTOR_TASK ; i++) {
		A[i] = rand() % 25 ;
	}
	free (M) ;

	// now we can make the tasks
	
	starpu_init(NULL) ;

	starpu_data_handle_t arrayProblem_handle, areasProblem_handle, solution_handle  ;
	starpu_data_handle_t bufferTaskA_handle[nbChain], bufferTaskB_handle[nbChain] ;

	starpu_vector_data_register(&arrayProblem_handle, STARPU_MAIN_RAM, (uintptr_t) A, nbChain * FACTOR_TASK, sizeof(int)) ;
	starpu_vector_data_register(&areasProblem_handle, STARPU_MAIN_RAM, (uintptr_t) givenAreas, nbChain, sizeof(int)) ;
	starpu_vector_data_register(&solution_handle, STARPU_MAIN_RAM, (uintptr_t) arraySolution, nbAreas, sizeof(int)) ;
	
	for (int t = 0 ; t < nbChain ; t++) {
		starpu_variable_data_register(bufferTaskA_handle+t, -1, (uintptr_t) 0, sizeof(int)) ;
		starpu_variable_data_register(bufferTaskB_handle+t, -1, (uintptr_t) 0, sizeof(int)) ;
	}
	if (mode == NAIVE) {
		fprintf(stderr, "Launching naive submission of tasks (should be uncorrect)\n");
		submissionNaive(arrayProblem_handle, areasProblem_handle, solution_handle, bufferTaskA_handle, bufferTaskB_handle, nbChain) ;
	} else if (mode == BARRIER) {
		fprintf(stderr, "TODO\n") ;
	} else if (mode == COLORING_LOCK) {
		fprintf(stderr, "TODO\n") ;
	} else if (mode == COLORING_BUFFER) {
		fprintf(stderr, "TODO\n") ;
	}
	
	starpu_task_wait_for_all() ;

	starpu_data_unregister(arrayProblem_handle) ;
	starpu_data_unregister(areasProblem_handle) ;
	starpu_data_unregister(solution_handle) ;

	for(int i = 0 ; i < nbChain ; i++) {
		starpu_data_unregister(bufferTaskA_handle[i]) ;
		starpu_data_unregister(bufferTaskB_handle[i]) ;	
	}
	
	// now we verify the solution is ok

	starpu_shutdown() ;	
	
	fprintf(stderr, "computing done ... now verifying the solution\n") ;	
	int * realSolution = createSolutionOfProblem(nbChain, givenAreas, nbAreas, A) ;
	
	for (int a = 0 ; a < nbAreas ; a++) {
		fprintf(stderr, "%d - %d\n", realSolution[a], arraySolution[a]);
		if (realSolution[a] != arraySolution[a]) {
			fprintf(stderr, "error : the index %d is not correct : %d for real versus %d for parallel\n", a, realSolution[a], arraySolution[a]) ;
			return 1 ;	
		}
	}
	fprintf(stderr, "all good ... ending normally\n") ;
	return 0 ;
}


