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
	fprintf(stderr, "reading data\n");
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
	fprintf(stderr, "transmitting data\n");
	// now we sleep
	usleep(1000) ;

	// now we transmit the value
	
	bufferWriter[0] = bufferReader[0] ;

	// and now we sleep again
	
	usleep(1000) ;
}

#define REPS 10000

void cpu_func_write_data(void * buffers [], void * cl_arg) {
	// first, we recover all the pieces of data
 	fprintf(stderr, "writing data\n");	
	int * bufferReader = (int*) STARPU_VARIABLE_GET_PTR(buffers[0]) ; // the value to write
	int * areasForWriting = (int*) STARPU_VECTOR_GET_PTR(buffers[1]) ; // the array of the location for writing
	int * solutionArray = (int*) STARPU_VECTOR_GET_PTR(buffers[2]) ; // the array to write : coherency has to be ensured by hand

	int index = (int)((uintptr_t) cl_arg) ;
	
	for (int i = 0 ; i < REPS ; i++) {
		solutionArray[areasForWriting[index]] += (*bufferReader) ;
		solutionArray[areasForWriting[index]] -= (*bufferReader) ;
	}
	solutionArray[areasForWriting[index]] += (*bufferReader) ;
}

void cpu_empty_func(void * buffers [], void * cl_arg) {
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

struct starpu_codelet cl_task_writer_naive = {
	.cpu_funcs = {cpu_func_write_data},
	.nbuffers = 3,
	.modes = {STARPU_R, STARPU_R, STARPU_R} // in fact, the last R is a read-write mode, but we manage data coherency by hand
} ;

struct starpu_codelet cl_task_writer_barrier = {
	.cpu_funcs = {cpu_func_write_data},
	.nbuffers = 4,
	.modes = {STARPU_R, STARPU_R, STARPU_R, STARPU_R} // in fact, the third R is a read-write mode, but we manage data coherency by hand
		// also, the last R is only for having a barrier with other tasks
} ;

struct starpu_codelet cl_task_empty_barrier = {
	.cpu_funcs = {cpu_empty_func},
	.nbuffers = 2,
	.modes = {STARPU_RW, STARPU_RW}
} ;

void submissionNaive(starpu_data_handle_t arrayProblem_handle, starpu_data_handle_t areasProblem_handle, starpu_data_handle_t solution_handle,
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

		task->handles[0] = bufferTaskA_handle[t] ;
		task->handles[1] = bufferTaskB_handle[t] ;

		starpu_task_submit(task) ;

		task = starpu_task_create() ;
		task->cl = &cl_task_writer_naive ;
		task->cl_arg = (void*) (uintptr_t) t ;
		task->cl_arg_size = sizeof(int) ;
		task->synchronous = 0 ;

		task->handles[0] = bufferTaskB_handle[t] ;
		task->handles[1] = areasProblem_handle ;
		task->handles[2] = solution_handle ;
		starpu_task_submit(task) ;
	}
}


#include "sc_hypervisor.h"
static int * counter_tasks_by_color = NULL ;
static int * counter_tasks_ended_by_color = NULL ;
static int * counter_tasks_in_exec_by_color = NULL ;
static int * counter_task_unlocked_by_color = NULL ;
static int nb_cpus_total = 0 ;
static int * nb_cpus_by_color = NULL ;
static int nbCols = 0 ;
static unsigned * contexts ;

static starpu_pthread_mutex_t * mutexes ;

void coloration_resize_ctxs (unsigned *sched_ctxs, int nsched_ctxs, int *workers, int nworkers) {
	// Require explicit resizing
	fprintf(stderr, "resizeing\n");
	int one_with_cpus = -1;
	for (int c = 0 ; c < nbCols ; c++)
		if (nb_cpus_by_color[c])
			one_with_cpus = c ;
	// here we know that one_with_cpus is the one with the cpus
	
	// we search for a new color
	fprintf(stderr, "research a new color\n");
	int colorLeaving = -1 ;
	for (int c = 0 ; c < nbCols ; c++) {
		fprintf(stderr, "color %d, tot : %d ; end : %d, ex : %d, unl : %d\n", c, counter_tasks_by_color[c], counter_tasks_ended_by_color[c], counter_tasks_in_exec_by_color[c], counter_task_unlocked_by_color[c]) ; 
		if (counter_tasks_by_color[c] - counter_tasks_ended_by_color[c] > 0 ) {
			colorLeaving = c ;
		}
	}
	if (colorLeaving >= 0) {
		// there is again a color with tasks to execute
		// we give to it all the cpus
		// recover all the cpus
		int workerids[nb_cpus_total] ;
		starpu_worker_get_ids_by_type(STARPU_CPU_WORKER, workerids,nb_cpus_total) ;
		starpu_sched_ctx_add_workers(workerids,nb_cpus_total, contexts[colorLeaving]) ;
		starpu_sched_ctx_remove_workers(workerids, nb_cpus_total, contexts[one_with_cpus]) ;
	}
}

struct sc_hypervisor_policy coloration_policy =
{
	.handle_poped_task = NULL,
	.handle_pushed_task = NULL,
	.handle_idle_cycle = NULL,
	.handle_idle_end = NULL,
	.handle_post_exec_hook = NULL,
	.resize_ctxs = coloration_resize_ctxs,
	.size_ctxs = NULL,
	.custom = 1,
	.name = "coloration"
};

void func_prologue_task_writer(void * arg) {
	int color = (int) (uintptr_t) arg ;
	STARPU_PTHREAD_MUTEX_LOCK(mutexes + color) ;
	counter_tasks_in_exec_by_color[color] ++ ;
	counter_task_unlocked_by_color[color] -- ;
	STARPU_PTHREAD_MUTEX_UNLOCK(mutexes + color) ;	
}

void func_epilogue_task_writer(void * arg) {
	int color = (int) (uintptr_t) arg ;
	STARPU_PTHREAD_MUTEX_LOCK(mutexes + color) ;
	counter_tasks_in_exec_by_color[color] -- ;
	counter_tasks_ended_by_color[color] ++ ;
	int leave = counter_tasks_by_color[color] - counter_tasks_ended_by_color[color] ;
	STARPU_PTHREAD_MUTEX_UNLOCK(mutexes + color) ;
	if (!leave)
		sc_hypervisor_resize_ctxs(NULL, 0, NULL, 0) ;

}

void func_epilogue_task_transmitter(void * arg) {
	int color = (int) (uintptr_t) arg ;
        STARPU_PTHREAD_MUTEX_LOCK(mutexes + color) ;
	counter_task_unlocked_by_color[color] ++ ;
        STARPU_PTHREAD_MUTEX_UNLOCK(mutexes + color) ;
}


int submissionByContext(starpu_data_handle_t arrayProblem_handle, starpu_data_handle_t areasProblem_handle, starpu_data_handle_t solution_handle,
        		starpu_data_handle_t * bufferTaskA_handle, starpu_data_handle_t * bufferTaskB_handle, int nbChain, int * colors) {
	// first step is to invert colors, ie instead of having an array colors like colors[i] gave the colors of the task i ; having an array col_inverse like col_inverse[i] give the tasks of color i
	int maxCol = 0 ;
	for (int t = 0 ; t < nbChain ; t++)
		if (colors[t] > maxCol)
			maxCol = colors[t] ;
	nbCols = maxCol + 1 ; // we have in nbCols the number of colors
	// now we will try to have the array nbByCol with nbByCol[i] is the number of tasks of color i
	
	counter_tasks_by_color = calloc(nbCols, sizeof(int)) ;
	counter_tasks_ended_by_color = calloc(nbCols, sizeof(int)) ;
	counter_tasks_in_exec_by_color = calloc(nbCols, sizeof(int)) ;
	counter_task_unlocked_by_color = calloc(nbCols, sizeof(int)) ;
	
	nb_cpus_total = starpu_cpu_worker_get_count() ;
	nb_cpus_by_color = calloc(nbCols, sizeof(int)) ;
	mutexes = calloc(nbCols, sizeof(starpu_pthread_mutex_t)) ;

	for (int t = 0 ; t < nbChain ; t++) {
		counter_tasks_by_color[colors[t]] ++ ;
	}

	// now we make the array taskOfColor like taskOfColor[i] is the array with the indexes of each task of color i
	// the advantage of doing this is to divide the time of initializing and the time of submission :
	// with this, we have during the submission of the task only to see this array, and not computing it at the submission
	
	int ** taskOfColor = calloc(nbCols, sizeof(int*)) ;
	for (int c = 0 ; c < nbCols ; c++) {
		// we fulfill the index c of taskOfColor
		int curColIndex = 0 ;
		taskOfColor[c] = calloc(counter_tasks_by_color[c], sizeof(int)) ;
		for(int i = 0 ; i < nbChain ; i++) {
			if (colors[i] == c) {
				taskOfColor[c][curColIndex++] = i ;
			}
		}
	}
	// here all is initiate
	// first we push all the tasks A and B


	// The next step is to create one context by color
	contexts = calloc(nbCols + 1, sizeof(unsigned)) ; // the last context is the big one

	contexts[nbCols] = starpu_sched_ctx_create(NULL, -1, "ctx", STARPU_SCHED_CTX_POLICY_NAME, "lws", 0) ;
	for (int i = 0 ; i < nbCols ; i++) {
		char name[25] ;
		sprintf(name, "ctx%d", i) ;
		contexts[i] =  starpu_sched_ctx_create(NULL, -(i == 0)  , name, STARPU_SCHED_CTX_POLICY_NAME, "lws", 0) ;		
		STARPU_PTHREAD_MUTEX_INIT(mutexes + i, NULL) ;
	}
	// And also a big context for the uncolored tasks


	sc_hypervisor_init(&coloration_policy) ;
        for (int i = 0 ; i < nbCols + 1 ; i++) {
		sc_hypervisor_register_ctx(contexts[i], 0.0) ;
	}
	
	//sc_hypervisor_size_ctxs(contexts, nbCols+1, workerids, starpu_cpu_worker_get_count());
	
	for ( int t = 0 ; t < nbChain ; t++) {
		struct starpu_task * task = starpu_task_create() ;
		task->cl = &cl_task_reader ;
		task->cl_arg = (void*) (uintptr_t) t ;
		task->cl_arg_size = sizeof(int) ;
		task->synchronous = 0 ;

		task->handles[0] = arrayProblem_handle ;
		task->handles[1] = bufferTaskA_handle[t] ;

		starpu_task_submit_to_ctx(task, contexts[nbCols]) ;

		task = starpu_task_create() ;

		task->cl = &cl_task_transmitter ;
		task->synchronous = 0 ;

		task->handles[0] = bufferTaskA_handle[t] ;
		task->handles[1] = bufferTaskB_handle[t] ;
		task->callback_func = func_epilogue_task_transmitter ;
		task->callback_arg = (void*) (uintptr_t) colors[t] ;
		starpu_task_submit_to_ctx(task, contexts[nbCols]) ;

	}

	// and now we push the C tasks
	
	// color by color
	for (int c = 0 ; c < nbCols ; c++) {
		// we push all the tasks of color c
		// Don't forget to use a new context
		for (int t = 0 ; t < counter_tasks_by_color[c] ; t++) {
			struct starpu_task * task = starpu_task_create() ;
			task->cl = &cl_task_writer_naive ;
			task->cl_arg = (void*) (uintptr_t) taskOfColor[c][t] ;
			task->cl_arg_size = sizeof(int) ;
			task->synchronous = 0 ;

		        task->handles[0] = bufferTaskB_handle[taskOfColor[c][t]] ;
                	task->handles[1] = areasProblem_handle ;
                	task->handles[2] = solution_handle ;
			task->prologue_callback_func = func_prologue_task_writer ;
			task->callback_func = func_epilogue_task_writer ;
			task->prologue_callback_arg = task->callback_arg = (void*)(uintptr_t) c ;	
			starpu_task_submit_to_ctx(task, contexts[c]) ;
		}
		// now we can free things
		free(taskOfColor[c]) ;
	}
	free(taskOfColor);
	// all tasks are pushed
}

int submissionBarrier(starpu_data_handle_t arrayProblem_handle, starpu_data_handle_t areasProblem_handle, starpu_data_handle_t solution_handle,
        		starpu_data_handle_t * bufferTaskA_handle, starpu_data_handle_t * bufferTaskB_handle, int nbChain, int * colors) {
	// first step is to invert colors, ie instead of having an array colors like colors[i] gave the colors of the task i ; having an array col_inverse like col_inverse[i] give the tasks of color i
	int maxCol = 0 ;
	for (int t = 0 ; t < nbChain ; t++)
		if (colors[t] > maxCol)
			maxCol = colors[t] ;
	int nbCols = maxCol + 1 ; // we have in nbCols the number of colors
	// now we will try to have the array nbByCol with nbByCol[i] is the number of tasks of color i
	int * nbByCol = calloc(nbCols, sizeof(int)) ;
	for (int t = 0 ; t < nbChain ; nbByCol[colors[t]]++, t++) ;
	
	// now we make the array taskOfColor like taskOfColor[i] is the array with the indexes of each task of color i
	// the advantage of doing this is to divide the time of initializing and the time of submission :
	// with this, we have during the submission of the task only to see this array, and not computing it at the submission
	
	int ** taskOfColor = calloc(nbCols, sizeof(int*)) ;
	for (int c = 0 ; c < nbCols ; c++) {
		// we fulfill the index c of taskOfColor
		int curColIndex = 0 ;
		taskOfColor[c] = calloc(nbByCol[c], sizeof(int)) ;
		for(int i = 0 ; i < nbChain ; i++) {
			if (colors[i] == c) {
				taskOfColor[c][curColIndex++] = i ;
			}
		}	
	}
	// here all is initiate
	// first we push all the tasks A and B

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

		task->handles[0] = bufferTaskA_handle[t] ;
		task->handles[1] = bufferTaskB_handle[t] ;

		starpu_task_submit(task) ;
	}

	// and now we push the C tasks
	
	// color by color
	starpu_data_handle_t barrier_handle[nbCols] ;
	int vector_handle[nbCols] ;
	for (int c = 0 ; c < nbCols ; c++)
		starpu_variable_data_register(barrier_handle+c, STARPU_MAIN_RAM, (uintptr_t) vector_handle + c, sizeof(int)) ;
	for (int c = 0 ; c < nbCols ; c++) {
		// we push all the tasks of color c
		for (int t = 0 ; t < nbByCol[c] ; t++) {
			struct starpu_task * task = starpu_task_create() ;
			task->cl = &cl_task_writer_barrier ;
			task->cl_arg = (void*) (uintptr_t) taskOfColor[c][t] ;
			task->cl_arg_size = sizeof(int) ;
			task->synchronous = 0 ;

		        task->handles[0] = bufferTaskB_handle[taskOfColor[c][t]] ;
                	task->handles[1] = areasProblem_handle ;
                	task->handles[2] = solution_handle ;
                	task->handles[3] = barrier_handle[c] ;
			
			starpu_task_submit(task) ;
		}
		// now we can free things
		free(taskOfColor[c]) ;
		// and we push the barrier task, if needed
		if (c < nbCols - 1) {
			struct starpu_task * task = starpu_task_create() ;
			task->synchronous = 0 ;
			task->cl = &cl_task_empty_barrier ;
			task->handles[0] = barrier_handle[c] ;
			task->handles[1] = barrier_handle[c+1] ;
			starpu_task_submit(task) ;
		}
	}
	free(taskOfColor) ;
	free(nbByCol) ;
	for (int c = 0 ; c < nbCols ; c++)
		starpu_data_unregister_submit(barrier_handle[c]) ;
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
	int maxC = 0 ;
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
		if (goodC > maxC)
			maxC = goodC ;
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
 	struct timespec start_t, end_t ;	
	clock_gettime(CLOCK_MONOTONIC, &start_t) ;
	if (mode == NAIVE) {
		fprintf(stderr, "Launching naive submission of tasks (should be uncorrect)\n");
		submissionNaive(arrayProblem_handle, areasProblem_handle, solution_handle, bufferTaskA_handle, bufferTaskB_handle, nbChain) ;
	} else if (mode == BARRIER) {
		fprintf(stderr, "Launching submission of tasks with barrier (should be correct)\n");
		submissionBarrier(arrayProblem_handle, areasProblem_handle, solution_handle, bufferTaskA_handle, bufferTaskB_handle, nbChain, colors) ;
	} else if (mode == COLORING_LOCK) {
		fprintf(stderr, "Launching submission of tasks by context, with one buffer (should be correct)\n");
		submissionByContext(arrayProblem_handle, areasProblem_handle, solution_handle, bufferTaskA_handle, bufferTaskB_handle, nbChain, colors) ;
	} else if (mode == COLORING_BUFFER) {
		fprintf(stderr, "TODO\n") ;
	}
	
	starpu_task_wait_for_all() ;
	clock_gettime(CLOCK_MONOTONIC, &end_t) ;
  	 fprintf(stderr, "We have %d colors\n", maxC + 1) ;
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
	int ret = 0 ;
	for (int a = 0 ; a < nbAreas ; a++) {
		fprintf(stderr, "%d - %d\n", realSolution[a], arraySolution[a]);
		if (realSolution[a] != arraySolution[a]) {
			fprintf(stderr, "error : the index %d is not correct : %d for real versus %d for parallel\n", a, realSolution[a], arraySolution[a]) ;
			ret = 1 ;	
		}
	}
	if (!ret)
		fprintf(stderr, "all good in %.3lf secs ... ending normally\n", end_t.tv_sec - start_t.tv_sec + ((end_t.tv_nsec - start_t.tv_nsec)/1000000000.0)) ;
	free(givenAreas) ;
	free(arraySolution) ;
	free(realSolution) ;
	free(colors) ;
	free(A) ;
	return ret ;
}


