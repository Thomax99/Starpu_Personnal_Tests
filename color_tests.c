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

#define FACTOR_TASK 32
#define WAIT_IN_READ 10000
// #define VERBOSE

static inline void starpu_task_submit_verify(struct starpu_task * task) {
	if (starpu_task_submit(task) != 0)
		fprintf(stderr, "WARNING : a task is not able to be executed\n") ;
}

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
#ifdef VERBOSE
	fprintf(stderr, "reading data\n");
#endif
	usleep(WAIT_IN_READ) ;
	(*buffer) = 0 ;
	// now we can make the resolution
	for (int i = 0 ; i < FACTOR_TASK ; i++) {
		(*buffer) += arrayProblem[index*FACTOR_TASK+i] ;
	}
}

#define WAITING_TIME 10000
#define RANDOM_FACT 3
void cpu_func_transmit_data(void * buffers[], void * cl_arg) {
	// first we recover the buffers for reading / writing
	
	int * bufferReader = (int *) STARPU_VARIABLE_GET_PTR(buffers[0]) ;
	int * bufferWriter = (int *) STARPU_VARIABLE_GET_PTR(buffers[1]) ;
#ifdef VERBOSE
	fprintf(stderr, "transmitting data\n");
#endif
	// now we sleep
	usleep((rand() % RANDOM_FACT) * WAITING_TIME) ;

	// now we transmit the value
	
	bufferWriter[0] = bufferReader[0] ;

	// and now we sleep again
	
	usleep((rand() % RANDOM_FACT) * WAITING_TIME) ;
}

#define REPS 100000
#define WRITE_SLEEP REPS
static int ** buffersByColor = NULL ; // buffersByColor[i] corresponds to the buffer posseded by the color i 

void real_func_write_data(int * solutionArray, int * areasForWriting, int index, int value) {
        for (int i = 0 ; i < REPS ; i++) {
                solutionArray[areasForWriting[index]] += value ;
                solutionArray[areasForWriting[index]] -= value ;
        }
        solutionArray[areasForWriting[index]] += value ;
	usleep(WRITE_SLEEP) ;
}

void cpu_func_write_data_multiBuffers(void * buffers [], void * cl_arg) {
	// first, we recover all the pieces of data
#ifdef VERBOSE
 	fprintf(stderr, "writing data\n");	
#endif
	int * bufferReader = (int*) STARPU_VARIABLE_GET_PTR(buffers[0]) ; // the value to write
	int * areasForWriting = (int*) STARPU_VECTOR_GET_PTR(buffers[1]) ; // the array of the location for writing
	int * colors = (int*) STARPU_VECTOR_GET_PTR(buffers[2]) ; // the colors 
	int index = (int)((uintptr_t) cl_arg) ;
	int * solutionArray = buffersByColor[colors[index]] ;

	real_func_write_data(solutionArray, areasForWriting, index, *bufferReader) ;
}

void cpu_func_write_data(void * buffers [], void * cl_arg) {
	// first, we recover all the pieces of data
#ifdef VERBOSE
 	fprintf(stderr, "writing data\n");	
#endif
	int * bufferReader = (int*) STARPU_VARIABLE_GET_PTR(buffers[0]) ; // the value to write
	int * areasForWriting = (int*) STARPU_VECTOR_GET_PTR(buffers[1]) ; // the array of the location for writing
	int * solutionArray = (int*) STARPU_VECTOR_GET_PTR(buffers[2]) ; // the array to write : coherency has to be ensured by hand
	int index = (int)((uintptr_t) cl_arg) ;
	real_func_write_data(solutionArray, areasForWriting, index, *bufferReader) ;
}

void cpu_empty_func(void * buffers [], void * cl_arg) {
}

struct starpu_codelet cl_task_reader = {
	.cpu_funcs = {cpu_func_read_data},
	.nbuffers = 2,
	.color = 0xFF0000,
	.modes = {STARPU_R, STARPU_W}
} ;

struct starpu_codelet cl_task_transmitter = {
	.cpu_funcs = {cpu_func_transmit_data},
	.nbuffers = 2,
	.color = 0x00FF00,
	.modes = {STARPU_R, STARPU_W}
} ;

struct starpu_codelet cl_task_writer_context_multibuffer = {
	.cpu_funcs = {cpu_func_write_data_multiBuffers},
	.nbuffers = 3,
	.color = 0x0000FF,
	.modes = {STARPU_R, STARPU_R, STARPU_R} // the last R is the color handle
} ;


struct starpu_codelet cl_task_writer_naive = {
	.cpu_funcs = {cpu_func_write_data},
	.nbuffers = 3,
	.color = 0x0000FF,
	.modes = {STARPU_R, STARPU_R, STARPU_R} // in fact, the last R is a read-write mode, but we manage data coherency by hand
} ;

struct starpu_codelet cl_task_writer_barrier = {
	.cpu_funcs = {cpu_func_write_data},
	.nbuffers = 4,
	.color = 0x0000FF,
	.modes = {STARPU_R, STARPU_R, STARPU_R, STARPU_R} // in fact, the third R is a read-write mode, but we manage data coherency by hand
		// also, the last R is only for having a barrier with other tasks
} ;

struct starpu_codelet cl_task_empty_barrier = {
	.cpu_funcs = {cpu_empty_func},
	.nbuffers = 2,
	.color = 0xFFFF00,
	.modes = {STARPU_RW, STARPU_RW}
} ;

void cpu_reverse_buffers(void * buffers[], void * cl_arg) {
	int nElem = (int)(uintptr_t) cl_arg ;

	int * first_buffer = (int*) STARPU_VECTOR_GET_PTR(buffers[0]) ;
	int * second_buffer = (int*) STARPU_VECTOR_GET_PTR(buffers[1]) ;
	for (int i = 0 ; i < nElem ; second_buffer[i] += first_buffer[i], i++) ; 
}

struct starpu_codelet cl_task_reverse_buffers = {
	.cpu_funcs = {cpu_reverse_buffers},
	.nbuffers = 2,
	.color = 0x00FFFF,
	.modes = {STARPU_R, STARPU_RW}
} ;


void submissionNaive(starpu_data_handle_t arrayProblem_handle, starpu_data_handle_t areasProblem_handle, starpu_data_handle_t solution_handle,
        		starpu_data_handle_t * bufferTaskA_handle, starpu_data_handle_t * bufferTaskB_handle, int nbChain) {
	for (int t = 0 ; t < nbChain ; t++) {

		struct starpu_task * task = starpu_task_create() ;
		task->cl = &cl_task_reader ;
		task->cl_arg = (void*) (uintptr_t) t ;
		task->cl_arg_size = sizeof(int) ;
		task->synchronous = 0 ;
		task->name = "READER" ;
		task->handles[0] = arrayProblem_handle ;
		task->handles[1] = bufferTaskA_handle[t] ;

		starpu_task_submit_verify(task) ;

		task = starpu_task_create() ;

		task->cl = &cl_task_transmitter ;
		task->synchronous = 0 ;
		task->name = "TRANSMITTER" ;
		task->handles[0] = bufferTaskA_handle[t] ;
		task->handles[1] = bufferTaskB_handle[t] ;

		starpu_task_submit_verify(task) ;

		task = starpu_task_create() ;
		task->cl = &cl_task_writer_naive ;
		task->cl_arg = (void*) (uintptr_t) t ;
		task->cl_arg_size = sizeof(int) ;
		task->synchronous = 0 ;
		task->name = "WRITER_NAIVE" ;
		task->handles[0] = bufferTaskB_handle[t] ;
		task->handles[1] = areasProblem_handle ;
		task->handles[2] = solution_handle ;
		starpu_task_submit_verify(task) ;
	}
}

#define NB_BUFS_MAX 3 
static int NB_BUFFERS = NB_BUFS_MAX ;

#include "sc_hypervisor.h"
#include <stdatomic.h>
static int * counter_tasks_by_color = NULL ;
static atomic_int * counter_tasks_ended_by_color = NULL ;
static atomic_int * counter_tasks_in_exec_by_color = NULL ;
int * counter_task_unlocked_by_color = NULL ;
static int nb_cpus_total = 0 ;
static int ** nb_cpus_by_color = NULL ;
static int ** index_cpus_by_color = NULL ; // the idea is the color c has cpus cpus_dispos[index_cpus_by_color[c][0]:index_cpus_by_color[c][0]+nb_cpu_by_color[c][0]] ; 
					//  cpus_dispos[index_cpus_by_color[c][1]:index_cpus_by_color[c][1]+nb_cpu_by_color[c][1]] ; ... ;
static int * cpus_dispos = NULL ;
static int nbCols = 0 ;
static unsigned * contexts ;
static int lastContext = -1 ; 
static starpu_pthread_mutex_t * mutexes, colMutex ;

static int leavingBuffers = 0, ** buffers_dispos = NULL ;
static int * cpu_partitions_dispos = NULL ;

void removeWorkers(int context) {
	for (int b = 0 ; b < NB_BUFFERS ; b++) {
		if (nb_cpus_by_color[context][b] > 0) {
			starpu_sched_ctx_remove_workers(&cpus_dispos[index_cpus_by_color[context][b]], nb_cpus_by_color[context][b], contexts[context]) ;
			nb_cpus_by_color[context][b] = 0 ;
			index_cpus_by_color[context][b] = -1;
		}
	}
	buffersByColor[context] = NULL ;
}

//give the cpus of contextFrom to contextTo
void transferCpus(int contextFrom, int contextTo) {
	// first we need to know if contextTo has already resources or no
	int hasAlreadyResources = buffersByColor[contextTo] != NULL ;

	if (hasAlreadyResources) {
		// we give to contextTo all the buffers of contextFrom ; but we don't give any buffer
		// we have to be carefull that we cannot write on the first index of the context directly
                for (int b = 0 ; b < NB_BUFFERS ; b++) {
                        if (nb_cpus_by_color[contextFrom][b] > 0) { // attribution of b
				// we need to find the location to put the cpus
				for (int b2 = 0 ; b2 < NB_BUFFERS ; b2 ++) { // at the location b2
					if (nb_cpus_by_color[contextTo][b2] == 0) {
		                                nb_cpus_by_color[contextTo][b2] = nb_cpus_by_color[contextFrom][b] ;
                                		index_cpus_by_color[contextTo][b2] = index_cpus_by_color[contextFrom][b] ; 
                                		starpu_sched_ctx_add_workers(&cpus_dispos[index_cpus_by_color[contextTo][b2]], nb_cpus_by_color[contextTo][b2], contexts[contextTo]) ;
						break ;
					}
				}
			}
		}
	} else {
		// we need to give also a buffer to the color
                for (int b = 0 ; b < NB_BUFFERS ; b++) {
                        if (nb_cpus_by_color[contextFrom][b] > 0) {
                                nb_cpus_by_color[contextTo][b] = nb_cpus_by_color[contextFrom][b] ;
                                index_cpus_by_color[contextTo][b] = index_cpus_by_color[contextFrom][b] ;
                                starpu_sched_ctx_add_workers(&cpus_dispos[index_cpus_by_color[contextTo][b]], nb_cpus_by_color[contextTo][b], contexts[contextTo]) ;
                        }
                }
                buffersByColor[contextTo] = buffersByColor[contextFrom] ;
	}
}

// 2nd version of coloration resize : we attribute resources to the better choice
void coloration_smart_resize_ctxs (unsigned *sched_ctxs, int nsched_ctxs, int *workers, int nworkers) {
	// again, we assume that the call is made by nshed_ctxs ctx
	// two possible cases :
	// else this is our first call, and we have to distribute the resources
	STARPU_PTHREAD_MUTEX_LOCK(&colMutex) ;
	//fprintf(stderr, "TODO : initialize nb_buffers ; and initialize buffers_dispos\n") ;
	//fprintf(stderr, "caller is %d and has %d tasks\n", nsched_ctxs, counter_tasks_by_color[nsched_ctxs] - counter_tasks_ended_by_color[nsched_ctxs]) ;
	int ntLeaving = counter_tasks_by_color[nsched_ctxs] - counter_tasks_ended_by_color[nsched_ctxs] ;
	if (ntLeaving > 0 && leavingBuffers > 0) {
		// first call to this, we need to distribute the resources
		// we can again distribute a buffer. We need to find which one
	//		fprintf(stderr, "there is buffers\n") ;
			for(int i = 0 ; i < NB_BUFFERS ; i++) {
				if (buffers_dispos[i]) {
	//				fprintf(stderr, "attribution of %d\n", i);
					// the buffer could be given
					// we give to our cpu the partition i of the cpus
					int nb_div = nb_cpus_total / NB_BUFFERS, leavingCpus = nb_cpus_total % NB_BUFFERS ;
					int index = i == 0 ? 0 : nb_div*i + leavingCpus ;
					int size = nb_div + (i==0)*leavingCpus ;
					index_cpus_by_color[nsched_ctxs][0] = index ;
					nb_cpus_by_color[nsched_ctxs][0] = size ;
					buffersByColor[nsched_ctxs] = buffers_dispos[i] ;
                                        starpu_sched_ctx_add_workers(&cpus_dispos[index_cpus_by_color[nsched_ctxs][0]], nb_cpus_by_color[nsched_ctxs][0], contexts[nsched_ctxs]) ;
					buffers_dispos[i] = NULL ;
					leavingBuffers -- ;
					break ;
				}
			}
	} else if (ntLeaving == 0){
		// we have end this color : we can attribute resources
		// we will evict the cpus of nsched_ctxs to an elected member
		// first we need to find which one is elected
		// then give it the cpus and the buffer
		// a good idea should be to make a function "transfers resources" to give well the things
		// TODO

	//	fprintf(stderr, "here we have %d (color %d end)\n", nb_cpus_by_color[nsched_ctxs][0], nsched_ctxs) ;
		int colorWithoutResources = -1, colorWithResources = -1 ;
		for (int c = 0 ; c < nbCols ; c++) {
			int nbTasksLeaving = counter_tasks_by_color[c] - counter_tasks_ended_by_color[c] ;
			int readyNExecuted = counter_task_unlocked_by_color[c] - counter_tasks_ended_by_color[c] - counter_tasks_in_exec_by_color[c] ;
			int hasResources = buffersByColor[c] != NULL ;
			if (nbTasksLeaving == 0)
				continue ; // if we have no task to execute, don't give any resource 
	//		fprintf(stderr, "color %d has %d tasks\n",c,  nbTasksLeaving) ; 	
			if (!hasResources) {
				if (colorWithoutResources >= 0) {
					// we need to compare the two colors to know which one is the best to execute
					int curReady = counter_task_unlocked_by_color[colorWithoutResources] - counter_tasks_ended_by_color[colorWithoutResources] - counter_tasks_in_exec_by_color[colorWithoutResources] ;
					int nLeaving = counter_tasks_by_color[colorWithoutResources] - counter_tasks_ended_by_color[colorWithoutResources] ;
					if (readyNExecuted > curReady || (readyNExecuted == curReady && nbTasksLeaving > nLeaving)) {
						colorWithoutResources = c ;
					}
				} else {
					// no colorWithoutResources ; we give it to this color
					colorWithoutResources = c ;
				}
			} else {
				// we only make a comparison if we have no colorWithoutResources
				if (colorWithoutResources < 0) {
					if (colorWithResources >= 0) {
						int curReady = counter_task_unlocked_by_color[colorWithResources] - counter_tasks_ended_by_color[colorWithResources] - counter_tasks_in_exec_by_color[colorWithResources] ;
						int nLeaving = counter_tasks_by_color[colorWithResources] - counter_tasks_ended_by_color[colorWithResources] ;
						if ( readyNExecuted > curReady || (readyNExecuted == curReady && nbTasksLeaving > nLeaving)) {
							colorWithResources = c ;
						}
					} else {
						colorWithResources = c ;
					}
				}
			}
		}
		// reattribution of resources
		int contextTo = colorWithoutResources >= 0 ? colorWithoutResources : colorWithResources ;
//		fprintf(stderr, "transfering from %d to %d\n", nsched_ctxs, contextTo) ;
		if (contextTo >= 0) {
			transferCpus(nsched_ctxs, contextTo) ;
		} else if (lastContext >= 0) {
			// work finished
                	starpu_sched_ctx_add_workers(cpus_dispos, nb_cpus_total, lastContext) ;
		}
		// in all case we remove the workers of the context caller
		removeWorkers(nsched_ctxs) ;
	}

	STARPU_PTHREAD_MUTEX_UNLOCK(&colMutex) ;
}

void coloration_resize_ctxs (unsigned *sched_ctxs, int nsched_ctxs, int *workers, int nworkers) {
	// Require explicit resizing
	// we assumes that the call is made like nsched_ctxs == the caller of this function
	int caller = nsched_ctxs ;
	
	// we search for a new color
	int colorLeaving = -1, color_working = -1 ;
	for (int c = 0 ; c < nbCols ; c++) {
		if (counter_tasks_by_color[c] - counter_tasks_ended_by_color[c] > 0  && nb_cpus_by_color[c][0] == 0) {
			// because it is possible to have different buffers, we have to verify that there is no cpu given to c
			colorLeaving = c ;
		} else if (counter_tasks_by_color[c] - counter_tasks_ended_by_color[c] > 0) {
			color_working = c ;
		}
	}

	// here we have several cases
	// colorLeaving is >= 0 if there is someone which has no cpu and again work
	// color_working is >= 0 if there is someone with cpus but again working
	//
	// If colorLeaving >= 0, we need to give all the cpus of c to colorLeaving
	// else if there is no color which has to start the work, but a color with cpus and again work, we give to it all the cpus
	// else, none need the cpus, we give them to the last context for making the reduction
	
	int contextTo = colorLeaving >= 0 ? colorLeaving : color_working ;
	if (contextTo >= 0) {
		transferCpus(caller, contextTo) ;
	} else if (lastContext >= 0) {
		// work finished
                starpu_sched_ctx_add_workers(cpus_dispos, nb_cpus_total, lastContext) ;
	}
	// in all case we remove the workers of the context caller
	removeWorkers(caller) ;	
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

struct sc_hypervisor_policy coloration_smart_policy =
{
        .handle_poped_task = NULL,
        .handle_pushed_task = NULL,
        .handle_idle_cycle = NULL,
        .handle_idle_end = NULL,
        .handle_post_exec_hook = NULL,
        .resize_ctxs = coloration_smart_resize_ctxs,
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
		sc_hypervisor_resize_ctxs(NULL, color, NULL, 0) ;

}

void func_epilogue_task_transmitter(void * arg) {
	int color = (int) (uintptr_t) arg ;
        STARPU_PTHREAD_MUTEX_LOCK(mutexes + color) ;
	counter_task_unlocked_by_color[color] ++ ;
        STARPU_PTHREAD_MUTEX_UNLOCK(mutexes + color) ;
	if (leavingBuffers > 0 && nb_cpus_by_color[color][0] == 0)
	      sc_hypervisor_resize_ctxs(NULL, color, NULL, 0) ;
}



static int * buffers_rec[NB_BUFS_MAX] ;
void liberateSubmissionByContextWithMultiBuffers(void) {
	// first we liberate the counters
	free (counter_tasks_by_color) ;
	free (counter_tasks_ended_by_color) ;
	free (counter_tasks_in_exec_by_color) ;
	free (counter_task_unlocked_by_color) ;
	
	// also the counters of cpus
	for(int c = 0 ; c < nbCols ; c++) {
		free (nb_cpus_by_color[c]) ;
		free (index_cpus_by_color[c]) ;
	}
	free (nb_cpus_by_color) ;
	free (index_cpus_by_color) ;
	free (cpus_dispos) ;

	// also destroying mutexes
	free (mutexes) ;

	// liberating also contextes
	/*for (int i = 0 ; i < nbCols + 1 ; i++)
		starpu_sched_ctx_delete(contexts[i]) ;
	starpu_sched_ctx_delete(lastContext) ;
	*/
	// here, deleting contexts create problems
	free(contexts) ;

	// don't forget buffers
	for (int i = 0 ; i < NB_BUFFERS ; i++)
		free(buffers_rec[i]) ;
	free(buffersByColor) ;
	sc_hypervisor_shutdown() ;
}

int submissionByContextWithMultiBuffers(starpu_data_handle_t arrayProblem_handle, starpu_data_handle_t areasProblem_handle, starpu_data_handle_t solution_handle,
        		starpu_data_handle_t * bufferTaskA_handle, starpu_data_handle_t * bufferTaskB_handle, int nbChain, int * colors, int nbElementOfBuffer, int isSmart) {
	// first step is to invert colors, ie instead of having an array colors like colors[i] gave the colors of the task i ; having an array col_inverse like col_inverse[i] give the tasks of color i
	int maxCol = 0 ;
	for (int t = 0 ; t < nbChain ; t++)
		if (colors[t] > maxCol)
			maxCol = colors[t] ;
	nbCols = maxCol + 1 ; // we have in nbCols the number of colors
	// now we will try to have the array nbByCol with nbByCol[i] is the number of tasks of color i
	
	counter_tasks_by_color = calloc(nbCols, sizeof(int)) ;
	counter_tasks_ended_by_color = calloc(nbCols, sizeof(atomic_int)) ;
	counter_tasks_in_exec_by_color = calloc(nbCols, sizeof(atomic_int)) ;
	counter_task_unlocked_by_color = calloc(nbCols, sizeof(int)) ;
	
	nb_cpus_total = starpu_cpu_worker_get_count() ;
	nb_cpus_by_color = calloc(nbCols, sizeof(int)) ;
	mutexes = calloc(nbCols, sizeof(starpu_pthread_mutex_t)) ;

	starpu_data_handle_t color_handle ;
	starpu_vector_data_register(&color_handle, STARPU_MAIN_RAM, (uintptr_t) colors, nbChain, sizeof(int)) ; 
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

	// here, we have all the coloration ok.
	// The next step is to create several buffers, also one context by color, and to attribute each buffer to one color
	
	// creation of the buffers
	
	buffersByColor = calloc(nbCols, sizeof(int*)) ;
	buffers_dispos = calloc(NB_BUFFERS, sizeof(int*)) ;
	starpu_data_handle_t buffers_handles[nbCols] ;
	cpus_dispos = calloc(nb_cpus_total, sizeof(int)) ;
	leavingBuffers = NB_BUFFERS ;
	//fprintf(stderr, "leaving buffers is %d (start)\n", leavingBuffers) ;
	for (int i = 0 ; i < NB_BUFFERS ; i++) {
		// first, the possessors of the buffers are the first colors
		buffers_rec[i] = buffers_dispos[i] = calloc(nbElementOfBuffer, sizeof(int)) ;
		if (!isSmart)
			buffersByColor[i] = buffers_rec[i] ;
		starpu_vector_data_register(buffers_handles + i, STARPU_MAIN_RAM, (uintptr_t) buffers_rec[i], nbElementOfBuffer, sizeof(int)) ;
	}

	// The next step is to create one context by color
	contexts = calloc(nbCols + 1, sizeof(unsigned)) ; // the last context is the big one

	// before creating the contexts we need the cpus
	starpu_worker_get_ids_by_type(STARPU_CPU_WORKER, cpus_dispos, nb_cpus_total) ; // at the beginning, the first color has the x first cpus
	// the second the x nexts and like this ...
	// but we have only one array with the cpus. the idea is that cpus_by_color[i] will point on a part of the array of cpus 
	
	contexts[nbCols] = starpu_sched_ctx_create(NULL, -1, "ctx", STARPU_SCHED_CTX_POLICY_NAME, "lws", 0) ; // this context is the one with all the tasks of type A and B
	// it can execute on all the cpus
	
	int nb_cpus_div = nb_cpus_total / NB_BUFFERS, leaving_cpus = nb_cpus_total % NB_BUFFERS ;
	
	// the first color has the nb_cpus_div + leaving_cpus first cpus
	// the next from 2 to NB_BUFFERS has the next nb_cpus_div 
	// the last ones have not any cpu
	
	nb_cpus_by_color = calloc(nbCols, sizeof(int*)) ;
	index_cpus_by_color = calloc(nbCols, sizeof(int*)) ;
	for (int i = 0 ; i < nbCols ; i++) {
		char name[25] ;
		sprintf(name, "ctx%d", i) ;
		nb_cpus_by_color[i] = calloc(NB_BUFFERS, sizeof(int)) ;
		index_cpus_by_color[i] = calloc(NB_BUFFERS, sizeof(int)) ;
		//fprintf(stderr, "at start, n is %d\n", counter_task_unlocked_by_color[i]) ;
		if (!isSmart) {
			if (i == 0) {
				// first color
				contexts[i] =  starpu_sched_ctx_create(cpus_dispos, nb_cpus_div + leaving_cpus, name, STARPU_SCHED_CTX_POLICY_NAME, "lws", 0) ;		
				nb_cpus_by_color[0][0] = nb_cpus_div + leaving_cpus ;
				index_cpus_by_color[0][0] = 0 ;
			} else if (i < NB_BUFFERS && !isSmart) {
				// the color has some cpus	
				contexts[i] =  starpu_sched_ctx_create(cpus_dispos + i*nb_cpus_div + leaving_cpus, nb_cpus_div, name, STARPU_SCHED_CTX_POLICY_NAME, "lws", 0) ;
				nb_cpus_by_color[i][0] = nb_cpus_div ;
				index_cpus_by_color[i][0] = i*nb_cpus_div + leaving_cpus ;
			} else {
				// no cpu
				contexts[i] =  starpu_sched_ctx_create(NULL, 0, name, STARPU_SCHED_CTX_POLICY_NAME, "lws", 0) ;
				nb_cpus_by_color[i][0] = 0 ;
			}
		} else {
			contexts[i] =  starpu_sched_ctx_create(NULL, 0, name, STARPU_SCHED_CTX_POLICY_NAME, "lws", 0) ;
			nb_cpus_by_color[i][0] = 0 ;
		}
		STARPU_PTHREAD_MUTEX_INIT(mutexes + i, NULL) ;
	}
	STARPU_PTHREAD_MUTEX_INIT(&colMutex, NULL) ;
	// now we can make the hypervisor
	sc_hypervisor_init(isSmart ? &coloration_smart_policy : &coloration_policy) ; 
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
		task->name = "READER" ;
		task->handles[0] = arrayProblem_handle ;
		task->handles[1] = bufferTaskA_handle[t] ;

		starpu_task_submit_to_ctx(task, contexts[nbCols]) ;

		task = starpu_task_create() ;

		task->cl = &cl_task_transmitter ;
		task->synchronous = 0 ;
		task->name = "TRANSMITTER" ;
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
			task->cl = &cl_task_writer_context_multibuffer ;
			task->cl_arg = (void*) (uintptr_t) taskOfColor[c][t] ;
			task->cl_arg_size = sizeof(int) ;
			task->synchronous = 0 ;
			task->name = "WRITER_MULTIBUFFER" ;
		        task->handles[0] = bufferTaskB_handle[taskOfColor[c][t]] ;
                	task->handles[1] = areasProblem_handle ;
                	task->handles[2] = color_handle ;
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
	
	// now we can push tasks for making the reduction
	// for now, we make a stupid one, we reverse each buffer on the global buffer
	// to do this, we create a new context without any cpus
	// when all the C tasks are ended, the hypervisor give all the cpus to this context
	
	lastContext = starpu_sched_ctx_create(NULL, 0, "lastContext", STARPU_SCHED_CTX_POLICY_NAME, "lws", 0) ;
        sc_hypervisor_register_ctx(lastContext, 0.0) ;
	for(int b = 0 ; b < NB_BUFFERS ; b++) {
		struct starpu_task * task = starpu_task_create() ;
		task->cl = &cl_task_reverse_buffers ;
		task->synchronous = 0 ;
		task->cl_arg = (void*) (uintptr_t) nbElementOfBuffer ;
		task->cl_arg_size = sizeof(int) ;
		task->handles[0] = buffers_handles[b] ;
		task->handles[1] = solution_handle ;
		task->name = "REVERSING" ;
		starpu_task_submit_to_ctx(task, lastContext) ;
	}
	starpu_data_unregister_submit(color_handle) ;
	for (int b = 0 ; b < NB_BUFFERS ; b++)
		starpu_data_unregister_submit(buffers_handles[b]) ;
}



int submissionByContext(starpu_data_handle_t arrayProblem_handle, starpu_data_handle_t areasProblem_handle, starpu_data_handle_t solution_handle,
        		starpu_data_handle_t * bufferTaskA_handle, starpu_data_handle_t * bufferTaskB_handle, int nbChain, int * colors, int nbElementOfBuffer, int isSmart) {
	NB_BUFFERS = 1 ;
	submissionByContextWithMultiBuffers(arrayProblem_handle, areasProblem_handle, solution_handle, bufferTaskA_handle, bufferTaskB_handle, nbChain, colors, nbElementOfBuffer, isSmart) ;
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
		task->name = "READER" ;
		task->handles[0] = arrayProblem_handle ;
		task->handles[1] = bufferTaskA_handle[t] ;
		starpu_task_submit_verify(task) ;

		task = starpu_task_create() ;
		task->cl = &cl_task_transmitter ;
		task->synchronous = 0 ;
		task->name = "TRANSMITTER" ;
		task->handles[0] = bufferTaskA_handle[t] ;
		task->handles[1] = bufferTaskB_handle[t] ;
		starpu_task_submit_verify(task) ;
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
			task->name = "WRITER_BARRIER" ;
		        task->handles[0] = bufferTaskB_handle[taskOfColor[c][t]] ;
                	task->handles[1] = areasProblem_handle ;
                	task->handles[2] = solution_handle ;
                	task->handles[3] = barrier_handle[c] ;
			
			starpu_task_submit_verify(task) ;
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
			task->name = "BARRIER" ;
			starpu_task_submit_verify(task) ;
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
	if (argc < 5) {
		fprintf(stderr, "We need 3 arguments : ./a.out nbAreas numberOfChainOfTasks submissionMode isSmart\nnbAreas is the number of different areas the tasks will write\nnumberOfChainOfTasks is the number of different tasks\nsubmissionMode is %d for naive, %d for barrier, %d for coloring with lock (todo), %d for bufferized coloring (todo)\n", NAIVE, BARRIER, COLORING_LOCK, COLORING_BUFFER);
		exit(1);
	}
	int nbAreas = atoi(argv[1]), nbChain = atoi(argv[2]), mode = atoi(argv[3]), isSmart = atoi(argv[4]) ;

	int * arraySolution = calloc(nbAreas, sizeof(int)) ;

	int * givenAreas = calloc(nbChain, sizeof(int)) ;
	if (argc >= 6)
		NB_BUFFERS = atoi(argv[5]) ;	
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
	
  	 fprintf(stderr, "We have %d colors\n", maxC + 1) ;
	if (starpu_init(NULL) != 0) {
		fprintf(stderr, "Error when initializing StarPU. shuting down now ...\n");
		exit(1) ;
	}

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
		submissionByContext(arrayProblem_handle, areasProblem_handle, solution_handle, bufferTaskA_handle, bufferTaskB_handle, nbChain, colors, nbAreas, isSmart) ;
	} else if (mode == COLORING_BUFFER) {
		fprintf(stderr, "Launching submission of tasks by context, with %d buffer(s) (should be correct)\n", NB_BUFFERS);
		submissionByContextWithMultiBuffers(arrayProblem_handle, areasProblem_handle, solution_handle, bufferTaskA_handle, bufferTaskB_handle, nbChain, colors, nbAreas, isSmart) ;
	}
	
	starpu_task_wait_for_all() ;
	clock_gettime(CLOCK_MONOTONIC, &end_t) ;

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
#ifdef VERBOSE
		fprintf(stderr, "%d - %d\n", realSolution[a], arraySolution[a]);
#endif 
		if (realSolution[a] != arraySolution[a]) {
			fprintf(stderr, "error : the index %d is not correct : %d for real versus %d for parallel\n", a, realSolution[a], arraySolution[a]) ;
			ret = 1 ;	
		}
	}
	double nsecs = end_t.tv_sec - start_t.tv_sec + ((end_t.tv_nsec - start_t.tv_nsec)/1000000000.0) ;
	long nusecs = nsecs * 1000. ;
	if (!ret)
		fprintf(stderr, "all good in %.3lf secs ... ending normally\n", end_t.tv_sec - start_t.tv_sec + ((end_t.tv_nsec - start_t.tv_nsec)/1000000000.0)) ;
	else
		fprintf(stderr, "error in %.3lf secs ... ending normally\n", end_t.tv_sec - start_t.tv_sec + ((end_t.tv_nsec - start_t.tv_nsec)/1000000000.0)) ;
	FILE * res = fopen("results.res", "a+") ;
	fprintf(res, "ok ? %d ; mode = %d ; na = %d ; nt = %d ; t = %lf\n", 1 - ret, mode, nbAreas, nbChain, nsecs) ;
	fclose(res) ;
	free(givenAreas) ;
	free(arraySolution) ;
	free(realSolution) ;
	free(colors) ;
	free(A) ;
	if (mode == COLORING_BUFFER) {
		liberateSubmissionByContextWithMultiBuffers() ;
	}
	return ret ;
}


