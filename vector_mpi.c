#include <starpu_mpi.h>
#include <inttypes.h>
#include <stdio.h>
#define SIZE (16384*16384)
#define ALPHA 3
#define NLEVELS 4 
#define NB_TOT_LEVELS 127 //2^(NLEVELS+1)-1 
#define NITER 10
//#define DEBUG
static int POWS[] =  {1, 2, 4, 8, 16, 32, 64, 128} ;
 
static int rank ;
int is_bubble(struct starpu_task *t, void *arg)
{
        uint64_t value = (uint64_t) arg ;
        // first 32 bits give the level, last give the number
 	      	 
        int level =  (int) value ;

#ifdef DEBUG
        fprintf(stderr, "in is bubble, lev is %d: %d\n", level, level < NLEVELS-1) ;
#endif
	return level < NLEVELS -1 ;
}

void axpy_kernel(void *buffers[], void * arg) {

	int * x = (int*) STARPU_VECTOR_GET_PTR(buffers[0]) ;
	int * y = (int*) STARPU_VECTOR_GET_PTR(buffers[1]) ;
	int size = STARPU_VECTOR_GET_NX(buffers[0]) ;
#ifdef DEBUG
	fprintf(stderr, "executing a bubble starting of size %d in rank %d\n", size, rank);
#endif
	for (int i = 0; i < size ;i++) {
#ifdef DEBUG2
		fprintf(stderr, "making %d : %d*%d+%d = %d\n", i, ALPHA, x[i], y[i], ALPHA*x[i] +y[i]) ;
#endif
		x[i] = ALPHA*x[i] + y[i] ;
	}
}

starpu_data_handle_t * handles_x ;
starpu_data_handle_t * handles_y ;
#include <mpi.h>
#include <math.h>


static struct starpu_codelet axpy_codelet = {
	.cpu_funcs = {axpy_kernel},
	.nbuffers = 2,
	.color=0X0000FF,
	.modes = {STARPU_RW, STARPU_R}
} ;

void gen_dag_func(struct starpu_task *t, void *arg) {
        uint64_t value = (uint64_t) arg ;
        // first 32 bits give the level, last give the number
#ifdef DEBUG
	fprintf(stderr, "generating DAG from proc %d\n", rank) ;
#endif
	int ret ;
	int level =  (int) ((( ( (uint64_t) 1) << 33) -1) & value) ;

        int index = (int) (value >> 32) ;
//#ifdef DEBUG
	fprintf(stderr, "gen dag : lev %d ind : %d %p ; rank %d\n", level, index, arg, rank) ;        
//#endif
        starpu_data_handle_t * subdata_x0 = &handles_x[POWS[level+1]-1 + 2*index] ;
        starpu_data_handle_t * subdata_x1 = &handles_x[POWS[level+1]-1 + 2*index+1] ;

        starpu_data_handle_t * subdata_y0 = &handles_y[POWS[level+1]-1 + 2*index] ;
        starpu_data_handle_t * subdata_y1 = &handles_y[POWS[level+1]-1 + 2*index+1] ;

        uint64_t argtask0 = ((2lu*index)<<32) + level + 1, argtask1 = ((2llu*((uint64_t)index)+1llu)<<32) + level + 1 ;
#ifdef DEBUG
	fprintf(stderr, "args are %p and %p\n", argtask0, argtask1) ;
#endif
	char name0[128], name1[128] ;
	sprintf(name0, "axpy_%d_%d", level+1, 2*index) ;
	sprintf(name1, "axpy_%d_%d", level+1, 2*index+1) ;

        ret = starpu_mpi_task_insert(MPI_COMM_WORLD, &axpy_codelet,
	//			STARPU_EXECUTE_ON_NODE, execute_on_node0,
                                STARPU_RW, *subdata_x0,
                                STARPU_R, *subdata_y0,
                                STARPU_BUBBLE_FUNC, &is_bubble,
                                STARPU_BUBBLE_GEN_DAG_FUNC, &gen_dag_func,
                                STARPU_BUBBLE_GEN_DAG_FUNC_ARG, (void*)(uintptr_t) argtask0,
                                STARPU_BUBBLE_FUNC_ARG, (void*)(uintptr_t) argtask0,
				STARPU_NAME, name0,
                                NULL) ;
        STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
	
	ret = starpu_mpi_task_insert(MPI_COMM_WORLD, &axpy_codelet,
	//			STARPU_EXECUTE_ON_NODE, execute_on_node1,
                                STARPU_RW, *subdata_x1,
                                STARPU_R, *subdata_y1,
                                STARPU_BUBBLE_FUNC, &is_bubble,
                                STARPU_BUBBLE_GEN_DAG_FUNC, &gen_dag_func,
                                STARPU_BUBBLE_GEN_DAG_FUNC_ARG, (void*)(uintptr_t) argtask1,
                                STARPU_BUBBLE_FUNC_ARG, (void*)(uintptr_t) argtask1,
				STARPU_NAME, name1,
                                NULL) ;
        STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
} 



static struct starpu_data_filter f = {
        .filter_func = starpu_vector_filter_block,
	.nchildren = 2
};

static int * possession_tree = NULL ;
int _check_result(int * x, int node, int from, int to, int act_ind_on_level, int max_at_this_level, int nb_at_this_level) {
	int cur_ind = max_at_this_level - nb_at_this_level + act_ind_on_level ;
#ifdef DEBUG
	fprintf(stderr, "looking from %d to %d at index %d (%d) on node %d\n", from, to, act_ind_on_level, cur_ind, node) ;
#endif
	if (possession_tree[cur_ind] != STARPU_MPI_MULTIPLE_NODE_WITH_ME && possession_tree[cur_ind] != node)
		return 1 ;
	if (possession_tree[cur_ind] == STARPU_MPI_MULTIPLE_NODE_WITH_ME) {
		int mid = (from+to)/2 ;
		return _check_result(x, node, from, mid, 2*act_ind_on_level, max_at_this_level + 2*nb_at_this_level, 2*nb_at_this_level) && _check_result(x, node, mid, to, 2*act_ind_on_level+1, max_at_this_level + 2*nb_at_this_level, 2*nb_at_this_level) ;
	}
	for (int i = from; i < to; i++) {
		int expected_result = (i%100) ;
		for (int it = 0 ; it < NITER; it++) {
			expected_result = ALPHA*expected_result+(i%100)+3 ;
		}
		if (x[i] != expected_result) {
#ifdef DEBUG
			fprintf(stderr, "result for i=%d is %d instead of %d (node %d)\n", i, x[i], expected_result, node );
#endif
			return 0 ;
		} else {
#ifdef DEBUG
			fprintf(stderr, "result for i=%d is ok : %d (node %d)\n", i, x[i], node );
#endif
		}
	}
	return 1 ;
}

int check_result(int * x, int node) {
	return _check_result(x, node, 0, SIZE, 0, 1, 1) ;
}

#include <assert.h>

int has_data(int level, int index, int nb_by_lev) {
	if (level == 0)
		return STARPU_MPI_MULTIPLE_NODE_WITH_ME ;
	if (level == 1) {
		if ( (index == 0 && rank <= 1) || (index == 1 && rank >=2)) {
			return STARPU_MPI_MULTIPLE_NODE_WITH_ME ;
		}
		return STARPU_MPI_MULTIPLE_NODE_WITHOUT_ME ;
	}
	return index < nb_by_lev/2 ? (index < nb_by_lev / 4 ? 0 : 1) : (index < (3*nb_by_lev) / 4 ? 2 :  3) ;
}

int main(int argc, char * argv[]) {
	starpu_fxt_autostart_profiling(0) ;

	int size, node = -1, ret;
	ret = starpu_mpi_init_conf(&argc, &argv, 1, MPI_COMM_WORLD, NULL);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_init");
	starpu_mpi_comm_rank(MPI_COMM_WORLD, &rank);
	starpu_mpi_comm_size(MPI_COMM_WORLD, &size);
	assert(size == 4) ;
	possession_tree = calloc(NB_TOT_LEVELS, sizeof(int)) ;
	handles_x = calloc(NB_TOT_LEVELS, sizeof(starpu_data_handle_t)) ;
	handles_y = calloc(NB_TOT_LEVELS, sizeof(starpu_data_handle_t)) ;
	int *x = NULL, *y = NULL ;
	x = calloc(SIZE, sizeof(int)), y = calloc(SIZE, sizeof(int)) ;
	for (int i = 0; i < SIZE; i++) {
		x[i] = (i%100) ;
		y[i] = (i%100)+3 ;
	}
	node = STARPU_MAIN_RAM ;

	starpu_vector_data_register(handles_x, node, (uintptr_t) x, SIZE, sizeof(int));
	starpu_vector_data_register(handles_y, node, (uintptr_t) y, SIZE, sizeof(int));
	possession_tree[0] = STARPU_MPI_MULTIPLE_NODE ;
	int l, ll ;
	int nb_by_lev = 1 ;
	int x_curstart_level = 0 ;
	for (l = 0; l < NLEVELS-1; l++) {
		for (ll = 0; ll < nb_by_lev; ll++) {
			possession_tree[x_curstart_level + ll] = has_data(l, ll, nb_by_lev) ;	
#ifdef DEBUG
			fprintf(stderr, "at level %d, %d is at %d\n", l, ll, possession_tree[x_curstart_level+ll]) ;
#endif
			starpu_data_partition_plan(handles_x[x_curstart_level + ll], &f, &handles_x[x_curstart_level+nb_by_lev + ll*2] ) ;	
			starpu_mpi_data_register(handles_x[x_curstart_level + ll], x_curstart_level + ll, possession_tree[x_curstart_level + ll] ) ;
			starpu_data_partition_plan(handles_y[x_curstart_level + ll], &f, &handles_y[x_curstart_level+nb_by_lev + ll*2] ) ;
			starpu_mpi_data_register(handles_y[x_curstart_level + ll], NB_TOT_LEVELS + x_curstart_level + ll, possession_tree[x_curstart_level + ll] );
		}
		x_curstart_level += nb_by_lev ;
		nb_by_lev *= 2 ;
	}
	for (ll = 0; ll < nb_by_lev; ll++) {
		possession_tree[x_curstart_level + ll] = has_data(l, ll, nb_by_lev) ;
#ifdef DEBUG
		fprintf(stderr, "at level %d, %d is at %d\n", l, ll, possession_tree[x_curstart_level+ll]) ;
#endif
		starpu_mpi_data_register(handles_x[x_curstart_level + ll], x_curstart_level + ll, possession_tree[x_curstart_level + ll]);
		starpu_mpi_data_register(handles_y[x_curstart_level + ll], NB_TOT_LEVELS + x_curstart_level + ll, possession_tree[x_curstart_level + ll]);	
	}
	starpu_fxt_start_profiling();
	for (int i = 0; i < NITER; i++) {
		ret = starpu_mpi_task_insert(MPI_COMM_WORLD, &axpy_codelet,
				STARPU_RW, handles_x[0],
				STARPU_R, handles_y[0],
				STARPU_BUBBLE_FUNC, &is_bubble,
				STARPU_BUBBLE_GEN_DAG_FUNC, &gen_dag_func,
				STARPU_BUBBLE_GEN_DAG_FUNC_ARG, NULL,
				STARPU_BUBBLE_FUNC_ARG, NULL,
				STARPU_NAME, "axpy_0_0",
				NULL) ;
        	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
	}
	starpu_task_wait_for_all() ;
	starpu_fxt_stop_profiling();

//	starpu_data_partition_clean(handles_x[0], 2, &handles_x[1] ) ;
//	starpu_data_partition_clean(handles_y[0], 2, &handles_y[1] ) ;
	nb_by_lev/=2 ;
	x_curstart_level -= nb_by_lev ;
	for (l = NLEVELS-2; l >=0; l--) {
		for (ll = 0; ll < nb_by_lev; ll++) {
			starpu_data_partition_clean(handles_x[x_curstart_level + ll], 2, &handles_x[x_curstart_level+nb_by_lev + ll*2] ) ;
			starpu_data_partition_clean(handles_y[x_curstart_level + ll], 2, &handles_y[x_curstart_level+nb_by_lev + ll*2] ) ;
		}
		nb_by_lev /= 2 ;
		x_curstart_level -= nb_by_lev ;
	}
	starpu_data_unregister(handles_x[0]) ;
	starpu_data_unregister(handles_y[0]) ;
	starpu_mpi_shutdown() ;
	if (node != -1) {
	if (check_result(x, rank))
		fprintf(stderr, "SUCCESS\n") ;
		else {
			fprintf(stderr, "FAILED\n");
		}
		free(x) ; free(y) ;
	}
	return 0 ;
}
