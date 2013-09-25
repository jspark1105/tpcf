/**
Copyright (c) 2013, Intel Corporation. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the Intel Corporation nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL INTEL CORPORATION BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <xmmintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h> 

#include <unistd.h>
#include <stdarg.h>
#include <nmmintrin.h>
#include <stdlib.h>
#include <pthread.h>
#include <math.h>
#include <float.h>
#include <omp.h>
#include "ia32intrin.h"

#ifdef __INTEL_OFFLOAD
#define DECLSPEC_TARGET_MIC __declspec(target(mic))
#else
#define DECLSPEC_TARGET_MIC
#endif

int global_argc;
char **global_argv;
char global_D_filename[256];
char global_R_filename[256];
#define PRINT_BLACK         
#define PRINT_RED           
#define PRINT_GREEN         
#define PRINT_BROWN         
#define PRINT_BLUE          
#define PRINT_MAGENTA       
#define PRINT_CYAN          
#define PRINT_GRAY          
#define PRINT_LIGHT_RED     
#define PRINT_LIGHT_GREEN   

#define MPI_DEBUGGING

#define SIMD_WIDTH 8
#define CORE_FREQUENCY (2.6*1000.0*1000.0*1000.0)

unsigned long long int MT_2[1] = {0};
unsigned long long int MT_3[1] = {0};
unsigned long long int MT_4[1] = {0};
unsigned long long int MT_5[1] = {0};
unsigned long long int MT_Z[1] = {0};

#define PCL_MIN(a,b) (((a) < (b)) ? (a) : (b))
#define PCL_MAX(a,b) (((a) > (b)) ? (a) : (b))

#define ERROR_PRINT() {printf("Error in file (%s) on line (%d) in function (%s)\n", __FILE__, __LINE__, __FUNCTION__); exit(123); }
#define ERROR_PRINT_STRING(abc) {printf("Error (%s) in file (%s) on line (%d) in function (%s)\n", abc, __FILE__, __LINE__, __FUNCTION__); exit(123); }
#define GET_POINT(Array, indexx, coordinate, total_number_of_points)     *(Array + (indexx) * DIMENSIONS + (coordinate))

char global_dirname[256];

int node_id;
int nnodes;
int nthreads;
#define MAX_THREADS 128
pthread_t threads[MAX_THREADS];
pthread_attr_t attr;
#define MY_BARRIER(threadid) barrier(threadid)
#define MPI_BARRIER(nodeid) MPI_Barrier(MPI_COMM_WORLD);

MPI_Request *recv_request;
MPI_Request *send_request_key;
MPI_Status *recv_status;

unsigned long long int global_memory_malloced = 0;

unsigned long long int global_time_total = 0, global_time_kdtree_d = 0, global_time_kdtree_r = 0, global_time_kdtree = 0, global_time_rr = 0, global_time_dr = 0, global_time_mpi = 0;
unsigned long long int global_time_per_thread_rr[MAX_THREADS] = {0};
unsigned long long int global_time_per_thread_dr[MAX_THREADS] = {0};






//#define POINTS_2D
#define TYPE float
#define DIMENSIONS 3
#define HIST_BINS 12
long long int global_number_of_galaxies = 0;
long long int global_number_of_galaxies_on_node_D;
long long int global_number_of_galaxies_on_node_R;
long long int global_galaxies_starting_index, global_galaxies_ending_index;

TYPE *global_Positions_D = NULL;
TYPE *global_Positions_R = NULL;
int global_starting_cell_index_D = -95123;
int global_ending_cell_index_D = -95123;
int global_starting_cell_index_R = -95123;
int global_ending_cell_index_R = -95123;


TYPE global_Lbox, global_rminL, global_rmaxL;
int global_nrbin;

int *global_Owner_D;
unsigned char *global_Required_D;
unsigned char *global_Required_D_For_R;

int *global_Owner_R;
unsigned char *global_Required_R;

typedef struct Gri
{
    TYPE **Positions;   
    int dimx;           
    int dimy;           
    int dimz;           

    TYPE Cell_Width[DIMENSIONS];
    TYPE Min[DIMENSIONS]; 
    TYPE Max[DIMENSIONS];
    TYPE Extent[DIMENSIONS];


    int number_of_uniform_subdivisions;
    int *Number_of_kd_subdivisions;

    TYPE **Bdry_X;
    TYPE **Bdry_Y;
    TYPE **Bdry_Z;

    int **Range;


    int **Count_Per_Thread;
    int *Count_Per_Cell;

    int *Start_Cell;
    int *End_Cell;
}Grid;

Grid global_grid_D;
Grid global_grid_R;
int **Ranges1;
int **Ranges2;
int *Ranges12_Max_Size;



long long int global_Easy[MAX_THREADS * 8] = {0};
long long int global_accumulated_easy = 0;

int global_spit_kdtree_to_file = 0;
int global_read_kdtree_from_file = 0;
unsigned long long int **local_Histogram_RR;
unsigned long long int **local_Histogram_DR;
unsigned long long int global_Histogram_DR[HIST_BINS];
unsigned long long int global_Histogram_RR[HIST_BINS];
unsigned long long int *global_Overall_Histogram_DR;
unsigned long long int *global_Overall_Histogram_RR;
double global_RR_over_RR[HIST_BINS];
double global_DR_over_RR[HIST_BINS];
long long int global_stat_total_interactions_rr = 0;
long long int global_stat_total_interactions_dr = 0;
long long int global_stat_useful_interactions_rr = 0;
long long int global_stat_useful_interactions_dr = 0;

TYPE **global_Aligned_Buffer;

TYPE global_rmax_2;
int global_dx;
int global_dy;
int global_dz;
TYPE *global_Rminarr;
TYPE *global_Rmaxarr;
TYPE *global_Rval;
TYPE *global_BinCorners;
TYPE *global_BinCorners2;
long long int *global_actual_sum_dr;
long long int *global_actual_sum_rr;

unsigned int ** global_Gather_Histogram0;
unsigned int ** global_Gather_Histogram1;
unsigned int ** global_RR_int0;
unsigned int ** global_RR_int1;
unsigned int ** global_DR_int0;
unsigned int ** global_DR_int1;
TYPE ** global_Pos1;
TYPE ** global_Bdry1_X;
TYPE ** global_Bdry1_Y;
TYPE ** global_Bdry1_Z;
void mpi_printf(char *formatstring, ...)
{
#ifdef MPI_DEBUGGING
    printf("(%s) --> [%d] ::: ", __FILE__, __LINE__);
    va_list arguments;
    va_start(arguments, formatstring);
    vfprintf(stdout, formatstring, arguments);
    va_end(arguments);
#endif
}

#include <stdarg.h>
void debug_printf(char *formatstring, ...)
{
#ifdef DETAILED_DEBUGGING
    va_list arguments;
    va_start(arguments, formatstring);
    vfprintf(stdout, formatstring, arguments);
    va_end(arguments);
#endif
}


void ParseArgs(int argc, char **argv)
{
    //The code expects 5 arguments... Some of them can be reduced, but
    //that's how it is right now...
    if (argc != 5)
    {
        printf("Usage ./a.out <D_File> <R_File> <nthreads> <nnodes> \n");
        exit(123);
    }

    //The code takes in the 'D_file' (filename of the fle with the coordinates of
    //the actual particles), followed by the '$_file' (filename of the
    //file which contains the same number of random particles (within
    //the same bounding box), followed by number of threaads and
    //followed by number of nodes. 
    //The code outputs RR and DR histograms -- 
    //Note that D file can be same as the R file, in which case DD and
    //DD will be printed out...
    
    //Note: For computation of RR, the code expoits the symmetry, and
    //only considers each pair just once. Hence, if D file === R file,
    //the DR output histograms will be twice that of RR histogram... 

    sscanf(argv[1], "%s", global_D_filename);
    sscanf(argv[2], "%s", global_R_filename);
    sscanf(argv[3], "%d", &nthreads);
    sscanf(argv[4], "%d", &nnodes);

    if (nthreads > MAX_THREADS) ERROR_PRINT_STRING("nthreads > MAX_THREADS");
    if (node_id == 0) printf("nnodes = %d\n", nnodes);
    if (node_id == 0) printf("nthreads = %d\n", nthreads);

//Paramters are set here -- so that they can be used later on...
    global_Lbox  = 100.0;
    global_rminL = 0.1;
    global_rmaxL = 10.0;
    global_nrbin = 10;

    int hist_bins = global_nrbin + 2;

    if ((hist_bins) != HIST_BINS)
    {
        ERROR_PRINT_STRING("Please change HIST_BINS or global_nrbin");
    }
}

#define GET_CELL_INDEX(x0, y0, z0) (x0 + y0*dimx + z0*dimxy)

///////////////////////////////////////////////////////////////////////////////////////////////////

void *my_malloc(size_t size)
{
   
    size_t sizee = size + 128;

    global_memory_malloced += sizee;

    unsigned char *X = (unsigned char *)malloc(sizee);


    unsigned long long int Y = (unsigned long long int)(X);

    while (Y % 64)
    {
        X++;
        Y =  (unsigned long long int)(X);
    }

    return ((void *)(X));
}


void *my_another_malloc(size_t size)
{

    global_memory_malloced += size;
    return (malloc(size));
}


void my_another_free(void *X, int sizee)
{
    global_memory_malloced -= sizee;
    free(X);
}


void my_free(void *X)
{
}


///////////////////////////////////////////////////////////////////////////////////////////////////


pthread_mutex_t barrier_mutex;
pthread_mutex_t complete_mutex;
pthread_cond_t barrier_cond;
pthread_cond_t complete_cond;

volatile static int _barrier_turn_ = 0;
volatile static int _barrier_go_1;
volatile static int _barrier_1[256];
volatile static int _barrier_go_2;
volatile static int _barrier_2[256];

static volatile __declspec(align(64)) unsigned _MY_BARRIER_TURN_ = 0;
static volatile __declspec(align(64)) unsigned _MY_BARRIER_COUNT_0 = 0;
static volatile __declspec(align(64)) unsigned _MY_BARRIER_COUNT_1 = 0;
static volatile __declspec(align(64)) unsigned _MY_BARRIER_FLAG_0 = 0;
static volatile __declspec(align(64)) unsigned _MY_BARRIER_FLAG_1 = 0;

static volatile int version = 0;
static volatile int gcount = 0;		

void old_barrier()
{
	pthread_mutex_lock(&complete_mutex);
	gcount += 1;
	if(gcount == nthreads){	
		gcount = 0;
		pthread_cond_broadcast(&complete_cond);	
	}
	else{
		pthread_cond_wait(&complete_cond,&complete_mutex);
	}
	pthread_mutex_unlock(&complete_mutex);
}

void barrier(int threadid=0)
{

  if (_barrier_turn_ == 0) {
    if (threadid == 0) {
      for (int i=1; i<nthreads; i++) {
        while(_barrier_1[i] == 0);
        _barrier_2[i] = 0;
      }
      _barrier_turn_ = 1;
      _barrier_go_2 = 0;
      _barrier_go_1 = 1;
    }
    else
    {
      _barrier_1[threadid] = 1;
      while(_barrier_go_1 == 0);
    }
  }
  else {
    if (threadid == 0) {
      for (int i=1; i<nthreads; i++) {
        while(_barrier_2[i] == 0);
        _barrier_1[i] = 0;
      }
      _barrier_turn_ = 0;
      _barrier_go_1 = 0;
      _barrier_go_2 = 1;
    }
    else {
      _barrier_2[threadid] = 1;
      while(_barrier_go_2 == 0);
    }
  }
}

extern int global_number_of_phases;




void barrier3(int threadid, int phase, int iteration)
{
    barrier(threadid);
}

#define CLAMP_ABOVE(x, value) (x = (x > (value)) ? (value) : x)
#define CLAMP_BELOW(x, value) (x = (x < (value)) ? (value) : x)
#define SWAP_INT_ADDR(X, Y) {  int *ZZZ = X; X = Y; Y = ZZZ;}
#define SWAP_TYPE_ADDR(X,Y) { TYPE *ZZZ = X; X = Y; Y = ZZZ;}
#define SWAP_TYPE(X,Y) { TYPE ZZZ = X; X = Y; Y = ZZZ;}


int Find_Min_Max_And_Separating_Axis(TYPE *Pos1, int count, TYPE *Pos2)
{
    if (DIMENSIONS != 3) ERROR_PRINT_STRING("DIMENSIONS != 3");

    TYPE Min[DIMENSIONS], Max[DIMENSIONS];

    Min[0] = Max[0] = Pos1[0];
    Min[1] = Max[1] = Pos1[1];
    Min[2] = Max[2] = Pos1[2];

    TYPE sum[DIMENSIONS] = {0,0,0};

    TYPE extent[DIMENSIONS];

    for(int i=0; i<count; i++)
    {
        sum[0] += Pos1[3*i+0];
        sum[1] += Pos1[3*i+1];
        sum[2] += Pos1[3*i+2];

        Min[0] = PCL_MIN(Min[0], Pos1[3*i+0]);
        Max[0] = PCL_MAX(Max[0], Pos1[3*i+0]);

        Min[1] = PCL_MIN(Min[1], Pos1[3*i+1]);
        Max[1] = PCL_MAX(Max[1], Pos1[3*i+1]);

        Min[2] = PCL_MIN(Min[2], Pos1[3*i+2]);
        Max[2] = PCL_MAX(Max[2], Pos1[3*i+2]);
    }

    extent[0] = (Max[0] - Min[0]); 
    extent[1] = (Max[1] - Min[1]); 
    extent[2] = (Max[2] - Min[2]); 

    TYPE value;
    int axis = -1;

    if ( (extent[0] >= extent[1]) && (extent[0] >= extent[2]))
    {
        value = sum[0]/count;
        axis = 0;
    }
    else if ( (extent[1] >= extent[2]) && (extent[1] >= extent[0]))
    {
        value = sum[1]/count;
        axis = 1;
    }
    else
    {
        value = sum[2]/count;
        axis = 2;
    }

    if (axis == -1) ERROR_PRINT();

    int left_side = 0; 
    for(int i=0; i<count; i++) left_side += (Pos1[3*i+axis] <= value);


    TYPE fraction = (left_side * 1.0)/count;
    TYPE threshold = 0.3295123;

    if (threshold >= 0.5) ERROR_PRINT();

    if  ((fraction < threshold) || (fraction > (1-threshold)))
    {
        TYPE old_value = value;
        int old_left_side = left_side;
        TYPE old_fraction = fraction;

        TYPE small_weight, large_weight;
        
        if (fraction < threshold)
        {
            small_weight = 2*fraction;
        }
        else
        {
            small_weight = 2*(1 - fraction);
        }
            
        if (small_weight < 0) ERROR_PRINT();
        if (small_weight > 1) ERROR_PRINT();
        large_weight = 1 - small_weight;

        if (small_weight > large_weight)
        {
            TYPE temp_weight = small_weight;
            small_weight = large_weight;
            large_weight = temp_weight;
        }

        if (fraction < threshold)
        {
            value = old_value *(large_weight) + Max[axis]*small_weight;
        }
        else if (fraction > (1-threshold))
        {
            value = old_value*large_weight + Min[axis]*small_weight;
        }
        else
        {
            ERROR_PRINT();
        }
    
        left_side = 0;
        for(int i=0; i<count; i++) left_side += (Pos1[3*i+axis] <= value);
        fraction = (left_side * 1.0)/count;

        debug_printf("Fraction being changed from (%.4f) --> (%.4f) ", old_fraction, fraction);

        if (old_fraction <  threshold)
        {
            if (fraction > (1-threshold))
            {
                TYPE diff0 = old_fraction;
                TYPE diff1 = (1-fraction);

                if (diff1 < diff0)
                {
                    debug_printf("  <><> Restored\n");
                    //Keep the old value's -- not able to re-order...
                    value = old_value;
                    left_side = old_left_side;
                    fraction = old_fraction;
                }
            }
        }
        else if (old_fraction > (1-threshold))
        {
            if (fraction < threshold)
            {
                TYPE diff0 = 1 - old_fraction;
                TYPE diff1 = fraction;

                if (diff1 < diff0)
                {
                    debug_printf("  <><> Restored\n");
                    //Keep the old value's -- not able to re-order...
                    value = old_value;
                    left_side = old_left_side;
                    fraction = old_fraction;
                }
            }
        }
        debug_printf("\n");
    }

    int mid = left_side;
    int left = 0;
    //printf("mid = %d ::: count = %d\n", mid, count);
    if (mid == count)
    {
        debug_printf("fds\n");
    }

    for(int i=0; i<count; i++)
    {
        if (Pos1[3*i+axis] <= value)
        {
            Pos2[3*left+0] = Pos1[3*i+0];
            Pos2[3*left+1] = Pos1[3*i+1];
            Pos2[3*left+2] = Pos1[3*i+2];
            left++;
        }
        else
        {
            Pos2[3*mid+0] = Pos1[3*i+0];
            Pos2[3*mid+1] = Pos1[3*i+1];
            Pos2[3*mid+2] = Pos1[3*i+2];
            mid++;
        }
    }

    if (left != left_side) ERROR_PRINT();
    if (mid != count) ERROR_PRINT();

    return (left_side);
}
           
void Compute_Min_Max_XYZ(TYPE *Pos,  int start_index, int end_index, TYPE *o_min_x, TYPE *o_min_y, TYPE *o_min_z, TYPE *o_max_x, TYPE *o_max_y, TYPE *o_max_z)
{
    //Data is stored in XXXXXXXX {count such elements} YYYYYYYY {count_such_elements} ZZZZZZZZZ {count_such_elements}
    int count = (end_index - start_index);

    TYPE min_x, max_x;
    TYPE min_y, max_y;
    TYPE min_z, max_z;

    min_x = max_x = Pos[3*start_index + 0*count + 0];
    min_y = max_y = Pos[3*start_index + 1*count + 0];
    min_z = max_z = Pos[3*start_index + 2*count + 0];

    for(int i=start_index; i<end_index; i++)
    {
        min_x = PCL_MIN(min_x, Pos[3*start_index + 0*count + (i - start_index)]);
        max_x = PCL_MAX(max_x, Pos[3*start_index + 0*count + (i - start_index)]);

        min_y = PCL_MIN(min_y, Pos[3*start_index + 1*count + (i - start_index)]);
        max_y = PCL_MAX(max_y, Pos[3*start_index + 1*count + (i - start_index)]);

        min_z = PCL_MIN(min_z, Pos[3*start_index + 2*count + (i - start_index)]);
        max_z = PCL_MAX(max_z, Pos[3*start_index + 2*count + (i - start_index)]);
    }

    *o_min_x = min_x;
    *o_min_y = min_y;
    *o_min_z = min_z;
    *o_max_x = max_x;
    *o_max_y = max_y;
    *o_max_z = max_z;
}

void Perform_Elaborate_Checking(Grid *grid)
{
#ifdef DETAILED_DEBUGGING
    for(int cell_id = 0; cell_id < grid->number_of_uniform_subdivisions; cell_id++)
    {
        for(int div = 0; div < grid->Number_of_kd_subdivisions[cell_id];  div++)
        {
            int start_index = grid->Range[cell_id][div + 0];
            int   end_index = grid->Range[cell_id][div + 1];

            int particles_in_this_cell = end_index - start_index;

            TYPE bdry_X_0 = grid->Bdry_X[cell_id][2*div + 0];
            TYPE bdry_X_1 = grid->Bdry_X[cell_id][2*div + 1];

            TYPE bdry_Y_0 = grid->Bdry_Y[cell_id][2*div + 0];
            TYPE bdry_Y_1 = grid->Bdry_Y[cell_id][2*div + 1];

            TYPE bdry_Z_0 = grid->Bdry_Z[cell_id][2*div + 0];
            TYPE bdry_Z_1 = grid->Bdry_Z[cell_id][2*div + 1];

            for(int j = 0; j < particles_in_this_cell; j++)
            {
                TYPE float_x = grid->Positions[cell_id][3*start_index + j + 0*particles_in_this_cell];
                TYPE float_y = grid->Positions[cell_id][3*start_index + j + 1*particles_in_this_cell];
                TYPE float_z = grid->Positions[cell_id][3*start_index + j + 2*particles_in_this_cell];

                if (float_x < bdry_X_0) 
                    ERROR_PRINT();
                if (float_x > bdry_X_1)
                    ERROR_PRINT();

                if (float_y < bdry_Y_0) 
                    ERROR_PRINT();
                if (float_y > bdry_Y_1)
                    ERROR_PRINT();

                if (float_z < bdry_Z_0) 
                    ERROR_PRINT();
                if (float_z > bdry_Z_1)
                    ERROR_PRINT();
            }
        }
    }
#endif
    
    debug_printf("PASSED KD-TREE CHECKS SUCCESSFULLY\n");

}

void Compress_Range(int *X, int *xcount)
{
    int count = *xcount;

    int i = 0;
    int j = 0;

    for(int i=0; i<count; i++)
    {
        if (X[2*i + 0] != X[2*i + 1])
        {
            X[2*j + 0] = X[2*i + 0];
            X[2*j + 1] = X[2*i + 1];
            j++;
        }
    }

    if (count != j)
    {
        *xcount = j;
    }
}


void Compute_KD_Tree(int threadid, TYPE *Pos, int number_of_particles, Grid *grid, unsigned char *Required)
{
    //The function actually performs the uniform subdivisions,
    //followed by non-uniform (Kd-tree) for each of the uniform cells
    //owned by the specific node (global_variable: node_id)...

    TYPE *Min = grid->Min;
    TYPE *Max = grid->Max;
    TYPE *Extent = grid->Extent;

    int dimx = grid->dimx;
    int dimy = grid->dimy;
    int dimz = grid->dimz;

    int dimxy = dimx * dimy;

    int dimxx = (dimx - 1);
    int dimyy = (dimy - 1);
    int dimzz = (dimz - 1);

    int total_number_of_cells = grid->number_of_uniform_subdivisions;
    int *Count = grid->Count_Per_Thread[threadid];

    for(int k=0; k<total_number_of_cells; k++) Count[k] = 0;

    int particles_per_thread = (number_of_particles % nthreads) ? (number_of_particles/nthreads+1): (number_of_particles/nthreads);
    int starting_index = particles_per_thread * threadid;
    int ending_index = starting_index + particles_per_thread;

    if (starting_index > number_of_particles) starting_index = number_of_particles;
    if (ending_index > number_of_particles) ending_index = number_of_particles;

    for(int i=starting_index; i<ending_index; i++)
    {
        TYPE float_x = Pos[3*i + 0];
        TYPE float_y = Pos[3*i + 1];
        TYPE float_z = Pos[3*i + 2];
            
        int cell_x = ((float_x - Min[0]) * dimx)/Extent[0];
        int cell_y = ((float_y - Min[1]) * dimy)/Extent[1];
        int cell_z = ((float_z - Min[2]) * dimz)/Extent[2];

        CLAMP_BELOW(cell_x, 0); CLAMP_ABOVE(cell_x, (dimxx));
        CLAMP_BELOW(cell_y, 0); CLAMP_ABOVE(cell_y, (dimyy));
        CLAMP_BELOW(cell_z, 0); CLAMP_ABOVE(cell_z, (dimzz));
        
        int cell_id = GET_CELL_INDEX(cell_x, cell_y, cell_z);
        Count[cell_id]++;

        if (!Required[cell_id]) 
        {
            printf("node_id = %d ::: i = %d\n", node_id, i);
            ERROR_PRINT();
        }
    }
    

    int cells_per_thread = (total_number_of_cells % nthreads) ? (total_number_of_cells/nthreads + 1) : (total_number_of_cells/nthreads);
    int starting_cell_index = threadid * cells_per_thread;
    int ending_cell_index = starting_cell_index +  cells_per_thread;

    if (starting_cell_index > total_number_of_cells) starting_cell_index = total_number_of_cells;
    if (  ending_cell_index > total_number_of_cells)   ending_cell_index = total_number_of_cells;

    MY_BARRIER(threadid);

    size_t sz = 0;
    for(int cell_id = starting_cell_index; cell_id < ending_cell_index; cell_id++)
    {
        int prev_sum = 0;
        for(int thr=0; thr < nthreads; thr++)
        {
            int new_sum = prev_sum + grid->Count_Per_Thread[thr][cell_id];
            grid->Count_Per_Thread[thr][cell_id] = prev_sum;
            prev_sum = new_sum;
        }

        int particles_in_this_cell = prev_sum;

        grid->Count_Per_Cell[cell_id]  = particles_in_this_cell;
        sz += DIMENSIONS * particles_in_this_cell  * sizeof(TYPE); //XXXX YYYY ZZZZ
    }

    {
        unsigned char *temp_memory = (unsigned char *)my_malloc(sz);
        unsigned char *temp2_memory = temp_memory;
        
        for(int cell_id = starting_cell_index; cell_id < ending_cell_index; cell_id++)
        {
            int particles_in_this_cell = grid->Count_Per_Cell[cell_id];
            
            grid->Positions[cell_id] = (TYPE *)(temp2_memory);  temp2_memory += (DIMENSIONS * particles_in_this_cell  * sizeof(TYPE));
        }

        if ((temp2_memory - temp_memory) != sz) ERROR_PRINT();
    }

    MY_BARRIER(threadid);

    for(int i=starting_index; i<ending_index; i++)
    {
        TYPE float_x = Pos[3*i + 0];
        TYPE float_y = Pos[3*i + 1];
        TYPE float_z = Pos[3*i + 2];
            
        int cell_x = ((float_x - Min[0]) * dimx)/Extent[0];
        int cell_y = ((float_y - Min[1]) * dimy)/Extent[1];
        int cell_z = ((float_z - Min[2]) * dimz)/Extent[2];

        CLAMP_BELOW(cell_x, 0); CLAMP_ABOVE(cell_x, (dimxx));
        CLAMP_BELOW(cell_y, 0); CLAMP_ABOVE(cell_y, (dimyy));
        CLAMP_BELOW(cell_z, 0); CLAMP_ABOVE(cell_z, (dimzz));
        
        int cell_id = GET_CELL_INDEX(cell_x, cell_y, cell_z);
        int index_to_write = grid->Count_Per_Thread[threadid][cell_id];

        int particles_in_this_cell = grid->Count_Per_Cell[cell_id];

#if 0
#else
        grid->Positions[cell_id][3*index_to_write + 0] = float_x;
        grid->Positions[cell_id][3*index_to_write + 1] = float_y;
        grid->Positions[cell_id][3*index_to_write + 2] = float_z;
#endif

        (grid->Count_Per_Thread[threadid][cell_id])++;
    }
    MY_BARRIER(threadid);


#if 1
    if (threadid == 0)
    {
        int summ = 0; for(int k=0; k<total_number_of_cells; k++) summ += (grid->Count_Per_Cell[k]);
        if (summ != number_of_particles) ERROR_PRINT();

        //for(int k=0; k<total_number_of_cells; k++) printf(" k = %d ::: Particles = %d\n", k, grid->Count_Per_Cell[k]);

        debug_printf("CHECKING SUCCESSFUL\n");
    }
#endif

    int start_cell = -1;
    int end_cell = -1;

    int particles_per_cell = number_of_particles/nthreads;
    {
        int summ = 0;
        int cell_id = 0;

        for(cell_id = 0; cell_id <=total_number_of_cells; cell_id++)
        {
            if (summ >= starting_index) 
            {
                start_cell = cell_id;
                break;
            }
            else summ += (grid->Count_Per_Cell[cell_id]);
        }

        for( ; cell_id <= total_number_of_cells; cell_id++)
        {
            if (summ >= ending_index)
            {
                end_cell = cell_id-1;
                break;
            }
            else summ += (grid->Count_Per_Cell[cell_id]);
        }
    }


    if (node_id == (nnodes-1))
    {
        if (threadid == (nthreads -1))
        {
            if (end_cell != (total_number_of_cells-1))
            {
                end_cell = (total_number_of_cells-1);
            }
        }
    }

    grid->Start_Cell[threadid] = start_cell;
    grid->End_Cell[threadid] = end_cell;

    if ((start_cell == -1) || (end_cell == -1)) 
    {
        printf("<%d> :::: number_of_particles = %d\n", node_id, number_of_particles); fflush(stdout);
        ERROR_PRINT();
    }

#if 1
    MY_BARRIER(threadid);
    if (threadid == 0)
    {
        for(int tid=0; tid < nthreads; tid++)
        {
            int points_for_this_thread = 0;
            for(int k=grid->Start_Cell[tid]; k<=grid->End_Cell[tid]; k++) points_for_this_thread += grid->Count_Per_Cell[k];
            debug_printf("tid = %2d ::: start_cell = %7d ::: end_cell = %7d ::: total_cells = %7d ::: points_for_this_thread = %d\n",  
                    tid, grid->Start_Cell[tid], grid->End_Cell[tid], total_number_of_cells, points_for_this_thread);
        }
    }
#endif


//////////////////////////////////////////////////////////////////////////////////////
//Phase-II Let's perform KD-Tree subdivision for each uniformly formed cell...
///////////////////////////////////////////////////////////////////////////////////////

    int threshold_particles_per_cell =  192;

    {
        MY_BARRIER(threadid);
        if (threadid == 0)
        {
            TYPE *Temp1 = (TYPE *)my_malloc(2 * nthreads * threshold_particles_per_cell * 3 * sizeof(TYPE));
            TYPE *Temp2 = Temp1;
            global_Aligned_Buffer = (TYPE **)my_malloc(nthreads * sizeof(TYPE *));
            for(int pp=0; pp<nthreads; pp++)
            {
                global_Aligned_Buffer[pp] = Temp2;
                Temp2 += 2 * 3 * threshold_particles_per_cell;
            }

            if ( (Temp2 - Temp1) != (2 * nthreads * threshold_particles_per_cell * 3)) ERROR_PRINT();
        }
        MY_BARRIER(threadid);
    }

    int cumm_particles = 0;

    int max_particles_for_this_thread = 0;
    for(int cell_id = 0; cell_id < start_cell; cell_id++) cumm_particles += grid->Count_Per_Cell[cell_id];
    for(int cell_id = start_cell; cell_id <= end_cell; cell_id++) max_particles_for_this_thread = PCL_MAX (max_particles_for_this_thread,  grid->Count_Per_Cell[cell_id]);

    TYPE *Temp_Pos = (TYPE *)my_another_malloc(max_particles_for_this_thread * 3 * sizeof(TYPE));

    {
        int maximum_number_of_particles = 0;
        int maximum_number_of_ranges = 0;
        debug_printf("node_id = %d ::: threadid = %d ::: start_cell = %d ::: end_cell = %d\n", node_id, threadid, start_cell, end_cell);

        for(int cell_id = start_cell; cell_id <= end_cell; cell_id++)
        {
            if (!Required[cell_id]) continue;

            int particles_in_current_cell = grid->Count_Per_Cell[cell_id];
            int *Ranges11 = Ranges1[threadid];
            int *Ranges22 = Ranges2[threadid];
            int number_of_ranges = 1; Ranges11[0] = 0; Ranges11[1] = particles_in_current_cell;
            int subdivision_required = 1;
            int iterations = 0;

            TYPE *Pos1 = grid->Positions[cell_id];

            TYPE *Pos2 = Temp_Pos;

            while (subdivision_required)
            {
                subdivision_required = 0;
                iterations++;
                int l = 0;
                for(int k=0; k<number_of_ranges; k++)
                {
                    int range_min = Ranges11[2*k + 0];
                    int range_max = Ranges11[2*k + 1];
                    if ( (range_max - range_min) > threshold_particles_per_cell)
                    {
                        subdivision_required++;
                
                        int left_side = Find_Min_Max_And_Separating_Axis(Pos1 + 3*range_min, (range_max-range_min), Pos2 + 3*range_min);

                        Ranges22[2*l+0] = range_min;
                        Ranges22[2*l+1] = range_min + left_side;
                        Ranges22[2*l+2] = range_min + left_side;
                        Ranges22[2*l+3] = range_max;
                        l+=2;
                    }
                    else
                    {
                        Ranges22[2*l+0] = range_min;
                        Ranges22[2*l+1] = range_max;
                        for(int j=range_min; j<range_max; j++)
                        {
                            Pos2[3*j+0] = Pos1[3*j+0];
                            Pos2[3*j+1] = Pos1[3*j+1];
                            Pos2[3*j+2] = Pos1[3*j+2];
                        }
                        l++;
                    }

                    if ( (2*l+4) >= Ranges12_Max_Size[threadid])
                    {
                        debug_printf("Realloc Called ::: Size Being Increased From (%d) --> ", Ranges12_Max_Size[threadid]);
                        Ranges12_Max_Size[threadid] *= 2;
                        debug_printf(" (%d)\n", Ranges12_Max_Size[threadid]);
                        int *temp_int = (int *)my_malloc(Ranges12_Max_Size[threadid] * 2 * sizeof(int));
                        int *temp_int1 = temp_int;
                        int *temp_int2 = temp_int + Ranges12_Max_Size[threadid];

                        for(int i=0; i<(2*number_of_ranges); i++) temp_int1[i] = Ranges11[i];
                        for(int i=0; i<(2*l); i++) temp_int2[i] = Ranges22[i];

                        my_free(Ranges11);
                        my_free(Ranges22);
                        Ranges1[threadid] = temp_int1;
                        Ranges2[threadid] = temp_int2;

                        Ranges11 = temp_int1;
                        Ranges22 = temp_int2;
                    }
                }

                number_of_ranges = l;
                SWAP_INT_ADDR(Ranges11, Ranges22);
                SWAP_TYPE_ADDR(Pos1, Pos2);
            }

            Compress_Range(Ranges11, &number_of_ranges);
            grid->Number_of_kd_subdivisions[cell_id] = number_of_ranges;


            TYPE *Dst_Pos = grid->Positions[cell_id];

            if ( (iterations % 2) == 0)
            {
                //Copy data from Pos1 to Pos2...

                //assert (Dst_Pos == Pos1)...
                if (Pos1 != Dst_Pos) ERROR_PRINT();

                for(int i=0; i<particles_in_current_cell; i++)
                {
                    Pos2[3*i + 0] = Pos1[3*i + 0];
                    Pos2[3*i + 1] = Pos1[3*i + 1];
                    Pos2[3*i + 2] = Pos1[3*i + 2];
                }
            }

            if (1)
            {
                //Data sits in Temp_Pos
                for(int i=0; i<number_of_ranges; i++)
                {
                    int local_start = Ranges11[2*i];
                    int   local_end = Ranges11[2*i+1];
                    int particles_in_this_kd_division = local_end - local_start;

                    for(int j=local_start; j<local_end; j++)
                    {
                        Dst_Pos[3*local_start + (j - local_start)  + 0 * particles_in_this_kd_division] = Temp_Pos[3*j+0];
                        Dst_Pos[3*local_start + (j - local_start)  + 1 * particles_in_this_kd_division] = Temp_Pos[3*j+1];
                        Dst_Pos[3*local_start + (j - local_start)  + 2 * particles_in_this_kd_division] = Temp_Pos[3*j+2];
                    }
                }
            }

            maximum_number_of_particles = PCL_MAX(maximum_number_of_particles, particles_in_current_cell);
            maximum_number_of_ranges = PCL_MAX(maximum_number_of_ranges, number_of_ranges);

            sz = (1+number_of_ranges) * sizeof(int) + 6 * number_of_ranges * sizeof(TYPE);
            unsigned char *X = (unsigned char *)my_malloc(sz);
            unsigned char *XX = X;

            grid->Range[cell_id]            = (int  *)(X); X += ((1+number_of_ranges) * sizeof( int));
            grid->Bdry_X[cell_id]           = (TYPE *)(X); X += (2 * number_of_ranges * sizeof(TYPE));
            grid->Bdry_Y[cell_id]           = (TYPE *)(X); X += (2 * number_of_ranges * sizeof(TYPE));
            grid->Bdry_Z[cell_id]           = (TYPE *)(X); X += (2 * number_of_ranges * sizeof(TYPE));

            if ( (X - XX) != sz) ERROR_PRINT();

            for(int i=0; i<number_of_ranges; i++) grid->Range[cell_id][i] = Ranges11[2*i]; grid->Range[cell_id][number_of_ranges] = particles_in_current_cell;
            for(int i=0; i<number_of_ranges; i++)
            {
                int start_index = grid->Range[cell_id][i+0];
                int   end_index = grid->Range[cell_id][i+1];

                TYPE min_x, max_x, min_y, max_y, min_z, max_z;
                Compute_Min_Max_XYZ(grid->Positions[cell_id], start_index, end_index, &min_x, &min_y, &min_z, &max_x, &max_y, &max_z);

                grid->Bdry_X[cell_id][2*i + 0] = min_x;
                grid->Bdry_X[cell_id][2*i + 1] = max_x;

                grid->Bdry_Y[cell_id][2*i + 0] = min_y;
                grid->Bdry_Y[cell_id][2*i + 1] = max_y;

                grid->Bdry_Z[cell_id][2*i + 0] = min_z;
                grid->Bdry_Z[cell_id][2*i + 1] = max_z;


            }
        }

    }

//#define KD_TREE_STATISTICS

#ifdef KD_TREE_STATISTICS

    {
        MY_BARRIER(threadid);
        if (threadid == 0)
        {
            int zero_cells = 0;
            int sum_particles = 0;
            int total_cells = 0;
            int max_particles = 0;
            int min_particles = number_of_particles + 1;
            for(int cell_id = 0; cell_id < grid->number_of_uniform_subdivisions; cell_id++)
            {
                //if (!grid->kd_subdivisions[cell_id]) debug_printf("cell_id = %d\n", cell_id);
                for(int div = 0; div < grid->Number_of_kd_subdivisions[cell_id];  div++, total_cells++)
                {
                    int particles_in_this_cell = grid->Range[cell_id][div+1] - grid->Range[cell_id][div+0];
                    //debug_printf("%d ::: %d ::: %d\n", cell_id, div, particles_in_this_cell);
                    zero_cells += (particles_in_this_cell == 0);
                    sum_particles += particles_in_this_cell;
                    min_particles = PCL_MIN(min_particles, particles_in_this_cell);
                    max_particles = PCL_MAX(max_particles, particles_in_this_cell);
                }
            }

            if (sum_particles != number_of_particles);
            PRINT_BLUE
            printf("node_id = %d ::: number_of_particles = %d ::: total_cells = %d ::: cells_with_zero_particles = %d ::: min_particles = %d ::: max_particles = %d ::: avg_particles = %.2lf \n", 
                    node_id, number_of_particles, total_cells, zero_cells, min_particles, max_particles, (sum_particles * 1.0)/total_cells);
            PRINT_BLACK

            Perform_Elaborate_Checking(grid);
    
            //Spit_KD_Tree_Into_File(grid, "kdtree.bin");
        }
    }
#endif

    my_another_free(Temp_Pos, (max_particles_for_this_thread * 3 * sizeof(TYPE)));
}
///////////////////////////////////////////////////////////////////////////////////////////////////





__declspec(align(64))
static unsigned char Remaining[][32] = {
  {   0,   0,   0,   0,    0,   0,   0,   0,    0,   0,   0,   0,    0,   0,   0,   0,    0,   0,   0,   0,    0,   0,   0,   0,    0,   0,   0,   0,  0, 0, 0, 0, },
  { 255, 255, 255, 255,    0,   0,   0,   0,    0,   0,   0,   0,    0,   0,   0,   0,    0,   0,   0,   0,    0,   0,   0,   0,    0,   0,   0,   0,  0, 0, 0, 0, },
  { 255, 255, 255, 255,  255, 255, 255, 255,    0,   0,   0,   0,    0,   0,   0,   0,    0,   0,   0,   0,    0,   0,   0,   0,    0,   0,   0,   0,  0, 0, 0, 0, },
  { 255, 255, 255, 255,  255, 255, 255, 255,  255, 255, 255, 255,    0,   0,   0,   0,    0,   0,   0,   0,    0,   0,   0,   0,    0,   0,   0,   0,  0, 0, 0, 0, },
  { 255, 255, 255, 255,  255, 255, 255, 255,  255, 255, 255, 255,  255, 255, 255, 255,    0,   0,   0,   0,    0,   0,   0,   0,    0,   0,   0,   0,  0, 0, 0, 0, },
  { 255, 255, 255, 255,  255, 255, 255, 255,  255, 255, 255, 255,  255, 255, 255, 255,  255, 255, 255, 255,    0,   0,   0,   0,    0,   0,   0,   0,  0, 0, 0, 0, },
  { 255, 255, 255, 255,  255, 255, 255, 255,  255, 255, 255, 255,  255, 255, 255, 255,  255, 255, 255, 255,  255, 255, 255, 255,    0,   0,   0,   0,  0, 0, 0, 0, },
  { 255, 255, 255, 255,  255, 255, 255, 255,  255, 255, 255, 255,  255, 255, 255, 255,  255, 255, 255, 255,  255, 255, 255, 255,  255, 255, 255, 255,  0, 0, 0, 0, },
  { 255, 255, 255, 255,  255, 255, 255, 255,  255, 255, 255, 255,  255, 255, 255, 255,  255, 255, 255, 255,  255, 255, 255, 255,  255, 255, 255, 255,  255, 255, 255, 255, },
};




TYPE my_log(TYPE x)
{
    return logf(x);
}

TYPE my_exp(TYPE x)
{
    return expf(x);
}



#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

    
void Compute_Min_Max_Dist_Sqr(TYPE *Range0_X, TYPE *Range0_Y, TYPE *Range0_Z, TYPE *Range1_X, TYPE *Range1_Y, TYPE *Range1_Z, 
        TYPE *o_min_dst_sqr, TYPE *o_max_dst_sqr)
{

    TYPE min_dst_sqr = 0;
    TYPE max_dst_sqr = 0;

    if (Range0_X[0] > Range1_X[1]) min_dst_sqr += (Range0_X[0] - Range1_X[1])*(Range0_X[0] - Range1_X[1]);
    if (Range1_X[0] > Range0_X[1]) min_dst_sqr += (Range1_X[0] - Range0_X[1])*(Range1_X[0] - Range0_X[1]);

    if (Range0_Y[0] > Range1_Y[1]) min_dst_sqr += (Range0_Y[0] - Range1_Y[1])*(Range0_Y[0] - Range1_Y[1]);
    if (Range1_Y[0] > Range0_Y[1]) min_dst_sqr += (Range1_Y[0] - Range0_Y[1])*(Range1_Y[0] - Range0_Y[1]);

    if (Range0_Z[0] > Range1_Z[1]) min_dst_sqr += (Range0_Z[0] - Range1_Z[1])*(Range0_Z[0] - Range1_Z[1]);
    if (Range1_Z[0] > Range0_Z[1]) min_dst_sqr += (Range1_Z[0] - Range0_Z[1])*(Range1_Z[0] - Range0_Z[1]);


    TYPE xmin = MIN(Range0_X[0], Range1_X[0]); TYPE xmax = MAX(Range0_X[1], Range1_X[1]);
    TYPE ymin = MIN(Range0_Y[0], Range1_Y[0]); TYPE ymax = MAX(Range0_Y[1], Range1_Y[1]);
    TYPE zmin = MIN(Range0_Z[0], Range1_Z[0]); TYPE zmax = MAX(Range0_Z[1], Range1_Z[1]);

    max_dst_sqr = (xmax - xmin) * (xmax - xmin) + (ymax - ymin) * (ymax - ymin) + (zmax - zmin) * (zmax - zmin);


    *o_min_dst_sqr = min_dst_sqr;
    *o_max_dst_sqr = max_dst_sqr;
}

void Compute_min_max_bin_id(TYPE *BinCorners2, int nrbin, TYPE min_dst_sqr, TYPE max_dst_sqr, int *o_min_bin_id, int *o_max_bin_id)
{

    int min_bin_id = -95123, max_bin_id = -95123;

    {
        for(int bin_id = 1; bin_id < (2 + nrbin); bin_id++)
        {
            if (BinCorners2[bin_id] > min_dst_sqr)
            {
                min_bin_id = bin_id - 1;
                break;
            }
        }


        if (min_bin_id == -95123)
        {
            min_bin_id = max_bin_id = (1+nrbin);
        }
        else
        {
            for(int bin_id = min_bin_id; bin_id < (2+nrbin); bin_id++)
            {
                if (BinCorners2[bin_id] > max_dst_sqr)
                {
                    max_bin_id = bin_id - 1;
                    break;
                }
            }
            if (max_bin_id == -95123)
            {
                max_bin_id = 1+nrbin;
            }
        }
    }

    *o_min_bin_id = min_bin_id;
    *o_max_bin_id = max_bin_id;

}


__attribute__((noinline))
DECLSPEC_TARGET_MIC
void Compute_Distance_And_Populate_Hist_1(TYPE *Pos0, int count0, TYPE *Pos1, int count1, TYPE *BinCorners2, int nrbin, 
        unsigned int *DD_int0, unsigned int *DD_int1, int min_bin_id, int max_bin_id, TYPE *Range1_X, TYPE *Range1_Y, TYPE *Range1_Z, int threadid)
{

    int self = (Pos0 == Pos1);
    int total_number_of_interactions = (self) ? ((count0 *(count1 - 1))/2) : (count0 * count1);
    DD_int0[min_bin_id] += total_number_of_interactions;
    global_Easy[8*threadid] += total_number_of_interactions;
}


__attribute__((noinline))
DECLSPEC_TARGET_MIC
void Compute_Distance_And_Populate_Hist_3(TYPE *Pos0, int count0, TYPE *Pos1, int count1, TYPE *BinCorners2, int nrbin, 
        unsigned int *DD_int0, unsigned int *DD_int1, int min_bin_id, int max_bin_id, TYPE *Range1_X, TYPE *Range1_Y, TYPE *Range1_Z, int threadid)
{

    TYPE separation_element1 = BinCorners2[min_bin_id+1];
    TYPE separation_element2 = BinCorners2[min_bin_id+2];

    __m256 xmm_separation_element1 = _mm256_set1_ps(separation_element1);
    __m256 xmm_separation_element2 = _mm256_set1_ps(separation_element2);

    __m128i xmm_result1 = _mm_set1_epi32(0);
    __m128i xmm_result2 = _mm_set1_epi32(0);

    int self = 0;
    if (Pos0 == Pos1) self = 1;

    if (self) 
    {
        for(int i=0; i<count0; i++)
        {
            int starting_index = (self) ? (i+1) : 0;
              
            __m256 xmm_x0 = _mm256_set1_ps(*(Pos0 + i + 0*count0));
            __m256 xmm_y0 = _mm256_set1_ps(*(Pos0 + i + 1*count0));
            __m256 xmm_z0 = _mm256_set1_ps(*(Pos0 + i + 2*count0));
              
            int particles_left = count1 - starting_index;
            int particles_left_prime = ((particles_left >> 3) << 3);
            int count1_prime = starting_index + particles_left_prime;

            for(int j=starting_index; j < count1_prime; j+=SIMD_WIDTH)
            {
                __m256 xmm_x1 = _mm256_loadu_ps(Pos1 + j + 0*count1);
                __m256 xmm_y1 = _mm256_loadu_ps(Pos1 + j + 1*count1);
                __m256 xmm_z1 = _mm256_loadu_ps(Pos1 + j + 2*count1);

                __m256 xmm_diff_X = _mm256_sub_ps(xmm_x0, xmm_x1);
                __m256 xmm_diff_Y = _mm256_sub_ps(xmm_y0, xmm_y1);
                __m256 xmm_diff_Z = _mm256_sub_ps(xmm_z0, xmm_z1);

                __m256 xmm_norm_2 = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(xmm_diff_X, xmm_diff_X), _mm256_mul_ps(xmm_diff_Y, xmm_diff_Y)), _mm256_mul_ps(xmm_diff_Z, xmm_diff_Z));

                __m256i t = _mm256_castps_si256(_mm256_cmp_ps(xmm_norm_2, xmm_separation_element1, _CMP_LT_OS));
                xmm_result1 = _mm_sub_epi32(xmm_result1, _mm256_castsi256_si128(t));
                xmm_result1 = _mm_sub_epi32(xmm_result1, _mm256_extractf128_si256(t, 1));
                t = _mm256_castps_si256(_mm256_cmp_ps(xmm_norm_2, xmm_separation_element2, _CMP_LT_OS));
                xmm_result2 = _mm_sub_epi32(xmm_result2, _mm256_castsi256_si128(t));
                xmm_result2 = _mm_sub_epi32(xmm_result2, _mm256_extractf128_si256(t, 1));
            }

            {
                int j = count1_prime;
                int remaining = count1 - count1_prime;
                __m256 xmm_anding = _mm256_load_ps((float *)(Remaining[remaining]));

                __m256 xmm_x1 = _mm256_loadu_ps(Pos1 + j + 0*count1);
                __m256 xmm_y1 = _mm256_loadu_ps(Pos1 + j + 1*count1);
                __m256 xmm_z1 = _mm256_loadu_ps(Pos1 + j + 2*count1);

                __m256 xmm_diff_X = _mm256_sub_ps(xmm_x0, xmm_x1);
                __m256 xmm_diff_Y = _mm256_sub_ps(xmm_y0, xmm_y1);
                __m256 xmm_diff_Z = _mm256_sub_ps(xmm_z0, xmm_z1);

                __m256 xmm_norm_2 = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(xmm_diff_X, xmm_diff_X), _mm256_mul_ps(xmm_diff_Y, xmm_diff_Y)), _mm256_mul_ps(xmm_diff_Z, xmm_diff_Z));

                __m256i t = _mm256_castps_si256(_mm256_and_ps(xmm_anding, _mm256_cmp_ps(xmm_norm_2, xmm_separation_element1, _CMP_LT_OS)));
                xmm_result1 = _mm_sub_epi32(xmm_result1, _mm256_castsi256_si128(t));
                xmm_result1 = _mm_sub_epi32(xmm_result1, _mm256_extractf128_si256(t, 1));
                t = _mm256_castps_si256(_mm256_and_ps(xmm_anding, _mm256_cmp_ps(xmm_norm_2, xmm_separation_element2, _CMP_LT_OS)));
                xmm_result2 = _mm_sub_epi32(xmm_result2, _mm256_castsi256_si128(t));
                xmm_result2 = _mm_sub_epi32(xmm_result2, _mm256_extractf128_si256(t, 1));
          
            }
        }
    }
    else 
    {
      
        for(int i=0; i<count0; i++)
        {
            __m256 xmm_x0 = _mm256_set1_ps(*(Pos0 + i + 0*count0));
            __m256 xmm_y0 = _mm256_set1_ps(*(Pos0 + i + 1*count0));
            __m256 xmm_z0 = _mm256_set1_ps(*(Pos0 + i + 2*count0));
              
            for(int j=0; j < count1; j+=SIMD_WIDTH)
            {
                __m256 xmm_x1 = _mm256_load_ps(Pos1 + j + 0*count1);
                __m256 xmm_y1 = _mm256_load_ps(Pos1 + j + 1*count1);
                __m256 xmm_z1 = _mm256_load_ps(Pos1 + j + 2*count1);

                __m256 xmm_diff_X = _mm256_sub_ps(xmm_x0, xmm_x1);
                __m256 xmm_diff_Y = _mm256_sub_ps(xmm_y0, xmm_y1);
                __m256 xmm_diff_Z = _mm256_sub_ps(xmm_z0, xmm_z1);

                __m256 xmm_norm_2 = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(xmm_diff_X, xmm_diff_X), _mm256_mul_ps(xmm_diff_Y, xmm_diff_Y)), _mm256_mul_ps(xmm_diff_Z, xmm_diff_Z));

              
                __m256i t = _mm256_castps_si256(_mm256_cmp_ps(xmm_norm_2, xmm_separation_element1, _CMP_LT_OS));
                xmm_result1 = _mm_sub_epi32(xmm_result1, _mm256_castsi256_si128(t));
                xmm_result1 = _mm_sub_epi32(xmm_result1, _mm256_extractf128_si256(t, 1));
                t = _mm256_castps_si256(_mm256_cmp_ps(xmm_norm_2, xmm_separation_element2, _CMP_LT_OS));
                xmm_result2 = _mm_sub_epi32(xmm_result2, _mm256_castsi256_si128(t));
                xmm_result2 = _mm_sub_epi32(xmm_result2, _mm256_extractf128_si256(t, 1));
            }
        }
    }

    {
        __declspec (align(64)) int Temp[8];
        _mm_store_si128((__m128i *)(Temp+0), xmm_result1);
        _mm_store_si128((__m128i *)(Temp+4), xmm_result2);
            
        int total =  (self) ? ((count0*(count1-1))/2) : (count0 * count1);
        int sum1 = Temp[0] + Temp[1] + Temp[2] + Temp[3];
        int sum2 = Temp[4] + Temp[5] + Temp[6] + Temp[7];

        DD_int0[min_bin_id+0] += sum1;
        DD_int0[min_bin_id+1] += sum2-sum1;
        DD_int0[min_bin_id+2] += (total - sum2);
    }
}

__attribute__((noinline))
DECLSPEC_TARGET_MIC
void Compute_Distance_And_Populate_Hist_2(TYPE *Pos0, int count0, TYPE *Pos1, int count1, TYPE *BinCorners2, int nrbin, 
        unsigned int *DD_int0, unsigned int *DD_int1, int min_bin_id, int max_bin_id, TYPE *Range1_X, TYPE *Range1_Y, TYPE *Range1_Z, int threadid)
{

    TYPE separation_element = BinCorners2[min_bin_id+1];
    __m256 xmm_separation_element = _mm256_set1_ps(separation_element);

    int self = 0;
    if (Pos0 == Pos1) self = 1;

    __m128i xmm_result_0 = _mm_set1_epi32(0);

    TYPE Range0_X[2], Range0_Y[2], Range0_Z[2];

    int to_be_subtracted = 0;

    if (self)
    {
        unsigned int something_to_check0 = DD_int0[min_bin_id + 2];
        unsigned int something_to_check1 = DD_int0[min_bin_id + 2];
        Compute_Distance_And_Populate_Hist_3(Pos0, count0, Pos1, count1, BinCorners2, nrbin, DD_int0, DD_int1, min_bin_id, max_bin_id+1, Range1_X, Range1_Y, Range1_Z, threadid);
        if (something_to_check0 != DD_int0[min_bin_id + 2]) ERROR_PRINT();
        if (something_to_check1 != DD_int0[min_bin_id + 2]) ERROR_PRINT();
    }

    else
    {
        for(int i=0; i<count0; i++)
        {
        
            __m256 xmm_x0_0 = _mm256_set1_ps(*(Pos0 + i + 0 + 0*count0));
            __m256 xmm_y0_0 = _mm256_set1_ps(*(Pos0 + i + 0 + 1*count0));
            __m256 xmm_z0_0 = _mm256_set1_ps(*(Pos0 + i + 0 + 2*count0));

            for(int j=0; j < count1; j+=SIMD_WIDTH)
            {
                __m256 xmm_x1 = _mm256_load_ps(Pos1 + j + 0*count1);
                __m256 xmm_y1 = _mm256_load_ps(Pos1 + j + 1*count1);
                __m256 xmm_z1 = _mm256_load_ps(Pos1 + j + 2*count1);

                __m256 xmm_diff_X_0 = _mm256_sub_ps(xmm_x0_0, xmm_x1);
                __m256 xmm_diff_Y_0 = _mm256_sub_ps(xmm_y0_0, xmm_y1);
                __m256 xmm_diff_Z_0 = _mm256_sub_ps(xmm_z0_0, xmm_z1);

                __m256 xmm_norm_2_0 = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(xmm_diff_X_0, xmm_diff_X_0), _mm256_mul_ps(xmm_diff_Y_0, xmm_diff_Y_0)), _mm256_mul_ps(xmm_diff_Z_0, xmm_diff_Z_0));

                __m256i t = _mm256_castps_si256(_mm256_cmp_ps(xmm_norm_2_0, xmm_separation_element, _CMP_LT_OS));
                xmm_result_0 = _mm_sub_epi32(xmm_result_0, _mm256_castsi256_si128(t));
                xmm_result_0 = _mm_sub_epi32(xmm_result_0, _mm256_extractf128_si256(t, 1));
            }
        }

        __m128i xmm_result = xmm_result_0;
        __declspec (align(64)) int Temp[4];
        _mm_store_si128((__m128i *)(Temp), xmm_result);
        
        int total =  (self) ? ((count0*(count1-1))/2) : (count0 * count1);
        int sum = Temp[0] + Temp[1] + Temp[2] + Temp[3];

        total -= to_be_subtracted;

        DD_int0[min_bin_id] += sum;
        DD_int0[min_bin_id+1] += (total - sum);
    }
}


__attribute__((noinline))
DECLSPEC_TARGET_MIC
void Compute_Distance_And_Populate_Hist_4(TYPE *Pos0, int count0, TYPE *Pos1, int count1, TYPE *BinCorners2, int nrbin, 
        unsigned int *DD_int0, unsigned int *DD_int1, int min_bin_id, int max_bin_id, TYPE *Range1_X, TYPE *Range1_Y, TYPE *Range1_Z, int threadid)
{

    TYPE separation_element1 = BinCorners2[min_bin_id+1];
    TYPE separation_element2 = BinCorners2[min_bin_id+2];
    TYPE separation_element3 = BinCorners2[min_bin_id+3];

    __m256 xmm_separation_element1 = _mm256_set1_ps(separation_element1);
    __m256 xmm_separation_element2 = _mm256_set1_ps(separation_element2);
    __m256 xmm_separation_element3 = _mm256_set1_ps(separation_element3);

    __m128i xmm_result1 = _mm_set1_epi32(0);
    __m128i xmm_result2 = _mm_set1_epi32(0);
    __m128i xmm_result3 = _mm_set1_epi32(0);

    int self = 0;
    if (Pos0 == Pos1) self = 1;

    if (self) 
    {
        for(int i=0; i<count0; i++)
        {
            int starting_index = i+1;
              
            __m256 xmm_x0 = _mm256_set1_ps(*(Pos0 + i + 0*count0));
            __m256 xmm_y0 = _mm256_set1_ps(*(Pos0 + i + 1*count0));
            __m256 xmm_z0 = _mm256_set1_ps(*(Pos0 + i + 2*count0));
              
            int particles_left = count1 - starting_index;
            int particles_left_prime = ((particles_left >> 3) << 3);
            int count1_prime = starting_index + particles_left_prime;

            for(int j=starting_index; j < count1_prime; j+=SIMD_WIDTH)
            {
                __m256 xmm_x1 = _mm256_loadu_ps(Pos1 + j + 0*count1);
                __m256 xmm_y1 = _mm256_loadu_ps(Pos1 + j + 1*count1);
                __m256 xmm_z1 = _mm256_loadu_ps(Pos1 + j + 2*count1);

                __m256 xmm_diff_X = _mm256_sub_ps(xmm_x0, xmm_x1);
                __m256 xmm_diff_Y = _mm256_sub_ps(xmm_y0, xmm_y1);
                __m256 xmm_diff_Z = _mm256_sub_ps(xmm_z0, xmm_z1);

                __m256 xmm_norm_2 = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(xmm_diff_X, xmm_diff_X), _mm256_mul_ps(xmm_diff_Y, xmm_diff_Y)), _mm256_mul_ps(xmm_diff_Z, xmm_diff_Z));

                __m256i t = _mm256_castps_si256(_mm256_cmp_ps(xmm_norm_2, xmm_separation_element1, _CMP_LT_OS));
                xmm_result1 = _mm_sub_epi32(xmm_result1, _mm256_castsi256_si128(t));
                xmm_result1 = _mm_sub_epi32(xmm_result1, _mm256_extractf128_si256(t, 1));
                t = _mm256_castps_si256(_mm256_cmp_ps(xmm_norm_2, xmm_separation_element2, _CMP_LT_OS));
                xmm_result2 = _mm_sub_epi32(xmm_result2, _mm256_castsi256_si128(t));
                xmm_result2 = _mm_sub_epi32(xmm_result2, _mm256_extractf128_si256(t, 1));
                t = _mm256_castps_si256(_mm256_cmp_ps(xmm_norm_2, xmm_separation_element3, _CMP_LT_OS));
                xmm_result3 = _mm_sub_epi32(xmm_result3, _mm256_castsi256_si128(t));
                xmm_result3 = _mm_sub_epi32(xmm_result3, _mm256_extractf128_si256(t, 1));
            }

            {
              
                int j = count1_prime;
                int remaining = count1 - count1_prime;
                __m256 xmm_anding = _mm256_load_ps((float *)(Remaining[remaining]));

                __m256 xmm_x1 = _mm256_loadu_ps(Pos1 + j + 0*count1);
                __m256 xmm_y1 = _mm256_loadu_ps(Pos1 + j + 1*count1);
                __m256 xmm_z1 = _mm256_loadu_ps(Pos1 + j + 2*count1);

                __m256 xmm_diff_X = _mm256_sub_ps(xmm_x0, xmm_x1);
                __m256 xmm_diff_Y = _mm256_sub_ps(xmm_y0, xmm_y1);
                __m256 xmm_diff_Z = _mm256_sub_ps(xmm_z0, xmm_z1);

                __m256 xmm_norm_2 = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(xmm_diff_X, xmm_diff_X), _mm256_mul_ps(xmm_diff_Y, xmm_diff_Y)), _mm256_mul_ps(xmm_diff_Z, xmm_diff_Z));

                __m256i t = _mm256_castps_si256(_mm256_and_ps(xmm_anding, _mm256_cmp_ps(xmm_norm_2, xmm_separation_element1, _CMP_LT_OS)));
                xmm_result1 = _mm_sub_epi32(xmm_result1, _mm256_castsi256_si128(t));
                xmm_result1 = _mm_sub_epi32(xmm_result1, _mm256_extractf128_si256(t, 1));
                t = _mm256_castps_si256(_mm256_and_ps(xmm_anding, _mm256_cmp_ps(xmm_norm_2, xmm_separation_element2, _CMP_LT_OS)));
                xmm_result2 = _mm_sub_epi32(xmm_result2, _mm256_castsi256_si128(t));
                xmm_result2 = _mm_sub_epi32(xmm_result2, _mm256_extractf128_si256(t, 1));
                t = _mm256_castps_si256(_mm256_and_ps(xmm_anding, _mm256_cmp_ps(xmm_norm_2, xmm_separation_element3, _CMP_LT_OS)));
                xmm_result3 = _mm_sub_epi32(xmm_result3, _mm256_castsi256_si128(t));
                xmm_result3 = _mm_sub_epi32(xmm_result3, _mm256_extractf128_si256(t, 1));
            }
        }
    }
    else 
    {
      
        for(int i=0; i<count0; i++)
        {
            __m256 xmm_x0 = _mm256_set1_ps(*(Pos0 + i + 0*count0));
            __m256 xmm_y0 = _mm256_set1_ps(*(Pos0 + i + 1*count0));
            __m256 xmm_z0 = _mm256_set1_ps(*(Pos0 + i + 2*count0));
              
            for(int j=0; j < count1; j+=SIMD_WIDTH)
            {
                __m256 xmm_x1 = _mm256_load_ps(Pos1 + j + 0*count1);
                __m256 xmm_y1 = _mm256_load_ps(Pos1 + j + 1*count1);
                __m256 xmm_z1 = _mm256_load_ps(Pos1 + j + 2*count1);

                __m256 xmm_diff_X = _mm256_sub_ps(xmm_x0, xmm_x1);
                __m256 xmm_diff_Y = _mm256_sub_ps(xmm_y0, xmm_y1);
                __m256 xmm_diff_Z = _mm256_sub_ps(xmm_z0, xmm_z1);

                __m256 xmm_norm_2 = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(xmm_diff_X, xmm_diff_X), _mm256_mul_ps(xmm_diff_Y, xmm_diff_Y)), _mm256_mul_ps(xmm_diff_Z, xmm_diff_Z));

                __m256i t = _mm256_castps_si256(_mm256_cmp_ps(xmm_norm_2, xmm_separation_element1, _CMP_LT_OS));
                xmm_result1 = _mm_sub_epi32(xmm_result1, _mm256_castsi256_si128(t));
                xmm_result1 = _mm_sub_epi32(xmm_result1, _mm256_extractf128_si256(t, 1));
                t = _mm256_castps_si256(_mm256_cmp_ps(xmm_norm_2, xmm_separation_element2, _CMP_LT_OS));
                xmm_result2 = _mm_sub_epi32(xmm_result2, _mm256_castsi256_si128(t));
                xmm_result2 = _mm_sub_epi32(xmm_result2, _mm256_extractf128_si256(t, 1));
                t = _mm256_castps_si256(_mm256_cmp_ps(xmm_norm_2, xmm_separation_element3, _CMP_LT_OS));
                xmm_result3 = _mm_sub_epi32(xmm_result3, _mm256_castsi256_si128(t));
                xmm_result3 = _mm_sub_epi32(xmm_result3, _mm256_extractf128_si256(t, 1));
          
            }
        }
    }


    {
        __declspec (align(64)) int Temp[12];
        _mm_store_si128((__m128i *)(Temp+0), xmm_result1);
        _mm_store_si128((__m128i *)(Temp+4), xmm_result2);
        _mm_store_si128((__m128i *)(Temp+8), xmm_result3);
    
            
        int total =  (self) ? ((count0*(count1-1))/2) : (count0 * count1);
        int sum1 = Temp[0] + Temp[1] + Temp[2] + Temp[3];
        int sum2 = Temp[4] + Temp[5] + Temp[6] + Temp[7];
        int sum3 = Temp[8] + Temp[9] + Temp[10] + Temp[11];

        DD_int0[min_bin_id+0] += sum1;
        DD_int0[min_bin_id+1] += sum2-sum1;
        DD_int0[min_bin_id+2] += sum3-sum2;
        DD_int0[min_bin_id+3] += (total - sum3);
    }
}


__attribute__((noinline))
DECLSPEC_TARGET_MIC
void Compute_Distance_And_Populate_Hist_N(TYPE *Pos0, int count0, TYPE *Pos1, int count1, TYPE *BinCorners2, int nrbin, 
        unsigned int *DD_int0, unsigned int *DD_int1, int min_bin_id, int max_bin_id, TYPE *Range1_X, TYPE *Range1_Y, TYPE *Range1_Z, int threadid)
{

    int self = 0;
    if (Pos0 == Pos1) self = 1;

    TYPE separation_element[HIST_BINS];
    __m256 xmm_separation_element[HIST_BINS];
    __m128i xmm_result[HIST_BINS];

    for(int k=min_bin_id; k<max_bin_id; k++)     separation_element[k-min_bin_id] = BinCorners2[k+1];
    for(int k=min_bin_id; k<max_bin_id; k++) xmm_separation_element[k-min_bin_id] = _mm256_set1_ps(separation_element[k-min_bin_id]);
    for(int k=min_bin_id; k<max_bin_id; k++) xmm_result[k-min_bin_id] = _mm_set1_epi32(0);

    int number_of_bins_minus_one = (max_bin_id - min_bin_id);

    if (self) 
    {
        for(int i=0; i<count0; i++)
        {
            int starting_index = i+1;
              
            __m256 xmm_x0 = _mm256_set1_ps(*(Pos0 + i + 0*count0));
            __m256 xmm_y0 = _mm256_set1_ps(*(Pos0 + i + 1*count0));
            __m256 xmm_z0 = _mm256_set1_ps(*(Pos0 + i + 2*count0));
              
            int particles_left = count1 - starting_index;
            int particles_left_prime = ((particles_left >> 3) << 3);
            int count1_prime = starting_index + particles_left_prime;

          
            for(int j=starting_index; j < count1_prime; j+=SIMD_WIDTH)
            {
                __m256 xmm_x1 = _mm256_loadu_ps(Pos1 + j + 0*count1);
                __m256 xmm_y1 = _mm256_loadu_ps(Pos1 + j + 1*count1);
                __m256 xmm_z1 = _mm256_loadu_ps(Pos1 + j + 2*count1);

                __m256 xmm_diff_X = _mm256_sub_ps(xmm_x0, xmm_x1);
                __m256 xmm_diff_Y = _mm256_sub_ps(xmm_y0, xmm_y1);
                __m256 xmm_diff_Z = _mm256_sub_ps(xmm_z0, xmm_z1);

                __m256 xmm_norm_2 = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(xmm_diff_X, xmm_diff_X), _mm256_mul_ps(xmm_diff_Y, xmm_diff_Y)), _mm256_mul_ps(xmm_diff_Z, xmm_diff_Z));

              
                for(int k=0; k<number_of_bins_minus_one; k++) 
                {
                  
                    __m256i t = _mm256_castps_si256(_mm256_cmp_ps(xmm_norm_2, xmm_separation_element[k], _CMP_LT_OS));
                    xmm_result[k] = _mm_sub_epi32(xmm_result[k], _mm256_castsi256_si128(t));
                    xmm_result[k] = _mm_sub_epi32(xmm_result[k], _mm256_extractf128_si256(t, 1));
                }
            }

            {
              
                int j = count1_prime;
                int remaining = count1 - count1_prime;
                __m256 xmm_anding = _mm256_load_ps((float *)(Remaining[remaining]));

                __m256 xmm_x1 = _mm256_loadu_ps(Pos1 + j + 0*count1);
                __m256 xmm_y1 = _mm256_loadu_ps(Pos1 + j + 1*count1);
                __m256 xmm_z1 = _mm256_loadu_ps(Pos1 + j + 2*count1);

                __m256 xmm_diff_X = _mm256_sub_ps(xmm_x0, xmm_x1);
                __m256 xmm_diff_Y = _mm256_sub_ps(xmm_y0, xmm_y1);
                __m256 xmm_diff_Z = _mm256_sub_ps(xmm_z0, xmm_z1);

                __m256 xmm_norm_2 = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(xmm_diff_X, xmm_diff_X), _mm256_mul_ps(xmm_diff_Y, xmm_diff_Y)), _mm256_mul_ps(xmm_diff_Z, xmm_diff_Z));

                for(int k=0; k<number_of_bins_minus_one; k++) 
                {
                    __m256i t = _mm256_castps_si256(_mm256_and_ps(xmm_anding, _mm256_cmp_ps(xmm_norm_2, xmm_separation_element[k], _CMP_LT_OS)));
                    xmm_result[k] = _mm_sub_epi32(xmm_result[k], _mm256_castsi256_si128(t));
                    xmm_result[k] = _mm_sub_epi32(xmm_result[k], _mm256_extractf128_si256(t, 1));
                }
            }
        }
    }
    else 
    {
        for(int i=0; i<count0; i++) 
        {
            __m256 xmm_x0 = _mm256_set1_ps(*(Pos0 + i + 0*count0));
            __m256 xmm_y0 = _mm256_set1_ps(*(Pos0 + i + 1*count0));
            __m256 xmm_z0 = _mm256_set1_ps(*(Pos0 + i + 2*count0));

            for(int j=0; j < count1; j+=SIMD_WIDTH)
            {
                __m256 xmm_x1 = _mm256_load_ps(Pos1 + j + 0*count1);
                __m256 xmm_y1 = _mm256_load_ps(Pos1 + j + 1*count1);
                __m256 xmm_z1 = _mm256_load_ps(Pos1 + j + 2*count1);

                __m256 xmm_diff_X = _mm256_sub_ps(xmm_x0, xmm_x1);
                __m256 xmm_diff_Y = _mm256_sub_ps(xmm_y0, xmm_y1);
                __m256 xmm_diff_Z = _mm256_sub_ps(xmm_z0, xmm_z1);

                __m256 xmm_norm_2 = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(xmm_diff_X, xmm_diff_X), _mm256_mul_ps(xmm_diff_Y, xmm_diff_Y)), _mm256_mul_ps(xmm_diff_Z, xmm_diff_Z));

                for(int k=0; k<number_of_bins_minus_one; k++) 
                {
                    __m256i t = _mm256_castps_si256(_mm256_cmp_ps(xmm_norm_2, xmm_separation_element[k], _CMP_LT_OS));
                    xmm_result[k] = _mm_sub_epi32(xmm_result[k], _mm256_castsi256_si128(t));
                    xmm_result[k] = _mm_sub_epi32(xmm_result[k], _mm256_extractf128_si256(t, 1));
                }
            }
        }
    }

    {
        __declspec (align(64)) int Temp[4*HIST_BINS];
        int sum[HIST_BINS];

        for(int k=0; k<number_of_bins_minus_one; k++) _mm_store_si128((__m128i *)(Temp+4*k), xmm_result[k]);
            
        int total =  (self) ? ((count0*(count1-1))/2) : (count0 * count1);

        for(int k=0; k<number_of_bins_minus_one; k++) sum[k] =  Temp[4*k + 0] + Temp[4*k + 1] + Temp[4*k + 2] + Temp[4*k + 3];
        sum[number_of_bins_minus_one] = total;

        DD_int0[min_bin_id+0] += sum[0];
        for(int k=1; k<=number_of_bins_minus_one; k++) DD_int0[min_bin_id+k] += (sum[k] - sum[k-1]);

    }
}

void  Actual_Update_Histogram_Self_Cross(TYPE *Pos0, int count0, TYPE *Pos1, int count1, TYPE *Range0_X, TYPE *Range0_Y, TYPE *Range0_Z, 
        TYPE *Range1_X, TYPE *Range1_Y, TYPE *Range1_Z, TYPE *BinCorners2, 
                    unsigned int *DD_int0, unsigned int *DD_int1, unsigned int *Gather_Histogram0, unsigned int *Gather_Histogram1, int nrbin, int threadid)
{

    int self = 0;
    if (Pos0 == Pos1) self = 1;
    if (self) { if (count0 != count1) ERROR_PRINT();}

    TYPE min_dist_sqr, max_dist_sqr;
    int min_bin_id, max_bin_id;

    Compute_Min_Max_Dist_Sqr(Range0_X, Range0_Y, Range0_Z, Range1_X, Range1_Y, Range1_Z, &min_dist_sqr, &max_dist_sqr);

    Compute_min_max_bin_id(BinCorners2, nrbin, min_dist_sqr, max_dist_sqr, &min_bin_id, &max_bin_id);

    int number_of_bins = max_bin_id - min_bin_id + 1;

    TYPE *Aligned_Buffer = global_Aligned_Buffer[threadid];

    TYPE LARGE_ENOUGH = 1000.0 * global_grid_D.Max[2];

    switch(number_of_bins)
    {
        case 1:
        {
            Compute_Distance_And_Populate_Hist_1(Pos0, count0, Pos1, count1, BinCorners2, nrbin, DD_int0, DD_int1, min_bin_id, max_bin_id, Range1_X, Range1_Y, Range1_Z, threadid);
            break;
        }

        case 2:
        {
	    unsigned long long int stime = __rdtsc();
            if (self)
            {
                Compute_Distance_And_Populate_Hist_2(Pos0, count0, Pos1, count1, BinCorners2, nrbin, DD_int0, DD_int1, min_bin_id, max_bin_id, Range1_X, Range1_Y, Range1_Z, threadid);
            }
            else
            {
                int count1_prime = 8*((count1 + 7)/8);
                for(int q=0; q<count1; q++) Aligned_Buffer[0*count1_prime + q] = Pos1[0*count1 + q]; for(int q=count1; q < count1_prime; q++) Aligned_Buffer[0*count1_prime + q] = LARGE_ENOUGH;
                for(int q=0; q<count1; q++) Aligned_Buffer[1*count1_prime + q] = Pos1[1*count1 + q]; for(int q=count1; q < count1_prime; q++) Aligned_Buffer[1*count1_prime + q] = LARGE_ENOUGH;
                for(int q=0; q<count1; q++) Aligned_Buffer[2*count1_prime + q] = Pos1[2*count1 + q]; for(int q=count1; q < count1_prime; q++) Aligned_Buffer[2*count1_prime + q] = LARGE_ENOUGH;
                Compute_Distance_And_Populate_Hist_2(Pos0, count0, Aligned_Buffer, count1_prime, BinCorners2, nrbin, DD_int0, DD_int1, min_bin_id, max_bin_id, Range1_X, Range1_Y, Range1_Z, threadid);
                DD_int0[max_bin_id] -= (count0 * (count1_prime - count1));
            }
	    if (threadid == 0) MT_2[0] += (__rdtsc() - stime);
            break;
        }

        case 3:
        {
	    unsigned long long int stime = __rdtsc();
            if (self)
            {
                Compute_Distance_And_Populate_Hist_3(Pos0, count0, Pos1, count1, BinCorners2, nrbin, DD_int0, DD_int1, min_bin_id, max_bin_id, Range1_X, Range1_Y, Range1_Z, threadid);
            }
            else
            {
                int count1_prime = 8*((count1 + 7)/8);
                for(int q=0; q<count1; q++) Aligned_Buffer[0*count1_prime + q] = Pos1[0*count1 + q]; for(int q=count1; q < count1_prime; q++) Aligned_Buffer[0*count1_prime + q] = LARGE_ENOUGH;
                for(int q=0; q<count1; q++) Aligned_Buffer[1*count1_prime + q] = Pos1[1*count1 + q]; for(int q=count1; q < count1_prime; q++) Aligned_Buffer[1*count1_prime + q] = LARGE_ENOUGH;
                for(int q=0; q<count1; q++) Aligned_Buffer[2*count1_prime + q] = Pos1[2*count1 + q]; for(int q=count1; q < count1_prime; q++) Aligned_Buffer[2*count1_prime + q] = LARGE_ENOUGH;
                Compute_Distance_And_Populate_Hist_3(Pos0, count0, Aligned_Buffer, count1_prime, BinCorners2, nrbin, DD_int0, DD_int1, min_bin_id, max_bin_id, Range1_X, Range1_Y, Range1_Z, threadid);
                DD_int0[max_bin_id] -= (count0 * (count1_prime - count1));
            }
	    if (threadid == 0) MT_3[0] += (__rdtsc() - stime);
            break;
        }

        case 4:
        {
	    unsigned long long int stime = __rdtsc();
            if (self)
            {
                Compute_Distance_And_Populate_Hist_4(Pos0, count0, Pos1, count1, BinCorners2, nrbin, DD_int0, DD_int1, min_bin_id, max_bin_id, Range1_X, Range1_Y, Range1_Z, threadid);
            }
            else
            {
                int count1_prime = 8*((count1 + 7)/8);
                for(int q=0; q<count1; q++) Aligned_Buffer[0*count1_prime + q] = Pos1[0*count1 + q]; for(int q=count1; q < count1_prime; q++) Aligned_Buffer[0*count1_prime + q] = LARGE_ENOUGH;
                for(int q=0; q<count1; q++) Aligned_Buffer[1*count1_prime + q] = Pos1[1*count1 + q]; for(int q=count1; q < count1_prime; q++) Aligned_Buffer[1*count1_prime + q] = LARGE_ENOUGH;
                for(int q=0; q<count1; q++) Aligned_Buffer[2*count1_prime + q] = Pos1[2*count1 + q]; for(int q=count1; q < count1_prime; q++) Aligned_Buffer[2*count1_prime + q] = LARGE_ENOUGH;
                Compute_Distance_And_Populate_Hist_4(Pos0, count0, Aligned_Buffer, count1_prime, BinCorners2, nrbin, DD_int0, DD_int1, min_bin_id, max_bin_id, Range1_X, Range1_Y, Range1_Z, threadid);
                DD_int0[max_bin_id] -= (count0 * (count1_prime - count1));
            }
	    if (threadid == 0) MT_4[0] += (__rdtsc() - stime);
            break;
        }

        default:
        {
	    unsigned long long int stime = __rdtsc();
            int count1_prime = 8*((count1 + 7)/8);
            if (self)
            {
                Compute_Distance_And_Populate_Hist_N(Pos0, count0, Pos1, count1, BinCorners2, nrbin, DD_int0, DD_int1, min_bin_id, max_bin_id, Range1_X, Range1_Y, Range1_Z, threadid);
            }
            else
            {
                for(int q=0; q<count1; q++) Aligned_Buffer[0*count1_prime + q] = Pos1[0*count1 + q]; for(int q=count1; q < count1_prime; q++) Aligned_Buffer[0*count1_prime + q] = LARGE_ENOUGH;
                for(int q=0; q<count1; q++) Aligned_Buffer[1*count1_prime + q] = Pos1[1*count1 + q]; for(int q=count1; q < count1_prime; q++) Aligned_Buffer[1*count1_prime + q] = LARGE_ENOUGH;
                for(int q=0; q<count1; q++) Aligned_Buffer[2*count1_prime + q] = Pos1[2*count1 + q]; for(int q=count1; q < count1_prime; q++) Aligned_Buffer[2*count1_prime + q] = LARGE_ENOUGH;
                Compute_Distance_And_Populate_Hist_N(Pos0, count0, Aligned_Buffer, count1_prime, BinCorners2, nrbin, DD_int0, DD_int1, min_bin_id, max_bin_id, Range1_X, Range1_Y, Range1_Z, threadid);
                DD_int0[max_bin_id] -= (count0 * (count1_prime - count1));
            }
	    if (threadid == 0) MT_5[0] += (__rdtsc() - stime);
            break;
        }
    }
}

void Update_Histogram_Self_Cross(TYPE *Pos0, int count0, TYPE *Bdry0_X, TYPE *Bdry0_Y, TYPE *Bdry0_Z, int *Range0, int ranges0, 
                                 TYPE *Pos1, int count1, TYPE *Bdry1_X, TYPE *Bdry1_Y, TYPE *Bdry1_Z, int *Range1, int ranges1, 
                                 TYPE *Range0_X, TYPE *Range0_Y, TYPE *Range0_Z, 
                                 TYPE *Range1_X, TYPE *Range1_Y, TYPE *Range1_Z, TYPE *BinCorners2, 
                                 unsigned int *DD_int0, unsigned int *DD_int1, unsigned int *Gather_Histogram0, unsigned int *Gather_Histogram1, int nrbin, int threadid)
{

    int self = 0;
    if (Pos0 == Pos1) self = 1;
    if (self) { if (count0 != count1) ERROR_PRINT();}

    TYPE min_dist_sqr, max_dist_sqr;
    int min_bin_id, max_bin_id;

    Compute_Min_Max_Dist_Sqr(Range0_X, Range0_Y, Range0_Z, Range1_X, Range1_Y, Range1_Z, &min_dist_sqr, &max_dist_sqr);

    Compute_min_max_bin_id(BinCorners2, nrbin, min_dist_sqr, max_dist_sqr, &min_bin_id, &max_bin_id);

    int number_of_bins = max_bin_id - min_bin_id + 1;

    if (number_of_bins == 1)
    {
        Compute_Distance_And_Populate_Hist_1(Pos0, count0, Pos1, count1, BinCorners2, nrbin, DD_int0, DD_int1, min_bin_id, max_bin_id, Range1_X, Range1_Y, Range1_Z, threadid);
        return;
    }

    TYPE Range2_X[2], Range2_Y[2], Range2_Z[2];
    TYPE Range3_X[2], Range3_Y[2], Range3_Z[2];

    for(int div_0 = 0; div_0 < ranges0; div_0++)
    {
        int count00 = Range0[div_0+1] - Range0[div_0+0];
        TYPE *Pos00 = Pos0 + 3*Range0[div_0];

        Range2_X[0] = Bdry0_X[2*div_0 + 0];
        Range2_X[1] = Bdry0_X[2*div_0 + 1];

        Range2_Y[0] = Bdry0_Y[2*div_0 + 0];
        Range2_Y[1] = Bdry0_Y[2*div_0 + 1];

        Range2_Z[0] = Bdry0_Z[2*div_0 + 0];
        Range2_Z[1] = Bdry0_Z[2*div_0 + 1];

        int div_1 = 0;
        if (self) div_1 = div_0; 
        for(; div_1 < ranges1; div_1++)
        {
            int count11 = Range1[div_1+1] - Range1[div_1+0];
            TYPE *Pos11 = Pos1 + 3*Range1[div_1];

            Range3_X[0] = Bdry1_X[2*div_1 + 0];
            Range3_X[1] = Bdry1_X[2*div_1 + 1];

            Range3_Y[0] = Bdry1_Y[2*div_1 + 0];
            Range3_Y[1] = Bdry1_Y[2*div_1 + 1];

            Range3_Z[0] = Bdry1_Z[2*div_1 + 0];
            Range3_Z[1] = Bdry1_Z[2*div_1 + 1];

            Actual_Update_Histogram_Self_Cross(Pos00, count00, Pos11, count11, Range2_X, Range2_Y, Range2_Z, Range3_X, Range3_Y, Range3_Z, BinCorners2, 
                    DD_int0, DD_int1, Gather_Histogram0, Gather_Histogram1, nrbin, threadid);
        }
    }
}




        
TYPE Find_Minimum_Distance_Between_Cells(TYPE *Range0_X, TYPE *Range0_Y, TYPE *Range0_Z, TYPE *Range1_X, TYPE *Range1_Y, TYPE *Range1_Z)
{
    TYPE min_dst_sqr = 0;

    if (Range0_X[0] > Range1_X[1]) min_dst_sqr += (Range0_X[0] - Range1_X[1])*(Range0_X[0] - Range1_X[1]);
    if (Range1_X[0] > Range0_X[1]) min_dst_sqr += (Range1_X[0] - Range0_X[1])*(Range1_X[0] - Range0_X[1]);

    if (Range0_Y[0] > Range1_Y[1]) min_dst_sqr += (Range0_Y[0] - Range1_Y[1])*(Range0_Y[0] - Range1_Y[1]);
    if (Range1_Y[0] > Range0_Y[1]) min_dst_sqr += (Range1_Y[0] - Range0_Y[1])*(Range1_Y[0] - Range0_Y[1]);

    if (Range0_Z[0] > Range1_Z[1]) min_dst_sqr += (Range0_Z[0] - Range1_Z[1])*(Range0_Z[0] - Range1_Z[1]);
    if (Range1_Z[0] > Range0_Z[1]) min_dst_sqr += (Range1_Z[0] - Range0_Z[1])*(Range1_Z[0] - Range0_Z[1]);

    return (min_dst_sqr);
}
  

void  Perform_DR_Helper(void *arg)
{
    int taskid = (int)((size_t)(arg));
    int threadid = omp_get_thread_num();

    Grid *grid_D = &global_grid_D;
    Grid *grid_R = &global_grid_R;

    TYPE *Extent = grid_D->Extent; 

    int dimx = grid_D->dimx;
    int dimy = grid_D->dimy;
    int dimz = grid_D->dimz;

    int dimxy = dimx * dimy;

    int dx = global_dx;
    int dy = global_dy;
    int dz = global_dz;

    TYPE *Cell_Width = grid_D->Cell_Width;

    TYPE Range0_X[2], Range0_Y[2], Range0_Z[2];
    TYPE Range1_X[2], Range1_Y[2], Range1_Z[2];

    int start_processing_cell_index = global_starting_cell_index_D;
    int   end_processing_cell_index = global_ending_cell_index_D;


    unsigned long long int threshold_for_accumulate_sum = (1<<29); threshold_for_accumulate_sum = (threshold_for_accumulate_sum << 1) - 1;

    unsigned long long int local_start_time, local_end_time;

    local_start_time = __rdtsc();

    long long int actual_sum = global_actual_sum_dr[16*threadid];
    long long int curr_accumulated_actual_sum = 0;

    unsigned int *Gather_Histogram0 = global_Gather_Histogram0[threadid];
    unsigned int *Gather_Histogram1 = global_Gather_Histogram1[threadid];
    unsigned int *DR_int0 = global_DR_int0[threadid];
    unsigned int *DR_int1 = global_DR_int1[threadid];

    TYPE *Pos1 = global_Pos1[threadid];
    TYPE *Bdry1_X = global_Bdry1_X[threadid];
    TYPE *Bdry1_Y = global_Bdry1_Y[threadid];
    TYPE *Bdry1_Z = global_Bdry1_Z[threadid];

    TYPE *BinCorners2 = global_BinCorners2;
    unsigned long long int *DR = local_Histogram_DR[threadid];

    int nrbin = global_nrbin;
    TYPE rmax_2 = global_rmax_2;

    int current_cell_index = global_starting_cell_index_D + taskid;
    {
        int x, y, z; z = current_cell_index / dimxy; int xy = current_cell_index % dimxy; y = xy / dimx; x = xy % dimx;


        if (!global_Required_D_For_R[current_cell_index]) ERROR_PRINT();



#if 0
#endif
        int objects_in_this_cell = grid_D->Count_Per_Cell[current_cell_index];
        int subdivisions_in_this_cell = grid_D->Number_of_kd_subdivisions[current_cell_index];
        int *Range0 = grid_D->Range[current_cell_index];
        TYPE *Bdry0_X = grid_D->Bdry_X[current_cell_index];
        TYPE *Bdry0_Y = grid_D->Bdry_Y[current_cell_index];
        TYPE *Bdry0_Z = grid_D->Bdry_Z[current_cell_index];

        TYPE *Pos0 = grid_D->Positions[current_cell_index];

        Range0_X[0] = (x*1.00)*Cell_Width[0]; Range0_X[1] = ((x+1)*1.00)*Cell_Width[0];
        Range0_Y[0] = (y*1.00)*Cell_Width[1]; Range0_Y[1] = ((y+1)*1.00)*Cell_Width[1];
        Range0_Z[0] = (z*1.00)*Cell_Width[2]; Range0_Z[1] = ((z+1)*1.00)*Cell_Width[2];

#if 0
        {
        }
#endif

        for(int zz = (z - dz); zz <= (z + dz); zz++)
        {
            for(int yy = (y - dy); yy <= (y + dy); yy++)
            {
                for(int xx = (x - dx); xx <= (x + dx); xx++)
                {
                    Range1_X[0] = (xx*1.00)*Cell_Width[0]; Range1_X[1] = ((xx+1)*1.00)*Cell_Width[0];
                    Range1_Y[0] = (yy*1.00)*Cell_Width[1]; Range1_Y[1] = ((yy+1)*1.00)*Cell_Width[1];
                    Range1_Z[0] = (zz*1.00)*Cell_Width[2]; Range1_Z[1] = ((zz+1)*1.00)*Cell_Width[2];

                    TYPE min_dist_2 = Find_Minimum_Distance_Between_Cells(Range0_X, Range0_Y, Range0_Z, Range1_X, Range1_Y, Range1_Z);

                    if (min_dist_2 > rmax_2) continue;

                    int xx_prime = xx, yy_prime = yy, zz_prime = zz;
                    if (xx < 0) xx_prime = xx + dimx; else if (xx >= (dimx)) xx_prime = xx - dimx;
                    if (yy < 0) yy_prime = yy + dimy; else if (yy >= (dimy)) yy_prime = yy - dimy;
                    if (zz < 0) zz_prime = zz + dimz; else if (zz >= (dimz)) zz_prime = zz - dimz;

                    if ( (xx_prime < 0) || (xx_prime >= dimx))  ERROR_PRINT();
                    if ( (yy_prime < 0) || (yy_prime >= dimy))  ERROR_PRINT();
                    if ( (zz_prime < 0) || (zz_prime >= dimz))  ERROR_PRINT();

                    int neighbor_cell_index = GET_CELL_INDEX(xx_prime, yy_prime, zz_prime);

                    if (!global_Required_D_For_R[neighbor_cell_index]) ERROR_PRINT();

                    int *Range1 = grid_R->Range[neighbor_cell_index];
                    int objects_in_neighboring_cell = grid_R->Count_Per_Cell[neighbor_cell_index];
                    int subdivisions_in_neighboring_cell = grid_R->Number_of_kd_subdivisions[neighbor_cell_index];

                    TYPE Delta[DIMENSIONS]; Delta[0] = Delta[1] = Delta[2] = 0.0;
                    if (xx < 0) Delta[0] = -Extent[0]; else if (xx >= (dimx)) Delta[0] = Extent[0];
                    if (yy < 0) Delta[1] = -Extent[1]; else if (yy >= (dimy)) Delta[1] = Extent[1];
                    if (zz < 0) Delta[2] = -Extent[2]; else if (zz >= (dimz)) Delta[2] = Extent[2];
                
                    for(int sub=0; sub < subdivisions_in_neighboring_cell; sub++)
                    {
                        TYPE *Actual_Dst = Pos1 + 3*Range1[sub];
                        TYPE *Actual_Src = grid_R->Positions[neighbor_cell_index] + 3*Range1[sub];
                        int count_of_objects = Range1[sub+1] - Range1[sub+0];
                        for(int j=0; j<DIMENSIONS; j++)
                        {
                            TYPE *Dst = Actual_Dst + j*count_of_objects;
                            TYPE *Src = Actual_Src + j*count_of_objects;
                            for(int i=0; i<count_of_objects; i++) Dst[i] = Src[i] + Delta[j];
                        }
                    }

                    for(int i=0; i < 2*subdivisions_in_neighboring_cell; i++) Bdry1_X[i] = grid_R->Bdry_X[neighbor_cell_index][i] + Delta[0];
                    for(int i=0; i < 2*subdivisions_in_neighboring_cell; i++) Bdry1_Y[i] = grid_R->Bdry_Y[neighbor_cell_index][i] + Delta[1];
                    for(int i=0; i < 2*subdivisions_in_neighboring_cell; i++) Bdry1_Z[i] = grid_R->Bdry_Z[neighbor_cell_index][i] + Delta[2];


                    long long int local_sum = ((long long int)(objects_in_this_cell) * ((long long int)(objects_in_neighboring_cell)));
                    actual_sum += local_sum;
                    curr_accumulated_actual_sum += local_sum;

                    Update_Histogram_Self_Cross(Pos0, objects_in_this_cell,         Bdry0_X, Bdry0_Y, Bdry0_Z, Range0, subdivisions_in_this_cell,
                                                Pos1, objects_in_neighboring_cell,  Bdry1_X, Bdry1_Y, Bdry1_Z, Range1, subdivisions_in_neighboring_cell, 
                                                Range0_X, Range0_Y, Range0_Z, 
                                                Range1_X, Range1_Y, Range1_Z, BinCorners2, DR_int0, DR_int1, Gather_Histogram0, Gather_Histogram1, nrbin, threadid);


                    if (curr_accumulated_actual_sum >= threshold_for_accumulate_sum)
                    {
                        unsigned long long int ttt = 0;
                        for(int i=0; i<=(1+nrbin); i++) ttt += DR_int0[i];
                        for(int i=0; i<=(1+nrbin); i++) ttt += DR_int1[i];

                        if (ttt != curr_accumulated_actual_sum)
                        {
                            printf("node_id = %d ::: threadid = %d :: curr_accumulated_actual_sum = %lld ::: ttt = %lld\n", node_id, threadid, curr_accumulated_actual_sum, ttt);
                            ERROR_PRINT();
                        }

                        curr_accumulated_actual_sum = 0;
                        for(int i=0; i<=(1+nrbin); i++) DR[i] += DR_int0[i];
                        for(int i=0; i<=(1+nrbin); i++) DR[i] += DR_int1[i];
                        for(int i=0; i<=(1+nrbin); i++) DR_int0[i] = 0;
                        for(int i=0; i<=(1+nrbin); i++) DR_int1[i] = 0;

                        for(int bin_id = 0; bin_id <= (1+nrbin); bin_id++)
                        {
                            for(int lane = 0; lane < SIMD_WIDTH; lane++)
                            {
                                DR[bin_id] += Gather_Histogram0[bin_id*SIMD_WIDTH + lane];
                                DR[bin_id] += Gather_Histogram1[bin_id*SIMD_WIDTH + lane];
                            }
                        }

                        for(int i=0; i<(SIMD_WIDTH*HIST_BINS); i++) Gather_Histogram0[i] = 0;
                        for(int i=0; i<(SIMD_WIDTH*HIST_BINS); i++) Gather_Histogram1[i] = 0;
                    }
                }
            }
        }
    }

    {
        for(int i=0; i<=(1+nrbin); i++) DR[i] += DR_int0[i];
        for(int i=0; i<=(1+nrbin); i++) DR[i] += DR_int1[i];
        for(int i=0; i<=(1+nrbin); i++) DR_int0[i] = 0;
        for(int i=0; i<=(1+nrbin); i++) DR_int1[i] = 0;

        for(int bin_id = 0; bin_id <= (1+nrbin); bin_id++)
        {
            for(int lane = 0; lane < SIMD_WIDTH; lane++)
            {
                DR[bin_id] += Gather_Histogram0[bin_id*SIMD_WIDTH + lane];
                DR[bin_id] += Gather_Histogram1[bin_id*SIMD_WIDTH + lane];
            }
        }
                
        for(int i=0; i<(SIMD_WIDTH*HIST_BINS); i++) Gather_Histogram0[i] = 0;
        for(int i=0; i<(SIMD_WIDTH*HIST_BINS); i++) Gather_Histogram1[i] = 0;
    }

    global_actual_sum_dr[16*threadid] = actual_sum;

    local_end_time = __rdtsc();
    global_time_per_thread_dr[threadid] += local_end_time - local_start_time;
}


void  Perform_RR_Helper(void *arg)
{
    int taskid = (int)((size_t)(arg));
    int threadid = omp_get_thread_num();

    Grid *grid = &global_grid_R;

    TYPE *Extent = grid->Extent;

    int dimx = grid->dimx;
    int dimy = grid->dimy;
    int dimz = grid->dimz;

    int dimxy = dimx * dimy;

    int dx = global_dx;
    int dy = global_dy;
    int dz = global_dz;

    TYPE *Cell_Width = grid->Cell_Width;

    TYPE Range0_X[2], Range0_Y[2], Range0_Z[2];
    TYPE Range1_X[2], Range1_Y[2], Range1_Z[2];

    int start_processing_cell_index = global_starting_cell_index_R;
    int   end_processing_cell_index = global_ending_cell_index_R;

    unsigned long long int threshold_for_accumulate_sum = (1<<29); threshold_for_accumulate_sum = (threshold_for_accumulate_sum << 1) - 1;

    unsigned long long int local_start_time, local_end_time;

    local_start_time = __rdtsc();

    long long int actual_sum = global_actual_sum_rr[16*threadid];
    long long int curr_accumulated_actual_sum = 0;

    unsigned int *Gather_Histogram0 = global_Gather_Histogram0[threadid];
    unsigned int *Gather_Histogram1 = global_Gather_Histogram1[threadid];
    unsigned int *RR_int0 = global_RR_int0[threadid];
    unsigned int *RR_int1 = global_RR_int1[threadid];

    TYPE *Pos1 = global_Pos1[threadid];
    TYPE *Bdry1_X = global_Bdry1_X[threadid];
    TYPE *Bdry1_Y = global_Bdry1_Y[threadid];
    TYPE *Bdry1_Z = global_Bdry1_Z[threadid];

    TYPE *BinCorners2 = global_BinCorners2;
    unsigned long long int *RR = local_Histogram_RR[threadid];

    int nrbin = global_nrbin;
    TYPE rmax_2 = global_rmax_2;

    int current_cell_index = global_starting_cell_index_R + taskid;
    {
        int x, y, z; z = current_cell_index / dimxy; int xy = current_cell_index % dimxy; y = xy / dimx; x = xy % dimx;


        if (!global_Required_R[current_cell_index]) ERROR_PRINT();

        int objects_in_this_cell = grid->Count_Per_Cell[current_cell_index];
        int subdivisions_in_this_cell = grid->Number_of_kd_subdivisions[current_cell_index];
        int *Range0 = grid->Range[current_cell_index];
        TYPE *Bdry0_X = grid->Bdry_X[current_cell_index];
        TYPE *Bdry0_Y = grid->Bdry_Y[current_cell_index];
        TYPE *Bdry0_Z = grid->Bdry_Z[current_cell_index];

        TYPE *Pos0 = grid->Positions[current_cell_index];

        Range0_X[0] = (x*1.00)*Cell_Width[0]; Range0_X[1] = ((x+1)*1.00)*Cell_Width[0];
        Range0_Y[0] = (y*1.00)*Cell_Width[1]; Range0_Y[1] = ((y+1)*1.00)*Cell_Width[1];
        Range0_Z[0] = (z*1.00)*Cell_Width[2]; Range0_Z[1] = ((z+1)*1.00)*Cell_Width[2];

        long long int local_sum = ((long long int)(objects_in_this_cell) * (long long int)(objects_in_this_cell-1))/2;

        actual_sum += local_sum;
        curr_accumulated_actual_sum += local_sum;

        Update_Histogram_Self_Cross(Pos0, objects_in_this_cell, Bdry0_X, Bdry0_Y, Bdry0_Z, Range0,  subdivisions_in_this_cell, 
                                    Pos0, objects_in_this_cell, Bdry0_X, Bdry0_Y, Bdry0_Z, Range0,  subdivisions_in_this_cell,
                                    Range0_X, Range0_Y, Range0_Z, 
                                    Range0_X, Range0_Y, Range0_Z, BinCorners2, RR_int0, RR_int1, Gather_Histogram0, Gather_Histogram1, nrbin, threadid);

        for(int zz = (z - dz); zz <= (z + dz); zz++)
        {
            for(int yy = (y - dy); yy <= (y + dy); yy++)
            {
                for(int xx = (x - dx); xx <= (x + dx); xx++)
                {
                    if ((xx == x) && (yy == y) && (zz == z)) continue;

                            
                    Range1_X[0] = (xx*1.00)*Cell_Width[0]; Range1_X[1] = ((xx+1)*1.00)*Cell_Width[0];
                    Range1_Y[0] = (yy*1.00)*Cell_Width[1]; Range1_Y[1] = ((yy+1)*1.00)*Cell_Width[1];
                    Range1_Z[0] = (zz*1.00)*Cell_Width[2]; Range1_Z[1] = ((zz+1)*1.00)*Cell_Width[2];

                    TYPE min_dist_2 = Find_Minimum_Distance_Between_Cells(Range0_X, Range0_Y, Range0_Z, Range1_X, Range1_Y, Range1_Z);

                    if (min_dist_2 > rmax_2) continue;


                    int xx_prime = xx, yy_prime = yy, zz_prime = zz;
                    if (xx < 0) xx_prime = xx + dimx; else if (xx >= (dimx)) xx_prime = xx - dimx;
                    if (yy < 0) yy_prime = yy + dimy; else if (yy >= (dimy)) yy_prime = yy - dimy;
                    if (zz < 0) zz_prime = zz + dimz; else if (zz >= (dimz)) zz_prime = zz - dimz;

                    if ( (xx_prime < 0) || (xx_prime >= dimx))  ERROR_PRINT();
                    if ( (yy_prime < 0) || (yy_prime >= dimy))  ERROR_PRINT();
                    if ( (zz_prime < 0) || (zz_prime >= dimz))  ERROR_PRINT();

                    int neighbor_cell_index = GET_CELL_INDEX(xx_prime, yy_prime, zz_prime);
                    if (neighbor_cell_index > current_cell_index) continue; 

                    if (!global_Required_R[neighbor_cell_index]) ERROR_PRINT();

                    int *Range1 = grid->Range[neighbor_cell_index];
                    int objects_in_neighboring_cell = grid->Count_Per_Cell[neighbor_cell_index];
                    int subdivisions_in_neighboring_cell = grid->Number_of_kd_subdivisions[neighbor_cell_index];

                    TYPE Delta[DIMENSIONS]; Delta[0] = Delta[1] = Delta[2] = 0.0;
                    if (xx < 0) Delta[0] = -Extent[0]; else if (xx >= (dimx)) Delta[0] = Extent[0];
                    if (yy < 0) Delta[1] = -Extent[1]; else if (yy >= (dimy)) Delta[1] = Extent[1];
                    if (zz < 0) Delta[2] = -Extent[2]; else if (zz >= (dimz)) Delta[2] = Extent[2];
                
                    for(int sub=0; sub < subdivisions_in_neighboring_cell; sub++)
                    {
                        TYPE *Actual_Dst = Pos1 + 3*Range1[sub];
                        TYPE *Actual_Src = grid->Positions[neighbor_cell_index] + 3*Range1[sub];
                        int count_of_objects = Range1[sub+1] - Range1[sub+0];
                        for(int j=0; j<DIMENSIONS; j++)
                        {
                            TYPE *Dst = Actual_Dst + j*count_of_objects;
                            TYPE *Src = Actual_Src + j*count_of_objects;
                            for(int i=0; i<count_of_objects; i++) Dst[i] = Src[i] + Delta[j];
                        }
                    }

                    for(int i=0; i < 2*subdivisions_in_neighboring_cell; i++) Bdry1_X[i] = grid->Bdry_X[neighbor_cell_index][i] + Delta[0];
                    for(int i=0; i < 2*subdivisions_in_neighboring_cell; i++) Bdry1_Y[i] = grid->Bdry_Y[neighbor_cell_index][i] + Delta[1];
                    for(int i=0; i < 2*subdivisions_in_neighboring_cell; i++) Bdry1_Z[i] = grid->Bdry_Z[neighbor_cell_index][i] + Delta[2];


                    long long int local_sum = ((long long int)(objects_in_this_cell) * ((long long int)(objects_in_neighboring_cell)));
                    actual_sum += local_sum;
                    curr_accumulated_actual_sum += local_sum;

                    Update_Histogram_Self_Cross(Pos0, objects_in_this_cell,         Bdry0_X, Bdry0_Y, Bdry0_Z, Range0, subdivisions_in_this_cell,
                                                Pos1, objects_in_neighboring_cell,  Bdry1_X, Bdry1_Y, Bdry1_Z, Range1, subdivisions_in_neighboring_cell, 
                                                Range0_X, Range0_Y, Range0_Z, 
                                                Range1_X, Range1_Y, Range1_Z, BinCorners2, RR_int0, RR_int1, Gather_Histogram0, Gather_Histogram1, nrbin, threadid);


                    if (curr_accumulated_actual_sum >= threshold_for_accumulate_sum)
                    {
                        unsigned long long int ttt = 0;
                        for(int i=0; i<=(1+nrbin); i++) ttt += RR_int0[i];
                        for(int i=0; i<=(1+nrbin); i++) ttt += RR_int1[i];

                        if (ttt != curr_accumulated_actual_sum)
                        {
                            printf("node_id = %d ::: threadid = %d :: curr_accumulated_actual_sum = %lld ::: ttt = %lld\n", node_id, threadid, curr_accumulated_actual_sum, ttt);
                            ERROR_PRINT();
                        }

                        curr_accumulated_actual_sum = 0;
                        for(int i=0; i<=(1+nrbin); i++) RR[i] += RR_int0[i];
                        for(int i=0; i<=(1+nrbin); i++) RR[i] += RR_int1[i];
                        for(int i=0; i<=(1+nrbin); i++) RR_int0[i] = 0;
                        for(int i=0; i<=(1+nrbin); i++) RR_int1[i] = 0;

                        for(int bin_id = 0; bin_id <= (1+nrbin); bin_id++)
                        {
                            for(int lane = 0; lane < SIMD_WIDTH; lane++)
                            {
                                RR[bin_id] += Gather_Histogram0[bin_id*SIMD_WIDTH + lane];
                                RR[bin_id] += Gather_Histogram1[bin_id*SIMD_WIDTH + lane];
                            }
                        }

                        for(int i=0; i<(SIMD_WIDTH*HIST_BINS); i++) Gather_Histogram0[i] = 0;
                        for(int i=0; i<(SIMD_WIDTH*HIST_BINS); i++) Gather_Histogram1[i] = 0;
                    }
                }
            }
        }
    }

    {
        for(int i=0; i<=(1+nrbin); i++) RR[i] += RR_int0[i];
        for(int i=0; i<=(1+nrbin); i++) RR[i] += RR_int1[i];
        for(int i=0; i<=(1+nrbin); i++) RR_int0[i] = 0;
        for(int i=0; i<=(1+nrbin); i++) RR_int1[i] = 0;

        for(int bin_id = 0; bin_id <= (1+nrbin); bin_id++)
        {
            for(int lane = 0; lane < SIMD_WIDTH; lane++)
            {
                RR[bin_id] += Gather_Histogram0[bin_id*SIMD_WIDTH + lane];
                RR[bin_id] += Gather_Histogram1[bin_id*SIMD_WIDTH + lane];
            }
        }
                
        for(int i=0; i<(SIMD_WIDTH*HIST_BINS); i++) Gather_Histogram0[i] = 0;
        for(int i=0; i<(SIMD_WIDTH*HIST_BINS); i++) Gather_Histogram1[i] = 0;
    }

    global_actual_sum_rr[16*threadid] = actual_sum;

    local_end_time = __rdtsc();
    global_time_per_thread_rr[threadid] += local_end_time - local_start_time;
}

#if 1

void Compute_Statistics_DR(void)
{
    int threadid = 0;
    int nrbin = global_nrbin;
    TYPE *Rminarr = global_Rminarr;
    TYPE *Rmaxarr = global_Rmaxarr;

    {
        int ngal = global_number_of_galaxies;

        for(int i=0; i<nthreads; i++)
        {
            for(int j=0; j<HIST_BINS; j++) 
            {
                global_Histogram_DR[j] += local_Histogram_DR[i][j];
            }
        }

        MPI_Gather(global_Histogram_DR, HIST_BINS, MPI_LONG_LONG_INT, global_Overall_Histogram_DR, HIST_BINS, MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);

        if (node_id == 0)
        {
            for(int j=0; j<HIST_BINS; j++) global_Histogram_DR[j] = 0;

            for(int k=0; k<nnodes; k++)
            {
                for(int j=0; j<HIST_BINS; j++)
                {
                    global_Histogram_DR[j] += global_Overall_Histogram_DR[k * HIST_BINS + j];
                }
            }

            global_stat_total_interactions_dr = 0; for(int j = 0; j < HIST_BINS; j++) global_stat_total_interactions_dr += global_Histogram_DR[j];

      
      #ifdef POINTS_2D
            double denominator_1 = (M_PI*ngal*(ngal))/1;
      #else
            double denominator_1 = (4.0*M_PI*ngal*(ngal))/3.00/1;
      #endif

            for(int bin_id=0; bin_id < nrbin; bin_id++)
            {
                double rmin = Rminarr[bin_id];
                double rmax = Rmaxarr[bin_id];
          #ifdef POINTS_2D
            double denominator = denominator_1 * (rmax*rmax - rmin*rmin);
          #else
            double denominator = denominator_1 * (rmax*rmax*rmax - rmin*rmin*rmin);
          #endif

                global_DR_over_RR[bin_id+1] = global_Histogram_DR[bin_id+1]/denominator;
            }
        }
    }

    if (threadid == 0)
    {
        unsigned long long int sum_time = 0;
        unsigned long long int min_time = global_time_per_thread_dr[0];
        unsigned long long int max_time = global_time_per_thread_dr[0];

        for(int kk=0; kk<nthreads; kk++)
        {
            sum_time += global_time_per_thread_dr[kk];
            min_time = PCL_MIN(min_time, global_time_per_thread_dr[kk]);
            max_time = PCL_MAX(max_time, global_time_per_thread_dr[kk]);
        }

        unsigned long long int avg_time = sum_time/nthreads;
        //printf("DR :: <<%d>> Avg. Time = %lld ::: Max Time = %lld ::: Ratio = %.4lf\n", node_id, avg_time, max_time, (max_time * 1.00)/avg_time);
    }

    long long int my_easy_sum = 0;
    if (threadid == 0)
    {

        for(int kk=0; kk<nthreads; kk++)
        {
            my_easy_sum += global_Easy[8*kk];
        }
    }

    MPI_BARRIER(node_id);

    if (threadid == 0)
    {
        long long int *Overall_global_Easy = (long long int *)malloc(nnodes * sizeof(long long int));
        MPI_Gather(&my_easy_sum, 1, MPI_LONG_LONG_INT, Overall_global_Easy, 1, MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);

        global_accumulated_easy = 0;
        for(int k = 0; k < nnodes; k++) global_accumulated_easy += Overall_global_Easy[k];

        PRINT_RED
        if (node_id == 0) { /*printf("<<%d>> ::: global_accumulated_easy = %lld\n", node_id, global_accumulated_easy); */ }
        PRINT_BLACK
    }
}


void Compute_Statistics_RR(void)
{
    int threadid = 0;
    int nrbin = global_nrbin;
    TYPE *Rminarr = global_Rminarr;
    TYPE *Rmaxarr = global_Rmaxarr;

    {
        int ngal = global_number_of_galaxies;
        for(int i=0; i<nthreads; i++)
        {
            for(int j=0; j<HIST_BINS; j++) 
            {
                global_Histogram_RR[j] += local_Histogram_RR[i][j];
            }
        }

        
        MPI_Gather(global_Histogram_RR, HIST_BINS, MPI_LONG_LONG_INT, global_Overall_Histogram_RR, HIST_BINS, MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);

        if (node_id == 0)
        {
            for(int j=0; j<HIST_BINS; j++) global_Histogram_RR[j] = 0;

            for(int k=0; k<nnodes; k++)
            {
                for(int j=0; j<HIST_BINS; j++)
                {
                    global_Histogram_RR[j] += global_Overall_Histogram_RR[k * HIST_BINS + j];
                }
            }

            global_stat_total_interactions_rr = 0; for(int j = 0; j < HIST_BINS; j++) global_stat_total_interactions_rr += global_Histogram_RR[j];

      #ifdef POINTS_2D
            double denominator_1 = (M_PI*ngal*(ngal-1))/2;
      #else
            double denominator_1 = (4.0*M_PI*ngal*(ngal-1))/3.00/2;
      #endif

            for(int bin_id=0; bin_id < nrbin; bin_id++)
            {
                double rmin = Rminarr[bin_id];
                double rmax = Rmaxarr[bin_id];
          #ifdef POINTS_2D
            double denominator = denominator_1 * (rmax*rmax - rmin*rmin);
          #else
            double denominator = denominator_1 * (rmax*rmax*rmax - rmin*rmin*rmin);
          #endif

            global_RR_over_RR[bin_id+1] = global_Histogram_RR[bin_id+1]/denominator;
            }
        }
    }

    if (threadid == 0)
    {
        unsigned long long int sum_time = 0;
        unsigned long long int min_time = global_time_per_thread_rr[0];
        unsigned long long int max_time = global_time_per_thread_rr[0];

        for(int kk=0; kk<nthreads; kk++)
        {
            sum_time += global_time_per_thread_rr[kk];
            min_time = PCL_MIN(min_time, global_time_per_thread_rr[kk]);
            max_time = PCL_MAX(max_time, global_time_per_thread_rr[kk]);
        }

        unsigned long long int avg_time = sum_time/nthreads;
        //printf("RR :: <<%d>> Avg. Time = %lld ::: Max Time = %lld ::: Ratio = %.4lf\n", node_id, avg_time, max_time, (max_time * 1.00)/avg_time);
    }

    MPI_BARRIER(node_id);
}

#endif
///////////////////////////////////////////////////////////////////////////////////////////////////

unsigned char *Number_Of_Ones;

void Compute_Number_of_Ones(void)
{
    Number_Of_Ones = (unsigned char *)malloc(256 * 9 * sizeof(unsigned char));

    for(int k=0; k<256; k++)
    {
        unsigned char *Dst = Number_Of_Ones + 9*k;
        Dst[0] = 0;

        for(int p=0; p<8; p++)
        {
            if ((k & (1<<(7-p))) != 0) Dst[++Dst[0]] = p;
        }

        if (Dst[0] > 8) ERROR_PRINT();
        if (Dst[0] == 8)
        {
            if (k !=255) ERROR_PRINT();
        }
     }
}

        
void Copy_Non_Changing_Data_From_D_To_R(void)
{
    if (DIMENSIONS != 3) ERROR_PRINT();

    for(int p = 0; p < DIMENSIONS; p++) global_grid_R.Min[p] = global_grid_D.Min[p];
    for(int p = 0; p < DIMENSIONS; p++) global_grid_R.Max[p] = global_grid_D.Max[p];
    for(int p = 0; p < DIMENSIONS; p++) global_grid_R.Extent[p] = global_grid_D.Extent[p];
    for(int p = 0; p < DIMENSIONS; p++) global_grid_R.Cell_Width[p] = global_grid_D.Cell_Width[p];
}

void Just_Compute_Min_Max(TYPE *Pos, int number_of_galaxies, int dimensions, TYPE *Answer_Min, TYPE *Answer_Max)
{
    for(int p=0; p<DIMENSIONS; p++) Answer_Min[p] = FLT_MAX;
    for(int p=0; p<DIMENSIONS; p++) Answer_Max[p] = -FLT_MAX;
    
    for(int k=0; k<number_of_galaxies; k++)
    {
        for(int p=0; p<DIMENSIONS; p++)
        {
            Answer_Min[p] = PCL_MIN(Answer_Min[p], Pos[3*k + p]);
            Answer_Max[p] = PCL_MAX(Answer_Max[p], Pos[3*k + p]);
        }
    }
}

void Just_Compute_Extents(int dimensions, TYPE *Answer_Min, TYPE *Answer_Max, TYPE *Answer_Extent, TYPE *Answer_Cell_Width, int dimx, int dimy, int dimz)
{
    for(int p=0; p<DIMENSIONS; p++) Answer_Extent[p] = Answer_Max[p] - Answer_Min[p];

    Answer_Cell_Width[0] = Answer_Extent[0] / dimx;
    Answer_Cell_Width[1] = Answer_Extent[1] / dimy;
    Answer_Cell_Width[2] = Answer_Extent[2] / dimz;
}


void Compute_Min_Max(TYPE *Pos, int number_of_galaxies, int dimensions, TYPE *Answer_Min, TYPE *Answer_Max, TYPE *Answer_Extent, int dimx, int dimy, int dimz, TYPE *Answer_Cell_Width)
{
    if (dimensions != DIMENSIONS) ERROR_PRINT_STRING("dimensions != DIMENSIONS");
    if (dimensions != 3) ERROR_PRINT_STRING("dimensions != 3");

    Just_Compute_Min_Max(Pos, number_of_galaxies, dimensions, Answer_Min, Answer_Max);

    Just_Compute_Extents(dimensions, Answer_Min, Answer_Max, Answer_Extent, Answer_Cell_Width, dimx, dimy, dimz);


}

void Populate_Grid(Grid *grid, int dimx, int dimy, int dimz)
{
    //The code takes in the assigned particles and scatters the
    //particles to the relevent cells. These cells are the ones that
    //were formed after uniform subdivision...

    grid->dimx = dimx;
    grid->dimy = dimy;
    grid->dimz = dimz;

    grid->number_of_uniform_subdivisions = (dimx * dimy * dimz);
    grid->Number_of_kd_subdivisions = (int *)my_malloc(grid->number_of_uniform_subdivisions * sizeof(int));
    for(int i=0; i<grid->number_of_uniform_subdivisions; i++) grid->Number_of_kd_subdivisions[i] = 0;

    grid->Positions      = (TYPE **)my_malloc(grid->number_of_uniform_subdivisions * sizeof(TYPE *));

    grid->Bdry_X = (TYPE **)my_malloc(grid->number_of_uniform_subdivisions * sizeof(TYPE *));
    grid->Bdry_Y = (TYPE **)my_malloc(grid->number_of_uniform_subdivisions * sizeof(TYPE *));
    grid->Bdry_Z = (TYPE **)my_malloc(grid->number_of_uniform_subdivisions * sizeof(TYPE *));

    grid->Range          = (int  **)my_malloc(grid->number_of_uniform_subdivisions * sizeof(int  *));

    grid->Count_Per_Thread = (int **)my_malloc(nthreads * sizeof(int *));

    int total_number_of_cells_prime =  grid->number_of_uniform_subdivisions;

    if (total_number_of_cells_prime % 64) total_number_of_cells_prime = (total_number_of_cells_prime/64 + 1) * 64;

    size_t sz = 0;
    sz += total_number_of_cells_prime * (nthreads) * sizeof(int);
    sz += (1+total_number_of_cells_prime) * (1) * sizeof(int);
    sz += (nthreads) * (1) * sizeof(int);
    sz += (nthreads) * (1) * sizeof(int);

    unsigned char *temp_memory = (unsigned char *)my_malloc(sz);
    unsigned char *temp2_memory = temp_memory;

    for(int i=0; i<nthreads; i++)
    {
        grid->Count_Per_Thread[i] = (int *)(temp2_memory); 
        temp2_memory += (total_number_of_cells_prime * sizeof(int));
    }

    grid->Count_Per_Cell = (int *)(temp2_memory); temp2_memory += ((1+total_number_of_cells_prime) * sizeof(int));


    grid->Start_Cell = (int *)(temp2_memory); temp2_memory += ((nthreads) * sizeof(int));
    grid->End_Cell = (int *)(temp2_memory); temp2_memory += ((nthreads) * sizeof(int));

    if ((temp2_memory - temp_memory) != (sz)) ERROR_PRINT();
}






void Allocate_Temporary_Arrays(void)
{
    Ranges1 = (int **)my_malloc(nthreads * sizeof(int *));
    Ranges2 = (int **)my_malloc(nthreads * sizeof(int *));
    Ranges12_Max_Size = (int *)my_malloc(nthreads * sizeof(int));

    for(int i=0; i<nthreads; i++) Ranges12_Max_Size[i] = 1024;
    for(int i=0; i<nthreads; i++)
    {
        Ranges1[i] = (int *)my_malloc(Ranges12_Max_Size[i] * sizeof(int));
        Ranges2[i] = (int *)my_malloc(Ranges12_Max_Size[i] * sizeof(int));
    }   
}


void Initialize_MPI_Mallocs(void)
{
    //This function allocates memory for data-structures that need to
    //hold MPI related data...
    static int calls = 0;
    calls++;
 
    if (calls == 1)
    {
        recv_request = (MPI_Request*)my_malloc(sizeof(MPI_Request)*(nnodes));
        send_request_key = (MPI_Request*)my_malloc(sizeof(MPI_Request)*(nnodes));
        recv_status  =  (MPI_Status*)my_malloc(sizeof(MPI_Status)*(nnodes));
    }
    else
    {
        ERROR_PRINT();
    }
}


    
void Read_D_R_File(char *filename, TYPE **i_Positions, long long int *i_number_of_galaxies_on_node)
{
    unsigned long long int stime = __rdtsc();
    long long int number_of_galaxies_on_node = -1;

    FILE *fp = fopen(filename, "rb");
    if (fp == NULL)  
    {
        printf("filename = %s :::", filename);
        ERROR_PRINT_STRING("File not found"); 
    }
    else
    {
        if (node_id == 0)
        {
            PRINT_GREEN
            printf("<%d> :: Reading %s\n", node_id, filename);
            PRINT_BLACK
        }
    }

    int valid_file = 1;

    if (fp)
    {
        unsigned char X;
        fread(&X, sizeof(char), 1, fp); if (X != 'P') valid_file = 0;
        fread(&X, sizeof(char), 1, fp); if (X != 'C') valid_file = 0;
        fread(&X, sizeof(char), 1, fp); if (X != 'L') valid_file = 0;
        fread(&X, sizeof(char), 1, fp); if (X != 'L') valid_file = 0;
        fread(&X, sizeof(char), 1, fp); if (X != 'B') valid_file = 0;
        fread(&X, sizeof(char), 1, fp); if (X != 'L') valid_file = 0;
        fread(&X, sizeof(char), 1, fp); if (X != '9') valid_file = 0;
        fread(&X, sizeof(char), 1, fp); if (X != '5') valid_file = 0;
        fread(&X, sizeof(char), 1, fp); if (X != '1') valid_file = 0;
        fread(&X, sizeof(char), 1, fp); if (X != '2') valid_file = 0;
        fread(&X, sizeof(char), 1, fp); if (X != '3') valid_file = 0;
    }
    else
    {
        valid_file = 0;
    }

    int local_number_of_galaxies;

    if (!valid_file) ERROR_PRINT_STRING("Something wrong in the input...\n");
    
    fread(&local_number_of_galaxies, sizeof(long long int), 1, fp);

    if (global_number_of_galaxies == 0)
    {
        global_number_of_galaxies = local_number_of_galaxies;
    }
    else
    {
        if (global_number_of_galaxies != local_number_of_galaxies) ERROR_PRINT_STRING("global_number_of_galaxies != local_number_of_galaxies");
    }


    long long int number_of_galaxies_per_node = (global_number_of_galaxies + nnodes - 1)/nnodes;

    global_galaxies_starting_index = number_of_galaxies_per_node * (node_id + 0);
    global_galaxies_ending_index   = number_of_galaxies_per_node * (node_id + 1);

    if (global_galaxies_starting_index > global_number_of_galaxies) global_galaxies_starting_index = global_number_of_galaxies;
    if (global_galaxies_ending_index > global_number_of_galaxies) global_galaxies_ending_index = global_number_of_galaxies;

    number_of_galaxies_on_node = global_galaxies_ending_index - global_galaxies_starting_index;

    size_t sz = (size_t)(number_of_galaxies_on_node) * (size_t)(DIMENSIONS) * (size_t)(sizeof(TYPE));

    TYPE *Positions = NULL;

    if (*i_Positions == NULL)
    {
        Positions = (TYPE *)my_malloc(sz);
    }
    else
    {
        Positions = *i_Positions;
    }

    mpi_printf("<<%d>>global_number_of_galaxies  = %lld ::: number_of_galaxies_on_node = %lld  ::: DIMENSIONS = %d\n", 
    node_id, global_number_of_galaxies, number_of_galaxies_on_node, DIMENSIONS);

    mpi_printf("global_galaxies_starting_index = %lld ::: global_galaxies_ending_index = %lld\n", global_galaxies_starting_index, global_galaxies_ending_index);

    if (valid_file)
    {
        mpi_printf("%s is a valid data file... Hence reading from it...\n", filename);
        size_t sz = (size_t)(number_of_galaxies_on_node) * (size_t)(sizeof(TYPE));
        TYPE *temp_memory = (TYPE *)my_malloc(sz);

        for(int j=0; j<DIMENSIONS; j++)
        {
            long int starting_offset = 11 + sizeof(long long int)  + j * global_number_of_galaxies * sizeof(TYPE) + global_galaxies_starting_index * sizeof(TYPE);
            int ret_val = fseek(fp, starting_offset, SEEK_SET);
            if (ret_val) ERROR_PRINT();

            size_t items_read = fread(temp_memory, sizeof(TYPE), number_of_galaxies_on_node, fp);
            
            if (items_read != number_of_galaxies_on_node) ERROR_PRINT();

            for(long long int i=0; i<number_of_galaxies_on_node; i++) 
                GET_POINT(Positions, i, j, number_of_galaxies_on_node) = temp_memory[i];
        }
        mpi_printf("<<%d>> %f %f %f\n", node_id, Positions[0 +3*79], Positions[1 + 3*79], Positions[2 + 3*79]);
    }
    else
    {
        ERROR_PRINT();
    }

    if (fp) fclose(fp);
    mpi_printf("nthreads = %d\n", nthreads);

    MPI_BARRIER(node_id);

    *i_Positions = Positions;
    *i_number_of_galaxies_on_node = number_of_galaxies_on_node;

    unsigned long long int etime = __rdtsc();
    if (node_id == 0) printf("Time Taken to read (%s) = %lld cycles (%.2lf seconds)\n", filename, (etime - stime), (etime - stime)/CORE_FREQUENCY);
}

void Initialize_Arrays(void)
{
    //This function allocates space for histograms (DR, RR) and some
    //local (per-thread) histograms...

    for(int i=0; i<HIST_BINS; i++) global_Histogram_RR[i] = 0;
    for(int i=0; i<HIST_BINS; i++) global_Histogram_DR[i] = 0;

    global_Overall_Histogram_RR = (unsigned long long int *)my_malloc(nnodes * HIST_BINS * sizeof(unsigned long long int));
    global_Overall_Histogram_DR = (unsigned long long int *)my_malloc(nnodes * HIST_BINS * sizeof(unsigned long long int));

    local_Histogram_RR = (unsigned long long int **)my_malloc(MAX_THREADS * sizeof(unsigned long long int *));
    local_Histogram_DR = (unsigned long long int **)my_malloc(MAX_THREADS * sizeof(unsigned long long int *));

    int  hist_bins = HIST_BINS;
    int hist_bins_prime = (((hist_bins + 32) >> 5)<<5); //This is to avoid cacheline conflicts and hence false sharing...
    mpi_printf("hist_bins_prime = %d\n", hist_bins_prime);


    for(int t=0; t<MAX_THREADS; t++)
    {
        local_Histogram_RR[t] = (unsigned long long int *)my_malloc(hist_bins_prime * sizeof(unsigned long long int));
        for(int k=0; k<hist_bins_prime; k++) local_Histogram_RR[t][k] = 0;
        local_Histogram_DR[t] = (unsigned long long int *)my_malloc(hist_bins_prime * sizeof(unsigned long long int));
        for(int k=0; k<hist_bins_prime; k++) local_Histogram_DR[t][k] = 0;
    }

#if 0
#endif

    #ifdef POINTS_2D
       mpi_printf("Running 2D datapoints --> 2D Auto-correlation\n");
    #else
       mpi_printf("Running 3D datapoints --> 3D Auto-correlation\n");
    #endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//STEP 3: ALLOCATE TEMPORARY ARRAYS...
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    Allocate_Temporary_Arrays();

    Initialize_MPI_Mallocs();
}

unsigned char **global_Required_during_initialization = NULL;
long long int *global_Weights_during_initialization = NULL;
int global_subdivision_per_node_during_initialization = -95123;
int *global_Count_Per_Cell_during_initialization = NULL;
int global_dimx_during_initialization;
int global_dimy_during_initialization;
int global_dimz_during_initialization;
int global_dimxy_during_initialization;
int global_number_of_subdivisions_during_initialization;
TYPE *global_Data_To_Send_during_initialization;
int *global_Prefix_Sum_Count_of_particles_to_send_during_initialization;
unsigned char *global_Local_Required_All_Nodes_during_initialization;
int *global_Send_Count_during_initialization;
TYPE *global_Local_Pos_during_initialization;
unsigned char *global_Template_during_initialization = NULL;
int *global_Template_Range_during_initialization = NULL;
int *global_Count_of_particles_to_send_during_initialization = NULL;

void *Compute_Weights_D_Parallel(void *arg1)
{
    int threadid = (int)(size_t)(arg1);
    long long int *Weights      = global_Weights_during_initialization;
    int subdivisions_per_node   = global_subdivision_per_node_during_initialization;
    int *Count_Per_Cell         = global_Count_Per_Cell_during_initialization;

    int dimx                    = global_dimx_during_initialization;
    int dimy                    = global_dimy_during_initialization;
    int dimz                    = global_dimz_during_initialization;
    int dimxy                   = global_dimxy_during_initialization;
     
    if (1)
    {
        int starting_cell_index = (node_id + 0) * subdivisions_per_node;
        int   ending_cell_index = (node_id + 1) * subdivisions_per_node;

        int cells = subdivisions_per_node;
        int cells_per_thread = (cells + nthreads - 1)/nthreads;


        int starting_cell_index_threadid = starting_cell_index + cells_per_thread * (threadid + 0);
        int   ending_cell_index_threadid = starting_cell_index + cells_per_thread * (threadid + 1);

        if (starting_cell_index_threadid > ending_cell_index) starting_cell_index_threadid = ending_cell_index;
        if (  ending_cell_index_threadid > ending_cell_index)   ending_cell_index_threadid = ending_cell_index;

        for(int current_cell_index = starting_cell_index_threadid; current_cell_index < ending_cell_index_threadid; current_cell_index++)
        {

            long long int local_weight = 0;
            int objects_in_this_cell = Count_Per_Cell[current_cell_index];

            local_weight += objects_in_this_cell;

            Weights[current_cell_index] = local_weight;
        }
    }

    return arg1;
}


void *Compute_Weights_R_Parallel(void *arg1)
{
    int threadid = (int)(size_t)(arg1);
    long long int *Weights      = global_Weights_during_initialization;
    int subdivisions_per_node   = global_subdivision_per_node_during_initialization;
    int *Count_Per_Cell         = global_Count_Per_Cell_during_initialization;

    int dimx                    = global_dimx_during_initialization;
    int dimy                    = global_dimy_during_initialization;
    int dimz                    = global_dimz_during_initialization;
    int dimxy                   = global_dimxy_during_initialization;
     
    if (1)
    {
        TYPE i_rminL = global_rminL;
        TYPE i_Lbox  = global_Lbox;
        TYPE i_rmaxL = global_rmaxL;

        TYPE rmax = i_rmaxL / i_Lbox;
        TYPE rmin = i_rminL / i_Lbox;

        TYPE rmax_2 = rmax * rmax;

        TYPE *Cell_Width = global_grid_R.Cell_Width;

        TYPE Range0_X[2], Range0_Y[2], Range0_Z[2];
        TYPE Range1_X[2], Range1_Y[2], Range1_Z[2];

        int dx = int(ceil((i_rmaxL/i_Lbox)*dimx)) + 0;
        int dy = int(ceil((i_rmaxL/i_Lbox)*dimy)) + 0;
        int dz = int(ceil((i_rmaxL/i_Lbox)*dimz)) + 0;

        int starting_cell_index = (node_id + 0) * subdivisions_per_node;
        int   ending_cell_index = (node_id + 1) * subdivisions_per_node;

        int cells = subdivisions_per_node;
        int cells_per_thread = (cells + nthreads - 1)/nthreads;


        int starting_cell_index_threadid = starting_cell_index + cells_per_thread * (threadid + 0);
        int   ending_cell_index_threadid = starting_cell_index + cells_per_thread * (threadid + 1);

        if (starting_cell_index_threadid > ending_cell_index) starting_cell_index_threadid = ending_cell_index;
        if (  ending_cell_index_threadid > ending_cell_index)   ending_cell_index_threadid = ending_cell_index;

        for(int current_cell_index = starting_cell_index_threadid; current_cell_index < ending_cell_index_threadid; current_cell_index++)
        {
            long long int local_weight = 0;
            int objects_in_this_cell = Count_Per_Cell[current_cell_index];

            int x, y, z; z = current_cell_index / dimxy; int xy = current_cell_index % dimxy; y = xy / dimx; x = xy % dimx;
    
            Range0_X[0] = (x*1.00)*Cell_Width[0]; Range0_X[1] = ((x+1)*1.00)*Cell_Width[0];
            Range0_Y[0] = (y*1.00)*Cell_Width[1]; Range0_Y[1] = ((y+1)*1.00)*Cell_Width[1];
            Range0_Z[0] = (z*1.00)*Cell_Width[2]; Range0_Z[1] = ((z+1)*1.00)*Cell_Width[2];

            local_weight += (objects_in_this_cell * (objects_in_this_cell-1))/2;

            int ccounter = 0;

            for(int zz = (z - dz); zz <= (z + dz); zz++)
            {
                for(int yy = (y - dy); yy <= (y + dy); yy++)
                {
                    int yy_prime = yy, zz_prime = zz;
                    if (yy < 0) yy_prime = yy + dimy; else if (yy >= (dimy)) yy_prime = yy - dimy;
                    if (zz < 0) zz_prime = zz + dimz; else if (zz >= (dimz)) zz_prime = zz - dimz;

	            int base_cell_index = GET_CELL_INDEX(0, yy_prime, zz_prime);
                    if (base_cell_index > current_cell_index) continue;
                    int ccounter = (zz - (z - dz)) * (2 * dy + 1) * (2 * dx + 1) + (yy - (y - dy)) * (2 * dx + 1) + 0;

                    for(int xx = (x - dx); xx <= (x + dx); xx++)
                    {
                        if (!global_Template_during_initialization[ccounter++]) continue;

                        int xx_prime = xx; 
                        if (xx < 0) xx_prime = xx + dimx; else if (xx >= (dimx)) xx_prime = xx - dimx;


                        int neighbor_cell_index = base_cell_index + xx_prime; 
                        if (neighbor_cell_index > current_cell_index) continue;

                        int objects_in_neighboring_cell = Count_Per_Cell[neighbor_cell_index];
                        local_weight += objects_in_this_cell * objects_in_neighboring_cell;
                    }
                }
            }

            Weights[current_cell_index] = local_weight;
        }
    }

    return arg1;
}


void Compute_Weights_D(int *Count_Per_Cell, long long *Weights, int subdivisions_per_node, int dimx, int dimy, int dimz, int dimxy)
{
    mpi_printf("Inside Compute_Weights_D\n");
    global_subdivision_per_node_during_initialization = subdivisions_per_node; 
    global_Weights_during_initialization = Weights;
    global_Count_Per_Cell_during_initialization = Count_Per_Cell;

    global_dimx_during_initialization = dimx;
    global_dimy_during_initialization = dimy;
    global_dimz_during_initialization = dimz;
    global_dimxy_during_initialization = dimxy;

    for(int i=1; i<nthreads; i++) pthread_create(&threads[i], NULL, Compute_Weights_D_Parallel, (void *)(i));
    Compute_Weights_D_Parallel(0);
    for(int i=1; i<nthreads; i++) pthread_join(threads[i], NULL);
}


void Compute_Weights_R(int *Count_Per_Cell, long long *Weights, int subdivisions_per_node, int dimx, int dimy, int dimz, int dimxy)
{
    mpi_printf("Inside Compute_Weights_R\n");
    global_subdivision_per_node_during_initialization = subdivisions_per_node; 
    global_Weights_during_initialization = Weights;
    global_Count_Per_Cell_during_initialization = Count_Per_Cell;

    global_dimx_during_initialization = dimx;
    global_dimy_during_initialization = dimy;
    global_dimz_during_initialization = dimz;
    global_dimxy_during_initialization = dimxy;

    for(int i=1; i<nthreads; i++) pthread_create(&threads[i], NULL, Compute_Weights_R_Parallel, (void *)(i));
    Compute_Weights_R_Parallel(0);
    for(int i=1; i<nthreads; i++) pthread_join(threads[i], NULL);
}
    
void Compute_Template_During_Initialization(Grid *grid, int dimx, int dimy, int dimz, int dimxy)
{

    TYPE i_rminL = global_rminL;
    TYPE i_Lbox  = global_Lbox;
    TYPE i_rmaxL = global_rmaxL;

    TYPE rmax = i_rmaxL / i_Lbox;
    TYPE rmin = i_rminL / i_Lbox;

    TYPE rmax_2 = rmax * rmax;

    TYPE *Cell_Width = grid->Cell_Width;

    TYPE Range0_X[2], Range0_Y[2], Range0_Z[2];
    TYPE Range1_X[2], Range1_Y[2], Range1_Z[2];

    int dx = int(ceil((i_rmaxL/i_Lbox)*dimx)) + 0;
    int dy = int(ceil((i_rmaxL/i_Lbox)*dimy)) + 0;
    int dz = int(ceil((i_rmaxL/i_Lbox)*dimz)) + 0;


    int x = dimx/2;
    int y = dimy/2;
    int z = dimz/2;

    int current_cell_index = GET_CELL_INDEX(x, y, z);

        
    Range0_X[0] = (x*1.00)*Cell_Width[0]; Range0_X[1] = ((x+1)*1.00)*Cell_Width[0];
    Range0_Y[0] = (y*1.00)*Cell_Width[1]; Range0_Y[1] = ((y+1)*1.00)*Cell_Width[1];
    Range0_Z[0] = (z*1.00)*Cell_Width[2]; Range0_Z[1] = ((z+1)*1.00)*Cell_Width[2];

    int cells_in_stencil = (2*dx +1) * (2*dy + 1) * (2*dz + 1);
    global_Template_during_initialization = (unsigned char *)malloc(cells_in_stencil * sizeof(unsigned char));
    global_Template_Range_during_initialization = (int *)malloc(2 * (2*dy + 1) * (2*dz + 1) * sizeof(int));

    for(int k=0; k<cells_in_stencil; k++) global_Template_during_initialization[k] = 0;

    int ccounter = 0;

    for(int zz = (z - dz); zz <= (z + dz); zz++)
    {
        for(int yy = (y - dy); yy <= (y + dy); yy++)
        {
            for(int xx = (x - dx); xx <= (x + dx); xx++, ccounter++)
            {
                if ((xx == x) && (yy == y) && (zz == z)) 
                {
                    global_Template_during_initialization[ccounter] = 0;
                    continue;
                }

                Range1_X[0] = (xx*1.00)*Cell_Width[0]; Range1_X[1] = ((xx+1)*1.00)*Cell_Width[0];
                Range1_Y[0] = (yy*1.00)*Cell_Width[1]; Range1_Y[1] = ((yy+1)*1.00)*Cell_Width[1];
                Range1_Z[0] = (zz*1.00)*Cell_Width[2]; Range1_Z[1] = ((zz+1)*1.00)*Cell_Width[2];

                TYPE min_dist_2 = Find_Minimum_Distance_Between_Cells(Range0_X, Range0_Y, Range0_Z, Range1_X, Range1_Y, Range1_Z);

                if (min_dist_2 > rmax_2) 
                {
                    global_Template_during_initialization[ccounter] = 0;
                    continue;
                }

                global_Template_during_initialization[ccounter] = 1;
            }
        }
    }

    if (ccounter != cells_in_stencil) ERROR_PRINT();

  
    {
        int ac = 0;
        for(int zz = (z - dz); zz <= (z + dz); zz++)
        {
            for(int yy = (y - dy); yy <= (y + dy); yy++)
            {
                int ccounter = (zz - (z - dz)) * (2*dx + 1) * (2*dx + 1) +  (yy - (y - dy)) * (2*dx + 1);
                int left_one = -dimx;
                int right_one = -dimx;
            
                for(int xx = (x - dx); xx <= (x + dx); xx++, ccounter++)
                {
                    if (global_Template_during_initialization[ccounter] == 1) 
                    {
                        left_one = xx - x;
                        break;
                    }
                }

                ccounter = (zz - (z - dz)) * (2*dx + 1) * (2*dx + 1) +  (yy - (y - dy)) * (2*dx + 1) + 2*dx;

                for(int xx = (x + dx); xx >= (x - dx); xx--, ccounter--)
                {
                    if (global_Template_during_initialization[ccounter] == 1) 
                    {
                        right_one = xx - x;
                        break;
                    }
                }

               if ((left_one == -dimx)) { if (right_one != -dimx) ERROR_PRINT(); }
               if ((left_one == -dimx)) 
               { 
               }
               else
               { 
                   if (left_one >= right_one) ERROR_PRINT();
                   if (left_one >= 0) ERROR_PRINT();
                   if (right_one <= 0) ERROR_PRINT();
               } 

                
               global_Template_Range_during_initialization[2*ac + 0] =  left_one;
               global_Template_Range_during_initialization[2*ac + 1] =  right_one;
               ac++;
            }
        }
    }
}

 
void *Compute_Required_D_For_R_Parallel(void *arg1)
{
    int threadid = (int)(size_t)(arg1);
    unsigned char *Required = global_Required_during_initialization[threadid];
    int number_of_subdivisions = global_number_of_subdivisions_during_initialization;

    int dimx                    = global_dimx_during_initialization;
    int dimy                    = global_dimy_during_initialization;
    int dimz                    = global_dimz_during_initialization;
    int dimxy                   = global_dimxy_during_initialization;
 
unsigned long long int stime = __rdtsc();
        if (1)
        {
            TYPE i_rminL = global_rminL;
            TYPE i_Lbox  = global_Lbox;
            TYPE i_rmaxL = global_rmaxL;

            TYPE rmax = i_rmaxL / i_Lbox;
            TYPE rmin = i_rminL / i_Lbox;

            TYPE rmax_2 = rmax * rmax;

            TYPE *Cell_Width = global_grid_D.Cell_Width;

            TYPE Range0_X[2], Range0_Y[2], Range0_Z[2];
            TYPE Range1_X[2], Range1_Y[2], Range1_Z[2];

            int dx = int(ceil((i_rmaxL/i_Lbox)*dimx)) + 0;
            int dy = int(ceil((i_rmaxL/i_Lbox)*dimy)) + 0;
            int dz = int(ceil((i_rmaxL/i_Lbox)*dimz)) + 0;

            int min_cell_id = (1<<29);
            int max_cell_id = -1;

            for(int current_cell_index = 0; current_cell_index < number_of_subdivisions; current_cell_index ++)
            {
                if (global_Owner_D[current_cell_index] != node_id) continue;
                min_cell_id = current_cell_index;
                break;
            }

            for(int current_cell_index = min_cell_id; current_cell_index < number_of_subdivisions; current_cell_index++)
            {
                if (global_Owner_D[current_cell_index] != node_id) 
                {
                    max_cell_id = current_cell_index - 1;
                    break;
                }
            }

            if (min_cell_id == (1<<29)) ERROR_PRINT();
            if (max_cell_id == -1)
            {
                if (node_id != (nnodes-1)) ERROR_PRINT();
                max_cell_id = number_of_subdivisions - 1;
            }
            

            int cells = max_cell_id - min_cell_id + 1;
            int cells_per_thread = (cells + nthreads - 1)/nthreads;

            int start_index = (threadid + 0) * cells_per_thread; if (start_index > cells) start_index = cells;
            int   end_index = (threadid + 1) * cells_per_thread; if (  end_index > cells)   end_index = cells;

            start_index += min_cell_id;
            end_index += min_cell_id;

unsigned long long int e2time = __rdtsc();
            for(int current_cell_index = start_index; current_cell_index < end_index; current_cell_index++)
            {
                if (global_Owner_D[current_cell_index] != node_id) ERROR_PRINT();

                Required[current_cell_index] = 1;
                int x, y, z; z = current_cell_index / dimxy; int xy = current_cell_index % dimxy; y = xy / dimx; x = xy % dimx;
        
                Range0_X[0] = (x*1.00)*Cell_Width[0]; Range0_X[1] = ((x+1)*1.00)*Cell_Width[0];
                Range0_Y[0] = (y*1.00)*Cell_Width[1]; Range0_Y[1] = ((y+1)*1.00)*Cell_Width[1];
                Range0_Z[0] = (z*1.00)*Cell_Width[2]; Range0_Z[1] = ((z+1)*1.00)*Cell_Width[2];


                int ccounter = 0;

                for(int zz = (z - dz); zz <= (z + dz); zz++)
                {
                    for(int yy = (y - dy); yy <= (y + dy); yy++)
                    {
                        for(int xx = (x - dx); xx <= (x + dx); xx++)
                        {
                            if (!global_Template_during_initialization[ccounter++]) continue;

                            int xx_prime = xx, yy_prime = yy, zz_prime = zz;
                            if (xx < 0) xx_prime = xx + dimx; else if (xx >= (dimx)) xx_prime = xx - dimx;
                            if (yy < 0) yy_prime = yy + dimy; else if (yy >= (dimy)) yy_prime = yy - dimy;
                            if (zz < 0) zz_prime = zz + dimz; else if (zz >= (dimz)) zz_prime = zz - dimz;

                            int neighbor_cell_index = GET_CELL_INDEX(xx_prime, yy_prime, zz_prime);

                            Required[neighbor_cell_index] = 1;
                        }
                    }
                }
            }
        }

        MY_BARRIER(threadid); 
unsigned long long int e1time = __rdtsc();

        int number_of_subdivisions_per_thread = (number_of_subdivisions + nthreads - 1)/nthreads;
        int start_index = (threadid + 0) * number_of_subdivisions_per_thread; if (start_index > number_of_subdivisions) start_index = number_of_subdivisions;
        int   end_index = (threadid + 1) * number_of_subdivisions_per_thread; if (end_index > number_of_subdivisions) end_index = number_of_subdivisions;

        for(int k = 0; k<nthreads; k++)
        {
            for(int j=start_index; j<end_index; j++)
            {
                if (global_Required_during_initialization[k][j] > 1) ERROR_PRINT();
                if (global_Required_during_initialization[k][j] == 1) global_Required_D_For_R[j] = 1;
            }
        }

        MY_BARRIER(threadid); 
unsigned long long int e2time = __rdtsc();


        return arg1;
}   


#if 0
#else

void *Compute_Required_R_Parallel(void *arg1)
{
    int threadid = (int)(size_t)(arg1);
    unsigned char *Required = global_Required_during_initialization[threadid];
    int number_of_subdivisions = global_number_of_subdivisions_during_initialization;

    int dimx                    = global_dimx_during_initialization;
    int dimy                    = global_dimy_during_initialization;
    int dimz                    = global_dimz_during_initialization;
    int dimxy                   = global_dimxy_during_initialization;
 
    int Index_Ranges[16];
    int ranges_found = 0;
unsigned long long int stime = __rdtsc();
        if (1)
        {
            TYPE i_rminL = global_rminL;
            TYPE i_Lbox  = global_Lbox;
            TYPE i_rmaxL = global_rmaxL;

            TYPE rmax = i_rmaxL / i_Lbox;
            TYPE rmin = i_rminL / i_Lbox;

            TYPE rmax_2 = rmax * rmax;

            TYPE *Cell_Width = global_grid_R.Cell_Width;

            TYPE Range0_X[2], Range0_Y[2], Range0_Z[2];
            TYPE Range1_X[2], Range1_Y[2], Range1_Z[2];

            int dx = int(ceil((i_rmaxL/i_Lbox)*dimx)) + 0;
            int dy = int(ceil((i_rmaxL/i_Lbox)*dimy)) + 0;
            int dz = int(ceil((i_rmaxL/i_Lbox)*dimz)) + 0;

            int min_cell_id = (1<<29);
            int max_cell_id = -1;

            for(int current_cell_index = 0; current_cell_index < number_of_subdivisions; current_cell_index ++)
            {
                if (global_Owner_R[current_cell_index] != node_id) continue;
                min_cell_id = current_cell_index;
                break;
            }

            for(int current_cell_index = min_cell_id; current_cell_index < number_of_subdivisions; current_cell_index++)
            {
                if (global_Owner_R[current_cell_index] != node_id) 
                {
                    max_cell_id = current_cell_index - 1;
                    break;
                }
            }

            if (min_cell_id == (1<<29)) ERROR_PRINT();
            if (max_cell_id == -1)
            {
                if (node_id != (nnodes-1)) ERROR_PRINT();
                max_cell_id = number_of_subdivisions - 1;
            }
            

            int cells = max_cell_id - min_cell_id + 1;
            int cells_per_thread = (cells + nthreads - 1)/nthreads;

            int start_index = (threadid + 0) * cells_per_thread; if (start_index > cells) start_index = cells;
            int   end_index = (threadid + 1) * cells_per_thread; if (  end_index > cells)   end_index = cells;

            start_index += min_cell_id;
            end_index += min_cell_id;

unsigned long long int e2time = __rdtsc();
            for(int current_cell_index = start_index; current_cell_index < end_index; current_cell_index++)
            {
                if (global_Owner_R[current_cell_index] != node_id) ERROR_PRINT();

                Required[current_cell_index] = 1;
                int x, y, z; z = current_cell_index / dimxy; int xy = current_cell_index % dimxy; y = xy / dimx; x = xy % dimx;
        
                int ac = 0;
                for(int zz = (z - dz); zz <= (z + dz); zz++)
                {
                    for(int yy = (y - dy); yy <= (y + dy); yy++, ac++)
                    {
                        int yy_prime = yy, zz_prime = zz;
                        if (yy < 0) yy_prime = yy + dimy; else if (yy >= (dimy)) yy_prime = yy - dimy;
                        if (zz < 0) zz_prime = zz + dimz; else if (zz >= (dimz)) zz_prime = zz - dimz;

			
                        int base_cell_index = GET_CELL_INDEX(0, yy_prime, zz_prime);
                        if (base_cell_index > current_cell_index) continue;

           		        int left_one = global_Template_Range_during_initialization[2*ac + 0];
           		        int right_one = global_Template_Range_during_initialization[2*ac + 1];

                        if (left_one == -dimx) continue;

			            Index_Ranges[2*0 + 0] = x + left_one;  
			            Index_Ranges[2*0 + 1] = x + right_one; 
			            ranges_found = 1;

			            if (Index_Ranges[2*0 + 1] >= dimx)
			            {
                            Index_Ranges[2*0 + 0] = 0;
                            Index_Ranges[2*0 + 1] = x + right_one - dimx;
                            Index_Ranges[2*1 + 0] = x + left_one;
                            Index_Ranges[2*1 + 1] = dimx - 1;
                            ranges_found = 2;
                        }
			            else if (Index_Ranges[2*0 + 0] < 0)
			            {
                            Index_Ranges[2*0 + 0] = 0;
                            Index_Ranges[2*0 + 1] = x + right_one;
                            Index_Ranges[2*1 + 0] = x + left_one + dimx;
                            Index_Ranges[2*1 + 1] = dimx - 1;
                            ranges_found = 2;
			            }


                        if (ranges_found == 1)
                        {
                            int neighbor_cell_index0 = base_cell_index + Index_Ranges[0];
                            int neighbor_cell_index1 = base_cell_index + Index_Ranges[1];
                            for(int p = neighbor_cell_index0; p<= neighbor_cell_index1; p++) 
                            {
                                if (p >= current_cell_index) break;
                                if (!Required[p]) Required[p] = 1;
                            }
                            
                        }
                        else if (ranges_found == 2)
                        {			
                            int neighbor_cell_index0 = base_cell_index + Index_Ranges[0];
                            int neighbor_cell_index1 = base_cell_index + Index_Ranges[1];
                            for(int p = neighbor_cell_index0; p<= neighbor_cell_index1; p++) 
                            {
                                if (p >= current_cell_index) break;
                                if (!Required[p]) Required[p] = 1;
                            }

                            int neighbor_cell_index2 = base_cell_index + Index_Ranges[2];
                            int neighbor_cell_index3 = base_cell_index + Index_Ranges[3];
                            for(int p = neighbor_cell_index2; p<= neighbor_cell_index3; p++) 
                            {
                                if (p >= current_cell_index) break;
                                if (!Required[p]) Required[p] = 1;
                            }
                        }
                    }
                }
            }
        }

        MY_BARRIER(threadid); 
unsigned long long int e1time = __rdtsc();
if ((node_id == 0) && (threadid == 0)) {/*printf("(%d) (%s) ::: Time Taken = %lld (%.2lf seconds)\n", __LINE__, __FILE__, (e1time - stime), (e1time - stime)/CORE_FREQUENCY);*/}

        int number_of_subdivisions_per_thread = (number_of_subdivisions + nthreads - 1)/nthreads;
        int start_index = (threadid + 0) * number_of_subdivisions_per_thread; if (start_index > number_of_subdivisions) start_index = number_of_subdivisions;
        int   end_index = (threadid + 1) * number_of_subdivisions_per_thread; if (end_index > number_of_subdivisions) end_index = number_of_subdivisions;

        for(int k = 0; k<nthreads; k++)
        {
            for(int j=start_index; j<end_index; j++)
            {
                if (global_Required_during_initialization[k][j] > 1) ERROR_PRINT();
                if (global_Required_during_initialization[k][j] == 1) global_Required_R[j] = 1;
            }
        }

        MY_BARRIER(threadid); 
unsigned long long int e2time = __rdtsc();
if ((node_id == 0) && (threadid == 0)) {/*printf("(%d) (%s) ::: Time Taken = %lld (%.2lf seconds)\n", __LINE__, __FILE__, (e2time - stime), (e2time - stime)/CORE_FREQUENCY);*/}


        return arg1;
}
#endif

void Compute_Required_R(int number_of_subdivisions, int dimx, int dimy, int dimz, int dimxy)
{
    mpi_printf("<<%d>> Inside Compute_Required_R\n", node_id);
    global_number_of_subdivisions_during_initialization = number_of_subdivisions;

    global_dimx_during_initialization = dimx;
    global_dimy_during_initialization = dimy;
    global_dimz_during_initialization = dimz;
    global_dimxy_during_initialization = dimxy;

    {
        for(int k = 0; k < nthreads; k++)
        {
            for(int t = 0; t < number_of_subdivisions; t++)
            {
                global_Required_during_initialization[k][t] = 0;
            }
        }
    }
            
unsigned long long int stime = __rdtsc();
    for(int i=1; i<nthreads; i++) pthread_create(&threads[i], NULL, Compute_Required_R_Parallel, (void *)(i));
    Compute_Required_R_Parallel(0);
    for(int i=1; i<nthreads; i++) pthread_join(threads[i], NULL);
unsigned long long int etime = __rdtsc();
}



void Compute_Required_D_For_R(int number_of_subdivisions, int dimx, int dimy, int dimz, int dimxy)
{
    mpi_printf("<<%d>> Inside Compute_Required_D_For_R\n", node_id);
    global_number_of_subdivisions_during_initialization = number_of_subdivisions;

    global_dimx_during_initialization = dimx;
    global_dimy_during_initialization = dimy;
    global_dimz_during_initialization = dimz;
    global_dimxy_during_initialization = dimxy;

    global_Required_during_initialization = (unsigned char **)malloc(nthreads * sizeof(unsigned char *));
    {
        unsigned char *TT1 = (unsigned char *)malloc(nthreads * number_of_subdivisions * sizeof(unsigned char));
        for(int k=0; k<(nthreads * number_of_subdivisions); k++) TT1[k] = 0;
        unsigned char *TT2 = TT1;
        for(int k=0; k<nthreads; k++)
        {
            global_Required_during_initialization[k] = TT2;
            TT2 += number_of_subdivisions;
        }

        if ( (TT2-TT1) != (nthreads * number_of_subdivisions)) ERROR_PRINT();
    }
            
unsigned long long int stime = __rdtsc();
    for(int i=1; i<nthreads; i++) pthread_create(&threads[i], NULL, Compute_Required_D_For_R_Parallel, (void *)(i));
    Compute_Required_D_For_R_Parallel(0);
    for(int i=1; i<nthreads; i++) pthread_join(threads[i], NULL);
unsigned long long int etime = __rdtsc();
}






void *Compute_Count_of_particles_to_send_Parallel(void *arg1)
{
    int threadid = (int)(size_t)(arg1);
    unsigned char *Local_Required_All_Nodes = global_Local_Required_All_Nodes_during_initialization;
    int *Count_of_particles_to_send = global_Count_of_particles_to_send_during_initialization;
    int number_of_subdivisions = global_number_of_subdivisions_during_initialization;
	int *Send_Count = global_Send_Count_during_initialization;

    int nnodes_per_thread = (nnodes + nthreads - 1)/nthreads;
    int start_node = (threadid + 0) * nnodes_per_thread;
    int   end_node = (threadid + 1) * nnodes_per_thread;

    if (start_node > nnodes) start_node = nnodes;
    if (  end_node > nnodes)   end_node = nnodes;

    int number_of_subdivisions_by_eight = number_of_subdivisions/8;
    if ((number_of_subdivisions_by_eight * 8) != number_of_subdivisions) number_of_subdivisions_by_eight++;

    for(int k=start_node; k < end_node; k++)
    {
        unsigned char *Src = Local_Required_All_Nodes + k*number_of_subdivisions_by_eight;
        int sum = 0;

        for(int m=0; m<number_of_subdivisions_by_eight; m++)
        {
            if (Src[m])
            {
                unsigned char value = Src[m];
                unsigned char *NOO = Number_Of_Ones + 9*value;
                int m_prime_base = m * 8;

                for(int j=1; j<=NOO[0]; j++)
                {
                    int p = NOO[j];
                    int m_prime =  m_prime_base + p;
                    sum += Send_Count[m_prime + 1] - Send_Count[m_prime];
                }
             }
         }

         Count_of_particles_to_send[k] =  sum;
    }

    return arg1;
}

void  Compute_Count_of_particles_to_send(unsigned char *Local_Required_All_Nodes, int *Count_of_particles_to_send, int *Send_Count, int number_of_subdivisions)
{

    global_Local_Required_All_Nodes_during_initialization = Local_Required_All_Nodes;
    global_Count_of_particles_to_send_during_initialization = Count_of_particles_to_send;
    global_number_of_subdivisions_during_initialization = number_of_subdivisions;
    global_Send_Count_during_initialization = Send_Count;

    for(int i=1; i<nthreads; i++) pthread_create(&threads[i], NULL, Compute_Count_of_particles_to_send_Parallel, (void *)(i));
    Compute_Count_of_particles_to_send_Parallel(0);
    for(int i=1; i<nthreads; i++) pthread_join(threads[i], NULL);

}


void *Prepare_Data_To_Send_Parallel(void *arg1)
{
    int threadid = (int)(size_t)(arg1);

	TYPE *Data_To_Send = global_Data_To_Send_during_initialization;
	int *Prefix_Sum_Count_of_particles_to_send = global_Prefix_Sum_Count_of_particles_to_send_during_initialization;
	unsigned char *Local_Required_All_Nodes = global_Local_Required_All_Nodes_during_initialization;
	int number_of_subdivisions = global_number_of_subdivisions_during_initialization;
	int *Send_Count = global_Send_Count_during_initialization;
	TYPE *Local_Pos = global_Local_Pos_during_initialization;

    int number_of_subdivisions_by_eight = number_of_subdivisions/8;
    if ((number_of_subdivisions_by_eight * 8) != number_of_subdivisions) number_of_subdivisions_by_eight++;

	int nnodes_per_thread = (nnodes + nthreads - 1)/nthreads;
	int start_node = (threadid + 0) * nnodes_per_thread; if (start_node > nnodes) start_node = nnodes;
	int   end_node = (threadid + 1) * nnodes_per_thread; if (  end_node > nnodes)   end_node = nnodes;

    for(int receiver  = start_node; receiver < end_node; receiver++)
    {
        TYPE *Src = Data_To_Send + Prefix_Sum_Count_of_particles_to_send[receiver] * 3;
        int offset = 0;
        unsigned char *Req = Local_Required_All_Nodes + receiver * number_of_subdivisions_by_eight;

        for(int k = 0; k<number_of_subdivisions_by_eight; k++)
        {
            if (Req[k])
            {
                unsigned char value = Req[k];
                int k_prime_base = 8*k;
                unsigned char *NOO = Number_Of_Ones + 9*value;

                for(int j=1; j<=NOO[0]; j++)
                {
                    int p = NOO[j];
                    int k_prime = k_prime_base + p;
			        size_t particles_to_send = (Send_Count[k_prime + 1] - Send_Count[k_prime]);
			        size_t  floats_to_send = 3 * particles_to_send;
			        size_t bytes_to_send = floats_to_send * sizeof(TYPE);

			        TYPE *Copying_Src = Local_Pos + 3*Send_Count[k_prime];
			        TYPE *Copying_Dst = Src + 3*offset;

#if 1
			        memcpy(Copying_Dst, Copying_Src, bytes_to_send);
			        offset += particles_to_send;
#else

                    for(int p=Send_Count[k_prime]; p < Send_Count[k_prime+1]; p++)
                    {
                        Src[3*offset + 0] = Local_Pos[3*p + 0];
                        Src[3*offset + 1] = Local_Pos[3*p + 1];
                        Src[3*offset + 2] = Local_Pos[3*p + 2];
                        offset++;
                    }
#endif
                 }
             }
        }
        if (offset != (Prefix_Sum_Count_of_particles_to_send[receiver+1] - Prefix_Sum_Count_of_particles_to_send[receiver])) ERROR_PRINT();
	}
	return arg1;
}
		

void Prepare_Data_To_Send(TYPE *Local_Pos, TYPE *Data_To_Send, int *Prefix_Sum_Count_of_particles_to_send, unsigned char *Local_Required_All_Nodes, int *Send_Count, int number_of_subdivisions)
{
	global_Data_To_Send_during_initialization = Data_To_Send;
	global_Prefix_Sum_Count_of_particles_to_send_during_initialization = Prefix_Sum_Count_of_particles_to_send;
	global_Local_Required_All_Nodes_during_initialization = Local_Required_All_Nodes;
	global_number_of_subdivisions_during_initialization = number_of_subdivisions;
	global_Send_Count_during_initialization = Send_Count;
	global_Local_Pos_during_initialization = Local_Pos;


    	for(int i=1; i<nthreads; i++) pthread_create(&threads[i], NULL, Prepare_Data_To_Send_Parallel, (void *)(i));
    	Prepare_Data_To_Send_Parallel(0);
    	for(int i=1; i<nthreads; i++) pthread_join(threads[i], NULL);
}

void  Convert_From_Byte_To_Bit(unsigned char *Required, int count)
{

    int count_prime = 0;
    int count_floored_to_multiple_of_eight = (count/8)*8;

    for(int k=0; k<count; k++)
    {
        if ( (Required[k] != 0) && (Required[k] != 1))
        {
            ERROR_PRINT();
        }
    }


    for(int k=0; k<count_floored_to_multiple_of_eight; k+= 8)
    {
        unsigned char value = (Required[k+0] << 7) + (Required[k+1] << 6) + 
                              (Required[k+2] << 5) + (Required[k+3] << 4) + 
                              (Required[k+4] << 3) + (Required[k+5] << 2) + 
                              (Required[k+6] << 1) + (Required[k+7] << 0) ;

         Required[count_prime]  = value;
         count_prime++;
     }

     if (count_floored_to_multiple_of_eight != count)
     {
        if (node_id == 0) { printf("nnodes is not a multiple of 8... A few new lines of code being executed...\n"); fflush(stdout); }
        int k = count_floored_to_multiple_of_eight;
        unsigned char value = 0;

        int remaining = count-count_floored_to_multiple_of_eight;

        if (remaining <= 0) ERROR_PRINT();
        if (remaining >= 8) ERROR_PRINT();
        for(int p=0; p<remaining; p++)
        {
            value += (Required[k+p] << (7-p));
        }
        Required[count_prime] = value;
     }

            

}


int *global_Prealloced_Send_Count = NULL;
int *global_Prealloced_Recv_Count = NULL;
int *global_Prealloced_Count_Per_Cell = NULL;

void Distribute_D_Particles_Amongst_Nodes(void)
{
    unsigned long long int stime = __rdtsc();
    Compute_Number_of_Ones();

    int dimx, dimy, dimz;
    int dimxy;
    int number_of_subdivisions;

    //The code below computes the number of uniform divisions of the
    //total grid... dimx, dimy, dimz... they are set in a way that
    //number_of_subdivisions is a multiple of nnodes...
    {
        int multiple = PCL_MAX(10 * nthreads, 100);  
        multiple = nthreads * ((multiple + nthreads - 1)/nthreads);

        int rough_estimate_for_total_number_of_cells = nnodes * multiple;

        dimx  = pow(rough_estimate_for_total_number_of_cells, (1.0/3.0));

        if ((dimx * dimx * dimx) < rough_estimate_for_total_number_of_cells) dimx++;
        dimy = dimx; 
        dimz = dimx;

        size_t another_approximate_for_particles_per_cell = global_number_of_galaxies / (dimx * dimy * dimz);

        int threshold = 500; //500
        if (another_approximate_for_particles_per_cell > threshold)
        {   
            rough_estimate_for_total_number_of_cells = global_number_of_galaxies/threshold;
            dimx  = pow(rough_estimate_for_total_number_of_cells, (1.0/3.0));
            if ((dimx * dimx * dimx) < rough_estimate_for_total_number_of_cells) dimx++;
            dimy = dimx;
            dimz = dimx;
        }

        if (dimx % 2) dimx++;
        dimy = dimx; 
        dimz = dimx;

        if (dimx % 2) ERROR_PRINT();
        if (dimy % 2) ERROR_PRINT();
        if (dimz % 2) ERROR_PRINT();

        number_of_subdivisions = dimx * dimy * dimz;

        dimxy = dimx * dimy;

        mpi_printf("dimx = %d ::: dimy = %d :: dimz = %d\n", dimx, dimy, dimz);

        number_of_subdivisions = ((number_of_subdivisions + nnodes - 1)/nnodes) * nnodes;
    }


    int subdivisions_per_node = (number_of_subdivisions + nnodes - 1)/nnodes;
    if ( (subdivisions_per_node * nnodes) != number_of_subdivisions) ERROR_PRINT();

    mpi_printf("number_of_subdivisions = %d ::: subdivisions_per_node = %d\n", number_of_subdivisions, subdivisions_per_node);

    if (number_of_subdivisions % nnodes) ERROR_PRINT();


    int *Send_Count = (int *)my_malloc((1 + number_of_subdivisions) * sizeof(int));
    int *Recv_Count = (int *)my_malloc((1 + number_of_subdivisions) * sizeof(int));
    int *Count_Per_Cell = (int *)my_malloc((1 + number_of_subdivisions) * sizeof(int));

    global_Prealloced_Send_Count = Send_Count;
    global_Prealloced_Recv_Count = Recv_Count;
    global_Prealloced_Count_Per_Cell = Count_Per_Cell;

    TYPE *MinMax = (TYPE *)my_malloc(nnodes * 6 * sizeof(TYPE));

    TYPE Local_Min[3], Local_Max[3];
    TYPE Local_MinMax[6];

    Just_Compute_Min_Max(global_Positions_D, global_number_of_galaxies_on_node_D, DIMENSIONS, Local_Min, Local_Max);

    mpi_printf("LMin = [%e %e %e] ::: [%e %e %e]\n", Local_Min[0], Local_Min[1], Local_Min[2], Local_Max[0], Local_Max[1], Local_Max[2]);


    //Each node computes Local_MinMax, and then that data is broadcast
    //to all nodes (using MPI_Allgather)...

#if 0
#else
    Local_MinMax[0] = Local_Min[0];
    Local_MinMax[1] = Local_Min[1];
    Local_MinMax[2] = Local_Min[2];
    Local_MinMax[3] = Local_Max[0];
    Local_MinMax[4] = Local_Max[1];
    Local_MinMax[5] = Local_Max[2];
    MPI_Allgather(Local_MinMax, 6, MPI_FLOAT, MinMax, 6, MPI_FLOAT, MPI_COMM_WORLD);
#endif

    for(int i=0; i<nnodes; i++)
    {
        Local_Min[0] = PCL_MIN(Local_Min[0], MinMax[6*i + 0]);
        Local_Min[1] = PCL_MIN(Local_Min[1], MinMax[6*i + 1]);
        Local_Min[2] = PCL_MIN(Local_Min[2], MinMax[6*i + 2]);

        Local_Max[0] = PCL_MAX(Local_Max[0], MinMax[6*i + 3]);
        Local_Max[1] = PCL_MAX(Local_Max[1], MinMax[6*i + 4]);
        Local_Max[2] = PCL_MAX(Local_Max[2], MinMax[6*i + 5]);
    }

    global_grid_D.Min[0] = Local_Min[0];
    global_grid_D.Min[1] = Local_Min[1];
    global_grid_D.Min[2] = Local_Min[2];

    global_grid_D.Max[0] = Local_Max[0];
    global_grid_D.Max[1] = Local_Max[1];
    global_grid_D.Max[2] = Local_Max[2];


    Just_Compute_Extents(DIMENSIONS, global_grid_D.Min, global_grid_D.Max, global_grid_D.Extent, global_grid_D.Cell_Width, dimx, dimy, dimz);

    //A Template is created that computes the neighboring cells that
    //need to be accessed...
    Compute_Template_During_Initialization(&global_grid_D, dimx, dimy, dimz, dimxy);

unsigned long long int e7time = __rdtsc();
if (node_id == 0) {/*printf("(%d) (%s) ::: Time Taken = %lld (%.2lf seconds)\n", __LINE__, __FILE__, (e7time - stime), (e7time - stime)/CORE_FREQUENCY);*/}

    {
        //The code below computes the number of particles per cell...
      
        for(int k=0; k<=number_of_subdivisions; k++) Send_Count[k] = 0;
        for(int k=0; k<=number_of_subdivisions; k++) Recv_Count[k] = 0;

        TYPE *Local_Pos = (TYPE *)malloc(global_number_of_galaxies_on_node_D * sizeof(TYPE) * DIMENSIONS);

        int dimxx = dimx - 1;
        int dimyy = dimy - 1;
        int dimzz = dimz - 1;

        TYPE *Min = global_grid_D.Min;
        TYPE *Max = global_grid_D.Max;
        TYPE *Extent  = global_grid_D.Extent;

        for(int i=0; i<global_number_of_galaxies_on_node_D; i++)
        {
            TYPE float_x = global_Positions_D[3*i + 0];
            TYPE float_y = global_Positions_D[3*i + 1];
            TYPE float_z = global_Positions_D[3*i + 2];
                
            int cell_x = ((float_x - Min[0]) * dimx)/Extent[0];
            int cell_y = ((float_y - Min[1]) * dimy)/Extent[1];
            int cell_z = ((float_z - Min[2]) * dimz)/Extent[2];

            CLAMP_BELOW(cell_x, 0); CLAMP_ABOVE(cell_x, (dimxx));
            CLAMP_BELOW(cell_y, 0); CLAMP_ABOVE(cell_y, (dimyy));
            CLAMP_BELOW(cell_z, 0); CLAMP_ABOVE(cell_z, (dimzz));
            
            int cell_id = GET_CELL_INDEX(cell_x, cell_y, cell_z);

            Send_Count[cell_id]++;
        }

unsigned long long int e8time = __rdtsc();
if (node_id == 0) {/*printf("(%d) (%s) ::: Time Taken = %lld (%.2lf seconds)\n", __LINE__, __FILE__, (e8time - stime), (e8time - stime)/CORE_FREQUENCY);*/}

        {
            int sum_so_far = 0;
            for(int k=0; k<number_of_subdivisions; k++)
            {
                int curr_value = Send_Count[k];
                Send_Count[k] = sum_so_far;
                sum_so_far += curr_value;
            }
            Send_Count[number_of_subdivisions] = sum_so_far;

            if (sum_so_far != global_number_of_galaxies_on_node_D) ERROR_PRINT();
        }



        for(int i=0; i<global_number_of_galaxies_on_node_D; i++)
        {
            TYPE float_x = global_Positions_D[3*i + 0];
            TYPE float_y = global_Positions_D[3*i + 1];
            TYPE float_z = global_Positions_D[3*i + 2];
                
            int cell_x = ((float_x - Min[0]) * dimx)/Extent[0];
            int cell_y = ((float_y - Min[1]) * dimy)/Extent[1];
            int cell_z = ((float_z - Min[2]) * dimz)/Extent[2];

            CLAMP_BELOW(cell_x, 0); CLAMP_ABOVE(cell_x, (dimxx));
            CLAMP_BELOW(cell_y, 0); CLAMP_ABOVE(cell_y, (dimyy));
            CLAMP_BELOW(cell_z, 0); CLAMP_ABOVE(cell_z, (dimzz));
            
            int cell_id = GET_CELL_INDEX(cell_x, cell_y, cell_z);

            Local_Pos[3* Send_Count[cell_id] + 0] = float_x;
            Local_Pos[3* Send_Count[cell_id] + 1] = float_y;
            Local_Pos[3* Send_Count[cell_id] + 2] = float_z;
            Send_Count[cell_id]++;
        }

        if (Send_Count[number_of_subdivisions-1] != global_number_of_galaxies_on_node_D) ERROR_PRINT();

        for(int k=(number_of_subdivisions-1); k>=1; k--) Send_Count[k] = Send_Count[k] - Send_Count[k-1];

unsigned long long int e9time = __rdtsc();
if (node_id == 0) {/*printf("(%d) (%s) ::: Time Taken = %lld (%.2lf seconds)\n", __LINE__, __FILE__, (e9time - stime), (e9time - stime)/CORE_FREQUENCY);*/}

#if 0
#endif

    #if 0
    #else
        MPI_Alltoall(Send_Count, subdivisions_per_node, MPI_INT, Recv_Count, subdivisions_per_node, MPI_INT, MPI_COMM_WORLD);
    #endif

unsigned long long int e10time = __rdtsc();
if (node_id == 0) {/*printf("(%d) (%s) ::: Time Taken = %lld (%.2lf seconds)\n", __LINE__, __FILE__, (e10time - stime), (e10time - stime)/CORE_FREQUENCY);*/}

#if 0
#endif


        {
            for(int k=0; k<number_of_subdivisions; k++) Count_Per_Cell[k] = 0;
            int *Dst = Count_Per_Cell + node_id * subdivisions_per_node;
            for(int k=0; k<nnodes; k++) 
            {
                int *Src = Recv_Count + k*subdivisions_per_node;
                for(int l=0; l<subdivisions_per_node; l++)
                {
                    Dst[l] += Src[l];
                }
            }

#if 0
#else
            int *Local_Count_Per_Cell = (int *)malloc(subdivisions_per_node * sizeof(int));
            int offfset = node_id * subdivisions_per_node;
            for(int k=0; k<subdivisions_per_node; k++) Local_Count_Per_Cell[k] = Count_Per_Cell[offfset + k];
            MPI_Allgather(Local_Count_Per_Cell, subdivisions_per_node, MPI_INT, Count_Per_Cell, subdivisions_per_node, MPI_INT, MPI_COMM_WORLD);

#endif
		
	#if 0
    #endif
        }

unsigned long long int e1time = __rdtsc();
if (node_id == 0) {/*printf("(%d) (%s) ::: Time Taken = %lld (%.2lf seconds)\n", __LINE__, __FILE__, (e1time - stime), (e1time - stime)/CORE_FREQUENCY);*/}

        long long int *Weights = (long long int *)malloc(number_of_subdivisions * sizeof(long long int));
        long long int *local_Weights = (long long int *)malloc(subdivisions_per_node * sizeof(long long int));

        //The code below computes the weight per cell... Refer to the SC'12 paper by Chhugani et al.
    
        Compute_Weights_D(Count_Per_Cell, Weights, subdivisions_per_node, dimx, dimy, dimz, dimxy);

        {
            int starting_cell_index = (node_id + 0) * subdivisions_per_node;
            int   ending_cell_index = (node_id + 1) * subdivisions_per_node;
            int cells = subdivisions_per_node;
            for(int k=0; k<cells; k++) local_Weights[k] = Weights[starting_cell_index + k];
        }

#if 0
#else
        MPI_Allgather(local_Weights, subdivisions_per_node, MPI_LONG_LONG_INT, Weights, subdivisions_per_node, MPI_LONG_LONG_INT, MPI_COMM_WORLD);
#endif

unsigned long long int e2time = __rdtsc();
if (node_id == 0) {/*printf("(%d) (%s) ::: Time Taken = %lld (%.2lf seconds)\n", __LINE__, __FILE__, (e2time - stime), (e2time - stime)/CORE_FREQUENCY);*/}


        //The code below computes the owner of each cell... Each cell
        //is owned by some node. For details, refer to the paper...

        global_Owner_D = (int *)malloc(number_of_subdivisions * sizeof(int));
        for(int k=0; k<number_of_subdivisions; k++) global_Owner_D[k] = -1; 

        {
            long long int sum_so_far = 0;
            for(int k=0; k<number_of_subdivisions; k++) sum_so_far  += Weights[k];
            long long int total_weights = sum_so_far;
            long long int weights_per_node = (sum_so_far + nnodes - 1)/nnodes;
            mpi_printf("total_weights = %lld ::: weights_per_node = %lld\n", total_weights, weights_per_node);
            fflush(stdout);
            int alloted_so_far = 0;

            sum_so_far = 0;
            for(int k=0; k<nnodes; k++)
            {
                long long int starting_index = (k + 0) * weights_per_node;
                long long int   ending_index = (k + 1) * weights_per_node;

                if (starting_index > total_weights ) starting_index = total_weights;
                if (  ending_index > total_weights)   ending_index = total_weights;

                while (sum_so_far < ending_index)
                {
                    sum_so_far += Weights[alloted_so_far];
                    global_Owner_D[alloted_so_far] = k;
                    alloted_so_far++;
                }
            }

            {

                for(int k = number_of_subdivisions-1; k >= 0; k--)
                {
                    if (global_Owner_D[k] == -1)
                    {
                        if (Weights[k] == 0) global_Owner_D[k] = nnodes-1;
                        else ERROR_PRINT();
                    }
                }

            }
        
            int *Temp1 = (int *)malloc(nnodes * sizeof(int)); for(int k=0; k<nnodes; k++) Temp1[k] = 0;
            int actual_subdivisions = dimx * dimy * dimz;
            for(int k=0; k<actual_subdivisions; k++) 
            {
                if (global_Owner_D[k] == -1) { printf("k = %d\n", k); ERROR_PRINT(); }
                Temp1[global_Owner_D[k]] += Count_Per_Cell[k];
            }

            {
                size_t final_sum = 0;
                int min = Temp1[0];
                int max = Temp1[0];
                for(int k=0; k<nnodes; k++) 
                {
                    final_sum += Temp1[k];
                    min = PCL_MIN(Temp1[k], min);
                    max = PCL_MAX(Temp1[k], max);
                }
                size_t avg = final_sum/nnodes;
                mpi_printf("Avg. = %d ::: max = %d ::: Ratio = %.2lf\n", (int)(avg), max, (max*1.0)/avg);

                if (final_sum != global_number_of_galaxies) ERROR_PRINT();
            }
        }

unsigned long long int e4time = __rdtsc();
if (node_id == 0) {/*printf("(%d) (%s) ::: Time Taken = %lld (%.2lf seconds)\n", __LINE__, __FILE__, (e4time - stime), (e4time - stime)/CORE_FREQUENCY);*/}


        global_Required_D = (unsigned char *)malloc(number_of_subdivisions *sizeof(unsigned char));
        global_Required_D_For_R = (unsigned char *)malloc(number_of_subdivisions *sizeof(unsigned char));
        for(int k=0; k<number_of_subdivisions; k++) global_Required_D[k] = 0;
        for(int k=0; k<number_of_subdivisions; k++) global_Required_D_For_R[k] = 0;

        for(int k = 0; k < number_of_subdivisions; k++)
        {
            if (global_Owner_D[k] == node_id) global_Required_D[k] = 1;
        }
        Compute_Required_D_For_R(number_of_subdivisions, dimx, dimy, dimz, dimxy);
#if 0
#endif


unsigned long long int e3time = __rdtsc();
if (node_id == 0) {/*printf("(%d) (%s) ::: Time Taken = %lld (%.2lf seconds)\n", __LINE__, __FILE__, (e3time - stime), (e3time - stime)/CORE_FREQUENCY); */}

        {
            int ssum = 0; 
            int particle_sum = 0;
            
            for(int k=0; k<number_of_subdivisions; k++) 
            {
                ssum += global_Required_D[k];
                if (global_Required_D[k]) particle_sum += Count_Per_Cell[k];
            }

            mpi_printf("node_id = %d ::: ssum = %d (number_of_subdivisions = %d) ::: particle_sum = %d (global_number_of_galxies = %lld)\n", 
                node_id, ssum, number_of_subdivisions, particle_sum, global_number_of_galaxies);
        }


        unsigned char *Local_Required_Bkp = (unsigned char *)malloc(number_of_subdivisions * sizeof(unsigned char));
        for(int k=0; k<number_of_subdivisions; k++) Local_Required_Bkp[k] = global_Required_D[k];

        //The code below tries to compress the data -- so that the
        //data transferred is reduced... Basically using 1-bit per
        //cell (as opposed to bytes). This leads to a marginal
        //increase in run-time, but dramatically reduces the data
        //transferred -- especially for large datasets and/or number
        //of nodes...

        Convert_From_Byte_To_Bit(global_Required_D, number_of_subdivisions);
        int number_of_subdivisions_by_eight = number_of_subdivisions/8;
        if ((number_of_subdivisions_by_eight * 8) != number_of_subdivisions) number_of_subdivisions_by_eight++;
        unsigned char *Local_Required_All_Nodes = (unsigned char *)malloc(nnodes * number_of_subdivisions_by_eight * sizeof(unsigned char));
        {

    #if 0
    #else
            MPI_Allgather(global_Required_D, number_of_subdivisions_by_eight, MPI_UNSIGNED_CHAR, 
                          Local_Required_All_Nodes, number_of_subdivisions_by_eight, MPI_UNSIGNED_CHAR, MPI_COMM_WORLD);
        
    #endif
        }
unsigned long long int e12time = __rdtsc();
if (node_id == 0) {/*printf("(%d) (%s) ::: Time Taken = %lld (%.2lf seconds)\n", __LINE__, __FILE__, (e12time - stime), (e12time - stime)/CORE_FREQUENCY);*/}

        {
            int sum_so_far = 0;
            for(int k=0; k<number_of_subdivisions; k++)
            {
                int curr_value = Send_Count[k];
                Send_Count[k] = sum_so_far;
                sum_so_far += curr_value;
            }
            Send_Count[number_of_subdivisions] = sum_so_far;

            if (sum_so_far != global_number_of_galaxies_on_node_D) ERROR_PRINT();
        }


        {
            int *Count_of_particles_to_send = (int *)malloc(nnodes * sizeof(int));
            int *Count_of_particles_to_recv = (int *)malloc(nnodes * sizeof(int));

            Compute_Count_of_particles_to_send(Local_Required_All_Nodes, Count_of_particles_to_send, Send_Count, number_of_subdivisions);
        #if 0      
        #endif
        #if 0
        #else
            MPI_Alltoall(Count_of_particles_to_send, 1, MPI_INT, Count_of_particles_to_recv, 1, MPI_INT, MPI_COMM_WORLD);
        #endif
unsigned long long int e11time = __rdtsc();
if (node_id == 0) {/*printf("(%d) (%s) ::: Time Taken = %lld (%.2lf seconds)\n", __LINE__, __FILE__, (e11time - stime), (e11time - stime)/CORE_FREQUENCY);*/}

             int total_particles_to_send;
             int total_particles_to_recv;
             {
                int sum0 = 0, sum1 = 0;
                for(int k=0; k<nnodes; k++) sum0 += Count_of_particles_to_send[k];
                for(int k=0; k<nnodes; k++) sum1 += Count_of_particles_to_recv[k];
                mpi_printf("node_id = %d ::: sum0 = %d ::: sum1 = %d\n", node_id, sum0, sum1);
                total_particles_to_send = sum0;
                total_particles_to_recv = sum1;
             }

             int *Prefix_Sum_Count_of_particles_to_send = (int *)malloc((nnodes + 1) * sizeof(int));
             int *Prefix_Sum_Count_of_particles_to_recv = (int *)malloc((nnodes + 1) * sizeof(int));
             {
                int sum = 0;
                for(int k=0; k<nnodes; k++)
                {
                    int value = Count_of_particles_to_send[k];
                    Prefix_Sum_Count_of_particles_to_send[k] = sum;
                    sum += value;
                }
                Prefix_Sum_Count_of_particles_to_send[nnodes] = sum;
                if (sum != total_particles_to_send) ERROR_PRINT();

                sum = 0;
                for(int k=0; k<nnodes; k++)
                {
                    int value = Count_of_particles_to_recv[k];
                    Prefix_Sum_Count_of_particles_to_recv[k] = sum;
                    sum += value;
                }
                Prefix_Sum_Count_of_particles_to_recv[nnodes] = sum;
                if (sum != total_particles_to_recv) ERROR_PRINT();
            }


            TYPE *Data_To_Send = (TYPE *)malloc(total_particles_to_send * 3 * sizeof(TYPE));
            TYPE *Data_To_Recv = (TYPE *)malloc(total_particles_to_recv * 3 * sizeof(TYPE));

#if 0 // <<----
#else

		Prepare_Data_To_Send(Local_Pos, Data_To_Send, Prefix_Sum_Count_of_particles_to_send, Local_Required_All_Nodes, Send_Count, number_of_subdivisions);
unsigned long long int e6time = __rdtsc();
if (node_id == 0) {/*printf("(%d) (%s) ::: Time Taken = %lld (%.2lf seconds)\n", __LINE__, __FILE__, (e6time - stime), (e6time - stime)/CORE_FREQUENCY);*/}

    for(int k=0; k<nnodes; k++) Count_of_particles_to_send[k] = Count_of_particles_to_send[k] * 3;
    for(int k=0; k<nnodes; k++) Count_of_particles_to_recv[k] = Count_of_particles_to_recv[k] * 3;
    for(int k=0; k<=nnodes; k++) Prefix_Sum_Count_of_particles_to_send[k] = Prefix_Sum_Count_of_particles_to_send[k] * 3;
    for(int k=0; k<=nnodes; k++) Prefix_Sum_Count_of_particles_to_recv[k] = Prefix_Sum_Count_of_particles_to_recv[k] * 3;
    MPI_Alltoallv(Data_To_Send, Count_of_particles_to_send, Prefix_Sum_Count_of_particles_to_send, MPI_FLOAT, 
                  Data_To_Recv, Count_of_particles_to_recv, Prefix_Sum_Count_of_particles_to_recv, MPI_FLOAT, MPI_COMM_WORLD);

#endif


             mpi_printf("Fully Finished :: (%d)\n", node_id);

             global_Positions_R = global_Positions_D;
             global_Positions_D = Data_To_Recv;

             global_number_of_galaxies_on_node_D = 0;
             int actual_subdivisions = dimx * dimy * dimz;

             int kmin = actual_subdivisions;
             int kmax = -1;
             for(int k=0; k<actual_subdivisions; k++) 
             {
                if (global_Owner_D[k] == -1) ERROR_PRINT();
                if (global_Owner_D[k] == node_id) 
                {
                    if (kmin > k) kmin = k;
                    if (kmax < k) kmax = k;
                }
             }

             global_starting_cell_index_D = kmin;
             global_ending_cell_index_D = kmax + 1;

             global_number_of_galaxies_on_node_D = total_particles_to_recv;

             mpi_printf("++ %d ++ global_starting_cell_index_D = %d :::  global_ending_cell_index_D = %d ::: global_number_of_galaxies_on_node_D = %lld\n", 
             node_id, global_starting_cell_index_D, global_ending_cell_index_D, global_number_of_galaxies_on_node_D);


             for(int k=global_starting_cell_index_D; k<global_ending_cell_index_D; k++) if (global_Owner_D[k] != node_id) ERROR_PRINT();
            for(int k=0; k<number_of_subdivisions; k++) global_Required_D[k] = Local_Required_Bkp[k];
        }

        free(Local_Pos);
    }

unsigned long long int e5time = __rdtsc();
if (node_id == 0) {/*printf("(%d) (%s) ::: Time Taken = %lld (%.2lf seconds)\n", __LINE__, __FILE__, (e5time - stime), (e5time - stime)/CORE_FREQUENCY);*/}

    PRINT_BLUE
    if (node_id == 0) mpi_printf("dimx = %d ::: dimy = %d ::: dimz = %d\n", dimx, dimy, dimz);
    PRINT_BLACK

    Populate_Grid(&global_grid_D, dimx, dimy, dimz);



#if 0
#endif

    if (node_id == 0)
    {
    PRINT_LIGHT_RED
        mpi_printf("global_Min = [%e %e %e] ::: global_Max = [%e %e %e]\n", global_grid_D.Min[0], global_grid_D.Min[1], global_grid_D.Min[2], global_grid_D.Max[0], global_grid_D.Max[1], global_grid_D.Max[2]);
        mpi_printf("global_Extent = [%e %e %e]\n", global_grid_D.Extent[0], global_grid_D.Extent[1], global_grid_D.Extent[2]);
        mpi_printf("Total Memory Allocated = %lld Bytes (%.2lf GB)\n", global_memory_malloced, global_memory_malloced/1000.0/1000.0/1000.0);
    PRINT_BLACK
    }

    unsigned long long int etime = __rdtsc();
    global_time_mpi += etime - stime;
}



void Distribute_R_Particles_Amongst_Nodes(void)
{
    unsigned long long int stime = __rdtsc();
    Compute_Number_of_Ones();

    int dimx, dimy, dimz;
    int dimxy;
    int number_of_subdivisions;

    {
        dimx = global_grid_D.dimx;
        dimy = global_grid_D.dimy;
        dimz = global_grid_D.dimz;

        if (dimx % 2) ERROR_PRINT();
        if (dimy % 2) ERROR_PRINT();
        if (dimz % 2) ERROR_PRINT();

        number_of_subdivisions = dimx * dimy * dimz;

        dimxy = dimx * dimy;

        if (node_id == 0) {/*printf("dimx = %d ::: dimy = %d :: dimz = %d\n", dimx, dimy, dimz);*/}

        number_of_subdivisions = ((number_of_subdivisions + nnodes - 1)/nnodes) * nnodes;
    }


    int subdivisions_per_node = (number_of_subdivisions + nnodes - 1)/nnodes;
    if ( (subdivisions_per_node * nnodes) != number_of_subdivisions) ERROR_PRINT();

    mpi_printf("number_of_subdivisions = %d ::: subdivisions_per_node = %d\n", number_of_subdivisions, subdivisions_per_node);

    if (number_of_subdivisions % nnodes) ERROR_PRINT();


    int *Send_Count = global_Prealloced_Send_Count;
    int *Recv_Count = global_Prealloced_Recv_Count;
    int *Count_Per_Cell = global_Prealloced_Count_Per_Cell;


    global_grid_R.Min[0] = global_grid_D.Min[0];
    global_grid_R.Min[1] = global_grid_D.Min[1];
    global_grid_R.Min[2] = global_grid_D.Min[2];

    global_grid_R.Max[0] = global_grid_D.Max[0];
    global_grid_R.Max[1] = global_grid_D.Max[1];
    global_grid_R.Max[2] = global_grid_D.Max[2];


unsigned long long int e7time = __rdtsc();
if (node_id == 0) {/*printf("(%d) (%s) ::: Time Taken = %lld (%.2lf seconds)\n", __LINE__, __FILE__, (e7time - stime), (e7time - stime)/CORE_FREQUENCY);*/}

    {
        for(int k=0; k<=number_of_subdivisions; k++) Send_Count[k] = 0;
        for(int k=0; k<=number_of_subdivisions; k++) Recv_Count[k] = 0;

        TYPE *Local_Pos = (TYPE *)malloc(global_number_of_galaxies_on_node_R * sizeof(TYPE) * DIMENSIONS);

        int dimxx = dimx - 1;
        int dimyy = dimy - 1;
        int dimzz = dimz - 1;

        TYPE *Min = global_grid_R.Min;
        TYPE *Max = global_grid_R.Max;
        TYPE *Extent  = global_grid_R.Extent;

        for(int i=0; i<global_number_of_galaxies_on_node_R; i++)
        {
            TYPE float_x = global_Positions_R[3*i + 0];
            TYPE float_y = global_Positions_R[3*i + 1];
            TYPE float_z = global_Positions_R[3*i + 2];
                
            int cell_x = ((float_x - Min[0]) * dimx)/Extent[0];
            int cell_y = ((float_y - Min[1]) * dimy)/Extent[1];
            int cell_z = ((float_z - Min[2]) * dimz)/Extent[2];

            CLAMP_BELOW(cell_x, 0); CLAMP_ABOVE(cell_x, (dimxx));
            CLAMP_BELOW(cell_y, 0); CLAMP_ABOVE(cell_y, (dimyy));
            CLAMP_BELOW(cell_z, 0); CLAMP_ABOVE(cell_z, (dimzz));
            
            int cell_id = GET_CELL_INDEX(cell_x, cell_y, cell_z);

            Send_Count[cell_id]++;
        }

unsigned long long int e8time = __rdtsc();
if (node_id == 0) {/*printf("(%d) (%s) ::: Time Taken = %lld (%.2lf seconds)\n", __LINE__, __FILE__, (e8time - stime), (e8time - stime)/CORE_FREQUENCY);*/}

        {
            int sum_so_far = 0;
            for(int k=0; k<number_of_subdivisions; k++)
            {
                int curr_value = Send_Count[k];
                Send_Count[k] = sum_so_far;
                sum_so_far += curr_value;
            }
            Send_Count[number_of_subdivisions] = sum_so_far;

            if (sum_so_far != global_number_of_galaxies_on_node_R) ERROR_PRINT();
        }


        for(int i=0; i<global_number_of_galaxies_on_node_R; i++)
        {
            TYPE float_x = global_Positions_R[3*i + 0];
            TYPE float_y = global_Positions_R[3*i + 1];
            TYPE float_z = global_Positions_R[3*i + 2];
                
            int cell_x = ((float_x - Min[0]) * dimx)/Extent[0];
            int cell_y = ((float_y - Min[1]) * dimy)/Extent[1];
            int cell_z = ((float_z - Min[2]) * dimz)/Extent[2];

            CLAMP_BELOW(cell_x, 0); CLAMP_ABOVE(cell_x, (dimxx));
            CLAMP_BELOW(cell_y, 0); CLAMP_ABOVE(cell_y, (dimyy));
            CLAMP_BELOW(cell_z, 0); CLAMP_ABOVE(cell_z, (dimzz));
            
            int cell_id = GET_CELL_INDEX(cell_x, cell_y, cell_z);

            Local_Pos[3* Send_Count[cell_id] + 0] = float_x;
            Local_Pos[3* Send_Count[cell_id] + 1] = float_y;
            Local_Pos[3* Send_Count[cell_id] + 2] = float_z;
            Send_Count[cell_id]++;
        }

        if (Send_Count[number_of_subdivisions-1] != global_number_of_galaxies_on_node_R) ERROR_PRINT();

        for(int k=(number_of_subdivisions-1); k>=1; k--) Send_Count[k] = Send_Count[k] - Send_Count[k-1];

unsigned long long int e9time = __rdtsc();
if (node_id == 0) {/*printf("(%d) (%s) ::: Time Taken = %lld (%.2lf seconds)\n", __LINE__, __FILE__, (e9time - stime), (e9time - stime)/CORE_FREQUENCY); */}

#if 0
#endif

    #if 0
    #else
        MPI_Alltoall(Send_Count, subdivisions_per_node, MPI_INT, Recv_Count, subdivisions_per_node, MPI_INT, MPI_COMM_WORLD);
    #endif

unsigned long long int e10time = __rdtsc();
if (node_id == 0) {/*printf("(%d) (%s) ::: Time Taken = %lld (%.2lf seconds)\n", __LINE__, __FILE__, (e10time - stime), (e10time - stime)/CORE_FREQUENCY);*/}

#if 0
#endif


        {
            for(int k=0; k<number_of_subdivisions; k++) Count_Per_Cell[k] = 0;
            int *Dst = Count_Per_Cell + node_id * subdivisions_per_node;
            for(int k=0; k<nnodes; k++) 
            {
                int *Src = Recv_Count + k*subdivisions_per_node;
                for(int l=0; l<subdivisions_per_node; l++)
                {
                    Dst[l] += Src[l];
                }
            }

#if 0
#else
            int *Local_Count_Per_Cell = (int *)malloc(subdivisions_per_node * sizeof(int));
            int offfset = node_id * subdivisions_per_node;
            for(int k=0; k<subdivisions_per_node; k++) Local_Count_Per_Cell[k] = Count_Per_Cell[offfset + k];
            MPI_Allgather(Local_Count_Per_Cell, subdivisions_per_node, MPI_INT, Count_Per_Cell, subdivisions_per_node, MPI_INT, MPI_COMM_WORLD);

#endif
		
	#if 0
    #endif
        }

unsigned long long int e1time = __rdtsc();
if (node_id == 0) {/*printf("(%d) (%s) ::: Time Taken = %lld (%.2lf seconds)\n", __LINE__, __FILE__, (e1time - stime), (e1time - stime)/CORE_FREQUENCY); */}

        long long int *Weights = (long long int *)malloc(number_of_subdivisions * sizeof(long long int));
        long long int *local_Weights = (long long int *)malloc(subdivisions_per_node * sizeof(long long int));

        Compute_Weights_R(Count_Per_Cell, Weights, subdivisions_per_node, dimx, dimy, dimz, dimxy);

        {
            int starting_cell_index = (node_id + 0) * subdivisions_per_node;
            int   ending_cell_index = (node_id + 1) * subdivisions_per_node;
            int cells = subdivisions_per_node;
            for(int k=0; k<cells; k++) local_Weights[k] = Weights[starting_cell_index + k];
        }

#if 0
#else
        MPI_Allgather(local_Weights, subdivisions_per_node, MPI_LONG_LONG_INT, Weights, subdivisions_per_node, MPI_LONG_LONG_INT, MPI_COMM_WORLD);
#endif

unsigned long long int e2time = __rdtsc();
if (node_id == 0) {/*printf("(%d) (%s) ::: Time Taken = %lld (%.2lf seconds)\n", __LINE__, __FILE__, (e2time - stime), (e2time - stime)/CORE_FREQUENCY); */}


        global_Owner_R = (int *)malloc(number_of_subdivisions * sizeof(int));
        for(int k=0; k<number_of_subdivisions; k++) global_Owner_R[k] = -1; 

        {
            long long int sum_so_far = 0;
            for(int k=0; k<number_of_subdivisions; k++) sum_so_far  += Weights[k];
            long long int total_weights = sum_so_far;
            long long int weights_per_node = (sum_so_far + nnodes - 1)/nnodes;
            mpi_printf("total_weights = %lld ::: weights_per_node = %lld\n", total_weights, weights_per_node);
            fflush(stdout);
            int alloted_so_far = 0;

            sum_so_far = 0;
            for(int k=0; k<nnodes; k++)
            {
                long long int starting_index = (k + 0) * weights_per_node;
                long long int   ending_index = (k + 1) * weights_per_node;

                if (starting_index > total_weights ) starting_index = total_weights;
                if (  ending_index > total_weights)   ending_index = total_weights;

                while (sum_so_far < ending_index)
                {
                    sum_so_far += Weights[alloted_so_far];
                    global_Owner_R[alloted_so_far] = k;
                    alloted_so_far++;
                }
            }

            {

                for(int k = number_of_subdivisions-1; k >= 0; k--)
                {
                    if (global_Owner_R[k] == -1)
                    {
                        if (Weights[k] == 0) global_Owner_R[k] = nnodes-1;
                        else ERROR_PRINT();
                    }
                }

            }
        
            int *Temp1 = (int *)malloc(nnodes * sizeof(int)); for(int k=0; k<nnodes; k++) Temp1[k] = 0;
            int actual_subdivisions = dimx * dimy * dimz;
            for(int k=0; k<actual_subdivisions; k++) 
            {
                if (global_Owner_R[k] == -1) { printf("k = %d\n", k); ERROR_PRINT(); }
                Temp1[global_Owner_R[k]] += Count_Per_Cell[k];
            }

            {
                size_t final_sum = 0;
                int min = Temp1[0];
                int max = Temp1[0];
                for(int k=0; k<nnodes; k++) 
                {
                    final_sum += Temp1[k];
                    min = PCL_MIN(Temp1[k], min);
                    max = PCL_MAX(Temp1[k], max);
                }
                size_t avg = final_sum/nnodes;
                mpi_printf("node_id = %d ::: Avg. = %d ::: min = %d ::: max = %d ::: Ratio = %.2lf\n", node_id, (int)(avg), min, max, (max*1.0)/avg);

                if (final_sum != global_number_of_galaxies) ERROR_PRINT();
            }
        }

unsigned long long int e4time = __rdtsc();
if (node_id == 0) {/*printf("(%d) (%s) ::: Time Taken = %lld (%.2lf seconds)\n", __LINE__, __FILE__, (e4time - stime), (e4time - stime)/CORE_FREQUENCY);*/}


        global_Required_R = (unsigned char *)malloc(number_of_subdivisions *sizeof(unsigned char));
        for(int k = 0; k < number_of_subdivisions; k++) global_Required_R[k] = 0;
        Compute_Required_R(number_of_subdivisions, dimx, dimy, dimz, dimxy);
#if 0
#endif

#if 1
        for(int k = 0; k < number_of_subdivisions; k++)
        {
            if ((global_Required_D_For_R[k] == 1) && (global_Required_R[k] == 0))
            {
                global_Required_R[k] = 1;
            }
        }
#endif
#if 0
#endif


unsigned long long int e3time = __rdtsc();
if (node_id == 0) {/*printf("(%d) (%s) ::: Time Taken = %lld (%.2lf seconds)\n", __LINE__, __FILE__, (e3time - stime), (e3time - stime)/CORE_FREQUENCY);*/}

        {
            int ssum = 0; 
            int particle_sum = 0;
            
            for(int k=0; k<number_of_subdivisions; k++) 
            {
                ssum += global_Required_R[k];
                if (global_Required_R[k]) particle_sum += Count_Per_Cell[k];
            }

            mpi_printf("node_id = %d ::: ssum = %d (number_of_subdivisions = %d) ::: particle_sum = %d (global_number_of_galxies = %lld)\n", 
                node_id, ssum, number_of_subdivisions, particle_sum, global_number_of_galaxies);
        }


        unsigned char *Local_Required_Bkp = (unsigned char *)malloc(number_of_subdivisions * sizeof(unsigned char));
        for(int k=0; k<number_of_subdivisions; k++) Local_Required_Bkp[k] = global_Required_R[k];

        Convert_From_Byte_To_Bit(global_Required_R, number_of_subdivisions);
        int number_of_subdivisions_by_eight = number_of_subdivisions/8;
        if ((number_of_subdivisions_by_eight * 8) != number_of_subdivisions) number_of_subdivisions_by_eight++;
        unsigned char *Local_Required_All_Nodes = (unsigned char *)malloc(nnodes * number_of_subdivisions_by_eight * sizeof(unsigned char));
        {

    #if 0
    #else
            MPI_Allgather(global_Required_R, number_of_subdivisions_by_eight, MPI_UNSIGNED_CHAR, 
                          Local_Required_All_Nodes, number_of_subdivisions_by_eight, MPI_UNSIGNED_CHAR, MPI_COMM_WORLD);
        
    #endif
        }
unsigned long long int e12time = __rdtsc();
if (node_id == 0) {/*printf("(%d) (%s) ::: Time Taken = %lld (%.2lf seconds)\n", __LINE__, __FILE__, (e12time - stime), (e12time - stime)/CORE_FREQUENCY);*/}

        for(int p=0; p<nnodes; p++) 
        {
        }

        {
            int sum_so_far = 0;
            for(int k=0; k<number_of_subdivisions; k++)
            {
                int curr_value = Send_Count[k];
                Send_Count[k] = sum_so_far;
                sum_so_far += curr_value;
            }
            Send_Count[number_of_subdivisions] = sum_so_far;

            if (sum_so_far != global_number_of_galaxies_on_node_R) ERROR_PRINT();
        }


        {
            int *Count_of_particles_to_send = (int *)malloc(nnodes * sizeof(int));
            int *Count_of_particles_to_recv = (int *)malloc(nnodes * sizeof(int));

            Compute_Count_of_particles_to_send(Local_Required_All_Nodes, Count_of_particles_to_send, Send_Count, number_of_subdivisions);
        #if 0      
        #endif
        #if 0
        #else
            MPI_Alltoall(Count_of_particles_to_send, 1, MPI_INT, Count_of_particles_to_recv, 1, MPI_INT, MPI_COMM_WORLD);
        #endif
unsigned long long int e11time = __rdtsc();
if (node_id == 0) {/*printf("(%d) (%s) ::: Time Taken = %lld (%.2lf seconds)\n", __LINE__, __FILE__, (e11time - stime), (e11time - stime)/CORE_FREQUENCY);*/}

             int total_particles_to_send;
             int total_particles_to_recv;
             {
                int sum0 = 0, sum1 = 0;
                for(int k=0; k<nnodes; k++) sum0 += Count_of_particles_to_send[k];
                for(int k=0; k<nnodes; k++) sum1 += Count_of_particles_to_recv[k];
                mpi_printf("node_id = %d ::: sum0 = %d ::: sum1 = %d\n", node_id, sum0, sum1);
                total_particles_to_send = sum0;
                total_particles_to_recv = sum1;
             }

             int *Prefix_Sum_Count_of_particles_to_send = (int *)malloc((nnodes + 1) * sizeof(int));
             int *Prefix_Sum_Count_of_particles_to_recv = (int *)malloc((nnodes + 1) * sizeof(int));
             {
                int sum = 0;
                for(int k=0; k<nnodes; k++)
                {
                    int value = Count_of_particles_to_send[k];
                    Prefix_Sum_Count_of_particles_to_send[k] = sum;
                    sum += value;
                }
                Prefix_Sum_Count_of_particles_to_send[nnodes] = sum;
                if (sum != total_particles_to_send) ERROR_PRINT();

                sum = 0;
                for(int k=0; k<nnodes; k++)
                {
                    int value = Count_of_particles_to_recv[k];
                    Prefix_Sum_Count_of_particles_to_recv[k] = sum;
                    sum += value;
                }
                Prefix_Sum_Count_of_particles_to_recv[nnodes] = sum;
                if (sum != total_particles_to_recv) ERROR_PRINT();
            }


            //Step L:
            TYPE *Data_To_Send = (TYPE *)malloc(total_particles_to_send * 3 * sizeof(TYPE));
            TYPE *Data_To_Recv = (TYPE *)malloc(total_particles_to_recv * 3 * sizeof(TYPE));

#if 0 // <<----
#else

		Prepare_Data_To_Send(Local_Pos, Data_To_Send, Prefix_Sum_Count_of_particles_to_send, Local_Required_All_Nodes, Send_Count, number_of_subdivisions);
unsigned long long int e6time = __rdtsc();
if (node_id == 0) {/*printf("(%d) (%s) ::: Time Taken = %lld (%.2lf seconds)\n", __LINE__, __FILE__, (e6time - stime), (e6time - stime)/CORE_FREQUENCY);*/}

    for(int k=0; k<nnodes; k++) Count_of_particles_to_send[k] = Count_of_particles_to_send[k] * 3;
    for(int k=0; k<nnodes; k++) Count_of_particles_to_recv[k] = Count_of_particles_to_recv[k] * 3;
    for(int k=0; k<=nnodes; k++) Prefix_Sum_Count_of_particles_to_send[k] = Prefix_Sum_Count_of_particles_to_send[k] * 3;
    for(int k=0; k<=nnodes; k++) Prefix_Sum_Count_of_particles_to_recv[k] = Prefix_Sum_Count_of_particles_to_recv[k] * 3;
    MPI_Alltoallv(Data_To_Send, Count_of_particles_to_send, Prefix_Sum_Count_of_particles_to_send, MPI_FLOAT, 
                  Data_To_Recv, Count_of_particles_to_recv, Prefix_Sum_Count_of_particles_to_recv, MPI_FLOAT, MPI_COMM_WORLD);

#endif


             mpi_printf("Fully Finished :: (%d)\n", node_id);

             global_Positions_R = Data_To_Recv;

             global_number_of_galaxies_on_node_R = 0;
             int actual_subdivisions = dimx * dimy * dimz;

             int kmin = actual_subdivisions;
             int kmax = -1;
             for(int k=0; k<actual_subdivisions; k++) 
             {
                if (global_Owner_R[k] == -1) ERROR_PRINT(); 
                if (global_Owner_R[k] == node_id) 
                {
                    if (kmin > k) kmin = k;
                    if (kmax < k) kmax = k;
                }
             }

             global_starting_cell_index_R = kmin;
             global_ending_cell_index_R = kmax + 1;

             global_number_of_galaxies_on_node_R = total_particles_to_recv;

             mpi_printf("++ %d ++ global_starting_cell_index_R = %d :::  global_ending_cell_index_R = %d ::: global_number_of_galaxies_on_node_R = %lld\n", 
             node_id, global_starting_cell_index_R, global_ending_cell_index_R, global_number_of_galaxies_on_node_R);


             for(int k=global_starting_cell_index_R; k<global_ending_cell_index_R; k++) if (global_Owner_R[k] != node_id) ERROR_PRINT();
            for(int k=0; k<number_of_subdivisions; k++) global_Required_R[k] = Local_Required_Bkp[k];
        }

        free(Local_Pos);
    }

unsigned long long int e5time = __rdtsc();
if (node_id == 0) {/*printf("(%d) (%s) ::: Time Taken = %lld (%.2lf seconds)\n", __LINE__, __FILE__, (e5time - stime), (e5time - stime)/CORE_FREQUENCY);*/}

    PRINT_BLUE
    if (node_id == 0) mpi_printf("dimx = %d ::: dimy = %d ::: dimz = %d\n", dimx, dimy, dimz);
    PRINT_BLACK

    Populate_Grid(&global_grid_R, dimx, dimy, dimz);

#if 0
#endif

    if (node_id == 0)
    {
    PRINT_LIGHT_RED
        mpi_printf("global_Min = [%e %e %e] ::: global_Max = [%e %e %e]\n", global_grid_R.Min[0], global_grid_R.Min[1], global_grid_R.Min[2], global_grid_R.Max[0], global_grid_R.Max[1], global_grid_R.Max[2]);
        mpi_printf("global_Extent = [%e %e %e]\n", global_grid_R.Extent[0], global_grid_R.Extent[1], global_grid_R.Extent[2]);
        mpi_printf("Total Memory Allocated = %lld Bytes (%.2lf GB)\n", global_memory_malloced, global_memory_malloced/1000.0/1000.0/1000.0);
    PRINT_BLACK
    }

    unsigned long long int etime = __rdtsc();
    global_time_mpi += etime - stime;
}
///////////////////////////////////////////////////////////////////////////////////////////////////

void *Compute_KD_Tree_Parallel_For_D(void *arg1)
{
    int threadid = (int)(size_t)(arg1);
    Compute_KD_Tree(threadid, global_Positions_D, global_number_of_galaxies_on_node_D, &global_grid_D, global_Required_D);
    return arg1;
}

void *Compute_KD_Tree_Parallel_For_R(void *arg1)
{
    int threadid = (int)(size_t)(arg1);
    Compute_KD_Tree(threadid, global_Positions_R, global_number_of_galaxies_on_node_R, &global_grid_R, global_Required_R);
    return arg1;
}


void Compute_KD_Tree_Acceleration_Data_Structure_For_D(void)
{
    //The code below compute the kd-tree for each of the cells that
    //are owned by the node. Like all time-consuming functions, this
    //is also parallelized to exploit the 'nthreads' threads...
    int nthreads_prime = 1 * nthreads; 
    unsigned long long int start_time = __rdtsc();
    //====
    for(int i=1; i<nthreads_prime; i++) pthread_create(&threads[i], NULL, Compute_KD_Tree_Parallel_For_D, (void *)(i));
    Compute_KD_Tree_Parallel_For_D(0);
    for(int i=1; i<nthreads_prime; i++) pthread_join(threads[i], NULL);
    //====
    unsigned long long int end_time = __rdtsc();
    global_time_kdtree_d += (end_time - start_time);
}

void Compute_KD_Tree_Acceleration_Data_Structure_For_R(void)
{
    int nthreads_prime = 1 * nthreads; 
    unsigned long long int start_time = __rdtsc();
    //====
    for(int i=1; i<nthreads_prime; i++) pthread_create(&threads[i], NULL, Compute_KD_Tree_Parallel_For_R, (void *)(i));
    Compute_KD_Tree_Parallel_For_R(0);
    for(int i=1; i<nthreads_prime; i++) pthread_join(threads[i], NULL);
    //====
    unsigned long long int end_time = __rdtsc();
    global_time_kdtree_r += (end_time - start_time);
}

void Report_Performance(void)
{
    if (node_id != 0) return;

    {
        PRINT_BLUE
            printf("=================== RR ================\n");
        PRINT_BLACK
        //++++++RR++++++++
        long long int total_sum_so_far = 0;
        long long int total_sum = 0;
        for(int bin_id=1; bin_id < (HIST_BINS-1); bin_id++) total_sum += global_Histogram_RR[bin_id];

        for(int bin_id=1; bin_id < (HIST_BINS-1); bin_id++)
        {
            total_sum_so_far += global_Histogram_RR[bin_id];
            printf("%2d :: %16lld (%19lld ::: %.4e %%)\n", bin_id, global_Histogram_RR[bin_id], total_sum_so_far, (total_sum_so_far*100.0)/total_sum);
        }

        printf("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n");
        for(int bin_id=1; bin_id < (HIST_BINS-1); bin_id++)
        {
        }

        global_stat_useful_interactions_rr = total_sum;
    }

    {
        PRINT_BLUE
            printf("=================== DR ================\n");
        PRINT_BLACK
        //++++++DR++++++++
        long long int total_sum_so_far = 0;
        long long int total_sum = 0;
        for(int bin_id=1; bin_id < (HIST_BINS-1); bin_id++) total_sum += global_Histogram_DR[bin_id];

        for(int bin_id=1; bin_id < (HIST_BINS-1); bin_id++)
        {
            total_sum_so_far += global_Histogram_DR[bin_id];
            printf("%2d :: %16lld (%19lld ::: %.4e %%)\n", bin_id, global_Histogram_DR[bin_id], total_sum_so_far, (total_sum_so_far*100.0)/total_sum);
        }

        printf("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n");
        for(int bin_id=1; bin_id < (HIST_BINS-1); bin_id++)
        {
        }

        global_stat_useful_interactions_dr = total_sum;
    }

    long long int ngal = global_number_of_galaxies;
    long long int actual_sum = global_stat_total_interactions_rr + global_stat_total_interactions_dr;
    global_time_kdtree = global_time_kdtree_d + global_time_kdtree_r;
    global_time_total = global_time_kdtree + global_time_rr + global_time_dr + global_time_mpi;


PRINT_BLUE
    printf("==================================================================================\n");
    printf("CORE_FREQUENCY\t\t\t = %.2lf GHz ::: nthreads = %d ::: nnodes = %d\n", (CORE_FREQUENCY/1000.0/1000.0/1000.0), nthreads, nnodes);
    printf("<<%d>>MPI_Time\t\t\t = %12lld cycles (%5.2lf seconds) (%5.2lf%%)\n", node_id, global_time_mpi, (global_time_mpi*1.0)/CORE_FREQUENCY, (global_time_mpi*100.0)/global_time_total);
PRINT_GRAY
    printf("<<%d>>KD-Tree (D) Construction Time\t = %12lld cycles (%5.2lf seconds) (%5.2lf%%)\n", node_id, global_time_kdtree_d, global_time_kdtree_d/CORE_FREQUENCY, (global_time_kdtree_d*100.0)/global_time_total);
    printf("<<%d>>KD-Tree (R) Construction Time\t = %12lld cycles (%5.2lf seconds) (%5.2lf%%)\n", node_id, global_time_kdtree_r, global_time_kdtree_r/CORE_FREQUENCY, (global_time_kdtree_r*100.0)/global_time_total);
PRINT_BLUE
    printf("<<%d>>KD-Tree Construction Time\t = %12lld cycles (%5.2lf seconds) (%5.2lf%%)\n", node_id, global_time_kdtree, global_time_kdtree/CORE_FREQUENCY, (global_time_kdtree*100.0)/global_time_total);
    printf("<<%d>>RR Time\t\t\t = %12lld cycles (%5.2lf seconds) (%5.2lf%%)\n", node_id, global_time_rr, global_time_rr/CORE_FREQUENCY, (global_time_rr*100.0)/global_time_total);
    printf("<<%d>>DR Time\t\t\t = %12lld cycles (%5.2lf seconds) (%5.2lf%%)\n", node_id, global_time_dr, global_time_dr/CORE_FREQUENCY, (global_time_dr*100.0)/global_time_total);
    printf("<<%d>>Total Time\t\t\t = %12lld cycles (%5.2lf seconds) (%5.2lf%%)\n", node_id, global_time_total, global_time_total/CORE_FREQUENCY, (global_time_total*100.0)/global_time_total);
    printf("==================================================================================\n");
PRINT_GREEN
{
	MT_Z[0] = MT_2[0] + MT_3[0] + MT_4[0] + MT_5[0];
	printf("----------------------------------------------\n");
	printf("----------------------------------------------\n");
}

PRINT_RED
}

void Report_Performance2(void)
{
    global_time_kdtree = global_time_kdtree_d + global_time_kdtree_r;
    printf("CORE_FREQUENCY = %.2lf GHz\n", CORE_FREQUENCY/1000.0/1000.0/1000.0);
    printf("KD-Tree Construction Time = %lld cycles (%.2lf seconds)\n", global_time_kdtree, global_time_kdtree/CORE_FREQUENCY);
    printf("RR Time = %lld cycles (%.2lf seconds)\n", global_time_rr, global_time_rr/CORE_FREQUENCY);
    printf("DR Time = %lld cycles (%.2lf seconds)\n", global_time_dr, global_time_dr/CORE_FREQUENCY);
    printf("Total Time = %lld cycles (%.2lf seconds)\n", global_time_total, global_time_total/CORE_FREQUENCY);
}

void Checking(void)
{
    if (node_id == (nnodes-1)) printf("<<node_id = %d>>> ::: SIMD_WIDTH = %d\n", node_id, SIMD_WIDTH);
}

void Initialize_MPI(int argc, char **argv)
{
    //This function does the PRE-mpi work... 
    int provided;

    MPI_Status stat;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided); /*START MPI */
    MPI_Comm_rank(MPI_COMM_WORLD, &node_id); /*DETERMINE RANK OF THIS PROCESSOR*/
    MPI_Comm_size(MPI_COMM_WORLD, &nnodes); /*DETERMINE TOTAL NUMBER OF PROCESSORS*/
    MPI_Errhandler_set(MPI_COMM_WORLD,MPI_ERRORS_RETURN);

    if (node_id == (nnodes-1)) { printf("Done Initialize_MPI()\n"); fflush(stdout);}
}

void Finalize_MPI(void)
{
    MPI_Finalize();

    if (node_id == (nnodes-1)) { printf("Done Finalize_MPI()\n"); fflush(stdout);}
}

#define IMHERE() 
#define IMHERE2() 

void Perform_Mandatory_Initializations(Grid *grid, size_t number_of_particles, TYPE i_Lbox, TYPE i_rminL, TYPE i_rmaxL, int nrbin)
{
    static int function_called_how_many_times = 0;
    function_called_how_many_times++;


    if (function_called_how_many_times > 1) 
    {
    }

    TYPE *Extent = grid->Extent;

    if (function_called_how_many_times  == 1)
    {
        global_actual_sum_rr = (long long int *)malloc(16 * nthreads *sizeof(long long int));
        for(int threadid = 0; threadid < nthreads; threadid++) global_actual_sum_rr[16*threadid] = 0;
        global_actual_sum_dr = (long long int *)malloc(16 * nthreads *sizeof(long long int));
        for(int threadid = 0; threadid < nthreads; threadid++) global_actual_sum_dr[16*threadid] = 0;
    }

    IMHERE();

    int dimx = grid->dimx;
    int dimy = grid->dimy;
    int dimz = grid->dimz;

    int number_of_uniform_subdivisions = grid->number_of_uniform_subdivisions;

    int maximum_number_of_particles = 0;
    int maximum_number_of_kd_subdivisions = 0;

    if (node_id == 0) {/*printf(" <<%d>> : number_of_uniform_subdivisions = %d\n", node_id, number_of_uniform_subdivisions);*/}
    for(int cell_id = 0; cell_id < number_of_uniform_subdivisions; cell_id++)
    {
        if (global_Required_R[cell_id])
        {
            int number_of_kd_subdivisions = grid->Number_of_kd_subdivisions[cell_id];
            int number_of_particles = grid->Range[cell_id][number_of_kd_subdivisions];
            maximum_number_of_kd_subdivisions = PCL_MAX(maximum_number_of_kd_subdivisions, number_of_kd_subdivisions);
            maximum_number_of_particles = PCL_MAX(maximum_number_of_particles, number_of_particles);
        }
    }

    IMHERE();
    int dimxy = dimx * dimy;

    if ((nrbin+2) != HIST_BINS)
    {
        ERROR_PRINT_STRING("Please change HIST_BINS or global_nrbin");
    }

    IMHERE();

    TYPE rmax = i_rmaxL / i_Lbox;
    TYPE rmin = i_rminL / i_Lbox;

    {
        TYPE max_dimension = grid->Max[0];
        max_dimension = PCL_MAX(grid->Max[1], max_dimension);
        max_dimension = PCL_MAX(grid->Max[2], max_dimension);

        rmax = rmax * max_dimension;
        rmin = rmin * max_dimension;
    }

    TYPE rmax_2 = rmax*rmax;
    TYPE rmin_2 = rmin*rmin;

    global_rmax_2 = rmax_2;

    global_Gather_Histogram0 = (unsigned int **)malloc(nthreads * sizeof(unsigned int *));
    global_Gather_Histogram1 = (unsigned int **)malloc(nthreads * sizeof(unsigned int *));
    global_RR_int0           = (unsigned int **)malloc(nthreads * sizeof(unsigned int *));
    global_RR_int1           = (unsigned int **)malloc(nthreads * sizeof(unsigned int *));
    global_DR_int0           = (unsigned int **)malloc(nthreads * sizeof(unsigned int *));
    global_DR_int1           = (unsigned int **)malloc(nthreads * sizeof(unsigned int *));
    global_Pos1              = (TYPE **)malloc(nthreads * sizeof(TYPE *));
    global_Bdry1_X           = (TYPE **)malloc(nthreads * sizeof(TYPE *));
    global_Bdry1_Y           = (TYPE **)malloc(nthreads * sizeof(TYPE *));
    global_Bdry1_Z           = (TYPE **)malloc(nthreads * sizeof(TYPE *));

    for(int threadid = 0; threadid < nthreads; threadid++)
    {
        int hist_bins_prime = (((HIST_BINS + 32) >> 5)<<5);
        size_t sz = 0;

        sz  += (SIMD_WIDTH * HIST_BINS) * sizeof(unsigned int);
        sz  += (SIMD_WIDTH * HIST_BINS) * sizeof(unsigned int);
        sz += (hist_bins_prime) * sizeof(int);
        sz += (hist_bins_prime) * sizeof(int);
        sz += (hist_bins_prime) * sizeof(int);
        sz += (hist_bins_prime) * sizeof(int);
        sz += maximum_number_of_particles * DIMENSIONS * sizeof(TYPE);
        sz += maximum_number_of_kd_subdivisions * 2 * sizeof(TYPE);
        sz += maximum_number_of_kd_subdivisions * 2 * sizeof(TYPE);
        sz += maximum_number_of_kd_subdivisions * 2 * sizeof(TYPE);


        unsigned char *temp_memory = (unsigned char *)my_malloc(sz);
        unsigned char *temp_memory2 = temp_memory;

        global_Gather_Histogram0[threadid] = (unsigned int *)(temp_memory2); temp_memory2 += (SIMD_WIDTH * HIST_BINS) * sizeof(unsigned int);
        global_Gather_Histogram1[threadid] = (unsigned int *)(temp_memory2); temp_memory2 += (SIMD_WIDTH * HIST_BINS) * sizeof(unsigned int);
        for(int i=0; i<(SIMD_WIDTH*HIST_BINS); i++) global_Gather_Histogram0[threadid][i] = 0;
        for(int i=0; i<(SIMD_WIDTH*HIST_BINS); i++) global_Gather_Histogram1[threadid][i] = 0;

        global_RR_int0[threadid] = (unsigned int *)(temp_memory2); temp_memory2 += (hist_bins_prime) * sizeof(int);
        global_RR_int1[threadid] = (unsigned int *)(temp_memory2); temp_memory2 += (hist_bins_prime) * sizeof(int);
        global_DR_int0[threadid] = (unsigned int *)(temp_memory2); temp_memory2 += (hist_bins_prime) * sizeof(int);
        global_DR_int1[threadid] = (unsigned int *)(temp_memory2); temp_memory2 += (hist_bins_prime) * sizeof(int);
        for(int i=0; i<=(1+nrbin); i++) global_RR_int0[threadid][i]  = 0;
        for(int i=0; i<=(1+nrbin); i++) global_RR_int1[threadid][i]  = 0;
        for(int i=0; i<=(1+nrbin); i++) global_DR_int0[threadid][i]  = 0;
        for(int i=0; i<=(1+nrbin); i++) global_DR_int1[threadid][i]  = 0;

        global_Pos1[threadid]     = (TYPE *)(temp_memory2); temp_memory2 += maximum_number_of_particles * DIMENSIONS * sizeof(TYPE);
        global_Bdry1_X[threadid]  = (TYPE *)(temp_memory2); temp_memory2 += maximum_number_of_kd_subdivisions * 2 * sizeof(TYPE);
        global_Bdry1_Y[threadid]  = (TYPE *)(temp_memory2); temp_memory2 += maximum_number_of_kd_subdivisions * 2 * sizeof(TYPE);
        global_Bdry1_Z[threadid]  = (TYPE *)(temp_memory2); temp_memory2 += maximum_number_of_kd_subdivisions * 2 * sizeof(TYPE);

        if ((temp_memory2 - temp_memory) != sz) ERROR_PRINT();
#if 0
#endif

        IMHERE();
    }

    IMHERE();
    {
        int threadid = 0;
        size_t sz = 0;

        sz += (nrbin * sizeof(TYPE));
        sz += (nrbin * sizeof(TYPE));
        sz += (nrbin * sizeof(TYPE));
        sz += (HIST_BINS * sizeof(TYPE));
        sz += (HIST_BINS * sizeof(TYPE));

    IMHERE();
        unsigned char *temp_memory = (unsigned char *)my_malloc(sz);
    IMHERE();
        unsigned char *temp2_memory = temp_memory;
    IMHERE();

        IMHERE();

        TYPE lrmin = my_log(rmin);
        TYPE lrmax = my_log(rmax);

        TYPE dlnr = (lrmax - lrmin)/nrbin;

        if ((node_id == 0) && (threadid == 0))  {/*printf("rmin = %f ::: rmax = %f ::: lrmin = %f ::: lrmax = %f ::: dlnr = %f\n", rmin, rmax, lrmin, lrmax, dlnr);*/}
        IMHERE();

        global_Rminarr  = (TYPE *)(temp2_memory); temp2_memory += (nrbin * sizeof(TYPE));
        global_Rmaxarr  = (TYPE *)(temp2_memory); temp2_memory += (nrbin * sizeof(TYPE));
        global_Rval     = (TYPE *)(temp2_memory); temp2_memory += (nrbin * sizeof(TYPE));
        global_BinCorners  = (TYPE *)(temp2_memory); temp2_memory += (HIST_BINS * sizeof(TYPE));
        global_BinCorners2 = (TYPE *)(temp2_memory); temp2_memory += (HIST_BINS * sizeof(TYPE));

        if ((temp2_memory - temp_memory) != sz) ERROR_PRINT();

        TYPE *Rminarr = global_Rminarr;
        TYPE *Rmaxarr = global_Rmaxarr;
        TYPE *Rval = global_Rval;
        TYPE *BinCorners = global_BinCorners;
        TYPE *BinCorners2 = global_BinCorners2;
    

        for(int i=1; i<nrbin; i++)     Rminarr[i] = my_exp(lrmin + i*dlnr); Rminarr[0] = rmin;
        for(int i=0; i<(nrbin-1); i++) Rmaxarr[i] = Rminarr[i+1];  Rmaxarr[nrbin-1] = rmax;
        for(int i=0; i<nrbin; i++)        Rval[i] = my_exp(lrmin + (i+0.5)*dlnr); 

        for(int i=0; i<(nrbin); i++) BinCorners[i] = Rminarr[i] * Rminarr[i];
        for(int i=(nrbin); i<(HIST_BINS); i++) BinCorners[i] = FLT_MAX; //Some large number...
        BinCorners2[0] = -FLT_MAX; for(int i=0; i<(1+nrbin); i++) BinCorners2[i+1] = BinCorners[i]; BinCorners2[nrbin+1] = rmax*rmax;

        int dx = int(ceil((i_rmaxL/i_Lbox)*dimx)) + 0;
        int dy = int(ceil((i_rmaxL/i_Lbox)*dimy)) + 0;
        int dz = int(ceil((i_rmaxL/i_Lbox)*dimz)) + 0;

        long long int actual_sum = 0;
        long long int curr_accumulated_actual_sum = 0;

        global_dx = dx;
        global_dy = dy;
        global_dz = dz;
    }
}


/*
 *
 * Break to banta hai -- KitKat time :)
 *
 *
 *
 */



void Perform_DR_TaskQ(void)
{
    unsigned long long int start_time = __rdtsc();
    IMHERE2();

    long dimensionSize[1], tileSize[1];
    int start_processing_cell_index = global_starting_cell_index_D; if (global_starting_cell_index_D < 0) ERROR_PRINT_STRING("global_starting_cell_index_D < 0");
    int   end_processing_cell_index = global_ending_cell_index_D;   if (global_ending_cell_index_D   < 0) ERROR_PRINT_STRING("global_ending_cell_index_D   < 0");
    long ntasks = end_processing_cell_index - start_processing_cell_index;

    IMHERE2();
    dimensionSize[0] = ntasks;  tileSize[0]=1;

    
#if 0
#endif

#if 0
#else
#endif

#pragma omp parallel for schedule(dynamic) num_threads(nthreads)
    for(int p=0; p<ntasks; p++) Perform_DR_Helper((void *)(p));


    MPI_BARRIER(node_id);


    unsigned long long int end_time = __rdtsc();
    global_time_dr += (end_time - start_time);

}




void Perform_RR_TaskQ(void)
{
    unsigned long long int start_time = __rdtsc();
    IMHERE2();

    long dimensionSize[1], tileSize[1];
    int start_processing_cell_index = global_starting_cell_index_R;
    int   end_processing_cell_index = global_ending_cell_index_R;
    long ntasks = end_processing_cell_index - start_processing_cell_index;

    IMHERE2();

    dimensionSize[0] = ntasks;  tileSize[0]=1;
    if (node_id == 0) { /*printf("<<%d>> ::: ntasks = %lld\n", node_id, ntasks); */}



#if 0
#else
#endif

#pragma omp parallel for schedule(dynamic) num_threads(nthreads)
    for(int p=0; p<ntasks; p++) Perform_RR_Helper((void *)(p));


    MPI_BARRIER(node_id);


    unsigned long long int end_time = __rdtsc();
    global_time_rr += (end_time - start_time);


}

void Perform_DRs(int argc, char **argv)
{
#define NUMBER_OF_RANDOM_FILES 1


    char *filename_D = global_D_filename; 
    char *filename_R = global_R_filename; 

    Initialize_Arrays();

    Read_D_R_File(filename_D, &global_Positions_D, &global_number_of_galaxies_on_node_D);
    Distribute_D_Particles_Amongst_Nodes();
    Compute_KD_Tree_Acceleration_Data_Structure_For_D();

    Copy_Non_Changing_Data_From_D_To_R();

    MPI_BARRIER(node_id);

    {
    }

    if (NUMBER_OF_RANDOM_FILES != 1) ERROR_PRINT();
    for(int k = 0; k < NUMBER_OF_RANDOM_FILES; k++)
    {
        Read_D_R_File(filename_R, &global_Positions_R, &global_number_of_galaxies_on_node_R);
        Distribute_R_Particles_Amongst_Nodes();
        Compute_KD_Tree_Acceleration_Data_Structure_For_R();
    
        Perform_Mandatory_Initializations(&global_grid_R, global_number_of_galaxies_on_node_R, global_Lbox, global_rminL, global_rmaxL, global_nrbin);

        Perform_RR_TaskQ();
        Perform_DR_TaskQ();
    }

    Compute_Statistics_RR();
    Compute_Statistics_DR();

    Report_Performance();
}



int main(int argc, char **argv)
{
    global_argc = argc;
    global_argv = argv;

    Initialize_MPI(argc, argv);

    ParseArgs(argc, argv);

    Perform_DRs(argc, argv);

    MPI_BARRIER(node_id);

    Checking();

    Finalize_MPI();
}

