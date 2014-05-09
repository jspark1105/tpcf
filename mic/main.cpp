/**
Copyright (c) 2013, Intel Corporation. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the <organization> nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#define MPI_COMPUTATION

#ifdef MPI_COMPUTATION
#include <mpi.h>
#endif

#include <stdio.h>
#include <float.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <stdlib.h>
#include <omp.h>

//#include "offload.h"

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
//#define ___rdtsc() 0
#define ___rdtsc() _rdtsc()

//#define HETERO_THRESHOLD (0.87)
#define HETERO_COMPUTATION

#ifdef HETERO_COMPUTATION
unsigned char *global_Template_during_hetero;
unsigned char *global_Template_during_hetero_RR_CPU;
unsigned char *global_Template_during_hetero_DR_CPU;
__attribute__ (( target (mic))) unsigned char *global_Template_during_hetero_RR_MIC;
__attribute__ (( target (mic))) unsigned char *global_Template_during_hetero_DR_MIC;

int global_hetero_cpu_number_of_D_cells_CPU;
int global_hetero_cpu_number_of_R_cells_CPU;
__attribute__ (( target (mic))) int global_hetero_cpu_number_of_D_cells_MIC;
__attribute__ (( target (mic))) int global_hetero_cpu_number_of_R_cells_MIC;
#endif

#define IMHERE2() 

//#define CORE_FREQUENCY_CPU (2.600 * 1000.0 * 1000.0 * 1000.0)
#define CORE_FREQUENCY_CPU (2.600 * 1000.0 * 1000.0 * 1000.0)
#define CORE_FREQUENCY_MIC (1.090 * 1000.0 * 1000.0 * 1000.0)
#define SIMD_WIDTH_MIC 16
#define SIMD_WIDTH_CPU  8
//#define SIMD_WIDTH_CPU  4

unsigned long long int MT_2_CPU[32] = {0};
unsigned long long int MT_3_CPU[32] = {0};
unsigned long long int MT_4_CPU[32] = {0};
unsigned long long int MT_5_CPU[32] = {0};
unsigned long long int MT_Z_CPU[32] = {0};
unsigned long long int NT_Z_CPU[32] = {0};

__attribute__ (( target (mic))) unsigned long long int MT_2_MIC[32] = {0};
__attribute__ (( target (mic))) unsigned long long int MT_3_MIC[32] = {0};
__attribute__ (( target (mic))) unsigned long long int MT_4_MIC[32] = {0};
__attribute__ (( target (mic))) unsigned long long int MT_5_MIC[32] = {0};
__attribute__ (( target (mic))) unsigned long long int MT_Z_MIC[32] = {0};
__attribute__ (( target (mic))) unsigned long long int NT_Z_MIC[32] = {0};
__attribute__ (( target (mic))) unsigned long long int OT_Z_MIC[32] = {0};
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
////////////////////// Useful Globals...  ////////////////////////////////

#define GLOBAL_THRESHOLD_PARTICLES_PER_CELL 128

#define PCL_MIN(a,b) (((a) < (b)) ? (a) : (b))
#define PCL_MAX(a,b) (((a) > (b)) ? (a) : (b))

#define ERROR_PRINT() {printf("Error in file (%s) on line (%d) in function (%s)\n", __FILE__, __LINE__, __FUNCTION__); exit(123); }
#define ERROR_PRINT_STRING(abc) {printf("Error (%s) in file (%s) on line (%d) in function (%s)\n", abc, __FILE__, __LINE__, __FUNCTION__); exit(123); }
#define GET_POINT(Array, indexx, coordinate, total_number_of_points)     *(Array + (indexx) * DIMENSIONS + (coordinate))

//extern "C" unsigned long long int read_tsc();
char global_dirname[256];

//int debug_cell_id = 127;

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
////////////////////// Parallelization related variables...  ////////////////////////////////

#define NTHREADS_MIC 240 //<<--- XXX

int node_id_CPU = 0;
int nnodes_CPU = 1;
int nthreads_CPU; 

__attribute__ (( target (mic))) int node_id_MIC = 0;
__attribute__ (( target (mic))) int nnodes_MIC = 1;
__attribute__ (( target (mic))) int nthreads_MIC = NTHREADS_MIC; 

#define MAX_THREADS_CPU 32
#define MAX_THREADS_MIC 240
//pthread_t threads[MAX_THREADS_CPU];
//pthread_attr_t attr;
//#define MY_BARRIER_CPU(threadid) barrier(threadid)

#ifndef MPI_COMPUTATION
#define MPI_BARRIER(nodeid) 
#endif

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
////////////////////// Timing related variables...  ///////////////////////////////////////////
unsigned long long int global_time_total_CPU = 0, global_time_rr_CPU = 0, global_time_dr_CPU = 0;
unsigned long long int global_time_per_thread_rr_CPU[MAX_THREADS_MIC] = {0};
unsigned long long int global_time_per_thread_dr_CPU[MAX_THREADS_MIC] = {0};

__attribute__ (( target (mic))) unsigned long long int global_time_total_MIC = 0, global_time_rr_MIC = 0, global_time_dr_MIC = 0;
__attribute__ (( target (mic))) unsigned long long int global_time_per_thread_rr_MIC[MAX_THREADS_MIC] = {0};
__attribute__ (( target (mic))) unsigned long long int global_time_per_thread_dr_MIC[MAX_THREADS_MIC] = {0};


#if 0
#define MPI_BARRIER(nodeid) MPI_Barrier(MPI_COMM_WORLD);
MPI_Request *recv_request;
MPI_Request *send_request_key;
MPI_Status *recv_status;
#endif

////////////////////// TPCF related variables...  ///////////////////////////////////////////

//#define POINTS_2D
#define TYPE float
#define DIMENSIONS 3
#define HIST_BINS 12
long long int global_number_of_galaxies_CPU = 0;
__attribute__ (( target (mic))) long long int global_number_of_galaxies_MIC = -1;

long long int global_number_of_galaxies_on_node_D;
long long int global_number_of_galaxies_on_node_R;
long long int global_galaxies_starting_index, global_galaxies_ending_index;

TYPE *global_Positions_D = NULL;
TYPE *global_Positions_R = NULL;

int global_starting_cell_index_D_CPU = -95123;
int global_ending_cell_index_D_CPU = -95123;
__attribute__ (( target (mic))) int global_starting_cell_index_D_MIC = -95123;
__attribute__ (( target (mic))) int global_ending_cell_index_D_MIC = -95123;

int global_starting_cell_index_R_CPU = -95123;
int global_ending_cell_index_R_CPU = -95123;
__attribute__ (( target (mic))) int global_starting_cell_index_R_MIC = -95123;
__attribute__ (( target (mic))) int global_ending_cell_index_R_MIC = -95123;


TYPE global_Lbox_CPU, global_rminL_CPU, global_rmaxL_CPU;
__attribute__ (( target (mic))) TYPE global_Lbox_MIC, global_rminL_MIC, global_rmaxL_MIC;

int global_nrbin_CPU;
__attribute__ (( target (mic))) int global_nrbin_MIC;

//unsigned char *global_Required_D_CPU = NULL;
unsigned char *global_Required_D_For_R_CPU = NULL;
__attribute__ (( target (mic))) unsigned char *global_Required_D_For_R_MIC;

unsigned char *global_Required_R_CPU = NULL;
__attribute__ (( target (mic))) unsigned char *global_Required_R_MIC;

#if 0
#define PADDED_ELEMENTS 32
typedef struct Sta
{
    TYPE Min[3];
    TYPE Max[3];
    int max_number_of_particles;
    int number_of_particles;
    unsigned char Padding[PADDED_ELEMENTS];
}Stat;
#endif


typedef struct Gri
{
    TYPE **Positions;   //We store the Positions for each cell together in an SOA format...
    int dimx;           //Number of Macro-Cells in the X-direction for the uniform subdivision stage...
    int dimy;           //Number of Macro-Cells in the Y-direction for the uniform subdivision stage... 
    int dimz;           //Number of Macro-Cells in the Z-direction for the uniform subdivision stage... 

    TYPE Cell_Width[DIMENSIONS];
    TYPE Min[DIMENSIONS]; 
    TYPE Max[DIMENSIONS];
    TYPE Extent[DIMENSIONS];

    //int max_particles_in_any_cell; //Max. particles in any Macro-cell...

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

//Stat *global_Bookkeeping;
Grid global_grid_D_CPU; __attribute__ (( target (mic))) Grid global_grid_D_MIC;
Grid global_grid_R_CPU; __attribute__ (( target (mic))) Grid global_grid_R_MIC;

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
long long int global_Easy_CPU[MAX_THREADS_CPU * 8] = {0};
long long int global_accumulated_easy_CPU = 0;
unsigned long long int **local_Histogram_RR_CPU;
unsigned long long int **local_Histogram_DR_CPU;
unsigned long long int global_Histogram_DR_CPU[HIST_BINS];
unsigned long long int global_Histogram_RR_CPU[HIST_BINS];
unsigned long long int *global_Overall_Histogram_DR_CPU;
unsigned long long int *global_Overall_Histogram_RR_CPU;
double global_RR_over_RR_CPU[HIST_BINS];
double global_DR_over_RR_CPU[HIST_BINS];
long long int global_stat_total_interactions_rr_CPU = 0;
long long int global_stat_total_interactions_dr_CPU = 0;
long long int global_stat_useful_interactions_rr_CPU = 0;
long long int global_stat_useful_interactions_dr_CPU = 0;
TYPE **global_Aligned_Buffer_CPU;



__attribute__ (( target (mic))) long long int global_Easy_MIC[MAX_THREADS_MIC * 8] = {0};
__attribute__ (( target (mic))) long long int global_accumulated_easy_MIC = 0;
__attribute__ (( target (mic))) unsigned long long int **local_Histogram_RR_MIC;
__attribute__ (( target (mic))) unsigned long long int **local_Histogram_DR_MIC;
__attribute__ (( target (mic))) unsigned long long int global_Histogram_DR_MIC[HIST_BINS];
__attribute__ (( target (mic))) unsigned long long int global_Histogram_RR_MIC[HIST_BINS];
__attribute__ (( target (mic))) unsigned long long int *global_Overall_Histogram_DR_MIC;
__attribute__ (( target (mic))) unsigned long long int *global_Overall_Histogram_RR_MIC;
__attribute__ (( target (mic))) double global_RR_over_RR_MIC[HIST_BINS];
__attribute__ (( target (mic))) double global_DR_over_RR_MIC[HIST_BINS];
__attribute__ (( target (mic))) long long int global_stat_total_interactions_rr_MIC = 0;
__attribute__ (( target (mic))) long long int global_stat_total_interactions_dr_MIC = 0;
__attribute__ (( target (mic))) long long int global_stat_useful_interactions_rr_MIC = 0;
__attribute__ (( target (mic))) long long int global_stat_useful_interactions_dr_MIC = 0;
__attribute__ (( target (mic))) TYPE **global_Aligned_Buffer_MIC;
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
TYPE global_rmax_2_CPU;
int global_dx_CPU;
int global_dy_CPU;
int global_dz_CPU;
TYPE *global_Rminarr_CPU;
TYPE *global_Rmaxarr_CPU;
TYPE *global_Rval_CPU;
TYPE *global_BinCorners_CPU;
TYPE *global_BinCorners2_CPU;



__attribute__ (( target (mic))) TYPE global_rmax_2_MIC;
__attribute__ (( target (mic))) int global_dx_MIC;
__attribute__ (( target (mic))) int global_dy_MIC;
__attribute__ (( target (mic))) int global_dz_MIC;
__attribute__ (( target (mic))) TYPE *global_Rminarr_MIC;
__attribute__ (( target (mic))) TYPE *global_Rmaxarr_MIC;
__attribute__ (( target (mic))) TYPE *global_Rval_MIC;
__attribute__ (( target (mic))) TYPE *global_BinCorners_MIC;
__attribute__ (( target (mic))) TYPE *global_BinCorners2_MIC;
__attribute__ (( target (mic))) TYPE *global_BinCorners3_MIC;

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
long long int *global_actual_sum_dr_CPU;
long long int *global_actual_sum_rr_CPU;
unsigned int ** global_Gather_Histogram0_CPU;
unsigned int ** global_Gather_Histogram1_CPU;
unsigned int ** global_RR_int0_CPU;
unsigned int ** global_RR_int1_CPU;
unsigned int ** global_DR_int0_CPU;
unsigned int ** global_DR_int1_CPU;
TYPE ** global_Pos1_CPU;
TYPE ** global_Bdry1_X_CPU;
TYPE ** global_Bdry1_Y_CPU;
TYPE ** global_Bdry1_Z_CPU;



__attribute__ (( target (mic))) long long int *global_actual_sum_dr_MIC;
__attribute__ (( target (mic))) long long int *global_actual_sum_rr_MIC;
__attribute__ (( target (mic))) unsigned int ** global_Gather_Histogram0_MIC;
__attribute__ (( target (mic))) unsigned int ** global_Gather_Histogram1_MIC;
__attribute__ (( target (mic))) unsigned int ** global_RR_int0_MIC;
__attribute__ (( target (mic))) unsigned int ** global_RR_int1_MIC;
__attribute__ (( target (mic))) unsigned int ** global_DR_int0_MIC;
__attribute__ (( target (mic))) unsigned int ** global_DR_int1_MIC;
__attribute__ (( target (mic))) TYPE ** global_Pos1_MIC;
__attribute__ (( target (mic))) TYPE ** global_Bdry1_X_MIC;
__attribute__ (( target (mic))) TYPE ** global_Bdry1_Y_MIC;
__attribute__ (( target (mic))) TYPE ** global_Bdry1_Z_MIC;

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


#define GET_CELL_INDEX(x0, y0, z0) (x0 + y0*dimx + z0*dimxy)

#ifdef __MIC__

#include <immintrin.h>

#define SIMD_WIDTH_LOG2_MIC 4

#define SIMDINTTYPE_MIC  __m512i //_M512
#define SIMDFPTYPE_MIC   __m512 //_M512
#define SIMDMASKTYPE_MIC __mmask 

#define _MM_SET1_INT_MIC(a) _mm512_set1_epi32(a) //_mm512_set_1to16_pi((a)) 
#define _MM_SET1_FP_MIC(a)  _mm512_set1_ps(a) //_mm512_set_1to16_ps((a)) 
#define _MM_STORE_MIC(Addr, value) _mm512_store_ps(Addr, value)


//__attribute__ (( target (mic))) float c_half_MIC = 0.5;
//__attribute__ (( target (mic))) SIMDFPTYPE_MIC xmm_half =  _MM_SET1_FP_MIC(c_half_MIC);



#define _MM_LOAD1_FP_MIC(Addr) _mm512_extload_ps((char *)(Addr),  _MM_UPCONV_PS_NONE, _MM_BROADCAST_1X16, _MM_HINT_NONE)
#define _MM_LOAD_FP_MIC(Addr) _mm512_load_ps(Addr)

#define _MM_PREFETCH1_MIC(Addr) 

//#define _MM_PREFETCH1_MIC(Addr) _mm_prefetch((char*)(Addr), _MM_HINT_T0)
//#define _MM_PREFETCH2_MIC(Addr) _mm_prefetch((char *)(Addr), _MM_HINT_T1)
//#define  _MM_PREFETCH1_EX_MIC(Addr)  _mm_prefetch((char *)(Addr), _MM_HINT_ET0);

#define _MM_SUB_FP_MIC(A,B) _mm512_sub_ps(A,B)
#define _MM_ADD_FP_MIC(A,B) _mm512_add_ps(A,B)
#define _MM_MUL_FP_MIC(A,B) _mm512_mul_ps(A,B)
#define _MM_ADD_INT_MIC(A,B) _mm512_add_epi32(A,B)
#define _MM_MASK_ADD_INT_MIC(xmm0, xmm1, k1) _mm512_mask_add_epi32(xmm0, k1, xmm0, xmm1);


#if 0
__attribute__ (( target (mic)))
inline SIMDINTTYPE_MIC _MM_MASK_ADD_INT_MIC(SIMDINTTYPE_MIC xmm0,  SIMDINTTYPE_MIC xmm1, SIMDMASKTYPE_MIC k1) 
{
  SIMDINTTYPE_MIC v1;
  v1 = _mm512_mask_add_epi32(xmm0, k1, xmm0, xmm1);
  return v1;
}
#endif

__attribute__ (( target (mic)))
inline SIMDFPTYPE_MIC _MM_LOADU_FP_MIC(float *X)
{
    __m512 vdst = _mm512_undefined();
    vdst = _mm512_extloadunpacklo_ps(vdst, (float *)(X), _MM_UPCONV_PS_NONE, _MM_HINT_NONE);
    vdst = _mm512_extloadunpackhi_ps(vdst, (float *)(X+16), _MM_UPCONV_PS_NONE, _MM_HINT_NONE);
    return vdst;
}

#define _MM_CMP_LT_MIC(xmm0, xmm1) _mm512_cmplt_ps_mask(xmm0, xmm1)
#if 0
__attribute__ (( target (mic)))
inline SIMDMASKTYPE_MIC _MM_CMP_LT_MIC(SIMDFPTYPE_MIC xmm0,  SIMDFPTYPE_MIC xmm1) 
{
  SIMDMASKTYPE_MIC k1;
  //k1 = _mm512_cmpnle_ps(xmm1, xmm0);
  k1 = _mm512_cmplt_ps_mask(xmm0, xmm1);
  return k1;
}
#endif


#define _MM_AND_MASK_MIC(A,B) _mm512_kand(A,B)
#if 0
__attribute__ (( target (mic)))
inline SIMDMASKTYPE_MIC _MM_AND_MASK_MIC(SIMDMASKTYPE_MIC xmm0,  SIMDMASKTYPE_MIC xmm1) 
{
  SIMDMASKTYPE_MIC k1;
  k1 = _mm512_kand(xmm0, xmm1);
  return k1;
}
#endif


#define _MM_HADD_MIC(A) _mm512_reduce_add_epi32(A)

#if 0
__attribute__ (( target (mic)))
inline int _MM_HADD_MIC(SIMDINTTYPE_MIC xmm0)
{
  int cnt;
  cnt = _mm512_reduce_add_epi32 (xmm0);
  return cnt;

}
#endif


#else



#include "ia32intrin.h"

#if (SIMD_WIDTH_CPU == 4)
#endif

#if (SIMD_WIDTH_CPU == 8)
#define SIMD_WIDTH_LOG2 3
#define SIMDMASKTYPE                __m256
#define SIMDFPTYPE                  __m256
#define _MM_SUB_FP                  _mm256_sub_ps
#define _MM_MUL_FP                  _mm256_mul_ps
#define _MM_ADD_FP                  _mm256_add_ps
#define SIMDINTTYPE                 __m256i
#define _MM_SET1_FP                 _mm256_set1_ps
#define _MM_LOAD_FP                 _mm256_load_ps
#define _MM_LOADU_FP                _mm256_loadu_ps
#define _MM_LOAD1_FP(a)             _mm256_set1_ps(*(a))

#define _MM_STORE_INT(A,B)          _mm_store_si128((__m128i *)(A), (B))
#define _MM_CMP_LT(A, B)            _mm256_cmp_ps(A, B, _CMP_LT_OS)


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



#endif



#endif


#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

__attribute__ (( target (mic))) 
void MICFunction_Compute_Min_Max_Dist_Sqr(TYPE *Range0_X, TYPE *Range0_Y, TYPE *Range0_Z, TYPE *Range1_X, TYPE *Range1_Y, TYPE *Range1_Z, 
        TYPE *o_min_dst_sqr, TYPE *o_max_dst_sqr)
{

    //return 123;
    TYPE min_dst_sqr = 0;
    TYPE max_dst_sqr = 0;

    if (Range0_X[0] > Range1_X[1]) min_dst_sqr += (Range0_X[0] - Range1_X[1])*(Range0_X[0] - Range1_X[1]);
    else if (Range1_X[0] > Range0_X[1]) min_dst_sqr += (Range1_X[0] - Range0_X[1])*(Range1_X[0] - Range0_X[1]);

    if (Range0_Y[0] > Range1_Y[1]) min_dst_sqr += (Range0_Y[0] - Range1_Y[1])*(Range0_Y[0] - Range1_Y[1]);
    else if (Range1_Y[0] > Range0_Y[1]) min_dst_sqr += (Range1_Y[0] - Range0_Y[1])*(Range1_Y[0] - Range0_Y[1]);

    if (Range0_Z[0] > Range1_Z[1]) min_dst_sqr += (Range0_Z[0] - Range1_Z[1])*(Range0_Z[0] - Range1_Z[1]);
    else if (Range1_Z[0] > Range0_Z[1]) min_dst_sqr += (Range1_Z[0] - Range0_Z[1])*(Range1_Z[0] - Range0_Z[1]);


    TYPE xmin = MIN(Range0_X[0], Range1_X[0]); TYPE xmax = MAX(Range0_X[1], Range1_X[1]);
    TYPE ymin = MIN(Range0_Y[0], Range1_Y[0]); TYPE ymax = MAX(Range0_Y[1], Range1_Y[1]);
    TYPE zmin = MIN(Range0_Z[0], Range1_Z[0]); TYPE zmax = MAX(Range0_Z[1], Range1_Z[1]);

    max_dst_sqr = (xmax - xmin) * (xmax - xmin) + (ymax - ymin) * (ymax - ymin) + (zmax - zmin) * (zmax - zmin);


    *o_min_dst_sqr = min_dst_sqr;
    *o_max_dst_sqr = max_dst_sqr;
}

__attribute__ (( target (mic))) 
void MICFunction_Compute_min_max_bin_id(TYPE *BinCorners2, int nrbin, TYPE min_dst_sqr, TYPE max_dst_sqr, int *o_min_bin_id, int *o_max_bin_id)
{
#ifdef __MIC__

#if 0
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
#endif

#if 0
    {
        SIMDFPTYPE_MIC xmm_min = _MM_LOAD1_FP_MIC(&min_dst_sqr);
        SIMDFPTYPE_MIC xmm_max = _MM_LOAD1_FP_MIC(&max_dst_sqr);

        int mask0 = _MM_CMP_LT_MIC(xmm_min, _MM_LOAD_FP_MIC(global_BinCorners3_MIC));
        int mask1 = _MM_CMP_LT_MIC(xmm_max, _MM_LOAD_FP_MIC(global_BinCorners3_MIC));

        int popcount0 = _mm_countbits_32(mask0);
        int popcount1 = _mm_countbits_32(mask1);

        if ( (min_bin_id + popcount0) != 15) 
        {
            printf("min_bin_id = %d ::: popcount0 = %d\n", min_bin_id, popcount0);
            //ERROR_PRINT();
        }
        if ( (max_bin_id + popcount1) != 15) 
        {
            printf("max_bin_id = %d ::: popcount1 = %d\n", max_bin_id, popcount1);
            //ERROR_PRINT();
        }
    }
#endif

    int min_bin_id = 15 - _mm_countbits_32(_MM_CMP_LT_MIC(_MM_LOAD1_FP_MIC(&min_dst_sqr), _MM_LOAD_FP_MIC(global_BinCorners3_MIC)));
    int max_bin_id = 15 - _mm_countbits_32(_MM_CMP_LT_MIC(_MM_LOAD1_FP_MIC(&max_dst_sqr), _MM_LOAD_FP_MIC(global_BinCorners3_MIC)));

    *o_min_bin_id = min_bin_id;
    *o_max_bin_id = max_bin_id;

#endif
}

void CPUFunction_Compute_Min_Max_Dist_Sqr(TYPE *Range0_X, TYPE *Range0_Y, TYPE *Range0_Z, TYPE *Range1_X, TYPE *Range1_Y, TYPE *Range1_Z, 
        TYPE *o_min_dst_sqr, TYPE *o_max_dst_sqr)
{

    //return 123;
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

void CPUFunction_Compute_min_max_bin_id(TYPE *BinCorners2, int nrbin, TYPE min_dst_sqr, TYPE max_dst_sqr, int *o_min_bin_id, int *o_max_bin_id)
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

//======================================
#ifndef __MIC__
#if (SIMD_WIDTH_CPU == 8)

void CPUFunction_Compute_Distance_And_Populate_Hist_1(TYPE *Pos0, int count0, TYPE *Pos1, int count1, TYPE *BinCorners2, int nrbin, 
        unsigned int *DD_int0, unsigned int *DD_int1, int min_bin_id, int max_bin_id, TYPE *Range1_X, TYPE *Range1_Y, TYPE *Range1_Z, int threadid)
{
    //Basically max_bin_id == min_bin_id...

    int self = (Pos0 == Pos1);
    int total_number_of_interactions = (self) ? ((count0 *(count1 - 1))/2) : (count0 * count1);
    DD_int0[min_bin_id] += total_number_of_interactions;
    global_Easy_CPU[8*threadid] += total_number_of_interactions;
}


void CPUFunction_Compute_Distance_And_Populate_Hist_3(TYPE *Pos0, int count0, TYPE *Pos1, int count1, TYPE *BinCorners2, int nrbin, 
        unsigned int *DD_int0, unsigned int *DD_int1, int min_bin_id, int max_bin_id, TYPE *Range1_X, TYPE *Range1_Y, TYPE *Range1_Z, int threadid)
{
#ifndef __MIC__
    TYPE separation_element1 = BinCorners2[min_bin_id+1];
    TYPE separation_element2 = BinCorners2[min_bin_id+2];

    SIMDFPTYPE xmm_separation_element1 = _MM_SET1_FP(separation_element1);
    SIMDFPTYPE xmm_separation_element2 = _MM_SET1_FP(separation_element2);

    __m128i xmm_result1 = _mm_set1_epi32(0);
    __m128i xmm_result2 = _mm_set1_epi32(0);

    int self = 0;
    if (Pos0 == Pos1) self = 1;

    if (self) 
    {
        for(int i=0; i<count0; i++)
        {
            int starting_index = (self) ? (i+1) : 0;
              
            SIMDFPTYPE xmm_x0 = _MM_LOAD1_FP(Pos0 + i + 0*count0);
            SIMDFPTYPE xmm_y0 = _MM_LOAD1_FP(Pos0 + i + 1*count0);
            SIMDFPTYPE xmm_z0 = _MM_LOAD1_FP(Pos0 + i + 2*count0);
              
            int particles_left = count1 - starting_index;
            int particles_left_prime = ((particles_left >> SIMD_WIDTH_LOG2) << SIMD_WIDTH_LOG2);
            int count1_prime = starting_index + particles_left_prime;

            for(int j=starting_index; j < count1_prime; j+=SIMD_WIDTH_CPU)
            {
                SIMDFPTYPE xmm_x1 = _MM_LOADU_FP(Pos1 + j + 0*count1);
                SIMDFPTYPE xmm_y1 = _MM_LOADU_FP(Pos1 + j + 1*count1);
                SIMDFPTYPE xmm_z1 = _MM_LOADU_FP(Pos1 + j + 2*count1);

                SIMDFPTYPE xmm_diff_X = _MM_SUB_FP(xmm_x0, xmm_x1);
                SIMDFPTYPE xmm_diff_Y = _MM_SUB_FP(xmm_y0, xmm_y1);
                SIMDFPTYPE xmm_diff_Z = _MM_SUB_FP(xmm_z0, xmm_z1);

                SIMDFPTYPE xmm_norm_2 = _MM_ADD_FP(_MM_ADD_FP(_MM_MUL_FP(xmm_diff_X, xmm_diff_X), _MM_MUL_FP(xmm_diff_Y, xmm_diff_Y)), _MM_MUL_FP(xmm_diff_Z, xmm_diff_Z));

                SIMDINTTYPE t = _mm256_castps_si256(_MM_CMP_LT(xmm_norm_2, xmm_separation_element1));
                xmm_result1 = _mm_sub_epi32(xmm_result1, _mm256_castsi256_si128(t));
                xmm_result1 = _mm_sub_epi32(xmm_result1, _mm256_extractf128_si256(t, 1));
                t = _mm256_castps_si256(_MM_CMP_LT(xmm_norm_2, xmm_separation_element2));
                xmm_result2 = _mm_sub_epi32(xmm_result2, _mm256_castsi256_si128(t));
                xmm_result2 = _mm_sub_epi32(xmm_result2, _mm256_extractf128_si256(t, 1));
            }

            {
                int j = count1_prime;
                int remaining = count1 - count1_prime;
                SIMDFPTYPE xmm_anding = _MM_LOAD_FP((float *)(Remaining[remaining]));

                SIMDFPTYPE xmm_x1 = _MM_LOADU_FP(Pos1 + j + 0*count1);
                SIMDFPTYPE xmm_y1 = _MM_LOADU_FP(Pos1 + j + 1*count1);
                SIMDFPTYPE xmm_z1 = _MM_LOADU_FP(Pos1 + j + 2*count1);

                SIMDFPTYPE xmm_diff_X = _MM_SUB_FP(xmm_x0, xmm_x1);
                SIMDFPTYPE xmm_diff_Y = _MM_SUB_FP(xmm_y0, xmm_y1);
                SIMDFPTYPE xmm_diff_Z = _MM_SUB_FP(xmm_z0, xmm_z1);

                SIMDFPTYPE xmm_norm_2 = _MM_ADD_FP(_MM_ADD_FP(_MM_MUL_FP(xmm_diff_X, xmm_diff_X), _MM_MUL_FP(xmm_diff_Y, xmm_diff_Y)), _MM_MUL_FP(xmm_diff_Z, xmm_diff_Z));

                SIMDINTTYPE t = _mm256_castps_si256(_mm256_and_ps(xmm_anding, _MM_CMP_LT(xmm_norm_2, xmm_separation_element1)));
                xmm_result1 = _mm_sub_epi32(xmm_result1, _mm256_castsi256_si128(t));
                xmm_result1 = _mm_sub_epi32(xmm_result1, _mm256_extractf128_si256(t, 1));
                t = _mm256_castps_si256(_mm256_and_ps(xmm_anding, _MM_CMP_LT(xmm_norm_2, xmm_separation_element2)));
                xmm_result2 = _mm_sub_epi32(xmm_result2, _mm256_castsi256_si128(t));
                xmm_result2 = _mm_sub_epi32(xmm_result2, _mm256_extractf128_si256(t, 1));
          
            }
        }
    }
    else 
    {
      
        for(int i=0; i<count0; i++)
        {
            SIMDFPTYPE xmm_x0 = _MM_LOAD1_FP(Pos0 + i + 0*count0);
            SIMDFPTYPE xmm_y0 = _MM_LOAD1_FP(Pos0 + i + 1*count0);
            SIMDFPTYPE xmm_z0 = _MM_LOAD1_FP(Pos0 + i + 2*count0);
              
            for(int j=0; j < count1; j+=SIMD_WIDTH_CPU)
            {
                SIMDFPTYPE xmm_x1 = _MM_LOAD_FP(Pos1 + j + 0*count1);
                SIMDFPTYPE xmm_y1 = _MM_LOAD_FP(Pos1 + j + 1*count1);
                SIMDFPTYPE xmm_z1 = _MM_LOAD_FP(Pos1 + j + 2*count1);

                SIMDFPTYPE xmm_diff_X = _MM_SUB_FP(xmm_x0, xmm_x1);
                SIMDFPTYPE xmm_diff_Y = _MM_SUB_FP(xmm_y0, xmm_y1);
                SIMDFPTYPE xmm_diff_Z = _MM_SUB_FP(xmm_z0, xmm_z1);

                SIMDFPTYPE xmm_norm_2 = _MM_ADD_FP(_MM_ADD_FP(_MM_MUL_FP(xmm_diff_X, xmm_diff_X), _MM_MUL_FP(xmm_diff_Y, xmm_diff_Y)), _MM_MUL_FP(xmm_diff_Z, xmm_diff_Z));

              
                SIMDINTTYPE t = _mm256_castps_si256(_MM_CMP_LT(xmm_norm_2, xmm_separation_element1));
                xmm_result1 = _mm_sub_epi32(xmm_result1, _mm256_castsi256_si128(t));
                xmm_result1 = _mm_sub_epi32(xmm_result1, _mm256_extractf128_si256(t, 1));
                t = _mm256_castps_si256(_MM_CMP_LT(xmm_norm_2, xmm_separation_element2));
                xmm_result2 = _mm_sub_epi32(xmm_result2, _mm256_castsi256_si128(t));
                xmm_result2 = _mm_sub_epi32(xmm_result2, _mm256_extractf128_si256(t, 1));
            }
        }
    }

    {
        __declspec (align(64)) int Temp[8];
        _MM_STORE_INT(Temp+0, xmm_result1);
        _MM_STORE_INT(Temp+4, xmm_result2);
            
        int total =  (self) ? ((count0*(count1-1))/2) : (count0 * count1);
        int sum1 = Temp[0] + Temp[1] + Temp[2] + Temp[3];
        int sum2 = Temp[4] + Temp[5] + Temp[6] + Temp[7];

        DD_int0[min_bin_id+0] += sum1;
        DD_int0[min_bin_id+1] += sum2-sum1;
        DD_int0[min_bin_id+2] += (total - sum2);
    }
#endif
}

void CPUFunction_Compute_Distance_And_Populate_Hist_2(TYPE *Pos0, int count0, TYPE *Pos1, int count1, TYPE *BinCorners2, int nrbin, 
        unsigned int *DD_int0, unsigned int *DD_int1, int min_bin_id, int max_bin_id, TYPE *Range1_X, TYPE *Range1_Y, TYPE *Range1_Z, int threadid)
{
    //Basically max_bin_id == min_bin_id + 1...

#ifndef __MIC__

    TYPE separation_element = BinCorners2[min_bin_id+1];
    SIMDFPTYPE xmm_separation_element = _MM_SET1_FP(separation_element);

    int self = 0;
    if (Pos0 == Pos1) self = 1;

    //if (self == 1) ERROR_PRINT();
        
    __m128i xmm_result_0 = _mm_set1_epi32(0);

    TYPE Range0_X[2], Range0_Y[2], Range0_Z[2];

    int to_be_subtracted = 0;

    if (self)
    {
        //printf("fdsf\n");
        unsigned int something_to_check0 = DD_int0[min_bin_id + 2];
        unsigned int something_to_check1 = DD_int0[min_bin_id + 2];
        CPUFunction_Compute_Distance_And_Populate_Hist_3(Pos0, count0, Pos1, count1, BinCorners2, nrbin, DD_int0, DD_int1, min_bin_id, max_bin_id+1, Range1_X, Range1_Y, Range1_Z, threadid);
        if (something_to_check0 != DD_int0[min_bin_id + 2]) ERROR_PRINT();
        if (something_to_check1 != DD_int0[min_bin_id + 2]) ERROR_PRINT();
    }

    else
    {
        for(int i=0; i<count0; i++)
        {
        
            SIMDFPTYPE xmm_x0_0 = _MM_LOAD1_FP(Pos0 + i + 0 + 0*count0);
            SIMDFPTYPE xmm_y0_0 = _MM_LOAD1_FP(Pos0 + i + 0 + 1*count0);
            SIMDFPTYPE xmm_z0_0 = _MM_LOAD1_FP(Pos0 + i + 0 + 2*count0);

            for(int j=0; j < count1; j+=SIMD_WIDTH_CPU)
            {
                SIMDFPTYPE xmm_x1 = _MM_LOAD_FP(Pos1 + j + 0*count1);
                SIMDFPTYPE xmm_y1 = _MM_LOAD_FP(Pos1 + j + 1*count1);
                SIMDFPTYPE xmm_z1 = _MM_LOAD_FP(Pos1 + j + 2*count1);

                SIMDFPTYPE xmm_diff_X_0 = _MM_SUB_FP(xmm_x0_0, xmm_x1);
                SIMDFPTYPE xmm_diff_Y_0 = _MM_SUB_FP(xmm_y0_0, xmm_y1);
                SIMDFPTYPE xmm_diff_Z_0 = _MM_SUB_FP(xmm_z0_0, xmm_z1);

                SIMDFPTYPE xmm_norm_2_0 = _MM_ADD_FP(_MM_ADD_FP(_MM_MUL_FP(xmm_diff_X_0, xmm_diff_X_0), _MM_MUL_FP(xmm_diff_Y_0, xmm_diff_Y_0)), _MM_MUL_FP(xmm_diff_Z_0, xmm_diff_Z_0));

                SIMDINTTYPE t = _mm256_castps_si256(_MM_CMP_LT(xmm_norm_2_0, xmm_separation_element));
                xmm_result_0 = _mm_sub_epi32(xmm_result_0, _mm256_castsi256_si128(t));
                xmm_result_0 = _mm_sub_epi32(xmm_result_0, _mm256_extractf128_si256(t, 1));
            }
        }

        __m128i xmm_result = xmm_result_0;
        __declspec (align(64)) int Temp[4];
        _MM_STORE_INT(Temp, xmm_result);
        
        int total =  (self) ? ((count0*(count1-1))/2) : (count0 * count1);
        int sum = Temp[0] + Temp[1] + Temp[2] + Temp[3];

        total -= to_be_subtracted;

        DD_int0[min_bin_id] += sum;
        DD_int0[min_bin_id+1] += (total - sum);
    }
#endif
}


void CPUFunction_Compute_Distance_And_Populate_Hist_4(TYPE *Pos0, int count0, TYPE *Pos1, int count1, TYPE *BinCorners2, int nrbin, 
        unsigned int *DD_int0, unsigned int *DD_int1, int min_bin_id, int max_bin_id, TYPE *Range1_X, TYPE *Range1_Y, TYPE *Range1_Z, int threadid)
{

#ifndef __MIC__
    TYPE separation_element1 = BinCorners2[min_bin_id+1];
    TYPE separation_element2 = BinCorners2[min_bin_id+2];
    TYPE separation_element3 = BinCorners2[min_bin_id+3];

    SIMDFPTYPE xmm_separation_element1 = _MM_SET1_FP(separation_element1);
    SIMDFPTYPE xmm_separation_element2 = _MM_SET1_FP(separation_element2);
    SIMDFPTYPE xmm_separation_element3 = _MM_SET1_FP(separation_element3);

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
              
            SIMDFPTYPE xmm_x0 = _MM_LOAD1_FP(Pos0 + i + 0*count0);
            SIMDFPTYPE xmm_y0 = _MM_LOAD1_FP(Pos0 + i + 1*count0);
            SIMDFPTYPE xmm_z0 = _MM_LOAD1_FP(Pos0 + i + 2*count0);
              
            int particles_left = count1 - starting_index;
            int particles_left_prime = ((particles_left >> SIMD_WIDTH_LOG2) << SIMD_WIDTH_LOG2);
            int count1_prime = starting_index + particles_left_prime;

            for(int j=starting_index; j < count1_prime; j+=SIMD_WIDTH_CPU)
            {
                SIMDFPTYPE xmm_x1 = _MM_LOADU_FP(Pos1 + j + 0*count1);
                SIMDFPTYPE xmm_y1 = _MM_LOADU_FP(Pos1 + j + 1*count1);
                SIMDFPTYPE xmm_z1 = _MM_LOADU_FP(Pos1 + j + 2*count1);

                SIMDFPTYPE xmm_diff_X = _MM_SUB_FP(xmm_x0, xmm_x1);
                SIMDFPTYPE xmm_diff_Y = _MM_SUB_FP(xmm_y0, xmm_y1);
                SIMDFPTYPE xmm_diff_Z = _MM_SUB_FP(xmm_z0, xmm_z1);

                SIMDFPTYPE xmm_norm_2 = _MM_ADD_FP(_MM_ADD_FP(_MM_MUL_FP(xmm_diff_X, xmm_diff_X), _MM_MUL_FP(xmm_diff_Y, xmm_diff_Y)), _MM_MUL_FP(xmm_diff_Z, xmm_diff_Z));

                SIMDINTTYPE t = _mm256_castps_si256(_MM_CMP_LT(xmm_norm_2, xmm_separation_element1));
                xmm_result1 = _mm_sub_epi32(xmm_result1, _mm256_castsi256_si128(t));
                xmm_result1 = _mm_sub_epi32(xmm_result1, _mm256_extractf128_si256(t, 1));
                t = _mm256_castps_si256(_MM_CMP_LT(xmm_norm_2, xmm_separation_element2));
                xmm_result2 = _mm_sub_epi32(xmm_result2, _mm256_castsi256_si128(t));
                xmm_result2 = _mm_sub_epi32(xmm_result2, _mm256_extractf128_si256(t, 1));
                t = _mm256_castps_si256(_MM_CMP_LT(xmm_norm_2, xmm_separation_element3));
                xmm_result3 = _mm_sub_epi32(xmm_result3, _mm256_castsi256_si128(t));
                xmm_result3 = _mm_sub_epi32(xmm_result3, _mm256_extractf128_si256(t, 1));
            }

            {
              
                int j = count1_prime;
                int remaining = count1 - count1_prime;
                SIMDFPTYPE xmm_anding = _MM_LOAD_FP((float *)(Remaining[remaining]));

                SIMDFPTYPE xmm_x1 = _MM_LOADU_FP(Pos1 + j + 0*count1);
                SIMDFPTYPE xmm_y1 = _MM_LOADU_FP(Pos1 + j + 1*count1);
                SIMDFPTYPE xmm_z1 = _MM_LOADU_FP(Pos1 + j + 2*count1);

                SIMDFPTYPE xmm_diff_X = _MM_SUB_FP(xmm_x0, xmm_x1);
                SIMDFPTYPE xmm_diff_Y = _MM_SUB_FP(xmm_y0, xmm_y1);
                SIMDFPTYPE xmm_diff_Z = _MM_SUB_FP(xmm_z0, xmm_z1);

                SIMDFPTYPE xmm_norm_2 = _MM_ADD_FP(_MM_ADD_FP(_MM_MUL_FP(xmm_diff_X, xmm_diff_X), _MM_MUL_FP(xmm_diff_Y, xmm_diff_Y)), _MM_MUL_FP(xmm_diff_Z, xmm_diff_Z));

                SIMDINTTYPE t = _mm256_castps_si256(_mm256_and_ps(xmm_anding, _MM_CMP_LT(xmm_norm_2, xmm_separation_element1)));
                xmm_result1 = _mm_sub_epi32(xmm_result1, _mm256_castsi256_si128(t));
                xmm_result1 = _mm_sub_epi32(xmm_result1, _mm256_extractf128_si256(t, 1));
                t = _mm256_castps_si256(_mm256_and_ps(xmm_anding, _MM_CMP_LT(xmm_norm_2, xmm_separation_element2)));
                xmm_result2 = _mm_sub_epi32(xmm_result2, _mm256_castsi256_si128(t));
                xmm_result2 = _mm_sub_epi32(xmm_result2, _mm256_extractf128_si256(t, 1));
                t = _mm256_castps_si256(_mm256_and_ps(xmm_anding, _MM_CMP_LT(xmm_norm_2, xmm_separation_element3)));
                xmm_result3 = _mm_sub_epi32(xmm_result3, _mm256_castsi256_si128(t));
                xmm_result3 = _mm_sub_epi32(xmm_result3, _mm256_extractf128_si256(t, 1));
            }
        }
    }
    else 
    {
      
        for(int i=0; i<count0; i++)
        {
            SIMDFPTYPE xmm_x0 = _MM_LOAD1_FP(Pos0 + i + 0*count0);
            SIMDFPTYPE xmm_y0 = _MM_LOAD1_FP(Pos0 + i + 1*count0);
            SIMDFPTYPE xmm_z0 = _MM_LOAD1_FP(Pos0 + i + 2*count0);
              
            for(int j=0; j < count1; j+=SIMD_WIDTH_CPU)
            {
                SIMDFPTYPE xmm_x1 = _MM_LOAD_FP(Pos1 + j + 0*count1);
                SIMDFPTYPE xmm_y1 = _MM_LOAD_FP(Pos1 + j + 1*count1);
                SIMDFPTYPE xmm_z1 = _MM_LOAD_FP(Pos1 + j + 2*count1);

                SIMDFPTYPE xmm_diff_X = _MM_SUB_FP(xmm_x0, xmm_x1);
                SIMDFPTYPE xmm_diff_Y = _MM_SUB_FP(xmm_y0, xmm_y1);
                SIMDFPTYPE xmm_diff_Z = _MM_SUB_FP(xmm_z0, xmm_z1);

                SIMDFPTYPE xmm_norm_2 = _MM_ADD_FP(_MM_ADD_FP(_MM_MUL_FP(xmm_diff_X, xmm_diff_X), _MM_MUL_FP(xmm_diff_Y, xmm_diff_Y)), _MM_MUL_FP(xmm_diff_Z, xmm_diff_Z));

                SIMDINTTYPE t = _mm256_castps_si256(_MM_CMP_LT(xmm_norm_2, xmm_separation_element1));
                xmm_result1 = _mm_sub_epi32(xmm_result1, _mm256_castsi256_si128(t));
                xmm_result1 = _mm_sub_epi32(xmm_result1, _mm256_extractf128_si256(t, 1));
                t = _mm256_castps_si256(_MM_CMP_LT(xmm_norm_2, xmm_separation_element2));
                xmm_result2 = _mm_sub_epi32(xmm_result2, _mm256_castsi256_si128(t));
                xmm_result2 = _mm_sub_epi32(xmm_result2, _mm256_extractf128_si256(t, 1));
                t = _mm256_castps_si256(_MM_CMP_LT(xmm_norm_2, xmm_separation_element3));
                xmm_result3 = _mm_sub_epi32(xmm_result3, _mm256_castsi256_si128(t));
                xmm_result3 = _mm_sub_epi32(xmm_result3, _mm256_extractf128_si256(t, 1));
          
            }
        }
    }


    {
        __declspec (align(64)) int Temp[12];
        _MM_STORE_INT(Temp+0, xmm_result1);
        _MM_STORE_INT(Temp+4, xmm_result2);
        _MM_STORE_INT(Temp+8, xmm_result3);
    
            
        int total =  (self) ? ((count0*(count1-1))/2) : (count0 * count1);
        int sum1 = Temp[0] + Temp[1] + Temp[2] + Temp[3];
        int sum2 = Temp[4] + Temp[5] + Temp[6] + Temp[7];
        int sum3 = Temp[8] + Temp[9] + Temp[10] + Temp[11];

        DD_int0[min_bin_id+0] += sum1;
        DD_int0[min_bin_id+1] += sum2-sum1;
        DD_int0[min_bin_id+2] += sum3-sum2;
        DD_int0[min_bin_id+3] += (total - sum3);
    }
#endif
}


__attribute__((noinline))
void CPUFunction_Compute_Distance_And_Populate_Hist_N(TYPE *Pos0, int count0, TYPE *Pos1, int count1, TYPE *BinCorners2, int nrbin, 
        unsigned int *DD_int0, unsigned int *DD_int1, int min_bin_id, int max_bin_id, TYPE *Range1_X, TYPE *Range1_Y, TYPE *Range1_Z, int threadid)
{
#ifndef __MIC__

    int self = 0;
    if (Pos0 == Pos1) self = 1;

    //SCompute_Distance_And_Populate_Hist_N(Pos0, count0, Pos1, count1, DD_int0, DD_int1, nrbin, BinCorners2, min_bin_id, max_bin_id, case_considered); return;

    //printf("min_bin_id = %d ::: max_bin_id = %d\n", min_bin_id, max_bin_id);
    TYPE separation_element[HIST_BINS];
    SIMDFPTYPE xmm_separation_element[HIST_BINS];
    __m128i xmm_result[HIST_BINS];

    for(int k=min_bin_id; k<max_bin_id; k++)     separation_element[k-min_bin_id] = BinCorners2[k+1];
    for(int k=min_bin_id; k<max_bin_id; k++) xmm_separation_element[k-min_bin_id] = _MM_SET1_FP(separation_element[k-min_bin_id]);
    for(int k=min_bin_id; k<max_bin_id; k++) xmm_result[k-min_bin_id] = _mm_set1_epi32(0);

    int number_of_bins_minus_one = (max_bin_id - min_bin_id);

    if (self) 
    {
        for(int i=0; i<count0; i++)
        {
            int starting_index = i+1;
              
            SIMDFPTYPE xmm_x0 = _MM_LOAD1_FP(Pos0 + i + 0*count0);
            SIMDFPTYPE xmm_y0 = _MM_LOAD1_FP(Pos0 + i + 1*count0);
            SIMDFPTYPE xmm_z0 = _MM_LOAD1_FP(Pos0 + i + 2*count0);
              
            int particles_left = count1 - starting_index;
            int particles_left_prime = ((particles_left >> SIMD_WIDTH_LOG2) << SIMD_WIDTH_LOG2);
            int count1_prime = starting_index + particles_left_prime;

          
            for(int j=starting_index; j < count1_prime; j+=SIMD_WIDTH_CPU)
            {
                SIMDFPTYPE xmm_x1 = _MM_LOADU_FP(Pos1 + j + 0*count1);
                SIMDFPTYPE xmm_y1 = _MM_LOADU_FP(Pos1 + j + 1*count1);
                SIMDFPTYPE xmm_z1 = _MM_LOADU_FP(Pos1 + j + 2*count1);

                SIMDFPTYPE xmm_diff_X = _MM_SUB_FP(xmm_x0, xmm_x1);
                SIMDFPTYPE xmm_diff_Y = _MM_SUB_FP(xmm_y0, xmm_y1);
                SIMDFPTYPE xmm_diff_Z = _MM_SUB_FP(xmm_z0, xmm_z1);

                SIMDFPTYPE xmm_norm_2 = _MM_ADD_FP(_MM_ADD_FP(_MM_MUL_FP(xmm_diff_X, xmm_diff_X), _MM_MUL_FP(xmm_diff_Y, xmm_diff_Y)), _MM_MUL_FP(xmm_diff_Z, xmm_diff_Z));

              
                for(int k=0; k<number_of_bins_minus_one; k++) 
                {
                  
                    SIMDINTTYPE t = _mm256_castps_si256(_MM_CMP_LT(xmm_norm_2, xmm_separation_element[k]));
                    xmm_result[k] = _mm_sub_epi32(xmm_result[k], _mm256_castsi256_si128(t));
                    xmm_result[k] = _mm_sub_epi32(xmm_result[k], _mm256_extractf128_si256(t, 1));
                }
            }

            {
              
                int j = count1_prime;
                int remaining = count1 - count1_prime;
                SIMDFPTYPE xmm_anding = _MM_LOAD_FP((float *)(Remaining[remaining]));

                SIMDFPTYPE xmm_x1 = _MM_LOADU_FP(Pos1 + j + 0*count1);
                SIMDFPTYPE xmm_y1 = _MM_LOADU_FP(Pos1 + j + 1*count1);
                SIMDFPTYPE xmm_z1 = _MM_LOADU_FP(Pos1 + j + 2*count1);

                SIMDFPTYPE xmm_diff_X = _MM_SUB_FP(xmm_x0, xmm_x1);
                SIMDFPTYPE xmm_diff_Y = _MM_SUB_FP(xmm_y0, xmm_y1);
                SIMDFPTYPE xmm_diff_Z = _MM_SUB_FP(xmm_z0, xmm_z1);

                SIMDFPTYPE xmm_norm_2 = _MM_ADD_FP(_MM_ADD_FP(_MM_MUL_FP(xmm_diff_X, xmm_diff_X), _MM_MUL_FP(xmm_diff_Y, xmm_diff_Y)), _MM_MUL_FP(xmm_diff_Z, xmm_diff_Z));

                for(int k=0; k<number_of_bins_minus_one; k++) 
                {
                    SIMDINTTYPE t = _mm256_castps_si256(_mm256_and_ps(xmm_anding, _MM_CMP_LT(xmm_norm_2, xmm_separation_element[k])));
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
            SIMDFPTYPE xmm_x0 = _MM_LOAD1_FP(Pos0 + i + 0*count0);
            SIMDFPTYPE xmm_y0 = _MM_LOAD1_FP(Pos0 + i + 1*count0);
            SIMDFPTYPE xmm_z0 = _MM_LOAD1_FP(Pos0 + i + 2*count0);

            for(int j=0; j < count1; j+=SIMD_WIDTH_CPU)
            {
                SIMDFPTYPE xmm_x1 = _MM_LOAD_FP(Pos1 + j + 0*count1);
                SIMDFPTYPE xmm_y1 = _MM_LOAD_FP(Pos1 + j + 1*count1);
                SIMDFPTYPE xmm_z1 = _MM_LOAD_FP(Pos1 + j + 2*count1);

                SIMDFPTYPE xmm_diff_X = _MM_SUB_FP(xmm_x0, xmm_x1);
                SIMDFPTYPE xmm_diff_Y = _MM_SUB_FP(xmm_y0, xmm_y1);
                SIMDFPTYPE xmm_diff_Z = _MM_SUB_FP(xmm_z0, xmm_z1);

                SIMDFPTYPE xmm_norm_2 = _MM_ADD_FP(_MM_ADD_FP(_MM_MUL_FP(xmm_diff_X, xmm_diff_X), _MM_MUL_FP(xmm_diff_Y, xmm_diff_Y)), _MM_MUL_FP(xmm_diff_Z, xmm_diff_Z));

                for(int k=0; k<number_of_bins_minus_one; k++) 
                {
                    SIMDINTTYPE t = _mm256_castps_si256(_MM_CMP_LT(xmm_norm_2, xmm_separation_element[k]));
                    xmm_result[k] = _mm_sub_epi32(xmm_result[k], _mm256_castsi256_si128(t));
                    xmm_result[k] = _mm_sub_epi32(xmm_result[k], _mm256_extractf128_si256(t, 1));
                }
            }
        }
    }

    {
        __declspec (align(64)) int Temp[4*HIST_BINS];
        int sum[HIST_BINS];

        for(int k=0; k<number_of_bins_minus_one; k++) _MM_STORE_INT(Temp+4*k, xmm_result[k]);
            
        int total =  (self) ? ((count0*(count1-1))/2) : (count0 * count1);

        for(int k=0; k<number_of_bins_minus_one; k++) sum[k] =  Temp[4*k + 0] + Temp[4*k + 1] + Temp[4*k + 2] + Temp[4*k + 3];
        sum[number_of_bins_minus_one] = total;

        DD_int0[min_bin_id+0] += sum[0];
        for(int k=1; k<=number_of_bins_minus_one; k++) DD_int0[min_bin_id+k] += (sum[k] - sum[k-1]);

    }
#endif
}
#endif
#endif
//======================================



__attribute__ (( target (mic))) 
void MICFunction_Compute_Distance_And_Populate_Hist_1(TYPE *Pos0, int count0, TYPE *Pos1, int count1, TYPE *BinCorners2, int nrbin, 
        unsigned int *DD_int0, unsigned int *DD_int1, int min_bin_id, int max_bin_id, TYPE *Range1_X, TYPE *Range1_Y, TYPE *Range1_Z, int threadid)
{
    //Basically max_bin_id == min_bin_id...

    int self = (Pos0 == Pos1);
    int total_number_of_interactions = (self) ? ((count0 *(count1 - 1))/2) : (count0 * count1);
    DD_int0[min_bin_id] += total_number_of_interactions;
    global_Easy_MIC[8*threadid] += total_number_of_interactions;
}

__attribute__ (( target (mic))) 
void MICFunction_Compute_Distance_And_Populate_Hist_3(TYPE *Pos0, int count0, TYPE *Pos1, int count1, TYPE *BinCorners2, int nrbin, 
        unsigned int *DD_int0, unsigned int *DD_int1, int min_bin_id, int max_bin_id, TYPE *Range1_X, TYPE *Range1_Y, TYPE *Range1_Z, int threadid)
{
#ifdef __MIC__


    int   c_one_MIC  = 1;
    SIMDINTTYPE_MIC xmm_one =  _MM_SET1_INT_MIC(c_one_MIC);

    TYPE separation_element1 = BinCorners2[min_bin_id+1];
    TYPE separation_element2 = BinCorners2[min_bin_id+2];

    SIMDFPTYPE_MIC xmm_separation_element1 = _MM_SET1_FP_MIC(separation_element1);
    SIMDFPTYPE_MIC xmm_separation_element2 = _MM_SET1_FP_MIC(separation_element2);

    SIMDINTTYPE_MIC xmm_result1 = _MM_SET1_INT_MIC(0);
    SIMDINTTYPE_MIC xmm_result2 = _MM_SET1_INT_MIC(0);

    int self = 0;
    if (Pos0 == Pos1) self = 1;

    if (self) 
    {
        for(int i=0; i<count0; i++)
        {
            int starting_index = (self) ? (i+1) : 0;
              
            SIMDFPTYPE_MIC xmm_x0 = _MM_LOAD1_FP_MIC(Pos0 + i + 0*count0);
            SIMDFPTYPE_MIC xmm_y0 = _MM_LOAD1_FP_MIC(Pos0 + i + 1*count0);
            SIMDFPTYPE_MIC xmm_z0 = _MM_LOAD1_FP_MIC(Pos0 + i + 2*count0);
              
            int particles_left = count1 - starting_index;
            int particles_left_prime = ((particles_left >> SIMD_WIDTH_LOG2_MIC) << SIMD_WIDTH_LOG2_MIC);
            int count1_prime = starting_index + particles_left_prime;

            for(int j=starting_index; j < count1_prime; j+=SIMD_WIDTH_MIC)
            {
                _MM_PREFETCH1_MIC(Pos1 + j + 0*count1 + SIMD_WIDTH_MIC);
                _MM_PREFETCH1_MIC(Pos1 + j + 1*count1 + SIMD_WIDTH_MIC);
                _MM_PREFETCH1_MIC(Pos1 + j + 2*count1 + SIMD_WIDTH_MIC);

                SIMDFPTYPE_MIC xmm_x1 = _MM_LOADU_FP_MIC(Pos1 + j + 0*count1);
                SIMDFPTYPE_MIC xmm_y1 = _MM_LOADU_FP_MIC(Pos1 + j + 1*count1);
                SIMDFPTYPE_MIC xmm_z1 = _MM_LOADU_FP_MIC(Pos1 + j + 2*count1);

                SIMDFPTYPE_MIC xmm_diff_X = _MM_SUB_FP_MIC(xmm_x0, xmm_x1);
                SIMDFPTYPE_MIC xmm_diff_Y = _MM_SUB_FP_MIC(xmm_y0, xmm_y1);
                SIMDFPTYPE_MIC xmm_diff_Z = _MM_SUB_FP_MIC(xmm_z0, xmm_z1);

                SIMDFPTYPE_MIC xmm_norm_2 = _MM_ADD_FP_MIC(_MM_ADD_FP_MIC(_MM_MUL_FP_MIC(xmm_diff_X, xmm_diff_X), _MM_MUL_FP_MIC(xmm_diff_Y, xmm_diff_Y)), _MM_MUL_FP_MIC(xmm_diff_Z, xmm_diff_Z));

                xmm_result1 = _MM_MASK_ADD_INT_MIC(xmm_result1, xmm_one, _MM_CMP_LT_MIC(xmm_norm_2, xmm_separation_element1));
                xmm_result2 = _MM_MASK_ADD_INT_MIC(xmm_result2, xmm_one, _MM_CMP_LT_MIC(xmm_norm_2, xmm_separation_element2));

                /*
                SIMDINTTYPE t = _mm256_castps_si256(_MM_CMP_LT(xmm_norm_2, xmm_separation_element1));
                xmm_result1 = _mm_sub_epi32(xmm_result1, _mm256_castsi256_si128(t));
                xmm_result1 = _mm_sub_epi32(xmm_result1, _mm256_extractf128_si256(t, 1));
                t = _mm256_castps_si256(_MM_CMP_LT(xmm_norm_2, xmm_separation_element2));
                xmm_result2 = _mm_sub_epi32(xmm_result2, _mm256_castsi256_si128(t));
                xmm_result2 = _mm_sub_epi32(xmm_result2, _mm256_extractf128_si256(t, 1));
                */
            }

            {
                int j = count1_prime;
                int remaining = count1 - count1_prime;
                SIMDMASKTYPE_MIC xmm_anding = (1 << remaining) - 1;

                SIMDFPTYPE_MIC xmm_x1 = _MM_LOADU_FP_MIC(Pos1 + j + 0*count1);
                SIMDFPTYPE_MIC xmm_y1 = _MM_LOADU_FP_MIC(Pos1 + j + 1*count1);
                SIMDFPTYPE_MIC xmm_z1 = _MM_LOADU_FP_MIC(Pos1 + j + 2*count1);

                SIMDFPTYPE_MIC xmm_diff_X = _MM_SUB_FP_MIC(xmm_x0, xmm_x1);
                SIMDFPTYPE_MIC xmm_diff_Y = _MM_SUB_FP_MIC(xmm_y0, xmm_y1);
                SIMDFPTYPE_MIC xmm_diff_Z = _MM_SUB_FP_MIC(xmm_z0, xmm_z1);

                SIMDFPTYPE_MIC xmm_norm_2 = _MM_ADD_FP_MIC(_MM_ADD_FP_MIC(_MM_MUL_FP_MIC(xmm_diff_X, xmm_diff_X), _MM_MUL_FP_MIC(xmm_diff_Y, xmm_diff_Y)), _MM_MUL_FP_MIC(xmm_diff_Z, xmm_diff_Z));
                xmm_result1 = _MM_MASK_ADD_INT_MIC(xmm_result1, xmm_one, _MM_AND_MASK_MIC(xmm_anding, _MM_CMP_LT_MIC(xmm_norm_2, xmm_separation_element1)));
                xmm_result2 = _MM_MASK_ADD_INT_MIC(xmm_result2, xmm_one, _MM_AND_MASK_MIC(xmm_anding, _MM_CMP_LT_MIC(xmm_norm_2, xmm_separation_element2)));

                /*
                SIMDINTTYPE t = _mm256_castps_si256(_mm256_and_ps(xmm_anding, _MM_CMP_LT(xmm_norm_2, xmm_separation_element1)));
                xmm_result1 = _mm_sub_epi32(xmm_result1, _mm256_castsi256_si128(t));
                xmm_result1 = _mm_sub_epi32(xmm_result1, _mm256_extractf128_si256(t, 1));
                t = _mm256_castps_si256(_mm256_and_ps(xmm_anding, _MM_CMP_LT(xmm_norm_2, xmm_separation_element2)));
                xmm_result2 = _mm_sub_epi32(xmm_result2, _mm256_castsi256_si128(t));
                xmm_result2 = _mm_sub_epi32(xmm_result2, _mm256_extractf128_si256(t, 1));
                */
          
            }
        }
    }
    else 
    {
      
        for(int i=0; i<count0; i++)
        {
            SIMDFPTYPE_MIC xmm_x0 = _MM_LOAD1_FP_MIC(Pos0 + i + 0*count0);
            SIMDFPTYPE_MIC xmm_y0 = _MM_LOAD1_FP_MIC(Pos0 + i + 1*count0);
            SIMDFPTYPE_MIC xmm_z0 = _MM_LOAD1_FP_MIC(Pos0 + i + 2*count0);
              
            for(int j=0; j < count1; j+=SIMD_WIDTH_MIC)
            {
                _MM_PREFETCH1_MIC(Pos1 + j + 0*count1 + SIMD_WIDTH_MIC);
                _MM_PREFETCH1_MIC(Pos1 + j + 1*count1 + SIMD_WIDTH_MIC);
                _MM_PREFETCH1_MIC(Pos1 + j + 2*count1 + SIMD_WIDTH_MIC);

                SIMDFPTYPE_MIC xmm_x1 = _MM_LOAD_FP_MIC(Pos1 + j + 0*count1);
                SIMDFPTYPE_MIC xmm_y1 = _MM_LOAD_FP_MIC(Pos1 + j + 1*count1);
                SIMDFPTYPE_MIC xmm_z1 = _MM_LOAD_FP_MIC(Pos1 + j + 2*count1);

                SIMDFPTYPE_MIC xmm_diff_X = _MM_SUB_FP_MIC(xmm_x0, xmm_x1);
                SIMDFPTYPE_MIC xmm_diff_Y = _MM_SUB_FP_MIC(xmm_y0, xmm_y1);
                SIMDFPTYPE_MIC xmm_diff_Z = _MM_SUB_FP_MIC(xmm_z0, xmm_z1);

                SIMDFPTYPE_MIC xmm_norm_2 = _MM_ADD_FP_MIC(_MM_ADD_FP_MIC(_MM_MUL_FP_MIC(xmm_diff_X, xmm_diff_X), _MM_MUL_FP_MIC(xmm_diff_Y, xmm_diff_Y)), _MM_MUL_FP_MIC(xmm_diff_Z, xmm_diff_Z));
                xmm_result1 = _MM_MASK_ADD_INT_MIC(xmm_result1, xmm_one, _MM_CMP_LT_MIC(xmm_norm_2, xmm_separation_element1));
                xmm_result2 = _MM_MASK_ADD_INT_MIC(xmm_result2, xmm_one, _MM_CMP_LT_MIC(xmm_norm_2, xmm_separation_element2));

                /*
                SIMDINTTYPE t = _mm256_castps_si256(_MM_CMP_LT(xmm_norm_2, xmm_separation_element1));
                xmm_result1 = _mm_sub_epi32(xmm_result1, _mm256_castsi256_si128(t));
                xmm_result1 = _mm_sub_epi32(xmm_result1, _mm256_extractf128_si256(t, 1));
                t = _mm256_castps_si256(_MM_CMP_LT(xmm_norm_2, xmm_separation_element2));
                xmm_result2 = _mm_sub_epi32(xmm_result2, _mm256_castsi256_si128(t));
                xmm_result2 = _mm_sub_epi32(xmm_result2, _mm256_extractf128_si256(t, 1));
                */
            }
        }
    }

    {
        int total =  (self) ? ((count0*(count1-1))/2) : (count0 * count1);
        /*
        __declspec (align(64)) int Temp[8];
        _MM_STORE_INT(Temp+0, xmm_result1);
        _MM_STORE_INT(Temp+4, xmm_result2);
            
        int sum1 = Temp[0] + Temp[1] + Temp[2] + Temp[3];
        int sum2 = Temp[4] + Temp[5] + Temp[6] + Temp[7];
        */
        int sum1 = _MM_HADD_MIC(xmm_result1);
        int sum2 = _MM_HADD_MIC(xmm_result2);

        DD_int0[min_bin_id+0] += sum1;
        DD_int0[min_bin_id+1] += sum2-sum1;
        DD_int0[min_bin_id+2] += (total - sum2);
    }

#endif
}



__attribute__ (( target (mic))) 
void MICFunction_Compute_Distance_And_Populate_Hist_2(TYPE *Pos0, int count0, TYPE *Pos1, int count1, TYPE *BinCorners2, int nrbin, 
        unsigned int *DD_int0, unsigned int *DD_int1, int min_bin_id, int max_bin_id, TYPE *Range1_X, TYPE *Range1_Y, TYPE *Range1_Z, int threadid)
{
#ifdef __MIC__

    int   c_one_MIC  = 1;
    SIMDINTTYPE_MIC xmm_one =  _MM_SET1_INT_MIC(c_one_MIC);
    //Basically max_bin_id == min_bin_id + 1...


    TYPE separation_element = BinCorners2[min_bin_id+1];
    SIMDFPTYPE_MIC xmm_separation_element = _MM_SET1_FP_MIC(separation_element);

    int self = 0;
    if (Pos0 == Pos1) self = 1;

    //if (self == 1) ERROR_PRINT();
        

    TYPE Range0_X[2], Range0_Y[2], Range0_Z[2];

    int to_be_subtracted = 0;

    if (self)
    {
        //printf("fdsf\n");
        unsigned int something_to_check0 = DD_int0[min_bin_id + 2];
        unsigned int something_to_check1 = DD_int0[min_bin_id + 2];
        MICFunction_Compute_Distance_And_Populate_Hist_3(Pos0, count0, Pos1, count1, BinCorners2, nrbin, DD_int0, DD_int1, min_bin_id, max_bin_id+1, Range1_X, Range1_Y, Range1_Z, threadid);
        if (something_to_check0 != DD_int0[min_bin_id + 2]) ERROR_PRINT();
        if (something_to_check1 != DD_int0[min_bin_id + 2]) ERROR_PRINT();
    }

    else
    {
        SIMDINTTYPE_MIC xmm_result_0 = _MM_SET1_INT_MIC(0);
        SIMDINTTYPE_MIC xmm_result_1 = _MM_SET1_INT_MIC(0);
        SIMDINTTYPE_MIC xmm_result_2 = _MM_SET1_INT_MIC(0);
        SIMDINTTYPE_MIC xmm_result_3 = _MM_SET1_INT_MIC(0);

        int count0_prime = ((count0>>2)<<2);
        for(int i=0; i<count0_prime; i += 4)
        {
            SIMDFPTYPE_MIC xmm_x0_0 = _MM_LOAD1_FP_MIC(Pos0 + i + 0 + 0*count0);
            SIMDFPTYPE_MIC xmm_y0_0 = _MM_LOAD1_FP_MIC(Pos0 + i + 0 + 1*count0);
            SIMDFPTYPE_MIC xmm_z0_0 = _MM_LOAD1_FP_MIC(Pos0 + i + 0 + 2*count0);

            SIMDFPTYPE_MIC xmm_x0_1 = _MM_LOAD1_FP_MIC(Pos0 + i + 1 + 0*count0);
            SIMDFPTYPE_MIC xmm_y0_1 = _MM_LOAD1_FP_MIC(Pos0 + i + 1 + 1*count0);
            SIMDFPTYPE_MIC xmm_z0_1 = _MM_LOAD1_FP_MIC(Pos0 + i + 1 + 2*count0);

            SIMDFPTYPE_MIC xmm_x0_2 = _MM_LOAD1_FP_MIC(Pos0 + i + 2 + 0*count0);
            SIMDFPTYPE_MIC xmm_y0_2 = _MM_LOAD1_FP_MIC(Pos0 + i + 2 + 1*count0);
            SIMDFPTYPE_MIC xmm_z0_2 = _MM_LOAD1_FP_MIC(Pos0 + i + 2 + 2*count0);

            SIMDFPTYPE_MIC xmm_x0_3 = _MM_LOAD1_FP_MIC(Pos0 + i + 3 + 0*count0);
            SIMDFPTYPE_MIC xmm_y0_3 = _MM_LOAD1_FP_MIC(Pos0 + i + 3 + 1*count0);
            SIMDFPTYPE_MIC xmm_z0_3 = _MM_LOAD1_FP_MIC(Pos0 + i + 3 + 2*count0);

            for(int j=0; j < count1; j+=SIMD_WIDTH_MIC)
            {
                _MM_PREFETCH1_MIC(Pos1 + j + 0*count1 + SIMD_WIDTH_MIC);
                _MM_PREFETCH1_MIC(Pos1 + j + 1*count1 + SIMD_WIDTH_MIC);
                _MM_PREFETCH1_MIC(Pos1 + j + 2*count1 + SIMD_WIDTH_MIC);

                SIMDFPTYPE_MIC xmm_x1 = _MM_LOAD_FP_MIC(Pos1 + j + 0*count1);
                SIMDFPTYPE_MIC xmm_y1 = _MM_LOAD_FP_MIC(Pos1 + j + 1*count1);
                SIMDFPTYPE_MIC xmm_z1 = _MM_LOAD_FP_MIC(Pos1 + j + 2*count1);

                SIMDFPTYPE_MIC xmm_diff_X_0 = _MM_SUB_FP_MIC(xmm_x0_0, xmm_x1);
                SIMDFPTYPE_MIC xmm_diff_Y_0 = _MM_SUB_FP_MIC(xmm_y0_0, xmm_y1);
                SIMDFPTYPE_MIC xmm_diff_Z_0 = _MM_SUB_FP_MIC(xmm_z0_0, xmm_z1);

                SIMDFPTYPE_MIC xmm_diff_X_1 = _MM_SUB_FP_MIC(xmm_x0_1, xmm_x1);
                SIMDFPTYPE_MIC xmm_diff_Y_1 = _MM_SUB_FP_MIC(xmm_y0_1, xmm_y1);
                SIMDFPTYPE_MIC xmm_diff_Z_1 = _MM_SUB_FP_MIC(xmm_z0_1, xmm_z1);

                SIMDFPTYPE_MIC xmm_diff_X_2 = _MM_SUB_FP_MIC(xmm_x0_2, xmm_x1);
                SIMDFPTYPE_MIC xmm_diff_Y_2 = _MM_SUB_FP_MIC(xmm_y0_2, xmm_y1);
                SIMDFPTYPE_MIC xmm_diff_Z_2 = _MM_SUB_FP_MIC(xmm_z0_2, xmm_z1);

                SIMDFPTYPE_MIC xmm_diff_X_3 = _MM_SUB_FP_MIC(xmm_x0_3, xmm_x1);
                SIMDFPTYPE_MIC xmm_diff_Y_3 = _MM_SUB_FP_MIC(xmm_y0_3, xmm_y1);
                SIMDFPTYPE_MIC xmm_diff_Z_3 = _MM_SUB_FP_MIC(xmm_z0_3, xmm_z1);

                SIMDFPTYPE_MIC xmm_norm_2_0 = _MM_ADD_FP_MIC(_MM_ADD_FP_MIC(_MM_MUL_FP_MIC(xmm_diff_X_0, xmm_diff_X_0), _MM_MUL_FP_MIC(xmm_diff_Y_0, xmm_diff_Y_0)), _MM_MUL_FP_MIC(xmm_diff_Z_0, xmm_diff_Z_0));
                SIMDFPTYPE_MIC xmm_norm_2_1 = _MM_ADD_FP_MIC(_MM_ADD_FP_MIC(_MM_MUL_FP_MIC(xmm_diff_X_1, xmm_diff_X_1), _MM_MUL_FP_MIC(xmm_diff_Y_1, xmm_diff_Y_1)), _MM_MUL_FP_MIC(xmm_diff_Z_1, xmm_diff_Z_1));
                SIMDFPTYPE_MIC xmm_norm_2_2 = _MM_ADD_FP_MIC(_MM_ADD_FP_MIC(_MM_MUL_FP_MIC(xmm_diff_X_2, xmm_diff_X_2), _MM_MUL_FP_MIC(xmm_diff_Y_2, xmm_diff_Y_2)), _MM_MUL_FP_MIC(xmm_diff_Z_2, xmm_diff_Z_2));
                SIMDFPTYPE_MIC xmm_norm_2_3 = _MM_ADD_FP_MIC(_MM_ADD_FP_MIC(_MM_MUL_FP_MIC(xmm_diff_X_3, xmm_diff_X_3), _MM_MUL_FP_MIC(xmm_diff_Y_3, xmm_diff_Y_3)), _MM_MUL_FP_MIC(xmm_diff_Z_3, xmm_diff_Z_3));

                xmm_result_0 = _MM_MASK_ADD_INT_MIC(xmm_result_0, xmm_one, _MM_CMP_LT_MIC(xmm_norm_2_0, xmm_separation_element));
                xmm_result_1 = _MM_MASK_ADD_INT_MIC(xmm_result_1, xmm_one, _MM_CMP_LT_MIC(xmm_norm_2_1, xmm_separation_element));
                xmm_result_2 = _MM_MASK_ADD_INT_MIC(xmm_result_2, xmm_one, _MM_CMP_LT_MIC(xmm_norm_2_2, xmm_separation_element));
                xmm_result_3 = _MM_MASK_ADD_INT_MIC(xmm_result_3, xmm_one, _MM_CMP_LT_MIC(xmm_norm_2_3, xmm_separation_element));

                /*
                SIMDINTTYPE_MIC t = _mm256_castps_si256(_MM_CMP_LT_MIC(xmm_norm_2_0, xmm_separation_element));
                xmm_result_0 = _mm_sub_epi32(xmm_result_0, _mm256_castsi256_si128(t));
                xmm_result_0 = _mm_sub_epi32(xmm_result_0, _mm256_extractf128_si256(t, 1));
                */
            }
        }

        for(int i=count0_prime; i<count0; i++)
        {
            SIMDFPTYPE_MIC xmm_x0_0 = _MM_LOAD1_FP_MIC(Pos0 + i + 0 + 0*count0);
            SIMDFPTYPE_MIC xmm_y0_0 = _MM_LOAD1_FP_MIC(Pos0 + i + 0 + 1*count0);
            SIMDFPTYPE_MIC xmm_z0_0 = _MM_LOAD1_FP_MIC(Pos0 + i + 0 + 2*count0);

            for(int j=0; j < count1; j+=SIMD_WIDTH_MIC)
            {
                _MM_PREFETCH1_MIC(Pos1 + j + 0*count1 + SIMD_WIDTH_MIC);
                _MM_PREFETCH1_MIC(Pos1 + j + 1*count1 + SIMD_WIDTH_MIC);
                _MM_PREFETCH1_MIC(Pos1 + j + 2*count1 + SIMD_WIDTH_MIC);

                SIMDFPTYPE_MIC xmm_x1 = _MM_LOAD_FP_MIC(Pos1 + j + 0*count1);
                SIMDFPTYPE_MIC xmm_y1 = _MM_LOAD_FP_MIC(Pos1 + j + 1*count1);
                SIMDFPTYPE_MIC xmm_z1 = _MM_LOAD_FP_MIC(Pos1 + j + 2*count1);

                SIMDFPTYPE_MIC xmm_diff_X_0 = _MM_SUB_FP_MIC(xmm_x0_0, xmm_x1);
                SIMDFPTYPE_MIC xmm_diff_Y_0 = _MM_SUB_FP_MIC(xmm_y0_0, xmm_y1);
                SIMDFPTYPE_MIC xmm_diff_Z_0 = _MM_SUB_FP_MIC(xmm_z0_0, xmm_z1);

                SIMDFPTYPE_MIC xmm_norm_2_0 = _MM_ADD_FP_MIC(_MM_ADD_FP_MIC(_MM_MUL_FP_MIC(xmm_diff_X_0, xmm_diff_X_0), _MM_MUL_FP_MIC(xmm_diff_Y_0, xmm_diff_Y_0)), _MM_MUL_FP_MIC(xmm_diff_Z_0, xmm_diff_Z_0));
                xmm_result_0 = _MM_MASK_ADD_INT_MIC(xmm_result_0, xmm_one, _MM_CMP_LT_MIC(xmm_norm_2_0, xmm_separation_element));

                /*
                SIMDINTTYPE_MIC t = _mm256_castps_si256(_MM_CMP_LT_MIC(xmm_norm_2_0, xmm_separation_element));
                xmm_result_0 = _mm_sub_epi32(xmm_result_0, _mm256_castsi256_si128(t));
                xmm_result_0 = _mm_sub_epi32(xmm_result_0, _mm256_extractf128_si256(t, 1));
                */
            }
        }


        int total =  (self) ? ((count0*(count1-1))/2) : (count0 * count1);
        /*
        __m128i xmm_result = xmm_result_0;
        __declspec (align(64)) int Temp[4];
        _MM_STORE_INT(Temp, xmm_result);
        int sum = Temp[0] + Temp[1] + Temp[2] + Temp[3];
        */
        xmm_result_0 = _MM_ADD_INT_MIC(xmm_result_0, _MM_ADD_INT_MIC(xmm_result_1, _MM_ADD_INT_MIC(xmm_result_2, xmm_result_3)));
        int sum = _MM_HADD_MIC(xmm_result_0);

        total -= to_be_subtracted;

        DD_int0[min_bin_id] += sum;
        DD_int0[min_bin_id+1] += (total - sum);
    }
#endif
}

__attribute__ (( target (mic))) 
void MICFunction_Compute_Distance_And_Populate_Hist_4(TYPE *Pos0, int count0, TYPE *Pos1, int count1, TYPE *BinCorners2, int nrbin, 
        unsigned int *DD_int0, unsigned int *DD_int1, int min_bin_id, int max_bin_id, TYPE *Range1_X, TYPE *Range1_Y, TYPE *Range1_Z, int threadid)
{
#ifdef __MIC__

    int   c_one_MIC  = 1;
    SIMDINTTYPE_MIC xmm_one =  _MM_SET1_INT_MIC(c_one_MIC);

    TYPE separation_element1 = BinCorners2[min_bin_id+1];
    TYPE separation_element2 = BinCorners2[min_bin_id+2];
    TYPE separation_element3 = BinCorners2[min_bin_id+3];

    SIMDFPTYPE_MIC xmm_separation_element1 = _MM_SET1_FP_MIC(separation_element1);
    SIMDFPTYPE_MIC xmm_separation_element2 = _MM_SET1_FP_MIC(separation_element2);
    SIMDFPTYPE_MIC xmm_separation_element3 = _MM_SET1_FP_MIC(separation_element3);

    SIMDINTTYPE_MIC xmm_result1 = _MM_SET1_INT_MIC(0);
    SIMDINTTYPE_MIC xmm_result2 = _MM_SET1_INT_MIC(0);
    SIMDINTTYPE_MIC xmm_result3 = _MM_SET1_INT_MIC(0);

    int self = 0;
    if (Pos0 == Pos1) self = 1;

    if (self) 
    {
        for(int i=0; i<count0; i++)
        {
            int starting_index = i+1;
              
            SIMDFPTYPE_MIC xmm_x0 = _MM_LOAD1_FP_MIC(Pos0 + i + 0*count0);
            SIMDFPTYPE_MIC xmm_y0 = _MM_LOAD1_FP_MIC(Pos0 + i + 1*count0);
            SIMDFPTYPE_MIC xmm_z0 = _MM_LOAD1_FP_MIC(Pos0 + i + 2*count0);
              
            int particles_left = count1 - starting_index;
            int particles_left_prime = ((particles_left >> SIMD_WIDTH_LOG2_MIC) << SIMD_WIDTH_LOG2_MIC);
            int count1_prime = starting_index + particles_left_prime;

            for(int j=starting_index; j < count1_prime; j+=SIMD_WIDTH_MIC)
            {
                _MM_PREFETCH1_MIC(Pos1 + j + 0*count1 + SIMD_WIDTH_MIC);
                _MM_PREFETCH1_MIC(Pos1 + j + 1*count1 + SIMD_WIDTH_MIC);
                _MM_PREFETCH1_MIC(Pos1 + j + 2*count1 + SIMD_WIDTH_MIC);

                SIMDFPTYPE_MIC xmm_x1 = _MM_LOADU_FP_MIC(Pos1 + j + 0*count1);
                SIMDFPTYPE_MIC xmm_y1 = _MM_LOADU_FP_MIC(Pos1 + j + 1*count1);
                SIMDFPTYPE_MIC xmm_z1 = _MM_LOADU_FP_MIC(Pos1 + j + 2*count1);

                SIMDFPTYPE_MIC xmm_diff_X = _MM_SUB_FP_MIC(xmm_x0, xmm_x1);
                SIMDFPTYPE_MIC xmm_diff_Y = _MM_SUB_FP_MIC(xmm_y0, xmm_y1);
                SIMDFPTYPE_MIC xmm_diff_Z = _MM_SUB_FP_MIC(xmm_z0, xmm_z1);

                SIMDFPTYPE_MIC xmm_norm_2 = _MM_ADD_FP_MIC(_MM_ADD_FP_MIC(_MM_MUL_FP_MIC(xmm_diff_X, xmm_diff_X), _MM_MUL_FP_MIC(xmm_diff_Y, xmm_diff_Y)), _MM_MUL_FP_MIC(xmm_diff_Z, xmm_diff_Z));

                xmm_result1 = _MM_MASK_ADD_INT_MIC(xmm_result1, xmm_one, _MM_CMP_LT_MIC(xmm_norm_2, xmm_separation_element1));
                xmm_result2 = _MM_MASK_ADD_INT_MIC(xmm_result2, xmm_one, _MM_CMP_LT_MIC(xmm_norm_2, xmm_separation_element2));
                xmm_result3 = _MM_MASK_ADD_INT_MIC(xmm_result3, xmm_one, _MM_CMP_LT_MIC(xmm_norm_2, xmm_separation_element3));
                /*
                SIMDINTTYPE_MIC t = _mm256_castps_si256(_MM_CMP_LT_MIC(xmm_norm_2, xmm_separation_element1));
                xmm_result1 = _mm_sub_epi32(xmm_result1, _mm256_castsi256_si128(t));
                xmm_result1 = _mm_sub_epi32(xmm_result1, _mm256_extractf128_si256(t, 1));
                t = _mm256_castps_si256(_MM_CMP_LT_MIC(xmm_norm_2, xmm_separation_element2));
                xmm_result2 = _mm_sub_epi32(xmm_result2, _mm256_castsi256_si128(t));
                xmm_result2 = _mm_sub_epi32(xmm_result2, _mm256_extractf128_si256(t, 1));
                t = _mm256_castps_si256(_MM_CMP_LT_MIC(xmm_norm_2, xmm_separation_element3));
                xmm_result3 = _mm_sub_epi32(xmm_result3, _mm256_castsi256_si128(t));
                xmm_result3 = _mm_sub_epi32(xmm_result3, _mm256_extractf128_si256(t, 1));
                */
            }

            {
              
                int j = count1_prime;
                int remaining = count1 - count1_prime;
                SIMDMASKTYPE_MIC xmm_anding = (1 << remaining) - 1;

                SIMDFPTYPE_MIC xmm_x1 = _MM_LOADU_FP_MIC(Pos1 + j + 0*count1);
                SIMDFPTYPE_MIC xmm_y1 = _MM_LOADU_FP_MIC(Pos1 + j + 1*count1);
                SIMDFPTYPE_MIC xmm_z1 = _MM_LOADU_FP_MIC(Pos1 + j + 2*count1);

                SIMDFPTYPE_MIC xmm_diff_X = _MM_SUB_FP_MIC(xmm_x0, xmm_x1);
                SIMDFPTYPE_MIC xmm_diff_Y = _MM_SUB_FP_MIC(xmm_y0, xmm_y1);
                SIMDFPTYPE_MIC xmm_diff_Z = _MM_SUB_FP_MIC(xmm_z0, xmm_z1);

                SIMDFPTYPE_MIC xmm_norm_2 = _MM_ADD_FP_MIC(_MM_ADD_FP_MIC(_MM_MUL_FP_MIC(xmm_diff_X, xmm_diff_X), _MM_MUL_FP_MIC(xmm_diff_Y, xmm_diff_Y)), _MM_MUL_FP_MIC(xmm_diff_Z, xmm_diff_Z));

                xmm_result1 = _MM_MASK_ADD_INT_MIC(xmm_result1, xmm_one, _MM_AND_MASK_MIC(xmm_anding, _MM_CMP_LT_MIC(xmm_norm_2, xmm_separation_element1)));
                xmm_result2 = _MM_MASK_ADD_INT_MIC(xmm_result2, xmm_one, _MM_AND_MASK_MIC(xmm_anding, _MM_CMP_LT_MIC(xmm_norm_2, xmm_separation_element2)));
                xmm_result3 = _MM_MASK_ADD_INT_MIC(xmm_result3, xmm_one, _MM_AND_MASK_MIC(xmm_anding, _MM_CMP_LT_MIC(xmm_norm_2, xmm_separation_element3)));
                /*
                SIMDINTTYPE_MIC t = _mm256_castps_si256(_mm256_and_ps(xmm_anding, _MM_CMP_LT_MIC(xmm_norm_2, xmm_separation_element1)));
                xmm_result1 = _mm_sub_epi32(xmm_result1, _mm256_castsi256_si128(t));
                xmm_result1 = _mm_sub_epi32(xmm_result1, _mm256_extractf128_si256(t, 1));
                t = _mm256_castps_si256(_mm256_and_ps(xmm_anding, _MM_CMP_LT_MIC(xmm_norm_2, xmm_separation_element2)));
                xmm_result2 = _mm_sub_epi32(xmm_result2, _mm256_castsi256_si128(t));
                xmm_result2 = _mm_sub_epi32(xmm_result2, _mm256_extractf128_si256(t, 1));
                t = _mm256_castps_si256(_mm256_and_ps(xmm_anding, _MM_CMP_LT_MIC(xmm_norm_2, xmm_separation_element3)));
                xmm_result3 = _mm_sub_epi32(xmm_result3, _mm256_castsi256_si128(t));
                xmm_result3 = _mm_sub_epi32(xmm_result3, _mm256_extractf128_si256(t, 1));
                */
            }
        }
    }
    else 
    {
      
        for(int i=0; i<count0; i++)
        {
            SIMDFPTYPE_MIC xmm_x0 = _MM_LOAD1_FP_MIC(Pos0 + i + 0*count0);
            SIMDFPTYPE_MIC xmm_y0 = _MM_LOAD1_FP_MIC(Pos0 + i + 1*count0);
            SIMDFPTYPE_MIC xmm_z0 = _MM_LOAD1_FP_MIC(Pos0 + i + 2*count0);
              
            for(int j=0; j < count1; j+=SIMD_WIDTH_MIC)
            {
                _MM_PREFETCH1_MIC(Pos1 + j + 0*count1 + SIMD_WIDTH_MIC);
                _MM_PREFETCH1_MIC(Pos1 + j + 1*count1 + SIMD_WIDTH_MIC);
                _MM_PREFETCH1_MIC(Pos1 + j + 2*count1 + SIMD_WIDTH_MIC);

                SIMDFPTYPE_MIC xmm_x1 = _MM_LOAD_FP_MIC(Pos1 + j + 0*count1);
                SIMDFPTYPE_MIC xmm_y1 = _MM_LOAD_FP_MIC(Pos1 + j + 1*count1);
                SIMDFPTYPE_MIC xmm_z1 = _MM_LOAD_FP_MIC(Pos1 + j + 2*count1);

                SIMDFPTYPE_MIC xmm_diff_X = _MM_SUB_FP_MIC(xmm_x0, xmm_x1);
                SIMDFPTYPE_MIC xmm_diff_Y = _MM_SUB_FP_MIC(xmm_y0, xmm_y1);
                SIMDFPTYPE_MIC xmm_diff_Z = _MM_SUB_FP_MIC(xmm_z0, xmm_z1);

                SIMDFPTYPE_MIC xmm_norm_2 = _MM_ADD_FP_MIC(_MM_ADD_FP_MIC(_MM_MUL_FP_MIC(xmm_diff_X, xmm_diff_X), _MM_MUL_FP_MIC(xmm_diff_Y, xmm_diff_Y)), _MM_MUL_FP_MIC(xmm_diff_Z, xmm_diff_Z));

                xmm_result1 = _MM_MASK_ADD_INT_MIC(xmm_result1, xmm_one, _MM_CMP_LT_MIC(xmm_norm_2, xmm_separation_element1));
                xmm_result2 = _MM_MASK_ADD_INT_MIC(xmm_result2, xmm_one, _MM_CMP_LT_MIC(xmm_norm_2, xmm_separation_element2));
                xmm_result3 = _MM_MASK_ADD_INT_MIC(xmm_result3, xmm_one, _MM_CMP_LT_MIC(xmm_norm_2, xmm_separation_element3));
                /*
                SIMDINTTYPE_MIC t = _mm256_castps_si256(_MM_CMP_LT_MIC(xmm_norm_2, xmm_separation_element1));
                xmm_result1 = _mm_sub_epi32(xmm_result1, _mm256_castsi256_si128(t));
                xmm_result1 = _mm_sub_epi32(xmm_result1, _mm256_extractf128_si256(t, 1));
                t = _mm256_castps_si256(_MM_CMP_LT_MIC(xmm_norm_2, xmm_separation_element2));
                xmm_result2 = _mm_sub_epi32(xmm_result2, _mm256_castsi256_si128(t));
                xmm_result2 = _mm_sub_epi32(xmm_result2, _mm256_extractf128_si256(t, 1));
                t = _mm256_castps_si256(_MM_CMP_LT_MIC(xmm_norm_2, xmm_separation_element3));
                xmm_result3 = _mm_sub_epi32(xmm_result3, _mm256_castsi256_si128(t));
                xmm_result3 = _mm_sub_epi32(xmm_result3, _mm256_extractf128_si256(t, 1));
                */
            }
        }
    }


    {
        int total =  (self) ? ((count0*(count1-1))/2) : (count0 * count1);
        /*
        __declspec (align(64)) int Temp[12];
        _MM_STORE_INT(Temp+0, xmm_result1);
        _MM_STORE_INT(Temp+4, xmm_result2);
        _MM_STORE_INT(Temp+8, xmm_result3);
    
            
        int sum1 = Temp[0] + Temp[1] + Temp[2] + Temp[3];
        int sum2 = Temp[4] + Temp[5] + Temp[6] + Temp[7];
        int sum3 = Temp[8] + Temp[9] + Temp[10] + Temp[11];
        */

        int sum1 = _MM_HADD_MIC(xmm_result1);
        int sum2 = _MM_HADD_MIC(xmm_result2);
        int sum3 = _MM_HADD_MIC(xmm_result3);

        DD_int0[min_bin_id+0] += sum1;
        DD_int0[min_bin_id+1] += sum2-sum1;
        DD_int0[min_bin_id+2] += sum3-sum2;
        DD_int0[min_bin_id+3] += (total - sum3);
    }
#endif
}

__attribute__ (( target (mic))) 
void MICFunction_Compute_Distance_And_Populate_Hist_N(TYPE *Pos0, int count0, TYPE *Pos1, int count1, TYPE *BinCorners2, int nrbin, 
        unsigned int *DD_int0, unsigned int *DD_int1, int min_bin_id, int max_bin_id, TYPE *Range1_X, TYPE *Range1_Y, TYPE *Range1_Z, int threadid)
{
#ifdef __MIC__

    int   c_one_MIC  = 1;
    SIMDINTTYPE_MIC xmm_one =  _MM_SET1_INT_MIC(c_one_MIC);

    int self = 0;
    if (Pos0 == Pos1) self = 1;

    //SCompute_Distance_And_Populate_Hist_N(Pos0, count0, Pos1, count1, DD_int0, DD_int1, nrbin, BinCorners2, min_bin_id, max_bin_id, case_considered); return;

    //printf("min_bin_id = %d ::: max_bin_id = %d\n", min_bin_id, max_bin_id);
    TYPE separation_element[HIST_BINS];
    SIMDFPTYPE_MIC xmm_separation_element[HIST_BINS];
    SIMDINTTYPE_MIC xmm_result[HIST_BINS];

    for(int k=min_bin_id; k<max_bin_id; k++)     separation_element[k-min_bin_id] = BinCorners2[k+1];
    for(int k=min_bin_id; k<max_bin_id; k++) xmm_separation_element[k-min_bin_id] = _MM_SET1_FP_MIC(separation_element[k-min_bin_id]);
    for(int k=min_bin_id; k<max_bin_id; k++) xmm_result[k-min_bin_id] = _MM_SET1_INT_MIC(0);

    int number_of_bins_minus_one = (max_bin_id - min_bin_id);

    if (self) 
    {
        for(int i=0; i<count0; i++)
        {
            int starting_index = i+1;
              
            SIMDFPTYPE_MIC xmm_x0 = _MM_LOAD1_FP_MIC(Pos0 + i + 0*count0);
            SIMDFPTYPE_MIC xmm_y0 = _MM_LOAD1_FP_MIC(Pos0 + i + 1*count0);
            SIMDFPTYPE_MIC xmm_z0 = _MM_LOAD1_FP_MIC(Pos0 + i + 2*count0);
              
            int particles_left = count1 - starting_index;
            int particles_left_prime = ((particles_left >> SIMD_WIDTH_LOG2_MIC) << SIMD_WIDTH_LOG2_MIC);
            int count1_prime = starting_index + particles_left_prime;

          
            for(int j=starting_index; j < count1_prime; j+=SIMD_WIDTH_MIC)
            {
                _MM_PREFETCH1_MIC(Pos1 + j + 0*count1 + SIMD_WIDTH_MIC);
                _MM_PREFETCH1_MIC(Pos1 + j + 1*count1 + SIMD_WIDTH_MIC);
                _MM_PREFETCH1_MIC(Pos1 + j + 2*count1 + SIMD_WIDTH_MIC);

                SIMDFPTYPE_MIC xmm_x1 = _MM_LOADU_FP_MIC(Pos1 + j + 0*count1);
                SIMDFPTYPE_MIC xmm_y1 = _MM_LOADU_FP_MIC(Pos1 + j + 1*count1);
                SIMDFPTYPE_MIC xmm_z1 = _MM_LOADU_FP_MIC(Pos1 + j + 2*count1);

                SIMDFPTYPE_MIC xmm_diff_X = _MM_SUB_FP_MIC(xmm_x0, xmm_x1);
                SIMDFPTYPE_MIC xmm_diff_Y = _MM_SUB_FP_MIC(xmm_y0, xmm_y1);
                SIMDFPTYPE_MIC xmm_diff_Z = _MM_SUB_FP_MIC(xmm_z0, xmm_z1);

                SIMDFPTYPE_MIC xmm_norm_2 = _MM_ADD_FP_MIC(_MM_ADD_FP_MIC(_MM_MUL_FP_MIC(xmm_diff_X, xmm_diff_X), _MM_MUL_FP_MIC(xmm_diff_Y, xmm_diff_Y)), _MM_MUL_FP_MIC(xmm_diff_Z, xmm_diff_Z));

              
                for(int k=0; k<number_of_bins_minus_one; k++) 
                {
                    xmm_result[k] = _MM_MASK_ADD_INT_MIC(xmm_result[k], xmm_one, _MM_CMP_LT_MIC(xmm_norm_2, xmm_separation_element[k]));
                    /*
                    SIMDINTTYPE_MIC t = _mm256_castps_si256(_MM_CMP_LT_MIC(xmm_norm_2, xmm_separation_element[k]));
                    xmm_result[k] = _mm_sub_epi32(xmm_result[k], _mm256_castsi256_si128(t));
                    xmm_result[k] = _mm_sub_epi32(xmm_result[k], _mm256_extractf128_si256(t, 1));
                    */
                }
            }

            {
              
                int j = count1_prime;
                int remaining = count1 - count1_prime;
                SIMDMASKTYPE_MIC xmm_anding = (1 << remaining) - 1;

                SIMDFPTYPE_MIC xmm_x1 = _MM_LOADU_FP_MIC(Pos1 + j + 0*count1);
                SIMDFPTYPE_MIC xmm_y1 = _MM_LOADU_FP_MIC(Pos1 + j + 1*count1);
                SIMDFPTYPE_MIC xmm_z1 = _MM_LOADU_FP_MIC(Pos1 + j + 2*count1);

                SIMDFPTYPE_MIC xmm_diff_X = _MM_SUB_FP_MIC(xmm_x0, xmm_x1);
                SIMDFPTYPE_MIC xmm_diff_Y = _MM_SUB_FP_MIC(xmm_y0, xmm_y1);
                SIMDFPTYPE_MIC xmm_diff_Z = _MM_SUB_FP_MIC(xmm_z0, xmm_z1);

                SIMDFPTYPE_MIC xmm_norm_2 = _MM_ADD_FP_MIC(_MM_ADD_FP_MIC(_MM_MUL_FP_MIC(xmm_diff_X, xmm_diff_X), _MM_MUL_FP_MIC(xmm_diff_Y, xmm_diff_Y)), _MM_MUL_FP_MIC(xmm_diff_Z, xmm_diff_Z));

                for(int k=0; k<number_of_bins_minus_one; k++) 
                {
                    xmm_result[k] = _MM_MASK_ADD_INT_MIC(xmm_result[k], xmm_one, _MM_AND_MASK_MIC(xmm_anding, _MM_CMP_LT_MIC(xmm_norm_2, xmm_separation_element[k])));
                    /*
                    SIMDINTTYPE_MIC t = _mm256_castps_si256(_mm256_and_ps(xmm_anding, _MM_CMP_LT_MIC(xmm_norm_2, xmm_separation_element[k])));
                    xmm_result[k] = _mm_sub_epi32(xmm_result[k], _mm256_castsi256_si128(t));
                    xmm_result[k] = _mm_sub_epi32(xmm_result[k], _mm256_extractf128_si256(t, 1));
                    */
                }
            }
        }
    }
    else 
    {
        for(int i=0; i<count0; i++) 
        {
            SIMDFPTYPE_MIC xmm_x0 = _MM_LOAD1_FP_MIC(Pos0 + i + 0*count0);
            SIMDFPTYPE_MIC xmm_y0 = _MM_LOAD1_FP_MIC(Pos0 + i + 1*count0);
            SIMDFPTYPE_MIC xmm_z0 = _MM_LOAD1_FP_MIC(Pos0 + i + 2*count0);

            for(int j=0; j < count1; j+=SIMD_WIDTH_MIC)
            {
                _MM_PREFETCH1_MIC(Pos1 + j + 0*count1 + SIMD_WIDTH_MIC);
                _MM_PREFETCH1_MIC(Pos1 + j + 1*count1 + SIMD_WIDTH_MIC);
                _MM_PREFETCH1_MIC(Pos1 + j + 2*count1 + SIMD_WIDTH_MIC);

                SIMDFPTYPE_MIC xmm_x1 = _MM_LOAD_FP_MIC(Pos1 + j + 0*count1);
                SIMDFPTYPE_MIC xmm_y1 = _MM_LOAD_FP_MIC(Pos1 + j + 1*count1);
                SIMDFPTYPE_MIC xmm_z1 = _MM_LOAD_FP_MIC(Pos1 + j + 2*count1);

                SIMDFPTYPE_MIC xmm_diff_X = _MM_SUB_FP_MIC(xmm_x0, xmm_x1);
                SIMDFPTYPE_MIC xmm_diff_Y = _MM_SUB_FP_MIC(xmm_y0, xmm_y1);
                SIMDFPTYPE_MIC xmm_diff_Z = _MM_SUB_FP_MIC(xmm_z0, xmm_z1);

                SIMDFPTYPE_MIC xmm_norm_2 = _MM_ADD_FP_MIC(_MM_ADD_FP_MIC(_MM_MUL_FP_MIC(xmm_diff_X, xmm_diff_X), _MM_MUL_FP_MIC(xmm_diff_Y, xmm_diff_Y)), _MM_MUL_FP_MIC(xmm_diff_Z, xmm_diff_Z));

                for(int k=0; k<number_of_bins_minus_one; k++) 
                {
                    xmm_result[k] = _MM_MASK_ADD_INT_MIC(xmm_result[k], xmm_one, _MM_CMP_LT_MIC(xmm_norm_2, xmm_separation_element[k]));
                    /*
                    SIMDINTTYPE_MIC t = _mm256_castps_si256(_MM_CMP_LT_MIC(xmm_norm_2, xmm_separation_element[k]));
                    xmm_result[k] = _mm_sub_epi32(xmm_result[k], _mm256_castsi256_si128(t));
                    xmm_result[k] = _mm_sub_epi32(xmm_result[k], _mm256_extractf128_si256(t, 1));
                    */
                }
            }
        }
    }

    {
        int total =  (self) ? ((count0*(count1-1))/2) : (count0 * count1);
        int sum[HIST_BINS];

        /*
        __declspec (align(64)) int Temp[4*HIST_BINS];
        for(int k=0; k<number_of_bins_minus_one; k++) _MM_STORE_INT(Temp+4*k, xmm_result[k]);
        for(int k=0; k<number_of_bins_minus_one; k++) sum[k] =  Temp[4*k + 0] + Temp[4*k + 1] + Temp[4*k + 2] + Temp[4*k + 3];
        */
        for(int k=0; k<number_of_bins_minus_one; k++) sum[k] =  _MM_HADD_MIC(xmm_result[k]);
        sum[number_of_bins_minus_one] = total;

        DD_int0[min_bin_id+0] += sum[0];
        for(int k=1; k<=number_of_bins_minus_one; k++) DD_int0[min_bin_id+k] += (sum[k] - sum[k-1]);
    }
#endif
}


void  CPUFunction_Actual_Update_Histogram_Self_Cross(TYPE *Pos0, int count0, TYPE *Pos1, int count1, TYPE *Range0_X, TYPE *Range0_Y, TYPE *Range0_Z, 
        TYPE *Range1_X, TYPE *Range1_Y, TYPE *Range1_Z, TYPE *BinCorners2, 
                    unsigned int *DD_int0, unsigned int *DD_int1, unsigned int *Gather_Histogram0, unsigned int *Gather_Histogram1, int nrbin, int threadid)
{
#ifndef __MIC__

    int self = 0;
    if (Pos0 == Pos1) self = 1;
    if (self) { if (count0 != count1) ERROR_PRINT();}

    TYPE min_dist_sqr, max_dist_sqr;
    int min_bin_id, max_bin_id;

    CPUFunction_Compute_Min_Max_Dist_Sqr(Range0_X, Range0_Y, Range0_Z, Range1_X, Range1_Y, Range1_Z, &min_dist_sqr, &max_dist_sqr);

    CPUFunction_Compute_min_max_bin_id(BinCorners2, nrbin, min_dist_sqr, max_dist_sqr, &min_bin_id, &max_bin_id);

    int number_of_bins = max_bin_id - min_bin_id + 1;

    TYPE *Aligned_Buffer = global_Aligned_Buffer_CPU[threadid];

    TYPE LARGE_ENOUGH = 1000.0 * global_grid_D_CPU.Max[2];

    //number_of_bins = 1;
    switch(number_of_bins)
    {
        case 1:
        {
            //Basically max_bin_id == min_bin_id...
            CPUFunction_Compute_Distance_And_Populate_Hist_1(Pos0, count0, Pos1, count1, BinCorners2, nrbin, DD_int0, DD_int1, min_bin_id, max_bin_id, Range1_X, Range1_Y, Range1_Z, threadid);
            break;
        }

#if (SIMD_WIDTH_CPU == 4)
        case 2:
        {
            unsigned long long int stime = ___rdtsc();
            //Basically there are 2 bins (min_bin_id, min_bin_id+1)...
            CPUFunction_Compute_Distance_And_Populate_Hist_2(Pos0, count0, Pos1, count1, BinCorners2, nrbin, DD_int0, DD_int1, min_bin_id, max_bin_id, Range1_X, Range1_Y, Range1_Z, threadid);
            if (threadid < 4) MT_2_CPU[8*threadid] += ___rdtsc() - stime;
            break;
        }

        case 3:
        {
            unsigned long long int stime = ___rdtsc();
            //Basically there are 3 bins (min_bin_id, min_bin_id+1, min_bin_id+2)...
            CPUFunction_Compute_Distance_And_Populate_Hist_3(Pos0, count0, Pos1, count1, BinCorners2, nrbin, DD_int0, DD_int1, min_bin_id, max_bin_id, Range1_X, Range1_Y, Range1_Z, threadid);
            if (threadid < 4) MT_3_CPU[8*threadid] += ___rdtsc() - stime;
            break;
        }

        case 4:
        {
            unsigned long long int stime = ___rdtsc();
            //Basically there are 4 bins (min_bin_id, min_bin_id+1, min_bin_id+2, min_bin_id+3)...
            CPUFunction_Compute_Distance_And_Populate_Hist_4(Pos0, count0, Pos1, count1, BinCorners2, nrbin, DD_int0, DD_int1, min_bin_id, max_bin_id, Range1_X, Range1_Y, Range1_Z, threadid);
            if (threadid < 4) MT_3_CPU[8*threadid] += ___rdtsc() - stime;
            break;
        }

        default:
        {
            unsigned long long int stime = ___rdtsc();
            CPUFunction_Compute_Distance_And_Populate_Hist_N(Pos0, count0, Pos1, count1, BinCorners2, nrbin, DD_int0, DD_int1, min_bin_id, max_bin_id, Range1_X, Range1_Y, Range1_Z, threadid);
            if (threadid < 4) MT_5_CPU[8*threadid] += ___rdtsc() - stime;
            break;
        }
#endif
#if (SIMD_WIDTH_CPU == 8)
        case 2:
        {
            unsigned long long int stime = ___rdtsc();
            //Basically there are 2 bins (min_bin_id, min_bin_id+1)...
            if (self)
            {
                CPUFunction_Compute_Distance_And_Populate_Hist_2(Pos0, count0, Pos1, count1, BinCorners2, nrbin, DD_int0, DD_int1, min_bin_id, max_bin_id, Range1_X, Range1_Y, Range1_Z, threadid);
            }
            else
            {
                //int count1_prime = 8*((count1 + 7)/8);
                int count1_prime = SIMD_WIDTH_CPU * ((count1 + SIMD_WIDTH_CPU - 1)/SIMD_WIDTH_CPU);
                for(int q=0; q<count1; q++) Aligned_Buffer[0*count1_prime + q] = Pos1[0*count1 + q]; for(int q=count1; q < count1_prime; q++) Aligned_Buffer[0*count1_prime + q] = LARGE_ENOUGH;
                for(int q=0; q<count1; q++) Aligned_Buffer[1*count1_prime + q] = Pos1[1*count1 + q]; for(int q=count1; q < count1_prime; q++) Aligned_Buffer[1*count1_prime + q] = LARGE_ENOUGH;
                for(int q=0; q<count1; q++) Aligned_Buffer[2*count1_prime + q] = Pos1[2*count1 + q]; for(int q=count1; q < count1_prime; q++) Aligned_Buffer[2*count1_prime + q] = LARGE_ENOUGH;
                CPUFunction_Compute_Distance_And_Populate_Hist_2(Pos0, count0, Aligned_Buffer, count1_prime, BinCorners2, nrbin, DD_int0, DD_int1, min_bin_id, max_bin_id, Range1_X, Range1_Y, Range1_Z, threadid);
                DD_int0[max_bin_id] -= (count0 * (count1_prime - count1));
            }
            if (threadid < 4) MT_2_CPU[8*threadid] += ___rdtsc() - stime;
            break;
        }

        case 3:
        {
            unsigned long long int stime = ___rdtsc();
            //Basically there are 3 bins (min_bin_id, min_bin_id+1, min_bin_id+2)...
            if (self)
            {
                CPUFunction_Compute_Distance_And_Populate_Hist_3(Pos0, count0, Pos1, count1, BinCorners2, nrbin, DD_int0, DD_int1, min_bin_id, max_bin_id, Range1_X, Range1_Y, Range1_Z, threadid);
            }
            else
            {
                //int count1_prime = 8*((count1 + 7)/8);
                int count1_prime = SIMD_WIDTH_CPU * ((count1 + SIMD_WIDTH_CPU - 1)/SIMD_WIDTH_CPU);
                for(int q=0; q<count1; q++) Aligned_Buffer[0*count1_prime + q] = Pos1[0*count1 + q]; for(int q=count1; q < count1_prime; q++) Aligned_Buffer[0*count1_prime + q] = LARGE_ENOUGH;
                for(int q=0; q<count1; q++) Aligned_Buffer[1*count1_prime + q] = Pos1[1*count1 + q]; for(int q=count1; q < count1_prime; q++) Aligned_Buffer[1*count1_prime + q] = LARGE_ENOUGH;
                for(int q=0; q<count1; q++) Aligned_Buffer[2*count1_prime + q] = Pos1[2*count1 + q]; for(int q=count1; q < count1_prime; q++) Aligned_Buffer[2*count1_prime + q] = LARGE_ENOUGH;
                CPUFunction_Compute_Distance_And_Populate_Hist_3(Pos0, count0, Aligned_Buffer, count1_prime, BinCorners2, nrbin, DD_int0, DD_int1, min_bin_id, max_bin_id, Range1_X, Range1_Y, Range1_Z, threadid);
                DD_int0[max_bin_id] -= (count0 * (count1_prime - count1));
            }
            if (threadid < 4) MT_3_CPU[8*threadid] += ___rdtsc() - stime;
            break;
        }

        case 4:
        {
            unsigned long long int stime = ___rdtsc();
            //Basically there are 4 bins (min_bin_id, min_bin_id+1, min_bin_id+2, min_bin_id+3)...
            if (self)
            {
                CPUFunction_Compute_Distance_And_Populate_Hist_4(Pos0, count0, Pos1, count1, BinCorners2, nrbin, DD_int0, DD_int1, min_bin_id, max_bin_id, Range1_X, Range1_Y, Range1_Z, threadid);
            }
            else
            {
                //int count1_prime = 8*((count1 + 7)/8);
                int count1_prime = SIMD_WIDTH_CPU * ((count1 + SIMD_WIDTH_CPU - 1)/SIMD_WIDTH_CPU);
                for(int q=0; q<count1; q++) Aligned_Buffer[0*count1_prime + q] = Pos1[0*count1 + q]; for(int q=count1; q < count1_prime; q++) Aligned_Buffer[0*count1_prime + q] = LARGE_ENOUGH;
                for(int q=0; q<count1; q++) Aligned_Buffer[1*count1_prime + q] = Pos1[1*count1 + q]; for(int q=count1; q < count1_prime; q++) Aligned_Buffer[1*count1_prime + q] = LARGE_ENOUGH;
                for(int q=0; q<count1; q++) Aligned_Buffer[2*count1_prime + q] = Pos1[2*count1 + q]; for(int q=count1; q < count1_prime; q++) Aligned_Buffer[2*count1_prime + q] = LARGE_ENOUGH;
                CPUFunction_Compute_Distance_And_Populate_Hist_4(Pos0, count0, Aligned_Buffer, count1_prime, BinCorners2, nrbin, DD_int0, DD_int1, min_bin_id, max_bin_id, Range1_X, Range1_Y, Range1_Z, threadid);
                DD_int0[max_bin_id] -= (count0 * (count1_prime - count1));
            }
            if (threadid < 4) MT_4_CPU[8*threadid] += ___rdtsc() - stime;
            break;
        }

        default:
        {
            //int count1_prime = 8*((count1 + 7)/8);
            unsigned long long int stime = ___rdtsc();
            int count1_prime = SIMD_WIDTH_CPU * ((count1 + SIMD_WIDTH_CPU - 1)/SIMD_WIDTH_CPU);
            if (self)
            {
                CPUFunction_Compute_Distance_And_Populate_Hist_N(Pos0, count0, Pos1, count1, BinCorners2, nrbin, DD_int0, DD_int1, min_bin_id, max_bin_id, Range1_X, Range1_Y, Range1_Z, threadid);
            }
            else
            {
                for(int q=0; q<count1; q++) Aligned_Buffer[0*count1_prime + q] = Pos1[0*count1 + q]; for(int q=count1; q < count1_prime; q++) Aligned_Buffer[0*count1_prime + q] = LARGE_ENOUGH;
                for(int q=0; q<count1; q++) Aligned_Buffer[1*count1_prime + q] = Pos1[1*count1 + q]; for(int q=count1; q < count1_prime; q++) Aligned_Buffer[1*count1_prime + q] = LARGE_ENOUGH;
                for(int q=0; q<count1; q++) Aligned_Buffer[2*count1_prime + q] = Pos1[2*count1 + q]; for(int q=count1; q < count1_prime; q++) Aligned_Buffer[2*count1_prime + q] = LARGE_ENOUGH;
                CPUFunction_Compute_Distance_And_Populate_Hist_N(Pos0, count0, Aligned_Buffer, count1_prime, BinCorners2, nrbin, DD_int0, DD_int1, min_bin_id, max_bin_id, Range1_X, Range1_Y, Range1_Z, threadid);
                DD_int0[max_bin_id] -= (count0 * (count1_prime - count1));
            }
            if (threadid < 4) MT_5_CPU[8*threadid] += ___rdtsc() - stime;
            break;
        }
#endif
    }
#endif
}

__attribute__ (( target (mic))) 
void  MICFunction_Actual_Update_Histogram_Self_Cross(TYPE *Pos0, int count0, TYPE *Pos1, int count1, TYPE *Range0_X, TYPE *Range0_Y, TYPE *Range0_Z, 
        TYPE *Range1_X, TYPE *Range1_Y, TYPE *Range1_Z, TYPE *BinCorners2, 
                    unsigned int *DD_int0, unsigned int *DD_int1, unsigned int *Gather_Histogram0, unsigned int *Gather_Histogram1, int nrbin, int threadid)
{
    unsigned long long int stime = ___rdtsc();
    int self = 0;
    if (Pos0 == Pos1) self = 1;
    if (self) { if (count0 != count1) ERROR_PRINT();}

    TYPE min_dist_sqr, max_dist_sqr;
    int min_bin_id, max_bin_id;

    MICFunction_Compute_Min_Max_Dist_Sqr(Range0_X, Range0_Y, Range0_Z, Range1_X, Range1_Y, Range1_Z, &min_dist_sqr, &max_dist_sqr);

    MICFunction_Compute_min_max_bin_id(BinCorners2, nrbin, min_dist_sqr, max_dist_sqr, &min_bin_id, &max_bin_id);

    int number_of_bins = max_bin_id - min_bin_id + 1;

    TYPE *Aligned_Buffer = global_Aligned_Buffer_MIC[threadid];

    TYPE LARGE_ENOUGH = 1000.0 * global_grid_D_MIC.Max[2];

    switch(number_of_bins)
    {
        case 1:
        {
            //Basically max_bin_id == min_bin_id...
            MICFunction_Compute_Distance_And_Populate_Hist_1(Pos0, count0, Pos1, count1, BinCorners2, nrbin, DD_int0, DD_int1, min_bin_id, max_bin_id, Range1_X, Range1_Y, Range1_Z, threadid);
            break;
        }

        case 2:
        {
            unsigned long long int stime = ___rdtsc();
            //Basically there are 2 bins (min_bin_id, min_bin_id+1)...
            if (self)
            {
                MICFunction_Compute_Distance_And_Populate_Hist_2(Pos0, count0, Pos1, count1, BinCorners2, nrbin, DD_int0, DD_int1, min_bin_id, max_bin_id, Range1_X, Range1_Y, Range1_Z, threadid);
            }
            else
            {
                //int count1_prime = 8*((count1 + 7)/8);
                int count1_prime = SIMD_WIDTH_MIC * ((count1 + SIMD_WIDTH_MIC - 1)/SIMD_WIDTH_MIC);
                for(int q=0; q<count1; q++) Aligned_Buffer[0*count1_prime + q] = Pos1[0*count1 + q]; for(int q=count1; q < count1_prime; q++) Aligned_Buffer[0*count1_prime + q] = LARGE_ENOUGH;
                for(int q=0; q<count1; q++) Aligned_Buffer[1*count1_prime + q] = Pos1[1*count1 + q]; for(int q=count1; q < count1_prime; q++) Aligned_Buffer[1*count1_prime + q] = LARGE_ENOUGH;
                for(int q=0; q<count1; q++) Aligned_Buffer[2*count1_prime + q] = Pos1[2*count1 + q]; for(int q=count1; q < count1_prime; q++) Aligned_Buffer[2*count1_prime + q] = LARGE_ENOUGH;
                MICFunction_Compute_Distance_And_Populate_Hist_2(Pos0, count0, Aligned_Buffer, count1_prime, BinCorners2, nrbin, DD_int0, DD_int1, min_bin_id, max_bin_id, Range1_X, Range1_Y, Range1_Z, threadid);
                DD_int0[max_bin_id] -= (count0 * (count1_prime - count1));
            }
            if (threadid < 4) MT_2_MIC[8*threadid] += ___rdtsc() - stime;
            break;
        }

        case 3:
        {
            unsigned long long int stime = ___rdtsc();
            //Basically there are 3 bins (min_bin_id, min_bin_id+1, min_bin_id+2)...
            if (self)
            {
                MICFunction_Compute_Distance_And_Populate_Hist_3(Pos0, count0, Pos1, count1, BinCorners2, nrbin, DD_int0, DD_int1, min_bin_id, max_bin_id, Range1_X, Range1_Y, Range1_Z, threadid);
            }
            else
            {
                //int count1_prime = 8*((count1 + 7)/8);
                int count1_prime = SIMD_WIDTH_MIC * ((count1 + SIMD_WIDTH_MIC - 1)/SIMD_WIDTH_MIC);
                for(int q=0; q<count1; q++) Aligned_Buffer[0*count1_prime + q] = Pos1[0*count1 + q]; for(int q=count1; q < count1_prime; q++) Aligned_Buffer[0*count1_prime + q] = LARGE_ENOUGH;
                for(int q=0; q<count1; q++) Aligned_Buffer[1*count1_prime + q] = Pos1[1*count1 + q]; for(int q=count1; q < count1_prime; q++) Aligned_Buffer[1*count1_prime + q] = LARGE_ENOUGH;
                for(int q=0; q<count1; q++) Aligned_Buffer[2*count1_prime + q] = Pos1[2*count1 + q]; for(int q=count1; q < count1_prime; q++) Aligned_Buffer[2*count1_prime + q] = LARGE_ENOUGH;
                MICFunction_Compute_Distance_And_Populate_Hist_3(Pos0, count0, Aligned_Buffer, count1_prime, BinCorners2, nrbin, DD_int0, DD_int1, min_bin_id, max_bin_id, Range1_X, Range1_Y, Range1_Z, threadid);
                DD_int0[max_bin_id] -= (count0 * (count1_prime - count1));
            }
            if (threadid < 4) MT_3_MIC[8*threadid] += ___rdtsc() - stime;
            break;
        }

        case 4:
        {
            unsigned long long int stime = ___rdtsc();
            //Basically there are 4 bins (min_bin_id, min_bin_id+1, min_bin_id+2, min_bin_id+3)...
            if (self)
            {
                MICFunction_Compute_Distance_And_Populate_Hist_4(Pos0, count0, Pos1, count1, BinCorners2, nrbin, DD_int0, DD_int1, min_bin_id, max_bin_id, Range1_X, Range1_Y, Range1_Z, threadid);
            }
            else
            {
                //int count1_prime = 8*((count1 + 7)/8);
                int count1_prime = SIMD_WIDTH_MIC * ((count1 + SIMD_WIDTH_MIC - 1)/SIMD_WIDTH_MIC);
                for(int q=0; q<count1; q++) Aligned_Buffer[0*count1_prime + q] = Pos1[0*count1 + q]; for(int q=count1; q < count1_prime; q++) Aligned_Buffer[0*count1_prime + q] = LARGE_ENOUGH;
                for(int q=0; q<count1; q++) Aligned_Buffer[1*count1_prime + q] = Pos1[1*count1 + q]; for(int q=count1; q < count1_prime; q++) Aligned_Buffer[1*count1_prime + q] = LARGE_ENOUGH;
                for(int q=0; q<count1; q++) Aligned_Buffer[2*count1_prime + q] = Pos1[2*count1 + q]; for(int q=count1; q < count1_prime; q++) Aligned_Buffer[2*count1_prime + q] = LARGE_ENOUGH;
                MICFunction_Compute_Distance_And_Populate_Hist_4(Pos0, count0, Aligned_Buffer, count1_prime, BinCorners2, nrbin, DD_int0, DD_int1, min_bin_id, max_bin_id, Range1_X, Range1_Y, Range1_Z, threadid);
                DD_int0[max_bin_id] -= (count0 * (count1_prime - count1));
            }
            if (threadid < 4) MT_4_MIC[8*threadid] += ___rdtsc() - stime;
            break;
        }

        default:
        {
            //int count1_prime = 8*((count1 + 7)/8);
            unsigned long long int stime = ___rdtsc();
            int count1_prime = SIMD_WIDTH_MIC * ((count1 + SIMD_WIDTH_MIC - 1)/SIMD_WIDTH_MIC);
            if (self)
            {
                MICFunction_Compute_Distance_And_Populate_Hist_N(Pos0, count0, Pos1, count1, BinCorners2, nrbin, DD_int0, DD_int1, min_bin_id, max_bin_id, Range1_X, Range1_Y, Range1_Z, threadid);
            }
            else
            {
                for(int q=0; q<count1; q++) Aligned_Buffer[0*count1_prime + q] = Pos1[0*count1 + q]; for(int q=count1; q < count1_prime; q++) Aligned_Buffer[0*count1_prime + q] = LARGE_ENOUGH;
                for(int q=0; q<count1; q++) Aligned_Buffer[1*count1_prime + q] = Pos1[1*count1 + q]; for(int q=count1; q < count1_prime; q++) Aligned_Buffer[1*count1_prime + q] = LARGE_ENOUGH;
                for(int q=0; q<count1; q++) Aligned_Buffer[2*count1_prime + q] = Pos1[2*count1 + q]; for(int q=count1; q < count1_prime; q++) Aligned_Buffer[2*count1_prime + q] = LARGE_ENOUGH;
                MICFunction_Compute_Distance_And_Populate_Hist_N(Pos0, count0, Aligned_Buffer, count1_prime, BinCorners2, nrbin, DD_int0, DD_int1, min_bin_id, max_bin_id, Range1_X, Range1_Y, Range1_Z, threadid);
                DD_int0[max_bin_id] -= (count0 * (count1_prime - count1));
            }
            if (threadid < 4) MT_5_MIC[8*threadid] += ___rdtsc() - stime;
            break;
        }
    }
    unsigned long long int etime = ___rdtsc();
    unsigned long long int ttime = etime - stime;
    if (threadid < 4) OT_Z_MIC[8*threadid] += ttime;
}

__attribute__ (( target (mic))) 
void MICFunction_Update_Histogram_Self_Cross(TYPE *Pos0, int count0, TYPE *Bdry0_X, TYPE *Bdry0_Y, TYPE *Bdry0_Z, int *Range0, int ranges0, 
                                 TYPE *Pos1, int count1, TYPE *Bdry1_X, TYPE *Bdry1_Y, TYPE *Bdry1_Z, int *Range1, int ranges1, 
                                 TYPE *Range0_X, TYPE *Range0_Y, TYPE *Range0_Z, 
                                 TYPE *Range1_X, TYPE *Range1_Y, TYPE *Range1_Z, TYPE *BinCorners2, 
                                 unsigned int *DD_int0, unsigned int *DD_int1, unsigned int *Gather_Histogram0, unsigned int *Gather_Histogram1, int nrbin, int threadid)
{

    unsigned long long int stime = ___rdtsc();

    /////////////////////////////////////////////////////////////////
    ////Data stored in XXXXX YYYYYYYYY ZZZZZZZZZZ format...
    ///////////////////////////////////////////////////////////////////

    int self = 0;
    if (Pos0 == Pos1) self = 1;
    if (self) { if (count0 != count1) ERROR_PRINT();}

    TYPE min_dist_sqr, max_dist_sqr;
    int min_bin_id, max_bin_id;

    MICFunction_Compute_Min_Max_Dist_Sqr(Range0_X, Range0_Y, Range0_Z, Range1_X, Range1_Y, Range1_Z, &min_dist_sqr, &max_dist_sqr);

    MICFunction_Compute_min_max_bin_id(BinCorners2, nrbin, min_dist_sqr, max_dist_sqr, &min_bin_id, &max_bin_id);

    int number_of_bins = max_bin_id - min_bin_id + 1;

    if (number_of_bins == 1)
    {
        MICFunction_Compute_Distance_And_Populate_Hist_1(Pos0, count0, Pos1, count1, BinCorners2, nrbin, DD_int0, DD_int1, min_bin_id, max_bin_id, Range1_X, Range1_Y, Range1_Z, threadid);
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
        if (self) div_1 = div_0; //Keep in mind that if self is true, div_1 == div_0 :)
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

            MICFunction_Actual_Update_Histogram_Self_Cross(Pos00, count00, Pos11, count11, Range2_X, Range2_Y, Range2_Z, Range3_X, Range3_Y, Range3_Z, BinCorners2, 
                    DD_int0, DD_int1, Gather_Histogram0, Gather_Histogram1, nrbin, threadid);
        }
    }

    unsigned long long int etime = ___rdtsc();
    unsigned long long int ttime = etime - stime;
    if (threadid < 4) NT_Z_MIC[8*threadid] += ttime;
}
 
void CPUFunction_Update_Histogram_Self_Cross(TYPE *Pos0, int count0, TYPE *Bdry0_X, TYPE *Bdry0_Y, TYPE *Bdry0_Z, int *Range0, int ranges0, 
                                 TYPE *Pos1, int count1, TYPE *Bdry1_X, TYPE *Bdry1_Y, TYPE *Bdry1_Z, int *Range1, int ranges1, 
                                 TYPE *Range0_X, TYPE *Range0_Y, TYPE *Range0_Z, 
                                 TYPE *Range1_X, TYPE *Range1_Y, TYPE *Range1_Z, TYPE *BinCorners2, 
                                 unsigned int *DD_int0, unsigned int *DD_int1, unsigned int *Gather_Histogram0, unsigned int *Gather_Histogram1, int nrbin, int threadid)
{
#ifndef __MIC__

    unsigned long long int stime = ___rdtsc();

    /////////////////////////////////////////////////////////////////
    ////Data stored in XXXXX YYYYYYYYY ZZZZZZZZZZ format...
    ///////////////////////////////////////////////////////////////////

    int self = 0;
    if (Pos0 == Pos1) self = 1;
    if (self) { if (count0 != count1) ERROR_PRINT();}

    TYPE min_dist_sqr, max_dist_sqr;
    int min_bin_id, max_bin_id;

    CPUFunction_Compute_Min_Max_Dist_Sqr(Range0_X, Range0_Y, Range0_Z, Range1_X, Range1_Y, Range1_Z, &min_dist_sqr, &max_dist_sqr);

    CPUFunction_Compute_min_max_bin_id(BinCorners2, nrbin, min_dist_sqr, max_dist_sqr, &min_bin_id, &max_bin_id);

    int number_of_bins = max_bin_id - min_bin_id + 1;

    if (number_of_bins == 1)
    {
        CPUFunction_Compute_Distance_And_Populate_Hist_1(Pos0, count0, Pos1, count1, BinCorners2, nrbin, DD_int0, DD_int1, min_bin_id, max_bin_id, Range1_X, Range1_Y, Range1_Z, threadid);
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
        if (self) div_1 = div_0; //Keep in mind that if self is true, div_1 == div_0 :)
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

            CPUFunction_Actual_Update_Histogram_Self_Cross(Pos00, count00, Pos11, count11, Range2_X, Range2_Y, Range2_Z, Range3_X, Range3_Y, Range3_Z, BinCorners2, 
                    DD_int0, DD_int1, Gather_Histogram0, Gather_Histogram1, nrbin, threadid);
        }
    }

    unsigned long long int etime = ___rdtsc();
    unsigned long long int ttime = etime - stime;
    if (threadid < 4) NT_Z_CPU[8*threadid] += ttime;
#endif
}


TYPE CPUFunction_Find_Minimum_Distance_Between_Cells(TYPE *Range0_X, TYPE *Range0_Y, TYPE *Range0_Z, TYPE *Range1_X, TYPE *Range1_Y, TYPE *Range1_Z)
{
    //return 123;
    TYPE min_dst_sqr = 0;

    if (Range0_X[0] > Range1_X[1]) min_dst_sqr += (Range0_X[0] - Range1_X[1])*(Range0_X[0] - Range1_X[1]);
    if (Range1_X[0] > Range0_X[1]) min_dst_sqr += (Range1_X[0] - Range0_X[1])*(Range1_X[0] - Range0_X[1]);

    if (Range0_Y[0] > Range1_Y[1]) min_dst_sqr += (Range0_Y[0] - Range1_Y[1])*(Range0_Y[0] - Range1_Y[1]);
    if (Range1_Y[0] > Range0_Y[1]) min_dst_sqr += (Range1_Y[0] - Range0_Y[1])*(Range1_Y[0] - Range0_Y[1]);

    if (Range0_Z[0] > Range1_Z[1]) min_dst_sqr += (Range0_Z[0] - Range1_Z[1])*(Range0_Z[0] - Range1_Z[1]);
    if (Range1_Z[0] > Range0_Z[1]) min_dst_sqr += (Range1_Z[0] - Range0_Z[1])*(Range1_Z[0] - Range0_Z[1]);

    return (min_dst_sqr);
}


__attribute__ (( target (mic))) 
TYPE MICFunction_Find_Minimum_Distance_Between_Cells(TYPE *Range0_X, TYPE *Range0_Y, TYPE *Range0_Z, TYPE *Range1_X, TYPE *Range1_Y, TYPE *Range1_Z)
{
    //return 123;
    TYPE min_dst_sqr = 0;

    if (Range0_X[0] > Range1_X[1]) min_dst_sqr += (Range0_X[0] - Range1_X[1])*(Range0_X[0] - Range1_X[1]);
    if (Range1_X[0] > Range0_X[1]) min_dst_sqr += (Range1_X[0] - Range0_X[1])*(Range1_X[0] - Range0_X[1]);

    if (Range0_Y[0] > Range1_Y[1]) min_dst_sqr += (Range0_Y[0] - Range1_Y[1])*(Range0_Y[0] - Range1_Y[1]);
    if (Range1_Y[0] > Range0_Y[1]) min_dst_sqr += (Range1_Y[0] - Range0_Y[1])*(Range1_Y[0] - Range0_Y[1]);

    if (Range0_Z[0] > Range1_Z[1]) min_dst_sqr += (Range0_Z[0] - Range1_Z[1])*(Range0_Z[0] - Range1_Z[1]);
    if (Range1_Z[0] > Range0_Z[1]) min_dst_sqr += (Range1_Z[0] - Range0_Z[1])*(Range1_Z[0] - Range0_Z[1]);

    return (min_dst_sqr);
}

void  CPUFunction_Perform_DR_Helper(void *arg)
{
     int threadid = omp_get_thread_num();
     int taskid   = (int)((size_t)(arg));

    //printf("(%d) :: taskid = %d\n", threadid, taskid);

    Grid *grid_D = &global_grid_D_CPU;
    Grid *grid_R = &global_grid_R_CPU;

    TYPE *Extent = grid_D->Extent; //Both grids have the same extents...

    int dimx = grid_D->dimx;
    int dimy = grid_D->dimy;
    int dimz = grid_D->dimz;

    int dimxy = dimx * dimy;

    int dx = global_dx_CPU;
    int dy = global_dy_CPU;
    int dz = global_dz_CPU;

    TYPE *Cell_Width = grid_D->Cell_Width;

    TYPE Range0_X[2], Range0_Y[2], Range0_Z[2];
    TYPE Range1_X[2], Range1_Y[2], Range1_Z[2];

    int start_processing_cell_index = global_starting_cell_index_D_CPU;
    int   end_processing_cell_index = global_ending_cell_index_D_CPU;

    //int start_processing_cell_index_prime = (start_processing_cell_index + threadid);

    unsigned long long int threshold_for_accumulate_sum = (1<<29); threshold_for_accumulate_sum = (threshold_for_accumulate_sum << 1) - 1;

    unsigned long long int local_start_time, local_end_time;

    local_start_time = ___rdtsc();

    long long int actual_sum = global_actual_sum_dr_CPU[16*threadid];
    long long int curr_accumulated_actual_sum = 0;

    unsigned int *Gather_Histogram0 = global_Gather_Histogram0_CPU[threadid];
    unsigned int *Gather_Histogram1 = global_Gather_Histogram1_CPU[threadid];
    unsigned int *DR_int0 = global_DR_int0_CPU[threadid];
    unsigned int *DR_int1 = global_DR_int1_CPU[threadid];

    TYPE *Pos1 = global_Pos1_CPU[threadid];
    TYPE *Bdry1_X = global_Bdry1_X_CPU[threadid];
    TYPE *Bdry1_Y = global_Bdry1_Y_CPU[threadid];
    TYPE *Bdry1_Z = global_Bdry1_Z_CPU[threadid];

    TYPE *BinCorners2 = global_BinCorners2_CPU;
    unsigned long long int *DR = local_Histogram_DR_CPU[threadid];

    int nrbin = global_nrbin_CPU;
    TYPE rmax_2 = global_rmax_2_CPU;

    int current_cell_index = global_starting_cell_index_D_CPU + taskid;
    //Perform work on current_cell_index...
    {
        int x, y, z; z = current_cell_index / dimxy; int xy = current_cell_index % dimxy; y = xy / dimx; x = xy % dimx;

        //if (current_cell_index % 1000 == 23) printf("threadid = %d ::: current_cell_index = %d\n", threadid, current_cell_index);

        if (!global_Required_D_For_R_CPU[current_cell_index]) ERROR_PRINT();



#if 0
        if (current_cell_index != 419) continue;
        //if (current_cell_index % 900 == 0)
        {
            printf("x = %d ::: y = %d ::: z = %d\n", x, y, z);
            printf("Z = %d ::: ", z);
            for(int k=0; k<12; k++)
            {
                long long int local_sum = DD[k];
                local_sum += DD_int0[k];
                local_sum += DD_int1[k];
                printf("%lld ", local_sum); 
            }

            printf("\n");
        }
#endif
        int objects_in_this_cell = grid_D->Count_Per_Cell[current_cell_index];
        int subdivisions_in_this_cell = grid_D->Number_of_kd_subdivisions[current_cell_index];
        int *Range0 = grid_D->Range[current_cell_index];
        TYPE *Bdry0_X = grid_D->Bdry_X[current_cell_index];
        TYPE *Bdry0_Y = grid_D->Bdry_Y[current_cell_index];
        TYPE *Bdry0_Z = grid_D->Bdry_Z[current_cell_index];

        //if (current_cell_index == debug_cell_id) printf("NODE_ID = %d ::: objects_in_this_cell (%d) = %d\n", node_id, debug_cell_id, objects_in_this_cell);
        TYPE *Pos0 = grid_D->Positions[current_cell_index];

        Range0_X[0] = (x*1.00)*Cell_Width[0]; Range0_X[1] = ((x+1)*1.00)*Cell_Width[0];
        Range0_Y[0] = (y*1.00)*Cell_Width[1]; Range0_Y[1] = ((y+1)*1.00)*Cell_Width[1];
        Range0_Z[0] = (z*1.00)*Cell_Width[2]; Range0_Z[1] = ((z+1)*1.00)*Cell_Width[2];

#if 0
        {
            int neighbor_cell_index = current_cell_index;
            int subdivisions_in_neighboring_cell = grid_R->Number_of_kd_subdivisions[neighbor_cell_index];
            int *Range2 = grid_R->Range[neighnor_cell_index];

            TYPE Range2_X[2], Range2_Y[2], Range2_Z[2];
            Range2_X[0] = Range0_X[0]; Range2_X[1] = Range0_X[1];
            Range2_Y[0] = Range0_Y[0]; Range2_Y[1] = Range0_Y[1];
            Range2_Z[0] = Range0_Z[0]; Range2_Z[1] = Range0_Z[1];
                    
            TYPE *Bdry2_X = grid_R->Bdry_X[neighbor_cell_index];
            TYPE *Bdry2_Y = grid_R->Bdry_Y[neighbor_cell_index];
            TYPE *Bdry2_Z = grid_R->Bdry_Z[neighbor_cell_index];

            int objects_in_neighboring_cell = grid_R->Count_Per_Cell[neighbor_cell_index];
            TYPE *Pos2 = grid_R->Positions[neighbor_cell_index];

            long long int local_sum = ((long long int)(objects_in_this_cell) * (long long int)(objects_in_neighboring_cell));

            actual_sum += local_sum;
            curr_accumulated_actual_sum += local_sum;

            Update_Histogram_Self_Cross(Pos0, objects_in_this_cell,         Bdry0_X, Bdry0_Y, Bdry0_Z, Range0, subdivisions_in_this_cell,
                                        Pos2, objects_in_neighboring_cell,  Bdry2_X, Bdry2_Y, Bdry2_Z, Range2, subdivisions_in_neighboring_cell, 
                                        Range0_X, Range0_Y, Range0_Z, 
                                        Range2_X, Range2_Y, Range2_Z, BinCorners2, DR_int0, DR_int1, Gather_Histogram0, Gather_Histogram1, nrbin, threadid);
        }
#endif

        for(int zz = (z - dz); zz <= (z + dz); zz++)
        {
            for(int yy = (y - dy); yy <= (y + dy); yy++)
            {
                for(int xx = (x - dx); xx <= (x + dx); xx++)
                {
                    //Our neighbor is the (xx, yy, zz) cell...
                    //if ((xx == x) && (yy == y) && (zz == z)) continue; XXX This line is not required for DR...

                    if ((xx == 30) && (yy == 13) && (zz == 0)) 
                    {
                        //printf("asffsdfs\n");
                    }
                            
                    ////////////////////////////////////////////////////////////////////////////////////////
                    //Step A: Figure out if the nearest points between the grids is >= rmax...
                    ////////////////////////////////////////////////////////////////////////////////////////
                            
                    Range1_X[0] = (xx*1.00)*Cell_Width[0]; Range1_X[1] = ((xx+1)*1.00)*Cell_Width[0];
                    Range1_Y[0] = (yy*1.00)*Cell_Width[1]; Range1_Y[1] = ((yy+1)*1.00)*Cell_Width[1];
                    Range1_Z[0] = (zz*1.00)*Cell_Width[2]; Range1_Z[1] = ((zz+1)*1.00)*Cell_Width[2];

                    TYPE min_dist_2 = CPUFunction_Find_Minimum_Distance_Between_Cells(Range0_X, Range0_Y, Range0_Z, Range1_X, Range1_Y, Range1_Z);

                    if (min_dist_2 > rmax_2) continue;

                    ////////////////////////////////////////////////////////////////////////////////////////
                    //Step B: Collect Particles into consecutive positions and also respect PREIODICITY...
                    ////////////////////////////////////////////////////////////////////////////////////////

                    int xx_prime = xx, yy_prime = yy, zz_prime = zz;
                    if (xx < 0) xx_prime = xx + dimx; else if (xx >= (dimx)) xx_prime = xx - dimx;
                    if (yy < 0) yy_prime = yy + dimy; else if (yy >= (dimy)) yy_prime = yy - dimy;
                    if (zz < 0) zz_prime = zz + dimz; else if (zz >= (dimz)) zz_prime = zz - dimz;

                    if ( (xx_prime < 0) || (xx_prime >= dimx))  ERROR_PRINT();
                    if ( (yy_prime < 0) || (yy_prime >= dimy))  ERROR_PRINT();
                    if ( (zz_prime < 0) || (zz_prime >= dimz))  ERROR_PRINT();

                    int neighbor_cell_index = GET_CELL_INDEX(xx_prime, yy_prime, zz_prime);
                    //if (neighbor_cell_index > current_cell_index) continue; //XXX: Very important line...

                    if (!global_Required_D_For_R_CPU[neighbor_cell_index]) ERROR_PRINT();

                    int *Range1 = grid_R->Range[neighbor_cell_index];
                    int objects_in_neighboring_cell = grid_R->Count_Per_Cell[neighbor_cell_index];
                    int subdivisions_in_neighboring_cell = grid_R->Number_of_kd_subdivisions[neighbor_cell_index];

                    TYPE Delta[DIMENSIONS]; Delta[0] = Delta[1] = Delta[2] = 0.0;
                    if (xx < 0) Delta[0] = -Extent[0]; else if (xx >= (dimx)) Delta[0] = Extent[0];
                    if (yy < 0) Delta[1] = -Extent[1]; else if (yy >= (dimy)) Delta[1] = Extent[1];
                    if (zz < 0) Delta[2] = -Extent[2]; else if (zz >= (dimz)) Delta[2] = Extent[2];
                
                    ////////////////////////////////////////////////////////////////////////////////////////
                    //Step C: Copy Positions, Bdry too...
                    ////////////////////////////////////////////////////////////////////////////////////////
                            
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


                    ////////////////////////////////////////////////////////////////////////////////////////
                    //Step C: Now please perform computation without further ado...
                    ////////////////////////////////////////////////////////////////////////////////////////

                    long long int local_sum = ((long long int)(objects_in_this_cell) * ((long long int)(objects_in_neighboring_cell)));
                    actual_sum += local_sum;
                    curr_accumulated_actual_sum += local_sum;

                    CPUFunction_Update_Histogram_Self_Cross(Pos0, objects_in_this_cell,         Bdry0_X, Bdry0_Y, Bdry0_Z, Range0, subdivisions_in_this_cell,
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
                            printf("node_id_CPU = %d ::: threadid = %d :: curr_accumulated_actual_sum = %lld ::: ttt = %lld\n", node_id_CPU, threadid, curr_accumulated_actual_sum, ttt);
                            ERROR_PRINT();
                        }

                        curr_accumulated_actual_sum = 0;
                        for(int i=0; i<=(1+nrbin); i++) DR[i] += DR_int0[i];
                        for(int i=0; i<=(1+nrbin); i++) DR[i] += DR_int1[i];
                        for(int i=0; i<=(1+nrbin); i++) DR_int0[i] = 0;
                        for(int i=0; i<=(1+nrbin); i++) DR_int1[i] = 0;

#if 0
                        for(int bin_id = 0; bin_id <= (1+nrbin); bin_id++)
                        {
                            for(int lane = 0; lane < SIMD_WIDTH_CPU; lane++)
                            {
                                DR[bin_id] += Gather_Histogram0[bin_id*SIMD_WIDTH_CPU + lane];
                                DR[bin_id] += Gather_Histogram1[bin_id*SIMD_WIDTH_CPU + lane];
                            }
                        }

                        for(int i=0; i<(SIMD_WIDTH_CPU*HIST_BINS); i++) Gather_Histogram0[i] = 0;
                        for(int i=0; i<(SIMD_WIDTH_CPU*HIST_BINS); i++) Gather_Histogram1[i] = 0;
#endif
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

#if 0
        for(int bin_id = 0; bin_id <= (1+nrbin); bin_id++)
        {
            for(int lane = 0; lane < SIMD_WIDTH_CPU; lane++)
            {
                DR[bin_id] += Gather_Histogram0[bin_id*SIMD_WIDTH_CPU + lane];
                DR[bin_id] += Gather_Histogram1[bin_id*SIMD_WIDTH_CPU + lane];
            }
        }
                
        for(int i=0; i<(SIMD_WIDTH_CPU*HIST_BINS); i++) Gather_Histogram0[i] = 0;
        for(int i=0; i<(SIMD_WIDTH_CPU*HIST_BINS); i++) Gather_Histogram1[i] = 0;
#endif
    }

    global_actual_sum_dr_CPU[16*threadid] = actual_sum;

    local_end_time = ___rdtsc();
    global_time_per_thread_dr_CPU[threadid] += local_end_time - local_start_time;

    {
        if (taskid < 10)
        {
            //printf("taskid : %d :: ", taskid); for(int p = 0; p < global_nrbin; p++) printf(" %lld ", DR[p]); printf("\n");
        }
    }
}

__attribute__ (( target (mic))) 
void  MICFunction_Perform_DR_Helper(void *arg)
{
     int threadid = omp_get_thread_num();
     int taskid   = (int)((size_t)(arg));

    //printf("(%d) :: taskid = %d\n", threadid, taskid);

    Grid *grid_D = &global_grid_D_MIC;
    Grid *grid_R = &global_grid_R_MIC;

    TYPE *Extent = grid_D->Extent; //Both grids have the same extents...

    int dimx = grid_D->dimx;
    int dimy = grid_D->dimy;
    int dimz = grid_D->dimz;

    int dimxy = dimx * dimy;

    int dx = global_dx_MIC;
    int dy = global_dy_MIC;
    int dz = global_dz_MIC;

    TYPE *Cell_Width = grid_D->Cell_Width;

    TYPE Range0_X[2], Range0_Y[2], Range0_Z[2];
    TYPE Range1_X[2], Range1_Y[2], Range1_Z[2];

    int start_processing_cell_index = global_starting_cell_index_D_MIC;
    int   end_processing_cell_index = global_ending_cell_index_D_MIC;

    //int start_processing_cell_index_prime = (start_processing_cell_index + threadid);

    unsigned long long int threshold_for_accumulate_sum = (1<<29); threshold_for_accumulate_sum = (threshold_for_accumulate_sum << 1) - 1;

    unsigned long long int local_start_time, local_end_time;

    local_start_time = ___rdtsc();

    long long int actual_sum = global_actual_sum_dr_MIC[16*threadid];
    long long int curr_accumulated_actual_sum = 0;

    unsigned int *Gather_Histogram0 = global_Gather_Histogram0_MIC[threadid];
    unsigned int *Gather_Histogram1 = global_Gather_Histogram1_MIC[threadid];
    unsigned int *DR_int0 = global_DR_int0_MIC[threadid];
    unsigned int *DR_int1 = global_DR_int1_MIC[threadid];

    TYPE *Pos1 = global_Pos1_MIC[threadid];
    TYPE *Bdry1_X = global_Bdry1_X_MIC[threadid];
    TYPE *Bdry1_Y = global_Bdry1_Y_MIC[threadid];
    TYPE *Bdry1_Z = global_Bdry1_Z_MIC[threadid];

    TYPE *BinCorners2 = global_BinCorners2_MIC;
    unsigned long long int *DR = local_Histogram_DR_MIC[threadid];

    int nrbin = global_nrbin_MIC;
    TYPE rmax_2 = global_rmax_2_MIC;

    int current_cell_index = global_starting_cell_index_D_MIC + taskid;
    //Perform work on current_cell_index...
    {
        int x, y, z; z = current_cell_index / dimxy; int xy = current_cell_index % dimxy; y = xy / dimx; x = xy % dimx;

        //if (current_cell_index % 1000 == 23) printf("threadid = %d ::: current_cell_index = %d\n", threadid, current_cell_index);

        if (!global_Required_D_For_R_MIC[current_cell_index]) ERROR_PRINT();



#if 0
        if (current_cell_index != 419) continue;
        //if (current_cell_index % 900 == 0)
        {
            printf("x = %d ::: y = %d ::: z = %d\n", x, y, z);
            printf("Z = %d ::: ", z);
            for(int k=0; k<12; k++)
            {
                long long int local_sum = DD[k];
                local_sum += DD_int0[k];
                local_sum += DD_int1[k];
                printf("%lld ", local_sum); 
            }

            printf("\n");
        }
#endif
        int objects_in_this_cell = grid_D->Count_Per_Cell[current_cell_index];
        int subdivisions_in_this_cell = grid_D->Number_of_kd_subdivisions[current_cell_index];
        int *Range0 = grid_D->Range[current_cell_index];
        TYPE *Bdry0_X = grid_D->Bdry_X[current_cell_index];
        TYPE *Bdry0_Y = grid_D->Bdry_Y[current_cell_index];
        TYPE *Bdry0_Z = grid_D->Bdry_Z[current_cell_index];

        //if (current_cell_index == debug_cell_id) printf("NODE_ID = %d ::: objects_in_this_cell (%d) = %d\n", node_id, debug_cell_id, objects_in_this_cell);
        TYPE *Pos0 = grid_D->Positions[current_cell_index];

        Range0_X[0] = (x*1.00)*Cell_Width[0]; Range0_X[1] = ((x+1)*1.00)*Cell_Width[0];
        Range0_Y[0] = (y*1.00)*Cell_Width[1]; Range0_Y[1] = ((y+1)*1.00)*Cell_Width[1];
        Range0_Z[0] = (z*1.00)*Cell_Width[2]; Range0_Z[1] = ((z+1)*1.00)*Cell_Width[2];

#if 0
        {
            int neighbor_cell_index = current_cell_index;
            int subdivisions_in_neighboring_cell = grid_R->Number_of_kd_subdivisions[neighbor_cell_index];
            int *Range2 = grid_R->Range[neighnor_cell_index];

            TYPE Range2_X[2], Range2_Y[2], Range2_Z[2];
            Range2_X[0] = Range0_X[0]; Range2_X[1] = Range0_X[1];
            Range2_Y[0] = Range0_Y[0]; Range2_Y[1] = Range0_Y[1];
            Range2_Z[0] = Range0_Z[0]; Range2_Z[1] = Range0_Z[1];
                    
            TYPE *Bdry2_X = grid_R->Bdry_X[neighbor_cell_index];
            TYPE *Bdry2_Y = grid_R->Bdry_Y[neighbor_cell_index];
            TYPE *Bdry2_Z = grid_R->Bdry_Z[neighbor_cell_index];

            int objects_in_neighboring_cell = grid_R->Count_Per_Cell[neighbor_cell_index];
            TYPE *Pos2 = grid_R->Positions[neighbor_cell_index];

            long long int local_sum = ((long long int)(objects_in_this_cell) * (long long int)(objects_in_neighboring_cell));

            actual_sum += local_sum;
            curr_accumulated_actual_sum += local_sum;

            Update_Histogram_Self_Cross(Pos0, objects_in_this_cell,         Bdry0_X, Bdry0_Y, Bdry0_Z, Range0, subdivisions_in_this_cell,
                                        Pos2, objects_in_neighboring_cell,  Bdry2_X, Bdry2_Y, Bdry2_Z, Range2, subdivisions_in_neighboring_cell, 
                                        Range0_X, Range0_Y, Range0_Z, 
                                        Range2_X, Range2_Y, Range2_Z, BinCorners2, DR_int0, DR_int1, Gather_Histogram0, Gather_Histogram1, nrbin, threadid);
        }
#endif

        int ccounter = 0;

        for(int zz = (z - dz); zz <= (z + dz); zz++)
        {
            for(int yy = (y - dy); yy <= (y + dy); yy++)
            {
                int yy_prime = yy;
                int zz_prime = zz;
                if (yy < 0) yy_prime = yy + dimy; else if (yy >= (dimy)) yy_prime = yy - dimy;
                if (zz < 0) zz_prime = zz + dimz; else if (zz >= (dimz)) zz_prime = zz - dimz;
	            int base_cell_index = GET_CELL_INDEX(0, yy_prime, zz_prime);

                for(int xx = (x - dx); xx <= (x + dx); xx++)
                {
                    if (!global_Template_during_hetero_DR_MIC[ccounter++]) continue;

                    //Our neighbor is the (xx, yy, zz) cell...
#if 0

                    //if ((xx == x) && (yy == y) && (zz == z)) continue; XXX This line is not required for DR...

                    if ((xx == 30) && (yy == 13) && (zz == 0)) 
                    {
                        //printf("asffsdfs\n");
                    }
                            
                    ////////////////////////////////////////////////////////////////////////////////////////
                    //Step A: Figure out if the nearest points between the grids is >= rmax...
                    ////////////////////////////////////////////////////////////////////////////////////////
                            
                    Range1_X[0] = (xx*1.00)*Cell_Width[0]; Range1_X[1] = ((xx+1)*1.00)*Cell_Width[0];
                    Range1_Y[0] = (yy*1.00)*Cell_Width[1]; Range1_Y[1] = ((yy+1)*1.00)*Cell_Width[1];
                    Range1_Z[0] = (zz*1.00)*Cell_Width[2]; Range1_Z[1] = ((zz+1)*1.00)*Cell_Width[2];

                    TYPE min_dist_2 = MICFunction_Find_Minimum_Distance_Between_Cells(Range0_X, Range0_Y, Range0_Z, Range1_X, Range1_Y, Range1_Z);

                    if (min_dist_2 > rmax_2) continue;
#endif

                    ////////////////////////////////////////////////////////////////////////////////////////
                    //Step B: Collect Particles into consecutive positions and also respect PREIODICITY...
                    ////////////////////////////////////////////////////////////////////////////////////////

                    int xx_prime = xx;
                    if (xx < 0) xx_prime = xx + dimx; else if (xx >= (dimx)) xx_prime = xx - dimx;
#if 0
                    int xx_prime = xx, yy_prime = yy, zz_prime = zz;
                    if (xx < 0) xx_prime = xx + dimx; else if (xx >= (dimx)) xx_prime = xx - dimx;
                    if (yy < 0) yy_prime = yy + dimy; else if (yy >= (dimy)) yy_prime = yy - dimy;
                    if (zz < 0) zz_prime = zz + dimz; else if (zz >= (dimz)) zz_prime = zz - dimz;

                    if ( (xx_prime < 0) || (xx_prime >= dimx))  ERROR_PRINT();
                    if ( (yy_prime < 0) || (yy_prime >= dimy))  ERROR_PRINT();
                    if ( (zz_prime < 0) || (zz_prime >= dimz))  ERROR_PRINT();
#else
#endif


                    //int neighbor_cell_index = GET_CELL_INDEX(xx_prime, yy_prime, zz_prime);
                    int neighbor_cell_index = base_cell_index + xx_prime;
                    //if (neighbor_cell_index > current_cell_index) continue; //XXX: Very important line...

                    //if (!global_Required_D_For_R_MIC[neighbor_cell_index]) ERROR_PRINT();

                    Range1_X[0] = (xx*1.00)*Cell_Width[0]; Range1_X[1] = ((xx+1)*1.00)*Cell_Width[0];
                    Range1_Y[0] = (yy*1.00)*Cell_Width[1]; Range1_Y[1] = ((yy+1)*1.00)*Cell_Width[1];
                    Range1_Z[0] = (zz*1.00)*Cell_Width[2]; Range1_Z[1] = ((zz+1)*1.00)*Cell_Width[2];

                    int *Range1 = grid_R->Range[neighbor_cell_index];
                    int objects_in_neighboring_cell = grid_R->Count_Per_Cell[neighbor_cell_index];
                    int subdivisions_in_neighboring_cell = grid_R->Number_of_kd_subdivisions[neighbor_cell_index];

                    TYPE Delta[DIMENSIONS]; Delta[0] = Delta[1] = Delta[2] = 0.0;
                    if (xx < 0) Delta[0] = -Extent[0]; else if (xx >= (dimx)) Delta[0] = Extent[0];
                    if (yy < 0) Delta[1] = -Extent[1]; else if (yy >= (dimy)) Delta[1] = Extent[1];
                    if (zz < 0) Delta[2] = -Extent[2]; else if (zz >= (dimz)) Delta[2] = Extent[2];
                
                    ////////////////////////////////////////////////////////////////////////////////////////
                    //Step C: Copy Positions, Bdry too...
                    ////////////////////////////////////////////////////////////////////////////////////////
                            
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


                    ////////////////////////////////////////////////////////////////////////////////////////
                    //Step C: Now please perform computation without further ado...
                    ////////////////////////////////////////////////////////////////////////////////////////

                    long long int local_sum = ((long long int)(objects_in_this_cell) * ((long long int)(objects_in_neighboring_cell)));
                    actual_sum += local_sum;
                    curr_accumulated_actual_sum += local_sum;

                    MICFunction_Update_Histogram_Self_Cross(Pos0, objects_in_this_cell,         Bdry0_X, Bdry0_Y, Bdry0_Z, Range0, subdivisions_in_this_cell,
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
                            printf("node_id_MIC = %d ::: threadid = %d :: curr_accumulated_actual_sum = %lld ::: ttt = %lld\n", node_id_MIC, threadid, curr_accumulated_actual_sum, ttt);
                            ERROR_PRINT();
                        }

                        curr_accumulated_actual_sum = 0;
                        for(int i=0; i<=(1+nrbin); i++) DR[i] += DR_int0[i];
                        for(int i=0; i<=(1+nrbin); i++) DR[i] += DR_int1[i];
                        for(int i=0; i<=(1+nrbin); i++) DR_int0[i] = 0;
                        for(int i=0; i<=(1+nrbin); i++) DR_int1[i] = 0;

#if 0
                        for(int bin_id = 0; bin_id <= (1+nrbin); bin_id++)
                        {
                            for(int lane = 0; lane < SIMD_WIDTH_MIC; lane++)
                            {
                                DR[bin_id] += Gather_Histogram0[bin_id*SIMD_WIDTH_MIC + lane];
                                DR[bin_id] += Gather_Histogram1[bin_id*SIMD_WIDTH_MIC + lane];
                            }
                        }

                        for(int i=0; i<(SIMD_WIDTH_MIC*HIST_BINS); i++) Gather_Histogram0[i] = 0;
                        for(int i=0; i<(SIMD_WIDTH_MIC*HIST_BINS); i++) Gather_Histogram1[i] = 0;
#endif
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

#if 0
        for(int bin_id = 0; bin_id <= (1+nrbin); bin_id++)
        {
            for(int lane = 0; lane < SIMD_WIDTH_MIC; lane++)
            {
                DR[bin_id] += Gather_Histogram0[bin_id*SIMD_WIDTH_MIC + lane];
                DR[bin_id] += Gather_Histogram1[bin_id*SIMD_WIDTH_MIC + lane];
            }
        }
                
        for(int i=0; i<(SIMD_WIDTH_MIC*HIST_BINS); i++) Gather_Histogram0[i] = 0;
        for(int i=0; i<(SIMD_WIDTH_MIC*HIST_BINS); i++) Gather_Histogram1[i] = 0;
#endif
    }

    global_actual_sum_dr_MIC[16*threadid] = actual_sum;

    local_end_time = ___rdtsc();
    global_time_per_thread_dr_MIC[threadid] += local_end_time - local_start_time;

    {
        if (taskid < 10)
        {
            //printf("taskid : %d :: ", taskid); for(int p = 0; p < global_nrbin; p++) printf(" %lld ", DR[p]); printf("\n");
        }
    }
}

void  CPUFunction_Perform_RR_Helper( void *arg)
{

     int threadid = omp_get_thread_num();
     int taskid   = (int)((size_t)(arg));

     //printf("threadid = %d ::: taskid = %d\n", threadid, taskid);

    //printf("(%d) :: taskid = %d\n", threadid, taskid);

    Grid *grid = &global_grid_R_CPU;

    TYPE *Extent = grid->Extent;

    int dimx = grid->dimx;
    int dimy = grid->dimy;
    int dimz = grid->dimz;

    int dimxy = dimx * dimy;

    int dx = global_dx_CPU;
    int dy = global_dy_CPU;
    int dz = global_dz_CPU;

    TYPE *Cell_Width = grid->Cell_Width;

    TYPE Range0_X[2], Range0_Y[2], Range0_Z[2];
    TYPE Range1_X[2], Range1_Y[2], Range1_Z[2];

    int start_processing_cell_index = global_starting_cell_index_R_CPU;
    int   end_processing_cell_index = global_ending_cell_index_R_CPU;

    //int start_processing_cell_index_prime = (start_processing_cell_index + threadid);

    unsigned long long int threshold_for_accumulate_sum = (1<<29); threshold_for_accumulate_sum = (threshold_for_accumulate_sum << 1) - 1;

    unsigned long long int local_start_time, local_end_time;

    local_start_time = ___rdtsc();

    long long int actual_sum = global_actual_sum_rr_CPU[16*threadid];
    long long int curr_accumulated_actual_sum = 0;

    unsigned int *Gather_Histogram0 = global_Gather_Histogram0_CPU[threadid];
    unsigned int *Gather_Histogram1 = global_Gather_Histogram1_CPU[threadid];
    unsigned int *RR_int0 = global_RR_int0_CPU[threadid];
    unsigned int *RR_int1 = global_RR_int1_CPU[threadid];

    TYPE *Pos1 = global_Pos1_CPU[threadid];
    TYPE *Bdry1_X = global_Bdry1_X_CPU[threadid];
    TYPE *Bdry1_Y = global_Bdry1_Y_CPU[threadid];
    TYPE *Bdry1_Z = global_Bdry1_Z_CPU[threadid];

    TYPE *BinCorners2 = global_BinCorners2_CPU;
    unsigned long long int *RR = local_Histogram_RR_CPU[threadid];

    int nrbin = global_nrbin_CPU;
    TYPE rmax_2 = global_rmax_2_CPU;

    int current_cell_index = global_starting_cell_index_R_CPU + taskid;
    //Perform work on current_cell_index...
    {
        int x, y, z; z = current_cell_index / dimxy; int xy = current_cell_index % dimxy; y = xy / dimx; x = xy % dimx;

        //if (current_cell_index % 1000 == 23) printf("threadid = %d ::: current_cell_index = %d\n", threadid, current_cell_index);

        if (!global_Required_R_CPU[current_cell_index]) ERROR_PRINT();



#if 0
        if (current_cell_index != 419) continue;
        //if (current_cell_index % 900 == 0)
        {
            printf("x = %d ::: y = %d ::: z = %d\n", x, y, z);
            printf("Z = %d ::: ", z);
            for(int k=0; k<12; k++)
            {
                long long int local_sum = DD[k];
                local_sum += DD_int0[k];
                local_sum += DD_int1[k];
                printf("%lld ", local_sum); 
            }

            printf("\n");
        }
#endif
        int objects_in_this_cell = grid->Count_Per_Cell[current_cell_index];
        int subdivisions_in_this_cell = grid->Number_of_kd_subdivisions[current_cell_index];
        int *Range0 = grid->Range[current_cell_index];
        TYPE *Bdry0_X = grid->Bdry_X[current_cell_index];
        TYPE *Bdry0_Y = grid->Bdry_Y[current_cell_index];
        TYPE *Bdry0_Z = grid->Bdry_Z[current_cell_index];

        //if (current_cell_index == debug_cell_id) printf("NODE_ID = %d ::: objects_in_this_cell (%d) = %d\n", node_id, debug_cell_id, objects_in_this_cell);
        TYPE *Pos0 = grid->Positions[current_cell_index];

        Range0_X[0] = (x*1.00)*Cell_Width[0]; Range0_X[1] = ((x+1)*1.00)*Cell_Width[0];
        Range0_Y[0] = (y*1.00)*Cell_Width[1]; Range0_Y[1] = ((y+1)*1.00)*Cell_Width[1];
        Range0_Z[0] = (z*1.00)*Cell_Width[2]; Range0_Z[1] = ((z+1)*1.00)*Cell_Width[2];

        long long int local_sum = ((long long int)(objects_in_this_cell) * (long long int)(objects_in_this_cell-1))/2;

        actual_sum += local_sum;
        curr_accumulated_actual_sum += local_sum;

        CPUFunction_Update_Histogram_Self_Cross(Pos0, objects_in_this_cell, Bdry0_X, Bdry0_Y, Bdry0_Z, Range0,  subdivisions_in_this_cell, 
                                                Pos0, objects_in_this_cell, Bdry0_X, Bdry0_Y, Bdry0_Z, Range0,  subdivisions_in_this_cell,
                                                Range0_X, Range0_Y, Range0_Z, 
                                                Range0_X, Range0_Y, Range0_Z, BinCorners2, RR_int0, RR_int1, Gather_Histogram0, Gather_Histogram1, nrbin, threadid);

        for(int zz = (z - dz); zz <= (z + dz); zz++)
        {
            for(int yy = (y - dy); yy <= (y + dy); yy++)
            {
                for(int xx = (x - dx); xx <= (x + dx); xx++)
                {
                    //Our neighbor is the (xx, yy, zz) cell...
                    if ((xx == x) && (yy == y) && (zz == z)) continue;

                    if ((xx == 30) && (yy == 13) && (zz == 0)) 
                    {
                        //printf("asffsdfs\n");
                    }
                            
                    ////////////////////////////////////////////////////////////////////////////////////////
                    //Step A: Figure out if the nearest points between the grids is >= rmax...
                    ////////////////////////////////////////////////////////////////////////////////////////
                            
                    Range1_X[0] = (xx*1.00)*Cell_Width[0]; Range1_X[1] = ((xx+1)*1.00)*Cell_Width[0];
                    Range1_Y[0] = (yy*1.00)*Cell_Width[1]; Range1_Y[1] = ((yy+1)*1.00)*Cell_Width[1];
                    Range1_Z[0] = (zz*1.00)*Cell_Width[2]; Range1_Z[1] = ((zz+1)*1.00)*Cell_Width[2];

                    TYPE min_dist_2 = CPUFunction_Find_Minimum_Distance_Between_Cells(Range0_X, Range0_Y, Range0_Z, Range1_X, Range1_Y, Range1_Z);

                    if (min_dist_2 > rmax_2) continue;

                    ////////////////////////////////////////////////////////////////////////////////////////
                    //Step B: Collect Particles into consecutive positions and also respect PREIODICITY...
                    ////////////////////////////////////////////////////////////////////////////////////////

                    int xx_prime = xx, yy_prime = yy, zz_prime = zz;
                    if (xx < 0) xx_prime = xx + dimx; else if (xx >= (dimx)) xx_prime = xx - dimx;
                    if (yy < 0) yy_prime = yy + dimy; else if (yy >= (dimy)) yy_prime = yy - dimy;
                    if (zz < 0) zz_prime = zz + dimz; else if (zz >= (dimz)) zz_prime = zz - dimz;

                    if ( (xx_prime < 0) || (xx_prime >= dimx))  ERROR_PRINT();
                    if ( (yy_prime < 0) || (yy_prime >= dimy))  ERROR_PRINT();
                    if ( (zz_prime < 0) || (zz_prime >= dimz))  ERROR_PRINT();

                    int neighbor_cell_index = GET_CELL_INDEX(xx_prime, yy_prime, zz_prime);
                    if (neighbor_cell_index > current_cell_index) continue; //XXX: Very important line...

                    if (!global_Required_R_CPU[neighbor_cell_index]) ERROR_PRINT();

                    int *Range1 = grid->Range[neighbor_cell_index];
                    int objects_in_neighboring_cell = grid->Count_Per_Cell[neighbor_cell_index];
                    int subdivisions_in_neighboring_cell = grid->Number_of_kd_subdivisions[neighbor_cell_index];

                    TYPE Delta[DIMENSIONS]; Delta[0] = Delta[1] = Delta[2] = 0.0;
                    if (xx < 0) Delta[0] = -Extent[0]; else if (xx >= (dimx)) Delta[0] = Extent[0];
                    if (yy < 0) Delta[1] = -Extent[1]; else if (yy >= (dimy)) Delta[1] = Extent[1];
                    if (zz < 0) Delta[2] = -Extent[2]; else if (zz >= (dimz)) Delta[2] = Extent[2];
                
                    ////////////////////////////////////////////////////////////////////////////////////////
                    //Step C: Copy Positions, Bdry too...
                    ////////////////////////////////////////////////////////////////////////////////////////
                            
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


                    ////////////////////////////////////////////////////////////////////////////////////////
                    //Step C: Now please perform computation without further ado...
                    ////////////////////////////////////////////////////////////////////////////////////////

                    long long int local_sum = ((long long int)(objects_in_this_cell) * ((long long int)(objects_in_neighboring_cell)));
                    actual_sum += local_sum;
                    curr_accumulated_actual_sum += local_sum;

                    CPUFunction_Update_Histogram_Self_Cross(Pos0, objects_in_this_cell,         Bdry0_X, Bdry0_Y, Bdry0_Z, Range0, subdivisions_in_this_cell,
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
                            //printf("node_id = %d ::: threadid = %d :: curr_accumulated_actual_sum = %lld ::: ttt = %lld\n", node_id, threadid, curr_accumulated_actual_sum, ttt);
                            ERROR_PRINT();
                        }

                        curr_accumulated_actual_sum = 0;
                        for(int i=0; i<=(1+nrbin); i++) RR[i] += RR_int0[i];
                        for(int i=0; i<=(1+nrbin); i++) RR[i] += RR_int1[i];
                        for(int i=0; i<=(1+nrbin); i++) RR_int0[i] = 0;
                        for(int i=0; i<=(1+nrbin); i++) RR_int1[i] = 0;

#if 0
                        for(int bin_id = 0; bin_id <= (1+nrbin); bin_id++)
                        {
                            for(int lane = 0; lane < SIMD_WIDTH_CPU; lane++)
                            {
                                RR[bin_id] += Gather_Histogram0[bin_id*SIMD_WIDTH_CPU + lane];
                                RR[bin_id] += Gather_Histogram1[bin_id*SIMD_WIDTH_CPU + lane];
                            }
                        }

                        for(int i=0; i<(SIMD_WIDTH_CPU*HIST_BINS); i++) Gather_Histogram0[i] = 0;
                        for(int i=0; i<(SIMD_WIDTH_CPU*HIST_BINS); i++) Gather_Histogram1[i] = 0;
#endif
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

#if 0
        for(int bin_id = 0; bin_id <= (1+nrbin); bin_id++)
        {
            for(int lane = 0; lane < SIMD_WIDTH_CPU; lane++)
            {
                RR[bin_id] += Gather_Histogram0[bin_id*SIMD_WIDTH_CPU + lane];
                RR[bin_id] += Gather_Histogram1[bin_id*SIMD_WIDTH_CPU + lane];
            }
        }
                
        for(int i=0; i<(SIMD_WIDTH_CPU*HIST_BINS); i++) Gather_Histogram0[i] = 0;
        for(int i=0; i<(SIMD_WIDTH_CPU*HIST_BINS); i++) Gather_Histogram1[i] = 0;
#endif
    }

    global_actual_sum_rr_CPU[16*threadid] = actual_sum;

    local_end_time = ___rdtsc();
    global_time_per_thread_rr_CPU[threadid] += local_end_time - local_start_time;
}


__attribute__ (( target (mic))) 
void  MICFunction_Perform_RR_Helper( void *arg)
{

     int threadid = omp_get_thread_num();
     int taskid   = (int)((size_t)(arg));
     //printf("threadid = %d ::: taskid = %d\n", threadid, taskid);

    //printf("(%d) :: taskid = %d\n", threadid, taskid);

    Grid *grid = &global_grid_R_MIC;

    TYPE *Extent = grid->Extent;

    int dimx = grid->dimx;
    int dimy = grid->dimy;
    int dimz = grid->dimz;

    int dimxy = dimx * dimy;

    int dx = global_dx_MIC;
    int dy = global_dy_MIC;
    int dz = global_dz_MIC;

    TYPE *Cell_Width = grid->Cell_Width;

    TYPE Range0_X[2], Range0_Y[2], Range0_Z[2];
    TYPE Range1_X[2], Range1_Y[2], Range1_Z[2];

    int start_processing_cell_index = global_starting_cell_index_R_MIC;
    int   end_processing_cell_index = global_ending_cell_index_R_MIC;

    //int start_processing_cell_index_prime = (start_processing_cell_index + threadid);

    unsigned long long int threshold_for_accumulate_sum = (1<<29); threshold_for_accumulate_sum = (threshold_for_accumulate_sum << 1) - 1;

    unsigned long long int local_start_time, local_end_time;

    local_start_time = ___rdtsc();

    long long int actual_sum = global_actual_sum_rr_MIC[16*threadid];
    long long int curr_accumulated_actual_sum = 0;

    unsigned int *Gather_Histogram0 = global_Gather_Histogram0_MIC[threadid];
    unsigned int *Gather_Histogram1 = global_Gather_Histogram1_MIC[threadid];
    unsigned int *RR_int0 = global_RR_int0_MIC[threadid];
    unsigned int *RR_int1 = global_RR_int1_MIC[threadid];

    TYPE *Pos1 = global_Pos1_MIC[threadid];
    TYPE *Bdry1_X = global_Bdry1_X_MIC[threadid];
    TYPE *Bdry1_Y = global_Bdry1_Y_MIC[threadid];
    TYPE *Bdry1_Z = global_Bdry1_Z_MIC[threadid];

    TYPE *BinCorners2 = global_BinCorners2_MIC;
    unsigned long long int *RR = local_Histogram_RR_MIC[threadid];

    int nrbin = global_nrbin_MIC;
    TYPE rmax_2 = global_rmax_2_MIC;

    int current_cell_index = global_starting_cell_index_R_MIC + taskid;
    //Perform work on current_cell_index...
    {
        int x, y, z; z = current_cell_index / dimxy; int xy = current_cell_index % dimxy; y = xy / dimx; x = xy % dimx;

        //if (current_cell_index % 1000 == 23) printf("threadid = %d ::: current_cell_index = %d\n", threadid, current_cell_index);

        if (!global_Required_R_MIC[current_cell_index]) ERROR_PRINT();



#if 0
        if (current_cell_index != 419) continue;
        //if (current_cell_index % 900 == 0)
        {
            printf("x = %d ::: y = %d ::: z = %d\n", x, y, z);
            printf("Z = %d ::: ", z);
            for(int k=0; k<12; k++)
            {
                long long int local_sum = DD[k];
                local_sum += DD_int0[k];
                local_sum += DD_int1[k];
                printf("%lld ", local_sum); 
            }

            printf("\n");
        }
#endif
        int objects_in_this_cell = grid->Count_Per_Cell[current_cell_index];
        int subdivisions_in_this_cell = grid->Number_of_kd_subdivisions[current_cell_index];
        int *Range0 = grid->Range[current_cell_index];
        TYPE *Bdry0_X = grid->Bdry_X[current_cell_index];
        TYPE *Bdry0_Y = grid->Bdry_Y[current_cell_index];
        TYPE *Bdry0_Z = grid->Bdry_Z[current_cell_index];

        //if (current_cell_index == debug_cell_id) printf("NODE_ID = %d ::: objects_in_this_cell (%d) = %d\n", node_id, debug_cell_id, objects_in_this_cell);
        TYPE *Pos0 = grid->Positions[current_cell_index];

        Range0_X[0] = (x*1.00)*Cell_Width[0]; Range0_X[1] = ((x+1)*1.00)*Cell_Width[0];
        Range0_Y[0] = (y*1.00)*Cell_Width[1]; Range0_Y[1] = ((y+1)*1.00)*Cell_Width[1];
        Range0_Z[0] = (z*1.00)*Cell_Width[2]; Range0_Z[1] = ((z+1)*1.00)*Cell_Width[2];

        long long int local_sum = ((long long int)(objects_in_this_cell) * (long long int)(objects_in_this_cell-1))/2;

        actual_sum += local_sum;
        curr_accumulated_actual_sum += local_sum;

        MICFunction_Update_Histogram_Self_Cross(Pos0, objects_in_this_cell, Bdry0_X, Bdry0_Y, Bdry0_Z, Range0,  subdivisions_in_this_cell, 
                                                Pos0, objects_in_this_cell, Bdry0_X, Bdry0_Y, Bdry0_Z, Range0,  subdivisions_in_this_cell,
                                                Range0_X, Range0_Y, Range0_Z, 
                                                Range0_X, Range0_Y, Range0_Z, BinCorners2, RR_int0, RR_int1, Gather_Histogram0, Gather_Histogram1, nrbin, threadid);

        int ccounter = 0;

        for(int zz = (z - dz); zz <= (z + dz); zz++)
        {
            for(int yy = (y - dy); yy <= (y + dy); yy++)
            {
                int yy_prime = yy;
                int zz_prime = zz;

                if (yy < 0) yy_prime = yy + dimy; else if (yy >= (dimy)) yy_prime = yy - dimy;
                if (zz < 0) zz_prime = zz + dimz; else if (zz >= (dimz)) zz_prime = zz - dimz;
                int base_cell_index = GET_CELL_INDEX(0, yy_prime, zz_prime);

                for(int xx = (x - dx); xx <= (x + dx); xx++)
                {
                        
                    if (!global_Template_during_hetero_RR_MIC[ccounter++]) continue;
#if 0
                    //Our neighbor is the (xx, yy, zz) cell...
                    if ((xx == x) && (yy == y) && (zz == z)) continue;

                    if ((xx == 30) && (yy == 13) && (zz == 0)) 
                    {
                        //printf("asffsdfs\n");
                    }
                            
                    ////////////////////////////////////////////////////////////////////////////////////////
                    //Step A: Figure out if the nearest points between the grids is >= rmax...
                    ////////////////////////////////////////////////////////////////////////////////////////
                            
                    Range1_X[0] = (xx*1.00)*Cell_Width[0]; Range1_X[1] = ((xx+1)*1.00)*Cell_Width[0];
                    Range1_Y[0] = (yy*1.00)*Cell_Width[1]; Range1_Y[1] = ((yy+1)*1.00)*Cell_Width[1];
                    Range1_Z[0] = (zz*1.00)*Cell_Width[2]; Range1_Z[1] = ((zz+1)*1.00)*Cell_Width[2];

                    TYPE min_dist_2 = MICFunction_Find_Minimum_Distance_Between_Cells(Range0_X, Range0_Y, Range0_Z, Range1_X, Range1_Y, Range1_Z);

                    if (min_dist_2 > rmax_2) continue;
#else
#endif

                    ////////////////////////////////////////////////////////////////////////////////////////
                    //Step B: Collect Particles into consecutive positions and also respect PREIODICITY...
                    ////////////////////////////////////////////////////////////////////////////////////////

#if 0
                    int xx_prime = xx, yy_prime = yy, zz_prime = zz;
                    if (xx < 0) xx_prime = xx + dimx; else if (xx >= (dimx)) xx_prime = xx - dimx;
                    if (yy < 0) yy_prime = yy + dimy; else if (yy >= (dimy)) yy_prime = yy - dimy;
                    if (zz < 0) zz_prime = zz + dimz; else if (zz >= (dimz)) zz_prime = zz - dimz;

                    if ( (xx_prime < 0) || (xx_prime >= dimx))  ERROR_PRINT();
                    if ( (yy_prime < 0) || (yy_prime >= dimy))  ERROR_PRINT();
                    if ( (zz_prime < 0) || (zz_prime >= dimz))  ERROR_PRINT();

                    int neighbor_cell_index = GET_CELL_INDEX(xx_prime, yy_prime, zz_prime);
#else

                    int xx_prime = xx;
                    if (xx < 0) xx_prime = xx + dimx; else if (xx >= (dimx)) xx_prime = xx - dimx;
                    int neighbor_cell_index = base_cell_index + xx_prime;
#endif

                    if (neighbor_cell_index > current_cell_index) continue; //XXX: Very important line...

                    if (!global_Required_R_MIC[neighbor_cell_index]) ERROR_PRINT();

                    Range1_X[0] = (xx*1.00)*Cell_Width[0]; Range1_X[1] = ((xx+1)*1.00)*Cell_Width[0];
                    Range1_Y[0] = (yy*1.00)*Cell_Width[1]; Range1_Y[1] = ((yy+1)*1.00)*Cell_Width[1];
                    Range1_Z[0] = (zz*1.00)*Cell_Width[2]; Range1_Z[1] = ((zz+1)*1.00)*Cell_Width[2];

                    int *Range1 = grid->Range[neighbor_cell_index];
                    int objects_in_neighboring_cell = grid->Count_Per_Cell[neighbor_cell_index];
                    int subdivisions_in_neighboring_cell = grid->Number_of_kd_subdivisions[neighbor_cell_index];

                    TYPE Delta[DIMENSIONS]; Delta[0] = Delta[1] = Delta[2] = 0.0;
                    if (xx < 0) Delta[0] = -Extent[0]; else if (xx >= (dimx)) Delta[0] = Extent[0];
                    if (yy < 0) Delta[1] = -Extent[1]; else if (yy >= (dimy)) Delta[1] = Extent[1];
                    if (zz < 0) Delta[2] = -Extent[2]; else if (zz >= (dimz)) Delta[2] = Extent[2];
                
                    ////////////////////////////////////////////////////////////////////////////////////////
                    //Step C: Copy Positions, Bdry too...
                    ////////////////////////////////////////////////////////////////////////////////////////
                            
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


                    ////////////////////////////////////////////////////////////////////////////////////////
                    //Step C: Now please perform computation without further ado...
                    ////////////////////////////////////////////////////////////////////////////////////////

                    long long int local_sum = ((long long int)(objects_in_this_cell) * ((long long int)(objects_in_neighboring_cell)));
                    actual_sum += local_sum;
                    curr_accumulated_actual_sum += local_sum;

                    MICFunction_Update_Histogram_Self_Cross(Pos0, objects_in_this_cell,         Bdry0_X, Bdry0_Y, Bdry0_Z, Range0, subdivisions_in_this_cell,
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
                            //printf("node_id = %d ::: threadid = %d :: curr_accumulated_actual_sum = %lld ::: ttt = %lld\n", node_id, threadid, curr_accumulated_actual_sum, ttt);
                            ERROR_PRINT();
                        }

                        curr_accumulated_actual_sum = 0;
                        for(int i=0; i<=(1+nrbin); i++) RR[i] += RR_int0[i];
                        for(int i=0; i<=(1+nrbin); i++) RR[i] += RR_int1[i];
                        for(int i=0; i<=(1+nrbin); i++) RR_int0[i] = 0;
                        for(int i=0; i<=(1+nrbin); i++) RR_int1[i] = 0;

#if 0
                        for(int bin_id = 0; bin_id <= (1+nrbin); bin_id++)
                        {
                            for(int lane = 0; lane < SIMD_WIDTH_MIC; lane++)
                            {
                                RR[bin_id] += Gather_Histogram0[bin_id*SIMD_WIDTH_MIC + lane];
                                RR[bin_id] += Gather_Histogram1[bin_id*SIMD_WIDTH_MIC + lane];
                            }
                        }

                        for(int i=0; i<(SIMD_WIDTH_MIC*HIST_BINS); i++) Gather_Histogram0[i] = 0;
                        for(int i=0; i<(SIMD_WIDTH_MIC*HIST_BINS); i++) Gather_Histogram1[i] = 0;
#endif
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

#if 0
        for(int bin_id = 0; bin_id <= (1+nrbin); bin_id++)
        {
            for(int lane = 0; lane < SIMD_WIDTH_MIC; lane++)
            {
                RR[bin_id] += Gather_Histogram0[bin_id*SIMD_WIDTH_MIC + lane];
                RR[bin_id] += Gather_Histogram1[bin_id*SIMD_WIDTH_MIC + lane];
            }
        }
                
        for(int i=0; i<(SIMD_WIDTH_MIC*HIST_BINS); i++) Gather_Histogram0[i] = 0;
        for(int i=0; i<(SIMD_WIDTH_MIC*HIST_BINS); i++) Gather_Histogram1[i] = 0;
#endif
    }

    global_actual_sum_rr_MIC[16*threadid] = actual_sum;

    local_end_time = ___rdtsc();
    global_time_per_thread_rr_MIC[threadid] += local_end_time - local_start_time;
}


__attribute__ (( target (mic))) 
void MICFunction_Compute_Statistics_DR(void)
{
    //All nodes are going to call this function...
    int threadid = 0;
    int nrbin = global_nrbin_MIC;
    TYPE *Rminarr = global_Rminarr_MIC;
    TYPE *Rmaxarr = global_Rmaxarr_MIC;

    {
        int ngal = global_number_of_galaxies_MIC;
        //for(int j=0; j<HIST_BINS; j++) global_Histogram[j] = 0;
        //int ngal = global_number_of_galaxies;

        //global_stat_total_interactions_dr = 0;
        for(int i=0; i<nthreads_MIC; i++)
        {
            //global_stat_total_interactions_dr += global_actual_sum_dr[16*i];
            for(int j=0; j<HIST_BINS; j++) 
            {
                global_Histogram_DR_MIC[j] += local_Histogram_DR_MIC[i][j];
            }
        }

        //printf(">>>>node_id = %d ::: global_Histogram[1] = %lld\n", node_id, global_Histogram[1]);
        //printf(">>>>node_id = %d ::: global_Histogram[2] = %lld\n", node_id, global_Histogram[2]);
        //printf(">>>>node_id = %d ::: global_Histogram[3] = %lld\n", node_id, global_Histogram[3]);

        
#if 0
        MPI_Gather(global_Histogram_DR, HIST_BINS, MPI_LONG_LONG_INT, global_Overall_Histogram_DR, HIST_BINS, MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);
#else
        for(int j=0; j<HIST_BINS; j++) global_Overall_Histogram_DR_MIC[j] =  global_Histogram_DR_MIC[j];
#endif

        //if (node_id == 0)
        {
            for(int j=0; j<HIST_BINS; j++) global_Histogram_DR_MIC[j] = 0;

            for(int k=0; k<1; k++)
            {
                for(int j=0; j<HIST_BINS; j++)
                {
                    global_Histogram_DR_MIC[j] += global_Overall_Histogram_DR_MIC[k * HIST_BINS + j];
                }
            }

            global_stat_total_interactions_dr_MIC = 0; for(int j = 0; j < HIST_BINS; j++) global_stat_total_interactions_dr_MIC += global_Histogram_DR_MIC[j];

            //We have in some sense DR now... Let's now compute DR_over_RR...

            /////NOTE:::: In our implementation,  the total number of //interacting pairs is nC2, while in Hemant's original //implementation it is n*n...
      
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

                //printf("bin_id = %d ::: rmin = %e ::: rmax = %e ::: denom = %e\n", bin_id, rmin, rmax, denominator);

                global_DR_over_RR_MIC[bin_id+1] = global_Histogram_DR_MIC[bin_id+1]/denominator;
            }
        }
    }

    if (threadid == 0)
    {
        unsigned long long int sum_time = 0;
        unsigned long long int min_time = global_time_per_thread_dr_MIC[0];
        unsigned long long int max_time = global_time_per_thread_dr_MIC[0];

        for(int kk=0; kk<nthreads_MIC; kk++)
        {
            sum_time += global_time_per_thread_dr_MIC[kk];
            min_time = PCL_MIN(min_time, global_time_per_thread_dr_MIC[kk]);
            max_time = PCL_MAX(max_time, global_time_per_thread_dr_MIC[kk]);
        }

        unsigned long long int avg_time = sum_time/nthreads_MIC;
        printf("DR :: <<%d>> Avg. Time = %lld ::: Max Time = %lld ::: Ratio = %.4lf\n", node_id_MIC, avg_time, max_time, (max_time * 1.00)/avg_time);
    }

    long long int my_easy_sum = 0;
    if (threadid == 0)
    {

        for(int kk=0; kk<nthreads_MIC; kk++)
        {
            my_easy_sum += global_Easy_MIC[8*kk];
        }

        //printf("<<%d>> ::: global_easy_sum = %lld\n", node_id, easy_sum);
    }

    if (threadid == 0)
    {
        //long long int *Overall_global_Easy = (long long int *)malloc(1 * sizeof(long long int));
#if 0
        MPI_Gather(&my_easy_sum, 1, MPI_LONG_LONG_INT, Overall_global_Easy, 1, MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);
#else
        //Overall_global_Easy[0] = my_easy_sum;
#endif

        global_accumulated_easy_MIC = my_easy_sum;
        //for(int k = 0; k < nnodes; k++) global_accumulated_easy += Overall_global_Easy[k];

        PRINT_RED
        printf("<<%d>> ::: global_accumulated_easy_MIC = %lld\n", node_id_MIC, global_accumulated_easy_MIC);
        PRINT_BLACK
    }
}


__attribute__ (( target (mic))) 
void MICFunction_Compute_Statistics_RR(void)
{
    //All nodes are going to call this function...
    int threadid = 0;
    int nrbin = global_nrbin_MIC;
    TYPE *Rminarr = global_Rminarr_MIC;
    TYPE *Rmaxarr = global_Rmaxarr_MIC;

    {
        //for(int j=0; j<HIST_BINS; j++) global_Histogram[j] = 0;
        int ngal = global_number_of_galaxies_MIC;
        //global_stat_total_interactions_rr = 0;
        for(int i=0; i<nthreads_MIC; i++)
        {
            //global_stat_total_interactions_rr += global_actual_sum_rr[16*i];
            for(int j=0; j<HIST_BINS; j++) 
            {
                global_Histogram_RR_MIC[j] += local_Histogram_RR_MIC[i][j];
            }
        }

        //printf(">>>>node_id = %d ::: global_Histogram[1] = %lld\n", node_id, global_Histogram[1]);
        //printf(">>>>node_id = %d ::: global_Histogram[2] = %lld\n", node_id, global_Histogram[2]);
        //printf(">>>>node_id = %d ::: global_Histogram[3] = %lld\n", node_id, global_Histogram[3]);

        
#if 0
        MPI_Gather(global_Histogram_RR, HIST_BINS, MPI_LONG_LONG_INT, global_Overall_Histogram_RR, HIST_BINS, MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);
#else
        for(int j=0; j<HIST_BINS; j++) global_Overall_Histogram_RR_MIC[j] = global_Histogram_RR_MIC[j];
#endif

        //if (node_id == 0)
        {
            for(int j=0; j<HIST_BINS; j++) global_Histogram_RR_MIC[j] = 0;

            for(int k=0; k<1; k++)
            {
                for(int j=0; j<HIST_BINS; j++)
                {
                    global_Histogram_RR_MIC[j] += global_Overall_Histogram_RR_MIC[k * HIST_BINS + j];
                }
            }

            global_stat_total_interactions_rr_MIC = 0; for(int j = 0; j < HIST_BINS; j++) global_stat_total_interactions_rr_MIC += global_Histogram_RR_MIC[j];
            //We have in some sense DD now... Let's now compute DD_over_RR...

            /////NOTE:::: In our implementation,  the total number of //interacting pairs is nC2, while in Hemant's original //implementation it is n*n...
      
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

            if (node_id_MIC == 0) printf("bin_id = %d ::: rmin = %e ::: rmax = %e ::: denom = %e\n", bin_id, rmin, rmax, denominator);

            global_RR_over_RR_MIC[bin_id+1] = global_Histogram_RR_MIC[bin_id+1]/denominator;
            }
        }
    }

    if (threadid == 0)
    {
        unsigned long long int sum_time = 0;
        unsigned long long int min_time = global_time_per_thread_rr_MIC[0];
        unsigned long long int max_time = global_time_per_thread_rr_MIC[0];

        for(int kk=0; kk<nthreads_MIC; kk++)
        {
            sum_time += global_time_per_thread_rr_MIC[kk];
            min_time = PCL_MIN(min_time, global_time_per_thread_rr_MIC[kk]);
            max_time = PCL_MAX(max_time, global_time_per_thread_rr_MIC[kk]);
        }

        unsigned long long int avg_time = sum_time/nthreads_MIC;
        printf("RR :: <<%d>> Avg. Time = %lld ::: Max Time = %lld ::: Ratio = %.4lf\n", node_id_MIC, avg_time, max_time, (max_time * 1.00)/avg_time);
    }
}

void CPUFunction_Compute_Statistics_DR(void)
{
    //All nodes are going to call this function...
    int threadid = 0;
    int nrbin = global_nrbin_CPU;
    TYPE *Rminarr = global_Rminarr_CPU;
    TYPE *Rmaxarr = global_Rmaxarr_CPU;

    {
        int ngal = global_number_of_galaxies_CPU;
        //for(int j=0; j<HIST_BINS; j++) global_Histogram[j] = 0;
        //int ngal = global_number_of_galaxies;

        //global_stat_total_interactions_dr = 0;
        for(int i=0; i<nthreads_CPU; i++)
        {
            //global_stat_total_interactions_dr += global_actual_sum_dr[16*i];
            for(int j=0; j<HIST_BINS; j++) 
            {
                global_Histogram_DR_CPU[j] += local_Histogram_DR_CPU[i][j];
            }
        }

        //printf(">>>>node_id = %d ::: global_Histogram[1] = %lld\n", node_id, global_Histogram[1]);
        //printf(">>>>node_id = %d ::: global_Histogram[2] = %lld\n", node_id, global_Histogram[2]);
        //printf(">>>>node_id = %d ::: global_Histogram[3] = %lld\n", node_id, global_Histogram[3]);

        
#if 0
        MPI_Gather(global_Histogram_DR, HIST_BINS, MPI_LONG_LONG_INT, global_Overall_Histogram_DR, HIST_BINS, MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);
#else
        for(int j=0; j<HIST_BINS; j++) global_Overall_Histogram_DR_CPU[j] =  global_Histogram_DR_CPU[j];
#endif

        //if (node_id == 0)
        {
            for(int j=0; j<HIST_BINS; j++) global_Histogram_DR_CPU[j] = 0;

            for(int k=0; k<1; k++)
            {
                for(int j=0; j<HIST_BINS; j++)
                {
                    global_Histogram_DR_CPU[j] += global_Overall_Histogram_DR_CPU[k * HIST_BINS + j];
                }
            }

            global_stat_total_interactions_dr_CPU = 0; for(int j = 0; j < HIST_BINS; j++) global_stat_total_interactions_dr_CPU += global_Histogram_DR_CPU[j];

            //We have in some sense DR now... Let's now compute DR_over_RR...

            /////NOTE:::: In our implementation,  the total number of //interacting pairs is nC2, while in Hemant's original //implementation it is n*n...
      
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

                //printf("bin_id = %d ::: rmin = %e ::: rmax = %e ::: denom = %e\n", bin_id, rmin, rmax, denominator);

                global_DR_over_RR_CPU[bin_id+1] = global_Histogram_DR_CPU[bin_id+1]/denominator;
            }
        }
    }

    if (threadid == 0)
    {
        unsigned long long int sum_time = 0;
        unsigned long long int min_time = global_time_per_thread_dr_CPU[0];
        unsigned long long int max_time = global_time_per_thread_dr_CPU[0];

        for(int kk=0; kk<nthreads_CPU; kk++)
        {
            sum_time += global_time_per_thread_dr_CPU[kk];
            min_time = PCL_MIN(min_time, global_time_per_thread_dr_CPU[kk]);
            max_time = PCL_MAX(max_time, global_time_per_thread_dr_CPU[kk]);
        }

        unsigned long long int avg_time = sum_time/nthreads_CPU;
        printf("DR :: <<%d>> Avg. Time = %lld ::: Max Time = %lld ::: Ratio = %.4lf\n", node_id_CPU, avg_time, max_time, (max_time * 1.00)/avg_time);
    }

    long long int my_easy_sum = 0;
    if (threadid == 0)
    {

        for(int kk=0; kk<nthreads_CPU; kk++)
        {
            my_easy_sum += global_Easy_CPU[8*kk];
        }

        //printf("<<%d>> ::: global_easy_sum = %lld\n", node_id, easy_sum);
    }

    if (threadid == 0)
    {
        //long long int *Overall_global_Easy = (long long int *)malloc(1 * sizeof(long long int));
#if 0
        MPI_Gather(&my_easy_sum, 1, MPI_LONG_LONG_INT, Overall_global_Easy, 1, MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);
#else
        //Overall_global_Easy[0] = my_easy_sum;
#endif

        global_accumulated_easy_CPU = my_easy_sum;
        //for(int k = 0; k < nnodes; k++) global_accumulated_easy += Overall_global_Easy[k];

        PRINT_RED
        printf("<<%d>> ::: global_accumulated_easy_CPU = %lld\n", node_id_CPU, global_accumulated_easy_CPU);
        PRINT_BLACK
    }
}


void CPUFunction_Compute_Statistics_RR(void)
{
    //All nodes are going to call this function...
    int threadid = 0;
    int nrbin = global_nrbin_CPU;
    TYPE *Rminarr = global_Rminarr_CPU;
    TYPE *Rmaxarr = global_Rmaxarr_CPU;

    {
        //for(int j=0; j<HIST_BINS; j++) global_Histogram[j] = 0;
        int ngal = global_number_of_galaxies_CPU;
        //global_stat_total_interactions_rr = 0;
        for(int i=0; i<nthreads_CPU; i++)
        {
            //global_stat_total_interactions_rr += global_actual_sum_rr[16*i];
            for(int j=0; j<HIST_BINS; j++) 
            {
                global_Histogram_RR_CPU[j] += local_Histogram_RR_CPU[i][j];
            }
        }

        //printf(">>>>node_id = %d ::: global_Histogram[1] = %lld\n", node_id, global_Histogram[1]);
        //printf(">>>>node_id = %d ::: global_Histogram[2] = %lld\n", node_id, global_Histogram[2]);
        //printf(">>>>node_id = %d ::: global_Histogram[3] = %lld\n", node_id, global_Histogram[3]);

        
#if 0
        MPI_Gather(global_Histogram_RR, HIST_BINS, MPI_LONG_LONG_INT, global_Overall_Histogram_RR, HIST_BINS, MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);
#else
        for(int j=0; j<HIST_BINS; j++) global_Overall_Histogram_RR_CPU[j] = global_Histogram_RR_CPU[j];
#endif

        //if (node_id == 0)
        {
            for(int j=0; j<HIST_BINS; j++) global_Histogram_RR_CPU[j] = 0;

            for(int k=0; k<1; k++)
            {
                for(int j=0; j<HIST_BINS; j++)
                {
                    global_Histogram_RR_CPU[j] += global_Overall_Histogram_RR_CPU[k * HIST_BINS + j];
                }
            }

            global_stat_total_interactions_rr_CPU = 0; for(int j = 0; j < HIST_BINS; j++) global_stat_total_interactions_rr_CPU += global_Histogram_RR_CPU[j];
            //We have in some sense DD now... Let's now compute DD_over_RR...

            /////NOTE:::: In our implementation,  the total number of //interacting pairs is nC2, while in Hemant's original //implementation it is n*n...
      
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

            if (node_id_CPU == 0) printf("bin_id = %d ::: rmin = %e ::: rmax = %e ::: denom = %e\n", bin_id, rmin, rmax, denominator);

            global_RR_over_RR_CPU[bin_id+1] = global_Histogram_RR_CPU[bin_id+1]/denominator;
            }
        }
    }

    if (threadid == 0)
    {
        unsigned long long int sum_time = 0;
        unsigned long long int min_time = global_time_per_thread_rr_CPU[0];
        unsigned long long int max_time = global_time_per_thread_rr_CPU[0];

        for(int kk=0; kk<nthreads_CPU; kk++)
        {
            sum_time += global_time_per_thread_rr_CPU[kk];
            min_time = PCL_MIN(min_time, global_time_per_thread_rr_CPU[kk]);
            max_time = PCL_MAX(max_time, global_time_per_thread_rr_CPU[kk]);
        }

        unsigned long long int avg_time = sum_time/nthreads_CPU;
        printf("RR :: <<%d>> Avg. Time = %lld ::: Max Time = %lld ::: Ratio = %.4lf\n", node_id_CPU, avg_time, max_time, (max_time * 1.00)/avg_time);
    }
}


void ParseArgs(int argc, char **argv)
{
    if (argc != 4)
    {
        printf("Usage ./a.out <Dir_Name> <nthreads> <nnodes> \n");
        exit(123);
    }

    sscanf(argv[1], "%s", global_dirname);
    sscanf(argv[2], "%d", &nthreads_CPU);
    sscanf(argv[3], "%d", &nnodes_CPU);

    if (nthreads_CPU > MAX_THREADS_CPU) ERROR_PRINT_STRING("nthreads_CPU > MAX_THREADS_CPU");
    if (node_id_CPU == 0) printf("nnodes_CPU = %d\n", nnodes_CPU);
    if (node_id_CPU == 0) printf("nthreads_CPU = %d\n", nthreads_CPU);
    //if (node_id_CPU == 0) printf("HETERO_THRESHOLD = %f\n", HETERO_THRESHOLD);

//Paramters are set here -- so that they can be used later on...
    global_Lbox_CPU  = 100.0;
    global_rminL_CPU = 0.1;
    global_rmaxL_CPU = 10.0;
    global_nrbin_CPU = 10;

    int hist_bins = global_nrbin_CPU + 2;

    if ((hist_bins) != HIST_BINS)
    {
        ERROR_PRINT_STRING("Please change HIST_BINS or global_nrbin_CPU");
    }
}


void CPUFunction_Perform_Mandatory_Initializations(Grid *grid, TYPE i_Lbox, TYPE i_rminL, TYPE i_rmaxL, int nrbin)
{
    //XXX: Only 1 thread calls this function... This is evident from //the fact that there is no threadid passed to this function -- //but still :)
    int function_called_how_many_times = 0;
    function_called_how_many_times++;


    if (function_called_how_many_times > 1) 
    {
        //XXX: Can be called multiple times... A little bit of memory leak is there...
        //ERROR_PRINT();
        //for(int threadid = 0; threadid < nthreads_CPU; threadid++) global_actual_sum[16*threadid] = 0;
        //return;
    }

    TYPE *Extent = grid->Extent;

    if (function_called_how_many_times  == 1)
    {
        global_actual_sum_rr_CPU = (long long int *)malloc(16 * nthreads_CPU * sizeof(long long int));
        for(int threadid = 0; threadid < nthreads_CPU; threadid++) global_actual_sum_rr_CPU[16*threadid] = 0;

        global_actual_sum_dr_CPU = (long long int *)malloc(16 * nthreads_CPU * sizeof(long long int));
        for(int threadid = 0; threadid < nthreads_CPU; threadid++) global_actual_sum_dr_CPU[16*threadid] = 0;
    }

    //IMHERE();

    int dimx = grid->dimx;
    int dimy = grid->dimy;
    int dimz = grid->dimz;

    int number_of_uniform_subdivisions = grid->number_of_uniform_subdivisions;

    int maximum_number_of_particles = 0;
    int maximum_number_of_kd_subdivisions = 0;

    //if (node_id == 0) printf(" <<%d>> : number_of_uniform_subdivisions = %d\n", node_id, number_of_uniform_subdivisions);
    for(int cell_id = 0; cell_id < number_of_uniform_subdivisions; cell_id++)
    {
        if (global_Required_R_CPU[cell_id])
        {
            int number_of_kd_subdivisions = grid->Number_of_kd_subdivisions[cell_id];
            int number_of_particles = grid->Range[cell_id][number_of_kd_subdivisions];
            maximum_number_of_kd_subdivisions = PCL_MAX(maximum_number_of_kd_subdivisions, number_of_kd_subdivisions);
            maximum_number_of_particles = PCL_MAX(maximum_number_of_particles, number_of_particles);
        }
    }

    //IMHERE();
    int dimxy = dimx * dimy;

    if ((nrbin+2) != HIST_BINS)
    {
        ERROR_PRINT_STRING("Please change HIST_BINS or global_nrbin");
    }

    //IMHERE();

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

    global_rmax_2_CPU = rmax_2;

    global_Gather_Histogram0_CPU = (unsigned int **)malloc(nthreads_CPU * sizeof(unsigned int *));
    global_Gather_Histogram1_CPU = (unsigned int **)malloc(nthreads_CPU * sizeof(unsigned int *));
    global_RR_int0_CPU           = (unsigned int **)malloc(nthreads_CPU * sizeof(unsigned int *));
    global_RR_int1_CPU           = (unsigned int **)malloc(nthreads_CPU * sizeof(unsigned int *));
    global_DR_int0_CPU           = (unsigned int **)malloc(nthreads_CPU * sizeof(unsigned int *));
    global_DR_int1_CPU           = (unsigned int **)malloc(nthreads_CPU * sizeof(unsigned int *));
    global_Pos1_CPU              = (TYPE **)malloc(nthreads_CPU * sizeof(TYPE *));
    global_Bdry1_X_CPU           = (TYPE **)malloc(nthreads_CPU * sizeof(TYPE *));
    global_Bdry1_Y_CPU           = (TYPE **)malloc(nthreads_CPU * sizeof(TYPE *));
    global_Bdry1_Z_CPU           = (TYPE **)malloc(nthreads_CPU * sizeof(TYPE *));

    for(int threadid = 0; threadid < nthreads_CPU; threadid++)
    {
        int hist_bins_prime = (((HIST_BINS + 32) >> 5)<<5);
        size_t sz = 0;

        sz  += (SIMD_WIDTH_CPU * HIST_BINS) * sizeof(unsigned int);
        sz  += (SIMD_WIDTH_CPU * HIST_BINS) * sizeof(unsigned int);
        sz += (hist_bins_prime) * sizeof(int);
        sz += (hist_bins_prime) * sizeof(int);
        sz += (hist_bins_prime) * sizeof(int);
        sz += (hist_bins_prime) * sizeof(int);
        sz += maximum_number_of_particles * DIMENSIONS * sizeof(TYPE);
        sz += maximum_number_of_kd_subdivisions * 2 * sizeof(TYPE);
        sz += maximum_number_of_kd_subdivisions * 2 * sizeof(TYPE);
        sz += maximum_number_of_kd_subdivisions * 2 * sizeof(TYPE);


        unsigned char *temp_memory = (unsigned char *)malloc(sz);
        unsigned char *temp_memory2 = temp_memory;

        global_Gather_Histogram0_CPU[threadid] = (unsigned int *)(temp_memory2); temp_memory2 += (SIMD_WIDTH_CPU * HIST_BINS) * sizeof(unsigned int);
        global_Gather_Histogram1_CPU[threadid] = (unsigned int *)(temp_memory2); temp_memory2 += (SIMD_WIDTH_CPU * HIST_BINS) * sizeof(unsigned int);
        for(int i=0; i<(SIMD_WIDTH_CPU*HIST_BINS); i++) global_Gather_Histogram0_CPU[threadid][i] = 0;
        for(int i=0; i<(SIMD_WIDTH_CPU*HIST_BINS); i++) global_Gather_Histogram1_CPU[threadid][i] = 0;

        global_RR_int0_CPU[threadid] = (unsigned int *)(temp_memory2); temp_memory2 += (hist_bins_prime) * sizeof(int);
        global_RR_int1_CPU[threadid] = (unsigned int *)(temp_memory2); temp_memory2 += (hist_bins_prime) * sizeof(int);
        global_DR_int0_CPU[threadid] = (unsigned int *)(temp_memory2); temp_memory2 += (hist_bins_prime) * sizeof(int);
        global_DR_int1_CPU[threadid] = (unsigned int *)(temp_memory2); temp_memory2 += (hist_bins_prime) * sizeof(int);
        for(int i=0; i<=(1+nrbin); i++) global_RR_int0_CPU[threadid][i]  = 0;
        for(int i=0; i<=(1+nrbin); i++) global_RR_int1_CPU[threadid][i]  = 0;
        for(int i=0; i<=(1+nrbin); i++) global_DR_int0_CPU[threadid][i]  = 0;
        for(int i=0; i<=(1+nrbin); i++) global_DR_int1_CPU[threadid][i]  = 0;

        global_Pos1_CPU[threadid]     = (TYPE *)(temp_memory2); temp_memory2 += maximum_number_of_particles * DIMENSIONS * sizeof(TYPE);
        global_Bdry1_X_CPU[threadid]  = (TYPE *)(temp_memory2); temp_memory2 += maximum_number_of_kd_subdivisions * 2 * sizeof(TYPE);
        global_Bdry1_Y_CPU[threadid]  = (TYPE *)(temp_memory2); temp_memory2 += maximum_number_of_kd_subdivisions * 2 * sizeof(TYPE);
        global_Bdry1_Z_CPU[threadid]  = (TYPE *)(temp_memory2); temp_memory2 += maximum_number_of_kd_subdivisions * 2 * sizeof(TYPE);

        if ((temp_memory2 - temp_memory) != sz) ERROR_PRINT();
#if 0
        unsigned long long int *DD = local_Histogram[threadid];
        for(int i=0; i<=(1+nrbin); i++) DD[i] = 0;
#endif

        //IMHERE();
    }

#if 1
    //IMHERE();
    {
        int threadid = 0;
        size_t sz = 0;

        sz += (nrbin * sizeof(TYPE));
        sz += (nrbin * sizeof(TYPE));
        sz += (nrbin * sizeof(TYPE));
        sz += (HIST_BINS * sizeof(TYPE));
        sz += (HIST_BINS * sizeof(TYPE));

    //IMHERE();
        unsigned char *temp_memory = (unsigned char *)malloc(sz);
    //IMHERE();
        unsigned char *temp2_memory = temp_memory;
    //IMHERE();

        //IMHERE();

        TYPE lrmin = log(rmin);
        TYPE lrmax = log(rmax);

        TYPE dlnr = (lrmax - lrmin)/nrbin;

        if ((node_id_CPU == 0) && (threadid == 0))  printf("rmin = %f ::: rmax = %f ::: lrmin = %f ::: lrmax = %f ::: dlnr = %f\n", rmin, rmax, lrmin, lrmax, dlnr);
        //IMHERE();

        global_Rminarr_CPU     = (TYPE *)(temp2_memory); temp2_memory += (nrbin * sizeof(TYPE));
        global_Rmaxarr_CPU     = (TYPE *)(temp2_memory); temp2_memory += (nrbin * sizeof(TYPE));
        global_Rval_CPU        = (TYPE *)(temp2_memory); temp2_memory += (nrbin * sizeof(TYPE));
        global_BinCorners_CPU  = (TYPE *)(temp2_memory); temp2_memory += (HIST_BINS * sizeof(TYPE));
        global_BinCorners2_CPU = (TYPE *)(temp2_memory); temp2_memory += (HIST_BINS * sizeof(TYPE));

        if ((temp2_memory - temp_memory) != sz) ERROR_PRINT();

        TYPE *Rminarr = global_Rminarr_CPU;
        TYPE *Rmaxarr = global_Rmaxarr_CPU;
        TYPE *Rval = global_Rval_CPU;
        TYPE *BinCorners = global_BinCorners_CPU;
        TYPE *BinCorners2 = global_BinCorners2_CPU;
    
        //Equally dividing in the 'log' scale...

        for(int i=1; i<nrbin; i++)     Rminarr[i] = exp(lrmin + i*dlnr); Rminarr[0] = rmin;
        for(int i=0; i<(nrbin-1); i++) Rmaxarr[i] = Rminarr[i+1];  Rmaxarr[nrbin-1] = rmax;
        for(int i=0; i<nrbin; i++)        Rval[i] = exp(lrmin + (i+0.5)*dlnr); 

        for(int i=0; i<(nrbin); i++) BinCorners[i] = Rminarr[i] * Rminarr[i];
        for(int i=(nrbin); i<(HIST_BINS); i++) BinCorners[i] = FLT_MAX; //Some large number...
        BinCorners2[0] = -FLT_MAX; for(int i=0; i<(1+nrbin); i++) BinCorners2[i+1] = BinCorners[i]; BinCorners2[nrbin+1] = rmax*rmax;

        int dx = int(ceil((i_rmaxL/i_Lbox)*dimx)) + 0;
        int dy = int(ceil((i_rmaxL/i_Lbox)*dimy)) + 0;
        int dz = int(ceil((i_rmaxL/i_Lbox)*dimz)) + 0;


        global_dx_CPU = dx;
        global_dy_CPU = dy;
        global_dz_CPU = dz;
    }
#endif
}


int Check_For_MIC(void)
{
#if 1
    // Check if coprocessor(s) are installed and available
    int num_devices = _Offload_number_of_devices();
    printf("rank: %d ::: Offload num_devices = %d\n", node_id_CPU, num_devices);
    if (num_devices == 0) 
    {
        printf("*** FAIL TPCF -- target unavailable\n");
        return 1;
    }
#endif
    return 0;
}


/*
 *
 * Code written on 08/30/2012... Starting from memory transfers...
 * 
 */

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
size_t DataTransfer_Size_D_From_CPU_To_MIC_CPU;
size_t DataTransfer_Size_R_From_CPU_To_MIC_CPU;

int *Temp_Memory_D_CPU;
int *Temp_Memory_R_CPU;


__attribute__ (( target (mic))) size_t DataTransfer_Size_D_From_CPU_To_MIC_MIC;
__attribute__ (( target (mic))) size_t DataTransfer_Size_R_From_CPU_To_MIC_MIC;

__attribute__ (( target (mic))) int *Temp_Memory_D_MIC;
__attribute__ (( target (mic))) int *Temp_Memory_R_MIC;

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

__attribute__ (( target (mic))) 
void  MICFunction_Spit_Output(Grid *grid, size_t Data_Transfer_Size, int *i_Temp_Memory, const char *filename)
{
    FILE *fp = fopen(filename, "wb");
    if (fp == NULL) ERROR_PRINT();

    PRINT_LIGHT_RED
    printf("Spitting out %s\n", filename);
    PRINT_BLACK


    fwrite(i_Temp_Memory, Data_Transfer_Size, 1, fp);
    fclose(fp);
}
       
__attribute__ (( target (mic))) 
void  MICFunction_Copy_Temp_Memory_To_D_Or_R(Grid *grid, size_t Data_Transfer_Size, int *Temp_Memory)
{
    unsigned long long int stime = ___rdtsc();
    size_t sz, sz1, sz2, sz3;
    sz = *((size_t *)(Temp_Memory + 0)); if (sz != Data_Transfer_Size) ERROR_PRINT();
    sz1 = *((size_t *)(Temp_Memory + 2));
    sz2 = *((size_t *)(Temp_Memory + 4));
    sz3 = *((size_t *)(Temp_Memory + 6));

    int dimx, dimy, dimz, number_of_uniform_subdivisions;

    dimx = *(Temp_Memory + 8 + 0);
    dimy = *(Temp_Memory + 8 + 1);
    dimz = *(Temp_Memory + 8 + 2);
    number_of_uniform_subdivisions = *(Temp_Memory + 8 + 3);

    grid->dimx = dimx;
    grid->dimy = dimy;
    grid->dimz = dimz;
    grid->number_of_uniform_subdivisions  = number_of_uniform_subdivisions;

    grid->Cell_Width[0] = *((TYPE *)(Temp_Memory + 12 + 0));
    grid->Cell_Width[1] = *((TYPE *)(Temp_Memory + 12 + 1)); 
    grid->Cell_Width[2] = *((TYPE *)(Temp_Memory + 12 + 2)); 
                                                            
    grid->Min[0]        = *((TYPE *)(Temp_Memory + 12 + 3)); 
    grid->Min[1]        = *((TYPE *)(Temp_Memory + 12 + 4));
    grid->Min[2]        = *((TYPE *)(Temp_Memory + 12 + 5));
                                                           
    grid->Max[0]        = *((TYPE *)(Temp_Memory + 12 + 6));
    grid->Max[1]        = *((TYPE *)(Temp_Memory + 12 + 7));
    grid->Max[2]        = *((TYPE *)(Temp_Memory + 12 + 8));

    grid->Extent[0] = grid->Max[0] - grid->Min[0];
    grid->Extent[1] = grid->Max[1] - grid->Min[1];
    grid->Extent[2] = grid->Max[2] - grid->Min[2];

    int *CPC = Temp_Memory + 21;
#if 0
    grid->Count_Per_Cell = (int *)malloc(number_of_uniform_subdivisions * sizeof(int));
    for(int p = 0; p < number_of_uniform_subdivisions; p++) grid->Count_Per_Cell[p] = CPC[p];
#else
    grid->Count_Per_Cell = CPC;
#endif

        
    int *KDS = CPC + number_of_uniform_subdivisions;
#if 0
    grid->Number_of_kd_subdivisions = (int *)malloc(number_of_uniform_subdivisions * sizeof(int));
    for(int p = 0; p < number_of_uniform_subdivisions; p++) grid->Number_of_kd_subdivisions[p] = KDS[p];
#else
    grid->Number_of_kd_subdivisions = KDS;
#endif

    int *S3_Start = KDS + number_of_uniform_subdivisions;

    int range_fields = S3_Start[0];
    int total_number_of_kd_tree_nodes = S3_Start[1];
    size_t total_number_of_particles = *((size_t *)(S3_Start + 2));


    int *Range_Dst = S3_Start + 4;
    TYPE *Bdry_X = (TYPE *)(Range_Dst + range_fields);
    TYPE *Bdry_Y = Bdry_X + 2 * total_number_of_kd_tree_nodes;
    TYPE *Bdry_Z = Bdry_Y + 2 * total_number_of_kd_tree_nodes;
    TYPE *Pos    = Bdry_Z + 2 * total_number_of_kd_tree_nodes;
    unsigned char *X = (unsigned char *)(Pos + 3 * total_number_of_particles);
    unsigned char *Y = X + number_of_uniform_subdivisions;
    unsigned char *Z = Y + number_of_uniform_subdivisions;

    grid->Range     = (int  **)malloc(number_of_uniform_subdivisions * sizeof(int *));
    grid->Bdry_X    = (TYPE **)malloc(number_of_uniform_subdivisions * sizeof(TYPE *));
    grid->Bdry_Y    = (TYPE **)malloc(number_of_uniform_subdivisions * sizeof(TYPE *));
    grid->Bdry_Z    = (TYPE **)malloc(number_of_uniform_subdivisions * sizeof(TYPE *));
    grid->Positions = (TYPE **)malloc(number_of_uniform_subdivisions * sizeof(TYPE *));


#if 0
    TYPE *Actual_Bdry_X = (TYPE *)malloc(2 * total_number_of_kd_tree_nodes * sizeof(TYPE));
    TYPE *Actual_Bdry_Y = (TYPE *)malloc(2 * total_number_of_kd_tree_nodes * sizeof(TYPE));
    TYPE *Actual_Bdry_Z = (TYPE *)malloc(2 * total_number_of_kd_tree_nodes * sizeof(TYPE));
    int *Actual_Range   = (int  *)malloc(range_fields * sizeof(int));
    TYPE *Actual_Pos    = (TYPE *)malloc(3 * total_number_of_particles  * sizeof(TYPE));

    for(int p = 0; p < 2 * total_number_of_kd_tree_nodes; p++) Actual_Bdry_X[p] = Bdry_X[p];
    for(int p = 0; p < 2 * total_number_of_kd_tree_nodes; p++) Actual_Bdry_Y[p] = Bdry_Y[p];
    for(int p = 0; p < 2 * total_number_of_kd_tree_nodes; p++) Actual_Bdry_Z[p] = Bdry_Z[p];
    for(int p = 0; p <     range_fields; p++) Actual_Range [p] = Range_Dst[p];
    for(size_t q = 0; q < 3 * total_number_of_particles; q++) Actual_Pos[q] = Pos[q];
#else
    TYPE *Actual_Bdry_X = Bdry_X;
    TYPE *Actual_Bdry_Y = Bdry_Y;
    TYPE *Actual_Bdry_Z = Bdry_Z;
    int *Actual_Range = Range_Dst;
    TYPE *Actual_Pos = Pos;
#endif


    int range_fields_counter = 0;
    int total_number_of_kd_tree_nodes_counter = 0;
    size_t total_number_of_particles_counter = 0;

    for(int k = 0; k < number_of_uniform_subdivisions; k++)
    {
        if (grid->Number_of_kd_subdivisions[k])
        {
            grid->Range[k] = Actual_Range + range_fields_counter;
            range_fields_counter += (1 + grid->Number_of_kd_subdivisions[k]);

            grid->Bdry_X[k] = Actual_Bdry_X + 2 * total_number_of_kd_tree_nodes_counter;
            grid->Bdry_Y[k] = Actual_Bdry_Y + 2 * total_number_of_kd_tree_nodes_counter;
            grid->Bdry_Z[k] = Actual_Bdry_Z + 2 * total_number_of_kd_tree_nodes_counter;
            total_number_of_kd_tree_nodes_counter += grid->Number_of_kd_subdivisions[k];

            grid->Positions[k] = Actual_Pos + 3 * total_number_of_particles_counter;

            total_number_of_particles_counter += grid->Count_Per_Cell[k];
        }
        else
        {
            grid->Range[k] = NULL;
            grid->Bdry_X[k] = NULL;
            grid->Bdry_Y[k] = NULL;
            grid->Bdry_Z[k] = NULL;
            grid->Positions[k] = NULL;
        }
    }

#if 0
    global_Required_D_For_R = (unsigned char *)malloc(number_of_uniform_subdivisions * sizeof(unsigned char));
    global_Required_R       = (unsigned char *)malloc(number_of_uniform_subdivisions * sizeof(unsigned char));

    for(int k = 0; k < number_of_uniform_subdivisions; k++) global_Required_D_For_R[k] = X[k];
    for(int k = 0; k < number_of_uniform_subdivisions; k++) global_Required_R[k] = Y[k];
#else
    global_Required_D_For_R_MIC = X;
    global_Required_R_MIC = Y;
#endif


    global_starting_cell_index_D_MIC = *((int *)(Z +  0));
    global_ending_cell_index_D_MIC   = *((int *)(Z +  4));
    global_starting_cell_index_R_MIC = *((int *)(Z +  8));
    global_ending_cell_index_R_MIC   = *((int *)(Z + 12));

    if (range_fields_counter != range_fields) ERROR_PRINT_STRING("range_fields_counter != range_fields");
    if (total_number_of_kd_tree_nodes_counter != total_number_of_kd_tree_nodes) ERROR_PRINT_STRING("total_number_of_kd_tree_nodes_counter != total_number_of_kd_tree_nodes");
    if (total_number_of_particles_counter != total_number_of_particles) ERROR_PRINT_STRING("total_number_of_particles_counter != total_number_of_particles");

    unsigned long long int etime = ___rdtsc();
    unsigned long long int ttime = etime - stime;
    {
        double seconds = ttime/CORE_FREQUENCY_MIC;
        PRINT_LIGHT_RED
        if (node_id_MIC == 0) printf("<<%d>> : Time Taken To Read (%lld bytes) = %lld cycles (%.2lf seconds) ::: %.2lf GB/sec\n", node_id_MIC, sz, ttime, seconds, sz/seconds/1000.0/1000.0/1000.0);
        PRINT_BLACK
    }
}
       
void CPUFunction_Copy_Temp_Memory_To_D_Or_R(Grid *grid, size_t Data_Transfer_Size, int *Temp_Memory)
{
    unsigned long long int stime = ___rdtsc();
    size_t sz, sz1, sz2, sz3;
    sz = *((size_t *)(Temp_Memory + 0)); if (sz != Data_Transfer_Size) ERROR_PRINT();
    sz1 = *((size_t *)(Temp_Memory + 2));
    sz2 = *((size_t *)(Temp_Memory + 4));
    sz3 = *((size_t *)(Temp_Memory + 6));

    int dimx, dimy, dimz, number_of_uniform_subdivisions;

    dimx = *(Temp_Memory + 8 + 0);
    dimy = *(Temp_Memory + 8 + 1);
    dimz = *(Temp_Memory + 8 + 2);
    number_of_uniform_subdivisions = *(Temp_Memory + 8 + 3);

    grid->dimx = dimx;
    grid->dimy = dimy;
    grid->dimz = dimz;
    grid->number_of_uniform_subdivisions  = number_of_uniform_subdivisions;

    grid->Cell_Width[0] = *((TYPE *)(Temp_Memory + 12 + 0));
    grid->Cell_Width[1] = *((TYPE *)(Temp_Memory + 12 + 1)); 
    grid->Cell_Width[2] = *((TYPE *)(Temp_Memory + 12 + 2)); 
                                                            
    grid->Min[0]        = *((TYPE *)(Temp_Memory + 12 + 3)); 
    grid->Min[1]        = *((TYPE *)(Temp_Memory + 12 + 4));
    grid->Min[2]        = *((TYPE *)(Temp_Memory + 12 + 5));
                                                           
    grid->Max[0]        = *((TYPE *)(Temp_Memory + 12 + 6));
    grid->Max[1]        = *((TYPE *)(Temp_Memory + 12 + 7));
    grid->Max[2]        = *((TYPE *)(Temp_Memory + 12 + 8));

    grid->Extent[0] = grid->Max[0] - grid->Min[0];
    grid->Extent[1] = grid->Max[1] - grid->Min[1];
    grid->Extent[2] = grid->Max[2] - grid->Min[2];

    int *CPC = Temp_Memory + 21;
#if 0
    grid->Count_Per_Cell = (int *)malloc(number_of_uniform_subdivisions * sizeof(int));
    for(int p = 0; p < number_of_uniform_subdivisions; p++) grid->Count_Per_Cell[p] = CPC[p];
#else
    grid->Count_Per_Cell = CPC;
#endif

        
    int *KDS = CPC + number_of_uniform_subdivisions;
#if 0
    grid->Number_of_kd_subdivisions = (int *)malloc(number_of_uniform_subdivisions * sizeof(int));
    for(int p = 0; p < number_of_uniform_subdivisions; p++) grid->Number_of_kd_subdivisions[p] = KDS[p];
#else
    grid->Number_of_kd_subdivisions = KDS;
#endif

    int *S3_Start = KDS + number_of_uniform_subdivisions;

    int range_fields = S3_Start[0];
    int total_number_of_kd_tree_nodes = S3_Start[1];
    size_t total_number_of_particles = *((size_t *)(S3_Start + 2));


    int *Range_Dst = S3_Start + 4;
    TYPE *Bdry_X = (TYPE *)(Range_Dst + range_fields);
    TYPE *Bdry_Y = Bdry_X + 2 * total_number_of_kd_tree_nodes;
    TYPE *Bdry_Z = Bdry_Y + 2 * total_number_of_kd_tree_nodes;
    TYPE *Pos    = Bdry_Z + 2 * total_number_of_kd_tree_nodes;
    unsigned char *X = (unsigned char *)(Pos + 3 * total_number_of_particles);
    unsigned char *Y = X + number_of_uniform_subdivisions;
    unsigned char *Z = Y + number_of_uniform_subdivisions;

    grid->Range     = (int  **)malloc(number_of_uniform_subdivisions * sizeof(int *));
    grid->Bdry_X    = (TYPE **)malloc(number_of_uniform_subdivisions * sizeof(TYPE *));
    grid->Bdry_Y    = (TYPE **)malloc(number_of_uniform_subdivisions * sizeof(TYPE *));
    grid->Bdry_Z    = (TYPE **)malloc(number_of_uniform_subdivisions * sizeof(TYPE *));
    grid->Positions = (TYPE **)malloc(number_of_uniform_subdivisions * sizeof(TYPE *));


#if 0
    TYPE *Actual_Bdry_X = (TYPE *)malloc(2 * total_number_of_kd_tree_nodes * sizeof(TYPE));
    TYPE *Actual_Bdry_Y = (TYPE *)malloc(2 * total_number_of_kd_tree_nodes * sizeof(TYPE));
    TYPE *Actual_Bdry_Z = (TYPE *)malloc(2 * total_number_of_kd_tree_nodes * sizeof(TYPE));
    int *Actual_Range   = (int  *)malloc(range_fields * sizeof(int));
    TYPE *Actual_Pos    = (TYPE *)malloc(3 * total_number_of_particles  * sizeof(TYPE));

    for(int p = 0; p < 2 * total_number_of_kd_tree_nodes; p++) Actual_Bdry_X[p] = Bdry_X[p];
    for(int p = 0; p < 2 * total_number_of_kd_tree_nodes; p++) Actual_Bdry_Y[p] = Bdry_Y[p];
    for(int p = 0; p < 2 * total_number_of_kd_tree_nodes; p++) Actual_Bdry_Z[p] = Bdry_Z[p];
    for(int p = 0; p <     range_fields; p++) Actual_Range [p] = Range_Dst[p];
    for(size_t q = 0; q < 3 * total_number_of_particles; q++) Actual_Pos[q] = Pos[q];
#else
    TYPE *Actual_Bdry_X = Bdry_X;
    TYPE *Actual_Bdry_Y = Bdry_Y;
    TYPE *Actual_Bdry_Z = Bdry_Z;
    int *Actual_Range = Range_Dst;
    TYPE *Actual_Pos = Pos;
#endif


    int range_fields_counter = 0;
    int total_number_of_kd_tree_nodes_counter = 0;
    size_t total_number_of_particles_counter = 0;

    for(int k = 0; k < number_of_uniform_subdivisions; k++)
    {
        if (grid->Number_of_kd_subdivisions[k])
        {
            grid->Range[k] = Actual_Range + range_fields_counter;
            range_fields_counter += (1 + grid->Number_of_kd_subdivisions[k]);

            grid->Bdry_X[k] = Actual_Bdry_X + 2 * total_number_of_kd_tree_nodes_counter;
            grid->Bdry_Y[k] = Actual_Bdry_Y + 2 * total_number_of_kd_tree_nodes_counter;
            grid->Bdry_Z[k] = Actual_Bdry_Z + 2 * total_number_of_kd_tree_nodes_counter;
            total_number_of_kd_tree_nodes_counter += grid->Number_of_kd_subdivisions[k];

            grid->Positions[k] = Actual_Pos + 3 * total_number_of_particles_counter;

            total_number_of_particles_counter += grid->Count_Per_Cell[k];
        }
        else
        {
            grid->Range[k] = NULL;
            grid->Bdry_X[k] = NULL;
            grid->Bdry_Y[k] = NULL;
            grid->Bdry_Z[k] = NULL;
            grid->Positions[k] = NULL;
        }
    }

#if 0
    global_Required_D_For_R = (unsigned char *)malloc(number_of_uniform_subdivisions * sizeof(unsigned char));
    global_Required_R       = (unsigned char *)malloc(number_of_uniform_subdivisions * sizeof(unsigned char));

    for(int k = 0; k < number_of_uniform_subdivisions; k++) global_Required_D_For_R[k] = X[k];
    for(int k = 0; k < number_of_uniform_subdivisions; k++) global_Required_R[k] = Y[k];
#else
    global_Required_D_For_R_CPU = X;
    global_Required_R_CPU = Y;
#endif


    global_starting_cell_index_D_CPU = *((int *)(Z +  0));
    global_ending_cell_index_D_CPU   = *((int *)(Z +  4));
    global_starting_cell_index_R_CPU = *((int *)(Z +  8));
    global_ending_cell_index_R_CPU   = *((int *)(Z + 12));

    if (range_fields_counter != range_fields) ERROR_PRINT_STRING("range_fields_counter != range_fields");
    if (total_number_of_kd_tree_nodes_counter != total_number_of_kd_tree_nodes) ERROR_PRINT_STRING("total_number_of_kd_tree_nodes_counter != total_number_of_kd_tree_nodes");
    if (total_number_of_particles_counter != total_number_of_particles) ERROR_PRINT_STRING("total_number_of_particles_counter != total_number_of_particles");

    unsigned long long int etime = ___rdtsc();
    unsigned long long int ttime = etime - stime;
    {
        double seconds = ttime/CORE_FREQUENCY_CPU;
        PRINT_LIGHT_RED
        if (node_id_CPU == 0) printf("<<%d>> : Time Taken To Read (%lld bytes) = %lld cycles (%.2lf seconds) ::: %.2lf GB/sec\n", node_id_CPU, sz, ttime, seconds, sz/seconds/1000.0/1000.0/1000.0);
        PRINT_BLACK
    }
}

void Populate_Temp_Memory(int **Temp_Memory, size_t *DataTransfer_Size, const char *filename)
{
    FILE *fp = fopen(filename, "rb"); if (fp == NULL) { printf("File (%s) not found\n", filename); ERROR_PRINT();}

    fread(DataTransfer_Size, sizeof(size_t), 1, fp);
    //*Temp_Memory = (int *)malloc((*DataTransfer_Size));
    *Temp_Memory = (int *)_mm_malloc((*DataTransfer_Size), (2*1024*1024));
    fseek(fp, 0, SEEK_SET);
    fread((*Temp_Memory), (*DataTransfer_Size), 1, fp);

    fclose(fp);
    PRINT_LIGHT_RED
    printf("Completely Read  %s\n", filename);
    PRINT_BLACK
}


void Load_Temp_Memory_From_File(void)
{
    Populate_Temp_Memory(&Temp_Memory_D_CPU, &DataTransfer_Size_D_From_CPU_To_MIC_CPU, "jch_D.bin");
    Populate_Temp_Memory(&Temp_Memory_R_CPU, &DataTransfer_Size_R_From_CPU_To_MIC_CPU, "jch_R.bin");
}


__attribute__ (( target (mic))) 
void MICFunction_Parse_Global_Variables(unsigned char *Global_Variables)
{
#ifdef __MIC__
    TYPE *X = (TYPE *)(Global_Variables);
    global_Lbox_MIC = X[0];
    global_rminL_MIC = X[1];
    global_rmaxL_MIC = X[2];

    int *Y = (int *)(Global_Variables + 3*sizeof(TYPE));
    global_nrbin_MIC = Y[0];
    node_id_MIC = Y[1];
    nnodes_MIC = Y[2];
    global_number_of_galaxies_MIC = *((long long int *)(Global_Variables + 24));
    
    global_hetero_cpu_number_of_D_cells_MIC = *((int *)(Global_Variables + 32));
    global_hetero_cpu_number_of_R_cells_MIC = *((int *)(Global_Variables + 36));

    {
        TYPE i_rminL = global_rminL_MIC;
        TYPE i_Lbox  = global_Lbox_MIC;
        TYPE i_rmaxL = global_rmaxL_MIC;

        int dimx = global_grid_D_MIC.dimx;
        int dimy = global_grid_D_MIC.dimy;
        int dimz = global_grid_D_MIC.dimz;
    
        int dx = int(ceil((i_rmaxL/i_Lbox)*dimx)) + 0;
        int dy = int(ceil((i_rmaxL/i_Lbox)*dimy)) + 0;
        int dz = int(ceil((i_rmaxL/i_Lbox)*dimz)) + 0;

        int cells_in_stencil = (2*dx +1) * (2*dy + 1) * (2*dz + 1);

        global_Template_during_hetero_RR_MIC = (unsigned char *)malloc(cells_in_stencil * sizeof(unsigned char));
        global_Template_during_hetero_DR_MIC = (unsigned char *)malloc(cells_in_stencil * sizeof(unsigned char));
    
        for(int p = 0; p < cells_in_stencil; p++) global_Template_during_hetero_RR_MIC[p] = Global_Variables[40 + 0*cells_in_stencil + p];
        for(int p = 0; p < cells_in_stencil; p++) global_Template_during_hetero_DR_MIC[p] = Global_Variables[40 + 1*cells_in_stencil + p];
    }

    printf("global_Lbox_MIC = %f ::: global_rminL_MIC = %f ::: global_rmaxL_MIC = %f ::: global_nrbin_MIC = %d\n", global_Lbox_MIC, global_rminL_MIC, global_rmaxL_MIC, global_nrbin_MIC);
#endif
}



__attribute__ (( target (mic))) 
void MICFunction_Perform_Mandatory_Initializations(Grid *grid, TYPE i_Lbox, TYPE i_rminL, TYPE i_rmaxL, int nrbin)
{
    //XXX: Only 1 thread calls this function... This is evident from //the fact that there is no threadid passed to this function -- //but still :)
    int function_called_how_many_times = 0;
    function_called_how_many_times++;


    if (function_called_how_many_times > 1) 
    {
        //XXX: Can be called multiple times... A little bit of memory leak is there...
        //ERROR_PRINT();
        //for(int threadid = 0; threadid < nthreads_CPU; threadid++) global_actual_sum[16*threadid] = 0;
        //return;
    }

    TYPE *Extent = grid->Extent;

    if (function_called_how_many_times  == 1)
    {
        global_actual_sum_rr_MIC = (long long int *)malloc(16 * nthreads_MIC * sizeof(long long int));
        for(int threadid = 0; threadid < nthreads_MIC; threadid++) global_actual_sum_rr_MIC[16*threadid] = 0;

        global_actual_sum_dr_MIC = (long long int *)malloc(16 * nthreads_MIC * sizeof(long long int));
        for(int threadid = 0; threadid < nthreads_MIC; threadid++) global_actual_sum_dr_MIC[16*threadid] = 0;
    }

    //IMHERE();

    int dimx = grid->dimx;
    int dimy = grid->dimy;
    int dimz = grid->dimz;

    int number_of_uniform_subdivisions = grid->number_of_uniform_subdivisions;

    int maximum_number_of_particles = 0;
    int maximum_number_of_kd_subdivisions = 0;

    //if (node_id == 0) printf(" <<%d>> : number_of_uniform_subdivisions = %d\n", node_id, number_of_uniform_subdivisions);
    for(int cell_id = 0; cell_id < number_of_uniform_subdivisions; cell_id++)
    {
        if (global_Required_R_MIC[cell_id])
        {
            int number_of_kd_subdivisions = grid->Number_of_kd_subdivisions[cell_id];
            int number_of_particles = grid->Range[cell_id][number_of_kd_subdivisions];
            maximum_number_of_kd_subdivisions = PCL_MAX(maximum_number_of_kd_subdivisions, number_of_kd_subdivisions);
            maximum_number_of_particles = PCL_MAX(maximum_number_of_particles, number_of_particles);
        }
    }

    //IMHERE();
    int dimxy = dimx * dimy;

    if ((nrbin+2) != HIST_BINS)
    {
        ERROR_PRINT_STRING("Please change HIST_BINS or global_nrbin");
    }

    //IMHERE();

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

    global_rmax_2_MIC = rmax_2;

    global_Gather_Histogram0_MIC = (unsigned int **)malloc(nthreads_MIC * sizeof(unsigned int *));
    global_Gather_Histogram1_MIC = (unsigned int **)malloc(nthreads_MIC * sizeof(unsigned int *));
    global_RR_int0_MIC           = (unsigned int **)malloc(nthreads_MIC * sizeof(unsigned int *));
    global_RR_int1_MIC           = (unsigned int **)malloc(nthreads_MIC * sizeof(unsigned int *));
    global_DR_int0_MIC           = (unsigned int **)malloc(nthreads_MIC * sizeof(unsigned int *));
    global_DR_int1_MIC           = (unsigned int **)malloc(nthreads_MIC * sizeof(unsigned int *));
    global_Pos1_MIC              = (TYPE **)malloc(nthreads_MIC * sizeof(TYPE *));
    global_Bdry1_X_MIC           = (TYPE **)malloc(nthreads_MIC * sizeof(TYPE *));
    global_Bdry1_Y_MIC           = (TYPE **)malloc(nthreads_MIC * sizeof(TYPE *));
    global_Bdry1_Z_MIC           = (TYPE **)malloc(nthreads_MIC * sizeof(TYPE *));

    for(int threadid = 0; threadid < nthreads_MIC; threadid++)
    {
        int hist_bins_prime = (((HIST_BINS + 32) >> 5)<<5);
        size_t sz = 0;

        sz  += (SIMD_WIDTH_MIC * HIST_BINS) * sizeof(unsigned int);
        sz  += (SIMD_WIDTH_MIC * HIST_BINS) * sizeof(unsigned int);
        sz += (hist_bins_prime) * sizeof(int);
        sz += (hist_bins_prime) * sizeof(int);
        sz += (hist_bins_prime) * sizeof(int);
        sz += (hist_bins_prime) * sizeof(int);
        sz += maximum_number_of_particles * DIMENSIONS * sizeof(TYPE);
        sz += maximum_number_of_kd_subdivisions * 2 * sizeof(TYPE);
        sz += maximum_number_of_kd_subdivisions * 2 * sizeof(TYPE);
        sz += maximum_number_of_kd_subdivisions * 2 * sizeof(TYPE);


        unsigned char *temp_memory = (unsigned char *)malloc(sz);
        unsigned char *temp_memory2 = temp_memory;

        global_Gather_Histogram0_MIC[threadid] = (unsigned int *)(temp_memory2); temp_memory2 += (SIMD_WIDTH_MIC * HIST_BINS) * sizeof(unsigned int);
        global_Gather_Histogram1_MIC[threadid] = (unsigned int *)(temp_memory2); temp_memory2 += (SIMD_WIDTH_MIC * HIST_BINS) * sizeof(unsigned int);
        for(int i=0; i<(SIMD_WIDTH_MIC*HIST_BINS); i++) global_Gather_Histogram0_MIC[threadid][i] = 0;
        for(int i=0; i<(SIMD_WIDTH_MIC*HIST_BINS); i++) global_Gather_Histogram1_MIC[threadid][i] = 0;

        global_RR_int0_MIC[threadid] = (unsigned int *)(temp_memory2); temp_memory2 += (hist_bins_prime) * sizeof(int);
        global_RR_int1_MIC[threadid] = (unsigned int *)(temp_memory2); temp_memory2 += (hist_bins_prime) * sizeof(int);
        global_DR_int0_MIC[threadid] = (unsigned int *)(temp_memory2); temp_memory2 += (hist_bins_prime) * sizeof(int);
        global_DR_int1_MIC[threadid] = (unsigned int *)(temp_memory2); temp_memory2 += (hist_bins_prime) * sizeof(int);
        for(int i=0; i<=(1+nrbin); i++) global_RR_int0_MIC[threadid][i]  = 0;
        for(int i=0; i<=(1+nrbin); i++) global_RR_int1_MIC[threadid][i]  = 0;
        for(int i=0; i<=(1+nrbin); i++) global_DR_int0_MIC[threadid][i]  = 0;
        for(int i=0; i<=(1+nrbin); i++) global_DR_int1_MIC[threadid][i]  = 0;

        global_Pos1_MIC[threadid]     = (TYPE *)(temp_memory2); temp_memory2 += maximum_number_of_particles * DIMENSIONS * sizeof(TYPE);
        global_Bdry1_X_MIC[threadid]  = (TYPE *)(temp_memory2); temp_memory2 += maximum_number_of_kd_subdivisions * 2 * sizeof(TYPE);
        global_Bdry1_Y_MIC[threadid]  = (TYPE *)(temp_memory2); temp_memory2 += maximum_number_of_kd_subdivisions * 2 * sizeof(TYPE);
        global_Bdry1_Z_MIC[threadid]  = (TYPE *)(temp_memory2); temp_memory2 += maximum_number_of_kd_subdivisions * 2 * sizeof(TYPE);

        if ((temp_memory2 - temp_memory) != sz) ERROR_PRINT();
#if 0
        unsigned long long int *DD = local_Histogram[threadid];
        for(int i=0; i<=(1+nrbin); i++) DD[i] = 0;
#endif

        //IMHERE();
    }

#if 1
    //IMHERE();
    {
        int threadid = 0;
        size_t sz = 0;

        sz += (nrbin * sizeof(TYPE));
        sz += (nrbin * sizeof(TYPE));
        sz += (nrbin * sizeof(TYPE));
        sz += (HIST_BINS * sizeof(TYPE));
        sz += (HIST_BINS * sizeof(TYPE));

    //IMHERE();
        unsigned char *temp_memory = (unsigned char *)malloc(sz);
    //IMHERE();
        unsigned char *temp2_memory = temp_memory;
    //IMHERE();

        //IMHERE();

        TYPE lrmin = log(rmin);
        TYPE lrmax = log(rmax);

        TYPE dlnr = (lrmax - lrmin)/nrbin;

        if ((node_id_MIC == 0) && (threadid == 0))  printf("rmin = %f ::: rmax = %f ::: lrmin = %f ::: lrmax = %f ::: dlnr = %f\n", rmin, rmax, lrmin, lrmax, dlnr);
        //IMHERE();

        global_Rminarr_MIC     = (TYPE *)(temp2_memory); temp2_memory += (nrbin * sizeof(TYPE));
        global_Rmaxarr_MIC     = (TYPE *)(temp2_memory); temp2_memory += (nrbin * sizeof(TYPE));
        global_Rval_MIC        = (TYPE *)(temp2_memory); temp2_memory += (nrbin * sizeof(TYPE));
        global_BinCorners_MIC  = (TYPE *)(temp2_memory); temp2_memory += (HIST_BINS * sizeof(TYPE));
        global_BinCorners2_MIC = (TYPE *)(temp2_memory); temp2_memory += (HIST_BINS * sizeof(TYPE));

        if ((temp2_memory - temp_memory) != sz) ERROR_PRINT();

        TYPE *Rminarr = global_Rminarr_MIC;
        TYPE *Rmaxarr = global_Rmaxarr_MIC;
        TYPE *Rval = global_Rval_MIC;
        TYPE *BinCorners = global_BinCorners_MIC;
        TYPE *BinCorners2 = global_BinCorners2_MIC;
    
        //Equally dividing in the 'log' scale...

        for(int i=1; i<nrbin; i++)     Rminarr[i] = exp(lrmin + i*dlnr); Rminarr[0] = rmin;
        for(int i=0; i<(nrbin-1); i++) Rmaxarr[i] = Rminarr[i+1];  Rmaxarr[nrbin-1] = rmax;
        for(int i=0; i<nrbin; i++)        Rval[i] = exp(lrmin + (i+0.5)*dlnr); 

        for(int i=0; i<(nrbin); i++) BinCorners[i] = Rminarr[i] * Rminarr[i];
        for(int i=(nrbin); i<(HIST_BINS); i++) BinCorners[i] = FLT_MAX; //Some large number...
        BinCorners2[0] = -FLT_MAX; for(int i=0; i<(1+nrbin); i++) BinCorners2[i+1] = BinCorners[i]; BinCorners2[nrbin+1] = rmax*rmax;

        for(int p = 0; p < HIST_BINS; p++)
        {
            printf("%d :: %f\n", p, BinCorners2[p]);
        }

        global_BinCorners3_MIC = (TYPE *)_mm_malloc((16 * sizeof(TYPE)), 64);
        for(int p = 0; p < HIST_BINS; p++)
        {
            global_BinCorners3_MIC[p] = BinCorners2[p];
        }
        for(int p = HIST_BINS; p < 16; p++)
        {
            global_BinCorners3_MIC[p] = FLT_MAX;
        }

        if (HIST_BINS >= 14) ERROR_PRINT();


        int dx = int(ceil((i_rmaxL/i_Lbox)*dimx)) + 0;
        int dy = int(ceil((i_rmaxL/i_Lbox)*dimy)) + 0;
        int dz = int(ceil((i_rmaxL/i_Lbox)*dimz)) + 0;


        global_dx_MIC = dx;
        global_dy_MIC = dy;
        global_dz_MIC = dz;
    }
#endif
}

void CPUFunction_Initialize_Arrays(void)
{
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//STEP 2: ALLOCATE AND INITIALIZE THE HISTOGRAMS...
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    for(int i=0; i<HIST_BINS; i++) global_Histogram_RR_CPU[i] = 0;
    for(int i=0; i<HIST_BINS; i++) global_Histogram_DR_CPU[i] = 0;

    global_Overall_Histogram_RR_CPU = (unsigned long long int *)malloc(HIST_BINS * sizeof(unsigned long long int));
    global_Overall_Histogram_DR_CPU = (unsigned long long int *)malloc(HIST_BINS * sizeof(unsigned long long int));

    local_Histogram_RR_CPU = (unsigned long long int **)malloc(MAX_THREADS_CPU * sizeof(unsigned long long int *));
    local_Histogram_DR_CPU = (unsigned long long int **)malloc(MAX_THREADS_CPU * sizeof(unsigned long long int *));

    int  hist_bins = HIST_BINS;
    int hist_bins_prime = (((hist_bins + 32) >> 5)<<5); //This is to avoid cacheline conflicts and hence false sharing...
    //mpi_printf("hist_bins_prime = %d\n", hist_bins_prime);


    for(int t=0; t<MAX_THREADS_CPU; t++)
    {
        local_Histogram_RR_CPU[t] = (unsigned long long int *)malloc(hist_bins_prime * sizeof(unsigned long long int));
        for(int k=0; k<hist_bins_prime; k++) local_Histogram_RR_CPU[t][k] = 0;
        local_Histogram_DR_CPU[t] = (unsigned long long int *)malloc(hist_bins_prime * sizeof(unsigned long long int));
        for(int k=0; k<hist_bins_prime; k++) local_Histogram_DR_CPU[t][k] = 0;
    }


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//STEP 3: ALLOCATE TEMPORARY ARRAYS...
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    //CPUFunction_Allocate_Temporary_Arrays();
}

__attribute__ (( target (mic))) 
void MICFunction_Initialize_Arrays(void)
{
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//STEP 2: ALLOCATE AND INITIALIZE THE HISTOGRAMS...
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    for(int i=0; i<HIST_BINS; i++) global_Histogram_RR_MIC[i] = 0;
    for(int i=0; i<HIST_BINS; i++) global_Histogram_DR_MIC[i] = 0;

    global_Overall_Histogram_RR_MIC = (unsigned long long int *)malloc(HIST_BINS * sizeof(unsigned long long int));
    global_Overall_Histogram_DR_MIC = (unsigned long long int *)malloc(HIST_BINS * sizeof(unsigned long long int));

    local_Histogram_RR_MIC = (unsigned long long int **)malloc(MAX_THREADS_MIC * sizeof(unsigned long long int *));
    local_Histogram_DR_MIC = (unsigned long long int **)malloc(MAX_THREADS_MIC * sizeof(unsigned long long int *));

    int  hist_bins = HIST_BINS;
    int hist_bins_prime = (((hist_bins + 32) >> 5)<<5); //This is to avoid cacheline conflicts and hence false sharing...
    //mpi_printf("hist_bins_prime = %d\n", hist_bins_prime);


    for(int t=0; t<MAX_THREADS_MIC; t++)
    {
        local_Histogram_RR_MIC[t] = (unsigned long long int *)malloc(hist_bins_prime * sizeof(unsigned long long int));
        for(int k=0; k<hist_bins_prime; k++) local_Histogram_RR_MIC[t][k] = 0;
        local_Histogram_DR_MIC[t] = (unsigned long long int *)malloc(hist_bins_prime * sizeof(unsigned long long int));
        for(int k=0; k<hist_bins_prime; k++) local_Histogram_DR_MIC[t][k] = 0;
    }


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//STEP 3: ALLOCATE TEMPORARY ARRAYS...
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    //MICFunction_Allocate_Temporary_Arrays();
}

void CPUFunction_Allocated_Aligned_Buffer(void)
{
    int threshold_particles_per_cell =  GLOBAL_THRESHOLD_PARTICLES_PER_CELL;

    {
        {
#if 0
            TYPE *Temp1 = (TYPE *)malloc(256 + 2 * nthreads_CPU * threshold_particles_per_cell * 3 * sizeof(TYPE));
            {
                unsigned long long int XX = (unsigned long long int)(Temp1);
                unsigned long long int XX0 = XX;
                while (XX % 64) XX += 4;
                Temp1 += (XX - XX0)/4;
            }
#else
            TYPE *Temp1 = (TYPE *)(_mm_malloc(2 * nthreads_CPU * threshold_particles_per_cell * 3 * sizeof(TYPE), 64));
#endif

            TYPE *Temp2 = Temp1;
            //printf("Temp1 = %lld\n", (long long int)(Temp1));
            global_Aligned_Buffer_CPU = (TYPE **)malloc(nthreads_CPU * sizeof(TYPE *));
            for(int pp=0; pp<nthreads_CPU; pp++)
            {
                global_Aligned_Buffer_CPU[pp] = Temp2;
                Temp2 += 2 * 3 * threshold_particles_per_cell;
            }

            if ( (Temp2 - Temp1) != (2 * nthreads_CPU * threshold_particles_per_cell * 3)) ERROR_PRINT();
        }
    }
}


__attribute__ (( target (mic))) 
void MICFunction_Allocated_Aligned_Buffer(void)
{
    int threshold_particles_per_cell =  GLOBAL_THRESHOLD_PARTICLES_PER_CELL;

    {
        {
            TYPE *Temp1 = (TYPE *)malloc(256 + 2 * nthreads_MIC * threshold_particles_per_cell * 3 * sizeof(TYPE));
            {
                unsigned long long int XX = (unsigned long long int)(Temp1);
                unsigned long long int XX0 = XX;
                while (XX % 64) XX += 4;
                Temp1 += (XX - XX0)/4;
            }

            TYPE *Temp2 = Temp1;
            printf("Temp1 = %lld\n", (long long int)(Temp1));
            global_Aligned_Buffer_MIC = (TYPE **)malloc(nthreads_MIC * sizeof(TYPE *));
            for(int pp=0; pp<nthreads_MIC; pp++)
            {
                global_Aligned_Buffer_MIC[pp] = Temp2;
                Temp2 += 2 * 3 * threshold_particles_per_cell;
            }

            if ( (Temp2 - Temp1) != (2 * nthreads_MIC * threshold_particles_per_cell * 3)) ERROR_PRINT();
        }
    }
}

void CPUFunction_Perform_DR_TaskQ(void)
{
    unsigned long long int start_time = ___rdtsc();
 
#ifdef HETERO_COMPUTATION
    {
        //global_starting_cell_index_D_CPU = global_starting_cell_index_D_CPU; //XXX Not changed...
        global_ending_cell_index_D_CPU = global_starting_cell_index_D_CPU + global_hetero_cpu_number_of_D_cells_CPU;
    }
#endif

   IMHERE2();
///////////////////////////////////////////////////////////////////////////////////////////
//Step 2: Perform_DD
///////////////////////////////////////////////////////////////////////////////////////////

    long dimensionSize[1], tileSize[1];
    int start_processing_cell_index = global_starting_cell_index_D_CPU; if (global_starting_cell_index_D_CPU < 0) ERROR_PRINT_STRING("global_starting_cell_index_D_CPU < 0");
    int   end_processing_cell_index = global_ending_cell_index_D_CPU;   if (global_ending_cell_index_D_CPU   < 0) ERROR_PRINT_STRING("global_ending_cell_index_D_CPU   < 0");
    long ntasks = end_processing_cell_index - start_processing_cell_index;

    IMHERE2();
    //dimensionSize[0] = ntasks;  tileSize[0]=1;

    //if (node_id == 0) printf("ntasks = %lld\n", ntasks);
    
    if (node_id_CPU == 0) printf("CPU ::: <<%d>> ::: ntasks = %lld\n", node_id_CPU, ntasks);
    //taskQInit(nthreads_CPU, ntasks);
#if 0
    printf("ntasks = %lld\n", ntasks);
    printf("start_processing_cell_index = %d ::: end_processing_cell_index = %d\n", start_processing_cell_index, end_processing_cell_index);
    printf("node_id = %d ::: \n", node_id);
#endif

#if 0
    taskQEnqueueGrid((TaskQTask)(Perform_DD_Helper), 0, 1, dimensionSize, tileSize);
#else
#if 0
    for(int p=0; p<ntasks; p++) taskQEnqueueTask1((TaskQTask1)(Perform_DR_Helper), 0, (void *)(p));
    taskQWait();
#else

#pragma omp parallel for schedule(dynamic) num_threads(nthreads_CPU)
    for(int p=0; p<ntasks; p++)
        CPUFunction_Perform_DR_Helper((void*)(p));
#endif
#endif
    //unsigned long long int start_time = read_tsc();

    unsigned long long int end_time = ___rdtsc();
    global_time_dr_CPU += (end_time - start_time);
}

__attribute__ (( target (mic))) 
void MICFunction_Perform_DR_TaskQ(void)
{
    unsigned long long int start_time = ___rdtsc();
    IMHERE2();

#ifdef HETERO_COMPUTATION
    {
        global_starting_cell_index_D_MIC = global_starting_cell_index_D_MIC + global_hetero_cpu_number_of_D_cells_MIC;
        //global_ending_cell_index_D_MIC = global_ending_cell_index_D_MIC;
    }
#endif

///////////////////////////////////////////////////////////////////////////////////////////
//Step 2: Perform_DD
///////////////////////////////////////////////////////////////////////////////////////////

    long dimensionSize[1], tileSize[1];
    int start_processing_cell_index = global_starting_cell_index_D_MIC; if (global_starting_cell_index_D_MIC < 0) ERROR_PRINT_STRING("global_starting_cell_index_D_MIC < 0");
    int   end_processing_cell_index = global_ending_cell_index_D_MIC;   if (global_ending_cell_index_D_MIC   < 0) ERROR_PRINT_STRING("global_ending_cell_index_D_MIC   < 0");
    long ntasks = end_processing_cell_index - start_processing_cell_index;

    IMHERE2();
    //dimensionSize[0] = ntasks;  tileSize[0]=1;

    //if (node_id == 0) printf("ntasks = %lld\n", ntasks);
    if (node_id_MIC == 0) printf("MIC ::: <<%d>> ::: ntasks = %lld\n", node_id_MIC, ntasks);
    
    //taskQInit(nthreads_CPU, ntasks);
#if 0
    printf("ntasks = %lld\n", ntasks);
    printf("start_processing_cell_index = %d ::: end_processing_cell_index = %d\n", start_processing_cell_index, end_processing_cell_index);
    printf("node_id = %d ::: \n", node_id);
#endif

#if 0
    taskQEnqueueGrid((TaskQTask)(Perform_DD_Helper), 0, 1, dimensionSize, tileSize);
#else
#if 0
    for(int p=0; p<ntasks; p++) taskQEnqueueTask1((TaskQTask1)(Perform_DR_Helper), 0, (void *)(p));
    taskQWait();
#else

#pragma omp parallel for schedule(dynamic) num_threads(nthreads_MIC)
    for(int p=0; p<ntasks; p++)
        MICFunction_Perform_DR_Helper((void*)(p));
#endif
#endif
    //unsigned long long int start_time = read_tsc();

    unsigned long long int end_time = ___rdtsc();
    global_time_dr_MIC += (end_time - start_time);
}

void CPUFunction_Perform_RR_TaskQ(void)
{
#ifndef __MIC__
    unsigned long long int start_time = ___rdtsc();
  
#ifdef HETERO_COMPUTATION
    {
        //global_starting_cell_index_R_CPU = global_starting_cell_index_R_CPU; //XXX Not changed...
        global_ending_cell_index_R_CPU = global_starting_cell_index_R_CPU + global_hetero_cpu_number_of_R_cells_CPU;
    }
#endif
    //IMHERE2();
///////////////////////////////////////////////////////////////////////////////////////////
//Step 2: Perform_DD
///////////////////////////////////////////////////////////////////////////////////////////

    //long dimensionSize[1], tileSize[1];
    int start_processing_cell_index = global_starting_cell_index_R_CPU;
    int   end_processing_cell_index = global_ending_cell_index_R_CPU;
    long ntasks = end_processing_cell_index - start_processing_cell_index;

    IMHERE2();

    //dimensionSize[0] = ntasks;  tileSize[0]=1;
    if (node_id_CPU == 0) printf("CPU ::: <<%d>> ::: ntasks = %lld\n", node_id_CPU, ntasks);
    //printf("ntasks = %lld\n", ntasks);


    //printf("node_id = %d ::: \n", node_id);

#if 0
    taskQEnqueueGrid((TaskQTask)(Perform_DD_Helper), 0, 1, dimensionSize, tileSize);
#else
#if 0
     for(int p=0; p<ntasks; p++) taskQEnqueueTask1((TaskQTask1)(Perform_RR_Helper), 0, (void *)(p));
     taskQWait();
#else

     //printf("nthreads_CPU = %d\n", nthreads_CPU);
#pragma omp parallel for schedule(dynamic) num_threads(nthreads_CPU)
//#pragma omp parallel for num_threads(nthreads_CPU)
    for(int p=0; p<ntasks; p++) 
        CPUFunction_Perform_RR_Helper((void*)(p));
#endif
#endif
    //unsigned long long int start_time = read_tsc();

    unsigned long long int end_time = ___rdtsc();
    global_time_rr_CPU += (end_time - start_time);

    //Compute_Statistics();
#endif
}

__attribute__ (( target (mic))) 
void MICFunction_Perform_RR_TaskQ(void)
{
    unsigned long long int start_time = ___rdtsc();
 
#ifdef HETERO_COMPUTATION
    {
        global_starting_cell_index_R_MIC = global_starting_cell_index_R_MIC + global_hetero_cpu_number_of_R_cells_MIC;
        //global_ending_cell_index_R_MIC = global_ending_cell_index_R_MIC;
    }
#endif

    //IMHERE2();
///////////////////////////////////////////////////////////////////////////////////////////
//Step 2: Perform_DD
///////////////////////////////////////////////////////////////////////////////////////////

    //long dimensionSize[1], tileSize[1];
    int start_processing_cell_index = global_starting_cell_index_R_MIC;
    int   end_processing_cell_index = global_ending_cell_index_R_MIC;
    long ntasks = end_processing_cell_index - start_processing_cell_index;

    IMHERE2();


    //dimensionSize[0] = ntasks;  tileSize[0]=1;
    if (node_id_MIC == 0) printf("MIC ::: <<%d>> ::: ntasks = %lld\n", node_id_MIC, ntasks);
    //printf("ntasks = %lld\n", ntasks);


    //printf("node_id = %d ::: \n", node_id);

#if 0
    taskQEnqueueGrid((TaskQTask)(Perform_DD_Helper), 0, 1, dimensionSize, tileSize);
#else
#if 0
     for(int p=0; p<ntasks; p++) taskQEnqueueTask1((TaskQTask1)(Perform_RR_Helper), 0, (void *)(p));
     taskQWait();
#else

#pragma omp parallel for schedule(dynamic) num_threads(nthreads_MIC)
//#pragma omp parallel for num_threads(nthreads_CPU)
    for(int p=0; p<ntasks; p++)
        MICFunction_Perform_RR_Helper((void*)(p));
#endif
#endif
    //unsigned long long int start_time = read_tsc();

    unsigned long long int end_time = ___rdtsc();
    global_time_rr_MIC += (end_time - start_time);

    //Compute_Statistics();
}

void CPUFunction_Report_Performance(void)
{
    if (node_id_CPU != 0) return;

    {
        PRINT_BLUE
            printf("=================== RR ================\n");
        PRINT_BLACK
        //++++++RR++++++++
        long long int total_sum_so_far = 0;
        long long int total_sum = 0;
        for(int bin_id=1; bin_id < (HIST_BINS-1); bin_id++) total_sum += global_Histogram_RR_CPU[bin_id];

        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        for(int bin_id=1; bin_id < (HIST_BINS-1); bin_id++)
        {
            total_sum_so_far += global_Histogram_RR_CPU[bin_id];
            printf("%2d :: %16lld (%19lld ::: %.4e %%)\n", bin_id, global_Histogram_RR_CPU[bin_id], total_sum_so_far, (total_sum_so_far*100.0)/total_sum);
        }

        printf("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n");
        for(int bin_id=1; bin_id < (HIST_BINS-1); bin_id++)
        {
            printf("%2d :: %.6lf \n", bin_id, (global_RR_over_RR_CPU[bin_id]-1.0));
        }
        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        global_stat_useful_interactions_rr_CPU = total_sum;
    }

    {
        PRINT_BLUE
            printf("=================== DR ================\n");
        PRINT_BLACK
        //++++++DR++++++++
        long long int total_sum_so_far = 0;
        long long int total_sum = 0;
        for(int bin_id=1; bin_id < (HIST_BINS-1); bin_id++) total_sum += global_Histogram_DR_CPU[bin_id];

        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        for(int bin_id=1; bin_id < (HIST_BINS-1); bin_id++)
        {
            total_sum_so_far += global_Histogram_DR_CPU[bin_id];
            printf("%2d :: %16lld (%19lld ::: %.4e %%)\n", bin_id, global_Histogram_DR_CPU[bin_id], total_sum_so_far, (total_sum_so_far*100.0)/total_sum);
        }

        printf("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n");
        for(int bin_id=1; bin_id < (HIST_BINS-1); bin_id++)
        {
            printf("%2d :: %.6lf \n", bin_id, (global_DR_over_RR_CPU[bin_id]-1.0));
        }
        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        global_stat_useful_interactions_dr_CPU = total_sum;
    }

    long long int ngal_CPU = global_number_of_galaxies_CPU;
    long long int actual_sum_CPU = global_stat_total_interactions_rr_CPU + global_stat_total_interactions_dr_CPU;
    //global_time_kdtree = global_time_kdtree_d + global_time_kdtree_r;
    global_time_total_CPU = global_time_rr_CPU + global_time_dr_CPU;


PRINT_BLUE
    printf("==================================================================================\n");
    printf("CORE_FREQUENCY_CPU\t\t\t = %.2lf GHz ::: nthreads_CPU = %d ::: nnodes_CPU = %d\n", (CORE_FREQUENCY_CPU/1000.0/1000.0/1000.0), nthreads_CPU, nnodes_CPU);
    //printf("<<%d>>MPI_Time\t\t\t = %12lld cycles (%5.2lf seconds) (%5.2lf%%)\n", node_id, global_time_mpi, (global_time_mpi*1.0)/CORE_FREQUENCY, (global_time_mpi*100.0)/global_time_total);
PRINT_GRAY
    //printf("<<%d>>KD-Tree (D) Construction Time\t = %12lld cycles (%5.2lf seconds) (%5.2lf%%)\n", node_id, global_time_kdtree_d, global_time_kdtree_d/CORE_FREQUENCY, (global_time_kdtree_d*100.0)/global_time_total);
    //printf("<<%d>>KD-Tree (R) Construction Time\t = %12lld cycles (%5.2lf seconds) (%5.2lf%%)\n", node_id, global_time_kdtree_r, global_time_kdtree_r/CORE_FREQUENCY, (global_time_kdtree_r*100.0)/global_time_total);
PRINT_BLUE
    //printf("<<%d>>KD-Tree Construction Time\t = %12lld cycles (%5.2lf seconds) (%5.2lf%%)\n", node_id, global_time_kdtree, global_time_kdtree/CORE_FREQUENCY, (global_time_kdtree*100.0)/global_time_total);
    printf("<<%d>>RR Time\t\t\t = %12lld cycles (%5.2lf seconds) (%5.2lf%%)\n", node_id_CPU, global_time_rr_CPU, global_time_rr_CPU/CORE_FREQUENCY_CPU, (global_time_rr_CPU*100.0)/global_time_total_CPU);
    printf("<<%d>>DR Time\t\t\t = %12lld cycles (%5.2lf seconds) (%5.2lf%%)\n", node_id_CPU, global_time_dr_CPU, global_time_dr_CPU/CORE_FREQUENCY_CPU, (global_time_dr_CPU*100.0)/global_time_total_CPU);
    printf("<<%d>>Total Time\t\t\t = %12lld cycles (%5.2lf seconds) (%5.2lf%%)\n", node_id_CPU, global_time_total_CPU, global_time_total_CPU/CORE_FREQUENCY_CPU, (global_time_total_CPU*100.0)/global_time_total_CPU);
    printf("==================================================================================\n");

PRINT_LIGHT_RED
{
    MT_Z_CPU[0] = MT_2_CPU[0] + MT_3_CPU[0] + MT_4_CPU[0] + MT_5_CPU[0];
    MT_Z_CPU[8] = MT_2_CPU[8] + MT_3_CPU[8] + MT_4_CPU[8] + MT_5_CPU[8];
    MT_Z_CPU[16] = MT_2_CPU[16] + MT_3_CPU[16] + MT_4_CPU[16] + MT_5_CPU[16];
    MT_Z_CPU[24] = MT_2_CPU[24] + MT_3_CPU[24] + MT_4_CPU[24] + MT_5_CPU[24];
    printf("MT_2_CPU = [%.2lf ::: %.2lf ::: %.2lf ::: %.2lf]\n", MT_2_CPU[0]/CORE_FREQUENCY_CPU, MT_2_CPU[8]/CORE_FREQUENCY_CPU, MT_2_CPU[16]/CORE_FREQUENCY_CPU,  MT_2_CPU[24]/CORE_FREQUENCY_CPU);
    printf("MT_3_CPU = [%.2lf ::: %.2lf ::: %.2lf ::: %.2lf]\n", MT_3_CPU[0]/CORE_FREQUENCY_CPU, MT_3_CPU[8]/CORE_FREQUENCY_CPU, MT_3_CPU[16]/CORE_FREQUENCY_CPU,  MT_3_CPU[24]/CORE_FREQUENCY_CPU);
    printf("MT_4_CPU = [%.2lf ::: %.2lf ::: %.2lf ::: %.2lf]\n", MT_4_CPU[0]/CORE_FREQUENCY_CPU, MT_4_CPU[8]/CORE_FREQUENCY_CPU, MT_4_CPU[16]/CORE_FREQUENCY_CPU,  MT_4_CPU[24]/CORE_FREQUENCY_CPU);
    printf("MT_5_CPU = [%.2lf ::: %.2lf ::: %.2lf ::: %.2lf]\n", MT_5_CPU[0]/CORE_FREQUENCY_CPU, MT_5_CPU[8]/CORE_FREQUENCY_CPU, MT_5_CPU[16]/CORE_FREQUENCY_CPU,  MT_5_CPU[24]/CORE_FREQUENCY_CPU);
    printf("-----------------------------------------------------------------------------------------------------\n");
    printf("MT_Z_CPU = [%.2lf ::: %.2lf ::: %.2lf ::: %.2lf]\n", MT_Z_CPU[0]/CORE_FREQUENCY_CPU, MT_Z_CPU[8]/CORE_FREQUENCY_CPU, MT_Z_CPU[16]/CORE_FREQUENCY_CPU,  MT_Z_CPU[24]/CORE_FREQUENCY_CPU);
    printf("NT_Z_CPU = [%.2lf ::: %.2lf ::: %.2lf ::: %.2lf]\n", NT_Z_CPU[0]/CORE_FREQUENCY_CPU, NT_Z_CPU[8]/CORE_FREQUENCY_CPU, NT_Z_CPU[16]/CORE_FREQUENCY_CPU,  NT_Z_CPU[24]/CORE_FREQUENCY_CPU);
    printf("-----------------------------------------------------------------------------------------------------\n");
}

PRINT_RED
    printf("<<%d>>Total Time Taken = %lld cycles (%.2lf seconds)\n", node_id_CPU, global_time_total_CPU, (global_time_total_CPU*1.0)/CORE_FREQUENCY_CPU);
    //printf("Total Time Taken (without Kd-Tree Construction) = %lld cycles (%.2lf seconds) ::: %.2lf cycles/actual_interaction\n", (global_time_total-global_time_kdtree), ((global_time_total-global_time_kdtree)*1.0)/CORE_FREQUENCY, (1.00*(global_time_total-global_time_kdtree))/actual_sum);

    //global_time_total = global_time_total - global_time_kdtree;

    printf("<<%d>> :::    global_accumulated_easy_CPU_over_all_the_nodes  = %15lld\n", node_id_CPU, global_accumulated_easy_CPU);
    printf("<<%d>> ::: global_useful_interactions_CPU = %15lld\n", node_id_CPU, (global_stat_useful_interactions_rr_CPU + global_stat_useful_interactions_dr_CPU));
    printf("<<%d>> ::: global_total_interactions_CPU  = %15lld\n", node_id_CPU, actual_sum_CPU);
    //printf("<<%d>> ::: Total Num of Interactions_CPU  = %15lld ::: Total Number of Galaxies = %lld ::: Interactions Per Galaxy = %d (%.2lf%%) ::: ", node_id_CPU, actual_sum_CPU, ngal_CPU, (int)(actual_sum_CPU/ngal_CPU), (actual_sum_CPU*100.0)/(1.0*ngal_CPU*ngal_CPU));
    //printf("<<%d>>Time Per Actual Interaction = %.2lf cycles\n", node_id_CPU, (nnodes_CPU * nthreads_CPU * global_time_total_CPU*1.0)/actual_sum_CPU);

#if 0
    printf("<<%d>>Total Number of Interactions = %lld ::: Total Number of Galaxies = %lld ::: Total Interactions Per Galaxy = %d (%.2lf%%) :::", 
            node_id, actual_sum, ngal, (int)(actual_sum/ngal), (actual_sum*100.0)/(1.0*ngal*ngal));
    printf("Time Per Interaction = %.2lf cycles\n", (global_time_total*1.0)/actual_sum);
#endif

#if 0
    double target_galaxies = 1000*1000*1000.0;
    double percentage_of_neighbors_tested_against = (actual_sum*100.0)/(1.0*ngal*ngal);
    double cycles_per_interaction = (global_time_total*1.0)/(actual_sum);
    double cycles_taken = (target_galaxies * target_galaxies * percentage_of_neighbors_tested_against/100.0) * cycles_per_interaction;
    double seconds_taken = cycles_taken / CORE_FREQUENCY;
    double hours_taken = seconds_taken/3600;
    double days_taken = hours_taken/24;

PRINT_BLUE
    printf("At this rate, time to process 1B galaxies = %.3lf hours (%.1lf days)\n", hours_taken, days_taken);
PRINT_BLACK
#endif

}
__attribute__ (( target (mic))) 
void MICFunction_Report_Performance(void)
{
    if (node_id_MIC != 0) return;

    {
        PRINT_BLUE
            printf("=================== RR ================\n");
        PRINT_BLACK
        //++++++RR++++++++
        long long int total_sum_so_far = 0;
        long long int total_sum = 0;
        for(int bin_id=1; bin_id < (HIST_BINS-1); bin_id++) total_sum += global_Histogram_RR_MIC[bin_id];

        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        for(int bin_id=1; bin_id < (HIST_BINS-1); bin_id++)
        {
            total_sum_so_far += global_Histogram_RR_MIC[bin_id];
            printf("%2d :: %16lld (%19lld ::: %.4e %%)\n", bin_id, global_Histogram_RR_MIC[bin_id], total_sum_so_far, (total_sum_so_far*100.0)/total_sum);
        }

        printf("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n");
        for(int bin_id=1; bin_id < (HIST_BINS-1); bin_id++)
        {
            printf("%2d :: %.6lf \n", bin_id, (global_RR_over_RR_MIC[bin_id]-1.0));
        }
        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        global_stat_useful_interactions_rr_MIC = total_sum;
    }

    {
        PRINT_BLUE
            printf("=================== DR ================\n");
        PRINT_BLACK
        //++++++DR++++++++
        long long int total_sum_so_far = 0;
        long long int total_sum = 0;
        for(int bin_id=1; bin_id < (HIST_BINS-1); bin_id++) total_sum += global_Histogram_DR_MIC[bin_id];

        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        for(int bin_id=1; bin_id < (HIST_BINS-1); bin_id++)
        {
            total_sum_so_far += global_Histogram_DR_MIC[bin_id];
            printf("%2d :: %16lld (%19lld ::: %.4e %%)\n", bin_id, global_Histogram_DR_MIC[bin_id], total_sum_so_far, (total_sum_so_far*100.0)/total_sum);
        }

        printf("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n");
        for(int bin_id=1; bin_id < (HIST_BINS-1); bin_id++)
        {
            printf("%2d :: %.6lf \n", bin_id, (global_DR_over_RR_MIC[bin_id]-1.0));
        }
        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        global_stat_useful_interactions_dr_MIC = total_sum;
    }

    long long int ngal_MIC = global_number_of_galaxies_MIC;
    long long int actual_sum_MIC = global_stat_total_interactions_rr_MIC + global_stat_total_interactions_dr_MIC;
    //global_time_kdtree = global_time_kdtree_d + global_time_kdtree_r;
    global_time_total_MIC = global_time_rr_MIC + global_time_dr_MIC;


PRINT_BLUE
    printf("==================================================================================\n");
    printf("CORE_FREQUENCY_MIC\t\t\t = %.2lf GHz ::: nthreads_MIC = %d ::: nnodes_MIC = %d\n", (CORE_FREQUENCY_MIC/1000.0/1000.0/1000.0), nthreads_MIC, nnodes_MIC);
    //printf("<<%d>>MPI_Time\t\t\t = %12lld cycles (%5.2lf seconds) (%5.2lf%%)\n", node_id, global_time_mpi, (global_time_mpi*1.0)/CORE_FREQUENCY, (global_time_mpi*100.0)/global_time_total);
PRINT_GRAY
    //printf("<<%d>>KD-Tree (D) Construction Time\t = %12lld cycles (%5.2lf seconds) (%5.2lf%%)\n", node_id, global_time_kdtree_d, global_time_kdtree_d/CORE_FREQUENCY, (global_time_kdtree_d*100.0)/global_time_total);
    //printf("<<%d>>KD-Tree (R) Construction Time\t = %12lld cycles (%5.2lf seconds) (%5.2lf%%)\n", node_id, global_time_kdtree_r, global_time_kdtree_r/CORE_FREQUENCY, (global_time_kdtree_r*100.0)/global_time_total);
PRINT_BLUE
    //printf("<<%d>>KD-Tree Construction Time\t = %12lld cycles (%5.2lf seconds) (%5.2lf%%)\n", node_id, global_time_kdtree, global_time_kdtree/CORE_FREQUENCY, (global_time_kdtree*100.0)/global_time_total);
    printf("<<%d>>RR Time\t\t\t = %12lld cycles (%5.2lf seconds) (%5.2lf%%)\n", node_id_MIC, global_time_rr_MIC, global_time_rr_MIC/CORE_FREQUENCY_MIC, (global_time_rr_MIC*100.0)/global_time_total_MIC);
    printf("<<%d>>DR Time\t\t\t = %12lld cycles (%5.2lf seconds) (%5.2lf%%)\n", node_id_MIC, global_time_dr_MIC, global_time_dr_MIC/CORE_FREQUENCY_MIC, (global_time_dr_MIC*100.0)/global_time_total_MIC);
    printf("<<%d>>Total Time\t\t\t = %12lld cycles (%5.2lf seconds) (%5.2lf%%)\n", node_id_MIC, global_time_total_MIC, global_time_total_MIC/CORE_FREQUENCY_MIC, (global_time_total_MIC*100.0)/global_time_total_MIC);
    printf("==================================================================================\n");

PRINT_LIGHT_RED
{
    MT_Z_MIC[0] = MT_2_MIC[0] + MT_3_MIC[0] + MT_4_MIC[0] + MT_5_MIC[0];
    MT_Z_MIC[8] = MT_2_MIC[8] + MT_3_MIC[8] + MT_4_MIC[8] + MT_5_MIC[8];
    MT_Z_MIC[16] = MT_2_MIC[16] + MT_3_MIC[16] + MT_4_MIC[16] + MT_5_MIC[16];
    MT_Z_MIC[24] = MT_2_MIC[24] + MT_3_MIC[24] + MT_4_MIC[24] + MT_5_MIC[24];
    printf("MT_2_MIC = [%.2lf ::: %.2lf ::: %.2lf ::: %.2lf]\n", MT_2_MIC[0]/CORE_FREQUENCY_MIC, MT_2_MIC[8]/CORE_FREQUENCY_MIC, MT_2_MIC[16]/CORE_FREQUENCY_MIC,  MT_2_MIC[24]/CORE_FREQUENCY_MIC);
    printf("MT_3_MIC = [%.2lf ::: %.2lf ::: %.2lf ::: %.2lf]\n", MT_3_MIC[0]/CORE_FREQUENCY_MIC, MT_3_MIC[8]/CORE_FREQUENCY_MIC, MT_3_MIC[16]/CORE_FREQUENCY_MIC,  MT_3_MIC[24]/CORE_FREQUENCY_MIC);
    printf("MT_4_MIC = [%.2lf ::: %.2lf ::: %.2lf ::: %.2lf]\n", MT_4_MIC[0]/CORE_FREQUENCY_MIC, MT_4_MIC[8]/CORE_FREQUENCY_MIC, MT_4_MIC[16]/CORE_FREQUENCY_MIC,  MT_4_MIC[24]/CORE_FREQUENCY_MIC);
    printf("MT_5_MIC = [%.2lf ::: %.2lf ::: %.2lf ::: %.2lf]\n", MT_5_MIC[0]/CORE_FREQUENCY_MIC, MT_5_MIC[8]/CORE_FREQUENCY_MIC, MT_5_MIC[16]/CORE_FREQUENCY_MIC,  MT_5_MIC[24]/CORE_FREQUENCY_MIC);
    printf("-----------------------------------------------------------------------------------------------------\n");
    printf("MT_Z_MIC = [%.2lf ::: %.2lf ::: %.2lf ::: %.2lf]\n", MT_Z_MIC[0]/CORE_FREQUENCY_MIC, MT_Z_MIC[8]/CORE_FREQUENCY_MIC, MT_Z_MIC[16]/CORE_FREQUENCY_MIC,  MT_Z_MIC[24]/CORE_FREQUENCY_MIC);
    printf("OT_Z_MIC = [%.2lf ::: %.2lf ::: %.2lf ::: %.2lf]\n", OT_Z_MIC[0]/CORE_FREQUENCY_MIC, OT_Z_MIC[8]/CORE_FREQUENCY_MIC, OT_Z_MIC[16]/CORE_FREQUENCY_MIC,  OT_Z_MIC[24]/CORE_FREQUENCY_MIC);
    printf("NT_Z_MIC = [%.2lf ::: %.2lf ::: %.2lf ::: %.2lf]\n", NT_Z_MIC[0]/CORE_FREQUENCY_MIC, NT_Z_MIC[8]/CORE_FREQUENCY_MIC, NT_Z_MIC[16]/CORE_FREQUENCY_MIC,  NT_Z_MIC[24]/CORE_FREQUENCY_MIC);
    printf("-----------------------------------------------------------------------------------------------------\n");
}

PRINT_RED
    printf("<<%d>>Total Time Taken = %lld cycles (%.2lf seconds)\n", node_id_MIC, global_time_total_MIC, (global_time_total_MIC*1.0)/CORE_FREQUENCY_MIC);
    //printf("Total Time Taken (without Kd-Tree Construction) = %lld cycles (%.2lf seconds) ::: %.2lf cycles/actual_interaction\n", (global_time_total-global_time_kdtree), ((global_time_total-global_time_kdtree)*1.0)/CORE_FREQUENCY, (1.00*(global_time_total-global_time_kdtree))/actual_sum);

    //global_time_total = global_time_total - global_time_kdtree;

    printf("<<%d>> :::    global_accumulated_easy_MIC = %15lld\n", node_id_MIC, global_accumulated_easy_MIC);
    printf("<<%d>> ::: global_useful_interactions_MIC = %15lld\n", node_id_MIC, (global_stat_useful_interactions_rr_MIC + global_stat_useful_interactions_dr_MIC));
    printf("<<%d>> ::: global_total_interactions_MIC  = %15lld\n", node_id_MIC, actual_sum_MIC);
    //printf("<<%d>> ::: Total Num of Interactions_MIC  = %15lld ::: Total Number of Galaxies = %lld ::: Interactions Per Galaxy = %d (%.2lf%%) ::: ", node_id_MIC, actual_sum_MIC, ngal_MIC, (int)(actual_sum_MIC/ngal_MIC), (actual_sum_MIC*100.0)/(1.0*ngal_MIC*ngal_MIC));
    //printf("<<%d>>Time Per Actual Interaction = %.2lf cycles\n", node_id_MIC, (nnodes_MIC * nthreads_MIC * global_time_total_MIC*1.0)/actual_sum_MIC);

#if 0
    printf("<<%d>>Total Number of Interactions = %lld ::: Total Number of Galaxies = %lld ::: Total Interactions Per Galaxy = %d (%.2lf%%) :::", 
            node_id, actual_sum, ngal, (int)(actual_sum/ngal), (actual_sum*100.0)/(1.0*ngal*ngal));
    printf("Time Per Interaction = %.2lf cycles\n", (global_time_total*1.0)/actual_sum);
#endif

#if 0
    double target_galaxies = 1000*1000*1000.0;
    double percentage_of_neighbors_tested_against = (actual_sum*100.0)/(1.0*ngal*ngal);
    double cycles_per_interaction = (global_time_total*1.0)/(actual_sum);
    double cycles_taken = (target_galaxies * target_galaxies * percentage_of_neighbors_tested_against/100.0) * cycles_per_interaction;
    double seconds_taken = cycles_taken / CORE_FREQUENCY;
    double hours_taken = seconds_taken/3600;
    double days_taken = hours_taken/24;

PRINT_BLUE
    printf("At this rate, time to process 1B galaxies = %.3lf hours (%.1lf days)\n", hours_taken, days_taken);
PRINT_BLACK
#endif

}

int global_cpu_grids_populated = 0;

void CPU_Function_Populate_CPU_Grids(int *Packet_D_CPU, size_t length_D, int *Packet_R_CPU, size_t length_R)
{
    if (!global_cpu_grids_populated)
    {
        if (DataTransfer_Size_D_From_CPU_To_MIC_CPU != (length_D * sizeof(int))) ERROR_PRINT();
        if (DataTransfer_Size_R_From_CPU_To_MIC_CPU != (length_R * sizeof(int))) ERROR_PRINT();

        if (Temp_Memory_D_CPU != Packet_D_CPU) ERROR_PRINT();
        if (Temp_Memory_R_CPU != Packet_R_CPU) ERROR_PRINT();

        CPUFunction_Copy_Temp_Memory_To_D_Or_R(&global_grid_D_CPU, DataTransfer_Size_D_From_CPU_To_MIC_CPU, Temp_Memory_D_CPU); if (node_id_CPU == 0) printf("Populated CPU grid_D Successfully\n");
        CPUFunction_Copy_Temp_Memory_To_D_Or_R(&global_grid_R_CPU, DataTransfer_Size_R_From_CPU_To_MIC_CPU, Temp_Memory_R_CPU); if (node_id_CPU == 0) printf("Populated CPU grid_R Successfully\n");
        global_cpu_grids_populated = 1;
    }
}

void CPUFunction_Perform_TPCF_On_CPU(int *Packet_D_CPU, size_t length_D, int *Packet_R_CPU, size_t length_R, unsigned char *Global_Variables, TYPE *Answer)
{

    if (!global_cpu_grids_populated)
    {
        ERROR_PRINT_STRING("How come CPU grid_D/_R are already populated...?\n");
        if (DataTransfer_Size_D_From_CPU_To_MIC_CPU != (length_D * sizeof(int))) ERROR_PRINT();
        if (DataTransfer_Size_R_From_CPU_To_MIC_CPU != (length_R * sizeof(int))) ERROR_PRINT();

        if (Temp_Memory_D_CPU != Packet_D_CPU) ERROR_PRINT();
        if (Temp_Memory_R_CPU != Packet_R_CPU) ERROR_PRINT();

        CPUFunction_Copy_Temp_Memory_To_D_Or_R(&global_grid_D_CPU, DataTransfer_Size_D_From_CPU_To_MIC_CPU, Temp_Memory_D_CPU); printf("Populated CPU grid_D Successfully\n");
        CPUFunction_Copy_Temp_Memory_To_D_Or_R(&global_grid_R_CPU, DataTransfer_Size_R_From_CPU_To_MIC_CPU, Temp_Memory_R_CPU); printf("Populated CPU grid_R Successfully\n");
        global_cpu_grids_populated = 1;
    }

    //printf("Done on CPU...\n");

    //CPUFunction_Spit_Output(&global_grid_D_CPU, DataTransfer_Size_D_From_CPU_To_MIC_CPU, Temp_Memory_D_CPU, "jch_D4.bin");
    //CPUFunction_Spit_Output(&global_grid_R_CPU, DataTransfer_Size_R_From_CPU_To_MIC_CPU, Temp_Memory_R_CPU, "jch_R4.bin");

    //XXX: Not required... MICFunction_Parse_Global_Variables(Global_Variables);

    unsigned long long int stime = ___rdtsc();
    CPUFunction_Perform_Mandatory_Initializations(&global_grid_R_CPU, global_Lbox_CPU, global_rminL_CPU, global_rmaxL_CPU, global_nrbin_CPU);
    CPUFunction_Initialize_Arrays();
    CPUFunction_Allocated_Aligned_Buffer();
    unsigned long long int etime = ___rdtsc();
    unsigned long long int ttime = etime - stime;
    if (node_id_CPU == 0) printf(" <%d> ::: Time Taken = %lld cycles (%.2lf seconds)\n", node_id_CPU, ttime, ttime/CORE_FREQUENCY_MIC);

    //---------------------------------
    CPUFunction_Perform_RR_TaskQ();
    CPUFunction_Perform_DR_TaskQ();
    CPUFunction_Compute_Statistics_RR();
    CPUFunction_Compute_Statistics_DR();
    CPUFunction_Report_Performance();
    Answer[0] = 95123;
}


__attribute__ (( target (mic))) 
void MICFunction_Perform_TPCF_On_MIC(int *Packet_D_MIC, size_t length_D, int *Packet_R_MIC, size_t length_R, unsigned char *Global_Variables, TYPE *Answer)
{

    DataTransfer_Size_D_From_CPU_To_MIC_MIC = length_D * sizeof(int);
    DataTransfer_Size_R_From_CPU_To_MIC_MIC = length_R * sizeof(int);

    Temp_Memory_D_MIC = Packet_D_MIC;
    Temp_Memory_R_MIC = Packet_R_MIC;

    MICFunction_Copy_Temp_Memory_To_D_Or_R(&global_grid_D_MIC, DataTransfer_Size_D_From_CPU_To_MIC_MIC, Temp_Memory_D_MIC); printf("Populated MIC grid_D Successfully\n");
    MICFunction_Copy_Temp_Memory_To_D_Or_R(&global_grid_R_MIC, DataTransfer_Size_R_From_CPU_To_MIC_MIC, Temp_Memory_R_MIC); printf("Populated MIC grid_R Successfully\n");

    //MICFunction_Spit_Output(&global_grid_D_MIC, DataTransfer_Size_D_From_CPU_To_MIC_MIC, Temp_Memory_D_MIC, "jch_D3.bin");
    //MICFunction_Spit_Output(&global_grid_R_MIC, DataTransfer_Size_R_From_CPU_To_MIC_MIC, Temp_Memory_R_MIC, "jch_R3.bin");

    MICFunction_Parse_Global_Variables(Global_Variables);

    unsigned long long int stime = ___rdtsc();
    MICFunction_Perform_Mandatory_Initializations(&global_grid_R_MIC, global_Lbox_MIC, global_rminL_MIC, global_rmaxL_MIC, global_nrbin_MIC);
    MICFunction_Initialize_Arrays();
    MICFunction_Allocated_Aligned_Buffer();
    unsigned long long int etime = ___rdtsc();
    unsigned long long int ttime = etime - stime;
    printf("Time Taken = %lld cycles (%.2lf seconds)\n", ttime, ttime/CORE_FREQUENCY_MIC);

    //---------------------------------
    MICFunction_Perform_RR_TaskQ();
    MICFunction_Perform_DR_TaskQ();
    MICFunction_Compute_Statistics_RR();
    MICFunction_Compute_Statistics_DR();
    MICFunction_Report_Performance();
    Answer[0] = 95123;
}

size_t Fill_Up_Global_Variables_Packet(unsigned char **i_Global_Variables)
{
    int sz0 = 0;
    int cells_in_stencil = 0;

    {
    
        TYPE i_rminL = global_rminL_CPU;
        TYPE i_Lbox  = global_Lbox_CPU;
        TYPE i_rmaxL = global_rmaxL_CPU;

        int dimx = global_grid_D_CPU.dimx;
        int dimy = global_grid_D_CPU.dimy;
        int dimz = global_grid_D_CPU.dimz;
    
        int dx = int(ceil((i_rmaxL/i_Lbox)*dimx)) + 0;
        int dy = int(ceil((i_rmaxL/i_Lbox)*dimy)) + 0;
        int dz = int(ceil((i_rmaxL/i_Lbox)*dimz)) + 0;
    
        cells_in_stencil = (2*dx +1) * (2*dy + 1) * (2*dz + 1);
        sz0 = cells_in_stencil * sizeof(unsigned char);
    }

    size_t sz = 6 * sizeof(TYPE) + sizeof(long long int) + 2 * sizeof(int) + 2 * sz0;
    unsigned char *Global_Variables = (unsigned char *)malloc(sz);
    *i_Global_Variables = Global_Variables;

    TYPE *X = (TYPE *)(Global_Variables);
    X[0] = global_Lbox_CPU;
    X[1] = global_rminL_CPU;
    X[2] = global_rmaxL_CPU;
    int *Y = (int *)(Global_Variables + 3*sizeof(TYPE));
    Y[0] = global_nrbin_CPU;
    Y[1] = node_id_CPU;
    Y[2] = nnodes_CPU;
    *((long long int *)(Global_Variables + 24)) = global_number_of_galaxies_CPU;
    *((int *)(Global_Variables + 32)) = global_hetero_cpu_number_of_D_cells_CPU;
    *((int *)(Global_Variables + 36)) = global_hetero_cpu_number_of_R_cells_CPU;

    for(int p = 0; p < cells_in_stencil; p++) Global_Variables[40 + 0*cells_in_stencil + p] = global_Template_during_hetero_RR_CPU[p];
    for(int p = 0; p < cells_in_stencil; p++) Global_Variables[40 + 1*cells_in_stencil + p] = global_Template_during_hetero_DR_CPU[p];

    if (sz != (40 + 2*cells_in_stencil)) ERROR_PRINT();
    return (sz);
}

#ifdef HETERO_COMPUTATION

#ifdef HETERO_COMPUTATION


int global_hetero_dimx;
int global_hetero_dimy;
int global_hetero_dimz;
int global_hetero_dimxy;
int global_hetero_number_of_uniform_subdivisions;

void Compute_Template_During_Hetero(Grid *grid, int dimx, int dimy, int dimz, int dimxy)
{

    TYPE i_rminL = global_rminL_CPU;
    TYPE i_Lbox  = global_Lbox_CPU;
    TYPE i_rmaxL = global_rmaxL_CPU;

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
    global_Template_during_hetero = (unsigned char *)malloc(cells_in_stencil * sizeof(unsigned char));
    global_Template_during_hetero_RR_CPU = (unsigned char *)malloc(cells_in_stencil * sizeof(unsigned char));
    global_Template_during_hetero_DR_CPU = (unsigned char *)malloc(cells_in_stencil * sizeof(unsigned char));

    for(int k=0; k<cells_in_stencil; k++) global_Template_during_hetero[k] = 0;

    int ccounter = 0;

    for(int zz = (z - dz); zz <= (z + dz); zz++)
    {
        for(int yy = (y - dy); yy <= (y + dy); yy++)
        {
            for(int xx = (x - dx); xx <= (x + dx); xx++, ccounter++)
            {
                //Our neighbor is the (xx, yy, zz) cell...
                if ((xx == x) && (yy == y) && (zz == z)) 
                {
                    global_Template_during_hetero[ccounter] = 0;
                    continue;
                }

                ////////////////////////////////////////////////////////////////////////////////////////
                //Step A: Figure out if the nearest points between the grids is >= rmax...
                ////////////////////////////////////////////////////////////////////////////////////////
                            
               
                Range1_X[0] = (xx*1.00)*Cell_Width[0]; Range1_X[1] = ((xx+1)*1.00)*Cell_Width[0];
                Range1_Y[0] = (yy*1.00)*Cell_Width[1]; Range1_Y[1] = ((yy+1)*1.00)*Cell_Width[1];
                Range1_Z[0] = (zz*1.00)*Cell_Width[2]; Range1_Z[1] = ((zz+1)*1.00)*Cell_Width[2];

                TYPE min_dist_2 = CPUFunction_Find_Minimum_Distance_Between_Cells(Range0_X, Range0_Y, Range0_Z, Range1_X, Range1_Y, Range1_Z);

                if (min_dist_2 > rmax_2) 
                {
                    global_Template_during_hetero[ccounter] = 0;
                    continue;
                }

                global_Template_during_hetero[ccounter] = 1;
            }
        }
    }

    if (ccounter != cells_in_stencil) ERROR_PRINT();

    ccounter = 0;
    for(int zz = (z - dz); zz <= (z + dz); zz++)
    {
        for(int yy = (y - dy); yy <= (y + dy); yy++)
        {
            for(int xx = (x - dx); xx <= (x + dx); xx++, ccounter++)
            {
                //Our neighbor is the (xx, yy, zz) cell...
                    
                global_Template_during_hetero_DR_CPU[ccounter] = global_Template_during_hetero[ccounter];
                global_Template_during_hetero_RR_CPU[ccounter] = global_Template_during_hetero[ccounter];

                if ((xx == x) && (yy == y) && (zz == z)) 
                {
                    if (global_Template_during_hetero[ccounter] != 0) ERROR_PRINT();
                    global_Template_during_hetero_DR_CPU[ccounter] = 1;
                }
            }
        }
    }

    if (ccounter != cells_in_stencil) ERROR_PRINT();
}

long long int *global_Weights_during_hetero;
int *global_Count_Per_Cell_during_hetero;
int global_dimx_during_hetero;
int global_dimy_during_hetero;
int global_dimz_during_hetero;
int global_dimxy_during_hetero;
int global_ntasks_hetero;

void Compute_Hetero_Weights_R_Parallel(void *arg)
{
    //int threadid = omp_get_thread_num();
    int taskid   = (int)((size_t)(arg));
    long long int *Weights      = global_Weights_during_hetero;
    int *Count_Per_Cell         = global_Count_Per_Cell_during_hetero;

    int dimx                    = global_dimx_during_hetero;
    int dimy                    = global_dimy_during_hetero;
    int dimz                    = global_dimz_during_hetero;
    int dimxy                   = global_dimxy_during_hetero;
     
    if (1)
    {
        TYPE i_rminL = global_rminL_CPU;
        TYPE i_Lbox  = global_Lbox_CPU;
        TYPE i_rmaxL = global_rmaxL_CPU;

        TYPE rmax = i_rmaxL / i_Lbox;
        TYPE rmin = i_rminL / i_Lbox;

        TYPE rmax_2 = rmax * rmax;

        TYPE *Cell_Width = global_grid_R_CPU.Cell_Width;

        TYPE Range0_X[2], Range0_Y[2], Range0_Z[2];
        TYPE Range1_X[2], Range1_Y[2], Range1_Z[2];

        int dx = int(ceil((i_rmaxL/i_Lbox)*dimx)) + 0;
        int dy = int(ceil((i_rmaxL/i_Lbox)*dimy)) + 0;
        int dz = int(ceil((i_rmaxL/i_Lbox)*dimz)) + 0;

        int ntasks = global_ntasks_hetero;
//if (node_id == 0) printf("dx = %d ::: dy = %d ::: dz = %d\n", dx, dy, dz);
        int starting_cell_index = global_starting_cell_index_R_CPU;
        int   ending_cell_index = global_ending_cell_index_R_CPU;

        int cells = ending_cell_index - starting_cell_index;
        int cells_per_task = (cells + ntasks - 1)/ntasks;

        //printf("cells_per_thread = %d\n", cells_per_thread);

        int starting_cell_index_taskid = starting_cell_index + cells_per_task * (taskid + 0);
        int   ending_cell_index_taskid = starting_cell_index + cells_per_task * (taskid + 1);

        if (starting_cell_index_taskid > ending_cell_index) starting_cell_index_taskid = ending_cell_index;
        if (  ending_cell_index_taskid > ending_cell_index)   ending_cell_index_taskid = ending_cell_index;

//if (node_id == 0) printf("start_index = %d ::: end_index = %d\n", starting_cell_index_threadid, ending_cell_index_threadid);
        //for(int current_cell_index = starting_cell_index; current_cell_index < ending_cell_index; current_cell_index++)
        for(int current_cell_index = starting_cell_index_taskid; current_cell_index < ending_cell_index_taskid; current_cell_index++)
        {
            //printf("current_cell_index = %d\n", current_cell_index);
            //if (global_Owner[current_cell_index] != node_id) continue;

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

                    for(int xx = (x - dx); xx <= (x + dx); xx++)
                    {
                        if (!global_Template_during_hetero[ccounter++]) continue;
                        //Our neighbor is the (xx, yy, zz) cell...
                        //if ((xx == x) && (yy == y) && (zz == z)) continue;

                        ////////////////////////////////////////////////////////////////////////////////////////
                        //Step A: Figure out if the nearest points between the grids is >= rmax...
                        ////////////////////////////////////////////////////////////////////////////////////////
                        
                        //Range1_X[0] = (xx*1.00)*Cell_Width[0]; Range1_X[1] = ((xx+1)*1.00)*Cell_Width[0];
                        //Range1_Y[0] = (yy*1.00)*Cell_Width[1]; Range1_Y[1] = ((yy+1)*1.00)*Cell_Width[1];
                        //Range1_Z[0] = (zz*1.00)*Cell_Width[2]; Range1_Z[1] = ((zz+1)*1.00)*Cell_Width[2];

                        //TYPE min_dist_2 = Find_Minimum_Distance_Between_Cells(Range0_X, Range0_Y, Range0_Z, Range1_X, Range1_Y, Range1_Z);

                        //if (min_dist_2 > rmax_2) continue;

                        ////////////////////////////////////////////////////////////////////////////////////////
                        //Step B: Collect Particles into consecutive positions and also respect PREIODICITY...
                        ////////////////////////////////////////////////////////////////////////////////////////

                        int xx_prime = xx; //, yy_prime = yy, zz_prime = zz;
                        if (xx < 0) xx_prime = xx + dimx; else if (xx >= (dimx)) xx_prime = xx - dimx;
                        //if (yy < 0) yy_prime = yy + dimy; else if (yy >= (dimy)) yy_prime = yy - dimy;
                        //if (zz < 0) zz_prime = zz + dimz; else if (zz >= (dimz)) zz_prime = zz - dimz;

                        //if ( (xx_prime < 0) || (xx_prime >= dimx))  ERROR_PRINT();
                        //if ( (yy_prime < 0) || (yy_prime >= dimy))  ERROR_PRINT();
                        //if ( (zz_prime < 0) || (zz_prime >= dimz))  ERROR_PRINT();

                        int neighbor_cell_index = base_cell_index + xx_prime; //GET_CELL_INDEX(xx_prime, yy_prime, zz_prime);
                        if (neighbor_cell_index > current_cell_index) continue;

                        int objects_in_neighboring_cell = Count_Per_Cell[neighbor_cell_index];
                        local_weight += objects_in_this_cell * objects_in_neighboring_cell;
                    }
                }
            }

            Weights[current_cell_index] = local_weight;
        }
    }
}

void Compute_Hetero_Weights_R(int *Count_Per_Cell, long long *Weights, int starting_cell_index, int ending_cell_index, int dimx, int dimy, int dimz, int dimxy)
{
    if (node_id_CPU == 0) printf("Inside Compute_Hetero_Weights_R\n");
    global_Weights_during_hetero = Weights;
    global_Count_Per_Cell_during_hetero = Count_Per_Cell;

    global_dimx_during_hetero = dimx;
    global_dimy_during_hetero = dimy;
    global_dimz_during_hetero = dimz;
    global_dimxy_during_hetero = dimxy;

    int ntasks = 16 * nthreads_CPU;

    global_ntasks_hetero = ntasks;

    #pragma omp parallel for schedule(dynamic) num_threads(nthreads_CPU)
    for(int p = 0 ; p < ntasks; p++)
    {
        Compute_Hetero_Weights_R_Parallel((void *)(p));
    }
}

void CPU_Compute_Hetero_Weights(void)
{
    unsigned long long int stime = ___rdtsc();

    int ncores_cpu = nthreads_CPU/1; //1 thread  per core for CPU
    int ncores_mic = NTHREADS_MIC/4; //4 threads per core for MIC

    long long int cpu_potency = ncores_cpu * SIMD_WIDTH_CPU * CORE_FREQUENCY_CPU;
    long long int mic_potency = ncores_mic * SIMD_WIDTH_MIC * CORE_FREQUENCY_MIC;

    double actual_cpu_potency = cpu_potency * 1.00;
    double actual_mic_potency = mic_potency * 0.70; // original 0.75...

    double fraction_work_cpu = (actual_cpu_potency)/(actual_cpu_potency + actual_mic_potency);

    //fraction_work_cpu = 1.00; //[0..1]

    PRINT_BLUE
    if (node_id_CPU == 0) printf("<<%d>> ::: fraction_work_cpu = %lf\n", node_id_CPU, fraction_work_cpu);
    PRINT_BLACK


    global_hetero_dimx = global_grid_D_CPU.dimx;
    global_hetero_dimy = global_grid_D_CPU.dimy;
    global_hetero_dimz = global_grid_D_CPU.dimz;
    global_hetero_dimxy = global_hetero_dimx * global_hetero_dimy;
    global_hetero_number_of_uniform_subdivisions = global_grid_D_CPU.number_of_uniform_subdivisions;

    Compute_Template_During_Hetero(&global_grid_D_CPU, global_hetero_dimx, global_hetero_dimy, global_hetero_dimz, global_hetero_dimxy);


    size_t sz = global_hetero_number_of_uniform_subdivisions * sizeof(long long int);
    long long int *global_Hetero_Weights_D = (long long int *)malloc(sz);
    long long int *global_Hetero_Weights_R = (long long int *)malloc(sz);

//==========================================================================================================================================
    {
        long long int total_weight_D = 0;
        for(int cell_id = global_starting_cell_index_D_CPU; cell_id < global_ending_cell_index_D_CPU; cell_id++)
        {
            //printf("cell_id = %d\n", cell_id);
            //if (!global_Required_D_CPU[cell_id]) ERROR_PRINT();
            global_Hetero_Weights_D[cell_id] = global_grid_D_CPU.Count_Per_Cell[cell_id];
            total_weight_D += global_Hetero_Weights_D[cell_id];
        }

        if (node_id_CPU == 0) printf("Total Weight_D = %lld\n", total_weight_D);

        {
            double cpu_weight_D = (double)(total_weight_D) * fraction_work_cpu;
            double weight_accumulated_D = 0;
            int cell_id;
            for(cell_id = global_starting_cell_index_D_CPU; cell_id < global_ending_cell_index_D_CPU; cell_id++)
            {
                if (weight_accumulated_D >= cpu_weight_D) break;
                weight_accumulated_D += global_Hetero_Weights_D[cell_id];
            }

            //if (cell_id == global_ending_cell_index_D_CPU) ERROR_PRINT();
            global_hetero_cpu_number_of_D_cells_CPU = cell_id - global_starting_cell_index_D_CPU;
            if (node_id_CPU == 0) printf("<<%d>> ::: global_hetero_cpu_number_of_D_cells_CPU = %d (out of %d)::: Percentage = %.2lf %%\n", 
                    node_id_CPU, global_hetero_cpu_number_of_D_cells_CPU, (global_ending_cell_index_D_CPU - global_starting_cell_index_D_CPU), 
                    (global_hetero_cpu_number_of_D_cells_CPU*100.0)/(global_ending_cell_index_D_CPU - global_starting_cell_index_D_CPU));
        }
    }
//==========================================================================================================================================

    {
        //long long int *Local_Weight_R = (long long int *)malloc(nthreads_CPU * 16 * sizeof(long long int));
        //for(int k = 0; k < nthreads_CPU; k++) Local_Weight_R[16*k] = 0;

        //The function below fills up global_Hetero_Weights_R...
        Compute_Hetero_Weights_R(global_grid_R_CPU.Count_Per_Cell, global_Hetero_Weights_R, global_starting_cell_index_R_CPU, global_ending_cell_index_R_CPU, global_hetero_dimx, global_hetero_dimy, global_hetero_dimz, global_hetero_dimxy);
#if 0
        {
            for(int cell_id = global_starting_cell_index_R_CPU; cell_id < global_ending_cell_index_R_CPU; cell_id++)
            {
                printf("%d : %lld\n", cell_id, global_Hetero_Weights_R[cell_id]);
            }
        }
#endif
        long long int total_weight_R = 0;
        for(int cell_id = global_starting_cell_index_R_CPU; cell_id < global_ending_cell_index_R_CPU; cell_id++)
        {
            total_weight_R += global_Hetero_Weights_R[cell_id];
        }

        if (node_id_CPU == 0) printf("Total Weight_R = %lld\n", total_weight_R);
 
        {
            double cpu_weight_R = (double)(total_weight_R) * fraction_work_cpu;
            double weight_accumulated_R = 0;
            int cell_id;
            for(cell_id = global_starting_cell_index_R_CPU; cell_id < global_ending_cell_index_R_CPU; cell_id++)
            {
                if (weight_accumulated_R >= cpu_weight_R) break;
                weight_accumulated_R += global_Hetero_Weights_R[cell_id];
            }

            //if (cell_id == global_ending_cell_index_R_CPU) ERROR_PRINT();
            global_hetero_cpu_number_of_R_cells_CPU = cell_id - global_starting_cell_index_R_CPU;
            if (node_id_CPU == 0) printf("<<%d>> ::: global_hetero_cpu_number_of_R_cells_CPU = %d (out of %d)::: Percentage = %.2lf %%\n", 
                    node_id_CPU, global_hetero_cpu_number_of_R_cells_CPU, (global_ending_cell_index_R_CPU - global_starting_cell_index_R_CPU), 
                    (global_hetero_cpu_number_of_R_cells_CPU*100.0)/(global_ending_cell_index_R_CPU - global_starting_cell_index_R_CPU));
        }       
    }



    unsigned long long int etime = ___rdtsc();
    unsigned long long int ttime = etime - stime;

    if (node_id_CPU == 0) printf("Time Taken For Hetero Work Division = %lld cycles (%.2lf seconds)\n", ttime, ttime/CORE_FREQUENCY_CPU);
    //exit(123);
}

#endif

#endif

#ifdef MPI_COMPUTATION

#define PCL_MAX(a,b) (((a) > (b)) ? (a) : (b))

unsigned long long int global_time_kdtree_d = 0;
unsigned long long int global_time_kdtree_r = 0;


#define PTHREAD

#ifdef PTHREAD
#include <pthread.h> 
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

#endif  //PTHREAD


#ifdef PTHREAD
//global value
static volatile int version = 0;
static volatile int gcount = 0;		

void old_barrier()
{
	pthread_mutex_lock(&complete_mutex);
	gcount += 1;
	if(gcount == nthreads_CPU){	
		gcount = 0;
		pthread_cond_broadcast(&complete_cond);	
	}
	else{
		pthread_cond_wait(&complete_cond,&complete_mutex);
	}
	pthread_mutex_unlock(&complete_mutex);
}

//#define NULL_BARRIER
//#define PTHREAD_BARRIER
//#define CONDITIONAL_BARRIER
//#define SPIN_BARRIER
#define DH1_BARRIER
//#define DH2_BARRIER
//#define LIGHT_BARRIER
void barrier(int threadid=0)
{

//  printf("tid: %d\n", threadid);
#ifdef NULL_BARRIER
	
#endif
#ifdef PTHREAD_BARRIER
    pthread_barrier_wait(&mybarrier);
#endif
#ifdef CONDITIONAL_BARRIER
	pthread_mutex_lock(&barrier_mutex);
	gcount += 1;
	if(gcount == nthreads_CPU){	
		gcount = 0;
		pthread_cond_broadcast(&barrier_cond);	
	}
	else{
		pthread_cond_wait(&barrier_cond,&barrier_mutex);
	}
	pthread_mutex_unlock(&barrier_mutex);
#endif
#ifdef SPIN_BARRIER
	int myversion = version+1;
	pthread_mutex_lock(&barrier_mutex);
	if(gcount < nthreads_CPU - 1){
		gcount++;
	}
	else{   //last thread sets back to zero
		gcount = 0;
		version++;
	}
	pthread_mutex_unlock(&barrier_mutex);
	do {
            __asm { pause };
	} while ( myversion != version); 
#endif
#ifdef DH1_BARRIER
  if (_barrier_turn_ == 0) {
    if (threadid == 0) {
      for (int i=1; i<nthreads_CPU; i++) {
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
      for (int i=1; i<nthreads_CPU; i++) {
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
#endif
#ifdef DH2_BARRIER
  if (_MY_BARRIER_TURN_) {
    asm __volatile__ ("lock incl %[mem]" :: [mem] "m" (*&_MY_BARRIER_COUNT_0));
    if (threadid == 0) {
      while (_MY_BARRIER_COUNT_0 < nthreads_CPU);
      _MY_BARRIER_TURN_ = 0;
      _MY_BARRIER_COUNT_1 = 0;
      _MY_BARRIER_FLAG_1 = 0;
      _MY_BARRIER_FLAG_0 = 1;
    }
    else  {
      while (!_MY_BARRIER_FLAG_0);
    }
  }
  else {
    asm __volatile__ ("lock incl %[mem]" :: [mem] "m" (*&_MY_BARRIER_COUNT_1));
    if (threadid == 0) {
      while (_MY_BARRIER_COUNT_1 < nthreads_CPU);
        _MY_BARRIER_TURN_ = 1;
        _MY_BARRIER_COUNT_0 = 0;
        _MY_BARRIER_FLAG_0 = 0;
        _MY_BARRIER_FLAG_1 = 1;
      }
      else {
        while (!_MY_BARRIER_FLAG_1);
      }
  }
#endif
#ifdef LIGHT_BARRIER
	int myversion = version+1;
	if(gcount < nthreads_CPU - 1) {
         __asm { lock inc dword ptr [gcount] };
	}
	else{   //last thread sets back to zero
		gcount = 0;
		version++;
     }
     do {
         __asm { pause };
     } while ( myversion != version); 
	
#endif	
}

extern int global_number_of_phases;



#if 0
void barrier3(int threadid, int phase, int iteration)
{
    if (nthreads_CPU == 1) return;

    // This function assumes that nthreads_CPU >= 2
    int current_phase = (phase+1) + iteration * (global_number_of_phases+1);

    global_Timestamp[threadid*16] = current_phase;

    if (threadid == 0)
    {
        //threadid == (0) so no prev neighbor...
        
        while ( (global_Timestamp[16*(threadid+1)] < current_phase))
        {
        }
    }

    else if (threadid < (nthreads_CPU-1))
    {
        while ( (global_Timestamp[16*(threadid-1)] < current_phase) || (global_Timestamp[16*(threadid+1)] < current_phase))
        {
        }
    }
    else
    {
        //threadid == (nthreads_CPU-1) so no next neighbor...
        while ( (global_Timestamp[16*(threadid-1)] < current_phase))
        {
        }
    }
}
#else

void barrier3(int threadid, int phase, int iteration)
{
    barrier(threadid);
}

#endif

#endif
#define my_malloc malloc
#define my_free free

long long int global_number_of_galaxies = 0;
unsigned char *global_Required_D;
int **Ranges1;
int **Ranges2;
int *Ranges12_Max_Size;

#define MPI_BARRIER(nodeid) MPI_Barrier(MPI_COMM_WORLD);

MPI_Request *recv_request;
MPI_Request *send_request_key;
MPI_Status *recv_status;
   
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

void Allocate_Temporary_Arrays(void)
{
    Ranges1 = (int **)my_malloc(nthreads_CPU * sizeof(int *)); 
    Ranges2 = (int **)my_malloc(nthreads_CPU * sizeof(int *)); 
    Ranges12_Max_Size = (int *)my_malloc(nthreads_CPU * sizeof(int));

    for(int i=0; i<nthreads_CPU; i++) Ranges12_Max_Size[i] = 1024;
    for(int i=0; i<nthreads_CPU; i++) 
    {
        Ranges1[i] = (int *)my_malloc(Ranges12_Max_Size[i] * sizeof(int));
        Ranges2[i] = (int *)my_malloc(Ranges12_Max_Size[i] * sizeof(int));
    }
}

#define MY_BARRIER(threadid) barrier(threadid)


#define my_another_malloc malloc
#define my_another_free(X, Y) free(X)

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


#if 0

#endif
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
        //debug_printf("Ranges being compressed from (%d) ranges --> (%d) ranges\n", count, j);
        *xcount = j;
    }

    //Otherwise *xcount does not change, and it not re-written :)
}


void Compute_KD_Tree(int threadid, TYPE *Pos, int number_of_particles, Grid *grid, unsigned char *Required)
{
    //++ Note that all the threads are executing this piece of code...
/////////////////////////////////////////////////////////////////////////
//Phase-I Let's first perform a uniform subdivision...
/////////////////////////////////////////////////////////////////////////

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

    int particles_per_thread = (number_of_particles % nthreads_CPU) ? (number_of_particles/nthreads_CPU+1): (number_of_particles/nthreads_CPU);
    int starting_index = particles_per_thread * threadid;
    int ending_index = starting_index + particles_per_thread;

    if (starting_index > number_of_particles) starting_index = number_of_particles;
    if (ending_index > number_of_particles) ending_index = number_of_particles;

#if 0
    FILE *hp;
    if (node_id == 1) hp = fopen("MM", "w");
#endif
    
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
            printf("node_id = %d ::: i = %d\n", node_id_CPU, i);
            ERROR_PRINT();
        }
    #if 0
        if (cell_id == debug_cell_id) 
        { 
            if (node_id != 1) ERROR_PRINT(); 
            fprintf(hp, "%f %f %f\n", Pos[3*i+0], Pos[3*i+1], Pos[3*i+2]); 
        }
    #endif
    }
    
#if 0
    if (node_id == 1) fclose(hp);
    printf("< %d > Beta: %d (%d)\n", node_id, Count[debug_cell_id], debug_cell_id);
#endif


    int cells_per_thread = (total_number_of_cells % nthreads_CPU) ? (total_number_of_cells/nthreads_CPU + 1) : (total_number_of_cells/nthreads_CPU);
    int starting_cell_index = threadid * cells_per_thread;
    int ending_cell_index = starting_cell_index +  cells_per_thread;

    if (starting_cell_index > total_number_of_cells) starting_cell_index = total_number_of_cells;
    if (  ending_cell_index > total_number_of_cells)   ending_cell_index = total_number_of_cells;

    MY_BARRIER(threadid);

    //////////////////////////////////////////////////////////////////////////////////////////
    //Step B: Parallelized Prefix Computation in steps... There is a prefix sum per cell...
    //////////////////////////////////////////////////////////////////////////////////////////

    size_t sz = 0;
    for(int cell_id = starting_cell_index; cell_id < ending_cell_index; cell_id++)
    {
        int prev_sum = 0;
        for(int thr=0; thr < nthreads_CPU; thr++)
        {
            int new_sum = prev_sum + grid->Count_Per_Thread[thr][cell_id];
            grid->Count_Per_Thread[thr][cell_id] = prev_sum;
            prev_sum = new_sum;
        }

        //We have now computed the total number of particles that will fall in bin "cell_id"...
        int particles_in_this_cell = prev_sum;
        //if (particles_in_this_cell  > max_number_of_particles) max_number_of_particles = particles_in_this_cell;

        grid->Count_Per_Cell[cell_id]  = particles_in_this_cell;
        sz += DIMENSIONS * particles_in_this_cell  * sizeof(TYPE); //XXXX YYYY ZZZZ
    }

    // Each thread will now malloc some memory... Note that 'malloc' will only be called ONCE by each thread :)
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

    //////////////////////////////////////////////////////////////////////////////////////////
    //Step C: Now Go and Populate the Cells...
    //////////////////////////////////////////////////////////////////////////////////////////

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
        grid->Positions[cell_id][index_to_write + 0 * particles_in_this_cell] = float_x;
        grid->Positions[cell_id][index_to_write + 1 * particles_in_this_cell] = float_y;
        grid->Positions[cell_id][index_to_write + 2 * particles_in_this_cell] = float_z;

        grid->Theta_Phi[cell_id][index_to_write + 0 * particles_in_this_cell] = theta;
        grid->Theta_Phi[cell_id][index_to_write + 1 * particles_in_this_cell] = phi;
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

/////////////////////////////////////////////////////////////////////////
//Phase-II: Divide the cells between threads so that each thread gets
//some sizeable number of cells...
/////////////////////////////////////////////////////////////////////////

        
    int start_cell = -1;
    int end_cell = -1;

    int particles_per_cell = number_of_particles/nthreads_CPU;
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


    if (node_id_CPU == (nnodes_CPU-1))
    {
        if (threadid == (nthreads_CPU -1))
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
        printf("<%d> :::: number_of_particles = %d\n", node_id_CPU, number_of_particles); fflush(stdout);
        ERROR_PRINT();
    }

#if 1
    MY_BARRIER(threadid);
    if (threadid == 0)
    {
        for(int tid=0; tid < nthreads_CPU; tid++)
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

    int threshold_particles_per_cell =  GLOBAL_THRESHOLD_PARTICLES_PER_CELL;

#if 0
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
#endif

    int cumm_particles = 0;

    int max_particles_for_this_thread = 0;
    for(int cell_id = 0; cell_id < start_cell; cell_id++) cumm_particles += grid->Count_Per_Cell[cell_id];
    for(int cell_id = start_cell; cell_id <= end_cell; cell_id++) max_particles_for_this_thread = PCL_MAX (max_particles_for_this_thread,  grid->Count_Per_Cell[cell_id]);

    TYPE *Temp_Pos = (TYPE *)my_another_malloc(max_particles_for_this_thread * 3 * sizeof(TYPE));

    {
        int maximum_number_of_particles = 0;
        int maximum_number_of_ranges = 0;
        debug_printf("node_id = %d ::: threadid = %d ::: start_cell = %d ::: end_cell = %d\n", node_id_CPU, threadid, start_cell, end_cell);

        for(int cell_id = start_cell; cell_id <= end_cell; cell_id++)
        {
            if (!Required[cell_id]) continue;
            //We only process the cell if it will eventuall be required... All the particles on this node only fills up cells for which global_Required[cell_id] == 1. :)
            //debug_printf("cell_id = %d\n", cell_id);

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
                //debug_printf("Ranges1[%d] = ", number_of_ranges); for(int k=0; k<number_of_ranges; k++) printf("[%d %d]", Ranges11[2*k], Ranges11[2*k+1]); printf("\n");
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

            //At the end of this, data sits in Temp_Pos, so that it can be transposed (SOA'd to global_grid->Positions)

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


#if 0
                if (grid->Bdry_Theta[cell_id][2*i + 0] > grid->Bdry_Theta[cell_id][2*i + 1])          SWAP_TYPE(grid->Bdry_Theta[cell_id][2*i + 0], grid->Bdry_Theta[cell_id][2*i + 1]);
                if (grid->Bdry_Sin_Theta[cell_id][2*i + 0] >  grid->Bdry_Sin_Theta[cell_id][2*i + 1]) SWAP_TYPE(grid->Bdry_Sin_Theta[cell_id][2*i + 0], grid->Bdry_Sin_Theta[cell_id][2*i + 1]);
                if (grid->Bdry_Cos_Theta[cell_id][2*i + 0] > grid->Bdry_Cos_Theta[cell_id][2*i + 1])  SWAP_TYPE(grid->Bdry_Cos_Theta[cell_id][2*i + 0], grid->Bdry_Cos_Theta[cell_id][2*i + 1]);

                if (grid->Bdry_Phi[cell_id][2*i + 0] > grid->Bdry_Phi[cell_id][2*i + 1]) SWAP_TYPE(grid->Bdry_Phi[cell_id][2*i + 0],  grid->Bdry_Phi[cell_id][2*i + 1]);
                if (grid->Bdry_Sin_Phi[cell_id][2*i + 0] > grid->Bdry_Sin_Phi[cell_id][2*i + 1]) SWAP_TYPE(grid->Bdry_Sin_Phi[cell_id][2*i + 0], grid->Bdry_Sin_Phi[cell_id][2*i + 1]);
                if (grid->Bdry_Cos_Phi[cell_id][2*i + 0] > grid->Bdry_Cos_Phi[cell_id][2*i + 1]) SWAP_TYPE(grid->Bdry_Cos_Phi[cell_id][2*i + 0], grid->Bdry_Cos_Phi[cell_id][2*i + 1]);
#endif
            }
        }

        //grid->Maximum_number_of_particles[threadid] = maximum_number_of_particles;
        //grid->Maximum_number_of_ranges[threadid] = maximum_number_of_ranges;
    }

#define KD_TREE_STATISTICS

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
                    node_id_CPU, number_of_particles, total_cells, zero_cells, min_particles, max_particles, (sum_particles * 1.0)/total_cells);
            PRINT_BLACK

            Perform_Elaborate_Checking(grid);
    
            //Spit_KD_Tree_Into_File(grid, "kdtree.bin");
        }
    }
#endif

    my_another_free(Temp_Pos, (max_particles_for_this_thread * 3 * sizeof(TYPE)));
}

void Copy_Non_Changing_Data_From_D_To_R(void)
{
    if (DIMENSIONS != 3) ERROR_PRINT();

    for(int p = 0; p < DIMENSIONS; p++) global_grid_R_CPU.Min[p] = global_grid_D_CPU.Min[p];
    for(int p = 0; p < DIMENSIONS; p++) global_grid_R_CPU.Max[p] = global_grid_D_CPU.Max[p];
    for(int p = 0; p < DIMENSIONS; p++) global_grid_R_CPU.Extent[p] = global_grid_D_CPU.Extent[p];
    for(int p = 0; p < DIMENSIONS; p++) global_grid_R_CPU.Cell_Width[p] = global_grid_D_CPU.Cell_Width[p];
}


void Initialize_MPI_Mallocs(void)
{
    static int calls = 0;
    calls++;
 
    if (calls == 1)
    {
        recv_request = (MPI_Request*)my_malloc(sizeof(MPI_Request)*(nnodes_CPU));
        send_request_key = (MPI_Request*)my_malloc(sizeof(MPI_Request)*(nnodes_CPU));
        recv_status  =  (MPI_Status*)my_malloc(sizeof(MPI_Status)*(nnodes_CPU));
    }
    else
    {
        ERROR_PRINT();
    }
}

 
void Read_D_R_File(char *filename, TYPE **i_Positions, long long int *i_number_of_galaxies_on_node)
{
    unsigned long long int stime = ___rdtsc();
    long long int number_of_galaxies_on_node = -1;
    //srand(95123);

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//STEP 1: READ THE FILES...
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //Setup Array...
//   global_number_of_galaxies = 20000;
//if (global_number_of_galaxies % 20000) global_number_of_galaxies = (global_number_of_galaxies/20000+1)*20000;

    FILE *fp = fopen(filename, "rb");
    if (fp == NULL)  
    {
        printf("filename = %s :::", filename);
        ERROR_PRINT_STRING("File not found"); 
    }
    else
    {
        if (node_id_CPU == 0)
        {
            PRINT_GREEN
            printf("<%d> :: Reading %s\n", node_id_CPU, filename);
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
    //if (!valid_file) global_number_of_galaxies = 40; //100*1000; //1269891; // 100000;
    
    fread(&local_number_of_galaxies, sizeof(long long int), 1, fp);

    if (global_number_of_galaxies == 0)
    {
        global_number_of_galaxies = local_number_of_galaxies;
    }
    else
    {
        if (global_number_of_galaxies != local_number_of_galaxies) ERROR_PRINT_STRING("global_number_of_galaxies != local_number_of_galaxies");
    }


    long long int number_of_galaxies_per_node = (global_number_of_galaxies + nnodes_CPU - 1)/nnodes_CPU;

    global_galaxies_starting_index = number_of_galaxies_per_node * (node_id_CPU + 0);
    global_galaxies_ending_index   = number_of_galaxies_per_node * (node_id_CPU + 1);

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
    node_id_CPU, global_number_of_galaxies, number_of_galaxies_on_node, DIMENSIONS);

    mpi_printf("global_galaxies_starting_index = %lld ::: global_galaxies_ending_index = %lld\n", global_galaxies_starting_index, global_galaxies_ending_index);
    //DIMENSIONS is the number of dimensions in the dataset...

    if (valid_file)
    {
        #ifdef LARRY 
        printf("Skipping reading valid data file for Larry\n");
        #else
        mpi_printf("%s is a valid data file... Hence reading from it...\n", filename);
        size_t sz = (size_t)(number_of_galaxies_on_node) * (size_t)(sizeof(TYPE));
        TYPE *temp_memory = (TYPE *)my_malloc(sz);

        for(int j=0; j<DIMENSIONS; j++)
        {
            long int starting_offset = 11 + sizeof(long long int)  + j * global_number_of_galaxies * sizeof(TYPE) + global_galaxies_starting_index * sizeof(TYPE);
            //printf("starting_offset = %lld\n", starting_offset);
            int ret_val = fseek(fp, starting_offset, SEEK_SET);
            if (ret_val) ERROR_PRINT();

            size_t items_read = fread(temp_memory, sizeof(TYPE), number_of_galaxies_on_node, fp);
            
            if (items_read != number_of_galaxies_on_node) ERROR_PRINT();

            //printf("TM[0] = %f\n", temp_memory[0]);
            for(long long int i=0; i<number_of_galaxies_on_node; i++) 
                GET_POINT(Positions, i, j, number_of_galaxies_on_node) = temp_memory[i];
        }
        #endif

        mpi_printf("<<%d>> %f %f %f\n", node_id_CPU, Positions[0 +3*79], Positions[1 + 3*79], Positions[2 + 3*79]);
        if (nnodes_CPU == 1)
        {
            int cell_id;
            cell_id = 0 + 79; mpi_printf("cell_id = [%f %f %f]\n", Positions[3*cell_id + 0], Positions[3*cell_id + 1], Positions[3*cell_id + 2]);
            cell_id = 317473 + 79; mpi_printf("cell_id = [%f %f %f]\n", Positions[3*cell_id + 0], Positions[3*cell_id + 1], Positions[3*cell_id + 2]);
            cell_id = 634946 + 79; mpi_printf("cell_id = [%f %f %f]\n", Positions[3*cell_id + 0], Positions[3*cell_id + 1], Positions[3*cell_id + 2]);
            cell_id = 952419 + 79; mpi_printf("cell_id = [%f %f %f]\n", Positions[3*cell_id + 0], Positions[3*cell_id + 1], Positions[3*cell_id + 2]);
        }
        //free(temp_memory);
    }
    else
    {
    }

    if (fp) fclose(fp);
    mpi_printf("nthreads_CPU = %d\n", nthreads_CPU);

    MPI_BARRIER(node_id_CPU);
    //int to_print = 17; printf("Positions_3D[%d] = %f\n", to_print, Positions_3D[to_print]);

    *i_Positions = Positions;
    *i_number_of_galaxies_on_node = number_of_galaxies_on_node;

    unsigned long long int etime = ___rdtsc();
    if (node_id_CPU == 0) printf("Time Taken to read (%s) = %lld cycles (%.2lf seconds)\n", filename, (etime - stime), (etime - stime)/CORE_FREQUENCY_CPU);
}

void *Compute_KD_Tree_Parallel_For_D(void *arg1)
{
    int threadid = (int)(size_t)(arg1);

    //Set_Affinity(threadid);
    //MY_BARRIER(threadid);

///////////////////////////////////////////////////////////////////////////////////////////
//Step 1: Compute KD_Tree of data...
///////////////////////////////////////////////////////////////////////////////////////////
    //unsigned long long int start_time, end_time;
    //start_time = ___rdtsc();
    Compute_KD_Tree(threadid, global_Positions_D, global_number_of_galaxies_on_node_D, &global_grid_D_CPU, global_Required_D);
    //MY_BARRIER(threadid);
    //end_time = ___rdtsc();
    //if (threadid == 0) global_time_kdtree += (end_time - start_time);
    return arg1;
}

void *Compute_KD_Tree_Parallel_For_R(void *arg1)
{
    int threadid = (int)(size_t)(arg1);

    //Set_Affinity(threadid);
    //MY_BARRIER(threadid);

///////////////////////////////////////////////////////////////////////////////////////////
//Step 1: Compute KD_Tree of data...
///////////////////////////////////////////////////////////////////////////////////////////
    //unsigned long long int start_time, end_time;
    //start_time = ___rdtsc();
    Compute_KD_Tree(threadid, global_Positions_R, global_number_of_galaxies_on_node_R, &global_grid_R_CPU, global_Required_R_CPU);
    
    //MY_BARRIER(threadid);
    //end_time = ___rdtsc();
    //if (threadid == 0) global_time_kdtree += (end_time - start_time);
    return arg1;
}

void Compute_KD_Tree_Acceleration_Data_Structure_For_D(void)
{
    unsigned long long int start_time = ___rdtsc();
    //====
#if 0
    for(int i=1; i<nthreads; i++) pthread_create(&threads[i], NULL, Compute_KD_Tree_Parallel_For_D, (void *)(i));
    Compute_KD_Tree_Parallel_For_D(0);
    for(int i=1; i<nthreads; i++) pthread_join(threads[i], NULL);
#else
#pragma omp parallel for num_threads(nthreads_CPU)
    for(int i = 0; i < nthreads_CPU; i++)
    {
        Compute_KD_Tree_Parallel_For_D((void *)(i));
    }
#endif
    //====
    unsigned long long int end_time = ___rdtsc();
    global_time_kdtree_d += (end_time - start_time);
}

void Compute_KD_Tree_Acceleration_Data_Structure_For_R(void)
{
    unsigned long long int start_time = ___rdtsc();
    //====
#if 0
    for(int i=1; i<nthreads; i++) pthread_create(&threads[i], NULL, Compute_KD_Tree_Parallel_For_R, (void *)(i));
    Compute_KD_Tree_Parallel_For_R(0);
    for(int i=1; i<nthreads; i++) pthread_join(threads[i], NULL);
#else
#pragma omp parallel for num_threads(nthreads_CPU)
    for(int i = 0; i < nthreads_CPU; i++)
    {
        Compute_KD_Tree_Parallel_For_R((void *)(i));
    }
#endif
    //====
    unsigned long long int end_time = ___rdtsc();
    global_time_kdtree_r += (end_time - start_time);
}

 
void  Copy_D_Or_R_To_Temp_Memory(Grid *grid, size_t *Data_Transfer_Size, int **i_Temp_Memory)
{
    unsigned long long int stime = ___rdtsc();

    //DataTransfer_size_from_CPU_To_MIC = 0;

    //1st thing...
    //dimx(4) + dimy(4) + dimz(4) + number_of_uniform_subidvisions(4) + Cell_Width(3*TYPE) + Min(3*TYPE) + Max(3*TYPE) 
    size_t sz1 = 4 * sizeof(int) + 9 * sizeof(TYPE);

    int dimx = grid->dimx;
    int dimy = grid->dimy;
    int dimz = grid->dimz;

    int number_of_uniform_subdivisions = grid->number_of_uniform_subdivisions;

    //2nd thing... Count_Per_Cell + Number_of_Kd_subdivisions...
    size_t sz2 = 2 * number_of_uniform_subdivisions * sizeof(int);

    int total_number_of_kd_tree_nodes = 0;
    int range_fields = 0;
    size_t total_number_of_particles = 0;
    for(int k = 0; k < number_of_uniform_subdivisions; k++)
    {
        total_number_of_kd_tree_nodes += grid->Number_of_kd_subdivisions[k];
        if (grid->Number_of_kd_subdivisions[k] > 0)
        {
            range_fields += (1 + grid->Number_of_kd_subdivisions[k]);
        }

        total_number_of_particles += grid->Count_Per_Cell[k];
    }

    //3rd thing... Range + Bdry_X + Bdry_Y + Bdry_Z + Positions...
    //16 because: range_fields[4] | total_number_of_kd_tree_nodes[4] | total_number_of_particles[8]

    size_t sz3 = 16 + range_fields * sizeof(int) + 3 * total_number_of_kd_tree_nodes * 2 * sizeof(TYPE) + total_number_of_particles * 3 * sizeof(TYPE);

    //4 * 4 to store global_starting_cell_index_D/R and global_ending_cell_index_D/R
    size_t sz4 = 2 * number_of_uniform_subdivisions * sizeof(char) + 4 * 4; //sz4 is not stored :)

    size_t sz0 = 32; //length of packet, sz1, sz2, sz3...

    size_t sz = sz0 + sz1 + sz2 + sz3 + sz4;

    //printf("sz = %u bytes (%.2lf GB)\n", sz, sz/1000.0/1000.0/1000.0);

    /*
     *
     * PACKET FORMAT:
     * 8 Bytes: Length of Packet
     * 8 Bytes SZ1
     * 8 Bytes SZ2
     * 8 Bytes SZ3
     * SZ1 Bytes of Data... [ dimx + dimy + dimz + number_of_uniform_subdivisions + Cell_Width[3] + Min[3] + Max[3]
     * SZ2 Bytes of Data... [ Count_Per_Cell[number_of_uniform_subdivisions] + Number_of_Kd_subdivisions[number_of_uniform_subdivisions]...
     * SZ3 Bytes of Data... [ 16 + Range_Fields[range_fields] + 
     *
     */

    int *Temp_Memory = (int *)malloc(sz);

    {
        *((size_t *)(Temp_Memory + 0)) = sz;
        *((size_t *)(Temp_Memory + 2)) = sz1;
        *((size_t *)(Temp_Memory + 4)) = sz2;
        *((size_t *)(Temp_Memory + 6)) = sz3;
        *(Temp_Memory + 8 + 0) = dimx;
        *(Temp_Memory + 8 + 1) = dimy;
        *(Temp_Memory + 8 + 2) = dimz;
        *(Temp_Memory + 8 + 3) = number_of_uniform_subdivisions;
        *((TYPE *)(Temp_Memory + 12 + 0)) = grid->Cell_Width[0];
        *((TYPE *)(Temp_Memory + 12 + 1)) = grid->Cell_Width[1];
        *((TYPE *)(Temp_Memory + 12 + 2)) = grid->Cell_Width[2];

        *((TYPE *)(Temp_Memory + 12 + 3)) = grid->Min[0];
        *((TYPE *)(Temp_Memory + 12 + 4)) = grid->Min[1];
        *((TYPE *)(Temp_Memory + 12 + 5)) = grid->Min[2];

        *((TYPE *)(Temp_Memory + 12 + 6)) = grid->Max[0];
        *((TYPE *)(Temp_Memory + 12 + 7)) = grid->Max[1];
        *((TYPE *)(Temp_Memory + 12 + 8)) = grid->Max[2];

        int *CPC = Temp_Memory + 21;
        for(int p = 0; p < number_of_uniform_subdivisions; p++) CPC[p] = grid->Count_Per_Cell[p];

        int *KDS = CPC + number_of_uniform_subdivisions;
        for(int p = 0; p < number_of_uniform_subdivisions; p++) KDS[p] = grid->Number_of_kd_subdivisions[p];

        int *S3_Start = KDS + number_of_uniform_subdivisions;

        S3_Start[0] = range_fields;
        S3_Start[1] = total_number_of_kd_tree_nodes;
        *((size_t *)(S3_Start + 2)) = total_number_of_particles;

        int *Range_Dst = S3_Start + 4;
        TYPE *Bdry_X = (TYPE *)(Range_Dst + range_fields);
        TYPE *Bdry_Y = Bdry_X + 2 * total_number_of_kd_tree_nodes;
        TYPE *Bdry_Z = Bdry_Y + 2 * total_number_of_kd_tree_nodes;
        TYPE *Pos    = Bdry_Z + 2 * total_number_of_kd_tree_nodes;

        int range_fields_counter = 0;
        int total_number_of_kd_tree_nodes_counter = 0;
        size_t total_number_of_particles_counter = 0;

        for(int k = 0; k < number_of_uniform_subdivisions; k++)
        {
            if (grid->Number_of_kd_subdivisions[k])
            {
                for(int p = 0; p <= grid->Number_of_kd_subdivisions[k]; p++, range_fields_counter++)
                {
                    Range_Dst[range_fields_counter] = grid->Range[k][p];
                }

                for(int p = 0; p < grid->Number_of_kd_subdivisions[k]; p++, total_number_of_kd_tree_nodes_counter++)
                {
                    Bdry_X[2*total_number_of_kd_tree_nodes_counter + 0] = grid->Bdry_X[k][2*p + 0];
                    Bdry_X[2*total_number_of_kd_tree_nodes_counter + 1] = grid->Bdry_X[k][2*p + 1];

                    Bdry_Y[2*total_number_of_kd_tree_nodes_counter + 0] = grid->Bdry_Y[k][2*p + 0];
                    Bdry_Y[2*total_number_of_kd_tree_nodes_counter + 1] = grid->Bdry_Y[k][2*p + 1];

                    Bdry_Z[2*total_number_of_kd_tree_nodes_counter + 0] = grid->Bdry_Z[k][2*p + 0];
                    Bdry_Z[2*total_number_of_kd_tree_nodes_counter + 1] = grid->Bdry_Z[k][2*p + 1];
                }

                for(int p = 0; p < grid->Count_Per_Cell[k]; p++, total_number_of_particles_counter++)
                {
                    Pos[3 * total_number_of_particles_counter + 0] = grid->Positions[k][3*p + 0];
                    Pos[3 * total_number_of_particles_counter + 1] = grid->Positions[k][3*p + 1];
                    Pos[3 * total_number_of_particles_counter + 2] = grid->Positions[k][3*p + 2];
                }
            }
        }

        if (number_of_uniform_subdivisions % 8) ERROR_PRINT();

        unsigned char *X = (unsigned char *)(Pos + 3 * total_number_of_particles);
        unsigned char *Y = X + number_of_uniform_subdivisions;
        unsigned char *Z = Y + number_of_uniform_subdivisions;
        *((int *)(Z +  0)) = global_starting_cell_index_D_CPU;
        *((int *)(Z +  4)) = global_ending_cell_index_D_CPU;
        *((int *)(Z +  8)) = global_starting_cell_index_R_CPU;
        *((int *)(Z + 12)) = global_ending_cell_index_R_CPU;

        for(int k = 0; k < number_of_uniform_subdivisions; k++) X[k] = global_Required_D_For_R_CPU[k];
        for(int k = 0; k < number_of_uniform_subdivisions; k++) Y[k] = global_Required_R_CPU[k];

        if (range_fields_counter != range_fields) ERROR_PRINT_STRING("range_fields_counter != range_fields");
        if (total_number_of_kd_tree_nodes_counter != total_number_of_kd_tree_nodes) ERROR_PRINT_STRING("total_number_of_kd_tree_nodes_counter != total_number_of_kd_tree_nodes");
        if (total_number_of_particles_counter != total_number_of_particles) ERROR_PRINT_STRING("total_number_of_particles_counter != total_number_of_particles");
    }

    *i_Temp_Memory = Temp_Memory;
    *Data_Transfer_Size = sz;

    unsigned long long int etime = ___rdtsc();
    unsigned long long int ttime = etime - stime;
    {
        double seconds = ttime/CORE_FREQUENCY_CPU;
        PRINT_LIGHT_RED
        if (node_id_CPU == 0) printf("Time Taken To Transfer (%lld bytes) = %lld cycles (%.2lf seconds) ::: %.2lf GB/sec\n", sz, ttime, seconds, sz/seconds/1000.0/1000.0/1000.0);
        PRINT_BLACK
    }
}

void  Spit_Output(Grid *grid, size_t Data_Transfer_Size, int *i_Temp_Memory, const char *filename)
{
    FILE *fp = fopen(filename, "wb");
    if (fp == NULL) ERROR_PRINT();

    PRINT_LIGHT_RED
    printf("Spitting out %s\n", filename);
    PRINT_BLACK

    fwrite(i_Temp_Memory, Data_Transfer_Size, 1, fp);

    fclose(fp);
}


void Copy_DR_To_Temp_Memory(void)
{
    Copy_D_Or_R_To_Temp_Memory(&global_grid_D_CPU, &DataTransfer_Size_D_From_CPU_To_MIC_CPU, &Temp_Memory_D_CPU);
    Copy_D_Or_R_To_Temp_Memory(&global_grid_R_CPU, &DataTransfer_Size_R_From_CPU_To_MIC_CPU, &Temp_Memory_R_CPU);

    //Spit_Output(&global_grid_D_CPU, DataTransfer_Size_D_From_CPU_To_MIC_CPU, Temp_Memory_D_CPU, "jch_D.bin");
    //Spit_Output(&global_grid_R_CPU, DataTransfer_Size_R_From_CPU_To_MIC_CPU, Temp_Memory_R_CPU, "jch_R.bin");
}


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




int *global_Owner_R = NULL;
int global_memory_malloced = 0;
int global_subdivision_per_node_during_initialization;
long long int *global_Weights_during_initialization;
int *global_Count_Per_Cell_during_initialization;
int global_dimx_during_initialization;
int global_dimxy_during_initialization;
int global_dimy_during_initialization;
int global_dimz_during_initialization;
unsigned char **global_Required_during_initialization;
int global_number_of_subdivisions_during_initialization;
unsigned char *global_Local_Required_All_Nodes_during_initialization;
int *global_Count_of_particles_to_send_during_initialization;
int *global_Send_Count_during_initialization;
TYPE *global_Data_To_Send_during_initialization;
int *global_Prefix_Sum_Count_of_particles_to_send_during_initialization;
TYPE *global_Local_Pos_during_initialization;

unsigned long long int global_time_mpi = 0;
int *global_Owner_D = NULL;
int *global_Prealloced_Send_Count = NULL;
int *global_Prealloced_Recv_Count = NULL;
int *global_Prealloced_Count_Per_Cell = NULL;
unsigned char *global_Template_during_initialization;
int *global_Template_Range_during_initialization;


#if 0
void *Compute_Required_R_Parallel(void *arg1)
{
    int threadid = (int)(size_t)(arg1);
    unsigned char *Required = global_Required_during_initialization[threadid];
    int number_of_subdivisions = global_number_of_subdivisions_during_initialization;

    int dimx                    = global_dimx_during_initialization;
    int dimy                    = global_dimy_during_initialization;
    int dimz                    = global_dimz_during_initialization;
    int dimxy                   = global_dimxy_during_initialization;
 
unsigned long long int stime = read_tsc();
    //global_Owner is read-only, and basically owners have already been decided...
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

            //for(int current_cell_index = number_of_subdivisions-1; current_cell_index >= 0; current_cell_index --)
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
            

            //[min_cell_id... max_cell_id] are both inclusive :)
            int cells = max_cell_id - min_cell_id + 1;
            int cells_per_thread = (cells + nthreads - 1)/nthreads;

            int start_index = (threadid + 0) * cells_per_thread; if (start_index > cells) start_index = cells;
            int   end_index = (threadid + 1) * cells_per_thread; if (  end_index > cells)   end_index = cells;

            start_index += min_cell_id;
            end_index += min_cell_id;

unsigned long long int e2time = read_tsc();
//if (node_id == 0) printf("(%d) (%s) ::: Time Taken = %lld (%.2lf seconds)\n", __LINE__, __FILE__, (e2time - stime), (e2time - stime)/CORE_FREQUENCY);
//if (node_id == 0) printf("dx = %d ::: dy = %d ::: dz = %d\n", dx, dy, dz);
//if (node_id == 0) printf("start_index = %d ::: end_index = %d\n", start_index, end_index);
//if (node_id == 0) printf("min_cell_id = %d ::: max_cell_id = %d\n", min_cell_id, max_cell_id);
            //for(int current_cell_index = min_cell_id; current_cell_index <= max_cell_id; current_cell_index ++)
            for(int current_cell_index = start_index; current_cell_index < end_index; current_cell_index++)
            {
                if (global_Owner_R[current_cell_index] != node_id) ERROR_PRINT();

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
                        int yy_prime = yy, zz_prime = zz;
                        if (yy < 0) yy_prime = yy + dimy; else if (yy >= (dimy)) yy_prime = yy - dimy;
                        if (zz < 0) zz_prime = zz + dimz; else if (zz >= (dimz)) zz_prime = zz - dimz;

			            int base_cell_index = GET_CELL_INDEX(0, yy_prime, zz_prime);

                        for(int xx = (x - dx); xx <= (x + dx); xx++)
                        {
                            if (!global_Template_during_initialization[ccounter++]) continue;
                            //Our neighbor is the (xx, yy, zz) cell...
                            //if ((xx == x) && (yy == y) && (zz == z)) continue;

                            ////////////////////////////////////////////////////////////////////////////////////////
                            //Step A: Figure out if the nearest points between the grids is >= rmax...
                            ////////////////////////////////////////////////////////////////////////////////////////
                            
                            //Range1_X[0] = (xx*1.00)*Cell_Width[0]; Range1_X[1] = ((xx+1)*1.00)*Cell_Width[0];
                            //Range1_Y[0] = (yy*1.00)*Cell_Width[1]; Range1_Y[1] = ((yy+1)*1.00)*Cell_Width[1];
                            //Range1_Z[0] = (zz*1.00)*Cell_Width[2]; Range1_Z[1] = ((zz+1)*1.00)*Cell_Width[2];

                            //TYPE min_dist_2 = Find_Minimum_Distance_Between_Cells(Range0_X, Range0_Y, Range0_Z, Range1_X, Range1_Y, Range1_Z);

                            //if (min_dist_2 > rmax_2) continue;

                            ////////////////////////////////////////////////////////////////////////////////////////
                            //Step B: Collect Particles into consecutive positions and also respect PREIODICITY...
                            ////////////////////////////////////////////////////////////////////////////////////////

                            int xx_prime = xx; //, yy_prime = yy, zz_prime = zz;
                            if (xx < 0) xx_prime = xx + dimx; else if (xx >= (dimx)) xx_prime = xx - dimx;
                            //if (yy < 0) yy_prime = yy + dimy; else if (yy >= (dimy)) yy_prime = yy - dimy;
                            //if (zz < 0) zz_prime = zz + dimz; else if (zz >= (dimz)) zz_prime = zz - dimz;

                            //if ( (xx_prime < 0) || (xx_prime >= dimx))  ERROR_PRINT();
                            //if ( (yy_prime < 0) || (yy_prime >= dimy))  ERROR_PRINT();
                            //if ( (zz_prime < 0) || (zz_prime >= dimz))  ERROR_PRINT();

                            int neighbor_cell_index = base_cell_index + xx_prime; //GET_CELL_INDEX(xx_prime, yy_prime, zz_prime);
                            if (neighbor_cell_index > current_cell_index) continue;

                            Required[neighbor_cell_index] = 1;
                            printf("%d ", neighbor_cell_index);
                        }
                        printf("\n--- current_cell_index = %d ::: yy = %d ::: zz = %d \n", current_cell_index, yy, zz);
                    }
                }
            }
        }

        MY_BARRIER(threadid); 
unsigned long long int e1time = read_tsc();
if ((node_id == 0) && (threadid == 0))printf("(%d) (%s) ::: Time Taken = %lld (%.2lf seconds)\n", __LINE__, __FILE__, (e1time - stime), (e1time - stime)/CORE_FREQUENCY);

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
unsigned long long int e2time = read_tsc();
if ((node_id == 0) && (threadid == 0)) printf("(%d) (%s) ::: Time Taken = %lld (%.2lf seconds)\n", __LINE__, __FILE__, (e2time - stime), (e2time - stime)/CORE_FREQUENCY);


        return arg1;

}
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
unsigned long long int stime = ___rdtsc();
    //global_Owner is read-only, and basically owners have already been decided...
        if (1)
        {
            TYPE i_rminL = global_rminL_CPU;
            TYPE i_Lbox  = global_Lbox_CPU;
            TYPE i_rmaxL = global_rmaxL_CPU;

            TYPE rmax = i_rmaxL / i_Lbox;
            TYPE rmin = i_rminL / i_Lbox;

            TYPE rmax_2 = rmax * rmax;

            TYPE *Cell_Width = global_grid_R_CPU.Cell_Width;

            TYPE Range0_X[2], Range0_Y[2], Range0_Z[2];
            TYPE Range1_X[2], Range1_Y[2], Range1_Z[2];

            int dx = int(ceil((i_rmaxL/i_Lbox)*dimx)) + 0;
            int dy = int(ceil((i_rmaxL/i_Lbox)*dimy)) + 0;
            int dz = int(ceil((i_rmaxL/i_Lbox)*dimz)) + 0;

            int min_cell_id = (1<<29);
            int max_cell_id = -1;

            for(int current_cell_index = 0; current_cell_index < number_of_subdivisions; current_cell_index ++)
            {
                if (global_Owner_R[current_cell_index] != node_id_CPU) continue;
                min_cell_id = current_cell_index;
                break;
            }

            //for(int current_cell_index = number_of_subdivisions-1; current_cell_index >= 0; current_cell_index --)
            for(int current_cell_index = min_cell_id; current_cell_index < number_of_subdivisions; current_cell_index++)
            {
                if (global_Owner_R[current_cell_index] != node_id_CPU) 
                {
                    max_cell_id = current_cell_index - 1;
                    break;
                }
            }

            if (min_cell_id == (1<<29)) ERROR_PRINT();
            if (max_cell_id == -1)
            {
                if (node_id_CPU != (nnodes_CPU-1)) ERROR_PRINT();
                max_cell_id = number_of_subdivisions - 1;
            }
            

            //[min_cell_id... max_cell_id] are both inclusive :)
            int cells = max_cell_id - min_cell_id + 1;
            int cells_per_thread = (cells + nthreads_CPU - 1)/nthreads_CPU;

            int start_index = (threadid + 0) * cells_per_thread; if (start_index > cells) start_index = cells;
            int   end_index = (threadid + 1) * cells_per_thread; if (  end_index > cells)   end_index = cells;

            start_index += min_cell_id;
            end_index += min_cell_id;

unsigned long long int e2time = ___rdtsc();
//if (node_id == 0) printf("(%d) (%s) ::: Time Taken = %lld (%.2lf seconds)\n", __LINE__, __FILE__, (e2time - stime), (e2time - stime)/CORE_FREQUENCY);
//if (node_id == 0) printf("dx = %d ::: dy = %d ::: dz = %d\n", dx, dy, dz);
//if (node_id == 0) printf("start_index = %d ::: end_index = %d\n", start_index, end_index);
//if (node_id == 0) printf("min_cell_id = %d ::: max_cell_id = %d\n", min_cell_id, max_cell_id);
            //for(int current_cell_index = min_cell_id; current_cell_index <= max_cell_id; current_cell_index ++)
            for(int current_cell_index = start_index; current_cell_index < end_index; current_cell_index++)
            {
                if (global_Owner_R[current_cell_index] != node_id_CPU) ERROR_PRINT();

                Required[current_cell_index] = 1;
                int x, y, z; z = current_cell_index / dimxy; int xy = current_cell_index % dimxy; y = xy / dimx; x = xy % dimx;
        
                //Range0_X[0] = (x*1.00)*Cell_Width[0]; Range0_X[1] = ((x+1)*1.00)*Cell_Width[0];
                //Range0_Y[0] = (y*1.00)*Cell_Width[1]; Range0_Y[1] = ((y+1)*1.00)*Cell_Width[1];
                //Range0_Z[0] = (z*1.00)*Cell_Width[2]; Range0_Z[1] = ((z+1)*1.00)*Cell_Width[2];


                //int ccounter = 0;

		
                int ac = 0;
                for(int zz = (z - dz); zz <= (z + dz); zz++)
                {
                    for(int yy = (y - dy); yy <= (y + dy); yy++, ac++)
                    {
                        //if ((current_cell_index == 20) && (yy == 0) && (zz == 0))
                        //{
                            //printf("X\n");
                        //}
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
                            
                            //if (neighbor_cell_index0 > apex) ERROR_PRINT();
                            //if (neighbor_cell_index1 > apex) ERROR_PRINT();

                        #if 0
                            if (node_id == 0)
                            {
                                for(int p = neighbor_cell_index0; p<= neighbor_cell_index1; p++) 
                                {
                                    if (p >= current_cell_index) continue;
                                    printf("%d ", p);
                                }
                            }
                        #endif
                        }
                        else if (ranges_found == 2)
                        {			
                            //int apex = base_cell_index + x;
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


                        #if 0
                            if (node_id == 0)
                            {
                                for(int p = neighbor_cell_index0; p<= neighbor_cell_index1; p++) 
                                {
                                    if (p >= current_cell_index) continue;
                                    printf("%d ", p);
                                }
                                for(int p = neighbor_cell_index2; p<= neighbor_cell_index3; p++) 
                                {
                                    if (p >= current_cell_index) continue;
                                    printf("%d ", p);
                                }
                            }
                        #endif
                        }
                        
                        //printf("\n--- current_cell_index = %d ::: yy = %d ::: zz = %d \n", current_cell_index, yy, zz);

#if 0
                        for(int xx = (x - dx); xx <= (x + dx); xx++)
                        {
                            if (!global_Template_during_initialization[ccounter++]) continue;
                            //Our neighbor is the (xx, yy, zz) cell...
                            //if ((xx == x) && (yy == y) && (zz == z)) continue;

                            ////////////////////////////////////////////////////////////////////////////////////////
                            //Step A: Figure out if the nearest points between the grids is >= rmax...
                            ////////////////////////////////////////////////////////////////////////////////////////
                            
                            //Range1_X[0] = (xx*1.00)*Cell_Width[0]; Range1_X[1] = ((xx+1)*1.00)*Cell_Width[0];
                            //Range1_Y[0] = (yy*1.00)*Cell_Width[1]; Range1_Y[1] = ((yy+1)*1.00)*Cell_Width[1];
                            //Range1_Z[0] = (zz*1.00)*Cell_Width[2]; Range1_Z[1] = ((zz+1)*1.00)*Cell_Width[2];

                            //TYPE min_dist_2 = Find_Minimum_Distance_Between_Cells(Range0_X, Range0_Y, Range0_Z, Range1_X, Range1_Y, Range1_Z);

                            //if (min_dist_2 > rmax_2) continue;

                            ////////////////////////////////////////////////////////////////////////////////////////
                            //Step B: Collect Particles into consecutive positions and also respect PREIODICITY...
                            ////////////////////////////////////////////////////////////////////////////////////////

                            int xx_prime = xx; //, yy_prime = yy, zz_prime = zz;
                            if (xx < 0) xx_prime = xx + dimx; else if (xx >= (dimx)) xx_prime = xx - dimx;
                            //if (yy < 0) yy_prime = yy + dimy; else if (yy >= (dimy)) yy_prime = yy - dimy;
                            //if (zz < 0) zz_prime = zz + dimz; else if (zz >= (dimz)) zz_prime = zz - dimz;

                            //if ( (xx_prime < 0) || (xx_prime >= dimx))  ERROR_PRINT();
                            //if ( (yy_prime < 0) || (yy_prime >= dimy))  ERROR_PRINT();
                            //if ( (zz_prime < 0) || (zz_prime >= dimz))  ERROR_PRINT();

                            int neighbor_cell_index = base_cell_index + xx_prime; //GET_CELL_INDEX(xx_prime, yy_prime, zz_prime);
                            if (neighbor_cell_index > current_cell_index) continue;

                            Required[neighbor_cell_index] = 1;
                        }
#endif
                    }
                }
            }
        }

        MY_BARRIER(threadid); 
unsigned long long int e1time = ___rdtsc();
if ((node_id_CPU == 0) && (threadid == 0))printf("(%d) (%s) ::: Time Taken = %lld (%.2lf seconds)\n", __LINE__, __FILE__, (e1time - stime), (e1time - stime)/CORE_FREQUENCY_CPU);

        int number_of_subdivisions_per_thread = (number_of_subdivisions + nthreads_CPU - 1)/nthreads_CPU;
        int start_index = (threadid + 0) * number_of_subdivisions_per_thread; if (start_index > number_of_subdivisions) start_index = number_of_subdivisions;
        int   end_index = (threadid + 1) * number_of_subdivisions_per_thread; if (end_index > number_of_subdivisions) end_index = number_of_subdivisions;

        for(int k = 0; k<nthreads_CPU; k++)
        {
            for(int j=start_index; j<end_index; j++)
            {
                if (global_Required_during_initialization[k][j] > 1) ERROR_PRINT();
                if (global_Required_during_initialization[k][j] == 1) global_Required_R_CPU[j] = 1;
            }
        }

        MY_BARRIER(threadid); 
unsigned long long int e2time = ___rdtsc();
if ((node_id_CPU == 0) && (threadid == 0)) printf("(%d) (%s) ::: Time Taken = %lld (%.2lf seconds)\n", __LINE__, __FILE__, (e2time - stime), (e2time - stime)/CORE_FREQUENCY_CPU);


        return arg1;
}
#endif


void Compute_Required_R(int number_of_subdivisions, int dimx, int dimy, int dimz, int dimxy)
{
    mpi_printf("<<%d>> Inside Compute_Required_R\n", node_id_CPU);
    global_number_of_subdivisions_during_initialization = number_of_subdivisions;

    global_dimx_during_initialization = dimx;
    global_dimy_during_initialization = dimy;
    global_dimz_during_initialization = dimz;
    global_dimxy_during_initialization = dimxy;

    //XXX: global_Required_during_initialization is already malloced... Just need to reset it...
    //global_Required_during_initialization = (unsigned char **)malloc(nthreads * sizeof(unsigned char *));
    {
        for(int k = 0; k < nthreads_CPU; k++)
        {
            for(int t = 0; t < number_of_subdivisions; t++)
            {
                global_Required_during_initialization[k][t] = 0;
            }
        }
    }
            
unsigned long long int stime = ___rdtsc();
#if 0
    for(int i=1; i<nthreads; i++) pthread_create(&threads[i], NULL, Compute_Required_R_Parallel, (void *)(i));
    Compute_Required_R_Parallel(0);
    for(int i=1; i<nthreads; i++) pthread_join(threads[i], NULL);
#else
#pragma omp parallel for num_threads(nthreads_CPU)
    for(int i = 0; i < nthreads_CPU; i++)
    {
        Compute_Required_R_Parallel((void *)(i));
    }
#endif
unsigned long long int etime = ___rdtsc();
//if (node_id == 0) printf("(%d) (%s) ::: Time Taken = %lld (%.2lf seconds)\n", __LINE__, __FILE__, (etime - stime), (etime - stime)/CORE_FREQUENCY);
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
        TYPE i_rminL = global_rminL_CPU;
        TYPE i_Lbox  = global_Lbox_CPU;
        TYPE i_rmaxL = global_rmaxL_CPU;

        TYPE rmax = i_rmaxL / i_Lbox;
        TYPE rmin = i_rminL / i_Lbox;

        TYPE rmax_2 = rmax * rmax;

        TYPE *Cell_Width = global_grid_R_CPU.Cell_Width;

        TYPE Range0_X[2], Range0_Y[2], Range0_Z[2];
        TYPE Range1_X[2], Range1_Y[2], Range1_Z[2];

        int dx = int(ceil((i_rmaxL/i_Lbox)*dimx)) + 0;
        int dy = int(ceil((i_rmaxL/i_Lbox)*dimy)) + 0;
        int dz = int(ceil((i_rmaxL/i_Lbox)*dimz)) + 0;

//if (node_id == 0) printf("dx = %d ::: dy = %d ::: dz = %d\n", dx, dy, dz);
        int starting_cell_index = (node_id_CPU + 0) * subdivisions_per_node;
        int   ending_cell_index = (node_id_CPU + 1) * subdivisions_per_node;

        int cells = subdivisions_per_node;
        int cells_per_thread = (cells + nthreads_CPU - 1)/nthreads_CPU;

        //printf("cells_per_thread = %d\n", cells_per_thread);

        int starting_cell_index_threadid = starting_cell_index + cells_per_thread * (threadid + 0);
        int   ending_cell_index_threadid = starting_cell_index + cells_per_thread * (threadid + 1);

        if (starting_cell_index_threadid > ending_cell_index) starting_cell_index_threadid = ending_cell_index;
        if (  ending_cell_index_threadid > ending_cell_index)   ending_cell_index_threadid = ending_cell_index;

//if (node_id == 0) printf("start_index = %d ::: end_index = %d\n", starting_cell_index_threadid, ending_cell_index_threadid);
        //for(int current_cell_index = starting_cell_index; current_cell_index < ending_cell_index; current_cell_index++)
        for(int current_cell_index = starting_cell_index_threadid; current_cell_index < ending_cell_index_threadid; current_cell_index++)
        {
            //printf("current_cell_index = %d\n", current_cell_index);
            //if (global_Owner[current_cell_index] != node_id) continue;

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

                    for(int xx = (x - dx); xx <= (x + dx); xx++)
                    {
                        if (!global_Template_during_initialization[ccounter++]) continue;
                        //Our neighbor is the (xx, yy, zz) cell...
                        //if ((xx == x) && (yy == y) && (zz == z)) continue;

                        ////////////////////////////////////////////////////////////////////////////////////////
                        //Step A: Figure out if the nearest points between the grids is >= rmax...
                        ////////////////////////////////////////////////////////////////////////////////////////
                        
                        //Range1_X[0] = (xx*1.00)*Cell_Width[0]; Range1_X[1] = ((xx+1)*1.00)*Cell_Width[0];
                        //Range1_Y[0] = (yy*1.00)*Cell_Width[1]; Range1_Y[1] = ((yy+1)*1.00)*Cell_Width[1];
                        //Range1_Z[0] = (zz*1.00)*Cell_Width[2]; Range1_Z[1] = ((zz+1)*1.00)*Cell_Width[2];

                        //TYPE min_dist_2 = Find_Minimum_Distance_Between_Cells(Range0_X, Range0_Y, Range0_Z, Range1_X, Range1_Y, Range1_Z);

                        //if (min_dist_2 > rmax_2) continue;

                        ////////////////////////////////////////////////////////////////////////////////////////
                        //Step B: Collect Particles into consecutive positions and also respect PREIODICITY...
                        ////////////////////////////////////////////////////////////////////////////////////////

                        int xx_prime = xx; //, yy_prime = yy, zz_prime = zz;
                        if (xx < 0) xx_prime = xx + dimx; else if (xx >= (dimx)) xx_prime = xx - dimx;
                        //if (yy < 0) yy_prime = yy + dimy; else if (yy >= (dimy)) yy_prime = yy - dimy;
                        //if (zz < 0) zz_prime = zz + dimz; else if (zz >= (dimz)) zz_prime = zz - dimz;

                        //if ( (xx_prime < 0) || (xx_prime >= dimx))  ERROR_PRINT();
                        //if ( (yy_prime < 0) || (yy_prime >= dimy))  ERROR_PRINT();
                        //if ( (zz_prime < 0) || (zz_prime >= dimz))  ERROR_PRINT();

                        int neighbor_cell_index = base_cell_index + xx_prime; //GET_CELL_INDEX(xx_prime, yy_prime, zz_prime);
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
        if (node_id_CPU == 0) { printf("nnodes is not a multiple of 8... A few new lines of code being executed...\n"); fflush(stdout); }
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

#if 0
    for(int i=1; i<nthreads; i++) pthread_create(&threads[i], NULL, Compute_Weights_R_Parallel, (void *)(i));
    Compute_Weights_R_Parallel(0);
    for(int i=1; i<nthreads; i++) pthread_join(threads[i], NULL);
#else
#pragma omp parallel for num_threads(nthreads_CPU)
    for(int i = 0; i < nthreads_CPU; i++)
    {
        Compute_Weights_R_Parallel((void *)(i));
    }
#endif
}
 
void *Compute_Count_of_particles_to_send_Parallel(void *arg1)
{
    int threadid = (int)(size_t)(arg1);
    unsigned char *Local_Required_All_Nodes = global_Local_Required_All_Nodes_during_initialization;
    int *Count_of_particles_to_send = global_Count_of_particles_to_send_during_initialization;
    int number_of_subdivisions = global_number_of_subdivisions_during_initialization;
	int *Send_Count = global_Send_Count_during_initialization;

    int nnodes_per_thread = (nnodes_CPU + nthreads_CPU - 1)/nthreads_CPU;
    int start_node = (threadid + 0) * nnodes_per_thread;
    int   end_node = (threadid + 1) * nnodes_per_thread;

    if (start_node > nnodes_CPU) start_node = nnodes_CPU;
    if (  end_node > nnodes_CPU)   end_node = nnodes_CPU;

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

#if 0
    for(int i=1; i<nthreads; i++) pthread_create(&threads[i], NULL, Compute_Count_of_particles_to_send_Parallel, (void *)(i));
    Compute_Count_of_particles_to_send_Parallel(0);
    for(int i=1; i<nthreads; i++) pthread_join(threads[i], NULL);
#else
#pragma omp parallel for num_threads(nthreads_CPU)
    for(int i = 0; i < nthreads_CPU; i++)
    {
        Compute_Count_of_particles_to_send_Parallel((void *)(i));
    }
#endif

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

	int nnodes_per_thread = (nnodes_CPU + nthreads_CPU - 1)/nthreads_CPU;
	int start_node = (threadid + 0) * nnodes_per_thread; if (start_node > nnodes_CPU) start_node = nnodes_CPU;
	int   end_node = (threadid + 1) * nnodes_per_thread; if (  end_node > nnodes_CPU)   end_node = nnodes_CPU;

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


#if 0
    	for(int i=1; i<nthreads; i++) pthread_create(&threads[i], NULL, Prepare_Data_To_Send_Parallel, (void *)(i));
    	Prepare_Data_To_Send_Parallel(0);
    	for(int i=1; i<nthreads; i++) pthread_join(threads[i], NULL);
#else

#pragma omp parallel for num_threads(nthreads_CPU)
    for(int i = 0; i < nthreads_CPU; i++)
    {
        Prepare_Data_To_Send_Parallel((void *)(i));
    }
#endif
} 

void Populate_Grid(Grid *grid, int dimx, int dimy, int dimz)
{
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

    grid->Count_Per_Thread = (int **)my_malloc(nthreads_CPU * sizeof(int *));

    //global_grid.count_per_thread actually stores the number of particles each thread computes for each bin...
    int total_number_of_cells_prime =  grid->number_of_uniform_subdivisions;

    if (total_number_of_cells_prime % 64) total_number_of_cells_prime = (total_number_of_cells_prime/64 + 1) * 64;

    size_t sz = 0;
    sz += total_number_of_cells_prime * (nthreads_CPU) * sizeof(int);
    sz += (1+total_number_of_cells_prime) * (1) * sizeof(int);
    sz += (nthreads_CPU) * (1) * sizeof(int);
    sz += (nthreads_CPU) * (1) * sizeof(int);

    unsigned char *temp_memory = (unsigned char *)my_malloc(sz);
    unsigned char *temp2_memory = temp_memory;

    for(int i=0; i<nthreads_CPU; i++)
    {
        grid->Count_Per_Thread[i] = (int *)(temp2_memory); 
        temp2_memory += (total_number_of_cells_prime * sizeof(int));
    }

    //global_grid.count_per_cell actually stores the number of particles in each cell...
    grid->Count_Per_Cell = (int *)(temp2_memory); temp2_memory += ((1+total_number_of_cells_prime) * sizeof(int));


    grid->Start_Cell = (int *)(temp2_memory); temp2_memory += ((nthreads_CPU) * sizeof(int));
    grid->End_Cell = (int *)(temp2_memory); temp2_memory += ((nthreads_CPU) * sizeof(int));

    if ((temp2_memory - temp_memory) != (sz)) ERROR_PRINT();
}


void Distribute_R_Particles_Amongst_Nodes(void)
{
    unsigned long long int stime = ___rdtsc();
    Compute_Number_of_Ones();

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//STEP 1: ALLOCATE GRID...
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    int dimx, dimy, dimz;
    int dimxy;
    int number_of_subdivisions;

    {
        //Copy grid dimensions from grid.D...
        dimx = global_grid_D_CPU.dimx;
        dimy = global_grid_D_CPU.dimy;
        dimz = global_grid_D_CPU.dimz;

        if (dimx % 2) ERROR_PRINT();
        if (dimy % 2) ERROR_PRINT();
        if (dimz % 2) ERROR_PRINT();

        //dimx = dimy = dimz = 8;
        number_of_subdivisions = dimx * dimy * dimz;

        dimxy = dimx * dimy;

        if (node_id_CPU == 0) printf("dimx = %d ::: dimy = %d :: dimz = %d\n", dimx, dimy, dimz);

        // We are artificially increasing the number of subdivisions for ease of communication... Will reduce it later :)
        number_of_subdivisions = ((number_of_subdivisions + nnodes_CPU - 1)/nnodes_CPU) * nnodes_CPU;
    }


    int subdivisions_per_node = (number_of_subdivisions + nnodes_CPU - 1)/nnodes_CPU;
    if ( (subdivisions_per_node * nnodes_CPU) != number_of_subdivisions) ERROR_PRINT();

    mpi_printf("number_of_subdivisions = %d ::: subdivisions_per_node = %d\n", number_of_subdivisions, subdivisions_per_node);

    if (number_of_subdivisions % nnodes_CPU) ERROR_PRINT();


    int *Send_Count = global_Prealloced_Send_Count;
    int *Recv_Count = global_Prealloced_Recv_Count;
    int *Count_Per_Cell = global_Prealloced_Count_Per_Cell;


    ////////////////////////////////////////////////////
    //STEP 1: Copy MinMax from grid_D...
    ////////////////////////////////////////////////////
            
    global_grid_R_CPU.Min[0] = global_grid_D_CPU.Min[0];
    global_grid_R_CPU.Min[1] = global_grid_D_CPU.Min[1];
    global_grid_R_CPU.Min[2] = global_grid_D_CPU.Min[2];

    global_grid_R_CPU.Max[0] = global_grid_D_CPU.Max[0];
    global_grid_R_CPU.Max[1] = global_grid_D_CPU.Max[1];
    global_grid_R_CPU.Max[2] = global_grid_D_CPU.Max[2];


unsigned long long int e7time = ___rdtsc();
if (node_id_CPU == 0) printf("(%d) (%s) ::: Time Taken = %lld (%.2lf seconds)\n", __LINE__, __FILE__, (e7time - stime), (e7time - stime)/CORE_FREQUENCY_CPU);
    ////////////////////////////////////////////////////
    //STEP 2: Compute the Send_Count...
    ////////////////////////////////////////////////////

    {
        for(int k=0; k<=number_of_subdivisions; k++) Send_Count[k] = 0;
        for(int k=0; k<=number_of_subdivisions; k++) Recv_Count[k] = 0;

        TYPE *Local_Pos = (TYPE *)malloc(global_number_of_galaxies_on_node_R * sizeof(TYPE) * DIMENSIONS);

        int dimxx = dimx - 1;
        int dimyy = dimy - 1;
        int dimzz = dimz - 1;

        TYPE *Min = global_grid_R_CPU.Min;
        TYPE *Max = global_grid_R_CPU.Max;
        TYPE *Extent  = global_grid_R_CPU.Extent;

        //Step A... Populate Histogram...
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

unsigned long long int e8time = ___rdtsc();
if (node_id_CPU == 0) printf("(%d) (%s) ::: Time Taken = %lld (%.2lf seconds)\n", __LINE__, __FILE__, (e8time - stime), (e8time - stime)/CORE_FREQUENCY_CPU);
        //mpi_printf("jch -- %d (%d)\n", Send_Count[debug_cell_id], debug_cell_id);
        //Step B... Compute Prefix-Sum... It's like the cummulative sum...

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


        //Step C... Scatter the data...

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

        //Step D... Undo the prefix sums...

        //for(int i=0; i<global_number_of_galaxies_on_node*3; i++) global_Positions[i] = Local_Pos[i];
        for(int k=(number_of_subdivisions-1); k>=1; k--) Send_Count[k] = Send_Count[k] - Send_Count[k-1];

unsigned long long int e9time = ___rdtsc();
if (node_id_CPU == 0) printf("(%d) (%s) ::: Time Taken = %lld (%.2lf seconds)\n", __LINE__, __FILE__, (e9time - stime), (e9time - stime)/CORE_FREQUENCY_CPU);

#if 0
        for(int l=0; l<nnodes; l++)
        {
            int temp_start = (l+0)*subdivisions_per_node;
            int   temp_end = (l+1)*subdivisions_per_node;
            int yas = 0;
            for(int k=temp_start; k<temp_end; k++) yas += Send_Count[k];
            mpi_printf("<%d> ::: L = %d ::: yas = %d\n", node_id, l, yas);
        }
#endif

        //Step E... Gather the counts...
    #if 0
        for(int k=0; k<nnodes; k++)
        {
            MPI_Gather(Send_Count + k * subdivisions_per_node, subdivisions_per_node, MPI_INT, Recv_Count, subdivisions_per_node, MPI_INT, k, MPI_COMM_WORLD);
        }
    #else
        MPI_Alltoall(Send_Count, subdivisions_per_node, MPI_INT, Recv_Count, subdivisions_per_node, MPI_INT, MPI_COMM_WORLD);
        //MPI_AllGather(Send_Count + k * subdivisions_per_node, subdivisions_per_node, MPI_INT, Recv_Count, subdivisions_per_node, MPI_INT, k, MPI_COMM_WORLD);
    #endif

unsigned long long int e10time = ___rdtsc();
if (node_id_CPU == 0) printf("(%d) (%s) ::: Time Taken = %lld (%.2lf seconds)\n", __LINE__, __FILE__, (e10time - stime), (e10time - stime)/CORE_FREQUENCY_CPU);

#if 0
        {
            int *TTTT = (int *)malloc(number_of_subdivisions * sizeof(int));
            int another_sum = 0;
            for(int k=0; k<number_of_subdivisions; k++) another_sum += Recv_Count[k];
            TTTT[node_id] = another_sum;
            for(int i=0; i<nnodes; i++)
	        {
                MPI_Bcast(TTTT + i, 1, MPI_INT, i, MPI_COMM_WORLD);
            }

            mpi_printf("<><> node_id = %d :::another_sum = %d\n", node_id, another_sum);
            int yas = 0; for(int i=0; i<nnodes; i++) yas += TTTT[i]; if (yas != global_number_of_galaxies) ERROR_PRINT();
        }
#endif


        //Step F... Compute Count_Per_Cell...
        {
            for(int k=0; k<number_of_subdivisions; k++) Count_Per_Cell[k] = 0;
            int *Dst = Count_Per_Cell + node_id_CPU * subdivisions_per_node;
            for(int k=0; k<nnodes_CPU; k++) 
            {
                int *Src = Recv_Count + k*subdivisions_per_node;
                for(int l=0; l<subdivisions_per_node; l++)
                {
                    Dst[l] += Src[l];
                }
            }

#if 0
            for(int i=0; i<nnodes; i++)
	        { 
                MPI_Bcast(Count_Per_Cell + i*subdivisions_per_node, subdivisions_per_node, MPI_INT, i, MPI_COMM_WORLD);
            }
#else
            int *Local_Count_Per_Cell = (int *)malloc(subdivisions_per_node * sizeof(int));
            int offfset = node_id_CPU * subdivisions_per_node;
            for(int k=0; k<subdivisions_per_node; k++) Local_Count_Per_Cell[k] = Count_Per_Cell[offfset + k];
            MPI_Allgather(Local_Count_Per_Cell, subdivisions_per_node, MPI_INT, Count_Per_Cell, subdivisions_per_node, MPI_INT, MPI_COMM_WORLD);

#endif
		
	#if 0
            //mpi_printf("jchjchjch -- %d(%d)\n", debug_cell_id, Count_Per_Cell[debug_cell_id]);
            //At this point, each node has the Recv_Count...
            {
                int yas = 0; for(int k=0; k<number_of_subdivisions; k++) yas += Count_Per_Cell[k]; 
                if (yas != global_number_of_galaxies) 
                {
                    mpi_printf("node_id = %d ::: yas = %d ::: global_number_of_galaxies = %lld\n", node_id, yas, global_number_of_galaxies);
                    ERROR_PRINT();
                }
            }
         #endif
        }

unsigned long long int e1time = ___rdtsc();
if (node_id_CPU == 0) printf("(%d) (%s) ::: Time Taken = %lld (%.2lf seconds)\n", __LINE__, __FILE__, (e1time - stime), (e1time - stime)/CORE_FREQUENCY_CPU);
        //Step G: For each cell that I own, compute the list of subdivisions that I may need to access during the execution of the algorithm...

        long long int *Weights = (long long int *)malloc(number_of_subdivisions * sizeof(long long int));
        long long int *local_Weights = (long long int *)malloc(subdivisions_per_node * sizeof(long long int));

        //XXX: Already Computed in distribute_D.cpp ::: Compute_Template_During_Initialization(&global_grid_R, dimx, dimy, dimz, dimxy);
    
        Compute_Weights_R(Count_Per_Cell, Weights, subdivisions_per_node, dimx, dimy, dimz, dimxy);

        {
            int starting_cell_index = (node_id_CPU + 0) * subdivisions_per_node;
            int   ending_cell_index = (node_id_CPU + 1) * subdivisions_per_node;
            int cells = subdivisions_per_node;
            for(int k=0; k<cells; k++) local_Weights[k] = Weights[starting_cell_index + k];
        }

#if 0
        for(int i=0; i<nnodes; i++)
	    {
            MPI_Bcast(Weights + i*subdivisions_per_node, subdivisions_per_node, MPI_LONG_LONG_INT, i, MPI_COMM_WORLD);
        }
#else
        MPI_Allgather(local_Weights, subdivisions_per_node, MPI_LONG_LONG_INT, Weights, subdivisions_per_node, MPI_LONG_LONG_INT, MPI_COMM_WORLD);
#endif

unsigned long long int e2time = ___rdtsc();
if (node_id_CPU == 0) printf("(%d) (%s) ::: Time Taken = %lld (%.2lf seconds)\n", __LINE__, __FILE__, (e2time - stime), (e2time - stime)/CORE_FREQUENCY_CPU);
        //Now we have the weights of all the cells... Let's divide them equally amongst the nodes...


        //Step G... Compute Owner of each Cell...
        global_Owner_R = (int *)malloc(number_of_subdivisions * sizeof(int));
        for(int k=0; k<number_of_subdivisions; k++) global_Owner_R[k] = -1; //Initialized to -1, i.e. not owned by anyone :)

        {
            //printf("ABCD\n");
            long long int sum_so_far = 0;
            for(int k=0; k<number_of_subdivisions; k++) sum_so_far  += Weights[k];
            long long int total_weights = sum_so_far;
            long long int weights_per_node = (sum_so_far + nnodes_CPU - 1)/nnodes_CPU;
            mpi_printf("total_weights = %lld ::: weights_per_node = %lld\n", total_weights, weights_per_node);
            fflush(stdout);
            int alloted_so_far = 0;

            sum_so_far = 0;
            //Divide cells per node here :) At least in theory :)
            for(int k=0; k<nnodes_CPU; k++)
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
                        if (Weights[k] == 0) global_Owner_R[k] = nnodes_CPU-1;
                        else ERROR_PRINT();
                    }
                }

            }
        
            int *Temp1 = (int *)malloc(nnodes_CPU * sizeof(int)); for(int k=0; k<nnodes_CPU; k++) Temp1[k] = 0;
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
                for(int k=0; k<nnodes_CPU; k++) 
                {
                    final_sum += Temp1[k];
                    min = PCL_MIN(Temp1[k], min);
                    max = PCL_MAX(Temp1[k], max);
                    //mpi_printf("[%d]--%d ", k, Temp1[k]); 
                }
                //mpi_printf("\n");
                size_t avg = final_sum/nnodes_CPU;
                mpi_printf("node_id = %d ::: Avg. = %d ::: min = %d ::: max = %d ::: Ratio = %.2lf\n", node_id_CPU, (int)(avg), min, max, (max*1.0)/avg);

                if (final_sum != global_number_of_galaxies) ERROR_PRINT();
            }
        }

unsigned long long int e4time = ___rdtsc();
if (node_id_CPU == 0) printf("(%d) (%s) ::: Time Taken = %lld (%.2lf seconds)\n", __LINE__, __FILE__, (e4time - stime), (e4time - stime)/CORE_FREQUENCY_CPU);


        global_Required_R_CPU = (unsigned char *)malloc(number_of_subdivisions *sizeof(unsigned char));
        for(int k = 0; k < number_of_subdivisions; k++) global_Required_R_CPU[k] = 0;
        ///////  WE will now again look at neighbors of the alloted nodes, and figure out which nodes are required...

        //XXX: For R, we need to look at neighbors to figure out what all particles are required...
        Compute_Required_R(number_of_subdivisions, dimx, dimy, dimz, dimxy);
#if 0
        for(int k = 0; k < number_of_subdivisions; k++)
        {
            if (global_Owner_R[k] == node_id) global_Required_R[k] = 1;
        }
#endif

#if 1
        for(int k = 0; k < number_of_subdivisions; k++)
        {
            if ((global_Required_D_For_R_CPU[k] == 1) && (global_Required_R_CPU[k] == 0))
            {
                global_Required_R_CPU[k] = 1;
            }
        }
#endif
#if 0
#endif


unsigned long long int e3time = ___rdtsc();
if (node_id_CPU == 0) printf("(%d) (%s) ::: Time Taken = %lld (%.2lf seconds)\n", __LINE__, __FILE__, (e3time - stime), (e3time - stime)/CORE_FREQUENCY_CPU);

        {
            int ssum = 0; 
            int particle_sum = 0;
            
            for(int k=0; k<number_of_subdivisions; k++) 
            {
                ssum += global_Required_R_CPU[k];
                if (global_Required_R_CPU[k]) particle_sum += Count_Per_Cell[k];
            }

            mpi_printf("node_id = %d ::: ssum = %d (number_of_subdivisions = %d) ::: particle_sum = %d (global_number_of_galxies = %lld)\n", 
                node_id_CPU, ssum, number_of_subdivisions, particle_sum, global_number_of_galaxies);
        }


        //Step I: BroadCast this global_Required to eveyone -- so that they all start sending out the required particles...

        unsigned char *Local_Required_Bkp = (unsigned char *)malloc(number_of_subdivisions * sizeof(unsigned char));
        for(int k=0; k<number_of_subdivisions; k++) Local_Required_Bkp[k] = global_Required_R_CPU[k];

        Convert_From_Byte_To_Bit(global_Required_R_CPU, number_of_subdivisions);
        int number_of_subdivisions_by_eight = number_of_subdivisions/8;
        if ((number_of_subdivisions_by_eight * 8) != number_of_subdivisions) number_of_subdivisions_by_eight++;
        unsigned char *Local_Required_All_Nodes = (unsigned char *)malloc(nnodes_CPU * number_of_subdivisions_by_eight * sizeof(unsigned char));
        {

    #if 0
            unsigned char *Dst = Local_Required_All_Nodes + node_id * number_of_subdivisions;
            for(int k = 0; k < number_of_subdivisions; k++) Dst[k] = global_Required[k];
            for(int k=0; k<nnodes; k++)
            {
                MPI_Bcast(Local_Required_All_Nodes + k*number_of_subdivisions, number_of_subdivisions, MPI_CHAR, k, MPI_COMM_WORLD);
            }
    #else
            //MPI_Allgather(global_Required, number_of_subdivisions, MPI_CHAR, Local_Required_All_Nodes, number_of_subdivisions, MPI_CHAR, MPI_COMM_WORLD);
            MPI_Allgather(global_Required_R_CPU, number_of_subdivisions_by_eight, MPI_UNSIGNED_CHAR, 
                          Local_Required_All_Nodes, number_of_subdivisions_by_eight, MPI_UNSIGNED_CHAR, MPI_COMM_WORLD);
        
    #endif
        }
unsigned long long int e12time = ___rdtsc();
if (node_id_CPU == 0) printf("(%d) (%s) ::: Time Taken = %lld (%.2lf seconds)\n", __LINE__, __FILE__, (e12time - stime), (e12time - stime)/CORE_FREQUENCY_CPU);

        for(int p=0; p<nnodes_CPU; p++) 
        {
            //mpi_printf("Required of cell_id (%d) == %d\n", debug_cell_id, Local_Required_All_Nodes[p*number_of_subdivisions + debug_cell_id]);
        }

        //Step J... Compute Prefix-Sum... It's like the cummulative sum...

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


        //Step K:
        {
            // I am node_id and I need to send data to each node, and also figure out how much will I get from every node...
            int *Count_of_particles_to_send = (int *)malloc(nnodes_CPU * sizeof(int));
            int *Count_of_particles_to_recv = (int *)malloc(nnodes_CPU * sizeof(int));

            Compute_Count_of_particles_to_send(Local_Required_All_Nodes, Count_of_particles_to_send, Send_Count, number_of_subdivisions);
      #if 0      
            for(int k=0; k<nnodes; k++)
            {
                unsigned char *Src = Local_Required_All_Nodes + k*number_of_subdivisions;
                int sum = 0;

                for(int m=0; m<number_of_subdivisions; m++)
                {
                    if (Src[m])
                    {
                        sum += Send_Count[m+1] - Send_Count[m];
                        //if (m == debug_cell_id) mpi_printf(" (%d) ::: XYZ -- %d\n", node_id, Send_Count[m+1] - Send_Count[m]);
                    }
                }

                Count_of_particles_to_send[k] =  sum;
             }

            for(int k=0; k<nnodes; k++)
            {
                //mpi_printf("<%d> ::: Count_of_particles_to_send[%d] = %d\n", node_id, k , Count_of_particles_to_send[k]);
            }

        #endif
        #if 0
             for(int k=0; k<nnodes; k++)
             {
                    MPI_Gather(Count_of_particles_to_send + k, 1, MPI_INT, Count_of_particles_to_recv, 1, MPI_INT, k, MPI_COMM_WORLD);
             }
        #else
            MPI_Alltoall(Count_of_particles_to_send, 1, MPI_INT, Count_of_particles_to_recv, 1, MPI_INT, MPI_COMM_WORLD);
        #endif
unsigned long long int e11time = ___rdtsc();
if (node_id_CPU == 0) printf("(%d) (%s) ::: Time Taken = %lld (%.2lf seconds)\n", __LINE__, __FILE__, (e11time - stime), (e11time - stime)/CORE_FREQUENCY_CPU);

             int total_particles_to_send;
             int total_particles_to_recv;
             {
                int sum0 = 0, sum1 = 0;
                for(int k=0; k<nnodes_CPU; k++) sum0 += Count_of_particles_to_send[k];
                for(int k=0; k<nnodes_CPU; k++) sum1 += Count_of_particles_to_recv[k];
                mpi_printf("node_id = %d ::: sum0 = %d ::: sum1 = %d\n", node_id_CPU, sum0, sum1);
                total_particles_to_send = sum0;
                total_particles_to_recv = sum1;
             }

             int *Prefix_Sum_Count_of_particles_to_send = (int *)malloc((nnodes_CPU + 1) * sizeof(int));
             int *Prefix_Sum_Count_of_particles_to_recv = (int *)malloc((nnodes_CPU + 1) * sizeof(int));
             {
                //Prefix Sum Count_of_particles_to_send...
                int sum = 0;
                for(int k=0; k<nnodes_CPU; k++)
                {
                    int value = Count_of_particles_to_send[k];
                    Prefix_Sum_Count_of_particles_to_send[k] = sum;
                    sum += value;
                }
                Prefix_Sum_Count_of_particles_to_send[nnodes_CPU] = sum;
                if (sum != total_particles_to_send) ERROR_PRINT();

                sum = 0;
                for(int k=0; k<nnodes_CPU; k++)
                {
                    int value = Count_of_particles_to_recv[k];
                    Prefix_Sum_Count_of_particles_to_recv[k] = sum;
                    sum += value;
                }
                Prefix_Sum_Count_of_particles_to_recv[nnodes_CPU] = sum;
                if (sum != total_particles_to_recv) ERROR_PRINT();
            }


            //Step L:
            TYPE *Data_To_Send = (TYPE *)malloc(total_particles_to_send * 3 * sizeof(TYPE));
            TYPE *Data_To_Recv = (TYPE *)malloc(total_particles_to_recv * 3 * sizeof(TYPE));

#if 0 // <<----
            for(int sender=0; sender < nnodes; sender++)
            {
                size_t recv_count_in_floats = Count_of_particles_to_recv[sender] * 3;

                TYPE *Addr = Data_To_Recv + Prefix_Sum_Count_of_particles_to_recv[sender] * 3;
                MPI_Irecv(Addr, recv_count_in_floats, MPI_FLOAT, sender, sender, MPI_COMM_WORLD, recv_request + sender);
            }

            //MPI_BARRIER(node_id);

            //Prepare the data to send, and send it...

#if 0


            for(int receiver  = 0; receiver < nnodes; receiver++)
            {
                TYPE *Src = Data_To_Send + Prefix_Sum_Count_of_particles_to_send[receiver] * 3;
                int offset = 0;
                char *Req = Local_Required_All_Nodes + receiver * number_of_subdivisions;

                for(int k = 0; k<number_of_subdivisions; k++)
                {
                    if (Req[k])
                    {
                    #if 0
                        if (k == debug_cell_id) mpi_printf("Alpha ::: %d --> %d ( %d ::: %d)\n", node_id, receiver, k, Send_Count[k+1]-Send_Count[k]);
                        if (k == debug_cell_id)
                        {
                            char filename[1024];
                            sprintf(filename, "NN_%d", node_id);
                            FILE *fp = fopen(filename, "w");
                            for(int p=Send_Count[k]; p < Send_Count[k+1]; p++)
                            {
                                fprintf(fp, "%f %f %f\n", Local_Pos[3*p + 0], Local_Pos[3*p + 1], Local_Pos[3*p + 2]);
                            }
                            fclose(fp);
                        }
                    #endif
                        for(int p=Send_Count[k]; p < Send_Count[k+1]; p++)
                        {
                            Src[3*offset + 0] = Local_Pos[3*p + 0];
                            Src[3*offset + 1] = Local_Pos[3*p + 1];
                            Src[3*offset + 2] = Local_Pos[3*p + 2];
                            offset++;
                        }
                     }
                 }

                 if (offset != (Prefix_Sum_Count_of_particles_to_send[receiver+1] - Prefix_Sum_Count_of_particles_to_send[receiver])) ERROR_PRINT();
                 size_t send_count_in_floats = offset* 3;
                 MPI_Isend(Src, send_count_in_floats, MPI_FLOAT, receiver, node_id, MPI_COMM_WORLD,  send_request_key + receiver);
             }
#else
		Prepare_Data_To_Send(Local_Pos, Data_To_Send, Prefix_Sum_Count_of_particles_to_send, Local_Required_All_Nodes, Send_Count, number_of_subdivisions);
unsigned long long int e6time = ___rdtsc();
if (node_id == 0) printf("(%d) (%s) ::: Time Taken = %lld (%.2lf seconds)\n", __LINE__, __FILE__, (e6time - stime), (e6time - stime)/CORE_FREQUENCY_CPU);

        for(int receiver  = 0; receiver < nnodes; receiver++)
		{
            TYPE *Src = Data_To_Send + Prefix_Sum_Count_of_particles_to_send[receiver] * 3;
            int offset = Prefix_Sum_Count_of_particles_to_send[receiver+1] - Prefix_Sum_Count_of_particles_to_send[receiver];
            size_t send_count_in_floats = offset* 3;
            MPI_Isend(Src, send_count_in_floats, MPI_FLOAT, receiver, node_id, MPI_COMM_WORLD,  send_request_key + receiver);
		}
#endif

             int key_messages_remaining = nnodes;

             while (key_messages_remaining > 0)
             {
                int msg_index;
                MPI_Waitany(nnodes, recv_request, &msg_index, recv_status);
                key_messages_remaining--;
             }

             MPI_Waitall(nnodes, send_request_key, MPI_STATUSES_IGNORE); 
#else

		Prepare_Data_To_Send(Local_Pos, Data_To_Send, Prefix_Sum_Count_of_particles_to_send, Local_Required_All_Nodes, Send_Count, number_of_subdivisions);
unsigned long long int e6time = ___rdtsc();
if (node_id_CPU == 0) printf("(%d) (%s) ::: Time Taken = %lld (%.2lf seconds)\n", __LINE__, __FILE__, (e6time - stime), (e6time - stime)/CORE_FREQUENCY_CPU);

    for(int k=0; k<nnodes_CPU; k++) Count_of_particles_to_send[k] = Count_of_particles_to_send[k] * 3;
    for(int k=0; k<nnodes_CPU; k++) Count_of_particles_to_recv[k] = Count_of_particles_to_recv[k] * 3;
    for(int k=0; k<=nnodes_CPU; k++) Prefix_Sum_Count_of_particles_to_send[k] = Prefix_Sum_Count_of_particles_to_send[k] * 3;
    for(int k=0; k<=nnodes_CPU; k++) Prefix_Sum_Count_of_particles_to_recv[k] = Prefix_Sum_Count_of_particles_to_recv[k] * 3;
    MPI_Alltoallv(Data_To_Send, Count_of_particles_to_send, Prefix_Sum_Count_of_particles_to_send, MPI_FLOAT, 
                  Data_To_Recv, Count_of_particles_to_recv, Prefix_Sum_Count_of_particles_to_recv, MPI_FLOAT, MPI_COMM_WORLD);

#endif


             mpi_printf("Fully Finished :: (%d)\n", node_id_CPU);

             global_Positions_R = Data_To_Recv;

             global_number_of_galaxies_on_node_R = 0;
             int actual_subdivisions = dimx * dimy * dimz;

             int kmin = actual_subdivisions;
             int kmax = -1;
             for(int k=0; k<actual_subdivisions; k++) 
             {
                if (global_Owner_R[k] == -1) ERROR_PRINT(); //XXX: Every node's data should have an owner :)
                if (global_Owner_R[k] == node_id_CPU) 
                {
                    if (kmin > k) kmin = k;
                    if (kmax < k) kmax = k;
                }
             }

             global_starting_cell_index_R_CPU = kmin;
             global_ending_cell_index_R_CPU = kmax + 1;

             global_number_of_galaxies_on_node_R = total_particles_to_recv;

             mpi_printf("++ %d ++ global_starting_cell_index_R = %d :::  global_ending_cell_index_R = %d ::: global_number_of_galaxies_on_node_R = %lld\n", 
             node_id_CPU, global_starting_cell_index_R_CPU, global_ending_cell_index_R_CPU, global_number_of_galaxies_on_node_R);


             for(int k=global_starting_cell_index_R_CPU; k<global_ending_cell_index_R_CPU; k++) if (global_Owner_R[k] != node_id_CPU) ERROR_PRINT();
            for(int k=0; k<number_of_subdivisions; k++) global_Required_R_CPU[k] = Local_Required_Bkp[k];
        }

        free(Local_Pos);
    }

unsigned long long int e5time = ___rdtsc();
if (node_id_CPU == 0) printf("(%d) (%s) ::: Time Taken = %lld (%.2lf seconds)\n", __LINE__, __FILE__, (e5time - stime), (e5time - stime)/CORE_FREQUENCY_CPU);

    PRINT_BLUE
    if (node_id_CPU == 0) mpi_printf("dimx = %d ::: dimy = %d ::: dimz = %d\n", dimx, dimy, dimz);
    PRINT_BLACK

    Populate_Grid(&global_grid_R_CPU, dimx, dimy, dimz);


    //MPI_BARRIER(node_id);

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//STEP 5: COMPUTE BOUNDING_BOX OF THE DATASET...
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#if 0
    Compute_Min_Max(global_Positions, global_number_of_galaxies, DIMENSIONS, global_grid.Min, global_grid.Max, global_grid.Extent, 
               global_grid.dimx, global_grid.dimy, global_grid.dimz, global_grid.Cell_Width);

#endif

    if (node_id_CPU == 0)
    {
    PRINT_LIGHT_RED
        mpi_printf("global_Min = [%e %e %e] ::: global_Max = [%e %e %e]\n", global_grid_R_CPU.Min[0], global_grid_R_CPU.Min[1], global_grid_R_CPU.Min[2], global_grid_R_CPU.Max[0], global_grid_R_CPU.Max[1], global_grid_R_CPU.Max[2]);
        mpi_printf("global_Extent = [%e %e %e]\n", global_grid_R_CPU.Extent[0], global_grid_R_CPU.Extent[1], global_grid_R_CPU.Extent[2]);
        mpi_printf("Total Memory Allocated = %lld Bytes (%.2lf GB)\n", global_memory_malloced, global_memory_malloced/1000.0/1000.0/1000.0);
    PRINT_BLACK
    }

    unsigned long long int etime = ___rdtsc();
    global_time_mpi += etime - stime;
}









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
#if 0
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
#endif

//if (node_id == 0) printf("dx = %d ::: dy = %d ::: dz = %d\n", dx, dy, dz);
        int starting_cell_index = (node_id_CPU + 0) * subdivisions_per_node;
        int   ending_cell_index = (node_id_CPU + 1) * subdivisions_per_node;

        int cells = subdivisions_per_node;
        int cells_per_thread = (cells + nthreads_CPU - 1)/nthreads_CPU;

        //printf("cells_per_thread = %d\n", cells_per_thread);

        int starting_cell_index_threadid = starting_cell_index + cells_per_thread * (threadid + 0);
        int   ending_cell_index_threadid = starting_cell_index + cells_per_thread * (threadid + 1);

        if (starting_cell_index_threadid > ending_cell_index) starting_cell_index_threadid = ending_cell_index;
        if (  ending_cell_index_threadid > ending_cell_index)   ending_cell_index_threadid = ending_cell_index;

//if (node_id == 0) printf("start_index = %d ::: end_index = %d\n", starting_cell_index_threadid, ending_cell_index_threadid);
        //for(int current_cell_index = starting_cell_index; current_cell_index < ending_cell_index; current_cell_index++)
        for(int current_cell_index = starting_cell_index_threadid; current_cell_index < ending_cell_index_threadid; current_cell_index++)
        {
            //printf("current_cell_index = %d\n", current_cell_index);
            //if (global_Owner[current_cell_index] != node_id) continue;

            long long int local_weight = 0;
            int objects_in_this_cell = Count_Per_Cell[current_cell_index];

#if 0
            int x, y, z; z = current_cell_index / dimxy; int xy = current_cell_index % dimxy; y = xy / dimx; x = xy % dimx;
    
            Range0_X[0] = (x*1.00)*Cell_Width[0]; Range0_X[1] = ((x+1)*1.00)*Cell_Width[0];
            Range0_Y[0] = (y*1.00)*Cell_Width[1]; Range0_Y[1] = ((y+1)*1.00)*Cell_Width[1];
            Range0_Z[0] = (z*1.00)*Cell_Width[2]; Range0_Z[1] = ((z+1)*1.00)*Cell_Width[2];
#endif

            //local_weight += (objects_in_this_cell * (objects_in_this_cell-1))/2;
            local_weight += objects_in_this_cell;

#if 0
            int ccounter = 0;

            for(int zz = (z - dz); zz <= (z + dz); zz++)
            {
                for(int yy = (y - dy); yy <= (y + dy); yy++)
                {
                    for(int xx = (x - dx); xx <= (x + dx); xx++)
                    {
                        if (!global_Template_during_initialization[ccounter++]) continue;
                        //Our neighbor is the (xx, yy, zz) cell...
                        //if ((xx == x) && (yy == y) && (zz == z)) continue;

                        ////////////////////////////////////////////////////////////////////////////////////////
                        //Step A: Figure out if the nearest points between the grids is >= rmax...
                        ////////////////////////////////////////////////////////////////////////////////////////
                        
                        //Range1_X[0] = (xx*1.00)*Cell_Width[0]; Range1_X[1] = ((xx+1)*1.00)*Cell_Width[0];
                        //Range1_Y[0] = (yy*1.00)*Cell_Width[1]; Range1_Y[1] = ((yy+1)*1.00)*Cell_Width[1];
                        //Range1_Z[0] = (zz*1.00)*Cell_Width[2]; Range1_Z[1] = ((zz+1)*1.00)*Cell_Width[2];

                        //TYPE min_dist_2 = Find_Minimum_Distance_Between_Cells(Range0_X, Range0_Y, Range0_Z, Range1_X, Range1_Y, Range1_Z);

                        //if (min_dist_2 > rmax_2) continue;

                        ////////////////////////////////////////////////////////////////////////////////////////
                        //Step B: Collect Particles into consecutive positions and also respect PREIODICITY...
                        ////////////////////////////////////////////////////////////////////////////////////////

                        int xx_prime = xx, yy_prime = yy, zz_prime = zz;
                        if (xx < 0) xx_prime = xx + dimx; else if (xx >= (dimx)) xx_prime = xx - dimx;
                        if (yy < 0) yy_prime = yy + dimy; else if (yy >= (dimy)) yy_prime = yy - dimy;
                        if (zz < 0) zz_prime = zz + dimz; else if (zz >= (dimz)) zz_prime = zz - dimz;

                        //if ( (xx_prime < 0) || (xx_prime >= dimx))  ERROR_PRINT();
                        //if ( (yy_prime < 0) || (yy_prime >= dimy))  ERROR_PRINT();
                        //if ( (zz_prime < 0) || (zz_prime >= dimz))  ERROR_PRINT();

                        int neighbor_cell_index = GET_CELL_INDEX(xx_prime, yy_prime, zz_prime);
                        if (neighbor_cell_index > current_cell_index) continue;

                        int objects_in_neighboring_cell = Count_Per_Cell[neighbor_cell_index];
                        //local_weight += objects_in_this_cell * objects_in_neighboring_cell;
                        local_weight += objects_in_this_cell;
                    }
                }
            }
#endif

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

#if 0
    for(int i=1; i<nthreads; i++) pthread_create(&threads[i], NULL, Compute_Weights_D_Parallel, (void *)(i));
    Compute_Weights_D_Parallel(0);
    for(int i=1; i<nthreads; i++) pthread_join(threads[i], NULL);
#else
#pragma omp parallel for num_threads(nthreads_CPU)
    for(int i=0; i<nthreads_CPU; i++) 
    {
        Compute_Weights_D_Parallel((void *)(i));
    }
#endif
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
 
unsigned long long int stime = ___rdtsc();
    //global_Owner is read-only, and basically owners have already been decided...
        if (1)
        {
            TYPE i_rminL = global_rminL_CPU;
            TYPE i_Lbox  = global_Lbox_CPU;
            TYPE i_rmaxL = global_rmaxL_CPU;

            TYPE rmax = i_rmaxL / i_Lbox;
            TYPE rmin = i_rminL / i_Lbox;

            TYPE rmax_2 = rmax * rmax;

            TYPE *Cell_Width = global_grid_D_CPU.Cell_Width;

            TYPE Range0_X[2], Range0_Y[2], Range0_Z[2];
            TYPE Range1_X[2], Range1_Y[2], Range1_Z[2];

            int dx = int(ceil((i_rmaxL/i_Lbox)*dimx)) + 0;
            int dy = int(ceil((i_rmaxL/i_Lbox)*dimy)) + 0;
            int dz = int(ceil((i_rmaxL/i_Lbox)*dimz)) + 0;

            int min_cell_id = (1<<29);
            int max_cell_id = -1;

            for(int current_cell_index = 0; current_cell_index < number_of_subdivisions; current_cell_index ++)
            {
                if (global_Owner_D[current_cell_index] != node_id_CPU) continue;
                min_cell_id = current_cell_index;
                break;
            }

            //for(int current_cell_index = number_of_subdivisions-1; current_cell_index >= 0; current_cell_index --)
            for(int current_cell_index = min_cell_id; current_cell_index < number_of_subdivisions; current_cell_index++)
            {
                if (global_Owner_D[current_cell_index] != node_id_CPU) 
                {
                    max_cell_id = current_cell_index - 1;
                    break;
                }
            }

            if (min_cell_id == (1<<29)) ERROR_PRINT();
            if (max_cell_id == -1)
            {
                if (node_id_CPU != (nnodes_CPU-1)) ERROR_PRINT();
                max_cell_id = number_of_subdivisions - 1;
            }
            

            //[min_cell_id... max_cell_id] are both inclusive :)
            int cells = max_cell_id - min_cell_id + 1;
            int cells_per_thread = (cells + nthreads_CPU - 1)/nthreads_CPU;

            int start_index = (threadid + 0) * cells_per_thread; if (start_index > cells) start_index = cells;
            int   end_index = (threadid + 1) * cells_per_thread; if (  end_index > cells)   end_index = cells;

            start_index += min_cell_id;
            end_index += min_cell_id;

unsigned long long int e2time = ___rdtsc();
//if (node_id == 0) printf("(%d) (%s) ::: Time Taken = %lld (%.2lf seconds)\n", __LINE__, __FILE__, (e2time - stime), (e2time - stime)/CORE_FREQUENCY);
//if (node_id == 0) printf("dx = %d ::: dy = %d ::: dz = %d\n", dx, dy, dz);
//if (node_id == 0) printf("start_index = %d ::: end_index = %d\n", start_index, end_index);
//if (node_id == 0) printf("min_cell_id = %d ::: max_cell_id = %d\n", min_cell_id, max_cell_id);
            //for(int current_cell_index = min_cell_id; current_cell_index <= max_cell_id; current_cell_index ++)
            for(int current_cell_index = start_index; current_cell_index < end_index; current_cell_index++)
            {
                if (global_Owner_D[current_cell_index] != node_id_CPU) ERROR_PRINT();

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
                            //Our neighbor is the (xx, yy, zz) cell...
                            //if ((xx == x) && (yy == y) && (zz == z)) continue;

                            ////////////////////////////////////////////////////////////////////////////////////////
                            //Step A: Figure out if the nearest points between the grids is >= rmax...
                            ////////////////////////////////////////////////////////////////////////////////////////
                            
                            //Range1_X[0] = (xx*1.00)*Cell_Width[0]; Range1_X[1] = ((xx+1)*1.00)*Cell_Width[0];
                            //Range1_Y[0] = (yy*1.00)*Cell_Width[1]; Range1_Y[1] = ((yy+1)*1.00)*Cell_Width[1];
                            //Range1_Z[0] = (zz*1.00)*Cell_Width[2]; Range1_Z[1] = ((zz+1)*1.00)*Cell_Width[2];

                            //TYPE min_dist_2 = Find_Minimum_Distance_Between_Cells(Range0_X, Range0_Y, Range0_Z, Range1_X, Range1_Y, Range1_Z);

                            //if (min_dist_2 > rmax_2) continue;

                            ////////////////////////////////////////////////////////////////////////////////////////
                            //Step B: Collect Particles into consecutive positions and also respect PREIODICITY...
                            ////////////////////////////////////////////////////////////////////////////////////////

                            int xx_prime = xx, yy_prime = yy, zz_prime = zz;
                            if (xx < 0) xx_prime = xx + dimx; else if (xx >= (dimx)) xx_prime = xx - dimx;
                            if (yy < 0) yy_prime = yy + dimy; else if (yy >= (dimy)) yy_prime = yy - dimy;
                            if (zz < 0) zz_prime = zz + dimz; else if (zz >= (dimz)) zz_prime = zz - dimz;

                            //if ( (xx_prime < 0) || (xx_prime >= dimx))  ERROR_PRINT();
                            //if ( (yy_prime < 0) || (yy_prime >= dimy))  ERROR_PRINT();
                            //if ( (zz_prime < 0) || (zz_prime >= dimz))  ERROR_PRINT();

                            int neighbor_cell_index = GET_CELL_INDEX(xx_prime, yy_prime, zz_prime);
                            //if (neighbor_cell_index > current_cell_index) continue;

                            Required[neighbor_cell_index] = 1;
                        }
                    }
                }
            }
        }

        MY_BARRIER(threadid); 
unsigned long long int e1time = ___rdtsc();
//if (node_id == 0) printf("(%d) (%s) ::: Time Taken = %lld (%.2lf seconds)\n", __LINE__, __FILE__, (e1time - stime), (e1time - stime)/CORE_FREQUENCY);

        int number_of_subdivisions_per_thread = (number_of_subdivisions + nthreads_CPU - 1)/nthreads_CPU;
        int start_index = (threadid + 0) * number_of_subdivisions_per_thread; if (start_index > number_of_subdivisions) start_index = number_of_subdivisions;
        int   end_index = (threadid + 1) * number_of_subdivisions_per_thread; if (end_index > number_of_subdivisions) end_index = number_of_subdivisions;

        for(int k = 0; k<nthreads_CPU; k++)
        {
            for(int j=start_index; j<end_index; j++)
            {
                if (global_Required_during_initialization[k][j] > 1) ERROR_PRINT();
                if (global_Required_during_initialization[k][j] == 1) global_Required_D_For_R_CPU[j] = 1;
            }
        }

        MY_BARRIER(threadid); 
unsigned long long int e2time = ___rdtsc();
//if (node_id == 0) printf("(%d) (%s) ::: Time Taken = %lld (%.2lf seconds)\n", __LINE__, __FILE__, (e2time - stime), (e2time - stime)/CORE_FREQUENCY);


        return arg1;
}


void Compute_Required_D_For_R(int number_of_subdivisions, int dimx, int dimy, int dimz, int dimxy)
{
    mpi_printf("<<%d>> Inside Compute_Required_D_For_R\n", node_id_CPU);
    global_number_of_subdivisions_during_initialization = number_of_subdivisions;

    global_dimx_during_initialization = dimx;
    global_dimy_during_initialization = dimy;
    global_dimz_during_initialization = dimz;
    global_dimxy_during_initialization = dimxy;

    global_Required_during_initialization = (unsigned char **)malloc(nthreads_CPU * sizeof(unsigned char *));
    {
        unsigned char *TT1 = (unsigned char *)malloc(nthreads_CPU * number_of_subdivisions * sizeof(unsigned char));
        for(int k=0; k<(nthreads_CPU * number_of_subdivisions); k++) TT1[k] = 0;
        unsigned char *TT2 = TT1;
        for(int k=0; k<nthreads_CPU; k++)
        {
            global_Required_during_initialization[k] = TT2;
            TT2 += number_of_subdivisions;
        }

        if ( (TT2-TT1) != (nthreads_CPU * number_of_subdivisions)) ERROR_PRINT();
    }
            
unsigned long long int stime = ___rdtsc();
#if 0
    for(int i=1; i<nthreads; i++) pthread_create(&threads[i], NULL, Compute_Required_D_For_R_Parallel, (void *)(i));
    Compute_Required_D_For_R_Parallel(0);
    for(int i=1; i<nthreads; i++) pthread_join(threads[i], NULL);
#else
#pragma omp parallel for num_threads(nthreads_CPU)
    for(int i = 0; i < nthreads_CPU; i++)
    {
        Compute_Required_D_For_R_Parallel((void *)(i));
    }
#endif
unsigned long long int etime = ___rdtsc();
//if (node_id == 0) printf("(%d) (%s) ::: Time Taken = %lld (%.2lf seconds)\n", __LINE__, __FILE__, (etime - stime), (etime - stime)/CORE_FREQUENCY);
}

void Compute_Template_During_Initialization(Grid *grid, int dimx, int dimy, int dimz, int dimxy)
{

    TYPE i_rminL = global_rminL_CPU;
    TYPE i_Lbox  = global_Lbox_CPU;
    TYPE i_rmaxL = global_rmaxL_CPU;

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
                //Our neighbor is the (xx, yy, zz) cell...
                if ((xx == x) && (yy == y) && (zz == z)) 
                {
                    global_Template_during_initialization[ccounter] = 0;
                    continue;
                }

                ////////////////////////////////////////////////////////////////////////////////////////
                //Step A: Figure out if the nearest points between the grids is >= rmax...
                ////////////////////////////////////////////////////////////////////////////////////////
                            
               
                Range1_X[0] = (xx*1.00)*Cell_Width[0]; Range1_X[1] = ((xx+1)*1.00)*Cell_Width[0];
                Range1_Y[0] = (yy*1.00)*Cell_Width[1]; Range1_Y[1] = ((yy+1)*1.00)*Cell_Width[1];
                Range1_Z[0] = (zz*1.00)*Cell_Width[2]; Range1_Z[1] = ((zz+1)*1.00)*Cell_Width[2];

                TYPE min_dist_2 = CPUFunction_Find_Minimum_Distance_Between_Cells(Range0_X, Range0_Y, Range0_Z, Range1_X, Range1_Y, Range1_Z);

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
            
                //[left_one .. right_one]
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
                    //ERROR_PRINT();
               }
               else
               { 
                   if (left_one >= right_one) ERROR_PRINT();
                   if (left_one >= 0) ERROR_PRINT();
                   if (right_one <= 0) ERROR_PRINT();
               } 

                
               global_Template_Range_during_initialization[2*ac + 0] =  left_one;
               global_Template_Range_during_initialization[2*ac + 1] =  right_one;
               //printf("ac = %d ::: left_one = %d ::: right_one = %d\n", ac, left_one, right_one);
               ac++;
            }
        }
    }
}

void Distribute_D_Particles_Amongst_Nodes(void)
{
    unsigned long long int stime = ___rdtsc();
    Compute_Number_of_Ones();

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//STEP 1: ALLOCATE GRID...
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    int dimx, dimy, dimz;
    int dimxy;
    int number_of_subdivisions;

    {
        int multiple = PCL_MAX(10 * nthreads_CPU, 100);  
        multiple = nthreads_CPU * ((multiple + nthreads_CPU - 1)/nthreads_CPU);

        int rough_estimate_for_total_number_of_cells = nnodes_CPU * multiple;

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

        //dimx = dimy = dimz = 8;
        number_of_subdivisions = dimx * dimy * dimz;

        dimxy = dimx * dimy;

        if (node_id_CPU == 0) printf("dimx = %d ::: dimy = %d :: dimz = %d\n", dimx, dimy, dimz);

        // We are artificially increasing the number of subdivisions for ease of communication... Will reduce it later :)
        number_of_subdivisions = ((number_of_subdivisions + nnodes_CPU - 1)/nnodes_CPU) * nnodes_CPU;
    }


    int subdivisions_per_node = (number_of_subdivisions + nnodes_CPU - 1)/nnodes_CPU;
    if ( (subdivisions_per_node * nnodes_CPU) != number_of_subdivisions) ERROR_PRINT();

    mpi_printf("number_of_subdivisions = %d ::: subdivisions_per_node = %d\n", number_of_subdivisions, subdivisions_per_node);

    if (number_of_subdivisions % nnodes_CPU) ERROR_PRINT();


    int *Send_Count = (int *)my_malloc((1 + number_of_subdivisions) * sizeof(int));
    int *Recv_Count = (int *)my_malloc((1 + number_of_subdivisions) * sizeof(int));
    int *Count_Per_Cell = (int *)my_malloc((1 + number_of_subdivisions) * sizeof(int));

    global_Prealloced_Send_Count = Send_Count;
    global_Prealloced_Recv_Count = Recv_Count;
    global_Prealloced_Count_Per_Cell = Count_Per_Cell;

    TYPE *MinMax = (TYPE *)my_malloc(nnodes_CPU * 6 * sizeof(TYPE));

    TYPE Local_Min[3], Local_Max[3];
    TYPE Local_MinMax[6];

    Just_Compute_Min_Max(global_Positions_D, global_number_of_galaxies_on_node_D, DIMENSIONS, Local_Min, Local_Max);

    mpi_printf("LMin = [%e %e %e] ::: [%e %e %e]\n", Local_Min[0], Local_Min[1], Local_Min[2], Local_Max[0], Local_Max[1], Local_Max[2]);



    ////////////////////////////////////////////////////
    //STEP 1: Compute and Broadcast Min/Max... /////////
    ////////////////////////////////////////////////////
            
#if 0
    MinMax[6*node_id + 0] = Local_Min[0];
    MinMax[6*node_id + 1] = Local_Min[1];
    MinMax[6*node_id + 2] = Local_Min[2];
    MinMax[6*node_id + 3] = Local_Max[0];
    MinMax[6*node_id + 4] = Local_Max[1];
    MinMax[6*node_id + 5] = Local_Max[2];
    for(int i=0; i<nnodes_CPU; i++)
	{
        MPI_Bcast(MinMax + 6*i, 6, MPI_FLOAT, i, MPI_COMM_WORLD);
    }
#else
    Local_MinMax[0] = Local_Min[0];
    Local_MinMax[1] = Local_Min[1];
    Local_MinMax[2] = Local_Min[2];
    Local_MinMax[3] = Local_Max[0];
    Local_MinMax[4] = Local_Max[1];
    Local_MinMax[5] = Local_Max[2];
    MPI_Allgather(Local_MinMax, 6, MPI_FLOAT, MinMax, 6, MPI_FLOAT, MPI_COMM_WORLD);
#endif

    for(int i=0; i<nnodes_CPU; i++)
    {
        Local_Min[0] = PCL_MIN(Local_Min[0], MinMax[6*i + 0]);
        Local_Min[1] = PCL_MIN(Local_Min[1], MinMax[6*i + 1]);
        Local_Min[2] = PCL_MIN(Local_Min[2], MinMax[6*i + 2]);

        Local_Max[0] = PCL_MAX(Local_Max[0], MinMax[6*i + 3]);
        Local_Max[1] = PCL_MAX(Local_Max[1], MinMax[6*i + 4]);
        Local_Max[2] = PCL_MAX(Local_Max[2], MinMax[6*i + 5]);
    }

    global_grid_D_CPU.Min[0] = Local_Min[0];
    global_grid_D_CPU.Min[1] = Local_Min[1];
    global_grid_D_CPU.Min[2] = Local_Min[2];

    global_grid_D_CPU.Max[0] = Local_Max[0];
    global_grid_D_CPU.Max[1] = Local_Max[1];
    global_grid_D_CPU.Max[2] = Local_Max[2];


    Just_Compute_Extents(DIMENSIONS, global_grid_D_CPU.Min, global_grid_D_CPU.Max, global_grid_D_CPU.Extent, global_grid_D_CPU.Cell_Width, dimx, dimy, dimz);

    Compute_Template_During_Initialization(&global_grid_D_CPU, dimx, dimy, dimz, dimxy);

unsigned long long int e7time = ___rdtsc();
if (node_id_CPU == 0) printf("(%d) (%s) ::: Time Taken = %lld (%.2lf seconds)\n", __LINE__, __FILE__, (e7time - stime), (e7time - stime)/CORE_FREQUENCY_CPU);
    ////////////////////////////////////////////////////
    //STEP 2: Compute the Send_Count...
    ////////////////////////////////////////////////////

    {
        for(int k=0; k<=number_of_subdivisions; k++) Send_Count[k] = 0;
        for(int k=0; k<=number_of_subdivisions; k++) Recv_Count[k] = 0;

        TYPE *Local_Pos = (TYPE *)malloc(global_number_of_galaxies_on_node_D * sizeof(TYPE) * DIMENSIONS);

        int dimxx = dimx - 1;
        int dimyy = dimy - 1;
        int dimzz = dimz - 1;

        TYPE *Min = global_grid_D_CPU.Min;
        TYPE *Max = global_grid_D_CPU.Max;
        TYPE *Extent  = global_grid_D_CPU.Extent;

        //Step A... Populate Histogram...
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

unsigned long long int e8time = ___rdtsc();
if (node_id_CPU == 0) printf("(%d) (%s) ::: Time Taken = %lld (%.2lf seconds)\n", __LINE__, __FILE__, (e8time - stime), (e8time - stime)/CORE_FREQUENCY_CPU);
        //mpi_printf("jch -- %d (%d)\n", Send_Count[debug_cell_id], debug_cell_id);
        //Step B... Compute Prefix-Sum... It's like the cummulative sum...

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


        //Step C... Scatter the data...

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

        //Step D... Undo the prefix sums...

        //for(int i=0; i<global_number_of_galaxies_on_node*3; i++) global_Positions[i] = Local_Pos[i];
        for(int k=(number_of_subdivisions-1); k>=1; k--) Send_Count[k] = Send_Count[k] - Send_Count[k-1];

unsigned long long int e9time = ___rdtsc();
if (node_id_CPU == 0) printf("(%d) (%s) ::: Time Taken = %lld (%.2lf seconds)\n", __LINE__, __FILE__, (e9time - stime), (e9time - stime)/CORE_FREQUENCY_CPU);

#if 0
        for(int l=0; l<nnodes_CPU; l++)
        {
            int temp_start = (l+0)*subdivisions_per_node;
            int   temp_end = (l+1)*subdivisions_per_node;
            int yas = 0;
            for(int k=temp_start; k<temp_end; k++) yas += Send_Count[k];
            mpi_printf("<%d> ::: L = %d ::: yas = %d\n", node_id, l, yas);
        }
#endif

        //Step E... Gather the counts...
    #if 0
        for(int k=0; k<nnodes_CPU; k++)
        {
            MPI_Gather(Send_Count + k * subdivisions_per_node, subdivisions_per_node, MPI_INT, Recv_Count, subdivisions_per_node, MPI_INT, k, MPI_COMM_WORLD);
        }
    #else
        MPI_Alltoall(Send_Count, subdivisions_per_node, MPI_INT, Recv_Count, subdivisions_per_node, MPI_INT, MPI_COMM_WORLD);
        //MPI_AllGather(Send_Count + k * subdivisions_per_node, subdivisions_per_node, MPI_INT, Recv_Count, subdivisions_per_node, MPI_INT, k, MPI_COMM_WORLD);
    #endif

unsigned long long int e10time = ___rdtsc();
if (node_id_CPU == 0) printf("(%d) (%s) ::: Time Taken = %lld (%.2lf seconds)\n", __LINE__, __FILE__, (e10time - stime), (e10time - stime)/CORE_FREQUENCY_CPU);

#if 0
        {
            int *TTTT = (int *)malloc(number_of_subdivisions * sizeof(int));
            int another_sum = 0;
            for(int k=0; k<number_of_subdivisions; k++) another_sum += Recv_Count[k];
            TTTT[node_id] = another_sum;
            for(int i=0; i<nnodes_CPU; i++)
	        {
                MPI_Bcast(TTTT + i, 1, MPI_INT, i, MPI_COMM_WORLD);
            }

            mpi_printf("<><> node_id = %d :::another_sum = %d\n", node_id, another_sum);
            int yas = 0; for(int i=0; i<nnodes_CPU; i++) yas += TTTT[i]; if (yas != global_number_of_galaxies) ERROR_PRINT();
        }
#endif


        //Step F... Compute Count_Per_Cell...
        {
            for(int k=0; k<number_of_subdivisions; k++) Count_Per_Cell[k] = 0;
            int *Dst = Count_Per_Cell + node_id_CPU * subdivisions_per_node;
            for(int k=0; k<nnodes_CPU; k++) 
            {
                int *Src = Recv_Count + k*subdivisions_per_node;
                for(int l=0; l<subdivisions_per_node; l++)
                {
                    Dst[l] += Src[l];
                }
            }

#if 0
            for(int i=0; i<nnodes_CPU; i++)
	        { 
                MPI_Bcast(Count_Per_Cell + i*subdivisions_per_node, subdivisions_per_node, MPI_INT, i, MPI_COMM_WORLD);
            }
#else
            int *Local_Count_Per_Cell = (int *)malloc(subdivisions_per_node * sizeof(int));
            int offfset = node_id_CPU * subdivisions_per_node;
            for(int k=0; k<subdivisions_per_node; k++) Local_Count_Per_Cell[k] = Count_Per_Cell[offfset + k];
            MPI_Allgather(Local_Count_Per_Cell, subdivisions_per_node, MPI_INT, Count_Per_Cell, subdivisions_per_node, MPI_INT, MPI_COMM_WORLD);

#endif
		
	#if 0
            //mpi_printf("jchjchjch -- %d(%d)\n", debug_cell_id, Count_Per_Cell[debug_cell_id]);
            //At this point, each node has the Recv_Count...
            {
                int yas = 0; for(int k=0; k<number_of_subdivisions; k++) yas += Count_Per_Cell[k]; 
                if (yas != global_number_of_galaxies) 
                {
                    mpi_printf("node_id_CPU = %d ::: yas = %d ::: global_number_of_galaxies = %lld\n", node_id_CPU, yas, global_number_of_galaxies);
                    ERROR_PRINT();
                }
            }
         #endif
        }

unsigned long long int e1time = ___rdtsc();
if (node_id_CPU == 0) printf("(%d) (%s) ::: Time Taken = %lld (%.2lf seconds)\n", __LINE__, __FILE__, (e1time - stime), (e1time - stime)/CORE_FREQUENCY_CPU);
        //Step G: For each cell that I own, compute the list of subdivisions that I may need to access during the execution of the algorithm...

        long long int *Weights = (long long int *)malloc(number_of_subdivisions * sizeof(long long int));
        long long int *local_Weights = (long long int *)malloc(subdivisions_per_node * sizeof(long long int));

        //J`:s` //`u``Compute_Template_During_Initialization(&global_grid_D, dimx, dimy, dimz, dimxy);
    
        Compute_Weights_D(Count_Per_Cell, Weights, subdivisions_per_node, dimx, dimy, dimz, dimxy);

        {
            int starting_cell_index = (node_id_CPU + 0) * subdivisions_per_node;
            int   ending_cell_index = (node_id_CPU + 1) * subdivisions_per_node;
            int cells = subdivisions_per_node;
            for(int k=0; k<cells; k++) local_Weights[k] = Weights[starting_cell_index + k];
        }

#if 0
        for(int i=0; i<nnodes_CPU; i++)
	    {
            MPI_Bcast(Weights + i*subdivisions_per_node, subdivisions_per_node, MPI_LONG_LONG_INT, i, MPI_COMM_WORLD);
        }
#else
        MPI_Allgather(local_Weights, subdivisions_per_node, MPI_LONG_LONG_INT, Weights, subdivisions_per_node, MPI_LONG_LONG_INT, MPI_COMM_WORLD);
#endif

unsigned long long int e2time = ___rdtsc();
if (node_id_CPU == 0) printf("(%d) (%s) ::: Time Taken = %lld (%.2lf seconds)\n", __LINE__, __FILE__, (e2time - stime), (e2time - stime)/CORE_FREQUENCY_CPU);
        //Now we have the weights of all the cells... Let's divide them equally amongst the nodes...


        //Step G... Compute Owner of each Cell...
        global_Owner_D = (int *)malloc(number_of_subdivisions * sizeof(int));
        for(int k=0; k<number_of_subdivisions; k++) global_Owner_D[k] = -1; //Initialized to -1, i.e. not owned by anyone :)

        {
            //printf("ABCD\n");
            long long int sum_so_far = 0;
            for(int k=0; k<number_of_subdivisions; k++) sum_so_far  += Weights[k];
            long long int total_weights = sum_so_far;
            long long int weights_per_node = (sum_so_far + nnodes_CPU - 1)/nnodes_CPU;
            mpi_printf("total_weights = %lld ::: weights_per_node = %lld\n", total_weights, weights_per_node);
            fflush(stdout);
            int alloted_so_far = 0;

            sum_so_far = 0;
            //Divide cells per node here :) At least in theory :)
            for(int k=0; k<nnodes_CPU; k++)
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
                        if (Weights[k] == 0) global_Owner_D[k] = nnodes_CPU-1;
                        else ERROR_PRINT();
                    }
                }

            }
        
            int *Temp1 = (int *)malloc(nnodes_CPU * sizeof(int)); for(int k=0; k<nnodes_CPU; k++) Temp1[k] = 0;
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
                for(int k=0; k<nnodes_CPU; k++) 
                {
                    final_sum += Temp1[k];
                    min = PCL_MIN(Temp1[k], min);
                    max = PCL_MAX(Temp1[k], max);
                    //mpi_printf("[%d]--%d ", k, Temp1[k]); 
                }
                //mpi_printf("\n");
                size_t avg = final_sum/nnodes_CPU;
                mpi_printf("Avg. = %d ::: max = %d ::: Ratio = %.2lf\n", (int)(avg), max, (max*1.0)/avg);

                if (final_sum != global_number_of_galaxies) 
                {
                    printf("final_sum = %lld ::: global_number_of_galaxies = %lld\n", final_sum, global_number_of_galaxies);
                    ERROR_PRINT();
                }
            }
        }

unsigned long long int e4time = ___rdtsc();
if (node_id_CPU == 0) printf("(%d) (%s) ::: Time Taken = %lld (%.2lf seconds)\n", __LINE__, __FILE__, (e4time - stime), (e4time - stime)/CORE_FREQUENCY_CPU);


        global_Required_D = (unsigned char *)malloc(number_of_subdivisions *sizeof(unsigned char));
        global_Required_D_For_R_CPU = (unsigned char *)malloc(number_of_subdivisions *sizeof(unsigned char));
        for(int k=0; k<number_of_subdivisions; k++) global_Required_D[k] = 0;
        for(int k=0; k<number_of_subdivisions; k++) global_Required_D_For_R_CPU[k] = 0;
        ///////  WE will now again look at neighbors of the alloted nodes, and figure out which nodes are required...

        //XXX: For D, we only require the nodes for which global_Owner is me -- i.e. node_id_CPU...
        //XXX: But we will still compute the neighboring nodes //required, and store it for later -- something that R will need...
        for(int k = 0; k < number_of_subdivisions; k++)
        {
            if (global_Owner_D[k] == node_id_CPU) global_Required_D[k] = 1;
        }
        Compute_Required_D_For_R(number_of_subdivisions, dimx, dimy, dimz, dimxy);
#if 0
#endif


unsigned long long int e3time = ___rdtsc();
if (node_id_CPU == 0) printf("(%d) (%s) ::: Time Taken = %lld (%.2lf seconds)\n", __LINE__, __FILE__, (e3time - stime), (e3time - stime)/CORE_FREQUENCY_CPU);

        {
            int ssum = 0; 
            int particle_sum = 0;
            
            for(int k=0; k<number_of_subdivisions; k++) 
            {
                ssum += global_Required_D[k];
                if (global_Required_D[k]) particle_sum += Count_Per_Cell[k];
            }

            mpi_printf("node_id_CPU = %d ::: ssum = %d (number_of_subdivisions = %d) ::: particle_sum = %d (global_number_of_galxies = %lld)\n", 
                node_id_CPU, ssum, number_of_subdivisions, particle_sum, global_number_of_galaxies);
        }


        //Step I: BroadCast this global_Required to eveyone -- so that they all start sending out the required particles...

        unsigned char *Local_Required_Bkp = (unsigned char *)malloc(number_of_subdivisions * sizeof(unsigned char));
        for(int k=0; k<number_of_subdivisions; k++) Local_Required_Bkp[k] = global_Required_D[k];

        Convert_From_Byte_To_Bit(global_Required_D, number_of_subdivisions);
        int number_of_subdivisions_by_eight = number_of_subdivisions/8;
        if ((number_of_subdivisions_by_eight * 8) != number_of_subdivisions) number_of_subdivisions_by_eight++;
        unsigned char *Local_Required_All_Nodes = (unsigned char *)malloc(nnodes_CPU * number_of_subdivisions_by_eight * sizeof(unsigned char));
        {

    #if 0
            unsigned char *Dst = Local_Required_All_Nodes + node_id_CPU * number_of_subdivisions;
            for(int k = 0; k < number_of_subdivisions; k++) Dst[k] = global_Required[k];
            for(int k=0; k<nnodes_CPU; k++)
            {
                MPI_Bcast(Local_Required_All_Nodes + k*number_of_subdivisions, number_of_subdivisions, MPI_CHAR, k, MPI_COMM_WORLD);
            }
    #else
            //MPI_Allgather(global_Required, number_of_subdivisions, MPI_CHAR, Local_Required_All_Nodes, number_of_subdivisions, MPI_CHAR, MPI_COMM_WORLD);
            MPI_Allgather(global_Required_D, number_of_subdivisions_by_eight, MPI_UNSIGNED_CHAR, 
                          Local_Required_All_Nodes, number_of_subdivisions_by_eight, MPI_UNSIGNED_CHAR, MPI_COMM_WORLD);
        
    #endif
        }
unsigned long long int e12time = ___rdtsc();
if (node_id_CPU == 0) printf("(%d) (%s) ::: Time Taken = %lld (%.2lf seconds)\n", __LINE__, __FILE__, (e12time - stime), (e12time - stime)/CORE_FREQUENCY_CPU);

        for(int p=0; p<nnodes_CPU; p++) 
        {
            //mpi_printf("Required of cell_id (%d) == %d\n", debug_cell_id, Local_Required_All_Nodes[p*number_of_subdivisions + debug_cell_id]);
        }

        //Step J... Compute Prefix-Sum... It's like the cummulative sum...

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


        //Step K:
        {
            // I am node_id_CPU and I need to send data to each node, and also figure out how much will I get from every node...
            int *Count_of_particles_to_send = (int *)malloc(nnodes_CPU * sizeof(int));
            int *Count_of_particles_to_recv = (int *)malloc(nnodes_CPU * sizeof(int));

            Compute_Count_of_particles_to_send(Local_Required_All_Nodes, Count_of_particles_to_send, Send_Count, number_of_subdivisions);
      #if 0      
            for(int k=0; k<nnodes_CPU; k++)
            {
                unsigned char *Src = Local_Required_All_Nodes + k*number_of_subdivisions;
                int sum = 0;

                for(int m=0; m<number_of_subdivisions; m++)
                {
                    if (Src[m])
                    {
                        sum += Send_Count[m+1] - Send_Count[m];
                        //if (m == debug_cell_id) mpi_printf(" (%d) ::: XYZ -- %d\n", node_id_CPU, Send_Count[m+1] - Send_Count[m]);
                    }
                }

                Count_of_particles_to_send[k] =  sum;
             }

            for(int k=0; k<nnodes_CPU; k++)
            {
                //mpi_printf("<%d> ::: Count_of_particles_to_send[%d] = %d\n", node_id, k , Count_of_particles_to_send[k]);
            }

        #endif
        #if 0
             for(int k=0; k<nnodes_CPU; k++)
             {
                    MPI_Gather(Count_of_particles_to_send + k, 1, MPI_INT, Count_of_particles_to_recv, 1, MPI_INT, k, MPI_COMM_WORLD);
             }
        #else
            MPI_Alltoall(Count_of_particles_to_send, 1, MPI_INT, Count_of_particles_to_recv, 1, MPI_INT, MPI_COMM_WORLD);
        #endif
unsigned long long int e11time = ___rdtsc();
if (node_id_CPU == 0) printf("(%d) (%s) ::: Time Taken = %lld (%.2lf seconds)\n", __LINE__, __FILE__, (e11time - stime), (e11time - stime)/CORE_FREQUENCY_CPU);

             int total_particles_to_send;
             int total_particles_to_recv;
             {
                int sum0 = 0, sum1 = 0;
                for(int k=0; k<nnodes_CPU; k++) sum0 += Count_of_particles_to_send[k];
                for(int k=0; k<nnodes_CPU; k++) sum1 += Count_of_particles_to_recv[k];
                mpi_printf("node_id_CPU = %d ::: sum0 = %d ::: sum1 = %d\n", node_id_CPU, sum0, sum1);
                total_particles_to_send = sum0;
                total_particles_to_recv = sum1;
             }

             int *Prefix_Sum_Count_of_particles_to_send = (int *)malloc((nnodes_CPU + 1) * sizeof(int));
             int *Prefix_Sum_Count_of_particles_to_recv = (int *)malloc((nnodes_CPU + 1) * sizeof(int));
             {
                //Prefix Sum Count_of_particles_to_send...
                int sum = 0;
                for(int k=0; k<nnodes_CPU; k++)
                {
                    int value = Count_of_particles_to_send[k];
                    Prefix_Sum_Count_of_particles_to_send[k] = sum;
                    sum += value;
                }
                Prefix_Sum_Count_of_particles_to_send[nnodes_CPU] = sum;
                if (sum != total_particles_to_send) ERROR_PRINT();

                sum = 0;
                for(int k=0; k<nnodes_CPU; k++)
                {
                    int value = Count_of_particles_to_recv[k];
                    Prefix_Sum_Count_of_particles_to_recv[k] = sum;
                    sum += value;
                }
                Prefix_Sum_Count_of_particles_to_recv[nnodes_CPU] = sum;
                if (sum != total_particles_to_recv) ERROR_PRINT();
            }


            //Step L:
            TYPE *Data_To_Send = (TYPE *)malloc(total_particles_to_send * 3 * sizeof(TYPE));
            TYPE *Data_To_Recv = (TYPE *)malloc(total_particles_to_recv * 3 * sizeof(TYPE));

#if 0 // <<----
            for(int sender=0; sender < nnodes_CPU; sender++)
            {
                size_t recv_count_in_floats = Count_of_particles_to_recv[sender] * 3;

                TYPE *Addr = Data_To_Recv + Prefix_Sum_Count_of_particles_to_recv[sender] * 3;
                MPI_Irecv(Addr, recv_count_in_floats, MPI_FLOAT, sender, sender, MPI_COMM_WORLD, recv_request + sender);
            }

            //MPI_BARRIER(node_id);

            //Prepare the data to send, and send it...

#if 0


            for(int receiver  = 0; receiver < nnodes_CPU; receiver++)
            {
                TYPE *Src = Data_To_Send + Prefix_Sum_Count_of_particles_to_send[receiver] * 3;
                int offset = 0;
                char *Req = Local_Required_All_Nodes + receiver * number_of_subdivisions;

                for(int k = 0; k<number_of_subdivisions; k++)
                {
                    if (Req[k])
                    {
                    #if 0
                        if (k == debug_cell_id) mpi_printf("Alpha ::: %d --> %d ( %d ::: %d)\n", node_id, receiver, k, Send_Count[k+1]-Send_Count[k]);
                        if (k == debug_cell_id)
                        {
                            char filename[1024];
                            sprintf(filename, "NN_%d", node_id);
                            FILE *fp = fopen(filename, "w");
                            for(int p=Send_Count[k]; p < Send_Count[k+1]; p++)
                            {
                                fprintf(fp, "%f %f %f\n", Local_Pos[3*p + 0], Local_Pos[3*p + 1], Local_Pos[3*p + 2]);
                            }
                            fclose(fp);
                        }
                    #endif
                        for(int p=Send_Count[k]; p < Send_Count[k+1]; p++)
                        {
                            Src[3*offset + 0] = Local_Pos[3*p + 0];
                            Src[3*offset + 1] = Local_Pos[3*p + 1];
                            Src[3*offset + 2] = Local_Pos[3*p + 2];
                            offset++;
                        }
                     }
                 }

                 if (offset != (Prefix_Sum_Count_of_particles_to_send[receiver+1] - Prefix_Sum_Count_of_particles_to_send[receiver])) ERROR_PRINT();
                 size_t send_count_in_floats = offset* 3;
                 MPI_Isend(Src, send_count_in_floats, MPI_FLOAT, receiver, node_id_CPU, MPI_COMM_WORLD,  send_request_key + receiver);
             }
#else
		Prepare_Data_To_Send(Local_Pos, Data_To_Send, Prefix_Sum_Count_of_particles_to_send, Local_Required_All_Nodes, Send_Count, number_of_subdivisions);
unsigned long long int e6time = ___rdtsc();
if (node_id_CPU == 0) printf("(%d) (%s) ::: Time Taken = %lld (%.2lf seconds)\n", __LINE__, __FILE__, (e6time - stime), (e6time - stime)/CORE_FREQUENCY_CPU);

        for(int receiver  = 0; receiver < nnodes_CPU; receiver++)
		{
            TYPE *Src = Data_To_Send + Prefix_Sum_Count_of_particles_to_send[receiver] * 3;
            int offset = Prefix_Sum_Count_of_particles_to_send[receiver+1] - Prefix_Sum_Count_of_particles_to_send[receiver];
            size_t send_count_in_floats = offset* 3;
            MPI_Isend(Src, send_count_in_floats, MPI_FLOAT, receiver, node_id_CPU, MPI_COMM_WORLD,  send_request_key + receiver);
		}
#endif

             int key_messages_remaining = nnodes_CPU;

             while (key_messages_remaining > 0)
             {
                int msg_index;
                MPI_Waitany(nnodes_CPU, recv_request, &msg_index, recv_status);
                key_messages_remaining--;
             }

             MPI_Waitall(nnodes_CPU, send_request_key, MPI_STATUSES_IGNORE); 
#else

		Prepare_Data_To_Send(Local_Pos, Data_To_Send, Prefix_Sum_Count_of_particles_to_send, Local_Required_All_Nodes, Send_Count, number_of_subdivisions);
unsigned long long int e6time = ___rdtsc();
if (node_id_CPU == 0) printf("(%d) (%s) ::: Time Taken = %lld (%.2lf seconds)\n", __LINE__, __FILE__, (e6time - stime), (e6time - stime)/CORE_FREQUENCY_CPU);

    for(int k=0; k<nnodes_CPU; k++) Count_of_particles_to_send[k] = Count_of_particles_to_send[k] * 3;
    for(int k=0; k<nnodes_CPU; k++) Count_of_particles_to_recv[k] = Count_of_particles_to_recv[k] * 3;
    for(int k=0; k<=nnodes_CPU; k++) Prefix_Sum_Count_of_particles_to_send[k] = Prefix_Sum_Count_of_particles_to_send[k] * 3;
    for(int k=0; k<=nnodes_CPU; k++) Prefix_Sum_Count_of_particles_to_recv[k] = Prefix_Sum_Count_of_particles_to_recv[k] * 3;
    MPI_Alltoallv(Data_To_Send, Count_of_particles_to_send, Prefix_Sum_Count_of_particles_to_send, MPI_FLOAT, 
                  Data_To_Recv, Count_of_particles_to_recv, Prefix_Sum_Count_of_particles_to_recv, MPI_FLOAT, MPI_COMM_WORLD);

#endif


             mpi_printf("Fully Finished :: (%d)\n", node_id_CPU);

             global_Positions_R = global_Positions_D;
             global_Positions_D = Data_To_Recv;

             global_number_of_galaxies_on_node_D = 0;
             int actual_subdivisions = dimx * dimy * dimz;

             int kmin = actual_subdivisions;
             int kmax = -1;
             for(int k=0; k<actual_subdivisions; k++) 
             {
                if (global_Owner_D[k] == -1) ERROR_PRINT();
                if (global_Owner_D[k] == node_id_CPU) 
                {
                    if (kmin > k) kmin = k;
                    if (kmax < k) kmax = k;
                }
             }

             global_starting_cell_index_D_CPU = kmin;
             global_ending_cell_index_D_CPU = kmax + 1;

             global_number_of_galaxies_on_node_D = total_particles_to_recv;

             mpi_printf("++ %d ++ global_starting_cell_index_D = %d :::  global_ending_cell_index_D = %d ::: global_number_of_galaxies_on_node_D = %lld\n", 
             node_id_CPU, global_starting_cell_index_D_CPU, global_ending_cell_index_D_CPU, global_number_of_galaxies_on_node_D);


             for(int k=global_starting_cell_index_D_CPU; k<global_ending_cell_index_D_CPU; k++) if (global_Owner_D[k] != node_id_CPU) ERROR_PRINT();
            for(int k=0; k<number_of_subdivisions; k++) global_Required_D[k] = Local_Required_Bkp[k];
        }

        free(Local_Pos);
    }

unsigned long long int e5time = ___rdtsc();
if (node_id_CPU == 0) printf("(%d) (%s) ::: Time Taken = %lld (%.2lf seconds)\n", __LINE__, __FILE__, (e5time - stime), (e5time - stime)/CORE_FREQUENCY_CPU);

    PRINT_BLUE
    if (node_id_CPU == 0) mpi_printf("dimx = %d ::: dimy = %d ::: dimz = %d\n", dimx, dimy, dimz);
    PRINT_BLACK

    Populate_Grid(&global_grid_D_CPU, dimx, dimy, dimz);


    //MPI_BARRIER(node_id);

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//STEP 5: COMPUTE BOUNDING_BOX OF THE DATASET...
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#if 0
    Compute_Min_Max(global_Positions, global_number_of_galaxies, DIMENSIONS, global_grid.Min, global_grid.Max, global_grid.Extent, 
               global_grid.dimx, global_grid.dimy, global_grid.dimz, global_grid.Cell_Width);

#endif

    if (node_id_CPU == 0)
    {
    PRINT_LIGHT_RED
        mpi_printf("global_Min = [%e %e %e] ::: global_Max = [%e %e %e]\n", global_grid_D_CPU.Min[0], global_grid_D_CPU.Min[1], global_grid_D_CPU.Min[2], global_grid_D_CPU.Max[0], global_grid_D_CPU.Max[1], global_grid_D_CPU.Max[2]);
        mpi_printf("global_Extent = [%e %e %e]\n", global_grid_D_CPU.Extent[0], global_grid_D_CPU.Extent[1], global_grid_D_CPU.Extent[2]);
        mpi_printf("Total Memory Allocated = %lld Bytes (%.2lf GB)\n", global_memory_malloced, global_memory_malloced/1000.0/1000.0/1000.0);
    PRINT_BLACK
    }

    unsigned long long int etime = ___rdtsc();
    global_time_mpi += etime - stime;
}

    
void Parse_Files_And_Populate_Temp_Memory(void)
{
    //=== Step 1... Initialize MPI_Mallocs...

    unsigned long long int stime = ___rdtsc();
    Allocate_Temporary_Arrays();
    Initialize_MPI_Mallocs();

    char filename_D[256];
    char filename_R[256];

    sprintf(filename_D, "%s/input_D.bin", global_dirname);
    Read_D_R_File(filename_D, &global_Positions_D, &global_number_of_galaxies_on_node_D);
    Distribute_D_Particles_Amongst_Nodes();
    Compute_KD_Tree_Acceleration_Data_Structure_For_D();

    Copy_Non_Changing_Data_From_D_To_R();

    MPI_BARRIER(node_id_CPU);

#define NUMBER_OF_RANDOM_FILES 1

    for(int k = 0; k < NUMBER_OF_RANDOM_FILES; k++)
    {
        sprintf(filename_R, "%s/input_R_%d.bin", global_dirname, k);
        //sprintf(filename_R, "%s/input_D.bin", global_dirname);
        Read_D_R_File(filename_R, &global_Positions_R, &global_number_of_galaxies_on_node_R);
        Distribute_R_Particles_Amongst_Nodes();
        Compute_KD_Tree_Acceleration_Data_Structure_For_R();
    
        //Perform_Mandatory_Initializations(&global_grid_R, global_number_of_galaxies_on_node_R, global_Lbox, global_rminL, global_rmaxL, global_nrbin);

        Copy_DR_To_Temp_Memory();
    }

    MPI_BARRIER(node_id_CPU);
    //exit(123);

    unsigned long long int etime = ___rdtsc();
    unsigned long long int ttime = etime - stime;
    if (node_id_CPU == 0) printf("<95123> <<%d> MPI + KD_Tree Time = %lld cycles (%.2lf seconds)\n", node_id_CPU, ttime, ttime/CORE_FREQUENCY_CPU);



}
#endif

void Perform_DRs2(int argc, char **argv)
{

    if (node_id_CPU == 0) { printf("GLOBAL_THRESHOLD_PARTICLES_PER_CELL = %d\n", GLOBAL_THRESHOLD_PARTICLES_PER_CELL); fflush(stdout); }

#ifndef MPI_COMPUTATION
    Load_Temp_Memory_From_File();
#else
    Parse_Files_And_Populate_Temp_Memory();
#endif

    size_t length_D = DataTransfer_Size_D_From_CPU_To_MIC_CPU/sizeof(int);
    size_t length_R = DataTransfer_Size_R_From_CPU_To_MIC_CPU/sizeof(int);
    int *Packet_D_CPU = Temp_Memory_D_CPU;
    int *Packet_R_CPU = Temp_Memory_R_CPU;


    CPU_Function_Populate_CPU_Grids(Packet_D_CPU, length_D, Packet_R_CPU, length_R);

#ifdef HETERO_COMPUTATION
    CPU_Compute_Hetero_Weights();
#endif
   
    unsigned char *Global_Variables = NULL; 
    size_t length_in_bytes_of_global_variables = Fill_Up_Global_Variables_Packet(&Global_Variables);

    TYPE *Answer_on_MIC = (TYPE *)malloc(sizeof(TYPE)); Answer_on_MIC[0] = 0;
    TYPE *Answer_on_CPU = (TYPE *)malloc(sizeof(TYPE)); Answer_on_CPU[0] = 0;

    //if (node_id_CPU == 0)
    {
        printf("<<%d>> DataTransfer_Size_D_From_CPU_To_MIC_CPU = %lld Bytes (%.2lf GB) ::: length_D = %lld\n", 
                node_id_CPU, DataTransfer_Size_D_From_CPU_To_MIC_CPU, DataTransfer_Size_D_From_CPU_To_MIC_CPU/1000/1000.0/1000.0,  length_D);
        printf("<<%d>> DataTransfer_Size_R_From_CPU_To_MIC_CPU = %lld Bytes (%.2lf GB) ::: length_R = %lld\n", 
                node_id_CPU, DataTransfer_Size_R_From_CPU_To_MIC_CPU, DataTransfer_Size_R_From_CPU_To_MIC_CPU/1000/1000.0/1000.0,  length_R);
        printf("<<%d>> length_in_bytes_of_global_variables = %lld Bytes (%.2lf GB)\n", node_id_CPU, length_in_bytes_of_global_variables, length_in_bytes_of_global_variables/1000.0/1000.0/1000.0);
	size_t stotal = DataTransfer_Size_D_From_CPU_To_MIC_CPU + DataTransfer_Size_R_From_CPU_To_MIC_CPU + length_in_bytes_of_global_variables;
	printf("<<%d>> Total Transfer Size = %lld Bytes (%.2lf GB)\n", node_id_CPU, stotal, stotal/1000.0/1000.0/1000.0);
    }

//XXXXX
#if 1
//if (node_id_CPU == 169) return;
//if (node_id_CPU == 31) return;
//if (node_id_CPU == 87) return;
//if (node_id_CPU == 228) return;
//if (node_id_CPU == 74) return;
//if (node_id_CPU == 86) return;
//if (node_id_CPU == 192) return;
//if (node_id_CPU == 216) return;
//if (node_id_CPU == 240) return;
//if (node_id_CPU == 234) return;
//if (node_id_CPU == 258) return;
//if (node_id_CPU == 75) return;
//if (node_id_CPU == 190) return;
//if (node_id_CPU == 193) return;
#endif
    unsigned long long int stime = ___rdtsc();
#if 1
    //__Offload_report(1);
#if 1
    #pragma offload target(mic:0) \
        in(Packet_D_CPU : length(length_D)) in(Packet_R_CPU: length(length_R)) in(Global_Variables : length(length_in_bytes_of_global_variables)) \
        out(Answer_on_MIC : length(1)) \
        signal(Answer_on_MIC)


    {
        //signal(Answer_on_MIC)
        //MIC Activity...
        MICFunction_Perform_TPCF_On_MIC(Packet_D_CPU, length_D, Packet_R_CPU, length_R, Global_Variables, Answer_on_MIC);
    }
#endif
#endif
    {
        //CPU Activity...
        CPUFunction_Perform_TPCF_On_CPU(Packet_D_CPU, length_D, Packet_R_CPU, length_R, Global_Variables, Answer_on_CPU);
        //printf("Waiting for stuff to finish :)...\n"); fflush(stdout);
    }

#if 1
#if 1
    #pragma  offload target(mic:0) wait(Answer_on_MIC)
#endif
#endif

    {
        unsigned long long int etime = ___rdtsc();
        unsigned long long int ttime = etime - stime;
        double seconds = ttime/CORE_FREQUENCY_CPU;
        size_t data_transferred = length_D * sizeof(int) + length_R * sizeof(int);
        printf("data_transferred = %lld ::: Time = %lld cycles (%.2lf seconds) ::: BW = %.2lf GB/sec\n", data_transferred, ttime, seconds, data_transferred/seconds/1000.0/1000.0/1000.0);

    }

    //MPI_BARRIER(node_id_CPU);
}

#ifdef MPI_COMPUTATION
void Initialize_MPI(int argc, char **argv)
{
    int provided;

    MPI_Status stat;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided); /*START MPI */
    MPI_Comm_rank(MPI_COMM_WORLD, &node_id_CPU); /*DETERMINE RANK OF THIS PROCESSOR*/
    MPI_Comm_size(MPI_COMM_WORLD, &nnodes_CPU); /*DETERMINE TOTAL NUMBER OF PROCESSORS*/
    MPI_Errhandler_set(MPI_COMM_WORLD,MPI_ERRORS_RETURN);

    if (node_id_CPU == (nnodes_CPU - 1)) { printf("Done Initialize_MPI()\n"); fflush(stdout);}
}

void Finalize_MPI(void)
{
    MPI_Finalize();

    if (node_id_CPU == (nnodes_CPU - 1)) { printf("Done Finalize_MPI()\n"); fflush(stdout);}
}


#endif

void Perform_Actual_DRs2(int argc, char **argv)
{
	MPI_BARRIER(node_id_CPU);
	unsigned long long int stime = ___rdtsc();

	Perform_DRs2(argc, argv);

	MPI_BARRIER(node_id_CPU);
	unsigned long long int etime = ___rdtsc();
	unsigned long long int ttime = etime - stime;

	if (node_id_CPU == 0)
	{
		printf("TotalTotalTotal Time = %lld cycles (%.2lf seconds)\n", ttime, ttime/CORE_FREQUENCY_CPU);
	}
}

int main(int argc, char **argv)
{
#ifdef MPI_COMPUTATION
    Initialize_MPI(argc, argv);
#endif

    ParseArgs(argc, argv);

    int mic_available = Check_For_MIC();
    MPI_BARRIER(node_id_CPU);

    Perform_Actual_DRs2(argc, argv);

#ifdef MPI_COMPUTATION
    Finalize_MPI();
#endif
}
