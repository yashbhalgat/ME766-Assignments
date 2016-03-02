/********************************************************
 * Author:  Yash Sanjay Bhalgat
 * Roll no: 13D070014
 * Course:  ME766
********************************************************/ 

#include <iostream>
#include <cstdlib>
#include <mpi.h>
#include <omp.h>

using namespace std;

#define SIZE 1000

int main(int argc, char* argv[])
{
	float *A, *B, *C;
	int max_t;
	double start, end;
	
	size_t ROWA = SIZE, ROWB = SIZE, COLA = SIZE, COLB = SIZE;
	int errNum;
	
	A = new float[ROWA*COLA];
	B = new float[ROWB*ROWB];
	C = new float[ROWA*COLB];
	
	for(int i=0; i<(SIZE*SIZE); i++){
		A[i] = rand()%100;
		B[i] = rand()%100;
	}
	
	start=MPI_Wtime();
	max_t = omp_get_max_threads();
	cout<<max_t<<endl;
	#pragma omp parallel for shared (A, B, C, ROWA, COLA, COLB) //num_threads(4) instead do export OMP_NUM_THREADS=1
	
	for(int i = 0; i < ROWA; i++){
		for(int j = 0; j < COLB; j++){
			C[j + i*COLB] = 0;
			for(int k = 0; k < COLA; k++){
				C[j + i*COLB] += A[i*COLA + k]*B[j + COLB*k];   // dot-product
			}    
		} 
	}
	
	end=MPI_Wtime();
	
	cout<<"Time taken to multiply "<<SIZE<<" x "<<SIZE<<" matrix : "<<(end-start)<<" seconds"<<endl;	
	
	delete [] A;
	delete [] B;
	delete [] C;
}
