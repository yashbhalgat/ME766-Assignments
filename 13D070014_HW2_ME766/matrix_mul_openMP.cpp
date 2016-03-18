/**********************************************************************
 * Author:  	Yash Sanjay Bhalgat
 * Roll no: 	13D070014
 * Course:  	ME766
 * Description: Large Matrix Multiplication using the openMP library
 * Tested:		tested on test matrix provided by Prof. Gopalakrishnan
**********************************************************************/ 

#include <iostream>
#include <cstdlib>
#include <mpi.h>
#include <omp.h>
#include <time.h>

// mpi.h included to give exact time of execution
// MPI_Wtime() give more accurate result than omp_get_wtime()

#define SIZE 500

using namespace std;

int main()
{
	float *A, *B, *C;
	int max_t;
	double start, end;
	
	size_t ROWA = SIZE, ROWB = SIZE, COLA = SIZE, COLB = SIZE;
	int errNum;
	
	A = new float[ROWA*COLA];
	B = new float[ROWB*ROWB];
	C = new float[ROWA*COLB];
	
	// srand (time(NULL));
	
	// for(int i=0; i<(SIZE*SIZE); i++){
	// 	A[i] = rand()%100;
	// 	B[i] = rand()%100;
	// }
	
	// test matrix as provided by Prof. Gopalakrishnan
	for(int i = 1; i <= SIZE; i++){
		for(int j = 1; j <= SIZE; j++){
			A[(j-1) + (i-1)*SIZE] = i+j;
			B[(j-1) + (i-1)*SIZE] = i*j;
		}
	}

	start = MPI_Wtime();
	
	#pragma omp parallel for shared (A, B, C, ROWA, COLA, COLB) //execute `export OMP_NUM_THREADS=1`
	
	for(int i = 0; i < ROWA; i++){
		for(int j = 0; j < COLB; j++){
			C[j + i*COLB] = 0;
			for(int k = 0; k < COLA; k++){
				C[j + i*COLB] += A[i*COLA + k]*B[j + COLB*k];   // dot-product
			}    
		} 
	}
	
	end = MPI_Wtime();
	
	cout<<"Time taken by openMP to multiply matrices of size "<<SIZE<<" x "<<SIZE<<" is : "<< (end-start) <<" secs"<<endl;
	
	// for(int i=0; i<SIZE; ++i) {
 //        for(int j=0; j<SIZE; ++j) {
 //            cout << C[j + i*SIZE] << " ";
 //        }
 //        cout << endl;
 //    }

    // release memory just to avoid the core faults in case of large matrices
	delete [] A;
	delete [] B;
	delete [] C;
}
