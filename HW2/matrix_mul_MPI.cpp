/************************************************************
 * Author:  	Yash Sanjay Bhalgat
 * Roll no: 	13D070014
 * Course:  	ME766
 * Description: Large Matrix Multiplication using MPI library
 * Tested:		tested on test matrix provided by Prof. Gopalakrishnan
************************************************************/ 

#include <iostream>
#include <cstdlib>
#include <mpi.h>
#include <time.h>

#define SIZE 10000

using namespace std;

void display_matrices(int mype, double* C, double* A, double* B);

int main(int argc, char* argv[])
{
	int mype, no_of_threads;
	int tag = 0;
	double start, end;
	MPI_Status status;    
	
	double *A, *B, *C;
	int submat_size;
	int start_address;
	
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &mype);
	MPI_Comm_size(MPI_COMM_WORLD, &no_of_threads);

	cout << "Comm size : " << no_of_threads << endl;
	
	submat_size = SIZE/no_of_threads;
	
	//srand(time(NULL));
	
	B = new double [SIZE*SIZE];
	// if(mype == 0) {
	// 	A = new double [SIZE*SIZE];
	// 	C = new double [SIZE*SIZE];
	// 	for(int i=0; i<SIZE*SIZE; ++i) {
	// 		A[i] = rand()%100;
	// 		B[i] = rand()%100;
	// 	}

	// }
	// else {
	// 	A = new double [SIZE*submat_size];
	// 	C = new double [SIZE*submat_size];
	// 	for(int i=0; i<SIZE*submat_size; ++i) {
	// 		C[i] = 0.0;
	// 	}
	// }
	
	if(mype == 0) {
		A = new double [SIZE*SIZE];
		C = new double [SIZE*SIZE];
		for(int i = 1; i <= SIZE; i++){
			for(int j = 1; j <= SIZE; j++){
				// test matrix given by Prof. Gopalakrishnan
				A[(j-1) + (i-1)*SIZE] = i+j;
				B[(j-1) + (i-1)*SIZE] = i*j;
			}
		}
	}
	else {
		A = new double [SIZE*submat_size];
		C = new double [SIZE*submat_size];
		for(int i=0; i<SIZE*submat_size; ++i) {
			C[i] = 0.0;
		}
	}

	
	start = MPI_Wtime();
		
	if(mype == 0) {
		start_address = SIZE*submat_size;
		for(int i=1; i<no_of_threads; ++i) {
			MPI_Send(A + start_address, SIZE*submat_size, MPI_DOUBLE, i, tag, MPI_COMM_WORLD);
			MPI_Send(B, SIZE*SIZE, MPI_DOUBLE, i, tag+1, MPI_COMM_WORLD);
			start_address += SIZE*submat_size;
		}
	}
	else {
		MPI_Recv(A, SIZE*submat_size, MPI_DOUBLE, 0, tag, MPI_COMM_WORLD, &status);
		MPI_Recv(B, SIZE*SIZE, MPI_DOUBLE, 0, tag+1, MPI_COMM_WORLD, &status);
	}
	
	
	// MPI_Bcast(B, SIZE*SIZE, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	
	// Standard multiplication method
	for(int i = 0; i < submat_size; i++)
	{
		for(int j = 0; j < SIZE; j++)
		{
			C[j + i*SIZE] = 0;
			for(int k = 0; k < SIZE; k++)
			{
				C[j + i*SIZE] += A[i*SIZE + k]*B[j + SIZE*k];
				
			}   
		}
	}
	
	if(mype == 0) {
		start_address = SIZE*submat_size;
		for(int i=1; i<no_of_threads; ++i) {
			MPI_Recv(C + start_address, SIZE*submat_size, MPI_DOUBLE, i, 2, MPI_COMM_WORLD, &status);
			start_address += SIZE*submat_size;
		}
	}
	else {
		MPI_Send(C, SIZE*submat_size, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
	}
	
	end = MPI_Wtime();
	
	cout<<"Using the MPI library for mat_multiplication of size: "<<SIZE<<" x "<<SIZE<<endl;
	cout << "Total time taken by process " << mype <<" = " << end-start << endl;
	
	display_matrices(mype, C, A, B);
	
	MPI_Finalize();
}

void display_matrices(int mype, double* C, double* A, double* B){
	if(mype == 0) {
	    if(SIZE > 10) {
	        cout << "Size is to large to display..." << endl;
	    }
	    else {
	        for(int i=0; i<SIZE; ++i) {
	            for(int j=0; j<SIZE; ++j) {
	                cout << C[j + i*SIZE] << " ";
	            }
	            cout << endl;
	        }
	        for(int i=0; i<SIZE; ++i) {
	            for(int j=0; j<SIZE; ++j) {
	                cout << A[j + i*SIZE] << " ";
	            }
	            cout << endl;
	        }
	        for(int i=0; i<SIZE; ++i) {
	            for(int j=0; j<SIZE; ++j) {
	                cout << B[j + i*SIZE] << " ";
	            }
	            cout << endl;
	        }
	    }
	}
}
