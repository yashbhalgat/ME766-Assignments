#include <iostream>
#include <cstdlib>

using namespace std;

#define SIZE 10

int matrix_multiply(double A[], double B[], int ROWA, int COLA, int ROWB, int COLB, double C[])
{
  if(COLA != ROWB) return -1;              // Check size - Cannot multiply matrix
  
  
  for(int i = 0; i < ROWA; i++)
  {
    for(int j = 0; j < COLB; j++)
    {
      C[j + i*COLB] = 0;
      for(int k = 0; k < COLA; k++)
      {
        C[j + i*COLB] += A[i*COLA + k]*B[j + COLB*k];          // dot-product
      }    
    } 
  }
  
  return 0;
}



int main(int argc, char* argv[])
{
    /*if (argc != 2)
    {
        cout << "Require matrix size as command line argument" << endl;
        cout << "Usage: ./a.out [size_of_matrix]" << endl;
        exit(1);
    }*/
    
    clock_t start, end;
    int err_num;
    
    double *A = new double[SIZE*SIZE];
    double *B = new double[SIZE*SIZE];
    double *C = new double[SIZE*SIZE];
    
    for(int i=0; i<(SIZE*SIZE); i++)
    {
        A[i] = rand()%100;
        B[i] = rand()%100;
    }

    start = clock();
    err_num = matrix_multiply(A, B, SIZE, SIZE, SIZE, SIZE, C);
    end = clock();
    
    cout<<"Time taken to multiply "<< SIZE << " X " << SIZE << " matrix : "<<(double)(end-start)/CLOCKS_PER_SEC<<" seconds"<< endl;
    
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
    
    delete [] A;
    delete [] B;
    delete [] C;
    
    return 0;
}
