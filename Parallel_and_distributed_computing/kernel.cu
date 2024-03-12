
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_functions.h>





#include <stdio.h>
#include <iostream>
#include <time.h> 
#include <random>

using namespace std;




cudaError_t Mean(float *c, float *a, float *b, unsigned int size);

void getGpuProperties(void) {
	int MB = 1024 * 1024, KB = 1024;
	float GHz = 1000000.0;
	cudaDeviceProp pdev;

	cout << "GPU Properties : " << endl;
	cout << "-------------------------------------------------------------------------- " << endl;

	cudaGetDeviceProperties(&pdev, 0);
	cout << "Name                                :  " << pdev.name << "               | \n";
	cout << "Capability                          :  " << pdev.major << '.' << pdev.minor << "                               | \n";
	cout << "Global Memory                       :  " << (pdev.totalGlobalMem) / MB << " MB                           | \n";
	cout << "Constant Memory                     :  " << (pdev.totalConstMem) / KB << " KB                             | \n";
	cout << "Shared Memory                       :  " << (pdev.sharedMemPerBlock) / KB << " KB                             | \n";
	cout << "Block Registers                     :  " << (pdev.regsPerBlock) / KB << " KB                             | \n";
	cout << "Clock                               :  " << (pdev.clockRate / GHz) << " GHz                         | \n";
	cout << "Processors                          :  " << pdev.multiProcessorCount << "                                | \n";
	cout << "Cores                               :  " << 8 * pdev.multiProcessorCount << "                               | \n";
	cout << "WarpSize(True paralel Threads)      :  " << pdev.warpSize << "                                | \n";
	cout << "Grid Size                           :  " << pdev.maxGridSize[0] << " " << pdev.maxGridSize[1] << " " << pdev.maxGridSize[2] << "            | \n";
	cout << "Block Size                          :  " << pdev.maxThreadsDim[0] << " " << pdev.maxThreadsDim[1] << " " << pdev.maxThreadsDim[2] << "                      |\n";
	cout << "Threads / Block                     :  " << pdev.maxThreadsPerBlock << "                              |\n";
	cout << "-------------------------------------------------------------------------- " << endl;
	
	cout << "\n\n";
}

__device__ float sum(float *arr,int col,int Width) {
	
	float sum;
	sum = 0;
	
	for (int k = 0; k < Width; k++) {

		sum = sum + arr[k  *  Width + col];
	
	
	
	return sum;
}


__global__ void KernelMean(float*  A, float*  B, float*  C, int Width)
{

		int col = threadIdx.x;
		
		
		
		float A_sum;
		float B_sum = 0;
		/*
		for (int k = 0; k < Width; k++) {
			
			A_sum = A_sum + A[k  *  Width + col];
			B_sum = B_sum + B[k  *  Width + col];
		}
		*/
		A_sum = sum(A, col, Width,);
		A_sum = sum(B, col, Width, );


		float mean_A = (A_sum / Width);
		
		float mean_B = (B_sum / Width);

		

		C[col] = mean_A+ mean_B;
}

float *createPinaka(int num)
{
	int i = 0;
	float *ptr;

	ptr = (float *)malloc(sizeof(float)*num);

	
	
		for (i = 0; i < num; i++)
		{
			ptr[i] = 0.0;
		}
	
	return ptr;
}

float *randomArray(int num,int pin) 
{
	default_random_engine generator;
	uniform_real_distribution<float> distribution(0,10);
	uniform_real_distribution<float> distribution1(0.001, 0.999);
	float *ptr = createPinaka(num);

	int j = 0;
	

	for (j = 0; j < num; j++)
	{
		ptr[j] = pin*(j+1)*distribution(generator)*distribution1(generator);
	}
	return ptr; 
}

float *initArray(int num)
{

	float *ptr = createPinaka(num);

	int j = 0;
	

	for (j = 0; j < num; j++)
	{
		ptr[j] = 0;
	}
	return ptr;
}

void printArray2D(float* array, int Width) {
	for (int i = 0; i < Width; i++) {
		cout << "[ ";
		for (int j = 0; j < Width; j++) {
			cout << array[i  *  Width + j] << "  ";

		}

		cout << "] " << endl;


	}
}

int main()
{
	
	getGpuProperties();
	
	cout << endl;
	
	
	unsigned int WIDTH;
	cout << "Give N = arrays will be (N*N)";
	cin >> WIDTH;

	float* A;
	A = randomArray(WIDTH*WIDTH,2);
	float* B;
	B = randomArray(WIDTH*WIDTH,3);
    float* C;
	C = initArray(WIDTH);



	

    // Add vectors in parallel.
    cudaError_t cudaStatus = Mean(A, B, C, WIDTH);
    
	cout << "               A                   " << endl;
	printArray2D(A,WIDTH);
	cout << "               B                   " << endl;
	printArray2D(B,WIDTH);

	cout << " C = | ";
	for (int i = 0; i < WIDTH; i++) {
		cout << C[i] << "  " << " | ";
	}
    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    
	int c;
	cin >> c;

	free(A);
	free(B);
	free(C);
    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t Mean(float *c, float *a, float *b, unsigned int size)
{
    float *dev_a = 0;
    float *dev_b = 0;
    float *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    
	
	
    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * size * sizeof(float));
   
    cudaStatus = cudaMalloc((void**)&dev_a, size*size * sizeof(float));

	
    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(float));
	
    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_c,c, size*size * sizeof(float), cudaMemcpyHostToDevice);

   
    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(float), cudaMemcpyHostToDevice);
    
	int* d_size=0;
	
	//cudaStatus = cudaMemcpy(d_size, &size, 1 * sizeof(int), cudaMemcpyHostToDevice);
	dim3 dimThread(size);
	dim3 dimBlock(1, 1);

	cudaMemcpy(dev_a, a, size*size * sizeof(float), cudaMemcpyHostToDevice);
    // Launch a kernel on the GPU with one thread for each element.
	KernelMean <<< 1, size>>>(dev_c, dev_a, dev_b,size);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
   
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(b, dev_b, size * sizeof(float), cudaMemcpyDeviceToHost);
    

	cudaFree(dev_c);
	cudaFree(dev_a);
	cudaFree(dev_b);
	return cudaStatus;   
}


