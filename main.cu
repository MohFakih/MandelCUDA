#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "qdbmp.h"

#define X_DIM 1024
#define Y_DIM 1024
#define	X_OFFSET 0.0
#define Y_OFFSET 0.0
#define SCALE 1.0
#define BITS_PER_PIXEL 24
#define MAX_ITER 255

__global__ void mandelbrot(int *grid) {
	if (threadIdx.x >= Y_DIM || blockIdx.x >= X_DIM) {
		return;
	}
	double x = 2 - 4.0*threadIdx.x/(SCALE * X_DIM);
	double y = 2 - 4.0*blockIdx.x/(SCALE * Y_DIM);
	double z_real = 0;
	double z_imag = 0;
	double ab=0;
	int i = 0;
	double a2 = 0;
	double b2 = 0;
	while (i < MAX_ITER && a2 + b2 <= 4) {
		z_real = a2 - b2 + x;
		z_imag = 2 * ab + y;
		i++;
		a2 = z_real * z_real;
		b2 = z_imag * z_imag;
		ab = z_real * z_imag;
	}
	grid[threadIdx.x * Y_DIM + blockIdx.x] = i;
}

int main() {
	int * GPUgrid;
	int * HOSTgrid;
	cudaMalloc(&GPUgrid, X_DIM * Y_DIM * sizeof(int));
	HOSTgrid = (int*)malloc(X_DIM * Y_DIM * sizeof(int));
	if (Y_DIM > 1024) {
		std::cout << "GPU may not handle more than 1024 threads." << std::endl;
		return 0;
	}
	mandelbrot <<<X_DIM, Y_DIM>> > (GPUgrid);
	cudaDeviceSynchronize();
	cudaMemcpy(HOSTgrid, GPUgrid, X_DIM*Y_DIM * sizeof(int),cudaMemcpyDeviceToHost);
	BMP* bmp = BMP_Create(X_DIM, Y_DIM, BITS_PER_PIXEL);
	bool hasError = false;
	for (int i = 0; i < X_DIM; i++) {
		for (int j = 0; j < Y_DIM; j++) {
			//std::cout << HOSTgrid[i*X_DIM + j] << " ";
			BMP_SetPixelRGB(bmp, i, j, HOSTgrid[i*X_DIM + j], HOSTgrid[i*X_DIM + j], HOSTgrid[i*X_DIM + j]);
			if (BMP_GetError() != BMP_OK)
			{
				std::cout << BMP_GetErrorDescription() << "XXXX" << BMP_GetError();
				hasError = true;
				break;
			}
		}
		if (hasError) {
			break;
		}
		//std::cout << std::endl;
	}
	BMP_WriteFile(bmp, "test.bmp");
	BMP_Free(bmp);
	free(HOSTgrid);
	cudaFree(GPUgrid);
	return 0;
}