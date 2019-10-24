#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "qdbmp.h"

// Parameters of the render

// Dimensions of the rendered image, Default 1024 x 1024
#define X_DIM 1024
#define Y_DIM 1024
// Offset from the center. Default (0, 0)
#define	X_OFFSET 0.0
#define Y_OFFSET 0.0
// Scale (zoom) level. Default 1
#define SCALE 1.0
// Precision of the iterations. Default 80
#define MAX_ITER 255

__global__ void mandelbrot(int *grid) {
	int i = 32 * blockIdx.x + threadIdx.x;
	int j = 32 * blockIdx.y + threadIdx.y;
	if (i >= Y_DIM || j >= X_DIM) {
		return;
	}
	double x = X_OFFSET + (4.0*i/(X_DIM)-2)/SCALE;
	double y = Y_OFFSET + (4.0*j/(Y_DIM)-2)/SCALE;
	double z_real = 0;
	double z_imag = 0;
	double ab=0;
	int steps = 0;
	double a2 = 0;
	double b2 = 0;
	while (steps < MAX_ITER && a2 + b2 <= 4) {
		z_real = a2 - b2 + x;
		z_imag = 2 * ab + y;
		steps++;
		a2 = z_real * z_real;
		b2 = z_imag * z_imag;
		ab = z_real * z_imag;
	}
	grid[i*Y_DIM + j] = steps;
}

int main() {
	int * GPUgrid;
	int * HOSTgrid;
	cudaMalloc(&GPUgrid, X_DIM * Y_DIM * sizeof(int));
	HOSTgrid = (int*)malloc(X_DIM * Y_DIM * sizeof(int));
	dim3 blocksDim((X_DIM+32)/32, (Y_DIM + 32)/ 32);
	dim3 threadsDim(32, 32);
	mandelbrot <<<blocksDim, threadsDim>> > (GPUgrid);
	cudaDeviceSynchronize();
	cudaMemcpy(HOSTgrid, GPUgrid, X_DIM*Y_DIM * sizeof(int),cudaMemcpyDeviceToHost);
	BMP* bmp = BMP_Create(X_DIM, Y_DIM, 24);
	bool hasError = false;
	for (int i = 0; i < X_DIM; i++) {
		for (int j = 0; j < Y_DIM; j++) {
			//std::cout << HOSTgrid[i*X_DIM + j] << " ";
			// You can play around with these for a custom color scheme.
			int R = 255 * (1- (1.0*(HOSTgrid[i*X_DIM + j]) / MAX_ITER));
			int G = 255 * (1.0*(HOSTgrid[i*X_DIM + j]) / MAX_ITER);
			int B = 255 * (1.0*(HOSTgrid[i*X_DIM + j]) / MAX_ITER);
			BMP_SetPixelRGB(bmp, i, j, R, G, B);
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
	BMP_WriteFile(bmp, "mandelbrot.bmp");
	BMP_Free(bmp);
	free(HOSTgrid);
	cudaFree(GPUgrid);
	return 0;
}