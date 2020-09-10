/*This file contains the kernel code for the GPU Implementation of R3 and R2 Algorithms*/

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "npp.h"
#include <stdio.h>
#include <math.h>
#include "kernel.h"
#include <cstdlib>
#include <cmath>


#define BLOCK_DIM 512

//R3 kernel code
__global__ void cudaR3(vs_t* viewshed, elev_t* elev, elev_t observer_elev, int minX, int maxX, int minY, int maxY, int observerX, int observerY, int ncols)
{
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;


	int width = (maxX - minX) + 1;
	int height = (maxY - minY) + 1;

	if (width <= col || height <= row)
	{
		return;
	}







	int x = col + minX;
	int y = row + minY;

	int index = (x + y * ncols);

	if (x == observerX && y == observerY)
	{
		viewshed[index] = 1;
	}

	int x1 = observerX;
	int y1 = observerY;
	int x2 = x;
	int y2 = y;

	int delta_x(x2 - x1);
	// if x1 == x2, then it does not matter what we set here
	signed char const ix((delta_x > 0) - (delta_x < 0));
	delta_x = std::abs(delta_x) << 1;

	int delta_y(y2 - y1);
	// if y1 == y2, then it does not matter what we set here
	signed char const iy((delta_y > 0) - (delta_y < 0));
	delta_y = std::abs(delta_y) << 1;


	float maxGradient = -10000;

	if (delta_x >= delta_y)
	{
		// error may go below zero
		int error(delta_y - (delta_x >> 1));

		while (x1 != x2)
		{
			if ((error >= 0) && (error || (ix > 0)))
			{
				error -= delta_x;
				y1 += iy;
			}
			// else do nothing

			error += delta_y;
			x1 += ix;


			int deltaY = y1 - observerY;
			int deltaX = x1 - observerX;
			float dist2 = deltaX * deltaX + deltaY * deltaY;
			int currentIndex = (x1 + y1 * ncols);
			double diff_elev = elev[currentIndex] - observer_elev;
			float gradient = (diff_elev * diff_elev) / dist2;
			if (diff_elev < 0) gradient *= -1;

			if (y1 == y && x1 == x)
			{
				if (gradient > maxGradient)
				{
					viewshed[index] = 1;
				}
				else
				{
					viewshed[index] = 0;
				}
			}
			else
			{
				if (gradient > maxGradient)
				{
					maxGradient = gradient;
				}
			}

		}
	}
	else
	{
		// error may go below zero
		int error(delta_x - (delta_y >> 1));

		while (y1 != y2)
		{
			if ((error >= 0) && (error || (iy > 0)))
			{
				error -= delta_y;
				x1 += ix;
			}
			// else do nothing

			error += delta_x;
			y1 += iy;

			int deltaY = y1 - observerY;
			int deltaX = x1 - observerX;
			float dist2 = deltaX * deltaX + deltaY * deltaY;

			int currentIndex = (x1 + y1 * ncols);

			double diff_elev = elev[currentIndex] - observer_elev;
			float gradient = (diff_elev * diff_elev) / dist2;
			if (diff_elev < 0) gradient *= -1;
			if (y1 == y && x1 == x)
			{
				if (gradient > maxGradient)
				{
					viewshed[index] = 1;
				}
				else
				{
					viewshed[index] = 0;
				}
			}
			else
			{
				if (gradient > maxGradient)
				{
					maxGradient = gradient;
				}
			}

		}
	}
}

//R2 Kernel code
__global__ void cudaR2(vs_t* viewshed, elev_t* elev, elev_t observer_elev, int minX, int maxX, int minY, int maxY, int observerX, int observerY, int ncols)
{

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	int width = (maxX - minX) + 1;
	int height = (maxY - minY) + 1;

	int totalCell = ((width + height) * 2) - 4;

	if (idx >= totalCell)
	{
		return;
	}

	int x, y;
	if (idx < width)
	{
		x = minX + idx;
		y = minY;
		//return;
	}
	else if (idx >= width && idx < width + height - 1)
	{
		x = maxX;
		y = minY + (idx + 1) - width;
		//return;
	}
	else if (idx >= width + height - 1 && idx < width + height + width - 2)
	{
		x = maxX - ((idx + 1) - (width + height - 1));
		y = maxY;
	}
	else if (idx >= width + height + width - 2 && idx < totalCell)
	{
		x = minX;
		y = maxY - ((idx + 1) - (width + height + width - 2));
	}


	int x1 = observerX;
	int y1 = observerY;
	int x2 = x;
	int y2 = y;

	int delta_x(x2 - x1);
	// if x1 == x2, then it does not matter what we set here
	signed char const ix((delta_x > 0) - (delta_x < 0));
	delta_x = std::abs(delta_x) << 1;

	int delta_y(y2 - y1);
	// if y1 == y2, then it does not matter what we set here
	signed char const iy((delta_y > 0) - (delta_y < 0));
	delta_y = std::abs(delta_y) << 1;


	float maxGradient = -10000;

	if (delta_x >= delta_y)
	{
		// error may go below zero
		int error(delta_y - (delta_x >> 1));

		while (x1 != x2)
		{
			if ((error >= 0) && (error || (ix > 0)))
			{
				error -= delta_x;
				y1 += iy;
			}
			// else do nothing

			error += delta_y;
			x1 += ix;


			int currentIndex = (x1 + y1 * ncols);
			int deltaY = y1 - observerY;
			int deltaX = x1 - observerX;
			float dist2 = deltaX * deltaX + deltaY * deltaY;

			double diff_elev = elev[currentIndex] - observer_elev;
			float gradient = (diff_elev * diff_elev) / dist2;
			if (diff_elev < 0) gradient *= -1;

			if (gradient > maxGradient)
			{
				maxGradient = gradient;
				viewshed[currentIndex] = 1;
			}
			else
			{
				viewshed[currentIndex] = 0;
			}
		}
	}
	else
	{
		// error may go below zero
		int error(delta_x - (delta_y >> 1));

		while (y1 != y2)
		{
			if ((error >= 0) && (error || (iy > 0)))
			{
				error -= delta_y;
				x1 += ix;
			}
			// else do nothing

			error += delta_x;
			y1 += iy;

			int currentIndex = (x1 + y1 * ncols);

			int deltaY = y1 - observerY;
			int deltaX = x1 - observerX;
			float dist2 = deltaX * deltaX + deltaY * deltaY;

			double diff_elev = elev[currentIndex] - observer_elev;
			float gradient = (diff_elev * diff_elev) / dist2;
			if (diff_elev < 0) gradient *= -1;

			if (gradient > maxGradient)
			{
				maxGradient = gradient;
				viewshed[currentIndex] = 1;
			}
			else
			{
				viewshed[currentIndex] = 0;
			}

		}
	}
}

//Wrapper for R3 kernel
void cudaR3Wrapper(vs_t * viewshed, elev_t * elev, elev_t observer_elev, int minX, int maxX, int minY, int maxY, int observerX, int observerY, int ncols)
{
	int width = (maxX - minX) + 1;
	int height = (maxY - minY) + 1;

	dim3 dimBlock(32, 32);

	dim3 dimGrid((int)std::ceil((float)((float)width / (float)dimBlock.x)), (int)std::ceil((float)((float)height / (float)dimBlock.y)));
	cudaR3 << <dimGrid, dimBlock >> > (viewshed, elev, observer_elev, minX, maxX, minY, maxY, observerX, observerY, ncols);
}

//Wrapper for R2 kernel
void cudaR2Wrapper(vs_t * viewshed, elev_t * elev, elev_t observer_elev, int minX, int maxX, int minY, int maxY, int observerX, int observerY, int ncols)
{
	int width = (maxX - minX) + 1;
	int height = (maxY - minY) + 1;


	int totalCell = ((width + height) * 2) - 4;


	int size = (int)std::ceil((float)((float)totalCell / (float)1024));
	cudaR2 << <size, 1024 >> > (viewshed, elev, observer_elev, minX, maxX, minY, maxY, observerX, observerY, ncols);
}

