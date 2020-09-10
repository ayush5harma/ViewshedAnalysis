/*This file contains the host code for kernel launch along with independent CPU implementation of R3 and R2 and testing performance to compare the runtime result for each*/

#include <iostream>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <algorithm>
#include <vector>
#include <map>
#include <time.h>

/* for CUDA Runtime */
#include "npp.h"
#include "kernel.h"
#include "helper_cuda.h"

/* for writing viewshed to picture */
#include <windows.h>
#include "EasyBMP.h"
#include "EasyBMP_DataStructures.h"
#include "EasyBMP_BMP.h"


using namespace std;

double PCFreq = 0.0;
__int64 CounterStart = 0;

int radiusGlob = 0;
int currenIteration = 0;
int iterationCount = 1;
ofstream resultFile;
#define MEMORYMETRICS

unsigned long int SIZECONV = 1024 * 1024;

/**
* Starts the performance counter
**/
void startCounter()
{
	LARGE_INTEGER li;
	if (!QueryPerformanceFrequency(&li))
		std::cout << "QueryPerformanceFrequency failed!\n";

	PCFreq = double(li.QuadPart) / 1000.0;

	QueryPerformanceCounter(&li);
	CounterStart = li.QuadPart;
}

/**
* Stops the performance counter and returns the result
**/
double getCounter()
{
	LARGE_INTEGER li;
	QueryPerformanceCounter(&li);
	return double(li.QuadPart - CounterStart) / PCFreq;
}


// Writes GPU computed viewshed output to image
void writeViewshedToPicture(vs_t * *viewshed, int ncols, int nrows, char* fileName)
{
	RGBApixel visibleColor;
	visibleColor.Red = 255;
	visibleColor.Green = 255;
	visibleColor.Blue = 255;     //white

	RGBApixel notVisibleColor;
	notVisibleColor.Red = 0;
	notVisibleColor.Green = 0;
	notVisibleColor.Blue = 0;    //black
	BMP img;
	img.SetSize(ncols, nrows);
	/** write output **/
	for (int i = 0; i < nrows; i++)
	{
		for (int j = 0; j < ncols; j++)
		{
			if (viewshed[i][j] == 1)
			{
				img.SetPixel(j, i, visibleColor);
			}
			else
			{
				img.SetPixel(j, i, notVisibleColor);
			}
		}
	}
	img.WriteToFile(fileName);
}


void writeViewshedToCsv(vs_t * *viewshed, int ncols, int nrows, char* fileName)
{
	std::ofstream myfile;
	myfile.open(fileName);
	for (int i = 0; i < nrows; i++)
	{

		for (int j = 0; j < ncols; j++)
		{
			if (viewshed[i][j] == 1)
			{
				myfile << "1,";
			}
			else
			{
				myfile << "0,";
			}
		}
		myfile << "\n";
	}
	myfile.close();
}





// Writes CPU computed viewshed ouput to image
void writeViewshedToPicture(vs_t * viewshed, int ncols, int nrows, char* fileName)
{
	RGBApixel visibleColor;
	visibleColor.Red = 255;
	visibleColor.Green = 255;
	visibleColor.Blue = 255;  //white

	RGBApixel notVisibleColor;
	notVisibleColor.Red = 0;
	notVisibleColor.Green = 0;
	notVisibleColor.Blue = 0;  //black
	BMP img;
	img.SetSize(ncols, nrows);
	for (int i = 0; i < nrows; i++)
	{
		for (int j = 0; j < ncols; j++)
		{
			int index = i * ncols + j;
			if (viewshed[index] == 1)
			{
				img.SetPixel(j, i, visibleColor);
			}
			else
			{
				img.SetPixel(j, i, notVisibleColor);
			}
		}
	}
	img.WriteToFile(fileName);
}
void writeViewshedToCsv(vs_t * viewshed, int ncols, int nrows, char* fileName) {
	std::ofstream myfile;
	myfile.open(fileName);

	for (int i = 0; i < nrows; i++)
	{
		for (int j = 0; j < ncols; j++)
		{
			int index = i * ncols + j;
			if (viewshed[index] == 1)
			{
				myfile << "1,";
			}
			else
			{
				myfile << "0,";
			}
		}
		myfile << "\n";
	}
	myfile.close();

}




double CPUR3(int nrows, int ncols, int radius, int observerX, int observerY, int observer_ht, elev_t * *elev)
{
	vs_t** viewshed;
	elev_t observer_elev;

	observer_elev = elev[observerY][observerX] + observer_ht;

	/** alloc viewshed matrix **/
	viewshed = new vs_t * [nrows];
	for (int i = 0; i < nrows; i++)
		viewshed[i] = new vs_t[ncols];

	//CALCULATE AREA OF INTEREST
	int minY = max(0, observerY - radius);
	int maxY = min(nrows - 1, observerY + radius);
	int minX = max(0, observerX - radius);
	int maxX = min(ncols - 1, observerX + radius);


	int width = (maxX - minX) + 1;
	int height = (maxY - minY) + 1;


#ifdef MEMORYMETRICS

	if (currenIteration == 1)
	{
		long elevMemSize = sizeof(elev_t) * ncols * nrows / SIZECONV;
		long viewshedMemSize = ncols * nrows * sizeof(vs_t) / SIZECONV;

		//-----------------Memory Information on Console---------------------//

		std::cout << "CPU R3 Memory Consumption for " << nrows << "*" << ncols << std::endl;
		std::cout << "Elevation Data  Memory Consumption(@CPU): " << elevMemSize << std::endl;
		std::cout << "Viewshed Memory Consumption(@CPU): " << viewshedMemSize << std::endl;
		std::cout << "Total Memory Consumption(@CPU): " << viewshedMemSize + elevMemSize << std::endl;


		resultFile << "CPU R3 Memory Consumption for " << nrows << "*" << ncols << std::endl;
		resultFile << "Elevation Data  Memory Consumption(@CPU): " << elevMemSize << std::endl;
		resultFile << "Viewshed Memory Consumption(@CPU): " << viewshedMemSize << std::endl;
		std::cout << "Total Memory Consumption(@CPU): " << viewshedMemSize + elevMemSize << std::endl;
	}

#endif

	startCounter();


	for (int y = minY; y <= maxY; y++)
	{
		for (int x = minX; x <= maxX; x++)
		{
			if (x == observerX && y == observerY)
			{
				viewshed[y][x] = 1;
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

					double diff_elev = elev[y1][x1] - observer_elev;
					float gradient = (diff_elev * diff_elev) / dist2;
					if (diff_elev < 0) gradient *= -1;

					if (y1 == y && x1 == x)
					{
						if (gradient > maxGradient)
						{
							viewshed[y][x] = 1;
						}
						else
						{
							viewshed[y][x] = 0;
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

					double diff_elev = elev[y1][x1] - observer_elev;
					float gradient = (diff_elev * diff_elev) / dist2;
					if (diff_elev < 0) gradient *= -1;
					if (y1 == y && x1 == x)
					{
						if (gradient > maxGradient)
						{
							viewshed[y][x] = 1;
						}
						else
						{
							viewshed[y][x] = 0;
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
	}

	//Get the result counter
	double result = getCounter();

	if (currenIteration == iterationCount)
	{
		char buffer[100];
		sprintf(buffer, "CPU-R3-%d-%d.bmp", observerX, observerY);

		writeViewshedToPicture(viewshed, ncols, nrows, buffer);

	}
	if (currenIteration == iterationCount)
	{
		char buffer[100];
		sprintf(buffer, "CPU-R3-%d-%d.csv", observerX, observerY);
		writeViewshedToCsv(viewshed, ncols, nrows, buffer);

	}

	/** delete matrices */
	for (int i = 0; i < nrows; i++) {
		//delete[] elev[i];
		delete[] viewshed[i];
	}
	//delete[] elev;
	delete[] viewshed;



	//Return counter
	return result;
}

void iterateLine(int x, int y, int observerX, int observerY, elev_t * *elev, vs_t * *viewshed, elev_t observer_elev)
{
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

			double diff_elev = elev[y1][x1] - observer_elev;
			float gradient = (diff_elev * diff_elev) / dist2;
			if (diff_elev < 0) gradient *= -1;

			if (gradient > maxGradient)
			{
				maxGradient = gradient;
				viewshed[y1][x1] = 1;
			}
			else
			{
				viewshed[y1][x1] = 0;
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

			double diff_elev = elev[y1][x1] - observer_elev;
			float gradient = (diff_elev * diff_elev) / dist2;
			if (diff_elev < 0) gradient *= -1;

			if (gradient > maxGradient)
			{
				maxGradient = gradient;
				viewshed[y1][x1] = 1;
			}
			else
			{
				viewshed[y1][x1] = 0;
			}

		}
	}
}

double CPUR2(int nrows, int ncols, int radius, int observerX, int observerY, int observer_ht, elev_t * *elev)
{

	vs_t** viewshed;
	elev_t observer_elev;

	observer_elev = elev[observerY][observerX] + observer_ht;

	//CALCULATE AREA OF INTEREST
	int minY = max(0, observerY - radius);
	int maxY = min(nrows - 1, observerY + radius);
	int minX = max(0, observerX - radius);
	int maxX = min(ncols - 1, observerX + radius);


	int width = (maxX - minX) + 1;
	int height = (maxY - minY) + 1;

	int x = minX;
	int y = minY;

#ifdef MEMORYMETRICS

	if (currenIteration == 1)
	{
		//-----------------Memory Information on Console---------------------//

		long elevMemSize = sizeof(elev_t) * ncols * nrows / SIZECONV;
		long viewshedMemSize = ncols * nrows * sizeof(vs_t) / SIZECONV;

		std::cout << "CPU R2 Memory Consumption for " << nrows << "*" << ncols << std::endl;
		std::cout << "Elevation Data  Memory Consumption(@CPU): " << elevMemSize << std::endl;
		std::cout << "Viewshed Memory Consumption(@CPU): " << viewshedMemSize << std::endl;
		std::cout << "Total Memory Consumption(@CPU): " << viewshedMemSize + elevMemSize << std::endl;


		resultFile << "CPU R2 Memory Consumption for " << nrows << "*" << ncols << std::endl;
		resultFile << "Elevation Data  Memory Consumption(@CPU): " << elevMemSize << std::endl;
		resultFile << "Viewshed Memory Consumption(@CPU): " << viewshedMemSize << std::endl;
		resultFile << "Total Memory Consumption(@CPU): " << viewshedMemSize + elevMemSize << std::endl;
	}

#endif

	startCounter();

	/** alloc viewshed matrix **/
	viewshed = new vs_t * [nrows];
	for (int i = 0; i < nrows; i++)
		viewshed[i] = new vs_t[ncols];

	viewshed[observerY][observerX] = 1;

	for (int y = maxY, x = maxX; y >= minY; y--) { // right border (going up)
		iterateLine(x, y, observerX, observerY, elev, viewshed, observer_elev);
	}
	for (int x = maxX, y = minY; x >= minX; x--) { // top border (going left)
		iterateLine(x, y, observerX, observerY, elev, viewshed, observer_elev);
	}
	for (int y = minY, x = minX; y <= maxY; y++) { // left border (going down)
		iterateLine(x, y, observerX, observerY, elev, viewshed, observer_elev);
	}
	for (int x = minX, y = maxY; x <= maxX; x++) { // bottom border (going right)
		iterateLine(x, y, observerX, observerY, elev, viewshed, observer_elev);
	}

	//Get the result counter
	double result = getCounter();

	if (currenIteration == iterationCount)
	{
		char buffer[100];
		sprintf(buffer, "CPU-R2-%d-%d.bmp", observerX, observerY);
		writeViewshedToPicture(viewshed, ncols, nrows, buffer);
	}
	if (currenIteration == iterationCount)
	{
		char buffer[100];
		sprintf(buffer, "CPU-R2-%d-%d.csv", observerX, observerY);
		writeViewshedToCsv(viewshed, ncols, nrows, buffer);

	}

	/** delete matrices */
	for (int i = 0; i < nrows; i++) {
		//delete[] elev[i];
		delete[] viewshed[i];
	}
	//delete[] elev;
	delete[] viewshed;

	//Return counter
	return result;
}



// Host code for kernel.cu
double GPUR3(int nrows, int ncols, int radius, int observerX, int observerY, int observer_ht, elev_t * elev)
{
	elev_t observer_elev;

	observer_elev = elev[observerY * ncols + observerX] + observer_ht;

	//CALCULATE AREA OF INTEREST
	int minY = max(0, observerY - radius);
	int maxY = min(nrows - 1, observerY + radius);
	int minX = max(0, observerX - radius);
	int maxX = min(ncols - 1, observerX + radius);

#ifdef MEMORYMETRICS

	if (currenIteration == 1)
	{

		long elevMemSize = sizeof(elev_t) * ncols * nrows / SIZECONV;
		long viewshedMemSize = ncols * nrows * sizeof(vs_t) / SIZECONV;

		std::cout << "GPU R3 Memory Consumption for " << nrows << "*" << ncols << std::endl;
		std::cout << "Elevation Data  Memory Consumption(@CPU): " << elevMemSize << std::endl;
		std::cout << "Viewshed Memory Consumption(@CPU): " << viewshedMemSize << std::endl;
		std::cout << "Total Memory Consumption(@CPU): " << viewshedMemSize + elevMemSize << std::endl;
		std::cout << "Elevation Data  Memory Consumption(@GPU): " << elevMemSize << std::endl;
		std::cout << "Viewshed Memory Consumption(@GPU): " << viewshedMemSize << std::endl;
		std::cout << "Total Memory Consumption(@GPU): " << viewshedMemSize + elevMemSize << std::endl;


		resultFile << "GPU R3 Memory Consumption for " << nrows << "*" << ncols << std::endl;
		resultFile << "Elevation Data  Memory Consumption(@CPU): " << elevMemSize << std::endl;
		resultFile << "Viewshed Memory Consumption(@CPU): " << viewshedMemSize << std::endl;
		resultFile << "Total Memory Consumption(@CPU): " << viewshedMemSize + elevMemSize << std::endl;
		resultFile << "Elevation Data  Memory Consumption(@GPU): " << elevMemSize << std::endl;
		resultFile << "Viewshed Memory Consumption(@GPU): " << viewshedMemSize << std::endl;
		resultFile << "Total Memory Consumption(@GPU): " << viewshedMemSize + elevMemSize << std::endl;
	}

#endif

	startCounter();


	vs_t* viewshed = new vs_t[nrows * ncols];
	vs_t* d_viewshed;

	checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_viewshed), sizeof(vs_t) * nrows * ncols));
	elev_t* d_elev;
	checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_elev), sizeof(elev_t) * nrows * ncols));
	checkCudaErrors(cudaMemcpy(d_elev, elev, sizeof(elev_t) * nrows * ncols, cudaMemcpyHostToDevice));
	cudaR3Wrapper(d_viewshed, d_elev, observer_elev, minX, maxX, minY, maxY, observerX, observerY, ncols);
	checkCudaErrors(cudaMemcpy(viewshed, d_viewshed, sizeof(vs_t) * nrows * ncols, cudaMemcpyDeviceToHost));

	//Get the result counter
	double result = getCounter();

	if (currenIteration == iterationCount)
	{
		char buffer[100];
		sprintf(buffer, "GPU-R3-%d-%d.bmp", observerX, observerY);
		writeViewshedToPicture(viewshed, ncols, nrows, buffer);

	}
	if (currenIteration == iterationCount)
	{
		char buffer[100];
		sprintf(buffer, "GPU-R3-%d-%d.csv", observerX, observerY);
		writeViewshedToCsv(viewshed, ncols, nrows, buffer);
	}

	delete[] viewshed;
	checkCudaErrors(cudaFree(d_elev));
	checkCudaErrors(cudaFree(d_viewshed));
	//Return counter
	return result;
}

double GPUR2(int nrows, int ncols, int radius, int observerX, int observerY, int observer_ht, elev_t * elev)
{
	elev_t observer_elev;
	observer_elev = elev[observerY * ncols + observerX] + observer_ht;

	//CALCULATE AREA OF INTEREST
	int minY = max(0, observerY - radius);
	int maxY = min(nrows - 1, observerY + radius);
	int minX = max(0, observerX - radius);
	int maxX = min(ncols - 1, observerX + radius);

#ifdef MEMORYMETRICS
	if (currenIteration == 1)
	{

		long elevMemSize = sizeof(elev_t) * ncols * nrows / SIZECONV;
		long viewshedMemSize = ncols * nrows * sizeof(vs_t) / SIZECONV;

		std::cout << "GPU R3 Memory Consumption for " << nrows << "*" << ncols << std::endl;
		std::cout << "Elevation Data  Memory Consumption(@CPU): " << elevMemSize << std::endl;
		std::cout << "Viewshed Memory Consumption(@CPU): " << viewshedMemSize << std::endl;
		std::cout << "Total Memory Consumption(@CPU): " << viewshedMemSize + elevMemSize << std::endl;
		std::cout << "Elevation Data  Memory Consumption(@GPU): " << elevMemSize << std::endl;
		std::cout << "Viewshed Memory Consumption(@GPU): " << viewshedMemSize << std::endl;
		std::cout << "Total Memory Consumption(@GPU): " << viewshedMemSize + elevMemSize << std::endl;


		resultFile << "GPU R3 Memory Consumption for " << nrows << "*" << ncols << std::endl;
		resultFile << "Elevation Data  Memory Consumption(@CPU): " << elevMemSize << std::endl;
		resultFile << "Viewshed Memory Consumption(@CPU): " << viewshedMemSize << std::endl;
		resultFile << "Total Memory Consumption(@CPU): " << viewshedMemSize + elevMemSize << std::endl;
		resultFile << "Elevation Data  Memory Consumption(@GPU): " << elevMemSize << std::endl;
		resultFile << "Viewshed Memory Consumption(@GPU): " << viewshedMemSize << std::endl;
		resultFile << "Total Memory Consumption(@GPU): " << viewshedMemSize + elevMemSize << std::endl;
	}
#endif

	startCounter();
	vs_t* viewshed = new vs_t[nrows * ncols];
	vs_t* d_viewshed;
	checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_viewshed), sizeof(vs_t) * nrows * ncols));
	elev_t* d_elev;
	checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_elev), sizeof(elev_t) * nrows * ncols));
	checkCudaErrors(cudaMemcpy(d_elev, elev, sizeof(elev_t) * nrows * ncols, cudaMemcpyHostToDevice));
	cudaR2Wrapper(d_viewshed, d_elev, observer_elev, minX, maxX, minY, maxY, observerX, observerY, ncols);
	checkCudaErrors(cudaMemcpy(viewshed, d_viewshed, sizeof(vs_t) * nrows * ncols, cudaMemcpyDeviceToHost));

	//Get the result counter
	double result = getCounter();

	if (currenIteration == iterationCount)
	{
		char buffer[100];
		sprintf(buffer, "GPU-R2-%d-%d.bmp", observerX, observerY);
		writeViewshedToPicture(viewshed, ncols, nrows, buffer);
	}
	if (currenIteration == iterationCount)
	{
		char buffer[100];
		sprintf(buffer, "GPU-R2-%d-%d.csv", observerX, observerY);
		writeViewshedToCsv(viewshed, ncols, nrows, buffer);

	}

	delete[] viewshed;
	checkCudaErrors(cudaFree(d_elev));
	checkCudaErrors(cudaFree(d_viewshed));
	//Return counter
	return result;
}

// Main function.
int
main(int argc, char** argv)
{

	//Seed random
	srand(time(NULL));


	srand(time(NULL));

	int observer_ht = 15; // Observer height
	int nrowCounts;    // X values in loaded raster image
	int nColumnCounts;
	int radius;
	char elevPaths[1][200];
	int n = 10;
	double result;
	cout << "\n";
	cout << "Enter Elevation Path (i.e the path to the raster image) :- ";
	cin  >> elevPaths[0];

	cout << "Enter DEM Dimensions (say 3601x3601 then input should be 3601 3601):- ";
	cin >> nrowCounts >> nColumnCounts;
	cout << "Enter Sensor Radius(in Pixels):- ";
	cin >> radius;
	
	int observerXs[1] = { 0 };     // X coordinates of observer for respective raster image
	int observerYs[1] = { 0};
	int k = 0;
		cout << "Enter Cordinate for Sensor:- ";
		cin >> observerXs[0] >> observerYs[0];
		cout << "Select Algorithm \n 1.CPU R2 \n 2.GPU R2\n 3.CPU R3\n 4.GPU R3\n Enter Your Choice :-";
		cin >> k;

	



	double resultCPUR2 = 0;
	double resultGPUR2 = 0;

	double resultCPUR3 = 0;
	double resultGPUR3 = 0;

	resultFile.open("results.txt");

	for (int i = 0; i < 1; i++)
	{
		radiusGlob = radius;
		elev_t** elev;
		//Read elevation
		FILE* f = fopen(elevPaths[0], "rb");
		elev = new elev_t * [nrowCounts];
		for (int k = 0; k < nrowCounts; k++) {
			elev[k] = new elev_t[nColumnCounts];
			fread(reinterpret_cast<char*>(elev[k]), sizeof(elev_t), nColumnCounts, f);
		}
		fclose(f);


		elev_t* elev1D = new elev_t[nrowCounts * nColumnCounts];
		//Read elevation
		f = fopen(elevPaths[0], "rb");
		int readCount = 0;
		for (int k = 0; k < nrowCounts; k++) {

			fread(reinterpret_cast<char*>(elev1D + readCount), sizeof(elev_t), nColumnCounts, f);
			readCount += nColumnCounts;
		}

		fclose(f);

		resultCPUR2 = 0;
		resultGPUR2 = 0;

		resultCPUR3 = 0;
		resultGPUR3 = 0;

		event_t* d_events;


		checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_events), sizeof(event_t) * 3601 * 3601 * 3));


		//std::getchar();   //uncomment to halt after each image is processed

		for (int j = 0; j < iterationCount; j++)
		{

			currenIteration = j + 1;
			double result;

			switch (k) {
			case 1:
				std::cout << "Running test for CPU R2 with Raster Image " << i << std::endl;;
				resultFile << "Running test for CPU R2 with Raster Image " << i << std::endl;;
				result = CPUR2(nrowCounts, nColumnCounts, radius, observerXs[i], observerYs[i], observer_ht, elev);
				std::cout << "CPU R2 test time is " << result << std::endl << std::endl << std::endl;
				resultFile << "CPU R2 test time is " << result << std::endl << std::endl << std::endl;
				resultCPUR2 += result;
				break;
			case 2:
				std::cout << "Running test for GPU R2 with Raster Image " << i << std::endl;;
				resultFile << "Running test for GPU R2 with Raster Image " << i << std::endl;;
				result = GPUR2(nrowCounts, nColumnCounts, radius, observerXs[i], observerYs[i], observer_ht, elev1D);
				std::cout << "GPU R2 test time is " << result << std::endl << std::endl << std::endl;
				resultFile << "GPU R2 test time is " << result << std::endl << std::endl << std::endl;
				resultGPUR2 += result;
				break;
			case 3:
				std::cout << "Running test for CPU R3 with Raster Image " << i << std::endl;;
				resultFile << "Running test for CPU R3 with Raster Image " << i << std::endl;;
				result = CPUR3(nrowCounts, nColumnCounts, radius, observerXs[i], observerYs[i], observer_ht, elev);
				std::cout << "CPU R3 test time is " << result << std::endl << std::endl << std::endl;
				resultFile << "CPU R3 test time is " << result << std::endl << std::endl << std::endl;
				resultCPUR3 += result;
				break;
			case 4:
				std::cout << "Running test for GPU R3 with Rater Image " << i << std::endl;;
				resultFile << "Running test for GPU R3 with Raster Image " << i << std::endl;;
				result = GPUR3(nrowCounts, nColumnCounts, radius, observerXs[i], observerYs[i], observer_ht, elev1D);
				std::cout << "GPU R3 test time is " << result << std::endl << std::endl << std::endl;
				resultFile << "GPU R3 test time is " << result << std::endl << std::endl << std::endl;
				resultGPUR3 += result;
				break;




				

			}
		
			




		}

		delete elev1D;

		for (int k = 0; k < nrowCounts; k++) {
			delete[] elev[k];
		}
		delete[] elev;

	}



	resultFile.close();
	std::cout << "Finished" << endl;
	std::getchar();
	cudaDeviceReset();
	return 0;
}

// Disable reporting warnings on functions that were marked with deprecated.
#pragma warning( disable : 4996 )