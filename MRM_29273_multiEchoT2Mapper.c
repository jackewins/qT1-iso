

//Include standard libraries
#include <stdio.h>
#include <stdlib.h>
#include <math.h>


//Include my own libraries
#include "nifti1.h"
#include "nifti1_io.h"
#include "znzlib.h"

// prototype some functions for the simplex solver
void calculateSimplexT2Solution(float *generalSolution, float *multiEchoSignal, float *multiEchoTimes, float *initialGuess, int numEchoTimes);
float calculateFitResidual(float *guess, float *multiEchoSignal, float *multiEchoTimes, int numEchoTimes);
void calculateBestToWorst(int *bestToWorst, float **simplex, int numVertices);
float absoluteValue(float value);

int main(int argc, char** argv)
{
    
    //Load in the data
	char echoImageFiles[250][10000]; // ROI images
	char outputT2MapFile[10000]; // output T2 file name
	char outputMoMapFile[10000]; // output T2 file name
	
	char outputT2MapFile_lin[10000]; // output T2 file name
	char outputMoMapFile_lin[10000]; // output T2 file name
	
	float *echoTimes;
	float *echoSignals, *lnEchoSignals;
	float *t2Solution;
	float *initialGuess;
	float sumX, sumY, sumXY, sumXX;
	float a, d, denominator, e1, mo, slope, intercept, lnslope, t2;

	int numberOfEchoImages = atoi(argv[1]); // number of Images
	
	float backgroundNoise;
	int backgroundVoxels;
	
	int numberOfVoxelsInImage;
	int j,k;
	unsigned long long i,n;
	
	int sliceIndex, pointsPerSlice;
	
	printf("%d\n", numberOfEchoImages);
	
	echoTimes = (float *)calloc(numberOfEchoImages, sizeof(float));
	
	for (n=0; n<numberOfEchoImages; n++) {
		strcpy(echoImageFiles[n], argv[2+(n*2)]);
		echoTimes[n] = atof(argv[3+(n*2)]);
	}
	strcpy(outputT2MapFile, argv[4+(numberOfEchoImages-1)*2]);
	strcpy(outputMoMapFile, argv[5+(numberOfEchoImages-1)*2]);
	
	strcpy(outputT2MapFile_lin, argv[6+(numberOfEchoImages-1)*2]);
	strcpy(outputMoMapFile_lin, argv[7+(numberOfEchoImages-1)*2]);
	
	printf("%s\n", outputT2MapFile);
	printf("%s\n", outputMoMapFile);
	printf("%s\n", outputT2MapFile_lin);
	printf("%s\n", outputMoMapFile_lin);
	for (n=0; n<numberOfEchoImages; n++) printf("%s, %f\n", echoImageFiles[n], echoTimes[n]);
	
	// read in the first echo image to get the image information
	printf(". reading %s\n", echoImageFiles[0]);
	nifti_image* dataImage_img = nifti_image_read(echoImageFiles[0], 1);
	float* dataImage_img_data = (float *)dataImage_img->data;
	numberOfVoxelsInImage = dataImage_img->nx*dataImage_img->ny*dataImage_img->nz;
	printf("%d\n", numberOfVoxelsInImage);
	
	float scale = dataImage_img->scl_slope;
	float shift = dataImage_img->scl_inter;
	
	printf("%f %f\n", scale, shift);
	
	// set up an array to hold the raw image data, then read in each echo time image in turn
	float *rawECHOTimeData = (float *)calloc(numberOfEchoImages*numberOfVoxelsInImage, sizeof(float));
	for (i=0; i<numberOfVoxelsInImage; i++) rawECHOTimeData[i] = dataImage_img_data[i];
	
	for (n=1; n<numberOfEchoImages; n++) {
		printf(". reading %s\n", echoImageFiles[n]);
		nifti_image* dataImage_img = nifti_image_read(echoImageFiles[n], 1);
		float* dataImage_img_data = (float *)dataImage_img->data;
		for (i=0; i<numberOfVoxelsInImage; i++) rawECHOTimeData[(n*numberOfVoxelsInImage)+i] = dataImage_img_data[i];
		free(dataImage_img_data);
	}
	
	free(dataImage_img_data);
	
	// figure out the background noise
	backgroundNoise = 0.00;
	backgroundVoxels = 0;
	i = 100;
	while (backgroundVoxels < 50) {
		if (rawECHOTimeData[i] > 10) {
			backgroundNoise+=rawECHOTimeData[i];
			backgroundVoxels++;
		}
		i++;
	}
	backgroundNoise/=50.00;
	
	
	// now we index through the array and calculate the T2 values
	float *T2MapData = (float *)calloc(numberOfVoxelsInImage, sizeof(float));
	float *MoMapData = (float *)calloc(numberOfVoxelsInImage, sizeof(float));
	
	float *T2MapData_lin = (float *)calloc(numberOfVoxelsInImage, sizeof(float));
	float *MoMapData_lin = (float *)calloc(numberOfVoxelsInImage, sizeof(float));
	
	echoSignals = (float *)calloc(numberOfEchoImages, sizeof(float));
	lnEchoSignals = (float *)calloc(numberOfEchoImages, sizeof(float));
	t2Solution = (float *)calloc(2, sizeof(float));
	initialGuess = (float *)calloc(2, sizeof(float));
	
	// we're going to put in a slice counter here as well
	pointsPerSlice = dataImage_img->nx*dataImage_img->ny;
	sliceIndex = 1;
	printf(".   working on slice: 1\n");
	
	for (i=0; i<numberOfVoxelsInImage; i++) {
		
		// slice counter bit for anxiety reduction
		if (i >= sliceIndex*pointsPerSlice) {
			printf(".   working on slice: %d\n", sliceIndex+1);
			sliceIndex++;
		}
		
		if (rawECHOTimeData[i]>0.35*backgroundNoise) {
		
			for (n=0; n<numberOfEchoImages; n++) {
				echoSignals[n] = rawECHOTimeData[(n*numberOfVoxelsInImage)+i];
				lnEchoSignals[n] = log(rawECHOTimeData[(n*numberOfVoxelsInImage)+i]);
			}
		
			// calculate the straightforward linear solutions
			sumX = 0.00;
			sumY = 0.00;
			sumXY = 0.00;
			sumXX = 0.00;
			for (n=0; n<numberOfEchoImages; n++) {
				sumX += echoTimes[n];
				sumY += lnEchoSignals[n];
				sumXY += ( (echoTimes[n])*(lnEchoSignals[n]) );
				sumXX += ( (echoTimes[n])*(echoTimes[n]) );
			}
			
			d = (numberOfEchoImages*sumXX) - (sumX*sumX);
			a = (numberOfEchoImages*sumXY) - (sumX*sumY);
			
			if (a != 0.00 && d != 0.00) {
				slope = a/d;
				t2 = -1.00/slope;
				if (t2 > 2000) t2 = 2000;
				if (t2 < 0) t2 = 0.00;
				
				intercept = (sumY-slope*sumX)/numberOfEchoImages;
				if (intercept > 6.214) intercept = 6.214;
				mo = exp(intercept);
			}
			else {
				t2 = 0.00;
				intercept = 0.00;
				mo = 0.00;
			}
			
			T2MapData_lin[i] = t2;
			MoMapData_lin[i] = mo;
			
			// use this as the initial guess for the non-linear calcuation
			if (t2 != 0.00 && mo != 0.00) {
				initialGuess[0] = t2;
				initialGuess[1] = mo;
			}
			else{
				initialGuess[0] = 950.00;
				initialGuess[1] = 100.00;
			}
			
			
			calculateSimplexT2Solution(t2Solution, echoSignals, echoTimes, initialGuess, numberOfEchoImages);
			
			T2MapData[i] = (t2Solution[0]);
			MoMapData[i] = (t2Solution[1]);
			
			
		}
		else {
			T2MapData[i] = 0.00;
			MoMapData[i] = 0.00;
			
			T2MapData_lin[i] = 0.00;
			MoMapData_lin[i] = 0.00;
		}
			
		
	}
	
	// write out the resulting calculated maps (non-linear fit)
	nifti_image* t2Output_img = nifti_copy_nim_info(dataImage_img);		
	t2Output_img->data = T2MapData;
	nifti_set_filenames(t2Output_img, outputT2MapFile, 1, 1); //set the output name to the image object
	nifti_image_write(t2Output_img); //write the image.
	
	nifti_image* moOutput_img = nifti_copy_nim_info(dataImage_img);		
	moOutput_img->data = MoMapData;
	nifti_set_filenames(moOutput_img, outputMoMapFile, 1, 1); //set the output name to the image object
	nifti_image_write(moOutput_img); //write the image.
	
	// write out the resulting calculated maps (linear fit)
	nifti_image* t2OutputLin_img = nifti_copy_nim_info(dataImage_img);		
	t2OutputLin_img->data = T2MapData_lin;
	nifti_set_filenames(t2OutputLin_img, outputT2MapFile_lin, 1, 1); //set the output name to the image object
	nifti_image_write(t2OutputLin_img); //write the image.
	
	nifti_image* moOutputLin_img = nifti_copy_nim_info(dataImage_img);		
	moOutputLin_img->data = MoMapData_lin;
	nifti_set_filenames(moOutputLin_img, outputMoMapFile_lin, 1, 1); //set the output name to the image object
	nifti_image_write(moOutputLin_img); //write the image.
	
	
	// feee up memory
	free(rawECHOTimeData);
	free(T2MapData);
	free(MoMapData);
	free(T2MapData_lin);
	free(MoMapData_lin);
	free(lnEchoSignals);
	free(t2Solution);
	free(echoSignals);
	
	// and, exit
	return 0;
}


void calculateSimplexT2Solution(float *generalSolution, float *multiEchoSignal, float *multiEchoTimes, float *initialGuess, int numEchoTimes)
{
	
	float *simplexLine, **simplex, *simplexCentre, residual;
	float *reflection, *expansion, *shrink, *contraction;
	int *bestToWorst;
	float RHO, CHI, PSI, SIGMA, maxError, usual_delta, zero_term_delta, rtol;
	float t2PercentDifference;
	int best, worst, secondWorst;
	int continueSimplex;
	
	int NMAX, NROUND, numParams, numVertices, iterations;
	int i, j, k;
	

	// strike up the band, initialize the simplex
	RHO = 1.00;
	CHI = 2.00;
	PSI = 0.50;
	SIGMA = 0.50;
	usual_delta = 0.05;
	zero_term_delta = .00025;
	
	maxError = 0.001;
	NMAX = 15000;
	
	numParams = 2;  // Mo and T2
	numVertices = numParams + 1;
	
	
	bestToWorst = (int *)calloc(3, sizeof(int));
	
	simplexLine = (float *)calloc(numParams, sizeof(float));
	simplexCentre = (float *)calloc(numParams, sizeof(float));
	simplex = (float **)calloc(numVertices, sizeof(float *));
	for (i=0; i<numVertices; i++) simplex[i] = (float *)calloc(numVertices, sizeof(float));
	
	reflection = (float *)calloc(numVertices, sizeof(float));
	expansion = (float *)calloc(numVertices, sizeof(float));
	shrink = (float *)calloc(numVertices, sizeof(float));
	contraction = (float *)calloc(numVertices, sizeof(float));
	
	continueSimplex = 1;
	NROUND = 0;
	
	while (continueSimplex == 1 && NROUND<250) {
		NROUND++;
		
		for (i=0; i<numParams; i++) simplexLine[i] = initialGuess[i];
		residual = calculateFitResidual(simplexLine, multiEchoSignal, multiEchoTimes, numEchoTimes);
		
		for (i=0; i<numParams; i++) simplex[0][i] = simplexLine[i];
		simplex[0][numParams] = residual;
		
		for (j=1; j<numVertices; j++) {
			for (i=0; i<numParams; i++) simplexLine[i] = initialGuess[i];
			
			if (simplexLine[j-1] != 0.00) simplexLine[j-1] = absoluteValue( (1.00+usual_delta)*simplexLine[j-1] );
			else simplexLine[j-1] = zero_term_delta;
			residual = calculateFitResidual(simplexLine, multiEchoSignal, multiEchoTimes, numEchoTimes);
			
			
			for (i=0; i<numParams; i++) simplex[j][i] = simplexLine[i];
			simplex[j][numParams] = residual;
		}
		
		// now calculate the best, worst and second worst vertices of the simplex
		calculateBestToWorst(bestToWorst, simplex, numVertices);
		best = bestToWorst[0];
		worst = bestToWorst[2];
		secondWorst = bestToWorst[1];
		
		rtol = absoluteValue( simplex[best][numParams] - simplex[worst][numParams] );
		
		iterations = 0;
		
		while (rtol > maxError && iterations < NMAX) {
			iterations++;
			
			// calculate the centroid of the current simplex, ignoring the worst point
			for (i=0; i<numParams; i++) simplexCentre[i] = 0.00;
			for (j=0; j<numVertices; j++) {
				if (j != worst) {
					for (i=0; i<numParams; i++) simplexCentre[i] += simplex[j][i];
				}
			}
			for (i=0; i<numParams; i++) simplexCentre[i] /= numParams;
			
			// reflect the simplex through the face of the high point
			for (i=0; i<numParams; i++) reflection[i] = (1.00+RHO)*simplexCentre[i] - RHO*simplex[worst][i];
			reflection[numParams] = calculateFitResidual(reflection, multiEchoSignal, multiEchoTimes, numEchoTimes);
			
			// compare the reflection vertex with the best vertex in the exisiting simplex
			if (reflection[numParams] < simplex[best][numParams]) {
				// if the reflection was better, try an expansion in this direction and see how that is
				for (i=0; i<numParams; i++) expansion[i] = ( (1.00+RHO*CHI)*simplexCentre[i] - RHO*CHI*simplex[worst][i] );
				expansion[numParams] = calculateFitResidual(expansion, multiEchoSignal, multiEchoTimes, numEchoTimes);
				
				
				if (expansion[numParams] < reflection[numParams]) {
					for (i=0; i<numVertices; i++) simplex[worst][i] = expansion[i];
				}
				else {
					for (i=0; i<numVertices; i++) simplex[worst][i] = reflection[i];
				}
			} 
			else {
				if (reflection[numParams] < simplex[secondWorst][numParams]) {
					for (i=0; i<numVertices; i++) simplex[worst][i] = reflection[i];
				}
				else {
					if (reflection[numParams] < simplex[worst][numParams]) {
						// perform an outside contraction
						for (i=0; i<numParams; i++) contraction[i] = absoluteValue( (1.00+PSI*RHO)*simplexCentre[i] - RHO*PSI*simplex[worst][i] );
						contraction[numParams] = calculateFitResidual(contraction, multiEchoSignal, multiEchoTimes, numEchoTimes);
						
						if (contraction[numParams] <= reflection[numParams]) {
							for (i=0; i<numVertices; i++) simplex[worst][i] = contraction[i];
						}
						else {
							// perform a shrink of all vertices except the best
							for (j=0; j<numVertices; j++) {
								if (j != best) {
									for (i=0; i<numParams; i++) shrink[i] = absoluteValue( simplex[best][i] + SIGMA*(simplex[j][i]-simplex[best][i]) );
									shrink[numParams] = calculateFitResidual(shrink, multiEchoSignal, multiEchoTimes, numEchoTimes);
									for (i=0; i<numVertices; i++) simplex[j][i] = shrink[i];
								}
							}
						}
					}
					else {
						// perform an inside contraction
						for (i=0; i<numParams; i++) contraction[i] = absoluteValue( (1.00-PSI)*simplexCentre[i] + PSI*simplex[worst][i] );
						contraction[numParams] = calculateFitResidual(contraction, multiEchoSignal, multiEchoTimes, numEchoTimes);
						
						if (contraction[numParams] < simplex[worst][numParams]) {
							for (i=0; i<numVertices; i++) simplex[worst][i] = contraction[i];
						}
						else {
							// perform a shrink of all vertices except the best
							for (j=0; j<numVertices; j++) {
								if (j != best) {
									for (i=0; i<numParams; i++) shrink[i] = absoluteValue( simplex[best][i] + SIGMA*(simplex[j][i]-simplex[best][i]) );
									shrink[numParams] = calculateFitResidual(shrink, multiEchoSignal, multiEchoTimes, numEchoTimes);
									for (i=0; i<numVertices; i++) simplex[j][i] = shrink[i];
								}
							}
						}
					}
				}
				
			}
			
			calculateBestToWorst(bestToWorst, simplex, numVertices);
			best = bestToWorst[0];
			worst = bestToWorst[2];
			secondWorst = bestToWorst[1];
			
			rtol = absoluteValue( simplex[best][numParams] - simplex[worst][numParams] );
		}
		
		// determine if we need to keep going
		if (NROUND >= 2) t2PercentDifference = absoluteValue( 100.00*( (simplex[best][0]-initialGuess[0])/simplex[best][0]) );
		if (t2PercentDifference < 1.00) continueSimplex = 0;
		
		for (i=0; i<numParams; i++) initialGuess[i] = simplex[best][i];
		
	}
	
	generalSolution[0] = initialGuess[0];
	generalSolution[1] = initialGuess[1];
	
	if (generalSolution[0] > 2000) generalSolution[0] = 2000;
	if (generalSolution[1] > 500) generalSolution[1] = 500;
	
	
	// free up all the allocated memory
	free(simplexLine);
	free(simplexCentre);
	
	for (i=0; i<numVertices; i++) free(simplex[i]);
	free(simplex);
	
	free(reflection);
	free(expansion);
	free(shrink);
	free(contraction);
	free(bestToWorst);
	
}

float calculateFitResidual(float *guess, float *multiEchoSignal, float *multiEchoTimes, int numEchoTimes) 
{
	float residual;
	float theorteticalSignal;
	float weight;
	int i;
	
	residual = 0.00;
	for (i=0; i<numEchoTimes; i++) {
		theorteticalSignal = 0.00;
		weight = 1.00; //exp(-guess[0]/multiEchoTimes[i]);
		theorteticalSignal = guess[1]*exp(-multiEchoTimes[i]/guess[0]);
		residual += sqrt( weight * pow( (multiEchoSignal[i] - theorteticalSignal), 2.00) );
	}
	
	return residual;
}

float absoluteValue(float value) {
	float abs;
	
	if (value < 0) abs = -1.00*value;
	else abs = value;
	
	return abs;
	
}


void calculateBestToWorst(int *bestToWorst, float **simplex, int numVertices)
{
	int i, j, best, worst, secondWorst;
	
	
	best = 0;
	secondWorst = 0;
	worst = 0;
	
	for (i=0; i<numVertices; i++) {
		if (simplex[i][numVertices-1] < simplex[best][numVertices-1]) best = i;
		if (simplex[i][numVertices-1] > simplex[worst][numVertices-1]) worst = i;
	}
	
	secondWorst = best;
	for (i=0; i<numVertices; i++) {
		if (i != worst) {
			if (simplex[i][numVertices-1] > simplex[secondWorst][numVertices-1]) secondWorst = i;
		}
	}
	
	bestToWorst[0] = best;
	bestToWorst[1] = secondWorst;
	bestToWorst[2] = worst;
	
}
