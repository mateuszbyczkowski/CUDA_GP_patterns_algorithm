#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <stdio.h>

#include "thrust/host_vector.h"
#include "thrust/device_vector.h"
#include "thrust/scan.h"
#include "thrust/scatter.h"
#include "thrust/sort.h"
#include "thrust/functional.h"
#include "thrust/reduce.h"
#include "thrust/copy.h"
#include "thrust/remove.h"
#include "thrust/device_ptr.h"
#include <thrust/execution_policy.h>

#include <math.h>
#include "device_atomic_functions.h"
#include <set>

using namespace std;

__host__ __device__ long getV(int up, int down) {
	long result = 1;
	for (int i = up - down + 1; i <= up; i++) {
		result *= i;
	}
	return result;
}

__global__ void func(int* output, int* sequence, int* pattern, int* uniqueSequence,
					int numberOfDifferentInSequence, int numberOfDifferentInPattern, int patternSize, int sequenceSize) {
	//prepare variables
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	int localTid = tid;

	int* localUniqueSequence = new int[numberOfDifferentInSequence]();
	localUniqueSequence = uniqueSequence;

	//count variant for thread
	int* variant = new int[numberOfDifferentInPattern]();

	for (int x = 0; x < patternSize; x++) {
		int v = getV(numberOfDifferentInSequence - x - 1, numberOfDifferentInPattern - x - 1);
		int t = tid / v;
		variant[x] = localUniqueSequence[t];

		for (int i = t; i < numberOfDifferentInSequence; i++) {
			localUniqueSequence[i] = localUniqueSequence[i + 1];
		}

		localUniqueSequence[numberOfDifferentInSequence - 1] = 0;

		tid = tid % v;
	}

	int* finalPattern = new int[patternSize]();

	//subsitution
	for (int i = 0; i < patternSize; i++) {
		finalPattern[i] = variant[pattern[i] - 1];
	}
	
	//find pattern in sequence
	int counter = 0;
	int counter2 = 0;
	for (int i = 0; i < patternSize; i++) {
		for (int k = counter2; k < sequenceSize; k++) {
			if (counter <= i) {
				if (finalPattern[i] == sequence[k]) {
					counter2 = k + 1;
					counter++;
				}
			}
		}
	}

	/*localSequence = sequence;
	int localSequenceSize = sequenceSize;
	int counter = 0;
	int nextII = 0;
	for (int i = 0; i < patternSize; i++) {
		for (int ii = nextII; ii < localSequenceSize; ii++) {
			if (counter <= i) {
				if (finalPattern[i] == localSequence[ii]) {
					nextII = ii + 1;
					counter++;
				}
			}
		}
	}*/

	//pattern found
	if (counter == patternSize) {
		for (int i = 0; i < patternSize; i++) {
			output[localTid * patternSize + i] = finalPattern[i];
		}
	}

	__syncthreads();
}

int countUniqueLetters(thrust::host_vector<int> sequence) {
	set<int> uniqueLettersInSequence;

	for (int i = 0; i < sequence.size(); i++) {
		uniqueLettersInSequence.insert(sequence[i]);
	}

	return uniqueLettersInSequence.size();
}

thrust::host_vector<int> getUniqueSequence(thrust::host_vector<int> sequence) {
	set<int> uniqueInSequence;

	for (int i = 0; i < sequence.size(); i++) {
		uniqueInSequence.insert(sequence[i]);
	}
	thrust::host_vector<int> uniqueInSequenceVector(uniqueInSequence.begin(), uniqueInSequence.end());
	return uniqueInSequenceVector;
}

void projekt() {
	thrust::host_vector<int> h_tab_out;
	thrust::device_vector<int> d_tab_out;

	thrust::device_vector<int> sequence_device;
	thrust::device_vector<int> pattern_device;
	thrust::device_vector<int> uniqueSequence_device;

	thrust::host_vector<int> pattern;
	pattern.push_back(1);
	pattern.push_back(2);
	pattern.push_back(2);
	pattern.push_back(1);
	
	thrust::host_vector<int> sequence;
	sequence.push_back(1);
	sequence.push_back(2);
	sequence.push_back(4);
	sequence.push_back(3);
	sequence.push_back(5);
	sequence.push_back(3);
	sequence.push_back(6);
	sequence.push_back(2);
	sequence.push_back(1);

	int numberOfDifferentInSequence = countUniqueLetters(sequence);
	int numberOfDifferentInPattern = countUniqueLetters(pattern);

	int sequenceSize = sequence.size();
	int patternSize = pattern.size();

	int variantsNumber = getV(numberOfDifferentInSequence, numberOfDifferentInPattern);

	//---------------------------------------------------------------------------------------------------------------//
	cout << "Liczba roznych w patternie: " << numberOfDifferentInPattern << endl;
	cout << "Liczba roznych w sekwencji: " << numberOfDifferentInSequence << endl;
	cout << "Liczba wariancji: " << variantsNumber << endl;
	//---------------------------------------------------------------------------------------------------------------//

	thrust::host_vector<int> uniqueSequence = getUniqueSequence(sequence);

	dim3 dimBlock(variantsNumber);
	dim3 dimGrid(1);

	h_tab_out.resize(variantsNumber * patternSize);
	d_tab_out.resize(variantsNumber * patternSize);

	uniqueSequence_device.resize(uniqueSequence.size());
	sequence_device.resize(sequence.size());
	pattern_device.resize(pattern.size());

	uniqueSequence_device = uniqueSequence;
	sequence_device = sequence;
	pattern_device = pattern;

	func << <dimGrid, dimBlock >> > (
		d_tab_out.data().get(),
		sequence_device.data().get(),
		pattern_device.data().get(),
		uniqueSequence_device.data().get(),
		numberOfDifferentInSequence,
		numberOfDifferentInPattern,
		patternSize,
		sequenceSize
		);

	h_tab_out = d_tab_out; //Kopiowanie device->host

	for (int s = 0; s < h_tab_out.size(); s++) {
		if (s % patternSize == 0) {
			cout << endl;
		}
		cout << h_tab_out[s];
	}
}

int main() {
	projekt();
	return EXIT_SUCCESS;
}