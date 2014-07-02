/* 
 * File:   CUDAMatMult.h
 * Author: weichen_cheng
 *
 * Created on November 18, 2013, 10:25 AM
 */

#ifndef CUDAMATMULT_H
#define	CUDAMATMULT_H

#include <curand_kernel.h>

class AdjListGPU{	//the adjacency list
public:
	int height;
	int* adjW;
	int* adjW_size;
	int* adjW_cum_size;
};

class TadjListGPU{	//Transpose of the adjacency list
public:
	int height;
	int* TadjW;
	int* TadjW_size;
	int* TadjW_cum_size;
};

class IntegerMatrix{
public:
	int width;
	int height;
	int* elements;
};

class Matrix{
public:
	int width;
	int height;
	float* elements;
};

class Mask{
public:
	int width;
	int height;
	bool* elements;
};

//	calculate the momentum term
__global__ void CalculateMomentum(AdjListGPU GPU_A_Adj, Matrix A, Matrix B, float mr) {

	int row = blockIdx.x * blockDim.x + threadIdx.x;

	if(row>=A.height) return;
	
	int size = GPU_A_Adj.adjW_size[row], k;
	for(int j=0;j<size;j++){
		k = GPU_A_Adj.adjW[GPU_A_Adj.adjW_cum_size[row] + j];
		A.elements[row*A.width + k] += mr * B.elements[row*B.width + k];
		B.elements[row*B.width + k] = A.elements[row*A.width + k];
	}

}

__global__ void CUDAupdateProjW(const float* delta0, float* projW, int* batch_samples, int* input_seq, 
                                int voc_size, int proj_dim, int nGram, int sample_size, int sizeOfBatchBlock, float lr) {
	int row = blockIdx.x * blockDim.x + threadIdx.x;

	if(row>=proj_dim) return;
	
	for(int i=0;i<nGram-1;i++){
            for(int j=0;j<sample_size;j++){
                    projW[row*voc_size + input_seq[batch_samples[j]+i]] += lr * delta0[(proj_dim*i + row)*sizeOfBatchBlock + j];
            }
        }
        for(int i=0;i<nGram-1;i++){
            for(int j=0;j<sample_size;j++){
                    projW[row*voc_size + input_seq[batch_samples[j]+i]] = max(projW[row*voc_size + input_seq[batch_samples[j]+i]], 0.0f);
            }
        }
}

__global__ void loadBatchNGram(float* v0, const float* projW, const int* batch_samples, int* input_seq, 
                                int voc_size, int proj_dim, int nGram, int sample_size, int sizeOfBatchBlock, int input_scale) {
	int row = blockIdx.x * blockDim.x + threadIdx.x;

	if(row>=sample_size) return;
	
	for(int i=0;i<nGram-1;i++){
                for(int j=0;j<proj_dim;j++){
                        v0[(proj_dim*i + j)*sizeOfBatchBlock + row] = input_scale * projW[j*voc_size + input_seq[batch_samples[row]+i]];
                }
        }
}

__global__ void initializeW(AdjListGPU GPU_A_Adj, Matrix A, bool norm_weight_flag, int host_seed) {
	int row = blockIdx.x * blockDim.x + threadIdx.x;

	if(row>=A.height) return;
	
	unsigned int seed = row;
	curandState s;
	// seed a random number generator
	curand_init(seed + host_seed, 0, 0, &s);
	
	int size = GPU_A_Adj.adjW_size[row], k;
	int base = GPU_A_Adj.adjW_cum_size[row];
	
	for(int j=0;j<size;j++){
		k = GPU_A_Adj.adjW[base + j];
		A.elements[row*A.width + k] = curand_uniform(&s); //j+1;
	}

	if(norm_weight_flag){
		float sum = 0;
		for(int j=0;j<size;j++){
			k = GPU_A_Adj.adjW[base + j];
			sum += A.elements[row*A.width + k];
		}
		if(sum!=0)
			for(int j=0;j<size;j++)
				A.elements[row*A.width + GPU_A_Adj.adjW[base + j]] /= sum;
	}
}

// 	update and normalize W;
__global__ void UpdateW(AdjListGPU GPU_A_Adj, Matrix A, Matrix B, float lr, float reg_term, bool norm_weight_flag) {

	int row = blockIdx.x * blockDim.x + threadIdx.x;

	if(row>=A.height) return;

	int size = GPU_A_Adj.adjW_size[row], k;
	int base = GPU_A_Adj.adjW_cum_size[row];
	
	for(int j=0;j<size;j++){
		k = GPU_A_Adj.adjW[base + j];
		A.elements[row*A.width + k] += reg_term * A.elements[row*A.width + k];
		A.elements[row*A.width + k] += lr * B.elements[row*B.width + k];
		A.elements[row*A.width + k] = max(A.elements[row*A.width + k], 0.0f);
	}

	if(norm_weight_flag){
		float sum = 0;
		for(int j=0;j<size;j++){
			k = GPU_A_Adj.adjW[base + j];
			sum += A.elements[row*A.width + k];
		}
		if(sum!=0)
			for(int j=0;j<size;j++)
				A.elements[row*A.width + GPU_A_Adj.adjW[base + j]] /= sum;
	}
}

__global__ void CumMatMulABTKernel(Matrix A, Matrix B, Matrix C, Matrix D, Mask E, int sign) {

	float Dvalue = 0;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if(row>=D.height || col>=D.width) return;
	if(E.elements[row*E.width + col]==0) return;

	if(col<B.height){
		for (int e = 0; e < A.width; ++e)
			Dvalue += A.elements[row * A.width + e] * B.elements[col * B.width + e];
	}else{
		int adjusted_col = col-B.height;
		for (int e = 0; e < A.width; ++e)
			Dvalue += A.elements[row * A.width + e] * C.elements[adjusted_col * C.width + e];
	}
	D.elements[row * D.width + col] = D.elements[row * D.width + col] + sign * Dvalue;
}

//	Forward of sum node		D = A*[B; C];
__global__ void MatAdjMulKernel(AdjListGPU GPU_A_Adj, Matrix A, Matrix B, Matrix C, Matrix D, bool flag) {

	float Dvalue = 0;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if(row>=D.height || col>=D.width) return;

	int k;
	int size = GPU_A_Adj.adjW_size[row];
	if(flag){ // include output layer
		for(int j=0;j<size;j++){
			k = GPU_A_Adj.adjW[GPU_A_Adj.adjW_cum_size[row] + j];
			if(k<B.height)	
				Dvalue += A.elements[row*A.width + k] * B.elements[k*B.width + col];
			else
				Dvalue += A.elements[row*A.width + k] * C.elements[(k-B.height)*C.width + col];
		}
	}else{	// not include output layer
		for(int j=0;j<size;j++){
			k = GPU_A_Adj.adjW[GPU_A_Adj.adjW_cum_size[row] + j];
			Dvalue += A.elements[row*A.width + k] * B.elements[k*B.width + col];
		}
	}

	D.elements[row * D.width + col] = Dvalue;
}

//	calculate the change of bias
//	temporary shortcut code for saving the memory which is very sparse
__global__ void calculateDW2BiasKernel(float* A, int hA, int wA, const float* B, int hB, int wB) {

	int row = blockIdx.x * blockDim.x + threadIdx.x;

	if(row>=hA) return;

        float sum = 0;
        for(int i=0;i<wB;i++){
                sum += B[row*wB + i];
        }
        
        A[row] += sum;
}

//	update bias
//	temporary shortcut code for saving the memory which is very sparse
__global__ void UpdateW2BiasKernel(float* A, int hA, int wA, const float* B, int hB, int wB, float lr) {

	int row = blockIdx.x * blockDim.x + threadIdx.x;

	if(row>=hA) return;
        
        A[row] += lr * B[row];
        A[row] = max(A[row], 0.0f);
}

//	Forward of a special layer
//	temporary shortcut code for saving the memory which is very sparse
__global__ void Special1ForwardKernel(float* A, int hA, int wA, const float* B, int hB, int wB, const float* C, int hC, int wC, const float* D, int K) {

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if(row>=hA || col>=wA) return;

	float vle = D[row]; //int base = 1;
	for(int k=0;k<K;k++){
		//base *= (k+1);
		//vle += B[(row*K+k)*wB+col]/base;
            vle += C[row * wC + k] * B[(row*K+k)*wB+col];
	}
	A[row * wA + col] = vle;
}

//	Backward of a special layer
//	temporary shortcut code for saving the memory which is very sparse
__global__ void Special1BackwardKernel(float* A, int hA, int wA, const float* B, int hB, int wB, const float* C, int hC, int wC, int K) {

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if(row>=hB || col>=wB) return;

	float vle; //int base = 1;
	vle = B[row*wB+col];
	for(int k=0;k<K;k++){
		//base *= (k+1);
		A[(row*K+k)*wA+col] = C[row*wC + k] * vle;
	}
}

//	calculate the d_W of a special layer
//	temporary shortcut code for saving the memory which is very sparse
__global__ void CalculateSpecial1DWKernel(const float* A, int hA, int wA, const float* B, int hB, int wB, float* C, int hC, int wC, int K) {

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if(row>=hC || col>=wC) return;

        float vle = 0;
	for(int i=0;i<wA;i++){
            vle += A[row * wA + i] * B[(row*K+col)*wB+i];
	}
	C[row * wC + col] += vle;
}

//	update of a special layer
//	temporary shortcut code for saving the memory which is very sparse
__global__ void Special1UpdateWKernel(float* A, int hA, int wA, const float* B, int hB, int wB, float lr) {

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if(row>=hA || col>=wA) return;

	A[row * wA + col] += lr * B[row * wB + col];
        A[row * wA + col] = max(A[row * wA + col], 0.0f);
}


//	Forward of max node		D = max(A \* [B; C]);
__global__ void MatAdjMaxKernel(AdjListGPU GPU_A_Adj, Matrix A, Matrix B, Matrix C, Matrix D, IntegerMatrix E, bool flag) {

	float Dvalue = -1; int Didx = 0;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if(row>=D.height || col>=D.width) return;

	int k; float vle;
	int size = GPU_A_Adj.adjW_size[row];
	if(flag){ // include output layer
		for(int j=0;j<size;j++){
			k = GPU_A_Adj.adjW[GPU_A_Adj.adjW_cum_size[row] + j];
			if(k<B.height){
				vle = A.elements[row*A.width + k] * B.elements[k*B.width + col];
				if(vle>Dvalue){
					Dvalue = vle;
					Didx = k;
				}
			}else{
				vle = A.elements[row*A.width + k] * C.elements[(k-B.height)*C.width + col];
				if(vle>Dvalue){
					Dvalue = vle;
					Didx = k;
				}
			}
		}
	}else{	// not include output layer
		for(int j=0;j<size;j++){
			k = GPU_A_Adj.adjW[GPU_A_Adj.adjW_cum_size[row] + j];
			vle = A.elements[row*A.width + k] * B.elements[k*B.width + col];
			if(vle>Dvalue){
				Dvalue = vle;
				Didx = k;
			}
		}
	}

	D.elements[row * D.width + col] = Dvalue;
	E.elements[row * D.width + col] = Didx;
}

//	Backward of sum node
__global__ void MatAdjMulATBKernel(TadjListGPU GPU_AT_Adj, Matrix A, Matrix B, Matrix C) {

	float Cvalue = 0;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if(row>=C.height || col>=C.width) return;

	int k;
	int size = GPU_AT_Adj.TadjW_size[row];
	for(int j=0;j<size;j++){
		k = GPU_AT_Adj.TadjW[GPU_AT_Adj.TadjW_cum_size[row] + j];
		Cvalue += A.elements[k*A.width + row] * B.elements[k*B.width + col];
	}

	C.elements[row * C.width + col] = Cvalue;
}

//	Backward of max node
__global__ void MatAdjMaxATBKernel(TadjListGPU GPU_AT_Adj, Matrix A, Matrix B, Matrix C, IntegerMatrix E) {

	float Cvalue = 0;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if(row>=C.height || col>=C.width) return;

	int k;
	int size = GPU_AT_Adj.TadjW_size[row];
	for(int j=0;j<size;j++){
		k = GPU_AT_Adj.TadjW[GPU_AT_Adj.TadjW_cum_size[row] + j];
		if(E.elements[k*E.width + col] == row)
			Cvalue += A.elements[k*A.width + row] * B.elements[k*B.width + col];
	}

	C.elements[row * C.width + col] = Cvalue;
}

//	Forward of product node
__global__ void MatAdjProdKernel(AdjListGPU GPU_A_Adj, Matrix B, Matrix C, Matrix D) {

	float Dvalue = 1;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if(row>=D.height || col>=D.width) return;

	int size = GPU_A_Adj.adjW_size[row];
	for(int j=0;j<size;j++){
		int k = GPU_A_Adj.adjW[GPU_A_Adj.adjW_cum_size[row] + j];
		if(k<B.height)
			Dvalue *= B.elements[k*B.width + col];
		else
			Dvalue *= C.elements[(k-B.height)*C.width + col];
	}

	D.elements[row * D.width + col] = Dvalue;
}

// 	Backward of product node
__global__ void MatAdjTransposeProdKernel(AdjListGPU GPU_A_Adj, TadjListGPU GPU_AT_Adj, Matrix A, Matrix B, Matrix C, Matrix D, Matrix E) {

	float Evalue = 0;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if(row>=E.height || col>=E.width) return;

	int size = GPU_AT_Adj.TadjW_size[row];
	int base = GPU_AT_Adj.TadjW_cum_size[row];
	for(int j=0;j<size;j++){
		float err = A.elements[GPU_AT_Adj.TadjW[base + j]*A.width + col];
		float weight = 1;

		if(C.elements[row*C.width + col]!=0){
			weight = (B.elements[GPU_AT_Adj.TadjW[GPU_AT_Adj.TadjW_cum_size[row] + j]*B.width + col]/C.elements[row*C.width + col]);
			Evalue += err * weight;
		}else{
			
			int next_v = GPU_AT_Adj.TadjW[GPU_AT_Adj.TadjW_cum_size[row] + j];
			int size = GPU_A_Adj.adjW_size[next_v];
			int base2 = GPU_A_Adj.adjW_cum_size[next_v];
			for(int k=0;k<size;k++){
				int idx = GPU_A_Adj.adjW[base2 + k];
				if(idx!=row){
					if(idx<C.height)
						weight*=C.elements[idx*C.width + col];
					else
						weight*=D.elements[(idx-C.height)*D.width + col];
				}
			}
			Evalue += err * weight;
			
			//Evalue += 1;
		}
	}

	E.elements[row * E.width + col] = Evalue;
}


#endif	/* CUDAMATMULT_H */

