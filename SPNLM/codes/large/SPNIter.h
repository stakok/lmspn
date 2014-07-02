/* 
 * File:   SPNIter.h
 * Author: weichen_cheng
 *
 * Created on November 18, 2013, 10:24 AM
 */

#ifndef SPNITER_H
#define	SPNITER_H


#include <cstdlib>
#include <map>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <sstream>
#include "Tokenizer.h"

#define BLOCK_SIZE 32
#define PLUS 1
#define MINUS -1

#define TYPE_INPUT_LAYER 1
#define TYPE_PROD_LAYER 2
#define TYPE_SUM_LAYER 3
#define RAND 1
#define SEQUENTIAL 2

//#define GRAM_NUM 6

//#define K_POWER 10
#define INPUT_SCALE 1
#define VALID 1
#define SILENT 1

#include "CUDAMatMult.h"
#include "AdjMatrix.h"
#include <cublas_v2.h>
#include <cusparse_v2.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <limits>

using namespace std;
using namespace thrust;

bool sort_idx(std::pair<int, int> const & a, std::pair<int, int> const & b) {
  return a.second < b.second;
}

class SPN{
private: 
	vector<string> vocabulary;
	map<string, int> vocabularyId;
	vector<int> LM_data;
        device_vector<int> device_LM_data;
	vector<int> valid_data;
        device_vector<int> device_valid_data;
	vector<int> test_data;
        device_vector<int> device_test_data;
	device_vector<int> batch_samples;

	long int sizeOfData, sizeOfValidData, sizeOfTestData, sizeOfBatchBlock;
	vector<bool> outLayerFlag;	// marks whether this layer contains output layer
	vector<bool> biasFlag;		// marks whether this layer has to add bias
	vector<bool> maxModeFlag;		// marks whether this layer use max node
	vector<int*> v_out_maxIdx;

	device_vector<float> outLayers;	//store the output on device
	device_vector<float> outLayers_ones;	//store ones on device
	float* device_outLayers;
	vector<device_vector<float> > device_v_out; 	// The v_out in GPU device memory
	vector<int*> device_v_out_maxIdx; 	// The v_out_maxId in GPU device memory
	vector<device_vector<float> > device_delta;	// The delta in GPU device memory

	vector<DeviceConnAdjList> connAdjList;// The network connection
	vector<DeviceConnTadjList> connTadjList;	// The network connection

	vector<device_vector<float> > device_W;	// The W in GPU device memory
	device_vector<float> device_W2_bias;
        device_vector<float> device_d_W2_bias;
        //host_vector<float> W2_bias;
        device_vector<float> projW;
	device_vector<float> proj_d_W;
	vector<device_vector<bool> > device_W_mask;
	vector<device_vector<float> > device_d_W;
	vector<device_vector<float> > device_mom_d_W;

	vector<long int> numOfNodes; 	// number of nodes in each layer

	vector<int> typeOfLayer;	// 1: input layer, 2: product layer, 3: sum layer
	int numOfLayer;		// number of layers

	long int inputDim;			// The dimension of input
	long int outputDim;			// The dimension of output
	double learning_rate;
	double regulatory_term;
	vector<float> scale_factor;
	double momentum_rate;
	bool norm_weight_flag;
	static const float eps = 1e-15;
	string weights_file;
        int K_POWER;
        int GRAM_NUM;
        int rand_seed;

	clock_t start_time;

	Matrix GPU_A, GPU_B, GPU_C, GPU_D, GPU_E;
	IntegerMatrix GPU_Int_A;
	Mask GPU_MA;
	AdjListGPU GPU_A_Adj;
	TadjListGPU GPU_AT_Adj;

	cublasHandle_t cublas_handle;
	cublasStatus_t cublas_ret;
	cusparseHandle_t cusparse_handle;
	cusparseStatus_t cusparse_ret;
    	cusparseMatDescr_t cusparse_descr;
	
	void setGPU_Matrix(Matrix* Mtx, int height, int width, float* elements){
		Mtx->height = height; Mtx->width = width;
		Mtx->elements = elements;
	}

	void setGPU_IntMatrix(IntegerMatrix* Mtx, int height, int width, int* elements){
		Mtx->height = height; Mtx->width = width;
		Mtx->elements = elements;
	}

public:
	SPN(int num_i, int num_o){		// the parameter is the number of input and output dimensions.
		inputDim = num_i; outputDim = num_o;
		numOfLayer = 0;
		learning_rate = 0;
		regulatory_term = 0;
		norm_weight_flag = true;
		
		cublasCreate(&cublas_handle);
		cusparseCreate(&cusparse_handle);
		cusparse_ret= cusparseCreateMatDescr(&cusparse_descr); 
		if (cusparse_ret != CUSPARSE_STATUS_SUCCESS) cout << "Matrix descriptor initialization failed";
		cusparseSetMatType(cusparse_descr,CUSPARSE_MATRIX_TYPE_GENERAL);
		cusparseSetMatIndexBase(cusparse_descr,CUSPARSE_INDEX_BASE_ZERO);  
	}

	~SPN(){ 
		destructLayeredSPN();
		cublasDestroy(cublas_handle);
		cusparseDestroy(cusparse_handle);
 	}

	void predictingOnDataset(){
		cout << "[MSG] Testing perplexity on testing data: " << calculatePerplexity() << endl;
		//cout << "[MSG] Testing word error rate on testing data: " << 100-100*calculateAccuracy() << "%" << endl;
	}

	void learningOnDataset(int iterations, double lr, double mr, double reg_term, bool n_flag){
		
		loadWeights();

		this->learning_rate = lr;
		this->momentum_rate = mr;
		this->regulatory_term = reg_term;
		this->norm_weight_flag = n_flag;

		float sum_out[sizeOfBatchBlock], a_out[sizeOfBatchBlock];

		int b_layer;
		for(int i=0;i<numOfLayer;i++) 
			if(outLayerFlag.at(i)==1){
				b_layer = i;
				break;
			}

		device_vector<float> saved_delta;
		if(VALID==1) saved_delta.resize(numOfNodes.at(b_layer) * sizeOfBatchBlock, 0);

                host_vector<float> v_out_end, delta_end;
                v_out_end.resize(device_v_out.at(numOfLayer-1).size(), 0);
                delta_end.resize(device_delta.at(numOfLayer-1).size(), 0);
                
		//float pre_valid_ppl = 1e15;
                clock_t start_time_interval = clock();
                int hour = 1;

		double ppl_train = calculatePerplexity();
		cout << "[MSG] The perplexity of training dataset is " << ppl_train << endl;

		double ppl_valid = calculateValidationPerplexity();
		cout << "[MSG] The perplexity of validation dataset is " << ppl_valid << endl;
	
		double ppl_test = calculateTestingPerplexity();
		cout << "[MSG] The perplexity of testing dataset is " << ppl_test << endl;

		for(int times=0;times<iterations;times++){

			clock_t start_time_a_pass = clock();

			//if(times!=0 && times%50000==0) saveWeights(times);
			if((double)(clock()-start_time_interval)/CLOCKS_PER_SEC>3600){
                            
                            saveWeights(hour);
                            
                            double ppl_valid = calculateValidationPerplexity();
                            cout << "[MSG] The perplexity of validation dataset is " << ppl_valid << " at hour " << hour << endl;
			
                            double ppl_test = calculateTestingPerplexity();
                            cout << "[MSG] The perplexity of testing dataset is " << ppl_test << " at hour " << hour << endl;
   			    start_time_interval = clock();
                            
                            hour++;
                        }
			
                        clearD_W();
                        
                        double mean_out = 0; int num_non_zero = 0;
                        //int a_data_idx;

                        /*vector<int> rand_perm;
                        for(int batch_idx=0;batch_idx<ceil((float)sizeOfData/sizeOfBatchBlock)-1;batch_idx++){
                            rand_perm.push_back(batch_idx);
                        }
                        for(int i=0;i<3000;i++){
                            int i1 = rand()%rand_perm.size();
                            int i2 = rand()%rand_perm.size();
                            int tmp = rand_perm[i1];
                            rand_perm[i1] = rand_perm[i2];
                            rand_perm[i2] = tmp;
                        }*/
                        
                        //for(int batch_idx=0;batch_idx<ceil((float)sizeOfData/sizeOfBatchBlock)-1;batch_idx++){
				//copyBatchNgramFromHostToDevice(SEQUENTIAL, rand_perm[batch_idx]);
                        
                            copyBatchNgramFromHostToDevice(RAND, -1);
                            //copyBatchNgramFromHostToDevice(SEQUENTIAL, times % (int)ceil((float)sizeOfData/sizeOfBatchBlock) );
                        //copyBatchNgramFromHostToDevice(SEQUENTIAL, 0);

                            //cout << "[MSG] learning_rate: " << learning_rate << endl;
                                
			    //clearD_W();

                            if(VALID==1){
                                    /****
                                     * In this section, we take advantage of special structure of the network
                                     * so that we only need a single forward pass and a single backward.
                                     * The method is to keep the value that will be used in other pass.
                                     * The last layer is independently calculated and all the other layers are 
                                     * re-used. 
                                     * If your structure is not special, simply set VALID = 0. The algorithm
                                     * will be slower but generate valid result.
                                     */

                                    forwardPass(outputDim+1);

                                    v_out_end = device_v_out.at(numOfLayer-1);
                                    for(int d_idx=0;d_idx<sizeOfBatchBlock;d_idx++){
                                            sum_out[d_idx] = v_out_end[0*sizeOfBatchBlock + d_idx];
                                    }

                                    clearDelta();

                                    for(int d_idx=0;d_idx<sizeOfBatchBlock;d_idx++){
                                            /*a_data_idx = batch_idx*sizeOfBatchBlock + d_idx;
                                            if(a_data_idx>=sizeOfData){
                                                    device_delta.at(numOfLayer-1)[0*sizeOfBatchBlock + d_idx] = 0;
                                            }else{*/
                                                    delta_end[0*sizeOfBatchBlock + d_idx] = -1/sum_out[d_idx];
                                            //}
                                    }
                                    device_delta.at(numOfLayer-1) = delta_end;

                                    backwardPass(outputDim+1, numOfLayer-2, b_layer);

                                    //calculateD_W(PLUS, outputDim+1, b_layer, numOfLayer-2);

                                    cublas_ret = cublasScopy(cublas_handle, numOfNodes.at(b_layer) * sizeOfBatchBlock, raw_pointer_cast(device_delta.at(b_layer).data()), 1, raw_pointer_cast(saved_delta.data()), 1);

                                    if (cublas_ret != CUBLAS_STATUS_SUCCESS){
                                                    cout << "cublasScopy returned error code " << cublas_ret << ", line(" << __LINE__ << ")" << endl;
                                                    exit(EXIT_FAILURE);
                                    }

                                    forwardPass(0, b_layer+1, numOfLayer-1);

                                    v_out_end = device_v_out.at(numOfLayer-1);
                                    for(int d_idx=0;d_idx<sizeOfBatchBlock;d_idx++){
                                            a_out[d_idx] = v_out_end[0*sizeOfBatchBlock + d_idx];
                                    }

                                    clearDelta();

                                    for(int d_idx=0;d_idx<sizeOfBatchBlock;d_idx++){
                                            /*a_data_idx = batch_idx*sizeOfBatchBlock + d_idx;
                                            if(a_data_idx>=sizeOfData){
                                                    device_delta.at(numOfLayer-1)[0*sizeOfBatchBlock + d_idx] = 0;
                                            }else{*/
                                                    delta_end[0*sizeOfBatchBlock + d_idx] = 1/a_out[d_idx];
                                            //}
                                    }
                                    device_delta.at(numOfLayer-1) = delta_end;

                                    backwardPass(0, numOfLayer-2, b_layer);

                                    //calculateD_W(PLUS, 0, b_layer, numOfLayer-2);

                                    /*cudaDeviceSynchronize();

                                    host_vector<float> tmp1;
                                    host_vector<float> tmp2;
                                    host_vector<float> tmp3;
                                    tmp1 = device_delta.at(b_layer);
                                    tmp2 = saved_delta;
                                    tmp3.resize(tmp2.size(), 0);
                                    double min_vle = tmp1[0];
                                    double max_vle = tmp1[0];
                                    double mean_vle1 = 0, mean_vle2 = 0, mean_vle3 = 0;
                                    for(int i=0;i<tmp1.size();i++){
                                        tmp3[i] = tmp1[i] + tmp2[i];
                                        if(tmp1[i] < min_vle) min_vle = tmp1[i];
                                        if(tmp1[i] > max_vle) max_vle = tmp1[i];
                                        mean_vle1 += tmp1[i];
                                        mean_vle2 += tmp2[i];
                                        mean_vle3 += tmp3[i];
                                    }
                                    mean_vle1 /= tmp1.size();
                                    mean_vle2 /= tmp2.size();
                                    mean_vle3 /= tmp3.size();

                                    cudaDeviceSynchronize();*/

                                    //for(int i=0;i<5;i++) cout << tmp3[i] << " ";
                                    //cout << endl;
                                    /*cout << "saved delta (before) (" << min_vle << "," << max_vle << "," << mean_vle1 << ")" << endl;
                                    cout << "mean_vle1: " << mean_vle1 << endl;
                                    cout << "mean_vle2: " << mean_vle2 << endl;
                                    cout << "mean_vle3: " << mean_vle3 << endl;*/

                                    float alpha = 1;
                                    cublas_ret = cublasSaxpy(cublas_handle, numOfNodes.at(b_layer) * sizeOfBatchBlock, &alpha, raw_pointer_cast(saved_delta.data()), 1, raw_pointer_cast(device_delta.at(b_layer).data()), 1);

                                    if (cublas_ret != CUBLAS_STATUS_SUCCESS){
                                                    cout << "cublasScopy returned error code " << cublas_ret << ", line(" << __LINE__ << ")" << endl;
                                                    exit(EXIT_FAILURE);
                                    }

                                    /*min_vle = device_delta.at(b_layer)[0];
                                    max_vle = device_delta.at(b_layer)[0];
                                    double mean_vle = 0;
                                    for(int i=0;i<device_delta.at(b_layer).size();i++){
                                        if(device_delta.at(b_layer)[i] < min_vle) min_vle = device_delta.at(b_layer)[i];
                                        if(device_delta.at(b_layer)[i] > max_vle) max_vle = device_delta.at(b_layer)[i];
                                        mean_vle += device_delta.at(b_layer)[i];
                                    }
                                    mean_vle /= device_delta.at(b_layer).size();
                                    //for(int i=0;i<5;i++) cout << saved_delta[i] << " ";
                                    //cout << endl;
                                    cout << "saved delta (" << min_vle << "," << max_vle << "," << mean_vle << ")" << endl;*/

                                    //continue back propagating
                                    backwardPass(0, b_layer-1, 0); 	// the first parameter does not matter.
                                    //calculateD_W(PLUS, 0, 0, b_layer-1);
                                    calculateD_W(PLUS, 0, 0, 0);	//only update the last layer
                                    calculateUpdateSpecialWeights();        //update this layer
                            }else{

                                    float a_err[sizeOfBatchBlock], sum_err[sizeOfBatchBlock];
                                    double out[sizeOfBatchBlock][outputDim];

                                    forwardPass(0);

                                    if(SILENT==0) start_timing("[TIM] Start to tick!");
                                    if(SILENT==0) cout << "[MSG] Calculate the error." << endl;
                                    for(int d_idx=0;d_idx<sizeOfBatchBlock;d_idx++){
                                            a_out[d_idx] = device_v_out.at(numOfLayer-1)[0*sizeOfBatchBlock + d_idx];
                                            a_err[d_idx] = 1/(a_out[d_idx]);
                                    }
                                    if(SILENT==0) end_timing("[TIM] End of tick.");

                                    clearDelta();

                                    cudaMemcpy(raw_pointer_cast(device_delta.at(numOfLayer-1).data()), a_err, numOfNodes.at(numOfLayer-1) * sizeOfBatchBlock * sizeof(float), cudaMemcpyHostToDevice);
                                    backwardPass(0);	//0 - output data
                                    calculateD_W(PLUS, 0, 1, numOfLayer-1);

                                    for(int i_o=0;i_o<outputDim;i_o++){
                                            forwardPass(i_o+1);
                                            //copy the network output to CPU memory
                                            for(int d_idx=0;d_idx<sizeOfBatchBlock;d_idx++)
                                                    out[d_idx][i_o] = device_v_out.at(numOfLayer-1)[0*sizeOfBatchBlock + d_idx];
                                    }

                                    memset(sum_out, 0, sizeOfBatchBlock*sizeof(float));

                                    for(int d_idx=0;d_idx<sizeOfBatchBlock;d_idx++) sum_out[d_idx] = 0;
                                    for(int d_idx=0;d_idx<sizeOfBatchBlock;d_idx++)
                                            for(int i_o=0;i_o<outputDim;i_o++)
                                                    sum_out[d_idx] += out[d_idx][i_o];

                                    if(SILENT==0) start_timing("[TIM] Start to tick!");
                                    if(SILENT==0) cout << "[MSG] Calculate the error." << endl;
                                    for(int d_idx=0;d_idx<sizeOfBatchBlock;d_idx++){
                                            sum_err[d_idx] = 1/(sum_out[d_idx]);	
                                    }
                                    if(SILENT==0) end_timing("[TIM] End of tick.");

                                    for(int i_o=0;i_o<outputDim;i_o++){
                                            forwardPass(i_o+1);
                                            clearDelta();

                                            cudaMemcpy(raw_pointer_cast(device_delta.at(numOfLayer-1).data()), sum_err, numOfNodes.at(numOfLayer-1) * sizeOfBatchBlock * sizeof(float), cudaMemcpyHostToDevice);
                                            backwardPass(i_o+1);
                                            calculateD_W(MINUS, i_o+1, 1, numOfLayer-1);
                                    }

                                    //check the output
                                    for(int d_idx=0;d_idx<sizeOfBatchBlock;d_idx++){
                                            int all_zero = 1;
                                            for(int i=0;i<numOfNodes.at(numOfLayer-2);i++){
                                                    if(out[d_idx][i]>eps){
                                                            all_zero = 0;
                                                            break;
                                                    }
                                            }
                                            if(all_zero==1) cout << "[ERR] The output for data " << d_idx << " in the batch are all zero to this network !" << endl;
                                    }	

                            }

                            if(SILENT==0) start_timing("[TIM] Start to tick!");
                            if(SILENT==0) cout << "[MSG] Calculate the value of objective function." << endl;

                            for(int d_idx=0;d_idx<sizeOfBatchBlock;d_idx++){
                                    //a_data_idx = batch_idx*sizeOfBatchBlock + d_idx;
                                    //if(a_data_idx<sizeOfData && sum_out[d_idx]!=0 && a_out[d_idx]!=0){
                                    if(sum_out[d_idx]!=0 && a_out[d_idx]!=0){
                                            mean_out += log(a_out[d_idx]/sum_out[d_idx]);
                                            num_non_zero += 1;
                                    }
                            }

			//}

			if(SILENT==0) end_timing("[TIM] End of tick.");
			
                        //calculateMomentum();
                        updateSpecialWeights();
                        updateWeights(0);
                        updateProjW();

                        if(times%500==0){
                        //if(batch_idx%100==0){
                            cout << "[MSG] " << mean_out << "/" << num_non_zero << "=" << mean_out/num_non_zero << " @ " << times << " (" << mean_out << "/" << num_non_zero << ")" << " lr: " << learning_rate << endl;
                            cout << "[MSG] Execution time for a epoch: " << (double)(clock()-start_time_a_pass)/CLOCKS_PER_SEC << endl;                        
                        }
                        
			//if(times!=0 && times%50000==0){
                        /*if(times!=0 && times%1==0){*/
                        //if(times%2==0){
                        //        double ppl_train = calculatePerplexity();
			//	cout << "[MSG] The perplexity of training dataset is " << ppl_train << endl;
                        //}
                        
                        //if(times!=0 && times%10000==0){
				//double ppl_valid = calculateValidationPerplexity();
				//cout << "[MSG] The perplexity of validation dataset is " << ppl_valid << endl;
				
				/*if(ppl_valid>pre_valid_ppl || ppl_valid*1.0003>pre_valid_ppl){
					cout << "perplexity increased or doesn't decrease enough !" << endl;
                                        learning_rate /= 2;
                                        loadWeights();
					//exit(0);
				}else{
					saveWeights();
                                        pre_valid_ppl = ppl_valid;
				}*/
			//}
                                
                        //if(times!=0 && times%10000==0){
                        //    double ppl_test = calculateTestingPerplexity();
                        //    cout << "[MSG] The perplexity of testing dataset is " << ppl_test << endl;
                        //}
                        
			//if(times!=0 && times%150000==0) cout << "[MSG] The word error rate of validation dataset is " << 100-100*calculateValidationAccuracy() << "%" << endl;
                            
                            //if(learning_rate<1e-10) break;
		}
	}

	double calculateTestingProbability(){
		if(SILENT==0) start_timing("[TIM] Start to tick!");
		if(SILENT==0) cout << "[MSG] calculateTestingProbability()";

		double sum_out[sizeOfBatchBlock], a_out[sizeOfBatchBlock];
		host_vector<float> v_out_end;
		v_out_end.resize(device_v_out.at(numOfLayer-1).size(), 0);
		double prob = 0;
cout << "start..." << endl;
		for(int batch_idx=0;batch_idx<ceil((float)sizeOfTestData/sizeOfBatchBlock);batch_idx++){

			copyBatchTestNgramFromHostToDevice(SEQUENTIAL, batch_idx);

			if(VALID==1){
				int b_layer;
				for(int i=0;i<numOfLayer;i++) 
					if(outLayerFlag.at(i)==1){
						b_layer = i;
						break;
					}

				forwardPass(outputDim+1);

				v_out_end = device_v_out.at(numOfLayer-1);
				for(int d_idx=0;d_idx<sizeOfBatchBlock;d_idx++){
					sum_out[d_idx] = v_out_end[0*sizeOfBatchBlock + d_idx];
				}

				forwardPass(0, b_layer+1, numOfLayer-1);

				v_out_end = device_v_out.at(numOfLayer-1);
				for(int d_idx=0;d_idx<sizeOfBatchBlock;d_idx++){
					a_out[d_idx] = v_out_end[0*sizeOfBatchBlock + d_idx];
				}
			}else{
				double out[sizeOfBatchBlock][outputDim];

				forwardPass(0);
				
				for(int d_idx=0;d_idx<sizeOfBatchBlock;d_idx++){
					a_out[d_idx] = device_v_out.at(numOfLayer-1)[0*sizeOfBatchBlock + d_idx];
				}

				for(int i_o=0;i_o<outputDim;i_o++){
					forwardPass(i_o+1);
					
					for(int d_idx=0;d_idx<sizeOfBatchBlock;d_idx++)
						out[d_idx][i_o] = device_v_out.at(numOfLayer-1)[0*sizeOfBatchBlock + d_idx];
				}

				memset(sum_out, 0, sizeOfBatchBlock*sizeof(float));

				for(int d_idx=0;d_idx<sizeOfBatchBlock;d_idx++) sum_out[d_idx] = 0;
				for(int d_idx=0;d_idx<sizeOfBatchBlock;d_idx++)
					for(int i_o=0;i_o<outputDim;i_o++)
						sum_out[d_idx] += out[d_idx][i_o];
			}

			for(int d_idx=0;d_idx<sizeOfBatchBlock;d_idx++){
				int a_data_idx = batch_idx*sizeOfBatchBlock + d_idx;
				if(a_data_idx>=sizeOfTestData) break;
				prob += log((double)a_out[d_idx]/sum_out[d_idx]);
				cout << (double)a_out[d_idx]/sum_out[d_idx] << endl;
			}
		}
cout << "end..." << endl;
		cout << "Testing Perplexity information: " << prob << "/" << sizeOfTestData << endl;
		if(SILENT==0) end_timing("[TIM] End of tick.");
		return (double) exp(-prob/sizeOfTestData);
	}

        double calculateTestingPerplexity(){
		if(SILENT==0) start_timing("[TIM] Start to tick!");
		if(SILENT==0) cout << "[MSG] calculateTestingPerplexity()";

		double sum_out[sizeOfBatchBlock], a_out[sizeOfBatchBlock];
		host_vector<float> v_out_end;
		v_out_end.resize(device_v_out.at(numOfLayer-1).size(), 0);
		double prob = 0;
		for(int batch_idx=0;batch_idx<ceil((float)sizeOfTestData/sizeOfBatchBlock);batch_idx++){

			copyBatchTestNgramFromHostToDevice(SEQUENTIAL, batch_idx);

			if(VALID==1){
				int b_layer;
				for(int i=0;i<numOfLayer;i++) 
					if(outLayerFlag.at(i)==1){
						b_layer = i;
						break;
					}

				forwardPass(outputDim+1);

				v_out_end = device_v_out.at(numOfLayer-1);
				for(int d_idx=0;d_idx<sizeOfBatchBlock;d_idx++){
					sum_out[d_idx] = v_out_end[0*sizeOfBatchBlock + d_idx];
				}

				forwardPass(0, b_layer+1, numOfLayer-1);

				v_out_end = device_v_out.at(numOfLayer-1);
				for(int d_idx=0;d_idx<sizeOfBatchBlock;d_idx++){
					a_out[d_idx] = v_out_end[0*sizeOfBatchBlock + d_idx];
				}
			}else{
				double out[sizeOfBatchBlock][outputDim];

				forwardPass(0);
				
				for(int d_idx=0;d_idx<sizeOfBatchBlock;d_idx++){
					a_out[d_idx] = device_v_out.at(numOfLayer-1)[0*sizeOfBatchBlock + d_idx];
				}

				for(int i_o=0;i_o<outputDim;i_o++){
					forwardPass(i_o+1);
					
					for(int d_idx=0;d_idx<sizeOfBatchBlock;d_idx++)
						out[d_idx][i_o] = device_v_out.at(numOfLayer-1)[0*sizeOfBatchBlock + d_idx];
				}

				memset(sum_out, 0, sizeOfBatchBlock*sizeof(float));

				for(int d_idx=0;d_idx<sizeOfBatchBlock;d_idx++) sum_out[d_idx] = 0;
				for(int d_idx=0;d_idx<sizeOfBatchBlock;d_idx++)
					for(int i_o=0;i_o<outputDim;i_o++)
						sum_out[d_idx] += out[d_idx][i_o];
			}

			for(int d_idx=0;d_idx<sizeOfBatchBlock;d_idx++){
				int a_data_idx = batch_idx*sizeOfBatchBlock + d_idx;
				if(a_data_idx>=sizeOfTestData) break;
 				prob += log((double)a_out[d_idx]/sum_out[d_idx]);
			}
		}

		cout << "Testing Perplexity information: " << prob << "/" << sizeOfTestData << endl;
		if(SILENT==0) end_timing("[TIM] End of tick.");
		return (double) exp(-prob/sizeOfTestData);
	}
        
	double calculateValidationPerplexity(){
		if(SILENT==0) start_timing("[TIM] Start to tick!");
		if(SILENT==0) cout << "[MSG] calculateValidationPerplexity()";

		double sum_out[sizeOfBatchBlock], a_out[sizeOfBatchBlock];
		host_vector<float> v_out_end;
		v_out_end.resize(device_v_out.at(numOfLayer-1).size(), 0);
		double prob = 0;
		for(int batch_idx=0;batch_idx<ceil((float)sizeOfValidData/sizeOfBatchBlock);batch_idx++){
			copyBatchValidationNgramFromHostToDevice(SEQUENTIAL, batch_idx);

			if(VALID==1){
				int b_layer;
				for(int i=0;i<numOfLayer;i++) 
					if(outLayerFlag.at(i)==1){
						b_layer = i;
						break;
					}

				forwardPass(outputDim+1);

				v_out_end = device_v_out.at(numOfLayer-1);
				for(int d_idx=0;d_idx<sizeOfBatchBlock;d_idx++){
					sum_out[d_idx] = v_out_end[0*sizeOfBatchBlock + d_idx];
				}

				forwardPass(0, b_layer+1, numOfLayer-1);

				v_out_end = device_v_out.at(numOfLayer-1);
				for(int d_idx=0;d_idx<sizeOfBatchBlock;d_idx++){
					a_out[d_idx] = v_out_end[0*sizeOfBatchBlock + d_idx];
				}
			}else{
				double out[sizeOfBatchBlock][outputDim];

				forwardPass(0);
				
				for(int d_idx=0;d_idx<sizeOfBatchBlock;d_idx++){
					a_out[d_idx] = device_v_out.at(numOfLayer-1)[0*sizeOfBatchBlock + d_idx];
				}

				for(int i_o=0;i_o<outputDim;i_o++){
					forwardPass(i_o+1);
					
					for(int d_idx=0;d_idx<sizeOfBatchBlock;d_idx++)
						out[d_idx][i_o] = device_v_out.at(numOfLayer-1)[0*sizeOfBatchBlock + d_idx];
				}

				memset(sum_out, 0, sizeOfBatchBlock*sizeof(float));

				for(int d_idx=0;d_idx<sizeOfBatchBlock;d_idx++) sum_out[d_idx] = 0;
				for(int d_idx=0;d_idx<sizeOfBatchBlock;d_idx++)
					for(int i_o=0;i_o<outputDim;i_o++)
						sum_out[d_idx] += out[d_idx][i_o];
			}

			for(int d_idx=0;d_idx<sizeOfBatchBlock;d_idx++){
				int a_data_idx = batch_idx*sizeOfBatchBlock + d_idx;
				if(a_data_idx>=sizeOfValidData) break;
                                
 				prob += log((double)a_out[d_idx]/sum_out[d_idx]);
			}
		}
                
		cout << "Validation Perplexity information: " << prob << "/" << sizeOfValidData << endl;
		if(SILENT==0) end_timing("[TIM] End of tick.");
		return (double) exp(-prob/sizeOfValidData);
	}

	double calculatePerplexity(){
		if(SILENT==0) start_timing("[TIM] Start to tick!");
		if(SILENT==0) cout << "[MSG] calculatePerplexity()";

		double sum_out[sizeOfBatchBlock], a_out[sizeOfBatchBlock];
		host_vector<float> v_out_end;
		v_out_end.resize(device_v_out.at(numOfLayer-1).size(), 0);
		double prob = 0;
		for(int batch_idx=0;batch_idx<ceil((float)sizeOfData/sizeOfBatchBlock);batch_idx++){
			copyBatchNgramFromHostToDevice(SEQUENTIAL, batch_idx);

			if(VALID==1){
				int b_layer;
				for(int i=0;i<numOfLayer;i++) 
					if(outLayerFlag.at(i)==1){
						b_layer = i;
						break;
					}

				forwardPass(outputDim+1);

				v_out_end = device_v_out.at(numOfLayer-1);
				for(int d_idx=0;d_idx<sizeOfBatchBlock;d_idx++){
					sum_out[d_idx] = v_out_end[0*sizeOfBatchBlock + d_idx];
				}

				forwardPass(0, b_layer+1, numOfLayer-1);

				v_out_end = device_v_out.at(numOfLayer-1);
				for(int d_idx=0;d_idx<sizeOfBatchBlock;d_idx++){
					a_out[d_idx] = v_out_end[0*sizeOfBatchBlock + d_idx];
				}
			}else{
				double out[sizeOfBatchBlock][outputDim];

				forwardPass(0);
				
				for(int d_idx=0;d_idx<sizeOfBatchBlock;d_idx++){
					a_out[d_idx] = device_v_out.at(numOfLayer-1)[0*sizeOfBatchBlock + d_idx];
				}

				for(int i_o=0;i_o<outputDim;i_o++){
					forwardPass(i_o+1);
					
					for(int d_idx=0;d_idx<sizeOfBatchBlock;d_idx++)
						out[d_idx][i_o] = device_v_out.at(numOfLayer-1)[0*sizeOfBatchBlock + d_idx];
				}

				memset(sum_out, 0, sizeOfBatchBlock*sizeof(float));

				for(int d_idx=0;d_idx<sizeOfBatchBlock;d_idx++) sum_out[d_idx] = 0;
				for(int d_idx=0;d_idx<sizeOfBatchBlock;d_idx++)
					for(int i_o=0;i_o<outputDim;i_o++)
						sum_out[d_idx] += out[d_idx][i_o];
			}

			for(int d_idx=0;d_idx<sizeOfBatchBlock;d_idx++){
				int a_data_idx = batch_idx*sizeOfBatchBlock + d_idx;
				if(a_data_idx>=sizeOfData) break;
				//cout << "[Prob] " << a_out[d_idx]/sum_out[d_idx] << endl;
 				prob += log((double)a_out[d_idx]/sum_out[d_idx]);
			}
		}

		cout << "Perplexity information: " << prob << "/" << sizeOfData << endl;
		if(SILENT==0) end_timing("[TIM] End of tick.");
		return (double) exp(-prob/sizeOfData);
	}

	// compare the network output with the data output
	double calculateValidationAccuracy(){
		if(SILENT==0) start_timing("[TIM] Start to tick!");
		if(SILENT==0) cout << "[MSG] calculateValidationAccuracy()";

		double** out = new double*[sizeOfBatchBlock];
		for(int i=0;i<sizeOfBatchBlock;i++) out[i] = new double[outputDim];

		int hit = 0;

		host_vector<float> tmp_v_out;
		host_vector<float> tmp_W;

		for(int batch_idx=0;batch_idx<ceil((float)sizeOfValidData/sizeOfBatchBlock);batch_idx++){

			copyBatchValidationNgramFromHostToDevice(SEQUENTIAL, batch_idx);
			

			if(VALID==1){
				forwardPass(outputDim+1);

				tmp_v_out = device_v_out.at(numOfLayer-2);
				tmp_W = device_W.at(numOfLayer-2);

				for(int i_o=0;i_o<outputDim;i_o++){
					for(int d_idx=0;d_idx<sizeOfBatchBlock;d_idx++)
						out[d_idx][i_o] = tmp_v_out[i_o*sizeOfBatchBlock + d_idx] * tmp_W[0*sizeOfBatchBlock + i_o];
				}

			}else{
				for(int i_o=0;i_o<outputDim;i_o++){
					forwardPass(i_o+1);
					
					for(int d_idx=0;d_idx<sizeOfBatchBlock;d_idx++)
						out[d_idx][i_o] = device_v_out.at(numOfLayer-1)[0*sizeOfBatchBlock + d_idx];
				}
			}

			for(int d_idx=0;d_idx<sizeOfBatchBlock;d_idx++){
				int max_idx = -1; double max_vle = 0;
				for(int i_o=0;i_o<outputDim;i_o++) 
					if(out[d_idx][i_o]>max_vle){
						max_vle = out[d_idx][i_o];
						max_idx = i_o;
					}

				int a_data_idx = batch_idx*sizeOfBatchBlock + d_idx;
				if(a_data_idx>=sizeOfValidData) break;
				if(max_idx == valid_data.at(a_data_idx+GRAM_NUM-1))
					hit ++;
			}
		}

		for(int i=0;i<sizeOfBatchBlock;i++) delete[] out[i];
		delete[] out;

		if(SILENT==0) end_timing("[TIM] End of tick.");
		return (double)hit/sizeOfValidData;
	}

	// compare the network output with the data output
	double calculateAccuracy(){
		if(SILENT==0) start_timing("[TIM] Start to tick!");
		if(SILENT==0) cout << "[MSG] calculateAccuracy()";

		double** out = new double*[sizeOfBatchBlock];
		for(int i=0;i<sizeOfBatchBlock;i++) out[i] = new double[outputDim];

		int hit = 0;

		for(int batch_idx=0;batch_idx<ceil((float)sizeOfData/sizeOfBatchBlock);batch_idx++){

			copyBatchNgramFromHostToDevice(SEQUENTIAL, batch_idx);

			if(VALID==1){
				forwardPass(outputDim+1);

				for(int i_o=0;i_o<outputDim;i_o++){
					for(int d_idx=0;d_idx<sizeOfBatchBlock;d_idx++)
						out[d_idx][i_o] = device_v_out.at(numOfLayer-2)[i_o*sizeOfBatchBlock + d_idx] * device_W.at(numOfLayer-2)[0*sizeOfBatchBlock + i_o];
				}

			}else{
				for(int i_o=0;i_o<outputDim;i_o++){
					forwardPass(i_o+1);
					
					for(int d_idx=0;d_idx<sizeOfBatchBlock;d_idx++)
						out[d_idx][i_o] = device_v_out.at(numOfLayer-1)[0*sizeOfBatchBlock + d_idx];
				}
			}

			for(int d_idx=0;d_idx<sizeOfBatchBlock;d_idx++){
				int max_idx = -1; double max_vle = 0;
				for(int i_o=0;i_o<outputDim;i_o++) 
					if(out[d_idx][i_o]>max_vle){
						max_vle = out[d_idx][i_o];
						max_idx = i_o;
					}

				int a_data_idx = batch_idx*sizeOfBatchBlock + d_idx;
				if(a_data_idx>=sizeOfData) break;
				if(max_idx == LM_data.at(a_data_idx+GRAM_NUM-1))
					hit ++;
			}
		}

		for(int i=0;i<sizeOfBatchBlock;i++) delete[] out[i];
		delete[] out;

		if(SILENT==0) end_timing("[TIM] End of tick.");
		return (double)hit/sizeOfData;
	}

        void copyBatchTestNgramFromHostToDevice(int mode, int batch_idx){

		if(sizeOfBatchBlock>sizeOfTestData){
			sizeOfBatchBlock = sizeOfTestData;
			if(SILENT==0) cout << "[MSG] The size of testing data is smaller than batch size, set the batch size equal to testing data size" << endl;
		}

		if(SILENT==0) start_timing("[TIM] Start to tick!");
		if(SILENT==0) cout << "[MSG] copyBatchTestNgramFromHostToDevice()" << endl;
		device_v_out.at(0).clear();
		device_v_out.at(0).resize(inputDim * sizeOfBatchBlock, 0);
		outLayers.clear();
		outLayers.resize(outputDim * sizeOfBatchBlock, 0);

		if(SILENT==0) end_timing("[TIM] 	1. End of tick.");

		bool sampled[sizeOfTestData];
		memset(sampled, false, sizeOfTestData * sizeof(bool));
		batch_samples.clear();
		for(int d_idx=0;d_idx<sizeOfBatchBlock;d_idx++){
			int a_data_idx;
			if(mode==RAND){
				a_data_idx = rand()%sizeOfTestData;
				while(sampled[a_data_idx]) a_data_idx = rand()%sizeOfTestData;
				sampled[a_data_idx] = true;
				batch_samples.push_back(a_data_idx);
			}else if(mode==SEQUENTIAL){
				a_data_idx = batch_idx*sizeOfBatchBlock + d_idx;
				if(a_data_idx>=sizeOfTestData) break;
				batch_samples.push_back(a_data_idx);
			}

			outLayers[test_data.at(a_data_idx+GRAM_NUM-1)*sizeOfBatchBlock + d_idx] = INPUT_SCALE;
		}

                int voc_size = vocabulary.size();
                loadBatchNGram<<<(batch_samples.size() + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>
                        (raw_pointer_cast(device_v_out.at(0).data()), raw_pointer_cast(projW.data()), raw_pointer_cast(batch_samples.data()), 
                        raw_pointer_cast(device_test_data.data()), voc_size, (inputDim/(GRAM_NUM-1)), GRAM_NUM, batch_samples.size(), sizeOfBatchBlock, INPUT_SCALE);
                		
                /*host_vector<float> v_out0;
		v_out0.resize(device_v_out.at(0).size(), 0);
		int voc_size = vocabulary.size();
		int proj_dim = (inputDim/(GRAM_NUM-1));
		for(int i=0;i<GRAM_NUM-1;i++){
			for(int j=0;j<proj_dim;j++){
				for(int k=0;k<batch_samples.size();k++){
					v_out0[(proj_dim*i + j)*sizeOfBatchBlock + k] = INPUT_SCALE * projW[j*voc_size + test_data.at(batch_samples[k]+i)];
				}
			}
		}
		device_v_out.at(0) = v_out0;
              */  
                

		if(SILENT==0) end_timing("[TIM] End of tick.");
	}
        
	void copyBatchValidationNgramFromHostToDevice(int mode, int batch_idx){

		if(sizeOfBatchBlock>sizeOfValidData){
			sizeOfBatchBlock = sizeOfValidData;
			if(SILENT==0) cout << "[MSG] The size of validation data is smaller than batch size, set the batch size equal to validation data size" << endl;
		}

		if(SILENT==0) start_timing("[TIM] Start to tick!");
		if(SILENT==0) cout << "[MSG] copyBatchValidationNgramFromHostToDevice()" << endl;
		device_v_out.at(0).clear();
		device_v_out.at(0).resize(inputDim * sizeOfBatchBlock, 0);
		outLayers.clear();
		outLayers.resize(outputDim * sizeOfBatchBlock, 0);

		if(SILENT==0) end_timing("[TIM] 	1. End of tick.");

		bool sampled[sizeOfValidData];
		memset(sampled, false, sizeOfValidData * sizeof(bool));
		batch_samples.clear();
		for(int d_idx=0;d_idx<sizeOfBatchBlock;d_idx++){
			int a_data_idx;
			if(mode==RAND){
				a_data_idx = rand()%sizeOfValidData;
				while(sampled[a_data_idx]) a_data_idx = rand()%sizeOfValidData;
				sampled[a_data_idx] = true;
				batch_samples.push_back(a_data_idx);
			}else if(mode==SEQUENTIAL){
				a_data_idx = batch_idx*sizeOfBatchBlock + d_idx;
				if(a_data_idx>=sizeOfValidData) break;
				batch_samples.push_back(a_data_idx);
			}

			outLayers[valid_data.at(a_data_idx+GRAM_NUM-1)*sizeOfBatchBlock + d_idx] = INPUT_SCALE;
		}

                int voc_size = vocabulary.size();
                
                loadBatchNGram<<<(batch_samples.size() + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>
                        (raw_pointer_cast(device_v_out.at(0).data()), raw_pointer_cast(projW.data()), raw_pointer_cast(batch_samples.data()), 
                        raw_pointer_cast(device_valid_data.data()), voc_size, (inputDim/(GRAM_NUM-1)), GRAM_NUM, batch_samples.size(), sizeOfBatchBlock, INPUT_SCALE);
                
		/*host_vector<float> v_out0;
		v_out0.resize(device_v_out.at(0).size(), 0);
		int voc_size = vocabulary.size();
		int proj_dim = (inputDim/(GRAM_NUM-1));
		for(int i=0;i<GRAM_NUM-1;i++){
			for(int j=0;j<proj_dim;j++){
				for(int k=0;k<batch_samples.size();k++){
					v_out0[(proj_dim*i + j)*sizeOfBatchBlock + k] = INPUT_SCALE * projW[j*voc_size + valid_data.at(batch_samples[k]+i)];
				}
			}
		}
		device_v_out.at(0) = v_out0;*/

		if(SILENT==0) end_timing("[TIM] End of tick.");
	}

	void copyBatchNgramFromHostToDevice(int mode, int batch_idx){

		if(sizeOfBatchBlock>sizeOfData){
			sizeOfBatchBlock = sizeOfData;
			if(SILENT==0) cout << "[MSG] The size of data is smaller than batch size, set the batch size equal to data size" << endl;
		}

		if(SILENT==0) start_timing("[TIM] Start to tick!");
		if(SILENT==0) cout << "[MSG] copyBatchNgramFromHostToDevice()" << endl;

		device_v_out.at(0).clear();
		device_v_out.at(0).resize(inputDim * sizeOfBatchBlock, 0);
		outLayers.clear();
		outLayers.resize(outputDim * sizeOfBatchBlock, 0);

		if(SILENT==0) end_timing("[TIM] 	1. End of tick.");

		bool* sampled = new bool[sizeOfData];
		//memset(sampled, false, sizeOfData * sizeof(bool));
		batch_samples.clear();
		for(int d_idx=0;d_idx<sizeOfBatchBlock;d_idx++){
			int a_data_idx;
			if(mode==RAND){
				a_data_idx = rand()%sizeOfData;
				while(sampled[a_data_idx]) a_data_idx = rand()%sizeOfData;
				sampled[a_data_idx] = true;
				batch_samples.push_back(a_data_idx);
			}else if(mode==SEQUENTIAL){
				a_data_idx = batch_idx*sizeOfBatchBlock + d_idx;
				if(a_data_idx>=sizeOfData) break;
				batch_samples.push_back(a_data_idx);
			}

			outLayers[LM_data.at(a_data_idx+GRAM_NUM-1)*sizeOfBatchBlock + d_idx] = INPUT_SCALE;
		}
                free(sampled);
                
                int voc_size = vocabulary.size();
                loadBatchNGram<<<(batch_samples.size() + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>
                        (raw_pointer_cast(device_v_out.at(0).data()), raw_pointer_cast(projW.data()), raw_pointer_cast(batch_samples.data()), 
                        raw_pointer_cast(device_LM_data.data()), voc_size, (inputDim/(GRAM_NUM-1)), GRAM_NUM, batch_samples.size(), sizeOfBatchBlock, INPUT_SCALE);
                
		if(SILENT==0) end_timing("[TIM] End of tick.");
	}

	void forwardPass(int outIdx){ forwardPass(outIdx, 1, numOfLayer-1); } 	// begin from 1st layer if users do not specify the begin_layer

	void forwardPass(int outIdx, int begin_layer, int end_layer){
		if(SILENT==0) start_timing("[TIM] Start to tick!");
		if(SILENT==0) cout << "[MSG] forwardPass() with outIdx=" << outIdx << endl;

		for(int l_idx=begin_layer;l_idx<=end_layer;l_idx++){
			if(SILENT==0) cout << "[MSG] forwardPass() l_idx = " << l_idx << endl;
			if(SILENT==0) cout << "[MSG] matrix size " << numOfNodes.at(l_idx) << "x" << numOfNodes.at(l_idx-1) + (outLayerFlag.at(l_idx-1)?outputDim:0)  << " @ layer " << l_idx << endl;
                        
			//temporary shortcut code for saving the memory which is very sparse
			if(l_idx==3){
				
				dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
				dim3 dimGrid((sizeOfBatchBlock + dimBlock.x - 1) / dimBlock.x, (numOfNodes.at(l_idx) + dimBlock.y -1) / dimBlock.y);
				//Special1ForwardKernel<<<dimGrid, dimBlock>>>(raw_pointer_cast(device_v_out.at(l_idx).data()), numOfNodes.at(l_idx), sizeOfBatchBlock, raw_pointer_cast(device_v_out.at(l_idx-1).data()), numOfNodes.at(l_idx-1), sizeOfBatchBlock, K);
                                Special1ForwardKernel<<<dimGrid, dimBlock>>>(raw_pointer_cast(device_v_out.at(l_idx).data()), numOfNodes.at(l_idx), sizeOfBatchBlock, raw_pointer_cast(device_v_out.at(l_idx-1).data()), numOfNodes.at(l_idx-1), sizeOfBatchBlock, raw_pointer_cast(device_W.at(l_idx-1).data()), vocabulary.size(), K_POWER, raw_pointer_cast(device_W2_bias.data()), K_POWER);

				continue;
			}


			if(typeOfLayer[l_idx]==TYPE_PROD_LAYER){	//product layer
				int n = numOfNodes.at(l_idx);
				
				GPU_A_Adj.height = connAdjList.at(l_idx-1).getHeight();
				GPU_A_Adj.adjW = raw_pointer_cast(connAdjList.at(l_idx-1).adjConn.data());
				GPU_A_Adj.adjW_size = raw_pointer_cast(connAdjList.at(l_idx-1).adjConn_size.data());
				GPU_A_Adj.adjW_cum_size = raw_pointer_cast(connAdjList.at(l_idx-1).adjConn_cum_size.data());

				setGPU_Matrix(&GPU_B, numOfNodes.at(l_idx-1), sizeOfBatchBlock, raw_pointer_cast(device_v_out.at(l_idx-1).data()));
				prepareOutputLayer(outIdx);
				setGPU_Matrix(&GPU_C, outputDim, sizeOfBatchBlock, device_outLayers);
				setGPU_Matrix(&GPU_D, n, sizeOfBatchBlock, raw_pointer_cast(device_v_out.at(l_idx).data()));
				
				dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
				dim3 dimGrid((GPU_D.width + dimBlock.x - 1) / dimBlock.x, (GPU_D.height + dimBlock.y -1) / dimBlock.y);
				MatAdjProdKernel<<<dimGrid, dimBlock>>>(GPU_A_Adj, GPU_B, GPU_C, GPU_D);

				if(biasFlag.at(l_idx)){
					cout << "Bias term for product layer is not defined!";
					exit(0);
				}
			}
			if(typeOfLayer[l_idx]==TYPE_SUM_LAYER){	//sum layer

				if(maxModeFlag.at(l_idx) || outLayerFlag.at(l_idx-1)){
					int n = numOfNodes.at(l_idx);
					int m = numOfNodes.at(l_idx-1) + (outLayerFlag.at(l_idx-1)?outputDim:0);

					GPU_A_Adj.height = connAdjList.at(l_idx-1).getHeight();
					GPU_A_Adj.adjW = raw_pointer_cast(connAdjList.at(l_idx-1).adjConn.data());
					GPU_A_Adj.adjW_size = raw_pointer_cast(connAdjList.at(l_idx-1).adjConn_size.data());
					GPU_A_Adj.adjW_cum_size = raw_pointer_cast(connAdjList.at(l_idx-1).adjConn_cum_size.data());

					setGPU_Matrix(&GPU_A, n, m, raw_pointer_cast(device_W.at(l_idx-1).data()));
					setGPU_Matrix(&GPU_B, numOfNodes.at(l_idx-1), sizeOfBatchBlock, raw_pointer_cast(device_v_out.at(l_idx-1).data()));
					prepareOutputLayer(outIdx);
					setGPU_Matrix(&GPU_C, outputDim, sizeOfBatchBlock, device_outLayers);
					setGPU_Matrix(&GPU_D, n, sizeOfBatchBlock, raw_pointer_cast(device_v_out.at(l_idx).data()));
					if(maxModeFlag.at(l_idx)) setGPU_IntMatrix(&GPU_Int_A, n, sizeOfBatchBlock, device_v_out_maxIdx.at(l_idx));

					dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
					dim3 dimGrid((GPU_D.width + dimBlock.x - 1) / dimBlock.x, (GPU_D.height + dimBlock.y -1) / dimBlock.y);
					if(maxModeFlag.at(l_idx)==false){ 
						MatAdjMulKernel<<<dimGrid, dimBlock>>>(GPU_A_Adj, GPU_A, GPU_B, GPU_C, GPU_D, outLayerFlag.at(l_idx-1));
					}else{
						MatAdjMaxKernel<<<dimGrid, dimBlock>>>(GPU_A_Adj, GPU_A, GPU_B, GPU_C, GPU_D, GPU_Int_A, outLayerFlag.at(l_idx-1));
					}
				}else{
                                        /*if(l_idx==5){
                                            //cout << device_v_out.at(l_idx)[0] << endl;
                                            float sum = 0;
                                            host_vector<float> tmp;
                                            tmp = device_v_out.at(l_idx-1);
                                            for(int i=0;i<tmp.size();i++) {
                                                sum += tmp[i];
                                            } 
                                            device_v_out.at(l_idx)[0] = sum;
                                        }else{*/
                                            float alpha = 1, beta = 0;
                                            int n = numOfNodes.at(l_idx);
                                            int m = numOfNodes.at(l_idx-1);

                                            //cublas_ret = cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, sizeOfBatchBlock, n, m, &alpha, raw_pointer_cast(device_v_out.at(l_idx-1).data()), sizeOfBatchBlock, raw_pointer_cast(device_W.at(l_idx-1).data()), m, &beta, raw_pointer_cast(device_v_out.at(l_idx).data()), sizeOfBatchBlock);
                                            cublas_ret = cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, sizeOfBatchBlock, n, m, &alpha, raw_pointer_cast(device_v_out.at(l_idx-1).data()), sizeOfBatchBlock, raw_pointer_cast(device_W.at(l_idx-1).data()), m, &beta, raw_pointer_cast(device_v_out.at(l_idx).data()), sizeOfBatchBlock);

                                            if (cublas_ret != CUBLAS_STATUS_SUCCESS){
                                                    cout << "cublasSgemm returned error code " << cublas_ret << ", line(" << __LINE__ << ")" << endl;
                                                    exit(EXIT_FAILURE);
                                            }
                                        //}
				}

				if(biasFlag.at(l_idx)){
					if(SILENT==0) cout << "Layer " << l_idx << " add bias " << endl;
					host_vector<float> tmp_v_out;
					tmp_v_out = device_v_out.at(l_idx);
					for(int i=0;i<tmp_v_out.size();i++) tmp_v_out[i]++;
					device_v_out.at(l_idx) = tmp_v_out;
				}
			}

			//scale the output
			if(scale_factor.at(l_idx)!=1){
				float alpha = scale_factor.at(l_idx);
				cublas_ret = cublasSscal(cublas_handle, numOfNodes.at(l_idx)*sizeOfBatchBlock, &alpha, raw_pointer_cast(device_v_out.at(l_idx).data()), 1);
				if (cublas_ret != CUBLAS_STATUS_SUCCESS){
					cout << "cublasSscal returned error code " << cublas_ret << ", line(" << __LINE__ << ")" << endl;
					exit(EXIT_FAILURE);
				}
			}

			if(SILENT==0) end_timing("[TIM] 	End of tick.");
		}

		if(SILENT==0) end_timing("[TIM] End of tick.");
	}

	void backwardPass(int outIdx){ 
		backwardPass(outIdx, numOfLayer-2, 0); 
	}

	void backwardPass(int outIdx, int from_layer, int end_layer){
		if(SILENT==0) start_timing("[TIM] Start to tick!");
		if(SILENT==0) cout << "[MSG] backwardPass() with outIdx " << outIdx << endl;

		for(int l_idx=from_layer;l_idx>=end_layer;l_idx--){

			if(SILENT==0) cout << "[MSG] matrix size " << numOfNodes.at(l_idx+1) << "x" << numOfNodes.at(l_idx) + (outLayerFlag.at(l_idx)?outputDim:0) << " @ layer " << l_idx << endl;

			//temporary shortcut code for saving the memory which is very sparse
			if(l_idx==2){

				dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
				dim3 dimGrid((sizeOfBatchBlock + dimBlock.x - 1) / dimBlock.x, (numOfNodes.at(l_idx+1) + dimBlock.y -1) / dimBlock.y);
				Special1BackwardKernel<<<dimGrid, dimBlock>>>(raw_pointer_cast(device_delta.at(l_idx).data()), numOfNodes.at(l_idx), sizeOfBatchBlock, raw_pointer_cast(device_delta.at(l_idx+1).data()), numOfNodes.at(l_idx+1), sizeOfBatchBlock, raw_pointer_cast(device_W.at(l_idx).data()), vocabulary.size(), K_POWER, K_POWER);

				continue;
			}

			//scale the output
			if(scale_factor.at(l_idx)!=1){
				float alpha = scale_factor.at(l_idx);
				cublas_ret = cublasSscal(cublas_handle, numOfNodes.at(l_idx)*sizeOfBatchBlock, &alpha, raw_pointer_cast(device_delta.at(l_idx).data()), 1);
				if (cublas_ret != CUBLAS_STATUS_SUCCESS){
					cout << "cublasSscal returned error code " << cublas_ret << ", line(" << __LINE__ << ")" << endl;
					exit(EXIT_FAILURE);
				}

				if(SILENT==0) end_timing("[TIM] 1.	End of tick.");
			}
			
			if(typeOfLayer.at(l_idx+1)==TYPE_PROD_LAYER){	//product layer
				int n = numOfNodes.at(l_idx+1);

				GPU_A_Adj.height = connAdjList.at(l_idx).getHeight();
				GPU_A_Adj.adjW = raw_pointer_cast(connAdjList.at(l_idx).adjConn.data());
				GPU_A_Adj.adjW_size = raw_pointer_cast(connAdjList.at(l_idx).adjConn_size.data());
				GPU_A_Adj.adjW_cum_size = raw_pointer_cast(connAdjList.at(l_idx).adjConn_cum_size.data());

				GPU_AT_Adj.height = connTadjList.at(l_idx).getHeight();
				GPU_AT_Adj.TadjW = raw_pointer_cast(connTadjList.at(l_idx).TadjConn.data());
				GPU_AT_Adj.TadjW_size = raw_pointer_cast(connTadjList.at(l_idx).TadjConn_size.data());
				GPU_AT_Adj.TadjW_cum_size = raw_pointer_cast(connTadjList.at(l_idx).TadjConn_cum_size.data());

				setGPU_Matrix(&GPU_A, n, sizeOfBatchBlock, raw_pointer_cast(device_delta.at(l_idx+1).data()));
				setGPU_Matrix(&GPU_B, n, sizeOfBatchBlock, raw_pointer_cast(device_v_out.at(l_idx+1).data()));
				setGPU_Matrix(&GPU_C, numOfNodes.at(l_idx), sizeOfBatchBlock, raw_pointer_cast(device_v_out.at(l_idx).data()));
				prepareOutputLayer(outIdx);
				setGPU_Matrix(&GPU_D, outputDim, sizeOfBatchBlock, device_outLayers);
				setGPU_Matrix(&GPU_E, numOfNodes.at(l_idx), sizeOfBatchBlock, raw_pointer_cast(device_delta.at(l_idx).data()));
				dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
				dim3 dimGrid((GPU_E.width + dimBlock.x - 1) / dimBlock.x, (GPU_E.height + dimBlock.y -1) / dimBlock.y);
				MatAdjTransposeProdKernel<<<dimGrid, dimBlock>>>(GPU_A_Adj, GPU_AT_Adj, GPU_A, GPU_B, GPU_C, GPU_D, GPU_E);
			}

			if(typeOfLayer.at(l_idx+1)==TYPE_SUM_LAYER){	//sum layer
				int n = numOfNodes.at(l_idx+1);
				int m = numOfNodes.at(l_idx) + (outLayerFlag.at(l_idx)?outputDim:0);
				GPU_AT_Adj.height = connTadjList.at(l_idx).getHeight();
				GPU_AT_Adj.TadjW = raw_pointer_cast(connTadjList.at(l_idx).TadjConn.data());
				GPU_AT_Adj.TadjW_size = raw_pointer_cast(connTadjList.at(l_idx).TadjConn_size.data());
				GPU_AT_Adj.TadjW_cum_size = raw_pointer_cast(connTadjList.at(l_idx).TadjConn_cum_size.data());

				setGPU_Matrix(&GPU_A, n, m, raw_pointer_cast(device_W.at(l_idx).data()));
				setGPU_Matrix(&GPU_B, n, sizeOfBatchBlock, raw_pointer_cast(device_delta.at(l_idx+1).data()));
				setGPU_Matrix(&GPU_C, numOfNodes.at(l_idx), sizeOfBatchBlock, raw_pointer_cast(device_delta.at(l_idx).data()));
				if(maxModeFlag.at(l_idx+1)) setGPU_IntMatrix(&GPU_Int_A, n, sizeOfBatchBlock, device_v_out_maxIdx.at(l_idx+1));

				dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
				dim3 dimGrid((GPU_C.width + dimBlock.x - 1) / dimBlock.x, (GPU_C.height + dimBlock.y -1) / dimBlock.y);

				if(maxModeFlag.at(l_idx+1)==false){	//sum node
					if(outLayerFlag.at(l_idx)==true){
						MatAdjMulATBKernel<<<dimGrid, dimBlock>>>(GPU_AT_Adj, GPU_A, GPU_B, GPU_C);
					}else{
						float alpha = 1, beta = 0;
						int n = numOfNodes.at(l_idx+1);
						int m = numOfNodes.at(l_idx);

						cublas_ret = cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, sizeOfBatchBlock, m, n, &alpha, raw_pointer_cast(device_delta.at(l_idx+1).data()), sizeOfBatchBlock, raw_pointer_cast(device_W.at(l_idx).data()), m, &beta, raw_pointer_cast(device_delta.at(l_idx).data()), sizeOfBatchBlock);
					
						if (cublas_ret != CUBLAS_STATUS_SUCCESS){
							cout << "cublasSgemm returned error code " << cublas_ret << ", line(" << __LINE__ << ")" << endl;
							exit(EXIT_FAILURE);
						}
					}
				}else{					//max node
					MatAdjMaxATBKernel<<<dimGrid, dimBlock>>>(GPU_AT_Adj, GPU_A, GPU_B, GPU_C, GPU_Int_A);
				}
			}

/*host_vector<float> tmp;
tmp = device_delta.at(l_idx);
float max_vle = tmp[0], min_vle = tmp[0];
double mean_vle = 0;
for(int i=0;i<device_v_out.at(l_idx).size();i++){
	if(tmp[i]>max_vle) max_vle = tmp[i];
	if(tmp[i]<min_vle) min_vle = tmp[i];
        mean_vle += tmp[i];
}
mean_vle /= device_v_out.at(l_idx).size();
cout << "layer " << l_idx << " delta (" << min_vle << "," << max_vle << "," << mean_vle << ")" << endl;*/

/*if(l_idx==0){
    cout << "layer 1:" << endl;
    for(int i=0;i<numOfNodes.at(l_idx+1);i++){
        for(int j=0;j<min((int)sizeOfBatchBlock,5);j++){
            cout << device_delta.at(l_idx+1)[i*sizeOfBatchBlock + j] << " ";
        }
    }
    cout << endl;
    
    cout << "W:" << endl;
    for(int i=0;i<numOfNodes.at(l_idx+1);i++){
        for(int j=0;j<min((int)numOfNodes.at(l_idx), 1);j++){
            cout << device_W.at(l_idx)[i*numOfNodes.at(l_idx) + j] << " ";
        }
    }
    cout << endl;
    
    host_vector<float> tmp1;
    tmp1 = device_delta.at(l_idx+1);
    host_vector<float> tmp2;
    tmp2 = device_W.at(l_idx);
    host_vector<float> tmp_delta0;
    tmp_delta0.resize(device_delta.at(l_idx).size(), 0);
    cout << numOfNodes.at(l_idx) << endl;
    for(int i=0;i<numOfNodes.at(l_idx);i++){
        for(int j=0;j<sizeOfBatchBlock;j++){
            tmp_delta0[i*sizeOfBatchBlock + j] = 0;
            float a = 0;
            for(int k=0;k<numOfNodes.at(l_idx+1);k++){
                tmp_delta0[i*sizeOfBatchBlock + j] += tmp1[k*sizeOfBatchBlock + j] * tmp2[k*numOfNodes.at(l_idx) + i];
                a += tmp1[k*sizeOfBatchBlock + j] * tmp2[k*numOfNodes.at(l_idx) + i];
                cout << device_delta.at(l_idx+1)[k*sizeOfBatchBlock + j] << "*" << device_W.at(l_idx)[k*numOfNodes.at(l_idx) + i] << "=" << a << endl;
            }
            cout << tmp_delta0[i*sizeOfBatchBlock + j] << endl;
            cout << a << endl;
        }
    }
    
    cout << "layer 0:" << endl;
    for(int i=0;i<min((int)numOfNodes.at(l_idx),5);i++){
        for(int j=0;j<min((int)sizeOfBatchBlock,5);j++){
            cout << tmp_delta0[i*sizeOfBatchBlock + j] << " ";
        }
        cout << endl;
    }
    
    cout << "layer 0:" << endl;
    for(int i=0;i<min((int)numOfNodes.at(l_idx),5);i++){
        for(int j=0;j<min((int)sizeOfBatchBlock,5);j++){
            cout << device_delta.at(l_idx)[i*sizeOfBatchBlock + j] << " ";
        }
        cout << endl;
    }
}*/

			if(SILENT==0) end_timing("[TIM] 2.	End of tick.");
		}

		if(SILENT==0) end_timing("[TIM] 3. End of tick.");
	}

	void calculateD_W(int sign, int outIdx, int from_layer, int end_layer){
		if(SILENT==0) start_timing("[TIM] Start to tick!");
		if(SILENT==0) cout << "[MSG] calculateD_W()" << endl;

		// calculate d_W
		for(int l_idx=from_layer;l_idx<=end_layer;l_idx++){

			if(typeOfLayer[l_idx+1]==TYPE_SUM_LAYER){
				if(outLayerFlag.at(l_idx)){
					int n = numOfNodes.at(l_idx+1);
					int m = numOfNodes.at(l_idx) + (outLayerFlag.at(l_idx)?outputDim:0);

					setGPU_Matrix(&GPU_A, n, sizeOfBatchBlock, raw_pointer_cast(device_delta.at(l_idx+1).data()));
					setGPU_Matrix(&GPU_B, numOfNodes.at(l_idx), sizeOfBatchBlock, raw_pointer_cast(device_v_out.at(l_idx).data()));
					prepareOutputLayer(outIdx);
					setGPU_Matrix(&GPU_C, outputDim, sizeOfBatchBlock, device_outLayers);
					setGPU_Matrix(&GPU_D, n, m, raw_pointer_cast(device_d_W.at(l_idx).data()));

					GPU_MA.height = n; GPU_MA.width = m;
					GPU_MA.elements = raw_pointer_cast(device_W_mask.at(l_idx).data());
				
					dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
					dim3 dimGrid((GPU_D.width + dimBlock.x - 1) / dimBlock.x, (GPU_D.height + dimBlock.y -1) / dimBlock.y);
					CumMatMulABTKernel<<<dimGrid, dimBlock>>>(GPU_A, GPU_B, GPU_C, GPU_D, GPU_MA, sign);
				}else{
					int n = numOfNodes.at(l_idx+1);
					int m = numOfNodes.at(l_idx);
					
					float alpha = sign;
					float beta = 1;

					cublas_ret = cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, sizeOfBatchBlock, &alpha, raw_pointer_cast(device_v_out.at(l_idx).data()), sizeOfBatchBlock, raw_pointer_cast(device_delta.at(l_idx+1).data()), sizeOfBatchBlock, &beta, raw_pointer_cast(device_d_W.at(l_idx).data()), m);
					
					if (cublas_ret != CUBLAS_STATUS_SUCCESS){
						cout << "cublasSgemm returned error code " << cublas_ret << ", line(" << __LINE__ << ")" << endl;
						exit(EXIT_FAILURE);
					}
				}
			}
		}

		if(SILENT==0) end_timing("[TIM] End of tick.");
	}

	// Add a product layer at specific layer
	void addProdLayer(int numberOfProdNode, int l_idx, float scale){
		int n = numberOfProdNode;
		
		typeOfLayer.push_back(TYPE_PROD_LAYER);
		numOfNodes.push_back(n);

		device_v_out.push_back(device_vector<float>(sizeOfBatchBlock * n, 0));
		device_delta.push_back(device_vector<float>(sizeOfBatchBlock * n, 0));

		scale_factor.push_back(scale);
	}

	// read the weights for a connected product-layer at specific place
	void readProdLayerConnFromFile(string filename, int l_idx){
		if(SILENT==0) start_timing("[TIM] Start to tick!");
		if(SILENT==0) cout << "[MSG] readProdLayerConnFromFile()" << endl;

		string line;
		ifstream myfile(filename.c_str());
		long int m = numOfNodes.at(l_idx-1) + (outLayerFlag.at(l_idx-1)?outputDim:0);
			
		long int n = 0;

		if (!myfile)
		{
			cout << "[ERR] Unable to open file"; 
		}else{
			int cnt_non_zero = 0;
			//count the number of lines
			while ( myfile.good() )
			{
				getline (myfile,line);	// read a instance
				if(line.empty()) break;
				n++;

				vector<string> tokens;
				CTokenizer<CIsFromString>::Tokenize(tokens, line, CIsFromString(" "));
				cnt_non_zero += tokens.size();
			}

			device_W.push_back(device_vector<float>(0, 0));
			device_W_mask.push_back(device_vector<bool>(0, 0));
			device_d_W.push_back(device_vector<float>(0, 0));
			device_mom_d_W.push_back(device_vector<float>(0, 0));

			if(n!=numOfNodes.at(l_idx))
				cout << "[ERR] The number of node at layer " << l_idx << " does not match the line in the file" << endl;

			myfile.clear();
			myfile.seekg(0, ios::beg);

			connAdjList.at(l_idx-1).allocateMemory(n);

			connAdjList.at(l_idx-1).adjConn_cum_size.push_back(0);
			for(int i=0;i<n;i++){
				getline (myfile,line);	// read a instance
				vector<string> tokens;
				CTokenizer<CIsFromString>::Tokenize(tokens, line, CIsFromString(" "));
				connAdjList.at(l_idx-1).adjConn_size.push_back(0);
				for(int j=0;j<tokens.size();j++){
					if(atoi(tokens[j].c_str())>=m) cout << "[ERR] Weight out of range" << endl;
					
					connAdjList.at(l_idx-1).adjConn.push_back(atoi(tokens[j].c_str()));
					connAdjList.at(l_idx-1).adjConn_size[i] = connAdjList.at(l_idx-1).adjConn_size[i] + 1;
				}
				connAdjList.at(l_idx-1).adjConn_cum_size.push_back(connAdjList.at(l_idx-1).adjConn_cum_size[i] + connAdjList.at(l_idx-1).adjConn_size[i]);
			}

			connAdjList.at(l_idx-1).allocateMemory(m);

			//compute the transpose
			connTadjList.at(l_idx-1).TadjConn_cum_size.push_back(0);
			for(int i=0;i<m;i++){
				connTadjList.at(l_idx-1).TadjConn_size.push_back(0);
				for(int j=0;j<n;j++){
					for(int k=0;k<connAdjList.at(l_idx-1).adjConn_size[j];k++){
						int base = connAdjList.at(l_idx-1).adjConn_cum_size[j];
						if(connAdjList.at(l_idx-1).adjConn[base + k]==i){
							connTadjList.at(l_idx-1).TadjConn.push_back(j);
							connTadjList.at(l_idx-1).TadjConn_size[i]++;
							break;
						}
					}
				}
				connTadjList.at(l_idx-1).TadjConn_cum_size.push_back(connTadjList.at(l_idx-1).TadjConn_cum_size[i] + connTadjList.at(l_idx-1).TadjConn_size[i]);
			}

			myfile.close();
	
			if(SILENT==0) end_timing("[TIM] End of tick.");

			//buildAdjWeights(l_idx, TYPE_PROD_LAYER, n, m);
		}
	}

	void generateConnectionType1(){
		if(SILENT==0) start_timing("[TIM] Start to tick!");
		if(SILENT==0) cout << "[MSG] start to construct type-1 network" << endl;

		vector<host_vector<float> > adjConn;
		vector<host_vector<int> > adjConn_size;
		vector<host_vector<int> > adjConn_cum_size;
		for(int i=0;i<5;i++){
			adjConn.push_back(host_vector<float>());
			adjConn_size.push_back(host_vector<float>());
			adjConn_cum_size.push_back(host_vector<float>());
		}

		long int C = vocabulary.size();

		long int numOfConn = 0;
		connAdjList.at(0).allocateMemory(C);
		adjConn.at(0).resize(C*inputDim, 0);
		adjConn_size.at(0).resize(C, 0);
		adjConn_cum_size.at(0).resize(C+1, 0);
		connAdjList.at(0).adjConn.resize(C*inputDim, 0);
		connAdjList.at(0).adjConn_size.resize(C, 0);
		connAdjList.at(0).adjConn_cum_size.resize(C+1, 0);

		for(int i=0;i<C;i++){
			for(int j=0;j<inputDim;j++){
				numOfConn++;
				adjConn.at(0)[i * inputDim + j] = j;
			}
			adjConn_size.at(0)[i] = inputDim;
			adjConn_cum_size.at(0)[i + 1] = adjConn_cum_size.at(0)[i] + inputDim;
		}
		connAdjList.at(0).adjConn = adjConn.at(0);
		connAdjList.at(0).adjConn_size = adjConn_size.at(0);
		connAdjList.at(0).adjConn_cum_size = adjConn_cum_size.at(0);

		connAdjList.at(1).allocateMemory(C*K_POWER);
		adjConn.at(1).resize(C*((K_POWER+1)*K_POWER/2), 0);
		adjConn_size.at(1).resize(C*K_POWER, 0);
		adjConn_cum_size.at(1).resize(C*K_POWER+1, 0);
		connAdjList.at(1).adjConn.resize(C*((K_POWER+1)*K_POWER/2), 0);
		connAdjList.at(1).adjConn_size.resize(C*K_POWER, 0);
		connAdjList.at(1).adjConn_cum_size.resize(C*K_POWER+1, 0);

		int idx = 0;
		for(int i=0;i<C;i++){
			for(int j=0;j<K_POWER;j++){
				for(int k=0;k<j+1;k++){
					numOfConn++;
					adjConn.at(1)[idx++] = i;	
				}
				adjConn_size.at(1)[i*K_POWER+j] = j+1;
				adjConn_cum_size.at(1)[i*K_POWER+j+1] = adjConn_cum_size.at(1)[i*K_POWER+j] + j+1;
			}
		}
	
		connAdjList.at(1).adjConn = adjConn.at(1);
		connAdjList.at(1).adjConn_size = adjConn_size.at(1);
		connAdjList.at(1).adjConn_cum_size = adjConn_cum_size.at(1);

		connAdjList.at(2).allocateMemory(C);
		adjConn.at(2).resize(C*K_POWER, 0);
		adjConn_size.at(2).resize(C, 0);
		adjConn_cum_size.at(2).resize(C+1, 0);
		connAdjList.at(2).adjConn.resize(C*K_POWER, 0);
		connAdjList.at(2).adjConn_size.resize(C, 0);
		connAdjList.at(2).adjConn_cum_size.resize(C+1, 0);

		for(int i=0;i<C;i++){
			for(int j=0;j<K_POWER;j++){
				numOfConn++;
				adjConn.at(2)[i*K_POWER + j] = i*K_POWER + j;
			}
			adjConn_size.at(2)[i] = K_POWER;
			adjConn_cum_size.at(2)[i + 1] = adjConn_cum_size.at(2)[i] + K_POWER;
		}
		connAdjList.at(2).adjConn = adjConn.at(2);
		connAdjList.at(2).adjConn_size = adjConn_size.at(2);
		connAdjList.at(2).adjConn_cum_size = adjConn_cum_size.at(2);

		connAdjList.at(3).allocateMemory(C);
		adjConn.at(3).resize(C*2, 0);
		adjConn_size.at(3).resize(C, 0);
		adjConn_cum_size.at(3).resize(C+1, 0);
		connAdjList.at(3).adjConn.resize(C*2, 0);
		connAdjList.at(3).adjConn_size.resize(C, 0);
		connAdjList.at(3).adjConn_cum_size.resize(C+1, 0);

		for(int i=0;i<C;i++){
			numOfConn++;
			adjConn.at(3)[i*2] = i;
			adjConn.at(3)[i*2+1] = C+i;
			adjConn_size.at(3)[i] = 2;
			adjConn_cum_size.at(3)[i + 1] = adjConn_cum_size.at(3)[i] + 2;
		}

		connAdjList.at(3).adjConn = adjConn.at(3);
		connAdjList.at(3).adjConn_size = adjConn_size.at(3);
		connAdjList.at(3).adjConn_cum_size = adjConn_cum_size.at(3);

		connAdjList.at(4).allocateMemory(1);
		adjConn.at(4).resize(C, 0);
		adjConn_size.at(4).resize(1, 0);
		adjConn_cum_size.at(4).resize(2, 0);
		connAdjList.at(4).adjConn.resize(C, 0);
		connAdjList.at(4).adjConn_size.resize(1, 0);
		connAdjList.at(4).adjConn_cum_size.resize(2, 0);
		for(int i=0;i<1;i++){
			for(int j=0;j<C;j++){
				numOfConn++;
				adjConn.at(4)[i*C + j] = i*C + j;
			}

			adjConn_size.at(4)[i] = C;
			adjConn_cum_size.at(4)[i + 1] = adjConn_cum_size.at(4)[i] + C;
		}
		connAdjList.at(4).adjConn = adjConn.at(4);
		connAdjList.at(4).adjConn_size = adjConn_size.at(4);
		connAdjList.at(4).adjConn_cum_size = adjConn_cum_size.at(4);

		if(SILENT==0) end_timing("1. [TIM] End of tick.");

		cout << "[MSG] Number of connections: " << numOfConn << endl;

		vector<host_vector<float> > TadjConn;
		vector<host_vector<int> > TadjConn_size;
		vector<host_vector<int> > TadjConn_cum_size;
		for(int i=0;i<5;i++){
			TadjConn.push_back(host_vector<float>());
			TadjConn_size.push_back(host_vector<float>());
			TadjConn_cum_size.push_back(host_vector<float>());
		}

		for(int l_idx=0;l_idx<5;l_idx++){
			if(SILENT==0) start_timing("[TIM] Start to tick!");
			cout << "[MSG] find the transpose matrix for layer " << l_idx << endl;
			long int n = numOfNodes.at(l_idx+1);
			long int m = numOfNodes.at(l_idx) + (outLayerFlag.at(l_idx)?outputDim:0);
			vector<std::pair<int,int> > coor;
			
			//compute the transpose
			for(int i=0;i<adjConn_size.at(l_idx).size();i++){
				for(int j=0;j<adjConn_size.at(l_idx)[i];j++){
					coor.push_back(std::make_pair(i, adjConn.at(l_idx)[adjConn_cum_size.at(l_idx)[i]+j]));
				}
			}
			
			if(SILENT==0) cout << "[MSG] finished prepare coor whose size is " << coor.size() << endl;

			stable_sort(coor.begin(), coor.end(), sort_idx);

			if(SILENT==0) end_timing("2. [TIM] End of tick.");

			connTadjList.at(l_idx).allocateMemory(m);
			TadjConn_cum_size.at(l_idx).push_back(0);
			int idx = 0;
			TadjConn_size.at(l_idx).push_back(0);
			for(int i=0;i<coor.size();i++){
				if(idx!=coor[i].second){
					TadjConn_cum_size.at(l_idx).push_back(TadjConn_size.at(l_idx)[idx] + TadjConn_cum_size.at(l_idx)[idx]);
					TadjConn_size.at(l_idx).push_back(0);
					idx = coor[i].second;
				}
				if(idx==coor[i].second){
					TadjConn.at(l_idx).push_back(coor[i].first);
					TadjConn_size.at(l_idx)[idx]++;
				}
			}
			TadjConn_cum_size.at(l_idx).push_back(TadjConn_size.at(l_idx)[idx] + TadjConn_cum_size.at(l_idx)[idx]);

			connTadjList.at(l_idx).TadjConn = TadjConn.at(l_idx);
			connTadjList.at(l_idx).TadjConn_size = TadjConn_size.at(l_idx);
			connTadjList.at(l_idx).TadjConn_cum_size = TadjConn_cum_size.at(l_idx);

			cout << "[MSG] finished the transpose matrix for layer " << l_idx << endl;

			if(SILENT==0) end_timing("3. [TIM] End of tick.");
		}

		for(int l_idx=0;l_idx<5;l_idx++){
			long int n = numOfNodes.at(l_idx+1);
			long int m = numOfNodes.at(l_idx) + (outLayerFlag.at(l_idx)?outputDim:0);

			//temporary shortcut code for saving the memory which is very sparse
			if(l_idx==2){
				device_W.push_back(device_vector<float>(K_POWER*vocabulary.size(), 0));
				device_W_mask.push_back(device_vector<bool>(0, 0));
				device_d_W.push_back(device_vector<float>(K_POWER*vocabulary.size(), 0));
				device_mom_d_W.push_back(device_vector<float>(0, 0));
                                
                                //initial the weight
                                for(int i=0;i<vocabulary.size();i++){
                                    int base = 1;
                                    for(int k=0;k<K_POWER;k++){
                                        base *= (k+1);
                                        device_W.at(l_idx)[i*K_POWER + k] = 1.0/base;
                                        //device_W.at(l_idx)[i*K_POWER + k] = (double)rand()/RAND_MAX;
                                    }
                                }
                                
                                device_W2_bias.resize(vocabulary.size(), 0);
                                for(int i=0;i<device_W2_bias.size();i++)
                                    device_W2_bias[i] = 1;
                                device_d_W2_bias.resize(vocabulary.size(), 0);
                                
				continue;
			}

			//assign matrix for SUM node
			if(typeOfLayer.at(l_idx+1)==TYPE_SUM_LAYER){
				if(SILENT==0) cout << " assign weight memory for sum layer " << l_idx << endl;

				device_W.push_back(device_vector<float>(n*m, 0));
				device_W_mask.push_back(device_vector<bool>(n*m, 0));
				device_d_W.push_back(device_vector<float>(n*m, 0));
				device_mom_d_W.push_back(device_vector<float>(n*m, 0));

				GPU_A_Adj.height = connAdjList.at(l_idx).getHeight();
				GPU_A_Adj.adjW = raw_pointer_cast(connAdjList.at(l_idx).adjConn.data());
				GPU_A_Adj.adjW_size = raw_pointer_cast(connAdjList.at(l_idx).adjConn_size.data());
				GPU_A_Adj.adjW_cum_size = raw_pointer_cast(connAdjList.at(l_idx).adjConn_cum_size.data());

				setGPU_Matrix(&GPU_A, n, m, raw_pointer_cast(device_W.at(l_idx).data()));

				initializeW<<<(GPU_A.height + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(GPU_A_Adj, GPU_A, true, rand_seed);
				if(SILENT==0) end_timing("4. [TIM] End of tick.");
			}
			if(typeOfLayer.at(l_idx+1)==TYPE_PROD_LAYER){
				if(SILENT==0) cout << " assign weight memory for product layer " << l_idx << endl;

				device_W.push_back(device_vector<float>(0, 0));
				device_W_mask.push_back(device_vector<bool>(0, 0));
				device_d_W.push_back(device_vector<float>(0, 0));
				device_mom_d_W.push_back(device_vector<float>(0, 0));
			}
		}

		//preset the weights
		/*for(int i=0;i<adjConn_size.at(2).size();i++){
			int y = adjConn.at(2)[adjConn_cum_size.at(2)[i] + 0];
			device_W.at(2)[i*numOfNodes.at(2) + y] = 1;
			for(int j=1;j<adjConn_size.at(2)[i];j++){
				device_W.at(2)[i*numOfNodes.at(2) + adjConn.at(2)[adjConn_cum_size.at(2)[i] + j]] = device_W.at(2)[i*numOfNodes.at(2) + adjConn.at(2)[adjConn_cum_size.at(2)[i] + j-1]] / (j+1);
			}
		}*/

		for(int i=0;i<device_W.at(4).size();i++){
			device_W.at(4)[i] = 1;
		}
/*
                for(int i=0;i<min((int)numOfNodes.at(1), 5);i++){
                    for(int j=0;j<min((int)numOfNodes.at(0), 5);j++){
                        cout << device_W.at(0)[i*numOfNodes.at(0) + j] << " ";
                    }
                    cout << endl;
                }
*/
		if(SILENT==0) end_timing("6. [TIM] End of tick.");
	}

	void generateConnectionType3(int T, int P, int Rep){
		if(SILENT==0) start_timing("[TIM] Start to tick!");
		if(SILENT==0) cout << "[MSG] start to construct type-3 network" << endl;
		
		int L1 = T, L2 = P;
		
		vector<host_vector<float> > adjConn;
		vector<host_vector<int> > adjConn_size;
		vector<host_vector<int> > adjConn_cum_size;
		for(int i=0;i<5;i++){
			adjConn.push_back(host_vector<float>());
			adjConn_size.push_back(host_vector<float>());
			adjConn_cum_size.push_back(host_vector<float>());
		}

		int C = vocabulary.size();

		long int numOfConn = 0;
		connAdjList.at(0).allocateMemory(L1);
		adjConn.at(0).resize(L1*inputDim, 0);
		adjConn_size.at(0).resize(L1, 0);
		adjConn_cum_size.at(0).resize(L1+1, 0);
		connAdjList.at(0).adjConn.resize(L1*inputDim, 0);
		connAdjList.at(0).adjConn_size.resize(L1, 0);
		connAdjList.at(0).adjConn_cum_size.resize(L1+1, 0);

		for(int i=0;i<L1;i++){
			for(int j=0;j<inputDim;j++){
				numOfConn++;
				adjConn.at(0)[i * inputDim + j] = j;
			}
			adjConn_size.at(0)[i] = inputDim;
			adjConn_cum_size.at(0)[i + 1] = adjConn_cum_size.at(0)[i] + inputDim;
		}
		connAdjList.at(0).adjConn = adjConn.at(0);
		connAdjList.at(0).adjConn_size = adjConn_size.at(0);
		connAdjList.at(0).adjConn_cum_size = adjConn_cum_size.at(0);

		connAdjList.at(1).allocateMemory(L2);
		adjConn.at(1).resize(L2*L1, 0);
		adjConn_size.at(1).resize(L2, 0);
		adjConn_cum_size.at(1).resize(L2+1, 0);
		connAdjList.at(1).adjConn.resize(L2*L1, 0);
		connAdjList.at(1).adjConn_size.resize(L2, 0);
		connAdjList.at(1).adjConn_cum_size.resize(L2+1, 0);
		for(int i=0;i<L2;i++){
			for(int j=0;j<L1;j++){
				numOfConn++;
				adjConn.at(1)[i*L1 + j] = j;
			}
			adjConn_size.at(1)[i] = L1;
			adjConn_cum_size.at(1)[i + 1] = adjConn_cum_size.at(1)[i] + L1;
		}
		connAdjList.at(1).adjConn = adjConn.at(1);
		connAdjList.at(1).adjConn_size = adjConn_size.at(1);
		connAdjList.at(1).adjConn_cum_size = adjConn_cum_size.at(1);

		connAdjList.at(2).allocateMemory(Rep*C);
		adjConn.at(2).resize(Rep*C*L2, 0);
		adjConn_size.at(2).resize(Rep*C, 0);
		adjConn_cum_size.at(2).resize(Rep*C+1, 0);
		connAdjList.at(2).adjConn.resize(Rep*C*L2, 0);
		connAdjList.at(2).adjConn_size.resize(Rep*C, 0);
		connAdjList.at(2).adjConn_cum_size.resize(Rep*C+1, 0);
		for(int i=0;i<Rep*C;i++){
			for(int j=0;j<L2;j++){
				numOfConn++;
				adjConn.at(2)[i*L2 + j] = j;
			}
			adjConn_size.at(2)[i] = L2;
			adjConn_cum_size.at(2)[i + 1] = adjConn_cum_size.at(2)[i] + L2;
		}
		connAdjList.at(2).adjConn = adjConn.at(2);
		connAdjList.at(2).adjConn_size = adjConn_size.at(2);
		connAdjList.at(2).adjConn_cum_size = adjConn_cum_size.at(2);

		connAdjList.at(3).allocateMemory(C);
		adjConn.at(3).resize(C*(Rep+1), 0);
		adjConn_size.at(3).resize(C, 0);
		adjConn_cum_size.at(3).resize(C+1, 0);
		connAdjList.at(3).adjConn.resize(C*(Rep+1), 0);
		connAdjList.at(3).adjConn_size.resize(C, 0);
		connAdjList.at(3).adjConn_cum_size.resize(C+1, 0);
		for(int i=0;i<C;i++){
			for(int j=0;j<Rep;j++){
				numOfConn++;
				adjConn.at(3)[i*(Rep+1) + j] = i*Rep + j;
			}
			numOfConn++;
			adjConn.at(3)[i*(Rep+1) + Rep] = connAdjList.at(2).getHeight() + i;
			adjConn_size.at(3)[i] = (Rep+1);
			adjConn_cum_size.at(3)[i + 1] = adjConn_cum_size.at(3)[i] + (Rep+1);
		}
		connAdjList.at(3).adjConn = adjConn.at(3);
		connAdjList.at(3).adjConn_size = adjConn_size.at(3);
		connAdjList.at(3).adjConn_cum_size = adjConn_cum_size.at(3);

		connAdjList.at(4).allocateMemory(1);
		adjConn.at(4).resize(C, 0);
		adjConn_size.at(4).resize(1, 0);
		adjConn_cum_size.at(4).resize(2, 0);
		connAdjList.at(4).adjConn.resize(C, 0);
		connAdjList.at(4).adjConn_size.resize(1, 0);
		connAdjList.at(4).adjConn_cum_size.resize(2, 0);
		for(int i=0;i<1;i++){
			for(int j=0;j<C;j++){
				numOfConn++;
				adjConn.at(4)[i*C + j] = i*C + j;
			}

			adjConn_size.at(4)[i] = C;
			adjConn_cum_size.at(4)[i + 1] = adjConn_cum_size.at(4)[i] + C;
		}
		connAdjList.at(4).adjConn = adjConn.at(4);
		connAdjList.at(4).adjConn_size = adjConn_size.at(4);
		connAdjList.at(4).adjConn_cum_size = adjConn_cum_size.at(4);

		if(SILENT==0) end_timing("1. [TIM] End of tick.");

		cout << "[MSG] Number of connections: " << numOfConn << endl;

		if(SILENT==0) end_timing("2. [TIM] End of tick.");

		vector<host_vector<float> > TadjConn;
		vector<host_vector<int> > TadjConn_size;
		vector<host_vector<int> > TadjConn_cum_size;
		for(int i=0;i<5;i++){
			TadjConn.push_back(host_vector<float>());
			TadjConn_size.push_back(host_vector<float>());
			TadjConn_cum_size.push_back(host_vector<float>());
		}

		for(int l_idx=0;l_idx<5;l_idx++){
			if(SILENT==0) start_timing("[TIM] Start to tick!");
			cout << "[MSG] find the transpose matrix for layer " << l_idx << endl;
			long int n = numOfNodes.at(l_idx+1);
			long int m = numOfNodes.at(l_idx) + (outLayerFlag.at(l_idx)?outputDim:0);
			vector<std::pair<int,int> > coor;
			
			//compute the transpose
			for(int i=0;i<adjConn_size.at(l_idx).size();i++){
				for(int j=0;j<adjConn_size.at(l_idx)[i];j++){
					coor.push_back(std::make_pair(i, adjConn.at(l_idx)[adjConn_cum_size.at(l_idx)[i]+j]));
				}
			}
			
			if(SILENT==0) cout << "[MSG] finished prepare coor whose size is " << coor.size() << endl;

			stable_sort(coor.begin(), coor.end(), sort_idx);

			if(SILENT==0) end_timing("2. [TIM] End of tick.");

			connTadjList.at(l_idx).allocateMemory(m);
			TadjConn_cum_size.at(l_idx).push_back(0);
			int idx = 0;
			TadjConn_size.at(l_idx).push_back(0);
			for(int i=0;i<coor.size();i++){
				if(idx!=coor[i].second){
					TadjConn_cum_size.at(l_idx).push_back(TadjConn_size.at(l_idx)[idx] + TadjConn_cum_size.at(l_idx)[idx]);
					TadjConn_size.at(l_idx).push_back(0);
					idx = coor[i].second;
				}
				if(idx==coor[i].second){
					TadjConn.at(l_idx).push_back(coor[i].first);
					TadjConn_size.at(l_idx)[idx]++;
				}
			}
			TadjConn_cum_size.at(l_idx).push_back(TadjConn_size.at(l_idx)[idx] + TadjConn_cum_size.at(l_idx)[idx]);

			connTadjList.at(l_idx).TadjConn = TadjConn.at(l_idx);
			connTadjList.at(l_idx).TadjConn_size = TadjConn_size.at(l_idx);
			connTadjList.at(l_idx).TadjConn_cum_size = TadjConn_cum_size.at(l_idx);

			cout << "[MSG] finished the transpose matrix for layer " << l_idx << endl;

			if(SILENT==0) end_timing("3. [TIM] End of tick.");
		}


		for(int l_idx=0;l_idx<5;l_idx++){
			long int n = numOfNodes.at(l_idx+1);
			long int m = numOfNodes.at(l_idx) + (outLayerFlag.at(l_idx)?outputDim:0);
			//assign matrix for SUM node
			if(typeOfLayer.at(l_idx+1)==TYPE_SUM_LAYER){
				device_W.push_back(device_vector<float>(n*m, 0));
				device_W_mask.push_back(device_vector<bool>(n*m, 0));
				device_d_W.push_back(device_vector<float>(n*m, 0));
				device_mom_d_W.push_back(device_vector<float>(n*m, 0));

				GPU_A_Adj.height = connAdjList.at(l_idx).getHeight();
				GPU_A_Adj.adjW = raw_pointer_cast(connAdjList.at(l_idx).adjConn.data());
				GPU_A_Adj.adjW_size = raw_pointer_cast(connAdjList.at(l_idx).adjConn_size.data());
				GPU_A_Adj.adjW_cum_size = raw_pointer_cast(connAdjList.at(l_idx).adjConn_cum_size.data());

				setGPU_Matrix(&GPU_A, n, m, raw_pointer_cast(device_W.at(l_idx).data()));

				initializeW<<<(GPU_A.height + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(GPU_A_Adj, GPU_A, norm_weight_flag, time(NULL));
				if(SILENT==0) end_timing("4. [TIM] End of tick.");
			}
			if(typeOfLayer.at(l_idx+1)==TYPE_PROD_LAYER){
				device_W.push_back(device_vector<float>(0, 0));
				device_W_mask.push_back(device_vector<bool>(0, 0));
				device_d_W.push_back(device_vector<float>(0, 0));
				device_mom_d_W.push_back(device_vector<float>(0, 0));
			}
		}

		if(SILENT==0) end_timing("6. [TIM] End of tick.");
	}

	void generateConnectionType4(int L1, int L3, int L5, int Rep){
		if(SILENT==0) start_timing("[TIM] Start to tick!");
		if(SILENT==0) cout << "[MSG] start to construct type-4 network" << endl;
		
		vector<host_vector<float> > adjConn;
		vector<host_vector<int> > adjConn_size;
		vector<host_vector<int> > adjConn_cum_size;
		for(int i=0;i<8;i++){
			adjConn.push_back(host_vector<float>());
			adjConn_size.push_back(host_vector<float>());
			adjConn_cum_size.push_back(host_vector<float>());
		}

		int C = vocabulary.size();

		long int numOfConn = 0;
		connAdjList.at(0).allocateMemory(L1);
		adjConn.at(0).resize(L1*inputDim, 0);
		adjConn_size.at(0).resize(L1, 0);
		adjConn_cum_size.at(0).resize(L1+1, 0);
		connAdjList.at(0).adjConn.resize(L1*inputDim, 0);
		connAdjList.at(0).adjConn_size.resize(L1, 0);
		connAdjList.at(0).adjConn_cum_size.resize(L1+1, 0);

		for(int i=0;i<L1;i++){
			for(int j=0;j<inputDim;j++){
				numOfConn++;
				adjConn.at(0)[i * inputDim + j] = j;
			}
			adjConn_size.at(0)[i] = inputDim;
			adjConn_cum_size.at(0)[i + 1] = adjConn_cum_size.at(0)[i] + inputDim;
		}
		connAdjList.at(0).adjConn = adjConn.at(0);
		connAdjList.at(0).adjConn_size = adjConn_size.at(0);
		connAdjList.at(0).adjConn_cum_size = adjConn_cum_size.at(0);

		connAdjList.at(1).allocateMemory(L1 + L1*(L1-1)/2);
		adjConn.at(1).resize(L1 + L1*(L1-1), 0);
		adjConn_size.at(1).resize(L1 + L1*(L1-1)/2, 0);
		adjConn_cum_size.at(1).resize(L1 + L1*(L1-1)/2+1, 0);
		connAdjList.at(1).adjConn.resize(L1 + L1*(L1-1), 0);
		connAdjList.at(1).adjConn_size.resize(L1 + L1*(L1-1)/2, 0);
		connAdjList.at(1).adjConn_cum_size.resize(L1 + L1*(L1-1)/2+1, 0);

		int idx = 0;
		for(int j=0;j<L1;j++){
			numOfConn++;
			adjConn.at(1)[idx] = j;
			adjConn_size.at(1)[idx] = 1;
			adjConn_cum_size.at(1)[idx + 1] = adjConn_cum_size.at(1)[idx] + 1;
			idx++;
		}

		for(int i=0;i<L1;i++){
			for(int j=i+1;j<L1;j++){
				numOfConn++;
				adjConn.at(1)[adjConn_cum_size.at(1)[idx]] = i;
				adjConn.at(1)[adjConn_cum_size.at(1)[idx]+1] = j;
				adjConn_size.at(1)[idx] = 2;
				adjConn_cum_size.at(1)[idx + 1] = adjConn_cum_size.at(1)[idx] + 2;
				idx++;
			}
		}
		
		connAdjList.at(1).adjConn = adjConn.at(1);
		connAdjList.at(1).adjConn_size = adjConn_size.at(1);
		connAdjList.at(1).adjConn_cum_size = adjConn_cum_size.at(1);

		if(numOfNodes.at(2)!=L1 + L1*(L1-1)/2){
			cout << "Number of neurons does not match at Layer 2" << endl;
			exit(0);
		}

		connAdjList.at(2).allocateMemory(L3);
		adjConn.at(2).resize(L3*connAdjList.at(1).getHeight(), 0);
		adjConn_size.at(2).resize(L3, 0);
		adjConn_cum_size.at(2).resize(L3+1, 0);
		connAdjList.at(2).adjConn.resize(L3*connAdjList.at(1).getHeight(), 0);
		connAdjList.at(2).adjConn_size.resize(L3, 0);
		connAdjList.at(2).adjConn_cum_size.resize(L3+1, 0);

		for(int i=0;i<L3;i++){
			for(int j=0;j<adjConn_size.at(1).size();j++){
				numOfConn++;
				adjConn.at(2)[i * connAdjList.at(1).getHeight() + j] = j;
			}
			adjConn_size.at(2)[i] = connAdjList.at(1).getHeight();
			adjConn_cum_size.at(2)[i + 1] = adjConn_cum_size.at(2)[i] + connAdjList.at(1).getHeight();
		}
		connAdjList.at(2).adjConn = adjConn.at(2);
		connAdjList.at(2).adjConn_size = adjConn_size.at(2);
		connAdjList.at(2).adjConn_cum_size = adjConn_cum_size.at(2);
		
		connAdjList.at(3).allocateMemory(L3 + L3*(L3-1)/2);
		adjConn.at(3).resize(L3 + L3*(L3-1), 0);
		adjConn_size.at(3).resize(L3 + L3*(L3-1)/2, 0);
		adjConn_cum_size.at(3).resize(L3 + L3*(L3-1)/2+1, 0);
		connAdjList.at(3).adjConn.resize(L3 + L3*(L3-1), 0);
		connAdjList.at(3).adjConn_size.resize(L3 + L3*(L3-1)/2, 0);
		connAdjList.at(3).adjConn_cum_size.resize(L3 + L3*(L3-1)/2+1, 0);

		idx = 0;
		for(int j=0;j<L3;j++){
			numOfConn++;
			adjConn.at(3)[idx] = j;
			adjConn_size.at(3)[idx] = 1;
			adjConn_cum_size.at(3)[idx + 1] = adjConn_cum_size.at(3)[idx] + 1;
			idx++;
		}

		for(int i=0;i<L3;i++){
			for(int j=i+1;j<L3;j++){
				numOfConn++;
				adjConn.at(3)[adjConn_cum_size.at(3)[idx]] = i;
				adjConn.at(3)[adjConn_cum_size.at(3)[idx]+1] = j;
				adjConn_size.at(3)[idx] = 2;
				adjConn_cum_size.at(3)[idx + 1] = adjConn_cum_size.at(3)[idx] + 2;
				idx++;
			}
		}
		
		connAdjList.at(3).adjConn = adjConn.at(3);
		connAdjList.at(3).adjConn_size = adjConn_size.at(3);
		connAdjList.at(3).adjConn_cum_size = adjConn_cum_size.at(3);

		if(numOfNodes.at(4)!=L3 + L3*(L3-1)/2){
			cout << "Number of neurons does not match at Layer 4" << endl;
			exit(0);
		}


		connAdjList.at(4).allocateMemory(L5);
		adjConn.at(4).resize(L5*connAdjList.at(3).getHeight(), 0);
		adjConn_size.at(4).resize(L5, 0);
		adjConn_cum_size.at(4).resize(L5+1, 0);
		connAdjList.at(4).adjConn.resize(L5*connAdjList.at(3).getHeight(), 0);
		connAdjList.at(4).adjConn_size.resize(L5, 0);
		connAdjList.at(4).adjConn_cum_size.resize(L5+1, 0);

		for(int i=0;i<L5;i++){
			for(int j=0;j<adjConn_size.at(3).size();j++){
				numOfConn++;
				adjConn.at(4)[i * connAdjList.at(3).getHeight() + j] = j;
			}
			adjConn_size.at(4)[i] = connAdjList.at(3).getHeight();
			adjConn_cum_size.at(4)[i + 1] = adjConn_cum_size.at(4)[i] + connAdjList.at(3).getHeight();
		}
		connAdjList.at(4).adjConn = adjConn.at(4);
		connAdjList.at(4).adjConn_size = adjConn_size.at(4);
		connAdjList.at(4).adjConn_cum_size = adjConn_cum_size.at(4);

		connAdjList.at(5).allocateMemory(Rep*C);
		adjConn.at(5).resize(Rep*C*connAdjList.at(4).getHeight(), 0);
		adjConn_size.at(5).resize(Rep*C, 0);
		adjConn_cum_size.at(5).resize(Rep*C+1, 0);
		connAdjList.at(5).adjConn.resize(Rep*C*connAdjList.at(4).getHeight(), 0);
		connAdjList.at(5).adjConn_size.resize(Rep*C, 0);
		connAdjList.at(5).adjConn_cum_size.resize(Rep*C+1, 0);

		for(int i=0;i<Rep*C;i++){
			for(int j=0;j<adjConn_size.at(4).size();j++){
				numOfConn++;
				adjConn.at(5)[i*adjConn_size.at(4).size() + j] = j;
			}
			adjConn_size.at(5)[i] = connAdjList.at(4).getHeight();
			adjConn_cum_size.at(5)[i + 1] = adjConn_cum_size.at(5)[i] + connAdjList.at(4).getHeight();
		}
		connAdjList.at(5).adjConn = adjConn.at(5);
		connAdjList.at(5).adjConn_size = adjConn_size.at(5);
		connAdjList.at(5).adjConn_cum_size = adjConn_cum_size.at(5);

		connAdjList.at(6).allocateMemory(C);
		adjConn.at(6).resize(C*(Rep+1), 0);
		adjConn_size.at(6).resize(C, 0);
		adjConn_cum_size.at(6).resize(C+1, 0);
		connAdjList.at(6).adjConn.resize(C*(Rep+1), 0);
		connAdjList.at(6).adjConn_size.resize(C, 0);
		connAdjList.at(6).adjConn_cum_size.resize(C+1, 0);
		for(int i=0;i<C;i++){
			for(int j=0;j<Rep;j++){
				numOfConn++;
				adjConn.at(6)[i*(Rep+1) + j] = i*Rep + j;
			}
			numOfConn++;
			adjConn.at(6)[i*(Rep+1) + Rep] = connAdjList.at(5).getHeight() + i;
			adjConn_size.at(6)[i] = (Rep+1);
			adjConn_cum_size.at(6)[i + 1] = adjConn_cum_size.at(6)[i] + (Rep+1);
		}
		connAdjList.at(6).adjConn = adjConn.at(6);
		connAdjList.at(6).adjConn_size = adjConn_size.at(6);
		connAdjList.at(6).adjConn_cum_size = adjConn_cum_size.at(6);

		connAdjList.at(7).allocateMemory(1);
		adjConn.at(7).resize(C, 0);
		adjConn_size.at(7).resize(1, 0);
		adjConn_cum_size.at(7).resize(2, 0);
		connAdjList.at(7).adjConn.resize(C, 0);
		connAdjList.at(7).adjConn_size.resize(1, 0);
		connAdjList.at(7).adjConn_cum_size.resize(2, 0);
		for(int i=0;i<1;i++){
			for(int j=0;j<C;j++){
				numOfConn++;
				adjConn.at(7)[i*C + j] = i*C + j;
			}

			adjConn_size.at(7)[i] = C;
			adjConn_cum_size.at(7)[i + 1] = adjConn_cum_size.at(7)[i] + C;
		}
		connAdjList.at(7).adjConn = adjConn.at(7);
		connAdjList.at(7).adjConn_size = adjConn_size.at(7);
		connAdjList.at(7).adjConn_cum_size = adjConn_cum_size.at(7);

		if(SILENT==0) end_timing("1. [TIM] End of tick.");

		cout << "[MSG] Number of connections: " << numOfConn << endl;

		if(SILENT==0) end_timing("2. [TIM] End of tick.");

		vector<host_vector<float> > TadjConn;
		vector<host_vector<int> > TadjConn_size;
		vector<host_vector<int> > TadjConn_cum_size;
		for(int i=0;i<8;i++){
			TadjConn.push_back(host_vector<float>());
			TadjConn_size.push_back(host_vector<float>());
			TadjConn_cum_size.push_back(host_vector<float>());
		}

		for(int l_idx=0;l_idx<8;l_idx++){
			if(SILENT==0) start_timing("[TIM] Start to tick!");
			cout << "[MSG] find the transpose matrix for layer " << l_idx << endl;
			long int n = numOfNodes.at(l_idx+1);
			long int m = numOfNodes.at(l_idx) + (outLayerFlag.at(l_idx)?outputDim:0);
			vector<std::pair<int,int> > coor;
			
			//compute the transpose
			for(int i=0;i<adjConn_size.at(l_idx).size();i++){
				for(int j=0;j<adjConn_size.at(l_idx)[i];j++){
					coor.push_back(std::make_pair(i, adjConn.at(l_idx)[adjConn_cum_size.at(l_idx)[i]+j]));
				}
			}
			
			if(SILENT==0) cout << "[MSG] finished prepare coor whose size is " << coor.size() << endl;

			stable_sort(coor.begin(), coor.end(), sort_idx);

			if(SILENT==0) end_timing("2. [TIM] End of tick.");

			connTadjList.at(l_idx).allocateMemory(m);
			TadjConn_cum_size.at(l_idx).push_back(0);
			int idx = 0;
			TadjConn_size.at(l_idx).push_back(0);
			for(int i=0;i<coor.size();i++){
				if(idx!=coor[i].second){
					TadjConn_cum_size.at(l_idx).push_back(TadjConn_size.at(l_idx)[idx] + TadjConn_cum_size.at(l_idx)[idx]);
					TadjConn_size.at(l_idx).push_back(0);
					idx = coor[i].second;
				}
				if(idx==coor[i].second){
					TadjConn.at(l_idx).push_back(coor[i].first);
					TadjConn_size.at(l_idx)[idx]++;
				}
			}
			TadjConn_cum_size.at(l_idx).push_back(TadjConn_size.at(l_idx)[idx] + TadjConn_cum_size.at(l_idx)[idx]);

			connTadjList.at(l_idx).TadjConn = TadjConn.at(l_idx);
			connTadjList.at(l_idx).TadjConn_size = TadjConn_size.at(l_idx);
			connTadjList.at(l_idx).TadjConn_cum_size = TadjConn_cum_size.at(l_idx);

			cout << "[MSG] finished the transpose matrix for layer " << l_idx << endl;

			if(SILENT==0) end_timing("3. [TIM] End of tick.");
		}


		for(int l_idx=0;l_idx<8;l_idx++){
			long int n = numOfNodes.at(l_idx+1);
			long int m = numOfNodes.at(l_idx) + (outLayerFlag.at(l_idx)?outputDim:0);
			//assign matrix for SUM node
			if(typeOfLayer.at(l_idx+1)==TYPE_SUM_LAYER){
				device_W.push_back(device_vector<float>(n*m, 0));
				device_W_mask.push_back(device_vector<bool>(n*m, 0));
				device_d_W.push_back(device_vector<float>(n*m, 0));
				device_mom_d_W.push_back(device_vector<float>(n*m, 0));

				GPU_A_Adj.height = connAdjList.at(l_idx).getHeight();
				GPU_A_Adj.adjW = raw_pointer_cast(connAdjList.at(l_idx).adjConn.data());
				GPU_A_Adj.adjW_size = raw_pointer_cast(connAdjList.at(l_idx).adjConn_size.data());
				GPU_A_Adj.adjW_cum_size = raw_pointer_cast(connAdjList.at(l_idx).adjConn_cum_size.data());

				setGPU_Matrix(&GPU_A, n, m, raw_pointer_cast(device_W.at(l_idx).data()));

				initializeW<<<(GPU_A.height + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(GPU_A_Adj, GPU_A, norm_weight_flag, time(NULL));
				if(SILENT==0) end_timing("4. [TIM] End of tick.");
			}
			if(typeOfLayer.at(l_idx+1)==TYPE_PROD_LAYER){
				device_W.push_back(device_vector<float>(0, 0));
				device_W_mask.push_back(device_vector<bool>(0, 0));
				device_d_W.push_back(device_vector<float>(0, 0));
				device_mom_d_W.push_back(device_vector<float>(0, 0));
			}
		}

		if(SILENT==0) end_timing("6. [TIM] End of tick.");
	}

	// read the weights for a connected product-layer at specific place
	void readSumLayerConnFromFile(string filename, int l_idx){
		if(SILENT==0) start_timing("[TIM] Start to tick!");
		if(SILENT==0) cout << "[MSG] readSumLayerConnFromFile()" << endl;
	
		string line;
		ifstream myfile(filename.c_str());
		int n = 0;
		int m = numOfNodes.at(l_idx-1) + (outLayerFlag.at(l_idx-1)?outputDim:0);

		if (!myfile){
			cout << "[ERR] Unable to open file"; 
		}else{
			//count the number of lines
			int cnt_non_zero = 0;
			while ( myfile.good() )
			{
				getline (myfile,line);	// read a instance
				if(line.empty()) break;
				n++;

				vector<string> tokens;
				CTokenizer<CIsFromString>::Tokenize(tokens, line, CIsFromString(" "));
				cnt_non_zero += tokens.size();
			}

			if(n!=numOfNodes.at(l_idx))
				cout << "[ERR] The number of node (" << numOfNodes.at(l_idx) << ") at layer " << l_idx << " does not match the line (" << n << ") in the file" << endl;

			device_W.push_back(device_vector<float>(n * m, 0));
			device_W_mask.push_back(device_vector<bool>(n * m, 0));
			device_d_W.push_back(device_vector<float>(n * m, 0));
			device_mom_d_W.push_back(device_vector<float>(n * m, 0));

			//reset the to the head of file
			myfile.clear();
			myfile.seekg(0, ios::beg);

			connAdjList.at(l_idx-1).allocateMemory(n);

			cnt_non_zero = 0;
			connAdjList.at(l_idx-1).adjConn_cum_size.push_back(0);
			for(int i=0;i<n;i++){
				getline (myfile,line);	// read a instance
				vector<string> tokens;
				CTokenizer<CIsFromString>::Tokenize(tokens, line, CIsFromString(" "));
				connAdjList.at(l_idx-1).adjConn_size.push_back(0);
				for(int j=0;j<tokens.size();j++){
					if(atoi(tokens[j].c_str())>=m) cout << "[ERR] Weight out of range" << endl;
					device_W.at(l_idx-1)[i*m + atoi(tokens[j].c_str())] = (double)rand()/RAND_MAX;
					device_W_mask.at(l_idx-1)[i*m + atoi(tokens[j].c_str())] = true;
					connAdjList.at(l_idx-1).adjConn.push_back(atoi(tokens[j].c_str()));
					connAdjList.at(l_idx-1).adjConn_size[i] = connAdjList.at(l_idx-1).adjConn_size[i] + 1;
				}
				connAdjList.at(l_idx-1).adjConn_cum_size.push_back(connAdjList.at(l_idx-1).adjConn_cum_size[i] + connAdjList.at(l_idx-1).adjConn_size[i]);
				
				//normalize
				double z = 0;
				for(int j=0;j<tokens.size();j++) z += device_W.at(l_idx-1)[i*m + atoi(tokens[j].c_str())];
				for(int j=0;j<tokens.size();j++) device_W.at(l_idx-1)[i*m + atoi(tokens[j].c_str())]/=z;
			}

			connTadjList.at(l_idx-1).allocateMemory(m);
			//compute the transpose
			connTadjList.at(l_idx-1).TadjConn_cum_size.push_back(0);
			for(int i=0;i<m;i++){
				connTadjList.at(l_idx-1).TadjConn_size.push_back(0);
				for(int j=0;j<n;j++){
					for(int k=0;k<connAdjList.at(l_idx-1).adjConn_size[j];k++){
						int base = connAdjList.at(l_idx-1).adjConn_cum_size[j];
						if(connAdjList.at(l_idx-1).adjConn[base + k]==i){
							connTadjList.at(l_idx-1).TadjConn.push_back(j);
							connTadjList.at(l_idx-1).TadjConn_size[i]++;
							break;
						}
					}
				}
				connTadjList.at(l_idx-1).TadjConn_cum_size.push_back(connTadjList.at(l_idx-1).TadjConn_cum_size[i] + connTadjList.at(l_idx-1).TadjConn_size[i]);
			}

			myfile.close();

			if(SILENT==0) end_timing("[TIM] End of tick.");

			//buildAdjWeights(l_idx, TYPE_SUM_LAYER, n, m);
		}
	}
	
	void setOutputLayerFlag(int l_idx){ outLayerFlag.at(l_idx) = true; }
	void setBiasLayerFlag(int l_idx){ biasFlag.at(l_idx) = true; }
	void setMaxModeFlag(int l_idx){ maxModeFlag.at(l_idx) = true; }

	void initOutputLayer(){	// the input parameter is the number of possible output, which is the number of output dimension
		outLayers.resize(outputDim * sizeOfBatchBlock, 0);
		outLayers_ones.resize(outputDim * sizeOfBatchBlock, 0);
		thrust::fill(outLayers_ones.begin(), outLayers_ones.end(), INPUT_SCALE);
	}

	void prepareOutputLayer(int outIdx){
		if(SILENT==0) start_timing("[TIM] Start to tick!");
		if(SILENT==0) cout << "[MSG] prepareOutputLayer()" << endl;

		if(outIdx==0){
			device_outLayers = raw_pointer_cast(outLayers.data());
		}else if(outIdx==outputDim+1){
			device_outLayers = raw_pointer_cast(outLayers_ones.data());
		}
	}

	void addInputLayer(int NumOfNodes, float scale){
		if(SILENT==0) cout << "[MSG] add input layer" << endl;
		typeOfLayer.push_back(TYPE_INPUT_LAYER);
		numOfNodes.push_back(NumOfNodes);

		device_v_out.push_back(device_vector<float>(sizeOfBatchBlock * numOfNodes.at(0), 0));
		device_delta.push_back(device_vector<float>(sizeOfBatchBlock * numOfNodes.at(0), 0));

		scale_factor.clear();
		scale_factor.push_back(scale);
	}

	// Add a sum layer at specific layer
	void addSumLayer(int numberOfSumNode, int l_idx, float scale){
		typeOfLayer.push_back(TYPE_SUM_LAYER);
		numOfNodes.push_back(numberOfSumNode);

		device_v_out.push_back(device_vector<float>(sizeOfBatchBlock * numberOfSumNode, 0));

		if(maxModeFlag.at(l_idx)){
			v_out_maxIdx[l_idx] = (int*)malloc( sizeOfBatchBlock * numberOfSumNode * sizeof(float));	
			cudaMalloc(&device_v_out_maxIdx[l_idx], sizeOfBatchBlock * numberOfSumNode * sizeof(int));
		}

		device_delta.push_back(device_vector<float>(sizeOfBatchBlock * numberOfSumNode, 0));

		scale_factor.push_back(scale);
	}

	void initialLayers(int NumOfLayers){
		/* initialize layers */	
		for(int i=0;i<NumOfLayers+1;i++) v_out_maxIdx.push_back(NULL); // plus a input layer
		for(int i=0;i<NumOfLayers+1;i++) device_v_out_maxIdx.push_back(NULL); // plus a input layer

		for(int i=0;i<NumOfLayers+1;i++) outLayerFlag.push_back(false);
		for(int i=0;i<NumOfLayers+1;i++) biasFlag.push_back(false);
		for(int i=0;i<NumOfLayers+1;i++) maxModeFlag.push_back(false);
	}

	void initialWeights(int NumOfLayers){
		/* initialize weights */
		for(int i=0;i<NumOfLayers;i++) connAdjList.push_back(DeviceConnAdjList());
		for(int i=0;i<NumOfLayers;i++) connTadjList.push_back(DeviceConnTadjList());
		projW.resize((inputDim/(GRAM_NUM-1)) * vocabulary.size(), 0);
		for(int i=0;i<projW.size();i++) projW[i] = 2*(double)rand()/RAND_MAX;

		/*for(int i=0;i<inputDim/(GRAM_NUM-1);i++){
			for(int j=0;j<vocabulary.size();j++){
				projW[i*vocabulary.size()+j] = (double)(i+1+j+1)/(vocabulary.size()*inputDim/(GRAM_NUM-1));
			}
		}*/

                /*for(int i=0;i<min((int)(inputDim/(GRAM_NUM-1)), 5);i++){
			for(int j=0;j<min((int)vocabulary.size(), 5);j++){
                            cout << projW[i*vocabulary.size()+j] << " ";
                        }
                        cout << endl;
                }*/
	}

	void initialNetwork(int NumOfLayers){
		initialLayers(NumOfLayers);
		initialWeights(NumOfLayers);
		numOfLayer = NumOfLayers+1;
	}

	void calculateMomentum(){
		if(SILENT==0) start_timing("[TIM] Start to tick!");
		if(SILENT==0) cout << "[MSG] calculateMomentum()" << endl;
		for(int l_idx=1;l_idx<numOfLayer;l_idx++){
			if(typeOfLayer[l_idx]==TYPE_SUM_LAYER){
				int n = numOfNodes.at(l_idx);
				int m = numOfNodes.at(l_idx-1) + (outLayerFlag.at(l_idx-1)?outputDim:0);

				GPU_A_Adj.height = connAdjList.at(l_idx-1).getHeight();
				GPU_A_Adj.adjW = raw_pointer_cast(connAdjList.at(l_idx-1).adjConn.data());
				GPU_A_Adj.adjW_size = raw_pointer_cast(connAdjList.at(l_idx-1).adjConn_size.data());
				GPU_A_Adj.adjW_cum_size = raw_pointer_cast(connAdjList.at(l_idx-1).adjConn_cum_size.data());

				setGPU_Matrix(&GPU_A, n, m, raw_pointer_cast(device_d_W.at(l_idx-1).data()));
				setGPU_Matrix(&GPU_B, n, m, raw_pointer_cast(device_mom_d_W.at(l_idx-1).data()));

				dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
				dim3 dimGrid((GPU_A.width + dimBlock.x - 1) / dimBlock.x, (GPU_A.height + dimBlock.y -1) / dimBlock.y);
				CalculateMomentum<<<(GPU_A.height + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(GPU_A_Adj, GPU_A, GPU_B, momentum_rate);
			}
		}

		if(SILENT==0) end_timing("[TIM] End of tick.");
	}

	void updateProjW(){
		if(SILENT==0) start_timing("[TIM] Start to tick!");
		if(SILENT==0) cout << "[MSG] updateProjW()" << endl;

                int voc_size = vocabulary.size();
                CUDAupdateProjW<<<((inputDim/(GRAM_NUM-1)) + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>
                        (raw_pointer_cast(device_delta.at(0).data()), raw_pointer_cast(projW.data()), raw_pointer_cast(batch_samples.data()), 
                        raw_pointer_cast(device_LM_data.data()), vocabulary.size(), (inputDim/(GRAM_NUM-1)), GRAM_NUM, batch_samples.size(), 
                        sizeOfBatchBlock, learning_rate/sizeOfBatchBlock);
		
                /*host_vector<float> host_delta0;
		host_delta0 = device_delta.at(0);
		int voc_size = vocabulary.size();
		int proj_dim = (inputDim/(GRAM_NUM-1));
		proj_d_W.clear();
		proj_d_W.resize(projW.size(), 0);
		for(int i=0;i<GRAM_NUM-1;i++){
			for(int k=0;k<proj_dim;k++){
				for(int j=0;j<batch_samples.size();j++){
					//proj_d_W[k*voc_size + LM_data.at(batch_samples.at(j)+i)] += host_delta0[(proj_dim*i + k)*sizeOfBatchBlock + j];
					projW[k*voc_size + LM_data.at(batch_samples[j]+i)] += (learning_rate/sizeOfBatchBlock) * host_delta0[(proj_dim*i + k)*sizeOfBatchBlock + j];
					projW[k*voc_size + LM_data.at(batch_samples[j]+i)] = max(projW[k*voc_size + LM_data.at(batch_samples[j]+i)], 0.0f);
				}
			}
		}*/

		/*for(int i=0;i<projW.size();i++){
			projW[i] = projW[i] + (learning_rate/sizeOfBatchBlock) * proj_d_W[i];
			projW[i] = max(projW[i], 0.0f);
		}*/

		if(SILENT==0) end_timing("[TIM] End of tick.");
	}
        
        void calculateUpdateSpecialWeights(){
                if(SILENT==0) start_timing("[TIM] Start to tick!");
		if(SILENT==0) cout << "[MSG] calculateUpdateSpecialWeights()" << endl;
                
		//temporary shortcut code for saving the memory which is very sparse	
                int l_idx = 2;
                dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
                dim3 dimGrid((K_POWER + dimBlock.x - 1) / dimBlock.x, (vocabulary.size() + dimBlock.y -1) / dimBlock.y);
                CalculateSpecial1DWKernel<<<dimGrid, dimBlock>>>(raw_pointer_cast(device_delta.at(l_idx+1).data()), numOfNodes.at(l_idx+1), sizeOfBatchBlock, raw_pointer_cast(device_v_out.at(l_idx).data()), numOfNodes.at(l_idx), sizeOfBatchBlock, raw_pointer_cast(device_d_W.at(l_idx).data()), vocabulary.size(), K_POWER, K_POWER);
                
                if(SILENT==0) end_timing("[TIM] End of tick.");
                
                //calculate the change of bias
                calculateDW2BiasKernel<<<(numOfNodes.at(l_idx+1) + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(raw_pointer_cast(device_d_W2_bias.data()), numOfNodes.at(l_idx+1), 1, raw_pointer_cast(device_delta.at(l_idx+1).data()), numOfNodes.at(l_idx+1), sizeOfBatchBlock);
                
                if(SILENT==0) end_timing("[TIM] End of tick.");
                
	}
        
        void updateSpecialWeights(){
                if(SILENT==0) start_timing("[TIM] Start to tick!");
		if(SILENT==0) cout << "[MSG] updateSpecialWeights()" << endl;
                
		//temporary shortcut code for saving the memory which is very sparse	
                int l_idx = 2;
                dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
                dim3 dimGrid((K_POWER + dimBlock.x - 1) / dimBlock.x, (vocabulary.size() + dimBlock.y -1) / dimBlock.y);
                Special1UpdateWKernel<<<dimGrid, dimBlock>>>(raw_pointer_cast(device_W.at(l_idx).data()), vocabulary.size(), K_POWER, raw_pointer_cast(device_d_W.at(l_idx).data()), vocabulary.size(), K_POWER, learning_rate/sizeOfBatchBlock);
                
                /*for(int i=0;i<min((int)vocabulary.size(),5);i++){
                    for(int j=0;j<K_POWER;j++){
                        cout << device_W.at(l_idx)[i*K_POWER + j] << " ";
                    }
                    cout << endl;
                }*/
                
                /*float min_vle = device_W.at(l_idx)[0];
                float max_vle = device_W.at(l_idx)[0];
                for(int i=0;i<device_W.at(l_idx).size();i++){
                	if(device_W.at(l_idx)[i] < min_vle) min_vle = device_W.at(l_idx)[i];
                	if(device_W.at(l_idx)[i] > max_vle) max_vle = device_W.at(l_idx)[i];
                }
                cout << "W2: (";
                cout << scientific << min_vle;
                cout << ",";
                cout << scientific << max_vle;
                cout << ")" << endl;*/
                
                /*for(int i=0;i<min((int)vocabulary.size(),5);i++){
                    for(int j=0;j<K_POWER;j++){
                        cout << device_W.at(l_idx)[i*K_POWER + j] << " ";
                    }
                    cout << endl;
                }*/
                
                if(SILENT==0) end_timing("[TIM] End of tick.");
                
                //update bias
                UpdateW2BiasKernel<<<(numOfNodes.at(l_idx+1) + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(raw_pointer_cast(device_W2_bias.data()), numOfNodes.at(l_idx+1), 1, raw_pointer_cast(device_d_W2_bias.data()), numOfNodes.at(l_idx+1), 1, learning_rate/sizeOfBatchBlock);
               
                /*host_vector<float> W2_bias;
                W2_bias = device_W2_bias;
                host_vector<float> bias_delta;
                bias_delta = device_delta.at(l_idx+1);
                for(int i=0;i<numOfNodes.at(l_idx+1);i++){
                    W2_bias[i] = 0;
                    for(int j=0;j<sizeOfBatchBlock;j++){
                        W2_bias[i] += (learning_rate/sizeOfBatchBlock) * bias_delta[i*sizeOfBatchBlock + j];
                    }
                    W2_bias[i] = max(W2_bias[i], 0.0f);
                }
                device_W2_bias = W2_bias;*/
                
                /*min_vle = device_W2_bias[0];
                max_vle = device_W2_bias[0];
                for(int i=0;i<device_W2_bias.size();i++){
                	if(device_W2_bias[i] < min_vle) min_vle = device_W2_bias[i];
                	if(device_W2_bias[i] > max_vle) max_vle = device_W2_bias[i];
                }
                cout << "device_W2_bias: (";
                cout << scientific << min_vle;
                cout << ",";
                cout << scientific << max_vle;
                cout << ")" << endl;*/
                
                
                if(SILENT==0) end_timing("[TIM] End of tick.");
                
                
                /*cout << "device_W2_bias: " << endl;
                for(int i=0;i<min((int)numOfNodes.at(l_idx+1), 5);i++){
                    cout << device_W2_bias[i] << " ";
                }
                cout << endl;*/
              
	}

	void updateWeights(int l_idx){

		if(typeOfLayer[l_idx+1]==TYPE_SUM_LAYER){
			int n = numOfNodes.at(l_idx+1);
			int m = numOfNodes.at(l_idx) + (outLayerFlag.at(l_idx)?outputDim:0);

			GPU_A_Adj.height = connAdjList.at(l_idx).getHeight();
			GPU_A_Adj.adjW = raw_pointer_cast(connAdjList.at(l_idx).adjConn.data());
			GPU_A_Adj.adjW_size = raw_pointer_cast(connAdjList.at(l_idx).adjConn_size.data());
			GPU_A_Adj.adjW_cum_size = raw_pointer_cast(connAdjList.at(l_idx).adjConn_cum_size.data());

			setGPU_Matrix(&GPU_A, n, m, raw_pointer_cast(device_W.at(l_idx).data()));
			setGPU_Matrix(&GPU_B, n, m, raw_pointer_cast(device_d_W.at(l_idx).data()));
			UpdateW<<<(GPU_A.height + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(GPU_A_Adj, GPU_A, GPU_B, learning_rate/sizeOfBatchBlock, regulatory_term, norm_weight_flag);
		}
	}

	void updateWeights(){
		if(SILENT==0) start_timing("[TIM] Start to tick!");
		if(SILENT==0) cout << "[MSG] updateWeights()" << endl;

		for(int l_idx=0;l_idx<numOfLayer-1;l_idx++){
			updateWeights(l_idx);			
		}

		if(SILENT==0) end_timing("[TIM] End of tick.");
	}

	void readVocabulary(string filename, bool end_notation){
		string line;
		ifstream myfile(filename.c_str());

		if (!myfile)
		{
			cout << "[ERR] Unable to open file"; 
		}else{
			while ( myfile.good() )
			{
				getline (myfile,line);	// read a instance
				if(line.empty()) break;

				vocabularyId[line] = vocabulary.size();	
				vocabulary.push_back(line);
			}
			myfile.close();
		}
                if(end_notation){
                    vocabularyId["</s>"] = vocabulary.size();	
                        vocabulary.push_back("</s>");
                }
                int cnt = vocabulary.size();
		cout << "[MSG] load " << vocabulary.size() << " vocubularies." << endl;
	}

	void readTextData(string filename, bool end_notation){ 
		string line;
		ifstream myfile(filename.c_str());

		if (!myfile)
		{
			cout << "[ERR] Unable to open file"; 
		}else{
			while ( myfile.good() )
			{
				getline (myfile,line);	// read a instance
				if(line.empty()) break;

				vector<string> tokens;
				CTokenizer<CIsFromString>::Tokenize(tokens, line, CIsFromString(" "));
				for(int i=0;i<tokens.size();i++)
					LM_data.push_back(vocabularyId[tokens[i]]);
                                
                                if(end_notation){
                                    LM_data.push_back(vocabularyId["</s>"]);
                                }

			}
			myfile.close();
		}

		if(SILENT==0){
			cout << "[MSG] LM data: ";
			for(int i=0;i<min(15, (int)LM_data.size());i++)
				cout << LM_data.at(i) << " ";
			cout << endl;
		}

		sizeOfData = LM_data.size() - GRAM_NUM + 1;

                host_vector<int> host_LM_data;
                host_LM_data.resize(LM_data.size(), 0);
                for(int i=0;i<LM_data.size();i++) host_LM_data[i] = LM_data[i];
                device_LM_data = host_LM_data;
                
		cout << "[MSG] load training data with size " << LM_data.size() << "." << endl;
	}

	void readValidTextData(string filename, bool end_notation){ 
		string line;
		ifstream myfile(filename.c_str());

		if (!myfile)
		{
			cout << "[ERR] Unable to open file"; 
		}else{
			while ( myfile.good() )
			{
				getline (myfile,line);	// read a instance
				if(line.empty()) break;

				vector<string> tokens;
				CTokenizer<CIsFromString>::Tokenize(tokens, line, CIsFromString(" "));
				for(int i=0;i<tokens.size();i++)
					valid_data.push_back(vocabularyId[tokens[i]]);
			
                                if(end_notation){
                                    valid_data.push_back(vocabularyId["</s>"]);
                                }
                        }
			myfile.close();
		}

		if(SILENT==0){
			cout << "[MSG] validation data: ";
			for(int i=0;i<min(15, (int)valid_data.size());i++)
				cout << valid_data.at(i) << " ";
			cout << endl;
		}

		sizeOfValidData = valid_data.size() - GRAM_NUM + 1;

                host_vector<int> host_valid_data;
                host_valid_data.resize(valid_data.size(), 0);
                for(int i=0;i<valid_data.size();i++) host_valid_data[i] = valid_data[i];
                device_valid_data = host_valid_data;
                
		cout << "[MSG] load validation data with size " << valid_data.size() << "." << endl;
	}

        void readTestTextData(string filename, bool end_notation){ 
		string line;
		ifstream myfile(filename.c_str());

		if (!myfile)
		{
			cout << "[ERR] Unable to open file"; 
		}else{
			while ( myfile.good() )
			{
				getline (myfile,line);	// read a instance
				if(line.empty()) break;

				vector<string> tokens;
				CTokenizer<CIsFromString>::Tokenize(tokens, line, CIsFromString(" "));
				for(int i=0;i<tokens.size();i++)
					test_data.push_back(vocabularyId[tokens[i]]);
			
                                if(end_notation){
                                    test_data.push_back(vocabularyId["</s>"]);
                                }
                        }
			myfile.close();
		}

		if(SILENT==0){
			cout << "[MSG] testing data: ";
			for(int i=0;i<min(15, (int)test_data.size());i++)
				cout << test_data.at(i) << " ";
			cout << endl;
		}

		sizeOfTestData = test_data.size() - GRAM_NUM + 1;

                host_vector<int> host_test_data;
                host_test_data.resize(test_data.size(), 0);
                for(int i=0;i<test_data.size();i++) host_test_data[i] = test_data[i];
                device_test_data = host_test_data;
                
		cout << "[MSG] load testing data with size " << test_data.size() << "." << endl;
	}
        
	void loadWeights(){
		if(SILENT==0) start_timing("[TIM] Start to tick!");
		if(SILENT==0) cout << "[MSG] loadWeights()" << endl;

		ifstream myfile;
		
		myfile.open (weights_file.c_str(), ios::in | ios::binary);
		if (myfile.is_open()) {
			if(SILENT==0) cout << "[MSG] read the weights from file " << weights_file << endl;
			for(int l_idx=1;l_idx<numOfLayer;l_idx++){
                                if(l_idx==3){
                                    float* tmp_W = (float*)malloc(vocabulary.size() * K_POWER * sizeof(float));
                                    myfile.read((char*)tmp_W, vocabulary.size() * K_POWER * sizeof(float));
                                    cudaMemcpy(raw_pointer_cast(device_W.at(l_idx-1).data()), tmp_W, vocabulary.size() * K_POWER * sizeof(float), cudaMemcpyHostToDevice);
                                    free(tmp_W);
                                    continue;
                                }
                            
				if(typeOfLayer[l_idx]==TYPE_SUM_LAYER){
					int n = numOfNodes.at(l_idx);
					int m = numOfNodes.at(l_idx-1) + (outLayerFlag.at(l_idx-1)?outputDim:0);
					float* tmp_W = (float*)malloc(n * m * sizeof(float));
					myfile.read((char*)tmp_W, n * m * sizeof(float));
					cudaMemcpy(raw_pointer_cast(device_W.at(l_idx-1).data()), tmp_W, n * m * sizeof(float), cudaMemcpyHostToDevice);

					free(tmp_W);
				}
			}

			//load projW
                        float* tmp_projW =(float*)malloc((inputDim/(GRAM_NUM-1)) * vocabulary.size() * sizeof(float));
			projW.resize((inputDim/(GRAM_NUM-1)) * vocabulary.size(), 0);
			myfile.read((char*)tmp_projW, (inputDim/(GRAM_NUM-1)) * vocabulary.size() * sizeof(float));
			cudaMemcpy(raw_pointer_cast(projW.data()), tmp_projW, (inputDim/(GRAM_NUM-1)) * vocabulary.size() * sizeof(float), cudaMemcpyHostToDevice);
			free(tmp_projW);
                        
			//load bias
			float* tmp_bias =(float*)malloc(vocabulary.size() * sizeof(float));
			myfile.read((char*)tmp_bias, vocabulary.size() * sizeof(float));
			cudaMemcpy(raw_pointer_cast(device_W2_bias.data()), tmp_bias, vocabulary.size() * sizeof(float), cudaMemcpyHostToDevice);
			free(tmp_bias);
		}else{
			cout << "[MSG] Do not find the weights file " << weights_file << ". Use the random weights." << endl;
		}
		myfile.close();
		if(SILENT==0) end_timing("[TIM] End of tick.");
	}

	void saveWeights(int times){
		if(SILENT==0) start_timing("[TIM] Start to tick!");
		if(SILENT==0) cout << "[MSG] saveWeights()" << endl;

		stringstream num_weights_file;
		num_weights_file << weights_file << "." << times;

		ofstream myfile;
		myfile.open (num_weights_file.str().c_str(), ios::out | ios::binary);
		if (myfile.is_open()) {
			if(SILENT==0) cout << "[MSG] save the weights to file " << weights_file << endl;
			for(int l_idx=1;l_idx<numOfLayer;l_idx++){
                            
                                if(l_idx==3){
                                    float* tmp_W = (float*)malloc(vocabulary.size() * K_POWER * sizeof(float));
                                    cudaMemcpy(tmp_W, raw_pointer_cast(device_W.at(l_idx-1).data()), vocabulary.size() * K_POWER * sizeof(float), cudaMemcpyDeviceToHost);
                                    myfile.write ((char*)tmp_W, vocabulary.size() * K_POWER * sizeof(float));
                                    free(tmp_W);
                                    continue; 
                                }
                            
				if(typeOfLayer[l_idx]==TYPE_SUM_LAYER){
					int n = numOfNodes.at(l_idx);
					int m = numOfNodes.at(l_idx-1) + (outLayerFlag.at(l_idx-1)?outputDim:0);
					float* tmp_W = (float*)malloc(n * m * sizeof(float));
					cudaMemcpy(tmp_W, raw_pointer_cast(device_W.at(l_idx-1).data()), n * m * sizeof(float), cudaMemcpyDeviceToHost);
					myfile.write ((char*)tmp_W, n * m * sizeof(float));
					free(tmp_W);
				}
			}

			//save projW
			float* tmp_projW = (float*)malloc((inputDim/(GRAM_NUM-1)) * vocabulary.size() * sizeof(float));
			cudaMemcpy(tmp_projW, raw_pointer_cast(projW.data()), (inputDim/(GRAM_NUM-1)) * vocabulary.size() * sizeof(float), cudaMemcpyDeviceToHost);
			myfile.write((char*)tmp_projW, (inputDim/(GRAM_NUM-1)) * vocabulary.size() * sizeof(float));
			free(tmp_projW);
                        
			//save bias
			float* tmp_bias = (float*)malloc(vocabulary.size() * sizeof(float));
			cudaMemcpy(tmp_bias, raw_pointer_cast(device_W2_bias.data()), vocabulary.size() * sizeof(float), cudaMemcpyDeviceToHost);
			myfile.write((char*)tmp_bias, vocabulary.size() * sizeof(float));
			free(tmp_bias);
		}else{
			cout << "[Err] Can't write weights to file " << weights_file << endl;
		}
		myfile.close();
		if(SILENT==0) end_timing("[TIM] End of tick.");
	}

	void start_timing(string msg){
		cudaDeviceSynchronize();
		cout << msg << endl;
		start_time = clock();
	}

	void end_timing(string msg){
		cudaDeviceSynchronize();
		cout << msg << " (spends " << (double)(clock() - start_time)/CLOCKS_PER_SEC << " seconds)" << endl; 
	}

        void setWeightsFile(string file){ weights_file = file; };
	void setBatchSize(int sizeOfBatchBlock){ this->sizeOfBatchBlock = sizeOfBatchBlock; }
        void setK_POWER(int k_power){ this->K_POWER = k_power; }
        void setGRAM_NUM(int gram_num){ this->GRAM_NUM = gram_num; }
        void setRAND_SEED(int rand_seed){ this->rand_seed = rand_seed; }
        
	void clearD_W(){
		if(SILENT==0) start_timing("[TIM] Start to tick!");
		if(SILENT==0) cout << "[MSG] clearD_W()" << endl;
		for(int l_idx=1;l_idx<numOfLayer;l_idx++){
                        //temporary shortcut code for saving the memory which is very sparse
                        if(l_idx==3){
                            device_d_W.at(l_idx-1).clear();
                            device_d_W.at(l_idx-1).resize(vocabulary.size()*K_POWER, 0);
                            
                            continue;
                        }
			if(typeOfLayer[l_idx]==TYPE_SUM_LAYER){
				int n = numOfNodes.at(l_idx);
				int m = numOfNodes.at(l_idx-1) + (outLayerFlag.at(l_idx-1)?outputDim:0);
				device_d_W.at(l_idx-1).clear();
				device_d_W.at(l_idx-1).resize(m*n, 0);
			}
                }
		proj_d_W.clear();
		proj_d_W.resize((inputDim/(GRAM_NUM-1)) * vocabulary.size(), 0);
                
                device_d_W2_bias.clear();
                device_d_W2_bias.resize(vocabulary.size()*K_POWER, 0);
                
		if(SILENT==0) end_timing("[TIM] End of tick.");
	}
	
	void clearDelta(){
		if(SILENT==0) start_timing("[TIM] Start to tick!");
		if(SILENT==0) cout << "[MSG] clearDelta()" << endl;
		for(int l_idx=0;l_idx<numOfLayer;l_idx++){
			device_delta.at(l_idx).clear();
			device_delta.at(l_idx).resize(numOfNodes.at(l_idx) * sizeOfBatchBlock, 0);
		}
		if(SILENT==0) end_timing("[TIM] End of tick.");
	}
	
	void destructLayers(){
		for(int l_idx=1;l_idx<numOfLayer;l_idx++) {
			if(maxModeFlag.at(l_idx)){
				free(v_out_maxIdx[l_idx]);
				cudaFree(device_v_out_maxIdx[l_idx]);
			}
		}
	}

	void destructLayeredSPN(){
		destructLayers();
	}
};


#endif	/* SPNITER_H */

