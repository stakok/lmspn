/* 
 * File:   spn_train.cu
 * Author: weichen_cheng
 *
 * Created on November 18, 2013, 10:23 AM
 */

#include <cstdlib>
#include <iostream>
#include <iterator>
#include "SPNIter.h"

#include "INIReader/INIReader.cpp"

using namespace std;

int main(int argc, char* argv[]){

	if(argc!=2){
		cout << "Please specify the location of config.ini" << endl;
		return 0;
	}

	INIReader reader(argv[1]);
	if (reader.ParseError() < 0) {
		std::cout << "Can't load " << argv[1] << endl;
		return 1;
	}

	double learning_rate = reader.GetReal("NetworkParam", "learning_rate", 0.001);
	double momentum_rate = reader.GetReal("NetworkParam", "momentum_rate", 0.3);
	double regulatory_term = reader.GetReal("NetworkParam", "regulatory_term", 0.01);
	int iterations = reader.GetInteger("NetworkParam", "iterations", 1000);
	int in_dim = reader.GetInteger("NetworkParam", "in_dim", 4);
	int out_dim = reader.GetInteger("NetworkParam", "out_dim", 3);
	string train_file = reader.Get("NetworkParam", "train_file", "tr_data");
        string valid_file = reader.Get("NetworkParam", "valid_file", "valid_data");
	string test_file = reader.Get("NetworkParam", "test_file", "test_data");
	string weightsFile = reader.Get("NetworkParam", "weights_file", "weights_file");
	string vocab = reader.Get("NetworkParam", "vocabulary", "vocab");

	int NumOfLayers = reader.GetInteger("NetworkParam", "NumOfLayers", 0);
	int batchSize = reader.GetInteger("NetworkParam", "BatchSize", 0);
	int norm_weight = reader.GetInteger("NetworkParam", "norm_weight", 1);
	int GPUID = reader.GetInteger("NetworkParam", "GPUID", 0);
        int end_notation = reader.GetInteger("NetworkParam", "END_NOTATION", 0);
        int k_power = reader.GetInteger("NetworkParam", "K_POWER", 0);
        int gram_num = reader.GetInteger("NetworkParam", "GRAM_NUM", 0);
        int rand_seed = reader.GetInteger("NetworkParam", "RAND_SEED", 1);
	int out_prob = reader.GetInteger("NetworkParam", "OUT_PROB", 0);

	cudaSetDevice(GPUID);

	int Rep = reader.GetInteger("NetworkParam", "Rep", 2);

	/* initialize random seed: */
 	srand (rand_seed);

	// Training
	SPN spn(in_dim, out_dim);
        
        spn.setK_POWER(k_power);
        spn.setGRAM_NUM(gram_num);
        spn.setRAND_SEED(rand_seed);
        
	spn.readVocabulary(vocab, end_notation);
	
	spn.readTextData(train_file, end_notation);
	spn.readValidTextData(valid_file, end_notation);
        spn.readTestTextData(test_file, end_notation);

	spn.setWeightsFile(weightsFile);
	spn.setBatchSize(batchSize);

	//spn.shuffleData();
	spn.initialNetwork(NumOfLayers-1);
	spn.initOutputLayer();

	// build the network
	for(int l_idx=0;l_idx<NumOfLayers;l_idx++){

		stringstream layerTag;
		layerTag << "Layer" << l_idx;
		int layerType = reader.GetInteger(layerTag.str(), "Type", -1);
		
		int outputLayer = reader.GetInteger(layerTag.str(), "OutputVariable", 0);
		int biasLayer = reader.GetInteger(layerTag.str(), "AddBias", 0);

		if(layerType==TYPE_INPUT_LAYER){
			int NumOfNodes = reader.GetInteger(layerTag.str(), "NumOfNodes", 0);
			float scale = reader.GetReal(layerTag.str(), "Scale", 1);

			if(outputLayer){
				cout << "[MSG] Layer " << l_idx << " withOutputVariable" << endl;
				spn.setOutputLayerFlag(l_idx);
			}

			if(biasLayer){
				cout << "[MSG] Layer " << l_idx << " AddBias" << endl;
				spn.setBiasLayerFlag(l_idx);
			}

			spn.addInputLayer(NumOfNodes, scale);
		}

		if(layerType==TYPE_PROD_LAYER){
		
			if(outputLayer){
				cout << "[MSG] Layer " << l_idx << " withOutputVariable" << endl;
				spn.setOutputLayerFlag(l_idx);
			}

			if(biasLayer){
				cout << "[MSG] Layer " << l_idx << " AddBias" << endl;
				spn.setBiasLayerFlag(l_idx);
			}
			
			int NumOfNodes = reader.GetInteger(layerTag.str(), "NumOfNodes", 0);
			float scale = reader.GetReal(layerTag.str(), "Scale", 1);
			spn.addProdLayer(NumOfNodes, l_idx, scale);

			//read the adjacency list defined by user
			string adjW_file = reader.Get(layerTag.str(), "File_adjW", "");

			if(adjW_file.size()>0){
				cout << "[MSG] User defined " << adjW_file << endl;
				spn.readProdLayerConnFromFile(adjW_file, l_idx);
			}
		}

		if(layerType==TYPE_SUM_LAYER){
			if(outputLayer){
				cout << "[MSG] Layer " << l_idx << " withOutputVariable" << endl;
				spn.setOutputLayerFlag(l_idx);
			}

			if(biasLayer){
				cout << "[MSG] Layer " << l_idx << " AddBias" << endl;
				spn.setBiasLayerFlag(l_idx);
			}

			int NumOfNodes = reader.GetInteger(layerTag.str(), "NumOfNodes", 0);
			float scale = reader.GetReal(layerTag.str(), "Scale", 1);
			float maxMode = reader.GetInteger(layerTag.str(), "MaxMode", false);
			if(maxMode==true) spn.setMaxModeFlag(l_idx);
			spn.addSumLayer(NumOfNodes, l_idx, scale);

			//read the adjacency list defined by user
			string adjW_file = reader.Get(layerTag.str(), "File_adjW", "");
			if(adjW_file.size()>0){
				cout << "[MSG] User defined " << adjW_file << endl;
				spn.readSumLayerConnFromFile(adjW_file, l_idx);
			}
		}
	}

	//spn.generateConnectionType3(reader.GetInteger("Layer1", "NumOfNodes", 0), reader.GetInteger("Layer3", "NumOfNodes", 0), Rep);
	//spn.generateConnectionType4(reader.GetInteger("Layer1", "NumOfNodes", 0), reader.GetInteger("Layer3", "NumOfNodes", 0), reader.GetInteger("Layer5", "NumOfNodes", 0), Rep);
	spn.generateConnectionType1();

	if(out_prob==0){
		spn.learningOnDataset(iterations, learning_rate, momentum_rate, regulatory_term, norm_weight==0?false:true);
	}else{
		spn.loadWeights();
		spn.calculateTestingProbability();
	}

	//spn.saveWeights();

	return 0;
}
