/* 
 * File:   AdjMatrix.h
 * Author: weichen_cheng
 *
 * Created on November 18, 2013, 10:26 AM
 */

#ifndef ADJMATRIX_H
#define	ADJMATRIX_H

#include <thrust/device_vector.h>
using namespace thrust;

class DeviceConnAdjList{
private:
	int height;
public:
	device_vector<int> adjConn;
	device_vector<int> adjConn_size;
	device_vector<int> adjConn_cum_size;

	int getHeight(){ return height; };

	DeviceConnAdjList(){ 
		height = 0;
	}

	void allocateMemory(int h){
		height = h;
	}
};

class DeviceConnTadjList{
private:
	int height;
public:
	device_vector<int> TadjConn;
	device_vector<int> TadjConn_size;
	device_vector<int> TadjConn_cum_size;

	int getHeight(){ return height; };

	DeviceConnTadjList(){ 
		height = 0;
	}

	void allocateMemory(int h){
		height = h;
	}
};

#endif	/* ADJMATRIX_H */

