#pragma once

#include <opencv2/opencv.hpp>
#include "kalmanFilter.h"

using namespace cv;
using namespace std;

enum TrackState { New = 0, Tracked, Lost, Removed };

class STrack
{
public:
	STrack(vector<float> tlwh_, float score, int label, globalConfig *conf);
	~STrack();

	vector<float> static tlbr_to_tlwh(vector<float> &tlbr);
	void static multi_predict(vector<STrack*> &stracks, byte_kalman::KalmanFilter &kalman_filter);
	void trackPredict();
	void static_tlwh();
	void static_tlbr();
	vector<float> tlwh_to_xyah(vector<float> tlwh_tmp);
	void xyah_to_tlwh();
	vector<float> to_xyah();
	void mark_lost();
	void mark_removed();
	int next_id();
	int end_frame();
	
	void activate(byte_kalman::KalmanFilter &kalman_filter, int frame_id);//, int classId
	void re_activate(STrack &new_track, int frame_id, bool new_id = false);
	void update(STrack &new_track, int frame_id);

public:
	bool is_activated;
	bool isOverlapped_0;
	bool isOverlapped_1;
	bool isOverlapped_2;
	bool isOverlapped_3;
	int track_id;
	int state;
	int startClassId;
	int ClassId;
	vector<float> _tlwh;
	vector<float> tlwh;
	vector<float> tlbr;
	vector<float> tlwh_predict;
	int frame_id;
	int tracklet_len;
	int start_frame;
	float track_thresh;
	float high_thresh;
	float match_thresh; 
	float angle;
	float area;
	float area_prev;
	float a_prev;


	float w_max;
	float h_max;
	float h_prev;
	float w_prev;

	KAL_MEAN mean;
	KAL_MEAN mean_predict;
	KAL_COVA covariance_predict;
	KAL_MEAN mean_prev;
	KAL_COVA covariance;
	DETECTBOX delta;
	COV_R toDraw;
	float score;

//private:
	byte_kalman::KalmanFilter kalman_filter;
};
