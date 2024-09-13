#include "STrack.h"

STrack::STrack(vector<float> tlwh_, float score, int label, globalConfig *conf)
{
	_tlwh.resize(4);
	_tlwh.assign(tlwh_.begin(), tlwh_.end());

	tlwh_predict.resize(4);

	is_activated = false;
	track_id = 0;
	state = TrackState::New;
	
	tlwh.resize(4);
	tlbr.resize(4);

	static_tlwh();
	static_tlbr();
	frame_id = 0;
	tracklet_len = 0;
	this->score = score;
	this->startClassId = label;
	this->ClassId = label;
	switch(this->startClassId){
		case 0:
			this->high_thresh = conf->mot_c0_high_thresh;
			this->match_thresh = conf->mot_c0_match_thresh;
			this->track_thresh = conf->mot_c0_track_thresh;
		break;
		case 1:
			this->high_thresh = conf->mot_c1_high_thresh;
			this->match_thresh = conf->mot_c1_match_thresh;
			this->track_thresh = conf->mot_c1_track_thresh;
		break;
		case 2:
			this->high_thresh = conf->mot_c2_high_thresh;
			this->match_thresh = conf->mot_c2_match_thresh;
			this->track_thresh = conf->mot_c2_track_thresh;
		break;
		default:
			this->high_thresh = conf->mot_high_thresh;
			this->match_thresh = conf->mot_match_thresh;
			this->track_thresh = conf->mot_track_thresh;
		break;
	}
	area_prev = tlwh[2]*tlwh[3];
	h_prev = tlwh[3];
	start_frame = 0;
}

STrack::~STrack()
{
}

void STrack::activate(byte_kalman::KalmanFilter &kalman_filter, int frame_id)//, int classId
{
	this->kalman_filter = kalman_filter;
	this->track_id = this->next_id();

	vector<float> _tlwh_tmp(4);
	_tlwh_tmp[0] = this->_tlwh[0];
	_tlwh_tmp[1] = this->_tlwh[1];
	_tlwh_tmp[2] = this->_tlwh[2];
	_tlwh_tmp[3] = this->_tlwh[3];
	vector<float> xyah = tlwh_to_xyah(_tlwh_tmp);
	DETECTBOX xyah_box;
	xyah_box[0] = xyah[0];
	xyah_box[1] = xyah[1];
	xyah_box[2] = xyah[2];
	xyah_box[3] = xyah[3];
	auto mc = this->kalman_filter.initiate(xyah_box);
	this->mean = mc.first;
	this->covariance = mc.second;

	static_tlwh();
	static_tlbr();

	this->tracklet_len = 0;
	this->state = TrackState::Tracked;
	if (frame_id == 1)
	{
		this->is_activated = true;
		//this->classId = classId;
	}
	//this->is_activated = true;
	this->frame_id = frame_id;
	this->start_frame = frame_id;
}

void STrack::re_activate(STrack &new_track, int frame_id, bool new_id)
{
	vector<float> xyah = tlwh_to_xyah(new_track.tlwh);
	DETECTBOX xyah_box;
	xyah_box[0] = xyah[0];
	xyah_box[1] = xyah[1];
	xyah_box[2] = xyah[2];
	xyah_box[3] = xyah[3];
	auto mc = this->kalman_filter.update(this->mean, this->covariance, xyah_box);
	this->mean = mc.first;
	this->covariance = mc.second;

	static_tlwh();
	static_tlbr();

	this->tracklet_len = 0;
	this->state = TrackState::Tracked;
	this->is_activated = true;
	this->frame_id = frame_id;
	this->score = new_track.score;
	if (new_id)
		this->track_id = next_id();
}

void STrack::update(STrack &new_track, int frame_id)
{
	this->frame_id = frame_id;
	this->tracklet_len++;
	this->ClassId = new_track.startClassId;
	vector<float> xyah = tlwh_to_xyah(new_track.tlwh);
	DETECTBOX xyah_box;
	xyah_box[0] = xyah[0];
	xyah_box[1] = xyah[1];
	xyah_box[2] = xyah[2];
	xyah_box[3] = xyah[3];

	auto mc = this->kalman_filter.update(this->mean, this->covariance, xyah_box);
	this->toDraw = this->kalman_filter.covToImage(this->mean, this->covariance);
	this->angle = this->kalman_filter.getAngle(this->mean, this->covariance, this->toDraw.first);
	//this->delta = this->kalman_filter.getDelta(this->mean, this->covariance, xyah_box);
	this->mean_prev = this->mean;
	this->mean = mc.first;
	this->covariance = mc.second;

	static_tlwh();
	xyah_to_tlwh();
	static_tlbr();

	this->state = TrackState::Tracked;
	this->is_activated = true;

	this->score = new_track.score;
}

void STrack::static_tlwh()
{
	if (this->state == TrackState::New)
	{
		tlwh[0] = _tlwh[0];
		tlwh[1] = _tlwh[1];
		tlwh[2] = _tlwh[2];
		tlwh[3] = _tlwh[3];
		return;
	}

	tlwh[0] = mean[0];
	tlwh[1] = mean[1];
	tlwh[2] = mean[2];
	tlwh[3] = mean[3];

	tlwh[2] *= tlwh[3];
	tlwh[0] -= tlwh[2] / 2;
	tlwh[1] -= tlwh[3] / 2;
}

void STrack::static_tlbr()
{
	tlbr.clear();
	tlbr.assign(tlwh.begin(), tlwh.end());
	tlbr[2] += tlbr[0];
	tlbr[3] += tlbr[1];
}

vector<float> STrack::tlwh_to_xyah(vector<float> tlwh_tmp)
{
	vector<float> tlwh_output = tlwh_tmp;
	tlwh_output[0] += tlwh_output[2] / 2;
	tlwh_output[1] += tlwh_output[3] / 2;
	tlwh_output[2] /= tlwh_output[3];
	return tlwh_output;
}

void STrack::trackPredict(){
	kalman_filter.predict(mean_predict, covariance_predict);
}

void STrack::xyah_to_tlwh(){
	vector<float> tmp_ltwh = {0.0,0.0,0.0,0.0};
	isOverlapped_0 = false;
	isOverlapped_1 = false;
	isOverlapped_2 = false;
	mean_predict = mean;
	covariance_predict = covariance;
	kalman_filter.predict(mean_predict, covariance_predict);
	kalman_filter.predict(mean_predict, covariance_predict);

	tmp_ltwh[0] = mean_predict[0];
	tmp_ltwh[1] = mean_predict[1];
	tmp_ltwh[2] = mean_predict[2];
	tmp_ltwh[3] = mean_predict[3]+0.2*mean_predict[3];

	tmp_ltwh[2] *= tmp_ltwh[3];
	tmp_ltwh[0] -= tmp_ltwh[2] / 2;
	tmp_ltwh[1] -= tmp_ltwh[3] / 2;


	if(tmp_ltwh[3] > this->h_max){
		this->h_max = tmp_ltwh[3];
	}
	if(tmp_ltwh[2] > this->w_max){
		this->w_max = tmp_ltwh[2];
	}

	//float w_multiply = 1.0;
	if(((mean_prev[4]<0) && (mean[4]>0)) || ((mean_prev[4]>0) && (mean[4]<0)) || 
		((mean_prev[4]==0) && (mean[4]<0)) || ((mean_prev[4]==0) && (mean[4]>0)) || 
			((mean_prev[4]>0) && (mean[4]==0)) || ((mean_prev[4]<0) && (mean[4]==0))){
		if(mean[4]>0){
			isOverlapped_0 = true;
		}else{
			if(mean[4]==0){
				isOverlapped_1 = true;
			}else{
				isOverlapped_2 = true;
			}
		}
	}


	if(isOverlapped_0){
		tmp_ltwh[2] *=0.2;
	}
	if(isOverlapped_2){
		tmp_ltwh[0] -= 0.2*tmp_ltwh[2];
		tmp_ltwh[2] *=0.4;
	}
	/*if(mean[0]>mean_prev[0]){
		isOverlapped_0 = true;
	}else{
		if(mean[0]==0){
			isOverlapped_1 = true;
		}else{
			isOverlapped_2 = true;
		}
	}*/
	

	area = tmp_ltwh[3]*tmp_ltwh[2];

	if(area < area_prev){
		tmp_ltwh[3] = 0.2*tmp_ltwh[3] + 0.8*this->h_prev;
		tmp_ltwh[2] = 0.2*tmp_ltwh[2] + 0.8*this->w_prev;
	}/*else{
		if(((mean_prev[6]<0) && (mean_predict[6]>=0)) || ((mean_prev[6]>=0) && (mean_predict[6]<0))){
			//float w_prev = mean_prev[2]*mean_prev[3];
			if(w_prev < tmp_ltwh[2]){
				tmp_ltwh[2] = 0.2*tmp_ltwh[2] + 0.8*this->w_max;
				tmp_ltwh[0] = mean_predict[0] - tmp_ltwh[2] / 2;
			}
			if(h_prev < tmp_ltwh[3]){
				tmp_ltwh[3] = 0.2*tmp_ltwh[3] + 0.8*this->h_max;
				tmp_ltwh[1] = mean_predict[1] - tmp_ltwh[3] / 2;
			}
		}
	}*/
		tmp_ltwh[0] = mean_predict[0] - tmp_ltwh[2] / 2;
		tmp_ltwh[1] = mean_predict[1] - tmp_ltwh[3] / 2;
	if(mean_predict[4]>=0){
		this->tlwh_predict[0] = tmp_ltwh[0]-mean_predict[4];
		this->tlwh_predict[2] = tmp_ltwh[2]+4*abs(mean_predict[4]); 
	}else{
		this->tlwh_predict[0] = tmp_ltwh[0]+5*mean_predict[4];
		this->tlwh_predict[2] = tmp_ltwh[2]+6*abs(mean_predict[4]); 
	}

	//if(mean_predict[5]>=0){
		this->tlwh_predict[1] = tmp_ltwh[1];//+mean_predict[5];
		this->tlwh_predict[3] = tmp_ltwh[3];//+2.0*abs(mean_predict[5]); 
	//}else{
	//	this->tlwh_predict[1] = tmp_ltwh[1];//-mean_predict[5];
	//	this->tlwh_predict[3] = tmp_ltwh[3];//+2*abs(mean_predict[5]); 
	//}


	area_prev = area;
	h_prev = tmp_ltwh[3];
	w_prev = tmp_ltwh[2];
	a_prev = mean_predict[2];

}


vector<float> STrack::to_xyah()
{
	return tlwh_to_xyah(tlwh);
}

vector<float> STrack::tlbr_to_tlwh(vector<float> &tlbr)
{
	tlbr[2] -= tlbr[0];
	tlbr[3] -= tlbr[1];
	return tlbr;
}

void STrack::mark_lost()
{
	state = TrackState::Lost;
}

void STrack::mark_removed()
{
	state = TrackState::Removed;
}

int STrack::next_id()
{
	static int _count = 0;
	_count++;
	return _count;
}

int STrack::end_frame()
{
	return this->frame_id;
}

void STrack::multi_predict(vector<STrack*> &stracks, byte_kalman::KalmanFilter &kalman_filter)
{
	for (int i = 0; i < stracks.size(); i++)
	{
		if (stracks[i]->state != TrackState::Tracked)
		{
			stracks[i]->mean[7] = 0;
		}
		kalman_filter.predict(stracks[i]->mean, stracks[i]->covariance);
		//stracks[i]->mean_predict = stracks[i]->mean;
		stracks[i]->static_tlwh();
		//stracks[i]->xyah_to_tlwh();
		stracks[i]->static_tlbr();
	}
}