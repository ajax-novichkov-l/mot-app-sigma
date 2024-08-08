#pragma once

#include <cstddef>
#include <vector>

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
typedef Eigen::Matrix<float, 1, 4, Eigen::RowMajor> DETECTBOX;
typedef Eigen::Matrix<float, -1, 4, Eigen::RowMajor> DETECTBOXSS;
typedef Eigen::Matrix<float, 1, 128, Eigen::RowMajor> FEATURE;
typedef Eigen::Matrix<float, Eigen::Dynamic, 128, Eigen::RowMajor> FEATURESS;
//typedef std::vector<FEATURE> FEATURESS;

//Kalmanfilter
//typedef Eigen::Matrix<float, 8, 8, Eigen::RowMajor> KAL_FILTER;
typedef Eigen::Matrix<float, 1, 8, Eigen::RowMajor> KAL_MEAN;
typedef Eigen::Matrix<float, 8, 8, Eigen::RowMajor> KAL_COVA;
typedef Eigen::Matrix<float, 1, 4, Eigen::RowMajor> KAL_HMEAN;
typedef Eigen::Matrix<float, 4, 4, Eigen::RowMajor> KAL_HCOVA;
using KAL_DATA = std::pair<KAL_MEAN, KAL_COVA>;
using KAL_HDATA = std::pair<KAL_HMEAN, KAL_HCOVA>;

//main
using RESULT_DATA = std::pair<int, DETECTBOX>;

//tracker:
using TRACKER_DATA = std::pair<int, FEATURESS>;
using MATCH_DATA = std::pair<int, int>;
typedef struct t {
	std::vector<MATCH_DATA> matches;
	std::vector<int> unmatched_tracks;
	std::vector<int> unmatched_detections;
}TRACHER_MATCHD;

typedef struct _globalConfig{
    std::string path_ipuFw;
    std::string path_model;
    std::string path_imgages;
    std::string path_labels;
    std::string dataType;
    unsigned int edgeSizeH;
    unsigned int edgeSizeW;
    unsigned int iterations;
    bool out_boxes;
    bool out_dump;
    bool out_txt;
    bool out_nms;
    double  threshold_confidence;
    double  threshold_main;
    double  threshold_boxes;

	double mot_track_thresh;
	double mot_high_thresh;
	double mot_match_thresh;

	double mot_c0_track_thresh;
	double mot_c0_high_thresh;
	double mot_c0_match_thresh;

	double mot_c1_track_thresh;
	double mot_c1_high_thresh;
	double mot_c1_match_thresh;

	double mot_c2_track_thresh;
	double mot_c2_high_thresh;
	double mot_c2_match_thresh;

	unsigned int mot_fps;
	unsigned int mot_max_time_lost;
}globalConfig;

//linear_assignment:
typedef Eigen::Matrix<float, -1, -1, Eigen::RowMajor> DYNAMICM;
