#pragma once

#include "dataType.h"

namespace byte_kalman
{
	class KalmanFilter
	{
	public:
		KalmanFilter();
		KAL_DATA initiate(const DETECTBOX& measurement);
		void predict(KAL_MEAN& mean, KAL_COVA& covariance);
		void setupRatio(globalConfig *conf);
		KAL_HDATA project(const KAL_MEAN& mean, const KAL_COVA& covariance);
		KAL_DATA update(const KAL_MEAN& mean,
			const KAL_COVA& covariance,
			const DETECTBOX& measurement);
		COV_R covToImage(KAL_MEAN& mean, KAL_COVA& covariance);

		Eigen::Matrix<float, 1, -1> gating_distance(
			const KAL_MEAN& mean,
			const KAL_COVA& covariance,
			const std::vector<DETECTBOX>& measurements);
		float getAngle(KAL_MEAN& mean, KAL_COVA& covariance, float r1);
		Eigen::Matrix<float, 1, 8> innovation;
	private:
		Eigen::Matrix<float, 8, 8, Eigen::RowMajor> _motion_mat;
		Eigen::Matrix<float, 4, 8, Eigen::RowMajor> _update_mat;
		float _std_weight_position_x;
		float _std_weight_position_y;
		float _std_weight_position_a;
		float _std_weight_position_h;
		float _std_weight_velocity_x;
		float _std_weight_velocity_y;
		float _std_weight_velocity_a;
		float _std_weight_velocity_h;
		float _std_weight_position_i_a;
	};
}