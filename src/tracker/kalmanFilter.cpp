#include "kalmanFilter.h"
#include <Eigen/Cholesky>
#include <iostream>

namespace byte_kalman
{
	KalmanFilter::KalmanFilter()
	{
		int ndim = 4;
		double dt = 1.;
		//std::string sep = "\n----------------------------------------\n";

		_motion_mat = Eigen::MatrixXf::Identity(8, 8);
		for (int i = 0; i < ndim; i++) {
			_motion_mat(i, ndim + i) = dt;
		}
		_update_mat = Eigen::MatrixXf::Identity(4, 8);

		//Eigen::IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");
		//std::cout << _motion_mat.format(CleanFmt) << sep;
		//std::cout << _update_mat.format(CleanFmt) << sep;
	}

	void KalmanFilter::setupRatio(globalConfig *conf){
		this->_std_weight_position_x = (float)conf->mot_weight_position_x;
		this->_std_weight_position_y = (float)conf->mot_weight_position_y;
		this->_std_weight_position_a = (float)conf->mot_weight_position_a;
		this->_std_weight_position_h = (float)conf->mot_weight_position_h;

		this->_std_weight_velocity_x = (float)conf->mot_weight_velocity_x;
		this->_std_weight_velocity_y = (float)conf->mot_weight_velocity_y;
		this->_std_weight_velocity_a = (float)conf->mot_weight_velocity_a;
		this->_std_weight_velocity_h = (float)conf->mot_weight_velocity_h;
		/*std::cout << "_std_weight_position_x - " << this->_std_weight_position_x << std::endl;
		std::cout << "_std_weight_position_y - " << this->_std_weight_position_y << std::endl;
		std::cout << "_std_weight_position_a - " << this->_std_weight_position_a << std::endl;
		std::cout << "_std_weight_position_h - " << this->_std_weight_position_h << std::endl;*/
	}

	KAL_DATA KalmanFilter::initiate(const DETECTBOX &measurement)
	{
		DETECTBOX mean_pos = measurement;
		DETECTBOX mean_vel;
		for (int i = 0; i < 4; i++) mean_vel(i) = 0;

		KAL_MEAN mean;
		for (int i = 0; i < 8; i++) {
			if (i < 4) mean(i) = mean_pos(i);
			else mean(i) = mean_vel(i - 4);
		}

		KAL_MEAN std;
		std(0) = 2 * _std_weight_position_x * measurement[3];
		std(1) = 2 * _std_weight_position_y * measurement[3];
		std(2) = 1e-2;
		std(3) = 2 * _std_weight_position_h * measurement[3];
		std(4) = 10 * _std_weight_velocity_x * measurement[3];
		std(5) = 10 * _std_weight_velocity_y * measurement[3];
		std(6) = 1e-5;
		std(7) = 10 * _std_weight_velocity_h * measurement[3]; 

		KAL_MEAN tmp = std.array().square();
		KAL_COVA var = tmp.asDiagonal();
		return std::make_pair(mean, var);
	}

	void KalmanFilter::predict(KAL_MEAN &mean, KAL_COVA &covariance){
		//std::string sep = "\n*************************************************\n";
        //Eigen::IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");
		//revise the data;
		DETECTBOX std_pos;
		std_pos << _std_weight_position_x * mean(3), _std_weight_position_y * mean(3), _std_weight_position_a, _std_weight_position_h * mean(3);
		DETECTBOX std_vel;
		std_vel << _std_weight_velocity_x * mean(3), _std_weight_velocity_y * mean(3), _std_weight_velocity_a, _std_weight_velocity_h * mean(3);
		KAL_MEAN tmp;
		tmp.block<1, 4>(0, 0) = std_pos;
		tmp.block<1, 4>(0, 4) = std_vel;
		tmp = tmp.array().square();
		KAL_COVA motion_cov = tmp.asDiagonal();
		KAL_MEAN mean1 = this->_motion_mat * mean.transpose();
		//std::cout << mean1.format(CleanFmt) << sep;
		KAL_COVA covariance1 = this->_motion_mat * covariance *(_motion_mat.transpose());
		//std::cout << covariance1.format(CleanFmt) << sep;
		covariance1 += motion_cov;
		mean = mean1;
		covariance = covariance1;
        //std::cout << covariance.format(CleanFmt) << sep;
	}

	KAL_HDATA KalmanFilter::project(const KAL_MEAN &mean, const KAL_COVA &covariance){
		DETECTBOX std;
		std << _std_weight_position_x * mean(3), _std_weight_position_y * mean(3), _std_weight_position_i_a, _std_weight_position_h * mean(3);
		KAL_HMEAN mean1 = _update_mat * mean.transpose();
		KAL_HCOVA covariance1 = _update_mat * covariance * (_update_mat.transpose());
		Eigen::Matrix<float, 4, 4> diag = std.asDiagonal();
		diag = diag.array().square().matrix();
		covariance1 += diag;
		//    covariance1.diagonal() << diag;
		return std::make_pair(mean1, covariance1);
	}

	/*DETECTBOX KalmanFilter::getDelta(const KAL_MEAN &mean, const DETECTBOX &measurement){

	}*/

	KAL_DATA KalmanFilter::update(const KAL_MEAN &mean, const KAL_COVA &covariance, const DETECTBOX &measurement){
		KAL_HDATA pa = project(mean, covariance);
		KAL_HMEAN projected_mean = pa.first;
		KAL_HCOVA projected_cov = pa.second;
		//std::string sep = "\n*************************************************\n";
        //Eigen::IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");

		Eigen::Matrix<float, 4, 8> B = (covariance * (_update_mat.transpose())).transpose();
		Eigen::Matrix<float, 8, 4> kalman_gain = (projected_cov.llt().solve(B)).transpose(); // eg.8x4
		//std::cout << kalman_gain.format(CleanFmt) << sep;
		Eigen::Matrix<float, 1, 4> _innovation = measurement - projected_mean; //eg.1x4
		//std::cout << _innovation.format(CleanFmt) << sep;
		auto tmp = _innovation * (kalman_gain.transpose());
		//std::cout << tmp.format(CleanFmt) << sep;
		this->innovation = tmp;

		//std::cout << tmp.format(CleanFmt) << sep;
		KAL_MEAN new_mean = (mean.array() + tmp.array()).matrix();
		KAL_COVA new_covariance = covariance - kalman_gain * projected_cov*(kalman_gain.transpose());
		return std::make_pair(new_mean, new_covariance);
	}

	Eigen::Matrix<float, 1, -1>
		KalmanFilter::gating_distance(
			const KAL_MEAN &mean,
			const KAL_COVA &covariance,
			const std::vector<DETECTBOX> &measurements)
	{
		KAL_HDATA pa = this->project(mean, covariance);
		KAL_HMEAN mean1 = pa.first;
		KAL_HCOVA covariance1 = pa.second;

		DETECTBOXSS d(measurements.size(), 4);
		int pos = 0;
		for (DETECTBOX box : measurements) {
			d.row(pos++) = box - mean1;
		}
		Eigen::Matrix<float, -1, -1, Eigen::RowMajor> factor = covariance1.llt().matrixL();
		Eigen::Matrix<float, -1, -1> _tri = factor.triangularView<Eigen::Lower>().solve<Eigen::OnTheRight>(d).transpose();
		auto _d_tri = ((_tri.array())*(_tri.array())).matrix();
		auto square_maha = _d_tri.colwise().sum();
		return square_maha;
	}

	COV_R KalmanFilter::covToImage(KAL_MEAN& mean, KAL_COVA& covariance){
		//KAL_COVA _covariance = covariance;
		/*std::string sep = "\n----------------------------------------\n";
        Eigen::IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");
        std::cout << _covariance.format(CleanFmt) << sep;*/

		float r1 = 0, r2 = 0;
		float _tmp0 = (covariance(0, 0) + covariance(1, 1))/2.0;
		/*std::cerr << "elem 0 0 - " << _covariance(0, 0) << std::endl;
		std::cerr << "elem 1 1 - " << _covariance(1, 1) << std::endl;
		std::cerr << "_tmp0_kalman - " << _tmp0 << std::endl;*/
		float _tmp1 = sqrtf(powf(((covariance.coeff(0, 0) - covariance.coeff(1, 1))/2),2)+powf(covariance.coeff(0, 1),2));
		/*std::cerr << "_tmp1_kalman - " << _tmp1 << std::endl;*/
		r1 = _tmp0 + _tmp1;
		r2 = _tmp0 - _tmp1;
		/*std::cerr << "r1_kalman - " << r1 << std::endl;
		std::cerr << "r2_kalman - " << r2 << std::endl;*/
		return std::make_pair(r1, r2);
	}

	float KalmanFilter::getAngle(KAL_MEAN& mean, KAL_COVA& covariance, float r1){
		if((covariance(0, 1) == 0) && (covariance(0, 0)>=covariance(1, 1))){
			return 0;
		}
		if((covariance(0, 1) == 0) && (covariance(0, 0)<covariance(1, 1))){
			return 90;
		}
		return 57.2958*atan2(r1-covariance(0, 0), covariance(0, 1)); 
	}
}