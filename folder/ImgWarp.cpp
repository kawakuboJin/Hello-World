#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cv.h>
#include <cxcore.h>
#include <highgui.h>
#include <opencv2/opencv.hpp>


using namespace std;
using namespace cv;

void CalculateUVFromUV(const cv::Point2f& _src,const cv::Point2f& _srcCenter,
		const cv::Point2f& _dstCenter,float r_src, float r_dst,cv::Point2f& planeCoor){
	double x, y;
        y = -cos(float(_src.y) / r_src) * r_src;
        x = -cos(float(_src.x) / r_src) * sqrt(r_src * r_src - y * y);

	double l = sqrt(x*x + y*y);
	if ( l >= r_src) {
		cout << "bad l: " << l << "\tr_src: " << r_src << endl;
		l = r_src * 0.99;
	}
	double theta = asin(l / r_src);
	double L = tan(theta) * r_src;
	if (L < 1) {
		planeCoor = _dstCenter;
		return;
	}
	x *= L/l;
	y *= L/l;
	theta = atan(L / r_dst);
	l = r_dst * sin(theta);
        x *= l/L;
	y *= l/L;
	
	double cosv = - y / r_dst;
	double v = acos(cosv);//0~pi
	double cosu = - x / sqrt(r_dst * r_dst - y * y);
	double u = acos(cosu);
	planeCoor.x = (u - CV_PI / 2.0) * r_dst + _dstCenter.x;
	planeCoor.y = (v - CV_PI / 2.0) * r_dst + _dstCenter.y;

}

void CalculatePlane(const cv::Point2f& _src,const cv::Point2f& _srcCenter,
		const cv::Point2f& _dstCenter,float _r,cv::Point2f& planeCoor){
	cv::Point2f src = _src - _srcCenter;	
	double L = sqrt(_r*_r + src.dot(src));
	src.x *= _r / L;
	src.y *= _r / L;
	double theta = sqrt(src.dot(src)) / _r;
	if (theta > 1){
		planeCoor = cv::Point2f(0, 0);
	}
	else{
		if (theta == 0){
			planeCoor = _dstCenter;
		}
		else{
			double atheta = asin(theta);
			double temp = 2 * atheta / (CV_PI*theta);
			planeCoor = temp*src + _dstCenter;
		}
	}
}

void CalculateSphcial(const cv::Point2f& _src, const cv::Point2f& _srcCenter, float _r, cv::Point2f& sphCoor){

	if (_src.x<_r / 2 || _src.x>3 * _r / 2 || _src.y<_r / 3 || _src.y>_r * 5 / 3){
		sphCoor = cv::Point2f(-1, -1);
		return;
	}	

	cv::Point2f src = _src;
	src.x -= _r;
	src.y -= _r;

	double theta = sqrt(src.dot(src)) / _r*CV_PI / 2;
	if (theta == 0){
		sphCoor = _srcCenter;
	}
	else{
		double temp = tan(theta)*CV_PI / theta / 2.0;
		sphCoor = src*temp + _srcCenter;
	}
	
}

void GenerateRemapUV(const cv::Mat& uv_src, cv::Mat& uv_dst, double angleHorizon, double angleVertical, float scale){

	float r_src = uv_src.cols / CV_PI / 2.0;
	float r_dst = r_src * scale;
	cv::Size si(angleHorizon / 180.0 * CV_PI * r_dst, angleVertical / 180 * CV_PI * r_dst);
	cv::Mat tmpM(int(CV_PI * r_dst), int(CV_PI * r_dst * 2), uv_src.type(), cv::Scalar(0,0,0));
	cv::Mat dst;
	cv::Mat map(si, CV_32FC2);

	int offset_x, offset_y;
	offset_x = CV_PI * r_dst / 2.0 - si.width / 2.0;
	offset_y = CV_PI * r_dst / 2.0 - si.height / 2.0;
#pragma omp parallel for
	for (int i = 0; i < si.height; i++){
		for (int j = 0; j < si.width; j++){
			CalculateUVFromUV(cv::Point2f((j + offset_x), (i + offset_y)), cv::Point2f(si.width / 2, si.height / 2), 
				cv::Point2f(uv_src.cols / 2, uv_src.rows / 2), r_dst, r_src, map.at<cv::Point2f>(i, j));			
		}
	}	
	cv::remap(uv_src, dst, map, cv::Mat(), cv::INTER_CUBIC);
	cv::Mat res = tmpM(cv::Rect((offset_x + CV_PI * r_dst / 2.0), offset_y, map.cols, map.rows));
	dst.copyTo(res);
	cv::resize(tmpM, uv_dst, cv::Size(uv_src.cols,uv_src.rows));
	//uv_dst = tmpM;
}

void GenerateUndistort(const cv::Mat& _distort, cv::Mat& undistort, double angleHorizon, double angleVertical){

	cv::Mat M = cv::getRotationMatrix2D(cv::Point2f(_distort.cols / 2, _distort.rows / 2), -90, 1.0);
	M.at<double>(0, 2) += (_distort.rows - _distort.cols) / 2;
	M.at<double>(1, 2) += (_distort.cols - _distort.rows) / 2;
	cv::Mat distort;
	cv::warpAffine(_distort, distort, M, cv::Size(_distort.rows, _distort.cols));
	double r = distort.cols * 180 / angleHorizon / 2.0;
	cv::Size si(tan(angleHorizon / 360.0 * CV_PI) * r * 2, tan(angleVertical / 360.0 * CV_PI) * r * 2);
	cv::Mat map(si, CV_32FC2);

#pragma omp parallel for
	for (int i = 0; i < si.height; i++){
		for (int j = 0; j < si.width; j++){
			CalculatePlane(cv::Point2f(j, i), cv::Point2f(si.width / 2, si.height / 2), 
				cv::Point2f(distort.cols / 2, distort.rows / 2), r, map.at<cv::Point2f>(i, j));			
		}
	}	
	cv::remap(distort, undistort, map, cv::Mat(), cv::INTER_CUBIC);
}

void GenerateDistort(const cv::Mat& _undistort, cv::Mat& distort, double angleHorizon, double angleVertical){
	double r = _undistort.cols / tan(angleHorizon / 2 * CV_PI / 180) / 2;
	cv::Mat map(r * 2 * angleHorizon / 180 * 2, r * 2 * angleHorizon / 180 * 2, CV_32FC2);
#pragma omp parallel for
	for (int i = 0; i < map.rows; i++){
		for (int j = 0; j < map.cols; j++){
			CalculateSphcial(cv::Point2f(j, i), cv::Point2f(_undistort.cols / 2, _undistort.rows / 2), 
				r, map.at<cv::Point2f>(i, j));
		}
	}

	cv::Mat distortNoRotation;
	cv::remap(_undistort, distortNoRotation, map, cv::Mat(), cv::INTER_CUBIC);
	double l = r*angleHorizon / 180 * 2;
	distortNoRotation = distortNoRotation(cv::Rect(r - l / 2, r - l * angleVertical / angleHorizon / 2, 
		l, l * angleVertical / angleHorizon));

	cv::Mat M = cv::getRotationMatrix2D(cv::Point2f(distortNoRotation.cols / 2, distortNoRotation.rows / 2), 90, 1.0);
	M.at<double>(0, 2) += (distortNoRotation.rows - distortNoRotation.cols) / 2;
	M.at<double>(1, 2) += (distortNoRotation.cols - distortNoRotation.rows) / 2;
	cv::warpAffine(distortNoRotation, distort, M, cv::Size(distortNoRotation.rows, distortNoRotation.cols));
}

int main(int argc, char* argv[]){
	if (argc < 6) {
		cout << "usage:./bin inputImage angleH angleV outputScale outputImage" << endl;
		exit(-1);
	}
	cv::Mat distort, undistort;
	
	for (int i = 1; i <= 1; i++){
		char ch[100];
//		sprintf(ch, "D:\\Matlab\\Doc\\right_9_25\\image%d-740.jpg", i);
		distort = cv::imread(argv[1]);
  		GenerateRemapUV(distort, undistort, atoi(argv[2]), atoi(argv[3]), atof(argv[4]));
//		GenerateUndistort(distort, undistort, 90, 120);
//		sprintf(ch, "D:\\Matlab\\Doc\\right_9_25\\image%d-740_undistort.jpg", i);
		cv::imwrite(argv[5], undistort);
	}
/*
	for (int i = 1; i <= 1; i++){
		char ch[100];
//		sprintf(ch, "D:\\Matlab\\Doc\\right_9_25\\image%d-740_undistort.jpg", i);
		undistort = cv::imread(ch);
		GenerateDistort(undistort, distort, 90, 120);
//		sprintf(ch, "D:\\Matlab\\Doc\\right_9_25\\image%d-1.jpg", i);
		cv::imwrite(ch, distort);
	}*/
}
