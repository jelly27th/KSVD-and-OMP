#ifndef _UTILS_H
#define _UTILS_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <Eigen/Dense>

cv::Mat read_image(std::string filename);
Eigen::MatrixXd image_to_patches(cv::Mat image, int PATCH_SIZE);
Eigen::MatrixXd multipy(Eigen::MatrixXd D, Eigen::MatrixXd X);
cv::Mat patches_to_image(Eigen::MatrixXd patches, int PATCH_SIZE);

#endif