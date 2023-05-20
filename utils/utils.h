#ifndef _UTILS_H
#define _UTILS_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

#include <Eigen/Dense>
#include <Eigen/Core>
#include <fstream>

#include <omp.h>
#include <stdio.h>

cv::Mat read_image(std::string filename);
Eigen::MatrixXd image_to_patches(cv::Mat image, int PATCH_SIZE);
Eigen::MatrixXd multipy(Eigen::MatrixXd D, Eigen::MatrixXd X);
cv::Mat patches_to_image(Eigen::MatrixXd patches, int PATCH_SIZE);
void matrix_data(Eigen::MatrixXd data, std::string name, int size);
void mat_data(cv::Mat data, std::string name);
Eigen::MatrixXd matrix_norm(const Eigen::MatrixXd &D);
Eigen::MatrixXd anti_matrix_norm(const Eigen::MatrixXd &D, const Eigen::MatrixXd &origin);

#endif