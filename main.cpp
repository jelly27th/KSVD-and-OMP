#include <iostream>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

#include "utils.h"
#include "_omp.h"
#include "ksvd.h"

#define PATCH_SIZE (8)
#define ATOM_NUM (256)
#define ITER_NUM (10)

int main(int argc, char **argv) {

    std::cout << "reading image..." << std::endl;
    cv::Mat image = read_image("../image/house.png");
    
    std::cout << "image to patches..." << std::endl;
    Eigen::MatrixXd Y = image_to_patches(image, PATCH_SIZE);

    std::cout << "ksvd init..." << std::endl;
    Eigen::MatrixXd D = ksvd_iniitation(Y, ATOM_NUM);

    Eigen::MatrixXd X;

    std::cout << "ksvd train..." << std::endl;
    for (int i = 0; i < ITER_NUM; ++i) {
        
        _omp(Y, D, X);
        ksvd_update(Y, D, X);
    }

    std::cout << "recover image..." << std::endl;
    Eigen::MatrixXd patches = multipy(D, X);

    std::cout << "patches to image..." << std::endl;
    cv::Mat image2 = patches_to_image(patches, PATCH_SIZE);

    return 0;
}