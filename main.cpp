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
    mat_data(image, "image");
    
    std::cout << "image to patches..." << std::endl;
    Eigen::MatrixXd Y = image_to_patches(image, PATCH_SIZE);
    matrix_data(Y, "patches", 1024);

    std::cout << "ksvd init..." << std::endl;
    Eigen::MatrixXd D = ksvd_iniitation(Y, ATOM_NUM);
    matrix_data(D, "dictionary", 256);

    Eigen::MatrixXd X(D.cols(), Y.cols());

    std::cout << "ksvd train..." << std::endl;
    for (int i = 0; i < ITER_NUM; ++i) {
        
        std::cout << "train iter time: " << i << std::endl;
        
        std::cout << "omp start..." << std::endl;
        // _omp(Y, D, X);
        _omp1(Y, D, X);

        std::cout << "kvd update..." << std::endl;
        ksvd_update(Y, D, X);
    }

    std::cout << "recover image..." << std::endl;
    matrix_data(D, "dictionary", 256);
    matrix_data(X, "sparse", 1024);
    Eigen::MatrixXd patches = multipy(D, X);

    std::cout << "patches to image..." << std::endl;
    cv::Mat image2 = patches_to_image(patches, PATCH_SIZE);
    mat_data(image2, "recover");
    
    cv::imwrite("../result/recover.png", image2);
    cv::imshow("recover", image2);
    cv::waitKey(0);

    return 0;
}