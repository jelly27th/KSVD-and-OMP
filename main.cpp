#include <iostream>
#include <time.h>

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

#include "utils.h"
#include "ksvd.h"
#include "_omp.h"

#define PATCH_SIZE (8)
#define ATOM_NUM (256)
#define SPARSITY (50)
#define ITER_NUM (10)


int main(int argc, char **argv) {

    std::cout << "reading image..." << std::endl;
    cv::Mat image = read_image("../image/house.png");
    
    std::cout << "image to patches..." << std::endl;
    Eigen::MatrixXd Y = image_to_patches(image, PATCH_SIZE);
    Eigen::MatrixXd Y_norm = matrix_norm(Y);

    // ==============================================
    //  ksvd train start
    // ==============================================

    clock_t start, finish;
    double totaltime;
    start = clock();

    KsvdValues ksvdvalues = ksvd(Y_norm, ATOM_NUM, ITER_NUM, SPARSITY);

    finish = clock();
    totaltime = (double)(finish - start) / CLOCKS_PER_SEC;
    std::cout << "clock " << totaltime << "s" << std::endl;

    // ==================================================
    //     ksvd train end
    // ==================================================

    std::cout << "recover image..." << std::endl;
    Eigen::MatrixXd patches_norm = multipy(ksvdvalues.D, ksvdvalues.X);

    std::cout << "patches to image..." << std::endl;
    Eigen::MatrixXd patches = anti_matrix_norm(patches_norm);
    cv::Mat image2 = patches_to_image(patches, PATCH_SIZE);
    
    cv::imwrite("../image/recover.png", image2);
    cv::imshow("recover", image2);
    cv::waitKey(0);

    return 0;
}