#include <iostream>
#include <time.h>

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

#include "utils.h"
#include "ksvd.h"
#include "_omp.h"

#define PATCH_SIZE (8)
#define ATOM_NUM (256)
#define ITER_NUM (30)
#define SPARSITY (50)

int main(int argc, char **argv) {

    std::cout << "reading image..." << std::endl;
    cv::Mat image = read_image("../image/house.png");
    // mat_data(image, "image");
    
    std::cout << "image to patches..." << std::endl;
    Eigen::MatrixXd Y = image_to_patches(image, PATCH_SIZE);
    // matrix_data(Y, "patches", 1024);
    Eigen::MatrixXd Y_norm = matrix_norm(Y);
    matrix_data(Y_norm, "patches_norm", 1024);

    // ==============================================
    //  split line start
    //  ksvd train start
    // ==============================================

    clock_t start, finish;
    double totaltime;
    start = clock();

    KsvdValues ksvdvalues = ksvd(Y_norm, ATOM_NUM, ITER_NUM, SPARSITY);

    std::cout << "recover image..." << std::endl;
    matrix_data(ksvdvalues.D, "dictionary_update", 256);
    matrix_data(ksvdvalues.X, "sparse", 1024);
    Eigen::MatrixXd patches_norm = multipy(ksvdvalues.D, ksvdvalues.X);
    matrix_data(patches_norm, "patches_recover", 1024);

    finish = clock();
    totaltime = (double)(finish - start) / CLOCKS_PER_SEC;
    std::cout << "clock " << totaltime << "s" << std::endl;

    // ==================================================
    //     spilt line end
    //     ksvd train end
    // ==================================================
    std::cout << "patches to image..." << std::endl;
    Eigen::MatrixXd patches = anti_matrix_norm(patches_norm, Y);
    cv::Mat image2 = patches_to_image(patches, PATCH_SIZE);
    mat_data(image2, "recover");
    
    cv::imwrite("../result/recover.png", image2);
    cv::imshow("recover", image2);
    cv::waitKey(0);

    return 0;
}