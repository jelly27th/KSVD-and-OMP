#include "utils.h"

cv::Mat read_image(std::string filename) {
    cv::Mat image = cv::imread(filename, cv::IMREAD_GRAYSCALE);

    cv::Mat image_double;
    image.convertTo(image_double, CV_64F);
    cv::imshow("ori", image);

    return image_double;
}

Eigen::MatrixXd image_to_patches(cv::Mat image, int PATCH_SIZE) {

    int rows = image.rows;
    int cols = image.cols;
    int n_channels = image.channels();

    // 计算图像可以被分为多少个8*8的块
    int n_rows = rows / PATCH_SIZE;
    int n_cols = cols / PATCH_SIZE;

    // 将图像转换为double类型的矩阵
    cv::Mat double_image;
    // image.convertTo(double_image, CV_64F, 1.0 / 255);
    image.convertTo(double_image, CV_64F);

    Eigen::MatrixXd patches(n_channels * PATCH_SIZE * PATCH_SIZE, n_rows * n_cols);
    Eigen::VectorXd col(PATCH_SIZE * PATCH_SIZE);
    
    int index = 0;
    for (int i = 0; i < rows; i += PATCH_SIZE ) {
        for (int j = 0; j < cols; j += PATCH_SIZE) {

            col.setZero();
            int idx = 0;
            for (int x = i; x < i + PATCH_SIZE; x++) {
                for (int y = j; y < j + PATCH_SIZE; y++) {
                    col(idx++) = double_image.at<double>(x, y);
                }
            }
            patches.col(index++) = col;
        }
    }

    return patches;
}

Eigen::MatrixXd multipy(Eigen::MatrixXd D, Eigen::MatrixXd X) {
    return D * X;
}

cv::Mat patches_to_image(Eigen::MatrixXd patches, int PATCH_SIZE) {
    int num_patches = patches.cols();
    int image_size = sqrt(num_patches) * PATCH_SIZE;

    // Create output image
    cv::Mat image(image_size, image_size, CV_64F);

    // Loop over patches and copy to output image
    for (int i = 0; i < num_patches; i++)
    { 
        Eigen::VectorXd col = patches.col(i); 
        int cols = i % int(sqrt(num_patches));
        int rows = i / int(sqrt(num_patches));

        int idx = 0;
        for (int x = (rows*PATCH_SIZE); x < PATCH_SIZE + (rows*PATCH_SIZE); ++x) {
            for (int y = (cols * PATCH_SIZE); y < PATCH_SIZE + (cols * PATCH_SIZE); ++y) {
                image.at<double>(x, y) = col(idx++);
            }
        }
    }
    
    cv::Mat image_uint8;
    image.convertTo(image_uint8, CV_8U);

    return image_uint8;
}

void mat_data(cv::Mat data, std::string name)
{
    name = "../result/" + name;
    std::ofstream file(name);
    file << format(data, cv::Formatter::FMT_NUMPY) << std::endl;
    file.close();
}

void matrix_data(Eigen::MatrixXd data, std::string name, int size)
{
    name = "../result/" + name;
    std::ofstream file(name);
    Eigen::IOFormat fmt(size, 0, ", ", "\n", "[", "]");
    file << data.format(fmt) << std::endl;
    file.close();
}

/**
 * return l2 matrix
*/
Eigen::MatrixXd matrix_norm(const Eigen::MatrixXd &D) {
    Eigen::MatrixXd D_norm = D;
    for (int i = 0; i < D.cols(); i++) {
        double norm = D_norm.col(i).norm();
        D_norm.col(i) /= norm;
    }
    return D_norm;
}

Eigen::MatrixXd anti_matrix_norm(const Eigen::MatrixXd &D, const Eigen::MatrixXd &origin) {
    // Eigen::MatrixXd D_anti = D;
    // for (int i = 0; i < D.cols(); i++)
    // {
    //     double norm = origin.col(i).norm();
    //     D_anti.col(i) = D.col(i) * norm;
    // }
    // return D_anti;

    return D * 30;
}